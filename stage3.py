import os
import io
import re
import math
import numpy as np
import operator
import pandas as pd
import dsh
import traceback
import collections

import table_solver as tab
import calc_solver as calc
import fit_solver as fit


########## ERRORS ##############################

########## ERRORS ##############################

def report(where):
    print(f'[agent] {where} failed:\n{traceback.format_exc()}')   


########## PROMPTS #####################################
MODEL = os.environ.get('DSH_MODEL', dsh.DEFAULT_MODEL)
MAX_ROUNDS = 4
# worst case: try more with LLM with some temp
N_SAMPLES = 4
SAMPLE_TEMP = 0.7

def vote(candidates):
    # compare different answers on how often they came
    vals = [c for c in candidates if c is not None and math.isfinite(c)]
    if not vals:
        return None
    key = lambda v: float(f'{v:.9g}')
    counts = collections.Counter(key(v) for v in vals)
    best = max(counts.values())
    return next(v for v in vals if counts[key(v)] == best)


############# ROUTING #############################################
def route_tasks(tasks):
    calc_files = []
    fit_files = []
    table_files = []
    for i, task in enumerate(tasks):
        files = task['files']
        if not files:
            calc_files.append(i)
        elif 'train.csv' in {os.path.basename(p) for p in files}:
            fit_files.append(i)
        else:
            table_files.append(i)
    return calc_files, fit_files, table_files


# ========================================================================
#                           AGENT
# ========================================================================
class Agent:
    def __init__(self):
        self.llm = dsh.load_llm(MODEL)
        self.out = []
    
    def solve(self, tasks, seconds=40.0):
        self.out = [None] * len(tasks)
        try:                                              
            self.solve_fct(tasks, seconds)
        except Exception:
            report('solve')
        # in worst case: we return 0 for everything
        return [o if o is not None else {'id': t['id'], 'answer': 0}
                for o, t in zip(self.out, tasks)]


    def solve_fct(self, tasks, seconds=40.0):
        clock = dsh.Clock(seconds)
        calc_tasks, fit_tasks, table_tasks = route_tasks(tasks)
        

        ##################### table, fit and calc function ###########################
        def fit_llm(indices, hclock):
            unclear_fit_tasks = []
            for i in indices:
                try:
                    self.out[i] = {"id": tasks[i]['id'], 'answer': fit.solve_fit_task(tasks[i])}
                except fit.Unclear as e:
                    unclear_fit_tasks.append((i,e))
                except Exception as e:
                    print(f'[fit] {tasks[i]["id"]}: {type(e).__name__}: {e}')
            # finished all tasks or no time left
            if not unclear_fit_tasks or hclock.left() < 2: 
                return

            # for all unclear parts in exercises, we need the help of an llm to know which unclear part of the sentence is related to which column
            questions = [(i, e, ph) for i, e in unclear_fit_tasks for ph in e.phrases]
            convs = [[{'role': 'system', 'content': fit.SYSTEM_fit}, *fit.DEMO_fit, {'role': 'user', 'content': 'Columns:\n' + 
                       '\n'.join(f'{c}: {x}' for c, x in e.columns.items()) + f'\n\nPhrase: {ph}'}] for i, e, ph in questions]
            replies = self.llm.chat(convs, max_new_tokens=15, max_time=hclock.left() - 1, stop=['</call>'])
            for (i, e, ph), reply in zip(questions, replies):
                print(f'[fit LLM] {tasks[i]["id"]} | {ph!r} -> {reply!r}')

            # llm returns the right column, then answer deterministically
            # read the llm replies (columns): match if the llm replied with only one column
            columns_from_llm = {i: {} for i, _ in unclear_fit_tasks}                                      
            for (i, e, ph), reply in zip(questions, replies):
                codes_in_reply = [c for c in e.columns if re.search(rf'(?<!\w){re.escape(c)}(?!\w)', reply)]
                if len(codes_in_reply) == 1:                                
                    columns_from_llm[i][ph] = codes_in_reply[0]
            for i, e in unclear_fit_tasks:              
                try:
                    # solve each task again with the LLMs columns: if a phrase is still missing or the column does not work, we return 0
                    self.out[i] = {'id': tasks[i]['id'], 'answer': fit.solve_fit_task(tasks[i], columns_from_llm[i])}
                except Exception as err:
                    print(f'[fit] {tasks[i]["id"]} after mapping: {type(err).__name__}: {err}')


        def table_llm(indices, hclock):
            if not indices:
                return

            # First use the LLM only for unresolved column names, then let Python
            # execute the already-understood aggregation/filtering operation.
            requests = {}
            for i in indices:
                request = tab.column_request(tasks[i])
                if request is not None:
                    requests[i] = request

            if requests and hclock.left() > 9:
                try:
                    ids = list(requests)
                    replies = self.llm.chat(
                        [requests[i][1] for i in ids],
                        max_new_tokens=32,
                        max_time=max(2, min(12, hclock.left()))
                    )
                    for i, reply in zip(ids, replies):
                        issue = requests[i][0]
                        # Never allow a model to invent a column: use exact codes only.
                        candidates = list(issue.columns)
                        suggestion = reply.strip().strip('`').strip('"\'')
                        matches = [col for col in candidates if suggestion == col]
                        if len(matches) != 1:
                            matches = [col for col in candidates if re.fullmatch(re.escape(col), suggestion, re.I)]
                        if len(matches) != 1:
                            continue
                        hint = {(issue.filename, str(issue.phrase).lower().strip()): matches[0]}
                        answer = float(tab.solve_data({**tasks[i], '_column_hints': hint}))
                        if math.isfinite(answer):
                            self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
                except Exception:
                    pass

            unresolved = [i for i in indices if self.out[i] is None]
            if not unresolved:
                return

            prepared, conversations, original_prompts, contexts = {}, {}, {}, {}
            for i in unresolved:
                task = tasks[i]
                try:
                    tables, df, context = tab.prepare_table_semantic(task)
                except Exception as error:
                    try:
                        loaded = tab.read_files(task['files'], task['objective'])
                        tables = loaded[0] if isinstance(loaded, tuple) else loaded
                        names = [name for name in tables if name != 'data_dictionary.csv']
                        if not names:
                            continue
                        df = tables[names[0]]
                        context = 'Available CSVs: ' + repr({k: list(v.columns) for k, v in tables.items()})
                        context += '\nJoin/preparation problem: ' + str(error)[:160]
                    except Exception:
                        continue

                context = tab.table_context(task, tables, df, context)
                contexts[i] = context
                prepared[i] = (tables, df)
                question = 'Question: ' + task['objective'] + '\n\n' + context
                original_prompts[i] = question
                conversations[i] = (
                    [{'role': 'system', 'content': tab.SYSTEM_table}]
                    + tab.table_examples(task)
                    + [{'role': 'user', 'content': question}]
                )

            plan_active = list(conversations)
            plan_convs = {}

            plan_active = list(conversations)
            plan_convs = {}
            for i in plan_active:
                try:
                    plan_convs[i] = tab.prompt(tasks[i], contexts[i], prepared[i][1])
                except Exception:
                    pass
            plan_active = list(plan_convs)
            for attempt in range(2):
                if not plan_active or hclock.left() < 8:
                    break
                try:
                    replies = self.llm.chat([plan_convs[i] for i in plan_active], max_new_tokens=220,
                                            max_time=max(1, min(10, hclock.left() - 6)))
                except Exception:
                    break
                pending = []
                for i, reply in zip(plan_active, replies):
                    try:
                        plan = tab.parse(reply)
                        tab.validate_semantics(tasks[i], plan, prepared[i][1])
                        answer = tab.validate_table_result(tasks[i], tab.execute(plan, prepared[i][1]))
                        self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
                    except Exception as e:
                        plan_convs[i] = plan_convs[i] + [
                            {'role': 'assistant', 'content': reply[:500]},
                            {'role': 'user', 'content': 'PLAN ERROR: ' + str(e)[:180]
                             + '\nReturn a corrected SINGLE JSON plan. Use exact schema column codes. No Python.'}]
                        pending.append(i)
                plan_active = pending

            active = [i for i in conversations if self.out[i] is None]
 
            weak = {}

            def run_candidate(i, reply):
                tables, df = prepared[i]
                result = tab.evaluate_table_expression(reply, tables, df)
                weak[i] = result                                  
                tab.expression_intent(tasks[i], reply)
                return tab.validate_table_result(tasks[i], result)


            for attempt in range(3):
                if not active or self.llm is None or hclock.left() < 3:
                    break
                try:
                    max_time = max(1, hclock.left() - (5 if attempt == 0 else 1))
                    replies = self.llm.chat(
                        [conversations[i] for i in active],
                        max_new_tokens=300,
                        max_time=max_time
                    ) # worst case: let the llm try it more if time left and get the most probable answer by doing it mulitple times
                    samples = []
                    if attempt == 0 and hclock.left() > 8:
                        samples = self.llm.chat([conversations[i] for i in active for _ in range(N_SAMPLES)],
                            max_new_tokens=200, temperature=SAMPLE_TEMP, batch_size=16,
                             max_time=min(8, hclock.left() - 4))
                except Exception:
                    break
                pending = []
                for pos, (i, reply) in enumerate(zip(active, replies)):
                    candidates = [reply] + samples[pos * N_SAMPLES:(pos + 1) * N_SAMPLES]
                    results, greedy_error = [], None
                    for k, cand in enumerate(candidates):
                        try:
                            results.append(run_candidate(i, cand))
                        except Exception as error:
                            if k == 0:
                                greedy_error = error
                    if results: # deterministic answer still counts and counts double
                        det = tab.WEAK.get(tasks[i]['id'])
                        pool = ([det] * 2 if det is not None else []) + results
                        self.out[i] = {'id': tasks[i]['id'], 'answer': vote(pool)}
                        continue

                    columns = list(prepared[i][1].columns)
                    prompt = (
                        original_prompts[i]
                        + '\n\nYour previous expression failed: ' + str(greedy_error)[:230]
                        + '\nPrevious expression: ' + reply[:300]
                        + '\nFix the expression without dropping any question conditions.'
                        + '\nValid columns: ' + repr(columns)
                    )
                    conversations[i] = ([
                        {'role': 'system', 'content': tab.SYSTEM_table}] + tab.table_examples(tasks[i])
                        + [{'role': 'user', 'content': prompt}])
                    pending.append(i)
                active = pending

            for i in unresolved:
                if self.out[i] is None:
                    fallback = weak.get(i, tab.WEAK.get(tasks[i]['id']))
                    ok = fallback is not None and math.isfinite(fallback)
                    self.out[i] = {'id': tasks[i]['id'], 'answer': fallback if ok else 0}



        def calc_llm(indices, hclock):
            unresolved=[]
            for i in indices:
                try:
                    result=float(calc.calc_general(tasks[i]))
                    if not math.isfinite(result):raise ValueError('Non-finite answer')
                    self.out[i]={'id':tasks[i]['id'],'answer':result}
                except Exception:
                    unresolved.append(i)
            if not unresolved:
                return
            if hclock.left()<3:
                for i in unresolved:self.out[i]={'id':tasks[i]['id'],'answer':0}
                return
            convs=[]
            for i in unresolved:
                convs.append([{'role':'system','content':calc.SYSTEM_calc}]+calc.DEMO_calc+[{'role':'user','content':tasks[i]['objective']}])
            
            def rounds(conv_list, temperature):
                convs_=[list(c) for c in conv_list]
                final=['']*len(convs_)
                last_valid=[None]*len(convs_)
                active=list(range(len(convs_)))
                for attempt in range(MAX_ROUNDS):
                    if not active or hclock.left()<3:
                        break
                    try:
                        replies=self.llm.chat([convs_[j] for j in active],max_new_tokens=384,max_time=max(1,hclock.left()-1),stop=['</call>'],temperature=temperature,batch_size=16)
                    except Exception:
                        break
                    pending=[]
                    for j,reply in zip(active,replies):
                        final[j]=reply
                        call=dsh.parse_tool_call(reply)
                        if call is not None:
                            obs=calc.observe(call)
                            match=re.fullmatch(r'Result: ([-+\deE.]+)',obs)
                            if match:last_valid[j]=float(match[1])
                            convs_[j].append({'role':'assistant','content':reply})
                            convs_[j].append({'role':'user','content':obs})
                            pending.append(j)
                    active=pending
                answers=[]
                for j in range(len(convs_)):
                    answer=dsh.tagged_number(final[j])
                    if answer is None:answer=last_valid[j]
                    if answer is None and not dsh.parse_tool_call(final[j]):answer=dsh.last_number(final[j])
                    answers.append(answer)
                return answers
 
            answers=[[a] for a in rounds(convs,0.0)]                         
            if hclock.left()>8:                                              
                sampled=rounds([c for c in convs for _ in range(N_SAMPLES)],SAMPLE_TEMP)
                for j in range(len(convs)):
                    answers[j]+=sampled[j*N_SAMPLES:(j+1)*N_SAMPLES]
            for j,i in enumerate(unresolved):
                answer=vote(answers[j])
                self.out[i]={'id':tasks[i]['id'],'answer':float(answer) if answer is not None else 0}

        ################# starting the jobs ###################################
        table_unresolved = tab.check_and_solve_infinite(tasks, table_tasks ,self.out)
        jobs = [(name, handler_fct, task_cat) for name, handler_fct, task_cat in (('fit', fit_llm, fit_tasks), ('calc', calc_llm, calc_tasks), ('table', table_llm, table_unresolved))]
        # calculate how many task we still have to do
        tasks_left = sum(len(idx) for _, _, idx in jobs)
        for name, handler_fct, task_cat in jobs:
            if not task_cat:
                continue 
            # calculate how much time is left for the amount of tasks left in that category/job
            budget = clock.left() * len(task_cat) / tasks_left
            tasks_left -= len(task_cat)
            # not enough time left to do more, better to give back 0 than to throw exception
            # but the deterministic fit part is fast and can still run (in fit, before starting the llm, the time is checked again)
            if budget < 2 and name != 'fit': 
                continue
            try:                                          
                handler_fct(task_cat, dsh.Clock(budget, margin=0.5))
            except Exception:
                report(name)

        return self.out
