import os
import io
import re
import math
import numpy as np
import operator
import pandas as pd
import dsh
import traceback

import table_solver as tab
import calc_solver as calc
import fit_solver as fit


########## ERRORS ##############################

def report(where):
    print(f'[agent] {where} failed:\n{traceback.format_exc()}')   


########## PROMPTS #####################################
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
MAX_ROUNDS = 4


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
        self.llm_table = self.llm
        self.llm_calc = self.llm
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

            prepared, context, conversations = {}, {}, {}
            for i in indices:
                try:
                    tables, df, info = tab.prepare_table_semantic(tasks[i])
                    enriched = tab.table_context(tasks[i], tables, df, info)
                    prepared[i] = (tables, df)
                    context[i] = enriched
                    conversations[i] = tab.prompt(tasks[i], enriched, df)
                except Exception as e:
                    print(f'[table prep] {tasks[i]["id"]}: {type(e).__name__}: {e}')

            # Qwen only needs to plan. Python performs the calculations.
            active = list(conversations)
            if active and hclock.left() >= 4:
                for attempt in range(2):
                    if not active or hclock.left() < 3:
                        break
                    try:
                        replies = self.llm.chat(
                            [conversations[i] for i in active],
                            max_new_tokens=220 if attempt == 0 else 180,
                            max_time=max(1, hclock.left() - (7 if attempt == 0 else 3)),
                            temperature=0
                        )
                    except Exception as e:
                        print(f'[table plan generation] {type(e).__name__}: {e}')
                        break
                    pending = []
                    for i, reply in zip(active, replies):
                        try:
                            plan = tab.parse(reply)
                            _, df = prepared[i]
                            tab.validate_semantics(tasks[i], plan, df)
                            answer = tab.execute(plan, df)
                            answer = tab.validate_table_result(tasks[i], answer)
                            self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
                            if os.environ.get('DSH_TABLE_DEBUG'):
                                print(f'[table plan accepted] {tasks[i]["id"]}: {plan} -> {answer}')
                        except Exception as e:
                            print(f'[table plan rejected] {tasks[i]["id"]}: {type(e).__name__}: {e}')
                            prev = reply[:500]
                            conversation = tab.prompt(tasks[i], context[i], prepared[i][1])
                            conversation.extend([
                                {'role':'assistant', 'content':prev},
                                {'role':'user', 'content': 'PLAN ERROR: ' + str(e)[:180]
                                 + '\nReturn a corrected SINGLE JSON plan. Use exact schema column codes. No Python.'}
                            ])
                            conversations[i] = conversation
                            pending.append(i)
                    active = pending

            # Preserve the older pandas-expression fallback as a last resort.
            # Only the remaining hard tasks consume extra decoding time.
            active = [i for i in indices if i in prepared and self.out[i] is None]
            if active and hclock.left() > 5:
                convs = {}
                for i in active:
                    convs[i] = ([{'role':'system','content':tab.SYSTEM_table}]
                                + tab.table_examples(tasks[i])
                                + [{'role':'user','content':'Question: '+tasks[i]['objective']
                                    +'\n\n'+context[i]}])
                for attempt in range(2):
                    if not active or hclock.left() < 3:
                        break
                    try:
                        replies = self.llm.chat(
                            [convs[i] for i in active],
                            max_new_tokens=200,
                            max_time=max(1,hclock.left()-(3 if attempt==0 else 1)),
                            temperature=0
                        )
                    except Exception as e:
                        print(f'[table pandas generation] {type(e).__name__}: {e}')
                        break
                    pending = []
                    for i, reply in zip(active, replies):
                        try:
                            tables, df = prepared[i]
                            result = tab.evaluate_table_expression(reply, tables, df)
                            tab.expression_intent(tasks[i], reply)
                            result = tab.validate_table_result(tasks[i], result)
                            self.out[i] = {'id':tasks[i]['id'], 'answer':result}
                            if os.environ.get('DSH_TABLE_DEBUG'):
                                print(f'[table pandas accepted] {tasks[i]["id"]}: {reply[:450]} -> {result}')
                        except Exception as e:
                            prompt = ('Question: ' + tasks[i]['objective'] + '\n\n' + context[i]
                                + '\nPrevious pandas expression: ' + reply[:400]
                                + '\nError: ' + str(e)[:160]
                                + '\nReturn a corrected single pandas expression.')
                            convs[i]=[{'role':'system','content':tab.SYSTEM_table},
                                      {'role':'user','content':prompt}]
                            pending.append(i)
                    active=pending

            for i in indices:
                if self.out[i] is None:
                    self.out[i]={'id':tasks[i]['id'],'answer':0}


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
            final=['']*len(unresolved)
            last_valid=[None]*len(unresolved)
            active=list(range(len(unresolved)))
            for attempt in range(MAX_ROUNDS):
                if not active or hclock.left()<3:
                    break
                try:
                    replies=self.llm.chat([convs[j] for j in active],max_new_tokens=384,max_time=max(1,hclock.left()-1),stop=['</call>'])
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
                        convs[j].append({'role':'assistant','content':reply})
                        convs[j].append({'role':'user','content':obs})
                        pending.append(j)
                active=pending
            for j,i in enumerate(unresolved):
                answer=dsh.tagged_number(final[j])
                if answer is None:answer=last_valid[j]
                if answer is None and not dsh.parse_tool_call(final[j]):answer=dsh.last_number(final[j])
                self.out[i]={'id':tasks[i]['id'],'answer':float(answer) if answer is not None else 0}

        ################# deterministic first; model time to real unsolved work #################
        table_unresolved = []
        for i in table_tasks:
            try:
                answer = float(tab.solve_data(tasks[i]))
                if not math.isfinite(answer):
                    raise ValueError('Non-finite table answer')
                self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
            except Exception as error:
                table_unresolved.append(i)
                print(f'[table deterministic] {tasks[i]["id"]}: {type(error).__name__}: {str(error)[:130]}')
        fit_unresolved = []
        for i in fit_tasks:
            try:
                answer = float(fit.solve_fit_task(tasks[i]))
                if not math.isfinite(answer):
                    raise ValueError('Non-finite fit answer')
                self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
            except Exception:
                fit_unresolved.append(i)
        calc_unresolved = []
        for i in calc_tasks:
            try:
                answer = float(calc.calc_general(tasks[i]))
                if not math.isfinite(answer):
                    raise ValueError('Non-finite calc answer')
                self.out[i] = {'id': tasks[i]['id'], 'answer': answer}
            except Exception:
                calc_unresolved.append(i)

        # In the old version the time was allocated by the number of ALL tasks,
        # including easy fit/calc tasks which had already been solved. This gave
        # many table batches too little time for Qwen to finish an expression.
        if table_unresolved:
            reserve = (5 if fit_unresolved else 0) + (5 if calc_unresolved else 0)
            budget = max(0, clock.left() - reserve - 0.7)
            if budget >= 3:
                try:
                    table_llm(table_unresolved, dsh.Clock(budget, margin=0))
                except Exception:
                    report('table')

        if fit_unresolved and clock.left() > 2.5:
            try:
                fit_llm(fit_unresolved, dsh.Clock(max(1, clock.left() - (4 if calc_unresolved else 0)), margin=0))
            except Exception:
                report('fit')

        if calc_unresolved and clock.left() > 2.5:
            try:
                calc_llm(calc_unresolved, dsh.Clock(clock.left(), margin=0))
            except Exception:
                report('calc')

        return self.out
