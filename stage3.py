import os
import io
import re
import math
import numpy as np
import operator
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import dsh
import traceback

import table_solver as tab
import calc_solver as calc
import fit_solver as fit


########## ERRORS ##############################

def report(where):
    print(f'[agent] {where} failed:\n{traceback.format_exc()}')   


########## PROMPTS #####################################
MODEL = os.environ.get('DSH_MODEL', dsh.DEFAULT_MODEL)
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
            prepared={}
            convs=[]
            contexts={}
            for i in indices:
                task=tasks[i]
                try:
                    tables,df,context=tab.prepare_table_semantic(task)
                except Exception as error:
                    tables=tab.read_files(task['files'],task['objective'])
                    names=[name for name in tables if name!='data_dictionary.csv']
                    if not names:
                        self.out[i]={'id':task['id'],'answer':0}
                        prepared[i]=None
                        convs.append(None)
                        continue
                    df=tables[names[0]]
                    context='Preparation error: '+str(error)+'; data files: '+str({name:list(frame.columns) for name,frame in tables.items()})
                prepared[i]=(tables,df)
                convs.append([{'role':'system','content':tab.SYSTEM_table}]+tab.table_examples(task)+[{'role':'user','content':'Question: '+task['objective']+'\n\n'+context}])
            active=[j for j,i in enumerate(indices) if prepared.get(i) is not None]
            if not active:
                return
            for attempt in range(3):
                if not active or self.llm is None or clock.left()<3:
                    break
                try:
                    replies=self.llm.chat([convs[j] for j in active],max_new_tokens=320,max_time=max(1,hclock.left()-1))
                except Exception:
                    break
                pending=[]
                for j,reply in zip(active,replies):
                    i=indices[j]
                    try:
                        tables,df=prepared[i]
                        answer=tab.evaluate_table_expression(reply,tables,df)
                        answer=tab.validate_table_result(tasks[i],answer)
                        self.out[i]={'id':tasks[i]['id'],'answer':answer}
                    except Exception as error:
                        columns=list(prepared[i][1].columns)
                        original_question='Question: '+tasks[i]['objective']
                        original_context=convs[j][-1]['content'] if attempt==0 else contexts[j]
                        contexts[j]=original_context
                        convs[j]=[{'role':'system','content':tab.SYSTEM_table}]+[{'role':'user','content':(
                            original_context+'\n\nEarlier attempt failed: '+type(error).__name__+': '+str(error)[:250]+'. '
                            'Do not repeat this expression: '+reply[:240]+'. '
                            'Use exactly one simple scalar pandas expression. Available df columns: '+repr(columns)
                        )}]
                        pending.append(j)
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
            if clock.left()<3:
                for i in unresolved:self.out[i]={'id':tasks[i]['id'],'answer':0}
                return
            convs=[]
            for i in unresolved:
                convs.append([{'role':'system','content':calc.SYSTEM_calc}]+calc.DEMO_calc+[{'role':'user','content':tasks[i]['objective']}])
            final=['']*len(unresolved)
            last_valid=[None]*len(unresolved)
            active=list(range(len(unresolved)))
            for attempt in range(MAX_ROUNDS):
                if not active or clock.left()<3:
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
