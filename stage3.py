import os
import re
import math
import dsh
import table_solver as tab
import calc_solver as calc
import fit_solver as fit

MODEL = os.environ.get('DSH_MODEL', dsh.DEFAULT_MODEL)

MAX_ROUNDS = 4

SYSTEM_calc = (
    "Solve the numerical problem carefully. You can call <call tool=\"calculator\">EXPRESSION</call> "
    "with arithmetic and functions sqrt, log, exp, comb, perm, factorial, gcd, lcm, "
    "mean, median, stdev, pstdev, binom_pmf, binom_ge, binom_le, hypergeom, "
    "normal_cdf, bayes, days_between, date_add, weekday. "
    "The arguments to probability functions are probabilities between 0 and 1, not percentages. "
    "Decide whether the requested answer is a fraction or percentage, and whether a date range is inclusive. "
    "Compute in the tool rather than mental arithmetic. After a tool result return ANSWER: <number>. "
    "If an expression fails, correct it. Never output an unexplained guess."
)

DEMO_calc = [
    {'role':'user','content':'A price is 80 after a 20% reduction. What was the original price?'},
    {'role':'assistant','content':'<call tool="calculator">80/(1-0.20)</call>'},
    {'role':'user','content':'Result: 100'},
    {'role':'assistant','content':'ANSWER: 100'},
    {'role':'user','content':'A binomial random variable has n=12 and p=0.2. What is the probability of at least 3 successes?'},
    {'role':'assistant','content':'<call tool="calculator">binom_ge(12,0.2,3)</call>'},
    {'role':'user','content':'Result: 0.44165425152'},
    {'role':'assistant','content':'ANSWER: 0.44165425152'},
    {'role':'user','content':'A test has prevalence 2%, sensitivity 95% and false positive rate 3%. What is the probability of disease given a positive test, in percent?'},
    {'role':'assistant','content':'<call tool="calculator">100*bayes(0.02,0.95,0.03)</call>'},
    {'role':'user','content':'Result: 39.2561983471'},
    {'role':'assistant','content':'ANSWER: 39.2561983471'}
]

SYSTEM_table = (
    "You solve numerical CSV questions by using only the provided pandas DataFrames. "
    "Write the shortest correct pandas expression producing ONE number. "
    "df contains the main file joined to lookup tables when possible; tables contains original CSVs. "
    "The schema gives real names, semantic aliases, category codes, decoded meaning_* labels and example values. "
    "Use column names that appear in that schema. You may use meaning_* columns for named categories. "
    "Do not invent a column, category, row or number. Some narrative facts are irrelevant distractors. "
    "Apply every condition, including the correct denominator for percentages and exclusions of missing observations. "
    "For a rank, aggregate once per group and rank the group-level results descending with method='min'. "
    "For a derived ratio divide row by row before aggregating; ignore infinite and missing values. "
    "When two tables are joined, use the lookup fields only after matching the correct join key. "
    "Use df.loc[df['col'] > threshold, 'target'] rather than chained Boolean indexing. "
    "Prefer direct .mean(), .median(), .sum(), .max(), .min(), .count(), .nunique() or len. "
    "Return just one Python expression. No explanation, Markdown or imports. "
    "Supported names: df, tables, pd, np, len, round, abs, sum, min, max."
)
DEMO_table = [
        ('Question: What is the mean salary for staff older than 30? Columns: age, salary',"df.loc[df['age']>30,'salary'].mean()"),
        ('Question: What percentage of orders in region East have a recorded price? Columns: region, price',"100*df.loc[df['region']=='East','price'].notna().mean()"),
        ('Question: What is the largest total revenue among regions? Columns: region, revenue',"df.groupby('region')['revenue'].sum().max()"),
        ('Question: What is the rank of East by total revenue per region? Columns: region, revenue',"df.groupby('region')['revenue'].sum().rank(ascending=False,method='min').loc['East']"),
        ('Question: What is the median revenue per item? Columns: revenue, items',"(df['revenue']/df['items']).replace([np.inf,-np.inf],np.nan).median()"),
        ('Question: How many observations belong to a department with floor at least 2? Columns: dept, floor',"len(df.loc[df['floor']>=2])"),
        ('Question: What is the mean floor by metro line, taking the largest group mean? Columns: metro, floor',"df.groupby('metro')['floor'].mean().max()")
    ]

SYSTEM_fit = ()
DEMO_fit = []



def table_examples(task):
    question=task['objective'].lower()
    examples=DEMO_table
    if 'rank' in question:
        selections=(2,3)
    elif re.search(r'per (?:item|day|person|kilometre)|ratio|divided by',question):
        selections=(0,4)
    elif re.search(r'for each|group|by the',question):
        selections=(2,6)
    elif re.search(r'how many|count|belong',question):
        selections=(1,5)
    elif re.search(r'fraction|percentage|percent',question):
        selections=(0,1)
    else:
        selections=(0,2)
    messages=[]
    for index in selections:
        prompt,answer=examples[index]
        messages.extend(({'role':'user','content':prompt},{'role':'assistant','content':answer}))
    return messages

def route_tasks(tasks):
    calc_files=[]
    fit_files=[]
    table_files=[]
    for index,task in enumerate(tasks):
        files=task.get('files',{})
        names=list(files) if isinstance(files,dict) else [os.path.basename(str(file)) for file in files]
        if not names:
            calc_files.append(index)
        elif 'train.csv' in names or (task.get('type') or '').startswith('fit.'):
            fit_files.append(index)
        else:
            table_files.append(index)
    return calc_files,fit_files,table_files

def check_infinite(tasks,indices,out):
    remaining=[]
    for i in indices:
        try:
            try:
                answer=float(tab.solve_data(tasks[i]))
            except Exception:
                if (tasks[i].get('type') or '').startswith('table.') or (tasks[i].get('files') and 'train.csv' not in tasks[i]['files']):
                    answer=float(tab.solve_table_semantic(tasks[i]))
                else:
                    raise
            if math.isfinite(answer):
                out[i]={'id':tasks[i]['id'],'answer':answer}
                continue
        except Exception:
            pass
        remaining.append(i)
    return remaining

class Agent:
    def __init__(self):
        self.llm_calc=None
        self.llm_fit=None
        self.llm_table=None
        self.out=[]

    def solve(self,tasks,seconds=40.0):
        clock=dsh.Clock(seconds)
        self.out=[None]*len(tasks)
        calc_indices,fit_indices,table_indices=route_tasks(tasks)
        fit_unresolved=[]
        for i in fit_indices:
            try:
                answer=float(fit.solve_fit(tasks[i]))
                if not math.isfinite(answer):
                    raise ValueError("Non-finite fit")
                self.out[i]={'id':tasks[i]['id'],'answer':answer}
            except Exception:
                fit_unresolved.append(i)
        table_unresolved=check_infinite(tasks,table_indices,self.out)

        def fit_llm(indices):
            pass

        def table_llm(indices):
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
                convs.append([{'role':'system','content':SYSTEM_table}]+table_examples(task)+[{'role':'user','content':'Question: '+task['objective']+'\n\n'+context}])
            active=[j for j,i in enumerate(indices) if prepared.get(i) is not None]
            if not active:
                return
            if clock.left()>3:
                try:
                    self.llm_table=self.llm_table or self.llm_calc or dsh.load_llm(MODEL)
                except Exception:
                    self.llm_table=None
            for attempt in range(3):
                if not active or self.llm_table is None or clock.left()<3:
                    break
                try:
                    replies=self.llm_table.chat([convs[j] for j in active],max_new_tokens=320,max_time=max(1,clock.left()-1))
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
                        convs[j]=[{'role':'system','content':SYSTEM_table}]+[{'role':'user','content':(
                            original_context+'\n\nEarlier attempt failed: '+type(error).__name__+': '+str(error)[:250]+'. '
                            'Do not repeat this expression: '+reply[:240]+'. '
                            'Use exactly one simple scalar pandas expression. Available df columns: '+repr(columns)
                        )}]
                        pending.append(j)
                active=pending
            for i in indices:
                if self.out[i] is None:
                    self.out[i]={'id':tasks[i]['id'],'answer':0}

        def calc_llm(indices):
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
            try:
                self.llm_calc=self.llm_calc or self.llm_table or dsh.load_llm(MODEL)
            except Exception:
                for i in unresolved:self.out[i]={'id':tasks[i]['id'],'answer':0}
                return
            convs=[]
            for i in unresolved:
                convs.append([{'role':'system','content':SYSTEM_calc}]+DEMO_calc+[{'role':'user','content':tasks[i]['objective']}])
            final=['']*len(unresolved)
            last_valid=[None]*len(unresolved)
            active=list(range(len(unresolved)))
            for attempt in range(MAX_ROUNDS):
                if not active or clock.left()<3:
                    break
                try:
                    replies=self.llm_calc.chat([convs[j] for j in active],max_new_tokens=384,max_time=max(1,clock.left()-1),stop=['</call>'])
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

        table_llm(table_unresolved)
        calc_llm(calc_indices)
        fit_llm(fit_unresolved)
        
        for i,task in enumerate(tasks):
            if self.out[i] is None:
                self.out[i]={'id':task['id'],'answer':0}
                
        return self.out
