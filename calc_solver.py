import re
import math
import ast
import datetime as dt
import operator
import statistics

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



def binom_pmf(n,p,k):
    return math.comb(int(n),int(k))*float(p)**int(k)*(1-float(p))**(int(n)-int(k)) if 0<=k<=n else 0.0

def binom_ge(n,p,k):
    return sum(binom_pmf(n,p,j) for j in range(int(k),int(n)+1))

def binom_le(n,p,k):
    return sum(binom_pmf(n,p,j) for j in range(0,int(k)+1))

def dice_sum(n,s,faces=6):
    n=int(n);s=int(s);faces=int(faces)
    if n>30 or faces>100 or n<1 or faces<2:raise ValueError('Unsupported dice parameters')
    ways=[1]
    for iteration in range(n):
        nxt=[0]*(len(ways)+faces)
        for i,count in enumerate(ways):
            for face in range(1,faces+1):nxt[i+face]+=count
        ways=nxt
    return ways[s]/faces**n if 0<=s<len(ways) else 0.0

def days_between(a,b,include_both=0):
    a=parse_date(str(int(a))) if isinstance(a,(int,float)) else parse_date(a)
    b=parse_date(str(int(b))) if isinstance(b,(int,float)) else parse_date(b)
    return (b-a).days+int(include_both)

def date_add(a,days):
    a=parse_date(str(int(a))) if isinstance(a,(int,float)) else parse_date(a)
    return int((a+dt.timedelta(days=int(days))).strftime('%Y%m%d'))

def weekday(a):
    a=parse_date(str(int(a))) if isinstance(a,(int,float)) else parse_date(a)
    return a.isoweekday()

def parse_date(value):
    text=str(value).strip(' ,.?!')
    if re.fullmatch(r'\d{8}',text):return dt.datetime.strptime(text,'%Y%m%d').date()
    for fmt in ('%Y-%m-%d','%d %B %Y','%B %d, %Y','%B %d %Y','%d %b %Y','%b %d, %Y','%b %d %Y'):
        try:return dt.datetime.strptime(text,fmt).date()
        except ValueError:pass
    raise ValueError('Date not recognized: '+text)

MATH_OPERATIONS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod,ast.Pow:operator.pow}

def hypergeom(population, successes, draws, observed):
    population, successes, draws, observed = map(int, (population, successes, draws, observed))
    if not (0 <= successes <= population and 0 <= draws <= population):
        raise ValueError('Invalid hypergeometric parameters')
    if observed < max(0, draws - (population - successes)) or observed > min(draws, successes):
        return 0.0
    return math.comb(successes, observed) * math.comb(population - successes, draws - observed) / math.comb(population, draws)

def normal_cdf(value, mean=0, std=1):
    if std <= 0:
        raise ValueError('Standard deviation must be positive')
    return 0.5 * (1 + math.erf((value - mean) / (std * math.sqrt(2))))

def bayes(prior, sensitivity, false_positive):
    positive = prior * sensitivity + (1 - prior) * false_positive
    if positive == 0:
        raise ValueError('Zero probability conditioning event')
    return prior * sensitivity / positive

MATH_FUNCTIONS={'sqrt':math.sqrt,'exp':math.exp,'log':math.log,'log10':math.log10,'log1p':math.log1p,'abs':abs,'round':round,'ceil':math.ceil,'floor':math.floor,'factorial':math.factorial,'comb':math.comb,'perm':math.perm,'gcd':math.gcd,'lcm':math.lcm,'sum':sum,'max':max,'min':min,'mean':statistics.mean,'median':statistics.median,'pstdev':statistics.pstdev,'stdev':statistics.stdev,'binom_pmf':binom_pmf,'binom_ge':binom_ge,'binom_le':binom_le,'dice_sum':dice_sum,'days_between':days_between,'date_add':date_add,'weekday':weekday,'hypergeom':hypergeom,'normal_cdf':normal_cdf,'bayes':bayes}

def math_calculator(expr):
    if len(expr)>900:raise ValueError('Expression too long')
    tree=ast.parse(expr.strip(),mode='eval')
    if len(list(ast.walk(tree)))>150:raise ValueError('Expression too complex')
    def evaluate(node,depth=0):
        if depth>32:raise ValueError('Expression too deep')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float,str):return node.value
        if isinstance(node,(ast.List,ast.Tuple)):
            if len(node.elts)>300:raise ValueError('Too many elements')
            return [evaluate(x,depth+1) for x in node.elts]
        if isinstance(node,ast.UnaryOp) and type(node.op) in (ast.UAdd,ast.USub):
            x=evaluate(node.operand,depth+1);return x if isinstance(node.op,ast.UAdd) else -x
        if isinstance(node,ast.BinOp) and type(node.op) in MATH_OPERATIONS:
            a=evaluate(node.left,depth+1);b=evaluate(node.right,depth+1)
            if type(node.op)==ast.Pow and (abs(float(b))>1000 or abs(float(a))>1e100):raise ValueError('Power exceeds limits')
            return MATH_OPERATIONS[type(node.op)](a,b)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in MATH_FUNCTIONS and not node.keywords:
            return MATH_FUNCTIONS[node.func.id](*[evaluate(x,depth+1) for x in node.args])
        if isinstance(node,ast.Name) and node.id in ('pi','e'):return math.pi if node.id=='pi' else math.e
        raise ValueError('Unsupported mathematical expression')
    result=evaluate(tree.body)
    if isinstance(result,bool) or not isinstance(result,(int,float)) or not math.isfinite(float(result)):
        raise ValueError('Expected finite number')
    return float(result)

NUMBER_PATTERN=r'-?\d+(?:\.\d+)?'

DATE_PATTERN=r'(?:\d{4}-\d\d-\d\d|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{1,2},?\s+\d{4})'

def number_list(q):
    candidates=re.findall(r'(?<![\d.])-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,}',q)
    if not candidates:return None
    best=max(candidates,key=lambda x:len(x))
    return [float(x.strip()) for x in best.split(',')]

def solve_inline(q):
    if 'weighted mean' in q and 'coefficient' in q:
        pairs=re.findall(r'(-?\d+(?:\.\d+)?)\s+(?:in|for)\s+[^()]*?\(coefficient\s+(\d+)\)',q)
        if len(pairs)>=2:
            w=[(float(x),int(y)) for x,y in pairs]
            return sum(x*y for x,y in w)/sum(y for x,y in w)
    values=number_list(q)
    if values:
        if 'strictly above' in q or 'greater than' in q:
            m=re.search(r'(?:strictly above|greater than)\s+('+NUMBER_PATTERN+r')',q)
            if m:return sum(x>float(m[1]) for x in values)
        if re.search(r'\b(?:population )?standard deviation\b',q):return statistics.pstdev(values) if 'population' in q or 'divide by n' in q else statistics.stdev(values)
        if 'median' in q:return statistics.median(values)
        if 'mean' in q or 'average' in q:return statistics.mean(values)
        if 'total' in q or 'sum' in q:return sum(values)
    return None

def solve_dates(q):
    md=re.search(r'('+DATE_PATTERN+r')\s+(?:to|and ends on)\s+('+DATE_PATTERN+r')',q)
    if md and ('days' in q or 'long' in q):return days_between(md[1],md[2],'count both' in q)
    md=re.search(r'(\d+)\s+days after\s+('+DATE_PATTERN+r')',q)
    if md:return date_add(md[2],int(md[1]))
    md=re.search(r'(?:day of the week does|which day of the week does)\s+('+DATE_PATTERN+r')',q)
    if md:return weekday(md[1])
    return None

def solve_probability(q):
    pairs=re.findall(r'(-?\d+(?:\.\d+)?)\s+(?:euros|dollars|\$)\s+with probability\s+(\d+(?:\.\d+)?)%',q)
    if len(pairs)>=2 and 'expected' in q and abs(sum(float(p) for amount,p in pairs)-100)<0.001:
        return sum(float(x)*float(p)/100 for x,p in pairs)
    if 'dice' in q and 'add up to' in q:
        n=re.search(r'(\d+)\s+fair\s+(\w+)-sided dice',q)
        total=re.search(r'add up to\s+(\d+)',q)
        faces={'six':6,'four':4,'eight':8,'ten':10,'twelve':12,'twenty':20}
        if n and total:
            size=int(n[2]) if n[2].isdigit() else faces.get(n[2])
            if size:return dice_sum(int(n[1]),int(total[1]),size)
    if 'probability' in q and ('defective' in q or 'success' in q or 'fails' in q):
        prob=re.search(r'probability\s+('+NUMBER_PATTERN+r')',q)
        count=re.search(r'(?:among|out of|in)\s+(\d+)\s+(?:parts|trials|items|attempts)',q)
        threshold=re.search(r'(at least|at most|exactly)\s+(\d+)\s+(?:defective|success|fail)',q)
        if prob and count and threshold:
            p=float(prob[1]);n=int(count[1]);k=int(threshold[2])
            if threshold[1]=='at least':return binom_ge(n,p,k)
            if threshold[1]=='at most':return binom_le(n,p,k)
            return binom_pmf(n,p,k)
    return None

def solve_numbers(q):
    cleaned=q.replace('×','*').replace('−','-').replace('÷','/').replace('√','sqrt ')
    m=re.search(r'\b(?:compute|what is)\s+(.+?)(?:\?|\.(?!\d)|\s+(?:give|the answer|round|report))',cleaned)
    if m:
        expr=m[1].replace('minus','-').replace('plus','+').replace('times','*').replace('divided by','/')
        if re.fullmatch(r'[0-9\s.()+*/%-]+',expr) and re.search(r'[+*/%-]',expr):
            try:return math_calculator(expr)
            except (ValueError,ZeroDivisionError,OverflowError,SyntaxError):pass
    m=re.search(r'(?:sqrt\s*|square root of\s*)('+NUMBER_PATTERN+r')',cleaned)
    if m:return math.sqrt(float(m[1]))
    m=re.search(r'(?:base-10 logarithm|logarithm in base 10|log10) of\s+('+NUMBER_PATTERN+r')',q)
    if m:return math.log10(float(m[1]))
    m=re.search(r'(?:natural logarithm|logarithm in base e|ln) of\s+('+NUMBER_PATTERN+r')',q)
    if m:return math.log(float(m[1]))
    m=re.search(r'e raised to the power\s+('+NUMBER_PATTERN+r')',q)
    if m:return math.exp(float(m[1]))
    m=re.search(r'('+NUMBER_PATTERN+r')\s+raised to the power\s+(\d+)(?:\s*/\s*(\d+))?',q)
    if m:return float(m[1])**(int(m[2])/int(m[3]) if m[3] else int(m[2]))
    m=re.search(r'(?:least common multiple|lcm) of\s+(\d+)\s+and\s+(\d+)',q)
    if m:return math.lcm(int(m[1]),int(m[2]))
    m=re.search(r'(?:remainder|modulo) of (?:the division of\s+)?(\d+) by (\d+)',q)
    if m:return int(m[1])%int(m[2])
    if 'leave together' in q or 'simultaneously' in q:
        periods=re.findall(r'\bevery\s+(\d+)\s+(?:minutes|hours|seconds|days)\b',q)
        if len(periods)>=2:return math.lcm(*[int(x) for x in periods])
    m=re.search(r'ordered arrangements of (\d+) items.*?from (\d+)',q)
    if m:return math.perm(int(m[2]),int(m[1]))
    m=re.search(r'team of (\d+).*?from (\d+)',q)
    if m and ('order does not matter' in q or 'chosen' in q):return math.comb(int(m[2]),int(m[1]))
    return None

def solve_growth(q):
    match = re.search(r'(?:grew|went|increased|decreased) from\s+(' + NUMBER_PATTERN + r')\s+to\s+(' + NUMBER_PATTERN + r')', q)
    if match:
        old, new = map(float, match.groups())
        years = re.search(r'\bin\s+(\d+)\s+years\b', q)
        if years and any(word in q for word in ('annual', 'yearly', 'constant')):
            return 100 * ((new / old) ** (1 / int(years[1])) - 1)
        if 'percent' in q:
            return 100 * (new / old - 1)
    if any(term in q for term in ('original price', 'before tax', 'before vat', 'amount before vat')):
        money = re.search(r'([\d,.]+)\s+(?:euros|dollars)', q)
        rates = re.findall(r'(' + NUMBER_PATTERN + r')%', q)
        if money and len(rates) == 1:
            return float(money[1].replace(',', '')) / (1 + float(rates[0]) / 100)
    changes = re.findall(r'\b(rises?|falls?|increases?|decreases?) by\s+(' + NUMBER_PATTERN + r')%', q)
    if changes and 'overall' in q:
        factors = [1 + (-1 if action.startswith(('fall', 'decrease')) else 1) * float(percent) / 100 for action, percent in changes]
        return 100 * (math.prod(factors) - 1)
    match = re.search(r'([\d,]+)\s+euros.*?(' + NUMBER_PATTERN + r')%\s+(?:compound interest|per year|every year|of its value every year).*?after\s+(\d+)\s+years', q)
    if match:
        initial, rate, years = match.groups()
        rate = float(rate) / 100 * (-1 if 'loses' in q else 1)
        return float(initial.replace(',', '')) * (1 + rate) ** int(years)
    return None

def solve_general_word_math(q):
    match = re.search(r'\b(?:solve|find) the system\s+(.+?)\s+and\s+(.+?)(?:\.|\?)\s*what is the value of\s+([xy])', q)
    if match:
        rows=[]
        for equation in (match[1],match[2]):
            sides=equation.replace('−','-').split('=')
            if len(sides)!=2:
                return None
            left=sides[0].replace(' ','').replace('-','+-')
            terms=re.findall(r'([+-]?(?:\d+(?:\.\d+)?)?)\*?([xy])',left)
            if not terms:
                return None
            coefficient={'x':0.0,'y':0.0}
            for amount,variable in terms:
                coefficient[variable]+=float(amount) if amount not in ('','+','-') else (-1 if amount=='-' else 1)
            rows.append((coefficient['x'],coefficient['y'],float(sides[1])))
        a,b,c=rows[0]
        d,e,f=rows[1]
        determinant=a*e-b*d
        if determinant==0:
            return None
        return (c*e-b*f)/determinant if match[3]=='x' else (a*f-c*d)/determinant

    if re.search(r'\b(?:pump|tap|pipe|leak|drain|fill|empty)\b',q) and 'hour' in q:
        filling=re.findall(r'\bfills?\b.{0,50}?\bin\s+(\d+(?:\.\d+)?)\s+hours?',q)
        draining=re.findall(r'\bempt(?:y|ies|ied)\b.{0,50}?\bin\s+(\d+(?:\.\d+)?)\s+hours?',q)
        if filling and draining and 'how many hours' in q:
            net=sum(1/float(x) for x in filling)-sum(1/float(x) for x in draining)
            if net>0:
                return 1/net

    if re.search(r'\b(?:employs|employees|workers)\b',q) and 'hours a week' in q:
        workers=re.search(r'\bemploys\s+(\d+)\s+(?:people|employees|workers)',q)
        hours=re.search(r'(\d+(?:\.\d+)?)\s+hours a week at\s+(\d+(?:\.\d+)?)\s+(?:euros?|dollars?) an hour',q)
        weeks=re.search(r'\bover\s+(\d+)\s+weeks',q)
        bonuses=re.search(r'\bbonus of\s+(\d+(?:\.\d+)?)\s+(?:euros?|dollars?)',q)
        if all((workers,hours,weeks)) and 'total paid' in q:
            return int(workers[1])*(float(hours[1])*float(hours[2])*int(weeks[1])+(float(bonuses[1]) if bonuses else 0))

    conversion=re.search(r'\b1\s+([a-z]+)\s*=\s*(\d+(?:\.\d+)?)\s+([a-z]+)\b',q)
    desired=re.search(r'\bconvert\s+(\d+(?:\.\d+)?)\s+([a-z]+)\s+to\s+([a-z]+)',q)
    if conversion and desired:
        units={'feet':'foot','inches':'inch','metres':'metre','meters':'metre','m':'metre','kilometres':'kilometre','kilometers':'kilometre','km':'kilometre'}
        def standard(unit):
            return units.get(unit,unit[:-1] if unit.endswith('s') else unit)
        from_unit,to_unit=standard(conversion[1]),standard(conversion[3])
        source,target=standard(desired[2]),standard(desired[3])
        if (source,target)==(from_unit,to_unit):
            return float(desired[1])*float(conversion[2])
        if (source,target)==(to_unit,from_unit):
            return float(desired[1])/float(conversion[2])
    return None

def solve_general_probability(q):
    if re.search(r'\b(?:test|screening)\b',q) and re.search(r'\b(?:detects|sensitivity|wrongly flags|false positive)\b',q) and 'positive' in q:
        percentages=[float(p)/100 for p in re.findall(r'(\d+(?:\.\d+)?)\s*%',q)]
        if len(percentages)>=3:
            prior,sensitivity,false_positive=percentages[:3]
            denominator=prior*sensitivity+(1-prior)*false_positive
            if denominator>0 and all(0<=p<=1 for p in (prior,sensitivity,false_positive)):
                probability=prior*sensitivity/denominator
                return 100*probability if re.search(r'\bin percent\b|\bas a percent\b',q) else probability
    return None

def solve_general_growth(q):
    threshold=re.search(r'\b(?:reach|exceed|surpass)\s+(?:at least\s+)?([\d,]+(?:\.\d+)?)\s+(?:euros?|dollars?)',q)
    if not threshold or not re.search(r'\bhow many\s+(?:full\s+)?years\b',q):
        return None
    starting=re.search(r'\b(?:amount of|starts? (?:at|with)|initial (?:amount|balance) of|principal of)\s+([\d,]+(?:\.\d+)?)\s+(?:euros?|dollars?)',q)
    rate=re.search(r'(\d+(?:\.\d+)?)\s*%\s+(?:compound interest|per year|annual interest)',q)
    if not starting or not rate:
        return None
    principal=float(starting[1].replace(',',''))
    target=float(threshold[1].replace(',',''))
    interest=float(rate[1])/100
    if principal<=0 or interest<=0:
        return None
    if principal>=target:
        return 0
    years=max(0,math.ceil(math.log(target/principal)/math.log1p(interest)-1e-10))
    while principal*(1+interest)**years<target*(1-1e-12):
        years+=1
    return float(years)

def calc_general(task):
    q=task['objective'].lower().replace('−','-').replace('×','*').replace('÷','/').replace('√','sqrt ')
    for func in (solve_general_word_math,solve_general_probability,solve_general_growth,solve_dates,solve_probability,solve_inline,solve_growth,solve_numbers):
        try:
            value=func(q)
            if value is not None and math.isfinite(float(value)):
                return float(value)
        except (ValueError,OverflowError,ZeroDivisionError,ArithmeticError):pass
    raise ValueError('Question requires reasoning or unsupported operation')

def observe(call):
    name, arg = call
    if name != 'calculator':
        return f'Error: unknown tool {name!r}'
    try:
        value = math_calculator(arg.replace('math.', ''))
        return f'Result: {value:.12g}'
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError, OverflowError) as e:
        return f'Error: {type(e).__name__}: {e}'
