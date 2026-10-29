import os
import io
import re
import math
import ast
import numpy as np
import pandas as pd
import fit_solver as fit

SYSTEM_table = (
    "You are a data analyst translating one numerical CSV question into one executable pandas expression. "
    "You MUST calculate using the actual DataFrame; never guess a number. "
    "Return ONE single Python expression only, no commentary, code fence, import, print or ANSWER label. "
    "The DataFrame df is the observation-level table, with a verified left join to lookup tables when available. "
    "tables is a mapping from CSV filename to its original DataFrame. "
    "Use ONLY real column names from the schema; descriptions are semantic names, not Python column names. "
    "When category values are coded, filter using exact codes or provided meaning_* decoded columns. "
    "A question about readings AT sites is a filter on the joined site attributes. "
    "Some narrative facts are irrelevant distractors; do not turn them into filters. "  
    "Follow every qualifying condition. WHERE/whose/at least/below/between/after/during restrict the dataset BEFORE aggregation. "
    "If the question asks the share WITHIN a filtered group, first select that group, then calculate its fraction. "
    "For percentage multiply by 100; for fraction keep 0..1. Missing denominators are excluded as specified. "
    "For rank, aggregate per group before applying rank(ascending=False,method='min'). "
    "For a largest or second-largest GROUP statistic, group first and only then take max or nlargest. "
    "Distinguish mean of row-level ratios from ratio of column totals. For per-row ratios exclude zero and missing denominators. "
    "For weighted mean use sum(value*weight)/sum(valid weights), excluding rows with missing values or weights. "
    "For calendar filters convert to datetime and compare year/dates. "
    "Available names: df, tables, pd, np, len, round, abs, sum, min, max, float, int. "
    "Prefer pandas .loc, .groupby, .mean, .median, .sum, .count, .nunique, .quantile and .corr."
)

DEMO_table = [
        ('Question: What is the mean salary for staff older than 30? Columns: age, salary',"df.loc[df['age']>30,'salary'].mean()"),
        ('Question: What percentage of orders in region East have a recorded price? Columns: region, price',"100*df.loc[df['region']=='East','price'].notna().mean()"),
        ('Question: What is the largest total revenue among regions? Columns: region, revenue',"df.groupby('region')['revenue'].sum().max()"),
        ('Question: What is the rank of East by total revenue per region? Columns: region, revenue',"df.groupby('region')['revenue'].sum().rank(ascending=False,method='min').loc['East']"),
        ('Question: What is the median revenue per item? Columns: revenue, items',"(df['revenue']/df['items']).replace([np.inf,-np.inf],np.nan).median()"),
        ('Question: How many observations belong to a department with floor at least 2? Columns: dept, floor',"len(df.loc[df['floor']>=2])"),
        ('Question: What is the mean floor by metro line, taking the largest group mean? Columns: metro, floor',"df.groupby('metro')['floor'].mean().max()")
        ,('Question: What is the mean amount for records linked to a group with level at least 2? df has already been joined; columns: amount, level', "df.loc[df['level']>=2,'amount'].mean()")
        ,('Question: What fraction of records in group A have a recorded score? Columns: group, score', "df.loc[df['group']=='A','score'].notna().mean()")
        ,('Question: What is the mean amount for records dated on or after March 1, 2022? Columns: date, amount', "df.loc[pd.to_datetime(df['date'],errors='coerce')>=pd.Timestamp('2022-03-01'),'amount'].mean()")
        ,('Question: What is the 75th percentile of speed? Columns: speed', "df['speed'].quantile(0.75)")
        ,('Question: What is the population standard deviation of speed? Columns: speed', "df['speed'].std(ddof=0)")
    ]

DEMO_table += [
    ('Question: What is the Pearson correlation between income and expenses? Columns: income, expenses', "df['income'].corr(df['expenses'])"),
    ('Question: How many distinct customer IDs are there? Columns: customer_id', "df['customer_id'].nunique()"),
    ('Question: What is the 25th percentile of the weight? Columns: weight', "df['weight'].quantile(0.25)"),
    ('Question: What is the population variance of temperature? Columns: temp', "df['temp'].var(ddof=0)"),
    ('Question: How many distinct regions have price above 40? Columns: region, price', "df.loc[df['price'] > 40, 'region'].nunique()")
]

DEMO_table.extend([
    ('Question: Among rows with quantity over 4, what percentage has recorded cost? Columns: qty, cost',
     "100*df.loc[df['qty']>4,'cost'].notna().mean()"),
    ('Question: What is the mean amount for rows dated in 2024? Columns: day, amount',
     "df.loc[pd.to_datetime(df['day'],errors='coerce').dt.year==2024,'amount'].mean()"),
    ('Question: What is the second-largest sum of amount by region? Columns: region, amount',
     "df.groupby('region')['amount'].sum().nlargest(2).iloc[-1]"),
    ('Question: Find the percentage change in total revenue from 2023 to 2024. Columns: day, revenue',
     "100*(df.loc[pd.to_datetime(df['day']).dt.year==2024,'revenue'].sum()/df.loc[pd.to_datetime(df['day']).dt.year==2023,'revenue'].sum()-1)"),
    ('Question: What is the average price weighted by quantity, for category A? Columns: category, price, quantity',
     "(df.loc[(df['category']=='A')&df['price'].notna()&df['quantity'].notna(),'price']*df.loc[(df['category']=='A')&df['price'].notna()&df['quantity'].notna(),'quantity']).sum()/df.loc[(df['category']=='A')&df['price'].notna()&df['quantity'].notna(),'quantity'].sum()"),
    ('Question: What is total revenue divided by total expense, only for rows having both? Columns: revenue, expense',
     "df.loc[df['revenue'].notna()&df['expense'].notna(),'revenue'].sum()/df.loc[df['revenue'].notna()&df['expense'].notna(),'expense'].sum()"),
    ('Question: What is the average revenue for readings at sites with floor at least 3? Columns in already joined df: revenue, floor',
     "df.loc[df['floor']>=3,'revenue'].mean()"),
    ('Question: What fraction of records with a recorded grade have grade at least 60? Columns: grade',
     "(df.loc[df['grade'].notna(),'grade']>=60).mean()"),
    ('Question: What is the difference between the maximum and minimum group average cost? Columns: region, cost',
     "df.groupby('region')['cost'].mean().max()-df.groupby('region')['cost'].mean().min()"),
])


COMPARISON_PATTERN=r'(strictly greater than|strictly less than|greater than|less than|of at least|of at most|at least|at most|on or after|before|below|above|after|in|is)'
DATE_TEXT_PATTERN=r'(?:[a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{4}|\d{4})'


def table_examples(task):
    """Retrieve short demonstrations by operation, not memorized domain templates."""
    q = task['objective'].lower()
    files = task.get('files') or {}
    names = list(files) if isinstance(files, dict) else [os.path.basename(str(p)) for p in files]
    joined = sum(n != 'data_dictionary.csv' for n in names) > 1
    rules = [
        (r'\b(?:weighted mean|weighted average)\b', (21, 4, 0)),
        (r'\b(?:growth|percentage change|percent change)\b', (20, 9, 2)),
        (r'\b(?:second|third|top [0-9]+)\b', (19, 2, 3)),
        (r'\b(?:ratio of totals|total.+divided by total)\b', (22, 4, 1)),
        (r'\b(?:rank|ranking)\b', (3, 2, 19)),
        (r'\b(?:difference between.*(?:largest|smallest|average)|difference between.*mean)\b', (25, 2, 6)),
        (r'\b(?:on or after|before|after|during|dated|year|month)\b', (18, 9, 0)),
        (r'\b(?:percentage|percent|fraction|proportion|share)\b', (17, 24, 1)),
        (r'\b(?:correlation|pearson|spearman)\b', (12, 4, 14)),
        (r'\b(?:distinct|different|unique)\b', (13, 16, 1)),
        (r'\b(?:percentile|quantile|iqr|interquartile)\b', (14, 10, 11)),
        (r'\bvariance|standard deviation\b', (15, 11, 10)),
        (r'\b(?:per person|per item|per day|divided by|ratio)\b', (4, 22, 0)),
        (r'\b(?:for each|group|by the|by site|by region|across sites)\b', (2, 6, 25)),
        (r'\b(?:missing|recorded|not recorded)\b', (24, 17, 1)),
    ]
    selected = None
    for regex, indexes in rules:
        if re.search(regex, q):
            selected = indexes
            break
    if selected is None:
        selected = (23, 7, 2) if joined else (0, 1, 2)
    if joined and not any(i in (7, 23) for i in selected):
        selected = selected[:2] + (23,)
    messages = []
    for idx in selected:
        prompt, answer = DEMO_table[idx]
        messages.extend(({'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': answer}))
    return messages


def read_files(files, objective=''):
    return fit.read_files(files, objective)[0]


class ColumnUnclear(ValueError):
    """The deterministic query is understood, but a column name needs clarification."""

    def __init__(self, phrase, filename, columns):
        super().__init__(f"Unclear column {phrase!r} in {filename}")
        self.phrase = phrase
        self.filename = filename
        self.columns = columns


class Data:
    def __init__(self, task):
        # store the question, tables and data-dictionary descriptions for later matching
        self.t = task
        self.filters_used = 0
        self.column_hints = task.get('_column_hints', {})
        self.q = task['objective']
        self.lower = self.q.lower()
        self.tables = read_files(task['files'], self.q)
        # use data_dictionary.csv to translate coded column names into their meaning
        raw = self.tables.get('data_dictionary.csv')
        self.map = {}
        if raw is not None:
            for _, r in raw.iterrows():
                name, col, desc = str(r['file']),str(r['column']),str(r['description']).lower()
                self.map.setdefault(name, {})[col] = desc
        # exclude the data dictionary from the datasets to analyze
        self.files = [x for x in self.tables if x != 'data_dictionary.csv']
        if not self.files:
            raise ValueError('No data files')
        # The fact table normally has more observations than its lookup table.
        # Use column count to break row-count ties; do not assume CSV file order.
        self.main = max(self.files, key=lambda f: (len(self.tables[f]), len(self.tables[f].columns)))
        self.files.remove(self.main)
        self.files.insert(0, self.main)
        # drop duplicate if asked
        if re.search('duplicat|appear twice|count each .* once|one copy of each',self.lower):
            f=self.main
            frame=self.tables[f]; desc=self.map.get(f,{})
            cands=[c for c in frame.columns if re.search(r'\bidentifier\b|\bid\b',desc.get(c,'')) or desc.get(c,'').endswith(' number')]
            cands=cands or [c for c in frame.columns if c.lower()=='id']
            cands=[c for c in cands if frame[c].nunique()>=0.5*len(frame)]
            if not cands:
                old = self.column('id',f, required=False)
                if not old:
                    first = frame.columns[0]; d0=desc.get(first,'')
                    if 'identifier' in d0 or d0.endswith(' number'): 
                        old=first
                if old and frame[old].nunique() >= 0.5*len(frame): 
                    cands=[old]
            key=cands[0] if cands else None
            if key: self.tables[f] = self.tables[f].drop_duplicates(subset=[key])
            else: self.tables[f] = self.tables[f].drop_duplicates()


    def column(self, key, file=None, required=True):
        # find a CSV column using its name first, then its human-readable description
        file = file or self.main
        cols = list(self.tables[file].columns)
        descriptions = self.map.get(file, {})
        query = str(key).lower().strip()
        # first exact column-name matches, if not: approximate matching
        for col in cols:
            if col.lower() == query:
                return col
        exact = [col for col in cols if descriptions.get(col, '').strip() == query]
        if len(exact) == 1:
            return exact[0]
        tokens = lambda text: set(re.findall(r'[a-z0-9]+', text.lower())) - {'the', 'of', 'a', 'an', 'in', 'for', 'to', 'by', 'and', 'or', 'with'}
        wanted = tokens(query)
        # use word overlap only if no exact name or description is found
        scored = []
        for col in cols:
            description = descriptions.get(col, '')
            words = tokens(col + ' ' + description)
            overlap = len(wanted & words)
            score = overlap / max(len(wanted), 1)
            if score > 0:
                scored.append((score, overlap, col))
        if scored:
            scored.sort(reverse=True)
            best = scored[0]
            if best[0] == 1 and (len(scored) == 1 or scored[1][0] < best[0]):
                return best[2]
        if required:
            raise ValueError(f'Cannot identify column for {key!r} in {file}; columns: {cols}')
        return None
    def frame(self, file=None):
        return self.tables[file or self.main].copy()


def schema_terms(text):
    # extract significant normalized words for schema comparisons
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    stop = {'the', 'a', 'an', 'of', 'for', 'in', 'by', 'to', 'with', 'from', 'that', 'each', 'per', 'is', 'and', 'or', 'number', 'total', 'mean', 'average'}
    return {w[:-1] if w.endswith('s') and len(w) > 4 else w for w in words if w not in stop}


STOP_WORDS = set('the a an of in for to by and or with on whose which that what its their each is are was were as at from who have has'.split())

COMMON_TERMS = {'occupancy':'people', 'occupants':'people', 'persons':'people', 'person':'people', 'carrier':'company', 'haulier':'company', 'size':'number', 'duration':'length', 'stay':'stay', 'days':'day'}


def word_tokens(value):
    # Use morphological normalization and a few broadly useful equivalents, not dataset names.
    words = re.findall(r'[a-z0-9]+', str(value).lower())
    result = set()
    for word in words:
        if word in STOP_WORDS or word.isdigit():
            continue
        if word.endswith('s') and len(word) > 4:
            word = word[:-1]
        result.add(COMMON_TERMS.get(word, word))
    return result


def date_column(frame, col, question=''):
    # convert text values to Pandas dates, respecting day-first wording when specified
    s = frame[col].astype('string')
    prefer_day = bool(re.search(r'dd/mm/yyyy|day/month/year|day first|day-first',question,re.I))
    try:
        out = pd.to_datetime(s, dayfirst=prefer_day, errors='coerce', format='mixed')
    except (ValueError,TypeError):
        out = pd.to_datetime(s, dayfirst=prefer_day, errors='coerce')
    return out


def resolve_column(d, phrase, name=None, numeric=None, date=None, exclude=()):
    name = name or d.main
    frame = d.tables[name]
    query = str(phrase).lower().strip()
    # A small LLM can resolve an ambiguous phrase to an existing column.
    suggested = d.column_hints.get((name, query))
    if suggested in frame.columns and suggested not in exclude:
        series = frame[suggested]
        if numeric is not True or pd.api.types.is_numeric_dtype(series):
            if numeric is not False or not pd.api.types.is_numeric_dtype(series):
                return suggested
    query = re.sub(r'\s+(?:values|columns?|fields?)$', '', query)
    if query.startswith('year of the '):
        query = query[len('year of the '):]
        date = True
    terms = word_tokens(query)
    descriptions = d.map.get(name, {})
    scores = []
    for col in frame:
        if col in exclude:
            continue
        series = frame[col]
        if numeric is True and not pd.api.types.is_numeric_dtype(series):
            continue
        if numeric is False and pd.api.types.is_numeric_dtype(series):
            continue
        description = descriptions.get(col, '')
        label = f'{col} {description}'
        words = word_tokens(label)
        common = words & terms
        importance = lambda word: 0.2 if word in {'number','value','total','amount'} else 1.0
        numerator = sum(importance(w) for w in common)
        score = 5 * numerator / max(sum(importance(w) for w in terms), 0.01) + numerator / max(sum(importance(w) for w in words), 0.01)
        if description.lower().startswith(query):
            score += 2
        if query == col.lower():
            score += 15
        if query == description.strip().lower():
            score += 12
        if query in description.lower() and len(query) >= 3:
            score += 4
        if numeric is not False and re.search(r'\bidentifier\b|\bid\b', description.lower()) and not terms.intersection({'identifier', 'id'}):
            score -= 6
        if date is True:
            score += 5 if 'date' in words or 'birth' in words or 'day' in words else -9
        if date is False and ('date' in words or 'birth' in words) and not terms.intersection({'date','birth','grant','listing'}):
            score -= 2
        scores.append((score, col))
    scores.sort(reverse=True)
    if not scores or scores[0][0] <= 0:
        raise ColumnUnclear(phrase, name, {col:descriptions.get(col,'') for col in frame if col not in exclude})
    if len(scores)>1 and scores[0][0]-scores[1][0]<0.15:
        raise ColumnUnclear(phrase, name, {col:descriptions.get(col,'') for col in frame if col not in exclude})
    return scores[0][1]


def numeric_values(d,df,col):
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):return pd.to_numeric(s,errors='coerce')
    return pd.to_numeric(s.astype('string').str.replace(r'^\s*([-+]?\d+(?:[.,]\d+)?)\s*[^\d\s.,]{1,8}\s*$',r'\1',regex=True).str.replace(',', '.', regex=False),errors='coerce')


def clean_frame(d,frame,name):
    df=frame.copy()
    q=d.lower
    m=re.search(r'(?:a|an|the)\s+([\w\- ]+?)\s+of\s+-1\s+means|(?:the )?value -1 in the\s+([\w\- ]+?)\s+column',q)
    if m:
        phrase=(m.group(1) or m.group(2))
        try:
            col=resolve_column(d,phrase,name)
            if pd.api.types.is_numeric_dtype(df[col]):df[col]=df[col].mask(df[col].eq(-1))
            else:df[col]=df[col].mask(df[col].astype('string').isin(['-1','-1.0','-1,0']))
        except ValueError:pass
    return df


USED_VALUES=[]   
WEAK={}          


def category_mask(df, col, value, desc=''):
    USED_VALUES.append(str(value).strip().lower())
    s=df[col].astype('string').str.strip().str.lower()
    v=str(value).strip().lower()
    if s.eq(v).any():return s.eq(v)
    words=re.escape(v)
    found=re.findall(r'([a-z]{1,4}\d*\d|[a-z]{2,4})\s+'+words+r'(?=[,;)\s]|$)',desc.lower())
    if found:
        if len(set(found))==1:return s.eq(found[0])
    labels=re.findall(r'\b([a-z]+\d+|[a-z]{2,4})\s+([^,;()]+)',desc.lower())
    for code,label in labels:
        if v in label or label.strip() in v:return s.eq(code)
    if s.str.contains(re.escape(v),regex=True,na=False).any():return s.str.contains(re.escape(v),regex=True,na=False)
    raise ValueError(f'Cannot map categorical value {value!r} to {col!r}')


def filter_mask(d,df,phrase,name=None):
    d.filters_used += 1
    name=name or d.main
    text=str(phrase).lower().strip(' ,.?!')
    text=re.sub(r'^(?:the |a |an |with |of |whose |that |have |has )+', '',text)
    desc=d.map.get(name,{})
    m=re.search(r"(?:whose |with |the |a )?([\w\s-]+?)\s+(?:field\s+)?(?:is|equals|as)\s+['\"]([^'\"]+)['\"]",text)
    if m:
        col=resolve_column(d,m[1],name)
        return category_mask(df,col,m[2],desc.get(col,''))
    m=re.search(r"['\"]([^'\"]+)['\"]\s+(?:as|in)\s+(?:the\s+)?(.+?)(?:\s+field)?$",text)
    if m:
        col=resolve_column(d,m[2],name)
        return category_mask(df,col,m[1],desc.get(col,''))
    m=re.search(r'([\w\s-]+?)\s+(?:field\s+)?(?:is|equals|as)\s+([yn])\b',text)
    if m:
        col=resolve_column(d,m[1],name)
        return category_mask(df,col,m[2],desc.get(col,''))
    m=re.search(r'\b(?:born|listed|shipped|delivered|delivery date|shipment date|listing date|birth date)\s+(on or after|before|in)\s+('+DATE_TEXT_PATTERN+r')',text)
    if m:
        which=m.group(0).split(' '+m.group(1))[0].strip()
        word={'born':'birth','listed':'listing date','shipped':'shipment date','delivered':'delivery date'}.get(which,which)
        col=resolve_column(d,word,name,date=True)
        s=date_column(df,col,d.q)
        v=m.group(2)
        if re.fullmatch(r'\d{4}',v):return s.dt.year.eq(int(v))
        if re.fullmatch(r'[a-z]+\s+\d{4}',v):
            dt=pd.to_datetime(v);return (s.dt.year==dt.year)&(s.dt.month==dt.month)
        v=pd.to_datetime(v)
        return s>=v if m[1]=='on or after' else s<v if m[1]=='before' else s.eq(v)
    m=re.search(r'(.+?)\s+'+COMPARISON_PATTERN+r'\s+(-?\d+(?:\.\d+)?)(?:\s+(?:kg|km|months|years|metres|square|euros|points|percent|%|half-days))?',text)
    if m:
        key,op,value=m[1],m[2],float(m[3])
        key=re.sub(r'^(?:with |a |an |the |a number of |the number of )','',key)
        col=resolve_column(d,key,name)
        s=numeric_values(d,df,col)
        if op in ('strictly greater than','greater than','above'):return s>value
        if op in ('strictly less than','less than','below','before'):return s<value
        if op in ('at least','of at least','on or after'):return s>=value
        if op in ('at most','of at most'):return s<=value
        return s.eq(value)
    for col in df:
        dcol=desc.get(col,'')
        if 'coded (' not in dcol.lower():continue
        codes=dcol.lower().split('coded (',1)[1].split(')',1)[0]
        matches=re.findall(r'\b([a-z]{1,4}\d+|[a-z]{2,4})\s+([^,]+)',codes)
        for code,label in matches:
            if label.strip() and re.search(r'\b'+re.escape(label.strip())+r'\b',text):
                return category_mask(df,col,label.strip(),dcol)
    raise ValueError('Unsupported filter '+text)


def aggregate_values(values,how):
    x=pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan)
    how=how.lower()
    if '75th percentile' in how or 'quantile' in how:return float(x.quantile(.75))
    if 'population' in how and 'standard deviation' in how:return float(x.std(ddof=0))
    if 'standard deviation' in how:return float(x.std(ddof=1))
    if any(w in how for w in ('mean','average')):return float(x.mean())
    if 'median' in how:return float(x.median())
    if any(w in how for w in ('maximum','largest','highest','max ')):return float(x.max())
    if any(w in how for w in ('minimum','smallest','lowest','min ')):return float(x.min())
    if any(w in how for w in ('total','sum')):return float(x.sum())
    raise ValueError('No aggregation '+how)


def requested_metric(q):
    patterns=[r'what is the\s+(?P<how>standard deviation|75th percentile|maximum|minimum|average|mean|median|total|sum of|largest|smallest)\s+(?P<metric>.+?)\s+(?:over all|over the|of (?:the |all )|for the |of all )',r'what is its\s+(?P<how>maximum|minimum|average|mean|median|largest|smallest)',r'what is the\s+(?P<how>maximum|minimum|average|mean|median|total|sum of|largest|smallest)\s+(?P<metric>.+?)\s*(?:\(|\?)']
    for p in patterns:
        m=re.search(p,q)
        if m:return m.group('how'), (m.groupdict().get('metric') or '').strip()
    raise ValueError('No metric '+q)


def source_frame(d):
    name=d.main
    return clean_frame(d,d.frame(name),name)


def generic_stat(d):
    q=d.lower
    df=source_frame(d)
    m=re.search(r'(?:how many|for how many)\s+\w+\s+(?:have|has|is|are)\s+(?:a |an |the )?(no recorded|missing or not recorded|recorded|missing)\s+(.+?)(?:\?|\.)',q)
    if m:
        col=resolve_column(d,m[2],d.main)
        return float(df[col].isna().sum() if m[1] in ('no recorded','missing','missing or not recorded') else df[col].notna().sum())
    m=re.search(r'(?:for )?how many\s+\w+\s+is (?:the )?(.+?)\s+(?:missing|not recorded)', q)
    if m:
        col = resolve_column(d, m[1], d.main)
        return float(df[col].isna().sum())
    how,field=requested_metric(q)
    field=re.sub(r'\s*\([^)]*\)\s*$','',field)
    col=resolve_column(d,field,d.main,numeric=True)
    return aggregate_values(numeric_values(d,df,col),how+' '+q if 'standard deviation' in how else how)


def generic_derive(d):
    q=d.lower
    df=source_frame(d)
    m=re.search(r'consider (?:the |an? )?(.+?)(?:\(|\.)',q)
    description=m[1].strip() if m else ''
    fraction=re.search(r'\(the\s+(.+?)\s+divided by the\s+(.+?)\)',q)
    if fraction:
        a,b=[resolve_column(d,x,d.main,numeric=True) for x in fraction.groups()]
        v=numeric_values(d,df,a)/numeric_values(d,df,b)
        raw_dictionary=d.tables.get('data_dictionary.csv')
        if raw_dictionary is not None and 'unit' in raw_dictionary.columns:
            rows=raw_dictionary.loc[(raw_dictionary['file'].astype(str)==d.main)&(raw_dictionary['column'].astype(str)==a)]
            if len(rows):
                source_unit=str(rows.iloc[0]['unit']).lower().strip()
                result_question=q.split('what is its',1)[-1]
                wanted=re.search(r'\(in (.+?)\)',result_question)
                result_unit=wanted[1].lower() if wanted else ''
                scales={'thousand euros':1000.0,'million euros':1000000.0,'euros':1.0,'kg':1.0,'tonnes':1000.0}
                for src,srcscale in scales.items():
                    if source_unit==src:
                        for dest,destscale in scales.items():
                            if result_unit.startswith(dest+' per') and src!=dest and src.split()[-1]==dest.split()[-1]:
                                v=v*(srcscale/destscale)
                                break
                        break
    elif ' per ' in description:
        a,b=description.split(' per ',1)
        if b.startswith('day of '):
            b = 'length of ' + b[len('day of '):]
        v=numeric_values(d,df,resolve_column(d,a,d.main,numeric=True))/numeric_values(d,df,resolve_column(d,b,d.main,numeric=True))
    else:
        m=re.search(r'consider the\s+(.+?)\s+in\s+([\w]+)\s*\((?:1\s+)?(?:[\w]+)\s*=\s*([\d.]+)\s+[\w]+\)',q)
        if not m:raise ValueError('No derived formula')
        col=resolve_column(d,m[1],d.main,numeric=True)
        factor=float(m[3]);v=numeric_values(d,df,col)*(1/factor if 'tonne' in m[2] else factor)
    filtered = False
    m=re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\s+(.+?)\s*\(',q)
    if m:
        text=m[1]
        if 'listed' in m[0]:text='listed '+text
        if 'shipped' in m[0]:text='shipped '+text
        if 'born' in m[0]:text='born '+text
        v=v.loc[filter_mask(d,df,text)]
        filtered = True
    m=re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\s+(.+?)\?',q)
    if m and not filtered and not re.search(r'\b(?:round|report|give)\b',m[1]):
        text = re.sub(r'\s*\([^)]*\)\s*$', '', m[1]).strip()
        v = v.loc[filter_mask(d,df,text)]
    m=re.search(r'what is its\s+(maximum|minimum|median|mean|average|largest|smallest)',q)
    if not m:raise ValueError('No derived aggregation')
    return aggregate_values(v,m[1])


def generic_filter(d):
    q=d.lower
    df=source_frame(d)
    m=re.search(r'among the\s+\w+\s+(?:with|whose|of|born|shipped)\s+(.+?),\s+what\s+(percentage|fraction)(?:\s+\(between 0 and 1\))?\s+(?:have|belong|were|are)\s+(.+?)\?',q)
    if m:
        between=re.search(r'among the\s+\w+\s+(.+?),\s*what',q)[1]
        before=between if between.startswith(('born','shipped','listed')) else re.sub(r'^(?:with|whose|of)\s+','',between)
        df2=df.loc[filter_mask(d,df,before)]
        after=m[3]
        after=re.sub(r'^(?:(?:to|in|the)\s+)+','',after)
        mask=filter_mask(d,df2,after)
        return float(mask.mean()*(100 if m[2]=='percentage' else 1))
    m=re.search(r'what\s+(percentage|fraction)(?:\s+\(between 0 and 1\))?\s+(?:have|belong|were|are)\s+(.+?)\?',q)
    if m:
        return float(filter_mask(d,df,m[2]).mean()*(100 if m[1]=='percentage' else 1))
    m=re.search(r'how many\s+\w+\s+(?:born|listed|with|whose|that|of)\s+(.+?)\?',q)
    if m:
        f='born '+m[1] if 'born' in m[0] else m[1]
        return float(filter_mask(d,df,f).sum())
    how,metric=requested_metric(q)
    tail=q[q.find('what is the '):] if 'what is the ' in q else q
    m=re.search(r'\b(?:of the|over the)\s+\w+\s+(?:with|whose|that|shipped|born)\s+(.+?)\s*(?:\(|\?)',tail)
    if not m:
        m=re.search(r'\b(?:of the|over the)\s+\w+\s+of (?:the )?(.+?)\s*(?:\(|\?)',tail)
    if m:
        filt=m[1]
        if 'shipped' in m[0]:filt='shipped '+filt
        if 'born' in m[0]:filt='born '+filt
        df=df.loc[filter_mask(d,df,filt)]
    elif re.search(r'\b(?:of|over) the\s+\w+\s+(?:with|whose|that|born|shipped|listed)\b', tail):
        # Never answer a restricted statistic using the entire table.
        raise ValueError('Unparsed filter; needs LLM fallback')
    col=resolve_column(d,metric,d.main,numeric=True)
    return aggregate_values(numeric_values(d,df,col),how)


def generic_group(d):
    q = d.lower
    df = source_frame(d)
    restricted = re.search(r'consider only the\s+\w+\s+(?:with|of)\s+(.+?)\.\s+what', q)
    if restricted:
        df = df.loc[filter_mask(d, df, restricted[1])]
    elif 'consider only' in q:
        raise ValueError('Unparsed group restriction; needs LLM fallback')

    rank = re.search(r'rank the (.+?) values? by the (mean|median|total|sum|average|number of) (.+?)(?:,|\.)', q)
    if rank:
        group, how, metric = rank.groups()
        group = re.sub(r'\s+values?$', '', group)
    else:
        match = re.search(r'\b(?:rank the|compute the)\s+(?:(mean|median|total|average|number of)\s+)?(.+?)\s+(?:for each|by the|by)\s+(.+?)(?:\.|,)', q)
        if not match:
            raise ValueError('Unknown group operation')
        how, metric, group = match.groups()
        how = how or 'total'

    if group.startswith('year of the '):
        gc = resolve_column(d, group[len('year of the '):], date=True)
        labels = date_column(df, gc, d.q).dt.year
    else:
        gc = resolve_column(d, group)
        labels = df[gc]

    if how == 'number of':
        values = df.groupby(labels, dropna=True).size()
    else:
        col = resolve_column(d, metric, numeric=True)
        grouped = numeric_values(d, df, col).groupby(labels, dropna=True)
        values = grouped.mean() if how in ('mean', 'average') else grouped.median() if how == 'median' else grouped.sum()

    rank_target = re.search(r'what is the rank of (.+?)\?', q)
    if rank_target:
        rankings = values.rank(ascending=False, method='min')
        target = rank_target[1].strip().removeprefix('the ')
        if target.isdecimal():
            key = int(target)
        else:
            possible = category_mask(df, gc, target, d.map.get(d.main, {}).get(gc, ''))
            observed = labels[possible].dropna().unique()
            if len(observed) != 1:
                raise ValueError('Rank category ambiguous')
            key = observed[0]
        return float(rankings.loc[key])

    if 'difference between the largest and the smallest' in q:
        return float(values.max() - values.min())
    if 'largest' in q or 'maximum' in q:
        return float(values.max())
    if 'smallest' in q or 'minimum' in q:
        return float(values.min())
    raise ValueError('Unrecognized group output')


def generic_join(d):
    names=[n for n in d.files if n!=d.main]
    if not names:raise ValueError('Missing lookup')
    look=names[0]
    a=source_frame(d)
    b=clean_frame(d,d.frame(look),look)
    result,(lc,rc)=join_on_best_keys(a,b,look,d.map.get(d.main,{}),d.map.get(look,{}))
    suffix='__'+look.removesuffix('.csv')
    cols={c:(c+suffix if c in a.columns else c) for c in b.columns if c!=rc}
    q=d.lower
    group = re.search(r'group the \w+ by the (.+?) of their \w+ and compute the (mean|median|average|sum|total) (.+?) per group', q)
    if group:
        col = resolve_column(d, group[1], look, exclude=[rc])
        metric = resolve_column(d, group[3], d.main, numeric=True)
        values = numeric_values(d, result, metric).groupby(result[cols[col]], dropna=True)
        agg = group[2]
        result_values = values.mean() if agg in ('mean','average') else values.median() if agg=='median' else values.sum()
        if re.search(r'\b(largest|maximum)\b', q):
            return float(result_values.max())
        if re.search(r'\b(smallest|minimum)\b', q):
            return float(result_values.min())
        raise ValueError('Unknown grouped statistic')
    m=re.search(r'belong to an?\s+\w+\s+(?:with (?:an? )?|whose )(.+?)(?:\?|\s*\()',q)
    if not m:raise ValueError('No join lookup condition')
    phrase=m[1]
    cm=re.search(r'(.+?)\s+'+COMPARISON_PATTERN+r'\s+(.+)',phrase)
    if not cm:raise ValueError('No comparator in joined criterion')
    field,op,tail=cm.groups()
    field=re.sub(r'\s+(?:of|with an?|whose)\s*$','',field).strip()
    field=re.sub(r'^a\s+','',field)
    col=resolve_column(d,field,look,exclude=[rc])
    actual=result[cols[col]]
    number=re.match(r'(-?\d+(?:\.\d+)?)',tail)
    if number:
        v=float(number[1]);s=pd.to_numeric(actual,errors='coerce')
        if op in ('at least','of at least','on or after'):mask=s>=v
        elif op in ('below','strictly less than','less than','before'):mask=s<v
        elif op in ('at most','of at most'):mask=s<=v
        else:mask=s>v
    else:
        cat=re.search(r"['\"]([^'\"]+)['\"]",tail)
        if not cat:raise ValueError('No value in join criterion')
        mask=actual.astype('string').str.lower().eq(cat[1].lower())
    df=result[mask.fillna(False)]
    if re.search(r'how many\s+\w+\s+belong',q):return float(len(df))
    how,metric=requested_metric(q)
    metric=re.sub(r'\s+of the\s+\w+.*$','',metric)
    x=resolve_column(d,metric,d.main,numeric=True,exclude=[lc])
    return aggregate_values(numeric_values(d,df,x),how)


def generic_extra(d):
    q = d.lower
    query_tail = q.split('what is ', 1)[-1].split('how many ', 1)[-1]
    if re.search(r'\b(?:among the|consider only|restricted to|limited to)\b', query_tail) or re.search(r'\bover the \w+(?: \w+){0,2} (?:with|whose|that|of the)\b', query_tail):
        raise ValueError('Restricted extended statistic needs the flexible expression solver')
    df = source_frame(d)
    if 'weighted mean' in q or 'weighted average' in q:
        m = re.search(r'weighted (?:mean|average) of (?:the )?(.+?) (?:using|weighted by) (?:the )?(.+?)(?:\s+over all[^?]*)?\?', q)
        if m:
            value = resolve_column(d, m[1], numeric=True)
            weight = resolve_column(d, m[2], numeric=True, exclude=[value])
            a, w = numeric_values(d, df, value), numeric_values(d, df, weight)
            mask = a.notna() & w.notna()
            if not mask.any() or w[mask].sum() == 0:
                raise ValueError('Invalid or empty weights')
            return float((a[mask] * w[mask]).sum() / w[mask].sum())
    if re.search(r'\bproduct of\b', q):
        m = re.search(r'product of (?:the )?(.+?)(?:\s+over all[^?]*)?\?', q)
        if m:
            col = resolve_column(d, m[1], numeric=True)
            return float(numeric_values(d,df,col).prod(min_count=1))
    if re.search(r'\b(?:pearson |spearman )?(?:correlation|correlation coefficient)\b', q):
        match = re.search(r'correlation(?: coefficient)?\s+between\s+(?:the )?(.+?)\s+and\s+(?:the )?(.+?)(?:\?|\.|\s+over\s+all)', q)
        if not match:
            raise ValueError('Unable to identify the correlation variables')
        x = resolve_column(d, match[1], numeric=True)
        y = resolve_column(d, match[2], numeric=True, exclude=[x])
        how = 'spearman' if 'spearman' in q else 'pearson'
        return float(numeric_values(d, df, x).corr(numeric_values(d, df, y), method=how))
    if 'interquartile range' in q or re.search(r'\biqr\b', q):
        match = re.search(r'(?:interquartile range|iqr)\s+of\s+(?:the )?(.+?)(?:\s+over\s+all|\s+among\s+the|\?|\.)', q)
        if not match:
            raise ValueError('Unknown interquartile range column')
        col = resolve_column(d, match[1], numeric=True)
        x = numeric_values(d, df, col)
        return float(x.quantile(.75) - x.quantile(.25))
    if re.search(r'\b(?:percentile|quantile)\b', q):
        match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)\s+percentile\s+of\s+(?:the )?(.+?)(?:\s+over\s+all|\s+among\s+the|\s+for\s+the|\?|\.)', q)
        if match:
            quantile = int(match[1]) / 100
            col = resolve_column(d, match[2], numeric=True)
            return float(numeric_values(d, df, col).quantile(quantile))
    if re.search(r'\b(?:number of distinct|number of unique|how many distinct|how many different|how many unique)\b', q):
        match = re.search(r'(?:number of (?:distinct|unique)|how many (?:distinct|different|unique))\s+(?:values?\s+(?:of|in)\s+)?(?:the )?(.+?)(?:\s+are\s+there|\s+over\s+all|\s+in\s+the|\?|\.)', q)
        if match:
            try:   
                col = resolve_column(d, match[1], numeric=False)
            except ValueError:
                col = resolve_column(d, match[1])
            return float(df[col].nunique(dropna=True))
    if re.search(r'\b(?:population|sample) variance\b', q):
        match = re.search(r'\b(?:population|sample) variance\s+of\s+(?:the )?(.+?)(?:\s+over\s+all|\?|\.)', q)
        if match:
            col = resolve_column(d, match[1], numeric=True)
            return float(numeric_values(d, df, col).var(ddof=0 if 'population' in q else 1))
    if 'range' in q and 'maximum' in q and 'minimum' in q:
        match = re.search(r'range\s+of\s+(?:the )?(.+?)(?:\s+over\s+all|\?|\.)', q)
        if match:
            col = resolve_column(d, match[1], numeric=True)
            x = numeric_values(d, df, col)
            return float(x.max()-x.min())
    raise ValueError('No deterministic implementation for this table operation')


def generic_table(d):
    typ=d.t.get('type') or infer_type(d.t)
    funcs={'table.stat':generic_stat,'table.filter':generic_filter,'table.group':generic_group,'table.derive':generic_derive,'table.join':generic_join,'table.extra':generic_extra}
    if typ not in funcs:raise ValueError('Not table')
    return funcs[typ](d)


def join_key(series):
    value = series.astype('string').str.strip().str.lower()
    value = value.str.replace(r'^0+(?=\d)', '', regex=True)
    value = value.str.replace(r'^[a-z]+[-_ ]+(?=[a-z]*\d)', '', regex=True)
    value = value.str.replace(r'^[a-z]0*(?=\d+$)', '', regex=True)
    return value


def join_on_best_keys(left, right, right_name, left_descriptions=None, right_descriptions=None):
    left_descriptions = left_descriptions or {}
    right_descriptions = right_descriptions or {}
    candidates = []
    for lc in left:
        a = join_key(left[lc]).dropna()
        if a.empty or a.nunique()<2:
            continue
        for rc in right:
            b = join_key(right[rc]).dropna()
            if b.empty or b.nunique()<2 or b.nunique()!=len(b):
                continue
            overlap = a.isin(set(b)).mean()
            if overlap<0.55:
                continue
            sa = schema_terms(lc+' '+left_descriptions.get(lc,''))
            sb = schema_terms(rc+' '+right_descriptions.get(rc,''))
            semantic = len(sa&sb)
            name_equal = lc.lower()==rc.lower()
            if overlap<0.85 and not (semantic or name_equal):
                continue
            categorical = not pd.api.types.is_numeric_dtype(left[lc]) and not pd.api.types.is_numeric_dtype(right[rc])
            if not semantic and not name_equal and not categorical:
                continue
            score = overlap*3 + semantic*1.3 + (1 if name_equal else 0) + (0.3 if categorical else 0)
            candidates.append((score,overlap,lc,rc))
    if not candidates:
        raise ValueError('No validated many-to-one join')
    candidates.sort(reverse=True)
    if len(candidates)>1 and candidates[0][0]-candidates[1][0]<0.1:
        raise ValueError('Join keys ambiguous')
    _, overlap, lc,rc = candidates[0]
    a,b = left.copy(),right.copy()
    join_column='__join_tmp_key__'
    a[join_column]=join_key(a[lc]);b[join_column]=join_key(b[rc])
    result=a.merge(b, on=join_column,how='left',suffixes=('', '__'+right_name.removesuffix('.csv')),validate='many_to_one')
    return result.drop(columns=join_column),(lc,rc)


def infer_type(t):
    q=t['objective'].lower()
    if re.search(r'\b(?:how many|for how many)\b[^?]*\b(?:missing|recorded|not recorded)\b',q):
        return 'table.stat'
    files=t.get('files',{})
    names=list(files) if isinstance(files,dict) else [os.path.basename(str(name)) for name in files]
    if len([name for name in names if name!='data_dictionary.csv'])>1:return 'table.join'
    if re.search(r'\b(weighted mean|weighted average|product of)\b',q):return 'table.extra'
    if re.search(r'\b(correlation|interquartile range|iqr|distinct|different values|unique values|population variance|sample variance|\d+(?:st|nd|rd|th) percentile)\b',q):return 'table.extra'
    if 'logistic regression' in q:return 'fit.logistic'
    if 'linear regression' in q or 'least-squares' in q:return 'fit.linear'
    if re.search(r'\b(for each|rank the|group the|by each)\b',q):return 'table.group'
    if re.search(r'\b(consider the|divided by|ratio|per person|per item|per day|per kilometre)\b',q):return 'table.derive'
    if re.search(r'\b(among the|whose|that have|with a|with an|listed before|shipped after|born in)\b',q):return 'table.filter'
    return 'table.stat'


def normalize_question(question):
    replacements = [
        (r'\bno fewer than\b', 'at least'),
        (r'\bno greater than\b', 'at most'),
        (r'\bno more than\b', 'at most'),
        (r'\bno less than\b', 'at least'),
        (r'\bcompute its\b', 'what is its'),
        (r'\bcalculate its\b', 'what is its'),
        (r'(?<!and )\bcompute the\b(?![^.?!]*\b(?:for each|per group)\b)', 'what is the'),
        (r'(?<!and )\bcalculate the\b(?![^.?!]*\b(?:for each|per group)\b)', 'what is the'),
        (r'\bmore than\b', 'strictly greater than'),
        (r'\bfewer than\b', 'strictly less than'),
        (r'\bno more than\b', 'at most'),
        (r'\bgreater than or equal to\b', 'at least'),
        (r'\bless than or equal to\b', 'at most'),
        (r'\bstrictly above\b', 'strictly greater than'),
        (r'\babove\b', 'strictly greater than'),
        (r'\bstrictly below\b', 'strictly less than'),
        (r'\bunder\b', 'strictly less than'),
        (r'\b50th percentile\b', 'median'),
        (r'\bacross all\b', 'over all'),
        (r'\bthroughout all\b', 'over all'),
        (r'\bassociated with\b', 'belong to'),
        (r'\bcorresponding to\b', 'belong to'),
        (r'\bwhat percent\b', 'what percentage'),
        (r'\bwhat proportion\b', 'what fraction'),
        (r'\bhighest\b', 'largest'),
        (r'\blowest\b', 'smallest'),
        (r'\bdetermine the\b', 'what is the')
    ]
    text = question
    # Normalize general question phrasing so one query plan covers several formulations.
    text = re.sub(r'\brestrict attention to\b', 'consider only', text, flags=re.I)
    text = re.sub(r'\bare associated with\b', 'belong to', text, flags=re.I)
    text = re.sub(r'\bis associated with\b', 'belongs to', text, flags=re.I)
    text = re.sub(r'\bhaving\b', 'with', text, flags=re.I)
    text = re.sub(r'\bthat have\b', 'with', text, flags=re.I)
    text = re.sub(r'\b(?:calculate|determine|find)\s+how many\b', 'how many', text, flags=re.I)
    text = re.sub(r'\bcount how many\b', 'how many', text, flags=re.I)
    text = re.sub(r'\border the\b', 'rank the', text, flags=re.I)
    text = re.sub(r'\bby each\b', 'for each', text, flags=re.I)
    text = re.sub(r'\b(?:calculate|determine|find) the\b(?=[^.?!]*\b(?:for each|per group)\b)', 'compute the', text, flags=re.I)
    text = re.sub(r'\b(?:calculate|determine|find)\s+(the|its)\b', r'what is \1', text, flags=re.I)
    text = re.sub(r'(?:(?<=\. )|^)of the(?=\s+[^?]{1,120}?,\s*what\s+(?:percentage|fraction)\b)', 'among the', text, flags=re.I)
    for pattern, value in replacements:
        text = re.sub(pattern, value, text, flags=re.I)
    return text



def guard_unparsed_semantics(task, data, typ):
    """Fail closed rather than silently dropping a restriction or aggregation layer."""
    q = task['objective'].lower()
    if typ in {'table.stat', 'table.filter', 'table.extra', 'table.derive'}:
        filter_clues = (
            r'\bwhere\b',
            r'\b(?:after|before|during)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december|\d{4})',
            r'\b(?:classified as|marked as|restricted to|limited to)\b',
            r'\b(?:for|among|of) (?:the )?(?:regular|special|trial) (?:readings|records|observations)\b',
        )
        if data.filters_used == 0 and any(re.search(pattern, q) for pattern in filter_clues):
            raise ValueError('Unparsed subset condition; use table model')
    # A single-column max/mean is not a group-level maximum/mean.
    if typ not in {'table.group', 'table.join'}:
        if re.search(r'\b(?:among|across|between) (?:the )?(?:sites|regions|groups|categories|classifications)\b', q) and re.search(r'\b(?:largest|smallest|highest|lowest|difference|maximum|minimum)\b', q):
            raise ValueError('Unparsed grouped aggregation; use table model')
        if re.search(r'\b(?:second|third|top \d+|bottom \d+)[ -](?:highest|largest|lowest|smallest)\b', q):
            raise ValueError('Unparsed nth-ranked aggregation; use table model')
    # Percentage growth requires TWO periods, not the mean of one column.
    if typ not in {'table.extra', 'table.derive'} and re.search(r'\b(?:percentage|percent) (?:growth|change|increase|decrease)\b', q):
        raise ValueError('Unparsed growth operation; use table model')

_FULL_DATE = r'(?:\d{4}-\d{2}-\d{2}|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{1,2},?\s+\d{4})'
DATE_RESTRICTION = re.compile(
    r'\b(?:of|for|among|over) (?:all )?the [a-z]+s\s+(?P<part>(?:(?:dated|recorded|made|placed|registered)\s+)?'
    r'(?:(?:in|during) (?P<year>(?:19|20)\d\d)\b|(?P<cmp>on or after|on or before|after|before|since|until) (?P<date>' + _FULL_DATE + r')))')

def question_sentence(text):
    # get the sentence that holds the question
    head = text.split('?')[0]
    return re.split(r'(?<=[.!])\s+', head)[-1]

def apply_date_restriction(d):
    d.date_filtered = False
    q = question_sentence(d.lower)
    m = DATE_RESTRICTION.search(q)
    if not m:
        return
    frame = d.tables[d.main]
    parsed = {c: date_column(frame, c, d.q) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])}
    parsed = {c: s for c, s in parsed.items() if s.notna().sum() >= 0.8 * frame[c].notna().sum() > 0}
    if len(parsed) > 1:
        parsed = {c: s for c, s in parsed.items() if re.search(r'\bdate\b|\bday\b', d.map.get(d.main, {}).get(c, ''))}
    if len(parsed) == 1:
         _, dates = next(iter(parsed.items()))    
    else: 
        try:
            col = resolve_column(d, 'date', date=True)
        except ValueError:
            return
        dates = date_column(frame, col, d.q)                  
    if m['year']:
        mask = dates.dt.year == int(m['year'])
    else:
        ref = pd.to_datetime(m['date'], errors='coerce')
        if pd.isna(ref):
            return
        mask = {'after': dates > ref, 'before': dates < ref, 'on or after': dates >= ref, 'since': dates >= ref,
                'on or before': dates <= ref, 'until': dates <= ref}[m['cmp']]

    d.tables[d.main] = frame[mask.fillna(False)]
    d.filters_used += 1
    d.date_filtered = True
    # remove the date phrase so later it is not read as part of a column name
    d.lower = d.lower.replace(m['part'], '').replace('  ', ' ')
    d.q = re.sub(re.escape(m['part']), '', d.q, count=1, flags=re.I).replace('  ', ' ')


def unused_category_labels(d):
    sentences=re.split(r'(?<=[.!?])\s+',d.lower)
    scope=' '.join([question_sentence(d.lower)]+[s for s in sentences if re.search(r'consider only|among the|restrict|limited to',s)])
    bad=[]
    for cols in d.map.values():
        for desc in cols.values():
            if 'coded (' not in desc:continue
            for code,label in re.findall(r'\b([a-z]{1,4}\d+|[a-z]{2,4})\s+([^,]+)',desc.split('coded (',1)[1].split(')',1)[0]):
                label=label.strip()
                if len(label)>=3 and re.search(r'\b'+re.escape(label)+r'\b',scope):
                    if not any(label in u or u in label or u==code for u in USED_VALUES):bad.append(label)
    return bad

AGG = r'(mean|average|total|median|maximum|minimum|sum)'
TAIL = r'\s*(?:\(in [^)]*\))?\s*\??$'

def group_values(d, measure, group, how):
    if re.search(r"\b(?:with|whose|that|where|between|and)\b|\d", measure + ' ' + group):
        return None 
    df = source_frame(d)                                        
    mcol = resolve_column(d, measure, numeric=True)
    gcol = resolve_column(d, re.sub(r's$', '', group.strip()), numeric=False)
    if df[gcol].nunique() > min(50, 0.5 * len(df)):
        return None                                          
    vals = numeric_values(d, df, mcol)
    g = vals.groupby(df[gcol]).agg(lambda x: aggregate_values(x, how))
    return g.dropna()

def category_rows(d, df, text):
    for col, desc in d.map.get(d.main, {}).items():
        if col in df.columns and 'coded (' in desc:
            try:
                m = category_mask(df, col, text, desc)
                if m.fillna(False).any():
                    return m.fillna(False)
            except ValueError:
                pass
    return None

def generic_compare(d):
    q = question_sentence(d.lower)
    m = re.search(r'what is the (second|third)[ -](largest|smallest) ' + AGG + r' (.+?) (?:among|across|of|over|between) (?:all )?(?:the )?([a-z ]+?)' + TAIL, q)
    if m:
        g = group_values(d, m[4], m[5], m[3])
        if g is None or len(g) < 3:
            return None
        k = {'second': 1, 'third': 2}[m[1]]
        return float(g.sort_values(ascending=(m[2] == 'smallest')).iloc[k])
    m = re.search(r'what is the difference between the (largest|smallest) and (?:the )?(largest|smallest) ' + AGG + r' (.+?) (?:among|across|of|over|between) (?:all )?(?:the )?([a-z ]+?)' + TAIL, q)
    if m and m[1] != m[2]:
        g = group_values(d, m[4], m[5], m[3])
        if g is None or len(g) < 2:
            return None
        return float(g.max() - g.min()) if m[1] == 'largest' else float(g.min() - g.max())
    m = (re.search(r'what is the ' + AGG + r' (.+?) of (?:the )?(.+?) minus the ' + AGG + r' (.+?) of (?:the )?(.+?)' + TAIL, q)
         or re.search(r'what is the difference between the ' + AGG + r' (.+?) of (?:the )?(.+?) and the ' + AGG + r' (.+?) of (?:the )?(.+?)' + TAIL, q))
    if m and m[1] == m[4] and m[2] == m[5]:
        if re.search(r"\b(?:with|whose|that|where)\b|\d", m[2]):
            return None
        df = source_frame(d)
        col = resolve_column(d, m[2], numeric=True)
        a, b = category_rows(d, df, m[3]), category_rows(d, df, m[6])
        if a is None or b is None:
            return None
        d.filters_used += 1
        vals = numeric_values(d, df, col)
        return aggregate_values(vals[a], m[1]) - aggregate_values(vals[b], m[4])
    return None

def generic_count(d):
    q = question_sentence(d.lower)
    if re.search(r'missing|not recorded|no recorded|distinct|different|percentage|fraction|per |each|average|mean', q):
        return None
    m = re.search(r"how many ([a-z]+?) (?:are there )?(?:have|has|with|whose|that)\s+(.+?)(?:\s+are there)?\s*\??$", q)
    if not m or 'at least one' in m[2]:
        return None
    noun = m[1].rstrip('s')                                 
    if any(desc.startswith(noun) for desc in d.map.get(d.main, {}).values() if 'coded (' in desc):
        return None
    cond = re.sub(r"\s+of\s+(['\"])", r" is \1", m[2])        
    df = source_frame(d)
    mask = filter_mask(d, df, cond)
    return float(pd.Series(mask).fillna(False).astype(bool).sum())

def solve_new_patterns(t, strict=True):
    #checking out the added deterministic paths (counting and comparisons)
    USED_VALUES.clear()
    task = {**t, 'objective': normalize_question(t['objective'])}
    task['type'] = task.get('type') or infer_type(task)
    if not task['type'].startswith('table.') or task['type'] == 'table.join':
        return None
    data = Data(task)
    apply_date_restriction(data)
    for fn in (generic_compare, generic_count):
        try:
            result = fn(data)
        except (ValueError, KeyError):
            result = None
        if result is None or not np.isfinite(result):
            continue
        q = question_sentence(data.lower)
        if strict and not data.date_filtered and re.search(r'\b(?:in|during|after|before|since|until)\s+(?:\d{1,2}\s+[a-z]+\s+|[a-z]+\s+\d{1,2},?\s+)?(?:19|20)\d\d\b', q):
            return None
        if strict and unused_category_labels(data):
            return None
        return result
    return None

def solve_data(t, strict=True):
    #if solve_data fails, then other deterministic ways are tried
    try:
        return solve_data_old(t, strict)
    except Exception:
        result = solve_new_patterns(t, strict)
        if result is None:
            raise
        return result



def solve_data_old(t, strict=True):
    # Infer the operation after standardizing wording, not before.
    USED_VALUES.clear()
    task = {**t, 'objective': normalize_question(t['objective'])}
    typ = task.get('type') or infer_type(task)
    if not typ.startswith('table.'):
        raise ValueError('Not a table question')
    task['type'] = typ
    data = Data(task)
    apply_date_restriction(data) 
    result = generic_table(data)
    if not strict:
        return result
    guard_unparsed_semantics(task, data, typ)
    # A mathematically valid result is still wrong if an explicit restriction
    # was silently ignored by a pattern-based parser.
    if typ != 'table.join' and data.filters_used == 0:
        q = task['objective'].lower()
        if re.search(r'\bamong the\b|\bconsider only\b', q):
            raise ValueError('Unapplied subset restriction')
        if re.search(r'\b(?:strictly greater than|strictly less than|at least|at most)\s+-?\d', q):
            raise ValueError('Unapplied numeric filter')
        if re.search(r'\b(?:in|during|after|before|since|until)\s+(?:\d{1,2}\s+[a-z]+\s+|[a-z]+\s+\d{1,2},?\s+)?(?:19|20)\d\d\b', question_sentence(q)):
            raise ValueError('Unapplied date restriction')
        bad = unused_category_labels(data)          
        if bad:
            raise ValueError('Unapplied category restriction: ' + ', '.join(bad))
    return result


def evaluate_table_expression(expression,tables,df):
    expression=expression.strip()
    expression=re.sub(r'^```(?:python)?\s*|\s*```$', '', expression).strip()
    expression=re.sub(r'^answer\s*:\s*','',expression,flags=re.I)
    if len(expression)>3000:
        raise ValueError('Generated code too long')
    parsed=ast.parse(expression,mode='exec')
    if len(list(ast.walk(parsed)))>450:
        raise ValueError('Generated code too complex')
    values={'df':df,'tables':tables,'pd':pd,'np':np,'abs':abs,'round':round,
            'len':len,'float':float,'int':int,'sum':sum,'min':min,'max':max}
    forbidden={'read_pickle','read_parquet','read_sql','read_html','read_json','to_csv',
               'to_pickle','to_sql','to_parquet','to_excel','eval','query','pipe',
               'system','popen','open','getattr','setattr','globals','locals','exec','compile'}
    for node in ast.walk(parsed):
        if isinstance(node,ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden):
            raise ValueError('Unsupported function or attribute')
        if isinstance(node,(ast.ListComp,ast.SetComp,ast.DictComp,ast.GeneratorExp,ast.Lambda,
                            ast.NamedExpr,ast.Await,ast.Yield,ast.YieldFrom,ast.Subscript)):
            if not isinstance(node,ast.Subscript):
                raise ValueError('Unsupported expression syntax')
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in forbidden:
            raise ValueError('Unsupported function')
    statements=[]
    for node in parsed.body:
        if isinstance(node,ast.Import) and len(node.names)==1 and (node.names[0].name,node.names[0].asname) in (('pandas','pd'),('numpy','np')):
            continue
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name.startswith('_') or name in ('pd','np','tables'):
                raise ValueError('Invalid assignment')
            statements.append((name,node.value))
        elif isinstance(node,ast.Expr):
            value=node.value
            if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='print' and len(value.args)==1:
                value=value.args[0]
            statements.append((None,value))
        else:
            raise ValueError('Return expression or simple assignment only')
    if not statements or len(statements)>10:
        raise ValueError('Missing expression')
    class CsvVisitor(ast.NodeTransformer):
        def visit_Call(self,node):
            if (isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name)
                and node.func.value.id=='pd' and node.func.attr=='read_csv'
                and len(node.args)==1 and isinstance(node.args[0],ast.Constant)
                and isinstance(node.args[0].value,str)):
                filename=node.args[0].value
                if filename not in tables:
                    raise ValueError('CSV file not provided')
                return ast.copy_location(ast.Subscript(value=ast.Name(id='tables',ctx=ast.Load()),
                                                       slice=ast.Constant(value=filename),ctx=ast.Load()),node)
            return self.generic_visit(node)
    converter=CsvVisitor()
    import warnings
    last=None
    for target, node in statements:
        node=ast.fix_missing_locations(converter.visit(node))
        for subnode in ast.walk(node):
            if isinstance(subnode,ast.Name) and isinstance(subnode.ctx,ast.Load) and subnode.id not in values:
                raise ValueError('Unknown variable '+subnode.id)
        with warnings.catch_warnings():
            warnings.filterwarnings('error',message='Boolean Series key will be reindexed')
            result=eval(compile(ast.Expression(node),'<table>','eval'),{'__builtins__':{}},values)
        if target is not None:
            values[target]=result
        else:
            last=result
    if last is None:
        last=values.get('answer',values.get('result'))
    if isinstance(last,(pd.Series,pd.DataFrame,np.ndarray)):
        arr=np.asarray(last).reshape(-1)
        if len(arr)!=1:
            raise ValueError('Expression must return one number')
        last=arr[0]
    answer=float(last)
    if not math.isfinite(answer):
        raise ValueError('Non-finite result')
    return answer


def validate_table_result(task, answer):
    question=task['objective'].lower()
    if 'what is the rank of' in question and (answer < 1 or abs(answer-round(answer))>1e-8):
        raise ValueError('A rank must be a positive integer')
    if re.search(r'\bwhat fraction\b', question) and not 0 <= answer <= 1:
        raise ValueError('A fraction must be between 0 and 1')
    if re.search(r'\bwhat percentage\b', question) and not 0 <= answer <= 100:
        raise ValueError('A percentage must be between 0 and 100')
    if re.search(r'\bhow many\b|\bfor how many\b', question) and (answer < 0 or abs(answer-round(answer))>1e-8):
        raise ValueError('A count must be a nonnegative integer')
    return answer


def solve_table_semantic(task):
    return solve_data(task)


def check_and_solve_infinite(tasks,indices,out):
    remaining=[]
    for i in indices:
        try:
            answer=float(solve_data(tasks[i]))
            if math.isfinite(answer):
                out[i]={'id':tasks[i]['id'],'answer':answer}
                continue
        except Exception:
            try:  
                weak=float(solve_data(tasks[i],strict=False))
                if math.isfinite(weak):WEAK[tasks[i]['id']]=weak
            except Exception:
                pass
        remaining.append(i)
    return remaining


def prepare_table_semantic(task):
    # Give Qwen real column names, descriptions, examples, and decoded category labels.
    data = Data(task)
    tables = data.tables
    df = source_frame(data)
    origins = {c:(data.main,c) for c in df}
    joined = ''
    if len(data.files) > 1:
        lookup = data.files[1]
        original = clean_frame(data, data.frame(lookup), lookup)
        df,(left_key,right_key) = join_on_best_keys(
            df, original, lookup, data.map.get(data.main, {}), data.map.get(lookup, {}))
        suffix = '__' + lookup.removesuffix('.csv')
        for col in original:
            if col == right_key:
                continue
            result_col = col+suffix if col in data.tables[data.main].columns else col
            origins[result_col] = (lookup,col)
        joined = f'{left_key} -> {right_key} in {lookup}'

    rows = []
    for col in list(df.columns):
        name,source = origins.get(col,(data.main,col))
        desc = data.map.get(name, {}).get(source, '')
        coded = re.search(r'coded\s*\(([^)]*)\)', desc, re.I)
        if coded:
            pairs = re.findall(r'\b([A-Za-z0-9_-]+)\s+([^,;()]+)', coded[1])
            if pairs:
                mapping = {key.lower():label.strip() for key,label in pairs}
                decoded = df[col].astype('string').str.lower().map(mapping)
                if decoded.notna().any():
                    df['meaning_'+col] = decoded
                    rows.append(f'meaning_{col}: decoded labels for {col}')
        sample = df[col].dropna().astype(str).unique()[:3].tolist()
        rows.append(f'{col}: {desc}; examples {sample}')
    context = f'Main file: {data.main}; rows: {len(df)}; joined: {joined or "none"}\nColumns in df:\n'
    context += '\n'.join(rows)
    context += '\nOriginal CSVs: ' + repr({f:list(x.columns) for f,x in tables.items() if f!='data_dictionary.csv'})
    return tables, df, context


SYSTEM_TABLE_COLUMN = (
    "Map the phrase in the question to exactly one CSV column. "
    "Return ONLY the column code, exactly as printed in Columns. "
    "Use the descriptions, not just similarity of short column codes. "
    "Do not generate pandas, explanations or extra text."
)


def column_request(task):
    """If parsing failed on a column name, give the LLM a small matching problem."""
    try:
        solve_data(task)
    except ColumnUnclear as e:
        description = '\n'.join(f'{col}: {desc}' for col, desc in e.columns.items())
        question = (
            f"Task: {task['objective']}\n"
            f"File: {e.filename}\n"
            f"Columns:\n{description}\n"
            f"Phrase: {e.phrase}"
        )
        return e, [
            {'role': 'system', 'content': SYSTEM_TABLE_COLUMN},
            {'role': 'user', 'content': question},
        ]
    except Exception:
        pass
    return None


def expression_intent(task, expression):
    """Basic semantic checks on model code, especially missing constraints.

    These checks are intentionally conservative: failure requests model repair.
    They cannot prove that every logically valid expression is correct.
    """
    q = task['objective'].lower()
    expression = re.sub(r'^```(?:python)?\s*|\s*```$', '', expression.strip()).strip()
    expression = re.sub(r'^answer\s*:\s*', '', expression, flags=re.I)
    tree = ast.parse(expression, mode='exec')
    nodes = list(ast.walk(tree))
    methods = {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    numbers = {float(n.value) for n in nodes if isinstance(n, ast.Constant)
               and isinstance(n.value, (int, float)) and not isinstance(n.value, bool)}
    numbers |= {-float(n.operand.value) for n in nodes
                if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub)
                and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float))}
    texts = [str(n.value) for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    source = ast.unparse(tree).lower()
    has_compare = any(isinstance(n, (ast.Compare, ast.BoolOp)) for n in nodes)
    has_mul = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult) for n in nodes)
    has_div = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in nodes)
    has_sub = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Sub) for n in nodes)
    aggregation = {'mean', 'median', 'sum', 'count', 'size', 'max', 'min', 'nunique', 'std', 'var'}
    has_selection = has_compare or bool(methods & {'loc','iloc','query','isin','eq','gt','ge','lt','le','between','where','filter'})
    problems = []
    if re.search(r'\brank(?:ing)?\b', q) and 'rank' not in methods:
        problems.append('rank groups using .rank(ascending=False, method="min")')
    if re.search(r'\b(?:second|third) (?:highest|largest|lowest|smallest)\b', q) and not methods.intersection({'nlargest','nsmallest','sort_values','sort_index','rank'}):
        problems.append('select the requested second or third group-level statistic, not the absolute maximum')
    if 'standard deviation' in q and 'std' not in methods:
        problems.append('use .std(ddof=0 for population, ddof=1 for sample)')
    if re.search(r'\bmedian\b', q) and not methods.intersection({'median','nanmedian','quantile'}):
        problems.append('use the median rather than the mean or maximum')
    if re.search(r'\b(?:difference|gap) between\b',q) and not (has_sub or 'diff' in methods):
        problems.append('calculate the requested difference by subtraction')
    if re.search(r'\b(?:row by row|per observation|per record)\b', q) and re.search(r'\b(?:ratio|divided by)\b',q) and not (has_div or 'divide' in methods):
        problems.append('calculate a ratio on each row before aggregation')
    if re.search(r'\b(?:have|has|with|whose) (?:a )?(?:recorded|nonmissing|missing)\b',q) and re.search(r'\b(?:fraction|percentage|percent)\b',q) and not methods.intersection({'notna','isna','notnull','isnull'}):
        problems.append('explicitly test recorded/missing values with .notna() or .isna()')
    if 'population variance' in q or 'sample variance' in q:
        if 'var' not in methods:
            problems.append('compute variance with .var()')
    if 'population standard deviation' in q or 'population variance' in q:
        if not re.search(r'ddof\s*=\s*0', source) and not {'nanstd','nanvar','std','var'}.isdisjoint(methods):
            problems.append('use ddof=0 for population variance or standard deviation')
    if re.search(r'\b\d+(?:st|nd|rd|th) percentile\b',q) and not methods.intersection({'quantile','percentile'}):
        problems.append('compute the requested quantile')
    percentile = re.search(r'\b(\d{1,3})(?:st|nd|rd|th) percentile\b', q)
    if percentile and 'quantile' in methods:
        p = int(percentile[1])/100
        if not any(abs(v-p)<1e-10 for v in numbers) and not any(abs(v-100*p)<1e-10 for v in numbers):
            problems.append(f'use percentile {percentile[1]} -> quantile({p})')
    if re.search(r'\b(?:distinct|different sites|unique values|unique sites)\b', q) and not methods.intersection({'nunique','unique','drop_duplicates'}):
        problems.append('count distinct values rather than rows')
    if re.search(r'\b(?:pearson|spearman|correlation)\b', q) and not methods.intersection({'corr','corrcoef'}):
        problems.append('calculate the requested correlation')
    if re.search(r'\b(?:interquartile range|iqr)\b', q) and not methods.intersection({'quantile','percentile'}):
        problems.append('use 75th percentile minus 25th percentile')
    if re.search(r'\bweighted (?:mean|average)\b', q) and not ((has_mul or 'multiply' in methods) and (has_div or 'divide' in methods) and 'sum' in methods):
        problems.append('weighted mean = sum(value * weight) / sum(valid weights), not ordinary .mean()')
    if re.search(r'\b(?:ratio of totals|total .+ divided by total)\b', q) and not (has_div and 'sum' in methods):
        problems.append('ratio of totals needs a division between two aggregated sums')
    if re.search(r'\b(?:growth|percent change|percentage change)\b', q) and not (has_div and has_sub and 100.0 in numbers):
        problems.append('growth percent = 100 * (new_total / old_total - 1)')
    if re.search(r'\b(?:per group|by site|by region|across sites|among sites|across regions|per geographic region|by the region)\b',q) and 'groupby' not in methods and 'pivot_table' not in methods:
        # "for each" is sometimes contextual text, but then groupby is still expected.
        problems.append('calculate the statistic by group before taking the final scalar')
    if re.search(r'\b(?:what (?:percentage|percent)|percentage of|percent of|percentage share|percentage growth)\b',q) and 100.0 not in numbers:
        problems.append('multiply the fraction by 100 to return percent units')
    if re.search(r'\b(?:where|among the (?!sites\b|regions\b|readings\b\s*$|readings\b.*\?$)|whose|with .+ of at least|at sites|at least \d|at most \d|above \d|below \d|after [a-z]+ \d|during \d{4}|classified as|for regular readings|for special readings)\b',q) and not has_selection:
        problems.append('apply the specified row filter before aggregating')
    # Thresholds must appear in the code as numeric constants or strings.
    threshold_pattern = (r'(?:strictly greater than|strictly less than|greater than|less than|'
                         r'at least|at most|above|below|more than|fewer than|over|under)\s+'
                         r'(-?\d+(?:[.,]\d+)?)')
    for raw in re.findall(threshold_pattern, q):
        threshold = float(raw.replace(',','.'))
        if not any(abs(n-threshold)<1e-10 for n in numbers) and not any(raw in txt for txt in texts):
            problems.append('use the explicit threshold '+raw)
    # Date-year checks reduce silent full-table averages.
    if re.search(r'\b(?:dated|recorded|during|in|from|after|before)\s+(?:\d{4}|[a-z]+ \d{1,2},? \d{4})',q):
        mentioned = re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',q)
        for yr in mentioned:
            if yr not in source:
                problems.append('apply the stated calendar year '+yr)
    if 'missing' in q and re.search(r'\b(?:percentage|fraction)\b',q) and not methods.intersection({'isna','notna','isnull','notnull','count'}):
        problems.append('test missingness explicitly using .isna() or .notna()')
    if problems:
        raise ValueError('Missing or unchecked intent: '+'; '.join(dict.fromkeys(problems)))


def table_context(task, tables, df, original):
    """Give the model reliable category values, dtypes and sample rows."""
    lines = [original]
    for col in df.columns:
        series = df[col]
        nonmissing = series.dropna()
        if nonmissing.empty:
            lines.append(f'{col}: all missing')
            continue
        if pd.api.types.is_numeric_dtype(series):
            lines.append(f'{col}: numeric; nonmissing {len(nonmissing)}/{len(series)}; min {nonmissing.min()}, max {nonmissing.max()}')
        else:
            vc = nonmissing.astype(str).value_counts()
            if len(vc) <= 16:
                sample = [(k, int(v)) for k,v in vc.items()]
                lines.append(f'{col}: categories/value counts {sample}')
            else:
                lines.append(f'{col}: strings; example values {nonmissing.astype(str).iloc[:5].tolist()}')
    sample = pd.concat([df.head(2),df.tail(1)],axis=0).to_dict(orient='records')
    lines.append('Example rows (not the full data): '+str(sample)[:1000])
    lines.append('Use df for real calculations; examples are NOT sufficient to calculate the answer.')
    return '\n'.join(lines)[:5000]


# ================== STRUCTURED TABLE PLAN FALLBACK (V5) ==================

"""Typed table-plan executor for numerically answering CSV questions.

The small LLM selects columns and operations; every actual calculation, filter,
join and aggregation happens in pandas. No model-generated code is evaluated.
"""
import json
import math
import re

import numpy as np
import pandas as pd


class PlanError(ValueError):
    pass


SYSTEM = (
    'Convert a numerical question about CSV tables into ONE compact JSON calculation plan. '
    'Do NOT solve the question mentally. Python will execute your JSON over the FULL CSV tables. '
    'Use only exact column codes in SCHEMA. A joined column can be filtered like an ordinary column. '
    'Reply with ONE JSON object, no markdown, prose or Python. '
    'Allowed op: stat, group, fraction, ratio, row_ratio, weighted, growth, profit, correlation, difference. '
    'stat: {"op":"stat","col":"CODE","agg":"mean|sum|median|max|min|count|nunique|size|std|var|quantile|iqr|range|prod|missing|recorded",'
    '"q":0.75,"sample":true,"where":[]}. '
    'group: {"op":"group","col":"CODE","group":"CODE","agg":"sum|mean|median|size|count|nunique|max|min",'
    '"post":"max|min|second_max|second_min|spread|rank|difference|value", "target":"CODE_OR_CATEGORY","target2":"CODE_OR_CATEGORY","where":[]}. '
    'fraction: {"op":"fraction","col":"CODE_IF_SUM","agg":"size|sum",'
    '"num_where":[],"den_where":[],"percent":false}; apply numerator and denominator filters separately. '
    'ratio: {"op":"ratio","col":"NUM_CODE","col2":"DEN_CODE","where":[]}; ratio of sums on SAME valid rows. '
    'row_ratio: {"op":"row_ratio","col":"NUM_CODE","col2":"DEN_CODE","agg":"mean|median|sum","where":[]}; divide per row and drop zero denominators. '
    'weighted: {"op":"weighted","col":"VALUE_CODE","weight":"WEIGHT_CODE","where":[]}. '
    'growth: {"op":"growth","col":"CODE","date":"DATE_CODE","year1":2023,"year2":2024,"percent":true}. '
    'profit: {"op":"profit","col":"REVENUE_CODE","col2":"COST_CODE","agg":"sum|mean|per_row_mean",'
    '"weight":"STAFF_CODE_IF_DIVIDE_PER_ROW","where":[]}. '
    'difference: {"op":"difference","col":"CODE","group":"CODE","agg":"mean|sum", "target":"CAT1","target2":"CAT2"}. '
    'correlation: {"op":"correlation","col":"CODE","col2":"CODE","method":"pearson|spearman"}. '
    'Filter syntax: {"col":"CODE","cmp":">|>=|<|<=|==|!=|between|isna|notna|year|after|on_or_after|before|on_or_before",'
    '"value":NUMBER_OR_CATEGORY_OR_DATE,"value2":NUMBER_FOR_BETWEEN}. '
    'Do not invent codes. Missing values are excluded by pandas. '
    'For percentages set percent true; for fractions set false. '
    'To count rows use stat with agg=size (col optional). To count distinct sites use nunique on site ID. '
    'For percentage of rows matching a condition, use fraction num_where versus den_where. '
    'For GROUP rank, use target category and rank descending with min ties. '
    'For top group totals aggregate by group FIRST, then take post=max or second_max. '
    'For a date restriction use where date/year filters. ' 
)

DEMONSTRATIONS = [
 ('How many distinct sites have revenue above 200?  site_id: site; revenue: money',
  {"op":"stat","col":"site_id","agg":"nunique","where":[{"col":"revenue","cmp":">","value":200}]}),
 ('What fraction of readings with revenue recorded have revenue over 100? revenue: numeric',
  {"op":"fraction","num_where":[{"col":"revenue","cmp":">","value":100}],"den_where":[{"col":"revenue","cmp":"notna"}],"percent":False}),
 ('What is the percentage of readings with missing cost? cost: numeric',
  {"op":"fraction","num_where":[{"col":"cost","cmp":"isna"}],"den_where":[],"percent":True}),
 ('What is the maximum regional sum of revenue? region: region; revenue: amount',
  {"op":"group","group":"region","col":"revenue","agg":"sum","post":"max"}),
 ('What is the second-largest total of revenue by region? region: region; revenue: amount',
  {"op":"group","group":"region","col":"revenue","agg":"sum","post":"second_max"}),
 ('What is the rank of special by total revenue per classification? classification: classes; revenue: amount',
  {"op":"group","group":"classification","col":"revenue","agg":"sum","post":"rank","target":"special"}),
 ('What is the difference between the maximum and minimum average cost among sites? site: site; cost: amount',
  {"op":"group","group":"site","col":"cost","agg":"mean","post":"spread"}),
 ('What is the mean revenue for sites in region North? region: geographic region; revenue: amount',
  {"op":"stat","col":"revenue","agg":"mean","where":[{"col":"region","cmp":"==","value":"North"}]}),
 ('What is the difference in average revenue between trial and regular? class: classification; revenue: amount',
  {"op":"difference","group":"class","col":"revenue","agg":"mean","target":"trial","target2":"regular"}),
 ('What is the average revenue per unit calculated for each record? revenue: revenue; qty: units',
  {"op":"row_ratio","col":"revenue","col2":"qty","agg":"mean"}),
 ('What is the total revenue divided by total cost with both values present? revenue: revenue; cost: cost',
  {"op":"ratio","col":"revenue","col2":"cost"}),
 ('Compute growth percent in total revenue from 2023 to 2024. day: day; revenue: revenue',
  {"op":"growth","col":"revenue","date":"day","year1":2023,"year2":2024,"percent":True}),
 ('Find mean revenue weighted by quantity among regular readings. class: classification; revenue: amount; units: quantity',
  {"op":"weighted","col":"revenue","weight":"units","where":[{"col":"class","cmp":"==","value":"regular"}]}),
 ('Compute mean profit per staff member per reading. revenue: amount; cost: cost; staff: people',
  {"op":"profit","col":"revenue","col2":"cost","weight":"staff","agg":"per_row_mean"}),
 ('Calculate sample standard deviation of expense. expense: numeric',
  {"op":"stat","col":"expense","agg":"std","sample":True}),
]


def relevant_examples(question):
    q = question.lower()
    if re.search(r'weighted', q): inds = (12, 10)
    elif re.search(r'growth|percent change', q): inds = (11, 10)
    elif re.search(r'profit|net of|revenue minus', q): inds = (13, 10)
    elif re.search(r'rank|second|largest|highest|lowest|by site|by region|per region|across sites', q): inds = (3, 4, 5, 6)
    elif re.search(r'difference in|difference between.*(trial|regular|category)', q): inds = (8, 6)
    elif re.search(r'fraction|percentage|share', q): inds = (1, 2)
    elif re.search(r'per observation|per record|per unit|row by row', q): inds = (9, 10)
    elif re.search(r'divided by|ratio of totals', q): inds = (10, 9)
    elif re.search(r'standard deviation|variance', q): inds = (14, 7)
    elif re.search(r'distinct|unique', q): inds = (0, 7)
    else: inds = (7, 3)
    return [({'role':'user','content':s}, {'role':'assistant','content':json.dumps(a,separators=(',',':'))}) for i in inds for s,a in [DEMONSTRATIONS[i]]]


def prompt(task, context, frame):
    # Avoid including full rows: schemas, category codes and ranges fit in small prompt.
    q=task['objective']
    schema=context[:4200]
    examples=relevant_examples(q)
    return ([{'role':'system','content':SYSTEM}] +
            [message for pair in examples for message in pair] +
            [{'role':'user','content':'SCHEMA:\n'+schema+'\n\nQUESTION: '+q+'\nJSON plan:'}])


def parse(reply):
    text=str(reply).strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip()
    try:
        p=json.loads(text)
    except (json.JSONDecodeError,ValueError):
        start=text.find('{')
        if start < 0: raise PlanError('No JSON object')
        dec=json.JSONDecoder()
        try: p,_=dec.raw_decode(text[start:])
        except (ValueError,json.JSONDecodeError) as exc: raise PlanError('Invalid JSON: '+str(exc))
    if not isinstance(p,dict): raise PlanError('Plan must be a JSON object')
    return p


def _column(frame, plan, name, optional=False):
    value=plan.get(name)
    if value in (None,'', 'none') and optional:return None
    if not isinstance(value,str) or value not in frame.columns:
        raise PlanError(f'{name}: unknown column {value!r}; use exact schema code')
    return value


def _numeric(s):
    return pd.to_numeric(s,errors='coerce')


def _as_category_mask(series, value):
    v=str(value).strip().lower()
    return (series.astype('string').str.lower().str.strip()==v).fillna(False)


def _category_mask(frame,col,value):
    base=_as_category_mask(frame[col],value)
    decoded='meaning_'+col
    if decoded in frame.columns:
        base=base|_as_category_mask(frame[decoded],value)
    return base


def _filter_mask(frame, filters):
    if filters is None: filters=[]
    if not isinstance(filters,list) or len(filters)>12: raise PlanError('where must be a short list')
    mask=pd.Series(True,index=frame.index,dtype=bool)
    for condition in filters:
        if not isinstance(condition,dict): raise PlanError('Each filter must be a JSON object')
        col=_column(frame,condition,'col')
        cmp=str(condition.get('cmp','==')).lower().strip()
        series=frame[col]
        val=condition.get('value')
        if cmp in ('isna','missing'): this=series.isna()
        elif cmp in ('notna','recorded'): this=series.notna()
        elif cmp in ('year','year_eq'):
            date=pd.to_datetime(series,errors='coerce')
            this=date.dt.year.eq(int(val))
        elif cmp in ('after','before','on_or_after','on_or_before'):
            date=pd.to_datetime(series,errors='coerce')
            boundary=pd.Timestamp(val)
            this={'after':date.gt,'before':date.lt,'on_or_after':date.ge,'on_or_before':date.le}[cmp](boundary)
        elif cmp=='between':
            lo=float(val); hi=float(condition.get('value2'))
            this=_numeric(series).between(lo,hi,inclusive='both')
        elif cmp in ('==','=','eq','!=','ne','>','>=','<','<='):
            if cmp in ('==','=','eq','!=','ne'):
                if isinstance(val,(int,float)) and not isinstance(val,bool):
                    this=_numeric(series).eq(float(val))
                else:
                    this=_category_mask(frame,col,val)
                if cmp in ('!=','ne'): this=~this & series.notna()
            else:
                num=_numeric(series); x=float(val)
                this={'>':num.gt,'>=':num.ge,'<':num.lt,'<=':num.le}[cmp](x)
        else: raise PlanError('Unsupported comparison '+cmp)
        mask &= this.fillna(False).astype(bool)
    return mask


def _stat(frame, col, agg, plan):
    agg=str(agg or 'size').lower()
    if agg in ('size','rows','rowcount'):
        return float(len(frame))
    if col is None: raise PlanError('Aggregation needs a column')
    s=frame[col]
    if agg in ('missing','isna'): return float(s.isna().sum())
    if agg in ('recorded','notna','count'): return float(s.notna().sum())
    if agg in ('unique','distinct','nunique'): return float(s.nunique(dropna=True))
    nums=_numeric(s).dropna()
    if nums.empty: raise PlanError('No numeric observations')
    if agg in ('mean','average','avg'): return float(nums.mean())
    if agg in ('median',): return float(nums.median())
    if agg=='sum':return float(nums.sum())
    if agg=='max':return float(nums.max())
    if agg=='min':return float(nums.min())
    if agg=='prod':return float(nums.prod())
    if agg in ('std','stdev','standard_deviation'):return float(nums.std(ddof=1 if plan.get('sample') else 0))
    if agg in ('var','variance'):return float(nums.var(ddof=1 if plan.get('sample') else 0))
    if agg in ('quantile','percentile'):
        q=float(plan.get('q',.5));q=q/100 if q>1 else q
        if not 0<=q<=1:raise PlanError('Invalid quantile')
        return float(nums.quantile(q))
    if agg=='iqr':return float(nums.quantile(.75)-nums.quantile(.25))
    if agg in ('range','spread'):return float(nums.max()-nums.min())
    raise PlanError('Unsupported aggregation '+agg)


def _group(frame, plan):
    g=_column(frame,plan,'group')
    agg=str(plan.get('agg','size'))
    col=_column(frame,plan,'col',optional=agg=='size')
    grouped=frame.groupby(g,dropna=True)
    if agg=='size': values=grouped.size().astype(float)
    elif agg=='count': values=grouped[col].count().astype(float)
    elif agg in ('nunique','distinct'): values=grouped[col].nunique().astype(float)
    else:
        nums=_numeric(frame[col]);gr=nums.groupby(frame[g],dropna=True)
        operations={'sum':gr.sum,'mean':gr.mean,'median':gr.median,'max':gr.max,'min':gr.min,'std':gr.std}
        if agg not in operations: raise PlanError('Unsupported grouped aggregation '+agg)
        values=operations[agg]().dropna()
    if values.empty:raise PlanError('No groups')
    post=str(plan.get('post','max')).lower()
    if post=='max':return float(values.max())
    if post=='min':return float(values.min())
    if post in ('second_max','second_highest'):
        if len(values)<2:raise PlanError('Fewer than 2 groups')
        return float(values.sort_values(ascending=False,kind='stable').iloc[1])
    if post in ('second_min','second_lowest'):
        if len(values)<2:raise PlanError('Fewer than 2 groups')
        return float(values.sort_values(kind='stable').iloc[1])
    if post in ('spread','range','difference'):return float(values.max()-values.min())
    if post in ('rank','value'):
        target=plan.get('target')
        keys=values.index.astype(str).str.lower().str.strip()
        which=np.flatnonzero(keys == str(target).lower().strip())
        if len(which)!=1 and 'meaning_'+g in frame.columns:
            labels=frame[[g,'meaning_'+g]].dropna().drop_duplicates()
            matched=labels.loc[_category_mask(labels,'meaning_'+g,target),g]
            if len(matched)==1:
                which=np.flatnonzero(keys==str(matched.iloc[0]).lower().strip())
        if len(which)!=1:raise PlanError('Group '+str(target)+' not found')
        if post=='value':return float(values.iloc[which[0]])
        return float(values.rank(method='min',ascending=False).iloc[which[0]])
    if post=='sum':return float(values.sum())
    raise PlanError('Unsupported group post-operation '+post)


def execute(plan, frame):
    """Execute a verified plan on a joined DataFrame; never evaluate model code."""
    if not isinstance(plan,dict) or len(plan)>22:raise PlanError('Invalid plan')
    op=plan.get('op')
    if op not in ('stat','group','fraction','ratio','row_ratio','weighted','growth','profit','correlation','difference'):
        raise PlanError('Unknown operation '+str(op))
    if op=='fraction':
        den_mask=_filter_mask(frame,plan.get('den_where'))
        num=frame.loc[den_mask&_filter_mask(frame,plan.get('num_where'))]
        den=frame.loc[den_mask]
        agg=plan.get('agg','size')
        col=_column(frame,plan,'col',optional=agg=='size')
        n=_stat(num,col,agg,plan);d=_stat(den,col,agg,plan)
        if d==0:raise PlanError('Zero denominator')
        result=n/d*(100 if plan.get('percent') else 1)
    else:
        data=frame.loc[_filter_mask(frame,plan.get('where'))]
        if op=='stat':
            agg=plan.get('agg','size')
            result=_stat(data,_column(frame,plan,'col',optional=agg=='size'),agg,plan)
        elif op=='group':result=_group(data,plan)
        elif op=='correlation':
            col=_column(frame,plan,'col');col2=_column(frame,plan,'col2')
            method=plan.get('method','pearson')
            if method not in ('pearson','spearman'):raise PlanError('Invalid correlation method')
            result=_numeric(data[col]).corr(_numeric(data[col2]),method=method)
        elif op in ('ratio','row_ratio','weighted','profit'):
            col=_column(frame,plan,'col')
            rhs='weight' if op=='weighted' else 'col2'
            col2=_column(frame,plan,rhs)
            x=_numeric(data[col]);y=_numeric(data[col2]);valid=x.notna()&y.notna()
            if op=='weighted':
                x=x[valid];y=y[valid]
                if y.sum()==0:raise PlanError('Zero weight total')
                result=(x*y).sum()/y.sum()
            elif op=='ratio':
                x=x[valid];y=y[valid]
                if y.sum()==0:raise PlanError('Zero denominator sum')
                result=x.sum()/y.sum()
            elif op=='row_ratio':
                values=(x[valid]/y[valid].replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).dropna()
                result=_stat(pd.DataFrame({'result':values}),'result',plan.get('agg','mean'),plan)
            else:
                diff=(x[valid]-y[valid])
                mode=plan.get('agg','sum')
                if mode=='sum':result=diff.sum()
                elif mode=='mean':result=diff.mean()
                elif mode=='per_row_mean':
                    weight=_column(frame,plan,'weight')
                    z=_numeric(data[weight]).reindex(diff.index)
                    result=(diff/z.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).mean()
                else:raise PlanError('Invalid profit aggregation')
        elif op=='growth':
            c=_column(frame,plan,'col'); date=_column(frame,plan,'date')
            yrs=pd.to_datetime(data[date],errors='coerce').dt.year
            x=_numeric(data[c]); y1=int(plan['year1']);y2=int(plan['year2'])
            total_old=x[yrs==y1].sum();total_new=x[yrs==y2].sum()
            if total_old==0:raise PlanError('No valid baseline year revenue')
            result=(total_new/total_old-1)*(100 if plan.get('percent',True) else 1)
        elif op=='difference':
            g=_column(frame,plan,'group'); col=_column(frame,plan,'col');agg=plan.get('agg','mean')
            values=_numeric(data[col]).groupby(data[g]).agg(agg)
            keys=values.index.astype(str).str.lower().str.strip()
            terms=[]
            for target in ('target','target2'):
                which=np.flatnonzero(keys==str(plan.get(target,'')).lower().strip())
                if len(which)!=1 and 'meaning_'+g in data.columns:
                    labels=data[[g,'meaning_'+g]].dropna().drop_duplicates()
                    matched=labels.loc[_category_mask(labels,'meaning_'+g,plan.get(target)),g]
                    if len(matched)==1:
                        which=np.flatnonzero(keys==str(matched.iloc[0]).lower().strip())
                if len(which)!=1:raise PlanError('Group for '+target+' not found')
                terms.append(float(values.iloc[which[0]]))
            result=terms[0]-terms[1]
    if not math.isfinite(float(result)):raise PlanError('Nonfinite calculated result')
    return float(result)


def validate_semantics(task, plan, frame):
    """Reject common omissions without claiming formal natural-language equivalence."""
    q=task['objective'].lower()
    op=plan.get('op')
    if re.search(r'\b(?:per|by|across) (?:site|region|group|classification)s?\b|\bgroup readings\b|\bfor each site\b',q):
        if op not in ('group','difference') and not re.search(r'\bat sites? (?:on|in|with|whose)\b',q):
            raise PlanError('Question requires grouped aggregation')
    if re.search(r'\b(?:what fraction|what percentage|percentage of|fraction of|percentage share)\b',q):
        if op!='fraction':raise PlanError('Need a numerator and denominator for the requested fraction')
        wants_percent='percentage' in q or re.search(r'\bpercent\b',q)
        if bool(plan.get('percent',False))!=bool(wants_percent):
            raise PlanError('Percent versus fraction scaling mismatch')
    if re.search(r'\btotal\s+[\w -]{0,55}\s+divided by total\b',q) and op!='ratio':
        raise PlanError('Question requires a ratio of paired column totals, not row-level ratios')
    if re.search(r'\b(?:row by row|per record|per observation)\b',q) and re.search(r'\bdivided by\b',q) and op!='row_ratio':
        if not (op=='profit' and plan.get('agg')=='per_row_mean'):
            raise PlanError('Question requests a ratio on each row before averaging')
    if 'percentage share' in q and 'total recorded revenue' in q:
        if op!='fraction' or plan.get('agg')!='sum':
            raise PlanError('Revenue share must divide the selected revenue sum by total revenue')
    if re.search(r'\b(?:ranking|what rank)\b',q):
        if op!='group' or plan.get('post')!='rank':
            raise PlanError('Ranking requires group totals then descending rank')
    if re.search(r'\b(?:overall profit|total profit)\b',q):
        if op!='profit' or plan.get('agg')!='sum':
            raise PlanError('Total profit requires the sum of paired revenue minus cost')
    if re.search(r'\b(?:weighted average|weighted mean)\b',q) and op!='weighted':
        raise PlanError('Weighted mean requires values and weights')
    if re.search(r'\b(?:sample|population) standard deviation\b',q):
        if op!='stat' or plan.get('agg') not in ('std','stdev','standard_deviation'):
            raise PlanError('Use std sample/population aggregation')
        if bool(plan.get('sample')) != bool('sample' in q):
            raise PlanError('Sample std uses ddof=1, population std ddof=0')
    if re.search(r'\b(?:second highest|second largest|second-highest|second-largest)\b',q):
        if op!='group' or plan.get('post') not in ('second_max','second_highest'):
            raise PlanError('Second-largest group aggregation needed')
    if re.search(r'\b(?:distinct|unique|different sites)\b',q):
        if op=='stat' and plan.get('agg') not in ('nunique','distinct','unique'):
            raise PlanError('Distinct count needed')
    if re.search(r'\b(?:growth|percent change|percentage change)\b',q) and op!='growth':
        raise PlanError('Growth requires date-filtered totals of the two years')
    if re.search(r'\b(?:row by row|per observation|per record)\b',q) and re.search(r'\b(?:ratio|divided|profit)\b',q):
        if op not in ('row_ratio','profit'):
            raise PlanError('Question requires per-row calculation')
    # Check explicitly mentioned comparative thresholds against plan filters.
    fs=plan.get('where',[])+plan.get('num_where',[])+plan.get('den_where',[])
    if re.search(r'\bmedian\b',q) and not (op in ('stat','group') and plan.get('agg')=='median'):
        raise PlanError('Question requests median aggregation')
    if re.search(r'\b(?:range \(maximum minus minimum\)|interquartile range|\biqr\b)',q):
        if op!='stat' or plan.get('agg') not in ('range','iqr','spread'):
            raise PlanError('Question requests a range or an IQR')
    if re.search(r'\b(?:pearson|spearman|correlation)\b',q):
        if op!='correlation':raise PlanError('Question requests a correlation')
        if 'spearman' in q and plan.get('method')!='spearman':raise PlanError('Use Spearman correlation')
    if re.search(r'\b(?:missing|not recorded|recorded)\b',q) and op=='fraction':
        checks=plan.get('num_where',[])
        if 'missing' in q and not any(f.get('cmp') in ('isna','missing') for f in checks):
            raise PlanError('Missing percentage needs an isna numerator filter')
        if 'recorded' in q and 'not recorded' not in q and 'missing' not in q and not any(f.get('cmp') in ('notna','recorded') for f in checks):
            if 'total recorded' not in q:
                raise PlanError('Recorded percentage needs a notna numerator filter')
    if re.search(r'\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)',q):
        matched=re.search(r'\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)',q)
        found=any(f.get('cmp')=='between' and str(f.get('value'))==matched[1] and str(f.get('value2'))==matched[2] for f in fs)
        found=found or all(any(str(f.get('value'))==v for f in fs) for v in (matched[1],matched[2]))
        if not found:raise PlanError('Missing two-sided between condition')
    if re.search(r'\bafter\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',q):
        if not any(f.get('cmp') in ('after','on_or_after') for f in fs):
            raise PlanError('Missing date-after filter')
    if re.search(r'\b(?:for|as|among) (?:regular|trial|special)\b|\bclassified as (?:regular|trial|special)\b',q):
        requested=re.findall(r'\b(?:regular|trial|special)\b',q)
        all_values=[str(f.get('value','')).lower() for f in fs if isinstance(f,dict)] + [str(plan.get(k,'')).lower() for k in ('target','target2')]
        if not all(any(label==v for v in all_values) for label in requested):
            raise PlanError('Missing requested classification/category filter')
    vals=[]
    for f in fs:
        if not isinstance(f,dict):continue
        for v in (f.get('value'),f.get('value2')):
            try:vals.append(float(v))
            except (ValueError,TypeError):pass
    for raw in re.findall(r'\b(?:strictly greater than|strictly less than|at least|at most|more than|over|below|above|under|greater than|less than)\s+(\d+(?:\.\d+)?)\b',q):
        x=float(raw)
        if not any(abs(x-y)<1e-10 for y in vals):
            raise PlanError(f'Missing numeric threshold {raw}')
    # Verify that any literal category mentioned in the question appears in a
    # filter/target, rather than letting a plausible global statistic through.
    used_categories={str(f.get('value','')).lower().strip() for f in fs if isinstance(f,dict)}
    used_categories|={str(plan.get(k,'')).lower().strip() for k in ('target','target2')}
    stop_labels={'day','date','row','total','all','mean','sum','yes','no','true','false','missing','flat'}
    for code in frame.columns:
        values=frame[code]
        if pd.api.types.is_numeric_dtype(values) or values.nunique(dropna=True)>14:
            continue
        for label in values.dropna().astype(str).unique():
            val=label.lower().strip()
            if len(val)<3 or val in stop_labels or len(val)>35:
                continue
            if re.search(r'(?<!\w)'+re.escape(val)+r'(?!\w)',q) and val not in used_categories:
                # A categorical GROUP target is the only alternative to filtering.
                raise PlanError('Question names category '+repr(label)+'; add its filter or group target')
    for yr in re.findall(r'\b(?:19|20)\d{2}\b',q):
        if not any((f.get('cmp') in ('year','year_eq') and str(f.get('value'))==yr) or (f.get('cmp') in ('after','before','on_or_after','on_or_before') and str(f.get('value','')).startswith(yr)) for f in fs) and not (op=='growth' and yr in (str(plan.get('year1')),str(plan.get('year2')))):
            raise PlanError(f'Missing year restriction {yr}')
    return True