import os
import io
import re
import math
import ast
import numpy as np
import pandas as pd
import fit_solver as fit

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
    "df is already joined with a lookup table when possible; do not join again unless the context explicitly says no join was possible. "
    "Use df.loc[df['col'] > threshold, 'target'] rather than chained Boolean indexing. "
    "Prefer direct .mean(), .median(), .sum(), .max(), .min(), .count(), .nunique() or len. "
    "Apply every stated filter; do not compute an overall statistic when a subset is requested. "
    "Dates may need pd.to_datetime; use .notna() to exclude missing denominators. "
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
        ,('Question: What is the mean amount for records linked to a group with level at least 2? df has already been joined; columns: amount, level', "df.loc[df['level']>=2,'amount'].mean()")
        ,('Question: What fraction of records in group A have a recorded score? Columns: group, score', "df.loc[df['group']=='A','score'].notna().mean()")
        ,('Question: What is the mean amount for records dated on or after March 1, 2022? Columns: date, amount', "df.loc[pd.to_datetime(df['date'],errors='coerce')>=pd.Timestamp('2022-03-01'),'amount'].mean()")
        ,('Question: What is the 75th percentile of speed? Columns: speed', "df['speed'].quantile(0.75)")
        ,('Question: What is the population standard deviation of speed? Columns: speed', "df['speed'].std(ddof=0)")
    ]


COMPARISON_PATTERN=r'(strictly greater than|strictly less than|greater than|less than|of at least|of at most|at least|at most|on or after|before|below|above|after|in|is)'
DATE_TEXT_PATTERN=r'(?:[a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{4}|\d{4})'


def table_examples(task):
    """Choose two demonstrations; they are four user/assistant chat messages."""
    question = task['objective'].lower()
    files = task.get('files') or {}
    names = list(files) if isinstance(files, dict) else [os.path.basename(str(p)) for p in files]
    data_file_count = sum(name != 'data_dictionary.csv' for name in names)

    if data_file_count > 1:
        selected = (5, 7)
    elif 'rank' in question:
        selected = (2, 3)
    elif re.search(r'\d(?:st|nd|rd|th) percentile|standard deviation|quantile', question):
        selected = (10, 11)
    elif re.search(r'\b(?:born|listed|shipped|delivered|date of|dated)\b', question):
        selected = (0, 9)
    elif re.search(r'\bper\b|ratio|divided by', question):
        selected = (0, 4)
    elif re.search(r'\b(?:fraction|percentage|percent)\b', question):
        selected = (1, 8)
    elif re.search(r'\b(?:for each|group|by the)\b', question):
        selected = (2, 6)
    elif re.search(r'\b(?:how many|count|belong)\b', question):
        selected = (1, 5)
    else:
        selected = (0, 2)

    messages = []
    for idx in selected:
        prompt, answer = DEMO_table[idx]
        messages.extend((
            {'role': 'user', 'content': prompt},
            {'role': 'assistant', 'content': answer},
        ))
    return messages


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
        self.tables, _ = fit.read_files(task['files'], self.q)
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
            key=self.column('id',f, required=False)
            if not key:
                first=self.tables[f].columns[0]
                description=self.map.get(f,{}).get(first,'')
                if 'identifier' in description or description.endswith(' number'):
                    key=first
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
    query = re.sub(r'\s+(?:values?|columns?|fields?)$', '', query)
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

# check
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
    if s.str.contains(re.escape(v),regex=True,na=False).any():
        return s.str.contains(re.escape(v),regex=True,na=False)
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


def generic_table(d):
    typ=d.t.get('type') or infer_type(d.t)
    funcs={'table.stat':generic_stat,'table.filter':generic_filter,'table.group':generic_group,'table.derive':generic_derive,'table.join':generic_join}
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
    if 'logistic regression' in q:return 'fit.logistic'
    if 'linear regression' in q or 'least-squares' in q:return 'fit.linear'
    files=t.get('files',{})
    names=list(files) if isinstance(files,dict) else [os.path.basename(str(name)) for name in files]
    if len([name for name in names if name!='data_dictionary.csv'])>1:return 'table.join'
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
        (r'\bhighest\b', 'largest'),
        (r'\blowest\b', 'smallest'),
        (r'\bdetermine the\b', 'what is the')
    ]
    text = question
    # Normalize general question phrasing so one query plan covers several formulations.
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

def unused_category_labels(d):
    bad=[]
    for cols in d.map.values():
        for desc in cols.values():
            if 'coded (' not in desc:continue
            for code,label in re.findall(r'\b([a-z]{1,4}\d+|[a-z]{2,4})\s+([^,]+)',desc.split('coded (',1)[1].split(')',1)[0]):
                label=label.strip()
                if len(label)>=3 and re.search(r'\b'+re.escape(label)+r'\b',d.lower):
                    if not any(label in u or u in label or u==code for u in USED_VALUES):bad.append(label)
    return bad


def solve_data(t, strict=True):
    # Infer the operation after standardizing wording, not before.
    USED_VALUES.clear()
    task = {**t, 'objective': normalize_question(t['objective'])}
    typ = task.get('type') or infer_type(task)
    if not typ.startswith('table.'):
        raise ValueError('Not a table question')
    task['type'] = typ
    data = Data(task)
    result = generic_table(data)
    # A mathematically valid result is still wrong if an explicit restriction
    # was silently ignored by a pattern-based parser.
    if strict and typ != 'table.join' and data.filters_used == 0:
        q = task['objective'].lower()
        if re.search(r'\bamong the\b|\bconsider only\b', q):
            raise ValueError('Unapplied subset restriction')
        if re.search(r'\b(?:strictly greater than|strictly less than|at least|at most)\s+-?\d', q):
            raise ValueError('Unapplied numeric filter')
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
    """Reject generated expressions that omit essential, explicit task operations."""
    q = task['objective'].lower()
    expression = re.sub(r'^```(?:python)?\s*|\s*```$', '', expression.strip()).strip()
    expression = re.sub(r'^answer\s*:\s*', '', expression, flags=re.I)
    try:
        tree = ast.parse(expression, mode='exec')
    except SyntaxError:
        return
    nodes = list(ast.walk(tree))
    methods = {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    numbers = {float(n.value) for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))}
    numbers |= {-float(n.operand.value) for n in nodes if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Constant) and isinstance(n.operand.value, (int, float))}
    literals = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    has_division = any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div) for n in nodes)
    needed = []
    if re.search(r'\brank (?:of|the)\b', q) and 'rank' not in methods:
        needed.append('rank aggregated values with .rank()')
    if 'standard deviation' in q and 'std' not in methods:
        needed.append('use .std() for standard deviation')
    if '75th percentile' in q and not methods.intersection({'quantile', 'percentile'}):
        needed.append('use .quantile(0.75) for the percentile')
    if re.search(r'\bconsider\b[^?]+\bdivided by\b', q) and not has_division:
        needed.append('divide numerator by denominator per row')
    if re.search(r'\b(?:for each|per group)\b', q) and not methods.intersection({'groupby', 'pivot_table'}):
        needed.append('aggregate each group separately')
    if re.search(r'\bwhat (?:percentage|percent)\b', q) and 100.0 not in numbers:
        needed.append('multiply the fraction by 100 for a percentage')
    # Check explicit numeric thresholds, not unrelated historical numbers.
    pattern = r'(?:strictly greater than|strictly less than|greater than|less than|at least|at most|above|below|more than|fewer than)\s+(-?\d+(?:[.,]\d+)?)'
    for raw in re.findall(pattern, q):
        threshold = float(raw.replace(',', '.'))
        if threshold not in numbers and not any(raw in value for value in literals):
            needed.append(f'apply the threshold {raw}')
    if re.search(r'\bamong the\b', q) and not methods.intersection({'loc', 'isin', 'eq', 'where', 'groupby', 'query'}):
        if not any(isinstance(n, ast.Compare) for n in nodes):
            needed.append('filter to the specified population before aggregating')
    if needed:
        raise ValueError('Missing required operation: ' + '; '.join(dict.fromkeys(needed)))


def table_context(task, tables, df, original):
    """Give the LLM a compact view of full rows as well as separate column examples."""
    # Rows reveal which codes co-occur, helping prevent wrong filter/measure combinations.
    preview = df.head(2).to_dict(orient='records')
    summary = f'\nExample rows of df: {preview}'
    return original + summary[:1000]
