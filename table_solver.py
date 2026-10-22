import os
import io
import re
import math
import ast
import numpy as np
import pandas as pd

def read_files(files, objective=''):
    if isinstance(files, dict):
        sources = list(files.items())
    else:
        sources = [(os.path.basename(str(p)), open(p, encoding='utf-8').read()) for p in files]
    frames = {}
    for name, content in sources:
        sep = ';' if content.splitlines()[0].count(';') > content.splitlines()[0].count(',') else ','
        frame = pd.read_csv(io.StringIO(content), sep=sep, dtype=str, keep_default_na=False)
        frame.columns = frame.columns.str.strip()
        for col in frame:
            s = frame[col].astype('string').str.strip()
            s = s.mask(s.str.lower().isin(['', 'na','n/a','missing','null','nan','?','-']))
            s = s.mask(s.str.match(r'^-1(?:[.,]0+)?$', na=False)) if '-1' in objective and 'not recorded' in objective.lower() else s
            numeric = s.str.replace(r'\s*(?:kg|kwh|pts|eur|km|k€|€|%|°c|mj|C|k)\s*$', '', regex=True, flags=re.I).str.replace(',', '.', regex=False)
            parsed = pd.to_numeric(numeric, errors='coerce')
            if s.notna().sum() and parsed.notna().sum() == s.notna().sum():
                frame[col] = parsed
            else:
                frame[col] = s
        frames[name] = frame
    return frames

class Data:
    def __init__(self, task):
        self.t = task
        self.q = task['objective']
        self.lower = self.q.lower()
        self.tables = read_files(task['files'], self.q)
        raw = self.tables.get('data_dictionary.csv')
        self.map = {}
        self.descriptions = {}
        if raw is not None:
            for row_index, r in raw.iterrows():
                name, col, desc = str(r['file']),str(r['column']),str(r['description']).lower()
                self.map.setdefault(name, {})[col] = desc
                self.descriptions[(name, col)] = desc
        self.files = [x for x in self.tables if x != 'data_dictionary.csv']
        if not self.files: raise ValueError('No data files')
        if 'train.csv' in self.files: self.main = 'train.csv'
        else: self.main = self.files[0]
        if re.search('duplicat|appear twice|count each .* once|one copy of each',self.lower):
            f=self.main
            key=self.column('id',f, required=False)
            if key: self.tables[f] = self.tables[f].drop_duplicates(subset=[key])
            else: self.tables[f] = self.tables[f].drop_duplicates()

    def column(self, key, file=None, required=True):
        file = file or self.main
        cols = list(self.tables[file].columns)
        descriptions = self.map.get(file, {})
        query = str(key).lower().strip()

        for col in cols:
            if col.lower() == query:
                return col

        exact = [col for col in cols if descriptions.get(col, '').strip() == query]
        if len(exact) == 1:
            return exact[0]

        tokens = lambda text: set(re.findall(r'[a-z0-9]+', text.lower())) - {
            'the', 'of', 'a', 'an', 'in', 'for', 'to', 'by', 'and', 'or', 'with'
        }
        wanted = tokens(query)
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
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    stop = {'the', 'a', 'an', 'of', 'for', 'in', 'by', 'to', 'with', 'from', 'that', 'each', 'per', 'is', 'and', 'or', 'number', 'total', 'mean', 'average'}
    return {w[:-1] if w.endswith('s') and len(w) > 4 else w for w in words if w not in stop}

WORD_EQUIV = {
    'average':'mean', 'averaged':'mean', 'largest':'maximum', 'highest':'maximum',
    'smallest':'minimum', 'lowest':'minimum', 'total':'sum',
    'born':'birth', 'shipped':'shipment', 'listed':'listing',
    'mathematics':'math', 'meters':'metre', 'metres':'metre',
    'people':'occupancy', 'persons':'occupancy', 'occupants':'occupancy',
    'enrolled':'enrol', 'billed':'bill', 'invoiced':'bill',
    'loaned':'loan', 'borrowed':'borrow', 'count':'number'
}

STOP_WORDS = set('number of the a an for to on over in with by and or from is are which whose that what its their as per at all each value values of this these those it there have has how many does do up of was were measured include includes coded columns column written answer rows file dataset date given recorded missing used the who belong belonging where among consider only'.split())

def word_tokens(value):
    ss = re.findall(r'[a-z0-9]+',str(value).lower())
    return {WORD_EQUIV.get(s[:-1] if s.endswith('s') and len(s)>4 else s, s[:-1] if s.endswith('s') and len(s)>4 else s) for s in ss if s not in STOP_WORDS and not s.isdigit()}

def date_column(frame, col, question=''):
    s = frame[col].astype('string')
    prefer_day = bool(re.search(r'dd/mm/yyyy|day/month/year|day first|day-first',question,re.I))
    try:
        out = pd.to_datetime(s, dayfirst=prefer_day, errors='coerce', format='mixed')
    except (ValueError,TypeError):
        out = pd.to_datetime(s, dayfirst=prefer_day, errors='coerce')
    return out

def resolve_column(d, phrase, name=None, numeric=None, date=None, exclude=()):
    name=name or d.main
    frame=d.tables[name]
    pp=str(phrase).strip().lower()
    target = word_tokens(pp)
    rank=[]
    descs=d.map.get(name,{})
    for col in frame:
        if col in exclude:continue
        s=frame[col]
        if numeric is True and not pd.api.types.is_numeric_dtype(s):continue
        if numeric is False and pd.api.types.is_numeric_dtype(s):continue
        desc=descs.get(col,'')
        label = f'{col} {desc}'
        tok=word_tokens(label)
        shared=target & tok
        score=(len(shared)/max(1,len(target)))*5 + len(shared)/max(1,len(tok))
        if pp==str(col).lower():score+=15
        if pp==desc.lower().strip():score+=12
        if pp in desc.lower() and len(pp)>=3:score+=4
        if re.search(r'\b'+re.escape(pp)+r'\b',str(col).lower()):score+=8
        if date is True:
            if re.search(r'\b(?:date|day|birth|year|listed|granted|placed)\b',desc.lower()):score+=5
            else:score-=9
        if date is False and ('date' in tok or 'birth' in tok) and not target.intersection({'date','birth','grant','listing'}):score-=2
        rank.append((score,col))
    rank.sort(reverse=True)
    if not rank or rank[0][0]<=0:
        raise ValueError(f'No column for {phrase} in {name}')
    if len(rank)>1 and rank[0][0]-rank[1][0]<0.1:
        raise ValueError(f'Ambiguous column for {phrase}: {rank[:3]}')
    return rank[0][1]

def numeric_values(d,df,col):
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):return pd.to_numeric(s,errors='coerce')
    return pd.to_numeric(s.astype('string').str.replace(',','.',regex=False).str.replace(r'\s*(?:kg|kwh|pts|eur|km|k€|€|%|°c|mj|c|k)\s*$','',regex=True,case=False),errors='coerce')

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

def category_mask(df, col, value, desc=''):
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

COMPARISON_PATTERN=r'(strictly greater than|strictly less than|greater than|less than|of at least|of at most|at least|at most|on or after|before|below|above|after|in|is)'

DATE_TEXT_PATTERN=r'(?:[a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}\s+[a-z]+\s+\d{4}|[a-z]+\s+\d{4}|\d{4})'

def filter_mask(d,df,phrase,name=None):
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
    m=re.search(r'for how many\s+\w+\s+is the\s+(.+?)\s+(?:missing|not recorded)',q)
    if m:
        col=resolve_column(d,m[1],d.main);return float(df[col].isna().sum())
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
        v=numeric_values(d,df,resolve_column(d,a,d.main,numeric=True))/numeric_values(d,df,resolve_column(d,b,d.main,numeric=True))
    else:
        m=re.search(r'consider the\s+(.+?)\s+in\s+([\w]+)\s*\((?:1\s+)?(?:[\w]+)\s*=\s*([\d.]+)\s+[\w]+\)',q)
        if not m:raise ValueError('No derived formula')
        col=resolve_column(d,m[1],d.main,numeric=True)
        factor=float(m[3]);v=numeric_values(d,df,col)*(1/factor if 'tonne' in m[2] else factor)
    m=re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\s+(.+?)\s*\(',q)
    if m:
        text=m[1]
        if 'listed' in m[0]:text='listed '+text
        if 'shipped' in m[0]:text='shipped '+text
        if 'born' in m[0]:text='born '+text
        v=v.loc[filter_mask(d,df,text)]
    m=re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\s+(.+?)\?',q)
    if m and not re.search(r'\b(?:round|report|give)\b',m[1]):
        try:v=v.loc[filter_mask(d,df,m[1])]
        except ValueError:pass
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
    col=resolve_column(d,metric,d.main,numeric=True)
    return aggregate_values(numeric_values(d,df,col),how)

def generic_group(d):
    q=d.lower
    df=source_frame(d)
    m=re.search(r'(?:consider only the\s+\w+\s+with|consider only the\s+\w+\s+of)\s+(.+?)\.\s+what',q)
    if m:df=df.loc[filter_mask(d,df,m[1])]
    m=re.search(r'\b(?:rank the|compute the)\s+(?:(mean|median|total|average|number of)\s+)?(.+?)\s+(?:for each|by the|by)\s+(.+?)(?:\.|,)',q)
    if not m:raise ValueError('Requires general table analysis')
    agg,metric,group=m.groups()
    if group.startswith('year of the '):
        date_phrase=group[len('year of the '):]
        gc=resolve_column(d,date_phrase,d.main,date=True)
        groups=date_column(df,gc,d.q).dt.year
    else:
        gc=resolve_column(d,group,d.main)
        groups=df[gc]
    if agg and agg.startswith('number of'):
        values=df.groupby(groups,dropna=True).size()
    else:
        cc=resolve_column(d,metric,d.main,numeric=True)
        nums=numeric_values(d,df,cc)
        gb=nums.groupby(groups,dropna=True)
        agg=(agg or 'total').lower()
        if agg in ('mean','average'):values=gb.mean()
        elif agg=='median':values=gb.median()
        else:values=gb.sum()
    if 'difference between the largest and the smallest' in q:return float(values.max()-values.min())
    if 'largest' in q or 'maximum' in q:return float(values.max())
    if 'smallest' in q or 'minimum' in q:return float(values.min())
    raise ValueError('Unrecognized grouped output')

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
    m=re.search(r'(?:belong to a\s+\w+\s+(?:with an? |whose )|belong to a\s+\w+\s+whose )(.+?)(?:\?|\s*\()',q)
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

def lookup_column(data, phrase, name=None, numeric=None, exclude=()):
    name = name or data.main
    frame=data.tables[name]
    needle=str(phrase).strip().lower()
    tokens=schema_terms(needle)
    matches=[]
    for col in frame:
        if col in exclude:
            continue
        s=frame[col]
        if numeric is True and not pd.api.types.is_numeric_dtype(s):
            continue
        desc=data.map.get(name,{}).get(col,'')
        words=schema_terms(col+' '+desc)
        common=len(tokens&words)
        score=3*common/max(1,len(tokens))+common/max(1,len(words))
        if col.lower()==needle:
            score+=7
        if needle==desc.strip():
            score+=5
        if needle in desc and len(needle)>3:
            score+=2
        matches.append((score,col))
    matches.sort(reverse=True)
    if not matches or matches[0][0]<=0 or (len(matches)>1 and matches[0][0]-matches[1][0]<0.15):
        raise ValueError('Column uncertain: '+needle)
    return matches[0][1]

def table_numeric(df, col):
    s=df[col]
    if pd.api.types.is_numeric_dtype(s):
        return s
    return pd.to_numeric(s.astype('string').str.replace(',', '.', regex=False),errors='coerce')

def table_aggregation(series,how):
    s=pd.to_numeric(series,errors='coerce').replace([np.inf,-np.inf],np.nan)
    how=how.lower()
    if '75th percentile' in how or 'quantile' in how:
        return float(s.quantile(.75))
    if 'standard deviation' in how:
        return float(s.std(ddof=0 if 'population' in how else 1))
    if 'median' in how:
        return float(s.median())
    if 'mean' in how or 'average' in how:
        return float(s.mean())
    if 'minimum' in how or 'smallest' in how or 'lowest' in how:
        return float(s.min())
    if 'maximum' in how or 'largest' in how or 'highest' in how:
        return float(s.max())
    if 'sum' in how or 'total' in how:
        return float(s.sum())
    raise ValueError('Ambiguous aggregation')

def table_simple(task):
    data=Data(task)
    q=data.lower
    if len(data.files)!=1:
        raise ValueError('Multi-table task needs relational planning')
    if re.search(r'\b(for each|group|rank|among|whose|belong|by department|by store|by school|consider|over the .* with|over the .* listed|over the .* shipped|over the .* born|with an? .* of at|with an? .* strictly)\b',q):
        raise ValueError('Filtered/grouped/derived question needs query planning')
    df=data.frame()
    if re.search(r'\b(appear twice|count each .* once|duplicate)\b',q):
        df=df.drop_duplicates()
    missing=re.search(r'(?:how many|for how many)\s+\w+\s+(?:have|has|is|are)\s+(?:an? |the )?(missing|not recorded|no recorded|recorded)\s+(.+?)(?:\?|\.)',q)
    if missing:
        col=lookup_column(data,missing.group(2).strip())
        return float(df[col].notna().sum() if missing.group(1)=='recorded' else df[col].isna().sum())
    requested=re.search(r'what is the\s+(standard deviation|75th percentile|maximum|minimum|average|mean|median|total|sum of|largest|smallest)\s+(.+?)\s+(?:over all|of all|over the|of the|for all|\(|\?)',q)
    if not requested:
        raise ValueError('Question requires relational planning')
    how,phrase=requested.groups()
    col=lookup_column(data,phrase,numeric=True)
    return table_aggregation(table_numeric(df,col),how+' '+q)

def infer_type(t):
    q=t['objective'].lower()
    if 'logistic regression' in q:return 'fit.logistic'
    if 'linear regression' in q or 'least-squares' in q:return 'fit.linear'
    files=t.get('files',{})
    names=list(files) if isinstance(files,dict) else [os.path.basename(str(name)) for name in files]
    if len([name for name in names if name!='data_dictionary.csv'])>1:return 'table.join'
    if re.search(r'\b(for each|rank the|group the|by each)\b',q):return 'table.group'
    if re.search(r'\b(consider the|divided by|ratio|per person|per item|per day|per kilometre)\b',q):return 'table.derive'
    if re.search(r'\b(among the|whose|that have|with a|with an|listed before|shipped after|born in)\b',q):return 'table.filter'
    if re.search(r'\bof (?:the|a) [a-z -]{2,40}(?:site|school|store|department|carrier)\b',q):return 'table.filter'
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
    text=question
    for pattern,value in replacements:
        text=re.sub(pattern,value,text,flags=re.I)
    return text

def find_metric_column(data, phrase, file=None, numeric=None):
    file = file or data.main
    text = re.sub(r'\b(?:values?|fields?|of the|the|by|each)\b', ' ', phrase.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    alternative = text
    if 'school size' in text:
        alternative = 'pupils enrolled'
    elif 'carrier' in text:
        alternative = 'transport company'
    elif re.search(r'\bday of stay\b', text):
        alternative = 'length of the stay'
    for candidate in (alternative, text, phrase):
        try:
            return resolve_column(data, candidate, file, numeric=numeric)
        except ValueError:
            pass
    raise ValueError('Uncertain column: ' + phrase)

def match_category(series, description, value):
    values = series.astype('string').str.strip().str.lower()
    text = value.strip().lower()
    if values.eq(text).any():
        return text
    desc = description.lower()
    coded = re.search(r'coded\s*\(([^)]*)\)', desc)
    if coded:
        for piece in coded[1].split(','):
            parts = piece.strip().split(maxsplit=1)
            if len(parts) == 2:
                code, label = parts
                if text == label or (word_tokens(label) and word_tokens(label).issubset(word_tokens(text))):
                    return code.lower()
    raise ValueError('Unknown category ' + value)

def solve_table_missing(data):
    q = data.lower
    match = re.search(r'(?:how many|for how many)\s+\w+\s+(?:have|has|is|are)\s+(?:a\s+|the\s+)?(?:no recorded|missing(?: or not recorded)?|not recorded)\s+(.+?)\?', q)
    if not match:
        match = re.search(r'for how many\s+\w+\s+is\s+(?:the\s+)?(.+?)\s+(?:missing|not recorded)', q)
    if not match:
        return None
    col = find_metric_column(data, match[1])
    frame = source_frame(data)
    return float(frame[col].isna().sum())

def solve_table_rank(data):
    q = data.lower
    match = re.search(r'rank the (.+?) values by the\s+(mean|average|median|total|sum|number of)\s+(.+?)(?:,|\.)', q)
    target = re.search(r'what is the rank of (.+?)\?', q)
    if not match or not target:
        return None
    group_phrase, operation, metric_phrase = match.groups()
    if 'year of the' in group_phrase:
        date_phrase = group_phrase.split('year of the', 1)[1].strip()
        group_column = find_metric_column(data, date_phrase)
        group = date_column(source_frame(data), group_column, data.q).dt.year
        key = int(target[1].strip())
    else:
        group_column = find_metric_column(data, group_phrase)
        frame = source_frame(data)
        group = frame[group_column]
        if target[1].strip().isdigit():
            key = int(target[1].strip())
        else:
            key = match_category(group, data.map.get(data.main, {}).get(group_column, ''), target[1].strip().removeprefix('the '))
    frame = source_frame(data)
    if operation == 'number of':
        scores = frame.groupby(group, dropna=True).size()
    else:
        value_column = find_metric_column(data, metric_phrase, numeric=True)
        observations = numeric_values(data, frame, value_column)
        grouped = observations.groupby(group, dropna=True)
        if operation in ('mean', 'average'):
            scores = grouped.mean()
        elif operation == 'median':
            scores = grouped.median()
        else:
            scores = grouped.sum()
    rankings = scores.rank(ascending=False, method='min')
    if key not in rankings.index:
        candidates = [idx for idx in rankings.index if str(idx).lower() == str(key).lower()]
        if not candidates:
            raise ValueError('Requested category not in data')
        key = candidates[0]
    return float(rankings.loc[key])

def joined_data(data):
    other_files = [file for file in data.files if file != data.main]
    if len(other_files) != 1:
        raise ValueError('Requires exactly two relational tables')
    lookup_file = other_files[0]
    left = source_frame(data)
    right = clean_frame(data, data.frame(lookup_file), lookup_file)
    merged, (left_key, right_key) = join_on_best_keys(left, right, lookup_file, data.map.get(data.main, {}), data.map.get(lookup_file, {}))
    renamed = {col: col + '__' + lookup_file.removesuffix('.csv') if col in left.columns else col for col in right.columns if col != right_key}
    return merged, lookup_file, renamed

def solve_table_join_group(data):
    q = data.lower
    match = re.search(r'group the\s+\w+\s+by the (.+?) of their\s+\w+\s+and compute the\s+(mean|average|median|sum|total)\s+(.+?)\s+per group', q)
    if not match:
        return None
    requested = re.search(r'what is the\s+(largest|smallest|maximum|minimum) value', q)
    if not requested:
        return None
    merged, lookup_file, lookup_cols = joined_data(data)
    dimension = find_metric_column(data, match[1], lookup_file)
    measure = find_metric_column(data, match[3], data.main, numeric=True)
    grouped = numeric_values(data, merged, measure).groupby(merged[lookup_cols[dimension]], dropna=True)
    agg = match[2]
    values = grouped.mean() if agg in ('mean','average') else grouped.median() if agg == 'median' else grouped.sum()
    return float(values.max() if requested[1] in ('largest','maximum') else values.min())

def solve_table_join_filter(data):
    q = data.lower
    match = re.search(r'belong to an?\s+\w+\s+with\s+(?:an?\s+)?(.+?)\s+(of at least|at least|below|strictly less than|strictly greater than|at most|above|greater than|less than)\s+(-?\d+(?:\.\d+)?)', q)
    if not match:
        return None
    how = re.search(r'what is the (mean|average|median|total|sum|maximum|minimum)\s+(.+?)\s+of the', q)
    if not how and not re.search(r'how many\s+\w+', q):
        return None
    merged, lookup_file, lookup_cols = joined_data(data)
    filter_col = find_metric_column(data, match[1], lookup_file, numeric=True)
    threshold = float(match[3])
    observations = numeric_values(data, merged, lookup_cols[filter_col])
    operator_word = match[2]
    if operator_word in ('at least', 'of at least'):
        mask = observations >= threshold
    elif operator_word in ('below', 'strictly less than', 'less than'):
        mask = observations < threshold
    elif operator_word == 'at most':
        mask = observations <= threshold
    else:
        mask = observations > threshold
    selected = merged.loc[mask.fillna(False)]
    if how is None:
        return float(len(selected))
    metric_col = find_metric_column(data, how[2], data.main, numeric=True)
    return aggregate_values(numeric_values(data, selected, metric_col), how[1])

def solve_table_fraction(data):
    q = data.lower
    match = re.search(r'among the\s+\w+\s+whose\s+(.*?)\s+is\s+[\'\"]([^\'\"]+)[\'\"]\s*,\s*what\s+(fraction|percentage).*?\s+have\s+(?:a |an |the )?(.+?)\s+(on or after|before|after|at least|at most)\s+(.+?)\?', q)
    if not match:
        return None
    frame = source_frame(data)
    category_col = find_metric_column(data, match[1])
    category = match_category(frame[category_col], data.map.get(data.main, {}).get(category_col,''), match[2])
    rows = frame.loc[frame[category_col].astype('string').str.strip().str.lower().eq(category)]
    if rows.empty:
        raise ValueError('No matching category')
    target_column = find_metric_column(data, match[4])
    is_date = 'date' in match[4] or 'day' in match[4]
    if is_date:
        vals = date_column(rows, target_column, data.q)
        value = pd.to_datetime(match[6])
    else:
        vals = numeric_values(data, rows, target_column)
        value = float(match[6])
    comparison = match[5]
    if comparison == 'on or after' or comparison == 'at least':
        passed = vals >= value
    elif comparison == 'after':
        passed = vals > value
    elif comparison == 'at most':
        passed = vals <= value
    else:
        passed = vals < value
    return float(passed.mean() * (100 if match[3] == 'percentage' else 1))

def solve_table_ratio(data):
    q = data.lower
    match = re.search(r'consider the (.+?) per (.+?)\.', q)
    operation = re.search(r'what is its\s+(mean|average|median|minimum|maximum|sum|total)', q)
    if not match or not operation or re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\b', q):
        return None
    definition = re.search(r'\(the (.+?) divided by the (.+?)\)', q)
    if definition:
        numerator_phrase, denominator_phrase = definition.groups()
    else:
        numerator_phrase = match[1]
        denominator_phrase = re.sub(r'\s*\([^)]*\).*$', '', match[2]).strip()
    numerator = find_metric_column(data, numerator_phrase, numeric=True)
    denominator = find_metric_column(data, denominator_phrase, numeric=True)
    frame = source_frame(data)
    vals = numeric_values(data,frame,numerator) / numeric_values(data,frame,denominator)
    return aggregate_values(vals, operation[1])

def solve_table_extended(data):
    functions = (solve_table_missing, solve_table_rank, solve_table_join_group,
                 solve_table_join_filter, solve_table_fraction, solve_table_ratio)
    for function in functions:
        try:
            result = function(data)
            if result is not None and math.isfinite(float(result)):
                return float(result)
        except (ValueError, KeyError, IndexError, TypeError, ZeroDivisionError):
            continue
    return None

def solve_data(t):
    typ=t.get('type') or infer_type(t)
    if typ.startswith('fit.'):
        raise ValueError('Fit tasks belong to the fit solver')
    if not typ.startswith('table.'):
        raise ValueError('Unsupported type')
    normalized={**t,'objective':normalize_question(t['objective']),'type':typ}
    data=Data(normalized)
    q=data.lower
    if typ=='table.group' and 'consider only' in q:
        clause=re.search(r'(?:consider only the\s+\w+\s+with|consider only the\s+\w+\s+of)\s+(.+?)\.\s+what',q)
        if not clause:
            raise ValueError('Unrecognized group filter')
    if typ=='table.derive' and re.search(r'\bover the\s+\w+\s+with\b',q):
        clause=re.search(r'over the\s+\w+\s+(?:with|listed|shipped|born)\s+(.+?)\s*(?:\(|\?)',q)
        if not clause:
            raise ValueError('Unrecognized derived filter')
    if re.search(r'\b(?:no fewer|at minimum|not exceeding|exceeds|more than or equal)\b',q):
        raise ValueError('Unsupported relational phrase')
    candidate=solve_table_extended(data)
    if candidate is not None:
        return candidate
    if typ=='table.stat':
        try:
            return table_simple(normalized)
        except (ValueError,KeyError,IndexError,AttributeError,TypeError):
            pass
    return generic_table(data)


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

def semantic_tokens(text):
    text=re.sub(r'\([^)]*\)',' ',str(text).lower())
    changes={'avg':'average','highest':'maximum','lowest':'minimum','shipped':'shipment','born':'birth','bought':'purchase','paid':'payment','ages':'age','sales':'sale','amounts':'amount','quantities':'quantity','students':'student','customers':'customer','orders':'order'}
    stop={'the','a','an','of','in','to','by','on','is','was','are','each','and','for','from','or','with','whose','that','those','this','all','any','recorded','column','value','values','at','per','one','over','its','their'}
    words=re.findall(r'[a-z0-9]+',text)
    return {changes.get(w,w[:-1] if len(w)>4 and w.endswith('s') else w) for w in words if w not in stop and len(w)>1}

def semantic_column(data, text, merged, origins, numeric=None, exclude=()):
    target=semantic_tokens(text)
    choices=[]
    if not target:
        raise ValueError('Column phrase missing')
    for col in merged:
        if col in exclude or col.startswith('meaning_'):
            continue
        series=merged[col]
        if numeric is True and not pd.api.types.is_numeric_dtype(series):
            continue
        name,source=origins.get(col,(data.main,col))
        description=data.map.get(name,{}).get(source,'')
        words=semantic_tokens(col+' '+description)
        common=target&words
        if not common:
            continue
        score=2*len(common)/len(target)+len(common)/max(1,len(words))
        if str(col).lower()==text.lower().strip():
            score+=4
        if description.lower().strip()==text.lower().strip():
            score+=3
        choices.append((score,col))
    choices.sort(reverse=True)
    if not choices or choices[0][0]<1.3:
        raise ValueError('Cannot identify column '+text)
    if len(choices)>1 and choices[0][0]-choices[1][0]<0.15:
        raise ValueError('Ambiguous column '+text)
    return choices[0][1]

def decoded_category_columns(data, frame, origins):
    names=[]
    result=frame.copy()
    for column in list(frame):
        name,source=origins.get(column,(data.main,column))
        desc=data.map.get(name,{}).get(source,'')
        code_match=re.search(r'coded\s*\(([^)]*)\)',desc,flags=re.I)
        if not code_match:
            continue
        pairs={}
        for entry in code_match.group(1).split(','):
            tokens=entry.strip().split(maxsplit=1)
            if len(tokens)==2:
                pairs[tokens[0].strip().lower()]=tokens[1].strip()
        if not pairs:
            continue
        alias='meaning_'+re.sub('[^a-z0-9]+','_',column.lower()).strip('_')
        if alias in result:
            continue
        result[alias]=join_key(frame[column]).map({join_key(pd.Series([str(k)])).iloc[0]:v for k,v in pairs.items()})
        names.append((column,alias,pairs))
    return result,names

def semantic_table_data(task):
    data=Data(task)
    main=data.main
    frame=data.frame()
    origins={c:(main,c) for c in frame}
    joined=[]
    for filename in data.files:
        if filename==main:
            continue
        old_cols=set(frame)
        try:
            frame,(left,right)=join_on_best_keys(frame,data.frame(filename),filename,data.map.get(main,{}),data.map.get(filename,{}))
        except (ValueError,KeyError):
            continue
        suffix='__'+filename.removesuffix('.csv')
        for col in frame:
            if col not in old_cols:
                source=col[:-len(suffix)] if col.endswith(suffix) else col
                origins[col]=(filename,source)
        joined.append((filename,left,right))
    for col in frame:
        name,original=origins.get(col,(main,col))
        desc=data.map.get(name,{}).get(original,'')
        if re.search(r'\b(?:date|day of|date of|birth)\b',desc):
            if not pd.api.types.is_datetime64_any_dtype(frame[col]):
                parsed=date_column(frame,col,data.q)
                if parsed.notna().sum()>0 and parsed.notna().mean()>0.7:
                    frame[col]=parsed
    frame,labels=decoded_category_columns(data,frame,origins)
    return data,frame,origins,labels,joined

def semantic_aggregate(values,operation):
    numeric=pd.to_numeric(values,errors='coerce').replace([np.inf,-np.inf],np.nan)
    text=operation.lower()
    if 'percentile' in text:
        m=re.search(r'(\d+)(?:st|nd|rd|th)? percentile',text)
        if not m:
            raise ValueError('Percentile is missing')
        return float(numeric.quantile(int(m[1])/100))
    if 'standard deviation' in text:
        return float(numeric.std(ddof=0 if 'population' in text else 1))
    if 'median' in text:
        return float(numeric.median())
    if re.search(r'\b(mean|average)\b',text):
        return float(numeric.mean())
    if re.search(r'\b(sum|total)\b',text):
        return float(numeric.sum())
    if re.search(r'\b(maximum|largest|highest)\b',text):
        return float(numeric.max())
    if re.search(r'\b(minimum|smallest|lowest)\b',text):
        return float(numeric.min())
    raise ValueError('Unrecognized aggregation')

def semantic_filter(data,frame,origins,labels,fragment):
    text=fragment.lower()
    if re.search(r'\bbetween\b|\beither\b|\bexcept\b|\bexcluding\b',text):
        raise ValueError('Complex condition needs model')
    restrictions=[]
    pattern=r'(strictly greater than|strictly less than|greater than or equal to|less than or equal to|at least|at most|more than|less than|above|below|before|after|on or after|no less than|no more than)\s+(-?\d+(?:[.,]\d+)?)'
    for m in re.finditer(pattern,text):
        left=text[max(0,m.start()-130):m.start()]
        left=re.split(r'\b(?:with|whose|having|where|and|or|that|of)\b',left)[-1]
        left=re.sub(r'\b(?:a|an|the|is|are|was|were|number of)\b',' ',left).strip(' ,')
        if not left:
            raise ValueError('Filter column unclear')
        col=semantic_column(data,left,frame,origins)
        scalar=float(m[2].replace(',','.'))
        vals=frame[col]
        if pd.api.types.is_datetime64_any_dtype(vals):
            if scalar.is_integer() and 1800<=scalar<=2200:
                vals=vals.dt.year
            else:
                raise ValueError('Date comparison requires date')
        else:
            vals=pd.to_numeric(vals,errors='coerce')
        comparison=m[1]
        if comparison in ('strictly greater than','more than','above'):mask=vals>scalar
        elif comparison in ('strictly less than','less than','below','before'):mask=vals<scalar
        elif comparison in ('greater than or equal to','at least','on or after','no less than'):mask=vals>=scalar
        elif comparison in ('less than or equal to','at most','no more than'):mask=vals<=scalar
        else:mask=vals>scalar
        restrictions.append(mask)
    cat_patterns=[]
    for code_col,label_col,mapping in labels:
        for code,label in mapping.items():
            text_tokens=semantic_tokens(text)
            label_tokens=semantic_tokens(label)
            if label_tokens and label_tokens.issubset(text_tokens):
                cat_patterns.append((code_col,label_col,code,label))
    if len(cat_patterns)==1:
        col,label_col,code,label=cat_patterns[0]
        restrictions.append(join_key(frame[col]).eq(join_key(pd.Series([code])).iloc[0]))
    if len(cat_patterns)>1:
        raise ValueError('Multiple categorical conditions need model')
    for m in re.finditer(r"\b([a-z][a-z\s]{2,35})\s+(?:is|equals|as)\s+['\"]([^'\"]+)['\"]",text):
        col=semantic_column(data,m[1],frame,origins)
        restrictions.append(frame[col].astype('string').str.lower().str.strip().eq(m[2].lower()))
    if not restrictions:
        raise ValueError('No recognized conditions')
    mask=pd.Series(True,index=frame.index)
    for restriction in restrictions:
        mask=mask & restriction.fillna(False)
    return mask

def semantic_key_for_category(data,frame,origins,group_col,term):
    text=str(term).lower().strip(' .?').removeprefix('the ')
    if text.isdigit():
        return int(text)
    current=frame[group_col]
    matches=[value for value in current.dropna().unique() if str(value).strip().lower()==text]
    if len(matches)==1:
        return matches[0]
    name,original=origins.get(group_col,(data.main,group_col))
    description=data.map.get(name,{}).get(original,'')
    coding=re.search(r'coded\s*\(([^)]*)\)',description)
    if coding:
        for entry in coding[1].split(','):
            pair=entry.strip().split(maxsplit=1)
            if len(pair)==2 and semantic_tokens(pair[1]).issubset(semantic_tokens(text)):
                code=pair[0].lower()
                candidates=[v for v in current.dropna().unique() if join_key(pd.Series([v])).iloc[0]==join_key(pd.Series([code])).iloc[0]]
                if len(candidates)==1:
                    return candidates[0]
    raise ValueError('Unknown group '+term)

def semantic_group(data,frame,origins,labels):
    text=data.lower
    ranked=re.search(r'\brank the\s+(.+?)\s+values\s+by\s+the\s+(mean|average|median|sum|total|number of)\s+(.+?)(?:,|\.)',text)
    if ranked:
        group_phrase,method,metric=ranked.groups()
    else:
        grouped=re.search(r'\b(?:compute|calculate|find|determine)\s+(?:the\s+)?(mean|average|median|total|sum|number of)\s+(.+?)\s+(?:for each|by the|per)\s+(.+?)(?:,|\.|\?)',text)
        if not grouped:
            raise ValueError('Not a grouping question')
        method,metric,group_phrase=grouped.groups()
    group_phrase=re.sub(r'\bvalues?\b','',group_phrase).strip()
    frame2=frame
    only=re.search(r'\bconsider only\s+(?:the\s+)?(.+?)\.\s*(?:what|compute|find|determine)',text)
    if only:
        clause=only[1]
        if 'with ' in clause:
            clause=clause.split('with ',1)[1]
        frame2=frame.loc[semantic_filter(data,frame,origins,labels,clause)]
    if group_phrase.startswith('year of the '):
        column=semantic_column(data,group_phrase[len('year of the '):],frame,origins)
        dates=frame2[column]
        if not pd.api.types.is_datetime64_any_dtype(dates):
            dates=date_column(frame2,column,data.q)
        group=dates.dt.year
    else:
        column=semantic_column(data,group_phrase,frame,origins)
        group=frame2[column]
    if method=='number of':
        summary=frame2.groupby(group,dropna=True).size()
    else:
        metric_col=semantic_column(data,metric,frame,origins,numeric=True)
        vals=pd.to_numeric(frame2[metric_col],errors='coerce')
        grouped=vals.groupby(group,dropna=True)
        if method in ('mean','average'):summary=grouped.mean()
        elif method=='median':summary=grouped.median()
        else:summary=grouped.sum()
    rank_match=re.search(r'\b(?:what is|find|determine|compute)\s+(?:the\s+)?rank of\s+(.+?)\?',text)
    if rank_match:
        key=semantic_key_for_category(data,frame2,origins,column,rank_match[1])
        return float(summary.rank(ascending=False,method='min').loc[key])
    if re.search(r'difference between the largest and (?:the )?smallest',text):
        return float(summary.max()-summary.min())
    output=re.search(r'\b(?:what is|find|determine|compute)\s+(?:the\s+)?(largest|smallest|highest|lowest|maximum|minimum)\s+(?:of these values|of the results|group|value)',text)
    if output:
        return float(summary.max() if output[1] in ('largest','highest','maximum') else summary.min())
    raise ValueError('Group output not understood')

def semantic_ratio(data,frame,origins,labels):
    text=data.lower
    definition=re.search(r'\(the\s+(.+?)\s+divided by\s+(?:the\s+)?(.+?)\)',text)
    if not definition:
        raise ValueError('No explicit ratio definition')
    numerator=semantic_column(data,definition[1],frame,origins,numeric=True)
    denominator=semantic_column(data,definition[2],frame,origins,numeric=True)
    values=pd.to_numeric(frame[numerator],errors='coerce')/pd.to_numeric(frame[denominator],errors='coerce')
    operation=re.search(r'what is its\s+(mean|average|median|maximum|minimum|largest|smallest)',text)
    if not operation:
        raise ValueError('Missing ratio operation')
    filter_match=re.search(r'\bover the\s+\w+\s+(?:with|whose|that|listed|shipped|born)\s+(.+?)\s*(?:\?|\()',text)
    if filter_match:
        clause=filter_match[1]
        if re.search(r'\b(listed|shipped|born)\b',filter_match[0]):
            clause=filter_match[0].split('over the',1)[-1]
        mask=semantic_filter(data,frame,origins,labels,clause)
        values=values.loc[mask]
    unit_match=re.search(r'\(in (euros|thousand euros|tonnes|kg)(?:\s+per\s+[^)]*)?\)',text)
    if unit_match and 'data_dictionary.csv' in data.tables:
        raw=data.tables['data_dictionary.csv']
        if 'unit' in raw:
            name,source=origins[numerator]
            rows=raw.loc[(raw['file'].astype(str)==name)&(raw['column'].astype(str)==source)]
            if len(rows):
                source_unit=str(rows.iloc[0]['unit']).lower().strip()
                wanted=unit_match[1]
                units={'euros':1,'thousand euros':1000,'kg':1,'tonnes':1000}
                if source_unit in units and wanted in units and ({source_unit,wanted}<={'euros','thousand euros'} or {source_unit,wanted}<={'kg','tonnes'}):
                    values=values*units[source_unit]/units[wanted]
    return semantic_aggregate(values,operation[1])

def semantic_percentage(data,frame,origins,labels):
    text=data.lower
    pattern=r'\bamong\s+(?:the\s+)?(.+?),\s*what\s+(percentage|fraction)\b(?:\s+\([^)]*\))?\s+(?:have|belong|are|were|is)\s+(.+?)\?'
    match=re.search(pattern,text)
    if not match:
        raise ValueError('Not a percentage question')
    before,kind,after=match.groups()
    if 'with ' in before:before=before.split('with ',1)[1]
    if 'whose ' in before:before=before.split('whose ',1)[1]
    if 'of the ' in before:before=before.split('of the ',1)[1]
    population=frame.loc[semantic_filter(data,frame,origins,labels,before)]
    if population.empty:
        raise ValueError('Empty population')
    condition=semantic_filter(data,population,origins,labels,after)
    return float(condition.mean()*(100 if kind=='percentage' else 1))

def semantic_missing(data,frame,origins):
    text=data.lower
    missing=re.search(r'\b(?:how many|for how many)\s+\w+\s+(?:have|has|is|are)\s+(?:a |an |the )?(missing|not recorded|no recorded|recorded)\s+(.+?)\?',text)
    if missing:
        col=semantic_column(data,missing[2],frame,origins)
        return float(frame[col].notna().sum() if missing[1]=='recorded' else frame[col].isna().sum())
    missing=re.search(r'\bfor how many\s+\w+\s+is\s+(?:the\s+)?(.+?)\s+(?:missing|not recorded)\b',text)
    if missing:
        col=semantic_column(data,missing[1],frame,origins)
        return float(frame[col].isna().sum())
    raise ValueError('Not missing count')

def solve_table_semantic(task):
    data,frame,origins,labels,joined=semantic_table_data(task)
    question=data.q
    lower=question.lower()
    handlers=(
        lambda:semantic_missing(data,frame,origins),
        lambda:semantic_percentage(data,frame,origins,labels),
        lambda:semantic_group(data,frame,origins,labels),
        lambda:semantic_ratio(data,frame,origins,labels),
    )
    for solve in handlers:
        try:
            answer=float(solve())
            if math.isfinite(answer):return answer
        except (ValueError,KeyError,IndexError,TypeError,AttributeError,ZeroDivisionError):
            continue
    if 'rank of' in lower or 'rank the' in lower or re.search(r'\bjoin\b',lower):
        raise ValueError('Needs more precise group interpretation')
    if re.search(r'\b(?:how many|number of)\b',lower):
        count=re.search(r'\bhow many\s+\w+\s+(?:with|whose|that|have|are)\s+(.+?)\?',lower)
        if count:
            mask=semantic_filter(data,frame,origins,labels,count[1])
            return float(mask.sum())
    asking=re.search(r'\b(?:what is|compute|calculate|find|determine)\s+(?:the |its )?(maximum|minimum|largest|smallest|highest|lowest|mean|average|median|sum|total|standard deviation|\d+th percentile)\s+(.+?)(?:\?|\.)',lower)
    if not asking:
        raise ValueError('Unsupported question form')
    operation,rest=asking.groups()
    if re.search(r'\b(?:for each|grouped by|rank|per group)\b',rest):
        raise ValueError('Grouping needs explicit group parsing')
    portion=re.split(r'\b(?:of the|among the|over the|for the|where|with|whose|that)\b',rest)[0]
    portion=re.sub(r'\s*\([^)]*\).*','',portion).strip()
    col=semantic_column(data,portion,frame,origins,numeric=True)
    if re.search(r'\b(?:of the|among the|over the|for the|where|with|whose|that)\b',rest):
        filter_part=re.split(r'\b(?:of the|among the|over the|for the|where|with|whose|that)\b',rest,1)[1]
        try:
            mask=semantic_filter(data,frame,origins,labels,filter_part)
        except ValueError:
            if re.search(r'\b(?:over all|for all|all rows|all records)\b',rest):
                mask=pd.Series(True,index=frame.index)
            else:
                raise
        values=frame.loc[mask,col]
    else:
        values=frame[col]
    answer=semantic_aggregate(values,operation)
    if not math.isfinite(answer):
        raise ValueError('No finite numeric result')
    return answer

def prepare_table_semantic(task):
    data,frame,origins,labels,joined=semantic_table_data(task)
    notes=[]
    for col in list(frame):
        if col.startswith('meaning_'):
            continue
        name,source=origins.get(col,(data.main,col))
        description=data.map.get(name,{}).get(source,'')
        if description:
            cleaned=re.sub(r'\([^)]*\)',' ',description)
            words=[w for w in re.findall(r'[a-z0-9]+',cleaned.lower()) if w not in {'the','of','a','an','in','for','on','by','to','is','are','per','from'}]
            alias='_'.join(words[:5])
            if alias and alias not in frame and alias not in data.tables:
                frame[alias]=frame[col]
                notes.append(f'{alias} = {col} ({description})')
    detail=[]
    for col in frame:
        values=frame[col].dropna().astype(str).unique()[:3]
        name,source=origins.get(col,(data.main,col))
        desc=data.map.get(name,{}).get(source,'')
        detail.append(f'{col}: {desc or "derived field"}; dtype {frame[col].dtype}; examples {list(values)}')
    joined_info='; '.join(f'{name}: {left} -> {right}' for name,left,right in joined)
    code_info='; '.join(f'{original} codes: {mapping}; decoded as {label}' for original,label,mapping in labels)
    context=f'Main file: {data.main}; total rows: {len(frame)}\nJoined lookup: {joined_info or "none"}\nColumns in df:\n'+'\n'.join(detail)
    if code_info:context+='\nDecoded categories: '+code_info
    if notes:context+='\nSemantic aliases (both names work):\n'+'\n'.join(notes)
    context+='\nRaw tables: '+repr({name:list(value.columns) for name,value in data.tables.items() if name!='data_dictionary.csv'})
    return data.tables,frame,context
