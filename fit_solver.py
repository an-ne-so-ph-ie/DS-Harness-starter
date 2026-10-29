import operator
import re
import numpy as np
import pandas as pd
import io
import os
from sklearn.linear_model import LogisticRegression

SYSTEM_fit = (
        "Match a phrase from a task to one column of a table. "
        "Input: 1) name of the columns each as CODE: description, "
        "       2) phrase as Phrase: phrase."
        "Answer with the CODE only."
        )
DEMO_fit = [
    {'role': 'user', 'content': 'Columns:\nq1: Library branch\nq2: Age of the reader (years)\n'
                                'q3: Books taken home this month\n\nPhrase: reader age'},
    {'role': 'assistant', 'content': 'q2'},
    {'role': 'user', 'content': 'Columns:\nq1: Library branch\nq2: Age of the reader (years)\n'
                                'q3: Books taken home this month\n\nPhrase: monthly borrowing'},
    {'role': 'assistant', 'content': 'q3'},
]


class Unclear(ValueError):
    # if some part of a sentence are not completely clear and help of llm is needed to return the right column name
    def __init__(self, unresolved_phrases, candidate_columns):
        super().__init__(f"unclear columns: {unresolved_phrases}")
        self.phrases = unresolved_phrases  
        self.columns = candidate_columns 


def read_files(input_files, objective=''):

    source_files = input_files.items() if isinstance(input_files, dict) else (
        (os.path.basename(str(file_path)), open(file_path, encoding='utf-8').read()) for file_path in input_files)
    stated = re.findall(r"missing values are written((?:\s*(?:,|or|and)?\s*'[^']+')+)", objective, re.I)
    missing = set(['', 'na', 'n/a', 'missing', 'null', 'nan', '?', '-']) | {t.lower() for part in stated for t in re.findall(r"'([^']+)'", part)}
    tables_by_filename = {}
    for file_name, file_content in source_files:
        csv_header = file_content.splitlines()[0]
        # either , or ; depending on what is more often
        sep = ';' if csv_header.count(';') > csv_header.count(',') else ','
        # read file with sep
        df = pd.read_csv(io.StringIO(file_content), sep=sep, dtype=str, keep_default_na=False)
        df.columns = df.columns.str.strip()
        for col in df:
            s = df[col].astype('string').str.strip()
            s = s.mask(s.str.lower().isin(missing))
            # saves every unit if in format number xxxx (and currency sign in front)
            numeric = s.str.replace(re.compile(r"^[€$£]?\s*(-?\d+(?:[.,]\d+)?)\s*[^\d\s.,]{0,4}$"), r"\1", regex = True).str.replace(',', '.', regex=False)
            converted = pd.to_numeric(numeric, errors='coerce')
            df[col] = converted if s.notna().sum() == converted.notna().sum() else s
        tables_by_filename[file_name] = df

    # prepare dictionary of column names (or optional data_dictionary.csv)
    col_desc, d = {}, tables_by_filename.get('data_dictionary.csv')

    if d is not None: 
        dic_col_names = {c.lower().strip(): c for c in d.columns}
        col = d[dic_col_names.get('column') or dic_col_names.get('name') or dic_col_names.get('variable') or dic_col_names.get('code') or d.columns[0]].astype(str)
        # look for any description
        text = d[next((c for k, c in dic_col_names.items() if 'desc' in k), d.columns[-1])].astype(str)
        file = dic_col_names.get('file') or dic_col_names.get('table')
        if file:
            col_desc = dict(zip(zip(d[file].astype(str), col), text))
        else:
            col_desc = {(n, c): x for n, t in tables_by_filename.items() for c, x in zip(col, text) if c in t.columns}
    return tables_by_filename, col_desc

def clean_frame(df, objective, desc): # clean frame from table_solver but changed s.t. usable for fit
    m = re.search(r'(?:a|an|the)\s+([\w\- ]+?)\s+of\s+-1\s+means|(?:the )?value -1 in the\s+([\w\- ]+?)\s+column', objective.lower())
    hits = [c for c in df if m and stems(m[1] or m[2]) <= stems(desc.get(c, c))]
    if len(hits) == 1:
        df = df.copy()
        df[hits[0]] = df[hits[0]].mask(pd.to_numeric(df[hits[0]], errors='coerce').eq(-1))
    return df

comp_operators = {'at least': operator.ge, 'at most': operator.le, 'above': operator.gt, 'below': operator.lt,
       'greater than': operator.gt, 'strictly greater than': operator.gt, 'greater than or equal to': operator.ge,
       'less than': operator.lt, 'strictly less than': operator.lt, 'less than or equal to': operator.le}
comp_operators_pattern = '|'.join(sorted(comp_operators, key=len, reverse=True)) # longer are first so that they are looked at first
to_ignore_words = {'the', 'of', 'a', 'an', 'to', 'for', 'in', 'and', 'with', 'by', 'on', 'or', 'versus', 'per', 'is', 'was'}

# get the pattern of a normal regression task, condition and question
reg_sentence_pattern = re.compile(
        r'fit an? (?P<model>[^.]*?regression)(?P<opts>[^.]*?) of (?:the )?(?P<target>.+?) on (?:the )?'
    r'(?P<features>.+?),\s*using (?:(?:the |all )?[\w\s-]+? of )?(?:the )?train\.csv(?P<filter>(?:[^.]|\.(?=\d))*)\.', re.I | re.S)

condition_pattern = re.compile(
    rf"(?:with an? |whose )(?P<col>.+?) (?:(?:of|is) (?P<year>\d{{4}}) or (?P<dir>earlier|later)"
    rf"|(?:of |is )?'(?P<word>[^']+)'|(?:of |is )?(?P<op>{comp_operators_pattern}) (?P<num>-?\d+(?:\.\d+)?))", re.I) #check for comparison

question_patterns = [
    ('r2', r'\bR(?:Â?²|\^?2)\b|coefficient of determination|r-squared'), #R², RÂ², R^2, R2, coefficient of determination and r-squared
    ('intercept', r'\bintercept\b'),
    ('coef', r'coefficient of (?:the )?(?P<phrase>.+?)(?: \(|\?)'),
    ('count', rf'how many [\w\s-]+? of test\.csv have a predicted .+? (?:of )?(?P<op>{comp_operators_pattern}) (?P<thr>-?\d+(?:\.\d+)?)'),
    ('mean', r'mean predicted .+? (?:over|of|for) (?:the )?[\w\s-]+? of test\.csv(?P<cond>[^?]*)\?'),
    ('predict', r'predicted .+? for (?:the )?[\w\s-]+? (?P<id>[\w-]+) of test\.csv'),
]

def stems(text):
    # in lower letters, erase stuff in brackets, ignores the ignore_words, seperate in words
    return frozenset(w[:5] for w in re.findall(r'[a-z0-9]+', re.sub(r'\(.*?\)', ' ', text.lower())) if w not in to_ignore_words)
 
 
def condition(text):
    text = text.strip(' ,.;')
    if not text:
        return None
    # m stands for condition_match
    if not (m := condition_pattern.fullmatch(text)): # check if the extracted condition is in our pattern
        raise ValueError(f'unknown condition: {text!r}')
    if m['year']:
        return m['col'], operator.le if m['dir'].lower() == 'earlier' else operator.ge, float(m['year'])
    if m['word'] is not None:
        return m['col'], operator.eq, m['word']
    # for the rest: get the operator from comp_operators
    return m['col'], comp_operators[m['op'].lower()], float(m['num'])

def conditions(text): # idea: allow multiple separated conditions 
    text = text.strip(' ,.;')
    if not text:
        return []
    parts = re.split(r',?\s+and\s+(?=(?:an?|the|whose|with)\s)', text)
    out = []
    for i, part in enumerate(parts):
        part = part.strip(' ,')
        if i and not re.match(r'(?:with an? |whose )', part, re.I):
            part = 'with ' + part
        out.append(condition(part))
    return out



# takes task and returns patterns and a structured model
def parse_fit(objective):
    # looks for a regression fit pattern
    m = reg_sentence_pattern.search(objective)
    if not m or not re.search(r'logistic|linear|least.squares', m['model'], re.I):
        raise ValueError('no fit sentence')
    # check which question was asked
    question = next((s for s in re.split(r'(?<=[.?])\s+', objective[m.end():]) if s.endswith('?')), '')
    found = next(((name, q) for name, pat in question_patterns if (q := re.search(pat, question, re.I))), None)
    if not found:
        raise ValueError(f'unknown question: {question!r}')
    question_type, group = found[0], found[1].groupdict()
    return {'logistic': 'logistic' in m['model'].lower(), 
            'intercept': not re.search(r'without (?:an )?intercept', m['opts'], re.I),
            'target': m['target'].strip(),
            'features': [p.strip().removeprefix('the ').strip()
                         for p in re.split(r',\s*(?:and\s+)?|\s+and\s+', m['features']) if p.strip()],
            'train_filter': conditions(m['filter']),
            'leak': re.findall(r'Do not use the columns? described as "([^"]+)"', objective),
            'dedupe': bool(re.search(r'duplicat|appear twice|count each .* once|one copy of each', objective, re.I)),
            'question': question_type, 'coef': group.get('phrase'), 'id': group.get('id'),
            'count': (comp_operators[group['op'].lower()], float(group['thr'])) if 'op' in group else None,
            'test_filter': conditions(group['cond']) if 'cond' in group else None}

# solve task without using an llm 
def solve_fit_task(task, model_suggested_columns=None):
    spec = parse_fit(task['objective'])
    tables, entries = read_files(task['files'], task['objective'])
    if 'train.csv' not in tables or 'test.csv' not in tables:
        raise ValueError('train.csv or test.csv missing')
    train, test = (clean_frame(tables[n], task['objective'], {c: entries.get((n, c), c) for c in tables[n]}) for n in ('train.csv', 'test.csv'))

    def describe(name, df):
        # look for id column (or take first one) and get description and returns only allowed columns
        desc = {c: entries.get((name, c), c) for c in df.columns}
        named = [c for c, x in desc.items() if re.search(r'\bidentifier\b|\bid\b', x, re.I)]
        idc = named[0] if len(named) == 1 else df.columns[0]
        leak = {stems(x) for x in spec['leak']}
        allowed = {c: stems(x) for c, x in desc.items() if c != idc and stems(x) not in leak}
        return desc, idc, allowed

    def resolve(phrases, desc, allowed, df=None, binary=None, numeric=()):
        # every part of a sentence gets a column (only allowed columns), if unclear it gets added to Unclear
        phrases = list(dict.fromkeys(phrases))
         # model_suggested_columns from llm is priorized if exists
        out = {ph: model_suggested_columns[ph] for ph in phrases if (model_suggested_columns or {}).get(ph) in allowed}
 
        def score(phrase_word_stems, head, column_word_stems):
            # get a score/count for each combination (same word, how much is identical in phrase and column).
            shared_word_count = len(phrase_word_stems & column_word_stems)
            return (column_word_stems == phrase_word_stems, shared_word_count / len(phrase_word_stems), head <= column_word_stems, shared_word_count / max(len(column_word_stems), 1))

        # first very strong threshold, then easy to be higher
        for threshold in (0.5, 1e-9):                       
            while True:
                # only looks for columns that aren't alreadx matched
                free = {c: s for c, s in allowed.items() if c not in out.values()}
                best = []     
                # for all phrases that are not yet matched to a column                              
                for ph in (ph for ph in phrases if ph not in out):
                    # idea: change into little token, then get a score and then match if good enough
                    p = stems(ph)
                    if not p:
                        continue
                    head = stems(re.sub(r'\(.*?\)', ' ', ph).split()[-1])   # last word counts as the main noun for unclear cases
                    scored = sorted(((score(p, head, s), c) for c, s in free.items()), reverse=True)
                    if scored and (scored[0][0][0] or scored[0][0][1] >= threshold) and \
                            (len(scored) == 1 or scored[1][0] < scored[0][0]):   # unique first place is necessary
                        best.append((scored[0][0], ph, scored[0][1]))
                if not best:
                    break
                # only taking the safest combination
                _, ph, col = max(best)   
                out[ph] = col

        # check if any remaining column code could be related to a remaining phrase
        free = [c for c in allowed if c not in out.values()]
        for ph in (ph for ph in phrases if ph not in out):  
            words = re.findall(r'[a-z]+', ph.lower())
            hits = [c for c in free if any(w.startswith(t) for t in re.findall(r'[a-z]{3,}', c.lower()) for w in words)] # same prefix
            if len(hits) == 1: # (only if there is only one match)
                out[ph] = hits[0]
                free.remove(hits[0])
        # check for binary columns
        if binary in phrases and binary not in out and df is not None:   
            two_labels = [c for c in free if df[c].nunique() == 2 and
                          set(df[c].dropna().astype(str).str.strip().str.lower()) <= set(yn_labels) | {'0.0', '1.0'}]
            if len(two_labels) == 1:
                out[binary] = two_labels[0]
                free.remove(two_labels[0])    
        # match any remaining numeric columns                            
        left = [ph for ph in numeric if ph not in out]                    
        free_num = [c for c in free if df is not None and pd.api.types.is_numeric_dtype(df[c])]
        if len(left) == 1 and len(free_num) == 1:
            out[left[0]] = free_num[0]
        # if still unclear, then an llm will be asked
        unclear = [ph for ph in phrases if ph not in out]
        if unclear:
            raise Unclear(unclear, {c: desc[c] for c in allowed if c not in out.values()})
        return out

    def where(df, cond, col):
        # returns the applied condition
        if not cond:
            return df
        _, comp_fct, value = cond
        s = df[col]
        s = s.astype('string').str.strip().str.lower() if comp_fct is operator.eq else pd.to_numeric(s, errors='coerce')
        return df[comp_fct(s, str(value).lower() if comp_fct is operator.eq else value).fillna(False)]

    # check for True / False patterns
    yn_labels = {'yes': 1, 'no': 0, 'y': 1, 'n': 0, 'true': 1, 'false': 0, '1': 1, '0': 0}
    for pat in (r'\(\s*(\w+)\s*=\s*1\s*,\s*(\w+)\s*=\s*0\s*\)', r'^(\w+)\s*\(1\)\s*(?:versus|vs\.?|or)\s*(\w+)\s*\(0\)'):
        if lm := re.search(pat, spec['target'], re.I):
            yn_labels.update({lm[1].lower(): 1, lm[2].lower(): 0})

    desc, idc, allowed = describe('train.csv', train)

    # are duplicated to be deleted?
    if spec['dedupe']:
        train = train.drop_duplicates(subset=idc)

    # match task to columns
    target, feats, trf = spec['target'], spec['features'], spec['train_filter']
    #cols = resolve([target, *feats] + ([trf[0]] if trf else []), desc, allowed,
    #                df=train, binary=target if spec['logistic'] else None, numeric=feats)
    cols = resolve([target, *feats] + [c[0] for c in trf], desc, allowed, df=train, binary=target if spec['logistic'] else None, numeric=feats)
    if len({cols[p] for p in [target, *feats]}) < len(feats) + 1:
        raise ValueError('two phrases point at the same column')

    sm = re.search(r'(?:a|an|the)\s+([\w\- ]+?)\s+of\s+-1\s+means|(?:the )?value -1 in the\s+([\w\- ]+?)\s+column', task['objective'].lower())
    sc = {k.lower(): v for k, v in cols.items()}.get((sm[1] or sm[2]).strip()) if sm else None
    if sc is not None:
        unrec = lambda df: df.assign(**{sc: df[sc].mask(pd.to_numeric(df[sc], errors='coerce').eq(-1))}) if sc in df.columns else df
        train, test = unrec(train), unrec(test)


    # convert into training data (into numeric, normalizing etc) such that a regression can be perfomed
    #rows = where(train, trf, cols.get(trf[0]) if trf else None)
    rows = train
    for filter_col, comp_fct, value in trf:
        rows = where(rows,(filter_col, comp_fct, value),cols[filter_col])
    X = rows[[cols[f] for f in feats]].apply(pd.to_numeric, errors='coerce')
    y = rows[cols[target]]
    if spec['logistic'] and pd.to_numeric(y, errors='coerce').notna().sum() < y.notna().sum():
        y_num = y.astype('string').str.strip().str.lower().map(yn_labels)
        if (y_num.isna() & y.notna()).any():
            raise ValueError('unknown labels in the target')
    else:
        y_num = pd.to_numeric(y, errors='coerce')
    if spec['logistic'] and not set(y_num.dropna().unique()) <= {0, 1}:
        raise ValueError('the target is not 0/1')
    keep = X.notna().all(axis=1) & y_num.notna()        
    if keep.sum() < len(feats) + 2:
        raise ValueError('too few training rows')
    X, y = X[keep].to_numpy(float), y_num[keep].to_numpy(float)

    # fitting the regression: intercept b0, coefficient b
    ones = np.ones((len(X), 1 if spec['intercept'] else 0))  # 1 if intercept, else 0
    if not spec['logistic']:
        w = np.linalg.lstsq(np.hstack([ones, X]), y, rcond=None)[0]
        b0, b = (w[0], w[1:]) if spec['intercept'] else (0.0, w)
    # logitsitc regression
    else:   
        mu = X.mean(0) * spec['intercept']
        sd = np.where(X.std(0) > 0, X.std(0), 1.0)
        m = LogisticRegression(C=np.inf, tol=1e-10, max_iter=10000, fit_intercept=spec['intercept']).fit((X - mu) / sd, y)
        # scale back
        b = m.coef_[0] / sd
        b0 = m.intercept_[0] - b @ mu if spec['intercept'] else 0.0

    def predict(A):
        z = b0 + A @ b
        return 1 / (1 + np.exp(-z)) if spec['logistic'] else z
 
    # return answer depending on the question_type
    q = spec['question']
    if q == 'intercept':
        return float(b0)
    if q == 'coef':
        k = [i for i, f in enumerate(feats) if stems(f) == stems(spec['coef'])]
        if len(k) != 1:
            raise ValueError(f"coefficient of what? {spec['coef']!r}")
        return float(b[k[0]])
    if q == 'r2':
        if spec['logistic']:
            raise ValueError('no R² for a logistic regression')
        return float(1 - ((y - predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum())
 
    # predict/ mean/ count on the test data
    tdesc, tidc, tallowed = describe('test.csv', test)
    tcols = {}
    for f in feats:
        same = [c for c, x in tdesc.items() if x == desc[cols[f]]]
        tcols[f] = same[0] if len(same) == 1 else cols[f] if cols[f] in test.columns else None
        if tcols[f] is None:
            raise ValueError(f'no column in test.csv for {desc[cols[f]]!r}')
    if q == 'predict':  # if only one line is needed
        ids = test[tidc]
        hit = (ids.astype('string').str.strip().str.lower() == spec['id'].lower()).fillna(False) \
            | (pd.to_numeric(ids, errors='coerce') == pd.to_numeric(spec['id'], errors='coerce'))
        sel = test[hit].iloc[:1]
        if sel.empty:
            raise ValueError(f"no row {spec['id']} in test.csv")
    else: # if more test rows are needed
        tf = spec['test_filter']
        #sel = where(test, tf, resolve([tf[0]], tdesc, tallowed)[tf[0]] if tf else None)
        if tf:
            sel = test
            for filter_col, comp_fct, value in tf:
                resolved_filter = resolve([filter_col], tdesc, tallowed)[filter_col]
                sel = where(sel, (filter_col, comp_fct, value), resolved_filter)
        else:
            sel = test

    Xt = sel[[tcols[f] for f in feats]].apply(pd.to_numeric, errors='coerce')
    if q == 'predict' and Xt.isna().any(axis=None):
        raise ValueError('the test row has a missing feature')
    Xt = Xt.dropna().to_numpy(float)
    if len(Xt) == 0:
        raise ValueError('no test rows')
    p = predict(Xt)
    value = p[0] if q == 'predict' else p.mean() if q == 'mean' else spec['count'][0](p, spec['count'][1]).sum()
    if not np.isfinite(value):
        raise ValueError('not a finite number')
    return float(value)
