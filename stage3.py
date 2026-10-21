import os
import io
import re
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import dsh

MODEL = os.environ.get('DSH_MODEL', dsh.DEFAULT_MODEL)
MAX_ROUNDS = 4
SYSTEM_calc = (
    "Solve the problem step by step. You have a calculator: to use it, write\n"
    '<call tool="calculator">EXPRESSION</call>\n'
    "and stop; the result comes back in the next message. EXPRESSION is Python arithmetic: "
    "numbers, + - * / ** // %, parentheses, and the functions sqrt, exp, log (natural), log10, "
    "comb, perm, factorial, floor, ceil, round. Use the calculator for every computation, "
    "one call at a time. When you know the final answer, write a last line of the form "
    "ANSWER: <number>, rounded as the question asks."
)
DEMO_calc = [
    {'role': 'user', 'content': 'A jacket costs 80 euros. What does it cost after a 15% discount, then a 20% tax? Round to 2 decimals.'},
    {'role': 'assistant', 'content': 'The discounted price is 80 × (1 − 0.15), then the tax multiplies it by 1.20.\n<call tool="calculator">80 * (1 - 0.15) * 1.20</call>'},
    {'role': 'user', 'content': 'Result: 81.6'},
    {'role': 'assistant', 'content': 'The final price is 81.6 euros.\nANSWER: 81.6'},
]

SYSTEM_fit = ()
DEMO_fit = []

SYSTEM_table = ()
DEMO_table = []


def read_files(files, objective=''):
    sources = files.items() if isinstance(files, dict) else ((os.path.basename(str(p)), open(p, encoding='utf-8').read()) for p in files)
    tables = {}
    for name, content in sources:
        header = content.splitlines()[0]
        sep = ';' if header.count(';') > header.count(',') else ','
        df = pd.read_csv(io.StringIO(content), sep=sep, dtype=str, keep_default_na=False)
        df.columns = df.columns.str.strip()
        for col in df:
            s = df[col].astype('string').str.strip()
            s = s.mask(s.str.lower().isin(['', 'na', 'n/a', 'missing', 'null', 'nan', '?', '-']))
            if '-1' in objective and 'not recorded' in objective.lower():
                s = s.mask(s.str.match(r'^-1(?:[.,]0+)?$', na=False))
            numeric = s.str.replace(r'\s*(?:kg|kwh|pts|eur|km|€|%|°c|mj)\s*$', '', regex=True, flags=re.I).str.replace(',', '.', regex=False)
            converted = pd.to_numeric(numeric, errors='coerce')
            df[col] = converted if s.notna().sum() == converted.notna().sum() else s
        tables[name] = df
    return tables


def tokens(text):
    return set(re.findall(r'[a-z0-9]+', text.lower())) - {'the', 'of', 'a', 'an', 'to', 'for', 'in', 'and', 'with', 'by', 'on', 'number'}


class Data:
    def __init__(self, task):
        self.question = task['objective']
        self.q = self.question.lower()
        self.tables = read_files(task['files'], self.question)
        self.files = [name for name in self.tables if name != 'data_dictionary.csv']
        self.main = 'train.csv' if 'train.csv' in self.files else self.files[0]
        self.descriptions = {}
        if 'data_dictionary.csv' in self.tables:
            for _, row in self.tables['data_dictionary.csv'].iterrows():
                self.descriptions[(row['file'], row['column'])] = str(row['description']).lower()
        if re.search(r'duplicat|appear twice|count each .* once|one copy of each', self.q):
            df = self.tables[self.main]
            self.tables[self.main] = df.drop_duplicates(subset=df.columns[0])

    def column(self, phrase, file=None):
        file = file or self.main
        columns = self.tables[file].columns
        phrase = phrase.lower().strip(' .')
        for col in columns:
            if col.lower() == phrase or phrase in self.descriptions.get((file, col), ''):
                return col
        terms = tokens(phrase)
        scores = [(len(terms & tokens(col + ' ' + self.descriptions.get((file, col), ''))), col) for col in columns]
        score, col = max(scores)
        if score == 0:
            raise ValueError(f'Column not found: {phrase}')
        return col

    def frame(self, file=None):
        return self.tables[file or self.main].copy()

    def values(self, df, phrase, file=None):
        return pd.to_numeric(df[self.column(phrase, file)], errors='coerce')


def aggregate(values, question):
    if re.search(r'\b(?:mean|average)\b', question):
        return values.mean()
    if 'median' in question:
        return values.median()
    if re.search(r'\b(?:maximum|max|largest)\b', question):
        return values.max()
    if re.search(r'\b(?:minimum|min|smallest)\b', question):
        return values.min()
    if 'standard deviation' in question:
        return values.std()
    if '75th percentile' in question:
        return values.quantile(.75)
    return values.sum()


def select_rows(data, df, question, file=None):
    pattern = r'(?:with (?:a|an|the) |whose |of at least |of at most )([a-z][a-z\s-]*?)\s+(strictly greater than|strictly less than|of at least|of at most|at least|at most|below|before)\s+(-?\d+(?:\.\d+)?)'
    for phrase, op, number in re.findall(pattern, question):
        values = data.values(df, phrase, file)
        limit = float(number)
        if op == 'strictly greater than':
            df = df[values > limit]
        elif op in ('strictly less than', 'below', 'before'):
            df = df[values < limit]
        elif op in ('of at most', 'at most'):
            df = df[values <= limit]
        else:
            df = df[values >= limit]
    return df


def stat_task(data):
    q = data.q
    df = data.frame()
    question = q.split('?', 1)[0]
    match = re.search(r'(?:maximum|minimum|median|mean|average|total|sum|recorded|no recorded)\s+(.+?)(?:\s+over\b|\s+of the\b|\?|$)', question)
    phrase = match.group(1) if match else question.split('have', 1)[-1]
    values = data.values(df, phrase)
    if 'no recorded' in question:
        return values.isna().sum()
    if 'how many' in question:
        return values.count()
    return aggregate(values, question)


def derive_task(data):
    q = data.q
    df = select_rows(data, data.frame(), q)
    match = re.search(r'\(the (.+?) divided by the (.+?)\)', q)
    if match:
        values = data.values(df, match.group(1)) / data.values(df, match.group(2))
    elif 'tonnes' in q:
        values = data.values(df, 'load weight') / 1000
    elif 'megajoules' in q:
        values = data.values(df, 'electricity use') * 3.6
    else:
        raise ValueError('No derivation found')
    return aggregate(values.replace([np.inf, -np.inf], np.nan), q.split('what is')[-1])


def filter_task(data):
    q = data.q
    df = select_rows(data, data.frame(), q)
    match = re.search(r'(?:maximum|minimum|median|mean|average|sum of|total)\s+(.*?)(?:\s+of the\b|\s+of the readings|\?|$)', q)
    if not match:
        raise ValueError('No requested measurement found')
    return aggregate(data.values(df, match.group(1)), q)


def group_task(data):
    q = data.q
    df = select_rows(data, data.frame(), q)
    match = re.search(r'(?:for each|rank the)\s+(.*?)(?:\.|,|\?| by\b| values\b)', q)
    by = data.column(match.group(1))
    metric = re.search(r'\b(?:total|mean|median|average)\s+(.+?)\s+(?:for each|by)\b', q)
    if metric:
        values = data.values(df, metric.group(1))
        kind = metric.group(0)
        grouped = values.groupby(df[by])
        result = grouped.mean() if 'mean' in kind or 'average' in kind else grouped.median() if 'median' in kind else grouped.sum()
    else:
        result = df.groupby(by).size()
    if 'difference between the largest and the smallest' in q:
        return result.max() - result.min()
    if 'rank of' in q:
        target = re.search(r'rank of (.*?)(?:\?|\.)', q).group(1).strip()
        return result.rank(ascending=False, method='min').loc[target]
    return aggregate(result, q.split('what is')[-1])


def join_task(data):
    left, right = data.files[:2]
    a, b = data.frame(left), data.frame(right)
    match = max(((len(tokens(data.descriptions.get((left, x), x)) & tokens(data.descriptions.get((right, y), y))), x, y) for x in a for y in b))
    _, left_key, right_key = match
    a['_key'] = a[left_key].astype(str).str.lower().str.replace(r'^(site-|c0*)', '', regex=True).str.strip()
    b['_key'] = b[right_key].astype(str).str.lower().str.replace(r'^(site-|c0*)', '', regex=True).str.strip()
    df = a.merge(b, on='_key', how='inner', suffixes=('', '_right'))
    df = select_rows(data, df, data.q, right)
    match = re.search(r'(?:median|mean|average|total|sum of)\s+(.+?)\s+of the\b', data.q)
    if 'how many' in data.q:
        return len(df)
    return aggregate(data.values(df, match.group(1), left), data.q)


def fit_task(data, logistic):
    q = data.q
    match = re.search(r'\bof (.+?) on (.+?),\s+using\b', q)
    target = re.sub(r'\s*\(.*', '', match.group(1)).strip()
    feature_names = [s.strip().removeprefix('the ') for s in re.split(r',\s*|\s+and\s+', match.group(2))]
    train = data.frame('train.csv')
    year = re.search(r'year(?:\s+granted|\s+of\s+admission)?\s+of\s+(\d{4})\s+or\s+(earlier|later)', q)
    if year:
        years = data.values(train, 'year', 'train.csv')
        train = train[years <= int(year.group(1))] if year.group(2) == 'earlier' else train[years >= int(year.group(1))]
    features = [data.column(name, 'train.csv') for name in feature_names]
    ycol = data.column(target, 'train.csv')
    x = train[features].apply(pd.to_numeric, errors='coerce')
    y = train[ycol]
    if logistic:
        y = y.astype('string').str.lower().replace({'yes': 1, 'no': 0, 'y': 1, 'n': 0})
    y = pd.to_numeric(y, errors='coerce')
    keep = x.notna().all(axis=1) & y.notna()
    x, y = x[keep], y[keep]
    model = LogisticRegression(C=1e12, max_iter=5000, tol=1e-10) if logistic else LinearRegression()
    model.fit(x, y.astype(int) if logistic else y)
    tail = q.split('what is')[-1]
    if 'intercept' in tail:
        return float(np.asarray(model.intercept_).flat[0])
    if 'coefficient of' in tail:
        term = tail.split('coefficient of', 1)[1].split('?', 1)[0]
        index = feature_names.index(next(name for name in feature_names if name in term))
        return float(np.asarray(model.coef_).flatten()[index])
    if 'r²' in tail or 'r2' in tail:
        return model.score(x, y)
    test = data.frame('test.csv')
    case = re.search(r'(?:loan|flat|reading|customer|stay)\s+(\d+)\s+of test\.csv', q)
    if case:
        test = test[pd.to_numeric(test.iloc[:, 0], errors='coerce') == int(case.group(1))]
    test_features = [data.column(name, 'test.csv') for name in feature_names]
    preds = model.predict_proba(test[test_features].astype(float))[:, 1] if logistic else model.predict(test[test_features].astype(float))
    return float(np.mean(preds)) if 'mean predicted' in tail else float(preds[0])


def infer_type(task):
    q = task['objective'].lower()
    if 'logistic regression' in q:
        return 'fit.logistic'
    if 'linear regression' in q or 'least-squares' in q:
        return 'fit.linear'
    names = [os.path.basename(str(name)) for name in task['files']]
    if len([name for name in names if name != 'data_dictionary.csv']) > 1:
        return 'table.join'
    if 'for each' in q or 'rank the' in q:
        return 'table.group'
    if 'divided by' in q or 'tonnes' in q or 'megajoules' in q:
        return 'table.derive'
    if 'among the' in q or 'whose' in q or 'with a' in q:
        return 'table.filter'
    return 'table.stat'


def solve_data(task):
    kind = task.get('type') or infer_type(task)
    data = Data(task)
    actions = {
        'table.stat': stat_task,
        'table.filter': filter_task,
        'table.group': group_task,
        'table.join': join_task,
        'table.derive': derive_task,
    }
    if kind.startswith('fit.'):
        return fit_task(data, kind == 'fit.logistic')
    return actions[kind](data)


def observe(call):
    name, arg = call
    if name != 'calculator':
        return f'Error: unknown tool {name!r}; the only tool is calculator.'
    try:
        value = dsh.calculator(arg.replace('math.', ''))
    except (ValueError, ZeroDivisionError, OverflowError, TypeError) as e:
        return f'Error: {e}'
    return f'Result: {value:.10g}'


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


def check_infinite(tasks, indices, out):
    remaining = []
    for i in indices:
        task = tasks[i]
        try:
            answer = float(solve_data(task))
            if math.isfinite(answer):
                out[i] = {'id': task['id'], 'answer': answer}
                continue
        except Exception:
            pass
        remaining.append(i)
    return remaining


class Agent:
    def __init__(self):
        self.llm_calc = None
        self.llm_fit = None
        self.llm_table = None
        self.out = []

    def solve(self, tasks, seconds=40.0):
        clock = dsh.Clock(seconds)
        self.out = [None] * len(tasks)

        calc_files, fit_files, table_files = route_tasks(tasks)
        remaining_fit = check_infinite(tasks, fit_files, self.out)
        remaining_table = check_infinite(tasks, table_files, self.out)

        def fit_llm(indices):
            # todo
            return

        def table_llm(indices):
            # todo
            return 

        def calc_llm(indices):
            self.llm_calc = self.llm_calc or dsh.load_llm(MODEL)
            convs = []
            for i in indices:
                task = tasks[i]
                text = task['objective']
                if task['files']:
                    text += '\n\nThe files (first lines):\n' + dsh.file_preview(task['files'])
                convs.append([{'role': 'system', 'content': SYSTEM_calc}] + DEMO_calc + [{'role': 'user', 'content': text}])
            final = [''] * len(indices)
            active = list(range(len(indices)))
            for _ in range(MAX_ROUNDS):
                if not active or clock.left() < 3:
                    break
                replies = self.llm_calc.chat([convs[j] for j in active], max_new_tokens=384,
                                             max_time=clock.left() - 1, stop=['</call>'])
                next_active = []
                for j, reply in zip(active, replies):
                    if reply:
                      final[j] = reply
                    call = dsh.parse_tool_call(reply)
                    if call is None:
                        continue
                    convs[j].append({'role': 'assistant', 'content': reply})
                    convs[j].append({'role': 'user', 'content': observe(call)})
                    next_active.append(j)
                active = next_active
            for j, i in enumerate(indices):
                answer = dsh.tagged_number(final[j])
                if answer is None:
                    answer = dsh.last_number(final[j])
                self.out[i] = {'id': tasks[i]['id'], 'answer': answer if answer is not None else 0}

        calc_llm(calc_files)
        fit_llm(remaining_fit)
        table_llm(remaining_table)

        return self.out
