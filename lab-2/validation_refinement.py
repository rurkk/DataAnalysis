"""Небольшое уточнение параметров только на validation; финальный отчёт не изменяется."""
from pathlib import Path
import os
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '2'
import ast
import json
import time
import hashlib
import numpy as np
import pandas as pd
import nbformat
from scipy import sparse, linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SEED = 42


def run():
    threadpool_limits(limits=2)
    # Использую ровно ту же реализацию и метрики, что в основном отчёте.
    notebook = nbformat.read(ROOT / 'lab-2.ipynb', as_version=4)
    symbols = {'Recommender', 'histories', 'targets', 'user_metrics', 'evaluate'}
    found = set()
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        for node in ast.parse(cell.source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in symbols:
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<notebook>', 'exec'), globals())
                found.add(node.name)
    assert found == symbols
    train = pd.read_csv(DATA / 'train_events.csv', dtype={'StockCode': str})
    validation = pd.read_csv(DATA / 'validation_events.csv', dtype={'StockCode': str})
    truth, _, _, _ = targets(train, validation)
    # Сетка фиксируется до расчёта новых метрик; test не загружается.
    grid = [('ItemKNN', k) for k in [30, 10, 20, 50]]
    grid += [('EASE', reg) for reg in [1000, 300, 3000, 10000]]
    rows, users = [], []
    for method, parameter in grid:
        model = Recommender(train, method, parameter)
        agg, individual, _, elapsed = evaluate(model, truth, ks=(10,))
        rows.append({'method': method, 'parameter': parameter, 'fit_s': model.fit_seconds,
                     'evaluation_s': elapsed, **agg.iloc[0].to_dict()})
        individual['method'] = method
        individual['parameter'] = parameter
        users.append(individual)
        print(method, parameter, 'NDCG@10 =', round(rows[-1]['NDCG'], 6), flush=True)
    results = pd.DataFrame(rows)
    individual = pd.concat(users, ignore_index=True)
    intervals = []
    rng = np.random.default_rng(SEED)
    for method, base in [('ItemKNN', 30), ('EASE', 1000)]:
        group = results[results.method == method]
        winner = group.sort_values('NDCG', ascending=False, kind='stable').iloc[0]
        best = int(winner.parameter)
        pivot = individual[individual.method == method].pivot(index='user', columns='parameter', values='NDCG')
        delta = (pivot[best] - pivot[base]).to_numpy()
        means = np.array([rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(2000)])
        baseline = group[group.parameter == base].NDCG.iloc[0]
        intervals.append({'method': method, 'baseline_parameter': base, 'best_parameter': best,
                          'baseline_ndcg': baseline, 'best_ndcg': winner.NDCG,
                          'delta': delta.mean(), 'relative_pct': 100 * delta.mean() / baseline,
                          'ci_low': np.quantile(means, .025), 'ci_high': np.quantile(means, .975)})
    results.to_csv(DATA / 'validation_refinement.csv', index=False)
    individual.to_csv(DATA / 'validation_refinement_users.csv', index=False)
    comparison = pd.DataFrame(intervals)
    comparison.to_csv(DATA / 'validation_refinement_comparison.csv', index=False)
    manifest = {'seed': SEED, 'grid': grid, 'users': len(truth), 'test_loaded': False,
                'primary_metric': 'NDCG@10', 'bootstrap_repeats': 2000,
                'limitations': 'Intervals are exploratory: winner was selected on the same validation users.',
                'inputs_sha256': {name: hashlib.sha256((DATA / name).read_bytes()).hexdigest()
                                  for name in ['train_events.csv', 'validation_events.csv']}}
    (DATA / 'validation_refinement_protocol.json').write_text(json.dumps(manifest, indent=2))
    print(comparison.to_string(index=False))


if __name__ == '__main__':
    run()
