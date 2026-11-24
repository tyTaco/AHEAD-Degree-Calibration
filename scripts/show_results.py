"""Level 1: summarize stored per-seed JSON; no inference or writes."""
import json
import math
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'src/research_runs/fixed_degree_up_exploration_20261123/report.json'
DATASETS = ('digg', 'bitcoinalpha', 'wikipedia', 'mooc')
POLICIES = ('val_auroc', 'train_loss')
METHODS = ('original', 'none', 'continuous', 'fixed_1_1', 'c_pairwise', 'c_guarded')
METRICS = ('AUROC', 'AP', 'PR_AUC')


def summarize(report):
    """Calculate all 144 method summaries and 120 paired comparisons."""
    if set(report['datasets']) != set(DATASETS):
        raise ValueError('報告中的資料集不完整')
    results = []
    for name in DATASETS:
        data = report['datasets'][name]
        for policy in POLICIES:
            rows = data['results'][policy]
            if sorted(row['seed'] for row in rows) != list(range(5)):
                raise ValueError(f'{name}/{policy}: seeds 不是 0–4')
            for metric in METRICS:
                for method in METHODS:
                    values = [float(row['metrics'][method][metric]) for row in rows]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f'{name}/{policy}/{method}/{metric}: 非有限數值')
                    results.append(('method', name, policy, metric, method, mean(values), stdev(values)))
                for base in METHODS:
                    if base == 'fixed_1_1':
                        continue
                    values = [row['metrics']['fixed_1_1'][metric] - row['metrics'][base][metric] for row in rows]
                    avg, sd = mean(values), stdev(values)
                    recorded = data['summary'][policy][base][metric]
                    if not math.isclose(avg, recorded['mean'], rel_tol=0, abs_tol=1e-12) or not math.isclose(sd, recorded['sample_sd'], rel_tol=0, abs_tol=1e-12):
                        raise ValueError(f'{name}/{policy}/{base}/{metric}: 配對摘要不一致')
                    results.append(('paired', name, policy, metric, base, avg, sd))
    return results


def main():
    report = json.loads(REPORT.read_text(encoding='utf-8'))
    results = summarize(report)
    print('Level 1：僅依保存 JSON 逐 seed 重算；D 已探索，非獨立測試')
    print('樣本標準差 ddof=1；差值 = Fixed 1.1 − 對照；C 方法使用歷史標籤')
    print('type dataset policy metric method mean sample_sd')
    for kind, dataset, policy, metric, method, avg, sd in results:
        print(f'{kind:6} {dataset:13} {policy:10} {metric:6} {method:10} {avg:+.9f} {sd:.9f}')
    print('完成：144 個方法均值／樣本 SD、120 個配對差均值／樣本 SD；配對摘要核對通過。')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f'Level 1 失敗：{exc}', file=sys.stderr)
        sys.exit(1)
