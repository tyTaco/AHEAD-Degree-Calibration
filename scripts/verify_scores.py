"""Level 2: read-only recheck of the already-explored D scores."""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

# Must be set before importing torch or local research modules.
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

PACKAGE = Path(__file__).resolve().parents[1]
SOURCE = PACKAGE / 'src'
DATASETS = ('digg', 'bitcoinalpha', 'wikipedia', 'mooc')
POLICIES = ('val_auroc', 'train_loss')
METHODS = ('original', 'none', 'continuous', 'fixed_1_1', 'c_pairwise', 'c_guarded')
METRICS = ('AUROC', 'AP', 'PR_AUC')
REPORT_REL = Path('research_runs/fixed_degree_up_exploration_20261123/report.json')
PRIOR_REL = Path('research_runs/c_reference_pairwise_exploration_20261123/report.json')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked_file(path):
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f'缺少檔案或檔案是符號連結：{path}')
    return path


def resolve_asset_root(path):
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f'研究資產根目錄不是資料夾：{root}')
    if not (root / 'research_runs').is_dir() and (root / 'src/research_runs').is_dir():
        root = root / 'src'
    if not (root / 'research_runs').is_dir() or not (root / 'processed_data').is_dir():
        raise ValueError(f'根目錄必須包含 research_runs/ 與 processed_data/，或包含這些項目的 src/：{root}')
    return root


def verify(root, datasets):
    bundled = checked_file(SOURCE / REPORT_REL)
    external = checked_file(root / REPORT_REL)
    if sha256(bundled) != sha256(external):
        raise ValueError(f'封裝報告與外部結果 hash 不一致：{external}')
    report = json.loads(bundled.read_text(encoding='utf-8'))
    prior_path = checked_file(root / PRIOR_REL)
    if sha256(prior_path) != report['prior_report_sha256']:
        raise ValueError(f'C 選擇報告 hash 不一致：{prior_path}')
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    if sha256(checked_file(root / 'run_fixed_degree_up_development.py')) != report['code_sha256']:
        raise ValueError('外部固定倍率分析程式 hash 與報告不一致')
    for name, digest in prior['code_sha256'].items():
        if Path(name).name != name or sha256(checked_file(root / name)) != digest:
            raise ValueError(f'C 選擇程式 hash 不一致：{name}')
    # Imported code must match the original research tree; tests are not imported.
    for local in sorted(SOURCE.rglob('*.py')):
        if 'tests' in local.relative_to(SOURCE).parts:
            continue
        relative = local.relative_to(SOURCE)
        remote = checked_file(root / relative)
        if sha256(local) != sha256(remote):
            raise ValueError(f'封裝與研究程式 hash 不一致：{relative}')
    sys.path.insert(0, str(SOURCE))
    from run_fixed_degree_up_development import analyze

    print(f'外部研究資產：{root}')
    print(f'封存結果 SHA-256：{sha256(external)}；C 選擇報告及程式 hash：通過')
    print('本次只讀已探索 D；沿用 C 既存選擇，不重新執行 C bootstrap。')
    total = 0
    max_difference = 0.0
    for dataset in datasets:
        print(f'核對 {dataset}：checkpoint、輸入資料與 scores hash ...', flush=True)
        actual = analyze(root, dataset, prior['datasets'][dataset])
        expected = report['datasets'][dataset]
        for policy in POLICIES:
            actual_rows, expected_rows = actual['results'][policy], expected['results'][policy]
            if [r['seed'] for r in actual_rows] != list(range(5)) or [r['seed'] for r in expected_rows] != list(range(5)):
                raise ValueError(f'{dataset}/{policy}: seeds 不一致')
            for a, e in zip(actual_rows, expected_rows):
                for key in ('checkpoint_sha256', 'score_sha256'):
                    if a[key] != e[key]:
                        raise ValueError(f'{dataset}/{policy}/seed{a["seed"]}: {key} 不一致')
                for method in METHODS:
                    for metric in METRICS:
                        av, ev = float(a['metrics'][method][metric]), float(e['metrics'][method][metric])
                        if not math.isfinite(av) or not math.isfinite(ev):
                            raise ValueError(f'{dataset}/{policy}/seed{a["seed"]}/{method}/{metric}: 非有限數值')
                        difference = abs(av - ev)
                        if difference > 1e-12:
                            raise ValueError(f'{dataset}/{policy}/seed{a["seed"]}/{method}/{metric}: 指標不一致 {difference}')
                        max_difference = max(max_difference, difference)
                        total += 1
            print(f'  {policy}：seed 0–4、5 個 checkpoint／scores hash 和 90 項指標通過')
        print(f'  D 區間 {actual["development_bounds"]}；CSV 標籤／時間及 hash 核對通過')
    print(f'完成：{len(datasets)} 資料集，{total} 項逐值比較，最大絕對差 {max_difference:.12g}。')
    print('限制：D 已探索，C 標籤發布時間未知，舊 scores 無 event ID；不是獨立測試或 Level 3。')
    print('完整檔案雜湊與 CSV／OS 緩衝可能讀取其他區段 bytes；不評估舊 Test。')
    return total


def main():
    parser = argparse.ArgumentParser(description='唯讀核對已封存的逐筆分數，不訓練、不推論、不寫入。')
    parser.add_argument('--assets-root', required=True, type=Path, help='原研究 src/，或包含 src/ 的專案根目錄')
    parser.add_argument('--dataset', choices=('all',) + DATASETS, default='all', help='選單一資料集或 all')
    args = parser.parse_args()
    try:
        root = resolve_asset_root(args.assets_root)
        verify(root, DATASETS if args.dataset == 'all' else (args.dataset,))
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError, json.JSONDecodeError) as exc:
        print(f'Level 2 核對失敗：{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
