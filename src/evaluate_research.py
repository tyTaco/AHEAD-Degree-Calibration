"""Evaluate one sealed research run once, after an explicit Test decision.

This entry point does not train, select checkpoints, or overwrite prior outputs.
CPU is the default; using a GPU requires separate authorization.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import torch

from train_research import (
    build_research_model,
    collect_research_scores,
    file_sha256,
    research_input_files,
    research_output_path,
    research_ranking_metrics,
    set_research_seed,
    verify_all_selected_checkpoints,
    verify_selected_checkpoint,
)
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.utils import get_neighbor_sampler


def load_verified_run(project_dir, run_id):
    """Verify selection, source, dataset, environment and checkpoints without Test inference."""
    project = Path(project_dir).resolve()
    root = research_output_path(project, run_id)
    if not root.is_dir():
        raise ValueError("The requested research run does not exist")
    if (root / "final_test").exists() or (root / "final_test").is_symlink():
        raise FileExistsError("A final Test attempt already exists; refusing to repeat it")
    sealed_path = root / "selection_complete.json"
    provenance_path = root / "provenance.json"
    sealed = json.loads(sealed_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    if sealed.get("schema_version") != 1:
        raise ValueError("Unsupported selection manifest")
    if file_sha256(provenance_path) != sealed.get("provenance_sha256"):
        raise ValueError("Provenance manifest changed after selection")
    for key in ("dataset", "policy", "calibration_mode"):
        if sealed.get(key) != provenance.get(key):
            raise ValueError(f"Selection and provenance disagree on {key}")
    if sealed.get("policy") not in ("val_auroc", "train_loss"):
        raise ValueError("Invalid selection policy")
    if sealed.get("calibration_mode") not in ("original", "none", "continuous"):
        raise ValueError("Invalid calibration mode")
    dataset = sealed["dataset"]
    if not isinstance(dataset, str) or dataset in ("", ".", "..") or Path(dataset).name != dataset:
        raise ValueError("Invalid dataset name")
    current_versions = {"python": sys.version, "torch": str(torch.__version__),
                        "numpy": np.__version__, "pandas": pd.__version__,
                        "scikit_learn": sklearn.__version__}
    for key, current in current_versions.items():
        if provenance.get(key) != current:
            raise ValueError(f"Environment version changed: {key}")
    expected_files = research_input_files(project, dataset)
    fingerprints = provenance.get("files", {})
    if set(fingerprints) != set(expected_files):
        raise ValueError("The recorded input file list is incomplete or different")
    for name, path in expected_files.items():
        if not path.is_file() or file_sha256(path) != fingerprints[name].get("sha256"):
            raise ValueError(f"Input file changed after selection: {name}")

    runs = sealed.get("selected_runs")
    count = sealed.get("num_runs")
    if not isinstance(count, int) or isinstance(count, bool) or count < 1 or not isinstance(runs, list) or len(runs) != count:
        raise ValueError("Incomplete selected runs")
    if [run.get("seed") for run in runs] != list(range(count)):
        raise ValueError("Selected seeds are missing, duplicated or out of order")
    verified = []
    shared_config = None
    for run in runs:
        seed = run["seed"]
        name = f"unsupervised_AHEADGraphMixer_seed{seed}"
        relative = Path("models") / "AHEADGraphMixer" / dataset / name / f"{name}.pkl"
        checkpoint = root / relative
        record_path = checkpoint.parent / "selection.json"
        if run.get("name") != name or run.get("path") != str(relative):
            raise ValueError("Unexpected selected checkpoint path")
        if checkpoint.is_symlink() or checkpoint.resolve() != checkpoint:
            raise ValueError("Selected checkpoint must not use symlink redirection")
        if record_path.is_symlink() or record_path.resolve() != record_path:
            raise ValueError("Selection record must not use symlink redirection")
        if file_sha256(record_path) != run.get("selection_record_sha256"):
            raise ValueError("Selection record changed after sealing")
        record = json.loads(record_path.read_text())
        config = run.get("configuration")
        if not isinstance(config, dict):
            raise ValueError("Missing run configuration")
        common = {k: v for k, v in config.items() if k not in ("seed", "save_model_name")}
        if shared_config is None:
            shared_config = common
        elif common != shared_config:
            raise ValueError("Selected runs have inconsistent configurations")
        expected_metric = "validation_auroc" if sealed["policy"] == "val_auroc" else "mean_train_batch_loss"
        expected = {"policy": sealed["policy"], "dataset": dataset, "seed": seed,
                    "selected_epoch": run.get("epoch"), "selection_metric": expected_metric,
                    "selection_score": run.get("selection_score"),
                    "checkpoint": checkpoint.name, "checkpoint_sha256": run.get("checkpoint_sha256"),
                    "validation_labels_used_for_selection": sealed["policy"] == "val_auroc",
                    "calibration_mode": sealed["calibration_mode"], "configuration": config}
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError("Selection record does not match sealed run")
        if run.get("selection_metric") != expected_metric:
            raise ValueError("Selection metric does not match policy")
        if config.get("seed") != seed or config.get("save_model_name") != name:
            raise ValueError("Configuration seed or checkpoint name mismatch")
        if config.get("dataset_name") != dataset or config.get("model_name") != "AHEADGraphMixer":
            raise ValueError("Configuration dataset or model mismatch")
        if not isinstance(run.get("epoch"), int) or not 1 <= run["epoch"] <= config.get("num_epochs", 0):
            raise ValueError("Selected epoch is outside the configured training range")
        if not isinstance(run.get("selection_score"), (int, float)) or not np.isfinite(run["selection_score"]):
            raise ValueError("Selection score must be finite")
        verified.append(dict(run, path=checkpoint))
    verify_all_selected_checkpoints(verified)
    return root, sealed, provenance, verified


def evaluate_run(project_dir, run_id, device, confirm_test):
    if not confirm_test:
        raise ValueError("Final Test requires the explicit --confirm-test flag")
    project = Path(project_dir).resolve()
    if Path.cwd().resolve() != project:
        raise ValueError("Run this script from its src directory")
    root, sealed, provenance, selected_runs = load_verified_run(project, run_id)
    requested_device = torch.device(device)
    if requested_device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA devices are supported")
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    if requested_device.type == "cuda" and requested_device.index is not None and requested_device.index >= torch.cuda.device_count():
        raise ValueError("Requested CUDA device does not exist")
    mode = sealed["calibration_mode"]
    old_mode = os.environ.get("AHEAD_DEGREE_CALIB_MODE")
    if old_mode is not None and old_mode != mode:
        raise ValueError("Calibration environment conflicts with the sealed run")
    # A fresh, exclusive directory records even an interrupted Test attempt.
    final_dir = root / "final_test"
    final_dir.mkdir(exist_ok=False)
    with (final_dir / "evaluation_started.json").open("x") as f:
        json.dump({"selection_sha256": file_sha256(root / "selection_complete.json"),
                   "provenance_sha256": file_sha256(root / "provenance.json"),
                   "device": str(requested_device), "calibration_mode": mode,
                   "memory_policy": "selected_train_state_then_replay_validation_without_labels",
                   "evaluation_mask": "none", "num_runs": len(selected_runs)}, f, indent=2)
    os.environ["AHEAD_DEGREE_CALIB_MODE"] = mode
    try:
        first = selected_runs[0]["configuration"]
        node_features, edge_features, full_data, _, val_data, test_data = get_node_classification_data(
            dataset_name=sealed["dataset"], val_ratio=first["val_ratio"], test_ratio=first["test_ratio"])
        if np.any(np.diff(full_data.node_interact_times) < 0):
            raise ValueError("Interactions must be chronological")
        val_loader = get_idx_data_loader(list(range(len(val_data.src_node_ids))), first["batch_size"], False)
        test_loader = get_idx_data_loader(list(range(len(test_data.src_node_ids))), first["batch_size"], False)
        results = []
        for selected in selected_runs:
            config = SimpleNamespace(**selected["configuration"])
            config.device = str(requested_device)
            set_research_seed(selected["seed"], requested_device)
            sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=config.sample_neighbor_strategy,
                                           time_scaling_factor=config.time_scaling_factor, seed=1)
            model = build_research_model(config, node_features, edge_features, sampler, selected["seed"])
            verify_selected_checkpoint(selected["path"], selected["checkpoint_sha256"])
            model.load_state_dict(torch.load(selected["path"], map_location=requested_device, weights_only=True), strict=True)
            model = model.to(requested_device)
            model.memory_generator.grad_mask_dict = None
            collect_research_scores(model, sampler, val_loader, val_data, config.num_neighbors, config.time_gap, False)
            scores, labels = collect_research_scores(model, sampler, test_loader, test_data,
                                                      config.num_neighbors, config.time_gap, True)
            metrics = research_ranking_metrics(labels, scores)
            result = {"seed": selected["seed"], "selected_epoch": selected["epoch"],
                      "selection_metric": selected["selection_metric"], "policy": sealed["policy"],
                      "validation_labels_used_for_selection": sealed["policy"] == "val_auroc",
                      "checkpoint_sha256": selected["checkpoint_sha256"], "device": str(requested_device),
                      "calibration_mode": mode, "metrics": metrics}
            with (final_dir / f"seed{selected['seed']}_metrics.json").open("x") as f:
                json.dump(result, f, indent=2)
            for name, values in (("scores", scores), ("labels", labels)):
                with (final_dir / f"seed{selected['seed']}_{name}.npy").open("xb") as f:
                    np.save(f, values)
            results.append(result)
        summary = {}
        for metric in ("AUROC", "AP", "PR_AUC"):
            values = [item["metrics"][metric] for item in results]
            summary[metric] = {"mean": float(np.mean(values)),
                               "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        with (final_dir / "evaluation_complete.json").open("x") as f:
            json.dump({"num_runs": len(results), "summary": summary}, f, indent=2)
        return summary
    finally:
        if old_mode is None:
            os.environ.pop("AHEAD_DEGREE_CALIB_MODE", None)
        else:
            os.environ["AHEAD_DEGREE_CALIB_MODE"] = old_mode


def validate_paired_runs(verified_runs):
    """Require both prespecified policies to use the same inputs and settings."""
    if len(verified_runs) != 2:
        raise ValueError("Final comparison requires both sealed selection policies")
    policies = {sealed["policy"] for _, sealed, _, _ in verified_runs}
    if policies != {"val_auroc", "train_loss"}:
        raise ValueError("Expected one val_auroc run and one train_loss run")
    _, first, first_provenance, first_selected = verified_runs[0]
    _, second, second_provenance, second_selected = verified_runs[1]
    for key in ("dataset", "calibration_mode", "num_runs"):
        if first[key] != second[key]:
            raise ValueError(f"Paired runs disagree on {key}")
    if first_provenance["files"] != second_provenance["files"]:
        raise ValueError("Paired runs have different code or data fingerprints")
    if [item["seed"] for item in first_selected] != [item["seed"] for item in second_selected]:
        raise ValueError("Paired runs have different seeds")
    for left, right in zip(first_selected, second_selected):
        if left["configuration"] != right["configuration"]:
            raise ValueError(f"Paired runs have different settings for seed {left['seed']}")


def evaluate_paired_runs(project_dir, run_ids, device, confirm_test):
    if not confirm_test:
        raise ValueError("Final Test requires the explicit --confirm-test flag")
    if len(run_ids) != 2 or len(set(run_ids)) != 2:
        raise ValueError("Provide two distinct sealed run IDs")
    # Verify both policies, including all checkpoints, before either sees Test.
    verified = [load_verified_run(project_dir, run_id) for run_id in run_ids]
    validate_paired_runs(verified)
    summaries = {}
    for run_id, (_, sealed, _, _) in zip(run_ids, verified):
        summaries[sealed["policy"]] = evaluate_run(project_dir, run_id, device, True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", action="append", required=True,
                        help="Existing sealed run ID; provide once for each selection policy")
    parser.add_argument("--device", default="cpu", help="Evaluation device; CPU by default")
    parser.add_argument("--confirm-test", action="store_true", help="Explicitly evaluate Test once")
    args = parser.parse_args()
    summary = evaluate_paired_runs(Path(__file__).resolve().parent, args.run_id, args.device, args.confirm_test)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
