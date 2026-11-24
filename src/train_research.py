import os
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import hashlib
import random
import time
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F 
import pandas as pd
import sklearn
import networkx as nx

# --- Model Imports ---
# Make sure these model files exist in your `models/` directory
# from models.UnsupervisedGraphMixer import UnsupervisedGraphMixer
from models.AHEADGraphMixer import AHEADGraphMixer
# from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts

# --- Utility Imports ---
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler
from utils.DataLoader import get_idx_data_loader, get_node_classification_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_node_classification_args

# --- Evaluation Function Import ---
from evaluate_models_utils import evaluate_unsupervised_model_node_classification, extract_embeddings_and_scores
from sklearn.metrics import f1_score


def _tensor_max(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return None
        return float(x.detach().max().item())
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return float(np.max(x))
    return None

# Isolated research protocol. The historical train.py and saved baselines are untouched.
# Set AHEAD_RESEARCH_POLICY to val_auroc or train_loss and AHEAD_RESEARCH_RUN_ID
# to a fresh, single-component directory name under ./research_runs/.
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc


def research_output_path(project_dir, run_id):
    if not run_id or run_id in (".", "..") or Path(run_id).name != run_id:
        raise ValueError("AHEAD_RESEARCH_RUN_ID must be a single non-empty directory name")
    project = Path(project_dir).resolve()
    parent = project / "research_runs"
    # Disallow symlink redirection, including dangling symlinks and run-id links.
    if parent.is_symlink() or parent.resolve() != parent:
        raise ValueError("research_runs must not be a symlink")
    candidate = parent / run_id
    if candidate.is_symlink() or candidate.resolve().parent != parent:
        raise ValueError("Research run directory must not be redirected")
    root = candidate
    for name in ("saved_models", "saved_results", "saved_scores"):
        protected = (project / name).resolve()
        if root == protected or root.is_relative_to(protected):
            raise ValueError("Research output cannot use a protected baseline directory")
    return root


def collect_research_scores(model, neighbor_sampler, loader, data, num_neighbors, time_gap, with_labels):
    model.set_neighbor_sampler(neighbor_sampler)
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for indices in loader:
            idx = indices.numpy()
            score = model.compute_anomaly_score(
                src_node_ids=data.src_node_ids[idx], dst_node_ids=data.dst_node_ids[idx],
                node_interact_times=data.node_interact_times[idx], edge_ids=data.edge_ids[idx],
                num_neighbors=num_neighbors, time_gap=time_gap)
            if not isinstance(score, torch.Tensor) or score.numel() != len(idx):
                raise ValueError("Model must return one score for each interaction")
            if not torch.isfinite(score).all().item():
                raise ValueError("Model produced non-finite scores")
            if with_labels:
                scores.append(score.detach().cpu().reshape(-1))
                labels.append(torch.as_tensor(data.labels[idx]).cpu().reshape(-1))
    if not with_labels:
        return None, None
    if not scores:
        raise ValueError("Evaluation split is empty")
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_research_seed(seed, device):
    """Seed CPU without initializing CUDA; seed CUDA only for a CUDA run."""
    if torch.device(device).type == "cpu":
        random.seed(seed)
        np.random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
    else:
        set_random_seed(seed=seed)


def verify_selected_checkpoint(path, expected_sha256):
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("Selected checkpoint changed after selection; refusing final test")


def verify_all_selected_checkpoints(selected_runs):
    for selected in selected_runs:
        verify_selected_checkpoint(selected["path"], selected["checkpoint_sha256"])


def research_input_files(project_dir, dataset_name):
    project = Path(project_dir)
    code = ["train_research.py", "evaluate_research.py", "models/AHEADGraphMixer.py", "models/PartitionedGraphMixer.py",
            "models/GraphMixer.py", "models/Memorymodel.py", "models/modules/__init__.py",
            "models/modules/time_encoding.py", "utils/DataLoader.py", "utils/EarlyStopping.py",
            "utils/load_configs.py", "utils/utils.py", "utils/metrics.py", "evaluate_models_utils.py"]
    data = [f"processed_data/{dataset_name}/ml_{dataset_name}{suffix}" for suffix in (".csv", ".npy", "_node.npy")]
    return {relative: project / relative for relative in code + data}


def research_ranking_metrics(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be non-empty and aligned")
    if not np.isfinite(scores).all() or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Finite scores and both binary label classes are required")
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"AUROC": float(roc_auc_score(labels, scores)),
            "AP": float(average_precision_score(labels, scores)),
            "PR_AUC": float(auc(recall, precision))}


def build_research_model(args, node_features, edge_features, sampler, seed):
    return AHEADGraphMixer(
        node_raw_features=node_features, edge_raw_features=edge_features,
        neighbor_sampler=sampler, time_feat_dim=args.time_feat_dim,
        num_tokens=args.num_neighbors, memory_dim=172, message_dim=128,
        device=args.device, num_layers=args.num_layers, dropout=args.dropout,
        loss_type=args.loss_type, temperature=args.temperature, time_gap=args.time_gap,
        num_hard_negatives=args.num_hard_negatives,
        lambda_thresh_loss=args.lambda_thresh_loss, dataset_name=args.dataset_name,
        run_idx=seed)


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # Get command-line arguments
    args = get_node_classification_args()
    policy = os.environ.get("AHEAD_RESEARCH_POLICY")
    if policy not in ("val_auroc", "train_loss"):
        raise ValueError("Set AHEAD_RESEARCH_POLICY to val_auroc or train_loss")
    if args.model_name != "AHEADGraphMixer":
        raise ValueError("This research protocol supports AHEADGraphMixer only")
    if args.num_runs < 1 or args.num_epochs < 1 or args.batch_size < 1:
        raise ValueError("num_runs, num_epochs and batch_size must be positive")
    calibration_mode = os.environ.get("AHEAD_DEGREE_CALIB_MODE", "original")
    if calibration_mode not in ("original", "none", "continuous"):
        raise ValueError("Invalid AHEAD_DEGREE_CALIB_MODE")
    run_id = os.environ.get("AHEAD_RESEARCH_RUN_ID", "")
    if Path.cwd().resolve() != Path(project_root).resolve():
        raise ValueError("Run this script from its src directory so input paths are unambiguous")
    research_root = research_output_path(project_root, run_id)
    # Exclusive creation prevents silent reuse or overwriting of any previous run.
    research_root.mkdir(parents=True, exist_ok=False)
    fingerprints = {name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                    for name, path in research_input_files(project_root, args.dataset_name).items()}
    manifest = {"policy": policy, "dataset": args.dataset_name, "run_id": run_id,
                "calibration_mode": calibration_mode, "python": sys.version,
                "torch": str(torch.__version__), "numpy": np.__version__,
                "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
                "files": fingerprints}
    with (research_root / "provenance.json").open("x") as f:
        json.dump(manifest, f, indent=2)
    
    # --- Add custom arguments for new models if they are not in the config file ---
    if not hasattr(args, 'anomaly_threshold'):
        args.anomaly_threshold = 0.5 # Default threshold for AHEADGraphMixer
    if not hasattr(args, 'test_interval_epochs'):
        args.test_interval_epochs = 5 # Default interval for periodic testing
    if not hasattr(args, 'pseudo_grad_threshold'):
        args.pseudo_grad_threshold = 1.0 # Default value
    if not hasattr(args, 'save_data_epoch_interval'):
        args.save_data_epoch_interval = 2


    # --- Data Loading ---
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = \
        get_node_classification_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # Do not inspect split labels during selection setup. Ranking metrics validate
    # the selected evaluation split only when labels are explicitly required.
    if np.any(np.diff(full_data.node_interact_times) < 0):
        raise ValueError("Interactions must be chronological before evaluating memory")

    num_nodes = node_raw_features.shape[0]

    # --- Neighbor Sampler & Data Loader Initialization ---
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=1)
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy, time_scaling_factor=args.time_scaling_factor, seed=0)
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    # --- Main Loop for Multiple Runs ---
    selected_runs = []
    for run in range(args.num_runs):
        set_research_seed(run, args.device)
        if hasattr(train_neighbor_sampler, "reset_random_state"):
            train_neighbor_sampler.reset_random_state()
        if hasattr(full_neighbor_sampler, "reset_random_state"):
            full_neighbor_sampler.reset_random_state()
        args.seed = run
        args.save_model_name = f'unsupervised_{args.model_name}_seed{args.seed}'

        # --- Logger Setup ---
        for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        log_dir = research_root / "logs" / args.model_name / args.dataset_name / args.save_model_name
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{str(time.time())}.log"))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter); ch.setFormatter(formatter)
        logger.addHandler(fh); logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts for UNSUPERVISED training. **********")
        logger.info(f'Configuration is {args}')
        logger.info(f"dataset_name: {args.dataset_name}")

        logger.info(f"Research policy: {policy}; this entry never evaluates Test")
        logger.info("Validation selection clears the residual training-batch mask")
        logger.info("train_loss is an unlabeled selection proxy, not a validated anomaly-detection criterion")

        # Only AHEADGraphMixer is supported by this protocol.
        model = build_research_model(args, node_raw_features, edge_raw_features,
                                     train_neighbor_sampler, run)

        logger.info(f'Model -> {model}')
        logger.info(f'#parameters: {get_parameter_sizes(model) * 4 / 1024 / 1024:.2f} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        model = convert_to_gpu(model, device=args.device)

        # --- Early Stopping Setup ---
        save_model_folder = research_root / "models" / args.model_name / args.dataset_name / args.save_model_name
        save_model_folder.mkdir(parents=True, exist_ok=False)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # --- Initialize states ONLY ONCE before the entire training process ---
        logger.info("Initializing model states for the run...")
        if hasattr(model, 'memory'):
            model.memory.reset_memory()

        selected_epoch = None

        # --- Training Loop ---
        for epoch in range(args.num_epochs):

            # ============================================================
            # EPOCH Stage 1: Indicator / Gradient Collection
            # ============================================================
            logger.info(f"Epoch {epoch + 1}, Stage 1: Collecting indicators...")
            model.eval()
            model.reset_states()
            model.set_neighbor_sampler(train_neighbor_sampler)

                # --- [CRITICAL] Always start Stage 1 from a CLEAN memory state ---
                # reset model.memory (the real source of time rollback)
            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            logger.info("Warming up memory before indicator collection...")
            for train_data_indices in tqdm(train_idx_data_loader, ncols=120, desc="Memory Warm-up"):
                    # IMPORTANT: if your loader returns torch.Tensor indices
                idx = train_data_indices.numpy().copy()
                batch_src = train_data.src_node_ids[idx]
                batch_dst = train_data.dst_node_ids[idx]
                batch_ts  = train_data.node_interact_times[idx]
                batch_eid = train_data.edge_ids[idx]

                with torch.no_grad():
                    call_kwargs = {}
                    call_kwargs["num_neighbors"] = args.num_neighbors
                    call_kwargs["time_gap"] = args.time_gap
                    model.compute_anomaly_score(batch_src, batch_dst, batch_ts, batch_eid, **call_kwargs)


            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            indicator_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, desc=f"Epoch {epoch + 1} [Indicator Collection]")

            for batch_idx, train_data_indices in enumerate(indicator_loader_tqdm):
                idx = train_data_indices.numpy().copy()

                batch_src = train_data.src_node_ids[idx]
                batch_dst = train_data.dst_node_ids[idx]
                batch_ts  = train_data.node_interact_times[idx]
                batch_eid = train_data.edge_ids[idx]

                call_kwargs = {}
                call_kwargs['num_neighbors'] = args.num_neighbors
                call_kwargs['time_gap'] = args.time_gap
                grad_norms, residuals = model.compute_and_cache_indicators(
                    batch_src, batch_dst, batch_ts, batch_eid,
                    **call_kwargs
                )

            model.indicator_cache.update_global_cache()

            # ============================================================
            # EPOCH Stage 2: Training (must ALSO start from CLEAN state)
            # ============================================================
                
            logger.info(f"Epoch {epoch + 1}, Stage 2: Training model...")
            model.train()
            model.reset_states()
            model.set_neighbor_sampler(train_neighbor_sampler)

            # --- [CRITICAL] Reset memory before training ---
            if hasattr(model, "memory") and hasattr(model.memory, "reset_memory"):
                model.memory.reset_memory()

            save_logs_this_epoch = (epoch + 1) % 10 == 0
            train_total_loss = 0.0

            import gc
            gc.collect()
            if torch.device(args.device).type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))

            torch.autograd.set_detect_anomaly(True)

            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, desc=f"Epoch {epoch + 1} [Training]")
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):

                idx = train_data_indices.numpy().copy()

                batch_src_node_ids = train_data.src_node_ids[idx]
                batch_dst_node_ids = train_data.dst_node_ids[idx]
                batch_node_interact_times = train_data.node_interact_times[idx]
                batch_edge_ids = train_data.edge_ids[idx]

                optimizer.zero_grad()

                loss_kwargs = dict(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                    edge_ids=batch_edge_ids,
                    epoch=epoch + 1,
                    save_logs_this_epoch=save_logs_this_epoch,
                )


                loss_kwargs["num_neighbors"] = args.num_neighbors
                loss_kwargs["time_gap"] = args.time_gap

                out = model.compute_unsupervised_loss(**loss_kwargs)
                if isinstance(out, (tuple, list)) and len(out) == 2:
                    loss, mem_payload = out
                else:
                    loss, mem_payload = out, None

                loss.backward()
                optimizer.step()


                if mem_payload is not None:
                    with torch.no_grad():
                        src_nodes, dst_nodes, times, upd_src_m, upd_dst_m = mem_payload
                        model.memory.update_state(src_nodes, upd_src_m.detach(), times)
                        model.memory.update_state(dst_nodes, upd_dst_m.detach(), times)

                train_total_loss += loss.item()
                train_idx_data_loader_tqdm.set_description(f"Epoch: {epoch + 1} [Training], train loss: {loss.item():.4f}")

            train_avg_loss = train_total_loss / (batch_idx + 1) if (batch_idx + 1) > 0 else 0


            # Select using only the declared policy; never evaluate Test here.
            if not np.isfinite(train_avg_loss):
                raise ValueError("Non-finite training loss cannot be used for selection")
            if policy == "val_auroc":
                state_after_train = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                training_mask = model.memory_generator.grad_mask_dict
                model.memory_generator.grad_mask_dict = None
                if hasattr(full_neighbor_sampler, "reset_random_state"):
                    full_neighbor_sampler.reset_random_state()
                try:
                    val_scores, val_labels = collect_research_scores(
                        model, full_neighbor_sampler, val_idx_data_loader, val_data,
                        args.num_neighbors, args.time_gap, with_labels=True)
                    selection_score = float(roc_auc_score(val_labels, val_scores))
                finally:
                    model.load_state_dict(state_after_train, strict=True)
                    model.memory_generator.grad_mask_dict = training_mask
                higher_is_better = True
                selection_name = "validation_auroc"
            else:
                selection_score = float(train_avg_loss)
                higher_is_better = False
                selection_name = "mean_train_batch_loss"
            if not np.isfinite(selection_score):
                raise ValueError("Non-finite selection score")
            previous = early_stopping.best_score
            improved = previous is None or (selection_score > previous if higher_is_better else selection_score < previous)
            if improved:
                selected_epoch = epoch + 1
            early_stop = early_stopping.step([(selection_name, selection_score, higher_is_better)], model)
            logger.info(f"Epoch {epoch + 1}: train_loss={train_avg_loss:.8f}, {selection_name}={selection_score:.8f}, selected_epoch={selected_epoch}")
            if early_stop:
                logger.info("Early stopping triggered")
                break

        best_model_path = save_model_folder / f"{args.save_model_name}.pkl"
        if selected_epoch is None or not best_model_path.is_file():
            raise RuntimeError("No selected checkpoint available; refusing final test")
        checkpoint_hash = file_sha256(best_model_path)
        selection_record = {"policy": policy, "dataset": args.dataset_name,
                            "seed": int(args.seed), "selected_epoch": selected_epoch,
                            "selection_metric": selection_name,
                            "selection_score": float(early_stopping.best_score),
                            "checkpoint": best_model_path.name,
                            "checkpoint_sha256": checkpoint_hash,
                            "validation_labels_used_for_selection": policy == "val_auroc",
                            "calibration_mode": calibration_mode,
                            "configuration": vars(args).copy()}
        with (save_model_folder / "selection.json").open("x") as f:
            json.dump(selection_record, f, indent=2)
        selected_runs.append({"seed": int(args.seed), "name": args.save_model_name,
                              "path": best_model_path, "epoch": selected_epoch,
                              "checkpoint_sha256": checkpoint_hash,
                              "selection_record_sha256": file_sha256(save_model_folder / "selection.json"),
                              "selection_metric": selection_name,
                              "selection_score": float(early_stopping.best_score),
                              "configuration": vars(args).copy()})
        logger.info(f"Selection complete for seed {args.seed}; no Test evaluation yet")
        logger.info(f"Training and selection cost {time.time() - run_start_time:.2f} seconds")
        del model, optimizer

    # Seal all selection decisions before any separate Test evaluation.
    if len(selected_runs) != args.num_runs:
        raise RuntimeError("Incomplete checkpoint selection; refusing to seal run")
    verify_all_selected_checkpoints(selected_runs)
    sealed = {"schema_version": 1, "policy": policy, "dataset": args.dataset_name,
              "calibration_mode": calibration_mode, "num_runs": int(args.num_runs),
              "provenance_sha256": file_sha256(research_root / "provenance.json"),
              "selected_runs": [dict(item, path=str(item["path"].relative_to(research_root)))
                                for item in selected_runs]}
    with (research_root / "selection_complete.json").open("x") as f:
        json.dump(sealed, f, indent=2)
    logger.info("All seeds selected and sealed. Test has not been evaluated.")

    sys.exit()

