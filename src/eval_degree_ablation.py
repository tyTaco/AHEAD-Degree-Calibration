import os
import sys
import json
import numpy as np
import torch

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc

from utils.load_configs import get_node_classification_args
from utils.DataLoader import get_node_classification_data, get_idx_data_loader
from utils.utils import get_neighbor_sampler, set_random_seed
from models.AHEADGraphMixer import AHEADGraphMixer
from evaluate_models_utils import evaluate_unsupervised_model_node_classification


def main():
    sys.argv = [
        "eval_degree_ablation.py",
        "--dataset_name", "digg",
        "--model_name", "AHEADGraphMixer",
    ]
    args = get_node_classification_args()

    seed = int(os.environ.get("AHEAD_ABLATION_SEED", "0"))
    if not 0 <= seed <= 4:
        raise ValueError("Seed must be between 0 and 4")
    set_random_seed(seed)

    node_features, edge_features, full_data, train_data, val_data, test_data = (
        get_node_classification_data(
            dataset_name=args.dataset_name,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    )

    sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1
    )

    test_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False
    )

    checkpoint = (
        "saved_models/AHEADGraphMixer/digg/"
        f"unsupervised_AHEADGraphMixer_seed{seed}/"
        f"unsupervised_AHEADGraphMixer_seed{seed}.pkl"
    )

    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)

    checkpoint_state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    output_dir = "saved_results/degree_ablation/digg"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\\n========== SEED {seed}: CONTROLLED EVALUATION ==========")

    set_random_seed(seed)

    model = AHEADGraphMixer(
        node_raw_features=node_features,
        edge_raw_features=edge_features,
        neighbor_sampler=sampler,
        time_feat_dim=args.time_feat_dim,
        num_tokens=args.num_neighbors,
        memory_dim=172,
        message_dim=128,
        device=args.device,
        num_layers=args.num_layers,
        dropout=args.dropout,
        loss_type=args.loss_type,
        temperature=args.temperature,
        time_gap=args.time_gap,
        num_hard_negatives=args.num_hard_negatives,
        lambda_thresh_loss=args.lambda_thresh_loss,
        dataset_name=args.dataset_name,
        run_idx=seed
    )

    model.load_state_dict(checkpoint_state, strict=True)
    model = model.to(args.device)
    model.degree_calib_mode = "none"
    os.environ.pop("AHEAD_DEGREE_CALIB_MODE", None)

    recorded_degrees = []
    original_degree_fn = model._get_temporal_degree_from_sampler

    def record_degree(*args, **kwargs):
        degrees = original_degree_fn(*args, **kwargs)
        recorded_degrees.append(
            np.asarray(degrees).reshape(-1).copy()
        )
        return degrees

    model._get_temporal_degree_from_sampler = record_degree

    _, scores, labels = evaluate_unsupervised_model_node_classification(
        model=model,
        neighbor_sampler=sampler,
        evaluate_idx_data_loader=test_loader,
        evaluate_data=test_data,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap
    )

    raw_scores = scores.detach().cpu().numpy().reshape(-1)
    y_true = labels.detach().cpu().numpy().astype(int).reshape(-1)
    degrees = np.concatenate(recorded_degrees)

    if len(degrees) != len(raw_scores):
        raise RuntimeError("Degree and score lengths do not match")

    calibrated = {
        "original": raw_scores * np.where(degrees < 3, 0.9, 1.0),
        "none": raw_scores.copy(),
        "continuous": raw_scores * (
            1.0 - 0.1 * np.exp(-degrees / 3.0)
        )
    }

    results = {}

    for mode, y_score in calibrated.items():
        precision, recall, _ = precision_recall_curve(
            y_true, y_score
        )

        results[mode] = {
            "AUROC": float(roc_auc_score(y_true, y_score)),
            "AP": float(average_precision_score(y_true, y_score)),
            "PR_AUC": float(auc(recall, precision)),
            "num_samples": int(len(y_true)),
            "num_anomalies": int(y_true.sum())
        }

        np.savez_compressed(
            os.path.join(
                output_dir,
                f"seed{seed}_{mode}_controlled_details.npz"
            ),
            scores=y_score,
            labels=y_true,
            degrees=degrees
        )

    output_file = os.path.join(
        output_dir, f"seed{seed}_controlled_comparison.json"
    )

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\\n========== CONTROLLED SUMMARY ==========")

    for mode, result in results.items():
        print(
            f"{mode:12s} "
            f"AUROC={result['AUROC']:.6f} "
            f"AP={result['AP']:.6f} "
            f"PR-AUC={result['PR_AUC']:.6f}"
        )

    print(f"\\nSaved: {output_file}")


if __name__ == "__main__":
    main()
