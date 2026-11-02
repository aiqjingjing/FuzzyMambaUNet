import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image
from reference import FuzzyMambaUNet
from reference.data import sha256
from reference.metrics import case_metrics, summarize
from reference.run import (
    ROOT,
    load_checkpoint,
    make_dataset,
    make_loader,
    read_config,
    save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config",
        help="Override evaluation settings; default: checkpoint configuration",
    )
    parser.add_argument("--data-root", help="Override the configured data directory")
    parser.add_argument("--output", default="results/isic2018_seed42")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new --output directory")
    torch.set_num_threads(4)
    saved = load_checkpoint(args.checkpoint)
    local_config = read_config(args.config or ROOT / "experiments/val189/config.json")
    config = dict(local_config)
    if not args.config:
        for key in ("size", "threshold", "boundary_tolerance"):
            if key in saved.get("config", {}):
                config[key] = saved["config"][key]
    data_root = args.data_root or str(ROOT / local_config["data_root"])
    dataset = make_dataset(config, data_root, "test", verify=True)
    loader = make_loader(dataset, config)
    model = FuzzyMambaUNet(saved["model_config"])
    if model.config != saved["model_config"]:
        raise ValueError("Incomplete checkpoint model configuration")
    model.load_state_dict(saved["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    output.mkdir(parents=True, exist_ok=False)
    if args.save_predictions:
        (output / "predictions").mkdir()
    rows = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            probability = model(x.to(device))
            if not torch.isfinite(probability).all():
                raise FloatingPointError("Non-finite prediction")
            pred = probability[0, 0].cpu().numpy() >= config["threshold"]
            target = y[0, 0].numpy() >= 0.5
            row = case_metrics(pred, target, config["boundary_tolerance"])
            name = dataset.pairs[i][0].name
            row["image"] = name
            rows.append(row)
            if args.save_predictions:
                Image.fromarray(pred.astype(np.uint8) * 255).save(
                    output / "predictions" / (Path(name).stem + ".png")
                )
    metrics = summarize(rows)
    result = dict(
        seed=saved["seed"],
        epoch=saved["epoch"],
        checkpoint_sha256=sha256(args.checkpoint),
        threshold=config["threshold"],
        boundary_tolerance=config["boundary_tolerance"],
        DSC=100 * metrics["dice_global"],
        IoU=100 * metrics["foreground_iou_global"],
        BF1=100 * metrics["boundary_f1_case_mean"],
        metrics=metrics,
    )
    save_json(output / "summary.json", result)
    save_json(output / "config.json", config)
    with (output / "per_image.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        "DSC: {:.4f}% | IoU: {:.4f}% | BF1: {:.4f}%".format(
            result["DSC"], result["IoU"], result["BF1"]
        )
    )


if __name__ == "__main__":
    main()
