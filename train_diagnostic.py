import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from reference import FuzzyMambaUNet
from reference.data import check_disjoint, sha256
from reference.run import (
    ROOT,
    initialize_backbone,
    loss_fn,
    make_dataset,
    make_loader,
    read_config,
    save_checkpoint,
    save_json,
    seed_all,
    state_sha256,
    validation_loss,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(ROOT / "experiments/val189/config.json")
    )
    parser.add_argument("--data-root", help="Override the configured data directory")
    parser.add_argument("--pretrained", help="Local VMamba backbone checkpoint")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", default=str(ROOT / "experiments/val189/runs/seed_42")
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    if args.pretrained and args.from_scratch:
        parser.error("--pretrained and --from-scratch cannot be combined")
    if args.seed < 0 or args.seed >= 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    config = read_config(args.config)
    args.data_root = args.data_root or str(ROOT / config["data_root"])
    if not args.from_scratch:
        args.pretrained = args.pretrained or str(ROOT / config["pretrained"])
    output = Path(args.output)
    if output.exists():
        parser.error("Output already exists; choose a new --output directory")
    seed_all(args.seed)
    torch.set_num_threads(4)
    datasets = {
        split: make_dataset(
            config, args.data_root, split, augment=split == "train", verify=True
        )
        for split in ("train", "val", "test")
    }
    check_disjoint(datasets)
    model = FuzzyMambaUNet(config["model"])
    initialization = (
        initialize_backbone(model, args.pretrained) if args.pretrained else "random"
    )
    device = torch.device(args.device)
    model.to(device)
    train_loader = make_loader(datasets["train"], config, True, args.seed)
    val_loader = make_loader(datasets["val"], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["t_max"], eta_min=config["eta_min"]
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "config.json", config)
    split_dir = config.get("split_dir")
    provenance = dict(
        seed=args.seed,
        counts={k: len(v) for k, v in datasets.items()},
        initialization=initialization,
        pretrained_sha256=sha256(args.pretrained) if args.pretrained else None,
        split_sha256={s: sha256(ROOT / split_dir / (s + ".csv")) for s in datasets}
        if split_dir
        else None,
    )
    save_json(output / "run.json", provenance)
    history, best = [], float("inf")
    for epoch in range(1, config["epochs"] + 1):
        random.seed(args.seed + epoch)
        np.random.seed(args.seed + epoch)
        model.train()
        losses = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        value = validation_loss(model, val_loader, device)
        if value < best:
            best = value
            save_checkpoint(
                output / "best.pth",
                dict(
                    model=model.state_dict(),
                    model_config=model.config,
                    config=config,
                    seed=args.seed,
                    epoch=epoch,
                    val_loss=value,
                    model_sha256=state_sha256(model.state_dict()),
                ),
            )
        record = dict(
            epoch=epoch,
            train_loss=float(np.mean(losses)),
            val_loss=value,
            best_val_loss=best,
            lr=optimizer.param_groups[0]["lr"],
        )
        history.append(record)
        save_json(output / "history.json", history)
        print(json.dumps(record), flush=True)
    print("Best checkpoint:", output / "best.pth")


if __name__ == "__main__":
    main()
