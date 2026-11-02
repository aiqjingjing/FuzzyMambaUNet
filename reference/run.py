import hashlib
import json
import random
import ssl
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import LesionDataset, sha256

ROOT = Path(__file__).resolve().parents[1]


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config["size"] < 32 or config["size"] % 32 or config["size"] > 2048:
        raise ValueError("Image size must be divisible by 32 and between 32 and 2048")
    if config["epochs"] < 1 or config["batch_size"] < 1:
        raise ValueError("Epochs and batch size must be positive")
    if not 0 <= config["threshold"] <= 1 or config["boundary_tolerance"] < 0:
        raise ValueError("Invalid evaluation threshold or boundary tolerance")
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def split_directory(config):
    value = config.get("split_dir")
    return ROOT / value if value else None


def make_dataset(config, data_root, split, augment=False, verify=False):
    return LesionDataset(
        data_root, split, config["size"], augment, split_directory(config), verify
    )


def make_loader(dataset, config, training=False, seed=42):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"] if training else 1,
        shuffle=training,
        num_workers=config["num_workers"],
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
        worker_init_fn=worker_seed,
    )


def loss_fn(prediction, target):
    bce = F.binary_cross_entropy(prediction, target)
    p, t = prediction.flatten(1), target.flatten(1)
    dice = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    return bce + dice


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        loss = loss_fn(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / count


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def state_sha256(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(saved, dict)
        or not {"model", "model_config", "seed", "epoch"} <= saved.keys()
    ):
        raise ValueError("Invalid checkpoint fields")
    if (
        type(saved["seed"]) is not int
        or type(saved["epoch"]) is not int
        or saved["epoch"] < 1
    ):
        raise ValueError("Invalid checkpoint seed or epoch")
    if not isinstance(saved["model_config"], dict):
        raise ValueError("Invalid model configuration")
    state = saved["model"]
    if not isinstance(state, dict) or not state:
        raise ValueError("Invalid model state")
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Invalid model state entry")
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite checkpoint tensor: " + key)
    if "model_sha256" in saved and state_sha256(state) != saved["model_sha256"]:
        raise ValueError("Model state checksum mismatch")
    return saved


def initialize_backbone(model, path):
    settings = json.loads((ROOT / "pretrained.json").read_text())
    path = Path(path)
    if not path.exists():
        import certifi

        path.parent.mkdir(parents=True, exist_ok=True)
        context = ssl.create_default_context(cafile=certifi.where())
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".part", delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            with urlopen(settings["url"], timeout=60, context=context) as response:
                with temporary.open("wb") as handle:
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        handle.write(block)
            if sha256(temporary) != settings["sha256"]:
                raise ValueError("Downloaded backbone checksum mismatch")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if sha256(path) not in settings["accepted_source_sha256"]:
        raise ValueError("Backbone checkpoint checksum does not match pretrained.json")
    report = model.load_pretrained(path, mirror_decoder=True)
    if report["loaded_tensors"] < settings["min_loaded_tensors"]:
        raise ValueError("Insufficient matching backbone tensors")
    return {
        key: report[key]
        for key in ("loaded_tensors", "encoder_tensors", "decoder_tensors")
    }
