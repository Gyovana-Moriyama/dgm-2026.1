import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
INDEX_PATTERN = re.compile(r"img_(\d+)_")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "training-results" / "cvae" / "results"


@dataclass
class EvaluationResult:
    num_ssim_pairs: int
    ssim_mean: float
    ssim_std: float
    ssim_min: float
    ssim_max: float
    num_counterfactual_images: int
    num_reference_images: int
    fid: float


class ImageFolderDataset(Dataset):
    """Small image folder dataset returning uint8 RGB tensors for TorchMetrics FID."""

    def __init__(self, image_paths: list[Path], transform: transforms.Compose):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def list_images(folder: Path, include_pairs: bool = False) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")

    paths = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not include_pairs:
        paths = [path for path in paths if "_pair" not in path.stem]
    return sorted(paths)


def extract_index(path: Path) -> str | None:
    match = INDEX_PATTERN.search(path.name)
    if match is None:
        return None
    return match.group(1)


def pair_originals_and_counterfactuals(
    original_dir: Path, counterfactual_dir: Path
) -> list[tuple[Path, Path]]:
    originals = {
        extract_index(path): path
        for path in list_images(original_dir)
        if "_original" in path.stem and extract_index(path) is not None
    }
    counterfactuals = {
        extract_index(path): path
        for path in list_images(counterfactual_dir)
        if "_counterfactual" in path.stem and extract_index(path) is not None
    }

    shared_indices = sorted(set(originals).intersection(counterfactuals))
    return [(originals[index], counterfactuals[index]) for index in shared_indices]


def load_grayscale_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("L")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0).unsqueeze(0)


def compute_paired_ssim(pairs: Iterable[tuple[Path, Path]], device) -> list[dict[str, object]]:
    rows = []
    for original_path, counterfactual_path in tqdm(
        list(pairs), desc="Computing SSIM", leave=False
    ):
        original = load_grayscale_tensor(original_path).to(device)
        counterfactual = load_grayscale_tensor(counterfactual_path).to(device)

        if original.shape != counterfactual.shape:
            raise ValueError(
                "Paired images must have the same shape for SSIM: "
                f"{original_path} has {tuple(original.shape)}, "
                f"{counterfactual_path} has {tuple(counterfactual.shape)}"
            )

        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        rows.append(
            {
                "index": extract_index(original_path),
                "original": str(original_path),
                "counterfactual": str(counterfactual_path),
                "ssim": float(ssim_metric(original, counterfactual).item()),
            }
        )
    return rows


def fid_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.PILToTensor(),
        ]
    )


@torch.no_grad()
def update_fid(
    metric: FrechetInceptionDistance, image_paths: list[Path], device: torch.device, batch_size: int, num_workers: int, real: bool) -> None:
    dataset = ImageFolderDataset(image_paths, fid_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    label = "real reference" if real else "fake counterfactual"
    for batch in tqdm(loader, desc=f"Updating FID ({label})", leave=False):
        metric.update(batch.to(device), real=real)


def compute_fid(
    reference_images: list[Path],
    counterfactual_images: list[Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    if len(reference_images) < 2 or len(counterfactual_images) < 2:
        raise ValueError("FID requires at least two images in each image set.")

    metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    update_fid(metric, reference_images, device, batch_size, num_workers, real=True)
    update_fid(metric, counterfactual_images, device, batch_size, num_workers, real=False)
    return float(metric.compute().item())


def write_ssim_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["index", "original", "counterfactual", "ssim"]
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_json(result: EvaluationResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(asdict(result), file, indent=2)


def summarize_ssim(rows: list[dict[str, object]]) -> tuple[float, float, float, float]:
    if not rows:
        nan = float("nan")
        return nan, nan, nan, nan

    scores = np.array([row["ssim"] for row in rows], dtype=np.float64)
    return (
        float(scores.mean()),
        float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
        float(scores.min()),
        float(scores.max()),
    )

def ssim_metric_calculation(device, output_csv, original_dir, counterfactual_dir):
    pairs = pair_originals_and_counterfactuals(original_dir, counterfactual_dir)
    if not pairs:
        raise ValueError(
            "No matching original/counterfactual pairs were found. Expected names like "
            "img_000000_original.png and img_000000_counterfactual.png."
        )
    ssim_rows = compute_paired_ssim(pairs, device)
    write_ssim_csv(ssim_rows, output_csv)
    ssim_mean, ssim_std, ssim_min, ssim_max = summarize_ssim(ssim_rows)
    return ssim_mean, ssim_std, ssim_min, ssim_max, ssim_rows

def fid_metric_calculation(original_dir, counterfactual_dir, device, batch_size, num_workers):
    counterfactual_images = [
        path for path in list_images(counterfactual_dir) if "_counterfactual" in path.stem
    ]
    reference_images = list_images(original_dir)
    reference_images = [path for path in reference_images if "_original" in path.stem]

    fid = compute_fid(
        reference_images,
        counterfactual_images,
        device,
        batch_size,
        num_workers,
    )
    return fid, len(counterfactual_images), len(reference_images)


