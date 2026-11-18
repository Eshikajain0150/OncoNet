"""Dataset construction, splits, stain normalisation, augmentation and PyTorch datasets
(manuscript Section 3.1)."""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T


# =========================================================================== image I/O
def imread_rgb(path) -> np.ndarray:
    """Unicode-safe RGB reader (cv2.imread fails on some Windows paths)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_rgb(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Cannot encode {path}")
    buf.tofile(str(path))


def imread_gray(path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Cannot read mask: {path}")
    return img


def imwrite_gray(path, img: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise IOError(f"Cannot encode {path}")
    buf.tofile(str(path))


def resize_rgb(img: np.ndarray, size: int) -> np.ndarray:
    interp = cv2.INTER_AREA if min(img.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(img, (size, size), interpolation=interp)


# =========================================================================== discovery
def discover_class_folders(raw_dir, extensions) -> dict[str, list[Path]]:
    """Return {folder_name_lower: [image paths]} for every folder that holds images.
    Duplicate copies of the same file name inside identically named folders (nested
    archive copies) are counted once."""
    raw_dir = Path(raw_dir)
    exts = {e.lower() for e in extensions}
    found: dict[str, dict[str, Path]] = {}
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            folder = p.parent.name.lower()
            found.setdefault(folder, {})
            found[folder].setdefault(p.name, p)
    return {k: [v[n] for n in sorted(v)] for k, v in found.items()}


def _even_allocation(total: int, n_parts: int) -> list[int]:
    base = total // n_parts
    rem = total - base * n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def build_subset(cfg) -> pd.DataFrame:
    """Draw the task-specific 500-image subset (seeded) using the configured mapping."""
    ds = cfg["dataset"]
    folders = discover_class_folders(cfg["paths"]["raw_dir"], ds["image_extensions"])
    if not folders:
        raise FileNotFoundError(
            f"No images found under {cfg['paths']['raw_dir']}. Extract the Kaggle archive there.")
    rng = random.Random(cfg["seed"])
    rows = []
    for label_idx, cls in enumerate(ds["class_names"]):
        src = [f.lower() for f in ds["mapping"][cls]]
        missing = [f for f in src if f not in folders]
        if missing:
            raise FileNotFoundError(
                f"Folder(s) {missing} for class {cls} not found. Available: {sorted(folders)}")
        alloc = _even_allocation(int(ds["counts"][cls]), len(src))
        for folder, k in zip(src, alloc):
            pool = list(folders[folder])
            if k > len(pool):
                raise ValueError(f"Requested {k} images from '{folder}' but only {len(pool)} exist")
            for p in rng.sample(pool, k):
                rows.append({"src_path": str(p), "src_folder": folder,
                             "label": cls, "label_idx": label_idx})
    df = pd.DataFrame(rows)
    df = df.sort_values(["label_idx", "src_folder", "src_path"]).reset_index(drop=True)
    df.insert(0, "image_id", [f"img_{i:04d}" for i in range(len(df))])
    return df


def assign_splits(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """70/15/15 stratified hold-out + 5 stratified folds on the 425-image development pool."""
    seed = cfg["seed"]
    sp = cfg["dataset"]["split"]
    idx = np.arange(len(df))
    y = df["label_idx"].values
    tr, rest = train_test_split(idx, test_size=sp["val"] + sp["test"], stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=sp["test"] / (sp["val"] + sp["test"]),
                              stratify=y[rest], random_state=seed)
    split = np.empty(len(df), dtype=object)
    split[tr], split[va], split[te] = "train", "val", "test"
    df = df.copy()
    df["split"] = split
    dev = np.where(df["split"] != "test")[0]
    skf = StratifiedKFold(n_splits=cfg["dataset"]["cv_folds"], shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=int)
    for k, (_, f_idx) in enumerate(skf.split(dev, y[dev])):
        fold[dev[f_idx]] = k
    df["cv_fold"] = fold
    return df


def class_weights_from_counts(counts: list[int]) -> list[float]:
    """w_c = N / (C * n_c)  (Eq. 18), computed from the training partition only."""
    N, C = sum(counts), len(counts)
    return [N / (C * n) for n in counts]


# =========================================================================== Reinhard
class ReinhardNormalizer:
    """Reinhard et al. (2001) colour transfer in CIE-Lab space."""

    def __init__(self, mean=None, std=None):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)

    @staticmethod
    def lab_stats(img: np.ndarray):
        lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        return lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6

    def fit(self, img: np.ndarray):
        self.mean, self.std = self.lab_stats(img)
        return self

    def transform(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
        lab = (lab - m) / s * self.std + self.mean
        lab[..., 0] = np.clip(lab[..., 0], 0, 100)
        lab[..., 1:] = np.clip(lab[..., 1:], -127, 127)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


def choose_reference_image(paths) -> tuple[int, np.ndarray]:
    """Representative training image: Lab mean closest to the median Lab mean."""
    stats = np.stack([np.concatenate(ReinhardNormalizer.lab_stats(imread_rgb(p))) for p in paths])
    med = np.median(stats, axis=0)
    scale = stats.std(0) + 1e-6
    i = int(np.argmin((((stats - med) / scale) ** 2).sum(1)))
    return i, stats[i]


# =========================================================================== augmentation
class AddGaussianNoise(torch.nn.Module):
    """I' = I + eps, eps ~ N(0, sigma^2), sigma ~ U(0, max_std); images in [0, 1]."""

    def __init__(self, max_std: float):
        super().__init__()
        self.max_std = max_std

    def forward(self, img, *rest):
        if self.max_std > 0:
            sigma = float(torch.rand(1)) * self.max_std
            img = (img + torch.randn_like(img) * sigma).clamp(0, 1)
            img = tv_tensors.wrap(img, like=img) if isinstance(img, tv_tensors.TVTensor) else img
        return (img, *rest) if rest else img


def seg_transforms(cfg, train: bool, mean, std):
    a, size = cfg["augmentation"], cfg["preprocess"]["seg_size"]
    ops = [T.ToImage()]
    if train:
        ops += [
            T.RandomRotation(a["rotation_deg"]),
            T.RandomHorizontalFlip(a["hflip_p"]),
            T.RandomVerticalFlip(a["vflip_p"]),
            T.RandomResizedCrop(size, scale=tuple(a["rrc_scale"]), antialias=True),
            T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        ]
    ops += [T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops), _Post(cfg, train, mean, std)


class _Post:
    def __init__(self, cfg, train, mean, std):
        self.noise = AddGaussianNoise(cfg["augmentation"]["noise_std_max"]) if train else None
        self.norm = T.Normalize(mean, std)

    def __call__(self, img):
        img = torch.as_tensor(img)
        if self.noise is not None:
            img = self.noise(img)
        return self.norm(img)


def cls_transforms(cfg, train: bool, mean, std):
    a, size = cfg["augmentation"], cfg["preprocess"]["vit_size"]
    ops = [T.ToImage()]
    if train:
        ops += [
            T.RandomRotation(a["rotation_deg"]),
            T.RandomHorizontalFlip(a["hflip_p"]),
            T.RandomVerticalFlip(a["vflip_p"]),
            T.RandomResizedCrop(size, scale=tuple(a["rrc_scale"]), antialias=True),
            T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        ]
    else:
        ops += [T.Resize((size, size), antialias=True)]
    ops += [T.ToDtype(torch.float32, scale=True)]
    return T.Compose(ops), _Post(cfg, train, mean, std)


# =========================================================================== datasets
class SegDataset(Dataset):
    def __init__(self, df, img_dir, mask_dir, cfg, train, mean, std):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.mask_dir = None if mask_dir is None else Path(mask_dir)
        self.geo, self.post = seg_transforms(cfg, train, mean, std)

    def __len__(self):
        return len(self.df)

    def image_tensor(self, i):
        r = self.df.iloc[i]
        img = imread_rgb(self.img_dir / f"{r.image_id}.png")
        t = self.geo(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)))
        return self.post(t.as_subclass(torch.Tensor)), r.image_id

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = imread_rgb(self.img_dir / f"{r.image_id}.png")
        msk = (imread_gray(self.mask_dir / f"{r.image_id}.png") > 127).astype(np.uint8)
        img_t, msk_t = self.geo(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
                                tv_tensors.Mask(torch.from_numpy(msk)[None]))
        img_t = self.post(img_t.as_subclass(torch.Tensor))
        return img_t, msk_t.as_subclass(torch.Tensor).float(), r.image_id


class ROIDataset(Dataset):
    """Two-view (local 128 / global 256) ROI dataset; ``views`` selects which are returned."""

    def __init__(self, df, roi_dir, cfg, train, mean, std, views=("local", "global")):
        self.df = df.reset_index(drop=True)
        self.roi_dir = Path(roi_dir)
        self.views = views
        self.geo, self.post = cls_transforms(cfg, train, mean, std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        out = []
        for v in self.views:
            img = imread_rgb(self.roi_dir / v / f"{r.image_id}.png")
            t = self.geo(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)))
            out.append(self.post(t.as_subclass(torch.Tensor)))
        return out, int(r.label_idx), r.image_id


def collate_views(batch):
    views = [torch.stack([b[0][k] for b in batch]) for k in range(len(batch[0][0]))]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ids = [b[2] for b in batch]
    return views, y, ids


# =========================================================================== MixUp / CutMix
def mix_batch(views: list[torch.Tensor], y_onehot: torch.Tensor, cfg, rng: np.random.Generator):
    """With p=0.25 MixUp (alpha=0.2), else with p=0.25 CutMix (alpha=1.0) (Eq. 3).
    The same lambda / permutation / relative box is used for every view."""
    a = cfg["augmentation"]
    u = rng.random()
    B = y_onehot.shape[0]
    if B < 2:
        return views, y_onehot, None
    perm = torch.from_numpy(rng.permutation(B)).to(y_onehot.device)
    if u < a["mixup_p"]:
        lam = float(rng.beta(a["mixup_alpha"], a["mixup_alpha"]))
        views = [lam * v + (1 - lam) * v[perm] for v in views]
        return views, lam * y_onehot + (1 - lam) * y_onehot[perm], "mixup"
    if u < a["mixup_p"] + a["cutmix_p"]:
        lam = float(rng.beta(a["cutmix_alpha"], a["cutmix_alpha"]))
        rx, ry = rng.random(), rng.random()
        rw = math.sqrt(1 - lam)
        new_views = []
        lam_eff = lam
        for v in views:
            H, W = v.shape[-2:]
            cw, ch = int(W * rw), int(H * rw)
            cx, cy = int(rx * W), int(ry * H)
            x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
            y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
            v = v.clone()
            v[..., y1:y2, x1:x2] = v[perm][..., y1:y2, x1:x2]
            lam_eff = 1 - (x2 - x1) * (y2 - y1) / (W * H)
            new_views.append(v)
        return new_views, lam_eff * y_onehot + (1 - lam_eff) * y_onehot[perm], "cutmix"
    return views, y_onehot, None


# =========================================================================== manifest helpers
def load_manifest(cfg) -> pd.DataFrame:
    p = Path(cfg["paths"]["processed_dir"]) / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError("manifest.csv missing - run scripts/01_prepare_data.py first")
    return pd.read_csv(p)


def split_frames(df, fold: int | None = None, seed: int = 42, inner_val: float = 0.15):
    """Return (train, val, test) frames.
    fold=None -> the primary 350/75/75 hold-out.
    fold=k    -> CV fold k of the 425-image development pool: evaluation = fold k,
                 training = remaining four folds, of which a stratified 15% inner split
                 is used only for checkpoint selection. The 75-image test set is never used."""
    if fold is None:
        return (df[df.split == "train"].copy(), df[df.split == "val"].copy(),
                df[df.split == "test"].copy())
    dev = df[df.split != "test"]
    ev = dev[dev.cv_fold == fold].copy()
    rest = dev[dev.cv_fold != fold]
    tr_idx, va_idx = train_test_split(rest.index.values, test_size=inner_val,
                                      stratify=rest.label_idx.values, random_state=seed)
    return rest.loc[tr_idx].copy(), rest.loc[va_idx].copy(), ev
