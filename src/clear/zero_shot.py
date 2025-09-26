"""Zero-shot inference helpers tailored to chest X-ray experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize
from tqdm.auto import tqdm

from . import clip as clip_module
from .evaluation import bootstrap, evaluate, sigmoid
from .model import CLIP

__all__ = [
    "CXRTestDataset",
    "load_clip",
    "zeroshot_classifier",
    "predict",
    "run_single_prediction",
    "process_alt_labels",
    "run_softmax_eval",
    "run_experiment",
    "make_true_labels",
    "make",
    "ensemble_models",
    "run_zero_shot",
]


class CXRTestDataset(Dataset):
    """Dataset wrapper around an HDF5 archive containing CXR images."""

    def __init__(self, img_path: str | Path, transform: Optional[Compose] = None) -> None:
        self.img_path = Path(img_path)
        self._file = h5py.File(self.img_path, "r")
        self.img_dset = self._file["cxr"]
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.img_dset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = int(idx.item())

        img = self.img_dset[idx]  # (H, W)
        img = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1).float()

        if self.transform is not None:
            img = self.transform(img)

        return {"img": img}

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            finally:
                self._file = None

    def __del__(self) -> None:  # pragma: no cover - defensive resource cleanup
        self.close()


def _default_model_params(context_length: int) -> Dict[str, int]:
    return {
        "embed_dim": 768,
        "image_resolution": 320,
        "vision_layers": 12,
        "vision_width": 768,
        "vision_patch_size": 16,
        "context_length": context_length,
        "vocab_size": 49408,
        "transformer_width": 512,
        "transformer_heads": 8,
        "transformer_layers": 12,
    }


def load_clip(
    model_path: str | Path,
    *,
    pretrained: bool = False,
    context_length: int = 77,
    device: Optional[torch.device] = None,
    use_dinov2: bool = False,
    dinov2_model_name: str = "dinov2_vitb14",
    freeze_dinov2: bool = False,
    strict: bool = True,
) -> CLIP:
    """Load a CLIP checkpoint saved via ``state_dict`` serialization."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pretrained:
        model, _ = clip_module.load("ViT-B/32", device=device, jit=False)
    else:
        params = _default_model_params(context_length)
        model = CLIP(**params).to(device)

    if use_dinov2:
        backbone = torch.hub.load("facebookresearch/dinov2", dinov2_model_name, pretrained=True)
        backbone = backbone.to(device)

        with torch.no_grad():
            features = backbone(torch.randn(1, 3, 224, 224, device=device))
            backbone_dim = features.shape[-1]

        class DinoV2Visual(nn.Module):
            def __init__(self, base: nn.Module, input_dim: int, output_dim: int) -> None:
                super().__init__()
                self.base = base
                self.projection = nn.Linear(input_dim, output_dim)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                feats = self.base(x)
                return self.projection(feats)

            @property
            def conv1(self) -> nn.Module:  # pragma: no cover - compatibility shim
                return self.projection

        model.visual = DinoV2Visual(backbone, backbone_dim, model.visual.output_dim)

        if freeze_dinov2:
            for param in model.visual.base.parameters():
                param.requires_grad = False

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()
    return model


def zeroshot_classifier(
    classnames: Sequence[str],
    templates: Sequence[str],
    model: CLIP,
    *,
    context_length: int = 77,
    device: Optional[torch.device] = None,
    show_progress: bool = False,
) -> torch.Tensor:
    """Average text embeddings across templates for each class."""

    device = device or next(model.parameters()).device
    iterator: Iterable[str] = classnames
    if show_progress:
        iterator = tqdm(iterator, desc="zeroshot-classes")

    embeddings = []
    with torch.no_grad():
        for name in iterator:
            texts = [template.format(name) for template in templates]
            tokens = clip_module.tokenize(texts, context_length=context_length).to(device)
            class_embeds = model.encode_text(tokens)
            class_embeds = class_embeds / class_embeds.norm(dim=-1, keepdim=True)
            pooled = class_embeds.mean(dim=0)
            embeddings.append(pooled / pooled.norm())

    return torch.stack(embeddings, dim=1)


def predict(
    loader: DataLoader,
    model: CLIP,
    zeroshot_weights: torch.Tensor,
    *,
    to_numpy: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """Project images onto zero-shot weights and return logits."""

    device = next(model.parameters()).device
    predictions: List[np.ndarray] = []

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc="zero-shot")

    with torch.no_grad():
        for batch in iterator:
            images = batch["img"].to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ zeroshot_weights
            predictions.append(logits.cpu().numpy())

    stacked = np.concatenate(predictions, axis=0)
    return stacked if to_numpy else stacked


def run_single_prediction(
    classnames: Sequence[str],
    template: str,
    model: CLIP,
    loader: DataLoader,
    *,
    context_length: int = 77,
    softmax_eval: bool = True,
) -> np.ndarray:
    weights = zeroshot_classifier(
        classnames,
        [template],
        model,
        context_length=context_length,
    )
    logits = predict(loader, model, weights)
    if not softmax_eval:
        logits = sigmoid((logits - logits.mean()) / (logits.std() + 1e-8))
    return logits


def process_alt_labels(
    alt_labels_dict: Optional[Mapping[str, Sequence[str]]],
    cxr_labels: Sequence[str],
) -> Tuple[Optional[List[str]], Optional[Dict[str, int]]]:
    if alt_labels_dict is None:
        return None, None

    inverse: Dict[str, str] = {}
    for main, alternatives in alt_labels_dict.items():
        inverse[main] = main
        for alt in alternatives:
            inverse[alt] = main

    index_map = {label: idx for idx, label in enumerate(cxr_labels)}
    alt_labels = list(inverse.keys())
    alt_idx_map = {alt: index_map[inverse[alt]] for alt in alt_labels}
    return alt_labels, alt_idx_map


def run_softmax_eval(
    model: CLIP,
    loader: DataLoader,
    labels: Sequence[str],
    pair_template: Tuple[str, str],
    *,
    context_length: int = 77,
) -> np.ndarray:
    pos_template, neg_template = pair_template
    pos_logits = run_single_prediction(labels, pos_template, model, loader, context_length=context_length)
    neg_logits = run_single_prediction(labels, neg_template, model, loader, context_length=context_length)
    logits = np.stack([pos_logits, neg_logits], axis=-1)
    scores = np.exp(logits)
    return scores[..., 0] / scores.sum(axis=-1)


def run_experiment(
    model: CLIP,
    labels: Sequence[str],
    templates: Sequence[str | Tuple[str, str]],
    loader: DataLoader,
    y_true: np.ndarray,
    *,
    alt_labels_dict: Optional[Mapping[str, Sequence[str]]] = None,
    softmax_eval: bool = True,
    context_length: int = 77,
    use_bootstrap: bool = True,
    show_progress: bool = False,
) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]] | List[pd.DataFrame], np.ndarray]:
    alt_labels, alt_idx_map = process_alt_labels(alt_labels_dict, labels)
    eval_labels = alt_labels or list(labels)

    results: List = []
    last_pred: Optional[np.ndarray] = None

    iterator = templates
    if show_progress:
        iterator = tqdm(templates, desc="templates")

    for template in iterator:
        if softmax_eval:
            if not isinstance(template, tuple) or len(template) != 2:
                raise ValueError("Softmax evaluation expects template pairs")
            y_pred = run_softmax_eval(
                model,
                loader,
                eval_labels,
                template,
                context_length=context_length,
            )
        else:
            if not isinstance(template, str):
                raise ValueError("Non-softmax evaluation expects string templates")
            y_pred = run_single_prediction(
                eval_labels,
                template,
                model,
                loader,
                context_length=context_length,
                softmax_eval=False,
            )

        last_pred = y_pred

        if use_bootstrap:
            results.append(
                bootstrap(
                    y_pred,
                    y_true,
                    eval_labels,
                    label_idx_map=alt_idx_map,
                )
            )
        else:
            results.append(
                evaluate(y_pred, y_true, eval_labels, label_idx_map=alt_idx_map)
            )

    if last_pred is None:
        raise ValueError("No templates were provided")

    return results, last_pred


def make_true_labels(
    label_path: str | Path,
    labels: Sequence[str],
    *,
    cutlabels: bool = True,
) -> np.ndarray:
    df = pd.read_csv(label_path)
    if cutlabels:
        df = df.loc[:, labels]
    else:
        df = df.iloc[:, 1:]
    return df.to_numpy(dtype=np.float32)


def make(
    model_path: str | Path,
    cxr_filepath: str | Path,
    *,
    pretrained: bool = True,
    context_length: int = 77,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[CLIP, DataLoader]:
    model = load_clip(
        model_path,
        pretrained=pretrained,
        context_length=context_length,
    )

    transform = Compose(
        [
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(448 if pretrained else model.visual.input_resolution, interpolation=InterpolationMode.BICUBIC),
        ]
    )

    dataset = CXRTestDataset(cxr_filepath, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return model, loader


def ensemble_models(
    model_paths: Sequence[str | Path],
    cxr_filepath: str | Path,
    labels: Sequence[str],
    pair_template: Tuple[str, str],
    *,
    cache_dir: Optional[str | Path] = None,
    save_name: Optional[str] = None,
) -> Tuple[List[np.ndarray], np.ndarray]:
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    predictions: List[np.ndarray] = []

    for path in sorted(map(Path, model_paths)):
        cache_path: Optional[Path] = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            suffix = save_name or path.stem
            cache_path = cache_dir / f"{suffix}.npy"

        if cache_path is not None and cache_path.exists():
            preds = np.load(cache_path)
        else:
            model, loader = make(path, cxr_filepath)
            preds = run_softmax_eval(model, loader, labels, pair_template)
            if cache_path is not None:
                np.save(cache_path, preds)
        predictions.append(preds)

    stacked = np.stack(predictions, axis=0)
    return predictions, stacked.mean(axis=0)


def run_zero_shot(
    labels: Sequence[str],
    templates: Sequence[str | Tuple[str, str]],
    model_path: str | Path,
    *,
    cxr_filepath: str | Path,
    final_label_path: str | Path,
    alt_labels_dict: Optional[Mapping[str, Sequence[str]]] = None,
    softmax_eval: bool = True,
    context_length: int = 77,
    pretrained: bool = False,
    use_bootstrap: bool = True,
    cutlabels: bool = True,
    batch_size: int = 1,
    num_workers: int = 0,
) -> Tuple[List, np.ndarray]:
    model, loader = make(
        model_path,
        cxr_filepath,
        pretrained=pretrained,
        context_length=context_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    y_true = make_true_labels(final_label_path, labels, cutlabels=cutlabels)

    return run_experiment(
        model,
        labels,
        templates,
        loader,
        y_true,
        alt_labels_dict=alt_labels_dict,
        softmax_eval=softmax_eval,
        context_length=context_length,
        use_bootstrap=use_bootstrap,
    )

