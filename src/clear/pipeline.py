"""High-level pipelines for concept-based zero-shot inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .concepts import ConceptBank
from .model import CLIP

__all__ = [
    "PromptPair",
    "PromptSet",
    "ZeroShotResult",
    "ZeroShotPipeline",
    "build_prompt_embeddings",
]

# Type alias for embedding functions
EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass(frozen=True)
class PromptPair:
    """Positive/negative prompt collections for a single label."""

    positive: Sequence[str]
    negative: Sequence[str]

    def all_text(self) -> Tuple[str, ...]:
        return tuple(self.positive) + tuple(self.negative)


@dataclass(frozen=True)
class PromptSet:
    """Container mapping label names to prompt pairs."""

    prompts: Mapping[str, PromptPair]

    def texts(self) -> Tuple[str, ...]:
        seen = []
        for pair in self.prompts.values():
            seen.extend(pair.positive)
            seen.extend(pair.negative)
        # keep order but drop duplicates
        unique = []
        seen_set = set()
        for text in seen:
            if text not in seen_set:
                unique.append(text)
                seen_set.add(text)
        return tuple(unique)


@dataclass
class ZeroShotResult:
    """Outputs produced by :class:`ZeroShotPipeline`."""

    probabilities: Mapping[str, np.ndarray]
    logits: Mapping[str, np.ndarray]
    image_embeddings: np.ndarray


def _normalize_tensor(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return tensor / tensor.norm(dim=dim, keepdim=True).clamp_min_(1e-6)


def build_prompt_embeddings(
    prompts: PromptSet,
    embed_fn: EmbedFn,
    *,
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """Embed prompts using ``embed_fn`` and return torch tensors."""

    registry: Dict[str, torch.Tensor] = {}
    all_texts = prompts.texts()
    embeddings = embed_fn(all_texts)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(all_texts):
        raise ValueError("Embedding function must return a 2D array aligned with inputs")

    tensor_map: Dict[str, torch.Tensor] = {
        text: torch.from_numpy(vec.astype(np.float32)) for text, vec in zip(all_texts, embeddings)
    }

    for label, pair in prompts.prompts.items():
        pos = torch.stack([tensor_map[text] for text in pair.positive], dim=0)
        neg = torch.stack([tensor_map[text] for text in pair.negative], dim=0)
        if normalize:
            pos = _normalize_tensor(pos, dim=-1)
            neg = _normalize_tensor(neg, dim=-1)
        registry[label] = torch.stack((pos.mean(dim=0), neg.mean(dim=0)), dim=0)
    return registry


class ZeroShotPipeline:
    """End-to-end concept-based zero-shot inference pipeline."""

    def __init__(
        self,
        model: CLIP,
        concept_bank: ConceptBank,
        *,
        concept_chunk_size: int = 8192,
        device: Optional[torch.device] = None,
    ) -> None:
        if not concept_bank.has_clip_embeddings() or not concept_bank.has_llm_embeddings():
            raise ValueError("ConceptBank must contain both CLIP and LLM embeddings")

        self.model = model
        self.device = device or next(model.parameters()).device
        self.model.to(self.device).eval()

        self.concept_bank = concept_bank
        self.concept_chunk_size = concept_chunk_size
        # store embeddings on CPU as float32 to avoid exhausting accelerator memory
        self._clip_embeddings = torch.from_numpy(concept_bank.ensure_clip_embeddings()).float()
        self._llm_embeddings = torch.from_numpy(concept_bank.ensure_llm_embeddings()).float()

    def _iter_embedding_chunks(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
        clip = self._clip_embeddings
        llm = self._llm_embeddings
        for start in range(0, clip.shape[0], self.concept_chunk_size):
            stop = min(start + self.concept_chunk_size, clip.shape[0])
            clip_chunk = clip[start:stop].to(self.device)
            llm_chunk = llm[start:stop].to(self.device)
            yield _normalize_tensor(clip_chunk, dim=1), llm_chunk

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        visuals = self.model.encode_image(images.to(self.device))
        return _normalize_tensor(visuals, dim=-1)

    @torch.no_grad()
    def project_to_concept_space(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project normalized image features into the concept embedding space."""

        embeddings = torch.zeros(
            image_features.shape[0],
            self._llm_embeddings.shape[1],
            device=self.device,
            dtype=image_features.dtype,
        )
        for clip_chunk, llm_chunk in self._iter_embedding_chunks():
            scores = image_features @ clip_chunk.T
            embeddings += scores @ llm_chunk
        return _normalize_tensor(embeddings, dim=-1)

    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        prompts: PromptSet,
        *,
        embed_fn: EmbedFn,
        show_progress: bool = False,
    ) -> ZeroShotResult:
        label_embeddings = build_prompt_embeddings(prompts, embed_fn)
        # Convert to device tensors
        pos_matrix = torch.stack([emb[0] for emb in label_embeddings.values()], dim=0).to(self.device)
        neg_matrix = torch.stack([emb[1] for emb in label_embeddings.values()], dim=0).to(self.device)
        pos_matrix = _normalize_tensor(pos_matrix, dim=-1)
        neg_matrix = _normalize_tensor(neg_matrix, dim=-1)

        label_names = list(label_embeddings.keys())
        probs: MutableMapping[str, list[np.ndarray]] = {name: [] for name in label_names}
        logits: MutableMapping[str, list[np.ndarray]] = {name: [] for name in label_names}
        image_embeddings: list[np.ndarray] = []

        iterator = dataloader
        if show_progress:
            iterator = tqdm(dataloader, desc="zero-shot")

        for batch in iterator:
            images = batch["img"] if isinstance(batch, dict) else batch
            if isinstance(images, (tuple, list)):
                images = images[0]
            features = self.encode_images(images)
            embeddings = self.project_to_concept_space(features)

            pos = embeddings @ pos_matrix.T
            neg = embeddings @ neg_matrix.T
            stacked = torch.stack((pos, neg), dim=-1)
            prob = F.softmax(stacked, dim=-1)[..., 0]

            for idx, name in enumerate(label_names):
                probs[name].append(prob[:, idx].cpu().numpy())
                logits[name].append(pos[:, idx].cpu().numpy())

            image_embeddings.append(embeddings.cpu().numpy())

        concatenated_probs = {name: np.concatenate(chunks, axis=0) for name, chunks in probs.items()}
        concatenated_logits = {name: np.concatenate(chunks, axis=0) for name, chunks in logits.items()}
        image_matrix = np.concatenate(image_embeddings, axis=0)
        return ZeroShotResult(concatenated_probs, concatenated_logits, image_matrix)

