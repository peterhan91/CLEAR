"""Utilities for loading and working with radiological concept banks."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import json
import csv
import numpy as np

__all__ = ["ConceptBank"]


def _load_concept_list(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix in {".jsonl", ".json"}:
        concepts: List[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    payload = json.loads(line.replace("'", '"'))
                if isinstance(payload, dict):
                    value = payload.get("concept") or payload.get("text") or payload.get("observation")
                    if value is None:
                        raise ValueError(f"JSON line must contain a 'concept' field: {line[:80]}")
                    concepts.append(str(value))
                else:
                    concepts.append(str(payload))
        return concepts

    if path.suffix in {".csv", ".tsv"}:
        delimiter = "," if path.suffix == ".csv" else "\t"
        concepts = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            field = reader.fieldnames[0] if reader.fieldnames else None
            if field is None:
                raise ValueError(f"No header row found in {path}")
            for row in reader:
                concepts.append(str(row[field]))
        return concepts

    # default to newline-delimited text file
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _load_array(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "embeddings" in data:
            array = data["embeddings"]
        else:
            # fall back to the first entry
            array = data[data.files[0]]
    else:
        array = data
    return np.asarray(array, dtype=np.float32)


@dataclass(slots=True)
class ConceptBank:
    """Container for radiological concepts and associated embeddings.

    Parameters
    ----------
    concepts:
        Sequence of concept strings ordered consistently with the embedding matrices.
    clip_embeddings:
        Optional matrix of CLIP text-encoder features for each concept. Shape must
        be ``(num_concepts, clip_dim)`` when provided.
    llm_embeddings:
        Optional matrix of LLM embeddings for each concept. Shape must be
        ``(num_concepts, llm_dim)`` when provided.
    metadata:
        Optional dictionary of auxiliary metadata (e.g., dataset statistics).
    """

    concepts: Sequence[str]
    clip_embeddings: Optional[np.ndarray] = None
    llm_embeddings: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.concepts = tuple(self.concepts)
        n_concepts = len(self.concepts)

        if self.clip_embeddings is not None:
            if self.clip_embeddings.shape[0] != n_concepts:
                raise ValueError(
                    "clip_embeddings rows must match number of concepts"
                )
        if self.llm_embeddings is not None:
            if self.llm_embeddings.shape[0] != n_concepts:
                raise ValueError(
                    "llm_embeddings rows must match number of concepts"
                )

    @classmethod
    def from_files(
        cls,
        concepts_path: str | Path,
        *,
        clip_embeddings_path: str | Path | None = None,
        llm_embeddings_path: str | Path | None = None,
        metadata: Optional[dict] = None,
    ) -> "ConceptBank":
        """Load a concept bank from disk.

        ``concepts_path`` may be a JSON Lines file containing objects with a
        ``concept`` field, a CSV/TSV file, or a simple newline-delimited text
        file. Embedding paths may point to ``.npy`` or ``.npz`` files containing
        an array named ``embeddings`` (or the first array when unnamed).
        """

        concepts = _load_concept_list(Path(concepts_path))
        clip_embeddings = _load_array(Path(clip_embeddings_path)) if clip_embeddings_path else None
        llm_embeddings = _load_array(Path(llm_embeddings_path)) if llm_embeddings_path else None
        return cls(
            concepts=concepts,
            clip_embeddings=clip_embeddings,
            llm_embeddings=llm_embeddings,
            metadata=metadata or {},
        )

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    def has_clip_embeddings(self) -> bool:
        return self.clip_embeddings is not None

    def has_llm_embeddings(self) -> bool:
        return self.llm_embeddings is not None

    def ensure_clip_embeddings(self) -> np.ndarray:
        if self.clip_embeddings is None:
            raise ValueError("ConceptBank does not contain CLIP embeddings")
        return self.clip_embeddings

    def ensure_llm_embeddings(self) -> np.ndarray:
        if self.llm_embeddings is None:
            raise ValueError("ConceptBank does not contain LLM embeddings")
        return self.llm_embeddings

    def subset(self, indices: Sequence[int]) -> "ConceptBank":
        """Return a new ``ConceptBank`` containing only ``indices``."""

        concepts = [self.concepts[i] for i in indices]
        clip_embeddings = None
        llm_embeddings = None
        if self.clip_embeddings is not None:
            clip_embeddings = self.clip_embeddings[np.asarray(indices)]
        if self.llm_embeddings is not None:
            llm_embeddings = self.llm_embeddings[np.asarray(indices)]
        metadata = dict(self.metadata)
        metadata["subset_indices"] = list(indices)
        return ConceptBank(
            concepts=concepts,
            clip_embeddings=clip_embeddings,
            llm_embeddings=llm_embeddings,
            metadata=metadata,
        )

    def iter_chunks(self, chunk_size: int) -> Iterable[tuple[np.ndarray, Optional[np.ndarray]]]:
        """Yield concept embedding chunks to manage memory usage."""

        clip = self.ensure_clip_embeddings()
        llm = self.llm_embeddings
        for start in range(0, self.num_concepts, chunk_size):
            stop = min(start + chunk_size, self.num_concepts)
            clip_chunk = clip[start:stop]
            llm_chunk = llm[start:stop] if llm is not None else None
            yield clip_chunk, llm_chunk

