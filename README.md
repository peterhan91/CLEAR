# CLEAR

Concept-Level Embeddings for Auditable Radiology (CLEAR) packages the tooling behind the paper
*CLEAR: An Auditable Foundation Model for Radiology Grounded in Clinical Concepts*. The code base
follows a standard Python package layout so it is straightforward to install, extend, and import
from other projects.

## Features

- Thin wrapper around CLIP checkpoints with reliable download and tokenisation utilities.
- Concept-bank utilities for loading the radiological observation space (CLIP and LLM embeddings).
- A streaming zero-shot pipeline that projects images into the LLM concept space without requiring
  the full concept matrix in accelerator memory.
- Metric helpers for threshold search, F1, Matthews correlation coefficient analysis, and
  bootstrapped confidence intervals.
- Example scripts for benchmarking, model auditing, and concept bottleneck experiments.

## Installation

1. Ensure you are running Python 3.9 or newer.
2. (Optional) Create and activate a virtual environment.
3. Install the package in editable mode:

   ```bash
   pip install -e .[visualization]
   ```

   The optional `visualization` extra pulls in matplotlib for plotting utilities.

## Quick Start

```python
import numpy as np
from torch.utils.data import DataLoader

from clear import (
    ConceptBank,
    PromptPair,
    PromptSet,
    ZeroShotPipeline,
    load,
)

# 1. Load the pretrained CLEAR checkpoint and preprocessing transform.
model, preprocess = load("ViT-B/32", device="cpu", jit=False)

# 2. Instantiate the concept bank (paths point to your precomputed artefacts).
concept_bank = ConceptBank.from_files(
    "assets/concepts.jsonl",
    clip_embeddings_path="assets/concepts_clip_embeddings.npz",
    llm_embeddings_path="assets/concepts_llm_embeddings.npz",
)

# 3. Build a PyTorch dataloader that yields preprocessed images.
dataloader = DataLoader(...)

# 4. Define positive/negative prompt pairs for each label.
prompts = PromptSet({
    "Atelectasis": PromptPair(["atelectasis"], ["no atelectasis"]),
    "Pleural Effusion": PromptPair(["pleural effusion"], ["no pleural effusion"]),
})

# 5. Provide an embedding lookup that returns the LLM embeddings for prompts.
prompt_vectors = np.load("assets/prompt_embeddings.npy", allow_pickle=True).item()

def embed_prompts(texts):
    return np.stack([prompt_vectors[text] for text in texts], axis=0).astype("float32")

# 6. Run concept-based zero-shot inference.
pipeline = ZeroShotPipeline(model, concept_bank)
result = pipeline.predict(dataloader, prompts, embed_fn=embed_prompts)

print(result.probabilities["Atelectasis"][:5])
```

## Repository Layout

```
src/clear/        Core library modules (CLIP loader, concept bank, zero-shot pipeline, metrics).
examples/         Research scripts and notebooks rewritten to import the packaged modules.
scripts/          Standalone data preparation and legacy utilities.
pyproject.toml    Packaging metadata for building or installing CLEAR.
README.md         This document.
LICENSE           MIT licence text.
```

## Development

- Run formatting and linting before opening a pull request:
  ```bash
  ruff check .
  black src examples scripts
  ```
- Add tests for new functionality where practical.
- Keep large data artefacts outside the repository; the zero-shot helpers expect HDF5 datasets.

## License

All code in this repository is released under the MIT License. Portions derived from the OpenAI
CLIP repository retain the original copyright notice.
