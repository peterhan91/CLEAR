#!/usr/bin/env python3
import json
import pickle
import time
import argparse
import logging
import numpy as np
import ast
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from openai import AzureOpenAI
from tqdm import tqdm

# Local model imports
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Sentence transformers import
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# PCA import for dimension reduction
try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except ImportError:
    PCA_AVAILABLE = False
    PCA = None

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last token from the hidden states, handling padding correctly."""
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def cls_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the [CLS] token from the hidden states (first token for BERT models)."""
    return last_hidden_states[:, 0]

def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format query with instruction for instruct-based models."""
    return f'Instruct: {task_description}\nQuery: {query}'

# Model configurations for different embedding models
MODEL_CONFIGS = {
    "Qwen/Qwen3-Embedding-8B": {
        "max_seq_length": 8192,  # More practical default
        "embedding_dim": 4096,
        "pooling_method": "last_token",
        "use_flash_attention": True,
        "padding_side": "left",
        "task_description": "Generate embeddings for medical radiology concepts and observations",
        "supports_instructions": True,
        "use_transformers_direct": True,
        "model_kwargs": {"device_map": "auto", "torch_dtype": "auto"},
        "tokenizer_kwargs": {"padding_side": "left"}
    },
    "Salesforce/SFR-Embedding-Mistral": {
        "max_seq_length": 4096,
        "embedding_dim": 4096,  # Will be determined at runtime
        "pooling_method": "last_token",
        "use_flash_attention": False,
        "padding_side": "right",
        "task_description": "Given medical radiology concepts, retrieve relevant passages and generate embeddings",
        "supports_instructions": True,
        "use_transformers_direct": True,
        "model_kwargs": {"device_map": "auto"},
        "tokenizer_kwargs": {}
    },
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract": {
        "max_seq_length": 512,
        "embedding_dim": 768,
        "pooling_method": "cls_token",  # BERT uses [CLS] token
        "use_flash_attention": False,
        "padding_side": "right",
        "task_description": None,  # BERT doesn't use instructions
        "supports_instructions": False,
        "use_transformers_direct": True,
        "model_kwargs": {"device_map": "auto"},
        "tokenizer_kwargs": {}
    }
}

class RadiologyEmbeddingGenerator:
    """Generate embeddings for radiological concepts using Azure OpenAI, transformers, or sentence transformers."""
    
    def __init__(self, 
                 embedding_type: str = "local", 
                 # Azure OpenAI parameters
                 azure_endpoint: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 api_version: Optional[str] = None, 
                 embedding_model: str = "text-embedding-3-small",
                 # Local model parameters
                 local_model_name: str = "Qwen/Qwen3-Embedding-8B",
                 device: Optional[str] = None,
                 max_length: int = 8192,
                 batch_size: int = 32,
                 # Sentence transformer parameters
                 trust_remote_code: bool = True,
                 use_fp16: bool = True,
                 # PCA dimension reduction parameters
                 apply_pca: bool = False,
                 pca_dimensions: Optional[int] = None):

        self.embedding_type = embedding_type.lower()
        self.max_length = max_length
        self.batch_size = batch_size
        self.trust_remote_code = trust_remote_code
        self.use_fp16 = use_fp16
        self.apply_pca = apply_pca
        self.pca_dimensions = pca_dimensions
        self.pca_model = None
        self.original_embedding_dim = None
        
        if self.embedding_type == "azure":
            self._init_azure_client(azure_endpoint, api_key, api_version, embedding_model)
        elif self.embedding_type == "local":
            self._init_local_model(local_model_name, device)
        elif self.embedding_type == "sentence_transformers":
            self._init_sentence_transformer_model(local_model_name, device)
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Must be 'azure', 'local', or 'sentence_transformers'.")
        
        if self.apply_pca:
            if not PCA_AVAILABLE:
                raise ImportError("scikit-learn required for PCA. Install: pip install scikit-learn")
            if self.pca_dimensions is None:
                raise ValueError("pca_dimensions must be specified when apply_pca is True")
            if self.pca_dimensions not in [768, 1536]:
                raise ValueError("pca_dimensions must be 768 or 1536")
            
            self.original_embedding_dim = self.embedding_dim
            
            if self.embedding_dim > self.pca_dimensions:
                self.embedding_dim = self.pca_dimensions
                logger.info(f"PCA enabled: {self.original_embedding_dim} -> {self.pca_dimensions}")
            else:
                logger.warning(f"Embedding dim ({self.embedding_dim}) <= PCA target ({self.pca_dimensions}). PCA skipped.")
                self.apply_pca = False
        
        self.concept_embeddings = {}
    
    def _init_azure_client(self, azure_endpoint: str, api_key: str, api_version: str, embedding_model: str):
        if not all([azure_endpoint, api_key, api_version]):
            raise ValueError("Azure OpenAI requires azure_endpoint, api_key, and api_version")
        
        self.azure_endpoint = azure_endpoint
        self.api_key = api_key
        self.api_version = api_version
        self.embedding_model = embedding_model
        
        if embedding_model in ["text-embedding-ada-002", "text-embedding-3-small"]:
            self.embedding_dim = 1536
        elif embedding_model == "text-embedding-3-large":
            self.embedding_dim = 3072
        else:
            self.embedding_dim = 1536
        
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        logger.info(f"Azure OpenAI client initialized: {self.embedding_model}")
    
    def _init_local_model(self, model_name: str, device: Optional[str]):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Loading local model: {model_name} on {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=self.trust_remote_code)
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if self.use_fp16 and self.device.type == "cuda":
                self.model = self.model.half()
                logger.info("FP16 enabled")
            
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors='pt', max_length=self.max_length, truncation=True)
                test_input = {k: v.to(self.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                test_embedding = last_token_pool(test_output.last_hidden_state, test_input['attention_mask'])
                self.embedding_dim = test_embedding.shape[-1]
            
            self.local_model_name = model_name
            self.task_description = "Generate embeddings for medical radiology concepts and observations"
            
            logger.info(f"Local model loaded. Embedding dim: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load local model {model_name}: {e}")
            raise

    def _init_sentence_transformer_model(self, model_name: str, device: Optional[str]):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model_config = MODEL_CONFIGS.get(model_name, {})
        use_transformers_direct = self.model_config.get("use_transformers_direct", False)
        
        if use_transformers_direct:
            logger.info(f"Loading transformers: {model_name} on {self.device}")
            self._init_transformers_direct(model_name)
        else:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")
            
            logger.info(f"Loading sentence transformer: {model_name} on {self.device}")
            self._init_sentence_transformer_fallback(model_name)
    
    def _init_transformers_direct(self, model_name: str):
        try:
            model_kwargs = self.model_config.get("model_kwargs", {}).copy()
            tokenizer_kwargs = self.model_config.get("tokenizer_kwargs", {}).copy()
            
            if model_kwargs.get("torch_dtype") == "auto":
                if self.use_fp16 and self.device.type == "cuda" and any(x in model_name.lower() for x in ["8b", "7b"]):
                    model_kwargs["torch_dtype"] = torch.float16
                    logger.info("FP16 enabled for large model")
                else:
                    model_kwargs.pop("torch_dtype", None)
            
            if "device_map" not in model_kwargs:
                model_kwargs["device"] = self.device
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=self.trust_remote_code,
                **tokenizer_kwargs
            )
            self.model = AutoModel.from_pretrained(
                model_name, 
                trust_remote_code=self.trust_remote_code,
                **model_kwargs
            )
            
            if "device_map" not in model_kwargs:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            with torch.no_grad():
                test_input = self.tokenizer("test", return_tensors='pt', max_length=512, truncation=True)
                test_input = {k: v.to(self.model.device) for k, v in test_input.items()}
                test_output = self.model(**test_input)
                
                pooling_method = self.model_config.get("pooling_method", "last_token")
                if pooling_method == "cls_token":
                    test_embedding = cls_token_pool(test_output.last_hidden_state, test_input['attention_mask'])
                else:
                    test_embedding = last_token_pool(test_output.last_hidden_state, test_input['attention_mask'])
                
                self.embedding_dim = test_embedding.shape[-1]
            
            self.local_model_name = model_name
            self.task_description = self.model_config.get("task_description")
            self.supports_instructions = self.model_config.get("supports_instructions", False)
            self.pooling_method = self.model_config.get("pooling_method", "last_token")
            
            max_seq_length = self.model_config.get("max_seq_length", self.max_length)
            self.max_length = min(max_seq_length, self.max_length)
            
            logger.info(f"Model loaded. Dim: {self.embedding_dim}, Pooling: {self.pooling_method}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _init_sentence_transformer_fallback(self, model_name: str):
        """Initialize using sentence transformers as fallback."""
        try:
            # Prepare model and tokenizer kwargs
            model_kwargs = self.model_config.get("model_kwargs", {})
            tokenizer_kwargs = self.model_config.get("tokenizer_kwargs", {})
            
            # Override device in model_kwargs if not using device_map
            if "device_map" not in model_kwargs:
                model_kwargs["device"] = self.device
            
            # Enable FP16 for large models on GPU
            if self.use_fp16 and self.device.type == "cuda" and any(x in model_name.lower() for x in ["8b", "7b"]):
                model_kwargs["torch_dtype"] = torch.float16
                logger.info("Enabled FP16 for large model GPU acceleration")
            
            # Initialize sentence transformer
            self.model = SentenceTransformer(
                model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
                trust_remote_code=self.trust_remote_code
            )
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            # Set max sequence length if specified
            max_seq_length = self.model_config.get("max_seq_length")
            if max_seq_length:
                self.model.max_seq_length = max_seq_length
                logger.info(f"Set max sequence length to: {max_seq_length}")
            
            self.local_model_name = model_name
            self.task_description = self.model_config.get("task_description")
            self.supports_instructions = self.model_config.get("supports_instructions", False)
            
            logger.info(f"Sentence transformer model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model {model_name}: {e}")
            raise

    def _extract_unique_concepts_from_file(self, filepath: str) -> List[str]:
        logger.info(f"Loading and parsing concepts from {filepath}...")
        
        # Check file extension to determine format
        if filepath.endswith('.csv'):
            # Read CSV file with pandas
            df = pd.read_csv(filepath)
            
            # Extract concepts from the 'concept' column
            if 'concept' in df.columns:
                concepts = df['concept'].dropna().tolist()
            else:
                logger.error(f"CSV file {filepath} does not contain a 'concept' column")
                return []
                
            # Convert to lowercase and remove duplicates
            concepts = [str(concept).lower().strip() for concept in concepts]
            concepts = list(set(concepts))
            
        elif filepath.endswith('.json'):
            # Original JSON parsing logic
            with open(filepath, 'r') as f:
                results = json.load(f)

            concepts = []
            for result in tqdm(results):
                try:
                    re_dict = ast.literal_eval(result['model_output'])
                    concepts.extend(re_dict['observations'])
                except:
                    # print(result['model_output'])
                    continue
            concepts = [concept.lower() for concept in concepts]
            concepts = list(set(concepts))
        else:
            logger.error(f"Unsupported file format. Please provide a .csv or .json file")
            return []
            
        return concepts

    def _extract_concepts_with_indices_from_csv(self, filepath: str) -> Dict[int, str]:
        """Extract concepts with their original CSV indices preserved."""
        logger.info(f"Loading and parsing concepts with indices from {filepath}...")
        
        if not filepath.endswith('.csv'):
            logger.error(f"This method only supports CSV files")
            return {}
            
        # Read CSV file with pandas
        df = pd.read_csv(filepath)
        
        # Check required columns
        if 'concept' not in df.columns:
            logger.error(f"CSV file {filepath} does not contain a 'concept' column")
            return {}
        
        if 'concept_idx' not in df.columns:
            logger.error(f"CSV file {filepath} does not contain a 'concept_idx' column")
            return {}
        
        # Create mapping from concept_idx to concept text
        concept_mapping = {}
        for _, row in df.iterrows():
            concept_idx = int(row['concept_idx'])
            concept_text = str(row['concept']).lower().strip()
            
            # Only add non-empty concepts
            if concept_text and concept_text.lower() != 'nan':
                concept_mapping[concept_idx] = concept_text
        
        logger.info(f"Loaded {len(concept_mapping)} concepts with indices")
        return concept_mapping

    def get_openai_embedding(self, text: str, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Generate embedding using Azure OpenAI API.
        
        Args:
            text: Text to embed.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            Embedding vector or zero vector if failed.
        """
        if self.embedding_type != "azure":
            raise ValueError("This method is only for Azure OpenAI embeddings")
            
        cleaned_text = text.strip()
        if not cleaned_text:
            logger.warning("Empty text provided for embedding, returning zero vector.")
            return np.zeros(self.embedding_dim)
        
        # Basic text preprocessing for embedding
        cleaned_text = cleaned_text.replace("\\n", " ")  # Ensure literal \n are spaces
        cleaned_text = cleaned_text.replace("\n", " ")   # Ensure actual newlines are spaces
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=[cleaned_text],
                    model=self.embedding_model
                )
                embedding = np.array(response.data[0].embedding)
                return embedding
                
            except Exception as e:
                # Handle rate limits and other API errors
                error_message = str(e).lower()
                
                if "rate_limit" in error_message or "rate limit" in error_message:
                    wait_time = (2 ** attempt) + np.random.uniform(0, 1)
                    logger.warning(f"Rate limit hit for text: \"{cleaned_text[:50]}...\", waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                elif "invalid_request" in error_message or "invalid request" in error_message:
                    logger.error(f"Invalid request for text: \"{cleaned_text[:100]}...\". Error: {e}")
                    return np.zeros(self.embedding_dim) # Return zero vector for invalid requests
                else:
                    logger.error(f"Attempt {attempt + 1} failed for embedding generation of \"{cleaned_text[:50]}...\": {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt) # Slightly increasing backoff
                    
        logger.error(f"Failed to generate embedding for \"{cleaned_text[:50]}...\" after {max_retries} attempts, returning zero vector.")
        return np.zeros(self.embedding_dim) # Return zero vector if all retries fail
    
    def get_local_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using local transformer model.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector or zero vector if failed.
        """
        if self.embedding_type != "local":
            raise ValueError("This method is only for local model embeddings")
            
        try:
            cleaned_text = text.strip()
            if not cleaned_text:
                logger.warning("Empty text provided for embedding, returning zero vector.")
                return np.zeros(self.embedding_dim)
            
            # Format text for instruct models if needed
            if "instruct" in self.local_model_name.lower():
                formatted_text = get_detailed_instruct(self.task_description, cleaned_text)
            else:
                formatted_text = cleaned_text
            
            # Tokenize
            inputs = self.tokenizer(
                formatted_text, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize embedding
                embedding = F.normalize(embedding, p=2, dim=1)
                
                # Convert to numpy
                embedding_np = embedding.cpu().numpy().flatten()
                
            return embedding_np
            
        except Exception as e:
            logger.error(f"Error generating local embedding for \"{text[:50]}...\": {e}")
            return np.zeros(self.embedding_dim)
    
    def get_local_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts using local model for efficiency.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if self.embedding_type != "local":
            raise ValueError("This method is only for local model embeddings")
        
        if not texts:
            return []
        
        try:
            # Clean and format texts
            cleaned_texts = []
            for text in texts:
                cleaned_text = text.strip()
                if not cleaned_text:
                    cleaned_texts.append("")
                    continue
                    
                # Format text for instruct models if needed
                if "instruct" in self.local_model_name.lower():
                    formatted_text = get_detailed_instruct(self.task_description, cleaned_text)
                else:
                    formatted_text = cleaned_text
                cleaned_texts.append(formatted_text)
            
            # Tokenize batch
            inputs = self.tokenizer(
                cleaned_texts, 
                max_length=self.max_length, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                
                # Normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                # Convert to numpy
                embeddings_np = embeddings.cpu().numpy()
            
            # Handle empty texts
            result = []
            for i, text in enumerate(texts):
                if text.strip():
                    result.append(embeddings_np[i])
                else:
                    result.append(np.zeros(self.embedding_dim))
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    def get_sentence_transformer_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using sentence transformer model or direct transformers.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector or zero vector if failed.
        """
        if self.embedding_type != "sentence_transformers":
            raise ValueError("This method is only for sentence transformer embeddings")
            
        try:
            cleaned_text = text.strip()
            if not cleaned_text:
                logger.warning("Empty text provided for embedding, returning zero vector.")
                return np.zeros(self.embedding_dim)
            
            # Format text for models that support instructions
            if self.supports_instructions and self.task_description:
                if "SFR-Embedding-Mistral" in self.local_model_name:
                    # Special formatting for SFR-Embedding-Mistral
                    formatted_text = get_detailed_instruct(self.task_description, cleaned_text)
                else:
                    # General instruction formatting
                    formatted_text = f"{self.task_description}: {cleaned_text}"
            else:
                formatted_text = cleaned_text
            
            # Check if using direct transformers or sentence transformers
            if hasattr(self, 'tokenizer') and hasattr(self, 'pooling_method'):
                # Direct transformers approach
                return self._get_transformers_embedding(formatted_text)
            else:
                # Sentence transformers approach
                embedding = self.model.encode(formatted_text, convert_to_numpy=True, normalize_embeddings=True)
                return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating sentence transformer embedding for \"{text[:50]}...\": {e}")
            return np.zeros(self.embedding_dim)
    
    def _get_transformers_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using direct transformers approach."""
        # Tokenize
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use appropriate pooling method
            if self.pooling_method == "cls_token":
                embedding = cls_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            else:  # last_token
                embedding = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embedding
            embedding = F.normalize(embedding, p=2, dim=1)
            
            # Convert to numpy
            embedding_np = embedding.cpu().numpy().flatten()
        
        return embedding_np
    
    def get_sentence_transformer_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts using sentence transformer or direct transformers.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        if self.embedding_type != "sentence_transformers":
            raise ValueError("This method is only for sentence transformer embeddings")
        
        if not texts:
            return []
        
        try:
            # Clean and format texts
            formatted_texts = []
            empty_indices = []
            
            for i, text in enumerate(texts):
                cleaned_text = text.strip()
                if not cleaned_text:
                    formatted_texts.append("")
                    empty_indices.append(i)
                    continue
                    
                # Format text for models that support instructions
                if self.supports_instructions and self.task_description:
                    if "SFR-Embedding-Mistral" in self.local_model_name:
                        # Special formatting for SFR-Embedding-Mistral
                        formatted_text = get_detailed_instruct(self.task_description, cleaned_text)
                    else:
                        # General instruction formatting
                        formatted_text = f"{self.task_description}: {cleaned_text}"
                else:
                    formatted_text = cleaned_text
                
                formatted_texts.append(formatted_text)
            
            # Check if using direct transformers or sentence transformers
            if hasattr(self, 'tokenizer') and hasattr(self, 'pooling_method'):
                # Direct transformers approach
                return self._get_transformers_embeddings_batch(formatted_texts, empty_indices)
            else:
                # Sentence transformers approach
                embeddings = self.model.encode(
                    formatted_texts, 
                    convert_to_numpy=True, 
                    normalize_embeddings=True,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
                
                # Handle empty texts
                result = []
                for i, embedding in enumerate(embeddings):
                    if i in empty_indices:
                        result.append(np.zeros(self.embedding_dim))
                    else:
                        result.append(embedding.astype(np.float32))
                
                return result
            
        except Exception as e:
            logger.error(f"Error generating batch sentence transformer embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]
    
    def _get_transformers_embeddings_batch(self, formatted_texts: List[str], empty_indices: List[int]) -> List[np.ndarray]:
        """Generate embeddings using direct transformers batch approach."""
        # Tokenize batch
        inputs = self.tokenizer(
            formatted_texts, 
            max_length=self.max_length, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Use appropriate pooling method
            if self.pooling_method == "cls_token":
                embeddings = cls_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            else:  # last_token
                embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
        
        # Handle empty texts
        result = []
        for i, embedding in enumerate(embeddings_np):
            if i in empty_indices:
                result.append(np.zeros(self.embedding_dim))
            else:
                result.append(embedding.astype(np.float32))
        
        return result
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding using the configured method (Azure, local, or sentence transformers).
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector.
        """
        if self.embedding_type == "azure":
            return self.get_openai_embedding(text)
        elif self.embedding_type == "local":
            return self.get_local_embedding(text)
        elif self.embedding_type == "sentence_transformers":
            return self.get_sentence_transformer_embedding(text)
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")
    
    def generate_embeddings_for_concept_list(self, concept_list: List[str], 
                                             save_intermediate: bool = True,
                                             intermediate_file: str = "intermediate_radiology_embeddings.pickle") -> Dict[str, np.ndarray]:
        logger.info(f"Generating embeddings for {len(concept_list)} concepts using {self.embedding_type}")
        
        if not concept_list:
            logger.warning("No concepts provided for embedding generation.")
            return {}
        
        embeddings: Dict[str, np.ndarray] = {}
        failed_concepts_count = 0
        
        if self.embedding_type in ["local", "sentence_transformers"]:
            # Use efficient batching for local models and sentence transformers
            logger.info(f"Using batch processing with batch size: {self.batch_size}")
            
            # Process in batches
            for i in tqdm(range(0, len(concept_list), self.batch_size), desc="Processing batches"):
                batch = concept_list[i:i + self.batch_size]
                
                try:
                    if self.embedding_type == "local":
                        batch_embeddings = self.get_local_embeddings_batch(batch)
                    else:  # sentence_transformers
                        batch_embeddings = self.get_sentence_transformer_embeddings_batch(batch)
                    
                    # Assign embeddings to concepts
                    for concept, embedding in zip(batch, batch_embeddings):
                        if np.allclose(embedding, 0):
                            logger.warning(f"Failed to generate embedding for concept: \"{concept[:50]}...\" (received zero vector)")
                            failed_concepts_count += 1
                        embeddings[concept] = embedding
                    
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    # Assign zero vectors for the entire batch
                    for concept in batch:
                        embeddings[concept] = np.zeros(self.embedding_dim)
                        failed_concepts_count += 1
                
                # Fit PCA model if needed and we have enough samples
                if self.apply_pca and self.pca_model is None and len(embeddings) >= 100:
                    valid_embeddings = [emb for emb in embeddings.values() if not np.allclose(emb, 0)]
                    if len(valid_embeddings) >= 50:  # Need sufficient samples for PCA
                        self._fit_pca_model(valid_embeddings[:1000])  # Use up to 1000 samples for fitting
                
                # Save intermediate results periodically
                if save_intermediate and len(embeddings) % (self.batch_size * 10) == 0:
                    self._save_intermediate_results(embeddings, intermediate_file, is_partial_batch=True)
        
        else:  # Azure OpenAI - process individually
            with tqdm(total=len(concept_list), desc="Processing concepts") as pbar:
                for concept_text in concept_list:
                    try:
                        embedding = self.get_embedding(concept_text)
                        
                        # get_embedding returns a zero vector on failure, so embedding should not be None.
                        # We check if it's a zero vector to count failures.
                        if np.allclose(embedding, 0):
                            logger.warning(f"Failed to generate embedding for concept: \"{concept_text[:50]}...\" (received zero vector)")
                            failed_concepts_count += 1
                        
                        embeddings[concept_text] = embedding
                        
                    except Exception as e:
                        logger.error(f"Error processing concept \"{concept_text[:50]}...\": {e}")
                        embeddings[concept_text] = np.zeros(self.embedding_dim)
                        failed_concepts_count += 1
                    
                    # Fit PCA model if needed and we have enough samples
                    if self.apply_pca and self.pca_model is None and len(embeddings) >= 100:
                        valid_embeddings = [emb for emb in embeddings.values() if not np.allclose(emb, 0)]
                        if len(valid_embeddings) >= 50:  # Need sufficient samples for PCA
                            self._fit_pca_model(valid_embeddings[:1000])  # Use up to 1000 samples for fitting
                    
                    pbar.update(1)
                    
                    if save_intermediate and len(embeddings) % 100 == 0 and len(embeddings) > 0:
                        self._save_intermediate_results(embeddings, intermediate_file, is_partial_batch=True)
        
        # Apply PCA to all embeddings if configured
        if self.apply_pca and embeddings:
            # Fit PCA if not already fitted
            if self.pca_model is None:
                valid_embeddings = [emb for emb in embeddings.values() if not np.allclose(emb, 0)]
                if len(valid_embeddings) >= 10:  # Minimum samples for PCA
                    self._fit_pca_model(valid_embeddings)
            
            # Apply PCA reduction
            if self.pca_model is not None:
                embeddings = self._apply_pca_to_embeddings(embeddings)
        
        success_count = len(embeddings) - failed_concepts_count
        logger.info(f"Completed embedding generation for this batch. Success: {success_count}, Failed: {failed_concepts_count} (represented as zero vectors)")
        
        return embeddings
    
    def generate_embeddings_with_indices(self, csv_filepath: str, 
                                        save_intermediate: bool = True,
                                        intermediate_file: str = "intermediate_radiology_embeddings_indexed.pickle") -> Dict[int, np.ndarray]:
        """
        Generate embeddings for CSV concepts while preserving the original concept_idx mapping.
        
        Args:
            csv_filepath: Path to CSV file with 'concept' and 'concept_idx' columns.
            save_intermediate: Whether to save intermediate results.
            intermediate_file: Path to save intermediate results.
            
        Returns:
            Dictionary mapping concept_idx (int) to embedding vectors.
        """
        logger.info(f"Starting indexed embedding generation from {csv_filepath} using {self.embedding_type} method")
        
        # Extract concepts with their indices
        concept_mapping = self._extract_concepts_with_indices_from_csv(csv_filepath)
        
        if not concept_mapping:
            logger.warning("No concepts with indices found.")
            return {}
        
        # Get list of concept texts for batch processing
        concept_texts = list(concept_mapping.values())
        concept_indices = list(concept_mapping.keys())
        
        logger.info(f"Processing {len(concept_texts)} concepts with preserved indices")
        
        # Generate embeddings
        if self.embedding_type in ["local", "sentence_transformers"]:
            # Use efficient batching for local models and sentence transformers
            logger.info(f"Using batch processing with batch size: {self.batch_size}")
            
            indexed_embeddings = {}
            
            # Process in batches
            for i in tqdm(range(0, len(concept_texts), self.batch_size), desc="Processing batches"):
                batch_texts = concept_texts[i:i + self.batch_size]
                batch_indices = concept_indices[i:i + self.batch_size]
                
                try:
                    if self.embedding_type == "local":
                        batch_embeddings = self.get_local_embeddings_batch(batch_texts)
                    else:  # sentence_transformers
                        batch_embeddings = self.get_sentence_transformer_embeddings_batch(batch_texts)
                    
                    # Map embeddings back to their indices
                    for idx, embedding in zip(batch_indices, batch_embeddings):
                        indexed_embeddings[idx] = embedding
                    
                except Exception as e:
                    logger.error(f"Error processing batch starting at index {i}: {e}")
                    # Assign zero vectors for the entire batch
                    for idx in batch_indices:
                        indexed_embeddings[idx] = np.zeros(self.embedding_dim)
                
                # Fit PCA model if needed and we have enough samples
                if self.apply_pca and self.pca_model is None and len(indexed_embeddings) >= 100:
                    valid_embeddings = [emb for emb in indexed_embeddings.values() if not np.allclose(emb, 0)]
                    if len(valid_embeddings) >= 50:  # Need sufficient samples for PCA
                        self._fit_pca_model(valid_embeddings[:1000])  # Use up to 1000 samples for fitting
                
                # Save intermediate results periodically
                if save_intermediate and len(indexed_embeddings) % (self.batch_size * 10) == 0:
                    self._save_intermediate_indexed_results(indexed_embeddings, intermediate_file)
        
        else:  # Azure OpenAI - process individually
            indexed_embeddings = {}
            
            with tqdm(total=len(concept_texts), desc="Processing concepts") as pbar:
                for idx, concept_text in zip(concept_indices, concept_texts):
                    try:
                        embedding = self.get_embedding(concept_text)
                        indexed_embeddings[idx] = embedding
                        
                    except Exception as e:
                        logger.error(f"Error processing concept idx {idx}: {e}")
                        indexed_embeddings[idx] = np.zeros(self.embedding_dim)
                    
                    # Fit PCA model if needed and we have enough samples
                    if self.apply_pca and self.pca_model is None and len(indexed_embeddings) >= 100:
                        valid_embeddings = [emb for emb in indexed_embeddings.values() if not np.allclose(emb, 0)]
                        if len(valid_embeddings) >= 50:  # Need sufficient samples for PCA
                            self._fit_pca_model(valid_embeddings[:1000])  # Use up to 1000 samples for fitting
                    
                    pbar.update(1)
                    
                    if save_intermediate and len(indexed_embeddings) % 100 == 0:
                        self._save_intermediate_indexed_results(indexed_embeddings, intermediate_file)
        
        # Apply PCA to all embeddings if configured
        if self.apply_pca and indexed_embeddings:
            # Fit PCA if not already fitted
            if self.pca_model is None:
                valid_embeddings = [emb for emb in indexed_embeddings.values() if not np.allclose(emb, 0)]
                if len(valid_embeddings) >= 10:  # Minimum samples for PCA
                    self._fit_pca_model(valid_embeddings)
            
            # Apply PCA reduction
            if self.pca_model is not None:
                indexed_embeddings = self._apply_pca_to_embeddings(indexed_embeddings)
        
        logger.info(f"Completed indexed embedding generation. Total: {len(indexed_embeddings)} concepts")
        return indexed_embeddings
    
    def _fit_pca_model(self, embeddings_sample: List[np.ndarray]) -> None:
        if not self.apply_pca or self.pca_model is not None:
            return
        
        try:
            embedding_matrix = np.vstack(embeddings_sample)
            self.pca_model = PCA(n_components=self.pca_dimensions)
            self.pca_model.fit(embedding_matrix)
            
            explained_variance = sum(self.pca_model.explained_variance_ratio_)
            logger.info(f"PCA fitted on {len(embeddings_sample)} samples. Variance: {explained_variance:.4f}")
            
        except Exception as e:
            logger.error(f"PCA fitting error: {e}")
            self.apply_pca = False
    
    def _apply_pca_to_embeddings(self, embeddings: Dict) -> Dict:
        if not self.apply_pca or self.pca_model is None:
            return embeddings
        
        try:
            logger.info(f"Applying PCA to {len(embeddings)} embeddings...")
            
            keys = list(embeddings.keys())
            embedding_list = [embeddings[key] for key in keys]
            embedding_matrix = np.vstack(embedding_list)
            reduced_embeddings = self.pca_model.transform(embedding_matrix)
            
            reduced_dict = {}
            for i, key in enumerate(keys):
                reduced_dict[key] = reduced_embeddings[i].astype(np.float32)
            
            logger.info(f"PCA completed: {self.original_embedding_dim} -> {self.pca_dimensions}")
            return reduced_dict
            
        except Exception as e:
            logger.error(f"PCA error: {e}")
            return embeddings
    
    def _save_intermediate_indexed_results(self, indexed_embeddings: Dict[int, np.ndarray], filepath: str):
        """Save intermediate indexed results."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(indexed_embeddings, f)
            logger.debug(f"Saved {len(indexed_embeddings)} indexed embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save intermediate indexed results to {filepath}: {e}")
    
    def _save_intermediate_results(self, embeddings_to_save: Dict[str, np.ndarray], filepath: str, is_partial_batch: bool = False):
        """Save intermediate results. If it's a partial batch, it saves only that batch.
           If not partial (called from main after merging), it saves the complete set."""
        # The parameter is_partial_batch is not used in the current simplified implementation,
        # but kept for potential future logic.
        try:
            data_to_save = embeddings_to_save
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logger.debug(f"Saved {len(data_to_save)} embeddings to intermediate file {filepath}")
        except Exception as e:
            logger.error(f"Failed to save intermediate results to {filepath}: {e}")
    
    def save_embeddings(self, embeddings, filepath: str):
        """
        Save final embeddings to file. Handles both Dict[str, np.ndarray] and Dict[int, np.ndarray].
        
        Args:
            embeddings: Dictionary of concept embeddings (str->array or int->array).
            filepath: Output file path.
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Determine the type for logging
            if embeddings:
                key_type = type(list(embeddings.keys())[0])
                if key_type == int:
                    logger.info(f"Saved {len(embeddings)} indexed embeddings (concept_idx->embedding) to {filepath}")
                else:
                    logger.info(f"Saved {len(embeddings)} text-based embeddings (concept->embedding) to {filepath}")
            else:
                logger.info(f"Saved 0 embeddings to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save final embeddings to {filepath}: {e}")
    
    def load_embeddings(self, filepath: str):
        """
        Load embeddings from file. Returns Dict[str, np.ndarray] or Dict[int, np.ndarray].
        
        Args:
            filepath: Input file path.
            
        Returns:
            Dictionary of concept embeddings (str->array or int->array).
        """
        if not Path(filepath).exists():
            logger.warning(f"Embeddings file {filepath} not found. Returning empty dictionary.")
            return {}
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Determine the type for logging
            if embeddings:
                key_type = type(list(embeddings.keys())[0])
                if key_type == int:
                    logger.info(f"Loaded {len(embeddings)} indexed embeddings (concept_idx->embedding) from {filepath}")
                else:
                    logger.info(f"Loaded {len(embeddings)} text-based embeddings (concept->embedding) from {filepath}")
            else:
                logger.info(f"Loaded 0 embeddings from {filepath}")
            
            # Basic validation of loaded data
            if not isinstance(embeddings, dict):
                logger.error(f"Loaded data from {filepath} is not a dictionary. Returning empty.")
                return {}
            # Further checks can be added here if needed (e.g., value types)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to load embeddings from {filepath}: {e}")
            return {}
    
    def validate_embeddings(self, embeddings) -> Dict[str, any]:
        """
        Validate the generated embeddings. Handles both Dict[str, np.ndarray] and Dict[int, np.ndarray].
        
        Args:
            embeddings: Dictionary of concept embeddings (str->array or int->array).
            
        Returns:
            Validation statistics.
        """
        stats = {
            'total_concepts': 0,
            'zero_vectors': 0,
            'valid_vectors': 0,
            'dimension_consistency': True,
            'expected_dimension': self.embedding_dim,
            'original_dimension': self.original_embedding_dim if self.apply_pca else self.embedding_dim,
            'pca_applied': self.apply_pca,
            'success_rate': 0.0,
            'embedding_type': 'indexed' if embeddings and isinstance(list(embeddings.keys())[0], int) else 'text-based'
        }
        
        if not embeddings:
            logger.warning("No embeddings provided for validation.")
            return stats

        stats['total_concepts'] = len(embeddings)
        
        for concept_key, embedding in embeddings.items():
            if not isinstance(embedding, np.ndarray):
                # Format key display based on type
                key_display = f"concept_idx {concept_key}" if isinstance(concept_key, int) else f"concept \"{str(concept_key)[:50]}...\""
                logger.warning(f"Embedding for {key_display} is not a numpy array. Skipping validation for this item.")
                stats['dimension_consistency'] = False # Mark as inconsistent if any are bad
                continue

            if embedding.shape[0] != self.embedding_dim:
                stats['dimension_consistency'] = False
                key_display = f"concept_idx {concept_key}" if isinstance(concept_key, int) else f"concept \"{str(concept_key)[:50]}...\""
                logger.warning(f"Dimension mismatch for {key_display}: {embedding.shape[0]} vs {self.embedding_dim}")
            
            if np.allclose(embedding, 0):
                stats['zero_vectors'] += 1
            else:
                stats['valid_vectors'] += 1
        
        if stats['total_concepts'] > 0:
            stats['success_rate'] = stats['valid_vectors'] / stats['total_concepts']
        
        pca_info = ""
        if stats['pca_applied']:
            pca_info = f" (PCA reduced from {stats['original_dimension']} to {stats['expected_dimension']})"
        
        logger.info(f"Validation results ({stats['embedding_type']}): {stats['valid_vectors']}/{stats['total_concepts']} valid embeddings "
                   f"({stats['success_rate']:.2%} success rate). Zero vectors: {stats['zero_vectors']}.{pca_info}")
        if not stats['dimension_consistency']:
            logger.warning("Dimension inconsistency found in one or more embeddings.")
            
        return stats


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Generate radiological concept embeddings using Azure OpenAI API, local transformers, or sentence transformers. Supports both CSV and JSON input formats.")
    
    # Common arguments
    parser.add_argument('--concepts_file', type=str, required=True,
                       help='Path to CSV or JSON file containing radiological concepts.')
    parser.add_argument('--embedding_type', type=str, default='local', choices=['azure', 'local', 'sentence_transformers'],
                       help='Type of embedding method to use (default: local).')
    parser.add_argument('--output', type=str, default='radiology_embeddings.pickle',
                       help='Output file path for embeddings (default: radiology_embeddings.pickle).')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to intermediate embeddings file to resume from.')
    parser.add_argument('--validate', action='store_true',
                       help='Validate embeddings after generation/loading.')
    parser.add_argument('--intermediate_file_path', type=str, default='intermediate_radiology_embeddings.pickle',
                        help='Path for saving intermediate results (default: intermediate_radiology_embeddings.pickle)')
    parser.add_argument('--preserve_indices', action='store_true',
                       help='For CSV files: preserve concept_idx mapping in output (maps concept_idx to embeddings).')
    
    # Azure OpenAI arguments
    azure_group = parser.add_argument_group('Azure OpenAI options')
    azure_group.add_argument('--azure_endpoint', type=str,
                       help='Azure OpenAI endpoint URL (required for Azure type).')
    azure_group.add_argument('--api_key', type=str,
                       help='Azure OpenAI API key (required for Azure type).')
    azure_group.add_argument('--api_version', type=str, default='2024-02-01',
                       help='Azure OpenAI API version (default: 2024-02-01).')
    azure_group.add_argument('--model', type=str, default='text-embedding-3-small',
                       help='Azure OpenAI embedding model deployment name (default: text-embedding-3-small).')
    
    # Local model and sentence transformer arguments
    local_group = parser.add_argument_group('Local model and Sentence Transformer options')
    local_group.add_argument('--local_model', type=str, default='Qwen/Qwen3-Embedding-8B',
                       help='Model name: Qwen/Qwen3-Embedding-8B, Salesforce/SFR-Embedding-Mistral, '
                            'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (default: Qwen/Qwen3-Embedding-8B)')
    local_group.add_argument('--device', type=str, default=None, choices=['cuda', 'cpu', 'auto'],
                       help='Device to use for models (default: auto - uses CUDA if available).')
    local_group.add_argument('--max_length', type=int, default=8192,
                       help='Max sequence length (default: 8192)')
    local_group.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    local_group.add_argument('--trust_remote_code', action='store_true', default=True,
                       help='Trust remote code (default: True)')
    local_group.add_argument('--use_fp16', action='store_true', default=True,
                       help='Use FP16 for GPU acceleration (default: True)')
    
    # PCA options
    pca_group = parser.add_argument_group('PCA options')
    pca_group.add_argument('--apply_pca', action='store_true',
                       help='Apply PCA dimension reduction')
    pca_group.add_argument('--pca_dimensions', type=int, choices=[768, 1536],
                       help='Target dimension: 768 or 1536')
    
    args = parser.parse_args()
    
    all_embeddings = {}  # Can be Dict[str, np.ndarray] or Dict[int, np.ndarray]
    
    try:
        generator = RadiologyEmbeddingGenerator(
            embedding_type=args.embedding_type,
            azure_endpoint=args.azure_endpoint,
            api_key=args.api_key,
            api_version=args.api_version,
            embedding_model=args.model,
            local_model_name=args.local_model,
            device=args.device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            trust_remote_code=args.trust_remote_code,
            use_fp16=args.use_fp16,
            apply_pca=args.apply_pca,
            pca_dimensions=args.pca_dimensions
        )
        
        # Check if indexed processing is requested for CSV files
        if args.preserve_indices and args.concepts_file.endswith('.csv'):
            logger.info("Using indexed processing to preserve concept_idx mapping")
            
            # For indexed processing, we work directly with the CSV
            if args.resume and Path(args.resume).exists():
                logger.info(f"Resuming indexed processing from: {args.resume}")
                all_embeddings = generator.load_embeddings(args.resume)
                
                # Check if we need to process more concepts
                concept_mapping = generator._extract_concepts_with_indices_from_csv(args.concepts_file)
                
                # Check if the loaded embeddings are indexed (int keys) - safer check
                if all_embeddings:
                    first_key = list(all_embeddings.keys())[0]
                    if isinstance(first_key, int):
                        processed_indices = set(all_embeddings.keys())
                        remaining_indices = set(concept_mapping.keys()) - processed_indices
                        
                        if remaining_indices:
                            logger.info(f"Found {len(all_embeddings)} existing indexed embeddings. Processing {len(remaining_indices)} remaining concepts.")
                            # Note: For simplicity, we'll regenerate all for now. Could be optimized.
                            indexed_embeddings = generator.generate_embeddings_with_indices(
                                args.concepts_file,
                                save_intermediate=True,
                                intermediate_file=args.intermediate_file_path.replace('.pickle', '_indexed.pickle')
                            )
                            all_embeddings.update(indexed_embeddings)
                        else:
                            logger.info("All concepts from the CSV file are already processed with indices.")
                    else:
                        logger.warning("Resume file contains text-based embeddings but --preserve_indices was requested. Starting fresh.")
                        all_embeddings = generator.generate_embeddings_with_indices(
                            args.concepts_file,
                            save_intermediate=True,
                            intermediate_file=args.intermediate_file_path.replace('.pickle', '_indexed.pickle')
                        )
                else:
                    logger.info("Resume file is empty. Starting indexed processing from scratch.")
                    all_embeddings = generator.generate_embeddings_with_indices(
                        args.concepts_file,
                        save_intermediate=True,
                        intermediate_file=args.intermediate_file_path.replace('.pickle', '_indexed.pickle')
                    )
            else:
                logger.info("Starting indexed processing from scratch.")
                all_embeddings = generator.generate_embeddings_with_indices(
                    args.concepts_file,
                    save_intermediate=True,
                    intermediate_file=args.intermediate_file_path.replace('.pickle', '_indexed.pickle')
                )
        else:
            # Original processing method
            # Load all unique concepts that need to be processed
            all_target_concepts = generator._extract_unique_concepts_from_file(args.concepts_file)
            if not all_target_concepts:
                logger.info("No concepts to process from the input file.")
                # If validate is true and resume is specified, we might just validate the resumed file.
                if args.resume and args.validate:
                    logger.info(f"Attempting to load and validate from resume file: {args.resume}")
                    resumed_embeddings = generator.load_embeddings(args.resume)
                    if resumed_embeddings:
                        stats = generator.validate_embeddings(resumed_embeddings)
                        print_validation_stats(stats)
                    else:
                        logger.info(f"No embeddings found in resume file {args.resume} to validate.")
                return

            concepts_to_process = all_target_concepts
            
            if args.resume and Path(args.resume).exists():
                logger.info(f"Resuming from intermediate results: {args.resume}")
                existing_embeddings = generator.load_embeddings(args.resume)
                all_embeddings.update(existing_embeddings) # Start with existing embeddings
                
                # Determine remaining concepts
                processed_concepts = set(existing_embeddings.keys())
                concepts_to_process = [c for c in all_target_concepts if c not in processed_concepts]
                
                if not concepts_to_process:
                    logger.info("All concepts from the input file are already present in the resume file.")
                else:
                    logger.info(f"Found {len(existing_embeddings)} existing embeddings. Processing {len(concepts_to_process)} remaining concepts.")
            else:
                logger.info(f"Processing {len(concepts_to_process)} unique concepts from scratch.")

            if concepts_to_process:
                logger.info(f"Generating embeddings for {len(concepts_to_process)} concepts.")
                
                new_embeddings = generator.generate_embeddings_for_concept_list(
                    concepts_to_process,
                    save_intermediate=True,
                    intermediate_file=args.intermediate_file_path
                )
                all_embeddings.update(new_embeddings)
        
        # Save final combined results
        if all_embeddings: # Ensure there's something to save
            generator.save_embeddings(all_embeddings, args.output)
        else:
            logger.info("No embeddings were generated or loaded. Output file will not be created.")

        if args.validate:
            if all_embeddings:
                stats = generator.validate_embeddings(all_embeddings)
                print_validation_stats(stats)
            elif args.resume: # If no embeddings processed but resume was specified, validate that file
                 logger.info(f"No new embeddings processed. Validating resume file: {args.resume}")
                 resumed_embeddings = generator.load_embeddings(args.resume)
                 if resumed_embeddings:
                     stats = generator.validate_embeddings(resumed_embeddings)
                     print_validation_stats(stats)
                 else:
                     logger.info(f"Resume file {args.resume} empty or not found for validation.")
            else:
                logger.info("No embeddings to validate.")
        
        logger.info("Radiology concept embedding generation process completed!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}", exc_info=True)
        # raise # Optionally re-raise for debugging or allow graceful exit

def print_validation_stats(stats: Dict[str, any]):
    print(f"\nValidation Results:")
    print(f"Total concepts processed/loaded: {stats['total_concepts']}")
    print(f"Valid embeddings (non-zero): {stats['valid_vectors']}")
    print(f"Zero vectors (failures/empty): {stats['zero_vectors']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Dimension consistency: {'Consistent' if stats['dimension_consistency'] else 'INCONSISTENT (check logs)'}")
    
    if stats.get('pca_applied', False):
        print(f"Original dimension: {stats.get('original_dimension', 'Unknown')}")
        print(f"PCA reduced dimension: {stats['expected_dimension']}")
        print(f"PCA reduction applied: Yes")
    else:
        print(f"Embedding dimension: {stats['expected_dimension']}")
        print(f"PCA reduction applied: No")

if __name__ == "__main__":
    main() 