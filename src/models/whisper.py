import torch
import whisper
from torch import nn
import numpy as np
from tqdm import tqdm  # For WeakLearners fitting progress
from typing import Optional, Dict  # Added Dict for loading state
import os  # Added for path checking
import joblib  # Added for loading weak learners

from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.models.base_model import BaseModel, BaseMultimodalModel

from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel


# --- Weak Learners ---
class WeakLearners(nn.Module):
    """
    Wrapper for scikit-learn regression models to act as weak learners.
    Requires loading fitted models from a file before use in inference.
    Note: Sklearn models run on CPU. Data is moved accordingly during forward.
    """

    # Added default dimensions matching common Whisper base/BERT base
    def __init__(self, audio_dim: int = 512, text_dim: int = 768, device: str = "cpu"):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        # Device primarily for moving predictions back, sklearn runs on CPU
        self.device = torch.device(device)

        # Initialize sklearn model structures (placeholders until loaded)
        self.ridge_regressor = Ridge(alpha=1.0)
        self.svr = SVR()
        self.dtr = DecisionTreeRegressor()

        self.models = [self.ridge_regressor, self.svr, self.dtr]
        self.model_names = ["Ridge", "SVR", "DTR"]
        self.fitted = False  # Will be set to True after loading

    # Keep fit method for potential separate training script, but not used by WhisperModel/MultiModalWhisper directly
    def fit(self, train_loader: torch.utils.data.DataLoader):
        """Train weak learners using embeddings from the train_loader."""
        # ... (previous fit implementation remains unchanged) ...
        print("Fitting weak learners...")
        # ... (rest of fit implementation) ...

    def load_fitted(self, filepath: str) -> bool:
        """Loads fitted sklearn models from a joblib file."""
        if not os.path.exists(filepath):
            print(f"Error: Weak learners file not found at {filepath}")
            return False
        try:
            loaded_sklearn_models = joblib.load(filepath)
            if isinstance(loaded_sklearn_models, list) and len(
                loaded_sklearn_models
            ) == len(self.models):
                self.models = loaded_sklearn_models
                self.fitted = True
                print(f"  Weak learners loaded successfully from {filepath}.")
                return True
            else:
                print(
                    f"Error: Loaded file '{filepath}' does not contain the expected list of {len(self.models)} sklearn models."
                )
                self.fitted = False
                return False
        except ImportError:
            print(
                "Error: joblib is required to load weak learners but not installed. Run: pip install joblib"
            )
            self.fitted = False
            return False
        except Exception as e:
            print(f"Error loading weak learners from {filepath}: {e}")
            self.fitted = False
            return False

    def forward(
        self, audio_emb: torch.Tensor, text_emb: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate predictions from loaded weak learners.
        Input tensors (on any device) will be moved to CPU for sklearn.
        Output tensor (predictions) will be on self.device.
        """
        if not self.fitted:
            # Changed error message slightly
            raise RuntimeError(
                "Weak learners must be loaded using load_fitted() before calling forward."
            )

        # Prepare input for sklearn: move to CPU, detach, convert to numpy
        audio_np = audio_emb.detach().cpu().numpy()

        # Handle text embedding based on text_dim
        if self.text_dim > 0:
            if text_emb is None:
                # If text is expected (text_dim > 0) but not provided, pad with zeros
                print(
                    "Warning: Text embedding is None in WeakLearners forward pass, but text_dim > 0. Padding with zeros."
                )
                zeros_text = np.zeros((audio_np.shape[0], self.text_dim))
                combined_embeddings = np.hstack((audio_np, zeros_text))
            else:
                # Text is expected and provided
                text_np = text_emb.detach().cpu().numpy()
                if audio_np.shape[0] != text_np.shape[0]:
                    raise ValueError(
                        "Batch size mismatch between audio and text embeddings in WeakLearners forward."
                    )
                combined_embeddings = np.hstack((audio_np, text_np))
        else:
            # No text is expected (text_dim == 0)
            if text_emb is not None:
                print(
                    "Warning: Text embedding provided to WeakLearners forward pass, but text_dim is 0. Ignoring text."
                )
            combined_embeddings = audio_np

        # Get predictions from each loaded sklearn model
        all_preds = []
        # Use self.models which now contains the loaded models
        for model in self.models:
            preds = model.predict(combined_embeddings)
            # Convert numpy array predictions to float tensor
            all_preds.append(torch.from_numpy(preds).float())

        # Stack predictions along a new dimension (dim=1) -> (batch_size, num_weak_learners)
        # Move the final stacked tensor to the designated device (e.g., GPU if specified)
        stacked_preds = torch.stack(all_preds, dim=1).to(self.device)
        return stacked_preds


# --- Stacking Model (Meta-Learner) ---
class StackingMetaLearner(nn.Module):
    """
    A simple feed-forward network that learns to combine predictions
    from weak learners. Structure needs to match the saved weights.
    """

    def __init__(self, weak_output_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        # Check for valid dimensions
        if weak_output_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                "weak_output_dim and hidden_dim must be positive integers."
            )
        self.fc1 = nn.Linear(weak_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Predict a single score

    def load_state_dict_from_file(self, filepath: str, device: torch.device):
        """Loads the state dictionary from a .pth file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Meta-learner state file not found at {filepath}")
        try:
            state_dict = torch.load(filepath, map_location=device)
            self.load_state_dict(state_dict)
            print(f"  StackingMetaLearner state loaded successfully from {filepath}.")
        except Exception as e:
            print(f"Error loading StackingMetaLearner state_dict from {filepath}: {e}")
            # Consider adding more debug info here if needed
            # print("Expected keys:", self.state_dict().keys())
            # if 'state_dict' in locals(): print("Loaded keys:", state_dict.keys())
            raise RuntimeError(f"Failed to load StackingMetaLearner state: {e}") from e

    def forward(self, weak_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weak_outputs (torch.Tensor): Tensor of shape (batch_size, weak_output_dim)
                                         containing predictions from weak learners.
        Returns:
            torch.Tensor: Final prediction of shape (batch_size, 1).
        """
        x = self.relu(self.fc1(weak_outputs))
        x = self.fc2(x)
        return x


# --- Main Ensemble Model (Wrapper used internally now) ---
class SSLEnsembleModel(nn.Module):
    """
    Combines WeakLearners and a StackingMetaLearner.
    Assumes WeakLearners are loaded externally and MetaLearner state_dict is loaded externally.
    This class mainly defines the forward pass logic using the components.
    """

    def __init__(
        self, weak_learners: WeakLearners, stacking_meta_learner: StackingMetaLearner
    ):
        super().__init__()
        if not isinstance(weak_learners, WeakLearners) or not weak_learners.fitted:
            raise ValueError(
                "A pre-loaded WeakLearners instance must be provided to SSLEnsembleModel."
            )
        if not isinstance(stacking_meta_learner, StackingMetaLearner):
            raise ValueError("A StackingMetaLearner instance must be provided.")

        self.weak_learners = weak_learners
        self.stacking_meta_learner = stacking_meta_learner

    def forward(
        self, audio_emb: torch.Tensor, text_emb: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        1. Get predictions from weak learners.
        2. Pass weak learner predictions to the stacking meta-learner.
        """
        # Get predictions from weak learners (shape: batch_size, num_weak_outputs)
        # WeakLearners forward handles device movement internally
        weak_outputs = self.weak_learners(
            audio_emb, text_emb
        )  # Output is on weak_learners.device

        # Ensure weak_outputs are on the same device as the meta-learner before passing
        weak_outputs = weak_outputs.to(
            next(self.stacking_meta_learner.parameters()).device
        )

        # Get final prediction from the meta-learner
        final_output = self.stacking_meta_learner(
            weak_outputs
        )  # Output shape (batch_size, 1)
        return final_output


# --- Updated WhisperModel ---
class WhisperModel(BaseModel):
    """
    Wrapper for the Whisper model for audio processing.
    Can optionally load and use an SSLEnsembleModel (Weak + Meta learners)
    to predict a final score instead of just returning embeddings.

    Args:
        whisper_variant (str): Whisper model variant (e.g., "base.en").
        weights (str, optional): Path to pre-trained Whisper weights (if different from standard). Defaults to None.
        device (str, optional): Device ("cuda", "cpu"). Auto-detected if None.
        # --- New Args for Ensemble Prediction ---
        ssl_ensemble_config (Dict, optional): If provided, enables prediction mode. Must contain:
            'weak_learners_path' (str): Path to the fitted weak learners (.joblib file).
            'meta_learner_path' (str): Path to the trained meta learner (.pth file).
            'audio_dim' (int): Expected audio embedding dimension for weak learners.
            'text_dim' (int): Expected text embedding dimension (set to 0 if no text used).
            'hidden_dim' (int): Hidden dimension of the trained meta learner.
            Defaults to None (outputs embeddings only).
    """

    def __init__(
        self,
        whisper_variant: str = "base.en",
        weights: str = None,
        device: str = None,
        ssl_ensemble_config: Optional[Dict] = None,  # <-- New Arg
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.predict_mode = False  # Default to embedding mode
        self.ssl_ensemble_model = None

        # --- Load Whisper Backbone ---
        if weights is None:
            print(f"Loading Whisper model variant: {whisper_variant}")
            model = whisper.load_model(whisper_variant, device=self.device)
        else:
            print(f"Loading Whisper model from specified name/path: {weights}")
            model = whisper.load_model(weights, device=self.device)
        self.model = model  # The Whisper model itself
        self.model.eval()

        # --- Load SSL Ensemble if config provided ---
        if ssl_ensemble_config is not None:
            print("SSL Ensemble config provided. Initializing for prediction mode.")
            self.predict_mode = True
            required_keys = [
                "weak_learners_path",
                "meta_learner_path",
                "audio_dim",
                "text_dim",
                "hidden_dim",
            ]
            if not all(key in ssl_ensemble_config for key in required_keys):
                raise ValueError(
                    f"ssl_ensemble_config missing required keys: {required_keys}"
                )

            # 1. Initialize and load WeakLearners
            weak_learners = WeakLearners(
                audio_dim=ssl_ensemble_config["audio_dim"],
                text_dim=ssl_ensemble_config["text_dim"],
                device=self.device,  # Output of weak learners will be on this device
            )
            if not weak_learners.load_fitted(ssl_ensemble_config["weak_learners_path"]):
                raise RuntimeError("Failed to load weak learners for WhisperModel.")

            # 2. Initialize and load StackingMetaLearner
            meta_learner = StackingMetaLearner(
                weak_output_dim=len(weak_learners.models),  # Should be 3 typically
                hidden_dim=ssl_ensemble_config["hidden_dim"],
            ).to(
                self.device
            )  # Meta learner runs on the main device
            meta_learner.load_state_dict_from_file(
                ssl_ensemble_config["meta_learner_path"], device=self.device
            )
            meta_learner.eval()  # Set meta learner to eval mode

            # 3. Create the SSLEnsembleModel wrapper
            self.ssl_ensemble_model = SSLEnsembleModel(
                weak_learners=weak_learners, stacking_meta_learner=meta_learner
            )
            print("SSLEnsembleModel loaded successfully within WhisperModel.")
        else:
            print(
                "No SSL Ensemble config provided. WhisperModel will output embeddings."
            )

    def preprocess_audios(self, audios: list) -> torch.Tensor:
        """
        Accepts a list of audio waveforms (numpy.ndarray).
        Pads/trims each audio, then computes log-Mel spectrograms.
        Returns a batch of tensors ready for the encoder.
        """
        processed = []
        for audio in audios:
            # Ensure audio is numpy array (Whisper functions expect this)
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_proc = whisper.pad_or_trim(audio)
            # log_mel_spectrogram expects numpy, returns tensor on specified device
            mel = whisper.log_mel_spectrogram(audio_proc, device=self.device)
            processed.append(mel)
        if not processed:
            return torch.empty(
                (0, 80, 3000), device=self.device
            )  # Return empty tensor if input list is empty
        batch = torch.stack(processed)
        return batch

    def forward(self, audios: list) -> torch.Tensor:
        """
        Performs a forward pass: preprocesses audio, passes spectrograms
        through the Whisper encoder, and returns mean-pooled embeddings.
        """
        mels = self.preprocess_audios(audios)
        if mels.numel() == 0:  # Handle empty batch
            # Assuming base model dim is 512. Adjust if using different variant.
            embed_dim = self.model.encoder.ln_post.normalized_shape[0]
            return torch.empty((0, embed_dim), device=self.device)

        with torch.no_grad():
            # Encoder output shape: (batch_size, sequence_length, embed_dim)
            embeddings = self.model.encoder(mels)
            # Mean pool across the time dimension (dim=1)
            pooled_embeddings = embeddings.mean(dim=1)
        return pooled_embeddings

        # Decide output based on mode
        if self.predict_mode:
            if self.ssl_ensemble_model is None:
                # This shouldn't happen if init logic is correct, but defensive check
                raise RuntimeError(
                    "Predict mode is True but SSLEnsembleModel is not loaded."
                )
            # Ensure embeddings are on the device expected by the ensemble/weak learners
            pooled_embeddings = pooled_embeddings.to(self.device)
            # Pass embeddings to the ensemble model.
            # Since this is WhisperModel (audio only), pass None for text_emb.
            # The WeakLearners' forward method handles None based on its text_dim.
            with torch.no_grad():
                final_prediction = self.ssl_ensemble_model(
                    pooled_embeddings, text_emb=None
                )
            return final_prediction  # Shape (batch_size, 1)
        else:
            # Return embeddings, ensuring they are on the main device specified for this class
            return pooled_embeddings.to(self.device)  # Shape (batch_size, embed_dim)


class MultiModalWhisper(BaseMultimodalModel):
    """
    Multimodal model using Whisper for audio and a specified transformer for text.
    Can optionally load and use an SSLEnsembleModel (Weak + Meta learners)
    to predict a final score instead of just returning embeddings.

    Args:
        whisper_variant (str): Whisper model variant ("base.en", "base", etc.).
        text_model_type (str): Text model identifier (e.g., "bert-base-uncased", "none").
        weights (str, optional): Path or name for pre-trained Whisper weights. Defaults to None.
        device (str, optional): Device ("cuda", "cpu"). Auto-detected if None.
        # --- New Args for Ensemble Prediction ---
        ssl_ensemble_config (Dict, optional): If provided, enables prediction mode. Must contain:
            'weak_learners_path' (str): Path to the fitted weak learners (.joblib file).
            'meta_learner_path' (str): Path to the trained meta learner (.pth file).
            'audio_dim' (int): Expected audio embedding dimension for weak learners.
            'text_dim' (int): Expected text embedding dimension (must match text model if used, else 0).
            'hidden_dim' (int): Hidden dimension of the trained meta learner.
            Defaults to None (outputs embeddings only).
    """

    def __init__(
        self,
        whisper_variant: str = "base.en",
        text_model_type: str = "bert-base-uncased",
        weights: str = None,
        device: str = None,
        ssl_ensemble_config: Optional[Dict] = None,  # <-- New Arg
    ) -> None:
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.predict_mode = False  # Default to embedding mode
        self.ssl_ensemble_model = None
        self.use_text = text_model_type.lower() != "none"  # Flag based on config

        # --- Whisper Model ---
        if weights is None:
            print(
                f"Initializing MultiModalWhisper: Loading Whisper variant '{whisper_variant}'"
            )
            wm = whisper.load_model(whisper_variant, device=self.device)
        else:
            print(f"Initializing MultiModalWhisper: Loading Whisper from '{weights}'")
            wm = whisper.load_model(weights, device=self.device)
        self.whisper_model = wm
        self.whisper_model.eval()

        # --- Text Model ---
        print(f"Initializing MultiModalWhisper: Loading Text model '{text_model_type}'")
        if text_model_type == "bert-base-uncased":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_model = BertModel.from_pretrained("bert-base-uncased").to(
                self.device
            )
        elif text_model_type == "DeepPavlov/rubert-base-cased":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "DeepPavlov/rubert-base-cased"
            )
            self.text_model = AutoModel.from_pretrained(
                "DeepPavlov/rubert-base-cased"
            ).to(self.device)
        elif text_model_type.lower() == "none":
            print("Text model set to 'none'. Text processing will be skipped.")
            self.tokenizer = None
            self.text_model = None
        else:
            # Attempt to load using Auto classes as a fallback
            try:
                print(
                    f"Warning: Unsupported text_model_type '{text_model_type}'. Attempting Auto loading..."
                )
                self.tokenizer = AutoTokenizer.from_pretrained(text_model_type)
                self.text_model = AutoModel.from_pretrained(text_model_type).to(
                    self.device
                )
                print(f"Successfully loaded '{text_model_type}' using Auto classes.")
            except Exception as e:
                print(
                    f"  Warning: Failed to load text model '{text_model_type}'. Error: {e}"
                )
                print("           Text features will be disabled.")
                self.use_text = False  # Disable text if loading fails
        # else:
        #     print("  Text model set to 'none'. Text processing will be skipped.")

        # --- Load SSL Ensemble if config provided ---
        if ssl_ensemble_config is not None:
            print("SSL Ensemble config provided. Initializing for prediction mode.")
            self.predict_mode = True
            required_keys = [
                "weak_learners_path",
                "meta_learner_path",
                "audio_dim",
                "text_dim",
                "hidden_dim",
            ]
            if not all(key in ssl_ensemble_config for key in required_keys):
                raise ValueError(
                    f"ssl_ensemble_config missing required keys: {required_keys}"
                )

            # Validate text_dim consistency
            expected_text_dim = ssl_ensemble_config["text_dim"]
            if self.use_text and expected_text_dim == 0:
                print(
                    "Warning: Ensemble config expects text_dim=0, but a text model was loaded. Text embeddings will likely be ignored by the ensemble."
                )
            if not self.use_text and expected_text_dim > 0:
                print(
                    "Warning: Ensemble config expects text_dim > 0, but no text model is loaded/used. Weak learners will likely pad with zeros."
                )
            # We proceed, assuming WeakLearners handles the mismatch by padding or ignoring.

            # 1. Initialize and load WeakLearners
            weak_learners = WeakLearners(
                audio_dim=ssl_ensemble_config["audio_dim"],
                text_dim=ssl_ensemble_config["text_dim"],
                device=self.device,  # Output of weak learners will be on this device
            )
            if not weak_learners.load_fitted(ssl_ensemble_config["weak_learners_path"]):
                raise RuntimeError(
                    "Failed to load weak learners for MultiModalWhisper."
                )

            # 2. Initialize and load StackingMetaLearner
            meta_learner = StackingMetaLearner(
                weak_output_dim=len(weak_learners.models),  # Should be 3
                hidden_dim=ssl_ensemble_config["hidden_dim"],
            ).to(
                self.device
            )  # Meta learner runs on the main device
            meta_learner.load_state_dict_from_file(
                ssl_ensemble_config["meta_learner_path"], device=self.device
            )
            meta_learner.eval()

            # 3. Create the SSLEnsembleModel wrapper
            self.ssl_ensemble_model = SSLEnsembleModel(
                weak_learners=weak_learners, stacking_meta_learner=meta_learner
            )
            print("SSLEnsembleModel loaded successfully within MultiModalWhisper.")
        else:
            print(
                "No SSL Ensemble config provided. MultiModalWhisper will output embeddings."
            )

    def preprocess_audio(self, audios: list) -> torch.Tensor:
        """
        Accepts a list of audio waveforms (numpy.ndarray), processes them using Whisper:
         - padding/trimming,
         - log-Mel spectrogram computation,
         - passes through encoder, returns mean-pooled embeddings.
        Operates under torch.no_grad().
        """
        processed = []
        for audio in audios:
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio_proc = whisper.pad_or_trim(audio)
            # Compute mels on the whisper model's device
            mel = whisper.log_mel_spectrogram(
                audio_proc, device=self.whisper_model.device
            )
            processed.append(mel)

        if not processed:
            embed_dim = self.whisper_model.encoder.ln_post.normalized_shape[0]
            return torch.empty((0, embed_dim), device=self.device)

        mels = torch.stack(processed)
        with torch.no_grad():
            audio_embeddings = self.whisper_model.encoder(mels).mean(dim=1)
        return audio_embeddings

    def preprocess_text(self, texts: list) -> Optional[torch.Tensor]:
        """
        Accepts a list of strings and returns text embeddings using the
        selected tokenizer and text model ([CLS] token output).
        Operates under torch.no_grad(). Returns None if no text model is loaded.
        """
        if not self.text_model or not self.tokenizer:
            return None  # Return None if text processing is disabled
        if not texts:
            embed_dim = self.text_model.config.hidden_size
            return torch.empty((0, embed_dim), device=self.device)

        # Handle potential None or empty strings in the list robustly
        # Replace None or empty strings with a placeholder like "[PAD]" or check tokenizer behavior
        processed_texts = [
            (
                t
                if t
                else (
                    self.tokenizer.pad_token
                    if hasattr(self.tokenizer, "pad_token")
                    else "[PAD]"
                )
            )
            for t in texts
        ]

        inputs = self.tokenizer(
            processed_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # [CLS] token embedding is typically at index 0
            text_embeddings = self.text_model(**inputs).last_hidden_state[:, 0, :]
        return text_embeddings

    def forward(self, audios: list, texts: list) -> tuple:
        """
        Performs forward pass for both modalities using preprocessing methods.
        Returns a tuple (audio_embeddings, text_embeddings).
        text_embeddings can be None if text_model_type was 'none'.
        """
        audio_emb = self.preprocess_audio(audios)
        text_emb = self.preprocess_text(texts)
        return audio_emb, text_emb


# --- Weak Learners ---
class WeakLearners(nn.Module):
    """
    Wrapper for scikit-learn regression models to act as weak learners.
    Requires fitting on training data embeddings before use.
    Note: Sklearn models run on CPU. Data is moved accordingly.
    """

    # Added default dimensions matching common Whisper base/BERT base
    def __init__(self, audio_dim: int = 512, text_dim: int = 768, device: str = "cpu"):
        super().__init__()

        self.audio_dim = audio_dim
        self.text_dim = text_dim
        # Device primarily for moving predictions back, sklearn runs on CPU
        self.device = torch.device(device)

        # Initialize sklearn models
        self.ridge_regressor = Ridge(alpha=1.0)
        self.svr = SVR()
        self.dtr = DecisionTreeRegressor()

        self.models = [self.ridge_regressor, self.svr, self.dtr]
        self.model_names = ["Ridge", "SVR", "DTR"]
        self.fitted = False

    def fit(self, train_loader: torch.utils.data.DataLoader):
        """Train weak learners using embeddings from the train_loader."""
        print("Fitting weak learners...")

        all_audio_emb, all_text_emb, all_labels = [], [], []

        # Iterate through loader to get all embeddings (requires loader yields embeddings)
        pbar = tqdm(
            train_loader, desc="Extracting Embeddings for Weak Learners", leave=False
        )
        for batch_data in pbar:
            # Assuming loader yields (audio_emb, text_emb, labels)
            # Handle potential variations in loader output if necessary
            if len(batch_data) == 3:
                audio_emb, text_emb, labels = batch_data
            else:
                # Adapt if your loader yields something different (e.g., dict)
                raise ValueError(
                    "Expected train_loader to yield (audio_emb, text_emb, labels)"
                )

            # Ensure tensors are detached and moved to CPU
            all_audio_emb.append(audio_emb.detach().cpu().numpy())
            if (
                text_emb is not None
            ):  # Handle case where text embeddings might be missing
                all_text_emb.append(text_emb.detach().cpu().numpy())
            else:
                # If text_emb is None, append zeros of the correct shape
                # This requires text_dim to be known and non-zero if text is expected
                if self.text_dim > 0:
                    zeros_text = np.zeros((audio_emb.shape[0], self.text_dim))
                    all_text_emb.append(zeros_text)
                # else: text_dim is 0, no need to append anything for text

            all_labels.append(labels.detach().cpu().numpy())

        if not all_audio_emb or not all_labels:
            raise RuntimeError(
                "No embeddings or labels collected from train_loader! Check data loading."
            )

        # Concatenate all embeddings and labels
        all_audio_emb = np.vstack(all_audio_emb)
        all_labels = np.concatenate(all_labels)  # Use concatenate for 1D array

        # Combine audio and text embeddings
        if all_text_emb:  # If text embeddings were collected
            all_text_emb = np.vstack(all_text_emb)
            if all_audio_emb.shape[0] != all_text_emb.shape[0]:
                raise ValueError(
                    f"Mismatch in number of audio ({all_audio_emb.shape[0]}) and text ({all_text_emb.shape[0]}) samples."
                )
            combined_embeddings = np.hstack((all_audio_emb, all_text_emb))
            print(f"Combined embedding shape for fitting: {combined_embeddings.shape}")
        else:  # Only audio embeddings
            combined_embeddings = all_audio_emb
            print(
                f"Using only audio embeddings for fitting. Shape: {combined_embeddings.shape}"
            )

        if combined_embeddings.shape[0] != len(all_labels):
            raise ValueError(
                f"Mismatch between number of samples in embeddings ({combined_embeddings.shape[0]}) and labels ({len(all_labels)})."
            )

        # Fit each sklearn model
        print("Training sklearn models...")
        for model, name in zip(self.models, self.model_names):
            print(f"  Fitting {name}...")
            model.fit(combined_embeddings, all_labels)

        self.fitted = True
        print("Weak learners fitting completed.")

    def forward(
        self, audio_emb: torch.Tensor, text_emb: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Generate predictions from fitted weak learners.
        Input tensors should be on the correct device (e.g., GPU).
        They will be moved to CPU for sklearn, and predictions moved back to self.device.
        """
        if not self.fitted:
            raise RuntimeError(
                "Weak learners must be fitted before calling forward. Call fit()."
            )

        # Prepare input for sklearn: move to CPU, detach, convert to numpy
        audio_np = audio_emb.detach().cpu().numpy()

        if text_emb is not None:
            text_np = text_emb.detach().cpu().numpy()
            if audio_np.shape[0] != text_np.shape[0]:
                raise ValueError(
                    "Batch size mismatch between audio and text embeddings in forward pass."
                )
            combined_embeddings = np.hstack((audio_np, text_np))
        else:
            # If text_emb is None, use only audio (assuming model was trained this way if text_dim=0)
            if self.text_dim > 0:
                # If text was expected but not provided, behavior is undefined.
                # Option 1: Raise error. Option 2: Pad with zeros. Let's pad for now.
                print(
                    "Warning: Text embedding is None in forward pass, but text_dim > 0. Padding with zeros."
                )
                zeros_text = np.zeros((audio_np.shape[0], self.text_dim))
                combined_embeddings = np.hstack((audio_np, zeros_text))
            else:  # text_dim is 0, only use audio
                combined_embeddings = audio_np

        # Get predictions from each model
        all_preds = []
        for model in self.models:
            preds = model.predict(combined_embeddings)
            all_preds.append(torch.from_numpy(preds).float())

        # Stack predictions and move to the designated device
        stacked_preds = torch.stack(all_preds, dim=1).to(
            self.device
        )  # Shape: (batch_size, num_weak_learners)
        return stacked_preds


# --- Stacking Model (Meta-Learner) ---
class StackingMetaLearner(nn.Module):
    """
    A simple feed-forward network that learns to combine predictions
    from weak learners.
    """

    def __init__(self, weak_output_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(weak_output_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Predict a single MOS score

    def forward(self, weak_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            weak_outputs (torch.Tensor): Tensor of shape (batch_size, weak_output_dim)
                                         containing predictions from weak learners.
        Returns:
            torch.Tensor: Final prediction of shape (batch_size, 1).
        """
        x = self.relu(self.fc1(weak_outputs))
        x = self.fc2(x)
        return x


# --- Main Ensemble Model ---
class SSLEnsembleModel(nn.Module):
    """
    Combines WeakLearners (fitted sklearn models) and a StackingMetaLearner (PyTorch nn.Module).
    Requires pre-fitted WeakLearners instance.
    """

    def __init__(self, weak_learners: WeakLearners, hidden_dim: int = 256):
        super().__init__()
        if not isinstance(weak_learners, WeakLearners) or not weak_learners.fitted:
            raise ValueError("A pre-fitted WeakLearners instance must be provided.")

        self.weak_learners = weak_learners
        num_weak_outputs = len(
            weak_learners.models
        )  # Get number of outputs from weak learners
        self.stacking_meta_learner = StackingMetaLearner(
            weak_output_dim=num_weak_outputs, hidden_dim=hidden_dim
        )

    def forward(
        self, audio_emb: torch.Tensor, text_emb: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass through the ensemble.
        1. Get predictions from weak learners.
        2. Pass weak learner predictions to the stacking meta-learner.
        """
        # Get predictions from weak learners (shape: batch_size, num_weak_outputs)
        weak_outputs = self.weak_learners(audio_emb, text_emb)

        # Get final prediction from the meta-learner
        final_output = self.stacking_meta_learner(weak_outputs)
        return final_output
