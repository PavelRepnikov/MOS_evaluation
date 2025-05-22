# tests/test_models.py

import unittest
import torch
import os
import joblib
import whisper  # openai-whisper
import numpy as np
import traceback  # For printing tracebacks on failure
from transformers import AutoTokenizer, AutoModel

# Assuming your project structure allows these imports from the 'src' directory
# Adjust the import path if your structure is different
try:
    from src.models.whisper import WeakLearners, SSLEnsembleModel
except ImportError:
    # Handle case where test is run in a way that src is not directly importable
    # You might need to adjust sys.path if running tests from a different location
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.models.whisper import WeakLearners, SSLEnsembleModel


# --- Configuration (MUST MATCH YOUR TRAINED MODEL) ---
# Adjust these paths and parameters based on your actual setup

# Paths relative to the tests/ directory
TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
TEST_AUDIO_FILENAME = "booksent_2012_0005_023.wav"
TEST_TRANSCRIPT_FILENAME = "booksent_2012_0005_023.txt"
TEST_AUDIO_PATH = os.path.join(TEST_DATA_DIR, TEST_AUDIO_FILENAME)
TEST_TRANSCRIPT_PATH = os.path.join(TEST_DATA_DIR, TEST_TRANSCRIPT_FILENAME)

# Path to saved model weights (relative to the project root, likely one level up from tests/)
MODEL_WEIGHTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/weights")
)  # Adjust if weights are elsewhere
MODEL_BASENAME = "best_ssl_ensemble_model"  # Base name used during saving
MODEL_BASE_PATH = os.path.join(MODEL_WEIGHTS_DIR, MODEL_BASENAME)
META_LEARNER_PATH = f"{MODEL_BASE_PATH}_meta.pth"
WEAK_LEARNERS_PATH = f"{MODEL_BASE_PATH}_weak.joblib"

# Model training configuration used for the saved weights (CHANGE THESE TO YOUR VALUES)
WHISPER_VARIANT = "base.en"  # e.g., 'base.en', 'base', 'small', 'medium', 'large'
TEXT_MODEL_TYPE = "bert-base-uncased"  # e.g., 'bert-base-uncased', 'DeepPavlov/rubert-base-cased', 'none'
HIDDEN_DIM = 256  # Meta-learner hidden dimension used during training

# --- Device Setup ---
# Use CPU for testing by default for portability and simplicity, unless GPU is strictly needed.
DEVICE = torch.device("cpu")
# Uncomment the line below to use GPU if available and necessary for the test
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSSLEnsembleModelInference(unittest.TestCase):
    """
    Test suite for performing inference with a trained SSLEnsembleModel.
    Loads the model components (weak learners, meta learner) and helper models (Whisper, Text),
    processes a sample audio/text pair, and verifies the inference pipeline runs correctly.
    """

    # Class attributes to hold loaded models and data
    whisper_helper = None
    tokenizer = None
    text_helper = None
    model = None
    weak_learners = None  # Keep weak learners if needed for inspection

    audio_emb = None
    text_emb = None
    audio_dim = -1
    text_dim = -1
    use_text_features = False  # Flag to track if text features are active

    @classmethod
    def setUpClass(cls):
        """
        Load models and preprocess data once for all tests in this class.
        This avoids redundant loading for each test method.
        """
        print("\n" + "=" * 30)
        print(" Setting up Test Case: Loading Models and Preprocessing ")
        print("=" * 30)
        print(f"Using device: {DEVICE}")

        # --- Validate Paths ---
        print("Validating required file paths...")
        paths_to_check = {
            "Test Audio": TEST_AUDIO_PATH,
            "Test Transcript": TEST_TRANSCRIPT_PATH,
            "Meta Learner Weights": META_LEARNER_PATH,
            "Weak Learner Weights": WEAK_LEARNERS_PATH,
        }
        for name, path in paths_to_check.items():
            print(f"  Checking {name}: {path}")
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} file not found at the expected location: {path}"
                )
        print("All required files found.")

        # Determine if text features should be attempted based on config
        cls.use_text_features = TEXT_MODEL_TYPE.lower() != "none"

        # --- Load Helper Models ---
        try:
            print(f"Loading Whisper helper model: {WHISPER_VARIANT}...")
            cls.whisper_helper = whisper.load_model(WHISPER_VARIANT, device=DEVICE)
            cls.whisper_helper.eval()  # Set to evaluation mode
            print("  Whisper helper loaded.")
        except Exception as e:
            print(
                f"FATAL ERROR: Could not load Whisper helper model '{WHISPER_VARIANT}'. Cannot proceed."
            )
            raise RuntimeError(f"Whisper loading failed: {e}") from e

        # Load text model helper only if configured
        if cls.use_text_features:
            print(f"Attempting to load Text helper model: {TEXT_MODEL_TYPE}...")
            try:
                cls.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_TYPE)
                cls.text_helper = AutoModel.from_pretrained(TEXT_MODEL_TYPE).to(DEVICE)
                cls.text_helper.eval()  # Set to evaluation mode
                print("  Text model helper loaded successfully.")
            except Exception as e:
                print(f"\nWARNING: Could not load text model '{TEXT_MODEL_TYPE}': {e}")
                print("         Disabling text features for this test run.\n")
                cls.use_text_features = False  # Disable text features if loading fails
                cls.tokenizer = None
                cls.text_helper = None
        else:
            print("Text model type is 'none'. Skipping text model loading.")

        # --- Load and Preprocess Test Data ---
        print("Loading and preprocessing test audio...")
        try:
            test_audio = whisper.load_audio(TEST_AUDIO_PATH)
            # Use whisper utilities for padding/trimming and spectrogram calculation
            test_audio_proc = whisper.pad_or_trim(test_audio)
            # Calculate mel spectrogram on the device where the whisper helper model resides
            mel = whisper.log_mel_spectrogram(
                test_audio_proc, device=cls.whisper_helper.device
            )
            # Add batch dimension and ensure it's on the whisper helper's device for encoding
            cls.mel_batch = mel.unsqueeze(0).to(cls.whisper_helper.device)
            print(f"  Audio preprocessed, mel spectrogram shape: {cls.mel_batch.shape}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or preprocess audio file {TEST_AUDIO_PATH}: {e}"
            ) from e

        print("Loading test transcript...")
        try:
            with open(TEST_TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
                # Expecting format like: ID\tText...
                line = f.readline().strip()
                parts = line.split("\t", 1)
                cls.test_transcript = (
                    parts[1] if len(parts) == 2 else line
                )  # Use text part or whole line
            if not cls.test_transcript:
                raise ValueError("Transcript file is empty or format is incorrect.")
            print(f"  Transcript loaded: '{cls.test_transcript}'")
        except Exception as e:
            raise RuntimeError(
                f"Failed to read transcript file {TEST_TRANSCRIPT_PATH}: {e}"
            ) from e

        # --- Generate Embeddings and Determine Dimensions ---
        # This simulates the process needed before loading the SSLEnsembleModel,
        # as its structure might depend on these dimensions.
        print("Generating embeddings to determine dimensions...")
        with torch.no_grad():
            # Generate Audio Embedding
            try:
                # Encoder runs on whisper_helper's device, result moved to target DEVICE
                cls.audio_emb = (
                    cls.whisper_helper.encoder(cls.mel_batch).mean(dim=1).to(DEVICE)
                )
                cls.audio_dim = cls.audio_emb.shape[1]
            except Exception as e:
                raise RuntimeError(f"Failed to generate audio embedding: {e}") from e

            # Generate Text Embedding (only if text features are active)
            cls.text_emb = None
            if cls.use_text_features:
                if cls.tokenizer and cls.text_helper:  # Check again if they were loaded
                    try:
                        inputs = cls.tokenizer(
                            cls.test_transcript,
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=128,
                        )
                        # Move tokenized inputs to the device of the text helper model
                        inputs = {
                            k: v.to(cls.text_helper.device) for k, v in inputs.items()
                        }
                        # Get embedding (e.g., [CLS] token) on text helper's device
                        text_output = cls.text_helper(**inputs).last_hidden_state[
                            :, 0, :
                        ]
                        # Move final text embedding to the target DEVICE
                        cls.text_emb = text_output.to(DEVICE)
                        cls.text_dim = cls.text_emb.shape[1]
                    except Exception as e:
                        print(
                            f"\nWARNING: Error during text embedding generation in setUpClass: {e}"
                        )
                        print("         Disabling text features for safety.\n")
                        # If embedding fails here, disable text features
                        cls.use_text_features = False
                        cls.text_dim = 0
                        cls.text_emb = None
                else:
                    # This case indicates an issue if use_text_features was True but models aren't loaded
                    print(
                        "WARNING: Text features were intended but helper models not loaded. Disabling text."
                    )
                    cls.use_text_features = False
                    cls.text_dim = 0
            else:
                # Text features were not enabled or failed during loading
                cls.text_dim = 0

        # Final confirmation of dimensions being used
        print(
            f"Using dimensions for model loading: Audio={cls.audio_dim}, Text={cls.text_dim}"
        )
        if cls.audio_dim <= 0:
            raise ValueError(
                "Audio dimension determined to be zero or negative. Check audio embedding generation."
            )

        # --- Load Trained SSLEnsembleModel ---
        print("Loading trained SSLEnsembleModel components...")
        # 1. Initialize WeakLearners structure with determined dimensions
        try:
            cls.weak_learners = WeakLearners(cls.audio_dim, cls.text_dim, device=DEVICE)
            print(
                f"  Initialized WeakLearners structure (AudioDim={cls.audio_dim}, TextDim={cls.text_dim})"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize WeakLearners structure: {e}"
            ) from e

        # 2. Load fitted sklearn models from .joblib file
        try:
            loaded_sklearn_models = joblib.load(WEAK_LEARNERS_PATH)
            # Basic validation of the loaded object
            num_expected_weak = len(cls.weak_learners.models)
            if (
                isinstance(loaded_sklearn_models, list)
                and len(loaded_sklearn_models) == num_expected_weak
            ):
                cls.weak_learners.models = loaded_sklearn_models
                cls.weak_learners.fitted = True  # Mark as fitted
                print(
                    f"  Loaded {len(loaded_sklearn_models)} weak learners from {WEAK_LEARNERS_PATH} successfully."
                )
            else:
                raise TypeError(
                    f"File {WEAK_LEARNERS_PATH} did not contain the expected list of {num_expected_weak} sklearn models."
                )
        except ImportError:
            raise ImportError(
                "Joblib not installed. Please install (`pip install joblib`) to load weak learners."
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"Failed to load or validate weak learners from {WEAK_LEARNERS_PATH}: {e}"
            ) from e

        # 3. Initialize the main SSLEnsembleModel using the loaded weak learners
        try:
            cls.model = SSLEnsembleModel(
                weak_learners=cls.weak_learners,  # Pass the instance with loaded+fitted models
                hidden_dim=HIDDEN_DIM,  # Must match training
            ).to(
                DEVICE
            )  # Move the PyTorch part (meta-learner) to the device
            print("  Initialized SSLEnsembleModel with loaded weak learners.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SSLEnsembleModel: {e}") from e

        # 4. Load the StackingMetaLearner's trained state dictionary
        try:
            meta_state_dict = torch.load(META_LEARNER_PATH, map_location=DEVICE)
            cls.model.stacking_meta_learner.load_state_dict(meta_state_dict)
            print(
                f"  Loaded StackingMetaLearner state from {META_LEARNER_PATH} successfully."
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Meta learner state file not found: {META_LEARNER_PATH}"
            ) from None
        except Exception as e:
            # Provide more specific error context if possible
            print(f"Error loading state dict into StackingMetaLearner. Details: {e}")
            print(
                "       Check if HIDDEN_DIM matches the trained model and if the .pth file is valid."
            )
            # Optionally print model structure vs state dict keys here for debugging
            # print("Model keys:", cls.model.stacking_meta_learner.state_dict().keys())
            # print("Loaded keys:", meta_state_dict.keys())
            raise RuntimeError(f"Failed to load meta learner state dict: {e}") from e

        # Ensure the final model is in evaluation mode
        cls.model.eval()
        print("-" * 30)
        print(" Test Case Setup Complete ")
        print("-" * 30)

    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests in the class have run."""
        print("\n" + "=" * 30)
        print(" Tearing Down Test Case ")
        print("=" * 30)
        # Explicitly delete large objects if memory is a concern
        del cls.whisper_helper
        del cls.tokenizer
        del cls.text_helper
        del cls.model
        del cls.weak_learners
        del cls.audio_emb
        del cls.text_emb
        # Call garbage collection if needed
        # import gc
        # gc.collect()
        # if DEVICE.type == 'cuda':
        #     torch.cuda.empty_cache()
        print("Cleanup finished.")

    def test_inference_single_sample(self):
        """
        Tests the forward pass of the fully loaded SSLEnsembleModel
        with a single preprocessed audio/text sample and checks the output value.
        """
        print("\n--- Running Test: test_inference_single_sample ---")
        self.assertIsNotNone(
            self.model, "FAIL: SSLEnsembleModel was not loaded in setUpClass."
        )
        self.assertIsNotNone(
            self.audio_emb, "FAIL: Audio embedding was not generated in setUpClass."
        )
        # Check text embedding only if text features were successfully enabled
        if self.use_text_features:
            self.assertIsNotNone(
                self.text_emb,
                "FAIL: Text embedding is None even though text features were enabled.",
            )
            # Check batch size consistency (should be 1 for this test)
            if (
                self.text_emb is not None
            ):  # Ensure text_emb is not None before accessing shape
                self.assertEqual(
                    self.audio_emb.shape[0],
                    self.text_emb.shape[0],
                    "FAIL: Batch size mismatch between audio and text embeddings.",
                )
        else:
            self.assertIsNone(
                self.text_emb,
                "FAIL: Text embedding should be None when text features are disabled.",
            )

        # --- Prepare Inputs for Model ---
        # Ensure embeddings are on the correct device (should already be from setUpClass)
        audio_emb_input = self.audio_emb.to(DEVICE)
        text_emb_input = self.text_emb.to(DEVICE) if self.text_emb is not None else None

        # --- Perform Inference ---
        print("Performing model inference...")
        try:
            with torch.no_grad():  # Ensure no gradients are computed
                output = self.model(audio_emb_input, text_emb_input)
            print(f"  Inference successful. Output tensor shape: {output.shape}")
        except Exception as e:
            # Use fail to provide a clear error message and stop the test
            self.fail(
                f"Model inference failed with exception: {e}\n{traceback.format_exc()}"
            )

        # --- Assertions on Output ---
        print("Asserting output properties...")
        # 1. Check Type
        self.assertIsInstance(
            output,
            torch.Tensor,
            f"FAIL: Output type is {type(output)}, expected torch.Tensor.",
        )

        # 2. Check Shape (expecting batch_size=1, output_dim=1)
        self.assertEqual(
            output.ndim,
            2,
            f"FAIL: Expected output ndim to be 2 (batch, features), got {output.ndim}.",
        )
        self.assertEqual(
            output.shape[0],
            1,
            f"FAIL: Expected batch size 1 in output, got {output.shape[0]}.",
        )
        self.assertEqual(
            output.shape[1],
            1,
            f"FAIL: Expected output feature dimension 1, got {output.shape[1]}.",
        )

        # 3. Check Value (Scalar)
        try:
            model_output_value = output.squeeze().item()  # Get the single scalar value
            print(f"  Model Output Scalar Value: {model_output_value:.4f}")
        except Exception as e:
            self.fail(f"Failed to extract scalar value from output tensor: {e}")

        # 4. Check Finite Value
        self.assertTrue(
            np.isfinite(model_output_value),
            f"FAIL: Output value ({model_output_value}) is not finite (NaN or Inf).",
        )

        # 5. **** Added Specific Value Check ****
        expected_value = 3.7504
        # Use assertAlmostEqual for floating-point comparison
        # Choose precision (e.g., places=3 for 3 decimal places) or delta
        # places = 3
        # self.assertAlmostEqual(model_output_value, expected_value, places=places,
        #                        msg=f"FAIL: Output value {model_output_value:.4f} is not close enough to expected {expected_value} (checked to {places} places).")

        # OR check using a delta (absolute difference)
        delta = 0.001  # Allow a small tolerance
        self.assertAlmostEqual(
            model_output_value,
            expected_value,
            delta=delta,
            msg=f"FAIL: Output value {model_output_value:.4f} is not close enough to expected {expected_value} (allowed delta={delta}).",
        )
        print(f"  Output value is close to expected value {expected_value}.")

        # 6. Optional: Plausible Range Check (Uncomment and adjust if you have expectations)
        # expected_min, expected_max = 1.0, 5.0 # Example: MOS range
        # self.assertTrue(expected_min <= model_output_value <= expected_max,
        #                 f"FAIL: Output value {model_output_value:.4f} is outside the expected range [{expected_min}, {expected_max}].")

        print("--- test_inference_single_sample PASSED ---")


# --- Allows running the test file directly using 'python tests/test_models.py' ---
# (Keep the if __name__ == '__main__': block as before)
if __name__ == "__main__":
    print(
        "Running tests directly. Consider using 'python -m unittest discover tests' from project root."
    )
    unittest.main()
