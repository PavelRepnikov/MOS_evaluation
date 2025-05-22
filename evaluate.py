# evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import os
import argparse
import whisper
import joblib
from typing import Optional, Tuple, List
import itertools
import traceback

# Import the unified SOMOSDataset class
from src.datasets.somos import SOMOSDataset

# Import model components
from src.models.whisper import WeakLearners, SSLEnsembleModel

# Import Hugging Face transformers for text models
from transformers import AutoTokenizer, AutoModel

# --- Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# --- Collate Function for Single Score Data ---
def create_collate_fn(
    audio_embed_model: nn.Module,  # Helper model (e.g., Whisper) for embedding
    tokenizer: Optional[AutoTokenizer],
    text_model: Optional[nn.Module],
    # Remove the 'device' argument here, collate fn should return CPU tensors
    # device: torch.device,
):
    """
    Factory to create a collate function that closes over loaded models.
    Processes batches of (audio_path, text, score).
    Generates embeddings on-the-fly for evaluation for single items.
    Returns CPU tensors.
    """
    # Determine the devices of the helper models ONCE
    # These models might be on GPU for faster inference during collate
    audio_model_device = next(audio_embed_model.parameters()).device
    text_model_device = (
        next(text_model.parameters()).device if text_model else torch.device("cpu")
    )

    def collate_fn(batch: List[Tuple[str, str, torch.Tensor]]):
        """
        Processes a batch of (audio_path, text, score).
        Loads audio, generates spectrograms, gets embeddings (potentially on helper model device).
        Tokenizes text, gets embeddings (potentially on helper model device).
        Returns final embeddings and scores as CPU tensors.
        """
        # --- (Filter batch as before) ---
        valid_batch = [item for item in batch if item and len(item) == 3 and item[0]]
        if not valid_batch:
            return None, None, None
        audio_paths, texts, scores = zip(*valid_batch)

        # --- (Load audio and filter as before) ---
        audios = []
        valid_indices_audio = []
        for i, path in enumerate(audio_paths):
            try:
                audio = whisper.load_audio(path)
                audios.append(audio)
                valid_indices_audio.append(i)
            except Exception as e:
                print(f"Warning: Error loading {path}: {e}")
        if not audios:
            return None, None, None
        texts = [texts[i] for i in valid_indices_audio]
        scores = [
            scores[i] for i in valid_indices_audio
        ]  # scores are already tensors here

        # --- Audio Processing ---
        audio_embeddings_final = None
        try:
            processed_audios = [whisper.pad_or_trim(audio) for audio in audios]
            # Compute spectrograms on the audio model's device
            mel_spectrograms = [
                whisper.log_mel_spectrogram(audio, device=audio_model_device)
                for audio in processed_audios
            ]
            mel_batch = torch.stack(mel_spectrograms)

            with torch.no_grad():
                # Ensure mel_batch is on the audio model's device
                mel_batch = mel_batch.to(audio_model_device)
                # Compute embeddings on the audio model's device
                audio_embeddings = audio_embed_model.encoder(mel_batch).mean(dim=1)
                # **** Crucial Change: Move final embedding to CPU ****
                audio_embeddings_final = audio_embeddings.cpu()
        except Exception as e:
            print(f"Error processing audio in collate: {e}. Returning None for batch.")
            traceback.print_exc()
            return None, None, None

        # --- Text Processing ---
        text_embeddings_final = None
        if text_model is not None and tokenizer is not None and texts:
            processed_texts = [
                (
                    str(t)
                    if pd.notna(t)
                    else (
                        tokenizer.pad_token
                        if hasattr(tokenizer, "pad_token")
                        else "[PAD]"
                    )
                )
                for t in texts
            ]
            try:
                # Tokenize (returns CPU tensors by default)
                inputs = tokenizer(
                    processed_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128,
                )
                # Move tokenized inputs to the text model's device for inference
                inputs = {key: val.to(text_model_device) for key, val in inputs.items()}
                with torch.no_grad():
                    # Compute embeddings on the text model's device
                    text_embeddings = text_model(**inputs).last_hidden_state[:, 0, :]
                    # **** Crucial Change: Move final embedding to CPU ****
                    text_embeddings_final = text_embeddings.cpu()
            except Exception as e:
                print(f"Error during text embedding generation: {e}")
                text_embeddings_final = None  # Keep as None if error

        # --- Scores ---
        # Ensure scores are stacked and on CPU (they should be CPU tensors from __getitem__)
        scores_tensor_final = (
            torch.stack(scores).cpu() if scores else torch.empty(0, dtype=torch.float32)
        )  # Added dtype

        # --- Final Checks & Return CPU Tensors ---
        if audio_embeddings_final is None:
            return None, None, None

        final_len = audio_embeddings_final.shape[0]
        # Adjust None assignment if text embedding failed
        if text_model is not None and text_embeddings_final is None:
            print(
                "Collate Warning: Text embedding failed, returning None for text tensor."
            )
            # Assign a placeholder of the correct shape if needed downstream, or keep None
            # text_embeddings_final = None # Already is None

        # Check lengths before returning CPU tensors
        if (
            text_embeddings_final is not None and text_embeddings_final.shape[0] != final_len
        ):
            print(
                f"Collate Error: Length mismatch Audio ({final_len}) vs Text ({text_embeddings_final.shape[0]})"
            )
            return None, None, None
        if scores_tensor_final.shape[0] != final_len:
            print(
                f"Collate Error: Length mismatch Audio ({final_len}) vs Scores ({scores_tensor_final.shape[0]})"
            )
            return None, None, None

        # Return CPU tensors
        return audio_embeddings_final, text_embeddings_final, scores_tensor_final

    return collate_fn


# --- Evaluation Function for Single Scores (MSE, LCC, SRCC etc.) ---
def evaluate_single_scores(
    model: nn.Module,  # The SSLEnsembleModel instance
    dataloader: DataLoader,
    device: torch.device,
    metric_name: str = "Score",  # Name of the metric being evaluated (e.g., "MOS", "SBS")
    criterion: nn.Module = nn.MSELoss(),  # Loss function (e.g., MSE)
) -> Tuple[float, float, float, float, float]:
    """
    Evaluates the model based on single ground truth scores per sample.
    Calculates standard regression metrics: MSE, RMSE, Pearson LCC, Spearman SRCC, Kendall Tau.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():  # Disable gradient calculations
        pbar = tqdm(dataloader, desc=f"Evaluating Single {metric_name}", leave=False)
        for i, batch_data in enumerate(pbar):
            # Check if collate function returned usable data
            if batch_data is None or len(batch_data) != 3 or batch_data[0] is None:
                print(f"Warning: Skipping problematic batch {i} from collate_fn.")
                continue

            try:
                audio_emb, text_emb, labels = batch_data

                # Further checks on tensor validity and shape
                if audio_emb.shape[0] == 0 or labels is None or labels.shape[0] == 0:
                    print(
                        f"Warning: Skipping empty/incomplete batch {i} after processing."
                    )
                    continue
                if audio_emb.shape[0] != labels.shape[0]:
                    print(
                        f"Warning: Skipping batch {i} due to embedding/label count mismatch ({audio_emb.shape[0]} vs {labels.shape[0]})"
                    )
                    continue
                if text_emb is not None and audio_emb.shape[0] != text_emb.shape[0]:
                    print(
                        f"Warning: Skipping batch {i} due to audio/text embedding count mismatch ({audio_emb.shape[0]} vs {text_emb.shape[0]})"
                    )
                    continue

                # Move data to the evaluation device
                audio_emb = audio_emb.to(device)
                labels = labels.to(device)
                # Handle optional text embeddings
                text_emb = text_emb.to(device) if text_emb is not None else None

                # Check if text embeddings are required but missing
                # Access weak_learners instance attached to the main model
                if (
                    text_emb is None and hasattr(model, "weak_learners") and model.weak_learners.text_dim > 0
                ):
                    print(
                        f"Warning: Skipping batch {i}, missing text emb required by model (text_dim={model.weak_learners.text_dim})."
                    )
                    continue

                # Perform forward pass
                outputs = model(audio_emb, text_emb)
                outputs_squeezed = (
                    outputs.squeeze()
                )  # Remove trailing dimension if necessary

                # Final shape check before calculating loss
                if outputs_squeezed.shape != labels.shape:
                    print(
                        f"Warning: Skipping batch {i} due to output/label shape mismatch after squeeze ({outputs_squeezed.shape} vs {labels.shape})"
                    )
                    continue

                # Calculate loss for the batch
                loss = criterion(outputs_squeezed, labels)
                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

                # Store predictions and labels (move to CPU for numpy conversion)
                preds_np = outputs_squeezed.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Check for NaNs/Infs before storing
                if (
                    np.isnan(preds_np).any() or np.isnan(labels_np).any() or np.isinf(preds_np).any() or np.isinf(labels_np).any()
                ):
                    print(
                        f"Warning: NaN/Inf detected in predictions or labels for batch {i}. Skipping batch results."
                    )
                    # Roll back stats for this batch if needed
                    total_loss -= loss.item() * batch_size
                    num_samples -= batch_size
                    continue

                all_preds.extend(preds_np.tolist())
                all_labels.extend(labels_np.tolist())
                pbar.set_postfix(batch_loss=loss.item())  # Update progress bar

            except Exception as e:
                print(f"\nError during single score evaluation batch {i}: {e}")
                traceback.print_exc()  # Print detailed traceback
                print("Skipping this batch.")
                continue  # Continue to the next batch

    # --- Calculate Overall Metrics ---
    if num_samples == 0 or not all_labels or not all_preds:
        print(
            f"Error: No valid samples were processed for single {metric_name} evaluation."
        )
        # Return NaN for all metrics
        return tuple([float("nan")] * 5)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Final check and filter for NaNs/Infs that might have slipped through batch checks
    valid_mask = np.isfinite(all_preds) & np.isfinite(all_labels)
    if not np.all(valid_mask):
        removed_count = np.sum(~valid_mask)
        print(
            f"Warning: Removing {removed_count} non-finite values before calculating final metrics."
        )
        all_preds = all_preds[valid_mask]
        all_labels = all_labels[valid_mask]

    # Need at least 2 valid points for correlation calculations
    if len(all_preds) < 2:
        print(
            f"Error: Fewer than 2 valid data points remaining for {metric_name}. Cannot calculate correlation metrics."
        )
        mse = (
            mean_squared_error(all_labels, all_preds)
            if len(all_preds) > 0
            else float("nan")
        )
        rmse = np.sqrt(mse) if not np.isnan(mse) else float("nan")
        avg_loss = total_loss / num_samples if num_samples > 0 else float("nan")
        # Return metrics, correlations will be NaN
        return mse, rmse, float("nan"), float("nan"), float("nan")

    # Calculate final metrics
    mse = mean_squared_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    avg_loss = (
        total_loss / num_samples if num_samples > 0 else float("nan")
    )  # Avoid division by zero
    # Calculate correlations safely
    try:
        lcc, _ = stats.pearsonr(all_labels, all_preds)
    except ValueError:
        lcc = float("nan")
    try:
        srcc, _ = stats.spearmanr(all_labels, all_preds)
    except ValueError:
        srcc = float("nan")
    try:
        kendall_tau, _ = stats.kendalltau(all_labels, all_preds)
    except ValueError:
        kendall_tau = float("nan")

    # --- Print Results ---
    print(f"\n--- Single {metric_name} Evaluation Results ---")
    print(f"  Valid Samples Processed: {len(all_preds)}")
    print(f"  Avg Loss   : {avg_loss:.4f}")
    print(f"  MSE        : {mse:.4f}")
    print(f"  RMSE       : {rmse:.4f}")
    print(f"  LCC (Pearson): {lcc:.4f}")
    print(f"  SRCC (Spearman): {srcc:.4f}")
    print(f"  Kendall Tau: {kendall_tau:.4f}")
    print("-------------------------------------")

    # Optional: Display some example predictions vs ground truth
    print(f"\nExamples (Prediction vs. Ground Truth {metric_name}):")
    num_examples_to_show = min(5, len(all_preds))
    if num_examples_to_show > 0:
        # Select random examples
        example_indices = np.random.choice(
            len(all_preds), num_examples_to_show, replace=False
        )
        for i in example_indices:
            print(f"  Pred: {all_preds[i]:.2f}, GT: {all_labels[i]:.2f}")
    else:
        print("  No valid predictions to display.")

    return mse, rmse, lcc, srcc, kendall_tau


# --- Evaluation Function for Pairwise Ranking (using single score data) ---
def evaluate_pairwise_ranking(
    model: nn.Module,  # Expects the SSLEnsembleModel
    dataloader: DataLoader,  # Expects DataLoader with standard collate_fn
    device: torch.device,
) -> Tuple[float, int]:
    """
    Evaluates the model's pairwise ranking accuracy using data with single scores.
    Pairs are formed dynamically within batches (or across batches if specified).
    """
    model.eval()
    correct_order_count = 0
    total_comparisons = 0
    # Store valid samples directly
    valid_audio_embeddings_cpu = []
    valid_text_embeddings_cpu = []
    valid_scores_cpu = []
    processed_sample_count = 0
    skipped_batches_count = 0

    # Determine if text is required by the model
    text_required = hasattr(model, "weak_learners") and model.weak_learners.text_dim > 0
    print(f"Pairwise Ranking: Text embeddings required by model: {text_required}")

    print("Collecting all valid predictions first for pairwise comparison...")
    # --- Step 1: Collect only valid samples (audio, text if needed, score) ---
    with torch.no_grad():
        pbar_collect = tqdm(dataloader, desc="Collecting data for ranking", leave=False)
        for i, batch_data in enumerate(pbar_collect):
            if batch_data is None or len(batch_data) != 3 or batch_data[0] is None:
                # print(f"Debug: Skipping invalid batch {i} from collate.") # Optional debug
                skipped_batches_count += 1
                continue  # Skip bad batches completely

            audio_emb, text_emb, scores = batch_data  # These are CPU tensors now

            # Basic validation
            if audio_emb.shape[0] == 0 or scores is None or scores.shape[0] == 0:
                # print(f"Debug: Skipping empty batch {i}.") # Optional debug
                skipped_batches_count += 1
                continue
            if audio_emb.shape[0] != scores.shape[0]:
                print(
                    f"Warning: Skipping batch {i} in collection, audio/score mismatch ({audio_emb.shape[0]} vs {scores.shape[0]})"
                )
                skipped_batches_count += 1
                continue

            # *** Crucial Filtering Logic ***
            # If text is required, text_emb MUST NOT be None for this batch
            if text_required:
                if text_emb is None:
                    print(
                        f"Warning: Skipping batch {i} in collection, text required but text_emb is None."
                    )
                    skipped_batches_count += 1
                    continue
                # Additional check: text shape must match audio shape
                if audio_emb.shape[0] != text_emb.shape[0]:
                    print(
                        f"Warning: Skipping batch {i} in collection, audio/text shape mismatch ({audio_emb.shape[0]} vs {text_emb.shape[0]})"
                    )
                    skipped_batches_count += 1
                    continue
                # If we reach here, text is required, present, and shapes match
                valid_audio_embeddings_cpu.append(audio_emb)
                valid_text_embeddings_cpu.append(
                    text_emb
                )  # It's definitely not None here
                valid_scores_cpu.append(scores)
                processed_sample_count += audio_emb.shape[0]
            else:
                # Text is not required, just collect audio and scores
                valid_audio_embeddings_cpu.append(audio_emb)
                # We don't need to store text_emb if not required
                valid_scores_cpu.append(scores)
                processed_sample_count += audio_emb.shape[0]

    # --- Check if any valid data was collected ---
    if not valid_scores_cpu or processed_sample_count == 0:
        print(
            f"Error: No valid data collected for pairwise evaluation after processing {i + 1} batches (skipped {skipped_batches_count})."
        )
        return float("nan"), 0
    print(
        f"Collected data for {processed_sample_count} valid samples (skipped {skipped_batches_count} batches)."
    )

    # --- Concatenate collected valid data ---
    try:
        all_audio_embeddings_cpu = torch.cat(valid_audio_embeddings_cpu, dim=0)
        all_scores_cpu = torch.cat(valid_scores_cpu, dim=0)
        # Concatenate text embeddings only if they were collected
        if text_required:
            all_text_embeddings_cpu = torch.cat(valid_text_embeddings_cpu, dim=0)
            # Sanity check lengths after concat
            if len(all_audio_embeddings_cpu) != len(all_text_embeddings_cpu):
                raise RuntimeError(
                    "Internal Error: Length mismatch after concatenating valid text embeddings."
                )
        else:
            all_text_embeddings_cpu = None  # Explicitly None if not required/collected

        # Final length check
        if len(all_audio_embeddings_cpu) != len(all_scores_cpu):
            raise RuntimeError(
                "Internal Error: Length mismatch between audio and scores after concatenating valid data."
            )

    except Exception as e:
        print(f"Error concatenating collected valid data: {e}")
        traceback.print_exc()
        return float("nan"), 0

    # Clear intermediate lists to free memory
    del valid_audio_embeddings_cpu, valid_text_embeddings_cpu, valid_scores_cpu

    # --- Step 2: Get all predictions from the model (using collected valid data) ---
    all_predictions = []
    eval_batch_size = dataloader.batch_size
    n_total_valid_samples = len(all_audio_embeddings_cpu)
    print(
        f"Generating predictions for {n_total_valid_samples} collected valid samples..."
    )
    with torch.no_grad():
        pbar_predict = tqdm(
            range(0, n_total_valid_samples, eval_batch_size),
            desc="Generating Predictions",
            leave=False,
        )
        for i in pbar_predict:
            # Prepare batch data, moving to evaluation device
            batch_audio_emb = all_audio_embeddings_cpu[i: i + eval_batch_size].to(
                device
            )
            batch_text_emb = None
            # Only get text batch if text embeddings exist
            if all_text_embeddings_cpu is not None:
                batch_text_emb = all_text_embeddings_cpu[i: i + eval_batch_size].to(
                    device
                )

            try:
                # Perform forward pass for the batch
                preds = model(batch_audio_emb, batch_text_emb).squeeze()
                all_predictions.append(preds.cpu())  # Store predictions on CPU
            except Exception as e:
                print(
                    f"\nError during prediction generation for batch starting at index {i}: {e}"
                )
                traceback.print_exc()
                batch_len = batch_audio_emb.shape[0]
                all_predictions.append(
                    torch.full((batch_len,), float("nan"), dtype=torch.float32)
                )  # Append NaNs

    # --- Check predictions and convert to numpy ---
    if not all_predictions:
        print("Error: Failed to generate predictions.")
        return float("nan"), 0
    try:
        all_predictions = torch.cat(all_predictions, dim=0).numpy()
        all_scores = all_scores_cpu.numpy()  # Convert collected scores
    except Exception as e:
        print(f"Error concatenating predictions/scores: {e}")
        return float("nan"), 0

    # Length check
    if (
        len(all_predictions) != n_total_valid_samples or len(all_scores) != n_total_valid_samples
    ):
        print(
            f"Error: Mismatch after prediction gen: Preds={len(all_predictions)}, Scores={len(all_scores)}, Expected={n_total_valid_samples}"
        )
        return float("nan"), 0

    # --- Step 3: Compare pairs and calculate ranking accuracy ---
    # Filter NaNs/Infs *after* prediction generation
    valid_mask = np.isfinite(all_predictions) & np.isfinite(all_scores)
    if not np.all(valid_mask):
        removed_count = np.sum(~valid_mask)
        print(
            f"Warning: Removing {removed_count} samples with non-finite predictions or scores before pairwise comparison."
        )
        all_predictions = all_predictions[valid_mask]
        all_scores = all_scores[valid_mask]

    n_final_samples = len(all_scores)
    if n_final_samples < 2:
        print("Error: < 2 valid samples remaining for pairwise comparison.")
        return float("nan"), 0

    print(f"Comparing {n_final_samples} final valid samples pairwise...")
    total_pairs_generated = n_final_samples * (n_final_samples - 1) // 2
    pbar_compare = tqdm(
        itertools.combinations(range(n_final_samples), 2),
        desc="Comparing Pairs",
        total=total_pairs_generated,
        leave=False,
    )
    correct_order_count = 0
    total_comparisons = 0

    for i, j in pbar_compare:
        score1, score2 = all_scores[i], all_scores[j]
        # Only compare if ground truth scores are different
        if score1 != score2:
            total_comparisons += 1
            pred1, pred2 = all_predictions[i], all_predictions[j]
            if (pred1 > pred2) == (score1 > score2):
                correct_order_count += 1

    # --- Calculate and Print Results ---
    if total_comparisons == 0:
        print("\nWarning: No pairs with different GT scores found.")
        ranking_accuracy = float("nan")
    else:
        ranking_accuracy = correct_order_count / total_comparisons

    print("\n--- Pairwise Ranking Evaluation Results ---")
    print(f"  Total Valid Samples Collected: {processed_sample_count}")
    print(f"  Samples after NaN/Inf Filter: {n_final_samples}")
    print(f"  Total Comparisons Made (GT different): {total_comparisons}")
    print(f"  Correctly Ordered Pairs: {correct_order_count}")
    print(f"  Pairwise Ranking Accuracy: {ranking_accuracy:.4f}")
    print("-----------------------------------------")

    return ranking_accuracy, total_comparisons


# --- Main Execution Function ---
def main(args):

    # --- Initialize Helper Embedding Models ---
    # Loads Whisper and optional text model based on args for use in collate_fn
    print("Initializing helper models for embedding generation...")
    audio_embed_model_helper = None
    if args.embedding_model.lower() == "whisper":
        try:
            # Load the specific whisper variant requested
            audio_embed_model_helper = whisper.load_model(
                args.whisper_variant, device=DEVICE
            )
            audio_embed_model_helper.eval()  # Set to eval mode
            print(f"Loaded Whisper helper model: {args.whisper_variant}")
        except Exception as e:
            print(f"Error loading Whisper model '{args.whisper_variant}': {e}")
            return  # Cannot proceed without embedding model
    # Add elif blocks here for other future embedding models
    # elif args.embedding_model.lower() == 'hubert':
    #     # Load HuBERT model and preprocessing if needed
    #     print("Error: HuBERT embedding model helper not implemented yet.")
    #     return
    else:
        print(f"Error: Unsupported embedding model type '{args.embedding_model}'")
        return

    # Initialize text model helper (if specified)
    bert_tokenizer = None
    bert_embed_model = None
    if args.text_model.lower() != "none":
        print(f"Loading text model helper: {args.text_model}")
        try:
            bert_tokenizer = AutoTokenizer.from_pretrained(args.text_model)
            bert_embed_model = AutoModel.from_pretrained(args.text_model).to(
                DEVICE
            )  # Load to GPU if available
            bert_embed_model.eval()  # Set to eval mode
        except Exception as e:
            print(
                f"Warning: Failed to load text model '{args.text_model}'. Text embeddings will not be used. Error: {e}"
            )
            bert_tokenizer = None
            bert_embed_model = None
    else:
        print("Text model set to 'none'. Skipping text embedding generation.")

    # --- Create Collate Function ---
    # Always use the standard collate function for single scores
    print("\n--- Using Standard Collate Function ---")
    dynamic_collate_fn = create_collate_fn(
        audio_embed_model=audio_embed_model_helper,
        tokenizer=bert_tokenizer,
        text_model=bert_embed_model,
    )

    # --- Instantiate Correct Dataset ---
    # Use StdSOMOSDataset or SOMOSDataset based on args
    print(
        f"\n--- Preparing Dataset ({args.dataset_format} type, split '{args.split}', metric '{args.metric}') ---"
    )
    eval_dataset = None  # Initialize
    try:
        # Use the unified SOMOSDataset for both formats
        eval_dataset = SOMOSDataset(
            data_path=args.data_path,
            transcript_path=args.transcript_file,
            audio_base_dir=args.audio_dir,
            dataset_format=args.dataset_format,  # Pass the format specifier
            metric=args.metric,  # Pass the desired metric
            split=args.split,  # Pass split info (used by 'standard')
            test_size=args.test_size,  # Pass splitting params (used by 'standard')
            seed=args.seed,  # Pass splitting params (used by 'standard')
        )

        if len(eval_dataset) == 0:
            print(
                "Error: Dataset is empty after initialization. Check paths, dataset_format, metric name, and filtering logic."
            )
            return

    except FileNotFoundError as e:
        print(f"Error: Data or transcript file not found: {e}")
        return
    except KeyError as e:
        print(f"Error: Required column missing in data (check format/metric): {e}")
        return
    except ValueError as e:
        print(f"Error: Invalid argument for dataset: {e}")
        return  # Catch format/metric errors
    except Exception as e:
        print(f"Error creating dataset: {e}")
        traceback.print_exc()
        return

    # --- Determine Embedding Dimensions ---
    # This step is crucial for initializing the model structure correctly
    print("\n--- Determining Embedding Dimensions ---")
    audio_dim = -1
    text_dim = -1
    try:
        # Use a small batch size just for dimension checking to be faster
        dim_check_loader = DataLoader(
            eval_dataset,
            batch_size=min(args.batch_size, 4),  # Use a small batch
            shuffle=False,  # No need to shuffle for dim check
            collate_fn=dynamic_collate_fn,
            num_workers=0,  # Using 0 workers is often safer for complex collate functions
        )

        print("Getting embedding dimensions from first valid batch...")
        first_valid_batch = None
        for batch in dim_check_loader:
            if (
                batch is not None and len(batch) == 3 and batch[0] is not None and batch[0].shape[0] > 0
            ):
                first_valid_batch = batch
                break  # Stop after finding the first valid batch

        if first_valid_batch is None:
            raise RuntimeError(
                "DataLoader did not yield any valid batches for dimension check."
            )

        audio_emb_sample, text_emb_sample, _ = first_valid_batch

        # Get dimensions
        audio_dim = audio_emb_sample.shape[1]
        # Handle case where text model is 'none' or embedding failed in collate
        text_dim = text_emb_sample.shape[1] if text_emb_sample is not None else 0
        print(f"Determined dimensions: Audio={audio_dim}, Text={text_dim}")

        # Clear memory used by the dim check loader and batch
        del dim_check_loader, first_valid_batch, audio_emb_sample, text_emb_sample

    except Exception as e:
        print(f"Error during embedding dimension check: {e}")
        traceback.print_exc()
        return  # Cannot proceed without dimensions

    # --- Load Trained Model Components ---
    # Loads the pre-trained WeakLearners (sklearn models) and the StackingMetaLearner (PyTorch model)
    print("\n--- Loading Trained Model Components ---")
    meta_learner_path = f"{args.model_base_path}_meta.pth"
    weak_learners_path = f"{args.model_base_path}_weak.joblib"
    print(f"  Attempting to load Meta-Learner state from: {meta_learner_path}")
    print(f"  Attempting to load Weak Learners state from: {weak_learners_path}")

    # 1. Initialize WeakLearners structure (needs correct dimensions)
    try:
        # Ensure the class is available
        if "WeakLearners" not in globals():
            raise NameError("WeakLearners class definition not found.")
        # Pass determined dimensions and target device (sklearn runs on CPU, but prediction tensors moved)
        weak_learners = WeakLearners(audio_dim, text_dim, device=DEVICE)
    except Exception as e:
        print(f"Error initializing WeakLearners structure: {e}")
        return

    # 2. Load the FITTED sklearn models using joblib
    if not os.path.exists(weak_learners_path):
        print(f"Error: Weak learners file not found at {weak_learners_path}")
        print(
            "       Ensure the model was trained and saved correctly (including the .joblib file)."
        )
        return
    try:
        loaded_sklearn_models = joblib.load(weak_learners_path)
        # Validate the loaded object - should be a list of fitted sklearn models
        if isinstance(loaded_sklearn_models, list) and len(
            loaded_sklearn_models
        ) == len(weak_learners.models):
            # Assign the loaded (already fitted) models to the WeakLearners instance
            weak_learners.models = loaded_sklearn_models
            weak_learners.fitted = True  # Mark the instance as fitted
            print("  Weak learners loaded from .joblib and assigned successfully.")
        else:
            print(
                f"Error: Loaded file '{weak_learners_path}' does not contain the expected list of sklearn models."
            )
            return
    except ImportError:
        print(
            "Error: joblib library is required to load weak learners. Please install it (`pip install joblib`)."
        )
        return
    except Exception as e:
        print(f"Error loading weak learners from {weak_learners_path}: {e}")
        traceback.print_exc()
        return

    # 3. Initialize the main SSLEnsembleModel structure
    # This combines the (now loaded) weak learners with the meta-learner
    try:
        if "SSLEnsembleModel" not in globals():
            raise NameError("SSLEnsembleModel class definition not found.")
        # Pass the weak_learners instance containing the loaded+fitted models
        ssl_ensemble_model = SSLEnsembleModel(
            weak_learners=weak_learners,  # Pass the restored weak learners
            hidden_dim=args.hidden_dim,  # Must match the trained meta-learner's hidden dim
        ).to(
            DEVICE
        )  # Move the PyTorch part (meta-learner) to the target device
    except Exception as e:
        print(f"Error initializing SSLEnsembleModel structure: {e}")
        traceback.print_exc()
        return

    # 4. Load the StackingMetaLearner's state_dict into the ensemble model
    if not os.path.exists(meta_learner_path):
        print(f"Error: Meta-learner state file (.pth) not found at {meta_learner_path}")
        return
    try:
        # Load the state dict intended ONLY for the meta-learner part
        meta_state_dict = torch.load(meta_learner_path, map_location=DEVICE)
        # Load it into the meta-learner component within the main ensemble model
        ssl_ensemble_model.stacking_meta_learner.load_state_dict(meta_state_dict)
        print("  StackingMetaLearner state loaded into model successfully.")
    except Exception as e:
        print(
            f"Error loading StackingMetaLearner state_dict from {meta_learner_path}: {e}"
        )
        print(
            "       Check if the saved state matches the current StackingMetaLearner architecture (input/hidden dims)."
        )
        traceback.print_exc()
        return

    # --- Prepare Full Evaluation Data Loader ---
    # Uses the same dataset instance created earlier for dimension check
    print(
        f"\n--- Preparing Full Evaluation DataLoader ({args.dataset_format} type, split '{args.split}', metric '{args.metric}') ---"
    )
    try:
        # We already created eval_dataset, just create the loader
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No shuffling needed for evaluation
            collate_fn=dynamic_collate_fn,  # Use the standard collate function
            num_workers=0,  # Set num_workers=0 for safety with complex collate fns / models
            pin_memory=(
                True if DEVICE.type == "cuda" else False
            ),  # Optional: speed up CPU->GPU transfer
        )
        print(f"DataLoader ready for evaluation with {len(eval_dataset)} samples.")
    except Exception as e:
        print(f"Error creating evaluation DataLoader: {e}")
        traceback.print_exc()
        return

    # --- Run Evaluation ---
    # Performs both single score evaluation and pairwise ranking evaluation
    print(f"\n--- Starting Evaluation (Metric: {args.metric}) ---")
    try:
        # 1. Run standard single score evaluation (MSE, LCC, SRCC, etc.)
        print("\nRunning standard single score evaluation...")
        evaluate_single_scores(
            model=ssl_ensemble_model,
            dataloader=eval_loader,
            device=DEVICE,
            metric_name=args.metric.upper(),  # Pass metric name for logging
        )

        # 2. Run pairwise ranking evaluation using the same data
        print("\nRunning pairwise ranking evaluation (comparing all pairs)...")
        evaluate_pairwise_ranking(
            model=ssl_ensemble_model,
            dataloader=eval_loader,  # Use the same loader
            device=DEVICE,
        )

    except Exception as e:
        print(f"\nError during the evaluation loops: {e}")
        traceback.print_exc()  # Print full traceback for debugging

    print("\nEvaluation finished.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SSL Ensemble MOS/SBS prediction model. Performs single-score correlation and pairwise ranking evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Show defaults in help message
    )

    # Data Arguments
    parser.add_argument(
        "--dataset_format",
        type=str,
        default="standard",
        choices=[
            "standard",
            "presplit",
        ],  # Only accept dataset types with single scores per sample
        help="Type of dataset format ('standard' for single CSV like normalised_somos.csv with internal split, 'presplit' for pre-split files like train_mos_list.txt).",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/archive/normalised_somos.csv",
        help="Path to the primary data file (CSV for 'standard', specific split file for 'presplit'). File must contain single scores per utterance.",
    )
    parser.add_argument(
        "--transcript_file",
        type=str,
        default="data/archive/all_transcripts.txt",
        help="Path to the transcript file (tab-separated: ID\\tText).",
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default="data/archive/update_SOMOS_v2/update_SOMOS_v2/all_audios/all_wavs",
        help="Path to the base directory containing WAV audio files.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="MOS",
        choices=["MOS", "SBS"],
        help="Which score column to use as ground truth from the data file (e.g., 'new_scale'/'mean' for MOS, 'SBS' column if present).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Which dataset split to evaluate on. Used for 'standard' type's internal splitting, or for logging with 'presplit'.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Fraction of data for validation split (only used if dataset_format='standard' and split is 'train'/'val').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val splitting (only used if dataset_format='standard').",
    )
    # Model Loading Arguments
    parser.add_argument(
        "--model_base_path",
        type=str,
        default="data/weights/best_ssl_ensemble_model",
        help="Base path for the saved model files. Expects '<base_path>_meta.pth' and '<base_path>_weak.joblib'.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden dimension of the StackingMetaLearner (must match the dimension used during training).",
    )

    # Embedding Model Configuration (Must match the setup used for training the loaded model)
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="whisper",
        choices=["whisper"],  # Add more choices as needed
        help="Base audio embedding model architecture used for feature extraction (must match training).",
    )
    parser.add_argument(
        "--whisper_variant",
        type=str,
        default="base.en",
        help="Whisper model variant (e.g., 'base.en', 'base', 'small'). Only used if embedding_model is 'whisper'.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="bert-base-uncased",
        help="Text model identifier for text embeddings (e.g., 'bert-base-uncased', 'DeepPavlov/rubert-base-cased', or 'none' to disable text).",
    )

    # Evaluation Arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for the evaluation dataloader.",
    )

    args = parser.parse_args()

    # --- Argument Validation and Path Resolution ---
    print("\n--- Validating Arguments ---")
    valid = True
    # Check required file/directory existence
    if not os.path.exists(args.data_path):
        print(f"Error: Data path not found: {args.data_path}")
        valid = False
    if not os.path.exists(args.transcript_file):
        print(f"Error: Transcript file not found: {args.transcript_file}")
        valid = False

    # Check model files
    meta_path = f"{args.model_base_path}_meta.pth"
    weak_path = f"{args.model_base_path}_weak.joblib"
    if not os.path.exists(meta_path):
        print(f"Error: Meta learner file not found: {meta_path}")
        valid = False
    if not os.path.exists(weak_path):
        print(f"Error: Weak learners file not found: {weak_path}")
        valid = False

    # Resolve audio directory path if it's relative
    if not os.path.isabs(args.audio_dir):
        try:
            # Assumes evaluate.py is run from a location where this relative path makes sense
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abs_audio_dir = os.path.join(script_dir, args.audio_dir)
            args.audio_dir = os.path.normpath(
                abs_audio_dir
            )  # Normalize path (e.g., handles ..)
            print(f"Resolved relative audio directory to: {args.audio_dir}")
        except (
            NameError
        ):  # __file__ might not be defined (e.g., in interactive session)
            print(
                "Warning: Could not resolve relative audio directory path automatically. Assuming path is relative to current working directory."
            )
            # No change needed, os functions will use CWD

    if not os.path.isdir(args.audio_dir):
        print(
            f"Error: Audio directory not found or is not a directory: {args.audio_dir}"
        )
        valid = False

    # Exit if any validation checks failed
    if not valid:
        print("\nExiting due to invalid arguments or missing files.")
        exit(1)

    print("Arguments and paths seem valid.")

    # --- Print Configuration ---
    print("\n--- Configuration ---")
    # Print all arguments for clarity and reproducibility
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("---------------------")

    # --- Run Main Evaluation Logic ---
    main(args)
