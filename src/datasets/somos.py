# src/datasets/somos.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict


def _load_transcripts(path: str) -> Dict[str, str]:
    """Loads transcripts from a file into a dictionary."""
    transcripts = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t", 1)  # Split on the first tab
                if len(parts) == 2:
                    utt_id, text = parts
                    # Assume transcript keys do NOT have .wav extension
                    # Store the base ID (without extension) as the key
                    transcripts[utt_id.replace(".wav", "")] = text
                # else:
                #     print(f"Warning: Skipping malformed transcript line: {line.strip()} in {path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Transcript file not found at: {path}")
    except Exception as e:
        raise RuntimeError(f"Error loading transcripts from {path}: {e}")
    if not transcripts:
        print(
            f"Warning: No transcripts loaded from {path}. Check file format (expected: 'utt_id\\ttranscript')."
        )
    return transcripts


# --- Consolidated SOMOSDataset ---
class SOMOSDataset(Dataset):
    """
    PyTorch Dataset for SOMOS data, handling multiple formats.

    Loads audio paths, corresponding transcripts, and specified metric score (MOS or SBS).
    Can handle 'standard' format (e.g., normalised_somos.csv, requires internal splitting)
    or 'presplit' format (e.g., train_mos_list.txt, uses file as split).

    Args:
        data_path (str): Path to the primary data file (CSV or TXT).
        transcript_path (str): Path to the transcript file.
                               Expected format: "utterance_id\\ttranscript text\\n".
                               Utterance IDs in transcript file should NOT have .wav extension.
        audio_base_dir (str): Base directory containing the audio files (wav format).
        dataset_format (str): The format of the data_path file ('standard' or 'presplit').
        metric (str): The metric score to load ('MOS' or 'SBS').
        split (str): Defines the dataset split ('train', 'val', 'test', 'all').
                     Used for internal splitting if format is 'standard', ignored otherwise.
        test_size (float): Proportion for validation set if format is 'standard' and split is 'train'/'val'.
        seed (int): Random seed for splitting if format is 'standard'.
    """

    def __init__(
        self,
        data_path: str,
        transcript_path: str,
        audio_base_dir: str,
        dataset_format: str,  # Added parameter
        metric: str = "MOS",
        split: str = "train",
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.audio_base_dir = audio_base_dir
        self.metric = metric.upper()
        self.dataset_format = dataset_format.lower()
        self.split = split  # Used for standard splitting or just logging

        if self.dataset_format not in ["standard", "presplit"]:
            raise ValueError(
                f"Invalid dataset_format '{self.dataset_format}'. Choose 'standard' or 'presplit'."
            )
        if self.metric not in ["MOS", "SBS"]:
            raise ValueError(f"Invalid metric '{self.metric}'. Choose 'MOS' or 'SBS'.")

        # --- Load Metadata based on format ---
        print(
            f"Loading SOMOS dataset (format: {self.dataset_format}, metric: {self.metric}, split: {self.split}) from: {data_path}"
        )
        full_df = None
        id_col_in_file = None  # Original ID column name in the file
        score_col_in_file = None  # Original score column name in the file
        needs_internal_split = False
        id_has_extension_in_file = False  # Flag to track original ID format

        try:
            if self.dataset_format == "standard":
                if self.metric != "MOS":
                    # The standard format we know (normalised_somos.csv) only has MOS ('new_scale')
                    raise ValueError(
                        f"Metric '{self.metric}' not available in 'standard' dataset format (expected 'MOS')."
                    )

                full_df = pd.read_csv(data_path)
                id_col_in_file = full_df.columns[0]  # Assume first col is ID
                score_col_in_file = "new_scale"
                needs_internal_split = True

                # Check if required columns exist
                if id_col_in_file not in full_df.columns:
                    raise KeyError(
                        f"Expected ID column '{id_col_in_file}' (first col) not found."
                    )
                if score_col_in_file not in full_df.columns:
                    raise KeyError(
                        f"Required score column '{score_col_in_file}' not found for standard format."
                    )

            elif self.dataset_format == "presplit":
                # This format should have 'utteranceId', 'mean' (for MOS), 'SBS'
                # Read CSV, check for potential unnamed index column
                temp_df = pd.read_csv(data_path)
                if temp_df.columns[0].startswith("Unnamed:"):
                    print(
                        f"  Detected unnamed index column '{temp_df.columns[0]}', reading with index_col=0."
                    )
                    full_df = pd.read_csv(data_path, index_col=0)
                else:
                    full_df = temp_df

                id_col_in_file = "utteranceId"
                score_mapping = {"MOS": "mean", "SBS": "SBS"}
                score_col_in_file = score_mapping.get(self.metric)

                if (
                    score_col_in_file is None
                ):  # Should not happen due to earlier check, but defensive
                    raise ValueError(
                        f"Internal error: No score column mapping for metric '{self.metric}'."
                    )

                needs_internal_split = False  # File itself defines the split

                # Check if required columns exist
                if id_col_in_file not in full_df.columns:
                    raise KeyError(
                        f"Required ID column '{id_col_in_file}' not found in presplit format."
                    )
                if score_col_in_file not in full_df.columns:
                    raise KeyError(
                        f"Required score column '{score_col_in_file}' (for metric '{self.metric}') not found in presplit format."
                    )

            # --- Dataframe Processing (Common Steps) ---
            if full_df is None:  # Should only happen if format invalid
                raise ValueError("Internal error: full_df not loaded.")

            # 1. Standardize ID: Store the BASE ID (without .wav) in 'utterance_id' column
            #    and ensure it's a string.
            full_df["utterance_id"] = (
                full_df[id_col_in_file].astype(str).str.replace(".wav", "", regex=False)
            )

            # 2. Standardize Score: Copy the relevant score to a 'score' column
            full_df["score"] = full_df[score_col_in_file].astype(float)

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at: {data_path}")
        except KeyError as e:
            raise KeyError(
                f"Missing expected column in {data_path}: {e}. Found columns: {full_df.columns if full_df is not None else 'Error loading DF'}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Error loading or processing metadata from {data_path}: {e}"
            )

        # --- Load Transcripts ---
        # Transcript keys should be base IDs (no extension)
        self.transcripts = _load_transcripts(transcript_path)

        # --- Filter out samples missing transcripts or audio ---
        valid_indices = []
        missing_audio_count = 0
        missing_transcript_count = 0
        original_count = len(full_df)

        print("Filtering dataset...")
        for idx, row in full_df.iterrows():
            # Use the standardized 'utterance_id' (base ID without extension)
            utt_id_base = row["utterance_id"]

            # Transcript lookup uses the base ID directly
            has_transcript = utt_id_base in self.transcripts

            # Audio path always needs the .wav extension added to the base ID
            audio_path = os.path.join(self.audio_base_dir, f"{utt_id_base}.wav")
            has_audio = os.path.exists(audio_path)

            if has_transcript and has_audio:
                valid_indices.append(idx)
            elif not has_transcript:
                missing_transcript_count += 1
                # print(f"Debug: Missing transcript for base ID: {utt_id_base}") # Optional debug
            elif not has_audio:
                missing_audio_count += 1
                # print(f"Debug: Missing audio for path: {audio_path}") # Optional debug

        # Select only valid rows and essential columns
        filtered_df = full_df.loc[valid_indices, ["utterance_id", "score"]].copy()

        print(
            f"Original samples in file: {original_count}. Found transcripts for {len(self.transcripts)} IDs."
        )
        if missing_audio_count > 0:
            print(
                f"Warning: Could not find audio files for {missing_audio_count} utterances."
            )
        if missing_transcript_count > 0:
            print(
                f"Warning: Utterances missing transcripts: {missing_transcript_count}."
            )
        print(f"Filtered dataset size (valid audio and transcript): {len(filtered_df)}")

        if len(filtered_df) == 0:
            raise ValueError(
                "No valid samples found after filtering. Check paths, file contents, ID formats, and transcript keys."
            )

        # --- Apply Split if needed ('standard' format) ---
        if needs_internal_split:
            print(
                f"Applying internal split '{self.split}' (test_size={test_size}, seed={seed})..."
            )
            if self.split in ["train", "val"] and 0 < test_size < 1:
                if len(filtered_df) > 1:
                    train_df, val_df = train_test_split(
                        filtered_df,
                        test_size=test_size,
                        random_state=seed,
                        # Optional: Add stratification here if desired, e.g., on binned scores
                        # stratify=pd.cut(filtered_df['score'], bins=5, labels=False, duplicates='drop')
                    )
                    self.metadata = train_df if self.split == "train" else val_df
                else:  # Only 1 sample, cannot split
                    print(
                        "Warning: Only one valid sample. Using it for the requested split."
                    )
                    self.metadata = filtered_df
            elif self.split == "all":
                print("Split is 'all', using all valid samples.")
                self.metadata = filtered_df
            else:  # e.g., split='test' or invalid split for standard format
                print(
                    f"Warning: Split '{self.split}' not handled for internal splitting or invalid. Using all {len(filtered_df)} valid samples."
                )
                self.metadata = filtered_df
        else:
            # 'presplit' format - use all data from the file
            print(f"Using all {len(filtered_df)} valid samples from pre-split file.")
            self.metadata = filtered_df

        print(
            f"Final dataset size for split '{self.split}' / file: {len(self.metadata)}"
        )
        # Reset index for clean __getitem__ access
        self.metadata = self.metadata.reset_index(drop=True)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[str, str, torch.Tensor]:
        """
        Returns a single sample: audio path, transcript, and score tensor.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            Tuple[str, str, torch.Tensor]: A tuple containing:
                - audio_path (str): Full path to the audio file (always with .wav).
                - transcript (str): The corresponding transcript.
                - score (torch.Tensor): The score (MOS or SBS) as a float tensor.
        """
        row = self.metadata.iloc[index]
        # Use the standardized base ID (without extension)
        utt_id_base = str(row["utterance_id"])
        score = torch.tensor(row["score"], dtype=torch.float32)

        # Get transcript using the base ID
        transcript = self.transcripts.get(utt_id_base, "")

        # Construct audio path by adding .wav to the base ID
        audio_path = os.path.join(self.audio_base_dir, f"{utt_id_base}.wav")

        # Optional: Final check if audio path exists (should be redundant due to filtering)
        # if not os.path.exists(audio_path):
        #     print(f"Warning: Audio file missing at __getitem__ for index {index}: {audio_path}")
        #     # Handle error appropriately if needed (e.g., return None or raise)

        return audio_path, transcript, score


# --- SBSDataset class is removed ---
