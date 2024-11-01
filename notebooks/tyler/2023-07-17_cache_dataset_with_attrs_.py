import os, pickle, sys, torch

# horrible hack to get around this repo not being a proper python package
# Adjust the SCRIPT_DIR to point to where 'data_utils' and 'dataloaders' are located

SCRIPT_DIR = "/content/drive/MyDrive/2910monalisa/tyler_silent_speech"
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from dataloaders import LibrispeechDataset, cache_dataset
from datasets import load_dataset
import subprocess
from tqdm import tqdm

# On Colab, set ON_SHERLOCK to False
ON_SHERLOCK = False

# Load the LibriSpeech dataset using the datasets library
librispeech_datasets = load_dataset("librispeech_asr")
librispeech_train = torch.utils.data.ConcatDataset([
    librispeech_datasets["train.clean.100"],
    librispeech_datasets["train.clean.360"],
    librispeech_datasets["train.other.500"],
])
librispeech_clean_val = librispeech_datasets["validation.clean"]
librispeech_clean_test = librispeech_datasets["test.clean"]

# Adjust the path to 'normalizers.pkl'
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")
if not os.path.exists(normalizers_file):
    # Handle the case where the file doesn't exist
    # You might need to generate it or adjust the code to work without it
    print(f"'normalizers.pkl' not found at {normalizers_file}")
    # Optionally, you can skip loading normalizers or set default values
    mfcc_norm = None  # or some default value
    emg_norm = None
else:
    mfcc_norm, emg_norm = pickle.load(open(normalizers_file, "rb"))

text_transform = TextTransform(togglePhones=False)

# Set paths for Colab environment
# where is this?
sessions_dir = "/content/sessions_dir"  # og "/data/magneto/"
scratch_directory = "/content/scratch"  # og "/scratch"
gaddy_dir = "/content/gaddy_dir"        # og "/scratch/GaddyPaper/"

# Ensure the directories exist
os.makedirs(scratch_directory, exist_ok=True)

# Set cache directories
librispeech_train_cache = os.path.join(
    scratch_directory, "librispeech-cache", "2024-01-23_librispeech_noleak_train_phoneme_cache"
)
librispeech_val_cache = os.path.join(
    scratch_directory, "librispeech-cache", "2024-01-23_librispeech_noleak_val_phoneme_cache"
)
librispeech_test_cache = os.path.join(
    scratch_directory, "librispeech-cache", "2024-01-23_librispeech_noleak_test_phoneme_cache"
)

# Adjust the alignment directory
alignment_dir = os.path.join(scratch_directory, "librispeech-alignments")
if not os.path.exists(alignment_dir):
    # Handle the case where the alignment directory doesn't exist
    # You may need to download or generate the alignment data
    print(f"Alignment directory not found at {alignment_dir}")
    alignment_dirs = []  # Proceed without alignment data or handle accordingly
else:
    alignment_dirs = [os.path.join(alignment_dir, d) for d in os.listdir(alignment_dir)]

per_index_cache = True

# Proceed to cache datasets
# You may need to adjust or remove parameters related to alignment if not available
cached_speech_val = cache_dataset(
    librispeech_val_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)(
    librispeech_clean_val,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "dev" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_val

cached_speech_train = cache_dataset(
    librispeech_train_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)(
    librispeech_train,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "train" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_train

cached_speech_test = cache_dataset(
    librispeech_test_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)(
    librispeech_clean_test,
    text_transform,
    mfcc_norm,
    list(filter(lambda x: "test" in x, alignment_dirs)),
    skip_chapter_ids={
        127182, 127183, 127193, 127195, 128861, 141081, 141082, 141083, 141084
    }
)
del cached_speech_test