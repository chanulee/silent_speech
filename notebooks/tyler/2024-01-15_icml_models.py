##
# based on these two files:
# 2023-07-25_dtw_speech_silent_emg.py : best sEMG results
# 2023-08-24_brain_to_text_comp_split.py : most recent brain-to-text results, uses MONA name
# to run:
# CUDA_VISIBLE_DEVICES=0 taskset -c 0-7 python 2024-01-15_icml_models.py --no-dtw --no-supTcon
2
##
# %load_ext autoreload
# %autoreload 2
##
import os, subprocess

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync" # no OOM
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
hostname = subprocess.run("hostname", capture_output=True)
ON_SHERLOCK = hostname.stdout[:2] == b"sh"
# ON_sh03_11n03 = hostname.stdout[:10] == b"sh03-11n03"  # henderj A100 80GB
ON_sh03_11n03 = hostname.stdout[:10] == b"no-match"

import pytorch_lightning as pl, pickle
from pytorch_lightning.plugins.environments import SLURMEnvironment
import sys, warnings
import numpy as np
import logging, signal
import torchmetrics
import random, typer
from tqdm.auto import tqdm
from typing import List
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DistributedSampler
import torch.nn.functional as F

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)

from read_emg import (
    EMGDataset,
    PreprocessedEMGDataset,
    PreprocessedSizeAwareSampler,
    EMGDataModule,
    ensure_folder_on_scratch,
)
from architecture import Model, S4Model, H3Model, ResBlock, MONAConfig, MONA
from data_utils import combine_fixed_length, decollate_tensor
from transformer import TransformerEncoderLayer
from pytorch_lightning.loggers import NeptuneLogger

# import neptune, shutil
import neptune.new as neptune, shutil
import typer
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, GradientAccumulationScheduler
from pytorch_lightning.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
    PassThroughProfiler,
)
from pytorch_lightning.strategies import DDPStrategy
from data_utils import TextTransform, in_notebook
from typing import List
from collections import defaultdict
from enum import Enum
from magneto.preprocessing import ensure_data_on_scratch
from dataloaders import (
    LibrispeechDataset,
    EMGAndSpeechModule,
    DistributedStratifiedBatchSampler,
    StratifiedBatchSampler,
    cache_dataset,
    split_batch_into_emg_neural_audio,
    DistributedSizeAwareStratifiedBatchSampler,
    SizeAwareStratifiedBatchSampler,
    collate_gaddy_or_speech,
    collate_gaddy_speech_or_neural,
    DistributedSizeAwareSampler,
    T12DataModule,
    T12Dataset,
    NeuralDataset,
    T12CompDataModule,
    BalancedBinPackingBatchSampler,
)
from functools import partial
from contrastive import (
    cross_contrastive_loss,
    var_length_cross_contrastive_loss,
    nobatch_cross_contrastive_loss,
    supervised_contrastive_loss,
)
import glob, scipy
from warnings import warn
from helpers import (
    load_npz_to_memory,
    get_last_ckpt,
    get_neptune_run,
    nep_get,
    string_to_np_array,
)

##
DEBUG = False
# DEBUG = True

# TODO:
# # check if a SLURM re-queue
# neptune_logger.experiment["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]


# https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
# not sure if makes a difference when we use fp16 / bf16
# use TF32 cores on A100 (19-bit)
torch.set_float32_matmul_precision("high")  # highest (32-bit) by default

torch.backends.cudnn.allow_tf32 = True  # should be True by default
run_id = ""
ckpt_path = ""
# ckpt_path = '/scratch/2023-07-10T12:20:43.920850_gaddy/SpeechOrEMGToText-epoch=29-val/wer=0.469.ckpt'
# ckpt_path = "/scratch/2023-08-03T21:30:03.418151_gaddy/SpeechOrEMGToText-epoch=15-val/wer=0.547.ckpt"
# run_id = "GAD-493"

per_index_cache = True  # read each index from disk separately
# per_index_cache = False # read entire dataset from disk

isotime = datetime.now().isoformat()

if DEBUG:
    NUM_GPUS = 1
    limit_train_batches = 2
    limit_val_batches = 2  # will not run on_validation_epoch_end
    # NUM_GPUS = 2
    # limit_train_batches = None
    # limit_val_batches = None
    log_neptune = False
    n_epochs = 2
    # n_epochs = 200
    # precision = "32"
    # precision = "16-mixed"
    precision = "bf16-mixed"
    num_sanity_val_steps = 2
    grad_accum = 1
    logger_level = logging.DEBUG
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

else:
    NUM_GPUS = 1
    grad_accum = 2  # might need if run on 1 GPU
    # grad_accum = 1
    # precision = "16-mixed"
    precision = "bf16-mixed"
    limit_train_batches = None
    limit_val_batches = None
    log_neptune = True
    # log_neptune = False
    n_epochs = 200
    num_sanity_val_steps = 0  # may prevent crashing of distributed training
    logger_level = logging.WARNING


assert (
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] == "TRUE"
), "run this in shell: export NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE='TRUE'"

# load our data file paths and metadata:

if per_index_cache:
    cache_suffix = "_per_index"
else:
    cache_suffix = ""
if ON_SHERLOCK:
    sessions_dir = "/oak/stanford/projects/babelfish/magneto/"
    # TODO: bechmark SCRATCH vs LOCAL_SCRATCH ...?
    scratch_directory = os.environ["SCRATCH"]
    librispeech_directory = "/oak/stanford/projects/babelfish/magneto/librispeech-cache"
    # scratch_directory = os.environ["LOCAL_SCRATCH"]
    # gaddy_dir = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/"
    gaddy_dir = os.path.join(scratch_directory, "GaddyPaper")
    # scratch_lengths_pkl = os.path.join(
    #     scratch_directory, "2023-07-25_emg_speech_dset_lengths.pkl"
    # )
    # tmp_lengths_pkl = os.path.join("/tmp", "2023-07-25_emg_speech_dset_lengths.pkl")
    # if os.path.exists(scratch_lengths_pkl) and not os.path.exists(tmp_lengths_pkl):
    #     shutil.copy(scratch_lengths_pkl, tmp_lengths_pkl)
    t12_npz_path = os.path.join(scratch_directory, "2023-08-21_T12_dataset.npz")
    T12_dir = os.path.join(scratch_directory, "T12_data_v4")
    if len(os.sched_getaffinity(0)) > 16:
        print(
            "WARNING: if you are running more than one script, you may want to use `taskset -c 0-16` or similar"
        )
else:
    # on my local machine
    sessions_dir = "/data/magneto/"
    scratch_directory = "/scratch"
    gaddy_dir = "/scratch/GaddyPaper/"
    # t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-21_T12_dataset_per_sentence_z-score.npz"
    t12_npz_path = "/data/data/T12_data_v4/synthetic_audio/2023-08-22_T12_dataset_gaussian-smoothing.npz"
    T12_dir = "/data/data/T12_data_v4/"

print(f"CPU affinity: {os.sched_getaffinity(0)}")

data_dir = os.path.join(gaddy_dir, "processed_data/")
# lm_directory = os.path.join(gaddy_dir, "pretrained_models/librispeech_lm/")
lm_directory = "/oak/stanford/projects/babelfish/magneto/GaddyPaper/icml_lm/"
normalizers_file = os.path.join(SCRIPT_DIR, "normalizers.pkl")

if ON_sh03_11n03:
    lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["LOCAL_SCRATCH"])
    librispeech_directory = ensure_folder_on_scratch(
        librispeech_directory, os.environ["LOCAL_SCRATCH"]
    )
elif ON_SHERLOCK:
    # avoid race condition for owners nodes if 2 jobs on same node
    # also less load time since interruptable
    lm_directory = ensure_folder_on_scratch(lm_directory, os.environ["SCRATCH"])
    librispeech_directory = ensure_folder_on_scratch(
        librispeech_directory, os.environ["SCRATCH"]
    )

gpu_ram = torch.cuda.get_device_properties(0).total_memory / 1024**3
if not gpu_ram > 70:
    warn("expecting A100 80GB, may OOM with supTcon")
##
# base_bz was 24 per GPU when run on 4 GPUs
# of classes in each batch. and maybe overrepresents silent EMG
base_bz = 24 * 4
val_bz = 2  # terrible memory usage even at 8, I'm not sure why so bad...
# gaddy used max_len = 128000, we double because of LibriSpeech
# TODO: try 512000 and grad_accum=1 (prob OOM but might be faster!)
# also almost def won't work for supTcon + dtw
# max_len = 48000 # from best perf with 4 x V100
max_len = 128000  #
# max_len = 64000
# max_len = 256000 # OOM on A100 80GB w/ supTcon

##

app = typer.Typer()

togglePhones = False
learning_rate = 3e-4
seqlen = 600
white_noise_sd = 0
constant_offset_sd = 0
use_dtw = True
use_crossCon = True
use_supTcon = True
audio_lambda = 1.0
emg_lambda = 1.0
weight_decay = 0.1
latent_affine = True
# Gaddy is 16% silent EMG, 84% vocalized EMG, and we use LibriSpeech for the rest
# by utterance count, not by time
frac_semg = 1588 / (5477 + 1588)
frac_vocal = 1 - frac_semg
frac_semg /= 2
frac_vocal /= 2
frac_librispeech = 0.5
# TODO: should sweep librispeech ratios...
batch_class_proportions = np.array([frac_semg, frac_vocal, frac_librispeech])
latest_epoch = -1
matmul_tf32 = True


@app.command()
def update_configs(
    constant_offset_sd_cli: float = typer.Option(0, "--constant-offset-sd"),
    white_noise_sd_cli: float = typer.Option(0, "--white-noise-sd"),
    learning_rate_cli: float = typer.Option(3e-4, "--learning-rate"),
    debug_cli: bool = typer.Option(False, "--debug/--no-debug"),
    phonemes_cli: bool = typer.Option(False, "--phonemes/--no-phonemes"),
    use_dtw_cli: bool = typer.Option(use_dtw, "--dtw/--no-dtw"),
    use_crossCon_cli: bool = typer.Option(use_crossCon, "--crossCon/--no-crossCon"),
    use_supTcon_cli: bool = typer.Option(use_supTcon, "--supTcon/--no-supTcon"),
    grad_accum_cli: int = typer.Option(grad_accum, "--grad-accum"),
    precision_cli: str = typer.Option(precision, "--precision"),
    logger_level_cli: str = typer.Option("WARNING", "--logger-level"),
    base_bz_cli: int = typer.Option(base_bz, "--base-bz"),
    val_bz_cli: int = typer.Option(val_bz, "--val-bz"),
    max_len_cli: int = typer.Option(max_len, "--max-len"),
    seqlen_cli: int = typer.Option(seqlen, "--seqlen"),
    n_epochs_cli: int = typer.Option(n_epochs, "--n-epochs"),
    run_id_cli: str = typer.Option(run_id, "--run-id"),
    ckpt_path_cli: str = typer.Option(ckpt_path, "--ckpt-path"),
    audio_lambda_cli: float = typer.Option(audio_lambda, "--audio-lambda"),
    emg_lambda_cli: float = typer.Option(emg_lambda, "--emg-lambda"),
    frac_semg_cli: float = typer.Option(frac_semg, "--frac-semg"),
    frac_vocal_cli: float = typer.Option(frac_vocal, "--frac-vocal"),
    frac_librispeech_cli: float = typer.Option(frac_librispeech, "--frac-librispeech"),
    weight_decay_cli: float = typer.Option(weight_decay, "--weight-decay"),
    matmul_tf32_cli: bool = typer.Option(matmul_tf32, "--matmul-tf32/--no-matmul-tf32"),
    latent_affine_cli: bool = typer.Option(
        latent_affine, "--latent-affine/--no-latent-affine"
    ),
    # devices_cli: str = typer.Option(devices, "--devices"),
):
    """Update configurations with command-line values."""
    global constant_offset_sd, white_noise_sd, DEBUG, grad_accum, matmul_tf32
    global precision, logger_level, base_bz, val_bz, max_len, seqlen, n_epochs
    global learning_rate, devices, togglePhones, use_dtw, use_crossCon, use_supTcon
    global audio_lambda, latent_affine, weight_decay, run_id, ckpt_path, latest_epoch
    global emg_lambda, frac_semg, frac_vocal, frac_librispeech, batch_class_proportions

    # devices = devices_cli
    # try:
    #     devices = int(devices) # eg "2" -> 2
    # except:
    #     pass
    use_dtw = use_dtw_cli
    use_crossCon = use_crossCon_cli
    use_supTcon = use_supTcon_cli
    togglePhones = phonemes_cli
    learning_rate = learning_rate_cli
    constant_offset_sd = constant_offset_sd_cli
    white_noise_sd = white_noise_sd_cli
    DEBUG = debug_cli
    run_id = run_id_cli
    grad_accum = grad_accum_cli
    precision = precision_cli
    logger_level = getattr(logging, logger_level_cli.upper())
    base_bz = base_bz_cli
    val_bz = val_bz_cli
    max_len = max_len_cli
    seqlen = seqlen_cli
    audio_lambda = audio_lambda_cli
    emg_lambda = emg_lambda_cli
    latent_affine = latent_affine_cli
    weight_decay = weight_decay_cli
    ckpt_path = ckpt_path_cli
    n_epochs = n_epochs_cli
    matmul_tf32 = matmul_tf32_cli

    if (
        frac_semg != frac_semg_cli
        or frac_vocal != frac_vocal_cli
        or frac_librispeech != frac_librispeech_cli
    ):
        batch_class_proportions = np.array(
            [frac_semg_cli, frac_vocal_cli, frac_librispeech_cli]
        )
        print(f"batch_class_proportions: {batch_class_proportions}")

    print("Updated configurations using command-line arguments.")


torch.backends.cuda.matmul.allow_tf32 = matmul_tf32  # false by default

if __name__ == "__main__" and not in_notebook():
    try:
        app()
    except SystemExit as e:
        pass

if ckpt_path != "":
    raise NotImplementedError("TODO: implement output_directory for ckpt_path")

if run_id != "":
    MANUAL_RESUME = True
else:
    MANUAL_RESUME = False
    output_directory = os.path.join(scratch_directory, f"{isotime}_gaddy")

SLURM_REQUEUE = False
if not MANUAL_RESUME and ON_SHERLOCK:
    # output_directory = os.path.join(os.environ["SCRATCH"], f"{isotime}_gaddy")
    # see if this fixes pytorch lightning SLURM auto-resume
    output_directory = os.path.join(os.environ["SCRATCH"], os.environ["SLURM_JOB_ID"])
    if os.path.exists(output_directory):
        SLURM_REQUEUE = True
        run_id_file = os.path.join(output_directory, "run_id.txt")
        with open(run_id_file, "r") as file:
            run_id = file.read().strip()
        print(f"SLURM requeue detected, resuming run_id={run_id}")

if run_id != "":
    print("momentarily opening run in read-only mode to get hyperparams")
    run = get_neptune_run(run_id, project="neuro/Gaddy")
    hparams = nep_get(run, "training/hyperparams")
    print("Ignoring any command line flags and using hparams: ", hparams)
    max_len = hparams["max_len"]
    togglePhones = hparams["togglePhones"]
    output_directory = nep_get(run, "output_directory")
    ckpt_path, latest_epoch = get_last_ckpt(output_directory)
    batch_class_proportions = string_to_np_array(hparams["batch_class_proportions"])


# needed for using CachedDataset
emg_datamodule = EMGDataModule(
    data_dir,
    togglePhones,
    normalizers_file,
    max_len=max_len,
    collate_fn=collate_gaddy_or_speech,
    pin_memory=(not DEBUG),
    batch_size=val_bz,
)
emg_train = emg_datamodule.train

mfcc_norm, emg_norm = pickle.load(open(normalizers_file, "rb"))

if NUM_GPUS > 1:
    strategy = DDPStrategy(gradient_as_bucket_view=True, find_unused_parameters=True)
elif NUM_GPUS == 1:
    strategy = "auto"
else:
    strategy = "auto"

devices = NUM_GPUS


logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logger_level,
    format="%(message)s",
    force=True,
)

logging.debug("DEBUG mode")
if not log_neptune:
    logging.warning("not logging to neptune")
##
if NUM_GPUS > 1:
    num_workers = 0  # nccl backend doesn't support num_workers>0
    rank_key = "RANK" if "RANK" in os.environ else "LOCAL_RANK"
    bz = base_bz * NUM_GPUS
    if rank_key not in os.environ:
        rank = 0
    else:
        rank = int(os.environ[rank_key])
    logging.info(f"SETTING CUDA DEVICE ON RANK: {rank}")

    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    # we cannot call DistributedSampler before pytorch lightning trainer.fit() is called,
    # or we get this error:
    # RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.
    # always include at least one example of class 0 (silent EMG & parallel EMG & parallel Audio) in batch
    # always include at least one example of class 1 (EMG & Audio) in batch
    # TrainBatchSampler = partial(Distributed`SizeAwareStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS, max_len=max_len//8, always_include_class=1)
    # TrainBatchSampler = partial(DistributedStratifiedBatchSampler,
    #     num_replicas=NUM_GPUS)
    TrainBatchSampler = partial(
        BalancedBinPackingBatchSampler,
        num_replicas=NUM_GPUS,
        # in emg_speech_dset_lengths we divide length by 8
        max_len=max_len // 8,
        always_include_class=[0],
    )
    ValSampler = lambda: DistributedSampler(
        emg_datamodule.val, shuffle=False, num_replicas=NUM_GPUS
    )
    TestSampler = lambda: DistributedSampler(
        emg_datamodule.test, shuffle=False, num_replicas=NUM_GPUS
    )
else:
    # TrainBatchSampler = SizeAwareStratifiedBatchSampler
    TrainBatchSampler = partial(
        BalancedBinPackingBatchSampler,
        num_replicas=NUM_GPUS,
        # in emg_speech_dset_lengths we divide length by 8
        max_len=max_len // 8,
        always_include_class=[0],
    )
    # num_workers=32
    num_workers = 0  # prob better now that we're caching
    bz = base_bz
    ValSampler = None
    TestSampler = None
    rank = 0

if rank == 0:
    os.makedirs(output_directory, exist_ok=True)


# must run 2023-07-17_cache_dataset_with_attrs_.py first
librispeech_train_cache = os.path.join(
    librispeech_directory, "2024-01-23_librispeech_noleak_train_phoneme_cache"
)
librispeech_val_cache = os.path.join(
    librispeech_directory, "2024-01-23_librispeech_noleak_val_phoneme_cache"
)
librispeech_test_cache = os.path.join(
    librispeech_directory, "2024-01-23_librispeech_noleak_test_phoneme_cache"
)

speech_val = cache_dataset(
    librispeech_val_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)()
speech_train = cache_dataset(
    librispeech_train_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)()
speech_test = cache_dataset(
    librispeech_test_cache,
    LibrispeechDataset,
    per_index_cache,
    remove_attrs_before_save=["dataset"],
)()

datamodule = EMGAndSpeechModule(
    emg_datamodule.train,
    emg_datamodule.val,
    emg_datamodule.test,
    speech_train,
    speech_val,
    speech_test,
    bz=bz,
    val_bz=val_bz,
    num_replicas=NUM_GPUS,
    pin_memory=(not DEBUG),
    num_workers=num_workers,
    TrainBatchSampler=TrainBatchSampler,
    ValSampler=ValSampler,
    TestSampler=TestSampler,
    batch_class_proportions=batch_class_proportions,
)
steps_per_epoch = len(datamodule.TrainBatchSampler) // grad_accum

# assert steps_per_epoch > 100, "too few steps per epoch"
# assert steps_per_epoch < 1000, "too many steps per epoch"
##

os.makedirs(output_directory, exist_ok=True)

if MANUAL_RESUME or SLURM_REQUEUE:
    config = MONAConfig(**hparams)
    text_transform = TextTransform(togglePhones=config.togglePhones)
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1  # +1 for CTC blank token ( i think? )
else:
    text_transform = TextTransform(togglePhones=togglePhones)
    n_chars = len(text_transform.chars)
    num_outs = n_chars + 1  # +1 for CTC blank token ( i think? )
    config = MONAConfig(
        steps_per_epoch=steps_per_epoch,
        lm_directory=lm_directory,
        num_outs=num_outs,
        precision=precision,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        audio_lambda=audio_lambda,
        emg_lambda=emg_lambda,
        # neural_input_features=datamodule.train.n_features,
        neural_input_features=1,
        seqlen=seqlen,
        max_len=max_len,
        batch_size=base_bz,
        white_noise_sd=white_noise_sd,
        constant_offset_sd=constant_offset_sd,
        num_train_epochs=n_epochs,
        togglePhones=togglePhones,
        use_dtw=use_dtw,
        use_crossCon=use_crossCon,
        use_supTcon=use_supTcon,
        batch_class_proportions=batch_class_proportions,
        # d_inner=8,
        # d_model=8,
        fixed_length=True,
        weight_decay=weight_decay,
        latent_affine=latent_affine,
    )

model = MONA(config, text_transform, no_neural=True)

##
logging.info("made model")

callbacks = []

if config.emg_lambda > 0:
    monitor = "val/silent_emg_wer"
    save_top_k = 10
else:
    monitor = None  # save most recent epochs
    save_top_k = 1

if log_neptune:
    # need to store credentials in your shell env
    nep_key = os.environ["NEPTUNE_API_TOKEN"]
    neptune_kwargs = {
        "project": "neuro/Gaddy",
        "name": model.__class__.__name__,
        "tags": [
            model.__class__.__name__,
            isotime,
            f"fp{config.precision}",
        ],
    }
    if MANUAL_RESUME or SLURM_REQUEUE:
        print(f"==== RESUMING RUN FROM EPOCH {latest_epoch} ====")
        neptune_logger = NeptuneLogger(
            run=neptune.init_run(
                with_id=run_id,
                api_token=os.environ["NEPTUNE_API_TOKEN"],
                **neptune_kwargs,
            ),
            log_model_checkpoints=False,
        )
    else:
        neptune_logger = NeptuneLogger(
            api_key=nep_key, **neptune_kwargs, log_model_checkpoints=False
        )
        # TODO: get run_id from neptune_logger
        run_id = neptune_logger.experiment.fetch()["sys"]["id"]
        neptune_logger.log_hyperparams(vars(config))
        neptune_logger.experiment["isotime"] = isotime
        neptune_logger.experiment["hostname"] = hostname.stdout.decode().strip()
        neptune_logger.experiment["output_directory"] = output_directory
        if "SLURM_JOB_ID" in os.environ:
            neptune_logger.experiment["SLURM_JOB_ID"] = os.environ["SLURM_JOB_ID"]

    # save run_id for SLURM requeue
    run_id_file = os.path.join(output_directory, "run_id.txt")
    with open(run_id_file, "w") as file:
        file.write(run_id)

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val/emg_ctc_loss",
    #     mode="min",
    #     dirpath=output_directory,
    #     save_top_k=10,
    #     filename=model.__class__.__name__ + "-{epoch:02d}-{val/emg_ctc_loss:.3f}",
    # )
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode="min",
        dirpath=output_directory,
        save_top_k=save_top_k,  # TODO: try averaging weights afterwards to see if improve WER..?
        save_last=True,
        filename=model.__class__.__name__ + "-{epoch:02d}-{val/silent_emg_wer:.3f}",
    )
    callbacks.extend(
        [
            checkpoint_callback,
            pl.callbacks.LearningRateMonitor(logging_interval="epoch"),
            # pl.callbacks.LearningRateMonitor(logging_interval="step"), # good for troubleshooting warmup
        ]
    )
else:
    neptune_logger = None

trainer = pl.Trainer(
    max_epochs=config.num_train_epochs,
    devices=devices,
    accelerator="gpu",
    accumulate_grad_batches=config.gradient_accumulation_steps,
    # accelerator="cpu",
    gradient_clip_val=1,  # was 0.5 for best 26.x% run, gaddy used 10, llama 2 uses 1.0
    logger=neptune_logger,
    default_root_dir=output_directory,
    callbacks=callbacks,
    precision=config.precision,
    limit_train_batches=limit_train_batches,
    limit_val_batches=limit_val_batches,
    # strategy=strategy,
    # use_distributed_sampler=True,
    # use_distributed_sampler=False, # we need to make a custom distributed sampler
    # num_sanity_val_steps=num_sanity_val_steps,
    sync_batchnorm=True,
    strategy=strategy,
    # strategy='fsdp', # errors on CTC loss being used on half-precision.
    # also model only ~250MB of params, so fsdp may be overkill
    # check_val_every_n_epoch=10 # should give speedup of ~30% since validation is bz=1
    num_sanity_val_steps=0,
    # https://lightning.ai/docs/pytorch/stable/debug/debugging_intermediate.html#detect-autograd-anomalies
    # detect_anomaly=True # slooooow
    # don't requeue jobs on SLURM if killed (e.g. on owners partition)
    # plugins=[SLURMEnvironment(auto_requeue=False)],
    # actually, let's not use this since it's not clear if it works with neptune
    # instead, we'll use a signal handler to resubmit the job
    # so we will submit job with name "interactive" to bypass
    # pytorch lightning SLURM
)

# commented out to use pytorch lightning SLURM auto-resume instead
# def resubmit_job(sig, frame):
#     """
#     Function to resubmit the job with the same run_id
#     """
#     global run_id
#     if run_id:
#         print(f"Got SIGUSR1. Resubmitting job with run_id: {run_id}")
#         subprocess.run(
#             [
#                 "sbatch",
#                 "/home/users/tbenst/code/silent_speech/notebooks/tyler/2024-01-15_icml_models.py",
#                 "--run-id",
#                 run_id,
#             ]
#         )
#     else:
#         print("No run_id provided, cannot resubmit job.")


# if "SLURM_JOB_ID" in os.environ:
#     print("Job running under SLURM, setting up signal handler for SIGUSR1.")
#     signal.signal(signal.SIGUSR1, resubmit_job)

##
logging.info("about to fit")
print(f"Sanity check: {len(datamodule.train)} training samples")
print(f"Sanity check: {len(datamodule.train_dataloader())} training batches")
# epoch of 242 if only train...
if MANUAL_RESUME or SLURM_REQUEUE:
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
else:
    trainer.fit(model, datamodule=datamodule)

if log_neptune:
    final_ckpt_path = os.path.join(
        output_directory, f"finished-training_epoch={model.current_epoch}.ckpt"
    )
    trainer.save_checkpoint(final_ckpt_path)
    print(f"saved checkpoint to {final_ckpt_path}")

##
exit(0)
##
dl = datamodule.train_dataloader()
# dl = datamodule.val_dataloader()
for b in dl:
    break
b
##
split_batch_into_emg_neural_audio(b)
##
datamodule.setup()
##
tbs = datamodule.TrainBatchSampler
len(tbs)
##
# print(len(list(tbs.iter_batches(0))))
print(len(list(tbs.iter_batches(1))))
##
##
datamodule.TrainBatchSampler.set_epoch(0)
dl = datamodule.train_dataloader()
for b1, bat1 in enumerate(dl):
    pass
##
datamodule.TrainBatchSampler.set_epoch(1)
dl = datamodule.train_dataloader()
for b2, bat2 in enumerate(dl):
    pass
assert b1 == b2, f"b1={b1}, b2={b2}"

##
for b, batch in enumerate(dl):
    # check if we have a paired_idx as expected ("class 1")
    audio_only = batch["audio_only"]
    silent = batch["silent"]
    paired = np.logical_and(np.logical_not(audio_only), np.logical_not(silent))
    assert np.sum(paired) > 0, f"no paired examples in batch {b}"


##
bins1 = list(datamodule.TrainBatchSampler.iter_batches(0))
bins2 = list(datamodule.TrainBatchSampler.iter_batches(0))
bins1[0], bins2[0]
##
[datamodule.TrainBatchSampler.classes[i] for i in bins1[519]]
##
[datamodule.TrainBatchSampler.classes[i] for i in bins[382]]
##
datamodule.setup()
dl = datamodule.val_dataloader()
batches = []
for b, batch in enumerate(dl):
    batches.append(batch)
    if b > 1:
        break
##
batches[0]["text"], batches[1]["text"]
##
for k, v in batches[0].items():
    print(k, len(v))
##

batch = batches[1]


def print_pairs(x, y, text="x\t"):
    print(f"{text}\tpairs with")
    for e, ye in zip([e.sum() for e in x], [ye.sum() for ye in y]):
        print(f"{e:.3f}\t\t{ye}")


print_pairs(batch["raw_emg"], batch["text_int"], "silent emg")
print("\n")
emg_tup, neural_tup, audio_tup, idxs = split_batch_into_emg_neural_audio(batch)
emg, length_emg, emg_phonemes, y_length_emg, y_emg = emg_tup
neural, length_neural, neural_phonemes, y_length_neural, y_neural = neural_tup
audio, length_audio, audio_phonemes, y_length_audio, y_audio = audio_tup
(
    paired_emg_idx,
    paired_audio_idx,
    silent_emg_idx,
    parallel_emg_idx,
    parallel_audio_idx,
) = idxs
# check if emg & text still match up

print_pairs(emg, y_emg, "silent emg")
##
list(zip(np.arange(3), np.arange(5)))
##
td = EMGDataset(
    base_dir=None,
    dev=True,
    test=False,
    returnRaw=True,
    togglePhones=False,
    normalizers_file=normalizers_file,
)


##
td[2]
##
td.voiced_data_locations.keys()
##
