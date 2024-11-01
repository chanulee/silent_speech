# Silent Speech
This repo is to understand and learn form Mona Lisa.  
Major difference between the Mona Lisa project is that we want to try this on Colab, and integrate ear EEG.

### Resources
- repo: https://github.com/Leoputera2407/silent_speech
- LibriSpeech: https://huggingface.co/datasets/openslr/librispeech_asr
- Gaddy Dataset: https://doi.org/10.5281/zenodo.4064408
- paper: https://arxiv.org/abs/2403.05583

## Approach 1: Breaking down [Paper reproduction](#paper-reproduction) 
0. run `notebooks/tyler/2023-07-17_cache_dataset_with_attrs_.py` [See How ↓](#0-2023-07-17_cache_dataset_with_attrs_py) or [view file](notebooks/tyler/2023-07-17_cache_dataset_with_attrs_.py)
1. run `notebooks/tyler/batch_model_sweep.sh` (`2024-01-15_icml_models.py`)
2. run `notebooks/tyler/2024-01-26_icml_pred.py`
3. run `notebooks/tyler/batch_beam_search.sh` (`2024-01-26_icml_beams.py`)
4. run `notebooks/tyler/2024-01-28_icml_figures.py`
5. run `notebooks/tyler/2024-01-31_icml_TEST.py`

## 0. 2023-07-17_cache_dataset_with_attrs_.py
[view file (modified version)](notebooks/tyler/2023-07-17_cache_dataset_with_attrs_.py) or [view file (original tyler version)](https://github.com/Leoputera2407/tyler_silent_speech/blob/main/notebooks/tyler/2023-07-17_cache_dataset_with_attrs_.py)

### Spotted problems & ways to solve it
- [x] The current code is based on SLURM, running on SHERLOCK
  - [x] ```ON_SHERLOCK = False```
- [x] Loading and cacheing the Libri Dataset ```librispeech_datasets = load_dataset("librispeech_asr")``` keeps timing out on colab
  - [ ] Testing them again with seperated block
  - [ ] Or is there any other way to do this by downloading the full dataset in prior to running this line, and then cache them seperately? (Given that downloading part is the reason of timing out / 'manual approach' block on colab)
- [ ] **What is ```sessions_dir```, ```scratch_directory``` and ```gaddy_dir``` supposed to be?**
```python
sessions_dir = "/data/magneto/"
scratch_directory = "/scratch"
gaddy_dir = "/scratch/GaddyPaper/"
```
  - [ ] and where is the root directory of this /data/magneto and /scratch?
- [ ] ```"librispeech-cache"``` automatically generates when load_dataset run?
- [ ] ```"librispeech-alignments"``` and ```alignment_dir```
  - [ ] Do MFA to generate this alignment - shown below in this readme [Montreal forced aligner ↓](#montreal-forced-aligner)
- [ ] While running ```from dataloaders import LibrispeechDataset, cache_dataset```, an error occured as the ```os.environ["SCRATCH"]``` was not set.
  - [ ] ```os.environ["SCRATCH"] = "/scratch"```
  - [ ] But tyler_silent_speech repo does not set this... why? Is it set already somewhere else?
  -

## Approach 2: Breaking down [Brain-to-text '24 reproduction](#brain-to-text-24-reproduction)  
1. Train 10 models of the [Pytorch NPTL baseline RNN](https://github.com/cffan/neural_seq_decoder)
2. Run beam search with the 5-gram model. The average validation WER should be approximatel 14.6%
3. run `notebooks/tyler/2024-02-13_wiilet_competition.py`. The validation WER of finetuned LISA should be approximately 13.7% without finetuning, or 10.2% with finetuning.

## 1. Pytorch NPTL baseline RNN
### Spotted problems & ways to solve it
- [ ] Train this model... with what?

---

# Achive of original Repo
## MONA LISA

This repository contains code for training Multimodal Orofacial Neural Audio (MONA) and Large Language
Model (LLM) Integrated Scoring Adjustment
(LISA). Together, MONA LISA sets a new state-of-the art for decoding silent speech, achieving 7.3% WER on validation data for open vocabulary.

[See the preprint on arxiv](https://arxiv.org/abs/2403.05583).

### Paper reproduction
First you will need to download the [Gaddy 2020 dataset](https://doi.org/10.5281/zenodo.4064408) Then, the following scripts can be modified and run in order on SLURM or a local machine. An individual model trains on one A100 for 24-48 hours depending on loss functions (supTcon increases train time by ~75%). The full model sweep as done in the paper trains 60 models.
0) run `notebooks/tyler/2023-07-17_cache_dataset_with_attrs_.py`
1) run `notebooks/tyler/batch_model_sweep.sh` (`2024-01-15_icml_models.py`)
2) run `notebooks/tyler/2024-01-26_icml_pred.py`
3) run `notebooks/tyler/batch_beam_search.sh` (`2024-01-26_icml_beams.py`)
4) run `notebooks/tyler/2024-01-28_icml_figures.py`
5) run `notebooks/tyler/2024-01-31_icml_TEST.py`

### Brain-to-text '24 reproduction
1) Train 10 models of the [Pytorch NPTL baseline RNN](https://github.com/cffan/neural_seq_decoder)
2) Run beam search with the 5-gram model. The average validation WER should be approximatel 14.6%
3) run `notebooks/tyler/2024-02-13_wiilet_competition.py`. The validation WER of finetuned LISA should be approximately 13.7% without finetuning, or 10.2% with finetuning.

The [final competition WER was 8.9%](https://eval.ai/web/challenges/challenge-page/2099/leaderboard/4944), which at time of writing is rank 1.

## Environment Setup

### alternate setup
First build the `environment.yml`. Then, 
```
> conda install libsndfile -c conda-forge
> 
> pip install jiwer torchaudio matplotlib scipy soundfile absl-py librosa numba unidecode praat-textgrids g2p_en einops opt_einsum hydra-core pytorch_lightning "neptune-client==0.16.18"
```


## Explanation of model outputs for CTC loss
For each timestep, the network predicts probability of each of 38 characters ('abcdefghijklmnopqrstuvwxyz0123456789|_'), where `|` is word boundary, and `_` is the "blank token". The blank token is used to separate repeat letters like "ll" in hello: `[h,h,e,l,l,_,l,o]`. It can optionally be inserted elsewhere too, like `__hhhh_eeee_llll_lllooo___`

### Example prediction


**Target text**: after breakfast instead of working i decided to walk down towards the common

Example model prediction (argmax last dim) of shape `(1821, 38)`:

`______________________________________________________________a__f___tt__eerr|||b__rr_eaaakk___ff____aa____ss_tt___________________||____a_nd__|_ssttt___eaa_dd_||ooff||ww___o_rr_____kk_____ii___nngg________________________||_____a____t__||_______c______i___d_____eedd__________||tt___o__||_w_____a______l_kkk____________________||______o______w__t______________|||t____oowwwaarrrdddsss____||thhee_|||c_____o___mm__mm___oo_nn___________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________`

Beam search gives, ' after breakfast and stead of working at cided to walk owt towards the common ', which here is same as result from "best path decoding" (argmax), but in theory could be different since sums probability of multiple alignments and is therefore more accurate.


### Montreal forced aligner
Instructions for getting phoneme alignments


https://montreal-forced-aligner.readthedocs.io/en/latest/first_steps/index.html#first-steps-align-pretrained

```
> conda create -n mfa -c conda-forge montreal-forced-aligner
> mfa model download acoustic english_us_arpa
> mfa model download dictionary english_us_arpa
> mfa validate --single_speaker -j 32 /data/data/T12_data/synthetic_audio/TTS english_us_arpa english_us_arpa
> mfa model download g2p english_us_arpa
> mfa g2p --single_speaker /data/data/T12_data/synthetic_audio/TTS english_us_arpa ~/Documents/MFA/TTS/oovs_found_english_us_arpa.txt --dictionary_path english_us_arpa
> mfa model add_words english_us_arpa ~/mfa_data/g2pped_oovs.txt
> mfa adapt --single_speaker -j 32 /data/data/T12_data/synthetic_audio/TTS english_us_arpa english_us_arpa /data/data/T12_data/synthetic_audio/adapted_bark_english_us_arpa
> mfa validate --single_speaker -j 32 /data/data/T12_data/synthetic_audio/TTS english_us_arpa english_us_arpa
# ensure no OOV (I had to manually correct a transcript due to a `{`)
> mfa adapt --single_speaker -j 32 --output_directory /data/data/T12_data/synthetic_audio/TTS /data/data/T12_data/synthetic_audio/TTS english_us_arpa english_us_arpa /data/data/T12_data/synthetic_audio/adapted_bark_english_us_arpa

### misc

Fast transfer of cache on sherlock to local NVME
```
cd $MAG/librispeech
find . -type f | parallel -j 16 rsync -avPR {} $LOCAL_SCRATCH/librispeech/
```
find . -type f | parallel -j 16 rsync -avPR {} $SCRATCH/librispeech/
