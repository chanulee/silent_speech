##
import numpy as np
import sys, os
import jiwer, typer
import tiktoken, asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm.auto import tqdm

# horrible hack to get around this repo not being a proper python package
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.getcwd()))
sys.path.append(SCRIPT_DIR)
from data_utils import TextTransform
from pqdm.threads import pqdm
from helpers import calc_wer, get_neptune_run, nep_get, get_run_type
from data_utils import in_notebook
from time import sleep
from collections import OrderedDict
import logging, json

if in_notebook():
    # allow for nested event loops in jupyter notebooks
    import nest_asyncio

    nest_asyncio.apply()

logging.basicConfig(level=logging.INFO)

text_transform = TextTransform(togglePhones=False)


def clean_transcripts(transcripts):
    transcripts = list(map(text_transform.clean_text, transcripts))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    transcripts = transformation(transcripts)
    for i in range(len(transcripts)):
        if transcripts[i].startswith("the correct transcription is "):
            transcripts[i] = transcripts[i].replace("the correct transcription is ", "")

    return transcripts


def create_rescore_msg(predictions):
    rescore_msg = "\n".join([p for p in predictions])
    return rescore_msg


def get_labels_preds(run_id):
    run = get_neptune_run(run_id, project="neuro/Gaddy")
    output_directory = nep_get(run, "output_directory")
    hparams = nep_get(run, "training/hyperparams")
    togglePhones = hparams["togglePhones"]
    text_transform = TextTransform(togglePhones=togglePhones)
    num_beams = 5000
    path = os.path.join(output_directory, f"2024-01-28_top100_{num_beams}beams.npz")
    # path = os.path.join(output_directory, f"2024-01-27_top100_{num_beams}beams.npz")
    npz = np.load(path, allow_pickle=True)
    run_type = get_run_type(hparams)

    assert "emg_silent_test" in npz["dataset"]
    silent_idxs = np.where(npz["dataset"] == "emg_silent_test")[0]
    silent_predictions = npz["predictions"][silent_idxs]
    silent_beam_scores = npz["beam_scores"][silent_idxs]
    silent_labels = npz["sentences"][silent_idxs]
    non_zero = np.where(silent_labels != "")[0]
    silent_predictions = silent_predictions[non_zero]
    silent_beam_scores = silent_beam_scores[non_zero]
    silent_labels = silent_labels[non_zero]

    vocal_idxs = np.where(npz["dataset"] == "emg_vocal_test")[0]
    vocal_predictions = npz["predictions"][vocal_idxs]
    vocal_beam_scores = npz["beam_scores"][vocal_idxs]
    vocal_labels = npz["sentences"][vocal_idxs]
    non_zero = np.where(vocal_labels != "")[0]
    vocal_predictions = vocal_predictions[non_zero]
    vocal_beam_scores = vocal_beam_scores[non_zero]
    vocal_labels = vocal_labels[non_zero]

    audio_idxs = np.where(npz["dataset"] == "audio_test")[0]
    audio_predictions = npz["predictions"][audio_idxs]
    audio_beam_scores = npz["beam_scores"][audio_idxs]
    audio_labels = npz["sentences"][audio_idxs]
    non_zero = np.where(audio_labels != "")[0]
    audio_predictions = audio_predictions[non_zero]
    audio_beam_scores = audio_beam_scores[non_zero]
    audio_labels = audio_labels[non_zero]

    librispeech_idxs = np.where(npz["dataset"] == "librispeech_test")[0]
    librispeech_predictions = npz["predictions"][librispeech_idxs]
    librispeech_beam_scores = npz["beam_scores"][librispeech_idxs]
    librispeech_labels = npz["sentences"][librispeech_idxs]
    non_zero = np.where(librispeech_labels != "")[0]
    librispeech_beam_scores = librispeech_beam_scores[non_zero]
    librispeech_labels = librispeech_labels[non_zero]
    librispeech_predictions = librispeech_predictions[non_zero]
    return (
        silent_predictions,
        silent_labels,
        vocal_predictions,
        vocal_labels,
        audio_predictions,
        audio_labels,
        librispeech_predictions,
        librispeech_labels,
    )


def completion_coroutine(sys_msg, user_msg, model="gpt-3.5-turbo-16k-0613"):
    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        seed=20240130,
    )


async def gather_completions(coroutines, n_jobs=3):
    msgs = []
    for i in tqdm(range(0, len(coroutines), n_jobs), desc="Processing completions"):
        batch = coroutines[i : i + n_jobs]
        responses = await asyncio.gather(*batch)
        msgs.extend([response.choices[0].message.content for response in responses])
    return msgs


def batch_completions(predictions, sys_msg, n_jobs=3, model="gpt-3.5-turbo-16k-0613"):
    coroutines = []
    for pred in predictions:
        rescore_msg = create_rescore_msg(pred)
        cc = completion_coroutine(sys_msg, rescore_msg, model=model)
        coroutines.append(cc)
        sleep(0.05)
    # run the asynchronous gathering function
    return asyncio.run(gather_completions(coroutines, n_jobs=n_jobs))


DIRECT_SYS_MSG = """Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Choose the transcription that is most accurate, ensuring it is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond with the chosen transcription only, without any introductory text."""


def cor_clean_transcripts(transcripts):
    ret = []
    for transcript in transcripts:
        # split on 'TRANSCRIPT: '
        t = transcript.split("TRANSCRIPT: ")[-1]
        # remove leading and trailing whitespace
        t = t.strip()
        ret.append(t)
    ret = list(map(text_transform.clean_text, ret))
    transformation = jiwer.Compose([jiwer.RemovePunctuation(), jiwer.ToLowerCase()])
    ret = transformation(ret)
    return ret


def chain_of_reasoning_LISA(preds, labels, model):
    sys_msg = "Your task is to perform automatic speech recognition. Below are multiple candidate transcriptions, listed from most likely to least likely. Begin your response with a Chain of Reasoning, explaining your analysis and decision-making process in choosing the most accurate transcription. After your analysis, clearly indicate your final choice with the cue 'TRANSCRIPT: '. Ensure the transcription you choose is contextually and grammatically correct. Focus on key differences in the options that change the meaning or correctness. Avoid selections with repetitive or nonsensical phrases. In cases of ambiguity, select the option that is most coherent and contextually sound. Respond first with your reasoning, followed by 'TRANSCRIPT: ' and then the chosen transcription."

    lisa_predictions = batch_completions(
        [s[:10] for s in preds], sys_msg, model=model, n_jobs=5
    )

    bad_performance = []
    for i, text in enumerate(lisa_predictions):
        if "TRANSCRIPT: " not in text:
            bad_performance.append(i)

    # we give Chain of Reasoning more than it's fair share of leeway here
    # other approaches are better so this is really just for didactic purposes
    # to show that it's not a good idea for this task
    assert len(bad_performance) < 10  # allow 5% task failure rate
    lisa_filt_predictions = [
        p for i, p in enumerate(lisa_predictions) if i not in bad_performance
    ]
    filt_labels = [l for i, l in enumerate(labels) if i not in bad_performance]
    lisa_wer = calc_wer(
        cor_clean_transcripts(lisa_filt_predictions), filt_labels, text_transform
    )
    typer.echo(
        f"{run_id} WER with Chain of Reasoning ({model}) and excluding {bad_performance} lazy responses: {lisa_wer * 100:.2f}%"
    )

    return lisa_predictions


def direct_LISA(preds, labels, model, N=10):
    assert len(preds) == len(labels), f"{len(preds)=} {len(labels)=}"
    lisa_predictions = batch_completions(
        [s[:N] for s in preds], DIRECT_SYS_MSG, model=model, n_jobs=5
    )

    try:
        lisa_wer = calc_wer(
            cor_clean_transcripts(lisa_predictions), labels, text_transform
        )
        typer.echo(f"{run_id} WER with direct {N=} ({model}): {lisa_wer * 100:.2f}%")
    except Exception as e:
        typer.echo(f"Error calculating WER: {e}")

    return lisa_predictions


def save_finetuning_dset(preds, labels, save_path):
    dset = [(create_rescore_msg(p), l) for p, l in zip(preds, labels)]

    # Convert to JSONL format
    jsonl_data = []
    for user_msg, assistant_msg in dset:
        jsonl_data.append(
            {
                "messages": [
                    {"role": "system", "content": DIRECT_SYS_MSG},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
            }
        )

    # Save as a JSONL file
    jsonl_path = save_path
    with open(jsonl_path, "w") as f:
        for entry in jsonl_data:
            json.dump(entry, f)
            f.write("\n")

    return jsonl_path


##

client = AsyncOpenAI(
    # max_retries=100,
    # timeout=15,
)

run_silent_preds = OrderedDict()
run_silent_labels = OrderedDict()
run_vocal_preds = OrderedDict()
run_vocal_labels = OrderedDict()
run_audio_preds = OrderedDict()
run_audio_labels = OrderedDict()
run_librispeech_preds = OrderedDict()
run_librispeech_labels = OrderedDict()

# 2024-01-30 ensemble10
# includes 5 crossCon and 5 crossCon + DTW runs
# a priori chosen for best val performance
# run_ids = np.array(
#     [
#         "GAD-984",
#         "GAD-986",
#         "GAD-987",
#         "GAD-988",
#         "GAD-983",
#         "GAD-940",
#         "GAD-938",
#         "GAD-941",
#         "GAD-937",
#         "GAD-939",
#     ]
# )

run_ids = np.array(
    [
        "GAD-984",  # crossCon + DTW (256k)
        "GAD-992",  # crossCon + DTW (256k)
        "GAD-986",  # crossCon + DTW (256k)
        "GAD-996",  # crossCon + DTW (256k)
        "GAD-993",  # crossCon + DTW (256k)
        "GAD-987",  # crossCon + DTW (256k)
        "GAD-988",  # crossCon + DTW (256k)
        "GAD-940",  # crossCon (256k)
        "GAD-995",  # crossCon + DTW (256k)
        "GAD-983",  # crossCon + DTW (256k) # 2024-01-31 ensemble10
        # "GAD-938",  # crossCon (256k)
        # "GAD-994",  # crossCon + DTW (256k)
        # "GAD-941",  # crossCon (256k)
        # "GAD-937",  # crossCon (256k)
        # "GAD-939",  # crossCon (256k) 2024-01-31 ensemble15
    ]
)

for r in run_ids:
    (
        silent_pred,
        silent_labels,
        vocal_pred,
        vocal_labels,
        audio_pred,
        audio_labels,
        librispeech_pred,
        librispeech_labels,
    ) = get_labels_preds(r)
    run_silent_preds[r] = silent_pred
    run_silent_labels[r] = silent_labels
    run_vocal_preds[r] = vocal_pred
    run_vocal_labels[r] = vocal_labels
    run_audio_preds[r] = audio_pred
    run_audio_labels[r] = audio_labels
    run_librispeech_preds[r] = librispeech_pred
    run_librispeech_labels[r] = librispeech_labels

##
# Calculate WER for best MONA only
run_id = "GAD-984"  # (20.63% validation WER)
silent_pred = run_silent_preds[run_id]
vocal_pred = run_vocal_preds[run_id]
audio_pred = run_audio_preds[run_id]
librispeech_pred = run_librispeech_preds[run_id]
silent_best_pred = np.array([p[0] for p in silent_pred])
vocal_best_pred = np.array([p[0] for p in vocal_pred])
audio_best_pred = np.array([p[0] for p in audio_pred])
librispeech_best_pred = np.array([p[0] for p in librispeech_pred])
silent_wer = calc_wer(silent_best_pred, silent_labels, text_transform)
typer.echo(f"==== {run_id} ====")
typer.echo(f"silent EMG WER: {silent_wer * 100:.2f}%")
vocal_wer = calc_wer(vocal_best_pred, vocal_labels, text_transform)
typer.echo(f"vocal EMG WER: {vocal_wer * 100:.2f}%")
audio_wer = calc_wer(audio_best_pred, audio_labels, text_transform)
typer.echo(f"audio WER: {audio_wer * 100:.2f}%")
librispeech_wer = calc_wer(librispeech_best_pred, librispeech_labels, text_transform)
typer.echo(f"librispeech WER: {librispeech_wer * 100:.2f}%")
##
#### Ensemble top1 ####
# for each entry, stack the top pred for each run
ensemble_top1_silent_preds = []
for i in range(len(silent_pred)):
    ensemble_top1_silent_preds.append(
        np.stack([p[i][0] for p in run_silent_preds.values()])
    )

ensemble_top1_vocal_preds = []
for i in range(len(vocal_pred)):
    ensemble_top1_vocal_preds.append(
        np.stack([p[i][0] for p in run_vocal_preds.values()])
    )

ensemble_top1_audio_preds = []
for i in range(len(audio_pred)):
    ensemble_top1_audio_preds.append(
        np.stack([p[i][0] for p in run_audio_preds.values()])
    )

ensemble_top1_librispeech_preds = []
for i in range(len(librispeech_pred)):
    ensemble_top1_librispeech_preds.append(
        np.stack([p[i][0] for p in run_librispeech_preds.values()])
    )

#### finetuned ensemble top1####
# 12.21% WER
lisa_predictions = direct_LISA(
    ensemble_top1_silent_preds,
    silent_labels,
    # see 2024-01-31_icml_LISA.py for description of models
    # "ft:gpt-3.5-turbo-1106:personal::8mordMqf"  # a priori chosen for best performance
    # "gpt-3.5-turbo-16k-0613"
    # "ft:gpt-3.5-turbo-1106:personal::8nHXtaiS", # actually the best
    # "ft:gpt-3.5-turbo-1106:personal::8nHNRM88", # also better
    "ft:gpt-3.5-turbo-1106:personal::8n2fDmAy",
    N=15,
)


##
# 3.65%
lisa_predictions = direct_LISA(
    ensemble_top1_vocal_preds,
    silent_labels,
    "ft:gpt-3.5-turbo-1106:personal::8mordMqf",
)

##
# 2.61% WER
lisa_predictions = direct_LISA(
    ensemble_top1_audio_preds,
    silent_labels,
    "ft:gpt-3.5-turbo-1106:personal::8mordMqf",
)

##
# 5.68% WER
lisa_predictions = direct_LISA(
    ensemble_top1_librispeech_preds,
    librispeech_labels,
    "ft:gpt-3.5-turbo-1106:personal::8mordMqf",
)

##
from scipy.stats import spearmanr
import numpy as np

#
val_ranks = np.array([i for i in range(1, 11)])  # val rankings
test_ranks = np.array([9, 4, 5, 6, 1, 2, 3, 7, 10, 8])  # test rankings

# Calculate Spearman's rank correlation
rho, pval = spearmanr(val_ranks, test_ranks)

print(f"Spearman's rho: {rho:.3f}")
print(f"p-value: {pval:.3f}")

##
