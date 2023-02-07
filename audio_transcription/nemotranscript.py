# install Cython
# install nemo from source
# install all other dependencies
#
import os
import json
import shutil
import uuid
import tempfile
import subprocess
import re
import time
import traceback
import gradio as gr
import pytube as pt
import nemo.collections.asr as nemo_asr
import torch
import speech_to_text_buffered_infer_ctc as buffered_ctc
import speech_to_text_buffered_infer_rnnt as buffered_rnnt
from nemo.utils import logging

# Set NeMo cache dir as /tmp
from nemo import constants

os.environ[constants.NEMO_ENV_CACHE_DIR] = "/tmp/nemo/"


SAMPLE_RATE = 16000  # Default sample rate for ASR
BUFFERED_INFERENCE_DURATION_THRESHOLD = 60.0  # 60 second and above will require chunked inference.
CHUNK_LEN_IN_SEC = 20.0  # Chunk size
BUFFER_LEN_IN_SEC = 30.0  # Total buffer size

TITLE = "NeMo ASR Inference on Hugging Face"
DESCRIPTION = "Demo of all languages supported by NeMo ASR"
DEFAULT_EN_MODEL = "nvidia/stt_en_conformer_transducer_xlarge"
DEFAULT_BUFFERED_EN_MODEL = "nvidia/stt_en_conformer_transducer_large"

# Pre-download and cache the model in disk space
logging.setLevel(logging.ERROR)
tmp_model = nemo_asr.models.ASRModel.from_pretrained(DEFAULT_BUFFERED_EN_MODEL, map_location='cpu')
del tmp_model
logging.setLevel(logging.INFO)

MARKDOWN = f"""
# {TITLE}
## {DESCRIPTION}
"""

CSS = """
p.big {
  font-size: 20px;
}
"""

ARTICLE = """
<br><br>
<p class='big' style='text-align: center'>
    <a href='https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html' target='_blank'>NeMo ASR</a> 
    | 
    <a href='https://github.com/NVIDIA/NeMo#nvidia-nemo' target='_blank'>Github Repo</a>
</p>
"""

SUPPORTED_LANGUAGES = set([])
SUPPORTED_MODEL_NAMES = set([])

# HF models, grouped by language identifier
hf_filter = nemo_asr.models.ASRModel.get_hf_model_filter()
hf_filter.task = "automatic-speech-recognition"

hf_infos = nemo_asr.models.ASRModel.search_huggingface_models(model_filter=hf_filter)
for info in hf_infos:
    print("Model ID:", info.modelId)
    try:
        lang_id = info.modelId.split("_")[1]  # obtains lang id as str
    except Exception:
        print("WARNING: Skipping model id -", info)
        continue

    SUPPORTED_LANGUAGES.add(lang_id)
    SUPPORTED_MODEL_NAMES.add(info.modelId)

SUPPORTED_MODEL_NAMES = sorted(list(SUPPORTED_MODEL_NAMES))

# DEBUG FILTER
# SUPPORTED_MODEL_NAMES = list(filter(lambda x: "en" in x and "conformer_transducer_large" in x, SUPPORTED_MODEL_NAMES))

model_dict = {}
for model_name in SUPPORTED_MODEL_NAMES:
    try:
        iface = gr.Interface.load(f'models/{model_name}')
        model_dict[model_name] = iface

        # model_dict[model_name] = None
    except:
        pass

if DEFAULT_EN_MODEL in model_dict:
    # Preemptively load the default EN model
    if model_dict[DEFAULT_EN_MODEL] is None:
        model_dict[DEFAULT_EN_MODEL] = gr.Interface.load(f'models/{DEFAULT_EN_MODEL}')

SUPPORTED_LANG_MODEL_DICT = {}
for lang in SUPPORTED_LANGUAGES:
    for model_id in SUPPORTED_MODEL_NAMES:
        if ("_" + lang + "_") in model_id:
            # create new lang in dict
            if lang not in SUPPORTED_LANG_MODEL_DICT:
                SUPPORTED_LANG_MODEL_DICT[lang] = [model_id]
            else:
                SUPPORTED_LANG_MODEL_DICT[lang].append(model_id)

# Sort model names
for lang in SUPPORTED_LANG_MODEL_DICT.keys():
    model_ids = SUPPORTED_LANG_MODEL_DICT[lang]
    model_ids = sorted(model_ids)
    SUPPORTED_LANG_MODEL_DICT[lang] = model_ids


def get_device():
    gpu_available = torch.cuda.is_available()
    if gpu_available:
        return torch.cuda.get_device_name()
    else:
        return "CPU"


def parse_duration(audio_file):
    """
    FFMPEG to calculate durations. Libraries can do it too, but filetypes cause different libraries to behave differently.
    """
    process = subprocess.Popen(['ffmpeg', '-i', audio_file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, stderr = process.communicate()
    matches = re.search(
        r"Duration:\s{1}(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+\.\d+?),", stdout.decode(), re.DOTALL
    ).groupdict()

    duration = 0.0
    duration += float(matches['hours']) * 60.0 * 60.0
    duration += float(matches['minutes']) * 60.0
    duration += float(matches['seconds']) * 1.0
    return duration


def resolve_model_type(model_name: str) -> str:
    """
    Map model name to a class type, without loading the model. Has some hardcoded assumptions in
    semantics of model naming.
    """
    # Loss specific maps
    if 'hybrid' in model_name or 'hybrid_ctc' in model_name or 'hybrid_transducer' in model_name:
        return 'hybrid'
    elif 'transducer' in model_name or 'rnnt' in model_id:
        return 'transducer'
    elif 'ctc' in model_name:
        return 'ctc'

    # Model specific maps
    if 'jasper' in model_name:
        return 'ctc'
    elif 'quartznet' in model_name:
        return 'ctc'
    elif 'citrinet' in model_name:
        return 'ctc'
    elif 'contextnet' in model_name:
        return 'transducer'

    return None


def resolve_model_stride(model_name) -> int:
    """
    Model specific pre-calc of stride levels.
    Dont laod model to get such info.
    """
    if 'jasper' in model_name:
        return 2
    if 'quartznet' in model_name:
        return 2
    if 'conformer' in model_name:
        return 4
    if 'squeezeformer' in model_name:
        return 4
    if 'citrinet' in model_name:
        return 8
    if 'contextnet' in model_name:
        return 8

    return -1


def convert_audio(audio_filepath):
    """
    Transcode all mp3 files to monochannel 16 kHz wav files.
    """
    filedir = os.path.split(audio_filepath)[0]
    filename, ext = os.path.splitext(audio_filepath)

    if ext == 'wav':
        return audio_filepath

    out_filename = os.path.join(filedir, filename + '.wav')

    process = subprocess.Popen(
        ['ffmpeg', '-y', '-i', audio_filepath, '-ac', '1', '-ar', str(SAMPLE_RATE), out_filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )

    stdout, stderr = process.communicate()

    if os.path.exists(out_filename):
        return out_filename
    else:
        return None


def extract_result_from_manifest(filepath, model_name) -> (bool, str):
    """
    Parse the written manifest which is result of the buffered inference process.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = json.loads(line)
                data.append(line['pred_text'])
            except Exception as e:
                pass

    if len(data) > 0:
        return True, data[0]
    else:
        return False, f"Could not perform inference on model with name : {model_name}"


def build_html_output(s: str, style: str = "result_item_success"):
    return f"""
    <div class='result'>
        <div class='result_item {style}'>
          {s}
        </div>
    </div>
    """


def infer_audio(model_name: str, audio_file: str) -> str:
    """
    Main method that switches from HF inference for small audio files to Buffered CTC/RNNT mode for long audio files.
    Args:
        model_name: Str name of the model (potentially with / to denote HF models)
        audio_file: Path to an audio file (mp3 or wav)
    Returns:
        str which is the transcription if successful.
        str which is HTML output of logs.
    """
    # Parse the duration of the audio file
    duration = parse_duration(audio_file)

    if duration > BUFFERED_INFERENCE_DURATION_THRESHOLD:  # Longer than one minute; use buffered mode
        # Process audio to be of wav type (possible youtube audio)
        audio_file = convert_audio(audio_file)

        # If audio file transcoding failed, let user know
        if audio_file is None:
            return "Error:- Failed to convert audio file to wav."

        # Extract audio dir from resolved audio filepath
        audio_dir = os.path.split(audio_file)[0]

        # Next calculate the stride of each model
        model_stride = resolve_model_stride(model_name)

        if model_stride < 0:
            return f"Error:- Failed to compute the model stride for model with name : {model_name}"

        # Process model type (CTC/RNNT/Hybrid)
        model_type = resolve_model_type(model_name)

        if model_type is None:

            # Model type could not be infered.
            # Try all feasible options
            RESULT = None

            try:
                ctc_config = buffered_ctc.TranscriptionConfig(
                    pretrained_name=model_name,
                    audio_dir=audio_dir,
                    output_filename="output.json",
                    audio_type="wav",
                    overwrite_transcripts=True,
                    model_stride=model_stride,
                    chunk_len_in_secs=20.0,
                    total_buffer_in_secs=30.0,
                )

                buffered_ctc.main(ctc_config)
                result = extract_result_from_manifest('output.json', model_name)
                if result[0]:
                    RESULT = result[1]

            except Exception as e:
                pass

            try:
                rnnt_config = buffered_rnnt.TranscriptionConfig(
                    pretrained_name=model_name,
                    audio_dir=audio_dir,
                    output_filename="output.json",
                    audio_type="wav",
                    overwrite_transcripts=True,
                    model_stride=model_stride,
                    chunk_len_in_secs=20.0,
                    total_buffer_in_secs=30.0,
                )

                buffered_rnnt.main(rnnt_config)
                result = extract_result_from_manifest('output.json', model_name)[-1]

                if result[0]:
                    RESULT = result[1]
            except Exception as e:
                pass

            if RESULT is None:
                return f"Error:- Could not parse model type; failed to perform inference with model {model_name}!"

        elif model_type == 'ctc':

            # CTC Buffered Inference
            ctc_config = buffered_ctc.TranscriptionConfig(
                pretrained_name=model_name,
                audio_dir=audio_dir,
                output_filename="output.json",
                audio_type="wav",
                overwrite_transcripts=True,
                model_stride=model_stride,
                chunk_len_in_secs=20.0,
                total_buffer_in_secs=30.0,
            )

            buffered_ctc.main(ctc_config)
            return extract_result_from_manifest('output.json', model_name)[-1]

        elif model_type == 'transducer':

            # RNNT Buffered Inference
            rnnt_config = buffered_rnnt.TranscriptionConfig(
                pretrained_name=model_name,
                audio_dir=audio_dir,
                output_filename="output.json",
                audio_type="wav",
                overwrite_transcripts=True,
                model_stride=model_stride,
                chunk_len_in_secs=20.0,
                total_buffer_in_secs=30.0,
            )

            buffered_rnnt.main(rnnt_config)
            return extract_result_from_manifest('output.json', model_name)[-1]

        else:
            return f"Error:- Could not parse model type; failed to perform inference with model {model_name}!"

    else:
        # Obtain Gradio Model function from cache of models
        if model_name in model_dict:
            model = model_dict[model_name]

            if model is None:
                # Load the gradio interface
                # try:
                iface = gr.Interface.load(f'models/{model_name}')
                print(iface)
                # except:
                #     iface = None

                if iface is not None:
                    # Update model cache
                    model_dict[model_name] = iface
        else:
            model = None

        if model is not None:
            # Use HF API for transcription
            try:
                transcriptions = model(audio_file)
                return transcriptions
            except Exception as e:
                transcriptions = ""
                error = ""

                error += (
                    f"The model `{model_name}` is currently loading and cannot be used "
                    f"for transcription.<br>"
                    f"Please try another model or wait a few minutes."
                )

                return error

        else:
            error = (
                f"Error:- Could not find model {model_name} in list of available models : "
                f"{list([k for k in model_dict.keys()])}"
            )
            return error


def transcribe(microphone, audio_file, model_name):

    audio_data = None
    warn_output = ""
    if (microphone is not None) and (audio_file is not None):
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )
        audio_data = microphone

    elif (microphone is None) and (audio_file is None):
        warn_output = "ERROR: You have to either use the microphone or upload an audio file"

    elif microphone is not None:
        audio_data = microphone
    else:
        audio_data = audio_file

    if audio_data is not None:
        audio_duration = parse_duration(audio_data)
    else:
        audio_duration = None

    time_diff = None
    try:
        with tempfile.TemporaryDirectory() as tempdir:
            filename = os.path.split(audio_data)[-1]
            new_audio_data = os.path.join(tempdir, filename)
            shutil.copy2(audio_data, new_audio_data)

            if os.path.exists(audio_data):
                os.remove(audio_data)

            audio_data = new_audio_data

            # Use HF API for transcription
            start = time.time()
            transcriptions = infer_audio(model_name, audio_data)
            end = time.time()
            time_diff = end - start

    except Exception as e:
        transcriptions = ""
        warn_output = warn_output

        if warn_output != "":
            warn_output += "<br><br>"

        warn_output += (
            f"The model `{model_name}` is currently loading and cannot be used "
            f"for transcription.<br>"
            f"Please try another model or wait a few minutes."
        )

    # Built HTML output
    if warn_output != "":
        html_output = build_html_output(warn_output, style="result_item_error")
    else:
        if transcriptions.startswith("Error:-"):
            html_output = build_html_output(transcriptions, style="result_item_error")
        else:
            output = f"Successfully transcribed on {get_device()} ! <br>" f"Transcription Time : {time_diff: 0.3f} s"

            if audio_duration > BUFFERED_INFERENCE_DURATION_THRESHOLD:
                output += f""" <br><br>
                Note: Audio duration was {audio_duration: 0.3f} s, so model had to be downloaded, initialized, and then
                buffered inference was used. <br>
                """

            html_output = build_html_output(output)

    return transcriptions, html_output


def _return_yt_html_embed(yt_url):
    """ Obtained from https://huggingface.co/spaces/whisper-event/whisper-demo """
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def yt_transcribe(yt_url: str, model_name: str):
    """ Modified from https://huggingface.co/spaces/whisper-event/whisper-demo """
    if yt_url == "":
        text = ""
        html_embed_str = ""
        html_output = build_html_output(f"""
            Error:- No YouTube URL was provide !
            """, style='result_item_error')
        return text, html_embed_str, html_output

    yt = pt.YouTube(yt_url)
    html_embed_str = _return_yt_html_embed(yt_url)

    with tempfile.TemporaryDirectory() as tempdir:
        file_uuid = str(uuid.uuid4().hex)
        file_uuid = f"{tempdir}/{file_uuid}.mp3"

        # Download YT Audio temporarily
        download_time_start = time.time()

        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(filename=file_uuid)

        download_time_end = time.time()

        # Get audio duration
        audio_duration = parse_duration(file_uuid)

        # Perform transcription
        infer_time_start = time.time()

        text = infer_audio(model_name, file_uuid)

        infer_time_end = time.time()

    if text.startswith("Error:-"):
        html_output = build_html_output(text, style='result_item_error')
    else:
        html_output = f"""
        Successfully transcribed on {get_device()} ! <br>
        Audio Download Time : {download_time_end - download_time_start: 0.3f} s <br>
        Transcription Time : {infer_time_end - infer_time_start: 0.3f} s <br> 
        """

        if audio_duration > BUFFERED_INFERENCE_DURATION_THRESHOLD:
            html_output += f""" <br>
            Note: Audio duration was {audio_duration: 0.3f} s, so model had to be downloaded, initialized, and then
            buffered inference was used. <br>
            """

        html_output = build_html_output(html_output)

    return text, html_embed_str, html_output


def create_lang_selector_component(default_en_model=DEFAULT_EN_MODEL):
    """
    Utility function to select a langauge from a dropdown menu, and simultanously update another dropdown
    containing the corresponding model checkpoints for that language.
    Args:
        default_en_model: str name of a default english model that should be the set default.
    Returns:
        Gradio components for lang_selector (Dropdown menu) and models_in_lang (Dropdown menu)
    """
    lang_selector = gr.components.Dropdown(
        choices=sorted(list(SUPPORTED_LANGUAGES)), value="en", type="value", label="Languages", interactive=True,
    )
    models_in_lang = gr.components.Dropdown(
        choices=sorted(list(SUPPORTED_LANG_MODEL_DICT["en"])),
        value=default_en_model,
        label="Models",
        interactive=True,
    )

    def update_models_with_lang(lang):
        models_names = sorted(list(SUPPORTED_LANG_MODEL_DICT[lang]))
        default = models_names[0]

        if lang == 'en':
            default = default_en_model
        return models_in_lang.update(choices=models_names, value=default)

    lang_selector.change(update_models_with_lang, inputs=[lang_selector], outputs=[models_in_lang])

    return lang_selector, models_in_lang


"""
Define the GUI
"""
demo = gr.Blocks(title=TITLE, css=CSS)

with demo:
    header = gr.Markdown(MARKDOWN)

    with gr.Tab("Transcribe Audio"):
        with gr.Row() as row:
            file_upload = gr.components.Audio(source="upload", type='filepath', label='Upload File')
            microphone = gr.components.Audio(source="microphone", type='filepath', label='Microphone')

        lang_selector, models_in_lang = create_lang_selector_component()

        run = gr.components.Button('Transcribe')

        transcript = gr.components.Label(label='Transcript')
        audio_html_output = gr.components.HTML()

        run.click(
            transcribe, inputs=[microphone, file_upload, models_in_lang], outputs=[transcript, audio_html_output]
        )

    with gr.Tab("Transcribe Youtube"):
        yt_url = gr.components.Textbox(
            lines=1, label="Youtube URL", placeholder="Paste the URL to a YouTube video here"
        )

        lang_selector_yt, models_in_lang_yt = create_lang_selector_component(
            default_en_model=DEFAULT_BUFFERED_EN_MODEL
        )

        with gr.Row():
            run = gr.components.Button('Transcribe YouTube')
            embedded_video = gr.components.HTML()

        transcript = gr.components.Label(label='Transcript')
        yt_html_output = gr.components.HTML()

        run.click(
            yt_transcribe, inputs=[yt_url, models_in_lang_yt], outputs=[transcript, embedded_video, yt_html_output]
        )

    gr.components.HTML(ARTICLE)

demo.queue(concurrency_count=1)
demo.launch(enable_queue=True)

