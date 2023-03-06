import os
import sys
import gradio as gr
from pytube import YouTube
import nemo.collections.asr as nemo_asr
import librosa
import soundfile

TITLE = "NeMo ASR"
DESCRIPTION = "Demo of Various Models in NeMo ASR"
DEFAULT_EN_MODEL = "stt_en_conformer_transducer_large"

MARKDOWN = f"""
# {TITLE}
## {DESCRIPTION}
"""

SUPPORTED_MODEL_NAMES = sorted(
    mdl.pretrained_model_name
    for mdl in nemo_asr.models.ASRModel.list_available_models()
)

def resample_audio(file_path, sr=44100):   
    print(f"[INFO] Resampling audio file: {file_path}")
    y, sr = librosa.load(file_path, sr)
    conversion = file_path + ".wav"
    soundfile.write(conversion, y, sr, format="wav")
    return conversion

def download_youtube_audio(url):
    print(f"[INFO] Downloading audio from YouTube URL: {url}")
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    new_file = base.replace(" ", "")
    os.rename(out_file, new_file)
    return new_file

def transcribe(audio_file, microphone, model_name):
    print(f"[INFO] Transcribing using model {model_name}")
    warning_message=""
    if microphone and audio_file:
        warning_message = "You must either use the microphone or upload an audio file"
        raise ValueError(warning_message)
    elif microphone is not None:
        print("[INFO] Using microphone audio...")
        audio_path = microphone
    elif audio_file is not None:
        print(f"[INFO] Using uploaded audio file: {audio_file}")
        audio_path = resample_audio(audio_file)
    else:
        warning_message="You must either use the microphone or upload an audio file"
        raise ValueError(warning_message)

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    try:
        transcriptions = model.transcribe(paths2audio_files=[audio_path])
    except Exception as e:
        print(f"[ERROR] {e}")
        warning_message = warning_message + "\n\n"
        warning_message += (f"Error: {e}")
        transcriptions = [warning_message]
        #raise ValueError(transcriptions[0])
    
    return f'{transcriptions[0]}'

demo = gr.Blocks(title=TITLE)
with demo:
    header = gr.Markdown(MARKDOWN)

    with gr.Row() as row:
        with gr.Row() as interior_row:
            file_upload = gr.components.Audio(source="upload", type='filepath', label='Upload File')
            microphone = gr.components.Audio(source="microphone", type='filepath', label='Microphone')
        with gr.Column() as interior_column:
            youtube_upload = gr.Textbox(fn=download_youtube_audio, placeholder="Paste YouTube URL here...", label="YouTube URL", )
            run_youtube_upload = gr.components.Button('Upload YouTube Video')
            run_youtube_upload.click(download_youtube_audio, inputs=[youtube_upload], outputs=[file_upload])

    models = gr.components.Dropdown(
        choices=SUPPORTED_MODEL_NAMES,
        value=DEFAULT_EN_MODEL,
        label="Models",
        interactive=True,
    )
    transcript = gr.components.Label(label='Transcript')

    run = gr.components.Button('Transcribe')
    run.click(transcribe, inputs=[microphone, file_upload, models], outputs=[transcript])

demo.queue(concurrency_count=1)
demo.launch()
