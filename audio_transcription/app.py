######IMPORTANT READ THIS PLS!!!!!!!!!!
######
##for development mode in Gradio... run `gradio app.py` in your terminal - trust me...
##make sure you have all dependencies and actually ran setup.ipynb
########
from __future__ import unicode_literals
import os
import sys
import gradio as gr
from pytube import YouTube
import nemo.collections.asr as nemo_asr
import librosa
import soundfile

#todo: Split out UI from Model stuff (maybe a class)... keep get youtube in the ui though
# onclick may not know what happens but calls the class 

TITLE = "NeMo ASR"
DESCRIPTION = "Demo of Various Models in NeMo ASR"
DEFAULT_EN_MODEL = "stt_en_citrinet_512"

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

SUPPORTED_MODEL_NAMES = set([])
available_models = nemo_asr.models.ASRModel.list_available_models()

for mdl in available_models:
    ## Todo: Load in models by language
    ##mdlnameswewant = mdl.pretrained_model_name.startswith("stt_en") ## seemed good to start with english
    ##if (mdlnameswewant):
        SUPPORTED_MODEL_NAMES.add(mdl.pretrained_model_name)
    #continue

SUPPORTED_MODEL_NAMES = sorted(list(SUPPORTED_MODEL_NAMES))

def resample_audio(file, sr=44100):   
    #Todo: refactor standard outs to be prints
    sys.stdout.write("[INFO] Resampling Audio...\n\n") 
    y,s =librosa.load(file, sr)
    conversion = file + ".wav"
    soundfile.write(conversion, y, s, format="wav")
    return conversion

def get_youtube_audio(url):
    sys.stdout.write("[INFO] Getting YouTube Audio...\n\n") 
    yt = YouTube(url)
    video = yt.streams.filter(only_audio=True).first()
    out_file = video.download()
    base, ext = os.path.splitext(out_file)
    new_file = base.replace(" ", "")
    os.rename(out_file, new_file)
    audio = new_file
    print(['file', audio])
    return audio
    

def transcribe(microphone, audio_file, model_name):

    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    warn_output = ""
    if (audio_file is not None) and (microphone is not None):
        sys.stdout.write(f'[ERROR] User uploaded microphone and audio, microphone will be used and audio will be discarded from transcription...\n\n') 
        warn_output = (
            "WARNING: You've uploaded an audio file and used the microphone. "
            "The recorded file from the microphone will be used and the uploaded audio will be discarded.\n"
        )
        audio_data = microphone

    elif (microphone is None) and (audio_file is None):
        sys.stdout.write(f'[ERROR] User must upload an audio file to proceed with transcription...\n\n') 
        return "ERROR: You have to either use the microphone or upload an audio file"

    elif microphone is not None:
        sys.stdout.write("[INFO] Using Microphone Audio...\n\n") 
        audio_data = microphone

    else:
        sys.stdout.write("[INFO]  Using uploaded file...\n\n") 
        audio_data = resample_audio(audio_file)

    try:
        print(['audio data try', audio_data])
        # Use HF API for transcription
        sys.stdout.write("[INFO]  Transcribing...\n\n") 
        transcriptions = model.transcribe(paths2audio_files=[audio_data])

    except Exception as e:
        sys.stdout.write(f'[ERROR] {e}...') 
        warn_output = warn_output + "\n\n"
        warn_output += (
            f"Error: {e}"
        )
        transcriptions = [warn_output]

    return f'{transcriptions[0]}'


##Two funcs - one that cleans and one that transcribes. Transcribe doesnt do any conditional testing. The cleanup one would be all the if else stuff


demo = gr.Blocks(title=TITLE, css=CSS)
# Could create a variable like gr.Row as a variable and then call these
with demo:
    header = gr.Markdown(MARKDOWN)

    with gr.Row() as row:
        with gr.Row() as interior_row:
            file_upload = gr.components.Audio(source="upload", type='filepath', label='Upload File')
            microphone = gr.components.Audio(source="microphone", type='filepath', label='Microphone')
        with gr.Column() as interior_column:
            youtube_upload = gr.Textbox(fn=get_youtube_audio, placeholder="Paste YouTube URL here...", label="YouTube URL", )
            run_youtube_upload = gr.components.Button('Upload YouTube Video')
            run_youtube_upload.click(get_youtube_audio, inputs=[youtube_upload], outputs=[file_upload])

    models = gr.components.Dropdown(
        choices=SUPPORTED_MODEL_NAMES,
        value=DEFAULT_EN_MODEL,
        label="Models",
        interactive=True,
    )
    transcript = gr.components.Label(label='Transcript')

    run = gr.components.Button('Transcribe')
    run.click(transcribe, inputs=[microphone, file_upload, models], outputs=[transcript])

    gr.components.HTML(ARTICLE)

demo.queue(concurrency_count=1)
demo.launch()
