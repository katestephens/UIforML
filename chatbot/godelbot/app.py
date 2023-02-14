import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

filepath = 'knowledge.txt'

def get_knowledge(filepath=filepath):
    with open(filepath, 'r') as file:
        data = file.read().replace('\n', '')
    print(data)
    return data

KNOWLEDGE_INGEST = get_knowledge()

def predict(input, history=[]):
    top_p = 0.9
    min_length = 8
    max_length = 64

    instruction = 'Instruction: given a dialog context and related knowledge, you need to respond empathically and safely based on knowledge. You must greet the speaker.'
    knowledge = KNOWLEDGE_INGEST

    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge

    dialog_history = list(sum(history, ()))

    dialog_history.append(input)

    #print(s)

    dialog = ' EOS ' .join(dialog_history)

    #print(dialog)

    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"



    # tokenize the new input sentence
    input_ids = tokenizer(f"{query}", return_tensors='pt').input_ids


    output = model.generate(input_ids, min_length=int(
        min_length), max_length=int(max_length), top_p=top_p, do_sample=True).tolist()
    
  
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    history.append((input, response))

    return history, history
    
with gr.Blocks() as demo:
    prompt_state = gr.State([])
    with gr.Row() as row:
        out = gr.Chatbot()
    with gr.Column() as col:
        prompt = gr.Textbox(placeholder="Type your prompt here...", show_label=False,)
        prompt.submit(fn=predict, inputs=[prompt, prompt_state], outputs=[out, prompt_state], show_progress=True)
    greet_btn = gr.Button("submit")
    greet_btn.click(fn=predict, inputs=[prompt, prompt_state], outputs=[out, prompt_state])

demo.queue(concurrency_count=1)
demo.launch()