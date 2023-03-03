from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

def take_last_tokens(inputs, note_history, history):
    """Filter the last 128 tokens"""
    if inputs['input_ids'].shape[1] > 128:
        inputs['input_ids'] = torch.tensor([inputs['input_ids'][0][-128:].tolist()])
        inputs['attention_mask'] = torch.tensor([inputs['attention_mask'][0][-128:].tolist()])
        note_history = ['</s> <s>'.join(note_history[0].split('</s> <s>')[2:])]
        history = history[1:]
    return inputs, note_history, history

def add_note_to_history(note, note_history):
    """Add a note to the historical information"""
    note_history.append(note)
    note_history = '</s> <s>'.join(note_history)
    return [note_history]

def chat(message, history):
    history = history or []
    if history: 
        history_useful = ['</s> <s>'.join([str(a[0])+'</s> <s>'+str(a[1]) for a in history])]
    else:
        history_useful = []
    history_useful = add_note_to_history(message, history_useful)
    inputs = tokenizer(history_useful, return_tensors="pt", max_length=128)
    inputs, history_useful, history = take_last_tokens(inputs, history_useful, history)
    reply_ids = model.generate(**inputs, max_new_tokens=250)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    history_useful = add_note_to_history(response, history_useful)
    list_history = history_useful[0].split('</s> <s>')
    history.append((list_history[-2], list_history[-1]))
    return history, history

def clear_prompt():
    return ""

with gr.Blocks() as demo:
    prompt_state = gr.State([])
    with gr.Row() as row:
        out = gr.Chatbot()
    with gr.Column() as col:
        prompt = gr.Textbox(placeholder="Type your prompt here...", show_label=False)
        prompt.submit(fn=chat, inputs=[prompt, prompt_state], outputs=[out, prompt_state], show_progress=True)
        with gr.Row() as interior_row:
            clear_btn = gr.Button(value="Clear")
            clear_btn.click(fn=clear_prompt, outputs=prompt)
            submit_btn = gr.Button(value="Submit")
            submit_btn.click(fn=chat, inputs=[prompt, prompt_state], outputs=[out, prompt_state], show_progress=True)

demo.queue(concurrency_count=1)
demo.launch()