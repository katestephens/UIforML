from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, Conversation, ConversationalPipeline
import torch
import gradio as gr

tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

def predict(input, history=[]):
    # tokenize the new input sentence
    new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')
    print(["new user input ids", new_user_input_ids])

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([torch.LongTensor(history), new_user_input_ids], dim=-1)
    print(["bot input ids", bot_input_ids])

    # generate a response 
    history = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id).tolist()
    print(["history", history])
    # convert the tokens to text, and then split the responses into lines
    response = [tokenizer.decode(history[0]).replace("<s>","").split("</s>")]
    print(response)
    response = [(response[i], response[i+1]) for i in range(0, len(response), 2)]  # convert to tuples of list
    return response, history

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
    inputs = tokenizer(history_useful, return_tensors="pt")
    inputs, history_useful, history = take_last_tokens(inputs, history_useful, history)
    reply_ids = model.generate(**inputs)
    response = tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[0]
    history_useful = add_note_to_history(response, history_useful)
    list_history = history_useful[0].split('</s> <s>')
    history.append((list_history[-2], list_history[-1]))
    return history, history

with gr.Blocks() as demo:
    prompt_state = gr.State([])
    with gr.Row() as row:
        out = gr.Chatbot()
    with gr.Column() as col:
        prompt = gr.Textbox(placeholder="Type your prompt here...", show_label=False,)
        prompt.submit(fn=chat, inputs=[prompt, prompt_state], outputs=[out, prompt_state], show_progress=True)
        #out.change(fn=predict, inputs=[prompt, prompt_state], outputs=[out, prompt_state])
#demo = gr.Interface(fn=predict, inputs=["text", "state"], outputs=["chatbot", "state"])
demo.queue(concurrency_count=1)

demo.launch()