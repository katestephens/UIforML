import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class GodelChatbot:
    KNOWLEDGE_FILEPATH = 'knowledge.txt'
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.filepath = 'knowledge.txt'
        self.top_p = 0.9
        self.min_length = 8
        self.max_length = 64

    def get_knowledge(self, filepath=None):
        if filepath is None:
            filepath = self.filepath
        with open(filepath, 'r') as file:
            data = file.read().replace('\n', '')
        return data

    def predict(self, input, history=[]):
        instruction = 'Instruction: given a dialog context and related knowledge, you need to respond empathically and safely based on knowledge. You must greet the speaker.'
        knowledge = self.get_knowledge()
        if knowledge != '':
            knowledge = '[KNOWLEDGE] ' + knowledge
        dialog_history = list(sum(history, ()))
        dialog_history.append(input)
        dialog = ' EOS ' .join(dialog_history)
        query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
        input_ids = self.tokenizer(f"{query}", return_tensors='pt').input_ids
        output = self.model.generate(input_ids, min_length=int(
            self.min_length), max_length=int(self.max_length), top_p=self.top_p, do_sample=True).tolist()
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        history.append((input, response))
        return history, history

godel_chatbot = GodelChatbot('microsoft/GODEL-v1_1-base-seq2seq')

with gr.Blocks(title="Godel Chatbot") as demo:
    prompt_state = gr.State([])
    with gr.Row():
        out = gr.Chatbot()
    with gr.Column():
        prompt = gr.Textbox(placeholder="Type your prompt here...", show_label=False,)
        prompt.submit(fn=godel_chatbot.predict, inputs=[prompt, prompt_state], outputs=[out, prompt_state], show_progress=True)
    greet_btn = gr.Button("submit")
    greet_btn.click(fn=godel_chatbot.predict, inputs=[prompt, prompt_state], outputs=[out, prompt_state])

demo.queue(concurrency_count=1)
demo.launch()