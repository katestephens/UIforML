import gradio as gr

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

preset_examples = [
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'Does money buy happiness?', 'Chitchat'),
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'What is the goal of life?', 'Chitchat'),
    ('Instruction: given a dialog context, you need to response empathically.',
     '', 'What is the most interesing thing about our universe?', 'Chitchat'),
     ('Instruction: given a dialog context and related knowledge, you need to respond based on the knowledge.', 
     '''Determined is an open-source deep learning training platform that makes building models fast and easy. Determined enables you to: Train models faster using state-of-the-art distributed training, without changing your model code. Automatically find high-quality models with advanced hyperparameter tuning from the creators of Hyperband. Get more from your GPUs with smart scheduling and cut cloud GPU costs by seamlessly using preemptible instances. Track and reproduce your work with experiment tracking that works out-of-the-box, covering code versions, metrics, checkpoints, and hyperparameters.  Determined integrates these features into an easy-to-use, high-performance deep learning environment — which means you can spend your time building models instead of managing infrastructure.  To use Determined, you can continue using popular DL frameworks such as TensorFlow and PyTorch; you just need to update your model code to integrate with the Determined API. Determined was acquired by HPE in 2020.
     ''',
     'What is Determined?', 'Grounded Response Generation'
     ),
     (
        'Instruction: given a dialog context and related knowledge, you need to respond safely based on the knowledge.',
        '''Over-the-counter medications such as ibuprofen (Advil, Motrin IB, others), acetaminophen (Tylenol, others) and aspirin. Always consult with a doctor.
        ''',
        'I have a headache, what should I do?', "Grounded Response Generation"
     ),
]


def generate(instruction, knowledge, dialog, top_p, min_length, max_length):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"

    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, min_length=int(
        min_length), max_length=int(max_length), top_p=top_p, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(query)
    print(output)
    return output


def api_call_generation(instruction, knowledge, query, top_p, min_length, max_length):

    dialog = [
        query
    ]
    response = generate(instruction, knowledge, dialog,
                        top_p, min_length, max_length)

    return response


def change_example(choice):
    choice_idx = int(choice.split()[-1]) - 1
    instruction, knowledge, query, instruction_type = preset_examples[choice_idx]
    return [gr.update(lines=1, visible=True, value=instruction), gr.update(visible=True, value=knowledge), gr.update(lines=1, visible=True, value=query), gr.update(visible=True, value=instruction_type)]

def change_textbox(choice):
    if choice == "Chitchat":
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context, you need to response empathically.")
    elif choice == "Grounded Response Generation":
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context and related knowledge, you need to respond safely based on the knowledge.")
    else:
        return gr.update(lines=1, visible=True, value="Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.")


with gr.Blocks(title="Godel Goal-Based Dialogue") as demo:

    dropdown = gr.Dropdown(
        [f"Example {i+1}" for i in range(5)], label='Examples')
    with gr.Row():
        with gr.Column(scale=1):

            radio = gr.Radio(
                ["Q&A", "Chitchat", "Grounded Response Generation"], label="Instruction Type", value='Q&A'
            )
            instruction = gr.Textbox(lines=1, interactive=True, label="Instruction",
                                    value="Instruction: given a dialog context and related knowledge, you need to answer the question based on the knowledge.")
            radio.change(fn=change_textbox, inputs=radio, outputs=instruction)
        with gr.Column(scale=2):
            knowledge = gr.Textbox(lines=8, label="Knowledge")


    with gr.Row():

        with gr.Column(scale=1):
            top_p = gr.Slider(0, 1, value=0.9, label='top_p')
            min_length = gr.Number(8, label='min_length')
            max_length = gr.Number(
                500, label='max_length (should be larger than min_length)')

        with gr.Column(scale=2):
            response = gr.Textbox(label="Response", lines=6)
            query = gr.Textbox(lines=1, label="User Query")

    greet_btn = gr.Button("Generate")
    greet_btn.click(fn=api_call_generation, inputs=[
                    instruction, knowledge, query, top_p, min_length, max_length], outputs=response)
    dropdown.change(change_example, dropdown, [instruction, knowledge, query, radio])

demo.launch()
