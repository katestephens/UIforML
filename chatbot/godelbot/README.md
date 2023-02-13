# Chatbot
This chatbot template uses HuggingFace to get a pretrained model. We are using [Microsoft Godel](https://www.microsoft.com/en-us/research/project/godel/).

To run locally in a Miniconda env. I install determined because I use determined as my development platform. See [Determined AI](https://github.com/determined-ai/determined)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod a+x Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
##yes to all the prompts
##restart your shell
conda create -n chatbot python=3.9
conda activate chatbot
pip install determined gradio transformers
gradio godelbot.py
```


