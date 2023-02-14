# Chatbot
This chatbot template uses HuggingFace to get a pretrained model. We are using [Microsoft Godel](https://www.microsoft.com/en-us/research/project/godel/).

`godelbot.py` demonstrates an interface in which we provide knowledge and other parameters to the model in real time. 

`app.py` is a chatbot-style interface where the knowledge is already provided in a file (`knowledge.txt`) and is read into the app when it is initialized.

You can run both of these files using the same steps. You can also run them simultaneously in the same conda env.

## Development
You can run this application in a miniconda environment on your local machine. This is highly recommended so that you don't interfere with other python dependencies in other projects.

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
gradio app.py
```

Running `gradio app.py` will enable reloading for the UI so that you do not have to restart the app every time you make a UI change. Ensure you run that while you are doing development, otherwise it will be very cumbersome to restart the app manually.

## Deployment

Please follow the steps for deployment listed in the root directory's README.  You will want to use this dockerfile, build, and push to your docker registry. To run, make sure you
`docker run -p <port>:<port> <dockerregistry>/<image>:<version> python app.py` . As long as you have set your environment variables, this should work for you.  Please note that if you're using the microphone component in Gradio, you will need to move to HTTPS or create a SSL for the microphone to work. https://github.com/gradio-app/gradio/issues/2551