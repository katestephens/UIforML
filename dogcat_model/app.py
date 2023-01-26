import sys
from skimage import io
import matplotlib.pyplot as plt
from seldon_core.seldon_client import SeldonClient
import gradio as gr

def dog_or_cat(image):
    seldon_ip = "<replace_with_seldon_ip>"
    sc = SeldonClient(deployment_name="kt-deploy-update", namespace="seldon", gateway_endpoint=f"{seldon_ip}", gateway="istio")
    out = sc.predict(transport="rest", data=image)
    res = out.response['data']['ndarray'][0]
    return res

title_text="Dog or Cat"
description_text="Pick and example or upload an image of a dog or a cat to see if our model can correctly identify it!"
example_images=["Yogi.png","Zoey.png","cat.png"]

#Caching these examples for the demo
demo = gr.Interface(fn=dog_or_cat, inputs="image", outputs="text", title=title_text, description=description_text, examples=example_images, cache_examples=True)
demo.launch()