#%%
!pip install datasets[audio] gradio -q
# %%
from datasets import load_dataset

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds
#%%
example = minds[0]
example

# The intent_class is a classification category of the audio recording. To convert this number into a meaningful string, we can use the int2str() method
id2label = minds.features["intent_class"].int2str
id2label(example["intent_class"])

#%%
import gradio as gr

def generate_audio():
    example = minds.shuffle()[0]
    audio = example["audio"]
    return(
        audio["sampling_rate"],
        audio["array"]
    ),id2label(example["intent_class"])

with gr.Blocks() as demo:
    with gr.Column():
        for _ in range(4):
            audio, label = generate_audio()
            output = gr.Audio(audio, label=label)

demo.launch(debug=True)
# %%
import matplotlib.pyplot as plt
import librosa.display

plt.figure().set_figwidth(12)
example = minds.shuffle()[0]
audio = example["audio"]
librosa.display.waveshow(audio["array"], sr=audio["sampling_rate"])
# %%
