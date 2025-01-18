#%%

from transformers import pipeline
from datasets import load_dataset
from datasets import Audio

minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
#%%
# Audio classification with a pipeline

classifier = pipeline(
    "audio-classification",
    model="anton-l/xtreme_s_xlsr_300m_minds14",
)

example = minds[0]
classifier(example["audio"]["array"])
#%%
# Automatic speech recognition with a pipeline
asr = pipeline("automatic-speech-recognition")
asr(example["audio"]["array"])
# %%
# Audio generation with a pipeline

pipe  = pipeline("text-to-speech",model="suno/bark-small")
text = "Ladybugs have had important roles in culture and religion, being associated with luck, love, fertility and prophecy. "
pipe(text)
# %%
