import torch
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from langchain.llms import Ollama
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Function to transcribe audio
def transcribe_audio(audio_bytes):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    audio_input, _ = processor(
        audio_bytes,
        sampling_rate=16_000,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        logits = model(input_values=audio_input.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)
    return transcription



# Streamlit app
st.title("Audio Transcription")

audio_bytes = audio_recorder(pause_threshold=3.0, sample_rate=16_000)

# Button to trigger audio transcription
if st.button("Transcribe Audio"):
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")

        # Transcribe audio
        transcription = transcribe_audio(audio_bytes)

        if transcription:
            st.write("Transcription:")
            st.write(transcription)
        else:
            st.write("Error: Failed to transcribe audio.")
    else:
        st.write("Error: No audio recorded.")

