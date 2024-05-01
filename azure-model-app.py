import os
from dotenv import load_dotenv
import torch
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import numpy as np
from openai import AzureOpenAI

client = AzureOpenAI(
        api_key='',  
        api_version="2024-02-01",
        azure_endpoint=''
)
deployment_id = "whisper"

def transcribe_audio(file):

    # Convert audio bytes to numpy array
 #   audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

    # Normalize audio array
  #  audio_tensor = torch.tensor(audio_array, dtype=torch.float64) / 32768.0

    # Provide inputs to the processor
    #inputs = processor(audio=audio_tensor, sampling_rate=16000, return_tensors="pt")
   # input_features = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").input_features

   # generate token ids
   # predicted_ids = model.generate(input_features)
    # decode token ids to text
    #transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    transcription = client.audio.transcriptions.create(
        file=open(file, "rb"),            
        model=deployment_id
    )
    return transcription

# Streamlit app
st.title("Audio to Text Transcription..")

audio_bytes = audio_recorder(pause_threshold=3.0, sample_rate=16_000)
st.write(audio_bytes)

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")
    save_audio(audio_bytes,'output.wav')
    file='output.wav'
    transcription = transcribe_audio(file)

    if transcription:
        st.write("Transcription:")
        st.write(transcription)
    else:
        st.write("Error: Failed to transcribe audio.")
else:
    st.write("No audio recorded.")

