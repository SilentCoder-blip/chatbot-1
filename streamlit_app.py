import os
import openai_whisper as whisper
from groq import Groq
from gtts import gTTS
import streamlit as st

# Load Whisper model for speech-to-text
whisper_model = whisper.load_model("base")

# Initialize Groq client with your API key
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to get response from Groq LLM
def get_response_from_groq(user_input):
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to convert text to speech using Google TTS
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    response_audio_path = "response.mp3"
    tts.save(response_audio_path)
    return response_audio_path

# Main function for voice-chat interaction
def voice_chat(audio):
    # Step 1: Convert audio to text using Whisper
    transcription = whisper_model.transcribe(audio)["text"]
    
    # Step 2: Get LLM response using Groq API
    response_text = get_response_from_groq(transcription)
    
    # Step 3: Convert response text to speech
    response_audio = text_to_speech(response_text)
    
    return response_text, response_audio

# Streamlit interface
st.title("Real-time Voice-to-Voice Chatbot")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if audio_file is not None:
    st.audio(audio_file, format="audio/wav")
    
    # Process the audio when the button is pressed
    if st.button("Chat"):
        # Step 1: Save uploaded audio temporarily for Whisper processing
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_file.getbuffer())

        # Step 2: Process the audio
        response_text, response_audio = voice_chat("temp_audio.wav")
        
        # Display the transcription and LLM response
        st.subheader("Transcription:")
        st.text(response_text)
        
        # Step 3: Play the response audio
        st.subheader("Response:")
        st.audio(response_audio, format="audio/mp3")

        # Clean up the temporary audio file
        os.remove("temp_audio.wav")
