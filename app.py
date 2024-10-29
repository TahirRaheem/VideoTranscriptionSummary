import streamlit as st
import torch
from transformers import pipeline
import moviepy.editor as mp
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Load models
@st.cache(show_spinner=False)
def load_models():
    st.write("Loading models...")
    whisper_model = "openai/whisper-large"
    summarization_model = "facebook/bart-large-cnn"
    asr_pipeline = pipeline("automatic-speech-recognition", model=whisper_model)
    summarization_pipeline = pipeline("summarization", model=summarization_model)
    return asr_pipeline, summarization_pipeline

asr_pipeline, summarization_pipeline = load_models()

# Function to extract text from video
def extract_text_from_video(video_file):
    try:
        st.write("Extracting text from video...")
        video = mp.VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        video.audio.write_audiofile(audio_file, codec='pcm_s16le')

        transcription = asr_pipeline(audio_file)
        return transcription['text']
    except Exception as e:
        st.error(f"Error extracting text from video: {e}")
        return None

# Function to summarize text
def summarize_text(text):
    try:
        st.write("Summarizing text...")
        summary = summarization_pipeline(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        st.error(f"Error summarizing text: {e}")
        return None

# Streamlit UI
st.title("Video Transcription and Summarization")
video_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if video_file is not None:
    st.video(video_file)

    if st.button("Extract Text"):
        with st.spinner("Extracting text..."):
            text = extract_text_from_video(video_file)
            if text:
                st.text_area("Transcribed Text", text, height=200)

                if st.button("Summarize Text"):
                    with st.spinner("Summarizing text..."):
                        summary = summarize_text(text)
                        if summary:
                            st.text_area("Summary", summary, height=150)

st.write("Made with ❤️ using Streamlit")
