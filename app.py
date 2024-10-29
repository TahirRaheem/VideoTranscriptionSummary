# Import necessary libraries
import whisper
from transformers import pipeline
from moviepy.editor import VideoFileClip
import streamlit as st

# Load OpenAI's Whisper model for transcription
model = whisper.load_model("base")

def transcribe_video(video_path):
    # Load video and extract audio
    video = VideoFileClip(video_path)
    audio_path = "audio.wav"
    video.audio.write_audiofile(audio_path)
    
    # Transcribe audio to text
    result = model.transcribe(audio_path)
    transcription = result["text"]
    return transcription

# Load BART summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    # Handle long text by breaking it into smaller chunks if needed
    max_length = 512
    if len(text) > max_length:
        summary = ""
        for i in range(0, len(text), max_length):
            chunk = text[i:i+max_length]
            chunk_summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
            summary += chunk_summary[0]['summary_text'] + " "
        return summary.strip()
    else:
        # For shorter text, summarize directly
        summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return summary[0]['summary_text']

# Streamlit application layout
st.title("Video Transcription and Summarization App")
st.write("Upload a video file, and the app will generate a transcription and summarize the content.")

# File uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.video(uploaded_file)  # Display the uploaded video

    # Button to process the video
    if st.button("Transcribe & Summarize"):
        # Transcribe the uploaded video
        transcription = transcribe_video("temp_video.mp4")
        st.subheader("Transcription:")
        st.write(transcription)

        # Summarize the transcription
        summary = summarize_text(transcription)
        st.subheader("Summary:")
        st.write(summary)
