def extract_text_from_video(video_file):
    try:
        # Validate the video file type
        if not video_file.name.endswith(('.mp4', '.mov', '.avi')):
            st.error("Please upload a valid video file.")
            return None
        
        # Load video using moviepy
        video = mp.VideoFileClip(video_file)
        audio_file = "temp_audio.wav"
        
        # Ensure audio track exists
        if video.audio is None:
            st.error("The video file does not contain an audio track.")
            return None
        
        video.audio.write_audiofile(audio_file, codec='pcm_s16le')

        # Use Whisper model for transcription
        transcription = asr_pipeline(audio_file)
        
        if 'text' not in transcription:
            st.error("Error in transcription output format.")
            return None
        
        return transcription['text']
    except (FileNotFoundError, OSError) as e:
        st.error(f"Error opening video file: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None
