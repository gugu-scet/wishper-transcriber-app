import os
import math
import uuid
import datetime
import tempfile
import streamlit as st
from pydub import AudioSegment
import whisper

st.set_page_config(page_title="Whisper Multilingual Transcriber", layout="centered")

st.title("🎧 Whisper Multilingual Audio Transcriber")
st.markdown("Upload an audio file, select language (or auto-detect), and get transcript + subtitles.")

# --- Sidebar ---
st.sidebar.header("Settings")
model_name = st.sidebar.selectbox("Model size", ["tiny", "base", "small", "medium", "large"], index=2)
chunk_seconds = st.sidebar.slider("Chunk length (seconds)", 30, 180, 60, 10)
language = st.sidebar.selectbox(
    "Language",
    ["Auto-detect", "English", "Hindi (हिन्दी)", "Odia (ଓଡ଼ିଆ)", "Bengali (বাংলা)",
     "Tamil (தமிழ்)", "Telugu (తెలుగు)", "Marathi (मराठी)", "Gujarati (ગુજરાતી)",
     "Kannada (ಕನ್ನಡ)", "Urdu (اُردُو)"]
)

LANG_MAP = {
    "Auto-detect": None, "English": "en", "Hindi (हिन्दी)": "hi", "Odia (ଓଡ଼ିଆ)": "or",
    "Bengali (বাংলা)": "bn", "Tamil (தமிழ்)": "ta", "Telugu (తెలుగు)": "te",
    "Marathi (मराठी)": "mr", "Gujarati (ગુજરાતી)": "gu", "Kannada (ಕನ್ನಡ)": "kn", "Urdu (اُردُو)": "ur"
}

uploaded_audio = st.file_uploader("Upload your audio file", type=["mp3", "wav", "m4a", "ogg", "flac"])

def format_timestamp(seconds):
    td = datetime.timedelta(seconds=float(max(0, seconds)))
    total = int(td.total_seconds())
    ms = int((td.total_seconds() - total) * 1000)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

@st.cache_resource(show_spinner=False)
def load_model_cached(name):
    return whisper.load_model(name)

def transcribe_audio(file, model_name, lang_code, chunk_s):
    workdir = tempfile.mkdtemp()
    audio = AudioSegment.from_file(file)
    chunk_ms = chunk_s * 1000
    n = math.ceil(len(audio) / chunk_ms)
    model = load_model_cached(model_name)

    all_text, all_segments = [], []
    offset = 0

    for i in range(n):
        start, end = i * chunk_ms, min((i + 1) * chunk_ms, len(audio))
        chunk_path = os.path.join(workdir, f"chunk_{i}.wav")
        audio[start:end].export(chunk_path, format="wav")
        result = model.transcribe(chunk_path, language=lang_code)
        text = result.get("text", "").strip()
        if text:
            all_text.append(text)
        for seg in result.get("segments", []):
            all_segments.append((offset + seg["start"], offset + seg["end"], seg["text"].strip()))
        offset += (end - start) / 1000

    # Save files
    txt_path = os.path.join(workdir, "transcript.txt")
    srt_path = os.path.join(workdir, "subtitles.srt")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(" ".join(all_text))

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, (stt, end, tx) in enumerate(all_segments, 1):
            f.write(f"{i}\n{format_timestamp(stt)} --> {format_timestamp(end)}\n{tx}\n\n")

    return txt_path, srt_path, " ".join(all_text)

if uploaded_audio:
    st.audio(uploaded_audio)
    if st.button("Transcribe"):
        with st.spinner("Transcribing, please wait... ⏳"):
            txt_path, srt_path, text = transcribe_audio(
                uploaded_audio, model_name, LANG_MAP[language], chunk_seconds
            )
        st.success("✅ Transcription completed!")
        st.download_button("⬇️ Download Transcript (TXT)", open(txt_path, "rb"), "transcript.txt")
        st.download_button("⬇️ Download Subtitles (SRT)", open(srt_path, "rb"), "subtitles.srt")
        st.text_area("Transcript Preview", text[:5000] + ("..." if len(text) > 5000 else ""), height=250)
