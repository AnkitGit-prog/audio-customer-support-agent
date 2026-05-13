"""
Streamlit Frontend for the Audio Customer Support Agent.
"""

import io
import os
import time
import base64
import logging
import requests
# pyrefly: ignore [missing-import]
import streamlit as st

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 120

# Initialize Session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"

st.set_page_config(
    page_title="Audio Customer Support Agent",
    page_icon="[Voice]",
    layout="wide",
)

# Custom CSS
st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
      .stApp { background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); }
      .glass-card {
          background: rgba(255,255,255,0.07);
          border: 1px solid rgba(255,255,255,0.12);
          border-radius: 14px;
          padding: 1.6rem;
          backdrop-filter: blur(12px);
          margin-bottom: 1.2rem;
      }
      .response-bubble {
          background: rgba(108,99,255,0.15);
          border-left: 4px solid #6c63ff;
          padding: 1rem;
          border-radius: 0 10px 10px 0;
          color: #e0e0ff;
      }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def call_text_chat(question: str, session_id: str):
    return requests.post(
        f"{BACKEND_URL}/chat/text", 
        params={"session_id": session_id},
        json={"text": question}, 
        timeout=REQUEST_TIMEOUT
    ).json()

def call_audio_chat(audio_bytes: bytes, filename: str, session_id: str, stream: bool):
    files = {"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")}
    params = {"session_id": session_id, "stream": str(stream).lower()}
    response = requests.post(f"{BACKEND_URL}/chat/audio", files=files, params=params, timeout=REQUEST_TIMEOUT)
    return {
        "audio": response.content,
        "transcript": response.headers.get("X-Transcript", ""),
        "response": response.headers.get("X-Response", ""),
        "language": response.headers.get("X-Language", "en")
    }

def call_health():
    return requests.get(f"{BACKEND_URL}/health", timeout=5).json()

def render_audio_player(audio_bytes: bytes):
    b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio controls autoplay style="width:100%"><source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Agent Control")
    
    st.subheader("Session Settings")
    st.write(f"Current Session: `{st.session_state.session_id}`")
    if st.button("Reset Session"):
        st.session_state.session_id = f"session_{int(time.time())}"
        st.rerun()

    st.markdown("---")
    try:
        health_data = call_health()
        if health_data.get("status") == "ready":
            st.success("🟢 Backend Online")
        else:
            st.warning("🟠 Backend Degraded")
            st.json(health_data.get("components", {}))
    except:
        st.error("🔴 Backend Offline")
    
    st.markdown("---")
    st.info("Features:\n- Conversation Memory\n- Multi-lang Support\n- Streaming Audio\n- VAD (Silence Trimming)")

# Main UI
st.title("Audio Customer Support Agent")
st.write("Ask a question by voice or text. Now with conversation history!")

tab1, tab2 = st.tabs(["Text Chat", "Audio Chat"])

with tab1:
    user_q = st.text_input("Your Question", placeholder="e.g. What is your return policy?")
    if st.button("Ask Text"):
        if user_q:
            with st.spinner("Thinking..."):
                res = call_text_chat(user_q, st.session_state.session_id)
                st.markdown(f'<div class="response-bubble">{res["response"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question.")

with tab2:
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])
    stream_toggle = st.checkbox("Enable Streaming (Lower Latency)", value=False)
    
    if audio_file and st.button("Process Audio"):
        with st.spinner("Processing..."):
            result = call_audio_chat(audio_file.getvalue(), audio_file.name, st.session_state.session_id, stream_toggle)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📝 Transcription")
                st.info(result["transcript"] if result["transcript"] else "No transcription available.")
            with col2:
                st.markdown("### 🌍 Language")
                st.success(f"Detected: `{result['language']}`")
            
            st.markdown("### 🤖 AI Response")
            st.markdown(f'<div class="response-bubble">{result["response"]}</div>', unsafe_allow_html=True)
            
            st.markdown("### 🔊 Spoken Response")
            render_audio_player(result["audio"])
