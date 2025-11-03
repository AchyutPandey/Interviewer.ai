Interviewer.ai
This project is deployed at hugging face.
Live @ [Viper51/Interviewer.ai](https://huggingface.co/spaces/Viper51/Interviewer.ai)

Interviewer.ai is a Streamlit-based mock interview assistant that generates interview questions from a candidate's resume, asks them interactively (text or voice), evaluates answers, and produces a final score and transcript.

This repository contains a production-ready Streamlit app with optional Google GenAI integration (Gemini) and Google Cloud speech services. The app is designed to run locally or inside Docker and includes defensive fallbacks so it works even when cloud APIs or optional audio libraries are not available.

Highlights
- Check the Enable LLM checkbox to use the LLM
- Generate tailored interview questions from resume text
- Interactive interview flow with stateful transcript
- Optional voice input (microphone) and audio output (TTS)
- Deterministic local fallbacks so the app still runs without cloud APIs

Contents

- `Dockerfile` — container definition for production deployment
- `requirements.txt` — Python dependencies (core + optional integrations)
- `src/streamlit_app.py` — main Streamlit application

Quick links

- Run locally (development)
- Run in Docker (production)
- Configuration / environment variables
- Troubleshooting (including 403 / Axios errors)

---

## Tech stack

- Python 3.12
- Streamlit — web UI framework
- LangChain (optional) + `langchain-google-genai` — integration helpers for Google GenAI
- google-generativeai — Google GenAI client (Gemini)
- google-cloud-texttospeech — Google Cloud Text-to-Speech (optional)
- google-cloud-speech — Google Cloud Speech-to-Text (optional)
- PyPDF2 — PDF parsing (if you switch to upload PDFs)
- pydantic — runtime models / validation
- gTTS — fallback text-to-speech (optional)
- SpeechRecognition — fallback speech recognition (optional)
- streamlit-mic-recorder — browser microphone component (optional)
- ffmpeg, portaudio, libasound2 (native packages used in Dockerfile for audio support)
- Docker — containerization (Dockerfile included)

---

## File layout

```
Interviewer.ai/
├─ Dockerfile
├─ requirements.txt
└─ src/
	 └─ streamlit_app.py   # main app
```

---