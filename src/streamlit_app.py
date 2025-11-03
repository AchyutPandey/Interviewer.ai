import streamlit as st
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# Optional AI SDKs
try:
    import google.generativeai as genai
except Exception:
    genai = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import PromptTemplate
except Exception:
    ChatGoogleGenerativeAI = None
    PromptTemplate = None

from pydantic import BaseModel, Field
from typing import Optional
import os
try:
    from google.cloud import speech
except Exception:
    speech = None
try:
    from google.cloud import texttospeech
except Exception:
    texttospeech = None
try:
    from streamlit_mic_recorder import mic_recorder
except Exception:
    mic_recorder = None

# --- Pydantic Models (from your code) ---

class questions(BaseModel):
    questions: list[str] = Field(description="List of questions")

class introduction(BaseModel):
    intro: Optional[str] = Field(description="Give AI agent's intro")
    question: str = Field(description="Question asked by AI agent")
    followup: Optional[str] = Field(description="The followup question to user's answer")

class evaluation(BaseModel):
    marks: int = Field(description="Marks out of 100")
    followup: Optional[str] = Field(description="The followup question")
    review: Optional[str] = Field(description="Short Review of the answer")

# --- AI & Logic Functions (from your code) ---

@st.cache_resource
def get_llm(api_key):
    """Cached function to initialize the LLM."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  
        temperature=1.0,
        google_api_key=api_key
    )
    
@st.cache_resource
def get_models(_llm_model):
    """Cached function to get structured output models."""
    generate_questions_resume_model = _llm_model.with_structured_output(questions)
    intro_model = _llm_model.with_structured_output(introduction)
    evaluate_answers_model = _llm_model.with_structured_output(evaluation)
    return generate_questions_resume_model, intro_model, evaluate_answers_model

    
# def read_resume(uploaded_file):
#     """Reads a PDF file uploaded via Streamlit."""
#     try:
#         if PdfReader is None:
#             st.warning("PyPDF2 is not installed; resume text extraction disabled.")
#             return None
#         reader = PdfReader(uploaded_file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text() or ""
#         return text
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#         return None

def generate_questions_from_resume(resume_text, model):
    """Generates interview questions from resume text."""
    if PromptTemplate is None or model is None or not st.session_state.get('enable_llm', False):
        # Simple fallback
        questions = ["Tell me about your most significant project.", "Describe a challenging bug you fixed.", "How do you design for scalability?", "Which technologies are you most comfortable with?"]
        return questions

    parse_resume_prompt_template = PromptTemplate(
        template="""Generate 4-8 interview questions about the Experience and Projects section from this given text of from a resume.
Try to cover all projects and experience. Generate some conceptual questions too. Don't generate unnecessary questions.
Resume:\n{text}""",
        input_variables=['text']
    )
    try:
        if not st.session_state.get('enable_llm', False):
            raise RuntimeError('LLM disabled')
        generate_question_from_resume_chain = parse_resume_prompt_template | model
        output = generate_question_from_resume_chain.invoke({'text': resume_text})
        return getattr(output, 'questions', output)
    except Exception as e:
        st.warning(f"LLM question generation failed or disabled, using fallback: {e}")
        questions = ["Tell me about your most significant project.", "Describe a challenging bug you fixed.", "How do you design for scalability?", "Which technologies are you most comfortable with?"]
        return questions

def get_introduction(model):
    """Gets the AI's intro and first question."""
    if PromptTemplate is None or model is None or not st.session_state.get('enable_llm', False):
        return type('O', (), {'intro': "Hello, I'm Interviewer.AI. Please introduce yourself.", 'question': "Can you briefly introduce yourself?"})()

    introduction_prompt = PromptTemplate(template="""Introduce yourself to the user telling the user that you are Heisenberg, an AI agent. And ask the user to give introduction""")
    try:
        if not st.session_state.get('enable_llm', False):
            raise RuntimeError('LLM disabled')
        intro_chain = introduction_prompt | model
        output = intro_chain.invoke({})
        return output
    except Exception as e:
        st.warning(f"LLM intro generation failed or disabled: {e}")
        return type('O', (), {'intro': "Hello, I'm Interviewer.AI. Please introduce yourself.", 'question': "Can you briefly introduce yourself?"})()

def ask_followup(user_intro, model):
    """Asks a followup to the user's intro."""
    if PromptTemplate is None or model is None or not st.session_state.get('enable_llm', False):
        return "Thanks — could you tell me one achievement you're most proud of?"

    intro_followup = PromptTemplate(template="""The user has given the following introduction of himself/herself. Ask a followup about his intro to make the user comfortable. Intro given by the user: {intro}""",
                                    input_variables=['intro'])
    try:
        if not st.session_state.get('enable_llm', False):
            raise RuntimeError('LLM disabled')
        followup_chain = intro_followup | model
        output = followup_chain.invoke({'intro': user_intro})
        return getattr(output, 'followup', None)
    except Exception as e:
        st.warning(f"LLM followup generation failed or disabled: {e}")
        return "Could you tell me about a specific result from that experience?"

def evaluate_answer(question, answer, model):
    """Evaluates the user's answer."""
    if PromptTemplate is None or model is None or not st.session_state.get('enable_llm', False):
        # Simple heuristic evaluator
        score = 50
        review = "Thank you for your answer. Provide more details next time."
        followup = None
        if answer and len(answer.split()) > 50:
            score = 80
            review = "Good answer — you covered several points."
        elif answer and len(answer.split()) > 20:
            score = 65
            review = "Decent answer; add more concrete examples."
        return type('O', (), {'marks': score, 'review': review, 'followup': followup})()

    evaluate_answer_prompt = PromptTemplate(template="""You are given a question and an answer. Evaluate the answer honestly on the question out of 100.
Also generate a very short review on the answer telling the candidate about his answer. If he is wrong but close to the correct answer, give subtle hints.
If a good followup question can be asked generate it but only if it is a genuine question.\nQuestion: {question}\n\n Answer: {answer}""",
                                            input_variables=['question', 'answer'])
    try:
        if not st.session_state.get('enable_llm', False):
            raise RuntimeError('LLM disabled')
        evaluate_chain = evaluate_answer_prompt | model
        output = evaluate_chain.invoke({'question': question, 'answer': answer})
        return output
    except Exception as e:
        st.warning(f"LLM evaluation failed or disabled: {e}")
        score = 50
        review = "Thank you for your answer. Provide more details next time."
        followup = None
        if answer and len(answer.split()) > 50:
            score = 80
        elif answer and len(answer.split()) > 20:
            score = 65
        return type('O', (), {'marks': score, 'review': review, 'followup': followup})()

# --- MODIFIED Streamlit Audio/Visual Function ---

import io # Make sure 'import io' is at the top of your file
@st.cache_data
def speech_to_text(audio_bytes):
    """
    Transcribes audio bytes using Google Cloud Speech-to-Text
    and returns the transcribed text.
    """
    if speech is None:
        st.warning("google-cloud-speech library not found, transcription is disabled.")
        return None
        
    # Get the API key from the environment (where HF secrets put it)
    api_key = os.environ.get("GOOGLE_API_KEY") 
    
    # Check if the key exists
    if not api_key:
        st.error("GOOGLE_API_KEY not found in secrets. Cannot initialize STT.")
        return None
        
    # Pass the key explicitly to the client
    client_options = {"api_key": api_key}
    client = speech.SpeechClient(client_options=client_options)

    # Configure the audio
    # Note: streamlit-mic-recorder outputs WAV, which is LINEAR16
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.ENCODING_UNSPECIFIED,
        language_code="en-US",
        sample_rate_hertz=48000 # This is a common sample rate
    )

    try:
        # Detects speech in the audio file
        st.info("Transcribing audio... (this may take a moment)")
        response = client.recognize(config=config, audio=audio)
        
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            st.session_state.chat_history.append(f"**You:** {transcript}")
            return transcript
        else:
            st.warning("Could not understand audio.")
            return None
            
    except Exception as e:
        st.error(f"Error during speech-to-text: {e}")
        st.info("This usually means the 'Cloud Speech-to-Text API' is not enabled or your mic is not outputting the correct audio format.")
        return None

# --- REPLACED: Official Google Cloud TTS Function ---

@st.cache_data
def synthesize_speech(text):
    """
    Synthesizes speech from the given text using Google Cloud TTS
    and returns the audio content as bytes.
    """
    if texttospeech is None:
        st.warning("google-cloud-texttospeech library not found, audio playback is disabled.")
        return None

    # --- START OF FIX ---
    # Get the API key from the environment (where HF secrets put it)
    api_key = os.environ.get("GOOGLE_API_KEY") 
    
    # Check if the key exists
    if not api_key:
        st.error("GOOGLE_API_KEY not found in secrets. Cannot initialize TTS.")
        return None
        
    # Pass the key explicitly to the client
    client_options = {"api_key": api_key}
    client = texttospeech.TextToSpeechClient(client_options=client_options)
    # --- END OF FIX ---

    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    # Select the type of audio file you want
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response.audio_content

def text_to_speech_and_display(text, autoplay=True):
    """
    Displays the text and plays the synthesized audio.
    """
    if not text:
        return
        
    try:
        # 1. Display the caption in chat
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        st.session_state.chat_history.append(f"**Interviewer:** {text}")

        # 2. Synthesize speech
        if not st.session_state.get('audio_enabled', False):
            return
        audio_content = synthesize_speech(text)

        # 3. Display audio player
        if audio_content:
            st.audio(audio_content, format='audio/mp3', autoplay=autoplay)
        else:
            st.info("Audio generation is disabled or failed.")
            
    except Exception as e:
        # This will catch any API errors (like 403, 404, etc.)
        st.error(f"Error during text-to-speech: {e}")
        st.info("This usually means the 'Cloud Text-to-Speech API' is not enabled in your Google Cloud project.")
# --- END OF REPLACEMENT ---
# We are replacing it with a text_input


# --- Main Streamlit App ---

st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("Interviewer.AI")

# Initialize LLM and models
llm = None
gen_q_model = None
intro_model = None
eval_model = None

# First, load the key from the environment variable if genai is available
if genai is None or ChatGoogleGenerativeAI is None:
    st.warning("Google GenAI or LangChain wrappers not available. App will use deterministic fallbacks.")

if 'enable_llm' not in st.session_state:
    st.session_state.enable_llm = False

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
api_key_exists = bool(GOOGLE_API_KEY)

if not api_key_exists:
    st.warning("⚠️ GOOGLE_API_KEY not found in environment variables.")
    st.info("Add GOOGLE_API_KEY to your Hugging Face Space secrets to enable AI features.")

# LLM Enable Checkbox
enable_llm_checkbox = st.checkbox(
    "Enable LLM features (Not checking this box will result in using default hard-coded questions)",  
    value=st.session_state.enable_llm,
    disabled=not api_key_exists,
    help="AI-powered question generation and evaluation"
)
st.session_state.enable_llm = enable_llm_checkbox

# Initialize LLM if enabled
if st.session_state.enable_llm and api_key_exists:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = get_llm(GOOGLE_API_KEY)
        gen_q_model, intro_model, eval_model = get_models(llm)
        st.success("✅ LLM features enabled successfully")
    except Exception as e:
        st.error(f"❌ Could not initialize LLM: {e}")
        st.info("Check your API key and try again.")
        st.session_state.enable_llm = False
        llm = None
        gen_q_model = None
        intro_model = None
        eval_model = None

# Test API Button (AFTER initialization)
if st.button("Test Google API Connection"):
    if not st.session_state.enable_llm:
        st.error("❌ LLM features are not enabled. Check the checkbox above first.")
    elif llm is None:
        st.error("❌ LLM is not initialized. Check API key configuration.")
    else:
        try:
            with st.spinner("Testing API connection..."):
                test_response = llm.invoke("Say 'Hello' if you can hear me.")
                st.success("✅ SUCCESS! API is working correctly.")
                st.info(f"Response: {test_response.content if hasattr(test_response, 'content') else str(test_response)}")
        except Exception as e:
            st.error(f"❌ API call FAILED with error: {e}")
            st.info("This usually means: invalid API key, quota exceeded, or network issues.")

st.divider()

# --- Session State Initialization ---
if 'stage' not in st.session_state:
    st.session_state.stage = 'start'
if 'audio_enabled' not in st.session_state:
    st.session_state.audio_enabled = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'q_index' not in st.session_state:
    st.session_state.q_index = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
if 'total_marks' not in st.session_state:
    st.session_state.total_marks = 0
if 'num_questions' not in st.session_state:
    st.session_state.num_questions = 0

# --- App Logic (State Machine) ---

# --- STAGE 0: Start (File Upload) ---
if st.session_state.stage == 'start':
    st.info("Welcome! Please paste your resume text below to begin.")
    st.toggle(
        "Enable Audio Mode (AI Voice & Microphone)", 
        key='audio_enabled', 
        help="If ON, the AI will speak and you can answer with your voice. If OFF, it's text-only and you have to type your answer."
    )
    with st.form(key="resume_form"):
        resume_text_input = st.text_area("Paste your resume text here:", height=300)
        submit_button = st.form_submit_button("Start Interview")
    
    if submit_button and resume_text_input:
        if not resume_text_input.strip():
            st.error("Please paste your resume text.")
        else:
            # Save text to session state and move to the processing stage
            st.session_state.resume_text = resume_text_input
            st.session_state.stage = 'processing_resume'
            st.rerun()

# --- NEW STAGE 0.5: Process Resume (runs *after* file upload) ---
elif st.session_state.stage == 'processing_resume':
    with st.spinner("Analyzing your resume... This may take a moment."):
        try:
            resume_text = st.session_state.resume_text
            st.session_state.questions = generate_questions_from_resume(resume_text, gen_q_model)

            # 2. Get DUMMY AI Introduction
            intro_output = get_introduction(intro_model)
            
            st.session_state.current_question = intro_output.question
            intr_and_ques = f"{intro_output.intro}...{intro_output.question}"
            # 3. Move to next stage and display intro
            st.session_state.stage = 'awaiting_intro'
            
            text_to_speech_and_display(intr_and_ques)
            # text_to_speech_and_display(intro_output.question)
            
            # Clean up the resume text from session state
            if 'resume_text' in st.session_state:
                del st.session_state.resume_text
            st.rerun()
            # --- END: TEMPORARY TEST CODE ---
        
        except Exception as e:
            st.error(f"An error occurred during AI processing: {e}")
            st.session_state.stage = 'start'
# --- Main Interview Area (Stages > 0) ---
if st.session_state.stage not in ['start', 'processing_resume']:
    
    # --- Chat History Display ---
    st.subheader("Interview Transcript")
    chat_container = st.container(height=400) # Added height for scrolling
    with chat_container:
        for entry in reversed(st.session_state.chat_history):
            st.markdown(entry)

    try:
        st.divider()
    except Exception:
        st.markdown('---')

    # --- End Interview Button ---
    if st.button("End Interview", type="primary"):
        st.session_state.stage = 'finished'
        st.rerun()

    # --- REPLACEMENT: Text Input Area ---
    user_text = None # Initialize user_text
    is_disabled = (st.session_state.stage == 'finished')
    
    if mic_recorder is None:
        st.error("streamlit_mic_recorder library failed to import. Voice input is disabled.")
        st.info("Please add 'streamlit-mic-recorder' to your requirements.txt")
    
    elif is_disabled:
        st.info("Interview is finished. Start a new interview to speak.")
        
    else:
        if st.session_state.audio_enabled:
            st.write("Your turn to speak:")
            audio_bytes_dict = mic_recorder(
                start_prompt="Start Recording ⏺️",
                stop_prompt="Stop Recording ⏹️",
                key='recorder'
            )
            
            if audio_bytes_dict:
                # The component returns a dictionary, get the bytes
                audio_bytes = audio_bytes_dict['bytes']
                
                with st.spinner("Transcribing your answer..."):
                    # Use our NEW Google Cloud STT function
                    user_text = speech_to_text(audio_bytes)


        else:
        # --- TEXT-ONLY MODE (Text Input) ---
            with st.form(key="answer_form", clear_on_submit=True):
                answer = st.text_input("Your answer:", disabled=is_disabled)
                submit_button = st.form_submit_button(label="Submit Answer", disabled=is_disabled)
                
                if submit_button and answer:
                    user_text = answer
                    st.session_state.chat_history.append(f"**You:** {user_text}")
    # --- END OF REPLACEMENT ---


    # --- Process Submitted Text ---
    if user_text:
        # --- STAGE 1: Process User's Introduction ---
        if st.session_state.stage == 'awaiting_intro':
            with st.spinner("Thinking of a followup..."):
                followup = ask_followup(user_text, intro_model)
                st.session_state.current_question = followup
                text_to_speech_and_display(followup) # This now just displays text
                st.session_state.stage = 'awaiting_intro_followup'
                # st.rerun()
        
        # --- STAGE 2: Process Followup to Intro ---
        elif st.session_state.stage == 'awaiting_intro_followup':
            text_to_speech_and_display("OK, Great. Let's start the interview with questions from your resume.")
            st.session_state.stage = 'asking_question' # Move to main questions
            # st.rerun()

        # --- STAGE 4: Process Answer to a Main Question ---
        elif st.session_state.stage == 'awaiting_answer':
            with st.spinner("Evaluating your answer..."):
                question_asked = st.session_state.current_question
                # text_to_speech_and_display(question_asked)
                output = evaluate_answer(question_asked, user_text, eval_model)
                
                st.session_state.total_marks += output.marks
                st.session_state.num_questions += 1
                
                if output.review:
                    text_to_speech_and_display(output.review) # This now just displays text
                
                if output.followup:
                    st.session_state.current_question = output.followup
                    text_to_speech_and_display(output.followup) # This now just displays text
                    st.session_state.stage = 'awaiting_followup_answer'
                else:
                    st.session_state.q_index += 1
                    st.session_state.stage = 'asking_question'
                # st.rerun()

        # --- STAGE 5: Process Answer to a Followup Question ---
        elif st.session_state.stage == 'awaiting_followup_answer':
             with st.spinner("Evaluating your answer..."):
                question_asked = st.session_state.current_question
                output = evaluate_answer(question_asked, user_text, eval_model)
                
                st.session_state.total_marks += output.marks
                st.session_state.num_questions += 1
                
                if output.review:
                    text_to_speech_and_display(output.review) # This now just displays text
                
                st.session_state.q_index += 1
                st.session_state.stage = 'asking_question'
                # st.rerun()

    # --- STAGE 3: Ask a New Question ---
    # This runs when the page loads into this state, *before* user input
    if st.session_state.stage == 'asking_question':
        if st.session_state.q_index < len(st.session_state.questions):
            question = st.session_state.questions[st.session_state.q_index]
            st.session_state.current_question = question
            text_to_speech_and_display(question) # This now just displays text
            st.session_state.stage = 'awaiting_answer'
        else:
            text_to_speech_and_display("That's all the questions I have. Thank you!")
            st.session_state.stage = 'finished'
            st.rerun()

    # --- STAGE 6: Finished ---
    if st.session_state.stage == 'finished':
        st.balloons()
        st.success("Interview Complete!")
        
        final_score = 0
        if st.session_state.num_questions > 0:
            final_score = st.session_state.total_marks / st.session_state.num_questions
        
        st.subheader("Final Report")
        st.markdown(f"**Total Questions Answered:** {st.session_state.num_questions}")
        st.markdown(f"**Average Score:** {final_score:.2f} / 100")
        
        # Transcript is already shown above, but we can show it again
        st.subheader("Full Transcript")
        for entry in st.session_state.chat_history:
            st.markdown(entry)
            
        if st.button("Start New Interview"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()