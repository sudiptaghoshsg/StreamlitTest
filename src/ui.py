import streamlit as st
import os
import json
import time # For polling audio capture status
import numpy as np # For checking audio data (though not directly used in this version)
from dotenv import load_dotenv
from typing import Optional
import re

# Adjust import paths
try:
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration # Import audio modules
    from src.utils import HealHubUtilities
except ImportError:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.nlu_processor import SarvamMNLUProcessor, HealthIntent, NLUResult
    from src.response_generator import HealHubResponseGenerator
    from src.symptom_checker import SymptomChecker
    from src.audio_capture import CleanAudioCapture, SarvamSTTIntegration
    from src.utils import HealHubUtilities

# --- Environment and API Key Setup ---
load_dotenv()
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

# --- Session State Initialization ---
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_language_display' not in st.session_state: 
    st.session_state.current_language_display = 'English'
if 'current_language_code' not in st.session_state: 
    st.session_state.current_language_code = 'en-IN'
if 'text_query_input_area' not in st.session_state:
    st.session_state.text_query_input_area = ""

# Symptom Checker states
if 'symptom_checker_active' not in st.session_state:
    st.session_state.symptom_checker_active = False
if 'symptom_checker_instance' not in st.session_state:
    st.session_state.symptom_checker_instance = None
if 'pending_symptom_question_data' not in st.session_state:
    st.session_state.pending_symptom_question_data = None

# Voice Input states
if 'voice_input_stage' not in st.session_state:
    # Stages: None, "arming", "recording", "transcribing", "processing_stt"
    st.session_state.voice_input_stage = None 
if 'audio_capturer' not in st.session_state: 
    st.session_state.audio_capturer = None
if 'captured_audio_data' not in st.session_state:
    st.session_state.captured_audio_data = None

# --- Language Mapping ---
LANGUAGE_MAP = {
    "English": "en-IN", 
    "हिन्दी (Hindi)": "hi-IN", 
    "বাংলা (Bengali)": "bn-IN", 
    "मराठी (Marathi)": "mr-IN", 
    "ಕನ್ನಡ (Kannada)": "kn-IN",
    "தமிழ் (Tamil)": "ta-IN",
    "తెలుగు (Telugu)": "te-IN",
    "മലയാളം (Malayalam)": "ml-IN",
}

DISPLAY_LANGUAGES = list(LANGUAGE_MAP.keys())

# --- Helper Functions ---
def add_message_to_conversation(role: str, content: str, lang_code: Optional[str] = None):
    message = {"role": role, "content": content}
    if lang_code and role == "user":
        message["lang"] = lang_code 
    st.session_state.conversation.append(message)

def process_and_display_response(user_query_text: str, lang_code: str):
    if not SARVAM_API_KEY:
        st.error("API Key not configured.")
        add_message_to_conversation("system", "Error: API Key not configured.")
        st.session_state.voice_input_stage = None # Reset voice stage on error
        return

    nlu_processor = SarvamMNLUProcessor(api_key=SARVAM_API_KEY)
    response_gen = HealHubResponseGenerator(api_key=SARVAM_API_KEY)
    util = HealHubUtilities(api_key=SARVAM_API_KEY)
    user_lang = st.session_state.current_language_code
    try:
        # User message is now added *before* calling this function for both text and voice.
        # So, this function should not add the user message again.
        
        with st.spinner("🧠 Thinking..."):
            nlu_output: NLUResult = nlu_processor.process_transcription(user_query_text, source_language=lang_code)
            if nlu_output.intent == HealthIntent.SYMPTOM_QUERY and not nlu_output.is_emergency:
                st.session_state.symptom_checker_active = True
                st.session_state.symptom_checker_instance = SymptomChecker(nlu_result=nlu_output, api_key=SARVAM_API_KEY)
                st.session_state.symptom_checker_instance.prepare_follow_up_questions()
                st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
                if st.session_state.pending_symptom_question_data:
                    question_to_ask_raw = st.session_state.pending_symptom_question_data['question']
                    symptom_context_raw = st.session_state.pending_symptom_question_data['symptom_name']
                    question_to_ask_translated = util.translate_text(question_to_ask_raw, user_lang)
                    symptom_context_translated = util.translate_text(symptom_context_raw, user_lang)
                    add_message_to_conversation("assistant", f"{question_to_ask_translated}: {symptom_context_translated}")
                else:
                    generate_and_display_assessment()
            else:
                bot_response = response_gen.generate_response(user_query_text, nlu_output)
                translated_bot_response = util.translate_text(bot_response, user_lang)
                add_message_to_conversation("assistant", translated_bot_response)
                st.session_state.symptom_checker_active = False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Standardized error message
        add_message_to_conversation("system", f"Sorry, an error occurred while processing your request. Please try rephrasing or try again later. (Details: {str(e)})")
        st.session_state.symptom_checker_active = False # Reset states on error
        st.session_state.symptom_checker_instance = None
        st.session_state.pending_symptom_question_data = None
    finally:
        st.session_state.voice_input_stage = None # Always reset voice stage after processing or error

def handle_follow_up_answer(answer_text: str):
    util = HealHubUtilities(api_key=SARVAM_API_KEY)
    user_lang = st.session_state.current_language_code
    if st.session_state.symptom_checker_instance and st.session_state.pending_symptom_question_data:
        # Add user's follow-up answer to conversation log
        add_message_to_conversation("user", answer_text, lang_code=st.session_state.current_language_code.split('-')[0])
        
        question_asked = st.session_state.pending_symptom_question_data['question']
        symptom_name = st.session_state.pending_symptom_question_data['symptom_name']
        with st.spinner("Recording answer..."):
            st.session_state.symptom_checker_instance.record_answer(symptom_name, question_asked, answer_text)
            st.session_state.pending_symptom_question_data = st.session_state.symptom_checker_instance.get_next_question()
        if st.session_state.pending_symptom_question_data:
            question_to_ask_raw = st.session_state.pending_symptom_question_data['question']
            symptom_context_raw = st.session_state.pending_symptom_question_data['symptom_name']
            question_to_ask_translated = util.translate_text(question_to_ask_raw, user_lang)
            symptom_context_translated = util.translate_text(symptom_context_raw, user_lang)
            add_message_to_conversation("assistant", f"{symptom_context_translated}: {question_to_ask_translated}")
        else:
            generate_and_display_assessment()
    else: 
        st.warning("No pending question to answer or symptom checker not active.")
        st.session_state.symptom_checker_active = False
    st.session_state.voice_input_stage = None # Reset voice stage

# New callback function for text submission
def handle_text_submission():
    user_input = st.session_state.text_query_input_area # Read from session state key
    current_lang_code = st.session_state.current_language_code

    if not user_input: # Do nothing if input is empty
        return

    # Add the current user input to conversation log REGARDLESS of whether it's new or follow-up
    
    if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data:
        # handle_follow_up_answer will process the answer.
        # It should NOT add the user message again as it's already added above.
        handle_follow_up_answer(user_input) 
    else: 
        add_message_to_conversation("user", user_input, lang_code=current_lang_code.split('-')[0])
        if st.session_state.symptom_checker_active: # Reset if symptom checker was active but no pending q
             st.session_state.symptom_checker_active = False 
             st.session_state.symptom_checker_instance = None
             st.session_state.pending_symptom_question_data = None
        # process_and_display_response will process the new query.
        # It should NOT add the user message again.
        process_and_display_response(user_input, current_lang_code)
    
    st.session_state.text_query_input_area = "" # Clear the text area state for next render
    # No explicit st.rerun() here, on_click handles it for buttons.
    # If called from a non-button context that needs immediate UI update, rerun might be needed.

def generate_and_display_assessment():
    util = HealHubUtilities(api_key=SARVAM_API_KEY)
    user_lang = st.session_state.current_language_code
    if st.session_state.symptom_checker_instance:
        with st.spinner("🔬 Generating preliminary assessment..."):
            assessment = st.session_state.symptom_checker_instance.generate_preliminary_assessment()
            try:
                assessment_str = f"### {util.translate_text('Preliminary Health Assessment', user_lang)}:\n\n"
                assessment_str += f"**{util.translate_text('Summary', user_lang)}:** {util.translate_text(assessment.get('assessment_summary', 'N/A'), user_lang)}\n\n"
                assessment_str += f"**{util.translate_text('Suggested Severity', user_lang)}:** {util.translate_text(assessment.get('suggested_severity', 'N/A'), user_lang)}\n\n"
                assessment_str += f"**{util.translate_text('Recommended Next Steps', user_lang)}:**\n"
                next_steps = assessment.get('recommended_next_steps', 'N/A')
                if isinstance(next_steps, list): 
                    for step in next_steps: assessment_str += f"- {util.translate_text(step, user_lang)}\n"
                elif isinstance(next_steps, str): # This is the block to modify
                    ### Replace the original problematic f-string line here
                    # Split on punctuation marks (., !, ?) followed by whitespace
                    sentences = re.split(r'(?<=[.!?])\s+', next_steps.strip())
                    # Add bullet to each sentence
                    temp_steps = '\n- '.join(sentences).strip()
                    # remove leading bullet if present (e.g. if next_steps started with punctuation)
                    temp_steps = temp_steps.lstrip('- ')
                    # Append to assessment_str
                    assessment_str += f"{util.translate_text(temp_steps, user_lang)}\n"
                else: 
                    assessment_str += f"- {util.translate_text('N/A', user_lang)}\n"
                warnings = assessment.get('potential_warnings')
                if warnings and isinstance(warnings, list) and len(warnings) > 0 :
                    assessment_str += f"\n**{util.translate_text('Potential Warnings', user_lang)}:**\n"
                    for warning in warnings: assessment_str += f"- {util.translate_text(warning, user_lang)}\n"
                kb_points = assessment.get('relevant_kb_triage_points')
                if kb_points and isinstance(kb_points, list) and len(kb_points) > 0:
                    assessment_str += f"\n**{util.translate_text('Relevant Triage Points from Knowledge Base', user_lang)}:**\n"
                    for point in kb_points: assessment_str += f"- {util.translate_text(point, user_lang)}\n"
                assessment_str += f"\n\n**{util.translate_text('Disclaimer', user_lang)}:** {util.translate_text(assessment.get('disclaimer', 'Always consult a doctor for medical advice.'), user_lang)}"
                add_message_to_conversation("assistant", assessment_str)
            except Exception as e:
                st.error(f"Error formatting assessment: {e}")
                try:
                    raw_assessment_json = json.dumps(assessment, indent=2)
                    add_message_to_conversation("assistant", f"Could not format assessment. Raw data:\n```json\n{raw_assessment_json}\n```")
                except Exception as json_e:
                    add_message_to_conversation("assistant", f"Could not format or serialize assessment: {json_e}")
        st.session_state.symptom_checker_active = False
        st.session_state.symptom_checker_instance = None
        st.session_state.pending_symptom_question_data = None
    st.session_state.voice_input_stage = None # Reset voice stage

# --- Streamlit UI ---
def main_ui():
    st.set_page_config(page_title="HealHub Assistant", layout="wide", initial_sidebar_state="collapsed")
    # st.caption("Your AI healthcare companion. Supporting English and Popular Indic Languages.")

    # Enhanced Recording Visual Cue
    if st.session_state.voice_input_stage == "recording":
        st.info("🔴 Recording audio... Speak now. Silence or the 'Stop' button will end recording.", icon="🎤")

    if not SARVAM_API_KEY: 
        st.error("🚨 SARVAM_API_KEY not found. Please set it in your .env file for the application to function.")
        st.stop()

    col1, col2 = st.columns([6, 6])
    with col1:
        st.title("💬 HealHub Assistant")
        st.caption("Your AI healthcare companion. Supporting English and Popular Indic Languages.")
    with col2:
        st.markdown(f"<div style='height: 40px;'></div>", unsafe_allow_html=True)
        selected_lang_display = st.selectbox(
            "Select Language / भाषा चुनें / ভাষা নির্বাচন করুন / भाषा निवडा:",
            options=DISPLAY_LANGUAGES,
            index=DISPLAY_LANGUAGES.index(st.session_state.current_language_display),
            key='language_selector_widget' 
        )
    if selected_lang_display != st.session_state.current_language_display:
        st.session_state.current_language_display = selected_lang_display
        st.session_state.current_language_code = LANGUAGE_MAP[selected_lang_display]
        st.session_state.conversation = [] 
        st.session_state.symptom_checker_active = False; st.session_state.symptom_checker_instance = None; st.session_state.pending_symptom_question_data = None
        st.session_state.voice_input_stage = None
        st.rerun()

    current_lang_code_for_query = st.session_state.current_language_code

    st.markdown("### Conversation")
    chat_container = st.container(height=300) 
    with chat_container:
        util = HealHubUtilities(api_key=SARVAM_API_KEY)
        user_lang = st.session_state.current_language_code
        idx = 0
        for msg_data in st.session_state.conversation:
            idx += 1
            role = msg_data.get("role", "system"); content = msg_data.get("content", "")
            avatar = "🧑‍💻" if role == "user" else "⚕️"
            if role == "user":
                with st.chat_message(role, avatar=avatar):
                    lang_display = msg_data.get('lang', st.session_state.current_language_code.split('-')[0])
                    st.markdown(f"{content} *({lang_display})*")
            elif role == "assistant":
                with st.chat_message(role, avatar=avatar): 
                    st.markdown(content) 
                    speak_key = f"speak_{idx}"
                    if st.button("🔊 Speak", key=speak_key):
                        with st.spinner("Synthesizing speech..."):
                            audio_bytes = util.synthesize_speech(content, user_lang)
                            st.audio(audio_bytes, format="audio/wav")
            else:
                with st.chat_message(role, avatar="ℹ️"): st.markdown(content)

    is_recording = st.session_state.voice_input_stage == "recording"
    
    # Define column width ratios for the input area, send button, and voice recording button
    COLUMN_WIDTHS = [10, 1, 1]
    col1, col2, col3 = st.columns(COLUMN_WIDTHS)

    with col1:
        input_label = "Type your answer here..." if st.session_state.symptom_checker_active and st.session_state.pending_symptom_question_data else "Type your health query here..."
    
        # Text area widget - its current value is stored in st.session_state.text_query_input_area due to its key
        st.text_area(input_label, height=100, key="text_query_input_area", disabled=is_recording)
        
        # user_input = st.text_input("Type your query or use voice:", key="user_input")

    with col2:
        st.markdown(f"<div style='height: 55px;'></div>", unsafe_allow_html=True)
        st.button(
            "📤 Send", 
            use_container_width=True, 
            key="send_button_widget", # Key can be kept if useful for other logic, or removed
            disabled=is_recording,
            on_click=handle_text_submission # Assign the callback
        )

    with col3:
        st.markdown(f"<div style='height: 55px;'></div>", unsafe_allow_html=True)
        record_voice_button_text = "⏹️ Stop" if st.session_state.voice_input_stage == "recording" else "🎙️ Record"
        record_voice_button = st.button(record_voice_button_text, use_container_width=True, key="record_voice_button_widget")


    # --- Voice Input Logic (remains largely the same, but text submission is handled by callback) ---
    if record_voice_button: # This handles the click of the voice button
        if is_recording: 
            if st.session_state.audio_capturer:
                st.session_state.audio_capturer.stop_recording() 
            st.session_state.voice_input_stage = "transcribing" 
            st.rerun()
        else: 
            st.session_state.voice_input_stage = "recording"
            st.session_state.captured_audio_data = None 
            if st.session_state.audio_capturer is None: 
                try:
                    print("Initializing CleanAudioCapture...")
                    st.session_state.audio_capturer = CleanAudioCapture(sample_rate=48000)
                    print("CleanAudioCapture initialized successfully")
                except Exception as init_error:
                    st.error(f"Failed to initialize audio capture: {init_error}")
                    print(f"Detailed initialization error: {init_error}")
                    st.session_state.voice_input_stage = None
                    st.rerun()
            try:
                print("Starting audio recording...")
                st.session_state.audio_capturer.start_recording()
                print("Audio recording started successfully")
                if not st.session_state.conversation or st.session_state.conversation[-1].get("content") != "🎤 Voice recording started... Speak now. Silence will stop it, or click 'Stop & Process'.":
                    add_message_to_conversation("system", "🎤 Voice recording started... Speak now. Silence will stop it, or click 'Stop & Process'.")

            except Exception as e:
                st.error(f"Failed to start recording: {e}. Ensure microphone is connected and permissions are granted.")
                add_message_to_conversation("system", f"Error: Could not start voice recording. Please check microphone permissions. (Details: {e})")
                st.session_state.voice_input_stage = None
            st.rerun()

    # State machine for voice processing (remains the same)
    if st.session_state.voice_input_stage == "recording":
        if st.session_state.audio_capturer and not st.session_state.audio_capturer.is_recording:
            st.session_state.voice_input_stage = "transcribing"
            st.rerun()
        else:
            time.sleep(0.1) 
            if st.session_state.audio_capturer and not st.session_state.audio_capturer.is_recording: # Check again
                 st.session_state.voice_input_stage = "transcribing"
            st.rerun() 

    if st.session_state.voice_input_stage == "transcribing":
        cleaned_audio = None
        if st.session_state.audio_capturer:
            cleaned_audio = st.session_state.audio_capturer.get_cleaned_audio()
        if cleaned_audio is not None and len(cleaned_audio) > 0:
            if not st.session_state.conversation or st.session_state.conversation[-1].get("content") != "🎙️ Audio captured. Transcribing...":
                add_message_to_conversation("system", "🎙️ Audio captured. Transcribing...")
            st.session_state.captured_audio_data = cleaned_audio 
            st.session_state.voice_input_stage = "processing_stt"
        else:
            add_message_to_conversation("system", "⚠️ No valid audio captured. Please try again.")
            st.session_state.voice_input_stage = None
        st.rerun()

    if st.session_state.voice_input_stage == "processing_stt":
        if st.session_state.captured_audio_data is not None:
            stt_service = SarvamSTTIntegration(api_key=SARVAM_API_KEY)
            lang_for_stt = st.session_state.current_language_code 
            try:
                with st.spinner("Transcribing audio..."):
                    stt_result = stt_service.transcribe_audio(
                        st.session_state.captured_audio_data, sample_rate=16000, source_language=lang_for_stt
                    )
                transcribed_text = stt_result.get("transcription")
                if transcribed_text and transcribed_text.strip():
                    add_message_to_conversation("user", transcribed_text, lang_code=lang_for_stt.split('-')[0])
                    process_and_display_response(transcribed_text, lang_for_stt) 
                else:
                    add_message_to_conversation("system", "⚠️ STT failed to transcribe audio or returned empty. Please try again.")
            except Exception as e:
                st.error(f"STT Error: {e}")
                add_message_to_conversation("system", f"Sorry, an error occurred during voice transcription. Please try again. (Details: {e})")
            st.session_state.captured_audio_data = None 
            st.session_state.voice_input_stage = None 
            st.rerun()
        else: 
            st.session_state.voice_input_stage = None
            st.rerun()

    # The old `if send_button and user_query_text_from_area:` block is now removed,
    # as its logic is handled by the handle_text_submission callback.

if __name__ == "__main__":
    main_ui()
