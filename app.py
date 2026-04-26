"""
TalentScout Hiring Assistant - Main Streamlit Application
A conversational AI chatbot for initial candidate screening, powered by Groq.
"""

import streamlit as st
from datetime import datetime

from utils.groq_client import GroqClient
from utils.conversation import ConversationManager, Stage
from utils.validators import (
    validate_email,
    validate_phone,
    validate_name,
    validate_experience,
    is_exit_keyword,
)
from utils.storage import save_candidate
from utils.prompts import SYSTEM_PROMPT


# ----------------------------- Page config ---------------------------------- #
st.set_page_config(
    page_title="TalentScout Hiring Assistant",
    page_icon="🧑‍💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ----------------------------- Custom styling ------------------------------- #
st.markdown(
    """
    <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid #4F46E5;
            margin-bottom: 1.5rem;
        }
        .main-header h1 {
            color: #4F46E5;
            margin-bottom: 0.25rem;
        }
        .main-header p {
            color: #6B7280;
            font-size: 0.95rem;
            margin: 0;
        }
        .stage-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            background: #EEF2FF;
            color: #4F46E5;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .privacy-note {
            background: #F9FAFB;
            border-left: 3px solid #4F46E5;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            color: #4B5563;
            border-radius: 4px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Session state -------------------------------- #
def init_session_state():
    """Initialize Streamlit session state on first run."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationManager()
    if "groq_client" not in st.session_state:
        st.session_state.groq_client = None
    if "session_started" not in st.session_state:
        st.session_state.session_started = False
    if "session_ended" not in st.session_state:
        st.session_state.session_ended = False


init_session_state()


# ----------------------------- Sidebar -------------------------------------- #
with st.sidebar:
    st.header("⚙️ Configuration")

    # Read API key — prefer secrets, fall back to manual input.
    default_key = ""
    try:
        default_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        default_key = ""

    api_key = st.text_input(
        "Groq API Key",
        value=default_key,
        type="password",
        help="Get your free API key from https://console.groq.com/keys",
    )

    model = st.selectbox(
        "Model",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
        ],
        index=0,
        help="Groq-hosted models available on the free tier.",
    )

    st.markdown("---")
    st.markdown("### 📊 Progress")
    convo: ConversationManager = st.session_state.conversation
    progress = convo.progress()
    st.progress(progress, text=f"{int(progress * 100)}% complete")
    st.caption(f"Current stage: **{convo.stage.value.replace('_', ' ').title()}**")

    st.markdown("---")
    st.markdown(
        """
        <div class="privacy-note">
        🔒 <strong>Privacy:</strong> Your data is stored locally in anonymized form
        for this demo. Type <code>exit</code>, <code>quit</code>, or
        <code>bye</code> at any time to end the conversation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Start New Session", use_container_width=True):
        for key in [
            "messages",
            "conversation",
            "session_started",
            "session_ended",
        ]:
            st.session_state.pop(key, None)
        st.rerun()


# ----------------------------- Header --------------------------------------- #
st.markdown(
    """
    <div class="main-header">
        <h1>🧑‍💼 TalentScout Hiring Assistant</h1>
        <p>Your AI-powered first interview, available 24/7</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ----------------------------- Guard: API key ------------------------------- #
if not api_key:
    st.warning("👈 Please enter your **Groq API key** in the sidebar to begin.")
    st.info(
        "Don't have one yet? Get a **free** key at "
        "[console.groq.com/keys](https://console.groq.com/keys). "
        "Groq offers a generous free tier — perfect for this demo."
    )
    st.stop()


# ----------------------------- Initialize client ---------------------------- #
if (
    st.session_state.groq_client is None
    or st.session_state.groq_client.api_key != api_key
    or st.session_state.groq_client.model != model
):
    try:
        st.session_state.groq_client = GroqClient(api_key=api_key, model=model)
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}")
        st.stop()


# ----------------------------- Initial greeting ----------------------------- #
if not st.session_state.session_started:
    greeting = (
        "👋 Hello! I'm **TalentBot**, the AI hiring assistant for **TalentScout**, "
        "a recruitment agency specializing in technology placements.\n\n"
        "I'll be conducting your initial screening today. The process is quick "
        "(about 5–10 minutes) and covers:\n\n"
        "1. **A few quick questions** about you and your background\n"
        "2. **Your tech stack** — the technologies you're proficient in\n"
        "3. **3–5 short technical questions** tailored to your stack\n\n"
        "You can type `exit`, `quit`, or `bye` at any time to end the conversation.\n\n"
        "Ready to get started? **What is your full name?**"
    )
    st.session_state.messages.append({"role": "assistant", "content": greeting})
    st.session_state.session_started = True


# ----------------------------- Render history ------------------------------- #
for msg in st.session_state.messages:
    avatar = "🧑‍💼" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


# ----------------------------- Stage handlers ------------------------------- #
def handle_user_input(user_input: str) -> str:
    """
    Route user input through the current conversation stage.

    Each information-gathering stage validates the input deterministically
    (no LLM call needed for simple field collection — this is faster, more
    reliable, and saves tokens). LLM calls are reserved for technical question
    generation, evaluation, and free-form fallback.
    """
    convo: ConversationManager = st.session_state.conversation
    client: GroqClient = st.session_state.groq_client
    stage = convo.stage

    # Information-gathering stages: deterministic validation
    if stage == Stage.NAME:
        ok, cleaned, err = validate_name(user_input)
        if not ok:
            return f"⚠️ {err}\n\nCould you please share your full name? (e.g., *Jane Doe*)"
        convo.candidate.full_name = cleaned
        convo.advance()
        return (
            f"Nice to meet you, **{cleaned.split()[0]}**! 🤝\n\n"
            "What's the **best email address** to reach you at?"
        )

    if stage == Stage.EMAIL:
        ok, cleaned, err = validate_email(user_input)
        if not ok:
            return f"⚠️ {err}\n\nPlease provide a valid email address (e.g., *jane@example.com*)."
        convo.candidate.email = cleaned
        convo.advance()
        return "Got it. ✅ And your **phone number**? (Include country code if outside India.)"

    if stage == Stage.PHONE:
        ok, cleaned, err = validate_phone(user_input)
        if not ok:
            return f"⚠️ {err}\n\nPlease provide a valid phone number (e.g., *+91 98765 43210*)."
        convo.candidate.phone = cleaned
        convo.advance()
        return "Thanks! How many **years of professional experience** do you have? (e.g., *3* or *5.5*)"

    if stage == Stage.EXPERIENCE:
        ok, cleaned, err = validate_experience(user_input)
        if not ok:
            return f"⚠️ {err}\n\nPlease enter a number between 0 and 60 (e.g., *2* or *3.5*)."
        convo.candidate.years_experience = cleaned
        convo.advance()
        return (
            f"{cleaned} years — great. 💼\n\n"
            "Which **position(s)** are you interested in? "
            "(e.g., *Backend Engineer*, *ML Engineer*, *Full-Stack Developer*)"
        )

    if stage == Stage.POSITION:
        if len(user_input.strip()) < 2:
            return "⚠️ Could you share at least one role you're interested in?"
        convo.candidate.desired_position = user_input.strip()
        convo.advance()
        return (
            "Noted! 📍 What's your **current location**? (City, Country)"
        )

    if stage == Stage.LOCATION:
        if len(user_input.strip()) < 2:
            return "⚠️ Please share your city, or city and country."
        convo.candidate.location = user_input.strip()
        convo.advance()
        return (
            "Almost done with the basics! 🛠️\n\n"
            "Now, please tell me about your **tech stack** — the programming "
            "languages, frameworks, databases, and tools you're proficient in. "
            "List as many as you'd like, separated by commas.\n\n"
            "*Example: Python, Django, PostgreSQL, Docker, AWS*"
        )

    if stage == Stage.TECH_STACK:
        # Parse comma/newline separated tech stack
        raw = user_input.replace("\n", ",")
        techs = [t.strip() for t in raw.split(",") if t.strip()]
        if not techs:
            return "⚠️ Please list at least one technology you work with."
        if len(techs) > 25:
            return "⚠️ That's a lot! Please pick your **top 10–15** most-relevant technologies."

        convo.candidate.tech_stack = techs
        convo.advance()

        # Generate technical questions via LLM
        with st.spinner("Generating personalized technical questions..."):
            try:
                questions = client.generate_technical_questions(
                    tech_stack=techs,
                    years_experience=convo.candidate.years_experience or 0,
                    position=convo.candidate.desired_position or "",
                )
                convo.questions = questions
            except Exception as e:
                return (
                    f"⚠️ I had trouble generating questions ({e}). "
                    "Could you re-list your tech stack, or try again in a moment?"
                )

        # Present the first technical question
        first_q = convo.next_question()
        return (
            f"Excellent — **{', '.join(techs[:5])}"
            f"{'...' if len(techs) > 5 else ''}** is a solid stack! 🎯\n\n"
            f"I've prepared **{len(questions)} short technical questions** "
            "tailored to your skills and experience level. Take your time, "
            "and answer in your own words.\n\n"
            f"---\n\n**Question 1 of {len(questions)}:**\n\n{first_q}"
        )

    if stage == Stage.TECHNICAL_QA:
        # Store the answer for the current question
        convo.record_answer(user_input)

        next_q = convo.next_question()
        if next_q is not None:
            idx = convo.current_q_index  # 1-indexed after record + advance
            total = len(convo.questions)
            return f"Thanks for that answer. 👍\n\n---\n\n**Question {idx} of {total}:**\n\n{next_q}"

        # All questions answered
        convo.advance()
        return (
            "🎉 **That was the last one!**\n\n"
            "Thanks for your thoughtful answers. Before we wrap up — "
            "is there anything else you'd like the TalentScout team to know? "
            "(Type *no* or *done* to finish.)"
        )

    if stage == Stage.WRAP_UP:
        notes = user_input.strip()
        if notes.lower() not in {"no", "nothing", "nope", "done", "n", "na", "n/a"}:
            convo.candidate.additional_notes = notes
        convo.advance()

        # Persist candidate (anonymized)
        try:
            record_id = save_candidate(convo.candidate.to_dict())
        except Exception:
            record_id = "unsaved"

        first_name = (convo.candidate.full_name or "there").split()[0]
        return (
            f"Thank you so much, **{first_name}**! 🙌\n\n"
            "Your screening is complete. Here's what happens next:\n\n"
            "1. ✅ Our recruitment team will review your responses within "
            "**2–3 business days**.\n"
            "2. 📧 If your profile matches an open role, you'll receive an "
            "email at the address you provided to schedule a follow-up.\n"
            "3. 🔒 Your data is stored securely and only used for recruitment "
            "purposes, in line with GDPR principles.\n\n"
            f"*Reference ID: `{record_id}`*\n\n"
            "Best of luck with your job search! You can close this window or "
            "type `bye` to exit."
        )

    # Conversation already ended
    return (
        "Our screening session is complete. ✅ "
        "Click **Start New Session** in the sidebar to begin a fresh interview, "
        "or simply close this window. Take care!"
    )


# ----------------------------- Chat input ----------------------------------- #
if not st.session_state.session_ended:
    user_input = st.chat_input("Type your response here...")

    if user_input:
        # Echo user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        # Check exit keywords first — highest priority
        if is_exit_keyword(user_input):
            farewell = (
                "👋 No problem — thanks for stopping by **TalentScout**! "
                "Feel free to come back any time to complete your screening. "
                "Have a great day!"
            )
            st.session_state.messages.append({"role": "assistant", "content": farewell})
            st.session_state.session_ended = True
            with st.chat_message("assistant", avatar="🧑‍💼"):
                st.markdown(farewell)
            st.stop()

        # Otherwise route through stage handler
        try:
            response = handle_user_input(user_input)
        except Exception as e:
            response = (
                f"😕 Something went wrong on my end: `{e}`. "
                "Could you rephrase or try again?"
            )

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant", avatar="🧑‍💼"):
            st.markdown(response)

        # If the wrap-up just finished, mark ended
        if st.session_state.conversation.stage == Stage.ENDED:
            st.session_state.session_ended = True
            st.rerun()
else:
    st.success("Session complete. Click **Start New Session** in the sidebar to begin again.")
