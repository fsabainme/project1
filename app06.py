import streamlit as st
import asyncio
import nest_asyncio
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled

nest_asyncio.apply()
set_tracing_disabled(True)

st.set_page_config(page_title="AI Linear Algebra Tutor", page_icon="📐", layout="wide")

st.title("📐 AI Linear Algebra Tutor")
st.caption("Personalized learning, guided practice, and conceptual understanding in linear algebra.")

LEARNING_PATH = [
    {
        "module": "Vectors & Geometry",
        "goals": [
            "Understand vectors, magnitude, and direction",
            "Perform vector addition and scalar multiplication",
            "Interpret vectors geometrically",
        ],
        "skills": ["vector arithmetic", "dot product", "projection"],
    },
    {
        "module": "Matrices & Systems",
        "goals": [
            "Represent systems of equations as matrices",
            "Row-reduce using Gaussian elimination",
            "Interpret solutions (unique, infinite, none)",
        ],
        "skills": ["row reduction", "matrix multiplication", "inverse"],
    },
    {
        "module": "Linear Transformations",
        "goals": [
            "Connect matrices with transformations",
            "Visualize transformations in 2D/3D",
            "Understand composition and inverse transformations",
        ],
        "skills": ["transformations", "composition", "change of basis"],
    },
    {
        "module": "Vector Spaces & Bases",
        "goals": [
            "Define vector spaces and subspaces",
            "Identify spans, linear independence, and bases",
            "Compute dimension and basis changes",
        ],
        "skills": ["span", "linear independence", "basis"],
    },
    {
        "module": "Eigenvalues & Eigenvectors",
        "goals": [
            "Solve characteristic equations",
            "Compute eigenvalues/eigenvectors",
            "Interpret eigenspaces and diagonalization",
        ],
        "skills": ["characteristic polynomial", "diagonalization"],
    },
]

PRACTICE_BANK = {
    "Vectors & Geometry": [
        "Compute the dot product of v = (2, -1) and w = (3, 4).",
        "Find the projection of v = (3, 1) onto w = (2, 2).",
    ],
    "Matrices & Systems": [
        "Row-reduce the augmented matrix [[1, 2, 5], [2, 1, 4]].",
        "Determine if the matrix [[1, 2], [2, 4]] is invertible.",
    ],
    "Linear Transformations": [
        "Describe the transformation for the matrix [[0, -1], [1, 0]].",
        "What is the composition of scaling by 2 and rotation by 90°?",
    ],
    "Vector Spaces & Bases": [
        "Is {(1, 0, 1), (2, 1, 3)} linearly independent?",
        "Find a basis for the span of {(1, 0), (2, 3)}.",
    ],
    "Eigenvalues & Eigenvectors": [
        "Find eigenvalues of A = [[2, 1], [1, 2]].",
        "Determine if A = [[4, 1], [0, 2]] is diagonalizable.",
    ],
}

TUTOR_INSTRUCTIONS = """
You are an AI linear algebra tutor. Follow a structured learning plan:
1) Diagnose the learner's goal, level, and misconceptions.
2) Teach conceptually with geometric intuition and clear steps.
3) Use Socratic questioning before giving answers.
4) Provide short, accurate explanations with checks for understanding.
5) Offer practice problems and tailored hints.
6) Track progress and recommend the next module.
7) Avoid giving full solutions unless asked twice or in "Explain" mode.
"""

# Initialize session state variables
if "agent" not in st.session_state:
    st.session_state.agent = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "progress" not in st.session_state:
    st.session_state.progress = {module["module"]: False for module in LEARNING_PATH}
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

with st.sidebar:
    st.header("🔧 Tutor Configuration")
    api_key = st.text_input("GEMINI API Key", type="password")
    base_url = st.text_input(
        "Base URL",
        value="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model_name = st.text_input("Model", value="gemini-2.0-flash")

    st.divider()
    st.subheader("👩‍🎓 Learner Profile")
    learner_level = st.selectbox("Current level", ["High school", "Intro college", "Advanced"])
    learning_goal = st.text_input("Goal (e.g., exam prep, homework, intuition)")
    tutoring_mode = st.radio("Mode", ["Socratic", "Explain", "Practice"], horizontal=True)

    st.divider()
    st.subheader("📚 Module Focus")
    module_choice = st.selectbox("Focus module", [m["module"] for m in LEARNING_PATH])
    difficulty = st.select_slider("Practice difficulty", ["Warm-up", "Standard", "Challenge"])

    st.divider()
    if st.button("Initiate Tutor"):
        if not api_key:
            st.error("❌ API key is required.")
        else:
            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
            agent = Agent(
                name="Linear Algebra Tutor",
                instructions=TUTOR_INSTRUCTIONS,
                model=OpenAIChatCompletionsModel(model=model_name, openai_client=client),
            )
            st.session_state.agent = agent
            st.success("✅ Tutor initiated successfully!")

    st.divider()
    st.subheader("🚀 Quick Start")
    st.write(
        "Start a guided session with a ready-made prompt aligned to your module and mode."
    )
    if st.button("Start Module Session"):
        if not st.session_state.agent:
            st.warning("Initiate the tutor first to begin a session.")
        else:
            st.session_state.pending_prompt = (
                "Start a short lesson with a diagnostic question and a worked example."
            )
            st.success("Session prompt queued. Check the chat window.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Tutor Chat")
    if st.session_state.agent:
        if st.session_state.pending_prompt:
            user_input = st.session_state.pending_prompt
            st.session_state.pending_prompt = None
        else:
            user_input = None
        for sender, message in st.session_state.chat_history:
            with st.chat_message(sender.lower()):
                st.markdown(message)

        user_chat = st.chat_input("Ask a question or request a practice problem...")
        if user_chat:
            user_input = user_chat

        if user_input:
            st.session_state.chat_history.append(("User", user_input))
            with st.chat_message("user"):
                st.markdown(user_input)

            async def run_agent():
                context = (
                    f"Learner level: {learner_level}\n"
                    f"Goal: {learning_goal or 'Not specified'}\n"
                    f"Mode: {tutoring_mode}\n"
                    f"Module focus: {module_choice}\n"
                    f"Difficulty: {difficulty}\n"
                    f"Provide responses aligned with the selected mode and module."
                )
                prompt = f"{context}\n\nStudent message: {user_input}"
                return await Runner.run(st.session_state.agent, prompt)

            try:
                loop = asyncio.get_event_loop()
                response = loop.run_until_complete(run_agent())
                st.session_state.chat_history.append(("Assistant", response.final_output))

                with st.chat_message("assistant"):
                    st.markdown(response.final_output)
            except Exception as e:
                with st.chat_message("assistant"):
                    st.error(f"Tutor error: {e}")
    else:
        st.info("Enter your credentials and initiate the tutor to start learning.")

with col2:
    st.subheader("🧭 Learning Plan")
    st.write(
        "Follow the structured path below. Mark modules as complete as you master them."
    )

    for module in LEARNING_PATH:
        module_name = module["module"]
        with st.expander(module_name, expanded=False):
            st.markdown("**Goals**")
            st.write("\n".join([f"- {goal}" for goal in module["goals"]]))
            st.markdown("**Key skills**")
            st.write(", ".join(module["skills"]))
            st.session_state.progress[module_name] = st.checkbox(
                "Mark complete", value=st.session_state.progress[module_name], key=module_name
            )

    st.divider()
    st.subheader("🎯 Recommended Practice")
    st.write(f"Module: **{module_choice}**")
    for item in PRACTICE_BANK.get(module_choice, []):
        st.markdown(f"- {item}")

    st.divider()
    st.subheader("📈 Progress Summary")
    completed = [m for m, done in st.session_state.progress.items() if done]
    st.write(f"Completed modules: {len(completed)} / {len(LEARNING_PATH)}")
    if completed:
        st.write(", ".join(completed))
    else:
        st.write("No modules completed yet.")
