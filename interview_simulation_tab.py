# /resume-app/interview_simulation_tab.py
import streamlit as st

from interview_simulator import InterviewSimulator
from utils import clear_thread, get_memory, get_thread
from langchain_core.messages import HumanMessage


simulation_thread_name = "interview_simulation_thread_id"


def _get_simulation_thread():
    """
    Retrieves the thread associated with the interview simulation.

    Returns:
        dict: The thread dictionary.
    """
    return get_thread(simulation_thread_name)


def _reset_simulation_state():
    """
    Resets the state of the interview simulation.

    Clears the thread associated with the simulation.
    """
    clear_thread(simulation_thread_name)


def _run(agent, thread):
    """
    Starts the interview simulation.

    This function initializes the interview simulation by invoking the agent's state graph
    with the necessary context, including the resume, persona, job description, and interview questions.

    Args:
        agent: The InterviewSimulator object.
        thread: The thread dictionary.
    """
    resume = st.session_state["app_state"]["resume"]
    persona = st.session_state["app_state"]["persona"]
    job_description = st.session_state["app_state"]["job_description"]
    interview_questions = st.session_state["app_state"]["interview_questions"]

    _ = agent.graph.invoke(
        {
            "resume": resume,
            "job_description": job_description,
            "persona": persona,
            "interview_questions": interview_questions,
        },
        thread,
    )


def _resume_with_state_update(agent: InterviewSimulator, thread: dict):
    """
    Resumes the interview simulation with updated state.

    This function updates the state of the interview simulation with the user's input
    and then resumes the simulation by invoking the agent's state graph.

    Args:
        agent: The InterviewSimulator object.
        thread: The thread dictionary.
    """
    user_input = str(st.session_state["interview_simulation_response"])

    agent.graph.update_state(
        thread,
        {"messages": [HumanMessage(content=user_input)]},
        as_node="ask_question",
    )
    # Now call with None to resume with changed state
    _ = agent.graph.invoke(None, thread)


def _user_response():
    """
    Presents a chat input for the user to provide their response.

    This function uses Streamlit's chat_input component to allow the user to enter their response
    to the interviewer's question.
    """
    st.chat_input(key="interview_simulation_response")


def _show_message(message):
    """
    Displays a message in the chat interface.

    This function renders a message in the Streamlit chat interface,
    distinguishing between messages from the user and the assistant (interviewer).

    Args:
        message: The message to be displayed.
    """
    if message.type == "human":
        st.chat_message("user").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)


def _show_messages(agent, thread):
    """
    Displays the messages exchanged during the interview simulation.

    This function retrieves the messages from the interview simulation's state
    and displays them in the Streamlit chat interface. It also checks if the interview has ended
    and presents a chat input for the user to provide their response if the interview is ongoing.

    Args:
        agent: The InterviewSimulator object.
        thread: The thread dictionary.
    """
    messages = agent.graph.get_state(thread).values["messages"]
    st.session_state["app_state"]["interview_session"] = messages
    has_ended = (
        "ended" in agent.graph.get_state(thread).values
        and agent.graph.get_state(thread).values["ended"]
    )

    for message in messages:
        _show_message(message)

    if len(messages) > 0 and not has_ended:
        _user_response()


def render_interview_simulation_tab():
    """
    Renders the interview simulation tab in the Streamlit app.

    This tab allows users to simulate an interview with an AI interviewer,
    using the pre-generated interview questions and persona. The user can interact with the AI
    by providing their responses to the questions, and the AI will provide feedback on the answers.
    """
    if (
        "app_state" not in st.session_state
        or "interview_questions" not in st.session_state["app_state"]
    ):
        st.write(
            "No application state found. Have you uploaded a resume and generated questions?"
        )
        return

    st.subheader("Interview Simulation")
    st.markdown(
        "AI plays role of interviewer, choosing a question from the pre-generated list of questions. "
        "When the candidate answers, AI then provides feedback on the answer."
    )

    inteviewer = InterviewSimulator(checkpointer=get_memory())
    start = st.button("Start Interview")
    if start:
        _reset_simulation_state()
        _run(inteviewer, _get_simulation_thread())
    elif (
        "interview_simulation_response" in st.session_state
        and st.session_state["interview_simulation_response"] is not None
    ):
        _resume_with_state_update(inteviewer, _get_simulation_thread())

    _show_messages(inteviewer, _get_simulation_thread())
