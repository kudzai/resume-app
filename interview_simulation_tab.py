import streamlit as st

from interview_simulator import InterviewSimulator
from utils import clear_thread, get_memory, get_thread
from langchain_core.messages import HumanMessage


simulation_thread_name = "interview_simulation_thread_id"


def _get_simulation_thread():
    return get_thread(simulation_thread_name)


def reset_simulation_state():
    clear_thread(simulation_thread_name)


def run(agent, thread):
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
    user_input = str(st.session_state["interview_simulation_response"])

    with st.spinner("Thinking..."):
        agent.graph.update_state(
            thread,
            {"messages": [HumanMessage(content=user_input)]},
            as_node="ask_question",
        )
        # Now call with None to resume with changed state
        _ = agent.graph.invoke(None, thread)


def user_response():
    st.chat_input(key="interview_simulation_response")


def show_message(message):
    if message.type == "human":
        st.chat_message("user").write(message.content)
    else:
        st.chat_message("assistant").write(message.content)


def show_messages(agent, thread):
    messages = agent.graph.get_state(thread).values["messages"]
    has_ended = (
        "ended" in agent.graph.get_state(thread).values
        and agent.graph.get_state(thread).values["ended"]
    )

    for message in messages:
        show_message(message)

    if len(messages) > 0 and not has_ended:
        user_response()


def render_interview_simulation_tab():
    if (
        "app_state" not in st.session_state
        or "interview_questions" not in st.session_state["app_state"]
    ):
        st.write("No application state found. Have you uploaded a resume?")
        return

    inteviewer = InterviewSimulator(checkpointer=get_memory())
    start = st.button("Start Interview")
    if start:
        reset_simulation_state()
        run(inteviewer, _get_simulation_thread())
    elif (
        "interview_simulation_response" in st.session_state
        and st.session_state["interview_simulation_response"] is not None
    ):
        _resume_with_state_update(inteviewer, _get_simulation_thread())

    show_messages(inteviewer, _get_simulation_thread())
