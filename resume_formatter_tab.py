import streamlit as st

from resume_formatter import ResumeFormatter
from langchain_core.messages import HumanMessage
import uuid
from utils import get_memory, get_thread


def user_input_form():
    st.chat_input(key="user_format_input")


def run(agent, thread, format_style: str):
    if "updated_resume" in st.session_state["app_state"]:
        resume = st.session_state["app_state"]["updated_resume"]
    else:
        resume = st.session_state["app_state"]["resume"]

    job_description = st.session_state["app_state"]["job_description"]
    _ = agent.graph.invoke(
        {
            "resume": resume,
            "job_description": job_description,
            "format_style": format_style,
        },
        thread,
    )


def _resume_with_state_update(agent, thread):
    current_values = agent.graph.get_state(thread)
    user_input = str(st.session_state["user_format_input"])
    current_values.values["messages"].append(HumanMessage(content=user_input))

    agent.graph.update_state(thread, {"messages": [HumanMessage(content=user_input)]})
    # Now call with None to resume with changed state
    response = agent.graph.invoke(None, thread)
    formatted_resume = response["messages"][-1].content
    st.session_state["app_state"]["formatted_resume"] = formatted_resume


def show_message(message):
    if message.type == "human":
        st.chat_message("user").write(message.content)
    else:
        st.chat_message("assistant").code(message.content)


def show_messages(agent, thread):
    messages = agent.graph.get_state(thread).values["messages"]
    for message in messages:
        show_message(message)

    if len(messages) > 0:
        user_input_form()


def get_formatter_thread():
    name_of_thread = "formatter_thread_id"
    return get_thread(name_of_thread)


def render_resume_formatter_tab():
    if "app_state" not in st.session_state or (
        "updated_resume" not in st.session_state["app_state"]
        and "resume" not in st.session_state["app_state"]
    ):
        st.write("No application state found. Have you uploaded a resume?")
        return

    formatter = ResumeFormatter(checkpointer=get_memory())

    format_options = ["contract", "chronological"]
    format_style = st.selectbox("Format Style", options=[""] + format_options)
    if format_style in format_options:
        with st.spinner("Loading..."):
            run(formatter, get_formatter_thread(), format_style)
    if (
        "user_format_input" in st.session_state
        and st.session_state["user_format_input"] is not None
    ):
        _resume_with_state_update(formatter, get_formatter_thread())

    show_messages(formatter, get_formatter_thread())
