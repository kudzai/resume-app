from typing import List, TypedDict
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AnyMessage
import re
import json
from langgraph.checkpoint.sqlite import SqliteSaver
import streamlit as st

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
model_name = "meta/llama3-70b-instruct"


class ApplicationState(TypedDict):
    resume: str
    job_description: str
    criteria: List[str] | None
    decisions: List[dict] | None
    persona: str | None
    age_category: str | None
    interview_session: List[AnyMessage] | None


class Decision(TypedDict):
    decision: str
    reason: str
    persona: str
    age_category: str


def parse_resume(path_to_resume) -> str:
    loader = PyPDFLoader(path_to_resume)
    pages = loader.load()
    content = ""
    for page in pages:
        content += page.page_content

    return content


def get_model():
    return ChatNVIDIA(
        model=model_name,
        api_key=st.session_state["NVIDIA_API_KEY"],
        base_url=NVIDIA_BASE_URL,
        temperature=0.0,
    )


def _format_string(input_text: str) -> str:
    return input_text.replace("\n", " ")


def extra_json_object(input_text: str) -> dict | None:
    json_pattern = re.compile(r"\{[^}]+\}")
    json_match = json_pattern.search(_format_string(input_text))

    if json_match:
        json_str = json_match.group()
        return json.loads(json_str)

    return None


def extract_json_list(input_text: str) -> list[str] | None:
    list_pattern = re.compile(r"\[.*?\]")
    list_match = list_pattern.search(_format_string(input_text))

    if list_match:
        list_str = list_match.group()
        return json.loads(list_str)
    return None


@st.cache_resource
def get_memory():
    return SqliteSaver.from_conn_string(":memory:")


def get_thread(name_of_thread: str):
    # Thread id is used to manage multiple simultaneous chats
    if name_of_thread not in st.session_state:
        st.session_state[name_of_thread] = str(uuid.uuid4())
    return {"configurable": {"thread_id": st.session_state[name_of_thread]}}


def clear_thread(name_of_thread: str):
    if name_of_thread in st.session_state:
        del st.session_state[name_of_thread]
