from resume_matching_tab import render_resume_matching_tab
import streamlit as st
import os

os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]


if __name__ == "__main__":
    resume_matching_tab, skills_gap_tab = st.tabs(
        ["Evaluate Match", "Skills Gap Analysis"]
    )
    with resume_matching_tab:
        render_resume_matching_tab()
