from inteview_simulation_tab import render_interview_simulation_tab
from resume_matching_tab import render_resume_matching_tab
import streamlit as st
import os

from resume_tuning_tab import render_resume_tuning_tab

os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]


if __name__ == "__main__":
    (
        resume_matching_tab,
        touch_resume_tab,
        interview_simulation_tab,
        skills_gap_tab,
        format_resume_tab,
    ) = st.tabs(
        [
            "Evaluate Match",
            "Touch Resume",
            "Interview Simulation",
            "Skills Gap Analysis",
            "Format Resume",
        ]
    )
    with resume_matching_tab:
        render_resume_matching_tab()

    with touch_resume_tab:
        render_resume_tuning_tab()

    with interview_simulation_tab:
        render_interview_simulation_tab()

    with skills_gap_tab:
        st.write("Skills Gap Analysis")

    with format_resume_tab:
        st.write("Format Resume")
