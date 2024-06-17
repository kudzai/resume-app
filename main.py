# /resume-app/main.py
from interview_simulation_tab import render_interview_simulation_tab
from resume_matching_tab import render_resume_matching_tab
import streamlit as st

from resume_tuning_tab import render_resume_tuning_tab


with st.sidebar:
    if "NVIDIA_API_KEY" in st.secrets:
        st.session_state["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]

    st.title("Settings")
    api_key = st.text_input(
        "NVIDIA API Key",
        type="password",
        value=st.session_state.get("NVIDIA_API_KEY", ""),
    )
    st.markdown(
        "Please enter your NVIDIA API Key. Don't have one? You can get one [here](https://build.nvidia.com/)."
    )
    if api_key:
        st.session_state["NVIDIA_API_KEY"] = api_key


def main():
    """
    Main function for the Streamlit app.

    This function sets up the tabs for the different features of the app:
    - Resume Fit Check
    - Resume Tuning And Interview Preparation
    - Interview Simulation

    Each tab renders its corresponding functionality.
    """
    (resume_matching_tab, resume_tuning_and_prep_tab, interview_simulation_tab) = (
        st.tabs(
            [
                "Resume Fit Check",
                "Resume Tuning And Interview Preparation",
                "Interview Simulation",
            ]
        )
    )
    with resume_matching_tab:
        render_resume_matching_tab()

    with resume_tuning_and_prep_tab:
        render_resume_tuning_tab()

    with interview_simulation_tab:
        render_interview_simulation_tab()


if __name__ == "__main__":
    if "NVIDIA_API_KEY" not in st.session_state:
        st.write(
            "Looks like were are missing the api key. Please enter it in the sidebar."
        )
        st.stop()

    main()
