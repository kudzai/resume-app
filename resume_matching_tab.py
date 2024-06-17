# /resume-app/resume_matching_tab.py
from resume_screener import ResumeScreener
import streamlit as st
import os

from utils import ApplicationState


def upload_resume() -> str:
    """
    Handles resume upload from the user.

    This function presents a file uploader to the user, allowing them to upload a PDF resume.
    If a file is uploaded, it is saved to the 'data' directory and the file path is returned.

    Returns:
        str: The path to the uploaded resume file, or None if no file is uploaded.
    """
    uploaded_file = st.file_uploader(
        "Upload CV",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload your resume in pdf format",
    )
    if uploaded_file is not None:
        path = f"./data/{uploaded_file.name}"
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return path
    else:
        return None


def get_job_description():
    """
    Gets the job description from the user.

    This function presents a text area to the user, allowing them to enter the job description.

    Returns:
        str: The job description entered by the user.
    """
    return st.text_area(
        "Job Description",
        value=None,
        placeholder="Enter your job description",
        height=100,
        help="Enter the job description (at least 50 chars)",
    )


def get_criteria() -> list[str] | None:
    """
    Gets the criteria for matching from the user.

    This function presents a text area to the user, allowing them to enter criteria for matching.
    The user can separate multiple criteria using the '|' character.
    If no criteria are provided, the function returns None, indicating that criteria should be inferred from the job description.

    Returns:
        list[str] | None: A list of criteria entered by the user, or None if no criteria are provided.
    """
    criteria_str = st.text_area(
        "Criteria for matching",
        value=None,
        placeholder="Enter your criteria, (use | to separate each criteria). Leave blank to infer from job description",
        height=20,
        help="Enter the criteria. Separate each criteria with a '|'.",
    )

    if criteria_str is not None:
        return [x.strip() for x in criteria_str.split("|") if len(x.strip()) > 0]

    return None


def render_decisions(decisions: list[dict]):
    """
    Renders the decisions made for each individual criterion.

    This function displays the results of the matching process for each criterion,
    showing whether the resume matched the criterion and the reason for the decision.

    Args:
        decisions: A list of dictionaries, each representing a decision for a specific criterion.
    """
    st.subheader("Matching against individual criteria")
    for decision in decisions:
        if decision["decision"] == "fail":
            status = ":red[Not a match]"
        else:
            status = ":green[A match]"
        st.markdown(f"**{decision['criterion']} - {status}**")
        st.markdown(decision["reason"])
        st.divider()


def render_overall_decision(response):
    """
    Renders the overall decision about the resume's match with the job description.

    This function displays the overall decision, indicating whether the resume is a match or not,
    along with the reason for the decision.

    Args:
        response: A dictionary containing the overall decision and reason.
    """
    if response["decision"] == "fail":
        status = ":red[Not a match]"
    else:
        status = ":green[A match]"

    st.subheader(f"Overall Match - {status}")
    st.markdown(response["reason"])
    st.divider()


def render_resume_matching_tab():
    """
    Renders the resume matching tab in the Streamlit app.

    This tab allows users to upload a resume and job description, and then evaluate how well the resume matches the job description.
    Users can provide their own criteria for matching, or let the system infer criteria from the job description.
    The results are displayed in a user-friendly format, showing the decisions for each criterion and the overall decision.
    """
    st.subheader("Resume fit check")
    st.markdown(
        "Evaluates how well a resume matches a job description, against specified criteria. "
        "The matcher can also infer criteria from the job description."
    )
    # st.markdown("### Example files\n")
    # col1, col2 = st.columns(2)
    # with col1:
    #     with open("data/full-stack-engineer-jd.txt", "r") as f:
    #         st.download_button(
    #             "Download job description", f, file_name="job-description.txt"
    #         )

    # with col2:
    #     with open("data/john-doe-resume.pdf", "rb") as f:
    #         st.download_button("Download Resume", f, file_name="john-doe-resume.pdf")

    job_description = get_job_description()
    criteria = get_criteria()
    num_auto_generated_criteria = st.slider(
        "Number of Criteria",
        1,
        10,
        3,
        help="Number of criteria to generate automatically if none are provided",
    )
    resume_file_path = upload_resume()
    start = st.button("Run check")
    if start and resume_file_path is not None and job_description is not None:
        with st.spinner("Scoring ..."):
            screener = ResumeScreener()

            response = screener.graph.invoke(
                {
                    "path_to_resume": resume_file_path,
                    "job_description": job_description,
                    "criteria": criteria,
                    "num_auto_generated_criteria": num_auto_generated_criteria,
                }
            )

            # Always reset state after new CV
            app_state = ApplicationState(
                resume=response["resume"],
                job_description=response["job_description"],
                criteria=response["criteria"],
                decisions=response["decisions"],
                decision=response["decision"],
                reason=response["reason"],
            )

            st.session_state.app_state = app_state

    if "app_state" in st.session_state:
        app_state = st.session_state.app_state
        if "decision" in app_state:
            render_overall_decision(app_state)
        if "decisions" in app_state:
            render_decisions(app_state["decisions"])
