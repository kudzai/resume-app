import streamlit as st

from resume_doctor import ResumeDoctor


def _render_questions(questions_with_categories):
    if not questions_with_categories:
        st.write("No questions found.")
        return
    with st.expander("## Suggested Interview Questions"):
        categories = list(questions_with_categories.keys())
        for category in categories:
            st.markdown(f"### {category.capitalize()}")
            questions = questions_with_categories[category]
            for question in questions:
                st.markdown(f"- {question}")


def render_resume_tuning_tab():
    if "app_state" not in st.session_state:
        st.write("No application state found. Have you uploaded a resume?")
        return

    st.subheader("Resume Tuning")
    st.markdown(
        "- Updates the resume to better match the job description. \n"
        "- Creates an example persona for a likely interviewer given the job description. Age category is used for the persona.\n"
        "- Generates sample interview questions based on the job description and the persona.\n"
    )
    app_state = st.session_state["app_state"]
    age_options = ["Baby Boomers", "GenX", "GenY", "GenZ"]

    age_category = st.selectbox(
        "Age Category",
        [""] + age_options,
        placeholder="Select an age category",
    )

    update_resume = st.button(
        "Run Generation", disabled=age_category not in age_options
    )

    if update_resume and age_category in age_options:
        with st.spinner("Updating resume..."):
            resume_doctor = ResumeDoctor()

            response = resume_doctor.graph.invoke(
                {
                    "resume": app_state["resume"],
                    "job_description": app_state["job_description"],
                    "age_category": age_category,
                }
            )
            st.session_state.app_state["persona"] = response["persona"]
            st.session_state.app_state["updated_resume"] = response["updated_resume"]
            st.session_state.app_state["interview_questions"] = response[
                "interview_questions"
            ]
            st.session_state.app_state["age_category"] = age_category

    if "updated_resume" in st.session_state.app_state:
        with st.expander("## Updated Resume"):
            st.code(st.session_state.app_state["updated_resume"])

    if "interview_questions" in st.session_state.app_state:
        _render_questions(st.session_state.app_state["interview_questions"])
