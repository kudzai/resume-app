import streamlit as st

from resume_doctor import ResumeDoctor


def render_resume_tuning_tab():
    if "app_state" not in st.session_state:
        st.write("No application state found. Have you uploaded a resume?")
        return

    app_state = st.session_state["app_state"]
    age_options = ["Baby Boomers", "GenX", "GenY", "GenZ"]

    age_category = st.selectbox(
        "Age Category",
        [""] + age_options,
        placeholder="Select an age category",
    )

    update_resume = st.button("Update Resume", disabled=age_category not in age_options)

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

            st.markdown(st.session_state.app_state["interview_questions"])
