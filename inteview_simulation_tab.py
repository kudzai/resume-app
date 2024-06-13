import streamlit as st


def render_interview_simulation_tab():
    if (
        "app_state" not in st.session_state
        or "interview_questions" not in st.session_state["app_state"]
    ):
        st.write("No application state found. Have you uploaded a resume?")
        return

    st.markdown(
        st.session_state["app_state"]["interview_questions"], unsafe_allow_html=True
    )
