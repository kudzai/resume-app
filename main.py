from resume_screener import ResumeScreener
import streamlit as st


def upload_resume() -> str:
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
    return st.text_area(
        "Job Description",
        value=None,
        placeholder="Enter your job description",
        height=100,
        help="Enter the job description (at least 50 chars)",
    )


def get_criteria() -> list[str] | None:
    criteria_str = st.text_area(
        "Criteria for matching",
        value=None,
        placeholder="Enter your criteria, (use | to separate each criteria). Leave blank to infer from job description",
        height=20,
        help="Enter the criteria. Separate each criteria with a '|'.",
    )

    if criteria_str is not None:
        return [x.strip() for x in criteria_str.split("|")]

    return None


if __name__ == "__main__":
    st.title("How Do I Match?")
    job_description = get_job_description()
    criteria = get_criteria()
    resume_file_path = upload_resume()
    print(f"Criteria: {criteria}")
    print(f"resume_file_path: {resume_file_path}")
    print(f"job_description: {job_description}")
    start = st.button("Start")
    if start and resume_file_path is not None and job_description is not None:
        screener = ResumeScreener()

        response = screener.graph.invoke(
            {
                "path_to_resume": resume_file_path,
                "job_description": job_description,
                "criteria": [],
            }
        )
        print(response)
