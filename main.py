from resume_screener import ResumeScreener


if __name__ == "__main__":
    screener = ResumeScreener()

    response = screener.graph.invoke(
        {
            "path_to_resume": "resume.pdf",
            "job_description": "data scientist",
            "criteria": [],
        }
    )
    print(response)
