from typing import List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from utils import extra_json_object, get_model


class ResumeDoctorState(TypedDict):
    age_category: str
    resume: str
    job_description: str
    persona: str | None
    updated_resume: str | None
    interview_questions: List[dict] | None


class ResumeDoctor:
    def __init__(self):
        self.model = get_model()
        self.graph = self.build_graph()

    def build_graph(self):
        builder = StateGraph(ResumeDoctorState)

        self.SYSTEM_PROMPT = """
You are an expert resume reviever and writer. You have been asked to review a resume given a job description and a persona.

Here is the job description:
{job_description}

Here is the resume:
{resume}
        """
        self.PERSONA_GENERATION_PROMPT = """
Create an example persona for a likely interviewer for the given job description and age group. 
Your output should be just the persona, with no other comments.

Here is the age group:
{age_category}
"""

        self.REWRITE_RESUME_PROMPT = """
Tailor the given resume to appeal to the persona below. Keep the tone of the original resume.
Your output should be just the updated resume in markdown, with no other comments.

Here is the persona:
{persona}
"""
        self.INTERVIEW_QUESTIONS_PROMPT = """
For the job description, generate likely interview questions that the persona below is likely to ask. 
Group the questions into logical categories. Your output should just be json like this (no other comments):
{{
        "design": ["question 1", "question 2"],
        "coding": ["question 3", "question 4", "question 5"]
}}

Here is the persona:
{interview_questions}

"""

        builder.add_node("generate_persona", self.generate_persona)
        builder.add_node("update_resume", self.update_resume)
        builder.add_node(
            "generate_interview_questions", self.generate_interview_questions
        )
        builder.add_edge("generate_persona", "update_resume")
        builder.add_edge("update_resume", "generate_interview_questions")
        builder.add_edge("generate_interview_questions", END)

        builder.set_entry_point("generate_persona")

        return builder.compile()

    def _get_system_prompt(self, state: ResumeDoctorState) -> SystemMessage:
        return SystemMessage(
            content=self.SYSTEM_PROMPT.format(
                job_description=state["job_description"], resume=state["resume"]
            )
        )

    def generate_persona(self, state: ResumeDoctorState) -> ResumeDoctorState:
        messages = [
            self._get_system_prompt(state),
            HumanMessage(
                content=self.PERSONA_GENERATION_PROMPT.format(
                    age_category=state["age_category"]
                )
            ),
        ]

        response = self.model.invoke(messages)

        return {"persona": response.content}

    def update_resume(self, state: ResumeDoctorState) -> ResumeDoctorState:
        messages = [
            self._get_system_prompt(state),
            HumanMessage(
                content=self.REWRITE_RESUME_PROMPT.format(persona=state["persona"])
            ),
        ]

        response = self.model.invoke(messages)

        return {"updated_resume": response.content}

    def generate_interview_questions(
        self, state: ResumeDoctorState
    ) -> ResumeDoctorState:
        messages = [
            self._get_system_prompt(state),
            HumanMessage(
                content=self.INTERVIEW_QUESTIONS_PROMPT.format(
                    interview_questions=state["persona"]
                )
            ),
        ]

        response = self.model.invoke(messages)
        questions = extra_json_object(response.content)

        return {"interview_questions": questions}
