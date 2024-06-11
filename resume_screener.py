import json
import operator
import os
from typing import List, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import Annotated

from utils import extra_json_object, extract_json_list, get_model, parse_resume


class ScreeningDecision(TypedDict):
    reason: str
    decision: str


class ScreenerState(TypedDict):
    path_to_resume: str
    resume: str
    job_description: str
    criteria: List[str]
    decisions: Annotated[List[ScreeningDecision], operator.add]
    decision: str
    reason: str


class ResumeScreener:
    def __init__(self):
        self.model = get_model()

        self.SYSTEM_PROMPT = """
You are an expert resume reviever. You have been asked to review the compatibilty of a resume with a job description.
Here is the job description:
{job_description}

Here is the resume:
{resume}
        """

        self.REVIEW_AGAINST_CRITERIA_PROMPT = """
Please review the resume and job description using the criterion below. Answer with 'pass' if the resume is compatible with the job description, 
and 'fail' if the resume is not compatible with the job description. If the criteria is not relevant to the job description, answer with 'pass'. 
Give a reason for the decision.

Your output should be in the following format:
{{
    "decision": "pass or fail",
    "reason": "reason for the decision"
}}

### Screening Criteria
{criterion}
"""
        self.OVERALL_COMPATIBILITY_PROMPT = """
Given the individual criterion matching results below, please decide if the resume is an overrall match with the job description, giving an overall 
reason for the decision.
Your output should be in the following format:
{{
    "decision": "pass or fail",
    "reason": "reason for the decision"
}}


### Matching Results
{compatibilities}
"""

        self.CRITERIA_GENERATION_PROMPT = """
From the job description, generate 3 to 5 criteria that can be used to measure the compatibility of a resume to the job description.
Your output should just be a list of strings in the following format with no other text, and ranked in order of importance:
["criteria1", "criteria2", "criteria3"]
        """

        self.build_graph()

    def get_system_prompt(self, job_description: str, resume: str) -> str:
        return self.SYSTEM_PROMPT.format(job_description=job_description, resume=resume)

    def build_graph(self):
        builder = StateGraph(ScreenerState)
        builder.add_node("parse", self.parse_resume)
        builder.add_node("generate_criteria", self.generate_criteria)
        builder.add_node("evaluate_criteria", self.evaluate_criteria)
        builder.add_node("overall_decision", self.overall_decision)

        builder.add_conditional_edges(
            "parse",
            self.should_generate_criteria,
            {
                "criteria": "generate_criteria",
                "decisions": "evaluate_criteria",
            },
        )
        builder.add_edge("generate_criteria", "evaluate_criteria")
        builder.add_conditional_edges(
            "evaluate_criteria",
            self.should_evaluate_criteria,
            {
                "evaluate_criteria": "evaluate_criteria",
                "decision": "overall_decision",
            },
        )

        builder.add_edge("overall_decision", END)
        builder.set_entry_point("parse")
        self.graph = builder.compile()

    def overall_decision(self, state: ScreenerState) -> ScreenerState:
        compatibilities = []
        for i, decision in enumerate(state["decisions"]):
            decision["criterion"] = state["criteria"][i]
            compatibilities.append(decision)

        compatibilities = json.dumps(compatibilities)
        messages = [
            SystemMessage(
                content=self.get_system_prompt(
                    state["job_description"], state["resume"]
                )
            ),
            HumanMessage(
                content=self.OVERALL_COMPATIBILITY_PROMPT.format(
                    compatibilities=compatibilities
                )
            ),
        ]
        response = self.model.invoke(messages)
        parsed_response = extra_json_object(response.content)
        return {
            "decision": parsed_response["decision"],
            "reason": parsed_response["reason"],
        }

    def should_generate_criteria(self, state: ScreenerState) -> str:
        if state["criteria"] is None or len(state["criteria"]) == 0:
            return "criteria"

        return "decisions"

    def parse_resume(self, state: ScreenerState) -> ScreenerState:
        parsed_resume = parse_resume(state["path_to_resume"])
        if os.path.exists(state["path_to_resume"]):
            os.remove(state["path_to_resume"])
        return {"resume": parsed_resume}

    def generate_criteria(self, state: ScreenerState) -> ScreenerState:
        messages = [
            SystemMessage(
                content=self.get_system_prompt(
                    state["job_description"], state["resume"]
                )
            ),
            HumanMessage(content=self.CRITERIA_GENERATION_PROMPT),
        ]
        response = self.model.invoke(messages)

        parsed_response = extract_json_list(response.content)

        return {"criteria": parsed_response}

    def should_evaluate_criteria(self, state: ScreenerState) -> bool:
        if len(state["criteria"]) > len(state["decisions"]):
            return "evaluate_criteria"

        return "decision"

    def evaluate_criteria(self, state: ScreenerState) -> ScreenerState:
        last_entry = len(state["decisions"])
        next_criteria = state["criteria"][last_entry]

        messages = [
            SystemMessage(
                content=self.get_system_prompt(
                    state["job_description"], state["resume"]
                )
            ),
            HumanMessage(
                content=self.REVIEW_AGAINST_CRITERIA_PROMPT.format(
                    criterion=next_criteria
                )
            ),
        ]
        response = self.model.invoke(messages)

        parsed_response = extra_json_object(response.content)
        return {
            "decisions": [
                {
                    "decision": parsed_response["decision"],
                    "reason": parsed_response["reason"],
                }
            ]
        }
