# /resume-app/resume_screener.py
import json
import operator
import os
from typing import List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from typing_extensions import Annotated

from utils import extra_json_object, extract_json_list, get_model, parse_resume


class ScreeningDecision(TypedDict):
    """
    A dictionary representing a screening decision.

    Attributes:
        reason: The reason for the decision.
        decision: The decision itself, either "pass" or "fail".
    """

    reason: str
    decision: str


class ScreenerState(TypedDict):
    """
    A dictionary representing the state of the resume screener.

    Attributes:
        path_to_resume: The path to the resume file.
        resume: The parsed resume text.
        job_description: The job description text.
        criteria: A list of screening criteria.
        decisions: A list of screening decisions.
        decision: The overall decision, either "pass" or "fail".
        reason: The reason for the overall decision.
        num_auto_generated_criteria: The number of automatically generated criteria.
    """

    path_to_resume: str
    resume: str
    job_description: str
    criteria: List[str]
    decisions: Annotated[List[ScreeningDecision], operator.add]
    decision: str
    reason: str
    num_auto_generated_criteria: Optional[int]


class ResumeScreener:
    """
    A class that uses a language model to screen resumes against a job description.

    This class takes a resume file path and a job description as input.
    It then uses a language model to generate screening criteria,
    evaluate the resume against each criterion, and make an overall decision
    about the compatibility of the resume with the job description.
    """

    def __init__(self):
        """
        Initializes the ResumeScreener class.

        Loads the language model and sets up the system prompts and state graph.
        """
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
From the job description, generate not more than {num_criteria} criteria that can be used to measure the compatibility of a resume to the job description.
Your output should just be a list of strings in the following format with no other text, and ranked in order of importance:
["criteria1", "criteria2", "criteria3"]
        """

        self.build_graph()

    def get_system_prompt(self, job_description: str, resume: str) -> str:
        """
        Returns the system prompt for the language model.

        Args:
            job_description: The job description text.
            resume: The parsed resume text.

        Returns:
            The system prompt string.
        """
        return self.SYSTEM_PROMPT.format(job_description=job_description, resume=resume)

    def build_graph(self):
        """
        Builds the state graph for the resume screener.

        The state graph defines the flow of the screening process,
        including the different states and transitions between them.
        """
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
        """
        Makes the overall decision about the resume's compatibility.

        This method takes the results of the individual criterion evaluations
        and uses the language model to make an overall decision.

        Args:
            state: The current state of the screener.

        Returns:
            The updated state with the overall decision and reason.
        """
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
        """
        Determines whether to generate criteria or evaluate existing ones.

        This method checks if criteria have already been generated.
        If not, it returns "criteria" to indicate that criteria generation
        should be the next step. Otherwise, it returns "decisions" to
        indicate that criterion evaluation should be the next step.

        Args:
            state: The current state of the screener.

        Returns:
            "criteria" or "decisions" depending on the current state.
        """
        if (
            "criteria" not in state
            or state["criteria"] is None
            or len(state["criteria"]) == 0
        ):
            return "criteria"

        return "decisions"

    def parse_resume(self, state: ScreenerState) -> ScreenerState:
        """
        Parses the resume file.

        This method reads the resume file and parses it into plain text.

        Args:
            state: The current state of the screener.

        Returns:
            The updated state with the parsed resume text.
        """
        parsed_resume = parse_resume(state["path_to_resume"])
        if os.path.exists(state["path_to_resume"]):
            os.remove(state["path_to_resume"])
        return {"resume": parsed_resume}

    def generate_criteria(self, state: ScreenerState) -> ScreenerState:
        """
        Generates screening criteria from the job description.

        This method uses the language model to generate a list of criteria
        that can be used to evaluate the resume's compatibility with the job description.

        Args:
            state: The current state of the screener.

        Returns:
            The updated state with the generated criteria.
        """
        messages = [
            SystemMessage(
                content=self.get_system_prompt(
                    state["job_description"], state["resume"]
                )
            ),
            HumanMessage(
                content=self.CRITERIA_GENERATION_PROMPT.format(
                    num_criteria=(state["num_auto_generated_criteria"] or 3)
                )
            ),
        ]
        response = self.model.invoke(messages)

        parsed_response = extract_json_list(response.content)

        return {"criteria": parsed_response}

    def should_evaluate_criteria(self, state: ScreenerState) -> bool:
        """
        Determines whether to evaluate more criteria or make the overall decision.

        This method checks if all criteria have been evaluated.
        If not, it returns "evaluate_criteria" to indicate that more criteria
        should be evaluated. Otherwise, it returns "decision" to indicate
        that the overall decision should be made.

        Args:
            state: The current state of the screener.

        Returns:
            "evaluate_criteria" or "decision" depending on the current state.
        """
        if len(state["criteria"]) > len(state["decisions"]):
            return "evaluate_criteria"

        return "decision"

    def evaluate_criteria(self, state: ScreenerState) -> ScreenerState:
        """
        Evaluates the resume against a single criterion.

        This method uses the language model to evaluate the resume against
        the next criterion in the list. It records the decision and reason
        for each criterion evaluation.

        Args:
            state: The current state of the screener.

        Returns:
            The updated state with the new decision and reason.
        """
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
