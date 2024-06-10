from typing import List, TypedDict
from langgraph.graph import StateGraph, END
import operator
from typing_extensions import Annotated

from utils import parse_resume


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
        print("Overall decision...")
        print(state)
        return {"decision": "accept", "reason": "overall_decision_made"}

    def should_generate_criteria(self, state: ScreenerState) -> str:
        if state["criteria"] is None or len(state["criteria"]) == 0:
            return "criteria"

        return "decisions"

    def parse_resume(self, state: ScreenerState) -> ScreenerState:
        parsed_resume = parse_resume(state["path_to_resume"])
        return {"resume": parsed_resume}

    def generate_criteria(self, state: ScreenerState) -> ScreenerState:
        print("Generating criteria...")
        return {"criteria": ["first_criteria", "second_criteria", "third_criteria"]}

    def should_evaluate_criteria(self, state: ScreenerState) -> bool:
        if len(state["criteria"]) > len(state["decisions"]):
            return "evaluate_criteria"

        return "decision"

    def evaluate_criteria(self, state: ScreenerState) -> ScreenerState:
        print(f"decisions = {state['decisions']}")
        last_entry = len(state["decisions"])
        next_criteria = state["criteria"][last_entry]
        print(f"Evaluating criteria: {next_criteria}")
        return {
            "decisions": [
                {
                    "decision": "accept",
                    "reason": f"criteria_evaluated - {next_criteria}",
                }
            ]
        }
