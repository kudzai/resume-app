from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from utils import get_model
import operator


class ResumeFormatterState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    format_style: str
    resume: str
    job_description: str


class ResumeFormatter:
    def __init__(self, checkpointer=None):
        self.model = get_model()
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

        self.SYSTEM_PROMPT = """
You are an expert resume reviever. You have asked to re-write the 
resume in the format speficied by the user.
The resume should address the job description below.

Here is the job description:
{job_description}

Here is the resume:
{resume}
        """

        self.CONTRACT_STYLE_PROMPT = """
Format the resume given below to have the following sections:
1. Profile/Summary
2. Key skills
3. Current Role/Status
4. Contract Portfolio/Highlights (3 most relevant achievements for the job description)
5. Experience/Contract Summary
6. Education/Qualifications
7. Interests/Hobbies if any

The 'Experience/Contract Summary' section should be formatted as (ignoring all other information):
<start date> to <end date>: <company> <role>

'Education/Qualifications' section should be (ignoring all other information):
<course title in bold>: <name of institution>

Your response should be the resume in markdown format, with no other comments.
"""

        self.CHRONOLOGICAL_STYLE_PROMPT = """
Format the resume given below to have the following sections:
1. Profile/Summary
2. Key skills
3. Current Role/Status
4. Contract Portfolio/Highlights (3 most relevant achievements for the job description)
5. Experience/Contract Summary
6. Education/Qualifications
7. Interests/Hobbies if any


The 'Experience/Contract Summary' section should be formatted as:
<start date> to <end date>: <company> <role>

'Education/Qualifications' section should be (ignoring all other information):
<course title in bold>: <name of institution>

Your response should be the resume in markdown format, with no other comments.
"""

    def _build_graph(self):
        builder = StateGraph(ResumeFormatterState)

        builder.add_node("format_resume", self.format_resume)
        builder.add_node("check_user_input", self.check_user_input)
        builder.add_edge("format_resume", "check_user_input")
        builder.add_conditional_edges(
            "check_user_input",
            self.should_format_resume,
            {END: END, "format_resume": "format_resume"},
        )
        builder.set_entry_point("format_resume")

        return builder.compile(
            checkpointer=self.checkpointer, interrupt_after=["format_resume"]
        )

    def _get_system_prompt(self, state: ResumeFormatterState) -> SystemMessage:
        return SystemMessage(
            content=self.SYSTEM_PROMPT.format(
                job_description=state["job_description"], resume=state["resume"]
            )
        )

    def format_resume(self, state: ResumeFormatterState):
        system_message = self._get_system_prompt(state)
        messages = [system_message]
        first_message = []
        if len(state["messages"]) > 0:
            # If we are already chatting, just add all the messages to date
            messages += state["messages"]
        else:
            if state["format_style"] == "contract":
                prompt = self.CONTRACT_STYLE_PROMPT
            else:
                prompt = self.CHRONOLOGICAL_STYLE_PROMPT
            # Lets remember the first message specifying the style to use
            first_message = [HumanMessage(content=prompt)]
            messages += first_message

        response = self.model.invoke(messages)

        return {"messages": first_message + [AIMessage(content=response.content)]}

    def should_format_resume(self, state: ResumeFormatterState):
        user_input = state["messages"][-1].content
        if user_input.lower().strip() == "done":
            return "END"

        return "format_resume"

    def check_user_input(self, state: ResumeFormatterState):
        pass
