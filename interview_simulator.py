# /resume-app/interview_simulator.py
from typing import List, Literal, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, AIMessage
from utils import get_model
import operator
from typing import Annotated


class InterviewSimulatorState(TypedDict):
    """
    A dictionary representing the state of the interview simulator.

    Attributes:
        resume: The candidate's resume text.
        persona: The persona of the interviewer.
        job_description: The job description text.
        interview_questions: A list of interview questions, grouped by category.
        messages: A list of messages exchanged during the interview.
        last_question: The last question asked by the interviewer.
        ended: A boolean indicating whether the interview has ended.
    """

    resume: str
    persona: str
    job_description: str
    interview_questions: List[dict] | None
    messages: Annotated[list[AnyMessage], operator.add]
    last_question: str | None
    ended: bool | None


class InterviewSimulator:
    """
    A class that simulates an interview using a language model.

    This class takes a resume, job description, persona, and interview questions as input.
    It then uses a language model to simulate an interview between the interviewer and the candidate,
    generating responses for both parties based on the provided context.
    """

    def __init__(self, checkpointer):
        """
        Initializes the InterviewSimulator class.

        Args:
            checkpointer: A checkpointer object used to save and load the state of the interview.
        """
        self.model = get_model()
        self.checkpointer = checkpointer
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Builds the state graph for the interview simulator.

        The state graph defines the flow of the interview,
        including the different states and transitions between them.
        """
        builder = StateGraph(InterviewSimulatorState)

        self.SYSTEM_PROMPT = """
You are an expert interviewer, and you will take on the role of the gven persona. 
You have been asked to interview a candidate with the following resume given a job description.

Here is the job description:
{job_description}

Here is the resume:
{resume}

Here is your persona:
{persona}
"""

        self.INTRODUCTION_PROMPT = """
Start by introducing yourself, and welcoming the candidate to the interview. 
Be empathetic and friendly, but firm.

Also let the candidate they can reply with "DONE" to finish the interview.
"""

        self.SELECT_AND_ASK_QUESTION_PROMPT = """
Ask the candidate a question. Select from the following questions.
{questions}
Make sure not to pick the same question twice.
"""

        self.REVIEW_ANSWER_PROMPT = """
The candidate has answered the following question:
{question}

The candidate has given the following answer:
{answer}

Comment on the answer, directly addressing the candidate, using your own knowledge and experience, and the candidate's resume. Suggest how it can be improved if possible.
"""
        self.WRAP_UP_PROMPT = """
The candidate has finished the interview. Thank the candidate for their time and consideration.
"""
        builder.add_node("introduction", self.introduction)
        builder.add_node("ask_question", self.ask_question)
        builder.add_node("review_answer", self.review_answer)
        builder.add_node("pre_review_answer", self.pre_review_answer)
        builder.add_node("wrap_up", self.wrap_up)

        builder.add_edge("introduction", "ask_question")

        builder.add_edge("introduction", "ask_question")
        builder.add_conditional_edges("ask_question", self.should_end_or_review)

        builder.add_edge("pre_review_answer", "review_answer")
        builder.add_edge("review_answer", "ask_question")
        builder.add_edge("wrap_up", END)

        builder.set_entry_point("introduction")

        return builder.compile(
            checkpointer=self.checkpointer,
            interrupt_after=["introduction", "ask_question"],
            interrupt_before=[],
        )

    def pre_review_answer(self, state: InterviewSimulatorState):
        """
        A placeholder function for pre-reviewing the candidate's answer.

        This function is currently empty and does not perform any actions.
        It is included as a potential point for future enhancements.
        """
        pass

    def _get_system_prompt(self, state: InterviewSimulatorState) -> SystemMessage:
        """
        Returns the system prompt for the language model.

        This method constructs the system prompt based on the provided context,
        including the job description, resume, and persona.

        Args:
            state: The current state of the interview simulator.

        Returns:
            The system prompt string.
        """
        return SystemMessage(
            content=self.SYSTEM_PROMPT.format(
                job_description=state["job_description"],
                resume=state["resume"],
                persona=state["persona"],
            )
        )

    def introduction(self, state: InterviewSimulatorState) -> InterviewSimulatorState:
        """
        Generates the interviewer's introduction.

        This method uses the language model to generate the interviewer's introduction,
        following the instructions provided in the INTRODUCTION_PROMPT.

        Args:
            state: The current state of the interview simulator.

        Returns:
            The updated state with the interviewer's introduction added to the messages list.
        """
        messages = [
            self._get_system_prompt(state),
            HumanMessage(content=self.INTRODUCTION_PROMPT),
        ]

        response = self.model.invoke(messages)

        return {"messages": [AIMessage(content=response.content)]}

    def should_end_or_review(
        self, state: InterviewSimulatorState
    ) -> Literal["wrap_up", "pre_review_answer"]:
        """
        Determines whether to end the interview or review the candidate's answer.

        This method checks the last message from the candidate.
        If the message is "DONE", it returns "wrap_up" to indicate that the interview should end.
        Otherwise, it returns "pre_review_answer" to indicate that the candidate's answer should be reviewed.

        Args:
            state: The current state of the interview simulator.

        Returns:
            "wrap_up" or "pre_review_answer" depending on the candidate's last message.
        """
        last_response = state["messages"][-1]
        if last_response.content.upper() == "DONE":
            return "wrap_up"
        else:
            return "pre_review_answer"

    def review_answer(self, state: InterviewSimulatorState) -> InterviewSimulatorState:
        """
        Generates the interviewer's response to the candidate's answer.

        This method uses the language model to generate the interviewer's response,
        following the instructions provided in the REVIEW_ANSWER_PROMPT.

        Args:
            state: The current state of the interview simulator.

        Returns:
            The updated state with the interviewer's response added to the messages list.
        """
        messages = [
            self._get_system_prompt(state),
        ]
        if len(state["messages"]) > 0:
            messages += state["messages"]

        messages += [
            HumanMessage(
                content=self.REVIEW_ANSWER_PROMPT.format(
                    question=state["last_question"],
                    answer=state["messages"][-1].content,
                )
            )
        ]

        response = self.model.invoke(messages)

        return {"messages": [AIMessage(content=response.content)]}

    def ask_question(self, state: InterviewSimulatorState) -> InterviewSimulatorState:
        """
        Generates the next interview question.

        This method uses the language model to select and ask the next interview question,
        following the instructions provided in the SELECT_AND_ASK_QUESTION_PROMPT.

        Args:
            state: The current state of the interview simulator.

        Returns:
            The updated state with the new question added to the messages list and stored as the last_question.
        """
        messages = [
            self._get_system_prompt(state),
        ]
        if len(state["messages"]) > 0:
            messages += state["messages"]

        messages += [
            HumanMessage(
                content=self.SELECT_AND_ASK_QUESTION_PROMPT.format(
                    questions=state["interview_questions"],
                )
            )
        ]

        response = self.model.invoke(messages)
        return {
            "messages": [AIMessage(content=response.content)],
            "last_question": response.content,
        }

    def wrap_up(self, state: InterviewSimulatorState) -> InterviewSimulatorState:
        """
        Generates the interviewer's closing remarks.

        This method uses the language model to generate the interviewer's closing remarks,
        following the instructions provided in the WRAP_UP_PROMPT.

        Args:
            state: The current state of the interview simulator.

        Returns:
            The updated state with the interviewer's closing remarks added to the messages list and the ended flag set to True.
        """
        messages = [
            self._get_system_prompt(state),
        ]
        if len(state["messages"]) > 0:
            messages += state["messages"]

        messages += [HumanMessage(content=self.WRAP_UP_PROMPT)]

        response = self.model.invoke(messages)
        return {
            "messages": [AIMessage(content=response.content)],
            "ended": True,
        }
