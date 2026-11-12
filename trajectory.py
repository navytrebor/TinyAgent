from dataclasses import dataclass, field

from llm import Response


@dataclass
class Step:
    """A single step of the agent: a thought, then either an action/observation or an answer"""
    thought: str | None = None
    action: dict | None = None
    observation: str | None = None
    answer: str | None = None
    metadata: dict = field(default_factory=dict)


class Trajectory:
    """Records agent execution as a sequence of runs"""

    def __init__(self)->None:
        self.runs:list[dict] = []

    def initialize(self,query: str)->None:
        """Register a new run with the given query"""
        self.runs.append({"query":query,"steps":[]})

    def add(self,response:Response,observation:str|None=None)->None:
        """Record a step from a Response, optionally with an observation"""
        #Add thought
        step = Step(
            thought=response.reasoning,
            metadata=response.metadata,
        )

        #Add ACTION/OBSERVATION or ANSWER
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content

        self.runs[-1]["steps"].append(step)
