from llm import LLM, Response
from memory import Memory
from trajectory import Trajectory
from tools import Tools


class TinyAgent:
    """A minimal modular and educational agent framework"""

    def __init__(self, llm: LLM,memory: Memory,tools:Tools):
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.planner = None
        self.trajectory = Trajectory()

        # Build system prompt with all components
        system_prompt = "You are a helpful assistant.\n\n"
        system_prompt += self.tools.prompt
        self.memory.add("system", system_prompt)

    def run(self, task: str):
        """Run the agent on a task"""
        self.memory.add("user", task)
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str)->str:
        """Perform a single step of the agent on a task"""
        #messages = [{"role": "user", "content": task}]
        response = self.llm.generate(self.memory.get_messages(),tools=self.tools.schemas)
        self.memory.add("assistant",response.content,tool_call=response.tool_call)
        # Tool parsing
        response = self.tools.parse(response)
        #ANSWER: Stopping mechanism
        if self.tools.is_done(response):
            self.trajectory.add(response)
            return response.content

        return self._execute_action(response)
    
    def _execute_action(self,response:Response)->str:
        """Execute a tool action"""
        # ACTION: execute tools
        result = self.tools.execute(response)

        # OBSERVATION: add tool results to memory and display
        role,observation = self.tools.observation(result)
        self.memory.add(role,observation)
        self.trajectory.add(response,observation)
        return observation
