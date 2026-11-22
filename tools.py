from typing import Any, Callable
import json
from llm import Response
from toolbox import multiply

class Tools:
    """Tools registry for the Agent"""

    def __init__(self,requires_approval:list[str]=[]):
        """Initialize and select tools that require approval before execution"""
        self.registry = {}
        self.requires_approval = requires_approval
    
    def add_tool(self,name:str,func:Callable,description:str="")->None:
        """Register a tool that the agent can use.
        
        Arguments:
            name: The name of the tool.
            func: The function implementing the tool.
            description: A description of the tool.
            """
        self.registry[name] ={"function":func, "description":description}

    
    @property
    def schemas(self)->None:
        """Used only for native tool-calling"""
        return None
    
    @property
    def description(self)->str:
        """Get descriptions of all registered tools"""
        return "\n".join(f"{tool}:{self.registry[tool]['description']}"
                         for tool in self.registry)
    

    @property
    def prompt(self)->str:
        return f"""
        #Tools
        
        if needed you can only use the following tools to assist you
        in completing tasks:
        {self.description}
        
        To use a tool, respond with JSON:
        {{"tool:"name","kwargs":{{"param":"value"}}}}
        """
    

    def parse(self,response:Response)->Response:
            """Parse a JSON tool call from text"""
            text = response.content
            if '"tool":' in text or "'tool':" in text:
                start, end = text.find("{"), text.rfind("}") + 1
                tool_call = json.loads(text[start:end])
    
                # Add the parsed tool call to the response
                return Response(content=response.content, reasoning=response.reasoning, tool_call=tool_call,metadata=response.metadata)
            return response

    def execute(self,response:Response)->Any:
        """Run a registered tool.

        Arguments:
            response: A response containing a parsed tool call.
        """
        tool_call = response.tool_call
        name, kwargs = tool_call["tool"], tool_call.get("kwargs", {})

        # Human in the loop: ask before running dangerous tools.
        if name in self.registry and name in self.requires_approval:
            approval = input(f"Allow {name}? (y/n): ").strip().lower()
            if approval not in ("y", "yes"):
                return f"Tool {name} was denied by the user."

        # Handle registered tools.
        if name in self.registry:
            tool_func = self.registry[name]["function"]
            return tool_func(**kwargs)

        return f"Tool {name} not found."

    def observation(self,result:str)->tuple[str,str]:
        """Return the observation as a user"""
        return "user",f"OBSERVATION:{result}"

    def is_done(self,response:Response)->bool:
        """The TinyAgent's stopping mechanism"""
        if not response.tool_call:
            return True
        if response.tool_call["tool"] == "final_answer":
            response.content = response.tool_call.get("kwargs","")
            return True
        return False
    