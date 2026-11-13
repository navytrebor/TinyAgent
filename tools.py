from typing import Callable

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
        
        To use a tool, respons with JSON:
        {{"tool:"name","kwargs":{{"param":"value"}}}}
        """
    

