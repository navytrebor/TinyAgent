from typing import Callable

class Tools:
    """Tools registry for the Agent"""

    def __init__(self,requires_approval:list[str]=[]):
        """Initialize and select tools that require approval before execution"""
        self.registry = {}
        self.requires_approval = requires_approval
    
    def add_tool(self,name:str,func:Callable,description:str="")->None
        


