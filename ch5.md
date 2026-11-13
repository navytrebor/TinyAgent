# Tool usage, Learning, and Protocols
Tools, in the form of functions and APIs, allow LLMs to step into the real world and interact with any system exposed to the LLM.
Without tools, agents can only *think and plan* but not *act* autonomously.
The degree of autonomy also decides how tools are used. If there is no autonomy, but there is a fixed flow, then tools can be used in a predefined order. In contrast, systems with larger autonomy allow agents to choose which tools to use and when. 

## Tool Usage

### Tool creation

Although tools come in various forms, they are typically functions that can be accessed through either an API or some internal code. 
Creating tools can be straightforward, but the LLM should be taken into account when designing them.