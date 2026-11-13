# Tool usage, Learning, and Protocols
Tools, in the form of functions and APIs, allow LLMs to step into the real world and interact with any system exposed to the LLM.
Without tools, agents can only *think and plan* but not *act* autonomously.
The degree of autonomy also decides how tools are used. If there is no autonomy, but there is a fixed flow, then tools can be used in a predefined order. In contrast, systems with larger autonomy allow agents to choose which tools to use and when. 

## Tool Usage

### Tool creation

Although tools come in various forms, they are typically functions that can be accessed through either an API or some internal code. 
Creating tools can be straightforward, but the LLM should be taken into account when designing them.

### Tool Definition

After creating the tool, we need to inform the LLM what the tool does. There are several methods by which we can communicate this to the LLM:
- Learning: (Specific) ool definition and usage are learned during training
- prompting: the tool definition is shared with the LLM through structured prompts

#### Using natural language

As (reasoning) LLMs are becoming easier to steer and dynamically learn to use tools, we can share the tool definitions through prompts. When you have a small set of tools the LLM should use, you can share that through the system prompt:
```python
system_prompt = """
...
#Tool Definition
You can use the following tools:
-multiply(a,b): multiply two numbers
-divide(a,b): divides y by b

#Tool Usage
When you need to use them write:[function_name(value1,value2)]
"""
```
The only thing you need then is a small piece of code that recognizes when a tool is being called. This is called a **validator** or **parser** and requires the LLm to closely follow the usage instructions.
It can therefore be quite error-prone because small errors in following the instructions might result in incomplete calls to the tools.
if we re-register the tool with the prompt:
```python
tools = Tools()
tools.add_tool(
    name="multiply",
    func="multiply",
    description="Multiplies two numbers: multiply(a:str,b:str)"
)
#Show prompt
print(tools.prompt)
```
which gives us:
```python
#Tools
        
        if needed you can only use the following tools to assist you
        in completing tasks:
        multiply:Multiplies two numbers: multiply(a:str,b:str)
        
        To use a tool, respons with JSON:
        {"tool:"name","kwargs":{"param":"value"}}
```
#### Using structured function calling

We can define each tool in structured schemas. These are typically JSON schema objects that include function's name, description, and its parameters(types, required fields, ranges, etc.).
```python
tool_schema = {
    "type":"function",
    "function":{
        "name":"multiply",
        "description":"Multiply two numbers",
        "parameters":{
            "type":"object",
            "properties":{
                "a":{
                    "type":"number",
                    "description":"First number"
                },
                "b":{
                    "type":"number",
                    "description":"Second number"
                }
            },
            "required":[
                "a",
                "b"
            ]
        }
    }
}
```
Although we are using structured JSON schemas for communication, the LLm still has to interpret how to use the tools and when, which makes tool usage a difficult task. 

XXXXX Insert png Figure 5-6 XXXXX

Best practices for defining functions to take into account:
- Document your tool: extensive descriptions to allow your agent to have a better understanding of what the tool is capable of
- Minimize the number of tools
- Minimize the scope of a tool

### Tool Selection

In tool selection, that is where reasoning LLMs shine. As with context engineering, the process of discovering tools might be helped with methodologies like RAG, where you store all tool in schemas in a separate database for the LLM to discover. 

### Tool calling


