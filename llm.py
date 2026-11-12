import json
import urllib.request
from dataclasses import dataclass

@dataclass
class Response:
    content: str | None
    reasoning: str | None
    tool_call: dict | None
    metadata: dict


class LLM:

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "no_key",
        think: bool = False,
        temperature: float | None = None,
    ):
        """Initialize the LLM with model, base URL, API key, and think flag"""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.temperature = temperature

    def generate(
        self,
        messages: list[dict],
        tools: list | None = None,
    )-> Response:
        """Generate a response from the LLM given a list of messages"""
        #Build the request body
        body = {
            "model":self.model,
            "messages":messages,
        }

        #Tools and reasoning
        if tools:
            body["tools"] = tools
        if not self.think:
            body["reasoning_effort"] = "none"

        #POST to the OpenAi-compatible / chat / completions endpoint
        request = urllib.request.Request(
            f'{self.base_url}/chat/completions',
            data = json.dumps(body).encode(),
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )    
        with urllib.request.urlopen(request) as response:
            data = json.load(response)

        #Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls")
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model":data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"]
        }

        #Format Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata
        )

class EmbeddingModel:
    """Generate embeddings"""
    def __init__(
            self,
            model:str,
            base_url:str = "http://localhost:11434/v1",

    ):
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector """
        #POST to the OpenAI-compatible/embeddings endpoint
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"model":self.model,"input":text}).encode(),
            headers={"Content-Type":"application/json"}
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # Extract and return the embdedding
        return response["data"][0]["embedding"]
