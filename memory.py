from llm import LLM, EmbeddingModel

class Memory:
    '''Simple memory module to store conversation history'''

    def __init__(self):
        self.messages = []

    def add(self,role:str,content:str,tool_call:dict|None=None,**kwargs)->None:
        '''Add a message to memory'''
        message = {"role":role,"content":content}

        #Tool call
        if tool_call:
            message["tool_calls"] = [tool_call]

        # Append message to memory
        self.messages.append(message)

    def get_messages(self):
        '''Get all messages'''
        return self.messages

class TrimmingMemory(Memory):
    """Memory that keeps only the last two user/assistant turns"""

    def add(self, role:str, content:str, tool_call:dict|None=None, **kwargs)->None:
        # Add the new message first using the parent class (Memory)
        super().add(role, content, tool_call, **kwargs)

        # Then, keep system message plus the most recent two turns (4 messages)

        system=[message for message in self.messages if message["role"] == "system"]
        turns = [message for message in self.messages if message["role"] != "system"]

        self.messages = system + turns[-4:]

class SummarizationMemory(Memory):
    """memory that compresses past turns into a running summary via an LLM"""

    def __init__(self,llm: LLM):
        super().__init__()
        self.llm = llm

    def add(self,role:str,content:str,tool_call:dict|None=None,**kwargs)->None:
        """Add the new message first using the parent class (Memory)"""
        super().add(role, content, tool_call, **kwargs)

    # After each completed turn, update the running summary
        if role == "assistant":
        # Split the existing summary from the new conversation turns.
            summary = ""
            conversation = ""
            for message in self.messages:
                if message["role"] == "system":
                    summary += message["content"] + "\n"
                else:
                    conversation += f"{message['role']}:{message['content']}\n"

            # Ask the LLM to extend the summary with new conversation
            prompt = f"""Update the summary with the new conversation.
            Summary:{summary}
            Conversation:{conversation}
            Output the updated summary only
            """
            response = self.llm.generate([{"role":"user","content":prompt}])
            self.messages = [{"role":"system","content":response.content}]


class RAGMemory(Memory):
    """Long-Term memory with RAG"""

    def __init__(self,embedding_model:EmbeddingModel,documents:list[str]):
        super().__init__()
        self.embedding_model = embedding_model
        self.documents = documents
        self.embeddings = [self.embedding_model.embed(doc) for doc in documents]

    def add(self,role:str,content:str,**kwargs)->None:
        # Augment user queries with retrieved context before storin
        if role == "user":
            context = "\n".join(self.search(content))
            content = f"""Context:
            {context}
            Question:{content}
        """
        super().add(role, content, **kwargs)

    def search(self,query:str)->list[str]:
        """Return the top-k documents most similar to the query"""
        query_embed = self.embedding_model.embed(query)
        scores = [self._cosine(query_embed,embed) for embed in self.embeddings]
        ranked = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)
        return [self.documents[index] for index in ranked[:3]]

    def _cosine(self, a:list[float], b:list[float])->float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(y*y for y in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
    