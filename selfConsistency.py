from collections import Counter
from llm import LLM


llm = LLM(model="gemma3:12b",temperature=1)

answers = []
for attempt in range(1, 11):
    print(f"Generating response {attempt}/10...", flush=True)
    query = """Six friends (Mararten, Ilse, Sarah, Jor, Irene, and Chris)
    are sitting in a row.
    - Sarah is in seat 3.
    - Chris is sitting at one end of the row.
    - Jor is sitting immediately to the right of Chris.
    - Ilse is sitting exactly in the middle of Sarah and Maarten.
    - Irene is not stting next to Sarah.
    - Maarten is sitting somewhere to the left of Irene.

    In which seat is Maarten sitting?
    Let's think step by step and give back your answers only after 'Answer:'
    """

    # Generate response
    messages =[{"role": "user", "content": query}]
    response = llm.generate(messages)
    answer = response.content.split("Answer:")[-1].strip()
    answers.append(answer)

print(Counter(answers))
