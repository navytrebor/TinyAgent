from llm import LLM

llm = LLM(model="gemma3:12b",temperature=1)

test_cases = [
    ('III', 3),
    ('IV', 4),
    ('IX', 9),
    ('LVIII', 58),
    ('MCMXCIV', 1994),
    ('MMMCMXCIX', 3999),
    ("", 0),
    ("IIII", 0), #invalid: four in a row
    ("VV", 0), #invalid: V's cannot repeat
    ("IC", 0), #invalid: I can only subtract from V and X
    ("ABC", 0), #invalid: non-Roman characters
    ("MMMM", 0), #invalid: exceeds 3999
]


def verify(answer):
    try:
        #Extract the function
        namespace = {}
        exec(answer, namespace)
        fn = namespace["roman_to_int"]

        #Evaluate the function with some test cases
        score = sum(fn(inp) == expected for inp, expected in test_cases) / len(test_cases)
        return score
    except Exception:
        # Return 0 if the code is not valid python
        return 0.0

responses = []
for _ in range(10):
    query = "Write a Python function `roman_to_int` that converts" \
    "a Roman numeral string to an integer. The function should:" \
    "1. Handle standard Roman numerals from 1 to 3999." \
    "2. Correctly handle subtractive notation (e.g. IV=4, IX=9, XL=40, XC=90, CD=900)" \
    "3. Return 0 for invalid inputs (empty string, non-Roman characters or invalid patterns like III or VV)" \
    "Give only the function definition, no explanantion or markdown formatting."

    # Generate response
    response = llm.generate([{"role":"user", "content":query}])

    #Verify the response and compute a score
    score = verify(response.content)

    # Store the response and its score
    responses.append((response.content,score))

# Get highest score and the corresponding answer
best_answer, best_score = max(responses, key=lambda x:x[1])

# Show all scores
print([response[1] for response in responses])
