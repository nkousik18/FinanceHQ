CALCULATE_PROMPT_V2 = """\
Perform the requested calculation using only values from the loan document below.

Reply in this format:
**Given:** <list the values you are using>
**Steps:** <numbered calculation steps>
**Result:** <final answer with unit>

If any required value is missing, reply: "Cannot calculate — missing: <value name>"

Document Context:
{context}

Question: {question}

Calculation:"""
