COMPARE_PROMPT_V2 = """\
Compare the requested attributes using only what is stated in the loan document below.

Reply as a markdown table where possible:
| Attribute | Value A | Value B |
|-----------|---------|---------|

If one or both values are absent, write "N/A" in the cell and note it below the table.

Document Context:
{context}

Question: {question}

Comparison:"""
