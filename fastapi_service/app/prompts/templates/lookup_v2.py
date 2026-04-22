LOOKUP_PROMPT_V2 = """\
Extract the requested value from the loan document below. Reply in this format:

**Value:** <exact value from document, or "Not found">
**Source:** <quote the sentence or field where you found it>

Document Context:
{context}

Question: {question}

Answer:"""
