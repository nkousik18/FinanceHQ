EXPLAIN_PROMPT_V2 = """\
Explain the requested term or clause from the loan document in plain language.

Reply in this format:
**In simple terms:** <1–2 sentence plain-language explanation>
**From the document:** <direct quote of the relevant clause>

If the term is not in the document, say: "This is not covered in the provided document."

Document Context:
{context}

Question: {question}

Explanation:"""
