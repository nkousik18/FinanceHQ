SUMMARISE_PROMPT_V2 = """\
Summarise the loan document below using only the information provided.

Reply with bullet points under these headings (skip any heading with no data):
• **Applicant** — name, employment, income
• **Loan** — amount, tenure, interest rate, repayment mode
• **Financials** — assets, liabilities, net worth
• **Key flags** — anything unusual or worth noting

Document Context:
{context}

Question: {question}

Summary:"""
