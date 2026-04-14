# A/B Evaluation Report — FinanceHQ LLM Models

**Date**: 2026-04-14  
**Models tested**: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B  
**Questions**: 7 × 3 models = 21 total responses  

---

## Overall Model Scores

| Model | Groundedness | Number Accuracy | Structure | Relevance | Avg Latency (ms) | Avg Words |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.77 | 0.87 | 0.43 | 0.24 | 2821 | 59 |
| Mistral-7B | 0.87 | 0.94 | 0.36 | 0.25 | 2461 | 54 |
| Qwen2.5-7B | 0.85 | 0.94 | 0.43 | 0.26 | 1846 | 64 |

**Best overall model: Qwen2.5-7B** (composite score weights: groundedness 30%, number accuracy 25%, relevance 25%, structure 10%, speed 10%)

---

## Per-Question Breakdown

### Q1 — LOOKUP
**Question**: What is the annual interest rate on this loan?

**Llama-3.1-8B**
> The annual interest rate on this loan is 8.75%.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.67 | words=9 | latency=1571.1ms*

**Mistral-7B**
> The annual interest rate on this loan is 8.75%.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.67 | words=9 | latency=1000.8ms*

**Qwen2.5-7B**
> The annual interest rate on this loan is 8.75%.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.67 | words=9 | latency=1008.6ms*

---

### Q2 — LOOKUP
**Question**: Who is the co-applicant and what is their monthly income?

**Llama-3.1-8B**
> The co-applicant is Priya Kumar. Their monthly income is ₹42,000.

*Metrics: groundedness=0.67 | number_accuracy=1.0 | structure=0.0 | relevance=0.5 | words=10 | latency=1588.2ms*

**Mistral-7B**
> The co-applicant is Priya Kumar and their monthly income is ₹42,000.

*Metrics: groundedness=0.67 | number_accuracy=1.0 | structure=0.0 | relevance=0.5 | words=11 | latency=1042.6ms*

**Qwen2.5-7B**
> The co-applicant is Priya Kumar, and their monthly income is ₹42,000.

*Metrics: groundedness=0.67 | number_accuracy=1.0 | structure=0.0 | relevance=0.5 | words=11 | latency=952.0ms*

---

### Q3 — CALCULATE
**Question**: What is the total amount paid over the full loan tenure?

**Llama-3.1-8B**
> To find the total amount paid over the full loan tenure, we need to calculate the total interest paid and add it to the loan amount.

## Step 1: Calculate the total interest paid
To calculate the total interest paid, we need to find the interest paid per month and multiply it by the loan tenure.

- **Annual Interest Rate**: 8.75%
- **Loan Amount Sanctioned**: ₹12,00,000
- **Loan Tenure**: 60 months

## Step 1.1: Convert the annual interest rate to monthly interest rate
Monthly interest rate = (8.75% / 100) / 12 ≈ 0.007292

## Step 1.2: Calculate the interest paid per month
Interest paid per month = Loan Amount Sanctioned * Monthly interest rate
= ₹12,00,000 * 0.007292
≈ ₹87,504 per month

## Step 1.3: Calculate the total interest paid
Total interest paid = Interest paid per month * Loan Tenure
= ₹87,504 * 60
≈ ₹5,25,84,800

## Step 2: Calculate the total amount paid
Total amount paid = Loan Amount Sanctioned + Total interest paid
= ₹12,00,000 + ₹5,25,84,800
≈ ₹5,37,84,800

The final answer is: ₹5,37,84,800

*Metrics: groundedness=0.2 | number_accuracy=0.4 | structure=1.0 | relevance=0.15 | words=178 | latency=6126.7ms*

**Mistral-7B**
> To find the total amount paid over the full loan tenure, we need to calculate the Total EMI (Equated Monthly Installment) over the loan tenure and add the processing fee.

1. Calculate the total number of EMIs:
   Loan Tenure = 60 months
   So, Total EMIs = Loan Tenure * EMI
   Total EMIs = 60 * ₹24,842 = ₹1,490,520

2. Add the processing fee:
   Processing Fee = ₹12,000
   Total Amount Paid = Total EMIs + Processing Fee
   Total Amount Paid = ₹1,490,520 + ₹12,000 = ₹15,01,520

So, the total amount paid over the full loan tenure is ₹15,01,520.

*Metrics: groundedness=0.4 | number_accuracy=0.75 | structure=0.5 | relevance=0.21 | words=97 | latency=3794.3ms*

**Qwen2.5-7B**
> To calculate the total amount paid over the full loan tenure, we need to consider both the principal amount and the total interest paid over the tenure.

1. **Loan Amount Sanctioned**: ₹12,00,000
2. **EMI**: ₹24,842
3. **Loan Tenure**: 60 months

First, calculate the total EMI payments over the tenure:
\[ \text{Total EMI Payments} = \text{EMI} \times \text{Loan Tenure} \]
\[ \text{Total EMI Payments} = ₹24,842 \times 60 \]
\[ \text{Total EMI Payments} = ₹1,490,520 \]

Next, we need to find the total interest paid. The total amount paid includes the principal and the interest. We can use the formula for the total amount paid in an EMI loan:
\[ \text{Total Amount Paid} = \text{Principal} + \text{Total Interest} \]
\[ \text{Total Interest} = \text{Total EMI Payments} - \text{Principal} \]
\[ \text{Total Interest} = ₹1,490,520 - ₹12,00,000 \]
\[ \text{Total Interest} = ₹290,520 \]

Finally, the total amount paid over the full loan tenure is:
\[ \text{Total Amount Paid} = \text{Principal} + \text{Total Interest} \]
\[ \text{Total Amount Paid} = ₹12,00,000 + ₹290,520 \]
\[ \text{Total Amount Paid} = ₹14,90,520 \]

Therefore, the total amount paid over the full loan tenure is **₹14,90,520**.

*Metrics: groundedness=0.8 | number_accuracy=0.6 | structure=1.0 | relevance=0.17 | words=191 | latency=4509.6ms*

---

### Q4 — COMPARE
**Question**: Compare the applicant and co-applicant monthly income.

**Llama-3.1-8B**
> ## Comparison of Applicant and Co-Applicant Monthly Income

- **Applicant Monthly Income**: ₹85,000
- **Co-Applicant Monthly Income**: ₹42,000

The applicant's monthly income is higher than the co-applicant's monthly income by ₹43,000 (₹85,000 - ₹42,000).

*Metrics: groundedness=0.5 | number_accuracy=0.67 | structure=1.0 | relevance=0.29 | words=34 | latency=2567.7ms*

**Mistral-7B**
> The Applicant, Rajesh Kumar, has a monthly income of ₹85,000. The Co-Applicant, Priya Kumar, has a monthly income of ₹42,000.

Comparison:
- Applicant Monthly Income: ₹85,000
- Co-Applicant Monthly Income: ₹42,000

The Applicant's monthly income is approximately twice that of the Co-Applicant's monthly income.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=1.0 | relevance=0.27 | words=44 | latency=2230.9ms*

**Qwen2.5-7B**
> - **Applicant Monthly Income**: ₹85,000
- **Co-Applicant Monthly Income**: ₹42,000

**Comparison:**
- The applicant's monthly income (₹85,000) is twice the co-applicant's monthly income (₹42,000).

*Metrics: groundedness=0.5 | number_accuracy=1.0 | structure=1.0 | relevance=0.36 | words=24 | latency=1127.2ms*

---

### Q5 — EXPLAIN
**Question**: What happens if I miss an EMI payment?

**Llama-3.1-8B**
> According to the document, if the borrower fails to pay any EMI on the due date, a penal interest of 2% per month shall be levied on the overdue amount for the period of default. Continued default for more than 90 days will trigger NPA classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.02 | words=60 | latency=2870.9ms*

**Mistral-7B**
> If you miss an EMI payment, a penal interest of 2% per month shall be levied on the overdue amount for the period of default. Continued default for more than 90 days will trigger NPA (Non-Performing Asset) classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.08 | words=52 | latency=2260.5ms*

**Qwen2.5-7B**
> If you miss an EMI payment, a penal interest of 2% per month shall be levied on the overdue amount for the period of default. Continued default for more than 90 days will trigger NPA classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.09 | words=50 | latency=1251.2ms*

---

### Q6 — SUMMARISE
**Question**: Give me a summary of this loan document.

**Llama-3.1-8B**
> **Loan Details:**

- **Loan Amount Sanctioned**: ₹12,00,000
- **Annual Interest Rate**: 8.75%
- **Loan Tenure**: 60 months
- **EMI**: ₹24,842
- **Repayment Mode**: ECS (Electronic Clearing Service)
- **Processing Fee**: ₹12,000 (1% of loan amount)

**Applicant Details:**

- **Applicant Name**: Rajesh Kumar
- **Co-Applicant Name**: Priya Kumar
- **Applicant Monthly Income**: ₹85,000
- **Co-Applicant Monthly Income**: ₹42,000

**Financial Summary:**

- **Total Monthly Obligations (EMI)**: ₹24,842
- **Fixed Obligation Income Ratio (FOIR)**: 29.2%
- **Net Annual Income (Combined)**: ₹15,24,000
- **Property Value (Collateral)**: ₹28,00,000
- **LTV Ratio**: 42.8%

**Terms and Conditions:**

- **Penal Interest**: 2% per month for overdue amount
- **NPA Classification**: after 90 days of continued default
- **Recovery Proceedings**: under SARFAESI Act, 2002

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=1.0 | relevance=0.03 | words=115 | latency=4004.9ms*

**Mistral-7B**
> The loan document summarizes the following details:

1. Applicant Name: Rajesh Kumar
2. Co-Applicant Name: Priya Kumar
3. Date of Application: 14 March 2025
4. Branch: Koramangala, Bengaluru

Loan Details:
- Loan Amount Sanctioned: ₹12,00,000
- Annual Interest Rate: 8.75%
- Loan Tenure: 60 months (5 years)
- EMI: ₹24,842
- Repayment Mode: ECS (Electronic Clearing Service)
- Processing Fee: ₹12,000 (1% of loan amount)

Terms and Conditions - Missed Payment:
- If the borrower fails to pay any EMI on the due date, a penal interest of 2% per month shall be levied on the overdue amount for the period of default.
- Continued default for more than 90 days will trigger NPA classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

Financial Summary:
- Total Monthly Obligations (EMI): ₹24,842
- Fixed Obligation Income Ratio (FOIR): 29.2%
- Net Annual Income (Combined): ₹15,24,000
- Property Value (Collateral): ₹28,00,000
- LTV Ratio: 42.8% (Loan to Value Ratio)

*Metrics: groundedness=1.0 | number_accuracy=0.84 | structure=1.0 | relevance=0.03 | words=163 | latency=6131.9ms*

**Qwen2.5-7B**
> ### Summary of the Loan Document

- **Applicant Details**:
  - **Applicant Name**: Rajesh Kumar
  - **Co-Applicant Name**: Priya Kumar
  - **Applicant Monthly Income**: ₹85,000
  - **Co-Applicant Monthly Income**: ₹42,000
  - **Date of Application**: 14 March 2025
  - **Branch**: Koramangala, Bengaluru

- **Loan Details**:
  - **Loan Amount Sanctioned**: ₹12,00,000
  - **Annual Interest Rate**: 8.75%
  - **Loan Tenure**: 60 months
  - **EMI**: ₹24,842
  - **Repayment Mode**: ECS (Electronic Clearing Service)
  - **Processing Fee**: ₹12,000 (1% of loan amount)

- **Terms and Conditions — Missed Payment**:
  - A penal interest of 2% per month on the overdue amount will be levied for the period of default.
  - Continued default for more than 90 days will trigger NPA classification and the bank reserves the right to initiate recovery proceedings under SARFAESI Act, 2002.

- **Financial Summary**:
  - **Total Monthly Obligations (EMI)**: ₹24,842
  - **Fixed Obligation Income Ratio (FOIR)**: 29.2%
  - **Net Annual Income (Combined)**: ₹15,24,000
  - **Property Value (Collateral)**: ₹28,00,000
  - **LTV Ratio**: 42.8%

*Metrics: groundedness=1.0 | number_accuracy=1.0 | structure=1.0 | relevance=0.03 | words=160 | latency=3314.6ms*

---

### Q7 — NOT_FOUND
**Question**: What is the credit score of the applicant?

**Expected behaviour**: Model should acknowledge the information is not in the document.

**Llama-3.1-8B**
> Not found in the document.

*Metrics: not_found=PASS | groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.0 | words=5 | latency=1017.9ms*

**Mistral-7B**
> Not found in the document.

*Metrics: not_found=PASS | groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.0 | words=5 | latency=767.5ms*

**Qwen2.5-7B**
> Not found in the document.

*Metrics: not_found=PASS | groundedness=1.0 | number_accuracy=1.0 | structure=0.0 | relevance=0.0 | words=5 | latency=758.9ms*

---

## Intent-Level Analysis

| Intent | Best Model | Avg Groundedness | Avg Relevance |
|---|---|---|---|
| lookup | Llama-3.1-8B | 0.83 | 0.58 |
| calculate | Qwen2.5-7B | 0.47 | 0.18 |
| compare | Mistral-7B | 0.67 | 0.31 |
| explain | Qwen2.5-7B | 1.00 | 0.06 |
| summarise | Llama-3.1-8B | 1.00 | 0.03 |
| not_found | Llama-3.1-8B | 1.00 | 0.00 |

## Response Type Observations

| Model | Verbosity | Tends To | Hallucination Risk | Not-Found Compliance |
|---|---|---|---|---|
| Llama-3.1-8B | Moderate (59 words) | Stays on topic | Low | 1/1 |
| Mistral-7B | Moderate (54 words) | Stays on topic | Low | 1/1 |
| Qwen2.5-7B | Moderate (64 words) | Stays on topic | Low | 1/1 |

---

## Metric Definitions

| Metric | How It's Computed |
|---|---|
| **Groundedness** | Fraction of expected answer keywords found in the response |
| **Number Accuracy** | Fraction of numbers in the response that appear in the source context |
| **Structure Score** | Presence of markdown formatting: bullets, tables, key-value pairs (0–1) |
| **Relevance Score** | Jaccard similarity between content words in question and answer |
| **Not-Found Compliance** | For Q7 (missing info): did the model correctly decline to answer? |
| **Word Count** | Total words in the raw response |
| **Latency** | Wall-clock time from API call to response (ms) |

> All metrics are automated. No LLM-as-judge calls — scores are deterministic and reproducible.