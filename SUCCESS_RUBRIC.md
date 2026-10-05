**Success Rubric**

Quantitative system performance from challenge overview document: 

* Readability: ≥80% of outputs at ≤8th-grade reading level (Flesch-Kincaid).  
* Faithfulness: ≥85% factual fidelity on the human-annotated test set, verified against human labels.  
* Hallucination rate: \<10% of outputs contain a clinically meaningful unsupported claim.  
* Multi-agent vs. single-agent baseline: ≥15% absolute improvement in faithfulness score.

Verifier agent quality:

* Verifier agrees with human annotators ≥80% of the time on faithfulness labels (target Cohen's κ ≥ 0.6).

Other metrics to consider: 

* Medical jargon density  
  * Build Jargon Density Scorer: Implement a tokenizer and medical dictionary look-up to calculate the percentage of complex clinical terms per text passage.  
* Output length  
* Refusal criteria

| Criteria | Good | Moderate | Poor |
| :---- | :---- | :---- | :---- |
| **Readability** | ≥80% of outputs at ≤8th-grade reading level | 50-79% of outputs at ≤8th-grade reading level | \<50% of outputs at ≤8th-grade reading level |
| **Faithfulness** | ≥85% factual fidelity on the human-annotated test set | 70-84% factual fidelity on the human-annotated test set | \<70% factual fidelity on the human-annotated test set |
| **Hallucination rate** | \<10% of outputs contain a clinically meaningful unsupported claim | 10-20% of outputs contain a clinically meaningful unsupported claim | \>20% of outputs contain a clinically meaningful unsupported claim |
| **Multi-agent vs single-agent baseline** | ≥15% absolute improvement in faithfulness score | 7-14% absolute improvement in faithfulness score | 0-6% absolute improvement in faithfulness score |
| **Verifier agent quality** | Verifier agrees with human annotators ≥80% of the time on faithfulness labels | Verifier agrees with human annotators 70-84% of the time on faithfulness labels | Verifier agrees with human annotators \<70% of the time on faithfulness labels |

