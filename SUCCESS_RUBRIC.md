# Success Rubric and Task Spec

## Task Spec

* The system must be readable at an 8th-grade reading level or below.
* The system must be faithful to the sources of truth. This means the system must not add information, change existing information, or delete significant information. 
* The system must not hallucinate or add clinically meaningful unsupported claims, which can be potentially harmful.
* The system must show improvement over the single-agent baseline.

## Quantitative system performance from challenge overview document: 

* Readability: ≥80% of outputs at ≤8th-grade reading level (Flesch-Kincaid).  
* Faithfulness: ≥85% factual fidelity on the human-annotated test set, verified against human labels.  
* Hallucination rate: \<10% of outputs contain a clinically meaningful unsupported claim.  
* Multi-agent vs. single-agent baseline: ≥15% absolute improvement in faithfulness score.

### Verifier agent quality:

* Verifier agrees with human annotators ≥80% of the time on faithfulness labels (target Cohen's κ ≥ 0.6).

| Criteria | Good | Moderate | Poor |
| :---- | :---- | :---- | :---- |
| **Readability** | ≥80% of outputs at ≤8th-grade reading level | 50-79% of outputs at ≤8th-grade reading level | \<50% of outputs at ≤8th-grade reading level |
| **Faithfulness** | ≥85% factual fidelity on the human-annotated test set | 70-84% factual fidelity on the human-annotated test set | \<70% factual fidelity on the human-annotated test set |
| **Hallucination rate** | \<10% of outputs contain a clinically meaningful unsupported claim | 10-20% of outputs contain a clinically meaningful unsupported claim | \>20% of outputs contain a clinically meaningful unsupported claim |
| **Multi-agent vs single-agent baseline** | ≥15% absolute improvement in faithfulness score | 7-14.99% absolute improvement in faithfulness score | 0-6.99% absolute improvement in faithfulness score |
| **Verifier agent quality** | Verifier agrees with human annotators ≥80% of the time on faithfulness labels | Verifier agrees with human annotators 70-79.99% of the time on faithfulness labels | Verifier agrees with human annotators \<70% of the time on faithfulness labels |
