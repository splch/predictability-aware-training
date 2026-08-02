# Audit checklist — consensus conventions for a one-column workshop/arXiv ML-systems paper

Evidence codes: A = ML-methods cluster (MLSys/ICML/StickyMoE), B = systems cluster (acmart/IEEEtran), C = landmark cluster (OLMoE/MoE-Infinity/Switch/MoE-Beyond). "All 3" = agreed by all three analyst reports.

## Document class & preamble
1. Root at `\documentclass{article}`, no custom class. [all 3]
2. One column, 10pt, letterpaper. [target style; matches A-P3, C-OLMoE/Switch]
3. Core preamble: microtype, graphicx, subcaption, booktabs, amsmath (+amssymb/mathtools/amsthm for math-heavy papers), hyperref. [all 3, mandatory; ams extras A-P2/P3, C-OLMoE]
4. hyperref loaded late with `colorlinks=true` and muted colors (dark blue / blue!70!black / green!50!black), never default red boxes. [A-P3, C-OLMoE/Switch; B inherits from acmart]
5. natbib with numbered compressed citations (`[numbers,sort&compress]`); `\citep`/`\citet`. [target style; A-P3, C-OLMoE/Switch; B numeric equivalent]
6. BibTeX (`\bibliographystyle` + `\bibliography`), never biblatex, never manual thebibliography. [all 3; C-MoE-Beyond manual = anti-pattern]
7. `\bibliographystyle{plainnat}` + `\bibliography{references}` in the real paper (commented out in the bare template). [target style; A-P3, C]
8. System-name macro `\textsc{...}\xspace` used consistently. [A-P3, B-P3, C-OLMoE/MoE-Infinity]
9. `\newcommand{\theHalgorithm}{\arabic{algorithm}}` if algorithm floats are used. [A-P1/P2, C-MoE-Infinity]

## Abstract
10. Single paragraph, ~150–250 words. [all 3]
11. Rhetoric: context/problem → prior-work gap → "We propose X" + mechanism → headline quantitative results. [all 3, mandatory]
12. Headline numbers bolded or otherwise explicit. [A-P3 `\textbf`, C all papers; B quotes up-to-N×]
13. Artifact/code URL at the end of the abstract. [A-P2/P3, C 3/4; B partial]
14. Self-contained; no citation overload. [all 3]

## Introduction
15. Structure: broad context → problem → categorized prior remedies → proposal paragraph. [all 3]
16. Proposal names 2–3 components as "First… Second… Third…" or "(i)…(ii)…". [A-P1/P2, B all]
17. Ends with an explicit itemized contributions list (mandatory); bold-tagged bullets or bold run-in "Contribution N" paragraphs both acceptable. [all of B; A-P3; C 3/4]
18. Results-summary sentence with headline numbers in or right after the contributions. [all 3]
19. Page-1 motivating figure or table with headline numbers. [B-P2, C-OLMoE/MoE-Infinity/Switch]

## Section skeleton
20. Skeleton: Introduction → Background/Related Work → (Motivation) → Method → Experiments → Systems/Evaluation → Discussion/Limitations → Conclusion → bibliography → appendix. [all 3, mandatory]
21. Related Work organized by thematic categories ("system-level", "post-hoc", "training-time"), each closing with a contrast to the proposed method. [all 3]
22. Related Work position: early (template default) or late after Evaluation — pick one deliberately. [disagreement: B-P1/P3 + C-OLMoE/Switch late; A-P1/B-P2/C-MoE-Infinity early]
23. Method section opens with an "Overview" subsection + architecture figure, then one subsection per named component. [all of B, A-P1/P2]
24. Experiments open with "Experimental Setup" (models, datasets, hardware, workloads, hyperparameters) as the FIRST subsection. [all 3, mandatory]
25. Ablation studies are the LAST subsection of Experiments. [all 3, mandatory]
26. Baselines are NAMED prior systems, cited, and compared on shared metrics. [all of B, A-P2/P3, C-MoE-Infinity]
27. Explicit Limitations/Boundary section or paragraph before Conclusion. [A-P1/P3, B-P3, C-MoE-Beyond/OLMoE]
28. Unnumbered Acknowledgments before the bibliography. [all of B, C-OLMoE/Switch]
29. Appendix AFTER the bibliography, referenced from the main text. [A-P2/P3, B-P3, C 3/4, mandatory]

## Floats, tables, figures
30. Float placement from the `[t]` family (`[t]`, `[!t]`, `[ht]`). [all 3, mandatory]
31. `\centering` inside every float. [all 3, mandatory]
32. Figure captions BELOW the graphic; table captions ABOVE the table. [all 3 mandatory for figures; tables: A-P2/P3, all of B, C-OLMoE + target style. Alt below: A-P1, C 3/4]
33. `\label` immediately after `\caption`, prefixes `fig:`/`tab:`/`sec:`/`eq:` (+`app:`). [A-P2/P3, B-P1/P3, C all, mandatory]
34. Tables use booktabs ONLY: `\toprule`/`\midrule`/`\bottomrule`; zero `\hline`, no vertical rules. [A all, B-P2, C-OLMoE/Switch, mandatory]
35. Large tables: `\footnotesize` wrapper or `\resizebox{\linewidth}{!}{...}`; `\midrule`+`\multicolumn` cohort separators; `\textbf` best cells; metric-direction arrows `($\uparrow$)`. [C-OLMoE/MoE-Infinity/Switch, B-P2]
36. Captions open with a bold lead-in sentence and are self-contained (comparison, protocol, takeaway). [A-P2, C-OLMoE/Switch]

## Math, citations, cross-references
37. Numbered `equation`/`align` for every loss term, one equation per term, each `\label{eq:...}`'d and referenced from text. [A all, C-OLMoE/Switch, mandatory]
38. Semantic citation keys (`fedus2022switch`, `xue2024moeinfinity`). [all 3, mandatory]
39. Cross-references via `\cref`/`\Cref` (cleveref `[capitalize,noabbrev]`) or explicit `Figure~\ref{}` wrappers — one style consistently. [A-P2/P3 + C-OLMoE/MoE-Infinity vs B-P1/P3]
40. Multi-panel figures via subcaption `subfigure` environments, per-panel captions, sized in fractions of `\linewidth`. [A all, B-P1/P2, C-MoE-Beyond]
