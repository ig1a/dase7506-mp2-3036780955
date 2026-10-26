# Train-only parser development

Before validation, three diagnostic passes and two parser corrections were made; no classifier was fitted during diagnosis.

1. Initial train audit found false flags caused by currency dollar delimiters crossing lines, removal of spans joining unrelated fragments, percentage notation such as 8/40*100=20%, and quotient-with-remainder prose. The original 20-candidate sample contained 8 fully clean flags, 1 mixed true/false, and 11 false-flag cases.
2. Restrict single-dollar math spans to a line, preserve removed-span barriers/newlines, skip percent and remainder/rounding chunks. Recheck train only.
3. Tighten trailing LaTeX-unit removal to disallow residual arithmetic; use a distinct barrier for removed math spans. Recheck train and review the final deterministic 20-candidate sample. No validation data inspected in these steps.

Final parser SHA and audit gate are in diagnostic.json and manual-audit.json. Percentage equations and many prose/currency equations deliberately remain unknown; absence of a flag is not proof of valid reasoning. The audit is by the AI assistant, not independent human annotation. No rule changes after the single validation run.
