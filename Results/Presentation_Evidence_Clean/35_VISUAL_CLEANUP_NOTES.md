# Script 35 — visual cleanup notes

## Changes made

1. **Cohort flow corrected**
   The flow is now a true sequential intersection:
   same-person core → Hb → HbA1c → complete PHQ-9 → full X3 survey-analysis requirements.

2. **x / y / P corrected**
   `P` is no longer shown as a component of `y`.

   Working structure:
   `x_raw → T(x) → y_working=[A,G,A×G,X] → P`

   Final thesis-level `y → components → y_hat` reconstruction remains future work.

3. **A/G group figure corrected**
   Main descriptive figure now uses the NHANES survey design and shows 95% confidence intervals.
   Raw N is retained only as sample-size annotation.

4. **Discovery A/G coefficient figure corrected**
   The estimates are unchanged.
   The figure explicitly states that A and G have different predictor units and raw coefficient magnitudes are not directly comparable.

5. **CFA / loading styling corrected**
   Categorical model/item comparisons are shown as points rather than line trajectories.

6. **Fasting glucose sensitivity corrected**
   A single shared-axis HbA1c-vs-fasting coefficient plot is intentionally not generated.
   The two G definitions have different units; use the table to compare direction, uncertainty and inferential support.

## Integrity

Preserved core SHA256 before/after identical: **True**

No scientific definition was altered by this script.
