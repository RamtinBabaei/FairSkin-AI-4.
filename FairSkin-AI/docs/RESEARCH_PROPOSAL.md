# Mini research proposal

## Title
FairSkin-AI: Auditing and Mitigating Skin-Tone Performance Disparities in Dermatology Image Classification

## Research question
How consistently does a dermatology image classifier perform across Fitzpatrick skin-tone groups, and to what extent can simple data- and loss-level mitigation strategies reduce subgroup performance gaps without substantially degrading overall performance?

## Experimental plan
1. Train a transfer-learning baseline on Fitzpatrick17k using the dataset's three broad diagnostic partitions.
2. Report overall metrics and subgroup metrics for I-II, III-IV, and V-VI.
3. Quantify disparity using max-minus-min subgroup gaps and bootstrap confidence intervals.
4. Compare the baseline with class-weighted loss and intersectional class-by-group sampling.
5. Report both fairness and utility; do not claim clinical validity.

## Important limitations
Fitzpatrick type was designed around sun response rather than as a perfect representation of skin colour. Image-source bias, label noise, class imbalance, and missing/broken source images can affect conclusions. Results should be presented as a dataset/model audit rather than as evidence of clinical safety.
