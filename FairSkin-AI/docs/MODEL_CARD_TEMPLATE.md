# Model Card — FairSkin-AI

## Intended use
Research and education on subgroup performance auditing in dermatology image classification.

## Out-of-scope use
Clinical diagnosis, triage, treatment decisions, patient-facing deployment, or claims of safety/equity without independent clinical validation.

## Training data
Record exact dataset version, number of usable images after failed/broken URLs, class distribution, Fitzpatrick distribution, and split sizes.

## Model
Record architecture, pretrained-weight source, input resolution, training epochs, optimizer, learning rate, mitigation mode, and random seed.

## Evaluation
Report overall accuracy/balanced accuracy/macro F1 and the same metrics by skin group. Report subgroup sample sizes and max-minus-min gaps. Include bootstrap confidence intervals.

## Limitations
Document label noise, image-source bias, incomplete source URLs, limitations of Fitzpatrick type as a skin-tone proxy, class imbalance, domain shift, and absence of clinical validation.
