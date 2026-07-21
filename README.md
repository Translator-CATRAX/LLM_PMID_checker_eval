# Evaluation of LLM_PMID_Checker
## Set-up
### Data preparation
- create a data folder at the root of this folder
- download SemMedDB input data:
1. Releases from [here](https://github.com/Translator-CATRAX/LLM_PMID_Checker/tree/main)
2. SemMedDB KGX from [here](https://github.com/Translator-CATRAX/SemMedDB-KGX/)

## Methodology
 In Knowledge Graph (KG) machine learning, performance is rarely uniform across the graph. If only aggregate metrics measurements (like global F1-score) are performed, we will end up with _The majority class trap_, where a model appears highly performant because it excels at predicting common, high-degree nodes/classes, while failing catastrophically on the "long tail" of rare biological entities.

To build a robust test suite, we move away from simple random sampling and toward stratified multi-dimensional sampling.

Here is a framework for how to approach this problem.

### Identifying the sources of bias
Before building the sampler, we must formalize the dimensions that impact your model:

- **Semantic bias** (ontology/biolink classes): Some classes in the Biolink model (e.g., biolink:Gene) are much more densely connected and well-represented than others (e.g., biolink:Event). The model may learn _shortcuts_ based on class-specific patterns rather than true relational logic.
- **Structural bias** (node degree):
1. _High-degree nodes_ (hubs): These provide massive amounts of context but can lead to "over-smoothing" in Graph Neural Networks, making it hard to distinguish specific attributes.
2. Low-degree nodes (leaves): These lack structural context, making the model rely solely on the predicate/attribute logic, which might be harder if the features are sparse.
- **Attribute sparsity**: The distribution of True vs. False for a specific edge attribute (here the PMID) is likely highly imbalanced.

### Metrics for the sources of bias:



### Stratified sampling:
Instead of sampling edges randomly, we create a stratified grid where each datapoint in the grid represents a combination of features. We then sample an equal (or proportionally representative) number of samples from each datapoint (=bin of a given features combinaison).