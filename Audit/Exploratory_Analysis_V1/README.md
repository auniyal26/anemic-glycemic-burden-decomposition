# Exploratory Analysis V1

Status: frozen exploratory proof-of-concept

Complete A/G/P sample: 9,743

Authoritative source:
- raw NHANES XPT files in Data/NHANES
- SHA256 hashes in Audit/13_raw_data_manifest_sha256.csv

Preservation chain:
raw XPT -> rebuilt reference -> domain-informed A/G representations -> confounder-residualized PHQ item space -> standardized residual space -> ICA latent space

Core transform parameters are serialized in:
- Parameters/global_transform_parameters.json
- Parameters/0506_transform_parameters.json
- Parameters/0708_transform_parameters.json

Final exploratory latent solution:
- K = 3
- ICA seed = 42
- independently fit by NHANES cycle
- mathematically aligned only after independent fitting

Current defensible exploratory result:
- a cross-cycle stable somatic PHQ dimension is recovered;
- preliminary hematological association with that dimension requires survey-correct, confounder-aware and ordinal-factor confirmation;
- glycemic association is weaker;
- A×G is not yet reproducible.

Not confirmatory evidence:
- current p-values;
- causal interpretation;
- final number of PHQ factors;
- population-representative NHANES estimates.

Required next:
- DAG-based adjustment sets
- expanded X variables
- full NHANES survey design
- fasting-glucose/diabetes sensitivity
- ordinal CFA/WLSMV
- missingness and multiple-testing control
