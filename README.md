# TIMEOrdinalDAG
A unified framework for ordinal causal discovery and intervention effect estimation in wearable physiological time-series data.
TIMEOrdinalDAG is an open-source framework for ordinal causal analysis of wearable physiological time-series data.

The framework integrates:

- Subject-based Feature Augmentation for transforming multivariate time-series into lagged feature matrices;
- CART-based Supervised Discretization for converting continuous variables into ordinal states;
- Latent Gaussian DAG Learning using the Ordinal Structural EM (OSEM) algorithm;
- Ordinal Causal Effect (OCE) estimation for quantifying intervention effects between ordinal variables.

The proposed pipeline provides a unified workflow from time-series preprocessing to causal discovery and intervention analysis, enabling interpretable causal inference in wearable sensor data and sports science applications.

Code Execution Order

The scripts in `ContinueEffectsRcode/Application` should be executed in the following order:

`merge_minute_data.R`  
↓  
`final_data.R`  
↓  
`Binning.R`  
↓  
`Heart_boot.R`  
↓  
`Heart_applition.R`  

This order ensures proper data preprocessing, transformation, and final causal analysis.
