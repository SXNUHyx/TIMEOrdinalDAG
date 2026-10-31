# Implementation provenance

The supplied revision archive contained Python reference code and immutable hashes of the submitted R source. UPSTREAM_SHA256.json records that attachment baseline, not the modified checkout. The original submitted manuscript is retained in manuscript/sn-article_original.tex.

The revised R adapter executes the live ../ContinueEffectsRcode tree; it no longer uses an unmodified shadow copy. R thresholds and temporal guards were tested independently, and all R files were parsed. Full BiDAG execution remains blocked by missing CRAN/Bioconductor dependencies and is not claimed as validated.

The Python reference backend retains the latent Gaussian threshold model, Monte Carlo E-step and latent interval intervention, but uses greedy DAG search rather than BiDAG MCMC. Its numerical results must be explicitly identified as Python-reference results. It is not a drop-in numerical reproduction.

Revision corrections include a nonzero prespecified lag-one effect in the three-variable simulation and disabling oracle effect comparisons for heterogeneous random-offset populations. Known-zero no-path effects remain covered by tests. Full source modifications are reviewable in git diff.

The original-r adapter exports the R-learned latent structure and parameters; its OCE is evaluated by the shared Python latent-interval integral. It is not an end-to-end execution of OrdinalEffects.R. Pure R score/threshold tests do not establish original R effect-routine compatibility.
