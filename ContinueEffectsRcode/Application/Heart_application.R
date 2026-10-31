# Supersedes the submitted row-bootstrap/plotting script.
# The revised pipeline estimates effects inside each fitted DAG and resamples
# complete subjects; it never infers interventions from a display CPDAG.
filearg <- grep("^--file=",commandArgs(),value=TRUE)
if(!length(filearg)) stop("Use Rscript with input.csv and output_dir arguments")
script <- normalizePath(sub("^--file=","",filearg[1]))
source(file.path(dirname(script),"run_revision.R"))
