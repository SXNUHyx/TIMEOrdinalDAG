# Reproducible subject-split / temporal-constraint pipeline, original R backend.
# Usage: Rscript run_revision.R INPUT_MINUTE_PANEL.csv OUTPUT_DIRECTORY [extra flags]
# Input columns: subject,time,Steps,HR; use experiments/prepare_fitbit.py first.
args <- commandArgs(trailingOnly=TRUE)
if(length(args)<2L) stop("Usage: Rscript run_revision.R input.csv output_dir [extra flags]")
filearg <- grep("^--file=",commandArgs(),value=TRUE)
if(!length(filearg)) stop("Invoke using Rscript")
script <- normalizePath(sub("^--file=","",filearg[1]))
root <- normalizePath(file.path(dirname(script),"..",".."))
python <- Sys.getenv("TIMEORDINALDAG_PYTHON",unset="python3")
if(!nzchar(Sys.getenv("RSCRIPT"))) Sys.setenv(RSCRIPT=file.path(R.home("bin"),"Rscript"))
input <- normalizePath(args[1],mustWork=TRUE)
output <- normalizePath(args[2],mustWork=FALSE)
extra <- if(length(args)>2L) args[-c(1L,2L)] else character()
cmd <- c(file.path(root,"experiments","run.py"),"real","--backend","original-r",
         "--input",input,"--out",output,"--columns","Steps,HR","--calibrate",
         "--lag","2","--q","4","--scheme","cart","--penalties","2,6,10",
         "--mc","20","--bootstrap","500","--baselines","",extra)
status <- system2(python,shQuote(cmd))
if(status!=0L) stop("Revised analysis failed; inspect the error above. No backend substitution was made.")
