# Collect stacking data in a csv file

setwd(snakemake@input[[1]]);
source(snakemake@config[["assembling-script"]]);

setwd("../../..");
write.csv(loso_pred_res, file=snakemake@output[[1]]);
