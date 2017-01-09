# Collect stacking data in a csv file

setwd(file.path(getwd(), "scripts"));
source(".Rprofile");

setwd("../");
setwd(snakemake@input[[1]]);
source(snakemake@config[["assembling-script"]]);

setwd("../../..");
write.csv(loso_pred_res, file=snakemake@output[[1]]);
