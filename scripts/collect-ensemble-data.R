# Collect stacking data in a csv file

setwd(file.path(getwd(), "scripts"))
source(".Rprofile")
setwd("../")

setwd(snakemake@input[[1]])
source(snakemake@config[["ensemble-assembler"]])

library(plyr)
library(dplyr)
library(tidyr)
library(lubridate)

df <- assemble_predictions()

setwd("../../..")
write.csv(df, file=snakemake@output[[1]])
