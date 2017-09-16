## Script for taking data files from adaptively-weighted-ensemble repository

library(plyr)
library(dplyr)
library(tidyr)

#' Assembling code taken from adaptively-weighted-ensemble repository
#'
#' Assemble leave-one-season-out or test phase predictions made by kde, kcde, and sarima
#' models on training data.
#'
#' @param preds_path path to leave-one-season-out or test phase predictions.
#' @param regions character vector specifying regions for which to get predictions,
#'   "National" or "Regionk" for k in 1, ..., 10.
#' @param seasons character vector specifying seasons for which to get predictions,
#'   "2011/2012" or "2011-2012" or "*" for all seasons
#' @param models character vector specifying models for which to get predictions,
#'   "kde", "kcde", or "sarima"
#' @param prediction_targets character vector specifying prediction targets,
#'   "onset", "peak_week", "peak_inc", "ph_1_inc", ..., "ph_4_inc"
#' @param prediction_types character vector specifying prediction types,
#'   "log_score", "competition_log_score", or "bin_log_probs"
#'
#' @return a data frame with predictions.
#'
#' @export
assemble_predictions <- function(
  preds_path,
  regions = c("National", paste0("Region", 1:10)),
  seasons = "*",
  models = c("kde", "kcde", "sarima"),
  prediction_targets = c("onset", "peak_week", "peak_inc", "ph_1_inc", "ph_2_inc", "ph_3_inc", "ph_4_inc"),
  prediction_types = c("log_score", "competition_log_score", "bin_log_probs")
  ) {
  ## Seasons in format used in prediction file names
  seasons <- gsub("/", "-", seasons)

  ## names of files with prediction results to load
  model_region_season_combos <- expand.grid(
    model = models,
    region = regions,
    season = seasons,
    stringsAsFactors = TRUE
  )
  file_name_patterns <- apply(model_region_season_combos, 1,
    function(combo) {
      paste0(preds_path, "/", combo["model"], "-", combo["region"], "-", combo["season"], "*")
    })
  file_names <- Sys.glob(file_name_patterns)

  ## load the prediction results
  pred_res <- rbind.fill(lapply(
    file_names,
    function(file_path) {
      region_val <- names(unlist(sapply(regions, function(region_val) grep(paste0(region_val, "-"), file_path))))
      readRDS(file_path) %>%
        mutate(region = region_val)
    }
  ))

  ## narrow down to the specified prediction targets and types
  prediction_cols_to_keep_templates <- outer(prediction_targets, prediction_types,
    function(target, type) {
      paste0(target, "_", ifelse(type == "bin_log_probs", "bin_.*_log_prob", type))
    }) %>%
    as.vector()
  prediction_cols_to_keep <- lapply(
    prediction_cols_to_keep_templates,
    function(pattern) grep(pattern, names(pred_res))) %>%
    unlist()
  prediction_cols_to_keep <- names(pred_res)[prediction_cols_to_keep]
  target_pred_res <- pred_res %>%
    select_("model",
      "region",
      "analysis_time_season",
      "analysis_time_season_week",
      .dots = prediction_cols_to_keep)

  return(target_pred_res)
}


output_file <- snakemake@output[[1]]
input_dir <- snakemake@input[[1]]

df <- assemble_predictions(input_dir)
write.csv(df, gzfile(output_file))
