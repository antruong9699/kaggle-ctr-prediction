# Data source: Kaggle CTR Prediction Competition — https://www.kaggle.com/competitions/predicting-clicks

# Load necessary libraries
library(dplyr)
library(xgboost)
library(vtreat)

# Step 1: Load and preprocess data
analysis_data <- read.csv('/Users/nathan/Downloads/APAN 5200 PAC Competition/analysis_data.csv')
scoring_data <- read.csv('/Users/nathan/Downloads/APAN 5200 PAC Competition/scoring_data.csv')

# Set random seed for reproducibility
set.seed(617)

# Store IDs and CTR
analysis_ids <- analysis_data$id
scoring_ids <- scoring_data$id
ctr_values <- analysis_data$CTR
analysis_data$CTR <- NULL
analysis_data$id <- NULL
scoring_data$id <- NULL


# Create treatment plan for categorical variables
trt <- designTreatmentsZ(dframe = analysis_data,
                         varlist = names(analysis_data))
newvars <- trt$scoreFrame[trt$scoreFrame$code %in% c('clean','lev'),'varName']

# Prepare data using treatment plan
analysis_encoded <- vtreat::prepare(treatmentplan = trt, #vtreat 
                                    dframe = analysis_data,
                                    varRestriction = newvars)
scoring_encoded <- vtreat::prepare(treatmentplan = trt,
                                   dframe = scoring_data,
                                   varRestriction = newvars)

# Convert to matrix format for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(analysis_encoded), #xgb.DMatrix 
                            label = ctr_values)

# Cross-validation with early stopping
cv_model <- xgb.cv(
  data = train_matrix,
  params = list(
    objective = "reg:squarederror",
    eta = 0.01,
    max_depth = 4,
    gamma = 0,
    colsample_bytree = 0.6,
    min_child_weight = 1,
    subsample = 0.8
  ),
  nrounds = 1000,
  nfold = 5,
  metrics = "rmse",
  verbose = TRUE,
  seed = 617
)

# Get the best number of rounds
best_nrounds <- cv_model$best_iteration
print(cv_model$best_iteration)

# Train final model with best number of rounds
final_model <- xgb.train(
  data = train_matrix,
  params = list(
    objective = "reg:squarederror",
    eta = 0.01,
    max_depth = 4,
    gamma = 0,
    colsample_bytree = 0.6,
    min_child_weight = 1,
    subsample = 0.8
  ),
  nrounds = 1000
)

# Prepare test data
test_matrix <- as.matrix(scoring_encoded)

# Generate predictions
final_predictions <- predict(final_model, test_matrix)

# Create submission file
submission <- data.frame(
  id = scoring_ids,
  CTR = final_predictions
)

# Print best RMSE and iteration
print(paste("Best RMSE:", min(cv_model$evaluation_log$test_rmse_mean)))
print(paste("Best Iteration:", cv_model$best_iteration))

# Save submission
write.csv(submission, "xgbcv_predictions3.csv", row.names = FALSE)
