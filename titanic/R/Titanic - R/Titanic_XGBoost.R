library(dplyr)
library(CatEncoders)
library(caret)
library(xgboost)

# Load Data
train_data = read.csv("../../data/train.csv", stringsAsFactors = FALSE)
test_data = read.csv("../../data/test.csv", stringsAsFactors = FALSE)

# Merge Data
all_data = bind_rows(train_data, test_data)
all_data_clean <- all_data

# Fill NA in Numeric Features with median (Imputing)
all_data_clean$Age[is.na(all_data_clean$Age)] <- median(all_data_clean$Age, na.rm = TRUE)
all_data_clean$Fare[is.na(all_data_clean$Fare)] <- median(all_data_clean$Fare, na.rm = TRUE)
all_data_clean$Parch[is.na(all_data_clean$Parch)] <- median(all_data_clean$Parch, na.rm = TRUE)
all_data_clean$Pclass[is.na(all_data_clean$Pclass)] <- median(all_data_clean$Pclass, na.rm = TRUE)
all_data_clean$Age[is.na(all_data_clean$SibSp)] <- median(all_data_clean$SibSp, na.rm = TRUE)

# Scale all Numeric Features
preProcValues <- preProcess(all_data_clean[, -2], method = c("scale"))
all_data_clean <- predict(preProcValues, all_data_clean)
all_data_clean$PassengerId = as.integer(rownames(all_data_clean))

# Encode Categorical Features
enc = LabelEncoder.fit(all_data_clean$Name)
all_data_clean$Name <- transform(enc, all_data_clean$Name)
enc = LabelEncoder.fit(all_data_clean$Sex)
all_data_clean$Sex <- transform(enc, all_data_clean$Sex)
enc = LabelEncoder.fit(all_data_clean$Cabin)
all_data_clean$Cabin <- transform(enc, all_data_clean$Cabin)
enc = LabelEncoder.fit(all_data_clean$Embarked)
all_data_clean$Embarked <- transform(enc, all_data_clean$Embarked)
enc = LabelEncoder.fit(all_data_clean$Ticket)
all_data_clean$Ticket <- transform(enc, all_data_clean$Ticket)

# Split back into Train and Test Datasets
train_val_clean = all_data_clean[1:891,]
test_clean = all_data_clean[892:1309,]

# Extract a Stratififed Validation Set from the Train Dataset
set.seed(42)
splitIndex <- createDataPartition(train_clean$Survived, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train_clean = train_val_clean[splitIndex,]
val_clean = train_val_clean[-splitIndex,]

# Split Train, Validation and Test Datasets into X and y
X_train = as.data.frame(select(train_clean, -Survived))
y_train = train_clean$Survived
X_val = as.data.frame(select(val_clean, -Survived))
y_val = val_clean$Survived
X_test = as.data.frame(select(test_clean, -Survived))
y_test = test_clean$Survived

# Train XGBoost Model
model <- xgboost(data = data.matrix(X_train), label = data.matrix(y_train), max.depth = 3, eta = 1, nthread = 2, nround = 5, objective = "binary:logistic")

# Predict on the Cross Validation Set
pred_xgboost_val <- predict(fit_xgboost, data.matrix(X_val))
y_val_pred <- as.numeric(pred_xgboost_val > 0.5)

# Calculate Performance Metrics (https://www.r-bloggers.com/computing-classification-evaluation-metrics-in-r/)
# Confustion Matrix
cm = as.matrix(table(Actual = y_val, Predicted = y_val_pred)) # create the confusion matrix
cm

n = sum(cm) # number of instances
nc = nrow(cm) # number of classes
diag = diag(cm) # number of correctly classified instances per class 
rowsums = apply(cm, 1, sum) # number of instances per class
colsums = apply(cm, 2, sum) # number of predictions per class
p = rowsums / n # distribution of instances over the actual classes
q = colsums / n # distribution of instances over the predicted classes

# Accuracy
accuracy = sum(diag) / n 
accuracy 

# F1
precision = diag / colsums 
recall = diag / rowsums 
f1 = 2 * precision * recall / (precision + recall) 
data.frame(precision, recall, f1)

# Compute feature importance matrix
importance_matrix <- xgb.importance(colnames(X_train), model = fit_xgboost)

# Plotting
xgb.plot.importance(importance_matrix)

# Prediction on Test DataSet
pred_xgboost_test <- predict(fit_xgboost, data.matrix(X_test))
y_test <- as.numeric(pred_xgboost_test > 0.5)

# Creating the submitting file
submit <- data.frame(PassengerId = X_test$PassengerId, Survived = y_test)
write.csv(submit, file = "../../data/submission_R.csv", row.names = FALSE)

