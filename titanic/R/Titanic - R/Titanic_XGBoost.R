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

# Fill NA in Numeric Features with median (Imputing) -- could be done with caret?
all_data_clean$Age[is.na(all_data_clean$Age)] <- median(all_data_clean$Age, na.rm = TRUE)
all_data_clean$Fare[is.na(all_data_clean$Fare)] <- median(all_data_clean$Fare, na.rm = TRUE)
all_data_clean$Parch[is.na(all_data_clean$Parch)] <- median(all_data_clean$Parch, na.rm = TRUE)
all_data_clean$Pclass[is.na(all_data_clean$Pclass)] <- median(all_data_clean$Pclass, na.rm = TRUE)
all_data_clean$Age[is.na(all_data_clean$SibSp)] <- median(all_data_clean$SibSp, na.rm = TRUE)

# Scale all Numeric Features
preProcValues <- preProcess(all_data_clean[, -2], method = c("scale"))
all_data_clean <- predict(preProcValues, all_data_clean)
all_data_clean$PassengerId = as.integer(rownames(all_data_clean))

# Mutate Categorical Features to Factors
all_data_clean <- mutate(all_data_clean, Name = factor(Name))
all_data_clean <- mutate(all_data_clean, Sex = factor(Sex))
all_data_clean <- mutate(all_data_clean, Cabin = factor(Cabin))
all_data_clean <- mutate(all_data_clean, Embarked = factor(Embarked))
all_data_clean <- mutate(all_data_clean, Ticket = factor(Ticket))

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
train_clean = all_data_clean[1:891,]
test_clean = all_data_clean[892:1309,]

# Split Train and Test Datasets into X and y
X_train = as.data.frame(select(train_clean, -Survived))
y_train = as.data.frame(select(train_clean, Survived))
X_test = as.data.frame(select(test_clean, -Survived))
y_test = as.data.frame(select(test_clean, Survived))

# ref: https://www.kaggle.com/ammara/titanic-competition-using-xgboost
# Using the cross validation to estimate our error rate:
param <- list("objective" = "binary:logistic")
cv.nround <- 15
cv.nfold <- 3

# Train using Cross-Validation
xgboost_cv = xgb.cv(param = param, data = data.matrix(X_train), label = data.matrix(y_train), nfold = cv.nfold, nrounds = cv.nround)

# Fitting with the xgboost model
nround  = 15
fit_xgboost <- xgboost(param = param, data = data.matrix(X_train), label = data.matrix(y_train), nrounds=nround)

# Get the feature real names
names <- dimnames(X_train)[[2]]

# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = fit_xgboost)

# Plotting
xgb.plot.importance(importance_matrix)

# Predict on the Train Set and find the Best Cut-off
pred_xgboost_train <- predict(fit_xgboost, data.matrix(X_train))
proportion <- sapply(seq(.3,.7,.01),function(step) c(step,sum(ifelse(pred_xgboost_train<step,0,1)!=X_train)))

# Prediction on Test DataSet
pred_xgboost_test <- predict(fit_xgboost, data.matrix(X_test))
y_test <- ifelse(pred_xgboost_test<proportion[,which.min(proportion[2,])][1],0,1)

# Creating the submitting file
submit <- data.frame(PassengerId = X_test$PassengerId, Survived = y_test)
write.csv(submit, file = "../../data/submission_R.csv", row.names = FALSE)

