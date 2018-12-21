	--Create ModelHistory table
	IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ModelHistory' AND xtype='U') 
	CREATE TABLE [AdventureWorksDW2017].[ML].[ModelHistory](
        [time] [varchar](MAX) NULL,
        [model_name] [varchar](10) NULL, 
        [model_path] [varchar](MAX) NULL,
		[data_path] [varchar](MAX) NULL,
		[parameters_path] [varchar](MAX) NULL, 
		[data_length] [int] NULL, 
		[AUC_test] [float] NULL, 
		[GINI_test] [float] NULL, 
		[f1_max_test] [float] NULL,
		[recall_test] [float] NULL,
		[precision_test] [float] NULL,
		[accuracy_test] [float] NULL,
		[thr_test] [float] NULL) 

	--Query to get predict data
	DECLARE @input_query NVARCHAR(MAX) = N'SELECT TOP (1000)
       [MaritalStatus]
      ,[Gender]
      ,CAST([YearlyIncome] AS float) as YearlyIncome
      ,CAST([TotalChildren] AS float) as TotalChildren
      ,CAST([NumberChildrenAtHome] AS float) as NumberChildrenAtHome
      ,[EnglishEducation]
      ,[EnglishOccupation]
      ,[HouseOwnerFlag]
      ,CAST([NumberCarsOwned] as int) as NumberCarsOwned
      ,[CommuteDistance]
      ,CAST([Age] as float) as Age
      ,CAST([BikeBuyer] as int) as BikeBuyer
	  FROM [AdventureWorksDW2017].[dbo].[vTargetMail]'

	--Path where to save standard output and errors
	DECLARE
		@save_path varchar(max)
		SET @save_path = 'C:\Users\Public\Documents\AdventureWorks\'

	DECLARE
		@run_feature_selection int
		SET @run_feature_selection = 0

	--Python modeling script
	INSERT INTO [AdventureWorksDW2017].[ML].[ModelHistory]
	EXEC sp_execute_external_script
		 @language = N'Python'
		,@script = N'

#Import libraries
import pandas as pd
import pickle
import sys
import os
from sklearn.model_selection import train_test_split
from feature_selection import feature_selection
from training import train, bayesian_optimization, rfc_optimization, xgb_optimization
from testing import predict

#Create folder for saving results
save_path = os.path.join(save_path, "models")
if not os.path.exists(save_path):
    os.makedirs(save_path)

now = str(pd.datetime.now().strftime("%Y%m%d-%H%M%S"))
save_path = os.path.join(save_path, now)
os.makedirs(save_path)

#Save standard output and path to file
sys.stdout = open(os.path.join(save_path, "modeling_out.log"), "w")
sys.stderr = open(os.path.join(save_path, "modeling_err.log"), "w")

# Get data from input query
data = input_data
data = pd.get_dummies(data, columns = ["MaritalStatus", "Gender", "EnglishEducation", "EnglishOccupation", "CommuteDistance"])
print(data.head())

#Split dataset to train and test
def split_train_test(df):  
    X = df.drop(["BikeBuyer"], axis=1)
    y = df["BikeBuyer"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, stratify=y, random_state=1)
    return X, y, X_train, X_test, y_train, y_test


# Prepare data for training
X, y, X_train_preselection, X_test_preselection, y_train, y_test = split_train_test(data)
print("Ksztalt zbioru treningowego", X.shape)
if run_feature_selection:
    X_train, X_test = feature_selection(X_train_preselection, y_train, X_test_preselection, method="GA")
else:
    X_train, X_test = feature_selection(X_train_preselection, y_train, X_test_preselection, method="None")


# Choose the best model on test dataset
best_AUC_test = 0
for model_name in ["rfc"]:
    model = train(X_train, y_train, X_test, y_test, model_name=model_name) 

    (prediction_test, 
	AUC_test, 
	GINI_test, 
	f1_max_test, 
	recall_test, 
	precision_test, 
	accuracy_test, 
	thr_test) = predict(model, X_test, y_test, model_name=model_name)

    print(model_name, ": AUC=", AUC_test)	
    if AUC_test > best_AUC_test:
        best_AUC_test = AUC_test
        chosen_model = model
        chosen_model_name = model_name

model = chosen_model
model_name = chosen_model_name 
(prediction_test, 
AUC_test, 
GINI_test, 
f1_max_test, 
recall_test, 
precision_test, 
accuracy_test, 
thr_test) = predict(model, X_test, y_test, model_name=model_name) 

print()
print("Feature importances:")
feature_importances = pd.DataFrame({"feature": list(X_train), "importance": model.feature_importances_})
print(feature_importances.sort_values("importance", ascending=False).to_string()) 


# Save features, model and parameters
data_path = os.path.join(save_path, "train_data.csv")
X.head(0).to_csv(data_path, index=False)

model_path = os.path.join(save_path, "model")
pickle.dump(model, open(model_path, "wb"))

parameters_path = os.path.join(save_path, "parameters")
parameters = {"model_name": model_name, "features": list(X_train), "threshold": thr_test}
pickle.dump(parameters, open(parameters_path, "wb"))


# Generate output dataset for SQL procedure
OutputDataSet = pd.DataFrame({"time": [now],
                              "model_name": [model_name], 
                              "model_path": [model_path], 
							  "data_path": [data_path], 
							  "parameters_path": [parameters_path], 
							  "data_length": [len(X)],
							  "AUC_test": [AUC_test], 
							  "GINI_test": [GINI_test], 
							  "f1_max_test": [f1_max_test], 
							  "recall_test": [recall_test], 
							  "precision_test": [precision_test], 
							  "accuracy_test": [accuracy_test], 
							  "thr_test": [thr_test]})

OutputDataSet = OutputDataSet[["time", 
                               "model_name", 
							   "model_path", 
							   "data_path", 
							   "parameters_path", 
							   "data_length", 
							   "AUC_test", 
							   "GINI_test", 
							   "f1_max_test", 
							   "recall_test", 
							   "precision_test", 
							   "accuracy_test", 
							   "thr_test"]]
'
		--input
		,@input_data_1 = @input_query 
		,@input_data_1_name = N'input_data'
		--parameters
		,@params = N'@save_path varchar(max), @run_feature_selection int' 
		,@save_path = @save_path
		,@run_feature_selection = @run_feature_selection