from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pandas as pd

#Bayesian optimization
def bayesian_optimization(X_train, y_train, X_test, y_test, function, parameters):
    n_iterations = 50
    gp_params = {"alpha": 1e-4}
    
    BO = BayesianOptimization(function, parameters)
    BO.maximize(n_iter=n_iterations, **gp_params)
    best = BO.max
    print()
    print("Best parameters:")
    print(pd.DataFrame(best["params"], index=["value"]).T)
    
    return best     

def rfc_optimization(cv_splits, X_train, y_train):
    def function(n_estimators, max_depth, min_samples_split):
            return cross_val_score(RandomForestClassifier(n_estimators=int(max(n_estimators,0)), 
                                                           max_depth=int(max(max_depth,1)), 
                                                           min_samples_split=int(max(min_samples_split,2)), 
                                                           min_samples_leaf=1, 
                                                           min_weight_fraction_leaf=0.0, 
                                                           min_impurity_decrease=0.0, 
                                                           min_impurity_split=None, 
                                                           n_jobs=1, 
                                                           random_state=42,   
                                                           class_weight="balanced"),  
                                   X=X_train, 
                                   y=y_train, 
                                   cv=cv_splits,
                                   scoring="roc_auc",
                                   n_jobs=1).mean()

    parameters = {"n_estimators": (10, 1000),
                "max_depth": (1, 150),
                "min_samples_split": (2, 10)}
    
    return function, parameters

def xgb_optimization(cv_splits, eval_set, X_train, y_train):
    def function(eta, gamma, max_depth):
            return cross_val_score(xgb.XGBClassifier(objective="binary:logistic",
                                                    learning_rate=max(eta,0),
                                                    gamma=max(gamma,0),
                                                    max_depth=int(max_depth),                                               
                                                    min_child_weight=0,                                        
													subsample=0.35,                                                  colsample_bytree=0.6,                                       n_estimators=300,
                                                    seed=42,
                                                    nthread=1,
                                                    silent=True,
                                                    scale_pos_weight = len(y_train[y_train == 0])/len(y_train[y_train == 1])),  
                                   X=X_train, 
                                   y=y_train, 
                                   cv=cv_splits,
                                   scoring="roc_auc",
                                   fit_params={"early_stopping_rounds": 10, "eval_metric": "auc", "eval_set": eval_set},
                                   n_jobs=1).mean()

    parameters = {"eta": (0.001, 0.4),
                    "gamma": (0, 20),
                    "max_depth": (1, 2000)
                    }
    
    return function, parameters
	
#Train model
def train(X_train, y_train, X_test, y_test, model_name="xgb"):
    eval_set = [(X_train, y_train), (X_test, y_test)]
    cv_splits = 4
    
    if model_name == "rfc":
        function, parameters = rfc_optimization(cv_splits, X_train, y_train)
        tree_best = bayesian_optimization(X_train, y_train, X_test, y_test, function, parameters) 
        
        best_model = RandomForestClassifier(n_estimators=int(max(tree_best["params"]["n_estimators"],0)), 
                               max_depth=int(max(tree_best["params"]["max_depth"],1)), 
                               min_samples_split=int(max(tree_best["params"]["min_samples_split"],2)), 
                               min_samples_leaf=1, 
                               min_weight_fraction_leaf=0.0, 
                               min_impurity_decrease=0.0, 
                               min_impurity_split=None, 
                               n_jobs=4, 
                               random_state=42,   
                               class_weight="balanced")
        best_model.fit(X_train, y_train)
    elif model_name == "xgb":  
        function, parameters = xgb_optimization(cv_splits, eval_set, X_train, y_train)
        tree_best = bayesian_optimization(X_train, y_train, X_test, y_test, function, parameters) 
        
        best_model = xgb.XGBClassifier(objective="binary:logistic",
                                learning_rate=max(tree_best["params"]["eta"],0),
                                gamma=max(tree_best["params"]["gamma"],0),
                                max_depth=int(tree_best["params"]["max_depth"]),       
                                min_child_weight=0,
                                subsample=0.35, 
                                colsample_bytree=0.6, 
                                n_estimators=300, 

                                seed=42,
                                nthread=4,
                                silent=True,
                                scale_pos_weight = len(y_train[y_train == 0])/len(y_train[y_train == 1]))    
        best_model.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=True, early_stopping_rounds=10)
    
    return best_model