from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
import pandas as pd

# Feature selection
def feature_selection_GA(X_train, y_train, X_test): 
    estimator =	RandomForestClassifier(n_jobs=-1, 
                                        random_state=42,   
                                        class_weight="balanced")

    selector = GeneticSelectionCV(estimator,
                                  cv=StratifiedKFold(3, random_state=42),
                                  verbose=1,
                                  scoring="roc_auc",
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=20,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.2,
                                  tournament_size=5,
                                  caching=True,
                                  n_jobs=-1)
    
    selector = selector.fit(X_train, y_train)

    selected_features = pd.DataFrame({"feature": list(X_train), "selected": selector.support_})
    print()
    print(selected_features[selected_features["selected"] == True])
    
    return X_train.loc[:, selector.support_], X_test.loc[:, selector.support_]

def feature_selection(X_train, y_train, X_test, method="None"):
    if method == "None":
        return X_train, X_test
    elif method == "GA":
        return feature_selection_GA(X_train, y_train, X_test)