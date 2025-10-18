```
Goal
    Model training
    Split the data into training and testing set
```

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import random
import warnings
warnings.filterwarnings('ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y  
)

print("Traing set shape:", X_train.shape)  
print("Testing set shape:", X_test.shape) 
print("Traing Target distribute:", y_train.value_counts()) 

# Evaluate some simple Models: Logistic, Linear, Tree
def evaluate_simple_estimators(X_train, y_train, X_test, y_test):

    # Define Models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
    }
    
    # evaluation using ROC AUC
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  #  model Training
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # prediction
        auc_score = roc_auc_score(y_test, y_pred_proba)  #compute ROC AUC
        results[name] = auc_score  
        print(f"{name}: AUC = {auc_score:.3f}")  
    
    return results


print("Compare the simple models' results:")
simple_results = evaluate_simple_estimators(X_train, y_train, X_test, y_test)  

# Evaluate some ensemble Models: Random Forest, XGBoost, GBoost, CatBoost
def evaluate_ensemble_methods(X_train, y_train, X_test, y_test):

    # Define Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss'),
        'CatBoost': CatBoostClassifier(random_state=42, verbose=False)
    }
    
    # evaluation using ROC AUC
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)  #  model Training
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # prediction
        auc_score = roc_auc_score(y_test, y_pred_proba)  #compute ROC AUC
        results[name] = auc_score  
        print(f"{name}: AUC = {auc_score:.3f}") 
    
    return results


print("Compare the ensemble models' results:")
ensemble_results = evaluate_ensemble_methods(X_train, y_train, X_test, y_test)  

```
Goal
    Tune the hyperparameters of selected method: CatBoost
    Analyse the final result
```
def tune_catboost(X_train, y_train, X_test, y_test):

    # Define parameters space
    param_dist = {
        'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'depth': [4, 6, 8, 10, 12],
        'l2_leaf_reg': [1, 3, 5, 7, 10],
        'iterations': [500, 1000, 1500, 2000],
        'border_count': [32, 64, 128, 254],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    }
    
    # create CatBoost classifier
    catboost = CatBoostClassifier(
        random_state=42,
        verbose=False,
        auto_class_weights='Balanced'  
    )
    
    # Tune
    random_search = RandomizedSearchCV(
        estimator=catboost,
        param_distributions=param_dist,
        n_iter=20,  
        cv=3,  # fole-3 CV
        scoring='roc_auc',  # evaluate the ROC AUC result
        random_state=42,
        n_jobs=-1  # use all CPU Cores
    )
    
    
    random_search.fit(X_train, y_train)  
    
    
    print("Optimal hyperparameters:", random_search.best_params_)  
    print("Optimal Training Socre:", random_search.best_score_)  
    
    # Use optimal model to train the data
    best_model = random_search.best_estimator_  
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  
    test_auc = roc_auc_score(y_test, y_pred_proba)  
    print("Testing AUC Socre:", test_auc)  
    
    return best_model, random_search.best_params_

# tune_catboost
print("tune_catboost:")
best_catboost, best_params = tune_catboost(X_train, y_train, X_test, y_test)  

# evaluate_final_model
def evaluate_final_model(model, X_train, y_train, X_test, y_test):

    # traing set prediction
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]  
    train_auc = roc_auc_score(y_train, y_train_pred_proba)  
    
    # Testing set prediction
    y_test_pred_proba = model.predict_proba(X_test)[:, 1]  
    test_auc = roc_auc_score(y_test, y_test_pred_proba)  
    
    print(f"Traing set AUC: {train_auc:.3f}")  
    print(f"Testing set AUC: {test_auc:.3f}") 
    
    # plot ROC AUC curve
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)  
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)  
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_auc:.3f})', color='blue') 
    plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_auc:.3f})', color='red')  
    plt.plot([0, 1], [0, 1], 'k--')  
    plt.xlabel('FP rate')  
    plt.ylabel('TP rate') 
    plt.title('ROC curve')  
    plt.legend()  
    plt.grid(True)  
    plt.show()
    
    # draw confusion matrix
    y_test_pred = (y_test_pred_proba > 0.5).astype(int)  
    cm = confusion_matrix(y_test, y_test_pred)  
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  
    plt.title('Confusion Matrix')  
    plt.xlabel('Prediction label') 
    plt.ylabel('Real label')  
    plt.show()
    

    print("\n classification report:")
    print(classification_report(y_test, y_test_pred))  
    
    return train_auc, test_auc, y_test_pred_proba


print("evaluate_final_model:")
train_auc, test_auc, y_pred_proba = evaluate_final_model(best_catboost, X_train, y_train, X_test, y_test)  

# 6. feature importance analysis
def plot_feature_importance(model, feature_names, top_n=40):

    # obtaion feature importance result
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_  # obtaion feature importance result
    else:
        print("Model doesn't provide feature importance result")
        return
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)  
    
    top_features = feature_importance_df.head(20)  
    
    # plot top 20 important features
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_features['importance'])  
    plt.yticks(range(len(top_features)), top_features['feature'])  
    plt.xlabel('feature importance score') 
    plt.title('Top 20 important features')  
    plt.gca().invert_yaxis()  
    plt.tight_layout()
    plt.show()
    

    
    return feature_importance_df

print("feature importance analysis:")
feature_importance_df = plot_feature_importance(best_catboost, X_train.columns)  

```
Goal
    After validating the model, it is evaluated using the test dataset
```

# use test dataset to predict default risk.
test_pred_proba = best_catboost.predict_proba(test_final)[:, 1]

submission = pd.DataFrame({"SK_ID_CURR":ID, "TARGET":test_pred_proba})


# save the prediction results as csv
submission.to_csv("submission-catboost.csv", index = False)
