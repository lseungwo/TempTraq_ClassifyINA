#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import xgboost as xgb
from xgboost import (
    XGBClassifier,
    plot_importance,
    plot_tree)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import seaborn as sns
import matplotlib.pyplot as plt

import itertools
import warnings


def assign_labels(df, category_col = 'Category'):
    """Assigns labels as 0 or 1 based on the category_col Values"""
    pass

def remove_pediatric_patients(df, age_threshold):
    """Remove patients who can be considered pediatric patients (decides based on given age_thresholds)"""
    df = df[df['Age']>=age_threshold]
    return df

def remove_soc_first_fevers(df, tt_fever_start_col, soc_fever_start_col):
    """Remove fevers that were detected earlier by SOC"""
    mask = df[tt_fever_start_col] > df[soc_fever_start_col]
    mask |= df['SF_ID'].isna()

    return df[mask]

def get_features_df(df, features, label_col):
    print('Total Number of Unique TempTraq Fevers', df['TF_ID'].nunique())
    print(df\
          .groupby(label_col)\
          ['TF_ID'].nunique())
    
    df_features = df[features]

    df_features = df_features\
        .rename({'GenderCode': 'Gender', 'RaceName': 'Race', 'TT Fever Start (DPI)_new': 'Fever_Start_from_Infusion(TT)', 'TTemp_Max_TT_new': 'Max_Temp_Within_2Hrs_Fever_Onset'}, axis =1)

    labels = df_features[label_col]

    df_features = df_features.drop(label_col, axis = 1)

    for col in df_features.select_dtypes(include='object').columns:
        df_features[col] = df_features[col].astype('category')
        
    tf_ids = df['TF_ID']
    return tf_ids, df_features, labels


def custom_objective_function(preds, dtrain, alpha):
    """
    Custom objective function that:
    - Gives high penalty for missing any true positives (false negatives).
    - Maximizes the prediction of Class 0, while balancing recall for Class 1.
    """
    labels = dtrain.get_label()
    probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities

    grad = np.zeros_like(probs)
    hess = probs * (1 - probs)  # Standard logistic Hessian

    # High penalty for false negatives (missing true positives)
    # Particularly for Class 1, but you can adjust this to balance Class 0

    grad[labels == 1] = alpha * (probs[labels == 1] - 1)  # Push towards predicting 1 for Class 1

    # Moderate penalty for Class 0 False Positives (FP), while allowing some flexibility
    beta = 1  # Weight for FP penalty for Class 0
    grad[labels == 0] = beta * probs[labels == 0]  # Push towards predicting 0 for Class 0

    return grad, hess




class XGBoostEvaluator:
    def __init__(
        self,
        n_splits=10,
        test_size=0.2,
        early_stopping_rounds=10,
        scale_features=True,
        n_threshold_points=10  # Number of threshold points to evaluate
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.scale_features = scale_features
        self.n_threshold_points = n_threshold_points
        
        # Base parameters that won't be tuned
        self.base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr"
        }
        
        # Parameter grid for tuning
        self.param_grid = {
            'scale_pos_weight': [1, 2, 3, 4, 5, 6], 
            'alpha' :[1, 2, 3, 4, 5, 6], 
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'learning_rate': [0.01, 0.05, 0.1]
        }
    
    def calculate_metrics_at_threshold(self, y_true, y_pred_proba, threshold):
        """Calculate precision, recall, and F1 score at a given threshold"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Count true positives, false positives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))
       
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp/ (tn+fp) if (tn + fp) > 0 else 0 
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return precision, recall, f1, fpr

     def evaluate_single_split(self, X, y, random_state):                                                                              |      def evaluate_single_split(self, X, y, random_state):
        """Evaluate model on a single train-test split using StratifiedShuffleSplit"""                                                |          """Evaluate model on a single train-test split using StratifiedShuffleSplit"""
        def strat_shuff_split(X, y, random_state):                                                                                    |          # Initial train-test split using StratifiedShuffleSplit                                                                      
            # Initial train-test split using StratifiedShuffleSplit                                                                   |          sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)                                
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)                             |          train_idx, test_idx = next(sss.split(X, y))                                                                                  
            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]               |          y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]                                                                        
                                                                                                                                        |          
              # Scale features if requested                                                                                             |          # Scale features if requested                                                                                                
              if self.scale_features:                                                                                                   |          if self.scale_features:                                                                                                      
                  scaler = StandardScaler()                                                                                             |              scaler = StandardScaler()                                                                                                
                  X_train = scaler.fit_transform(X_train)                                                                               |              X_train = scaler.fit_transform(X_train)                                                                                  
                  X_test = scaler.transform(X_test)       ef evaluate_single_split(self, X, y, random_state):

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Generate parameter combinations
        param_combinations = [
            dict(zip(self.param_grid.keys(), v)) 
            for v in itertools.product(*self.param_grid.values())
        ]

        # Cross-validation splitter for parameter tuning
        cv_splitter = StratifiedShuffleSplit(
            n_splits=5,
            test_size=0.2,
            random_state=random_state
        )

        # Store results for each parameter combination
        cv_results_all = []

        # Perform cross-validation for each parameter combination
        for params in param_combinations:
            # Combine base parameters with current parameter set
            current_params = {**self.base_params, **params}

            cv_results = xgb.cv(
                params=current_params,
                dtrain=dtrain,
                num_boost_round=1000,
                folds=list(cv_splitter.split(X_train, y_train)),
                early_stopping_rounds=self.early_stopping_rounds,
                obj=custom_objective_function,
                verbose_eval=False,
                metrics=['aucpr']  # Changed to AUCPR to match base params
            )

            # Store results
            cv_results_all.append({
                'params': current_params,
                'cv_results': cv_results,
                'best_score': cv_results['test-aucpr-mean'].max(),  # Changed to AUCPR
                'best_iteration': len(cv_results)
            })

        # Find best parameters
        best_cv_result = max(cv_results_all, key=lambda x: x['best_score'])
        best_params = best_cv_result['params']
        best_num_rounds = best_cv_result['best_iteration']

        # Train final model with best parameters
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            obj = custom_objective_function,
            num_boost_round=best_num_rounds,
            evals=[(dtest, "test")],
            verbose_eval=False
        )

        # Predictions
        y_pred_proba = model.predict(dtest)
        
        # After getting predictions, calculate metrics across thresholds
        thresholds = np.linspace(0, 1, self.n_threshold_points)
        threshold_metrics = []
        
        for threshold in thresholds:
            precision, recall, f1 = self.calculate_metrics_at_threshold(
                y_test, y_pred_proba, threshold
            )
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        threshold_metrics_df = pd.DataFrame(threshold_metrics)
        
        # Calculate AUPRC
        auprc = average_precision_score(y_test, y_pred_proba)
        
        # Get feature importance
        importance_dict = {
            name: score for name, score in 
            zip(X.columns, model.get_score(importance_type='gain').values())
        }
        
        # Get predictions at optimal threshold
        optimal_threshold = 0.5
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        return {
            'threshold_metrics': threshold_metrics_df,
            'auprc': auprc,
            'feature_importance': importance_dict,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'best_params': best_params,
            'best_cv_score': best_cv_result['best_score']
        }
    
    def evaluate_multiple_splits(self, X, y):
        """Evaluate model across multiple train-test splits"""
        all_threshold_metrics = []
        feature_importances = []
        best_params_list = []
        
        for i in range(self.n_splits):
            if i%1==0:
                print(f'{i+1} split')
            split_result = self.evaluate_single_split(X, y, random_state=i)
            all_threshold_metrics.append(split_result['threshold_metrics'])
            feature_importances.append(split_result['feature_importance'])
            best_params_list.append(split_result['best_params'])
        
        # Calculate mean and std of metrics across all splits for each threshold
        threshold_metrics_mean = pd.concat(all_threshold_metrics).groupby('threshold').mean()
        threshold_metrics_std = pd.concat(all_threshold_metrics).groupby('threshold').std()
        
        # Find optimal thresholds for different criteria
        optimal_thresholds = {
            'f1': threshold_metrics_mean['f1'].idxmax(),
            'balanced': threshold_metrics_mean.apply(
                lambda x: abs(x['precision'] - x['recall']), axis=1
            ).idxmin()
        }
        
        
        # Aggregate feature importances
        importance_df = pd.DataFrame(feature_importances)
        mean_importance = importance_df.mean()
        std_importance = importance_df.std()
        
        # Aggregate feature importances
        importance_df = pd.DataFrame(feature_importances)
        mean_importance = importance_df.mean()
        std_importance = importance_df.std()
        
        # Rest of the aggregation code remains the same
        return {
            'threshold_metrics': {
                'mean': threshold_metrics_mean,
                'std': threshold_metrics_std
            },
            'optimal_thresholds': optimal_thresholds,
            'feature_importance': pd.DataFrame({
                'mean_importance': mean_importance,
                'std_importance': std_importance
            }).sort_values('mean_importance', ascending=False)\
            # ,'parameter_summary': pd.DataFrame({
            #     'mean_value': mean_params,
            #     'std_value': std_params
            # })
        }

def plot_results(results, save_fig = True):
    """Plot evaluation results including performance curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot PR curve with confidence intervals
    metrics_mean = results['threshold_metrics']['mean']
    metrics_std = results['threshold_metrics']['std']
    
    ax1.plot(metrics_mean['recall'], metrics_mean['precision'], 'b-', label='Mean PR curve')
    ax1.fill_between(
        metrics_mean['recall'],
        metrics_mean['precision'] - metrics_std['precision'],
        metrics_mean['precision'] + metrics_std['precision'],
        alpha=0.2
    )
    ax1.set_title('Precision-Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    
    # Plot F1 curve with confidence intervals
    ax2.plot(metrics_mean.index, metrics_mean['f1'], 'g-', label='Mean F1')
    ax2.fill_between(
        metrics_mean.index,
        metrics_mean['f1'] - metrics_std['f1'],
        metrics_mean['f1'] + metrics_std['f1'],
        alpha=0.2
    )
    ax2.axvline(results['optimal_thresholds']['f1'], color='r', linestyle='--', 
                label=f"Optimal F1 threshold: {results['optimal_thresholds']['f1']:.3f}")
    ax2.set_title('F1 Score vs Threshold')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    
    # Plot top feature importance
    top_features = results['feature_importance'].head(10)
    sns.barplot(x=top_features['mean_importance'], y=top_features.index, ax=ax3)
    ax3.set_title('Top 10 Feature Importance')
    ax3.set_xlabel('Mean Importance')
    
    # Plot metrics at optimal thresholds
    optimal_metrics = pd.DataFrame([
        {
            'Metric': metric,
            'Value': metrics_mean.loc[threshold][['precision', 'recall', 'f1']].values[0]
        }
        for metric, threshold in results['optimal_thresholds'].items()
        for metric_type in ['Precision', 'Recall', 'F1']
    ])
    sns.barplot(data=optimal_metrics, x='Value', y='Metric', ax=ax4)
    ax4.set_title('Metrics at Optimal Thresholds')

    plt.tight_layout()
    if save_fig:
        fig.savefig(f'{n_splits}_Results.png')
    return fig


# def custom_objective_penalize_FN0(preds, dtrain):
#     """
#     Custom objective function that:
#     - Strongly penalizes missing true negatives (Class 0 classified as 1).
#     - Tries to capture as many positives (Class 1) as possible.
#     """
#     labels = dtrain.get_label()
#     probs = 1 / (1 + np.exp(-preds))  # Sigmoid to get probabilities
    
#     grad = np.zeros_like(probs)
#     hess = probs * (1 - probs)  # Standard logistic Hessian
    
#     # Strong penalty for false positives (Class 0 misclassified as 1)
#     alpha = 4  # Adjust weight for false positives (FN for Class 0)
#     grad[labels == 0] = alpha * probs[labels == 0]  # Push towards predicting 0
    
#     # Encourage capturing positives (Class 1)
#     beta = 2  # Weight for FN for Class 1 (still strong, but lower than alpha)
#     grad[labels == 1] = beta * (probs[labels == 1] - 1)  # Push towards predicting 1

#     return grad, hess


# def custom_objective_penalize_FP1(preds, dtrain):
#     """
#     Custom objective function to maximize recall for Class 1.
#     - Strongly penalizes false negatives for Class 1 (missed serious disease cases).
#     - Optionally, encourages precision for Class 0 by penalizing false positives.
#     """
#     labels = dtrain.get_label()
#     probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities
    
#     grad = np.zeros_like(probs)
#     hess = probs * (1 - probs)  # Standard logistic Hessian
    
#     # Strong penalty for false negatives (missed Class 1)
#     alpha = 3  # Weight for FN penalty
#     grad[labels == 1] = alpha * (probs[labels == 1] - 1)  # Push towards predicting 1
    
#     # Moderate penalty for false positives (optional, to help precision for Class 0)
#     beta = 1.5  # Weight for FP penalty
#     grad[labels == 0] = beta * probs[labels == 0]  # Push towards predicting 0

#     return grad, hess



# def custom_objective_class0_recall(preds, dtrain):
#     """
#     Custom objective function that forces recall for Class 0 to be 1
#     while maximizing recall for Class 1.
#     """
#     labels = dtrain.get_label()
#     probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities

#     grad = np.zeros_like(probs)
#     hess = probs * (1 - probs)  # Standard logistic Hessian

#     # Penalty for Class 0 False Positives (FP)
#     # High penalty for false positives for Class 0
#     alpha = 5  # Weight for FP penalty for Class 0
#     grad[labels == 0] = alpha * probs[labels == 0]  # Push towards predicting 0 for Class 0

#     # Penalty for Class 1 False Negatives (FN)
#     # Maximize recall for Class 1 by penalizing false negatives
#     beta = 1  # Weight for FN penalty for Class 1
#     grad[labels == 1] = beta * (probs[labels == 1] - 1)  # Push towards predicting 1 for Class 1

#     return grad, hess

# def custom_objective_class0_maximization(preds, dtrain):
#     """
#     Custom objective function that:
#     - Gives high penalty for missing any true positives (false negatives).
#     - Maximizes the prediction of Class 0, while balancing recall for Class 1.
#     """
#     labels = dtrain.get_label()
#     probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities

#     grad = np.zeros_like(probs)
#     hess = probs * (1 - probs)  # Standard logistic Hessian

#     # High penalty for false negatives (missing true positives)
#     # Particularly for Class 1, but you can adjust this to balance Class 0
#     alpha = 5  # Weight for FN penalty for Class 1
#     grad[labels == 1] = alpha * (probs[labels == 1] - 1)  # Push towards predicting 1 for Class 1

#     # Moderate penalty for Class 0 False Positives (FP), while allowing some flexibility
#     beta = 2  # Weight for FP penalty for Class 0
#     grad[labels == 0] = beta * probs[labels == 0]  # Push towards predicting 0 for Class 0

#     return grad, hess

# def custom_objective_precision1_maximization(preds, dtrain):
#     """
#     Custom objective function that:
#     - Ensures recall (sensitivity) for class 1 is 1 (no false negatives).
#     - Maximizes precision for class 1 by reducing false positives.
#     """
#     labels = dtrain.get_label()
#     probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities

#     grad = np.zeros_like(probs)
#     hess = probs * (1 - probs)  # Standard logistic Hessian

#     # Strong penalty for false negatives (FN) to ensure recall 1 (Sensitivity 1)
#     alpha = 10  # High weight to prevent missing any true positives
#     grad[labels == 1] = alpha * (probs[labels == 1] - 1)  # Push towards predicting 1

#     # Strong penalty for false positives (FP) to maximize precision for Class 1
#     beta = 5  # Weight to discourage FP
#     grad[labels == 0] = beta * probs[labels == 0]  # Push towards predicting 0 for negatives

#     return grad, hess


# # Find the best threshold based on F1-score
# def find_best_threshold(thresholds, f1_scores):
#     idx_recall_1 = np.where(recall == 1)[0]  # Find all indexes where recall is 1
    
#     if len(idx_recall_1) == 0:
#         print("No threshold ensures recall=1. Defaulting to lowest threshold.")
#         return thresholds[0]  # Default to the lowest threshold
    
#     best_idx = idx_recall_1[np.argmax(precision[idx_recall_1])]  # Maximize precision at recall 1
#     return thresholds[best_idx], idx_recall_1



# def plot_roc_curve(y_true, y_scores):
#     """
#     Plots the ROC curve and computes AUC.

#     Parameters:
#     y_true (list or array): True class labels (0 or 1).
#     y_scores (list or array): Predicted probabilities or decision scores.
#     """
#     fpr, tpr, thresholds = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)  # Compute Area Under Curve (AUC)

#     plt.figure(figsize=(8, 6))
#     plt.plot(fpr, tpr, color='b', label=f'ROC Curve (AUC = {roc_auc:.4f})')
#     plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
#     # plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random Guessing")
#     plt.xlabel("False Positive Rate (FPR)")
#     plt.ylabel("True Positive Rate (TPR)")
#     plt.title("Receiver Operating Characteristic (ROC) Curve")
#     plt.legend()
#     plt.grid()
#     plt.show()

#     print(f"ROC AUC: {roc_auc:.4f}")





# def train_xgboost(data_features, labels):
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(data_features, labels, test_size=0.2, random_state=42)

#     # Define parameters
#     params = {
#         "objective": "binary:logistic",  # Standard binary classification objective
#         "eval_metric": "logloss",
#         "learning_rate": 0.1,
#         "max_depth": 4
#     }

#     # Convert data to DMatrix format
#     dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
#     dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

#     # Perform cross-validation
#     cv_results = xgb.cv(
#         params=params,
#         dtrain=dtrain,
#         num_boost_round=100,
#         nfold=5,  # 5-fold cross-validation
#         obj=custom_objective_class0_recall,  # Use custom loss function
#         early_stopping_rounds=10,
#         verbose_eval=10,
#         as_pandas=True
#     )

#     # Train the final model using the best number of boosting rounds from CV
#     best_num_boost_round = cv_results.shape[0]
#     bst = xgb.train(
#         params=params,
#         dtrain=dtrain,
#         num_boost_round=best_num_boost_round,
#         obj=custom_objective_class0_recall,
#         evals=[(dtest, "test")],
#         verbose_eval=10
#     )

#     # Predict probabilities
#     y_proba = bst.predict(dtest)

#     # Compute precision-recall curve
#     _, _, thresholds = precision_recall_curve(y_test, y_proba)


#     # # Make sure thresholds align with recall and precision
#     # thresholds = np.append(thresholds, 1.0)  # Add max threshold

#     recall_0 = []  # Store Recall for Class 0
#     precision_1 = []
#     recall_1 = []

#     for t in thresholds:
#         y_pred = (y_proba >= t).astype(int)  # Convert probabilities to binary

#         TN = np.sum((y_pred == 0) & (y_test == 0))
#         FP = np.sum((y_pred == 1) & (y_test == 0))
        
#         FN = np.sum((y_pred ==0) & (y_test==1))
#         TP = np.sum((y_pred == 1) & (y_test == 1))
#         p_1 = TP / (TP + FP+ 1e-9)
#         r_0 = TN / (TN + FP+ 1e-9)
#         r_1 = TP / (FN + TP + 1e-9)
#         recall_0.append(r_0) # Avoid division by zero
#         precision_1.append(p_1)
#         recall_1.append(r_1)
#         print(f"TP:{TP}, FP:{FP}, TN:{TN}, recall_0: {r_0}, recall_1: {r_1}")
        
        
#         print(f"Threshold : {t}")
#         # Confusion Matrix
#         cm = confusion_matrix(y_test, y_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         disp.plot(cmap=plt.cm.Blues)
#         plt.title("Confusion Matrix")
#         plt.show()



# def find_optimal_threshold_recall1(y_true, y_pred_proba):
#     """Find threshold that maximizes precision while maintaining recall=1"""
#     precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
#     # Find all points where recall is 1
#     recall_1_indices = np.where(recall >= 0.999)[0]  # Using 0.999 instead of 1 for numerical stability
    
#     if len(recall_1_indices) == 0:
#         return 0, 0, 0  # No threshold achieves recall=1
    
#     # Among those points, find the one with highest precision
#     best_idx = recall_1_indices[np.argmax(precision[recall_1_indices])]
    
#     # Get corresponding threshold (handle edge case where threshold doesn't exist)
#     threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
#     return threshold, precision[best_idx], recall[best_idx]


if __name__=='__main__':
    data = pd.read_excel('../Model/0204_Training_Target_Abx.xlsx')
    features = ['Age',
       'TTemp_Max_TT_new', 'TT Fever Start (DPI)_new', 'label',
       'TTemp__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)',
       'TTemp__ar_coefficient__coeff_9__k_10',
       'TTemp__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
       'TTemp__fft_coefficient__attr_"real"__coeff_24',
       'TTemp__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"']
    
    tf_ids, data_features, labels = get_features_data(data, features)
    # Usage example
    n_splits = 100
    evaluator = XGBoostEvaluator(n_splits=3, test_size=0.2)
    results = evaluator.evaluate_multiple_splits(data_features, labels)

    # Plot results
    plot_results(results)
    plt.show()
