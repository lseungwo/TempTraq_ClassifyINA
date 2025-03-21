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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
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

from datetime import datetime
import os

from model_logging import *

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


def custom_objective_function(preds, dtrain, alpha = 1, beta = 5):
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
    grad[labels == 0] = beta * probs[labels == 0]  # Push towards predicting 0 for Class 0

    return grad, hess




class XGBoostEvaluator:
    def __init__(
        self,
        n_splits=10,
        test_size=0.2,
        early_stopping_rounds=10,
        scale_features=True,
        n_threshold_points=10,  # Number of threshold points to evaluate
        gridSearch = True
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.scale_features = scale_features
        self.n_threshold_points = n_threshold_points
        self.gridSearch = gridSearch
        
        # Base parameters that won't be tuned
        self.base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": 2,
            'alpha': 1,
            'beta': 3
        }
        
        self.safe_params = {
            "max_depth": 4,
            'min_child_weight': 3,
            'gamma': 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "learning_rate": 0.1
        }
        
        # Parameter grid for tuning
        self.param_grid = {
            'max_depth': [3, 4, 5, 6],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'learning_rate': [0.01, 0.05, 0.1],
            # "scale_pos_weight": [1 ,2, 3, 4, 5, 6]
            # 'alpha': [1, 5, 6, 7, 10]
        }

        set_alpha = self.base_params.get('alpha', '')
        set_spw = self.base_params.get('scale_pos_weight', '')
        global training_file
        set_experiment(f'{label_name}_{training_file}_{set_alpha}_{set_spw}')
    
    def calculate_metrics_at_threshold(self, y_true, y_pred_proba, threshold):
        """Calculate precision, recall, and F1 score at a given threshold"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Count true positives, false positives, false negatives
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp/ (fp+tn) if (fp+tn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        aucpr = average_precision_score(y_true, y_pred_proba)
        
        return precision, recall, specificity, fpr, f1, aucpr

    def evaluate_single_split(self, X, y, random_state):
        """Evaluate model on a single train-test split using StratifiedShuffleSplit"""
        def stratified_shuffle_split(X, y, random_state=random_state):
            # Initial train-test split using StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)
            train_idx, test_idx = next(sss.split(X, y))

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            return X_train, X_test, y_train, y_test
        
        def scale_features(X_train, X_test):
            # Scale features if requested
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            return X_train, X_test
        
        def convert_to_dmatrix(X_train, y_train, X_test, y_test):
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            return dtrain, dtest


        def cross_validation_for_best_parameter(X_train, y_train, dtrain, param_combinations, test_size = 0.2, random_state = random_state, gridSearch = True):
            def cv(current_params):
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

            # Cross-validation splitter for parameter tuning
            cv_splitter = StratifiedShuffleSplit(
                n_splits=5,
                test_size=test_size,
                random_state=random_state
            )

            # Store results for each parameter combination
            cv_results_all = []

            # Perform cross-validation for each parameter combination
            if gridSearch:
                for params in param_combinations:
                    # Combine base parameters with current parameter set
                    current_params = {**self.base_params, **params}
                    cv(current_params)
            
            else:
                current_params = {**self.base_params, **self.safe_params}
                cv(current_params)

            return cv_results_all
        

        def train_model(dtrain, dtest, y_test, best_params, best_num_rounds):
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
                precision, recall, specificity, fpr, f1, aucpr = self.calculate_metrics_at_threshold(
                    y_test, y_pred_proba, threshold
                )
                threshold_metrics.append({
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'fpr': fpr,
                    'f1': f1,
                    'aucpr': aucpr
                })
            
            threshold_metrics_df = pd.DataFrame(threshold_metrics)
            return model, y_pred_proba, threshold_metrics_df
        

        X_train, X_test, y_train, y_test = stratified_shuffle_split(X, y, random_state)

        if self.scale_features:
            X_train, X_test = scale_features(X_train, X_test)

        dtrain, dtest = convert_to_dmatrix(X_train, y_train, X_test, y_test)



         # Generate parameter combinations
        param_combinations = [
            dict(zip(self.param_grid.keys(), v)) 
            for v in itertools.product(*self.param_grid.values())
        ]

        cv_results_all = cross_validation_for_best_parameter(X_train, y_train, dtrain, param_combinations, test_size = 0.2, random_state = random_state, gridSearch = self.gridSearch)
        
        # Find best parameters
        best_cv_result = max(cv_results_all, key=lambda x: x['best_score'])
        best_params = best_cv_result['params']
        best_num_rounds = best_cv_result['best_iteration']

        model, y_pred_proba, threshold_metrics_df = train_model(dtrain, dtest, y_test, best_params, best_num_rounds)

        best_params['num_boost_rounds'] = best_cv_result['best_iteration']
        

        # Get feature importance
        importance_dict = {
            name: score for name, score in 
            zip(X.columns, model.get_score(importance_type='gain').values())
        }

        # Get predictions at optimal threshold
        optimal_threshold = 0.5
        precision, recall, specificity, fpr, f1, aucpr = self.calculate_metrics_at_threshold(y_test, y_pred_proba, optimal_threshold) 
        optimal_threshold_performance = {
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'fpr': fpr,
                    'f1': f1,
                    'aucpr': aucpr
                }
        
       
        
        log_model(model, best_params, aucpr, dtrain, random_state)
        
        return {
            'optimal_thresholds_performance': optimal_threshold_performance,
            'threshold_metrics': threshold_metrics_df,
            'feature_importance': importance_dict,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'best_params': best_params,
            'cv_results_all': cv_results_all
        }
    
    def evaluate_multiple_splits(self, X, y):
        """Evaluate model across multiple train-test splits"""
        model_performance = []
        all_threshold_metrics = []
        feature_importances = []
        pred_results = []
        best_params = []
        cv_results = []
        
        for i in range(self.n_splits):
            if i%1==0:
                print(f'{i+1} split')
            split_result = self.evaluate_single_split(X, y, random_state=i)
            model_performance.append(split_result['optimal_thresholds_performance'])
            all_threshold_metrics.append(split_result['threshold_metrics'])
            feature_importances.append(split_result['feature_importance'])
            pred_results.append({'true':split_result['true_labels'], 'predicted':split_result['probabilities']})
            best_params.append(split_result['best_params'])
            cv_results.append(pd.DataFrame(split_result['cv_results_all']))
        
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
        
        model_performance = pd.DataFrame(model_performance)
        cv_results = pd.concat(cv_results)
        
        # Rest of the aggregation code remains the same
        return {
            'threshold_all' : pd.concat(all_threshold_metrics),
            'threshold_metrics': {
                'mean': threshold_metrics_mean,
                'std': threshold_metrics_std
            },
            'optimal_thresholds': optimal_thresholds,
            'feature_importance': pd.DataFrame({
                'mean_importance': mean_importance,
                'std_importance': std_importance
            }).sort_values('mean_importance', ascending=False),\
            'model_performance_at0.5': model_performance,
            'cv_results_agg' : cv_results,
            'pred_results': pd.DataFrame(pred_results)
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
        fig.savefig(f'{directory_path}/{n_splits}_Results.png')
    return fig


if __name__=='__main__':
    training_file = 'Training0321_1to4_filter_abx+engr.xlsx'
    df = pd.read_excel(training_file)
    label_name = 'label_engr'
    turn_warnings = False
    if not turn_warnings:
        warnings.filterwarnings("ignore")
    features = ['Age', 
       'TTemp_Max_TT_new', 'TT Fever Start (DPI)_new','TTemp_Interp__ar_coefficient__coeff_0__k_10',
       'TTemp_Interp__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
       'TTemp_Interp__fft_coefficient__attr_"abs"__coeff_1',
       'TTemp_Interp__fft_coefficient__attr_"angle"__coeff_30',
       'TTemp_Interp__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
       'TTemp_Interp__energy_ratio_by_chunks__num_segments_10__segment_focus_9', label_name]
    
    tf_ids, df_features, labels = get_features_df(df, features, label_name)
    # Usage example
    n_splits = 1000
    print(f'This script will build model {n_splits} times')
    evaluator = XGBoostEvaluator(n_splits= n_splits,  test_size=0.2,  scale_features = False, n_threshold_points=1000, gridSearch = True)
    print('Building and evaluating the model..')
    results = evaluator.evaluate_multiple_splits(df_features, labels)
       
    current_time =  datetime.now().replace(microsecond=0)

    # Specify the directory path you want to create
    directory_path = f"results/{current_time}"

    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    results['model_performance_at0.5'].to_csv(directory_path+'/performance_per_model.csv', index_label = 'Model#')
    results['cv_results_agg'].to_csv(directory_path +'/cv_results_agg.csv', index = False)
    results['pred_results'].to_csv(directory_path+'/pred_results.csv', index = False)
    results['threshold_all'].to_csv(directory_path+'/threshold_metrics.csv', index = False)
    results['feature_importance'].to_csv(directory_path+'/feature_importance.csv', index = False)



    print("\nModel Performance Summary:")
    print("-------------------------")
    print(f"mean_aucpr:{np.nanmean(results['model_performance_at0.5']['aucpr'])}")

    # Plot results
    plot_results(results)
    plt.show()