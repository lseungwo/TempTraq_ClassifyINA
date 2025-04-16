#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re

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

from joblib import Parallel, delayed
import time

from model_logging import *

import warnings

import yaml
import sqlite3

def get_config(file = 'config.yaml'):
    # config file
    with open(file, "r") as f:
        config = yaml.safe_load(f)
    print("Loaded Configuration:")
    for section, settings in config.items():
        print(f"\n[{section}]")
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"{key}: {value}")
        else:
            print(settings)

    return config


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

def get_features_df(df, features, label_col, sample_scale_weight = None):
    print('Total Number of Unique TempTraq Fevers', df['TF_ID'].nunique())
    print(df\
          .groupby(label_col)\
          ['TF_ID'].nunique())
    
    df_features = df[features]

    df_features = df_features\
        .rename({'GenderCode': 'Gender', 'RaceName': 'Race', 'TT Fever Start (DPI)_new': 'Fever_Start_from_Infusion(TT)', 'TTemp_Max_TT_new': 'Max_Temp_Within_2Hrs_Fever_Onset'}, axis =1)

    labels = df[label_col]

    for col in df_features.select_dtypes(include='object').columns:
        df_features[col] = df_features[col].astype('category')
        
    tf_ids = df['TF_ID']

    if sample_scale_weight:
        sample_scale_weight = df[sample_scale_weight]
    return tf_ids, df_features, labels, sample_scale_weight


class XGBoostEvaluator:
    def __init__(
        self,
        num_splits=10,
        test_size=0.5,
        early_stopping_rounds=10,
        scale_features = False,
        n_threshold_points=10,  # Number of threshold points to evaluate
        gridSearch = True,
        label_name = 'label_engr',
        scale_pos_weight = 1,
        fn_penalize = 1,
        fp_penalize = 1
    ):
        self.num_splits = num_splits
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds
        self.scale_features = scale_features
        self.n_threshold_points = n_threshold_points
        self.gridSearch = gridSearch
        self.label_name = re.findall(r'(abx2|engr)', label_name)[0].upper()
        self.fn_penalize = fn_penalize
        self.fp_penalize = fp_penalize

        # Base parameters that won't be tuned
        self.base_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "tree_method": "hist",  # Enables histogram-based tree growth for efficiency
            "n_jobs": n_cpus,  # Uses all available CPU cores
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
            'learning_rate': [0.01, 0.05, 0.1]
        }

    

        if self.label_name =='ENGR':
            set_experiment(f'{self.label_name}_fpp:{self.fp_penalize}_spw:{self.base_params["scale_pos_weight"]}')
        elif self.label_name == 'ABX2':
            set_experiment(f'{self.label_name}_fnp:{self.fn_penalize}_spw:{self.base_params["scale_pos_weight"]}')


    def custom_objective_function(self, preds, dtrain):
        """
        Custom objective function that:
        - Gives high penalty for missing any infection that needs antibiotics(can be false positive or false negative depending on the label assignement).
        """

        # https://machinelearningmastery.com/cost-sensitive-learning-for-imbalanced-classification/
        labels = dtrain.get_label()
        probs = 1 / (1 + np.exp(-preds))  # Sigmoid to convert raw scores to probabilities

        grad = np.zeros_like(probs)
        hess = probs * (1 - probs)  # Standard logistic Hessian

        # High penalty for false negatives (missing true positives)
        # Particularly for Class 1, but you can adjust this to balance Class 0
        grad[labels == 1] = self.fn_penalize * (probs[labels == 1] - 1)  # Push towards predicting 1 for Class 1

        # Moderate penalty for Class 0 False Positives (FP), while allowing some flexibility
        grad[labels == 0] = self.fp_penalize * probs[labels == 0]  # Push towards predicting 0 for Class 0

        return grad, hess
    
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

    def evaluate_single_split(self, X, y, random_state, sample_scale_weight):
        """Evaluate model on a single train-test split using StratifiedShuffleSplit"""
        def stratified_shuffle_split(X, y, random_state=random_state, sample_scale_weight=sample_scale_weight):
            # Initial train-test split using StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)
            train_idx, test_idx = next(sss.split(X, y))

            X_train, X_test, sample_scale_weight = X.iloc[train_idx], X.iloc[test_idx], sample_scale_weight.iloc[train_idx]
            y_train, y_test, = y.iloc[train_idx], y.iloc[test_idx]

            return X_train, X_test, y_train, y_test, sample_scale_weight
        
        def scale_features(X_train, X_test):
            # Scale features if requested
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            return X_train, X_test
        
        def convert_to_dmatrix(X_train, y_train, X_test, y_test, sample_scale_weight):
            # Convert to DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train, weight = sample_scale_weight, enable_categorical = True)
            dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical= True)

            return dtrain, dtest


        def cross_validation_for_best_parameter(X_train, y_train, dtrain, param_combinations, test_size = 0.2, random_state = random_state, gridSearch = True):
            
            # Cross-validation splitter for parameter tuning
            cv_splitter = StratifiedShuffleSplit(
                n_splits=5,
                test_size=test_size,
                random_state=random_state
            )

            def cv(current_params):
                print(f"[PID {os.getpid()}] Starting CV with params: {current_params}")
                cv_results = xgb.cv(
                        params=current_params,
                        dtrain=dtrain,
                        num_boost_round=1000,
                        folds=list(cv_splitter.split(X_train, y_train)),
                        early_stopping_rounds=self.early_stopping_rounds,
                        obj=self.custom_objective_function,
                        verbose_eval=False,
                        metrics=['aucpr']  # Changed to AUCPR to match base params
                    )

                    # Store results
                return({
                    'params': current_params,
                    'cv_results': cv_results,
                    'best_score': cv_results['test-aucpr-mean'].max(),  # Changed to AUCPR
                    'best_iteration': len(cv_results)
                })

           

            # Perform cross-validation for each parameter combination
            if gridSearch:
                # print(len([({**self.base_params, **params}) for params in param_combinations]))
                cv_results_all = Parallel(n_jobs=n_cpus)(delayed(cv)({**self.base_params, **params}) for params in param_combinations)
                store_all_cv_results(cv_results_all)
                best_cv_result = max(cv_results_all, key=lambda x: x['best_score'])

            else:
                current_params = {**self.base_params, **self.safe_params}
                cv_results_all = cv(current_params)
                store_all_cv_results(cv_results_all)
                
                best_cv_result = cv_results_all

                if gridSearch == 'bayes':
                    pass
        
            best_params = best_cv_result['params']
            best_num_rounds = best_cv_result['best_iteration']

            return best_params, best_num_rounds
        

        def train_model(dtrain, dtest, y_test, best_params, best_num_rounds):
            # Train final model with best parameters
            model = xgb.train(
                params=best_params,
                dtrain=dtrain,
                obj = self.custom_objective_function,
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
                    'threshold':threshold,
                    'precision': precision,
                    'recall': recall,
                    'specificity': specificity,
                    'fpr': fpr,
                    'f1': f1,
                    'aucpr': aucpr
                })
            
            threshold_metrics_df = pd.DataFrame(threshold_metrics)
            return model, y_pred_proba, threshold_metrics_df
        

        X_train, X_test, y_train, y_test, sample_scale_weight = stratified_shuffle_split(X, y, random_state, sample_scale_weight)

        if self.scale_features:
            X_train, X_test = scale_features(X_train, X_test)

        dtrain, dtest = convert_to_dmatrix(X_train, y_train, X_test, y_test, sample_scale_weight)

         # Generate parameter combinations
        param_combinations = [
            dict(zip(self.param_grid.keys(), v)) 
            for v in itertools.product(*self.param_grid.values())
        ]

        best_params, best_num_rounds = cross_validation_for_best_parameter(X_train, y_train, dtrain, param_combinations, test_size = 0.2, random_state = random_state, gridSearch = self.gridSearch)
        # Find best parameters
        
        model, y_pred_proba, threshold_metrics_df = train_model(dtrain, dtest, y_test, best_params, best_num_rounds)

        best_params['num_boost_rounds'] = best_num_rounds
        

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
        
       
        
        log_model(model, best_params, aucpr, dtrain, random_state, self.label_name)
        
        return {
            'optimal_thresholds_performance': optimal_threshold_performance,
            'threshold_metrics': threshold_metrics_df,
            'feature_importance': importance_dict,
            'probabilities': y_pred_proba,
            'true_labels': y_test,
            'best_params': best_params,
        }
    
    def evaluate_multiple_splits(self, X, y, sample_scale_weight):
        """Evaluate model across multiple train-test splits"""
        model_performance = []
        all_threshold_metrics = []
        feature_importances = []
        pred_results = []
        best_params = []
        for i in range(self.num_splits):
            if i%1==0:
                print(f'{i+1} split')
            split_result = self.evaluate_single_split(X, y, random_state=i, sample_scale_weight=sample_scale_weight)
            model_performance.append(split_result['optimal_thresholds_performance'])
            all_threshold_metrics.append(split_result['threshold_metrics'])
            feature_importances.append(split_result['feature_importance'])
            pred_results.append({'true':split_result['true_labels'], 'predicted':split_result['probabilities']})
            best_params.append(split_result['best_params'])
         
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
        fig.savefig(f'{directory_path}/{num_splits}_Results.png')
    return fig


if __name__=='__main__':
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
    config = get_config()
    print(f'CPU: {n_cpus}')

    # datasets
    training_file = config['datasets']

    # preprocessing
    scale_features = config['preprocessing']['scale_features']

    # experiment
    num_splits = config['experiment']['num_splits']
    test_size = config['experiment']['test_size']
    n_threshold_points = config['experiment']['n_threshold_points']
    gridSearch = config['experiment']['gridSearch']
    turn_warnings = config['experiment']['turn_warnings']

    # training specs
    scale_pos_weight = config['training_specs']['scale_pos_weight']
    fp_penalize = config['training_specs']['fp_penalize']
    fn_penalize = config['training_specs']['fn_penalize']
    sample_scale_weight = config['training_specs']['sample_scale_weight']
    early_stopping_rounds = config['training_specs']['early_stopping_rounds']
    

    features = config['features']
    label_name = config['labels']
    
    if not turn_warnings:
        warnings.filterwarnings("ignore")
    
    df = pd.read_excel(training_file)

    tf_ids, df_features, labels, sample_scale_weight = get_features_df(df, features, label_name, sample_scale_weight)

    # Usage example

    print(f'This script will build model {num_splits} times')
    evaluator = XGBoostEvaluator(num_splits= num_splits,
                                 test_size = test_size,
                                 early_stopping_rounds=early_stopping_rounds,
                                scale_features = scale_features,
                                n_threshold_points = n_threshold_points,
                                      gridSearch = gridSearch,
                                          label_name = label_name,
                                          scale_pos_weight = scale_pos_weight,
                                          fp_penalize=fp_penalize,
                                            fn_penalize = fn_penalize)
    
    print('Building and evaluating the model..')
    results = evaluator.evaluate_multiple_splits(df_features, labels, sample_scale_weight)
       
    current_time =  str(datetime.now().replace(microsecond=0)).replace(' ', '_')

    # Specify the directory path you want to create
    directory_path = f"results/{current_time}"

    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    results['model_performance_at0.5'].to_csv(directory_path+'/performance_per_model.csv', index_label = 'Model#')
    results['pred_results'].to_csv(directory_path+'/pred_results.csv', index = False)
    results['threshold_all'].to_csv(directory_path+'/threshold_metrics.csv', index = False)
    results['feature_importance'].to_csv(directory_path+'/feature_importance.csv', index = False)



    print("\nModel Performance Summary:")
    print("-------------------------")
    print(f"mean_aucpr:{np.nanmean(results['model_performance_at0.5']['aucpr'])}")

    # Plot results
    plot_results(results)
    plt.show()