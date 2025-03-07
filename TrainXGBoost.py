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

def get_features_df(df, features):
    print('Total Number of Unique TempTraq Fevers', df['TF_ID'].nunique())
    print(df\
          .groupby('label')\
          ['TF_ID'].nunique())
    
    df_features = df[features]

    df_features = df_features\
        .rename({'GenderCode': 'Gender', 'RaceName': 'Race', 'TT Fever Start (DPI)_new': 'Fever_Start_from_Infusion(TT)', 'TTemp_Max_TT_new': 'Max_Temp_Within_2Hrs_Fever_Onset'}, axis =1)

    labels = df_features['label']

    df_features = df_features.drop('label', axis = 1)

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
    alpha = 6  # Weight for FN penalty for Class 1
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

    def evaluate_single_split(self, X, y, random_state):
        """Evaluate model on a single train-test split using StratifiedShuffleSplit"""
        def strat_shuff_split(X, y, random_state):
            # Initial train-test split using StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.test_size, random_state=random_state)
            train_idx, test_idx = next(sss.split(X, y))

            X_train, X_test, y_train, y_test = X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]

            # Scale features if requested
            if self.scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            
            return X_train, X_test, y_train, y_test

        def grid_search(dtrain, random_state):
            xgb_model = XGBClassifier(**self.base_params)

            # Cross-validation splitter for parameter tuning
            cv_splitter = StratifiedShuffleSplit(
                n_splits=5,
                test_size=0.2,
                random_state=random_state
            )

                # Create GridSearchCV with custom objective function
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=self.param_grid,
                scoring='average_precision',  # AUCPR as the scoring metric
                cv=cv_splitter,  # Use StratifiedShuffleSplit as cross-validation
                verbose=1,
                n_jobs=-1  # Use all CPU cores for parallel computation
            )

            # Run the grid search
            grid_search.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            return best_params, best_score
    
    
        X_train, X_test, y_train, y_test = strat_shuff_split(X, y, random_state)

        # Convert to DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        best_params, best_score = grid_search(dtrain, random_state)


        
        # Extract alpha for final model training
        final_alpha = best_params.pop('alpha')


        # Create final objective function with best alpha
        def final_objective(preds, dtrain):
            return custom_objective_function(preds, dtrain, alpha=final_alpha)
        
        # Train final model with best parameters
        model = xgb.train(
            params=best_params,
            dtrain=dtrain,
            obj = final_objective,
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
            precision, recall, f1, fpr = self.calculate_metrics_at_threshold(
                y_test, y_pred_proba, threshold
            )
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr
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
        auprc_list = []

        
        for i in range(self.n_splits):
            if i%10==0:
                print(f'{i+1} split')
            split_result = self.evaluate_single_split(X, y, random_state=i)
            all_threshold_metrics.append(split_result['threshold_metrics'])
            feature_importances.append(split_result['feature_importance'])
            best_params_list.append(pd.DataFrame([split_result['best_params']]))
            auprc_list.append(split_result['auprc'])
        
        # Calculate mean and std of metrics across all splits for each threshold
        all_threshold_metrics = pd.concat(all_threshold_metrics)
        threshold_metrics_mean = all_threshold_metrics.groupby('threshold').mean()
        threshold_metrics_std = all_threshold_metrics.groupby('threshold').std()
        

        best_params_df = pd.concat(best_params_list)
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
        
        # Rest of the aggregation code remains the same
        return {
            'threshold_df': all_threshold_metrics,
            'threshold_metrics': {
                'mean': threshold_metrics_mean,
                'std': threshold_metrics_std
            },
            'optimal_thresholds': optimal_thresholds,
            'feature_importance': pd.DataFrame({
                'mean_importance': mean_importance,
                'std_importance': std_importance
            }).sort_values('mean_importance', ascending=False),
            'auprc': auprc_list,\
            'parameter_summary_df': best_params_df
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


if __name__=='__main__':
    df = pd.read_excel('0204_Training_Target_Abx.xlsx')
    features = ['Age',
       'TTemp_Max_TT_new', 'TT Fever Start (DPI)_new', 'label',
       'TTemp__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)',
       'TTemp__ar_coefficient__coeff_9__k_10',
       'TTemp__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
       'TTemp__fft_coefficient__attr_"real"__coeff_24',
       'TTemp__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"']
    
    tf_ids, df_features, labels = get_features_df(df, features)
    # Usage example
    n_splits = 100
    evaluator = XGBoostEvaluator(n_splits= n_splits, test_size=0.2, n_threshold_points=1000)
    results = evaluator.evaluate_multiple_splits(df_features, labels)
    
    print("\nModel Performance Summary:")
    print("-------------------------")
    print(f'mean_auprc:{np.nanmean(results["auprc"])}')

    # Plot results
    plot_results(results)
    plt.show()

    results['threshold_df'].to_csv(f'results/{n_splits}_threshold_metrics.csv', index = False)
    results['parameter_summary_df'].to_csv(f'results/{n_splits}_parameter_summary.csv', index = False)
    results['feature_importance'].to_csv(f'results/{n_splits}_feature_importance.csv')    
    pd.DataFrame({'auprc':results['auprc']}).to_csv(f'results/{n_splits}_auprc.csv', index = False)