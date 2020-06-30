import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import time
import os


def load_dataset(directory, participant, selected_column_names, screen_out_timestamps=None):
    _dataset = pd.read_csv('{dir}/{participant}.csv'.format(dir=directory, participant=participant)).replace([np.inf, -np.inf], np.nan).dropna(axis=0).drop_duplicates(subset='timestamp')
    if screen_out_timestamps is not None:
        _dataset = _dataset[~_dataset.timestamp.isin(screen_out_timestamps)]
    _features = _dataset[selected_column_names]
    _label = _dataset.gt_label.astype(int)
    return list(_dataset.timestamp), _features, _label


def participant_train_test_xgboost(participant, train_dir, test_dir):
    # train & test dataset
    ts, test_features, test_labels = load_dataset(directory=test_dir, participant=participant, selected_column_names=selected_feature_names)
    _, train_features, train_labels = load_dataset(directory=train_dir, participant=participant, selected_column_names=selected_feature_names, screen_out_timestamps=ts)

    # configure test dataset
    scaler = MinMaxScaler()
    scaler.fit(test_features)
    test_features_scaled = scaler.transform(test_features)
    test_features = pd.DataFrame(test_features_scaled, index=test_features.index, columns=test_features.columns)

    k_folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True)
    for idx, (train_indices, test_indices) in enumerate(splitter.split(train_features, train_labels)):
        x_train = train_features.iloc[train_indices]
        y_train = train_labels.iloc[train_indices]
        x_test = train_features.iloc[test_indices]
        y_test = train_labels.iloc[test_indices]
        k_folds.append((x_train, y_train, x_test, y_test))

    print('# Features : rows({rows}) cols({cols})'.format(rows=train_features.shape[0], cols=train_features.shape[1]))
    # print(train_features.head(), '\n')
    print('# Labels : stressed({stressed}) not-stressed({not_stressed})'.format(stressed=np.count_nonzero(train_labels == 1), not_stressed=np.count_nonzero(train_labels == 0)))
    # print(train_labels.head(), '\n')

    k_folds_sampled = []
    for idx, (x_train, y_train, x_test, y_test) in enumerate(k_folds):
        sampler = SMOTE()
        x_sample, y_sample = sampler.fit_resample(x_train, y_train)
        x_sample = pd.DataFrame(x_sample, columns=x_train.columns)
        y_sample = pd.Series(y_sample)
        k_folds_sampled.append((x_sample, y_sample, x_test, y_test))

    k_folds_scaled = []
    for x_train, y_train, x_test, y_test in k_folds_sampled:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train_scale = scaler.transform(x_train)
        x_test_scale = scaler.transform(x_test)
        x_train = pd.DataFrame(x_train_scale, index=x_train.index, columns=x_train.columns)
        x_test = pd.DataFrame(x_test_scale, index=x_test.index, columns=x_test.columns)
        k_folds_scaled.append((x_train, y_train, x_test, y_test))

    conf_mtx = np.zeros((2, 2))  # 2 X 2 confusion matrix
    train_data = xgb.DMatrix(data=train_features, label=train_labels.to_numpy())

    # Parameter tuning / grid search
    print('tuning parameters...')
    best_params = {'max_depth': 6, 'min_child_weight': 1, 'eta': .3, 'subsample': 1, 'colsample_bytree': 1, 'objective': 'binary:logistic', 'booster': 'gbtree', 'verbosity': 0, 'eval_metric': "auc"}
    grid_search_params = [(max_depth, min_child_weight) for max_depth in range(0, 12) for min_child_weight in range(0, 8)]
    current_test_auc = -float("Inf")
    tmp_params = None
    for max_depth, min_child_weight in grid_search_params:
        # Update our parameters
        best_params['max_depth'] = max_depth
        best_params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(best_params, train_data, nfold=5, metrics=['auc'], early_stopping_rounds=25)
        # Update best MAE
        mean_mae = cv_results['test-auc-mean'].max()
        if mean_mae > current_test_auc:
            current_test_auc = mean_mae
            tmp_params = (max_depth, min_child_weight)
    best_params['max_depth'] = tmp_params[0]
    best_params['min_child_weight'] = tmp_params[1]

    grid_search_params = [(subsample, colsample) for subsample in [i / 10. for i in range(7, 11)] for colsample in [i / 10. for i in range(7, 11)]]
    current_test_auc = -float("Inf")
    tmp_params = {'subsample': None, 'colsample_bytree': None}
    # We start by the largest values and go down to the smallest
    for sub_sample, col_sample in reversed(grid_search_params):
        # We update our parameters
        best_params['subsample'] = sub_sample
        best_params['colsample_bytree'] = col_sample
        # Run CV
        cv_results = xgb.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
        mean_mae = cv_results['test-auc-mean'].max()
        if mean_mae > current_test_auc:
            current_test_auc = mean_mae
            tmp_params = {'subsample': sub_sample, 'colsample_bytree': col_sample}
    best_params['subsample'] = tmp_params['subsample']
    best_params['colsample_bytree'] = tmp_params['colsample_bytree']

    current_test_auc = -float("Inf")
    tmp_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        # We update our parameters
        best_params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
        # Update best score
        mean_mae = cv_results['test-auc-mean'].max()
        if mean_mae > current_test_auc:
            current_test_auc = mean_mae
            tmp_params = eta
    best_params['eta'] = tmp_params

    current_test_auc = -float("Inf")
    tmp_params = None
    gamma_range = [i / 10.0 for i in range(0, 25)]
    for gamma in gamma_range:
        # We update our parameters
        best_params['gamma'] = gamma
        # Run and time CV
        cv_results = xgb.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
        # Update best score
        mean_mae = cv_results['test-auc-mean'].max()
        if mean_mae > current_test_auc:
            current_test_auc = mean_mae
            tmp_params = gamma
    best_params['gamma'] = tmp_params

    print('training and testing...')
    xgb_models = []  # This is used to store models for each fold.
    folds_scores_tmp = {'Accuracy (balanced)': [], 'F1 score': [], 'ROC AUC score': [], 'True Positive rate': [], 'True Negative rate': []}
    for x_train, y_train, x_test, y_test in k_folds_scaled:
        train_data = xgb.DMatrix(data=x_train, label=y_train.to_numpy())
        evaluation_data = xgb.DMatrix(data=x_test, label=y_test.to_numpy())
        test_data = xgb.DMatrix(data=test_features, label=test_labels.to_numpy())

        # docs : https://xgboost.readthedocs.io/en/latest/parameter.html
        results = {}
        booster = xgb.train(best_params, dtrain=train_data, num_boost_round=1000, early_stopping_rounds=25, evals=[(evaluation_data, 'test')], verbose_eval=False, evals_result=results)
        print('Fold evaluation results : ', results)
        predicted_probabilities = booster.predict(data=test_data, ntree_limit=booster.best_ntree_limit)
        predicted_labels = np.where(predicted_probabilities > 0.5, 1, 0)

        acc = balanced_accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, average='macro')
        roc_auc = roc_auc_score(test_labels, predicted_probabilities)
        tpr = recall_score(test_labels, predicted_labels)
        tnr = recall_score(test_labels, predicted_labels, pos_label=0)

        folds_scores_tmp['Accuracy (balanced)'].append(acc)
        folds_scores_tmp['F1 score'].append(f1)
        folds_scores_tmp['ROC AUC score'].append(roc_auc)
        folds_scores_tmp['True Positive rate'].append(tpr)
        folds_scores_tmp['True Negative rate'].append(tnr)

        conf_mtx += confusion_matrix(test_labels, predicted_labels)
        xgb_models.append(booster)

    folds_scores = {}
    for k, v in folds_scores_tmp.items():
        folds_scores[k] = np.mean(v)
    return best_params, folds_scores


if __name__ == '__main__':
    start_time = time.time()
    _train_dir = 'C:/Users/Kevin/Desktop/data-processing-v2/8. no-filter-v2'
    _test_dir = 'C:/Users/Kevin/Desktop/data-processing-v2/threshold-gridsearch/combined-filtered-dataset/test dataset'

    selected_feature_names = ['mean_nni', 'sdnn', 'rmssd', 'nni_50', 'lf', 'hf', 'lf_hf_ratio', 'sampen', 'ratio_sd2_sd1', 'sd2']

    all_params, all_scores = {}, {}
    params_cols, scores_cols = [], []
    for _filename in os.listdir(_train_dir):
        if not _filename.endswith('.csv'):
            continue
        _participant = _filename[:-4]
        print(_participant)
        all_params[_participant] = []
        all_scores[_participant] = []
        params, scores = participant_train_test_xgboost(participant=_participant, train_dir=_train_dir, test_dir=_test_dir)
        if len(params_cols) + len(scores_cols) == 0:
            params_cols = list(params.keys())
            params_cols.sort()
            scores_cols = list(scores.keys())
            scores_cols.sort()
        all_params[_participant] = params
        all_scores[_participant] = scores

    with open("test_scores.csv", "w+") as w_scores, open("test_params.csv", "w+") as w_params:
        w_params.write('Participant,{}\n'.format(','.join(params_cols)))
        w_scores.write('Participant,{}\n'.format(','.join(scores_cols)))
        for _participant in all_params:
            w_params.write(_participant)
            w_scores.write(_participant)
            for param_col in params_cols:
                w_params.write(',{}'.format(all_params[_participant][param_col]))
            for score_col in scores_cols:
                w_scores.write(',{}'.format(all_scores[_participant][score_col]))
            w_params.write('\n')
            w_scores.write('\n')
    print('completed!')
    print(' --- execution time : %s seconds --- ' % (time.time() - start_time))
