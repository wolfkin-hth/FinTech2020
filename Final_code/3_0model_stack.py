# coding = 'utf-8'

import numpy as np
import pandas as pd
from parameter import *
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

def lgb_model(train_X, label, test, params_lgb):
    fea_dict = {v: k for k, v in enumerate(train_X.columns)}
    train_X.columns = [fea_dict[v] for v in train_X.columns]
    test.columns = train_X.columns

    fea_dict = {v: k for k, v in fea_dict.items()}
    # lgb 模型
    cv_pred = np.zeros(test.shape[0])
    cv_best_auc_all = 0

    SKF = StratifiedKFold(n_splits=10, shuffle=True, random_state=2019)
    skf = SKF.split(train_X, label)

    fea_importances = pd.DataFrame({'column': train_X.columns})

    for i, (train_fold, validate) in enumerate(skf):
        print("model: lgb. fold: ", i, "training...")
        X_train, label_train = train_X.iloc[train_fold], label.iloc[train_fold]
        X_validate, label_validate = train_X.iloc[validate], label.iloc[validate]

        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params_lgb, dtrain, valid_sets=(dtrain, dvalid),
                        verbose_eval=50, early_stopping_rounds=300)
        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        cv_best_auc_all += bst.best_score['valid_1']['auc']

        fea_importance_temp = pd.DataFrame({
            'column': train_X.columns,
            'importance_' + str(i): bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        })

        fea_importances = fea_importances.merge(fea_importance_temp, how='left', on='column')

    cv_pred /= 10
    cv_best_auc_all /= 10
    # valid_best_auc_all /= 5
    print("lgb cv score for valid is: ", cv_best_auc_all)
    # print("lgb valid score for valid is: ", valid_best_auc_all)

    fea_importances['importance'] = (fea_importances['importance_0'] + fea_importances['importance_1'] +
                                     fea_importances['importance_2'] + fea_importances['importance_3'] +
                                     fea_importances['importance_4'] + fea_importances['importance_5'] +
                                     fea_importances['importance_6'] + fea_importances['importance_7'] +
                                     fea_importances['importance_8'] + fea_importances['importance_9']) / 10

    fea_importances = fea_importances[['column', 'importance']]
    fea_importances = fea_importances.sort_values(by='importance', ascending=False)
    fea_importances['column'] = fea_importances['column'].apply(lambda x: fea_dict[x])

    return cv_pred, fea_importances


if __name__ == '__main__':

    data = pd.read_csv(PROCESSING_DATA)

    xgb_stack = pd.read_csv(STACK_PATH['xgb'])
    lgb_stack = pd.read_csv(STACK_PATH['lgb'])

    data = pd.concat([data, lgb_stack, xgb_stack], axis=1)

    # 删除某一类别占比超过99%的列
    good_cols = list(data.columns)
    for col in data.columns:
        rate = data[col].value_counts(normalize=True, dropna=False).values[0]
        if rate > 0.999:
            good_cols.remove(col)
            print(col, rate)

    data = data[good_cols]
    train = data[data['isTest'] == -1]
    test = data[data['isTest'] == 1]

    submission = pd.DataFrame()
    submission['id'] = test['id']

    params_lgb = {
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'max_depth': -1,
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 16,
        'learning_rate': 0.1,
        'feature_fraction': 1.,
        'bagging_fraction': 1.,
        'reg_lambda': 0.5,
        'reg_alpha': 0.3,
        'random_state': 1024,
        'n_jobs': -1,
    }

    drop_fea = ['id', 'flag', 'isTest']
    features = [col for col in test.columns if col not in drop_fea]
    train_X = train[features]
    train_y = train['flag']

    print(train_X.shape, test[features].shape)

    result, feature_importance = lgb_model(train_X, train_y, test[features], params_lgb)

    print(train_X.shape, test[features].shape)

    print('\n features(importance=0): ')
    print(list(feature_importance[feature_importance['importance'] == 0]['column']))

    # saving result
    print('saving result...')
    submission['pred'] = result
    submission.to_csv(SUBMISSION, index=False, header=0, encoding='utf-8', sep='\t')
    print('done')
