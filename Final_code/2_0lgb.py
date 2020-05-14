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

    SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
    skf = SKF.split(train_X, label)

    fea_importances = pd.DataFrame({'column': train_X.columns})

    stack_train = np.zeros((train.shape[0], 1))
    stack_test = np.zeros((test.shape[0], 1))

    for i, (train_fold, validate) in enumerate(skf):
        print("model: lgb. fold: ", i, "training...")
        X_train, label_train = train_X.iloc[train_fold], label.iloc[train_fold]
        X_validate, label_validate = train_X.iloc[validate], label.iloc[validate]

        # for feat_1 in categorical_features:
        #     res = pd.DataFrame()
        #     temp = X_train[[feat_1]]
        #     count = temp.groupby([feat_1]).apply(lambda x: x['flag'].count()).reset_index(name=feat_1 + '_all')
        #     count1 = temp.groupby([feat_1]).apply(lambda x: x['flag'].sum()).reset_index(name=feat_1 + '_1')
        #     count[feat_1 + '_1'] = count1[feat_1 + '_1']
        #     count.fillna(value=0, inplace=True)
        #     count[feat_1 + '_rate'] = round(count[feat_1 + '_1'] / (count[feat_1 + '_all'] + 5), 5)
        #     count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
        #     count.fillna(value=0, inplace=True)
        #     res = res.append(count, ignore_index=True)
        #     # print(feat_1, ' over')
        #     X_train = pd.merge(X_train, res, how='left', on=feat_1)
        #     X_validate = pd.merge(X_train, res, how='left', on=feat_1)
        #     test = pd.merge(X_train, res, how='left', on=feat_1)

        # print(X_train.shape, test.shape)
        dtrain = lgb.Dataset(X_train, label_train)
        dvalid = lgb.Dataset(X_validate, label_validate, reference=dtrain)

        bst = lgb.train(params_lgb, dtrain, valid_sets=(dtrain, dvalid),
                        verbose_eval=50, early_stopping_rounds=100)
        cv_pred += bst.predict(test, num_iteration=bst.best_iteration)
        cv_best_auc_all += bst.best_score['valid_1']['auc']

        score_va = bst.predict(train_X.iloc[validate], num_iteration=bst.best_iteration)
        score_te = bst.predict(test, num_iteration=bst.best_iteration)
        stack_train[validate] += score_va[:, None]
        stack_test += score_te[:, None]

        fea_importance_temp = pd.DataFrame({
            'column': train_X.columns,
            'importance_' + str(i): bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        })

        fea_importances = fea_importances.merge(fea_importance_temp, how='left', on='column')

    cv_pred /= 5
    cv_best_auc_all /= 5
    # valid_best_auc_all /= 5
    print("lgb cv score for valid is: ", cv_best_auc_all)
    # print("lgb valid score for valid is: ", valid_best_auc_all)

    fea_importances['importance'] = (fea_importances['importance_0'] + fea_importances['importance_1'] +
                                     fea_importances['importance_2'] + fea_importances['importance_3'] +
                                     fea_importances['importance_4']) / 5

    fea_importances = fea_importances[['column', 'importance']]
    fea_importances = fea_importances.sort_values(by='importance', ascending=False)
    fea_importances['column'] = fea_importances['column'].apply(lambda x: fea_dict[x])

    stack_test /= 5
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()

    df_stack['lgb_prob'] = np.around(stack[:, 0], 6)
    return cv_pred, fea_importances, df_stack


if __name__ == '__main__':

    data = pd.read_csv(PROCESSING_DATA)

    # one-hot
    # user_property = ['gdr_cd', 'mrg_situ_cd', 'edu_deg_cd', 'acdm_deg_cd', 'deg_cd', 'atdd_type']
    # ind_fea = ['ic_ind', 'fr_or_sh_ind', 'dnl_mbl_bnk_ind', 'dnl_bind_cmb_lif_ind',
    #            'hav_car_grp_ind', 'hav_hou_grp_ind', 'l6mon_agn_ind', 'vld_rsk_ases_ind',
    #            'loan_act_ind', 'crd_card_act_ind']
    # categorical_features = user_property + ind_fea
    # for col in categorical_features:
    #     tmp = pd.get_dummies(data[col])
    #     tmp.columns = [col+'_'+str(x) for x in tmp.columns]
    #     data = data.join(tmp)
    #     data.drop(col, axis=1, inplace=True)

    # 增加cvr特征和count特征
    # features_count_cvr = pd.read_csv(FEATURES_COUNT_CVR)
    # data = pd.concat([data, features_count_cvr], axis=1)

    # tag_WOE
    # features_tag_WOE = pd.read_csv(FEATURES_TAG_WOE)
    # data = pd.concat([data, features_tag_WOE], axis=1)

    # 增加连续变量特征
    # features_nums_class = pd.read_csv(FEATURES_NUMS_CLASS)
    # data = pd.concat([data, features_nums_class], axis=1)

    # trd表的id分组特征
    features_trd_id = pd.read_csv(FEATURES_TRD_ID)
    data = pd.merge(data, features_trd_id, how='left', on='id')
    data.fillna(0, inplace=True)

    # trd表的时序特征
    features_trd_time = pd.read_csv(FEATURES_TRD_TIME)
    data = pd.merge(data, features_trd_time, how='left', on='id')
    data.fillna(0, inplace=True)

    # trd表的R特征
    features_trd_r = pd.read_csv(FEATURES_TRD_R)
    data = pd.merge(data, features_trd_r, how='left', on='id')
    data.fillna(-1, inplace=True)

    # trd表的时序特征2
    # features_trd_time2 = pd.read_csv(FEATURES_TRD_TIME2)
    # data = pd.merge(data, features_trd_time2, how='left', on='id')
    # data.fillna(0, inplace=True)

    # trd tfidf特征
    # tfidf_Trx_Cod1_Cd = pd.read_csv(FEATURE_TRD_TFIDF)
    # data = pd.merge(data, tfidf_Trx_Cod1_Cd, how='left', on='id')
    # data.fillna(0, inplace=True)

    # trd w2v特征
    # feature_trd_tfidf = pd.read_csv(FEATURE_TRD_W2V)
    # data = pd.merge(data, feature_trd_tfidf, how='left', on='id')
    # data.fillna(0, inplace=True)

    # trd 聚集度特征
    # feature_trd_agg = pd.read_csv(FEATURES_TRD_AGG)
    # data = pd.merge(data, feature_trd_agg, how='left', on='id')
    # data.fillna(0, inplace=True)

    # beh tfidf特征
    # feature_beh_tfidf = pd.read_csv(FEATURE_BEH_TFIDF)
    # data = pd.merge(data, feature_beh_tfidf, how='left', on='id')
    # data.fillna(0, inplace=True)

    # beh 表的id分组特征
    # features_beh_id = pd.read_csv(FEATURES_BEH_ID)
    # data = pd.merge(data, features_beh_id, how='left', on='id')
    # data.fillna(0, inplace=True)

    # beh w2v特征
    # feature_beh_tfidf = pd.read_csv(FEATURE_BEH_W2V)
    # data = pd.merge(data, feature_beh_tfidf, how='left', on='id')
    # data.fillna(0, inplace=True)


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
    # features = [col for col in test.columns if col not in drop_fea]
    # TOP100
    features = ['cur_credit_min_opn_dt_cnt', 'l1y_crd_card_csm_amt_dlm_cd', 'Dat_Flg1_Cd_per_count_amt_1',
                'perm_crd_lmt_cd',
                'Trx_Cod2_Cd54_recent_trd', 'cny_trx_amt_sum_6', 'Dat_Flg3_Cd1_recent_trd', 'acdm_deg_cd',
                'cur_debit_min_opn_dt_cnt', 'Dat_Flg3_Cd_per_count_amt_1', 'Trx_Cod1_Cd_amt_2',
                'Dat_Flg1_Cd_per_count_amt_0',
                'Trx_Cod1_Cd_per_count_amt_2', 'Trx_Cod1_Cd_per_day_times_2', 'Trx_Cod1_Cd_amt_0', 'gdr_cd',
                'Trx_Cod2_Cd_per_count_amt_36', 'pot_ast_lvl_cd', 'job_year', 'cny_trx_amt_sum_5',
                'hld_crd_card_grd_cd',
                'Trx_Cod1_Cd_per_count_amt_0', 'fr_or_sh_ind', 'age', '6_Dat_Flg3_Cd_times_1',
                'Dat_Flg1_Cd_per_day_times_1',
                'Trx_Cod1_Cd1_recent_trd', 'Trx_Cod2_Cd_per_day_times_2', 'Trx_Cod2_Cd_amt_36',
                'Dat_Flg1_Cd5_per_day_times_0',
                'per_day_trd_amt', 'Trx_Cod2_Cd53_recent_trd', 'Trx_Cod2_Cd5_per_day_times_26', 'his_lng_ovd_day',
                'hav_car_grp_ind', 'Dat_Flg1_Cd5_per_day_times_1', 'Trx_Cod2_Cd1_recent_trd',
                'Dat_Flg3_Cd_per_count_amt_0',
                'trd_amt', 'dnl_bind_cmb_lif_ind', 'Trx_Cod2_Cd6_per_day_times_32', 'Trx_Cod2_Cd6_per_day_times_54',
                'Trx_Cod2_Cd_amt_54', 'Dat_Flg1_Cd6_per_day_times_1', 'Trx_Cod2_Cd_per_count_amt_30',
                'Trx_Cod2_Cd6_per_day_times_2', 'Dat_Flg3_Cd_amt_1', 'per_count_trd_amt',
                'Trx_Cod2_Cd_per_day_times_46',
                'Dat_Flg3_Cd6_per_day_times_1', 'Trx_Cod1_Cd2_recent_trd', 'frs_agn_dt_cnt', 'edu_deg_cd',
                'Trx_Cod2_Cd_per_day_times_15', 'dnl_mbl_bnk_ind', 'Trx_Cod2_Cd29_recent_trd', 'Dat_Flg1_Cd_amt_1',
                'Trx_Cod2_Cd_amt_32', 'Trx_Cod2_Cd_per_count_amt_55', 'crd_card_act_ind',
                'Trx_Cod2_Cd_per_count_amt_32',
                'Dat_Flg3_Cd_per_day_times_1', 'Trx_Cod2_Cd_amt_53', 'Trx_Cod2_Cd32_recent_trd', 'Dat_Flg1_Cd_amt_0',
                'Trx_Cod2_Cd5_per_day_times_32', 'per_day_trd_times', 'Trx_Cod2_Cd_amt_55', 'Trx_Cod2_Cd_amt_30',
                'Trx_Cod2_Cd_amt_52', '5_Trx_Cod2_Cd_times_54', 'Trx_Cod2_Cd_per_day_times_40',
                'Trx_Cod2_Cd_per_count_amt_54',
                'Trx_Cod2_Cd_per_day_times_3', 'Trx_Cod1_Cd5_per_day_times_2', 'Dat_Flg3_Cd_times_1',
                'Trx_Cod2_Cd_per_day_times_54', 'Dat_Flg3_Cd5_per_day_times_1', '6_Trx_Cod2_Cd_times_10',
                'Trx_Cod2_Cd6_per_day_times_1', 'Trx_Cod2_Cd5_per_day_times_30', 'Trx_Cod2_Cd_per_count_amt_11',
                'Trx_Cod2_Cd_amt_24', 'Dat_Flg3_Cd_amt_0', 'Trx_Cod1_Cd6_per_day_times_2', 'Trx_Cod2_Cd30_recent_trd',
                'Trx_Cod2_Cd_per_day_times_16', '6_Trx_Cod2_Cd_times_54', 'Trx_Cod2_Cd6_per_day_times_55',
                'Trx_Cod2_Cd_per_day_times_32', 'Trx_Cod1_Cd0_recent_trd', 'Trx_Cod2_Cd_per_day_times_0',
                'Trx_Cod1_Cd_per_day_times_1', 'Trx_Cod1_Cd_per_count_amt_1', 'Trx_Cod2_Cd_per_count_amt_21',
                'Trx_Cod2_Cd_times_54', 'Dat_Flg1_Cd_per_day_times_0', 'Trx_Cod2_Cd5_per_day_times_54',
                'Trx_Cod2_Cd16_recent_trd',
                'Dat_Flg1_Cd0_recent_trd']
    # TOP150
    # features = ['cur_credit_min_opn_dt_cnt', 'l1y_crd_card_csm_amt_dlm_cd', 'perm_crd_lmt_cd', 'Dat_Flg1_Cd_per_count_amt_1',
    #  'cny_trx_amt_sum_6', 'cur_debit_min_opn_dt_cnt', 'Dat_Flg3_Cd_per_count_amt_1', 'acdm_deg_cd', 'Trx_Cod1_Cd_amt_2',
    #  '6_Dat_Flg3_Cd_times_1', 'Dat_Flg1_Cd_per_count_amt_0', 'Trx_Cod1_Cd_amt_0', 'Trx_Cod1_Cd_per_count_amt_2', 'age',
    #  'job_year', 'cny_trx_amt_sum_5', 'Trx_Cod2_Cd6_per_day_times_54', 'Trx_Cod1_Cd_per_day_times_2',
    #  'Trx_Cod2_Cd_per_count_amt_36', 'pot_ast_lvl_cd', 'gdr_cd', 'Trx_Cod2_Cd_amt_36', 'trd_amt',
    #  'Dat_Flg1_Cd5_per_day_times_0', 'hld_crd_card_grd_cd', 'Dat_Flg3_Cd_per_count_amt_0', 'per_day_trd_amt',
    #  'Trx_Cod1_Cd_per_count_amt_0', '6_Trx_Cod2_Cd_times_54', 'Dat_Flg1_Cd_amt_0', 'Trx_Cod1_Cd5_per_day_times_2',
    #  'Dat_Flg3_Cd6_per_day_times_1', 'per_day_trd_times', 'hav_car_grp_ind', 'Trx_Cod1_Cd6_per_day_times_0',
    #  'Trx_Cod2_Cd_per_count_amt_30', 'Trx_Cod2_Cd5_per_day_times_26', 'fr_or_sh_ind', 'Dat_Flg1_Cd5_per_day_times_1',
    #  'Dat_Flg1_Cd6_per_day_times_1', 'Dat_Flg1_Cd_per_day_times_1', 'frs_agn_dt_cnt', 'Trx_Cod2_Cd5_per_day_times_32',
    #  'Trx_Cod2_Cd_per_day_times_46', 'per_count_trd_amt', 'Trx_Cod2_Cd5_per_day_times_30', 'dnl_bind_cmb_lif_ind',
    #  'his_lng_ovd_day', 'Trx_Cod2_Cd_per_day_times_2', 'crd_card_act_ind', 'Trx_Cod2_Cd6_per_day_times_32',
    #  'Dat_Flg1_Cd_amt_1', 'Trx_Cod2_Cd_amt_54', 'Dat_Flg3_Cd5_per_day_times_1', 'Trx_Cod2_Cd_per_day_times_16',
    #  'Trx_Cod2_Cd_per_day_times_54', 'Trx_Cod2_Cd_amt_32', 'Trx_Cod1_Cd6_per_day_times_2',
    #  'Trx_Cod2_Cd_per_count_amt_54', 'Trx_Cod2_Cd6_per_day_times_1', 'Trx_Cod2_Cd_per_count_amt_32',
    #  'Trx_Cod2_Cd_per_day_times_30', 'Trx_Cod2_Cd6_per_day_times_2', 'Dat_Flg3_Cd_amt_1', 'edu_deg_cd',
    #  'Trx_Cod1_Cd5_per_day_times_0', 'Trx_Cod2_Cd_amt_55', 'Trx_Cod2_Cd_amt_30', 'dnl_mbl_bnk_ind', 'cur_credit_cnt',
    #  'Trx_Cod2_Cd_per_day_times_15', 'Trx_Cod2_Cd_per_day_times_3', 'Trx_Cod2_Cd_per_day_times_32',
    #  'Trx_Cod2_Cd_amt_52', 'Dat_Flg3_Cd_times_1', 'Trx_Cod2_Cd_per_day_times_40', 'Dat_Flg3_Cd_per_day_times_1',
    #  'Dat_Flg3_Cd_amt_0', 'Trx_Cod2_Cd6_per_day_times_29', 'Trx_Cod2_Cd_per_day_times_17',
    #  'Trx_Cod2_Cd_per_count_amt_41', 'Trx_Cod2_Cd_per_count_amt_55', 'Trx_Cod1_Cd5_per_day_times_1',
    #  'Trx_Cod2_Cd5_per_day_times_54', 'Trx_Cod2_Cd_per_day_times_12', 'Trx_Cod2_Cd_per_day_times_29',
    #  'Trx_Cod2_Cd_per_day_times_0', 'Trx_Cod2_Cd_per_day_times_53', '6_Trx_Cod2_Cd_times_10',
    #  'Trx_Cod2_Cd_per_count_amt_11', 'Dat_Flg1_Cd_per_day_times_0', '5_Trx_Cod2_Cd_times_54',
    #  'Trx_Cod2_Cd_per_count_amt_15', 'Trx_Cod2_Cd_amt_53', 'Trx_Cod2_Cd_per_day_times_26',
    #  'Trx_Cod2_Cd_per_count_amt_16', 'Trx_Cod1_Cd_per_day_times_0', 'Trx_Cod2_Cd_per_count_amt_12',
    #  'Trx_Cod2_Cd_times_54', 'Dat_Flg1_Cd6_per_day_times_0', 'Trx_Cod2_Cd6_per_day_times_55',
    #  'Trx_Cod2_Cd_per_count_amt_21', 'Trx_Cod2_Cd_per_count_amt_2', 'Trx_Cod2_Cd5_per_day_times_7',
    #  'confirm_rsk_ases_lvl_typ_cd', 'Trx_Cod2_Cd_per_day_times_19', 'Trx_Cod1_Cd_per_day_times_1',
    #  'Trx_Cod1_Cd_per_count_amt_1', 'Trx_Cod2_Cd_amt_15', 'Trx_Cod2_Cd_amt_24', '5_Dat_Flg1_Cd_times_1',
    #  'Trx_Cod2_Cd_per_day_times_1', 'Dat_Flg1_Cd_times_1', 'Trx_Cod2_Cd_per_count_amt_1',
    #  'Trx_Cod2_Cd_per_day_times_36', 'Trx_Cod2_Cd_per_count_amt_10', 'Trx_Cod2_Cd_amt_2',
    #  'Trx_Cod2_Cd5_per_day_times_2', 'Trx_Cod2_Cd_per_day_times_45', 'Dat_Flg3_Cd5_per_day_times_0',
    #  'Trx_Cod1_Cd_times_2', '6_Trx_Cod2_Cd_times_32', 'Trx_Cod2_Cd_per_count_amt_52', 'Trx_Cod2_Cd5_per_day_times_24',
    #  'Trx_Cod2_Cd6_per_day_times_30', 'Trx_Cod2_Cd_per_count_amt_25', 'Trx_Cod2_Cd6_per_day_times_7',
    #  'Trx_Cod2_Cd_per_count_amt_3', 'Trx_Cod2_Cd_amt_1', 'Trx_Cod2_Cd_amt_29', 'Trx_Cod2_Cd_per_count_amt_24',
    #  'Trx_Cod2_Cd6_per_day_times_19', 'Trx_Cod2_Cd6_per_day_times_11', 'Trx_Cod2_Cd_amt_19',
    #  'Trx_Cod2_Cd6_per_day_times_36', 'Trx_Cod2_Cd_amt_41', '6_Dat_Flg1_Cd_times_1', 'trd_count',
    #  'Trx_Cod2_Cd_per_count_amt_45', 'Trx_Cod2_Cd_per_day_times_41', 'Trx_Cod2_Cd_per_count_amt_19',
    #  'Trx_Cod2_Cd5_per_day_times_16', 'Trx_Cod2_Cd5_per_day_times_28', 'Trx_Cod2_Cd5_per_day_times_41',
    #  'Trx_Cod2_Cd_per_day_times_6', 'Trx_Cod2_Cd_per_day_times_10', 'Trx_Cod2_Cd6_per_day_times_9',
    #  'Trx_Cod2_Cd5_per_day_times_31', 'Trx_Cod2_Cd5_per_day_times_10', 'count_sum_6']

    train_X = train[features]
    train_y = train['flag']

    print(train_X.shape, test[features].shape)

    result, feature_importance, df_stack = lgb_model(train_X, train_y, test[features], params_lgb)

    print(train_X.shape, test[features].shape)

    print('\n most importance features: ')
    print(list(feature_importance.iloc[:150]['column']))
    print('\n features(importance=0): ')
    print(list(feature_importance[feature_importance['importance'] == 0]['column']))

    # saving result
    print('saving result...')
    submission['pred'] = result
    submission.to_csv(SUBMISSION, index=False, header=0, encoding='utf-8', sep='\t')
    print('done')

    df_stack.to_csv(STACK_PATH['lgb'], index=None, encoding='utf8')
    print('lgb特征已保存\n')
