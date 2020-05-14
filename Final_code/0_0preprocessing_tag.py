# coding = 'utf-8'

import os
import pandas as pd
from parameter import *
import method as me

if __name__ == '__main__':
    # 读取数据
    print('start user tag processing...')
    train_tag = pd.read_csv(os.path.join(RAW_DATA_PATH, '训练数据集_tag.csv'))
    test_tag = pd.read_csv(os.path.join(RAW_DATA_PATH, '评分数据集_tag_b.csv'))

    test_tag['flag'] = -1
    train_tag['isTest'] = -1
    test_tag['isTest'] = 1

    data = pd.concat([train_tag, test_tag])

    data['atdd_type'].replace({0.0: '0', 1.0: '1'}, inplace=True)
    cols = ['acdm_deg_cd', 'atdd_type']
    for col in cols:
        data[col].fillna(r'\N', inplace=True)
    cols = ['deg_cd', 'edu_deg_cd']
    for col in cols:
        data[col].fillna('~', inplace=True)

    # data['intact_rate'] = 0
    # for col in data.columns:
    #     if col not in ['id', 'flag']:
    #         if col in ['deg_cd', 'edu_deg_cd']:
    #             data['tmp'] = data[col].apply(lambda x: 1 if x not in [r'\N', '~'] else 0)
    #         elif col in ['acdm_deg_cd', 'atdd_type', 'confirm_rsk_ases_lvl_typ_cd', 'crd_card_act_ind',
    #                      'cust_inv_rsk_endu_lvl_cd', 'dnl_bind_cmb_lif_ind', 'dnl_mbl_bnk_ind',
    #                      'fin_rsk_ases_grd_cd', 'fr_or_sh_ind', 'frs_agn_dt_cnt', 'gdr_cd', 'hav_car_grp_ind',
    #                      'hav_hou_grp_ind', 'his_lng_ovd_day', 'hld_crd_card_grd_cd', 'ic_ind', 'job_year',
    #                      'l12_mon_fnd_buy_whl_tms', 'l12_mon_gld_buy_whl_tms', 'l12_mon_insu_buy_whl_tms',
    #                      'l12mon_buy_fin_mng_whl_tms', 'l1y_crd_card_csm_amt_dlm_cd', 'l6mon_agn_ind',
    #                      'l6mon_daim_aum_cd', 'loan_act_ind', 'mrg_situ_cd', 'ovd_30d_loan_tot_cnt',
    #                      'pot_ast_lvl_cd', 'tot_ast_lvl_cd', 'vld_rsk_ases_ind']:
    #             data['tmp'] = data[col].apply(lambda x: 1 if x != r'\N' else 0)
    #         else:
    #             data['tmp'] = data[col].apply(lambda x: 1 if x == -1 else 0)
    #         data['intact_rate'] += data['tmp']
    # data['intact_rate'] /= 41
    # del data['tmp']

    cols = ['confirm_rsk_ases_lvl_typ_cd', 'cust_inv_rsk_endu_lvl_cd', 'fin_rsk_ases_grd_cd',
            'frs_agn_dt_cnt', 'hld_crd_card_grd_cd', 'pot_ast_lvl_cd', 'tot_ast_lvl_cd']
    for col in cols:
        data[col].replace({r'\N': 0}, inplace=True)
        data[col] = data[col].astype(int)

    cols = ['job_year', 'l12_mon_fnd_buy_whl_tms', 'l12_mon_gld_buy_whl_tms',
            'l12_mon_insu_buy_whl_tms', 'l12mon_buy_fin_mng_whl_tms',
            'l1y_crd_card_csm_amt_dlm_cd', 'ovd_30d_loan_tot_cnt', 'his_lng_ovd_day']
    for col in cols:
        data[col].replace({r'\N': -1}, inplace=True)
        data[col] = data[col].astype(int)

    cols = ['acdm_deg_cd', 'deg_cd', 'atdd_type', 'crd_card_act_ind', 'edu_deg_cd',
            'fr_or_sh_ind', 'gdr_cd', 'hav_car_grp_ind', 'hav_hou_grp_ind', 'ic_ind',
            'l6mon_agn_ind', 'loan_act_ind', 'mrg_situ_cd', 'vld_rsk_ases_ind',
            'dnl_bind_cmb_lif_ind', 'dnl_mbl_bnk_ind']
    for col in cols:
        if data[col].dtype == 'object':
            data[col] = data[col].astype(str)
    me.labelEncoder_df(data, cols)

    print(data.info())

    print('saving data: processing tag...')

    data.to_csv(PROCESSING_DATA, index=False)

    print('done!')