# coding = 'utf-8'
# author = 'wolfkin'
# trd特征提取(F   M)

import pandas as pd
from parameter import *

data = pd.read_csv(PROCESSING_TRD_DATA)

data['count'] = 1

fea = data[['id']].drop_duplicates().reset_index(drop=True)

data.sort_values(by=['id', 'trx_tm'], ascending=True, inplace=True)

# 1. 总次数、总天数、总金额
trd_count = data.groupby('id')['count'].agg({'trd_count': 'sum'}).reset_index()
fea = pd.merge(fea, trd_count, how='left', on='id')

day_count = data.groupby('id')['date'].agg({'day_count': 'nunique'}).reset_index()
fea = pd.merge(fea, day_count, how='left', on='id')

trd_amt = data.groupby('id')['cny_trx_amt'].agg({'trd_amt': 'sum'}).reset_index()
fea = pd.merge(fea, trd_amt, how='left', on='id')

# 每个id最大金额、最小金额、金额gap
# trd_max = data.groupby('id')['cny_trx_amt'].agg({'trd_max': 'max'}).reset_index()
# fea = pd.merge(fea, trd_max, how='left', on='id')
#
# trd_min = data.groupby('id')['cny_trx_amt'].agg({'trd_min': 'min'}).reset_index()
# fea = pd.merge(fea, trd_min, how='left', on='id')
#
# fea['trd_gap'] = fea['trd_max'] - fea['trd_min']


# 2. 平均每天交易次数、平均每天交易金额、平均每次交易金额
fea['per_day_trd_times'] = fea['trd_count'] / fea['day_count']
fea['per_day_trd_amt'] = fea['trd_amt'] / fea['day_count']
fea['per_count_trd_amt'] = fea['trd_amt'] / fea['trd_count']

# 3. 交易方向、交易方式、一级二级代码、分别计总次数与平均每天次数
cols = ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd']
for col in cols:
    tmp_count = data.groupby(['id', col])['count'].agg({'sum'})
    tmp_count = tmp_count.unstack().reset_index()
    tmp_count.fillna(0, inplace=True)
    tmp = list(tmp_count.columns)
    tmp[0] = 'id'
    tmp[1:] = [col + '_times_' + str(x[1]) for x in tmp[1:]]
    tmp_count.columns = tmp

    fea = pd.merge(fea, tmp_count, how='left', on='id')
    for xcol in tmp_count.columns[1:]:
        fea[col+'_per_day_times_'+xcol.split('_')[-1]] = fea[xcol] / fea['day_count']

# 3. 交易方向、交易方式、一级二级代码、分别计金额数, 平均每天金额数和平均每次金额数
cols = ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd']
for col in cols:
    tmp_count = data.groupby(['id', col])['cny_trx_amt'].agg({'sum'})
    tmp_count = tmp_count.unstack().reset_index()
    tmp_count.fillna(0, inplace=True)
    tmp = list(tmp_count.columns)
    tmp[0] = 'id'
    tmp[1:] = [col + '_amt_' + str(x[1]) for x in tmp[1:]]
    tmp_count.columns = tmp
    fea = pd.merge(fea, tmp_count, how='left', on='id')

    for xcol in tmp_count.columns[1:]:
    #     fea[col+'_per_day_amt_'+xcol.split('_')[-1]] = fea[xcol] / fea['day_count']
        fea[col + '_per_count_amt_' + xcol.split('_')[-1]] = fea[xcol] / fea[col + '_times_' + xcol.split('_')[-1]]

    # 每个交易方向、交易方式、一级二级代码、分别计最大金额、最小金额、金额gap
    # tmp_count = data.groupby(['id', col])['cny_trx_amt'].agg({'max'})
    # tmp_count = tmp_count.unstack().reset_index()
    # tmp_count.fillna(0, inplace=True)
    # tmp = list(tmp_count.columns)
    # tmp[0] = 'id'
    # tmp[1:] = [col + '_amt_max_' + str(x[1]) for x in tmp[1:]]
    # tmp_count.columns = tmp
    # fea = pd.merge(fea, tmp_count, how='left', on='id')
    #
    # tmp_count = data.groupby(['id', col])['cny_trx_amt'].agg({'min'})
    # tmp_count = tmp_count.unstack().reset_index()
    # tmp_count.fillna(0, inplace=True)
    # tmp = list(tmp_count.columns)
    # tmp[0] = 'id'
    # tmp[1:] = [col + '_amt_min_' + str(x[1]) for x in tmp[1:]]
    # tmp_count.columns = tmp
    # fea = pd.merge(fea, tmp_count, how='left', on='id')
    #
    # for xcol in tmp_count.columns[1:]:
    #     fea[col + '_amt_gap_' + xcol.split('_')[-1]] = fea[col + '_amt_max_' + xcol.split('_')[-1]] - fea[xcol]

# for col in fea.columns[1:]:
#     fea[col+'_rank'] = fea[col].rank(method='min')
# fea.fillna(0, inplace=True)

# for col in fea.columns[1:]:
#     fea['Is_'+col+'_mean']=(fea[col]>fea[col].mean())*1
print(fea.info())
fea.to_csv(FEATURES_TRD_ID, index=False)
