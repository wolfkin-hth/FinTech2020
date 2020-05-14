# coding = 'utf-8'
# trd特征提取(F  M)--进一步细粒度，分两个月

import pandas as pd
from parameter import *

data = pd.read_csv(PROCESSING_TRD_DATA)

data['count'] = 1

fea = data[['id']].drop_duplicates().reset_index(drop=True)

data.sort_values(by=['id', 'trx_tm'], ascending=True, inplace=True)

data['day'] = data['day'] - data['day'].min()

# trd_time_min = data.groupby('id')['day'].agg({'trd_time_min': 'min'}).reset_index()
# fea = pd.merge(fea, trd_time_min, how='left', on='id')
#
# trd_time_max = data.groupby('id')['day'].agg({'trd_time_max': 'max'}).reset_index()
# fea = pd.merge(fea, trd_time_max, how='left', on='id')

# 分月份统计
data_5= data[data['day']<=30]
data_6= data[data['day']>30]

trd_sum_5 = data_5.groupby('id')['date'].agg({'day_count_5': 'nunique'}).reset_index()
fea = pd.merge(fea, trd_sum_5, how='left', on='id')

trd_sum_6 = data_6.groupby('id')['date'].agg({'day_count_6': 'nunique'}).reset_index()
fea = pd.merge(fea, trd_sum_6, how='left', on='id')

# fea['rate_nday_6/5'] = fea['day_count_6'] / (fea['day_count_5'] + 0.01)

cols = ['cny_trx_amt', 'count']
for col in cols:
    trd_sum_5 = data_5.groupby('id')[col].agg({col+'_sum_5': 'sum'}).reset_index()
    fea = pd.merge(fea, trd_sum_5, how='left', on='id')

    trd_sum_6 = data_6.groupby('id')[col].agg({col+'_sum_6': 'sum'}).reset_index()
    fea = pd.merge(fea, trd_sum_6, how='left', on='id')

    # fea['rate_'+col+'_6/5'] = fea[col+'_sum_6'] / (fea[col+'_sum_5'] + 0.01)

    # fea[col+'5_per_day_times'] = fea[col+'_sum_5'] / fea['day_count_5']
    # fea[col + '6_per_day_times'] = fea[col + '_sum_6'] / fea['day_count_6']

# fea['5_per_count_trd_amt'] = fea['cny_trx_amt_sum_5'] / fea['count_sum_5']
# fea['6_per_count_trd_amt'] = fea['cny_trx_amt_sum_6'] / fea['count_sum_6']

cols = ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd']
for col in cols:
    tmp_count = data_5.groupby(['id', col])['count'].agg({'sum'})
    tmp_count = tmp_count.unstack().reset_index()
    tmp_count.fillna(0, inplace=True)
    tmp1 = list(tmp_count.columns)
    tmp1[0] = 'id'
    tmp1[1:] = ['5_' + col + '_times_' + str(x[1]) for x in tmp1[1:]]
    tmp_count.columns = tmp1

    fea = pd.merge(fea, tmp_count, how='left', on='id')
    tmp3 = []
    for xcol in tmp_count.columns[1:]:
        fea[col+'5_per_day_times_'+xcol.split('_')[-1]] = fea[xcol] / fea['day_count_5']
        tmp3.append(col+'5_per_day_times_'+xcol.split('_')[-1])

    tmp_count = data_6.groupby(['id', col])['count'].agg({'sum'})
    tmp_count = tmp_count.unstack().reset_index()
    tmp_count.fillna(0, inplace=True)
    tmp2 = list(tmp_count.columns)
    tmp2[0] = 'id'
    tmp2[1:] = ['6_' + col + '_times_' + str(x[1]) for x in tmp2[1:]]
    tmp_count.columns = tmp2

    fea = pd.merge(fea, tmp_count, how='left', on='id')
    tmp4 = []
    for xcol in tmp_count.columns[1:]:
        fea[col + '6_per_day_times_' + xcol.split('_')[-1]] = fea[xcol] / fea['day_count_6']
        tmp4.append(col + '6_per_day_times_' + xcol.split('_')[-1])

    # for i in range(1, len(tmp1)):
    #     fea['rate_' + str(i) + '_6/5'] = fea[tmp2[i]] / (fea[tmp1[i]] + 0.01)
    # for i in range(len(tmp3)):
    #     fea['rate_xx' + str(i) + '_6/5'] = fea[tmp4[i]] / (fea[tmp3[i]] + 0.01)


fea.fillna(0.0, inplace=True)
print(fea.info())

fea.to_csv(FEATURES_TRD_TIME, index=False)