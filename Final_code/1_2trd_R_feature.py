# coding = 'utf-8'
# trd特征提取(R)

import pandas as pd
from parameter import *

data = pd.read_csv(PROCESSING_TRD_DATA)

data['count'] = 1

fea = data[['id']].drop_duplicates().reset_index(drop=True)

data.sort_values(by=['id', 'trx_tm'], ascending=True, inplace=True)

data['day'] = data['day'] - data['day'].min()

# 最近一次交易时间
recent_trd = data.groupby('id')['day'].agg({'recent_trd': 'max'}).reset_index()
recent_trd['recent_trd'] = 60-recent_trd['recent_trd']
recent_trd['Is_recent_trd_mean']=(recent_trd['recent_trd']>recent_trd['recent_trd'].mean())*1
fea = pd.merge(fea, recent_trd, how='left', on='id')

# 3. 交易方向、交易方式、一级二级代码、分别计最近一次时间
cols = ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd']
for col in cols:
    col_value = list(data[col].unique())
    for c_v in col_value:
        print(col, c_v)
        data_tmp = data[data[col] == c_v]
        recent_trd = data_tmp.groupby('id')['day'].agg({col+str(c_v)+'_recent_trd': 'max'}).reset_index()
        recent_trd[col+str(c_v)+'_recent_trd'] = 60 - recent_trd[col+str(c_v)+'_recent_trd']
        recent_trd['Is_recent_'+col+str(c_v)+'_mean'] = (recent_trd[col+str(c_v)+'_recent_trd'] > recent_trd[col+str(c_v)+'_recent_trd'].mean()) * 1
        fea = pd.merge(fea, recent_trd, how='left', on='id')

for col in fea.columns[1:]:
    if 'mean' in col:
        fea[col].fillna(-1, inplace=True)
    else:
        fea[col].fillna(-1, inplace=True)
print(fea.info())

fea.to_csv(FEATURES_TRD_R, index=False)