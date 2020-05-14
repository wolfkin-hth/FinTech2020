# coding = 'utf-8'
# author = 'wolfkin'
# 预处理trd

import os
import pandas as pd
from parameter import *
import method as me
from datetime import datetime
import time

if __name__ == '__main__':
    # 读取数据
    print('start trd processing...')
    train_trd = pd.read_csv(os.path.join(RAW_DATA_PATH, '训练数据集_trd.csv'))
    # test_trd_a = pd.read_csv(os.path.join(RAW_DATA_PATH, '评分数据集_trd.csv'))
    test_trd = pd.read_csv(os.path.join(RAW_DATA_PATH, '评分数据集_trd_b.csv'))

    test_trd['flag'] = -1
    # test_trd_a['flag'] = -1
    train_trd['isTest'] = -1
    test_trd['isTest'] = 1
    # test_trd_a['isTest'] = 1

    data = pd.concat([train_trd, test_trd])

    data['month'] = data['trx_tm'].apply(lambda x: int(x[5:7]))
    data['hour'] = data['trx_tm'].apply(lambda x: int(x[11:13]))
    data['day_1'] = data['trx_tm'].apply(lambda x: int(x[8:10]))
    data['date'] = data['trx_tm'].apply(lambda x: x[0:10])
    data['trx_tm'] = data['trx_tm'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    data['day'] = data['trx_tm'].apply(lambda x: x.dayofyear)
    data['weekday'] = data['trx_tm'].apply(lambda x: x.weekday())
    data['isWeekend'] = data['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
    data['trx_tm'] = data['trx_tm'].apply(lambda x: int(time.mktime(x.timetuple())))

    me.labelEncoder_df(data, ['Dat_Flg1_Cd', 'Dat_Flg3_Cd', 'Trx_Cod1_Cd', 'Trx_Cod2_Cd'])
    # 保存数据
    print(data.isnull().sum())
    print(data.info())

    print('saving data: processing trd...')
    data.to_csv(PROCESSING_TRD_DATA, index=False)
    print('done!')


