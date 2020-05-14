# coding = 'utf-8'
# 路径参数

import os

BASE_PATH = '../data'

RAW_DATA_PATH = os.path.join(BASE_PATH, 'RawData')
TEMP_DATA_PATH = os.path.join(BASE_PATH, "TempData")
ETL_DATA_PATH = os.path.join(BASE_PATH, 'EtlData')
RESULT_PATH = os.path.join(BASE_PATH, "Result")


SUBMISSION = os.path.join(RESULT_PATH, 'submission.txt')
#############################################################

# 预处理表
PROCESSING_DATA = os.path.join(TEMP_DATA_PATH, 'train_test_processing.csv')  # tag
PROCESSING_TRD_DATA = os.path.join(TEMP_DATA_PATH, 'train_test_trd_processing.csv')  # trd

# trd分组特征
FEATURES_TRD_ID = os.path.join(ETL_DATA_PATH, 'features_trd_id.csv')

# trd按时间分组特征
FEATURES_TRD_TIME = os.path.join(ETL_DATA_PATH, 'features_trd_time.csv')

# trd按R特征
FEATURES_TRD_R = os.path.join(ETL_DATA_PATH, 'features_trd_r.csv')

# 不同模型的stack特征
STACK_PATH = {}
models = ['lgb', 'xgb']
for mod in models:
    STACK_PATH[mod] = os.path.join(ETL_DATA_PATH, 'submission_'+mod+'.txt')
