# coding = 'utf-8'
# pipeline

import os

if __name__ == '__main__':
    print("start preprocessing")
    os.system("python 0_0preprocessing_tag.py")
    os.system("python 0_1preprocessing_trd.py")

    print("start feature engineering")
    os.system("python 1_0trd_id_feature.py")
    os.system("python 1_1trd_time_feature.py")
    os.system("python 1_2trd_R_feature.py")

    print("start model training")
    os.system("python 2_0lgb.py")
    os.system("python 2_1xgb.py")
    os.system("python 3_0model_stack.py")

    print("done!")