## **招商银行2020FinTech精英训练营数据赛道（信用风险评分）**

### B榜0.78422，rank10.   [详细方案分享](https://zhuanlan.zhihu.com/p/140017918)

### **代码说明**

代码为最终版成绩对应代码，为防止混乱(这写的也够乱的了)，方案调优过程中无用的策略没有提供代码。

1. 0_0preprocessing_tag.py: tag表预处理
2. 0_1preprocessing_trd.py: trd表预处理
3. 1_0trd_id_feature.py: trd表中特征提取(按id)
4. 1_1trd_time_feature.py: trd表中特征提取(按id并且分时段)
5. 1_2trd_R_feature.py： trd表中R类特征提取
6. 2_0lgb.py：单模lightgbm，五折交叉验证训练
7. 2_1xgb.py：单模xgboost，五折交叉验证训练
8. 3_0model_stack.py：模型融合，得到最终结果
9. parameter.py：路径参数文件
10. method.py：通用函数方法
11. **run.py**：一键执行pipeline, 预处理、特征提取、模型训练与融合，从原始数据得到最终结果。

### **环境与依赖库**

- Python 3.5.4
- Pandas 0.25.3
- Numpy 1.17.0
- Sklearn 0.21.3
- Lightgbm 2.2.3
- Xgboost 0.81

### **路径说明**

├─Final_code：py代码文件

└─data

​		├─RawData : 原始csv文件(包含训练集和测试集)

​		├─TempData ：预处理文件

​		├─EtlData ：特征文件

​		└─Result ：生成最终结果submission.csv

 