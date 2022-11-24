#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright (C) 2022 #
# @Time    : 2022/11/24 23:44
# @Author  : TaoistQu
# @Email   : qulei_20180331@163.com
# @File    : data_analysis.py
# @Software: PyCharm
import os,sys
sys.path.append(os.pardir)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#数据加载
job = pd.read_csv('D:\MyCode\AI\Pandas\data\job.csv')
#print(job.shape,job['city'].unique())

colums = ["positionName","companyShortName", "city", "companySize",
          "education", "financeStage","industryField", "salary",
          "workYear","companyLabelList","job_detail"]
job = job[colums].drop_duplicates()

#print(job.shape,job.head())
'''
####数据清洗
###提取数据分析师数据
cond = job["positionName"].str.contains("数据分析")
job = job[cond]
job.reset_index(inplace=True)
print(job.shape,job.head())
print(cond)
'''

'''
薪资转换
job["salary"] = job["salary"].str.lower().str.extract(r'(\d+)[k]-(\d+)k')\
.applymap(lambda x:int(x)).mean(axis=1)

'''

'''
job["job_detail"] =job["job_detail"].str.lower().fillna("") #将字符串小写化，并将缺失值赋值为空字符串
job["Python"] = job["job_detail"].map(lambda x:1 if
('python' in x) else 0)
job["SQL"] = job["job_detail"].map(lambda
x:1 if ('sql' in x) or ('hive' in x) else 0)
job["Tableau"] = job["job_detail"].map(lambda x:1 if
'tableau' in x else 0)
job["Excel"] = job["job_detail"].map(lambda x:1 if 'excel' in x else 0)
job['SPSS/SAS'] =job['job_detail'].map(lambda x:1 if ('spss'
in x) or ('sas' in x) else 0)


print(job.head())

'''

plt.figure(figsize=(12,9))
cities = job['city'].value_counts() # 统计城市工作数量
plt.barh(y = cities.index[::-1],
width = cities.values[::-1],
color = '#3c7f99')
plt.box(False) # 不显示边框
plt.title(label=' 各城市数据分析岗位的需求量 ',fontsize=32, weight='bold',
color='white',
backgroundcolor='#c5b783',pad =
30 )
plt.tick_params(labelsize = 16)
plt.grid(axis = 'x',linewidth = 0.5,color =
'#3c7f99')
plt.show()

