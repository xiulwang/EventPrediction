# EventPrediction

###目录：

1. 简介
2. 数据
3. 模块设计
	* feature抽取(`FeatureExtract.py`)
	* BaseLine(`LinaerRegression.py`)
	
## 1.简介
Event prediction based on Gdelt event data

## 2.数据
us_feature表，时间：20120101-20180111， 从event_statis表中抽取得到

## 3.模块设计
主要分为以下几个模块：

 - feature抽取(`FeatureExtract.py`)
 - BaseLine(`LinaerRegression.py`)

### 3.1 feature抽取

 - 从us_feature数据库中抽取各州数据，形成csv文件
 - 对于每一个州的csv数据，取相应的数据列，这里暂且只取`column_list = ['geo0_quad4_count_num','geo0_quad4_gold_avg']`
 - X：[T,D], T代表一共有多少个数据，D代表特征维度
 - Y : [T,7],T同上,7代表未来7天label值
 
### 3.2 BaseLine
#### 3.2.1 LinearRegression