# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/0d98ab73-b7d3-4f27-96ab-c2b90e1bc1fd)
```
max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/7855dadd-18c0-4163-95b1-b4b872eb6492)
```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
![image](https://github.com/user-attachments/assets/b2680e42-c79c-40a3-a8a9-b0adb6fe6f99)
## Min-Max Scaling
 ```
from sklearn.preprocessing import MinMaxScaler()
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/a582a841-9f31-4d7a-82fd-4a9b397c61db)
## Normalization
```
df2=pd.read_csv("bmi.csv")
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2
```
## Maximum absolute scaling
```
df3=pd.read_csv("bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/48979a4f-421c-4f7d-9208-51d98f3280f1)
## RobustScaler
```
df4=pd.read_csv("bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/4bc465bf-7b83-4f47-b170-6f5fc2cc28e7)
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('income(1) (1) (2).csv',na_values=[" ?"])
data
```
![image](https://github.com/user-attachments/assets/5f07aa82-50c7-462e-9e65-fffbd807e8bc)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/d14dfb7d-b96e-4281-854d-7c64affa8bb3)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/e41f23ae-a258-4ddd-b923-9c7fd810c08e)
```
data2 = data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/86e66b11-f835-4135-a4a3-06378bc02e5c)
```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/af61143f-863d-432b-b2cf-11f419269f5c)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/d7cba62d-5b5e-484e-a9e2-e403d3b1b29e)
```
data2
```
![image](https://github.com/user-attachments/assets/01a64651-e7b7-4f4d-b8e4-778bd8c5cd67)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/50fa3d37-a93c-4e88-ae0d-754069e420b6)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/594d6ecf-3977-45fa-af12-0aac273c050b)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/831bd978-dc02-4134-a018-3cc80da2fc00)
```
y=new_data['SalStat'].values
print(y)
```
[0 0 1 ... 0 0 0]
```
x = new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/b22d91b5-b5f4-4b5e-97dd-3e1e41a726cb)
```
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/966ea247-7c4c-450a-bad5-9695ccc50d6c)
```
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
```
![image](https://github.com/user-attachments/assets/5a850d30-3d65-494f-96f7-8bee7e6ba17e)
```
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)
```
0.8392087523483258
```
print('Misclassified samples: %d' % (test_y != prediction).sum())
```
Misclassified samples: 1455
```
data.shape

(31978, 13)
```
## FEATURE SELECTION TECHNIQUES
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/d7b3aa68-21ce-42aa-b735-9d5ac0d8edc6)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/5107b6e3-bcbb-43a5-8ca9-50cd43b6df51)
```
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
```
![image](https://github.com/user-attachments/assets/07fdd944-6d09-40b6-98f1-8500411c2257)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/bd6d4af0-f6f9-45b2-bc40-ddee7ab8318d)

# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is successful.
