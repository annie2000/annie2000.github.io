# Pandas 정리

## 사용법

### Series: 1차원 자료구조
배열/리스트와 같은 일련의 시퀀스 데이타를 정리
별도의 인덱스 레이블을 지정하지 않으면 자동적으로 0부터 시작되는 정수 인덱스 사용


```python
import pandas as pd
data = [1,3,5,7,9]
s = pd.Series(data)
```


```python
s
```




    0    1
    1    3
    2    5
    3    7
    4    9
    dtype: int64



### Data Frame

행과 열이 있는 데이블 데이터(Tabular Data) 처리
열을 Dict의 Key 로, 행을 Dict의 Value로 한 dictionary 데이타를 pd.DataFrame()을 사용해 자료구조화 함


```python
import pandas as pd
data ={'year': [ 2016, 2017, 2018], 
       'GDP rate': [2.8, 3.1, 3.0], 
       'GDP': ['1.637M','1.73M', '1.83M']
      }

df = pd.DataFrame(data)
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>GDP rate</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>2.8</td>
      <td>1.637M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>3.1</td>
      <td>1.73M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>3.0</td>
      <td>1.83M</td>
    </tr>
  </tbody>
</table>
</div>



### Panel

** Deprecated. and removed from Pandas 

3차원 자료 구조: Axis0(items), Axis1(major_axis), Axis2(minor_axis)등 3개의 축을 가지고 있다.

Axis0은 그 한 요소가 2차원의 DataFrame에 해당, Axis1은 DataFram의 행(row)에 해당되고 Axis2는 Dataframe의 열(Column)에 해당된다.


#### 데이타 액세스

인덱싱과 속성을 사용해 접근 : i.e., df['year'], df[df['year]>2016 등


```python
df['year']
```




    0    2016
    1    2017
    2    2018
    Name: year, dtype: int64




```python
df[df['year']>2016]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>GDP rate</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>3.1</td>
      <td>1.73M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>3.0</td>
      <td>1.83M</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>GDP rate</th>
      <th>GDP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016</td>
      <td>2.8</td>
      <td>1.637M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017</td>
      <td>3.1</td>
      <td>1.73M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>3.0</td>
      <td>1.83M</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>GDP rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.0</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2017.0</td>
      <td>2.966667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.0</td>
      <td>0.152753</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2016.0</td>
      <td>2.800000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.5</td>
      <td>2.900000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2017.0</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017.5</td>
      <td>3.050000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2018.0</td>
      <td>3.100000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sum()
```




    year                    6051
    GDP rate                 8.9
    GDP         1.637M1.73M1.83M
    dtype: object




```python
df.mean()
```




    year        2017.000000
    GDP rate       2.966667
    dtype: float64




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3 entries, 0 to 2
    Data columns (total 3 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   year      3 non-null      int64  
     1   GDP rate  3 non-null      float64
     2   GDP       3 non-null      object 
    dtypes: float64(1), int64(1), object(1)
    memory usage: 200.0+ bytes


#### 외부데이터 읽고 쓰기
pandas는 CSV 파일, 텍스트 파일, 엑셀 파일, SQL 데이타베이스, HDF5 포맷 등 다양한 외부 리소스에 데이타를 읽고 쓸 수 있는 기능을 제공


```python
import pandas as pd
df = pd.read_csv('/Users/catherine/Desktop/grade.csv')
```


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Korean</th>
      <th>English</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>80</td>
      <td>85</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>90</td>
      <td>100</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>75</td>
      <td>70</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
%matplotlib inline
import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('/Users/catherine/Desktop/grade.csv')
plt.bar(df.id, df['English'])
plt.show()
```


    
![png](output_20_0.png)
    



```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Korean</th>
      <th>English</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>80</td>
      <td>85</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>90</td>
      <td>100</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>75</td>
      <td>70</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0:, 1:5]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Korean</th>
      <th>English</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>80</td>
      <td>85</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>90</td>
      <td>100</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>75</td>
      <td>70</td>
      <td>65</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0:, 1:5].plot.bar()
```




    <AxesSubplot:>




    
![png](output_23_1.png)
    



```python
df.plot.bar()
```




    <AxesSubplot:>




    
![png](output_24_1.png)
    



```python
import seaborn as sns
%matplotlib inline
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Korean</th>
      <th>English</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>id</th>
      <td>1.000000</td>
      <td>-0.327327</td>
      <td>-0.500000</td>
      <td>-0.327327</td>
    </tr>
    <tr>
      <th>Korean</th>
      <td>-0.327327</td>
      <td>1.000000</td>
      <td>0.981981</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>English</th>
      <td>-0.500000</td>
      <td>0.981981</td>
      <td>1.000000</td>
      <td>0.981981</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>-0.327327</td>
      <td>1.000000</td>
      <td>0.981981</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
a= df.iloc[0:, 1:5].corr()
a
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Korean</th>
      <th>English</th>
      <th>Math</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Korean</th>
      <td>1.000000</td>
      <td>0.981981</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>English</th>
      <td>0.981981</td>
      <td>1.000000</td>
      <td>0.981981</td>
    </tr>
    <tr>
      <th>Math</th>
      <td>1.000000</td>
      <td>0.981981</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.heatmap(a, cmap = 'coolwarm', annot = True)
```




    <AxesSubplot:>




    
![png](output_27_1.png)
    


##  Python Pandas Interview Questions

### Define the Python Pandas?

Pandas is defined as an open-source library that provides high-performance data manipulation in Python. Pandas stands for Panel Data meaning econometrics from multidimensional data. 

### how can you calculate the standard deviation from the Series?


```python
import pandas as pd
import numpy as np
df = pd.Series(np.random.randint(0,7, size =10))
```


```python
df
```


```python
df.std() # std() is defined as a function for calculating the standard deviation of the given set of numbers, Dataframe, Column, and rows
```

### Define DataFrame in Pandas?

A DataFrame is a widely used data structure pf pandas and works with a two-dimentional array with labeled axes(rows and columns). DataFrame is defined as a standard way to store data and has two different indexes, i.e., row index and column index. It consist of the following properties
- The columns can be neterogeous types like int and bool
- it can be seen as a dictionary of Series structure where both the rows and columns are indexed. It is denoted as "columns" in the case of columns and "index" in case of row.

### What are the significant features of the pandas Library?

- Memory efficient
- Data Alignment
- Reshaping
- Merge and join
- Time Series

### Define the different ways a DataFrame can be created in Pandas?
- list
- dict of ndarrays


```python
# Define by list
import pandas as pd

a =['Python', 'Pandas']
info = pd.DataFrame(a)
info
```


```python
# Define by dict

import pandas as pd
info = {'ID': [101, 102, 103], 
       'Department': ['B.Sc', 'B.Tech', 'M.Tech']
       }

info = pd.DataFrame(info)
info
```

### Explain categorical data in Pandas

A categorical data is defined as a Pandas data tye that corresponds to a categorical variable in statistics. A categorical variable is generallly used to take a limited and usually fixed number of possible values.

Example
- Gender, country affiliation, blood type, social class, observation time, or rating via likert scales

### How will you create a series from dict in Pandas?


```python
import pandas as pd
import numpy as np

info = {'X': 0., 'Y':1., 'Z':2. }
a = pd.Series(info)
a
```

### how can we create a copy of the series in Pandas?

- pandas.Series.copy
- Series.copy(deep = True)
    -  If we set deep=True, the data will be copied, and the actual python objects will not be copied recursively, only the reference to the object will be copied.

### how will you create an empty DataFrame in Pandas?


```python
import pandas as pd
info = pd.DataFrame()
info
```

### how will you add a column to a pandas DataFrame?

We can add new column to an existing DataFrame. See the below code


```python
import pandas as pd
info = {'one': pd.Series([1,2,3,4,5], index = ['a', 'b', 'c', 'd', 'e']), 
       'two': pd.Series([1,2,3,4,5,6], index = ['a', 'b', 'c', 'd', 'e', 'f'])}

info = pd.DataFrame(info)
```


```python
info
```


```python
info['three']= pd.Series([20, 40, 60], index = ['a', 'b', 'c'])
info
```


```python
info['four'] = info['one'] + info['three']
```


```python
info
```

### how to add an index, row or column to a Pandas DataFrame?

Adding an index to a DataFrame
- Pandas allow adding the inoyts to te index argument if you create a DataFrame will make sure that you have the desired index
- By default,  

Adding rows to a DataFrame: we can use .loc, iloc to insert the rows in the DataFrame
- The loc works for the labels of the index
    - loc[4] => values of DataFrame that have an index labeled 4
- iloc works for the positions in the index. 
    - iloc[4] => the values of DataFrame that are present at index '4'
    
    
#### Remember    
- Select specific rows and/or columns using loc when using the row and column names
- Select specific rows and/or columns using iloc when using the positions in the table
- You can assign new values to a selection based on loc/iloc.


```python
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'age' :     [ 10, 22, 13, 21, 12, 11, 17],
    'section' : [ 'A', 'B', 'C', 'B', 'B', 'A', 'A'],
    'city' :    [ 'Gurgaon', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai', 'Delhi', 'Mumbai'],
    'gender' :  [ 'M', 'F', 'F', 'M', 'M', 'M', 'F'],
    'favourite_color' : [ 'red', np.NAN, 'yellow', np.NAN, 'black', 'green', 'red']
})

data
```


```python
data.loc[data.age >=15]
```


```python
data.loc[(data.age>=12) & (data.gender == 'M')]
```


```python
data.loc[1:3]
```

### how to select a subset of a DataFrame?


```python
data.iloc[0:3, 3:5] # iloc [행시작:행끝, 칼럼 시작: 칼럼 끝]
```

https://www.javatpoint.com/python-pandas-interview-questions


```python
import pandas as pd
df1 =pd.DataFrame({
    'name':['James', 'Jeff'],
    'Rank': [3, 2]})

df2 =pd.DataFrame({
    'name':['James', 'Jeff'],
    'Rank': [1, 3]})

a= pd.merge(df1, df2, on = 'name')
a
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Rank_x</th>
      <th>Rank_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>James</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jeff</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.DataFrame(a)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Rank</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python

```
