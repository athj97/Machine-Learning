```python
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import math
```


```python
f = pd.read_csv("spambase.data", header = None)
f.head()#Testing
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.64</td>
      <td>0.64</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.778</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.756</td>
      <td>61</td>
      <td>278</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>0.28</td>
      <td>0.50</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.28</td>
      <td>0.21</td>
      <td>0.07</td>
      <td>0.00</td>
      <td>0.94</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.132</td>
      <td>0.0</td>
      <td>0.372</td>
      <td>0.180</td>
      <td>0.048</td>
      <td>5.114</td>
      <td>101</td>
      <td>1028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.71</td>
      <td>0.0</td>
      <td>1.23</td>
      <td>0.19</td>
      <td>0.19</td>
      <td>0.12</td>
      <td>0.64</td>
      <td>0.25</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.143</td>
      <td>0.0</td>
      <td>0.276</td>
      <td>0.184</td>
      <td>0.010</td>
      <td>9.821</td>
      <td>485</td>
      <td>2259</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.137</td>
      <td>0.0</td>
      <td>0.137</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.63</td>
      <td>0.00</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>0.31</td>
      <td>0.63</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.135</td>
      <td>0.0</td>
      <td>0.135</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>3.537</td>
      <td>40</td>
      <td>191</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
random.seed(0)
train, test = train_test_split(f, train_size = 2/3)
test.head()#Testing
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3270</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.63</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1.769</td>
      <td>5</td>
      <td>46</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3464</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.188</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>3.900</td>
      <td>13</td>
      <td>78</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1168</th>
      <td>0.34</td>
      <td>0.00</td>
      <td>0.69</td>
      <td>0.0</td>
      <td>0.17</td>
      <td>0.51</td>
      <td>0.00</td>
      <td>0.51</td>
      <td>0.17</td>
      <td>0.17</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.170</td>
      <td>0.0</td>
      <td>1.275</td>
      <td>0.141</td>
      <td>0.0</td>
      <td>5.598</td>
      <td>78</td>
      <td>711</td>
      <td>1</td>
    </tr>
    <tr>
      <th>427</th>
      <td>0.00</td>
      <td>1.11</td>
      <td>1.11</td>
      <td>0.0</td>
      <td>1.11</td>
      <td>0.00</td>
      <td>2.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.146</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>2.058</td>
      <td>5</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4449</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.62</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.112</td>
      <td>0.0</td>
      <td>0.225</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1.866</td>
      <td>4</td>
      <td>28</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
train.head()#Testing
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.55</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.099</td>
      <td>0.0</td>
      <td>0.893</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>2.122</td>
      <td>16</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>232</th>
      <td>0.16</td>
      <td>0.32</td>
      <td>0.65</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.49</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.773</td>
      <td>0.08</td>
      <td>0.08</td>
      <td>6.586</td>
      <td>132</td>
      <td>955</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3974</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.58</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.817</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.640</td>
      <td>5</td>
      <td>146</td>
      <td>0</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.23</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.0</td>
      <td>0.23</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.043</td>
      <td>0.043</td>
      <td>0.0</td>
      <td>0.175</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.294</td>
      <td>11</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2980</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.111</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>4.446</td>
      <td>29</td>
      <td>209</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
bias_train = np.ones(train.shape[0]).reshape(-1, 1)
bias_test = np.ones(test.shape[0]).reshape(-1, 1)
```


```python
def Zscore(x, dataset):#Re-used from last homework(s)
    for i in range(dataset.shape[1] - 1):
        col = dataset.iloc[:, i].to_numpy()
        temp = (col - np.mean(col))/ (np.std(col, ddof = 1))
        x = np.append(x, temp.reshape(-1, 1), axis = 1)
    return x
```


```python
zscored_train = pd.DataFrame(Zscore(bias_train, train.iloc[:, :-1]))
zscored_test = pd.DataFrame(Zscore(bias_test, test.iloc[:, :-1]))
X = zscored_test.iloc[:, :-1].to_numpy()
Y = test.iloc[:, -1].to_numpy()
```


```python
temp_train = pd.DataFrame(train)
#temp_test = pd.DataFrame(test)
spam = temp_train[temp_train[temp_train.shape[1] - 1] == 1]
non_spam = temp_train[temp_train[temp_train.shape[1] - 1] == 0]
spam.head()#Testing
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>206</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.55</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.099</td>
      <td>0.0</td>
      <td>0.893</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>2.122</td>
      <td>16</td>
      <td>121</td>
      <td>1</td>
    </tr>
    <tr>
      <th>232</th>
      <td>0.16</td>
      <td>0.32</td>
      <td>0.65</td>
      <td>0.0</td>
      <td>0.32</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.773</td>
      <td>0.080</td>
      <td>0.08</td>
      <td>6.586</td>
      <td>132</td>
      <td>955</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.23</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.0</td>
      <td>0.23</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.043</td>
      <td>0.043</td>
      <td>0.0</td>
      <td>0.175</td>
      <td>0.000</td>
      <td>0.00</td>
      <td>1.294</td>
      <td>11</td>
      <td>66</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>0.00</td>
      <td>0.89</td>
      <td>1.14</td>
      <td>0.0</td>
      <td>0.12</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.25</td>
      <td>0.12</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.165</td>
      <td>0.0</td>
      <td>0.371</td>
      <td>0.061</td>
      <td>0.00</td>
      <td>2.878</td>
      <td>84</td>
      <td>475</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1319</th>
      <td>0.00</td>
      <td>0.27</td>
      <td>0.27</td>
      <td>0.0</td>
      <td>1.09</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.82</td>
      <td>0.54</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.085</td>
      <td>0.128</td>
      <td>0.00</td>
      <td>2.484</td>
      <td>20</td>
      <td>164</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
len(temp_train), len(spam), len(non_spam)#Testing
```




    (3067, 1201, 1866)




```python
non_spam.head()#Testing
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>48</th>
      <th>49</th>
      <th>50</th>
      <th>51</th>
      <th>52</th>
      <th>53</th>
      <th>54</th>
      <th>55</th>
      <th>56</th>
      <th>57</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3974</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.58</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.817</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.640</td>
      <td>5</td>
      <td>146</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2980</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.111</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.446</td>
      <td>29</td>
      <td>209</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3232</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.299</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.236</td>
      <td>13</td>
      <td>85</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3941</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.87</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.019</td>
      <td>0.019</td>
      <td>0.019</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.174</td>
      <td>35</td>
      <td>461</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3501</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000</td>
      <td>0.570</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.312</td>
      <td>11</td>
      <td>37</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 58 columns</p>
</div>




```python
def likelihood(val, mean, std):
    x1 = 1 / (std * (2 * math.pi) ** 0.5)
    x2 = np.exp((-1) * ((val - mean) ** 2) / (2 * std ** 2))
    return x1 * x2
```


```python
initprob_spam = spam.shape[0] / train.shape[0]
initprob_nspam = non_spam.shape[0] / train.shape[0]
print(initprob_spam, initprob_nspam)#Testing
spam_mean_li = np.mean(spam)
spam_std_li = np.std(spam)
non_spam_mean_li = np.mean(non_spam)
non_spam_std_li = np.std(non_spam)

#For spam samples
spam_pred = []
tmp_spam = []
for i in X:
    p1 = np.prod([likelihood(i[j], spam_mean_li[j], spam_std_li[j]) for j in range(len(i))])
    spam_pred.append(p1 * initprob_spam)

non_spam_pred = []
#For non_spam samples
for a in X:
    p2 = np.prod([likelihood(a[b], non_spam_mean_li[b], non_spam_std_li[b]) for b in range(len(a))])
    non_spam_pred.append(p2 * initprob_nspam)
```

    0.3915878708835996 0.6084121291164004
    


```python
#This cell is for testing for how many nonzero values are there in both the lists
c1 = 0
for i in range(len(spam_pred)):
    if i != 0:
        c1 += 1
print(len(spam_pred), c1)
c2 = 0
for i in range(len(non_spam_pred)):
    if i != 0:
        c2 += 1
print(len(non_spam_pred), c2)
```

    1534 1533
    1534 1533
    


```python
prediction = []
for i in range(len(spam_pred)):
    if spam_pred[i] < non_spam_pred[i]:
        prediction.append(1)
    else:
        prediction.append(0)
prediction = np.array(prediction)
len(prediction), len(Y)#Testing
```




    (1534, 1534)




```python
prediction[:20], Y[:20]
```




    (array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
     array([0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
           dtype=int64))




```python
tp, tn, fp, fn = 0, 0, 0, 0
for i in range(len(prediction)):
    if not prediction[i] == Y[i]:
        if prediction[i] == 1:
            fp += 1
        else:
            tp += 1
    else:
        if prediction[i] == 1:
            fn += 1
        else:
            tn += 1

#tp, tn, fp, fn
```


```python
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f_measure = 2 * precision * recall / (precision + recall)
```

If you get runtime error here(division by zero), please re-run the code.


```python
print(f'Accuracy is {accuracy * 100:.4}%')
print(f'Precision is {precision * 100:.4}%')
print(f'Recall is {recall * 100:.4}%')
print(f'F_score is {f_measure * 100:.4}%')
```

    Accuracy is 87.09%
    Precision is 84.27%
    Recall is 83.17%
    F_score is 83.72%
    
