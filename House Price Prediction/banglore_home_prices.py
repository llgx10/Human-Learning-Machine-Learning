#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)


# <h2 style='color:blue'>Data Load: Load banglore home prices into a dataframe</h2>

# In[74]:


df1 = pd.read_csv("HPPM.csv")
df1.head()


# In[75]:


df1.shape


# In[76]:


df1.columns


# In[77]:


df1['area_type'].unique()


# In[78]:


df1['area_type'].value_counts()


# **Drop features that are not required to build our model**

# In[79]:


df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')
df2.shape


# <h2 style='color:blue'>Data Cleaning: Handle NA values</h2>

# In[80]:


df2.isnull().sum()


# In[81]:


df2.shape


# In[82]:


df3 = df2.dropna()
df3.isnull().sum()


# In[83]:


df3.shape


# <h2 style='color:blue'>Feature Engineering</h2>

# **Add new feature(integer) for bhk (Bedrooms Hall Kitchen)**

# In[84]:


df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
df3.bhk.unique()


# **Explore total_sqft feature**

# In[85]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[86]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[87]:


def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None   


# In[88]:


df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
df4 = df4[df4.total_sqft.notnull()]
df4.head(2)


# **For below row, it shows total_sqft as 2475 which is an average of the range 2100-2850**

# In[89]:


df4.loc[30]


# **Add new feature called price per square feet**

# In[90]:


df5 = df4.copy()
df5['price_per_sqft'] = df5['price']*100000/df5['total_sqft']
df5.head()


# In[91]:


df5_stats = df5['price_per_sqft'].describe()
df5_stats


# **Examine locations which is a categorical variable.**

# In[92]:


df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5['location'].value_counts(ascending=False)
location_stats


# In[93]:


location_stats.values.sum()


# In[94]:


len(location_stats[location_stats>10])


# In[95]:


len(location_stats)


# In[96]:


len(location_stats[location_stats<=10])


# In[97]:


location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10


# In[98]:


len(df5.location.unique())


# In[99]:


df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[100]:


df5.head(10)


# <h2 style="color:blue">Outlier Removal Using Business Logic</h2>

# In[101]:


df5[df5.total_sqft/df5.bhk<300].head()


# In[102]:


df5.shape


# In[103]:


df6 = df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape


# <h2 style='color:blue'>Outlier Removal Using Standard Deviation and Mean</h2>

# In[104]:


df6.price_per_sqft.describe()


# In[105]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
df7.shape


# **Let's check if for a given location how does the 2 BHK and 3 BHK property prices look like**

# In[106]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[107]:


plot_scatter_chart(df7,"Hebbal")


# In[131]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
# df8 = df7.copy()
df8.shape


# **Plot same scatter chart again to visualize price_per_sqft for 2 BHK and 3 BHK properties**

# In[132]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[133]:


plot_scatter_chart(df8,"Hebbal")


# In[134]:


import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")


# <h2 style='color:blue'>Outlier Removal Using Bathrooms Feature</h2>

# In[135]:


df8.bath.unique()


# In[136]:


plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[137]:


df8[df8.bath>10]


# **It is unusual to have 2 more bathrooms than number of bedrooms in a home**

# In[138]:


df8[df8.bath>df8.bhk+2]


# In[139]:


df9 = df8[df8.bath<df8.bhk+2]
df9.shape


# In[140]:


df9.head(2)


# In[141]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# <h2 style='color:blue'>Use One Hot Encoding For Location</h2>

# In[142]:


dummies = pd.get_dummies(df10.location)
dummies.head(3)


# In[143]:


df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
df11.head()


# In[144]:


df12 = df11.drop('location',axis='columns')
df12.head(2)


# <h2 style='color:blue'>Build a Model Now...</h2>

# In[145]:


df12.shape


# In[146]:


X = df12.drop(['price'],axis='columns')
X.head(3)


# In[147]:


X.shape


# In[148]:


y = df12.price
y.head(3)


# In[149]:


len(y)


# In[174]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15,random_state=14)


# In[175]:


from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)


# In[228]:


y_pred5=lr_clf.predict(X_test)
df1= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred5})
df1


# <h2 style='color:blue'>Use K Fold cross validation to measure accuracy of our LinearRegression model</h2>

# In[179]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

am = ShuffleSplit(n_splits=5, test_size=0.15, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=am)


# <h2 style='color:blue'>Find best model using GridSearchCV</h2>

# In[184]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.15, random_state=10)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score'])

find_best_model_using_gridsearchcv(X,y)


# In[185]:


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


# In[187]:


sc_X = StandardScaler()
X_train1= sc_X.fit_transform(X_train)
X_test1 = sc_X.fit_transform(X_test)


# In[188]:


regressor1 = SVR(kernel='poly')
regressor1.fit(X_train1, y_train)


# In[189]:


y_pred1 = regressor1.predict(X_test1)


# In[190]:


df1= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred1})
df1


# In[191]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: " + str(mean_absolute_error(y_pred1, y_test)/10))
print("MSE: " + str(mean_squared_error(y_pred1, y_test)/10))


# In[192]:


regressor1.score(X_test1,y_test)


# In[193]:


plt.scatter(y_test,y_pred1)
plt.xlabel('Actual')
plt.ylabel('Predicted')


# In[194]:


from sklearn.ensemble import RandomForestRegressor

regressor2 = RandomForestRegressor(n_estimators=20, random_state=0)
regressor2.fit(X_train, y_train)
y_pred2 = regressor2.predict(X_test)


# In[195]:


df2= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred2})
df2


# In[196]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: " + str(mean_absolute_error(y_pred2, y_test)/10))
print("MSE: " + str(mean_squared_error(y_pred2, y_test)/10))


# In[197]:


regressor2.score(X_test,y_test)


# In[198]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# In[199]:


gbm = GradientBoostingRegressor()


# In[200]:


gbm.fit(X_train, y_train)


# In[201]:


y_pred4=gbm.predict(X_test)


# In[202]:


df4= pd.DataFrame({'Actual': y_test, 'Predicted': y_pred4})
df4


# In[203]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
print("MAE: " + str(mean_absolute_error(y_pred4, y_test)/10))
print("MSE: " + str(mean_squared_error(y_pred4, y_test)/10))


# In[204]:


gbm.score(X_test,y_test)


# **Based on above results we can say that Gradient boosting gives the best score. Hence we will use that.**

# <h2 style='color:blue'>Test the model for few properties</h2>

# In[221]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return gbm.predict([x])[0]


# In[222]:


df10.head()


# In[229]:


List = df10["location"].unique()


# In[224]:


predict_price('Bisuvanahalli',1000, 3, 3)


# In[ ]:





# In[225]:


predict_price('1st Phase JP Nagar',1000, 3, 3)


# In[226]:


predict_price('Indira Nagar',1000, 2, 2)


# In[227]:


predict_price('Indira Nagar',1000, 3, 3)


# In[ ]:




