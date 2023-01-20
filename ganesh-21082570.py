

# ## Importing the libraries 
# 

# In[1]:


from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# ## Importing dataset 
# 

# In[3]:


data_1=pd.read_csv('8f0c3adc-2577-450a-bdfa-85e2618dd64b_Data.csv')


# In[4]:


data_1.head()


# In[5]:


data_1.info()


# In[6]:


data_1.shape


# In[7]:


data_1.isnull().sum()


# In[8]:


data_1.describe()


# In[9]:


X = data_1
y = data_1['Series Name']


# In[10]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['Series Name'] = le.fit_transform(X['Series Name'])
y = le.transform(y)


# In[11]:


X.info()


# In[12]:


X.head()


# In[13]:


data_1.isnull().sum()


# In[14]:


# Remove rows with null values
data_1.dropna(inplace=True)


# In[15]:


data_1.isnull().sum()


# In[16]:


cols = X.columns


# ## Performing K-Means clustering 

# In[17]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# Generating the following sample data
X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0) 
# Perform the k-means clustering
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
# Plot the data points colored by cluster
plt.scatter(X[:,0], X[:,1], c=pred_y, cmap='viridis')
# Plot the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
# Add title and labels
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Centers and Cluster Membership ')
# Show the plot
plt.show()


# In[18]:


data_1


# In[19]:


data_1.rename(columns={'Country Name': 'Country_Name'}, inplace=True)
data_1.rename(columns={'Series Name': 'Series_Name'}, inplace=True)
data_1.rename(columns={'Series Code': 'Series_Code'}, inplace=True)
data_1.rename(columns={'Country Code': 'Country_Code'}, inplace=True)
data_1.rename(columns={'1990 [YR1990]': '1990'}, inplace=True)
data_1.rename(columns={'2000 [YR2000]': '2000'}, inplace=True)
data_1.rename(columns={'2012 [YR2012]': '2012'}, inplace=True)
data_1.rename(columns={'2013 [YR2013]': '2013'}, inplace=True)
data_1.rename(columns={'2014 [YR2014]': '2014'}, inplace=True)
data_1.rename(columns={'2015 [YR2015]': '2015'}, inplace=True)
data_1.rename(columns={'2016 [YR2016]': '2016'}, inplace=True)
data_1.rename(columns={'2017 [YR2017]': '2017'}, inplace=True)
data_1.rename(columns={'2018 [YR2018]': '2018'}, inplace=True)
data_1.rename(columns={'2019 [YR2019]': '2019'}, inplace=True)
data_1.rename(columns={'2020 [YR2020]': '2020'}, inplace=True)
data_1.rename(columns={'2021 [YR2021]': '2021'}, inplace=True)


# In[20]:


data_1.info()


# ## Pie Chart 

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt
# Sample data
data = [3870760, 3717750, 3744230, 3481190, 3426020, 3718370, 3588950, 3624770, 3557750]
keys=['2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']
# define Seaborn color palette for the utilisation
palette_color = sns.color_palette('mako')
# Set the size of the figure
plt.figure(figsize=(10, 5))  
# plotting data on the chart
plt.pie(data, labels=keys, colors=palette_color,
        autopct='%.0f%%')
# displaying the following chart
plt.show()


# ## Bar chart

# In[22]:


# data for the bar plot
years = [2012, 2013, 2014, 2015, 2016]
values = [130, 125, 120, 110, 100]
# Set the figure size
plt.figure(figsize=(10, 5))
# create the bar plot
plt.bar(years, values, color = 'red')
# add axis labels and a title
plt.xlabel('Years')
plt.ylabel('Values')
plt.title('Values by 5-year Intervals')
# show the plot
plt.show()


# In[23]:


# Create a dataframe with country and population data
data_2 = {'years': ['2012', '2013', '2014', '2015', '2016'],
        'values': [130, 125, 120, 110, 100]}

df= pd.DataFrame(data_2)
# Set the figure size
plt.figure(figsize=(10, 5))
# Create a bar chart with log scale y-axis
plt.bar(df['years'], df['values'], color = 'cyan')
plt.yscale("log")
plt.show()


# In[24]:


df['values'] = df['values']/df['values'].mean()*100
# Set the figure size
plt.figure(figsize=(10, 5))
plt.bar(df['years'], df['values'], color = 'violet')
plt.show()


# In[25]:


data_1


# In[26]:


# Checking for null values
print(data_1.isnull().sum())


# In[27]:


data_1 = data_1.fillna(0)


# In[28]:


data_1.head()


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# In[30]:


# Create a list of columns to be dropped
columns_to_drop = ['Series_Name', 'Series_Code', 'Country_Name', 'Country_Code']
# Drop the columns
data_3 = data_1.drop(columns_to_drop, axis=1)


# In[31]:


data_3.info()


# In[32]:


data_3.dtypes


# In[33]:


data_3 = data_3.replace(['..', 'nan'], [0, 0])


# In[34]:


data_3 = data_3.fillna(0)


# In[35]:


data_3.info()


# In[36]:


data_3


# ## Linear Regression

# In[37]:


X = data_3[['1990', '2000', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']]
y = data_3['2014']


# In[38]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a linear regression model
model = LinearRegression()
# Fit the model to the training data
model.fit(X_train, y_train)
# Predict the output for the test data
y_pred = model.predict(X_test)


# In[39]:


# Print the coefficients
print('Coefficients: ', model.coef_)
print('Intercept: ', model.intercept_)

# Evaluate the model using R^2 score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2 score : ", r2)


# ## Random Forest Regressor

# In[40]:


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a random forest regressor model
rf = RandomForestRegressor(n_estimators=100, random_state=0)
# Evaluate the model using R^2 score
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("R^2 score : ", r2)


# In[ ]:




