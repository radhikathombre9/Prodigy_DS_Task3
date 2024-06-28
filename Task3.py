# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 14:40:26 2024

@author: Radhika
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import warnings
warnings.filterwarnings('ignore')

Bank = pd.read_csv(r"D:\Nikhil Analytics\Prodigy Infotech\bank.csv",delimiter= ';')


Bank.head()

Bank.shape

Bank.columns

Bank.info()

Bank.describe()

Bank.isnull().sum()


# =============================================================================
# EXPLORATORY DATA ANALYSIS
# =============================================================================


# Histogram for Numerical columns

Bank.hist(bins=20, figsize=(14, 10), edgecolor='black')
plt.suptitle('Histograms of Numerical Columns')
plt.show()



# Marital Distribution
marital = Bank['marital'].value_counts().reset_index()
marital.columns = ['Marital Status', 'Count']
plt.figure(figsize=(7, 5))
plt.title('Marital Distribution')
plt.pie(marital['Count'], labels=marital['Marital Status'], autopct='%1.2f%%')
plt.show()

# Loan _Distribution
loan=Bank['loan'].value_counts().reset_index()
loan.columns = ['loan Status','Count']
plt.figure(figsize=(7,5))
color=['#c6538c','#00b3b3']
plt.pie(loan['Count'], labels=loan['loan Status'],autopct='%1.2f%%',colors=color)
plt.show()


# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(Bank['age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Balance Distribution
plt.figure(figsize=(10, 6))
sns.histplot(Bank['balance'], bins=30, kde=True)
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

# Scatter plot for Age vs Balance
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='balance', data=Bank)
plt.title('Age vs. Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()


# Education Wise Distribution
educ=Bank['education'].value_counts().reset_index()
educ.columns=['Count','education status']
plt.figure(figsize=(7,5))
color=['#fc0872','#82d40f','#ffd503','#1fcbff']
plt.title('Eduction wise Distribution')
plt.bar(educ['Count'],educ['education status'],color=color)
plt.show()



# Month wise loans
mon_loan=Bank.groupby(['month','loan']).size().reset_index()
mon_loan.rename(columns={0:'Count'},inplace=True)
mon_loan_yes=mon_loan[mon_loan['loan']=='yes']
mon_loan_no=mon_loan[mon_loan['loan']=='no']
plt.figure(figsize=(7,5))
plt.title('Month Wise Loans')
plt.xlabel('Month')
plt.ylabel('Count')
plt.bar(mon_loan_yes['month'],mon_loan_yes['Count'],label='yes',color='#fc0872')
plt.bar(mon_loan_no['month'],mon_loan_no['Count'],bottom=mon_loan_yes['Count'],label='no',color='#82d40f')
plt.legend()
plt.show()


# Correlation Matrix

numeric_cols = Bank.select_dtypes(include='number')
corr_matrix = numeric_cols.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# =============================================================================
# Pre Processing
# =============================================================================

Bank.drop(['pdays','previous','poutcome','y'],axis=1,inplace=True)


encoder=LabelEncoder()
Bank['job']=encoder.fit_transform(Bank['job'])
Bank['marital']=encoder.fit_transform(Bank['marital'])
Bank['education']=encoder.fit_transform(Bank['education'])
Bank['default']=encoder.fit_transform(Bank['default'])
Bank['housing']=encoder.fit_transform(Bank['housing'])
Bank['loan']=encoder.fit_transform(Bank['loan'])
Bank['contact']=encoder.fit_transform(Bank['contact'])
Bank['month']=encoder.fit_transform(Bank['month'])




x=Bank.drop('default',axis=1)
y=Bank['default']
     


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=45)

x_train.shape,x_test.shape


y_train.shape,y_test.shape     


# =============================================================================
# Model Building
# =============================================================================

model=DecisionTreeClassifier()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("Accuracy Score:",accuracy_score(y_test,y_pred))


# Checking Overfitting anfd Under Fitting
x_pred=model.predict(x_train)
print("Accuracy Score:",accuracy_score(y_train,x_pred)) # there is no overfitting and underfitting

conf_mat=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_mat,annot=True,square=True,fmt='d',cmap='PuRd')


print(classification_report(y_test,y_pred))

# =============================================================================
# output - DecisionTreeClassifier Model performing really well, scoring 1.0 on the 
# training data and 96% on the test data. This means it's good at making accurate predictions
# =============================================================================





























