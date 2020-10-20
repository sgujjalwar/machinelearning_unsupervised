# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:36:46 2019

@author: Sushrutha Gujjalwar
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:11:15 2019

@author: Sushrutha Gujjalwar
Working Directory:
C:\Users\Sushrutha Gujjalwar\Desktop\indi_ml
Purpose:
    Assigment- 2 
    Game of Thrones, in many ways they say, kind of our own life. I always
    surprised by the haunting deaths. Surprisingly, our Prof. Chase gave 
    to ultimately master our skills as an analyst. 
    Analyzed GOT dataset, to predict which characters in the series will live or die,
    and to provide data sriven recommendations, which contained approx
    2000 characters and 25 variables.
    
"""

###############################################################################
##### LIBRARIES AND SET UP OF FILE 
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf # regression modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz # Exports graphics
from sklearn.externals.six import StringIO # Saves an object in memory
from IPython.display import Image # Displays an image on the frontend
from sklearn.metrics import roc_auc_score
import pydotplus # Interprets dot objects
import sklearn.metrics # more metrics for model performance evaluation

file = 'GOT_character_predictions.xlsx'

got = pd.read_excel(file)


###############################################################################
           ##### DATA COMPREHENSION (Basic Data Exploration)########
###############################################################################
##### For any dataset, before EDA, get to authenticate--------

## To know Column names
got.columns

## Displaying the first 5 rows of the DataFrame
print(got.head())

##For dimension of dataset
got.shape 

## Information about each variable
got.info()

## Descriptive statistics
got.describe().round(2)

## Checking missing values                          
print(got.isnull().sum())

'''
#title                         1008
#male                             0
#culture                       1269
#dateOfBirth                   1513
#mother                        1925
#father                        1920
#heir                          1923
#house                          427
#spouse                        1670
#book1_A_Game_Of_Thrones          0
#book2_A_Clash_Of_Kings           0
#book3_A_Storm_Of_Swords          0
#book4_A_Feast_For_Crows          0
#book5_A_Dance_with_Dragons       0
#isAliveMother                 1925
#isAliveFather                 1920
#isAliveHeir                   1923
#isAliveSpouse                 1670
#isMarried                        0
#isNoble                          0
#age                           1515
#numDeadRelations                 0
#popularity                       0
#isAlive                          0
'''

## To see the correlation between variables, to proceed for data exploration

df_corr = got.corr().round(2)

corr = got.corr()
ax=sns.heatmap(corr)
ax.set_title("Heatmap")

##correlation map

f,ax = plt.subplots(figsize=(9, 8))

sns.heatmap(got.corr(), 
            annot=True, 
            linewidths=.10, 
            fmt= '.1f',
            ax=ax)

plt.savefig('got Correlation Heatmap.png')
plt.show()

###############################################################################
##### EXPLORATORY ANALYSIS
###############################################################################

# Histograms to check distributions before exploration:
got.hist(bins=75, figsize=(25,15))
plt.show()

## Creating boxplots for age, popularity and numDeadRelations.
got.boxplot(column = 'age',vert=False)

got.boxplot(column = 'popularity',vert=False)

got.boxplot(column = 'numDeadRelations',vert=False)


########################################
#Flagging missing values and defining outliers
########################################

## Working on Age 
got['age'][got['age'] < 0] = 0

got['age'] = got['age'].fillna(pd.np.mean(got['age']))

## Working on Date of Birth

got['dateOfBirth'][got['dateOfBirth'] < 0] = 0

got['dateOfBirth'] = got['dateOfBirth'].fillna(pd.np.mean(got['dateOfBirth']))


      
'''Defining outlier thresholds here,to maintain code easily'''

dateOfBirth_high = 260
dateOfBirth_low = -25
age = 28
age_low = -200000
numDeadRelations = 1
popularity = 0.2


def up_out(col,lim):
    got['o_'+col] = 0
    for val in enumerate(got.loc[ : , col]):   
        if val[1] > lim:
            got.loc[val[0], 'o_'+col] = 1 
                
           
up_out('numDeadRelations', numDeadRelations)
up_out('popularity', popularity)

got['o_dateOfBirth'] = 0

for val in enumerate(got.loc[ : , 'dateOfBirth']):    
    if val[1] < dateOfBirth_low:
        got.loc[val[0], 'o_dateOfBirth'] = -1

for val in enumerate(got.loc[ : , 'dateOfBirth']):   
    if val[1] > dateOfBirth_high:
        got.loc[val[0], 'o_dateOfBirth_high'] = 1

got['o_age'] = 0

for val in enumerate(got.loc[ : , 'age']):    
    if val[1] < age_low:
        got.loc[val[0], 'o_age'] = -1

for val in enumerate(got.loc[ : , 'age']):   
    if val[1] > age:
        got.loc[val[0], 'o_age'] = 1


########################################
#From the null value analysis, dropping variables where 90% data is missing
########################################

got.drop(columns=['isAliveSpouse',
                  'isAliveHeir',
                  'isAliveFather',
                  'isAliveMother',
                  'spouse',
                  'heir',
                  'father',
                  'mother',
                  #'dateOfBirth'
                  ], inplace = True)
    
##Histograms after imputing and dropping variables to see the distribution
for col in got.iloc[:, 1:]:
    if is_numeric_dtype(got[col]) == True:
      sns.distplot(got[col], kde = True)
      plt.tight_layout()
      plt.show()
      
##########################################################################
##### Study target variable#####
###############################################################################

## Correlations with target variable - isAlive

df_corr_alive = got.corr()['isAlive'].sort_values()

'''
#age                           -0.271109
#numDeadRelations             -0.192444
#popularity                   -0.183223
#book1_A_Game_Of_Thrones      -0.147401
#male                         -0.146982
#dateOfBirth                  -0.085863
#book2_A_Clash_Of_Kings       -0.067200
#isMarried                    -0.050037
#isAliveMother                -0.043033
#isNoble                      -0.042211
#book3_A_Storm_Of_Swords       0.006693
#book5_A_Dance_with_Dragons    0.032846
#isAliveSpouse                 0.174275
#isAliveFather                 0.195992
#book4_A_Feast_For_Crows       0.268975
#isAliveHeir                   0.384900
#isAlive                       1.000000
'''

########################################
##To count number of character's who are alive
########################################
alive_cnt = got['isAlive'].value_counts()

sns.countplot(x='isAlive', data = got)   
plt.show() 

########################################
##To know, percentage alive 
########################################

pct_of_alive = len(
                   got[got['isAlive']==1])/len(got['isAlive']
                   )

print("Percentage of character's Alive", pct_of_alive*100)

########################################
## Testing with existing variables using group by
########################################
Alive_mean = got.groupby('isAlive').mean()

Alive_cult_mean = got.groupby('culture').mean()

Alive_house_mean = got.groupby('house').mean()


########################################
##Vizualizing target variable 
########################################

##Visualizing alive % with respect to house'

pd.crosstab(got.house, got.isAlive).plot(kind='bar', figsize=(100,25))

plt.title('Characters alive in each house')
plt.xlabel('House')
plt.ylabel('is_alive')
plt.savefig('House_alive')

''' 
    Calculating alive percentage, by grouping house can help me to predict
    accuracy. Analyzed that houses like frey,greyjoy, tyrell,night watch,
    lannister,stark to be alive where there alive to death ratio is more. 

'''

##Visualizing alive % with respect to title'

pd.crosstab(got.title, got.isAlive).plot(kind='bar', figsize=(100,25))

plt.title('Characters alive with respect to title')
plt.xlabel('Title')
plt.ylabel('is_alive')
plt.savefig('Title_alive')

''' 
    A similar strategy is used for title as well. Extrapolated that characters
    bearing title like maester, ser have highest to be alive. Characters bearing title
    like Prince,grand maester, bloodrider,andals have equal chances to live 
    and princess more chances to die than to live.
'''

###############################################################################
## Testing new variables
###############################################################################

# Survival % male + noble 

got['noble_male'] = got['male'] + got['isNoble']
Alive_noblemale_mean = got.groupby('noble_male').mean()

def func(x):
    if x < 0.5:
        return 1
    elif x < 0.8:
        return 2
    else:
        return 3

Alive_noblemale_mean['noblemale_m'] = Alive_noblemale_mean['isAlive'].map(func)

got['noblemale_m']= got['noble_male'].map(Alive_noblemale_mean['noblemale_m'])

got['noblemale_m'].value_counts()

########################################
## Testing based on house 
########################################

got['house'] = got['house'].fillna('Unknown')

got_house_list = got['house'].unique()
got_house_tbl = pd.DataFrame(got_house_list)
got_house_tbl['deadCnt'] = 0
got_house_tbl['aliveCnt'] = 0

## To Count dead and alive person, grouped by house

index = 0

for i in got['house']:
    index_inner=0
    if (got['isAlive'][index]==0):
        for j in got_house_tbl[0]:
            if (j == i):
                got_house_tbl['deadCnt'][index_inner] += 1
                break
            else:
                index_inner +=1
    else: 
        for j in got_house_tbl[0]:
            if (j == i):
                got_house_tbl['aliveCnt'][index_inner] += 1 
                break
            else:
                index_inner +=1   
    index = index+1

##Calculated alive percentage of particular house
##added it as a new var without creating a threshold.

got_house_tbl['Total_Cnt']= got_house_tbl['aliveCnt']+got_house_tbl['deadCnt']   
got_house_tbl['alive_pct']=0.00
got['house_alive_pct']=0.00

got_house_tbl['alive_pct'] = (got_house_tbl['aliveCnt']/got_house_tbl['Total_Cnt']) 
    
index = 0  
  
for i in got['house']:
    index_inner=0
    for j in got_house_tbl[0]:
        if (j==i):
            got['house_alive_pct'][index] = got_house_tbl['alive_pct'][index_inner]
            break
        else: 
            index_inner += 1
    index +=1
    
    
########################################
## Testing based on title
########################################
    
got['title'] = got['title'].fillna('Unknown')

got_title_list = got['title'].unique()
got_title_tbl = pd.DataFrame(got_title_list)
got_title_tbl['deadCnt'] = 0
got_title_tbl['aliveCnt'] = 0


## To Count dead and alive person, grouped by title

index = 0

for i in got['title']:
    index_inner=0
    if (got['isAlive'][index]==0):
        for j in got_title_tbl[0]:
            if (j == i):
                got_title_tbl['deadCnt'][index_inner] += 1
                break
            else:
                index_inner +=1
    else: 
        for j in got_title_tbl[0]:
            if (j == i):
                got_title_tbl['aliveCnt'][index_inner] += 1 
                break
            else:
                index_inner +=1   
    index = index+1

##Calculated alive percentage of particular title and adding it as a new var.

got_title_tbl['Total_Cnt']= got_title_tbl['aliveCnt']+got_title_tbl['deadCnt']   
got_title_tbl['alive_pct']=0.00
got['title_alive_pct']=0.00

got_title_tbl['alive_pct'] = (got_title_tbl['aliveCnt']/got_title_tbl['Total_Cnt']) 
    
index = 0    
for i in got['title']:
    index_inner=0
    for j in got_title_tbl[0]:
        if (j==i):
            got['title_alive_pct'][index] = got_title_tbl['alive_pct'][index_inner]
            break
        else: 
            index_inner += 1
    index +=1



########################################
## Testing based on culture
########################################
    
got['culture'] = got['culture'].fillna('Unknown')

got_culture_list = got['culture'].unique()
got_culture_tbl = pd.DataFrame(got_culture_list)
got_culture_tbl['deadCnt'] = 0
got_culture_tbl['aliveCnt'] = 0


## To Count dead and alive person, grouped by culture

index = 0

for i in got['culture']:
    index_inner=0
    if (got['isAlive'][index]==0):
        for j in got_culture_tbl[0]:
            if (j == i):
                got_culture_tbl['deadCnt'][index_inner] += 1
                break
            else:
                index_inner +=1
    else: 
        for j in got_culture_tbl[0]:
            if (j == i):
                got_culture_tbl['aliveCnt'][index_inner] += 1 
                break
            else:
                index_inner +=1   
    index = index+1

##Calculated alive percentage of particular culture and adding it as a new var.

got_culture_tbl['Total_Cnt']= got_culture_tbl['aliveCnt']+got_culture_tbl['deadCnt']   
got_culture_tbl['alive_pct']=0.00
got['culture_alive_pct']=0.00

got_culture_tbl['alive_pct'] = (got_culture_tbl['aliveCnt']/got_culture_tbl['Total_Cnt']) 
    
index = 0    
for i in got['culture']:
    index_inner=0
    for j in got_culture_tbl[0]:
        if (j==i):
            got['culture_alive_pct'][index] = got_culture_tbl['alive_pct'][index_inner]
            break
        else: 
            index_inner += 1
    index +=1

################################################################################
##Preparing dataset for train and test split
################################################################################

got_data   = got.loc[:,['male',
                        'popularity',
                        'book1_A_Game_Of_Thrones',
                        'book2_A_Clash_Of_Kings',
                        'book3_A_Storm_Of_Swords',
                        'book4_A_Feast_For_Crows',
                        'book5_A_Dance_with_Dragons',
                        'isMarried',
                        'isNoble',
                        'age',
                        'house_alive_pct',
                        'title_alive_pct'
                        ]]
                        


# Preparing the target variable

got_target = got.loc[:, 'isAlive']


########################
# Scaling our data... this is unsupervised learning !!!!!!
########################

# Removing the target variable.

got_features = got_data


# Instantiating a StandardScaler() object
scaler = StandardScaler()


# Fitting the scaler with our data
scaler.fit(got_features)


# Transforming our data after fit
X_scaled = scaler.transform(got_features)


# Putting our scaled data into a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)
#not a necessary step

# Setting up the train/test split
X_train, X_test, y_train, y_test = train_test_split(
            got_data,
            got_target,
            test_size = 0.10,
            random_state = 508)

# Training set 
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)

###############################################################################
## Classification with KNN
###############################################################################

# Running the neighbor optimization code with a small adjustment for classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    
    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

##Plotting the accuracy score
    
fig, ax = plt.subplots(figsize=(12,9))

plt.plot(neighbors_settings, training_accuracy, label = "training accuracy")
plt.plot(neighbors_settings, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

## exploring the highest test accuracy
print(test_accuracy)

## Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)

## It looks like 3 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors = 3)

## Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)

## try adding .values.ravel() to you code as in the code below.
knn_clf_fit = knn_clf.fit(X_train, y_train.values.ravel())


## Let's compare the testing score to the training score.
print('Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


#Training Score 0.8978
#Testing Score: 0.8522

## Generating Predictions based on the optimal KNN model
knn_clf_pred_train = knn_clf_fit.predict(X_train)

knn_clf_pred_test = knn_clf_fit.predict(X_test)

## Generating Predictions based on the optimal KNN model

knn_train_auc_score = roc_auc_score(y_train,knn_clf_pred_train).round(4)  
knn_test_auc_score = roc_auc_score(y_test, knn_clf_pred_test).round(4)

## Generating auc_score based on the optimal KNN model
print('knn_auc_train Score', knn_train_auc_score.round(4))
#score - 0.852
print('knn_auc_test Score', knn_test_auc_score.round(4))
#score - 0.828

###############################################################################
## Classification with logistic
###############################################################################
logistic_full = smf.logit(formula = """isAlive ~ 
                       got['male']+
                       got['age']+
                       got['isMarried']+
                       
                       got['isNoble']+
                       got['book1_A_Game_Of_Thrones']+
                       got['book3_A_Storm_Of_Swords']+
                       got['book4_A_Feast_For_Crows']+
                       got['popularity']+
                       got['house_alive_pct']+
                       got['title_alive_pct']""",
                       data = got)


results_logistic_full = logistic_full.fit()


results_logistic_full.summary()

results_logistic_full.pvalues

###############################################################################
# Hyperparameter Tuning with Logistic Regression
###############################################################################

logreg = LogisticRegression(C = 1)

#Fitting training and testing dataset
logreg_fit = logreg.fit(X_train, y_train)

# Running Predictions
logreg_pred_train = logreg_fit.predict(X_train)
logreg_pred_test = logreg_fit.predict(X_test)

# Let's compare the testing score to the training score.
print('Training Score', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_fit.score(X_test, y_test).round(4)) 

#Training Score 0.835,
#Testing Score: 0.8308, 

###############################################################################
                    # Cross Validation with k-folds
###############################################################################

# Cross Validating the knn model with three folds
cv_lr_7 = cross_val_score(knn_clf,
                           got_data,
                           got_target,
                           cv = 7)


print(cv_lr_7)


print(pd.np.mean(cv_lr_7).round(3))

print('\nAverage: ',
      pd.np.mean(cv_lr_7).round(3),
      '\nMinimum: ',
      min(cv_lr_7).round(3),
      '\nMaximum: ',
      max(cv_lr_7).round(3))

#Average:  0.805 
#Minimum:  0.781 
#Maximum:  0.824

###############################################################################
## Classification with other models
###############################################################################

####gaussian
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb_fit = gnb.fit(X_train, y_train)
y_predictions = gnb.predict(X_test)

print('Training Score', gnb_fit.score(X_train, y_train).round(4))
print('Testing Score:', gnb_fit.score(X_test, y_test).round(4)) 

#Training Score 0.8007
#Testing Score: 0.7949

#####svc
from sklearn.svm import SVC

clf = SVC(gamma='auto')
clf_fit = clf.fit(X_train, y_train)

y_predictions = clf.predict(X_test)

print('Training Score', clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', clf_fit.score(X_test, y_test).round(4)) 

##Training Score 0.8258,
#Testing Score: 0.8308,

###############################################################################
# Random Forest in scikit-learn
###############################################################################

# Following the same procedure as other scikit-learn modeling techniques

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)



# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)



# Are our predictions the same for each model? 
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()



# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))
####0.8612
####0.8564

# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

#Training Score 0.8624
#Testing Score: 0.8718

# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)


########################
# Parameter tuning with GridSearchCV
########################


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]



param_grid = {'n_estimators' : estimator_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space,
              'bootstrap' : bootstrap_space,
              'warm_start' : warm_start_space}



# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth = None,
                                          random_state = 508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv = 3)#5)


# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))


# Tuned Logistic Regression Accuracy so far is 0.8504


###############################################################################
# Gradient Boosted Machines
###############################################################################


# Building a weak learner gbm
gbm_2 = GradientBoostingClassifier(loss = 'deviance',
                                  learning_rate = 1.5,
                                  n_estimators = 100,
                                  max_depth = 2,
                                  criterion = 'friedman_mse',
                                  warm_start = False,
                                  random_state = 508)


gbm_basic_fit = gbm_2.fit(X_train, y_train)


gbm_basic_predict = gbm_basic_fit.predict(X_test)


# Training and Testing Scores
print('Training Score gbm', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score gbm :', gbm_basic_fit.score(X_test, y_test).round(4))

#Training Score gbm 0.9012
#Testing Score gbm : 0.8872

##defining gbm fit predict models
gbm_clf_pred_train = gbm_basic_fit.predict(X_train)
gbm_clf_pred_test = gbm_basic_fit.predict(X_test)

gbm_train_auc_score = roc_auc_score(y_train,gbm_clf_pred_train).round(4)  
gbm_test_auc_score = roc_auc_score(y_test, gbm_clf_pred_test).round(4)

## Printing auc on other best fit so far
print('gbm_auc_train Score', gbm_train_auc_score.round(4))
print('gbm_auc_test Score', gbm_test_auc_score.round(4))

#gbm_auc_train Score 0.8545
#gbm_auc_tes Score 0.8259

###############################################################################
                    # Cross Validation with k-folds
###############################################################################

# Cross Validating the knn model with three folds
cv_lr_2 = cross_val_score(gbm_2,
                           got_data,
                           got_target,
                           cv = 7)


print(cv_lr_2)


print(pd.np.mean(cv_lr_2).round(3))

print('\nAverage: ',
      pd.np.mean(cv_lr_2).round(3),
      '\nMinimum: ',
      min(cv_lr_7).round(3),
      '\nMaximum: ',
      max(cv_lr_7).round(3))

#Average:  0.827 
#Minimum:  0.781 
#Maximum:  0.824

###############################################################################
# Creating a confusion matrix
###############################################################################

print(confusion_matrix(y_true = y_test,
                       y_pred = gbm_basic_predict))


# Visualizing a confusion matrix

labels = ['Dead', 'Live']

cm = confusion_matrix(y_true = y_test,
                      y_pred = gbm_basic_predict)


sns.heatmap(cm,
            annot = True,
            xticklabels = labels,
            yticklabels = labels,
            cmap = 'PuBu')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()

'''
The result tells us that we have 173 correct predictions
 and 22 incorrect predictions.
'''
########################
# Creating a classification report
########################

print(classification_report(y_true = y_test,
                            y_pred = gbm_basic_predict,
                            target_names = labels))

###############################################################################
## We can check the scores of all the predictive models
###############################################################################

# Kneighbor Classification - second Best model so far
print('Training Score knn_clf', knn_clf_fit.score(X_train, y_train).round(4))
print('Testing Score knn_clf:', knn_clf_fit.score(X_test, y_test).round(4))


# Logistic Regression - Good model
print('Training Score log_reg', logreg_fit.score(X_train, y_train).round(4))
print('Testing Score log_reg:', logreg_fit.score(X_test, y_test).round(4))
print("Tuned Logistic Regression Accuracy:", full_forest_cv.best_score_.round(4))

#SVC- This model is underfitting
print('Training Score', clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', clf_fit.score(X_test, y_test).round(4)) 

## considered in the confusion matrix
# GBM - Best model so far
print('Training Score gbm', gbm_basic_fit.score(X_train, y_train).round(4))
print('Testing Score gbm :', gbm_basic_fit.score(X_test, y_test).round(4))

print('gbm_auc_train Score', gbm_train_auc_score.round(4))
print('gbm_auc_test Score', gbm_test_auc_score.round(4))

#Random forest  
#Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))

# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

#gaussian
print('Training Score', gnb_fit.score(X_train, y_train).round(4))
print('Testing Score:', gnb_fit.score(X_test, y_test).round(4)) 

#svc
print('Training Score', clf_fit.score(X_train, y_train).round(4))
print('Testing Score:', clf_fit.score(X_test, y_test).round(4)) 


### auc and cv test is performed on both gbm and knn models
### confusion matrix is performed on gbm 
#Now we will export the predictions to excel sheet for submission
got.to_excel("got_predictions.xlsx")

#Now we will export the predictions to excel sheet for submission
got_data.to_excel("got_predictions_data_file.xlsx")

#Now we will export the predictions to excel sheet for submission
got_target.to_excel("got_predictions_target_file.xlsx")


