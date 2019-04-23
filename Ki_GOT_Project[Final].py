# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 00:37:56 2019

@author: Kisuc Kim

Purpose:
    Machine Learning individual assignment
"""

########################
# Libraries and Dataset
########################

# Loading libraries
import pandas as pd  # pandas
import seaborn as sns  # seaborn
import numpy as np  # numpy
import matplotlib.pyplot as plt  # matplot
import sklearn.metrics  # sklearn
from sklearn.model_selection import train_test_split  # train/test split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.neighbors import KNeighborsClassifier  # KNN Classification
from sklearn.model_selection import cross_val_score  # Cross Validation
from sklearn.tree import export_graphviz  # Exports graphics
from sklearn.externals.six import StringIO  # Saves an object in memory
from sklearn.tree import DecisionTreeClassifier  # Classification trees
from sklearn.model_selection import GridSearchCV  # GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# GradientBoostingClassifier
from sklearn.metrics import roc_auc_score  # roc_auc_score
from sklearn.metrics import confusion_matrix  # confusion_matrix
from sklearn.metrics import classification_report  # classification_report
import statsmodels.api as sm  # statsmodels.api
import statsmodels.formula.api as smf  # statsmodels.formula
from IPython.display import Image  # Displays an image on the frontend
import pydotplus  # Interprets dot objects

# Importing the dataset
got_df = pd.read_excel('GOT_character_predictions.xlsx')


###############################################################################
# Dataset Summary
###############################################################################

# General Summary of the Dataset
got_df.shape

got_df.info()

got_df.head(n=5)

got_df.describe().round(2)

got_df.corr().round(3)

##############################################################################
# Data Preparation
##############################################################################

# Drop unnecessary variables ('S.No','name')
got_df_data = got_df.drop(['S.No', 'name'], axis=1)

########################
# Impute missing values
########################

print(
    got_df_data
    .isnull()
    .sum()
)


for col in got_df_data:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if got_df_data[col].isnull().any():
        got_df_data['m_' + col] = got_df_data[col].isnull().astype(int)

########################
# Working with Categorical Variables
########################

# Check the count for non-numerical variables
print('title:', got_df_data['title'].count())
print('culture:', got_df_data['culture'].count())
print('mother:', got_df_data['mother'].count())
print('father:', got_df_data['father'].count())
print('heir:', got_df_data['heir'].count())
print('house:', got_df_data['house'].count())
print('spouse:', got_df_data['spouse'].count())


# Detailed-check the count for non-numerical variables with some values.
# More than 200 (10% of total)
print(got_df_data['title'].value_counts(dropna=True))
print('title Total # :', got_df_data['title'].count())
print('title Unique # :', got_df_data['title'].value_counts().count())

print(got_df_data['culture'].value_counts(dropna=True))
print('culture Total # :', got_df_data['culture'].count())
print('culture Unique # :', got_df_data['culture'].value_counts().count())

print(got_df_data['house'].value_counts(dropna=True))
print('house Total # :', got_df_data['house'].count())
print('house Unique # :', got_df_data['house'].value_counts().count())

print(got_df_data['spouse'].value_counts(dropna=True))
print('spouse Total # :', got_df_data['spouse'].count())
print('spouse Unique # :', got_df_data['spouse'].value_counts().count())


# Drop no longer necessary variables after converted to binary variables
# (misssing values)
got_df_data0 = got_df_data.drop(['mother', 'father', 'heir', 'spouse'], axis=1)


# Numbering for unique features in remaining non-numerical variables
got_df_data0['N_title'] = got_df_data0['title'].fillna(0)
labels1 = got_df_data0['title'].astype('category').cat.categories.tolist()
replace_map_comp1 = {
    'N_title': {
        k: v for k,
        v in zip(
            labels1,
            list(
                range(
                    1,
                    len(labels1) +
                    1)))}}
got_df_data0.replace(replace_map_comp1, inplace=True)

got_df_data0['N_culture'] = got_df_data0['culture'].fillna(0)
labels2 = got_df_data0['culture'].astype('category').cat.categories.tolist()
replace_map_comp2 = {
    'N_culture': {
        k: v for k,
        v in zip(
            labels2,
            list(
                range(
                    1,
                    len(labels2) +
                    1)))}}
got_df_data0.replace(replace_map_comp2, inplace=True)

got_df_data0['N_house'] = got_df_data0['house'].fillna(0)
labels3 = got_df_data0['house'].astype('category').cat.categories.tolist()
replace_map_comp3 = {
    'N_house': {
        k: v for k,
        v in zip(
            labels3,
            list(
                range(
                    1,
                    len(labels3) +
                    1)))}}
got_df_data0.replace(replace_map_comp3, inplace=True)


# Creating a new variable (Sum of the count appeared in book 1 to 5)
got_df_data0['Book_Appearance'] = 0
got_df_data0['Book_Appearance'] = got_df_data0['book1_A_Game_Of_Thrones']
+ got_df_data0['book2_A_Clash_Of_Kings']
+ got_df_data0['book3_A_Storm_Of_Swords']
+ got_df_data0['book4_A_Feast_For_Crows']
+ got_df_data0['book5_A_Dance_with_Dragons']

got_df_data0['Multi_Books'] = 0

for i in range(len(got_df_data0)):
    if got_df_data0.loc[i, 'Book_Appearance'] >= 2:
        got_df_data0.loc[i, 'Multi_Books'] = 1

# Creating a new variable (Sum of alive family)
got_df_data0['AliveFamily'] = 0
got_df_data0['AliveFamily'] = got_df_data0['isAliveMother'].fillna(0)
+ got_df_data0['isAliveFather'].fillna(0)
+ got_df_data0['isAliveHeir'].fillna(0)
got_df_data0['isAliveSpouse'].fillna(0)

got_df_data0['isAliveFamily'] = 0

for i in range(len(got_df_data0)):
    if got_df_data0.loc[i, 'AliveFamily'] >= 1:
        got_df_data0.loc[i, 'isAliveFamily'] = 1


# Drop no longer necessary variables (Categorical strings)
got_df_data1 = got_df_data0.drop(['title', 'culture', 'house'], axis=1)


########################
# Flagging outliers
########################

# Checking outliers with boxplot
sns.boxplot(got_df_data1['dateOfBirth'])
plt.show()
plt.clf()
sns.boxplot(got_df_data1['age'])
plt.show()
plt.clf()
sns.boxplot(got_df_data1['popularity'])
plt.show()
plt.clf()
sns.boxplot(got_df_data1['Book_Appearance'])
plt.show()
plt.clf()

# Setting benchmarks for outliers

dob_hi = 300
dob_lo = 0
age_lo = 0

Q1 = got_df_data1['popularity'].quantile(0.25)
Q3 = got_df_data1['popularity'].quantile(0.75)
IQR = Q3 - Q1
pop_hi = Q3 + 1.5 * IQR


# 'dateOfBirth' outliers
got_df_data1['out_dateOfBirth'] = 0

for val in enumerate(got_df_data1.loc[:, 'dateOfBirth']):

    if val[1] > dob_hi:
        got_df_data1.loc[val[0], 'out_dateOfBirth'] = 1


for val in enumerate(got_df_data1.loc[:, 'dateOfBirth']):

    if val[1] < dob_lo:
        got_df_data1.loc[val[0], 'out_dateOfBirth'] = 1


# 'age' outliers
got_df_data1['out_age'] = 0

for val in enumerate(got_df_data1.loc[:, 'age']):

    if val[1] < age_lo:
        got_df_data1.loc[val[0], 'out_age'] = 1


# 'popularity' outliers
got_df_data1['out_popularity'] = 0

for val in enumerate(got_df_data1.loc[:, 'popularity']):

    if val[1] > pop_hi:
        got_df_data1.loc[val[0], 'out_popularity'] = 1

# Set dictionary to handle weired values
weird_births = {-28: None,
                -25: None,
                298299: None,
                278279: None}

weird_ages = {-298001: None,
              -277980: None}

# Weired values converted to None
got_df_data1['dateOfBirth'].replace(weird_births, inplace=True)
got_df_data1['age'].replace(weird_ages, inplace=True)

# Reimpute missing values
for col in got_df_data1:

    """ Create columns that are 0s if a value was not missing and 1 if
    a value is missing. """

    if got_df_data1[col].isnull().any():
        got_df_data1['m_' + col] = got_df_data1[col].isnull().astype(int)

# Check data distribution for age (Fill with median)
sns.boxplot(got_df_data1['age'])
plt.show()
plt.clf()

got_df_data1['age'] = got_df_data1['age'].fillna(got_df_data1['age'].median())

# Check data distribution for age (Fill with 0 due to unknown measurement)
sns.boxplot(got_df_data1['dateOfBirth'])
plt.show()
plt.clf()

got_df_data1['dateOfBirth'] = got_df_data1['dateOfBirth'].fillna(0)

# Creating a new variable filled 0 (Binary variables whether alive or not
# with missing values)
got_df_data1['filled_isAliveMother'] = 0
got_df_data1['filled_isAliveMother'] = got_df_data1['isAliveMother'].fillna(0)

got_df_data1['filled_isAliveFather'] = 0
got_df_data1['filled_isAliveFather'] = got_df_data1['isAliveFather'].fillna(0)

got_df_data1['filled_isAliveHeir'] = 0
got_df_data1['filled_isAliveHeir'] = got_df_data1['isAliveHeir'].fillna(0)

got_df_data1['filled_isAliveSpouse'] = 0
got_df_data1['filled_isAliveSpouse'] = got_df_data1['isAliveSpouse'].fillna(0)


# Drop no longer necessary variables (with missing values)
got_df_data2 = got_df_data1.drop(['isAliveMother',
                                  'isAliveFather',
                                  'isAliveHeir',
                                  'isAliveSpouse'
                                  ], axis=1)


# Final missing value checking
print(
    got_df_data2
    .isnull()
    .sum()
)


##############################################################################
# Correlation Analysis
##############################################################################


# Using correlation to identify which variables we should consider in our
# Model.
df_corr = got_df_data2.corr().round(2)


print(df_corr)

########################
# Correlation Heatmap
########################

# Using palplot to view a color scheme
sns.palplot(sns.color_palette('coolwarm', 12))

fig, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(df_corr,
            cmap='coolwarm',
            annot=True,
            linewidths=0.1)

plt.show()


###############################################################################
# Developing a Classification Base
###############################################################################

############
# # Preparation
############

# Preparing a DataFrame witl all variables
got_df_data3 = got_df_data2.drop(['isAlive'], axis=1)

got_df_target = got_df_data2.loc[:, 'isAlive']


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(got_df_data3,
                                                    got_df_target,
                                                    test_size=0.10,
                                                    random_state=508,
                                                    stratify=got_df_target)


# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# Original Value Counts
got_df_target.value_counts()
got_df_target.sum() / got_df_target.count()


# Training set value counts
y_train.value_counts()
y_train.sum() / y_train.count()


# Testing set value counts
y_test.value_counts()
y_test.sum() / y_test.count()


# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
got_train = pd.concat([X_train, y_train], axis=1)

# Inputting hyperparameter arguments
logreg_1 = LogisticRegression(random_state=508)


logreg_1_fit = logreg_1.fit(X_train, y_train)


logreg_1_pred_train = logreg_1_fit.predict(X_train)
logreg_1_pred_test = logreg_1_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_1_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_1_fit.score(X_test, y_test).round(4))


# AUC:
print('Training AUC Score', roc_auc_score(
    y_train, logreg_1_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, logreg_1_pred_test).round(4))


########################
# Logistic Regression Modeling
########################

# Biserial point correlations
df_corr = got_df_data2.corr().round(2)
df_corr.loc['isAlive'].sort_values(ascending=False)


# Modeling based on the most correlated explanatory variable
logistic_small = smf.logit(
    formula="""isAlive ~ book4_A_Feast_For_Crows""",
    data=got_train)


results_logistic = logistic_small.fit()


results_logistic.summary()


###############################################################################
# WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK #
###############################################################################

# Working full model (LinAlgError: Singular matrix)

#logistic_work = smf.logit(formula="""isAlive ~ male +
#                                               dateOfBirth +
#                                               book1_A_Game_Of_Thrones +
#                                               book2_A_Clash_Of_Kings +
#                                               book3_A_Storm_Of_Swords +
#                                               book4_A_Feast_For_Crows +
#                                               book5_A_Dance_with_Dragons +
#                                               isMarried +
#                                               isNoble +
#                                               age +                                              
#                                               numDeadRelations +
#                                               popularity +
#                                               m_title +
#                                               m_culture +
#                                               m_dateOfBirth +
#                                               m_mother +
#                                               m_father +
#                                               m_heir +
#                                               m_house +
#                                               m_spouse +
#                                               m_isAliveMother +
#                                               m_isAliveFather +
#                                               m_isAliveHeir +
#                                               m_isAliveSpouse +
#                                               m_age +
#                                               N_title +
#                                               N_culture +
#                                               N_house +
#                                               Book_Appearance +
#                                               Multi_Books +
#                                               AliveFamily +
#                                               isAliveFamily +
#                                               out_dateOfBirth +
#                                               out_age +
#                                               out_popularity +
#                                               filled_isAliveMother +
#                                               filled_isAliveFather +
#                                               filled_isAliveHeir +
#                                               filled_isAliveSpouse
#                                               """,
#                          data=got_train)

#results_logistic_work = logistic_work.fit()


#results_logistic_work.summary()

###############################################################################
# WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK # WON'T WORK #
###############################################################################


# Significant model (Manually selelected based on assumptions)
logistic_sigf = smf.logit(formula="""isAlive ~ male +
                                                age +
                                                numDeadRelations +
                                                m_culture +
                                                N_culture +
                                                N_house +
                                                Book_Appearance +
                                                out_popularity
                                                """,data=got_train)


results_logistic_sigf = logistic_sigf.fit()


results_logistic_sigf.summary()

# Preparing a DataFrame with significant variables
got_df_data_final1 = got_df_data2.loc[:, ['male',
                                          'age',
                                          'numDeadRelations',
                                          'm_culture',
                                          'N_culture',
                                          'N_house',
                                          'Book_Appearance',
                                          'out_popularity']]

got_df_target_final1 = got_df_data2.loc[:, 'isAlive']


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(got_df_data_final1,
                                                    got_df_target_final1,
                                                    test_size=0.10,
                                                    random_state=508,
                                                    stratify=got_df_target_final1)


# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# Original Value Counts
got_df_target_final1.value_counts()
got_df_target_final1.sum() / got_df_target_final1.count()


# Training set value counts
y_train.value_counts()
y_train.sum() / y_train.count()


# Testing set value counts
y_test.value_counts()
y_test.sum() / y_test.count()


# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
got_train = pd.concat([X_train, y_train], axis=1)

# Inputting hyperparameter arguments
logreg_2 = LogisticRegression(C=1)


logreg_2_fit = logreg_2.fit(X_train, y_train)


logreg_2_pred_train = logreg_2_fit.predict(X_train)
logreg_2_pred_test = logreg_2_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_2_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_2_fit.score(X_test, y_test).round(4))


# Preparing the FINAL DataFrame with significant variables
got_df_data_final2 = got_df_data2.loc[:, ['male',
                                          'age',
                                          'numDeadRelations',
                                          'm_culture',
                                          'N_culture',
                                          'Book_Appearance',
                                          'out_popularity']]


got_df_target_final2 = got_df_data2.loc[:, 'isAlive']


# train_test_split
X_train, X_test, y_train, y_test = train_test_split(got_df_data_final2,
                                                    got_df_target_final2,
                                                    test_size=0.10,
                                                    random_state=508,
                                                    stratify=got_df_target_final2)


# Training set
print(X_train.shape)
print(y_train.shape)

# Testing set
print(X_test.shape)
print(y_test.shape)


# Original Value Counts
got_df_target_final2.value_counts()
got_df_target_final2.sum() / got_df_target_final2.count()


# Training set value counts
y_train.value_counts()
y_train.sum() / y_train.count()


# Testing set value counts
y_test.value_counts()
y_test.sum() / y_test.count()


# We need to merge our X_train and y_train sets so that they can be
# used in statsmodels
got_train = pd.concat([X_train, y_train], axis=1)

# Inputting hyperparameter arguments
logreg_3 = LogisticRegression(random_state=508)


logreg_3_fit = logreg_3.fit(X_train, y_train)


logreg_3_pred_train = logreg_3_fit.predict(X_train)
logreg_3_pred_test = logreg_3_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_3_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_3_fit.score(X_test, y_test).round(4))


# AUC:
print('Training AUC Score', roc_auc_score(
    y_train, logreg_3_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(y_test, logreg_3_pred_test).round(4))


# Creating a hyperparameter grid
C_space = pd.np.arange(0.1, 10, 0.1)
solver_space = ['newton-cg', 'lbfgs']

param_grid = {'C': C_space,
              'solver': solver_space}


# Building the model object one more time
logreg_object = LogisticRegression(random_state=508)


# Creating a GridSearchCV object
logreg_grid = GridSearchCV(logreg_object,
                           param_grid,
                           cv=3,
                           scoring='roc_auc',
                           return_train_score=False)


# Fit it to the training data
logreg_grid.fit(X_train, y_train)

print("Tuned Logistic Regression Parameter:", logreg_grid.best_params_)
print("Tuned Logistic Regression Accuracy:", logreg_grid.best_score_.round(4))

# Inputting hyperparameter arguments
logreg_optimal = LogisticRegression(C=0.3,
                                    solver='lbfgs',
                                    random_state=508)


logreg_optimal_fit = logreg_optimal.fit(X_train, y_train)


logreg_optimal_pred_train = logreg_optimal_fit.predict(X_train)
logreg_optimal_pred_test = logreg_optimal_fit.predict(X_test)


# Let's compare the testing score to the training score.
print(
    'LOG Training Score',
    logreg_optimal_fit.score(
        X_train,
        y_train).round(4))
print('LOG Testing Score:', logreg_optimal_fit.score(X_test, y_test).round(4))


###############################################################################
# Classification Base with KNN
###############################################################################


########################
# Developing a Classification Base with KNN
########################


# Running the neighbor optimization code with a small adjustment for
# classification
training_accuracy = []
test_accuracy = []

neighbors_settings = range(1, 51)


for n_neighbors in neighbors_settings:
    # build the model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train.values.ravel())

    # record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))

    # record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))


fig, ax = plt.subplots(figsize=(12, 9))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()


# Looking for the highest test accuracy
print(test_accuracy)


# Printing highest test accuracy
print(test_accuracy.index(max(test_accuracy)) + 1)


# It looks like 12 neighbors is the most accurate
knn_clf = KNeighborsClassifier(n_neighbors=14)


# Fitting and predicting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)
knn_clf_fit_pred_test = knn_clf_fit.predict(X_test)
knn_clf_fit_pred_train = knn_clf_fit.predict(X_train)

# Let's compare the testing score to the training score.
print('KNN Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('KNN Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))


###############################################################################
# Building Classification Trees
###############################################################################

c_tree = DecisionTreeClassifier(random_state=508)
c_tree_fit = c_tree.fit(X_train, y_train)


print('Training Score', c_tree_fit.score(X_train, y_train).round(4))
print('Testing Score:', c_tree_fit.score(X_test, y_test).round(4))


# Visualizing the tree
dot_data = StringIO()


export_graphviz(decision_tree=c_tree_fit,
                out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=got_df_data_final2.columns)


graph1 = pydotplus.graph_from_dot_data(dot_data.getvalue())


Image(graph1.create_png(),
      height=500,
      width=800)


###############################################################################
# Hyperparameter Tuning with GridSearchCV
###############################################################################


########################
# Optimizing for hyperparameters
########################


# Creating a hyperparameter grid
depth_space = pd.np.arange(1, 20)
leaf_space = pd.np.arange(1, 500)

param_grid = {'max_depth': depth_space,
              'min_samples_leaf': leaf_space}


# Building the model object one more time
c_tree_2_hp = DecisionTreeClassifier(random_state=508)


# Creating a GridSearchCV object
c_tree_2_hp_cv = GridSearchCV(c_tree_2_hp, param_grid, cv=3)


# Fit it to the training data
c_tree_2_hp_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", c_tree_2_hp_cv.best_params_)
print("Tuned Logistic Regression Accuracy:",
      c_tree_2_hp_cv.best_score_.round(4))


###############################################################################
# Visualizing the Tree
###############################################################################


# Building a tree model object with optimal hyperparameters
c_tree_optimal = DecisionTreeClassifier(random_state=508,
                                        max_depth=5,
                                        min_samples_leaf=7)


c_tree_optimal_fit = c_tree_optimal.fit(X_train, y_train)

c_tree_optimal_pred_test = c_tree_optimal_fit.predict(X_test)

dot_data = StringIO()


export_graphviz(decision_tree=c_tree_optimal_fit,
                out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=X_train.columns)


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png(),
      height=500,
      width=800)


# Feature importance function
########################

def plot_feature_importances(model, train=X_train, export=False):
    fig, ax = plt.subplots(figsize=(12, 9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


########################


plot_feature_importances(c_tree_optimal,
                         train=X_train)


# Scoring the gini model
print(
    'CTree Training Score',
    c_tree_optimal_fit.score(
        X_train,
        y_train).round(4))
print(
    'CTree Testing Score:',
    c_tree_optimal_fit.score(
        X_test,
        y_test).round(4))


###############################################################################
# Random Forest in scikit-learn
###############################################################################


# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators=500,
                                          criterion='gini',
                                          max_depth=None,
                                          min_samples_leaf=15,
                                          bootstrap=True,
                                          warm_start=False,
                                          random_state=508)


# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators=500,
                                             criterion='entropy',
                                             max_depth=None,
                                             min_samples_leaf=15,
                                             bootstrap=True,
                                             warm_start=False,
                                             random_state=508)


# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)


full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


# Are our predictions the same for each model?
pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))


full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()


# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))


# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))


# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test = full_entropy_fit.score(X_test, y_test)


########################
# Feature importance function
########################

def plot_feature_importances(model, train=X_train, export=False):
    fig, ax = plt.subplots(figsize=(12, 9))
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")


########################

plot_feature_importances(full_gini_fit,
                         train=X_train)


plot_feature_importances(full_entropy_fit,
                         train=X_train)


########################
# Parameter tuning with GridSearchCV
########################


# Creating a hyperparameter grid
estimator_space = pd.np.arange(100, 1350, 250)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['gini', 'entropy']
bootstrap_space = [True, False]
warm_start_space = [True, False]


param_grid = {'n_estimators': estimator_space,
              'min_samples_leaf': leaf_space,
              'criterion': criterion_space,
              'bootstrap': bootstrap_space,
              'warm_start': warm_start_space}


# Building the model object one more time
full_forest_grid = RandomForestClassifier(max_depth=None,
                                          random_state=508)


# Creating a GridSearchCV object
full_forest_cv = GridSearchCV(full_forest_grid, param_grid, cv=3)


# Fit it to the training data
full_forest_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter:", full_forest_cv.best_params_)
print("Tuned Logistic Regression Accuracy:",
      full_forest_cv.best_score_.round(4))


########################
# Building Random Forest Model Based on Best Parameters
########################

rf_optimal = RandomForestClassifier(bootstrap=False,
                                    criterion='gini',
                                    min_samples_leaf=16,
                                    n_estimators=350,
                                    warm_start=True)


rf_optimal.fit(X_train, y_train)


rf_optimal_pred = rf_optimal.predict(X_test)

rf_optimal_pred_proba = rf_optimal.predict_proba(X_test)

print('RF Training Score', rf_optimal.score(X_train, y_train).round(4))
print('RF Testing Score:', rf_optimal.score(X_test, y_test).round(4))


rf_optimal_train = rf_optimal.score(X_train, y_train)
rf_optimal_test = rf_optimal.score(X_test, y_test)


###############################################################################
# Gradient Boosted Machines
###############################################################################


# Building a weak learner gbm (with all defaults)
gbm_3 = GradientBoostingClassifier(loss='deviance',
                                   learning_rate=0.1,
                                   n_estimators=100,
                                   max_depth=3,
                                   criterion='friedman_mse',
                                   warm_start=False,
                                   random_state=508,
                                   )


gbm_3_fit = gbm_3.fit(X_train, y_train)


gbm_3_predict = gbm_3_fit.predict(X_test)


# Training and Testing Scores
print('Training Score', gbm_3_fit.score(X_train, y_train).round(4))
print('Testing Score:', gbm_3_fit.score(X_test, y_test).round(4))


gbm_basic_train = gbm_3_fit.score(X_train, y_train)
gmb_basic_test = gbm_3_fit.score(X_test, y_test)


########################
# Applying GridSearchCV
########################


# Creating a hyperparameter grid
learn_space = pd.np.arange(0.1, 1.6, 0.1)
estimator_space = pd.np.arange(50, 250, 50)
depth_space = pd.np.arange(1, 10)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate': learn_space,
              'max_depth': depth_space,
              'criterion': criterion_space,
              'n_estimators': estimator_space}


# Building the model object one more time
gbm_grid = GradientBoostingClassifier(random_state=508)


# Creating a GridSearchCV object
gbm_grid_cv = GridSearchCV(gbm_grid, param_grid, cv=3)


# Fit it to the training data
gbm_grid_cv.fit(X_train, y_train)


# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))


########################
# Building GBM Model Based on Best Parameters
########################

gbm_optimal = GradientBoostingClassifier(criterion='mae',
                                         learning_rate=0.3,
                                         max_depth=5,
                                         n_estimators=200,
                                         random_state=508)


gbm_optimal.fit(X_train, y_train)


gbm_optimal_score = gbm_optimal.score(X_test, y_test)


gbm_optimal_pred_train = gbm_optimal.predict(X_train)
gbm_optimal_pred_test = gbm_optimal.predict(X_test)


# Training and Testing Scores
print('GBM Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('GBM Testing Score:', gbm_optimal.score(X_test, y_test).round(4))


gbm_optimal_train = gbm_optimal.score(X_train, y_train)
gmb_optimal_test = gbm_optimal.score(X_test, y_test)


###############################################################################
# Comparing with all methodologies
###############################################################################

# Logistic Regression
print(
    'LOG Training Score',
    logreg_optimal_fit.score(
        X_train,
        y_train).round(4))
print('LOG Testing Score:', logreg_optimal_fit.score(X_test, y_test).round(4))

# K-nearest neighbor
print('KNN Training Score', knn_clf_fit.score(X_train, y_train).round(4))
print('KNN Testing Score:', knn_clf_fit.score(X_test, y_test).round(4))

# Classification Decision Tree
print(
    'CTree Training Score',
    c_tree_optimal_fit.score(
        X_train,
        y_train).round(4))
print(
    'CTree Testing Score:',
    c_tree_optimal_fit.score(
        X_test,
        y_test).round(4))

# Random Forest
print('RF Training Score', rf_optimal.score(X_train, y_train).round(4))
print('RF Testing Score:', rf_optimal.score(X_test, y_test).round(4))

# Gradient Boosted Machines
print('GBM Training Score', gbm_optimal.score(X_train, y_train).round(4))
print('GBM Testing Score:', gbm_optimal.score(X_test, y_test).round(4))

# AUC:
print('Training AUC Score', roc_auc_score(
    y_train, knn_clf_fit_pred_train).round(4))
print('Testing AUC Score', roc_auc_score(
    y_test, knn_clf_fit_pred_test).round(4))


###############################################################################
# Cross Validation with k-folds
###############################################################################

# Cross Validating the knn model with three folds
cv_gbm_3 = cross_val_score(knn_clf,
                           got_df_data_final2,
                           got_df_target_final2,
                           cv=3)


print('\nAverage: ',
      pd.np.mean(cv_gbm_3).round(3),
      '\nMinimum: ',
      min(cv_gbm_3).round(3),
      '\nMaximum: ',
      max(cv_gbm_3).round(3))


########################
# Creating a confusion matrix
########################

print(confusion_matrix(y_true=y_test,
                       y_pred=gbm_optimal_pred_test))


# Visualizing the confusion matrix
labels = ['0=not alive', '1=alive']

cm = confusion_matrix(y_true=y_test,
                      y_pred=gbm_optimal_pred_test)


sns.heatmap(cm,
            annot=True,
            xticklabels=labels,
            yticklabels=labels,
            cmap='Blues')


plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrix of the classifier')
plt.show()


########################
# Creating a classification report
########################

# Changing the labels on the classification report

labels = ['0=not alive', '1=alive']

print(classification_report(y_true=y_test,
                            y_pred=gbm_optimal_pred_test,
                            target_names=labels))

# Area under the AUC curve
print(roc_auc_score(y_test, knn_clf_fit_pred_test).round(3))


########################
# Saving Results
########################

# Saving model predictions
model_scores_df = pd.DataFrame({'Actual': y_test,
                                'LR_Predicted': logreg_optimal_pred_test,
                                'KNN_Predicted': knn_clf_fit_pred_test,
                                'CD_Predicted': c_tree_optimal_pred_test,
                                'RF_Predicted': rf_optimal_pred,
                                'GBM_Predicted': gbm_optimal_pred_test})


model_scores_df.to_excel("Kisuc_Kim_Model_Prediction.xlsx")

