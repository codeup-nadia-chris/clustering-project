import pandas as pd
import numpy as np


from sklearn.preprocessing import  PolynomialFeatures
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, explained_variance_score

# linear regressions
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor

# non-linear regressions
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import src.wrangle as wr
import src.clusters as cl



############## GLOBAL VARIABLES ###########
seed = 42 # random seed for random_states
features = ['garage_sqft', 'age','beds','garage','fireplace','bath',\
            'bed_bath_ratio','lot_sqft','tax_amount','hottub_spa', 'Orange',\
            'Ventura', 'LA','logerror']

# get zillow data
df = wr.get_zillow()


# remove unneeded columns and add dummy variables for county_name
df = wr.dummies(df)
df = df[features]


# leave the 

# split the data into 3 data sets and 3 target arrays
X_train, X_validate, X_test, y_train, y_validate, y_test = wr.full_split_zillow(df)

# get scaled X_train, X_validate, X_test sets
# standard scaler
X_train, X_validate, X_test = wr.standard_scale_zillow(X_train, X_validate, X_test)

# get a baseline value = median of the train set's target
baseline = y_train.mean()


###### GLOBAL EVALUATION VARS ##########

# DataFrame to keep model's evaluations
scores = pd.DataFrame(columns=['model_name', 'feature_name', 'R2_train', 'R2_validate'])

# create a dictionary of regression models
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting Regression': GradientBoostingRegressor(random_state=seed),
    'Decision Tree Regression': DecisionTreeRegressor(max_depth=4, random_state=seed),
    'Random Forest Regression':RandomForestRegressor(max_depth=4, random_state=seed),
    'LassoLars Regression':LassoLars(alpha=0.1)
    }


############### EVALUATION FUNCTIONS #############

def regression_errors(y_actual, y_predicted):
    '''
    returns r^2 score
    '''

    # adjucted R^2 score
    ADJR2 = explained_variance_score(y_actual, y_predicted)
    return round(ADJR2, 2)

############### MODELING FUNCTIONS ###############

def run_models(X_train, X_validate, y_train, y_validate, f_number):
    
    '''
    general function to run models with X_train and X_validate that were scaled
    '''
    feature_name = 'stand ' + str(f_number)
    for key in models:
        # create a model
        model = models[key]
        # fit the model
        model.fit(X_train, y_train)
        # predictions of the train set
        y_hat_train = model.predict(X_train)
        # predictions of the validate set
        y_hat_validate = model.predict(X_validate)


        # calculate scores train set
        R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        R2_val = regression_errors(y_validate, y_hat_validate)
        
        
        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [key, feature_name, R2, R2_val]

def run_polinomial(X1, X2, y_train, y_validate, f_number):
    i = 5 # first 6 features
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly.fit(X1.iloc[:, :i])
    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
        poly.transform(X1.iloc[:, :i]),
        columns=poly.get_feature_names(X1.iloc[:, :i].columns),
        index=X1.index)
    X1_poly = pd.concat([X1_poly, X1.iloc[:, i:]], axis=1)
    #X1_poly = pd.concat([X1_poly, X1], axis=1)

    #display(X1_poly.head(1)) #testing the columns

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
        poly.transform(X2.iloc[:, :i]),
        columns=poly.get_feature_names(X2.iloc[:, :i].columns),
        index=X2.index)
    X2_poly = pd.concat([X2_poly, X2.iloc[:, i:]], axis=1)
    #X2_poly = pd.concat([X2_poly, X2], axis=1)

    feature_name = 'poly '+ str(f_number)

    for key in models:
        # create a model
        model = models[key]
        # fit the model
        model.fit(X1_poly, y_train)
        # predictions of the train set
        y_hat_train = model.predict(X1_poly)
        # predictions of the validate set
        y_hat_validate = model.predict(X2_poly)

        # calculate scores train set
        R2 = regression_errors(y_train, y_hat_train)
        # calculate scores validation set
        R2_val = regression_errors(y_validate, y_hat_validate)
        

        # add the score results to the scores Data Frame
        scores.loc[len(scores.index)] = [key, feature_name, R2, R2_val]


def get_scores():
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    run_models(X_train, X_validate, y_train, y_validate, np.nan)
    run_polinomial(X_train.iloc[:, :-1], X_validate.iloc[:, :-1], y_train, y_validate,np.nan)
    return scores


############# RUN MODELS ON CLUSTERS ##############

X_train_num, X_validate_num, X_test_num = cl.add_numerical_clusters(X_train, X_validate, X_test)
X_train_loc, X_validate_loc, X_test_loc = cl.add_location_clusters(X_train, X_validate, X_test)

def check_numerical_clusters():
    '''
    run models on numerical cluster data frames
    '''
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    
    # create data frames based on clusters and run models
    for j in range(6):
        # separate by clusters
        X1 = X_train_num[X_train_loc.numerical_clusters == j]
        X2 = X_validate_num[X_validate_loc.numerical_clusters == j]
        # drop column location_cluster
        X1.drop(columns='numerical_clusters', inplace=True)
        X2.drop(columns='numerical_clusters', inplace=True)
        # separate y_train
        y1 = y_train[X1.index]
        y2 = y_validate[X2.index]
        
        # run models
        run_models(X1, X2, y1, y2, j)
        run_polinomial(X1.iloc[:, :-1], X2.iloc[:, :-1], y1, y2, j)
        
    return scores

def check_location_clusters():
    '''
    run models on location cluster data frames
    '''
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    
    # create data frames based on clusters and run models
    for j in range(6):
        # separate by clusters
        X1 = X_train_loc[X_train_loc.location_clusters == j]
        X2 = X_validate_loc[X_validate_loc.location_clusters == j]
        # drop column location_cluster
        X1.drop(columns='location_clusters', inplace=True)
        X2.drop(columns='location_clusters', inplace=True)
        # separate y_train
        y1 = y_train[X1.index]
        y2 = y_validate[X2.index]
        
        # run models
        run_models(X1, X2, y1, y2, j)
        run_polinomial(X1.iloc[:, :-1], X2.iloc[:, :-1], y1, y2, j)
        
    return scores

def get_cluster_scores():
    '''
    this function runs models on clustered subsets
    returns the data frame with first 10 results of 
    '''
    loc_clust = check_location_clusters().iloc[:10, :]
    num_clust = check_numerical_clusters().iloc[:10, :]
    cluster_results = pd.concat([loc_clust, num_clust], axis=1)
    columns = ['location_clusters', 'feature_name_loc', 'Location_R2_train', 'R2_val_loc',\
           'numerical_clusters', 'feature_name_num', 'Numerical_R2_train', 'R2_val_num']
    cluster_results.columns = columns
    columns2 = ['Location_R2_train', 'R2_val_loc', 'Numerical_R2_train', 'R2_val_num']
    cluster_results = cluster_results[columns2]
    
    return cluster_results