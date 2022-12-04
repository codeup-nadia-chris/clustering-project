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
features_counties = ['garage_sqft', 'age','beds','garage','fireplace','bath',\
            'bed_bath_ratio','lot_sqft','tax_amount','hottub_spa', 'logerror']

# get zillow data
df = wr.get_zillow()

# separate data based on location
la_city = df[df.county_name == 'LA_city'] # LA city
la = df[df.county_name == 'LA'] # LA county
ventura = df[df.county_name == 'Ventura'] # Ventura county
orange = df[df.county_name == 'Orange'] # Orange county

# remove unneeded columns in counties data sets

la_city = la_city[features_counties]
la = la[features_counties]
ventura = ventura[features_counties]
orange = orange[features_counties]

# remove unneeded columns and add dummy variables for county_name in the main data set
df = wr.dummies(df)
df = df[features]

#split_counties into train, validate, test data sets and target vars
XLA1, XLA2, XLA3, yla1, yla2, yla3 = wr.full_split_zillow(la)
XLC1, XLC2, XLC3, ylc1, ylc2, ylc3 = wr.full_split_zillow(la_city)
XO1, XO2, XO3, yo1, yo2, yo3 = wr.full_split_zillow(ventura)
XV1, XV2, XV3, yv1, yv2, yv3 = wr.full_split_zillow(orange)
# scale counties data sets
XLA1, XLA2, XLA3 = wr.standard_scale_zillow(XLA1, XLA2, XLA3, counties=True)
XLC1, XLC2, XLC3 = wr.standard_scale_zillow(XLC1, XLC2, XLC3, counties=True)
XO1, XO2, XO3 = wr.standard_scale_zillow(XO1, XO2, XO3, counties=True)
XV1, XV2, XV3 = wr.standard_scale_zillow(XV1, XV2, XV3, counties=True)

# split the main data into 3 data sets and 3 target arrays
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

def run_models(X_train, X_validate, y_train, y_validate, f_name='stand '):
    
    '''
    general function to run models with X_train and X_validate that were scaled
    '''
    feature_name = f_name # + str(f_number)
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

def run_polinomial(X1, X2, y_train, y_validate, f_name='poly '):
    '''
    
    '''
    f = ['beds', 'bath']
    poly = PolynomialFeatures(degree=3, include_bias=False, interaction_only=False)
    poly.fit(X1[f])
    # create a df with transformed features of the train set
    X1_poly = pd.DataFrame(
        poly.transform(X1[f]),
        columns=poly.get_feature_names(f),
        index=X1.index)
    X1_poly = pd.concat([X1_poly, X1[f]], axis=1)
    #X1_poly = pd.concat([X1_poly, X1], axis=1)

    #display(X1_poly.head(1)) #testing the columns

    # create a df with transformed features for the validate set
    X2_poly = pd.DataFrame(
        poly.transform(X2[f]),
        columns=poly.get_feature_names(X2[f].columns),
        index=X2.index)
    X2_poly = pd.concat([X2_poly, X2[f]], axis=1)
    #X2_poly = pd.concat([X2_poly, X2], axis=1)

    feature_name = f_name #+ str(f_number)

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
    run_models(X_train, X_validate, y_train, y_validate, f_name='stand ')
    run_polinomial(X_train.iloc[:, :-1], X_validate.iloc[:, :-1], y_train, y_validate)
    return scores.sort_values(by=['R2_train'], ascending=False).head(10)


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
        
    return scores.head(5)

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
        
    return scores.head(5)

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

######### RUN MODELS ON COUNTY DATA SETS
def get_counties_scores(): 
    # empty the scores data frame
    scores.drop(scores.index, axis=0, inplace=True)
    # la county
    run_models(XLA1, XLA2, yla1, yla2, f_name='la stand')
    run_polinomial(XLA1, XLA2, yla1, yla2, f_name='la poly')

    # la city
    run_models(XLC1, XLC2, ylc1, ylc2, f_name='la_city stand')
    run_polinomial(XLC1, XLC2, ylc1, ylc2, f_name='la_city poly')

    # orange county
    run_models(XO1, XO2, yo1, yo2, f_name='orange stand')
    run_polinomial(XO1, XO2, yo1, yo2, f_name='orange poly')

    # ventura county
    run_models(XV1, XV2, yv1, yv2, f_name='ventura stand')
    run_polinomial(XV1, XV2, yv1, yv2, f_name='ventura poly')
    
    return scores.sort_values(by=['R2_train', 'R2_validate'], ascending=False).head(10)

####### get the scores of the best model ###########
def get_final_scores():
    XLC1, XLC2, XLC3, ylc1, ylc2, ylc3 = wr.full_split_zillow(la_city)
    LC1, XLC2, XLC3 = wr.standard_scale_zillow(XLC1, XLC2, XLC3, counties=True)
    rf = RandomForestRegressor(max_depth=4, random_state=seed)
    rf.fit(XLC1, ylc1)
    y_hat_train = rf.predict(XLC1)
    y_hat_validate = rf.predict(XLC2)
    y_hat_test = rf.predict(XLC3)
    R2_train = regression_errors(ylc1, y_hat_train)
    R2_validate = regression_errors(ylc2, y_hat_validate)
    R2_test = regression_errors(ylc3, y_hat_test)
    final_scores = pd.DataFrame(columns=['model_name','train', 'validate', 'test'])
    final_scores.loc[len(final_scores.index)] = ['Random Forest Regressor', R2_train, R2_validate, R2_test]
    return final_scores