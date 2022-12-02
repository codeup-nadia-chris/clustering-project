'''
Create a file named evaluate.py that contains the following functions.

- plot_residuals(y, yhat): creates a residual plot
- regression_errors(y, yhat): returns the following values:
    sum of squared errors (SSE)
    explained sum of squares (ESS)
    total sum of squares (TSS)
    mean squared error (MSE)
    root mean squared error (RMSE)
- baseline_mean_errors(y): computes the SSE, MSE, and RMSE for the baseline model
- better_than_baseline(y, yhat): returns true if your model performs better than the baseline, otherwise false
'''

import seaborn as sns
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr, pearsonr
import pingouin as pg
import plotly.express as px
import matplotlib.pyplot as plt

import src.wrangle as wr


#############################################################
######   Vizualizations for Notebook Presentation    ########
#############################################################

#############     Correlation barchart       ##############

def correlation_viz():
    plt.figure(figsize=(20,20))
    df, _, __ = wr.split_zillow(wr.get_zillow())
    num_variables = df.columns.tolist()
    num_variables = num_variables[0:19]
    num_variables
    del num_variables[9:13]
    
    df['absolute_logerror'] = df['logerror'].abs()
    pearson_df = pearson_test_df(df, 'absolute_logerror', num_variables)
    pearson_df = pearson_df.sort_values(by = 'Minus_P', ascending=False)
    fig = px.bar(pearson_df, y='Minus_P', x='Potential_Target',
                 hover_data=['Keep'], color='Keep',
                 labels={'Potential_Target':'Feature Variables To Build Model On', 'Minus_P': '1 - P value'},height=400)
    fig.show()



#############################################################
######   Tests for Continuous Variable Correlation   ########
#############################################################

##############             Spearman            ##############
def spearman_test(df, target_var, test_var):
    '''alternative test for continuous to continuous correlation tests.
    Is used when correlation has a curilinear shape, rather than linear.
    Must have a roughly monotonic relationship.'''
    r, p_value = spearmanr(df[target_var], df[test_var])
    print(f'Spearman Correlation Coefficient of {test_var}: {r}\nP-value: {p_value:.3f}')


##############       Two Versions of Pearson           ##############

def pearson_test(df, target_var, test_var):
    '''default test for continuous to continuous correlation tests. 
    Handles linear relationships well'''
    r, p_value = pearsonr(df[target_var], df[test_var])
    print(f'Pearson Correlation Coefficient of {test_var}: {r}\nP-value: {p_value:.3f}')


def pearson_test_df(df, target_var, test_var_list):
    '''default test for continuous to continuous correlation tests. 
    Handles linear relationships well'''
    
    pearson_df = pd.DataFrame(
        {'Potential_Target':[],
         'Coefficient' :[],
         'P-Value' : [],
         'Minus_P' : [],
         'Keep' : [],})

    for item in test_var_list:
        r, p_value = pearsonr(df[target_var], df[item])
        if 1 - p_value >= 0.95:
            keeper = 'Yes'
        else:
            keeper = 'No'
        
        pearson_df = pearson_df.append(
        {'Potential_Target': item,
         'Coefficient' : r,
         'P-Value' : p_value,
         'Minus_P' : 1-p_value,
         'Keep' : keeper},
        ignore_index = True)
        
    return pearson_df


##############################################################
######   Tests for Categorical Variable Correlation   ########
##############################################################

##############              T-Test              ##############
def t_test(df, target_var, test_var):
    '''test for determining if there is a statistically significant 
    relationship between a categorical and continuous variable'''
    results = pg.ttest(df[target_var], df[test_var], correction=True)
    print(f'P-Val {test_var} = {results.iloc[0,3]:.3f}')



def plot_residuals(y, yhat):
    fig = sns.scatterplot(x = y, y = yhat)



def calc_performance(y, yhat, featureN = 2):
    # explained sum of squares
    ess = ((yhat - y.mean())**2).sum()
    
    # sum of squres errors
    sse = mean_squared_error(y, yhat)*len(y)
    
    # total sum of squares
    tss = ess+sse
    
    # mean sum of squares error
    mse = mean_squared_error(y,yhat)
    
    # rooted sum of squares 
    rmse = sqrt(mse)
    
    # R squared
    r2 = ess/tss
    # A second version of R Squared which may be more accurate
    r2v2 = r2_score(y, yhat)
    
    if featureN > 2:
        # Adjusted R Squared
        adjR2= 1-(1-r2v2)*(len(y)-1)/(len(y)-featureN-1)

        return ess, sse, tss, mse, rmse, r2v2, adjR2    
    
    else:
        return ess, sse, tss, mse, rmse, r2v2



def regression_errors(y, yhat, df=False, features=2):
    '''
    This module does the legwork for evaluating model efficacy. 
    The default argument 'df' will determine whether to pass a data frame with a dictionary 
    when called or to print the evaluation metrics. 
    The AdjR2Feature default argument allows for tuning an Adjusted R^2 value, which is more accurate
    than R^2 for models with multiple features/variables 

    '''
    
    
    if features <= 2:
        ess, sse, tss, mse, rmse, r2v2 = calc_performance(y, yhat)
        if df==False:
            print(f'''Model Performance
            ESS = {round(ess,5)}
            SSE = {round(sse,5)}
            TSS = {round(tss,5)}
            MSE = {round(mse,5)}
            RMSE = {round(rmse,5)}
            R2 = {round(r2v2,10)}''')
        

        else:
            df = pd.DataFrame()
        
            df ={
                'ESS' : round(ess,3),
                'SSE' : round(sse,3),
                'TSS' : round(tss,3),
                'MSE' : round(mse,3),
                'RMSE': round(rmse,3),
                'R2': round(r2v2,3)
                }
                
            return df

    else: 
        ess, sse, tss, mse, rmse, r2v2, adjR2 = calc_performance(y, yhat, features)
        if df==False:
            print(f'''Model Performance
            ESS = {round(ess,5)}
            SSE = {round(sse,5)}
            TSS = {round(tss,5)}
            MSE = {round(mse,5)}
            RMSE = {round(rmse,5)}
            R^2 = {round(r2v2,10)}
            AdjR^2 = {round(adjR2,5)}''')
        

        else:
            df = pd.DataFrame()
        
            df ={
                'ESS' : round(ess,3),
                'SSE' : round(sse,3),
                'TSS' : round(tss,3),
                'MSE' : round(mse,3),
                'RMSE': round(rmse,3),
                'R^2' : round(r2v2,3),
                'AdjR^2':round(adjR2,3)
                }
                
            return df



def evaluate_models(y, yhat):

    ess, sse, tss, mse, rmse, r2 = calc_performance(y, yhat)

    df = pd.DataFrame()
    
    df ={
            'ESS' : round(ess,3),
            'SSE' : round(sse,3),
            'TSS' : round(tss,3),
            'MSE' : round(mse,3),
            'RMSE': round(rmse,3),
            'AdjR^2': round(r2,3)
        }
    
        
    return df



def better_than_baseline(y, yhat):
    SSE, ESS, TSS, MSE, RMSE = regression_errors(y, yhat)
    
    SSE_baseline, MSE_baseline, RMSE_baseline = baseline_mean_errors(y)
    
    if SSE < SSE_baseline:
        print('My OSL model performs better than baseline')
    else:
        print('My OSL model performs worse than baseline. :( )')



''' 
making models for each county


# la county
# fit data to simple regression
lm.fit(la_x_train, la_y_train)

# make predictions
la_simple_model = lm.predict(la_x_train)

# orange county
# fit data to simple regression
lm.fit(or_x_train, or_y_train)

# make predictions
or_simple_model = lm.predict(or_x_train)

# la county
# fit data to simple regression
lm.fit(vent_x_train, vent_y_train)

# make predictions
vent_simple_model = lm.predict(vent_x_train)

'''
