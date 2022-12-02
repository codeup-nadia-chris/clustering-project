import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import src.wrangle as wr

########## GLOBAL VARIABLES #######

seed = 42

# features to use for clustering
location = ['latitude', 'longitude'] # 6 clusters
numerical = ['sqft', 'garage_sqft', 'lot_sqft', 'age'] # 7 clusters

df = wr.get_zillow()
train, validate, test = wr.split_zillow(df)
X_train, X_validate, X_test, y_train, y_validate, y_test = wr.full_split_zillow(df)
X_train, X_validate, X_test = wr.standard_scale_zillow(X_train, X_validate, X_test, clustering=True)

# get train_scaled to 
train_scaled = wr.standard_scale_one_df(train).iloc[:, :-2]

######### clustering exploration #########

def visualize_map(df):
    '''
    this function accepts a data frame as a parameter
    and print out a map of counties based on their latitude and longitude
    '''
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=train, x='longitude', y='latitude', s=1, hue='county_name')
    plt.title('longitude vs latitude')
    plt.yticks([])
    plt.xticks([])


######## find the best k for k-meean ######

def find_the_k(df:pd.DataFrame, k_min:int = 1, k_max:int = 10, list_of_features=None):
    '''
    function accepts a scaled data frame as a parameter,
    range for clusters and list of featured
    visualizes distance to the points for every cluster
    and returns a data frame with calculation results
    '''
    k_range = range(k_min, k_max+1)
    if list_of_features == None:
        list_of_features = df.columns.tolist()
    wcss = [] #Within-Cluster Sum of Square
    k_range = range(1,11)
    clustering = df[list_of_features]
    # run the loop with clusters from 1 to 10 to find the best n_clusters number
    for i in k_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(clustering)
        wcss.append(kmeans.inertia_)
    # compute the difference from one k to the next
    delta = [round(wcss[i] - wcss[i+1],0) for i in range(len(wcss)-1)]
    # compute the percent difference from one k to the next
    pct_delta = [round(((wcss[i] - wcss[i+1])/wcss[i])*100, 1) for i in range(len(wcss)-1)]
    
    # create a dataframe with all of our metrics to compare them across values of k: SSE, delta, pct_delta
    compare = pd.DataFrame(dict(k=k_range[0:-1], 
                             wcss=wcss[0:-1], 
                             delta=delta, 
                             pct_delta=pct_delta))

    # visualize points and distances between them
    plt.figure(figsize=(20, 8))
    # plot wcss to find the 'elbow'
    plt.subplot(1, 2, 1)
    plt.plot(k_range, wcss, color='#6d4ee9', marker='D')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distance to the points')
    #plt.xlim(start_point, end_point)

    # plot k with pct_delta
    plt.subplot(1, 2, 2)
    plt.plot(compare.k, compare.pct_delta, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Percent Change')
    plt.title('Change in distance %')
    plt.show()
    
    # return a data frame
    return compare

####### CLUSTERING FUNCTIONS #############

def run_clustering_location():
    '''
    the function create clusters based on latitude and longitude values
    return 3 arrays with cluster numbers that can be added to the train/validate/test data frames
    '''
    kmeans = KMeans(n_clusters=6, init='k-means++', random_state=seed)
    loc_train = kmeans.fit_predict(X_train[location])
    loc_validate = kmeans.predict(X_validate[location])
    loc_test = kmeans.predict(X_test[location])
    return loc_train, loc_validate, loc_test

def run_clustering_numerical():
    '''
    the function create clusters based on numerical columns
    return 3 arrays with cluster numbers that can be added to the train/validate/test data frames
    '''
    kmeans = KMeans(n_clusters=7, init='k-means++', random_state=seed)
    num_train = kmeans.fit_predict(X_train[numerical])
    num_validate = kmeans.predict(X_validate[numerical])
    num_test = kmeans.predict(X_test[numerical])
    return num_train, num_validate, num_test

 #### get values of clustering

loc_train, loc_validate, loc_test  = run_clustering_location()
num_train, num_validate, num_test =  run_clustering_numerical()

def add_clusters_to_train(df):
    df['location_clusters'] = loc_train.astype('uint8')
    df['numerical_clusters'] = num_train.astype('uint8')
    return df

train_with_clusters = add_clusters_to_train(train)

def viz_clustering_results():
    '''
    this function shows the results of location based and numerical based clusters
    '''

    # set colors for visuals
    colors = ['red', 'blue', 'green', 'magenta']
    palette = sns.set_palette(sns.color_palette(colors))

    ### plot the results
    plt.figure(figsize=(20, 8))
    plt.suptitle('Clustering results')

    # subplot 1 viz for numerical clusters
    plt.subplot(121)
    sns.scatterplot(data=train, x='longitude', y='latitude', hue='county_name', palette='Accent', s=300, alpha=0.1, legend=None)
    sns.scatterplot(data=train, x='longitude', y='latitude', hue='numerical_clusters', palette=palette, s=25, legend=None)
    plt.title('Background -> counties, dots -> numerical clusters')
    plt.yticks([])
    plt.xticks([])

    # subplot 2 viz for location clusters
    plt.subplot(122)
    sns.scatterplot(data=train, x='longitude', y='latitude', hue='county_name', palette='Accent', s=300, alpha=0.1, legend=None)
    sns.scatterplot(data=train, x='longitude', y='latitude', hue='location_clusters', palette=palette, s=25, legend=None)
    plt.title('Background -> counties, dots -> locations clusters')
    plt.yticks([])
    plt.xticks([])
    plt.show()


def add_location_clusters(train, validate, test): 
    '''
    the function accepts train, validate, test as parameters
    returns those sets with columns with clusters attached
    '''
    train['location_clusters'] = loc_train.astype('uint8')
    validate['location_clusters'] = loc_validate.astype('uint8')
    test['location_clusters'] = loc_test.astype('uint8')
    train['numerical_clusters'] = num_train.astype('uint8')
    validate['numerical_clusters'] = num_validate.astype('uint8')
    test['numerical_clusters'] = num_test.astype('uint8')
    
    return train, validate, test

