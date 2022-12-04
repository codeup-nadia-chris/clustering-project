import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import scipy.stats as stats

from sklearn.cluster import KMeans

import src.wrangle as wr

####### display options
pd.options.display.float_format = '{:,.3f}'.format
# define the default font sizes
plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

########## GLOBAL VARIABLES #######

seed = 42
alpha = 0.05



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

def viz_zip_counties(df):
    plt.figure(figsize=(12,6))
    sns.scatterplot(data=df, y='zip', x='logerror', hue='county_name')
    plt.title('Zip codes and counties')
    plt.xticks([])
    plt.yticks([])
    plt.show()

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
    kmeans1 = KMeans(n_clusters=7, init='k-means++', random_state=seed)
    num_train = kmeans1.fit_predict(X_train[numerical])
    num_validate = kmeans1.predict(X_validate[numerical])
    num_test = kmeans1.predict(X_test[numerical])
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
    returns those sets with columns with location clusters attached
    '''
    train['location_clusters'] = loc_train.astype('uint8')
    validate['location_clusters'] = loc_validate.astype('uint8')
    test['location_clusters'] = loc_test.astype('uint8')
    
    return train, validate, test

def add_numerical_clusters(train, validate, test):
    '''
    the function accepts train, validate, test as parameters
    returns those sets with columns with numerical clusters attached
    '''
    train['numerical_clusters'] = num_train.astype('uint8')
    validate['numerical_clusters'] = num_validate.astype('uint8')
    test['numerical_clusters'] = num_test.astype('uint8')

    return train, validate, test

############## EXPLORATION WITH CLUSTERS #######################
def test_clusters():
    '''
    the function displays the results of t-test for every cluster' mean vs logerror' mean
    '''
    # print the results of location clusters
    print('Location clusters')
    for i in range(6):
        twc = train_with_clusters[train_with_clusters.location_clusters == i]
        t, p = stats.ttest_1samp(twc.logerror, train.logerror.mean())
        if p < alpha:
            print(f'Cluater {i}: the difference in means is significant. P-value={round(p, 3)}')
        else:
            print(f'Cluater {i}: the difference in means is not significant. P-value={round(p, 3)}')
    print()

    
    # print the results of numerical clusters
    print('Numerical clusters')
    for i in range(7):
        twc = train_with_clusters[train_with_clusters.numerical_clusters == i]
        t, p = stats.ttest_1samp(twc.logerror, train.logerror.mean())
        if p < alpha:
            print(f'Cluster {i}: the difference in means is significant. P-value={round(p, 3)}')
        else:
            print(f'Cluster {i}: the difference in means is not significant. P-value={round(p, 3)}')
    print()


#def dummies_for_loc_clusters(loc_train, loc_validate, loc_test):
    '''
    creates dummies for the location clusters
    replaces signficant clusters with 1 and not significat with 0
    returns dummies for train, validate, test data sets
    '''
    # creaate dummies for location clusters
    
    #loc_train = loc_train.replace({0:1, 1:0, 2:0, 3:1, 4:1, 5:1})
    #loc_validate = loc_validate.replace({0:1, 1:0, 2:0, 3:1, 4:1, 5:1})
    #loc_test = loc_test.replace({0:1, 1:0, 2:0, 3:1, 4:1, 5:1})
    
    #return loc_train, loc_validate, loc_test

#def dummies_for_num_clusters(num_train, num_validate, num_test):
    '''
    creates dummies for the numerical clusters
    replaces signficant clusters with 1 and not significat with 0
    returns dummies for train, validate, test data sets
    '''
    # create dummies for numerical clusters
    
    #num_train = num_train.replace({2:0, 3:0, 4:1, 5:1, 6:0})
    #num_validate = num_validate.replace({2:0, 3:0, 4:1, 5:1, 6:0})
    #num_test = num_test.replace({2:0, 3:0, 4:1, 5:1, 6:0})
    
    #return num_train, num_validate, num_test

def viz_cluster_means():
    '''
    the function creates barplots with means of 
    location and numerical clusters
    '''
    plt.figure(figsize=(20, 6))
    plt.suptitle('Logerror means of the clusters')

    plt.subplot(121)
    sns.barplot(data=train_with_clusters, x='location_clusters', y='logerror')
    plt.title('Location clusters')

    plt.subplot(122)
    sns.barplot(data=train_with_clusters, x='numerical_clusters', y='logerror')
    plt.title('Numerical clusters')
    plt.show()