import quandl
import pandas_datareader.data as web
import datetime
import pandas as pd
import sklearn
import numpy as np
import scipy as sp
from operator import methodcaller
import time

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.model_selection import cross_val_score
import warnings; warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest

indicators = {
    'Selic': 'BCB/432',
    'Exchange Rate USD Sell': 'BCB/1',
    'Exchange Rate USD Buy': 'BCB/10813',
    'BM&F Gold gramme': 'BCB/4',
    'Bovespa total volume': 'BCB/8',
    'International Reserves': 'BCB/13621',
    'Bovespa index': 'BCB/7',
    'Foreign exchange operations balance': 'BCB/13961',
    'Nasdaq index': 'BCB/7810',
    'Dow Jones index': 'BCB/7809'
}

def get_stock_data(symbol, start_date, end_date):
    df = web.DataReader(symbol + '.SA', 'yahoo', start_date, end_date)
    return df

def download_market_data(start_date, end_date):
    df = quandl.get(list(indicators.values()), start_date=start_date , end_date=end_date)
    df.columns = list(indicators.keys())

    # to review - filling all NAs with data. some data, such as foreign exchange operations balance, is too sparse
    df = df.fillna(0)

    return df

def merge_datasets(input_df, market_df):
    df = pd.concat([input_df, market_df], join='inner', axis=1)
    return df

def get_tech_indicators(input_df):
    df = input_df.copy()
    for n in range(10,61,10):
        df['sma'+str(n)] = df['Adj Close'].rolling(window=n, center=False).mean()
        std =df['Adj Close'].rolling(window=n, center=False).std()
        df['bb_lower'+str(n)] = df['sma'+str(n)] - (2 * std)
        df['bb_upper'+str(n)] = df['sma'+str(n)] + (2 * std)
    return df


def create_labels(input_df, span, profit):
    df = input_df.copy()
    df.loc[:, 'Label'] = False
    for i in range(1,span+1):
        delta = df['Adj Close'] / df.shift(-i)['Adj Close'] -1
        df.loc[:, 'Label'] = (delta > profit).values | df.loc[:, 'Label'].values
    return df[:(df.shape[0]-span)]

def create_features(input_df, base):
    """ Receives a dataframe as input 
        Returns a new dataframe with ratios calculated
    """
    # avoid modifying in place
    df = input_df.copy() 
    # get all columns ubt the label
    cols = list(df.columns)
    if 'Label' in cols:
        cols.remove('Label')
    # create the new columns for the number of days
    for n in range(1,base+1):
        new_cols = list(map(lambda x: "{}-{}".format(x,n), cols))
        df[new_cols] = (df.loc[:, cols] / df.shift(n).loc[:, cols]) - 1
    
    # replace +inf with max and -inf with min, for each column
    for col in df.columns:
        if len(df[df[col] == np.inf]):
            df.loc[df[col] == np.inf, col] = df.loc[df[col] != np.inf, col].max()
        if len(df[df[col] == -np.inf]):
            df.loc[df[col] == -np.inf, col] = df.loc[df[col] != -np.inf, col].min()    
            
    # to review - filling all NAs with data, instead of dropping
    # need to understand why 50% of the data is returning NA
    # have to do with the foreign exchange data, dividing by 0
    df = df.fillna(0)
   
    # leave or remove original columns? for now I will leave them
    #return df.drop(cols, axis=1)
    return df

def split_features_labels(df):
    features = [x for x in df.columns if x != "Label"]
    X = df[features].values
    y = df['Label'].values
    return X, y

def gen_classifier(symbol, market_df, start_date, end_date, base=60, span=4, profit=.05):
    # 1. get stock data
    df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    # 2. add market data
    df = merge_datasets(df, market_df)
    # 3. calculate stock indicators
    df = get_tech_indicators(df)
    # 4. create features
    df = create_features(df, base=base)
    # 5. create labels
    df = create_labels(df, span=span, profit=profit)
    # 6. split features and labels
    X,  y = split_features_labels(df)
    print(np.bincount(y))
    # 7. scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 8. fit classifier 
    # knn with pca, train faster. can't lose this amount of time
    pca = PCA(n_components=10, random_state=42) #whiten=True
    X = pca.fit_transform(X)
    clf = kNN()
    clf.fit(X, y)
    # 9. calculate precision
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='precision')
    print("Precision for {}: {:0.2f} (+/- {:0.2f})".format(symbol, scores.mean(), scores.std() * 2))
    # 10. return
    return clf, scaler, pca, scores.mean()

def score_classifier(symbol, market_df, start_date, end_date, base=60, span=4, profit=.05):
    """ Adapted gen classifier function specifically to generate scores """

    # 1. get stock data
    df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    # 2. add market data
    df = merge_datasets(df, market_df)
    # 3. calculate stock indicators
    df = get_tech_indicators(df)
    # 4. create features
    df = create_features(df, base=base)
    # 5. create labels
    df = create_labels(df, span=span, profit=profit)
    # 6. split features and labels
    X,  y = split_features_labels(df)
    # 7. scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # 8. fit classifier 
    # knn with pca, train faster. can't lose this amount of time
    pca = PCA(n_components=10, random_state=42) #whiten=True
    X = pca.fit_transform(X)
    clf = kNN()
    clf.fit(X, y)
    # 9. calculate precision
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.1, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring='precision')
    return scores.mean(), scores.std(), np.bincount(y)[1]/len(y)

def prep_backtest_data(symbol, market_df, start_date, end_date, base=60):
    # 1. get stock data
    df = get_stock_data(symbol=symbol, start_date=start_date, end_date=end_date)
    # 2. add market data
    df = merge_datasets(df, market_df)
    # 3. calculate stock indicators
    df = get_tech_indicators(df)
    # 4. create features
    df = create_features(df, base=base)
    # return df
    return df

# special methods created for ga


"""
def create_labels(input_df, forward=19, profit_margin=.042, stop_loss=.020):
    #  need to optimize. vectorize label creation 

    df = input_df.copy()
    for row in range(df.shape[0]-forward):
        
        # initialize max and min ticks
        max_uptick = 0
        min_downtick = 0 
        # df.loc[:, 'Label'] = 0 # slower, why? go back to previous approach

        # move all days forward
        for i in range(1,forward+1):
            delta = (df.ix[row+i, 'Adj Close'] / df.ix[row, 'Adj Close'])-1
            if delta > max_uptick:
                max_uptick = delta
            if delta < min_downtick:
                min_downtick = delta

        # evaluate ticks against predefined strategy parameters
        if max_uptick >= profit_margin and min_downtick >= -stop_loss:
            df.ix[row,'Label'] = 1
        else:
	        df.ix[row, 'Label'] = 0

    return df.dropna()
"""
