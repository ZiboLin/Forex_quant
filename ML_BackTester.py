import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tpqoa as tpqoa 
from sklearn.linear_model import LogisticRegression
plt.style.use("seaborn")

class MLBacktester():
    def __init__(self,symbol,start,end,granularity,tc = 0.00007,lags = 5, train_ratio = 0.7):
        self.symbol = symbol
        self.start = start 
        self.end = end
        self.lags = lags
        self.train_ratio = train_ratio
        self.granularity = granularity
        self.tc = tc 
        self.lags = lags
        self.model = LogisticRegression(C = 1e6, max_iter = 100000, multi_class="ovr")
        self.results = None
        self.get_data()
        
    def get_data(self):
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument = self.symbol, start = self.start,end = self.end,
                             granularity = self.granularity, price = "M", localize = False)
        raw.drop(columns = ["o","h","l","volume","complete"], inplace = True)
        raw.rename(columns={"c":"price"},inplace = True)
        raw["returns"] = np.log(raw/raw.shift(1))
        self.data = raw
    
    def split_data(self, start, end):
        ''' Split the data into training set & test set.
        '''
        data = self.data.loc[start:end].copy()
        return data 
    
    def prepare_features(self, start, end):
        '''Prepares the feature columns for training set and test set 
        '''
        self.data_subset = self.split_data(start,end)
        self.feature_columns = []
        for lag in range(1, self.lags + 1):
            col = "lag{}".format(lag)
            self.data_subset[col] = self.data_subset["returns"].shift(lag)
            self.feature_columns.append(col)
        self.data_subset.dropna(inplace=True)
    
    def fit_model(self, start, end):
        self.prepare_features(start,end)
        self.model.fit(self.data_subset[self.feature_columns],np.sign(self.data_subset["returns"]))
        
    def test_strategy(self):
        #determining datetime for start, end and split (for training an testing period)
        full_data = self.data.copy()
        split_index = int(len(full_data) * self.train_ratio)
        split_date = full_data.index[split_index-1]
        train_start = full_data.index[0]
        test_end = full_data.index[-1]
        
        #fit the model on the training set
        self.fit_model(train_start,split_date)
        
        #prepare the test set 
        self.prepare_features(split_date, test_end)
        
        #make predictions on the test set
        predict = self.model.predict(self.data_subset[self.feature_columns])
        self.data_subset["pred"] = predict
        
        #calculate strategy returns
        self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]
        
        #determine the number of trades in each bar 
        self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()
        
        #subtract transaction/trading costs from pre-cost return 
        self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc
        
        self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
        self.data_subset["cstrategy"] = self.data_subset["strategy"].cumsum().apply(np.exp)
        self.results = self.data_subset
        
        perf = self.results["cstrategy"].iloc[-1]
        outperf = perf - self.results["creturns"].iloc[-1]
        
        return round(perf,6), round(outperf,6)
    
    def plot_results(self, plot_train_results = False):
        if self.results is None:
            print("Run test_strategy() first")
        else:
            title = "Logistic Regression: {} | tc = {} | lags = {}".format(self.symbol,self.tc, self.lags)
            self.results[["creturns","cstrategy"]].plot(title=title, figsize = (12,8))
        
        if plot_train_results is True:
            full_data = self.data.copy()
            split_index = int(len(full_data) * self.train_ratio)
            split_date = full_data.index[split_index-1]
            train_start = full_data.index[0]
            test_end = full_data.index[-1]

            #fit the model on the training set
            self.fit_model(train_start,split_date)

            #prepare the test set 
            self.prepare_features(train_start, split_date)

            #make predictions on the test set
            predict = self.model.predict(self.data_subset[self.feature_columns])
            self.data_subset["pred"] = predict

            #calculate strategy returns
            self.data_subset["strategy"] = self.data_subset["pred"] * self.data_subset["returns"]

            #determine the number of trades in each bar 
            self.data_subset["trades"] = self.data_subset["pred"].diff().fillna(0).abs()

            #subtract transaction/trading costs from pre-cost return 
            self.data_subset.strategy = self.data_subset.strategy - self.data_subset.trades * self.tc

            self.data_subset["creturns"] = self.data_subset["returns"].cumsum().apply(np.exp)
            self.data_subset["cstrategy"] = self.data_subset["strategy"].cumsum().apply(np.exp)
            self.results = self.data_subset
            
            title = "Logistic Regression: {} | tc = {} | lags = {}".format(self.symbol,self.tc, self.lags)
            self.results[["creturns","cstrategy"]].plot(title=title, figsize = (12,8))
            
    def hit_ratio(self):
        hits = np.sign(self.results.returns * self.results.pred.shift(0)).value_counts()
        hit_ratio = hits[1.0]/sum(hits)
        return hit_ratio

        