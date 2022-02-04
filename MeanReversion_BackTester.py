import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import tpqoa


class MeanRevBacktester:
    def __init__(self,symbol,dev,start,end,granularity,SMA,tc):
        self.symbol = symbol
        self.dev = dev
        self.start = start
        self.end = end
        self.granularity = granularity
        self.SMA = SMA #default should be 30
        self.tc = tc
        self.get_data()
        self.results = None
    
    def get_data(self):
        api = tpqoa.tpqoa("oanda.cfg")
        raw =  api.get_history(instrument=self.symbol, start = self.start, end = self.end,
               granularity = self.granularity, price = "M", localize = False )
        raw.drop(columns=["o","h","l","volume","complete"],inplace = True)
        raw.rename(columns={"c":"price"},inplace = True)
        raw["returns"] = np.log(raw/raw.shift(1))

        #prepare data
        raw["SMA"] = raw["price"].rolling(self.SMA).mean()
        raw["lower"] = raw["SMA"]-self.dev*raw["price"].rolling(self.SMA).std()
        raw["upper"] = raw["SMA"]+self.dev*raw["price"].rolling(self.SMA).std()
        
        
        self.data = raw
        return raw

    def set_parameters(self, SMA = None, dev = None):
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["price"].rolling(self.SMA).mean()
            self.data["lower"] = self.data["SMA"]-self.dev*self.data["price"].rolling(self.SMA).std()
            self.data["upper"] = self.data["SMA"]+self.dev*self.data["price"].rolling(self.SMA).std()
        if dev is not None:
            self.dev = dev
            self.data["lower"] = self.data["SMA"]-self.dev*self.data["price"].rolling(self.SMA).std()
            self.data["upper"] = self.data["SMA"]+self.dev*self.data["price"].rolling(self.SMA).std()
            
    def test_strategy(self):
        data = self.data.copy().dropna()
        data["distance"] = data.price - data.SMA
        data["position"] = np.where(data["price"] < data["lower"],1,np.nan)
        data["position"] = np.where(data["price"] > data["upper"],-1,data["position"])
        #if meet the SMA, go neutral , position*negative = negative
        data["position"] = np.where(data.distance*data.distance.shift(1)<0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)
        data["strategy"] = data.position.shift(1) * data["returns"]
        data.dropna(inplace = True)
        
        #determine when a trae takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        #absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        
        return round(perf,6), round(outperf,6)
    
    def plot_results(self):
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | SMA = {} | dev = {} | TC = {}".format(self.symbol,
                                                               self.SMA, self.dev, self.tc)
            self.results[["creturns","cstrategy"]].plot(title = title, figsize = (12,8))
    
    def update_and_run(self,boll):
        self.set_parameters(int(boll[0]),int(boll[1]))
        return -self.test_strategy()[0]
        
    
    def optimise_parameters(self,SMA_range,dev_range):
        opt = brute(self.update_and_run,(SMA_range,dev_range),finish=None)
        return opt,-self.update_and_run(opt)
        
    
    
        

