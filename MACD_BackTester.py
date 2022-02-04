import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute 
import tpqoa

class MACD_BackTester():
    def __init__(self,symbol,start,end,granularity,tc,EMA_short,EMA_long):
        self.symbol = symbol
        self.start = start
        self.end = end 
        self.granularity = granularity 
        self.tc = tc 
        self.EMA_long = EMA_long
        self.EMA_short = EMA_short
        
        self.get_computed_data()
        self.results = None
    
    def get_computed_data(self):
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument=self.symbol,start=self.start,end=self.end,
                             granularity=self.granularity,price="M",localize=False)
        raw.drop(columns=["o","h","l","volume","complete"],inplace=True)
        raw.rename(columns={"c":"price"},inplace=True)
        raw["returns"] = np.log(raw/raw.shift(1))
        
        #### EMA must be used 
        raw["MACD_short"] = raw["price"].ewm(span = self.EMA_short, adjust = False).mean()
        raw["MACD_long"] = raw["price"].ewm(span = self.EMA_long, adjust = False).mean()
        raw["MACD"] = raw["MACD_short"] - raw["MACD_long"]

        self.data = raw
        return raw
    
    def set_parameters(self, EMA_short = None, EMA_long = None):
        if EMA_short is not None:
            self.EMA_short = EMA_short
            self.data["MACD_short"] = self.data["price"].ewm(span = self.EMA_short, adjust = False).mean()
            self.data["MACD"] = self.data["MACD_short"] - self.data["MACD_long"]
        if EMA_long is not None:
            self.EMA_long = EMA_long
            self.data["MACD_long"] = self.data["price"].ewm(span = self.EMA_long, adjust = False).mean()
            self.data["MACD"] = self.data["MACD_short"] - self.data["MACD_long"]
    
    def test_strategy(self):
        data = self.data.copy().dropna()
        
        ####
        data["position"] = np.where(data["MACD"]>0,1,-1)
        ### to avoid false negative/positve signal, we can set an interval?
#         data["position"] = np.where(data["MACD"]>0.1,1,np.nan)
#         data["position"] = np.where(data["MACD"]<-0.1,-1,data["position"])
        data["position"] = data.position.ffill().fillna(0)
        ####
        
        data["strategy"]=data.position.shift(1)*data["returns"]
        data.dropna(inplace=True)
        
        #determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        #substract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades*self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        
        return round(perf,6),round(outperf,6)
    
    def update_and_run(self,EMA):
        self.set_parameters(int(EMA[0]),int(EMA[1]))
        return -self.test_strategy()[0]
        
    def optimise_parameters(self,EMA_short_range,EMA_long_range):
        opt = brute(self.update_and_run,(EMA_short_range,EMA_long_range),finish=None)
        return opt,-self.update_and_run(opt)
    
    def hit_ratio(self):
        hits = np.sign(self.results.returns * self.results.position.shift(1)).value_counts()
        hit_ratio = hits[1.0]/sum(hits)
        return hit_ratio
        
    def plot_results(self,plot_MACD=False):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | EMA_short = {} | EMA_long = {} | granularity = {} | TC = {}".format(self.symbol,self.EMA_short,self.EMA_long,self.granularity,self.tc)
            self.results[["creturns","cstrategy"]].plot(title=title, figsize=(12,8))
            
        if plot_MACD is True:
            plt.figure(figsize=(15,5))
            plt.title("Price chart")
            plt.plot(self.data["price"])
            plt.show()
            
            plt.figure(figsize=(15,5))
            plt.title("MACD chart")
            plt.plot(self.data["MACD"])
            plt.axhline(0,linestyle="--")
            plt.show()
        