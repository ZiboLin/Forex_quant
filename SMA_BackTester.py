import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tpqoa
from scipy.optimize import brute

#(1)SMA class
# Once finish, optimise the SMA_S and SMA_L so that it produce the best result 
class SMABackTester():
    def __init__(self, symbol, SMA_S, SMA_L, start, end, tc, granularity):
        self.symbol = symbol
        self.SMA_S = SMA_S
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.tc = tc
        self.granularity = granularity   
        self.results = None

        self.get_data()

        self.prepare_data()
#         else:
#             self.data = data
#             self.data["returns"] = np.log(raw/raw.shift(1))

    def __repr__(self):
        return "SMABackTester(symbol = {}, SMA_S = {}, SMA_L ={}, start= {}, end = {})".format(self.symbol, self.SMA_S, self.SMA_L, self.start, self.end)
    
    def get_data(self):
        ''' Import data using Oanada api
        '''
        #Get data directly from oanda api with granularity 
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument = self.symbol, start = self.start, end = self.end,
                             granularity = self.granularity, price = "M", localize = False)
        raw.drop(columns=["o","h","l","volume","complete"],inplace = True)
        # If resample is needed
        #raw = raw.resample("20min",label = "right").last().dropna().iloc[:-1]
        #
        raw.rename(columns={"c":"price"},inplace = True)
        raw["returns"] = np.log(raw/raw.shift(1))
        self.data = raw #self.data is not defined until get_data() called
        
    def prepare_data(self):
        ''' Prepares the data for SMA strategy backtesting
        '''
        data = self.data.copy()
        data["SMA_S"] = data["price"].rolling(self.SMA_S).mean()
        data["SMA_L"] = data["price"].rolling(self.SMA_L).mean()
        self.data = data 
    
    def set_parameters(self, SMA_S = None, SMA_L = None):
        ''' Updates SMA parameters and then the prepared dataset
        '''
        if SMA_S is not None:
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L is not None:
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
            
    def test_strategy(self):
        data = self.data.copy().dropna()
        #data["position"] = int(-1) #not necessary, added for testing purpose
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        #return from the strategy 
        data["strategy"] = data["position"].shift(1)*data["returns"]
        data.dropna(inplace = True) #maybe not neccessary 
        
        #determine when a trade takes place 
        data["trades"] = data.position.diff().fillna(0).abs()
             
        #subtract trading cost from returns when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        #calculate the absolute performance of the strategy 
        perf = data["cstrategy"].iloc[-1]
        #out-/underperformance of strategy 
        outperf = perf - data["creturns"].iloc[-1]
        return round(perf,6), round(outperf,6)
    
    def plot_results(self):
        if self.results is None:
            print("No results to plot yet, run test_strategy() first")
        else:
            title = "{} | SMA_S = {} | SMA_L = {} | TC = {}".format(self.symbol, self.SMA_S, self.SMA_L, self.tc)
            self.results[["creturns","cstrategy"]].plot(title = title, figsize=(12,8))
        
    def update_and_run(self, SMA):
        ''' Updates SMA parameters using tuple and returns the negative absolute performance
            for optimise_paramters()
        '''
        self.set_parameters(int(SMA[0]), int(SMA[1]))
        return -self.test_strategy()[0] #only return the performance 
    
    def optimise_parameters(self, SMA1_range, SMA2_range):
        '''pass in SMA1_range, SMA2_range: tuple
        tuple of the form(start, end, step size)
        '''
    
        opt = brute(self.update_and_run,(SMA1_range,SMA2_range), finish=None)
        return opt, -self.update_and_run(opt) #why return negative?