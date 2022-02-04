import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
import tpqoa

class RSI_BackTester:
    def __init__(self, symbol, start, end,granularity, period, tc):
        '''
        Compare to other back tester class, a hit ratio function is added
        '''
        self.symbol = symbol
        self.start = start 
        self.end = end
        self.granularity = granularity
        self.period = period
        self.tc = tc
        self.get_computed_data()
        self.results = None
    
    def computeRSI (self, data, time_window): #(data["price"],days)
        diff = data.diff(1).dropna()      
        #this preservers dimensions off diff values
        up_chg = 0 * diff
        down_chg = 0 * diff
        
        # up change is equal to the positive difference, otherwise equal to zero
        up_chg[diff > 0] = diff[ diff>0 ]
        # down change is equal to negative deifference, otherwise equal to zero
        down_chg[diff < 0] = diff[ diff < 0 ]

        # values are related to exponential decay
        # we set com=time_window-1 so we get decay alpha=1/time_window
        up_chg_avg   = up_chg.ewm(com=time_window-1 , min_periods=time_window).mean()
        down_chg_avg = down_chg.ewm(com=time_window-1 , min_periods=time_window).mean()

        rs = abs(up_chg_avg/down_chg_avg)
        rsi = 100 - 100/(1+rs)
        return rsi

    def get_computed_data(self):
        api = tpqoa.tpqoa("oanda.cfg")
        raw = api.get_history(instrument = self.symbol, start = self.start, end = self.end,
                             granularity = self.granularity, price = "M", localize = False)
        raw.drop(columns=["o","h","l","volume","complete"],inplace = True)
        raw.rename(columns={"c":"price"},inplace = True)
        raw["returns"] = np.log(raw/raw.shift(1))
        
        #prepare data 
        #Don't need EMA, but used for reference 
        #raw["EMA"] = raw["price"].ewm(span = self.ema_span, adjust = False).mean()
        #raw["SMA"] = raw["price"].rolling(50).mean()
        raw["RSI"] = self.computeRSI(raw["price"],self.period)
        
        self.data = raw
        return raw
        
    def set_parameters(self,period = None):
        if period is not None:
            self.period = period
            self.data["RSI"] = self.computeRSI(self.data["price"],self.period)
    
    def test_strategy(self):
        data = self.data.copy().dropna()
        data["position"]=np.where(data["RSI"]>70,-1,np.nan)
        data["position"]=np.where(data["RSI"]<30,1,data["position"])
        data["position"]=data.position.ffill().fillna(0)
        data["strategy"]=data.position.shift(1)*data["returns"]
        data.dropna(inplace = True)
        
        #determine when a trade takes place
        data["trades"] = data.position.diff().fillna(0).abs()
        
        #substract transaction costs from return when trade takes place
        data.strategy = data.strategy - data.trades * self.tc
        
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        #absolute performance of the strategy 
        perf = data["cstrategy"].iloc[-1]
        outperf = perf - data["creturns"].iloc[-1]
        
        return round(perf,6),round(outperf,6)
    
    def update_and_run(self,period):
        self.set_parameters(period)
        return self.test_strategy()[0]
    
    def optimise_parameters(self):
        '''
            self-defined with a range of 1 to 500
        '''
        max_ret = -99999
        opt = -99999
        for i in range(2,500):
            ret = self.update_and_run(i)
            if ret > max_ret:
                opt = i
                max_ret = ret

        return opt, self.update_and_run(opt)
                
    def hit_ratio(self):
        hits = np.sign(self.results.returns * self.results.position.shift(1)).value_counts()
        hit_ratio = hits[1.0] / sum(hits)
        return hit_ratio


    def plot_results(self,plot_RSI = False):
        if self.results is None:
            print("No results to plot yet. Run a strategy")
        else:
            title = "{} | period = {} | granularity = {} | TC = {}".format(self.symbol,
                                                                           self.period,self.granularity,self.tc)
            self.results[["creturns","cstrategy"]].plot(title=title, figsize=(12,8))
        
        if plot_RSI is True:
            plt.figure(figsize=(15,5))
            plt.title("Price chart")
            plt.plot(self.data["price"])
            plt.show()

            plt.figure(figsize=(15,5))
            plt.title("RSI chart")
            plt.plot(self.data["RSI"])
            plt.axhline(0,linestyle="--",alpha=0.1)
            plt.axhline(20,linestyle="--",alpha=0.5)
            plt.axhline(30,linestyle="--")
            plt.axhline(70,linestyle="--")
            plt.axhline(80,linestyle="--",alpha=0.5)
            plt.axhline(100,linestyle="--",alpha=0.1)
            plt.show()
