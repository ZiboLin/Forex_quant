import pandas as pd 
import numpy as np 
import tpqoa as tpqoa 
import matplotlib.pyplot as plt 
from DNNModel import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
import pickle
plt.style.use("seaborn")
pd.set_option('display.float_format', lambda x: '%.5f' % x)

class DNN_Backtester():
	def __init__(self,symbol,start,end,granularity,window = 50,SMA_short = 10,SMA_long = 50,
    	EMA_short = 12, EMA_long = 26,tc = 0.00007,lags = 5, train_ratio = 0.7):
		self.symbol = symbol
		self.start = start 
		self.end = end
		self.train_ratio = train_ratio
		self.granularity = granularity
		self.tc = tc 
		self.lags = lags
		self.window	= window
		self.SMA_short = SMA_short
		self.SMA_long = SMA_long
		self.EMA_short = EMA_short    #For MACD index 
		self.EMA_long = EMA_long

		self.results = None
		self.get_data()
        
	def get_data(self):
		# raw = pd.read_csv("DNN_data.csv", parse_dates = ["time"], index_col = "time")
		# raw.rename(columns={"{}".format(self.symbol):"price"}, inplace = True)
		# raw["returns"] = np.log(raw["price"] / raw["price"].shift())

		##############################
		api = tpqoa.tpqoa("oanda.cfg")
		raw = api.get_history(instrument = self.symbol, start = self.start,end = self.end,
                             granularity = self.granularity, price = "M", localize = False)
		raw.drop(columns = ["o","h","l","volume","complete"], inplace = True)
		raw.rename(columns={"c":"price"},inplace = True)
		raw["returns"] = np.log(raw/raw.shift(1))
		##############################

		raw["Dir"] = np.where(raw["returns"] > 0, 1, 0)  #When position return = 1, negative return = 0
		raw["SMA"] = raw["price"].rolling(self.SMA_short).mean() - raw["price"].rolling(self.SMA_long).mean() 
		raw["Boll"] = (raw["price"] - raw["price"].rolling(self.window).mean())/raw["price"].rolling(self.window).std()
		raw["MACD"] = raw["price"].ewm(span = self.EMA_short, adjust = False).mean() - raw["price"].ewm(span = self.EMA_long, adjust = False).mean()
		raw["Min"] = raw["price"].rolling(self.window).min()/raw["price"] - 1
		raw["Max"] = raw["price"].rolling(self.window).max()/raw["price"] - 1
		raw["Mom"] = raw["returns"].rolling(3).mean()
		raw["Vol"] = raw["returns"].rolling(self.window).std()
		raw.dropna(inplace = True)
		############ optinal ############
		#raw["RSI"]
		#raw["KDJ"]
		############

		############ Prepare data ############
		#self.features = ["Dir", "SMA", "Boll", "MACD", "Min", "Max", "Mom", "Vol"]
		self.features = ["Dir", "SMA", "Boll", "Min", "Max", "Mom", "Vol"]
		self.feature_columns = []
		for f in self.features:
			for lag in range(1,self.lags + 1):
				col = "{}_lag_{}".format(f,lag)
				raw[col] = raw[f].shift(lag)
				self.feature_columns.append(col)

		raw.dropna(inplace = True)
		self.data = raw

	def test_strategy(self):
		full_data = self.data.copy()
		split = int(len(full_data) * self.train_ratio)
		train = full_data.iloc[:split].copy()
		test = full_data.iloc[split:].copy()
		mu,std = train.mean(), train.std() # train set parameters (mu, std) for standardization
		#for saving model
		self.save_mu = mu
		self.save_std = std

		train_s = (train - mu) / std # standardization of train set features
		test_s = (test - mu) / std  # standardization of test set features (with train set parameters!!!)

		set_seeds(100)
		# fitting a DNN model with 3 Hidden Layers (50 nodes each) and dropout regularization
		self.model = create_model(hl = 3, hu = 50, dropout = True, input_dim = len(self.feature_columns))
		self.model.fit(x = train_s[self.feature_columns], y = train["Dir"], epochs = 50, verbose = False,
			validation_split = 0.2, shuffle = False, class_weight = cw(train))
		#self.model.evaluate(test_s[self.feature_columns],train["Dir"])  # evaluate the fit on the train set
		
		####### test results ######
		self.model.evaluate(test_s[self.feature_columns],test["Dir"])
		test["proba"] = self.model.predict(test_s[self.feature_columns])
		test["position"] = np.where(test.proba < 0.47, -1, np.nan) # 1. short where proba < 0.47
		test["position"] = np.where(test.proba > 0.53, 1, test.position) # 2. long where proba > 0.53

		######  Only trade in high-voliotility time zone  ######
		# test.index = test.index.tz_localize("UTC")
		# test["NYTime"] = test.index.tz_convert("America/New_York")
		# test["hour"] = test.NYTime.dt.hour
		# test["position"] = np.where(~test.hour.between(2,12), 0, test.position) # 3. neutral in non-busy hours

		test["position"] = test.position.ffill().fillna(0)  # 4. in all other cases: hold position

		test["strategy"] = test["position"] * test["returns"]
		test["creturns"] = test["returns"].cumsum().apply(np.exp) # Without trading cost 
		test["cstrategy"] = test["strategy"].cumsum().apply(np.exp)

		test["trades"] = test.position.diff().abs()
		test["strategy_net"] = test.strategy - test.trades * self.tc 
		test["cstrategy_net"] = test["strategy_net"].cumsum().apply(np.exp) #With trading cost 

		self.results = test
		self.train_results = train 


		perf = self.results["cstrategy_net"].iloc[-1]
		outperf = perf - self.results["creturns"].iloc[-1]

		return round(perf,6), round(outperf,6)

	def plot_train_predict(self):
		#Predict the probabilities
		pred = self.model.predict(train_s[feature_columns])
		plt.hist(pred, bins = 50)
		plt.show()

	def plot_test_predict(self):
		pred = self.model.predict(test_s[feature_columns])
		plt.hist(pred, bins = 50)
		plt.show()

	def plot_results(self):
		if self.results is None:
			print("Run test_strategy() first")
		else:
			title = "DNN: {} | tc = {} | lags = {}".format(self.symbol,self.tc,self.lags)
			self.results[["creturns","cstrategy","cstrategy_net"]].plot(figsize = (12,8))
			plt.show()

	def hit_ratio(self):
		hits = np.sign(self.results.returns * self.results.position).value_counts()
		hit_ratio = hits[1.0]/sum(hits)
		return hit_ratio

	def save_model(self): 
		self.model.save("DNN_model")
		##########################
		params = {"mu":self.save_mu, "std":self.save_std}
		pickle.dump(params, open("params.pkl", "wb"))

