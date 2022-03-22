import pandas as pd
import numpy as np
import tpqoa
from datetime import datetime, timedelta
import time
import keras
import pickle

class DNNTrader(tpqoa.tpqoa):
	def __init__(self,conf_file,instrument, bar_length, window,
	 lags, model, mu, std, units):
		super().__init__(conf_file) # so all info can be used across the whole class 
		self.instrument = instrument
		self.bar_length = pd.to_timedelta(bar_length)
		self.tick_data = pd.DataFrame()
		self.raw_data = None
		self.data = None 
		self.last_bar = None
		self.units = units
		self.position = 0
		self.profits = []

		#*****************add strategy-specific attributes here******************
		self.window = window
		self.lags = lags
		self.model = model
		self.mu = mu
		self.std = std
        #************************************************************************

	def get_most_recent(self,days = 6): #use 5 days to avoid long public holiday
		while True: #repeat until we get all historical bars 
			time.sleep(2)
			now = datetime.utcnow()
			now = now - timedelta(microseconds = now.microsecond)
			past = now - timedelta(days = days)
			df = self.get_history(instrument = self.instrument, start = past, end = now,
	                               granularity = "S5", price = "M", localize = False).c.dropna().to_frame()
			df.rename(columns = {"c":self.instrument}, inplace = True)
			#we use 5s because it is the most stable setting in oanda
			#when resample historical data, we use dropna() instead of ffill()
			#because there may be weekend in between, if we use live trading we can use ffill()
			df = df.resample(self.bar_length, label = "right").last().dropna().iloc[:-1] #get the last bar in that timeframe,exclude the last bar because not completed 
			self.raw_data = df.copy()
			self.last_bar = self.raw_data.index[-1]
			#accept, if less than [bar_length] has elapased since the last full historiacal bar and now 
			if pd.to_datetime(datetime.utcnow()).tz_localize("UTC") - self.last_bar < self.bar_length:
				# NEW -> Start Time of Trading Session 
				self.start_time = pd.to_datetime(datetime.utcnow()).tz_localize("UTC") 
				break

	def start_trading(self, days, max_attempts = 5, wait = 20, wait_increase = 0): #Error Handling
		attempt = 0
		success = False
		while True:
			try:
				self.get_most_recent(days)
				self.stream_data(self.instrument)
			except Exception as e:
				print(e, end = " | ")
			else:
				success = True 
				break
			finally:
				attempt += 1
				print("Attempt: {}".format(attempt), end = "\n")
				if success == False:
					if attempt >= max_attempts:
						print("max_attempts reached !")
						try: #try to terminate session
							time.sleep(wait)
							self.terminate_session(cause = "Unexpected Session Stop ( too many errors).")
						except Exception as e:
							print(e, end = " | ")
							print("Could not terminate_session properly!")
						finally:
							break
					else: # try again 
						time.sleep(wait)
						wait += wait_increase
						self.tick_data = pd.DataFrame()

	def on_success(self,time,bid,ask):
		print(self.ticks, end = " ")
		# collect and store tick data
		recent_tick = pd.to_datetime(time)

		#define stop because of low market violatility
		# if recent_tick.time() >= pd.to_datetime("17:30").time():
		# 	self.stop_stream = True

		############## Define Stop ##############
		# if self.ticks >= 100:
		# 	self.terminate_session(cause = "Scheduled Session End.")
		# 	return 
		#########################################

		df = pd.DataFrame({self.instrument:(ask+bid)/2},index = [recent_tick])
		self.tick_data = self.tick_data.append(df)
		#if a time longer than the bar_length has elapsed between last full bar and the most recent time
		if recent_tick - self.last_bar > self.bar_length:
			self.resample_and_join()
			self.define_strategy()
			self.execute_trades()

	def resample_and_join(self):
		#append the most recent ticks(resample) to self.data 
		self.raw_data = self.raw_data.append(self.tick_data.resample(self.bar_length,
			label="right").last().ffill().iloc[:-1])
		self.tick_data = self.tick_data.iloc[-1:] #only keep the latest tick(next bar)
		self.last_bar = self.raw_data.index[-1] #update time of last full bar 

	def define_strategy(self):
		df = self.raw_data.copy()
		#******************** define your strategy here ************************
		#create features
		df = df.append(self.tick_data) # append latest tick (== open price of current bar)
		df["returns"] = np.log(df[self.instrument] / df[self.instrument].shift())
		df["Dir"] = np.where(df["returns"] > 0, 1, 0)
		df["SMA"] = df[self.instrument].rolling(self.window).mean() - df[self.instrument].rolling(150).mean()
		df["Boll"] = (df[self.instrument] - df[self.instrument].rolling(self.window).mean()) / df[self.instrument].rolling(self.window).std()
		df["MACD"] = df[self.instrument].ewm(span = 12, adjust = False).mean() - df[self.instrument].ewm(span = 26, adjust = False).mean()
		df["Min"] = df[self.instrument].rolling(self.window).min() / df[self.instrument] - 1
		df["Max"] = df[self.instrument].rolling(self.window).max() / df[self.instrument] - 1
		df["Mom"] = df["returns"].rolling(3).mean()
		df["Vol"] = df["returns"].rolling(self.window).std()
		df.dropna(inplace = True)
		
		# create lags
		self.cols = []
		features = ["Dir", "SMA", "Boll", "Min", "Max", "Mom", "Vol"]

		for f in features:
			for lag in range(1, self.lags + 1):
				col = "{}_lag_{}".format(f, lag)
				df[col] = df[f].shift(lag)
				self.cols.append(col)
		df.dropna(inplace = True)
		
		# standardization
		df_s = (df - self.mu) / self.std
		# predict
		df["proba"] = self.model.predict(df_s[self.cols])
		
		#determine positions
		df = df.loc[self.start_time:].copy() # starting with first live_stream bar (removing historical bars)
		df["position"] = np.where(df.proba < 0.47, -1, np.nan)
		df["position"] = np.where(df.proba > 0.53, 1, df.position)
		df["position"] = df.position.ffill().fillna(0) # start with neutral position if no strong signal
		#***********************************************************************
		
		self.data = df.copy()

	def execute_trades(self):
		if self.data["position"].iloc[-1] == 1:
			if self.position == 0:
				order = self.create_order(self.instrument, self.units, suppress = True, ret = True)
				self.report_trade(order, "GOING LONG")
			elif self.position == -1:
				order = self.create_order(self.instrument, self.units * 2, suppress = True, ret = True) 
				self.report_trade(order, "GOING LONG")
			self.position = 1
		elif self.data["position"].iloc[-1] == -1: 
			if self.position == 0:
				order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
				self.report_trade(order, "GOING SHORT")
			elif self.position == 1:
				order = self.create_order(self.instrument, -self.units * 2, suppress = True, ret = True)
				self.report_trade(order, "GOING SHORT")
			self.position = -1
		elif self.data["position"].iloc[-1] == 0: 
			if self.position == -1:
				order = self.create_order(self.instrument, self.units, suppress = True, ret = True) 
				self.report_trade(order, "GOING NEUTRAL")
			elif self.position == 1:
				order = self.create_order(self.instrument, -self.units, suppress = True, ret = True)
				self.report_trade(order, "GOING NEUTRAL")
			self.position = 0

	def report_trade(self, order, going):
		time = order["time"]
		units = order["units"]
		price = order["price"]
		pl = float(order["pl"])
		self.profits.append(pl)
		cumpl = sum(self.profits)
		print("\n" + 100* "-")
		print("{} | {}".format(time, going))
		print("{} | units = {} | price = {} | P&L = {} | Cum P&L = {}".format(time, units, price, pl, cumpl))
		print(100 * "-" + "\n")

	def terminate_session(self, cause): #NEW
		self.stop_stream = True
		if self.position != 0:
			close_order = self.create_order(self.instrument, units = -self.position * self.units,
				suppress = True, ret = True)
			self.report_trade(close_order, "GOING NEUTRAL")
			self.position = 0
		print(cause)



#Only execute when directly called from terminal 
if __name__ == "__main__": 
	# Loading the model
	model = keras.models.load_model("DNN_model")

	# Loading mu and std
	params = pickle.load(open("params.pkl", "rb"))
	mu = params["mu"]
	std = params["std"]

	#initilise the class 
	trader = DNNTrader("oanda.cfg", "EUR_USD", bar_length = "15min",
	                   window = 50, lags = 5, model = model, mu = mu, std = std, units = 5000)

	# trader.get_most_recent()
	# trader.stream_data(trader.instrument)
	trader.start_trading(days=6)

	if trader.position != 0:
		close_order = trader.create_order(trader.instrument, units = -trader.position * trader.units, 
			suppress = True, ret = True)
		trader.report_trade(close_order,"Going NEUTRAL")
		trader.position = 0 
