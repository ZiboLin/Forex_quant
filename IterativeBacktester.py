#IterativeBacktester.py 
from IterativeBase import *

class IterativeBacktester(IterativeBase):
	''' Class for iterative(event-driven) bakctesting of trading strategies.
	'''

	#helper method 
	def go_long(self,bar,units = None, amount = None):
		if self.position == -1:
			self.buy_instrument(bar, units = -self.units) #if short position, go neutral fist
		if units:
			self.buy_instrument(bar,units = units)
		elif amount:
			if amount == "all":
				amount = self.current_balance
			self.buy_instrument(bar,amount = amount) # go long

	#helper method
	def go_short(self,bar,units = None, amount = None):
		if self.position == 1:
			self.sell_instrument(bar, units = self.units) #if long position, go neutral first
		if units:
			self.sell_instrument(bar, units = units)
		elif amount:
			if amount == "all":
				amount = self.current_balance
			self.sell_instrument(bar, amount = amount) # go short

	def test_sma_strategy(self, SMA_S, SMA_L):
		stm = "Testing SMA strategy | {} | SMA_S = {} & SMA_L = {}".format(self.symbol,SMA_S,SMA_L)
		print("-" * 75)
		print(stm)
		print("-" * 75)

		#reset 
		self.position = 0 #initial neutral position 
		self.trades = 0 
		self.current_balance = self.initial_balance
		self.get_data() #reset dataset 

		#prepare data 
		self.data["SMA_S"] = self.data["price"].rolling(SMA_S).mean()
		self.data["SMA_L"] = self.data["price"].rolling(SMA_L).mean()
		self.data.dropna(inplace = True)

		for bar in range(len(self.data)-1): #all bars
			if self.data["SMA_S"].iloc[bar] > self.data["SMA_L"].iloc[bar]: #signal to go long
				if self.position in [0,-1]:
					self.go_long(bar,amount="all") # go long with full amount 
					self.position = 1 #long position
			elif self.data["SMA_S"].iloc[bar] < self.data["SMA_L"].iloc[bar]:
				if self.position in [0,1]:
					self.go_short(bar,amount = "all") 
					self.position = -1
		self.close_pos(bar+1) #close position at the last bar 

	def test_con_strategy(self,window = 1):
		stm = "Testing Contrarian strategy | {} | Window = {}".format(self.symbol,window)
		print("-" * 75)
		print(stm)
		print("-" * 75)

		#reset
		self.position = 0 
		self.trades = 0 
		self.current_balance = self.initial_balance 
		self.get_data()

		#prepare data
		self.data["rolling_returns"] = self.data["returns"].rolling(window).mean()
		self.data.dropna(inplace = True)

		#Contrarian strategy
		for bar in range(len(self.data)-1): 
			if self.data["rolling_returns"].iloc[bar] <= 0: #signal to go long
				if self.position in [0,-1]:
					self.go_long(bar, amount = "all")
					self.position = 1
			elif self.data["rolling_returns"].iloc[bar] >0: #signal to go short
				if self.position in [0,1]:
					self.go_short(bar, amount = "all") #go short with full amount
					self.position = -1 
		self.close_pos(bar+1)

	def test_boll_strategy(self, SMA, dev):
		stm = "Testing Bollinger Bands Strategy | {} | SMA = {} & dev = {}".format(self.symbol, SMA, dev)
		print("-" * 75)
		print(stm)
		print("-" * 75)

		#reset 
		self.position = 0 
		self.trades = 0 
		self.current_balance = self.initial_balance 
		self.get_data()

		#prepare data 
		self.data["SMA"] = self.data["price"].rolling(SMA).mean()
		self.data["Lower"] = self.data["SMA"] - self.data["price"].rolling(SMA).std() * dev
		self.data["Upper"] = self.data["SMA"] + self.data["price"].rolling(SMA).std() * dev
		self.data.dropna(inplace = True)

		#Bollinger strategy
		for bar in range(len(self.data)-1): 
			if self.position == 0: 
				if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]:
					self.go_long(bar, amount = "all") #go long with full amount
					self.position = 1
				elif self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]:
					self.go_short(bar, amount = "all") 
					self.position = -1
			elif self.position == 1: # When long
				if self.data["price"].iloc[bar] > self.data["SMA"].iloc[bar]:
					if self.data["price"].iloc[bar] > self.data["Upper"].iloc[bar]:
						self.go_short(bar,amount = "all")
						self.position = -1
					else:
						self.sell_instrument(bar, units = self.units)
						self.position = 0 
			elif self.position == -1:
				if self.data["price"].iloc[bar] < self.data["SMA"].iloc[bar]:
					if self.data["price"].iloc[bar] < self.data["Lower"].iloc[bar]:
						self.go_long(bar, amount = "all") 
						self.position = 1
					else: 
						self.buy_instrument(bar, units = -self.units)
						self.position = 0 
		self.close_pos(bar+1)























