# Ricequant量化交易平台
# 日期：2013-01-01到 2016-10-04，日回测
# 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
 
# 在这部分编写技术分析模块
def RSI(N1=6, N2=12, N3=24):
    """
    RSI 相对强弱指标
    """
    LC = REF(CLOSE, 1)
    RSI1 = SMA(MAX(CLOSE - LC, 0), N1, 1) / SMA(ABS(CLOSE - LC), N1, 1) * 100
    RSI2 = SMA(MAX(CLOSE - LC, 0), N2, 1) / SMA(ABS(CLOSE - LC), N2, 1) * 100
    RSI3 = SMA(MAX(CLOSE - LC, 0), N3, 1) / SMA(ABS(CLOSE - LC), N3, 1) * 100
 
    return RSI1, RSI2, RSI3
 
# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
def init(context):
    reg_indicator('RSI', RSI, '1d', win_size=40)
    context.buy = []
    context.sell = []
    context.hold = []
    context.s_sell = []
 
# before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
def before_trading(context):
    context.buy = []
    context.sell = []
    context.hold = []
    stocks,_ = get_all_stocks(context)
     
    for stock in context.portfolio.positions.keys():
        RSI1,RSI2,RSI3 = get_indicator(stock, 'RSI')
        if RSI1>80 and REF(RSI1,1) > REF(RSI2,1) and RSI1 < RSI2:
            context.sell.append(stock)
        else:
            context.hold.append(stock)
     
    for stock in context.s_sell:
        if stock not in context.portfolio.positions.keys():
            context.s_sell.remove(stock)
             
    if len(context.hold) >= 10:
        return None
         
    for stock in stocks:
        RSI1,RSI2,RSI3 = get_indicator(stock, 'RSI')
        if RSI1<20 and REF(RSI1,1) < REF(RSI2,1) and RSI1 > RSI2:
            context.buy.append(stock)
            if stock in context.sell:
                context.sell.remove(stock)
 
 
# 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
def handle_bar(context, bar_dict):
    for stock in context.sell:
        order = order_target_percent(stock,0)
        if order.unfilled_quantity != 0:
            context.s_sell.append(stock)
    for stock in context.s_sell:
        order_target_value(stock,0)
     
    if len(context.hold)+len(context.buy) == 0:
        return None
    weight = 1/(len(context.hold)+len(context.buy))
     
    for stock in context.hold:
        order_target_percent(stock,weight)
    for stock in context.buy:
        order_target_percent(stock,weight)
 
# after_trading函数会在每天交易结束后被调用，当天只会被调用一次
def after_trading(context):
    pass
 
def get_all_stocks(context):
    all_stocks = all_instruments("CS").order_book_id
    will_end = []
    trade = []
    for stock in all_stocks:
        ins = instruments(stock)
        if ins is None:
            pass
        else:
            start = ins.listed_date
            end = ins.de_listed_date
            if (start-context.now).days < 0:
                if 0< (end - context.now).days <30:
                    will_end.append(stock)
                elif 30 < (end - context.now).days and is_suspended(stock) == False:
                    trade.append(stock)
    #print(len(trade),len(will_end))
    return trade,will_end