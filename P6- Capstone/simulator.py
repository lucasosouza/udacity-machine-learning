import quandl
import pandas_datareader.data as web
import datetime
import pandas as pd
import sklearn
import numpy as np
import scipy as sp
from operator import methodcaller
import time

class Operation():
    def __init__(self, symbol, price, qty, start_date, span, end_date=None):
        self.symbol = symbol
        self.price = price
        self.qty = qty
        self.start_date = start_date
        self.end_date = end_date
        self.gain_loss = 0
        self.days_left = span
        
    def close(self, end_date, sell_price):
        self.end_date = end_date
        self.gain_loss = (sell_price / self.price) -1

    def export(self):
        return {
            'symbol': self.symbol,
            'price': self.price,
            'qty': self.qty,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'span': self.days_left
        }

    
    def report(self):
        print("Symbol: {}, Start: {}, End: {}, Gain_loss: {:.2f}%, R$ {:.2f}, Days Left: {}".format(
                self.symbol, self.start_date, self.end_date, self.gain_loss*100, 
                self.price*self.qty*self.gain_loss, self.days_left))

class Analyst():
    def __init__(self, symbol, data, clf, scaler, pca, score):
        self.symbol = symbol
        self.data = data.copy()
        self.clf = clf
        self.scaler = scaler
        self.pca = pca
        self.score = score
            
    def predict(self, days):
        """ Predict for all days in advance """
        X = self.data.loc[days, :].values	
        # this is the only line that could justify a data leakage - no partial fit for now
        # self.scaler.partial_fit(X)
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        y_pred = self.clf.predict(X)
        self.signals = pd.DataFrame(y_pred, index=days)

    def check_signal(self, day):
        return self.signals.loc[day].item()
    
    def get_var(self, day, var):
        """ Return any var of the dataset """
        return self.data.loc[day, var]
                
class Operator():
    def __init__(self, strategy, analysts, capital=0, start_date='2015-01-01', end_date='2015-12-31', op_days=[]):
        self.strategy = strategy
        self.analysts = analysts
        self.capital = capital
        self.period = [d for d in pd.date_range(start=start_date, end=end_date, freq='D') if d in op_days]
        self.operations = []

    def sort_analysts(self):
        self.analysts = sorted(self.analysts, key=lambda a: a.score, reverse=True)
        
    def run(self):
        # sort analysts by score, and request prediction
        self.sort_analysts()
        for analyst in self.analysts:
            analyst.predict(self.period)
        for day in self.period:
            # check if there any open operations that needs to be closed
            self.check_operations(day)
            # wrap up if last daydd
            if day == self.period[-1]:
                for operation in self.operations:
                    if not operation.end_date:
                        self.sell(day, operation)
            # if not, proceed with trading strategy
            else:
                # loop through all analysts
                for analyst in self.analysts:
                    if analyst.check_signal(day):
                        if self.capital > 0:
                            self.buy(day, analyst.symbol)                        

    def check_operations(self, day):
        for operation in self.operations:
            analyst = next(filter(lambda a:a.symbol == operation.symbol, self.analysts))
            span, profit, loss = self.strategy
            if not operation.end_date:
                # remove one more day
                operation.days_left -= 1
                # calc valuation, for that specific symbol
                valuation = (analyst.get_var(day, 'Adj Close') / operation.price)-1
                # sell if it reaches the target or the ends the number of days
                if valuation >= profit or valuation <= loss or operation.days_left<=0:
                    self.sell(day, operation)

    def buy(self, day, symbol):
        span, _, _ = self.strategy
        analyst = next(filter(lambda a:a.symbol == symbol, self.analysts))
        price = analyst.get_var(day, 'Adj Close')
        qty = self.capital / price
        # update capital
        self.capital -= qty * price
        # open operation
        self.operations.append(
            Operation(symbol = symbol, price = price, qty = qty, start_date = day, span=span))
        # change start and end date of operations
        
    def sell(self, day, operation):
        analyst = next(filter(lambda a:a.symbol == operation.symbol, self.analysts))
        price = analyst.get_var(day, 'Adj Close')
        # print('triggered at: {:.2f}, selling at: {:.2f}'.format(self.data.loc[day, 'Adj Close'], price))
        # update capital
        self.capital += operation.qty * price
        # close operation
        operation.close(day, price)

"""    
    # 'realistic version'. buy and sell prices are the opening prices for the next day, not the closing price. don't have access to adjusted open value
    
    def buy(self, day, symbol):
        span, _, _ = self.strategy
        analyst = next(filter(lambda a:a.symbol == symbol, self.analysts))
        next_day = self.period[self.period.index(day) + 1]
        price = analyst.get_var(next_day, 'Open')
        # print('triggered at: {:.2f}, buying at: {:.2f}'.format(self.data.loc[day, 'Adj Close'], price))
        qty = self.capital / price
        # update capital
        self.capital -= qty * price
        # open operation
        self.operations.append(
            Operation(symbol = symbol, price = price, qty = qty, start_date = day, span=span))
        # change start and end date of operations
        
    def sell(self, day, operation):
        analyst = next(filter(lambda a:a.symbol == operation.symbol, self.analysts))
        # get open price for the next day. if last day, get closing price for the day
        if day == self.period[-1]:
            price = analyst.get_var(day, 'Close')
        else:
            next_day = self.period[self.period.index(day) + 1]
            price = analyst.get_var(next_day, 'Open')
        # print('triggered at: {:.2f}, selling at: {:.2f}'.format(self.data.loc[day, 'Adj Close'], price))
        # update capital
        self.capital += operation.qty * price
        # close operation
        operation.close(day, price)


    def check_signal(self, day):
        # get X
        X = self.get_X(day)
        # scale and extract principal components
        self.scaler.partial_fit(self.get_X(day))
        X = self.scaler.transform(X)
        X = self.pca.transform(X)
        # get label
        label = self.clf.predict(X)
        return label
"""        