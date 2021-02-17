import numpy as np
import quandl


quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

'''
all_mid_prices = np.loadtxt('mid_data.txt', dtype=float)
print(all_mid_prices)'''

''' Data Collection'''

print('\n \n')
# Separates the dates from the metal price
def import_data(time_or_price, metal_data):
    metal = [] # creates array to append to and return
    count = 0
    while(count != (len(metal_data))):
        metal.append(metal_data[count][time_or_price])
        # Uses the time_or_price parameter to select column and individually appends the data into the array
        count = count + 1
    np.array(metal)
    return metal;

# Variables
date = 0
price = 1
# Imports data from quandl into an array
gold_data = quandl.get("LBMA/GOLD")
gold_usd_am = import_data(price,quandl.get('LBMA/GOLD', column_index='1', returns='numpy'))
gold_usd_pm = import_data(price,quandl.get('LBMA/GOLD', column_index='2', returns='numpy'))
metal_date = import_data(date,quandl.get('LBMA/GOLD', column_index='1', returns='numpy'))

print(np.array(gold_usd_am))
print('\n \n')
print(np.array(metal_date))
print('\n \n')
print(np.array(gold_usd_pm))
print('\n \n')