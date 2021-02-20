import numpy as np
import quandl


quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

'''
all_mid_prices = np.loadtxt('mid_data.txt', dtype=float)
print(all_mid_prices)'''

''' Data Collection'''

print('\n \n')
# Import table data
gold_data = quandl.get("LBMA/GOLD", returns='numpy')
mid_prices = []
count = 0

# Getting Open and close data
gold_usd_open = np.nan_to_num(gold_data[6])

gold_usd_close = gold_data['USD (PM)']

dateindex = gold_data['Date']

print(gold_usd_open)


# Getting mid values the open and close data
while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count]) / 2.0 # Typical mid point between two data entries
    if (gold_usd_open[count] == 0 or gold_usd_close[count] == 0):
        mid_prices.append(temp * 2.0) # if one data entries for either morn or even is not entered the data entry that was entered will be taken
    else: mid_prices.append(temp)
    count += 1

count = 0

# rare case if both data entries are 0 then the previous and next data entry will act as the data points
while(count < len(mid_prices)-1):
    if (mid_prices[count] == 0):
        temp = (mid_prices[count-1] + mid_prices[count+1])/2.0
        mid_prices[count] = temp

    count += 1



'''
print(np.array(mid_prices))
print('\n \n')
'''

