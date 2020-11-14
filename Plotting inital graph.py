
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import *
import quandl
from matplotlib.font_manager import FontProperties
quandl.ApiConfig.api_key = 'Gv9VqzUx_24QFyuG267H'

print ("Defining instruments ...")

plt.style.use('ggplot')
fig=plt.figure()

# data
''' Yearly Data'''
# series = quandl.get("LBMA/GOLD",trim_start="1979-01-01", trim_end="2019-11-01", collapse="annual") 
series = quandl.get("LBMA/GOLD")
mid_prices= []
count = 0

# Getting Open and close data
gold_usd_open = series['USD (AM)']
gold_usd_close = np.nan_to_num(series['USD (PM)'], nan=0.0)

# Getting mid values the open and close data
while (count < len(gold_usd_open)):
    temp = (gold_usd_open[count] + gold_usd_close[count])/2.0

    if(gold_usd_open[count] == 0 or gold_usd_close[count] == 0 ):
        mid_prices.append(temp*2.0)

    else:
        mid_prices.append(temp)
    count = count + 1


print ("Plotting ...")

dateaxis = series.reset_index()['Date']
plt.plot(dateaxis, mid_prices)
plt.xlabel("Year")
plt.ylabel("Prices")
myFont = FontProperties()
plt.show()
