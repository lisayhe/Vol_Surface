import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm

from scipy import interpolate
#from scipy.stats import norm

def readNPlot():
        
    # file name
    excel_file = 'data_apple.xlsx'
    # read data into a data frame
    df = pd.read_excel(excel_file)
    # create the 'Mid' variable
    df['Mid'] = df[['Bid','Ask']].mean(axis=1)
    
    
    # see some of the data
    df.head()
    
    # define strikes and maturities
    #strikes = np.sort(df_calls.Strike.unique())
    strikes = np.arange(170., 210. + 2.5, 2.5)
    maturities = np.sort(df.Maturity_days.unique())
    #print(strikes)
    #print(maturities)
    
    df_calls = df[df['Option_type'] == 'Call'][['Maturity_days', 'Strike', 'Mid']]
    df_calls.head()
    
    # define a grid for the surface
    X, Y = np.meshgrid(strikes, maturities)
    callPrices = np.empty([len(maturities), len(strikes)])
    
    # we use linear interpolation for missing strikes
    for i in range(len(maturities)):
        s = df_calls[df_calls.Maturity_days == maturities[i]]['Strike']
        price = df_calls[df_calls.Maturity_days == maturities[i]]['Mid']
        f = interpolate.interp1d(s, price, bounds_error=False, fill_value=0)
        callPrices[i, :] = f(strikes) 
       
    # plot the surface
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    #ax.plot_wireframe(X, Y, callPrices, rstride=1, cstride=1)
    ax.plot_surface(X, Y, callPrices, cmap=cm.coolwarm)
    ax.set_ylabel('Maturity (days)') 
    ax.set_xlabel('Strike') 
    ax.set_zlabel('C(K, T)')
    ax.set_title('Apple Calls')
    #plt.savefig('appleCallSurface.png')
    plt.show()
    
    return maturities, strikes, callPrices

readNPlot()