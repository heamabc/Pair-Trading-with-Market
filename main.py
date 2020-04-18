# import library
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import re
import math	
from sklearn.decomposition import PCA
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import warnings
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")

#==================================================== Parameters ==================================================================
lookback_list = [120]
entry_level_list = [1.5, 1.6]
exit_level_list = [0, 0.5]
capital = 10000000
transaction_cost = 0.03
input_directory = r'/kaggle/input/sp500fina4380/data.csv'
output_directory = '/content'
end_date = '12/31/2010'
#==================================================== Parameters ==================================================================



# List all tickers
regex_pat = re.compile(r'_.*')
Tickers = data.columns.str.replace(regex_pat, '').unique()

def data_cleaning(end_date):
    # Read data and slicing
    data = pd.read_csv(os.path.join(dirname, filename), index_col='date')
    data = data.loc[:end_date]

    # Calculate effect of split
    splitFactor_cols = [ele + '_splitFactor' for ele in Tickers]
    splitFactor_df = data[splitFactor_cols]
    splitFactor_df = splitFactor_df.cumprod()
    splitFactor_df.columns = Tickers

    # Calculate effect of split cash dividend
    divCash_cols = [ele + '_divCash' for ele in Tickers]
    divCash_df = data[divCash_cols]
    divCash_df = divCash_df.cumsum()
    divCash_df.columns = Tickers

    # Slice open, close, volumne df
    open_cols = [ele + '_open' for ele in Tickers]
    close_cols = [ele + '_close' for ele in Tickers]
    volume_cols = [ele + '_volume' for ele in Tickers]

    open_df = data[open_cols]
    close_df = data[close_cols]
    volume_df = data[volume_cols]

    open_df.columns = Tickers
    close_df.columns = Tickers
    volume_df.columns = Tickers

    open_df = open_df * splitFactor_df + divCash_df * splitFactor_df
    close_df = close_df * splitFactor_df + divCash_df * splitFactor_df
    volume_df = volume_df * splitFactor_df

    return open_df, close_df, volume_df



def backtester(capital, transaction_cost, beta, signal, return_df):
    # Calculate weight for all stocks
    weight = beta.copy()
    weight[weight.notnull()] = 1
    weight = (weight.T/beta.count(axis=1)).T
    weight['SPY'] = weight['TT']

    # Position
    position = signal.cumsum()
    
    
    # Actual beta position on SPY
    beta_position = pd.DataFrame(0, columns = signal.columns, index=signal.index)

    # For position of stock, capture the beta of the stock when the position start. Then the position for SPY = position of stock * -1 * beta captured
    for col in beta_position:
        tmp = position[col]
        lists = np.split(tmp,np.where(tmp == 0)[0])
        lists = [ele[1:] for ele in lists if len(ele) > 1]

        first_signal = [ele[0] for ele in lists]
        first_ind = [ele.index[0] for ele in lists]

        for j in range(len(first_ind)):
            beta_position[col].loc[lists[j].index] = beta[col].loc[first_ind[j]] * first_signal[j] * -1
            
    # Position for SPY
    position['SPY'] = beta_position.sum(axis=1)
    
    # Transacation Cost
    # 1/n changed
    # 
    transaction_cost_df = position.iloc[:,:-1].copy()
    transaction_cost_df = pd.DataFrame(np.where((weight.iloc[:,:-1].diff() != 0) | (transaction_cost_df != transaction_cost_df.shift(1)), -transaction_cost,0), columns = Tickers[:-1], index = signal.index)
    transaction_cost_df['SPY'] = np.where((position['SPY'].diff() != 0) | (weight['TT'].diff() != 0), -transaction_cost, 0)

    # return minus transaction cost
    return_with_tc = return_df.loc[signal.index] + transaction_cost_df

    # Individual stocks return
    daily_return = return_with_tc * position
    cum_return = (daily_return + 1).cumprod()

    # Portfolio return
    port_daily_return = (daily_return * weight).sum(axis=1)
    port_cum_return = (port_daily_return + 1).cumprod()

    # Portfolio value
    port_value = port_cum_return * capital

    # By Pari return
    pairs_SPY_return = (beta_position.T * return_with_tc['SPY']).T
    pairs_daily_return =  pairs_SPY_return + daily_return
    pairs_cum_return = (pairs_daily_return + 1).cumprod()

    # Relationship : pairs_SPY_return.sum(axis=1) == daily_return['SPY']
    
    
    return daily_return, cum_return, pairs_daily_return, pairs_cum_return, port_daily_return, port_cum_return, port_value

# time series of the open price
def transform_ln_price(open_df):
    open_df_transpose = open_df.T

    def price_0_func(x):
        if (open_df_transpose.values[0] is None) or (x.values[0] is None):
            return None
        else:
            return open_df_transpose.loc[x.name, x.values[0]]

    open_orignal_df = pd.DataFrame(open_df_transpose.apply(lambda x: x.first_valid_index(), axis=1)).apply(price_0_func,axis=1)

    ln_open_df = np.log(open_df/open_orignal_df)
    return ln_open_df

# generating signals based on OLS regression of every stock against SPY
def compute_betas_by_ols_stock(stock,SPY):
    SPY = sm.add_constant(SPY, prepend=True)
    ols = sm.OLS(stock, SPY).fit()
    resid = sm.OLS(stock,SPY).fit().resid
    if np.isnan(stock).any() == True:
        return ols.params[1],resid,1
    else: 
        return ols.params[1],resid, ts.adfuller(resid)[1]
    #returns betas, residuals, p-values for ADF test
# series is the our formation dataframe, we need to only include relevant dates here
def compute_betas_by_ols(series, SPY):
    fittedvalues= np.zeros((1, series.shape[1]))
    residuals = np.zeros((lookback, series.shape[1]))
    #alphas = []
    pvalues= np.zeros((1, series.shape[1]))
    for ii in range(0, series.shape[1]):
        f, r, p= compute_betas_by_ols_stock(series.iloc[:,ii],SPY)
        fittedvalues[:, ii] = f
        residuals[:, ii] = r
        pvalues[:,ii] = p
        #alphas.append(a)
        #pvalues[ii, :] = p
    return fittedvalues, residuals, pvalues#, alphas, pvalues
def generate_signals(date, lookback, entry_level, exit_level):
    date_row = ln_open_df.index.get_loc(date)
    if date_row<= lookback:
        print ('The stated lookback period is larger than the earliest entry in the data')
        return
    df = ln_open_df.iloc[date_row-lookback:date_row]
    betas,residuals,pvalues = compute_betas_by_ols(df.iloc[:,:-1],df.iloc[:,-1]) 
    residual_sd = residuals.std(axis = 0)
    residual_mean = residuals.mean(axis = 0) # just for sanity check, mean should be = 0
    today_residual = residuals[-1]
    today_sscore = (today_residual-residual_mean)/residual_sd #creating s-score for each stock, sscore is a vector for one day
    today_signal,yesterday_exposure = signal_from_residual(today_sscore,pvalues, entry_level, exit_level)
    today_exposure = yesterday_exposure+today_signal
    global exposure
    exposure = np.vstack((exposure,today_exposure))
    return today_signal, betas,pvalues, today_sscore 
def run_strategy(startdate,enddate,lookback, entry_level, exit_level):
    startdate_row = ln_open_df.index.get_loc(startdate)
    enddate_row = ln_open_df.index.get_loc(enddate)
    for i in tqdm(range(startdate_row, enddate_row+1)):
        today_signal, today_betas,today_pvalues, today_sscore = generate_signals(ln_open_df.index[i], lookback, entry_level, exit_level)
        if i == startdate_row:
            all_signals = today_signal
            all_betas = today_betas
            all_pvalues = today_pvalues
            all_sscores= today_sscore
        else: 
            all_signals = np.vstack((all_signals,today_signal))
            all_betas = np.vstack((all_betas,today_betas))
            all_pvalues = np.vstack((all_pvalues,today_pvalues))
            all_sscores = np.vstack((all_sscores,today_sscore))
    global exposure
    exposure = exposure[1:]
    return all_signals,exposure,all_betas, all_pvalues, all_sscores
def signal_from_residual(today_sscore,pvalues, entry_level, exit_level):
    signal = []
    if len(exposure.shape) == 1:
        yesterday_exposure = exposure
    else:
        yesterday_exposure = exposure[-1]
    for i in range(len(today_sscore)):
        if pvalues[0,i] >=0.05 and yesterday_exposure[i] == 1:
            signal.append(-1)
        elif pvalues[0,i] >=0.05 and yesterday_exposure[i] == -1:
            signal.append(1)
        elif pvalues[0,i]>= 0.05:
            signal.append(0)
        elif today_sscore[i]>entry_level and yesterday_exposure[i]==0:
            signal.append(1)
        elif today_sscore[i]<-entry_level and yesterday_exposure[i]==0:
            signal.append(-1)
        elif today_sscore[i]<= exit_level and yesterday_exposure[i]==1:
            signal.append(-1)
        elif today_sscore[i]>= exit_level and yesterday_exposure[i]==-1:
            signal.append(1)
        else:
            signal.append(0)
    return signal, yesterday_exposure

# Convert the numpy array result to Dataframe with the same index and columns
def transform_result(df, ln_open_df, startdate, enddate):
    #  Exclude SPY column, and slice the index to the backtesting period
    return pd.DataFrame(df, columns = Tickers[:-1], index = ln_open_df.index[ln_open_df.index.get_loc(startdate):ln_open_df.index.get_loc(enddate)+1])


def sharpe(port_daily_return, pairs_daily_return):
    
    sharpe_port = np.sqrt(252) * (port_daily_return.mean() / port_daily_return.std())
    sharpe_pairs = np.sqrt(252) * (pairs_daily_return.mean() / pairs_daily_return.std()) 
    
    return sharpe_pairs,sharpe_port

def drawdown(pairs_cum_return, port_cum_return):
    # Daily drawdown for pairs
    expanding_Max = pairs_cum_return.expanding(min_periods=1).max()
    Daily_Drawdown = pairs_cum_return/expanding_Max - 1.0
    
    # Find the maximum drawdown for every day
    Max_Daily_Drawdown = Daily_Drawdown.expanding(min_periods=1).min()
    
    # Daily drawdown for portfolio
    expanding_Max_port = port_cum_return.expanding(min_periods=1).max()
    Daily_Drawdown_port = port_cum_return/expanding_Max_port.values - 1.0
    
    # Find the maximum drawdown for every day
    Max_Daily_Drawdown_port = Daily_Drawdown_port.expanding(min_periods=1).min()
    
    return Max_Daily_Drawdown, Max_Daily_Drawdown_port

def performance(pairs_cum_return, pairs_daily_return,port_cum_return,port_daily_return):
    data = pairs_cum_return.reset_index()
    d = data.to_numpy()
    annualized_return = []
    df = DataFrame(pairs_daily_return)
    annualized_volatility = df.std()*np.sqrt(252)
    ddd = DataFrame(port_cum_return)
    firstt = ddd.first_valid_index()
    lastt = ddd.last_valid_index()

    return_periodd = ddd.loc[firstt:lastt]
    annualized_return_port = ((ddd.iloc[-1] )**(365/return_periodd.shape[0]))-1
    dff = DataFrame(port_daily_return)
    annualized_volatility_port = dff.std()*np.sqrt(252)

    for i in range(1,d.shape[1]):
        a = np.column_stack((d[:,0],d[:,i]))
        a = DataFrame(a)
        a=a.rename(columns={0: "Date"})
        a = a.set_index('Date')
        first = a.first_valid_index()
        last = a.last_valid_index()
        return_period = a.loc[first:last,1]
        if last == None:
            annualized_return.append(None)
            continue
        cumu = a.loc[last].iloc[0] - 1
        this = ((1 + cumu )**(365/return_period.shape[0]))-1
        annualized_return.append(this)

    annualized_return = DataFrame(annualized_return)
    dd = data.drop(['date'],axis = 1)
    name = dd.columns
    annualized_return['name'] = name
    annualized_return = annualized_return.set_index('name')
    return annualized_return,annualized_volatility,annualized_return_port,annualized_volatility_port

def generate_output(directory):
    # Transformation
    performance_df = pd.DataFrame()
    performance_df['Sharpe'] = sharpe_pairs
    performance_df['Annualized_return'] = annualized_return
    performance_df['Annualized_volatility'] = annualized_volatility
    performance_df['Maximum_drawdown'] = max_dd

    # Add portfolio metrics 
    Max_Daily_Drawdown['Portfolio'] = Max_Daily_Drawdown_port
    daily_return['Portfolio'] = port_daily_return
    cum_return['Portfolio'] = port_cum_return
    
    # our parameters set
    params = (lookback, entry_level, exit_level)
    
    # Save results to excel
    writer = pd.ExcelWriter(directory + "/" + str(params) + '.xlsx')
    
    daily_return.to_excel(writer, sheet_name = str(params) + " daily_return")
    cum_return.to_excel(writer, sheet_name = str(params) + " cum_return")
    pairs_daily_return.to_excel(writer, sheet_name = str(params) + " pairs_daily_return")
    pairs_cum_return.to_excel(writer, sheet_name = str(params) + " pairs_cum_return")
    Max_Daily_Drawdown.to_excel(writer, sheet_name = str(params) + " Max_Drawdown")
    performance_df.to_excel(writer, sheet_name = str(params) + " performance")
    
    writer.save()
    
    return

#==================================================== Start Backtest ==================================================================
open_df, close_df, volume_df = data_cleaning(end_date)
return_df = open_df.pct_change()
ln_open_df = transform_ln_price(open_df)


for lookback in lookback_list:
    for entry_level in entry_level_list:
        for exit_level in exit_level_list:
            
            # Setting variables for backtesting
            exposure = np.zeros(len(Tickers)-1)
            startdate = ln_open_df.iloc[lookback + 1].name
            enddate = ln_open_df.iloc[-1].name
            
            # Generate Signal, exposure, betas, pvalues, standard scores of our strategy
            signals, exposure, betas, pvalues, sscores = run_strategy(startdate,enddate,lookback, entry_level, exit_level)
            signal_df = transform_result(signals, ln_open_df, startdate, enddate)
            exposure_df = transform_result(exposure, ln_open_df, startdate, enddate)
            betas_df = transform_result(betas, ln_open_df, startdate, enddate)
            pvalues_df = transform_result(pvalues, ln_open_df, startdate, enddate)
            sscores_df = transform_result(sscores, ln_open_df, startdate, enddate)
            
            # Generate returns of our strategy
            daily_return, cum_return, pairs_daily_return, pairs_cum_return, port_daily_return, port_cum_return, port_value = backtester(capital, transaction_cost, betas_df, signal_df, return_df)
            
            # Generate metrics for our strategy
            sharpe_pairs,sharpe_port = sharpe(port_daily_return, pairs_daily_return)
            Max_Daily_Drawdown,Max_Daily_Drawdown_port = drawdown(pairs_cum_return, port_cum_return)
            annualized_return,annualized_volatility, annualized_return_port,annualized_volatility_port = performance(pairs_cum_return, pairs_daily_return,port_cum_return,port_daily_return)

            sharpe_pairs['Portfolio'] = sharpe_port.iloc[0]
            annualized_return.loc['Portfolio'] = annualized_return_port.iloc[0]
            annualized_return = annualized_return.iloc[:,0]
            annualized_volatility['Portfolio'] = annualized_volatility_port.iloc[0]

            max_dd = Max_Daily_Drawdown.iloc[-1]
            max_dd['Portfolio'] = Max_Daily_Drawdown_port.iloc[-1,0]


            generate_output(output_directory)
