import numpy as np 
import pandas as pd 
import re
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===================================================== Parameters ====================================================
input_directory = r"/content/drive/My Drive/FINA 4380/data.csv"


start_date = '1/3/2005'
end_date='6/30/2005'
#end_date = '12/31/2010'

lookback = 60
entry_level = 2.0
exit_level = 0.0
p_value = 0.05

transaction_cost = 0.001

class data_generation:
    
    def __init__(self, input_directory, start_date, end_date):
        self.input_directory = input_directory
        self.start_date = start_date
        self.end_date = end_date
        
    def transform_ln_price(self, open_df):

        # If the first valid index is none, return first element, else return none
        def first_valid_index(series):
          first_index = series.first_valid_index()
          if first_index != None:
            return series.loc[first_index]
          else:
            return None

        ln_open_df = np.log(open_df/open_df.apply(first_valid_index))
        ln_open_np = ln_open_df.values

        return ln_open_np
        
    def output_data(self):
        
        # Read data and slicing
        data = pd.read_csv(self.input_directory, index_col=0)
        data = data.loc[self.start_date:self.end_date]
        
        # Tickers
        regex_pat = re.compile(r'_.*')
        Tickers = data.columns.str.replace(regex_pat, '').unique()

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

        # Return
        open_return_df = open_df.pct_change()
        close_return_df = close_df.pct_change()

        ln_open_np = self.transform_ln_price(open_df)

        return ln_open_np, open_df, close_df, volume_df, open_return_df, close_return_df, Tickers
        
class strategy:
    
    def __init__(self, lookback, entry_level, exit_level, p_value):
        self.lookback = lookback
        self.entry_level = entry_level
        self.exit_level = exit_level
        self.p_value = p_value
        
    def coint_and_resid(self, X_df, Y):
        # Initialize numpy array to match the shape
        betas = np.zeros((1, X_df.shape[1]))
        standardized_residuals = np.zeros((1, X_df.shape[1]))
        pvalues = np.ones((1, X_df.shape[1]))

        for j in range(X_df.shape[1]):
            X = X_df[:,j]
            
            # If there is any nan, cannot perform OLS and we will exclude them from trading (set the pvalue be one from the last part)
            if np.isnan(X).any():
                continue

            X = sm.add_constant(X)
            result = sm.OLS(Y,X).fit()
            
            betas[0][j] = result.params[1]
            standardized_residuals[0][j] = result.resid_pearson[-1]
            pvalues[0][j] = adfuller(result.resid)[1]

        return pvalues, betas, standardized_residuals
    
    def position_generator(self, tdy_pvalues, tdy_betas, tdy_standardized_residuals, ytd_position):
        tdy_position = np.where((tdy_pvalues > self.p_value), 0,
                        np.where((ytd_position == 0 ) & (abs(tdy_standardized_residuals) > self.entry_level), -np.sign(tdy_standardized_residuals),
                        np.where((ytd_position == 1) & (tdy_standardized_residuals>self.exit_level), 0,
                        np.where((ytd_position == -1) & (tdy_standardized_residuals<self.exit_level), 0,
                                ytd_position))))
        
        return tdy_position
    
    def transform_np(self, data, open_df):
        return pd.DataFrame(data[1:], index=open_df.index[self.lookback:], columns=open_df.columns[:-1])

    def generate_equal_weight(self, n):
        number_of_stocks = betas.shape[1]
        weight = [1/(2*number_of_stocks)] * (number_of_stocks)
        weight.append(1/2)

        return weight
    
    def main(self, ln_open_np, open_df):
        
        # Initialize empty array
        pvalues = np.ones((1,ln_open_np.shape[1]-1))
        standardized_residuals = np.ones((1,ln_open_np.shape[1]-1))
        betas = np.ones((1,ln_open_np.shape[1]-1))
        beta_position = np.zeros((1,ln_open_np.shape[1]-1))
        position = np.zeros((1,ln_open_np.shape[1]-1))
        
        # For each day, slice back 60 days
        for i in tqdm(range(len(ln_open_np)-self.lookback)):
            sliced_data_np = ln_open_np[i:self.lookback+i]
            
            # Compute everyday pval, betas, residuals
            tdy_pvalues, tdy_betas, tdy_standardized_residuals = self.coint_and_resid(sliced_data_np[:,:-1], sliced_data_np[:,-1])
            
            # Compute everyday position
            ytd_position = position[-1]
            tdy_position = self.position_generator(tdy_pvalues, tdy_betas, tdy_standardized_residuals, ytd_position)

            ytd_beta_position = beta_position[-1]
            tdy_beta_position = np.where((ytd_position == tdy_position), ytd_beta_position, 
                                        np.where((ytd_position == 0) & (tdy_position !=0), tdy_position * -1 * tdy_betas,
                                        0))
            
            # Stack the data to numpy
            pvalues = np.vstack((pvalues, tdy_pvalues))
            standardized_residuals = np.vstack((standardized_residuals, tdy_standardized_residuals))
            betas = np.vstack((betas, tdy_betas))
            position = np.vstack((position, tdy_position))
            beta_position = np.vstack((beta_position, tdy_beta_position))
            
        # Convert numpy to pandas
        pvalues = self.transform_np(pvalues, open_df)
        standardized_residuals = self.transform_np(standardized_residuals, open_df)
        betas = self.transform_np(betas, open_df)
        position = self.transform_np(position, open_df)
        beta_position = self.transform_np(beta_position, open_df)
        
        # Generate SPY position
        SPY_position = beta_position.sum(axis=1)
        position['SPY'] = SPY_position

        weight = self.generate_equal_weight(betas.shape[1])
            
        return pvalues, standardized_residuals, betas, position, beta_position, weight
        
class backtest:
    
    def __init__(self, transaction_cost):
        self.transaction_cost = transaction_cost
        
    def main(self, position, open_return_df, weight):

        sliced_open_return_df = open_return_df.loc[position.index]

        # signal is simply the difference of position and some adjustments
        signal = position.diff()
        signal.iloc[0] = position.iloc[0]

        # If we are long, we minus transaction cost, if we are short, we add transaction cost
        transaction_cost_df = -signal * -1 * self.transaction_cost
        
        # return minus transaction cost
        return_with_tc = sliced_open_return_df - transaction_cost_df
        
        daily_return = return_with_tc * position
        
        # Special adjustment for SPY
        # SPY return = original position return + signal return
        # signal return = signal position * (transaction cost)
        SPY_sign_signal = np.where(signal['SPY'] == 0, 0,
                          np.where(signal['SPY'] > 0, 1, -1))
        signal_return = signal['SPY'] * ( -SPY_sign_signal * transaction_cost)
        daily_return['SPY'] = sliced_open_return_df['SPY'] + signal_return
        
        culmulative_return = (transaction_cost_df + 1).cumprod()

        # Portfolio return
        port_daily_return = (daily_return * weight).sum(axis=1)
        port_culmulative_return = (port_daily_return + 1).cumprod()

        # Max Drawdown
        daily_culmulative_max = port_culmulative_return.expanding().max()
        daily_culmulative_min = port_culmulative_return.expanding().min()
        maximum_drawdown = daily_culmulative_min/daily_culmulative_max-1
        
        return daily_return, culmulative_return, port_daily_return, port_culmulative_return, maximum_drawdown

# ===================================================== Main ====================================================
generate_data = data_generation(input_directory, start_date, end_date)
ln_open_np, open_df, close_df, volume_df, open_return_df, close_return_df, Tickers = generate_data.output_data()

cointegration_strategy = strategy(lookback, entry_level, exit_level, p_value)
pvalues, standardized_residuals, betas, position, beta_position, weight = cointegration_strategy.main(ln_open_np, open_df)

cointegration_backtest = backtest(transaction_cost)
daily_return, culmulative_return, port_daily_return, port_culmulative_return, maximum_drawdown = cointegration_backtest.main(position, open_return_df, weight)

writer = pd.ExcelWriter(r'/content/drive/My Drive/Statistical Arbitrage/result.xlsx')

daily_return.to_excel(writer, 'daily_return')
culmulative_return.to_excel(writer, 'culmulative_return')
port_daily_return.to_excel(writer, 'port_daily_return')
port_culmulative_return.to_excel(writer, 'port_culmulative_return')
maximum_drawdown.to_excel(writer, 'max_drawdown')

writer.save()
