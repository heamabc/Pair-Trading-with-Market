import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class data_generation:
    
    def __init__(self, input_directory, start_date, end_date):
        self.input_directory = input_directory
        self.start_date = start_date
        self.end_date = end_date
        
    def transform_ln_price(self, open_df):
        open_df_transpose = open_df.T

        def price_0_func(x):
            if (open_df_transpose.values[0] is None) or (x.values[0] is None):
                return None
            else:
                return open_df_transpose.loc[x.name, x.values[0]]

        open_orignal_df = pd.DataFrame(open_df_transpose.apply(lambda x: x.first_valid_index(), axis=1)).apply(price_0_func,axis=1)

        ln_open_df = np.log(open_df/open_orignal_df)
        return ln_open_df
        
    def output_data(self):
        
        # Read data and slicing
        data = pd.read_csv(self.input_directory, index_col='date')
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
        
        ln_open_df = self.transform_ln_price(open_df)

        return ln_open_df.values, open_df, close_df, volume_df, Tickers
        
        
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
                        np.where((ytd_position == 0 ) & (abs(tdy_standardized_residuals) > self.entry_level), np.sign(tdy_standardized_residuals),
                        np.where((ytd_position == 1) & (tdy_standardized_residuals<self.exit_level), 0,
                        np.where((ytd_position == -1) & (tdy_standardized_residuals>self.exit_level), 0,
                                ytd_position))))

        return tdy_position
    
    def main(self, ln_open_np):
        
        pvalues = np.ones((1,ln_open_df.shape[1]-1))
        standardized_residuals = np.ones((1,ln_open_df.shape[1]-1))
        betas = np.ones((1,ln_open_df.shape[1]-1))
        position = np.zeros((1,ln_open_df.shape[1]-1))
        
        
        for i in tqdm(range(len(ln_open_np)-self.lookback)):
            sliced_data_np = ln_open_np[i:self.lookback+i]
            tdy_pvalues, tdy_betas, tdy_standardized_residuals = self.coint_and_resid(sliced_data_np[:,:-1], sliced_data_np[:,-1])
            
            ytd_position = position[-1]
            tdy_position = self.position_generator(tdy_pvalues, tdy_betas, tdy_standardized_residuals, ytd_position)
            
            
            
            # Stack the data to numpy
            pvalues = np.vstack((pvalues, tdy_pvalues))
            standardized_residuals = np.vstack((standardized_residuals, tdy_standardized_residuals))
            betas = np.vstack((betas, tdy_betas))
            position = np.vstack((tdy_position, position))

        return pvalues, standardized_residuals, betas, position
        
        
input_directory = r"/kaggle/input/sp500fina4380/data.csv"

start_date = '1/3/2005'
#end_date = '12/31/2010'
end_date = '12/29/2005'

lookback = 60
entry_level = 2.0
exit_level = 0.0
p_value = 0.05

generate_data = data_generation(input_directory, start_date, end_date)
ln_open_np, open_df, close_df, volume_df, Tickers = generate_data.output_data()

cointegration_strategy = strategy(lookback, entry_level, exit_level, p_value)
pvalues, standardized_residual, betas, position = cointegration_strategy.main(ln_open_np)



