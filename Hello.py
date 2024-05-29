### Below are all the functions for the streamlit app ###

# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf  # Install using: pip install yfinance
from datetime import datetime, timedelta
import pandas_datareader as pdr
#from IPython.display import display
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


# Equities Functions

## Equities Data Preprocess Function
st.set_page_config(layout="wide")

def process_equities_data(file_name):
    """
    Process an Excel file of data for analysis.
    This function reads the data, ensures the datetime is in the correct format, sorts the data, renames certain columns, and fills NaN values.
    
    Parameters:
    file_name (str): The name of the Excel file to process.
    
    Returns:
    df (pd.DataFrame): The processed DataFrame.
    """
    # Read the data
    df = pd.read_excel(file_name)

    # Make sure datetime is in the right format
    df["Date"] = pd.to_datetime(df["Date"])

    # Reverse data sorting so that it is in ascending order 
    df = df.sort_values('Date', ascending=True)

    # Find columns by partial name and rename them
    column_names = {
        df.filter(like='Price Earnings Ratio').columns[0]: 'Price Earnings Ratio',
        df.filter(like='Price/Cash Flow').columns[0]: 'Price/Cash Flow',
        df.filter(like='Price to Sales Ratio').columns[0]: 'Price to Sales Ratio',
        df.filter(like='Price/EBITDA').columns[0]: 'Price/EBITDA',
        df.filter(like='Price to Book Ratio').columns[0]: 'Price to Book Ratio',
        df.filter(like='EV To Trailing 12M Sales').columns[0]: 'EV To Trailing 12M Sales',
        df.filter(like='Enterprise Value/EBITDA ').columns[0]: 'Enterprise Value/EBITDA',
        df.filter(like='Dividend 12 Month Yld - Gross').columns[0]: 'Dividend 12 Month Yld - Gross',
        df.filter(like='Gross Margin').columns[0]: 'Gross Margin',
        df.filter(like='Profit Margin').columns[0]: 'Profit Margin',
        df.filter(like='Return on Assets').columns[0]: 'Return on Assets',
        df.filter(like='Return on Common Equity').columns[0]: 'Return on Common Equity',
        df.filter(like='Dividend Payout Ratio').columns[0]: 'Dividend Payout Ratio',
        df.filter(like='Total Debt to EV').columns[0]: 'Total Debt to EV',
        df.filter(like='Net Debt/EBITDA').columns[0]: 'Net Debt/EBITDA',
        df.filter(like='Total Debt to Total Equity').columns[0]: 'Total Debt to Total Equity',
        df.filter(like='Total Debt to Total Asset').columns[0]: 'Total Debt to Total Asset',
        df.filter(like='Close Price').columns[0]: 'Close Price'
    }
    df.rename(columns=column_names, inplace=True)
    
    # Fill NaN values using forward and backward fill
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')

    # Add annual return column
    df['Annual Return'] = df['Close Price'].pct_change(periods=12)

    return df


## Equities Score Calculation Functions

def calculate_valuation(data, years):
    """
    Calculate the average percentile of value indicators.
    This function selects a timeframe using the 'years' variable, calculates the percentile of the first, second, and third value indicators, and then calculates the average of these three percentiles.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the value indicators.
    """
    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of first value indicator (Price Earnings Ratio)
    last_pe = data['Price Earnings Ratio'].iloc[-1]
    pe_percentile = stats.percentileofscore(data['Price Earnings Ratio'], last_pe, kind='rank',nan_policy='omit')

    # Calculate percentile of second value indicator (Price/Cash Flow)
    last_pcash = data['Price/Cash Flow'].iloc[-1]
    pcash_percentile = 100 - stats.percentileofscore(data['Price/Cash Flow'], last_pcash, kind='rank', nan_policy='omit')

    # Calculate percentile of third value indicator (Price to Sales Ratio)
    last_psale = data['Price to Sales Ratio'].iloc[-1]
    psale_percentile = 100 - stats.percentileofscore(data['Price to Sales Ratio'], last_psale, kind='rank', nan_policy='omit')

    # Calculate percentile of fourth value indicator (Price/EBITA)
    last_pebita = data['Price/EBITDA'].iloc[-1]
    pebita_percentile = 100 - stats.percentileofscore(data['Price/EBITDA'], last_pebita, kind='rank', nan_policy='omit')

    # Calculate percentile of fifth value indicator (Price to Book Ratio)
    last_pb = data['Price to Book Ratio'].iloc[-1]
    pb_percentile = 100 - stats.percentileofscore(data['Price to Book Ratio'], last_pb, kind='rank', nan_policy='omit')

    # Calculate percentile of sixth value indicator (EV To Trailing 12M Sales)
    last_ev = data['EV To Trailing 12M Sales'].iloc[-1]
    ev_percentile = 100 - stats.percentileofscore(data['EV To Trailing 12M Sales'], last_ev, kind='rank', nan_policy='omit')

    # Calculate percentile of seventh value indicator (Enterprise Value/EBITDA)
    last_ev_ebitda = data['Enterprise Value/EBITDA'].iloc[-1]
    ev_ebitda_percentile = 100 - stats.percentileofscore(data['Enterprise Value/EBITDA'], last_ev_ebitda, kind='rank', nan_policy='omit')

    # Calculate percentile of eigth value indicator (Dividend 12 Month Yld - Gross)
    last_div = data['Dividend 12 Month Yld - Gross'].iloc[-1]
    div_percentile = stats.percentileofscore(data['Price Earnings Ratio'], last_div, kind='rank', nan_policy='omit')

    # Calculate the average of the 8 percentiles
    average_percentile = (pe_percentile + pcash_percentile + psale_percentile + pebita_percentile + pb_percentile + ev_percentile + ev_ebitda_percentile + div_percentile) / 8

    return average_percentile


def calculate_growth(data, years):
    """
    Calculate the average percentile of growth indicators.
    This function selects a timeframe using the 'years' variable, calculates the percentile of the growth indicators, and then calculates the average of the percentiles.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the growth indicators.

    Returns:
    average_percentile (float): The average percentile of the growth indicators.
    """
    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of first growth indicator (Gross Margin)
    last_grm = data['Gross Margin'].iloc[-1]
    grm_percentile = stats.percentileofscore(data['Gross Margin'], last_grm, kind='rank', nan_policy='omit') 

    # Calculate percentile of second growth indicator (Profit Margin)
    last_prm = data['Profit Margin'].iloc[-1]
    prm_percentile = stats.percentileofscore(data['Profit Margin'], last_prm, kind='rank', nan_policy='omit')

    # Calculate percentile of third growth indicator (Return on Assets)
    last_roa = data['Return on Assets'].iloc[-1]
    roa_percentile = stats.percentileofscore(data['Return on Assets'], last_roa, kind='rank', nan_policy='omit')

    # Calculate percentile of fourth growth indicator (Return on Common Equity)
    last_roe = data['Return on Common Equity'].iloc[-1]
    roe_percentile = stats.percentileofscore(data['Return on Common Equity'], last_roe, kind='rank', nan_policy='omit')

    # Calculate percentile of fifth growth indicator (Dividend Payout Ratio)
    last_dpr = data['Dividend Payout Ratio'].iloc[-1]
    dpr_percentile = stats.percentileofscore(data['Dividend Payout Ratio'], last_dpr, kind='rank', nan_policy='omit')

    # Calculate the average of the 5 percentiles
    average_percentile = (grm_percentile + prm_percentile + roa_percentile + roe_percentile + dpr_percentile) / 5

    return average_percentile


def calculate_leverage(data, years):
    """
    Calculate the average percentile of leverage indicators.
    This function selects a timeframe using the 'years' variable, calculates the percentile of the sentiment indicators, and then calculates the average of the percentiles.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the leverage indicators.

    Returns:
    average_percentile (float): The average percentile of the leverage indicators.
    """
    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of first leverage indicator (Total Debt to EV)
    last_dev = data['Total Debt to EV'].iloc[-1]
    dev_percentile = 100 - stats.percentileofscore(data['Total Debt to EV'], last_dev, kind='rank', nan_policy='omit')

    # Calculate percentile of second leverage indicator (Net Debt/EBITDA)
    last_de_ebitda= data['Net Debt/EBITDA'].iloc[-1]
    de_ebitda_percentile = 100 - stats.percentileofscore(data['Net Debt/EBITDA'], last_de_ebitda, kind='rank', nan_policy='omit')

    # Calculate percentile of third leverage indicator (Total Debt to Total Equity)
    last_debt_equity= data['Total Debt to Total Equity'].iloc[-1]
    debt_equity_percentile = 100 - stats.percentileofscore(data['Total Debt to Total Equity'], last_debt_equity, kind='rank', nan_policy='omit')

    # Calculate percentile of fourth leverage indicator (Total Debt to Total Asset)
    last_debt_asset= data['Total Debt to Total Asset'].iloc[-1]
    debt_asset_percentile = 100 - stats.percentileofscore(data['Total Debt to Total Asset'], last_debt_asset, kind='rank', nan_policy='omit')

    # Calculate the average of the 4 percentiles
    average_percentile = (dev_percentile + de_ebitda_percentile + debt_equity_percentile + debt_asset_percentile) / 4
    
    # Inverse the percentile because lower beta and volatility is considered better
    return average_percentile


def calculate_final_score(data, years):
    """
    Calculate the final score of an asset class
    This function calculates the final score of an asset class based on the value, growth, and leverage scores.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the value, growth, and leverage indicators.

    Returns:
    final_score (float): The final score of the asset class.
    """
    value_score = calculate_valuation(data, years)
    growth_score = calculate_growth(data, years)
    leverage_score = calculate_leverage(data, years)
    
    final_score = (value_score * valuation_weight) + (growth_score * growth_weight) + (leverage_score * leverage_weight)

    return final_score


## Equities Display Results Functions

def plot_equities_scores(data, names, years):
    """
    Plot the scores of the asset classes using Plotly.
    This function calculates the scores for each category and creates a bar plot to visualize the scores.

    Parameters:
    data (list): A list of DataFrames containing the value, growth, and leverage indicators for each asset class.
    names (list): A list of strings containing the names of each asset class.

    Returns:
    A bar plot of the scores for each asset class using Plotly.
    """
    # Create a dictionary to store the scores for each category
    scores = {'Final': [], 'Valuation': [], 'Growth': [], 'Leverage': []}
    
    # Calculate the scores for each category
    for df in data:
        scores['Valuation'].append(calculate_valuation(df, years))
        scores['Growth'].append(calculate_growth(df, years))
        scores['Leverage'].append(calculate_leverage(df, years))
        scores['Final'].append(calculate_final_score(df, years))

    # Create a dataframe from the scores dictionary and add asset names as a column
    scores_df = pd.DataFrame(scores, index=names)
    scores_df.reset_index(inplace=True)
    scores_df.rename(columns={'index': 'Asset Class'}, inplace=True)

    figs=[]
    # Create a separate bar plot for each category
    for category in scores:
        # Create a new column for color based on the score ranking
        scores_df = scores_df.sort_values(by=category, ascending=True)  # Keep ascending sort
        scores_df['color'] = ['Other Scores' for _ in range(len(scores_df))]  # Default 'Other Scores'
        scores_df.iloc[-3:, scores_df.columns.get_loc('color')] = ['Third Highest Score', 'Second Highest Score', 'Highest Score']  # Modify the last 3
    
        # Create the bar plot
        fig = px.bar(scores_df, y='Asset Class', x=category, title=f'Scores of Asset Classes for {category}: {years}yr Timeframe', 
                     color='color', orientation='h', 
                     color_discrete_map={'Highest Score': 'navy', 'Second Highest Score': 'royalblue', 'Third Highest Score': 'skyblue', 'Other Scores': 'lightgray'},
                     hover_data={'Asset Class': True, category: True, 'color': False},
                     text=category)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='inside')
        figs.append(fig)
    
    return figs


# Fixed Income Functions

## Fixed Income Data Preprocess Function
def process_fi_data(file_name):
    """
    Process an Excel file of data for analysis.
    This function reads the data, ensures the datetime is in the correct format, sorts the data, renames certain columns, and fills NaN values.
    
    Parameters:
    file_name (str): The name of the Excel file to process.
    
    Returns:
    df (pd.DataFrame): The processed DataFrame.
    """
    # Read the data
    df = pd.read_excel(file_name)

    # Make sure datetime is in the right format
    df["Date"] = pd.to_datetime(df["Date"])

    # Reverse data sorting so that it is in ascending order 
    df = df.sort_values('Date', ascending=True)

    # Find columns by partial name and rename them
    column_names = {
        df.filter(like='Index Price').columns[0]: 'Index Price',
        df.filter(like='Index Coupon').columns[0]: 'Index Coupon',
        df.filter(like='Index Time to Maturity').columns[0]: 'Index Time to Maturity',
        df.filter(like='Index OAS').columns[0]: 'Index OAS',
        df.filter(like='Index OAD').columns[0]: 'Index OAD',
        df.filter(like='Index OAC').columns[0]: 'Index OAC',
        df.filter(like='Index Yield to Worst').columns[0]: 'Index Yield to Worst',
        df.filter(like='Index Yield to Maturity').columns[0]: 'Index Yield to Maturity',
        df.filter(like='Index Bid Spread').columns[0]: 'Index Spread',
        df.filter(like='Index Total Return').columns[0]: 'Index Monthly Return(%)'
    }
    df.rename(columns=column_names, inplace=True)

    df['Annual Return'] = df['Index Price'].pct_change(periods=12)
    
    # Fill NaN values using forward and backward fill
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df = df.fillna(0)

    return df


## Fixed Income Score Calculation Functions

def calculate_fixed_income_value(data, years):
    """
    Calculate the percentiles of value fixed income indicators.
    This function selects a timeframe using the 'years' variable and calculates the percentiles.
    """

    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of Index Price
    last_price = data['Index Price'].iloc[-1]
    # Take the inverse of the percentile because lower price is better
    price_percentile = round(100 - stats.percentileofscore(data['Index Price'], last_price, nan_policy='omit'), 2)

    # Calculate percentile of Index Yield to Maturity
    last_yield_maturity = data['Index Yield to Maturity'].iloc[-1]
    yield_to_maturity_percentile = round(stats.percentileofscore(data['Index Yield to Maturity'], last_yield_maturity, nan_policy='omit'), 2)

    # Calculate percentile of Index Yield to Worst
    last_yield_worst = data['Index Yield to Worst'].iloc[-1]
    yield_to_worst_percentile = round(stats.percentileofscore(data['Index Yield to Worst'], last_yield_worst, nan_policy='omit'), 2)

    avg_percentile = (price_percentile + yield_to_maturity_percentile + yield_to_worst_percentile) / 3

    return avg_percentile


def calculate_fixed_income_volatility(data, years):
    """
    Calculate the percentile of volatility fixed income indicators.
    This function selects a timeframe using the 'years' variable and calculates the percentile.
    """

    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of Index OAS
    last_oas = data['Index OAS'].iloc[-1]
    oas_percentile = round(stats.percentileofscore(data['Index OAS'], last_oas, nan_policy='omit'), 2)

    # Calculate percentile of Index OAD
    last_oad = data['Index OAD'].iloc[-1]
    oad_percentile = round(stats.percentileofscore(data['Index OAD'], last_oad, nan_policy='omit'), 2)

    # Calculate percentile of Index OAC
    last_oac = data['Index OAC'].iloc[-1]
    oac_percentile = round(stats.percentileofscore(data['Index OAC'], last_oac, nan_policy='omit'), 2)

    # Calculate percentile of Index Spread
    last_spread = data['Index Spread'].iloc[-1]
    spread_percentile = round(stats.percentileofscore(data['Index Spread'], last_spread, nan_policy='omit'), 2)

    avg_percentile = (oas_percentile + oad_percentile + oac_percentile + spread_percentile) / 4

    return avg_percentile


def calculate_fixed_income_coupon(data, years):
    """
    Calculate the percentile of income fixed income indicators.
    This function selects a timeframe using the 'years' variable and calculates the percentile.
    """

    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of Index Coupon
    last_coupon = data['Index Coupon'].iloc[-1]
    coupon_percentile = round(stats.percentileofscore(data['Index Coupon'], last_coupon, nan_policy='omit'), 2)

    return coupon_percentile


def calculate_fixed_income_duration(data, years):
    """
    Calculate the percentile of duration fixed income indicators.
    This function selects a timeframe using the 'years' variable and calculates the percentile.
    """

    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of Index Time to Maturity
    last_maturity = data['Index Time to Maturity'].iloc[-1]
    time_to_maturity_percentile = round(stats.percentileofscore(data['Index Time to Maturity'], last_maturity, nan_policy='omit'), 2)

    return time_to_maturity_percentile


## Fixed Income Display Results Functions

def plot_fixed_income_scores(data, names, years):
    """Plots scores of fixed income asset classes with top 3 highlighting."""
    scores = {'Value': [], 'Volatility': [], 'Coupon': [], 'Duration': []}

    # Calculate the scores for each category
    for df in data:
        scores['Value'].append(calculate_fixed_income_value(df, years))
        scores['Volatility'].append(calculate_fixed_income_volatility(df, years))
        scores['Coupon'].append(calculate_fixed_income_coupon(df, years))
        scores['Duration'].append(calculate_fixed_income_duration(df, years))
    
    scores_df = pd.DataFrame(scores, index=names)
    scores_df.reset_index(inplace=True)
    scores_df.rename(columns={'index': 'Asset Class'}, inplace=True)

    figs = []  # Store all figure objects

    for category in scores:
        # Color assignment for top 3
        scores_df = scores_df.sort_values(by=category, ascending=True)
        scores_df['color'] = ['Other Scores' for _ in range(len(scores_df))]
        scores_df.iloc[-3:, scores_df.columns.get_loc('color')] = ['Third Highest Score', 'Second Highest Score', 'Highest Score']

        fig = px.bar(scores_df, y='Asset Class', x=category, color='color', orientation='h',
                     title=f'Scores of Asset Classes for {category}',
                     color_discrete_map={'Highest Score': 'navy', 'Second Highest Score': 'royalblue', 'Third Highest Score': 'skyblue', 'Other Scores': 'lightgray'},
                     text=scores_df[category].round(2))

        figs.append(fig)  # Add the figure to the list

    return figs  # Return the list of figures, i.e. figures[0] for Value, figures[1] for Volatility, etc.


# Backtesting Functions

## Using Average Returns

def calculate_return(df_list, names):
    """Calculates average annual returns over specified timeframes.

    Args:
        df_list (list): A list of DataFrames containing the 'Annual Return' column.
        timeframes (list): List of timeframes (in years) for average calculation.

    Returns:
        pd.DataFrame: DataFrame with average annual returns for each timeframe.
    """

    all_returns = []  # This will store the average returns for each dataset
    time_frames = [1, 3, 5, 10]

    for df in df_list:
        returns = []  # This will store the average returns for the current dataset across time frames
        latest_date = pd.to_datetime(df['Date']).max()

        for years in time_frames:
            start_date = latest_date - pd.DateOffset(years=years)
            # Filter the DataFrame based on the time frame
            filtered_df = df[(pd.to_datetime(df['Date']) > start_date) & (pd.to_datetime(df['Date']) <= latest_date)]

            # Calculate the average return for the current time frame using the 'Annual Return(%)' column
            avg_return = round(filtered_df['Annual Return'].mean(), 2)
            returns.append(avg_return)

        # Append the calculated average returns for this dataset to the all_returns list
        all_returns.append(returns)

    # Create DataFrame for visualization
    results_df = pd.DataFrame(all_returns, index=names, columns=time_frames).transpose()

    # Reset the index to make it a regular column and rename it to 'Timeframe(yr)'
    results_df = results_df.reset_index().rename(columns={'index': 'Timeframe(yr)'})

    return results_df


def plot_avg_return_heatmap(df_list, names):
    """Generates a heatmap of average annual returns."""
    
    results = calculate_return(df_list, names)

    # Ensure 'Timeframe(yr)' is a column and not an index
    if 'Timeframe(yr)' not in results.columns:
        results = results.reset_index().rename(columns={'index': 'Timeframe(yr)'})
        
    df_heatmap = results.set_index('Timeframe(yr)').transpose()

    # Create Heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.heatmap(df_heatmap, annot=True, fmt='.2%', cmap='RdBu', cbar_kws={'label': 'Average Return (%)'}, ax=ax) 
    ax.set_title('Average Annual Returns Across Timeframes (Compounded)')  
    ax.set_xlabel('Timeframe (Years)') 
    ax.set_ylabel('Asset Class') 

    return fig


## Using Sharpe Ratio

def get_treasury_bill_rate():
    # Get the current date
    today = pd.to_datetime('today')
    
    # If today is a weekend (Saturday or Sunday), set the date to the last Friday
    if today.weekday() >= 5:
        last_trading_day = today - pd.Timedelta(days=today.weekday() - 4)
    else:
        last_trading_day = today
    
    # Attempt to download data until we get valid data (handles holidays and weekends)
    while True:
        tbill_data = yf.download("^IRX", start=last_trading_day, end=last_trading_day + pd.Timedelta(days=1))
        if not tbill_data.empty:
            tbill_rate = tbill_data['Close'].iloc[-1] / 100  # Convert percentage to decimal
            return tbill_rate
        last_trading_day -= pd.Timedelta(days=1)  # Move to the previous day


def calculate_sharpe_ratio(df_list, names):
    """Calculates Sharpe ratios over specified timeframes."""
    all_sharpes = []  
    time_frames = [1, 3, 5, 10]  # Keep consistent with return calculation

    risk_free_rate = get_treasury_bill_rate()

    for df in df_list:
        sharpes = [] 
        latest_date = pd.to_datetime(df['Date']).max()

        for years in time_frames:
            start_date = latest_date - pd.DateOffset(years=years)
            filtered_df = df[(pd.to_datetime(df['Date']) > start_date) & (pd.to_datetime(df['Date']) <= latest_date)]
            
            returns = filtered_df['Annual Return']  
            excess_return = returns - risk_free_rate
            sharpe_ratio = round((excess_return.mean() / returns.std()) * np.sqrt(12), 2)  
            sharpes.append(sharpe_ratio)

        all_sharpes.append(sharpes)

    results_df = pd.DataFrame(all_sharpes, index=names, columns=time_frames).transpose()
    results_df = results_df.reset_index().rename(columns={'index': 'Timeframe(yr)'})
    return results_df


def plot_sharpe_ratio_heatmap(df_list, names):
    """Generates a heatmap of Sharpe ratios."""
    
    results = calculate_sharpe_ratio(df_list, names)

    # Ensure 'Timeframe(yr)' is a column and not an index
    if 'Timeframe(yr)' not in results.columns:
        results = results.reset_index().rename(columns={'index': 'Timeframe(yr)'})
        
    df_heatmap = results.set_index('Timeframe(yr)').transpose()

    # Create Heatmap using Seaborn
    fig, ax = plt.subplots(figsize=(10, 6))  
    sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='RdBu', cbar_kws={'label': 'Sharpe Ratio'}, ax=ax) 
    ax.set_title('Sharpe Ratio Across Timeframes')  
    ax.set_xlabel('Timeframe (Years)') 
    ax.set_ylabel('Asset Class') 

    return fig



#Bangyangs updates above this line



# Equities Ratios Visualization

## This is a configurable plot that can plot the average ratio of any column between all the equities
def plot_equities_valuation(df_list, df_names, ratio, timeframe_years=None):
    data = []
    for df, name in zip(df_list, df_names):
        if timeframe_years:
            df_filtered = df[df['Date'] >= pd.to_datetime('today') - pd.DateOffset(years=timeframe_years)]
            avg_ratio = df_filtered[ratio].mean()
        else:
            avg_ratio = df[ratio].mean()
        data.append({'Asset Class': name, 'Average Ratio': avg_ratio})

    df_temp = pd.DataFrame(data)  
    df_temp = df_temp.sort_values(by='Average Ratio', ascending=True)  

    # Updated Color Logic (correcting order for ascending sort)
    df_temp['color'] = 'Other Scores' # Default color
    df_temp.iloc[-3:, df_temp.columns.get_loc('color')] = ['Third Highest Score', 'Second Highest Score', 'Highest Score'] 
    
    fig = px.bar(df_temp, x='Average Ratio', y='Asset Class', color='color', orientation='h',
                 title=f"Average {ratio} Valuation (Last {timeframe_years} Years)" if timeframe_years else f"Average {ratio} Valuation",
                 labels={'x': ratio, 'y': 'Asset Class'},
                 color_discrete_map={'Highest Score': 'navy', 'Second Highest Score': 'royalblue', 
                 'Third Highest Score': 'skyblue', 'Other Scores': 'lightgray'})
    return fig

# Fixed Income Duration Visualization

## This function plots all the fixed income's yield vs duration over a specified year parameter
## Can be a useful visualization tool of the fixed income landscape
def plot_yield_duration(df_list, df_names, indicator, time_frame_years):
    fig = go.Figure()  # Using graph_objects for more flexibility

    for df, name in zip(df_list, df_names):
        # Filter for timeframe
        df_filtered = df[df['Date'] >= pd.to_datetime('today') - pd.DateOffset(years=time_frame_years)]
        fig.add_trace(go.Scatter(
            x=df_filtered['Index OAD'],  # Using OAD to Treasury
            y=df_filtered[indicator], 
            mode='markers',
            name=name
        ))

    fig.update_layout(title= indicator+" vs. Duration over the Last "+str(time_frame_years)+" Years",
                      xaxis_title='Option Adjusted Duration',
                      yaxis_title=indicator)
    return fig

## Yield vs Duration Table Visualization
def plot_yield_duration_table(df_list, df_names, timeframe_years):
    data = []
    for df, name in zip(df_list, df_names):
        if timeframe_years:
            df_filtered = df[df['Date'] >= pd.to_datetime('today') - pd.DateOffset(years=timeframe_years)]
            avg_yield = df_filtered['Index Yield to Maturity'].mean()
            avg_duration = df_filtered['Index OAD'].mean()
        else:
            avg_yield = df['Index Yield to Maturity'].mean()
            avg_duration = df['Index OAD'].mean()
        data.append({'Asset Class': name, 'Avg. Yield': avg_yield, 'Avg. Duration': avg_duration})

    df_results = pd.DataFrame(data)
    df_results = df_results.sort_values(by='Avg. Yield', ascending=False)  # Sort by yield
    df_results = df_results.reset_index(drop=True)  # Reset index
    df_results = df_results.round(2)  # Round to 2 decimal places

    return df_results

### Below this is streamlit dashboard ###

# Initialize session state for weights
if 'valuation_weight' not in st.session_state:
    st.session_state['valuation_weight'] = 0.4
if 'growth_weight' not in st.session_state:
    st.session_state['growth_weight'] = 0.4
if 'leverage_weight' not in st.session_state:
    st.session_state['leverage_weight'] = 0.2

# Function to adjust weights
def adjust_weights(changed_key):
    total = st.session_state[changed_key]
    if changed_key == 'valuation_weight':
        st.session_state.growth_weight = (1 - total) / 2
        st.session_state.leverage_weight = (1 - total) / 2
    elif changed_key == 'growth_weight':
        st.session_state.valuation_weight = (1 - total) / 2
        st.session_state.leverage_weight = (1 - total) / 2
    elif changed_key == 'leverage_weight':
        st.session_state.valuation_weight = (1 - total) / 2
        st.session_state.growth_weight = (1 - total) / 2

# Define file processing and plotting functions here...

## Import equities data
large_cap_growth = process_equities_data("streamlit/data/RLG_10y_monthly.xlsx")
large_cap_value = process_equities_data("streamlit/data/RLV_10y_monthly.xlsx")
mid_cap_growth = process_equities_data("streamlit/data/RDG_10y_monthly.xlsx")
mid_cap_value = process_equities_data("streamlit/data/RMV_10y_monthly.xlsx")
small_cap_growth = process_equities_data("streamlit/data/RUO_10y_monthly.xlsx")
small_cap_value = process_equities_data("streamlit/data/RUJ_10y_monthly.xlsx")
int_growth = process_equities_data("streamlit/data/MXEA000G_10y_monthly.xlsx")
int_value = process_equities_data("streamlit/data/MXEA000V_10y_monthly.xlsx")
emerging_markets_equity = process_equities_data("streamlit/data/MXEF_10y_monthly.xlsx")
small_cap_int = process_equities_data("streamlit/data/SBERWUU_10y_monthly.xlsx")

## Store the dataframes in a list
equities = [large_cap_growth, large_cap_value, mid_cap_growth, mid_cap_value, small_cap_growth, small_cap_value, int_growth, int_value, emerging_markets_equity, small_cap_int]

## Set names of the asset classes
equities_names = ['Large Cap Growth', 'Large Cap Value', 'Mid Cap Growth', 'Mid Cap Value', 'Small Cap Growth', 'Small Cap Value', 'Interational Growth', 'International Value', 'Emerging Markets Equity', 'Small Cap International']

## set default year parameter
years = 10

## set default weights parameters
valuation_weight = 0.4
growth_weight = 0.4
leverage_weight = 0.2


# Fixed Income Results

## Import fixed income data

core_bond = process_fi_data("streamlit/data/10y_monthly_core_bond.xlsx")
emerging_bond = process_fi_data("streamlit/data/10y_monthly_emerging_bond.xlsx")
floating_rate_bond = process_fi_data("streamlit/data/10y_monthly_floating_rate.xlsx")
high_yield_bond = process_fi_data("streamlit/data/10y_monthly_high_yield.xlsx")
short_term_bond = process_fi_data("streamlit/data/10y_monthly_short_term.xlsx")
tips = process_fi_data("streamlit/data/10y_monthly_tips.xlsx")

## Store the dataframes in a list
fixed_income = [core_bond, emerging_bond, floating_rate_bond, high_yield_bond, short_term_bond, tips]

## Set names of the asset classes
fixed_income_names = ['Core Bond', 'Emerging Bond', 'Floating Rate Bond', 'High Yield Bond', 'Short Term Bond', 'TIPS']

#This section defines the upload file function. 

# Define paths to existing files
equity_files_paths = {
    'Large Cap Growth': "streamlit/data/RLG_10y_monthly.xlsx",
    'Large Cap Value': "streamlit/data/RLV_10y_monthly.xlsx",
    'Mid Cap Growth': "streamlit/data/RDG_10y_monthly.xlsx",
    'Mid Cap Value': "streamlit/data/RMV_10y_monthly.xlsx",
    'Small Cap Growth': "streamlit/data/RUO_10y_monthly.xlsx",
    'Small Cap Value': "streamlit/data/RUJ_10y_monthly.xlsx",
    'International Growth': "streamlit/data/MXEA000G_10y_monthly.xlsx",
    'International Value': "streamlit/data/MXEA000V_10y_monthly.xlsx",
    'Emerging Markets Equity': "streamlit/data/MXEF_10y_monthly.xlsx",
    'Small Cap International': "streamlit/data/SBERWUU_10y_monthly.xlsx"
}

# Define a dictionary to hold uploaded data
uploaded_equity_data = {}

# Set names of the asset classes
equities_names = list(equity_files_paths.keys())

# Create file uploaders for each equity type
with st.sidebar:
    with st.popover("Upload New Equity Data Files Here"):
        st.header("Upload New Equity Data Files")
        for name in equities_names:
            file = st.file_uploader(f"Upload {name} Data", type=["xlsx"], key=name)
            if file is not None:
                # Process and store the uploaded file
                uploaded_equity_data[name] = process_equities_data(file)
                # Save the uploaded file to replace the existing one
                with open(equity_files_paths[name], 'wb') as f:
                    f.write(file.getbuffer())
                st.success(f"{name} data has been successfully uploaded and replaced.")

        # Load and display the updated dataframes
        if uploaded_equity_data:
            st.header("Updated Equities Data")
            for name, df in uploaded_equity_data.items():
                st.write(f"**{name}**")
                st.dataframe(df)


# Streamlit dashboard layout
if __name__ == '__main__':
    st.title('Pacific Life Asset Classes Dashboard')
    tab1, tab2, tab3, tab4 = st.tabs(["Equities Analysis", "Fixed Income Analysis", "Backtesting", "Further Analysis"])
    # Tab 1: Equities Scores
    with tab1:
        st.header("Equity Assets Analysis")
        with st.popover("Edit Settings for Analysis Here"):
            st.subheader("Select the Number of Years")
            equities_years = st.slider("Select the number of years:", min_value=0, max_value=10, value=10, step=1, key='equities_years')
        
            st.subheader("Adjust Weights")
        
            # Use st.columns to arrange the weights side by side
            col1, col2, col3 = st.columns(3)

            with col1:
                valuation_weight = st.number_input("Valuation Weight", min_value=0.0, max_value=1.0, value=st.session_state.get('valuation_weight', 0.33), key='valuation_weight', on_change=adjust_weights, args=('valuation_weight',))

            with col2:
                growth_weight = st.number_input("Growth Weight", min_value=0.0, max_value=1.0, value=st.session_state.get('growth_weight', 0.33), key='growth_weight', on_change=adjust_weights, args=('growth_weight',))

            with col3:
                leverage_weight = st.number_input("Leverage Weight", min_value=0.0, max_value=1.0, value=st.session_state.get('leverage_weight', 0.34), key='leverage_weight', on_change=adjust_weights, args=('leverage_weight',))

            st.write(f"Total: {st.session_state.valuation_weight + st.session_state.growth_weight + st.session_state.leverage_weight}")

        # Button to update plots
        #if st.button('Update Plots'):
        #no need for button. 
        st.subheader("Equity Asset Scores")
        equities_figs = plot_equities_scores(equities, equities_names, equities_years)
        #this version will seperate the plots into two columns.
        tabcol1,tabcol2 = st.columns(2)
        for i, fig in enumerate(equities_figs):
            if i % 2 == 0:
                with tabcol1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with tabcol2:
                    st.plotly_chart(fig, use_container_width=True)
        #this version will generate plots in a line.        #
        #for fig in equities_figs:
        #    st.plotly_chart(fig, use_container_width=True)


    # Tab 2: Fixed Income Scores
    with tab2:
        st.header("Fixed Income Assets Analysis")
        
        with st.popover(" Edit Settings for Analysis Here"):
            st.subheader("Select the Number of Years")
            fi_years = st.slider("Select the number of years:", min_value=0, max_value=10, value=10, step=1, key='fi_years')

        st.subheader("Fixed Income Asset Scores")
        fixed_income_figs = plot_fixed_income_scores(fixed_income, fixed_income_names, fi_years)
        
        #this generates columns based on number of figs:
        #columns = st.columns(len(fixed_income_figs))

        #this version seperates the plots into columnss based on number of figs       
        #for col, fig in zip(columns, fixed_income_figs):
         #   with col:
         #       st.plotly_chart(fig, use_container_width=True)
        
        #this generates the columns for two columns
        tab2col1, tab2col2, = st.columns(2)

        #This version seperates the plots into two columns
        for i, fig in enumerate(fixed_income_figs):
            if i % 2 == 0:
                with tab2col1:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                with tab2col2:
                    st.plotly_chart(fig, use_container_width=True)
       
        #This version will generate plots in a line    
        #for fig in fixed_income_figs:
        #    st.plotly_chart(fig, use_container_width=True)


    # Tab 3: Backtesting
    with tab3:
        
        #name the columns so we can nest the headings and plots inthem
        coleq, colfi = st.columns(2)

        with coleq:# anything tabbed under this goes in the first column
            st.subheader("Equities Backtesting")
            st.subheader("Average Annual Returns")
            equities_returns_fig = plot_avg_return_heatmap(equities, equities_names)
            st.pyplot(equities_returns_fig)

            st.subheader("Sharpe Ratio")
            equities_sharpe_fig = plot_sharpe_ratio_heatmap(equities, equities_names)
            st.pyplot(equities_sharpe_fig)

        with colfi:#anything tabbed under this goes in the second column
            st.subheader("Fixed Income Backtesting")
            st.subheader("Average Annual Returns")
            fi_return_fig = plot_avg_return_heatmap(fixed_income, fixed_income_names)
            st.pyplot(fi_return_fig)

            st.subheader("Sharpe Ratio")
            fi_sharpe_fig = plot_sharpe_ratio_heatmap(fixed_income, fixed_income_names)
            st.pyplot(fi_sharpe_fig)


    # Tab 4: Further Analysis
    with tab4:
        st.header("Further Analysis")

        st.subheader("Equities Ratios Visualization")
        st.write("This is a configurable plot that can plot the average ratio of any column between all the equities.")
        with st.popover("Change Equity Ratio Settings Here"):
            ratio = st.selectbox("Select a ratio to plot:", equities[0].columns, key='equities_select_ratio')
            timeframe_years1 = st.slider("Select the number of years:", min_value=0, max_value=10, value=10, step=1, key='timeframe_years1')
       
        equity_valuation = plot_equities_valuation(equities, equities_names, ratio, timeframe_years1)
        st.plotly_chart(equity_valuation, use_container_width=True)

        st.subheader("Fixed Income Duration Visualization")
        st.write("This function plots all the fixed income's yield vs duration over a specified year parameter.")
        st.write("Can be a useful visualization tool of the fixed income landscape.")
        with st.popover ("Change Fixed Index Duration Settings Here"): 
            timeframe_years2 = st.slider("Select the number of years:", min_value=0, max_value=10, value=10, step=1, key='timeframe_years2')
            indicator = st.selectbox("Select an indicator to plot:", ['Index Yield to Maturity', 'Index Yield to Worst', 'Index OAS', 'Index OAD', 'Index OAC', 'Index Spread'], key='fixed_income_select')
        yield_duration = plot_yield_duration(fixed_income, fixed_income_names, indicator, timeframe_years2)
        st.plotly_chart(yield_duration, use_container_width=False)

        st.subheader("Yield vs Duration Table Visualization")
        st.write("This table shows the average yield and duration of each fixed income asset class over a specified year parameter.")
        with st.popover("Change Yield Duration Settings Here"):   
            timeframe_years3 = st.slider("Select the number of years:", min_value=0, max_value=10, value=10, step=1, key='timeframe_years3')
            yield_duration_table = plot_yield_duration_table(fixed_income, fixed_income_names, timeframe_years3)
        st.write(yield_duration_table, use_container_width=True)

