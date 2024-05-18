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

def calculate_valuation(data):
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


def calculate_growth(data):
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


def calculate_leverage(data):
    """
    Calculate the average percentile of sentiment indicators.
    This function selects a timeframe using the 'years' variable, calculates the percentile of the sentiment indicators, and then calculates the average of the percentiles.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the sentiment indicators.

    Returns:
    average_percentile (float): The average percentile of the sentiment indicators.
    """
    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of first sentiment indicator (Total Debt to EV)
    last_dev = data['Total Debt to EV'].iloc[-1]
    dev_percentile = 100 - stats.percentileofscore(data['Total Debt to EV'], last_dev, kind='rank', nan_policy='omit')

    # Calculate percentile of second sentiment indicator (Net Debt/EBITDA)
    last_de_ebitda= data['Net Debt/EBITDA'].iloc[-1]
    de_ebitda_percentile = 100 - stats.percentileofscore(data['Net Debt/EBITDA'], last_de_ebitda, kind='rank', nan_policy='omit')

    # Calculate percentile of third sentiment indicator (Total Debt to Total Equity)
    last_debt_equity= data['Total Debt to Total Equity'].iloc[-1]
    debt_equity_percentile = 100 - stats.percentileofscore(data['Total Debt to Total Equity'], last_debt_equity, kind='rank', nan_policy='omit')

    # Calculate percentile of fourth sentiment indicator (Total Debt to Total Asset)
    last_debt_asset= data['Total Debt to Total Asset'].iloc[-1]
    debt_asset_percentile = 100 - stats.percentileofscore(data['Total Debt to Total Asset'], last_debt_asset, kind='rank', nan_policy='omit')

    # Calculate the average of the 4 percentiles
    average_percentile = (dev_percentile + de_ebitda_percentile + debt_equity_percentile + debt_asset_percentile) / 4
    
    # Inverse the percentile because lower beta and volatility is considered better
    return average_percentile


def calculate_final_score(data):
    """
    Calculate the final score of an asset class
    This function calculates the final score of an asset class based on the value, growth, and sentiment scores.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the value, growth, and sentiment indicators.

    Returns:
    final_score (float): The final score of the asset class.
    """
    value_score = calculate_valuation(data)
    growth_score = calculate_growth(data)
    sentiment_score = calculate_leverage(data)
    
    final_score = (value_score * valuation_weight) + (growth_score * growth_weight) + (sentiment_score * leverage_weight)

    return final_score


## Equities Display Results Functions

def plot_equities_scores(data, names):
    """
    Plot the scores of the asset classes using Plotly.
    This function calculates the scores for each category and creates a bar plot to visualize the scores.

    Parameters:
    data (list): A list of DataFrames containing the value, growth, and sentiment indicators for each asset class.
    names (list): A list of strings containing the names of each asset class.

    Returns:
    A bar plot of the scores for each asset class using Plotly.
    """
    # Create a dictionary to store the scores for each category
    scores = {'Final': [], 'Valuation': [], 'Growth': [], 'Leverage': []}
    
    # Calculate the scores for each category
    for df in data:
        scores['Valuation'].append(calculate_valuation(df))
        scores['Growth'].append(calculate_growth(df))
        scores['Leverage'].append(calculate_leverage(df))
        scores['Final'].append(calculate_final_score(df))

    # Create a dataframe from the scores dictionary and add asset names as a column
    scores_df = pd.DataFrame(scores, index=names)
    scores_df.reset_index(inplace=True)
    scores_df.rename(columns={'index': 'Asset Class'}, inplace=True)

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
    
    return fig


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

def calculate_fixed_income_value(data):
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


def calculate_fixed_income_volatility(data):
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


def calculate_fixed_income_coupon(data):
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


def calculate_fixed_income_duration(data):
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

def plot_fixed_income_scores(data, names):
    """Plots scores of fixed income asset classes with top 3 highlighting."""
    scores = {'Value': [], 'Volatility': [], 'Coupon': [], 'Duration': []}

    # Calculate the scores for each category
    for df in data:
        scores['Value'].append(calculate_fixed_income_value(df))
        scores['Volatility'].append(calculate_fixed_income_volatility(df))
        scores['Coupon'].append(calculate_fixed_income_coupon(df))
        scores['Duration'].append(calculate_fixed_income_duration(df))
    
    scores_df = pd.DataFrame(scores, index=names)
    scores_df.reset_index(inplace=True)
    scores_df.rename(columns={'index': 'Asset Class'}, inplace=True)

    figures = []  # Store all figure objects

    for category in scores:
        # Color assignment for top 3
        scores_df = scores_df.sort_values(by=category, ascending=True)
        scores_df['color'] = ['Other Scores' for _ in range(len(scores_df))]
        scores_df.iloc[-3:, scores_df.columns.get_loc('color')] = ['Third Highest Score', 'Second Highest Score', 'Highest Score']

        fig = px.bar(scores_df, y='Asset Class', x=category, color='color', orientation='h',
                     title=f'Scores of Asset Classes for {category}',
                     color_discrete_map={'Highest Score': 'navy', 'Second Highest Score': 'royalblue', 'Third Highest Score': 'skyblue', 'Other Scores': 'lightgray'},
                     text=scores_df[category].round(2))

        figures.append(fig)  # Add the figure to the list

    return figures  # Return the list of figures, i.e. figures[0] for Value, figures[1] for Volatility, etc.


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
    tbill_data = yf.download("^IRX", start=pd.to_datetime('today') - pd.Timedelta(days=1), end=pd.to_datetime('today'))
    tbill_rate = tbill_data['Close'].iloc[-1] / 100  # Convert percentage to decimal

    return tbill_rate


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


#categories = {
#    "RLG": "large_cap_growth",
#    "RLV": "large_cap_value",
#    "RDG": "mid_cap_growth",
#    "RMV": "mid_cap_value",
#    "RUO": "small_cap_growth",
#    "RUJ": "small_cap_value",
#    "MXEA000G": "international_growth",
#    "MXEA000V": "international_value",
#    "SBERWUU": "international_small_cap",
#    "MXEF": "emerging_markets"
#}

#uploaded_file = st.sidebar.file_uploader("Upload to replace equity data:", type=['xlsx'])

#if uploaded_file:
#    file_category = None
#    # Determine which category the file belongs to based on its name
#    for key, value in categories.items():
#        if key in uploaded_file.name:
#            file_category = value
#            break
    
#    if file_category:
#        # Process and update the specific category data
#        equity_data[file_category] = process_data(uploaded_file)
#        st.success(f"Updated data for {file_category}.")
#    else:
#        st.error("The uploaded file does not match any recognized category.")


#Bangyangs updates above this line


# Equities Results

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

## set weights parameters
valuation_weight = 0.4
growth_weight = 0.4
leverage_weight = 0.2

## Plot the equities scores
plot_equities_scores(equities, equities_names)


## Backtesting on Equities

## Rolling Annual Return Backtesting
plot_avg_return_heatmap(equities, equities_names)

## Sharpe Ratio Backtesting
plot_sharpe_ratio_heatmap(equities, equities_names)


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

## Set default years parameter
years = 10

## Plot the fixed income scores
plot_fixed_income_scores(fixed_income, fixed_income_names)

## Backtesting on Fixed Income

## Rolling Annual Return Backtesting
plot_avg_return_heatmap(fixed_income, fixed_income_names)

## Sharpe Ratio Backtesting
plot_sharpe_ratio_heatmap(fixed_income, fixed_income_names)


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
    fig.show()

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
    fig.show()

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

    print(df_results.to_string(index=False))  # Improved table formatting (optional)


### Below this is streamlit dashboard ###

# Initialize session state for weights
if 'valuation_weight' not in st.session_state:
    st.session_state['valuation_weight'] = 0.4
if 'growth_weight' not in st.session_state:
    st.session_state['growth_weight'] = 0.4
if 'leverage_weight' not in st.session_state:
    st.session_state['leverage_weight'] = 0.2

# Function to validate weights
def validate_weights():
    total = st.session_state.valuation_weight + st.session_state.growth_weight + st.session_state.leverage_weight
    if total != 1.0:
        diff = 1.0 - total
        st.session_state.sentiment_weight += diff
        st.error("The total of weights has been adjusted to 1.0 by modifying the sentiment weight.")

# Define file processing and plotting functions here...

# Load and process data
equities_data_files = [
    "streamlit/data/RLG_10y_monthly.xlsx", "streamlit/data/RLV_10y_monthly.xlsx",
    "streamlit/data/RDG_10y_monthly.xlsx", "streamlit/data/RMV_10y_monthly.xlsx",
    "streamlit/data/RUO_10y_monthly.xlsx", "streamlit/data/RUJ_10y_monthly.xlsx",
    "streamlit/data/MXEA000G_10y_monthly.xlsx", "streamlit/data/MXEA000V_10y_monthly.xlsx",
    "streamlit/data/MXEF_10y_monthly.xlsx", "streamlit/data/SBERWUU_10y_monthly.xlsx"
]

equities_names = [
    'Large Cap Growth', 'Large Cap Value', 'Mid Cap Growth', 'Mid Cap Value', 'Small Cap Growth', 
    'Small Cap Value', 'International Growth', 'International Value', 'Emerging Markets Equity', 
    'Small Cap International'
]

equities = [process_equities_data(file) for file in equities_data_files]

fixed_income_data_files = [
    "streamlit/data/10y_monthly_core_bond.xlsx", "streamlit/data/10y_monthly_emerging_bond.xlsx",
    "streamlit/data/10y_monthly_floating_rate.xlsx", "streamlit/data/10y_monthly_high_yield.xlsx",
    "streamlit/data/10y_monthly_short_term.xlsx", "streamlit/data/10y_monthly_tips.xlsx"
]

fixed_income_names = [
    'Core Bond', 'Emerging Bond', 'Floating Rate Bond', 'High Yield Bond', 
    'Short Term Bond', 'TIPS'
]

fixed_income = [process_fi_data(file) for file in fixed_income_data_files]

# Streamlit dashboard layout
if __name__ == '__main__':
    st.title('Financial Data Visualization and Backtesting')
    tab1, tab2, tab3, tab4 = st.tabs(["Asset Scores", "🗃 Settings", "EDA", "📈 Charts"])

    # Tab 1: Asset Scores
    with tab1:
        st.header("Asset Scores")
        years = st.slider("Select the number of years:", min_value=5, max_value=10, value=10, step=5)

        st.subheader("Equity Asset Scores")
        equities_figs = plot_fixed_income_scores(equities, equities_names)
        for fig in equities_figs:
            st.plotly_chart(fig)

        st.subheader("Fixed Income Asset Scores")
        fixed_income_figs = plot_fixed_income_scores(fixed_income, fixed_income_names)
        for fig in fixed_income_figs:
            st.plotly_chart(fig)

    # Tab 2: Settings
    with tab2:
        st.header("Settings")
        st.subheader("Adjust Weights")
        st.number_input("Valuation Weight", min_value=0.0, max_value=1.0, value=st.session_state.valuation_weight, key='valuation_weight', on_change=validate_weights)
        st.number_input("Growth Weight", min_value=0.0, max_value=1.0, value=st.session_state.growth_weight, key='growth_weight', on_change=validate_weights)
        st.number_input("Leverage Weight", min_value=0.0, max_value=1.0, value=st.session_state.leverage_weight, key='leverage_weight', on_change=validate_weights)

        st.write(f"Total: {st.session_state.valuation_weight + st.session_state.growth_weight + st.session_state.leverage_weight}")

    # Tab 3: EDA (Exploratory Data Analysis)
    with tab3:
        st.header("Exploratory Data Analysis")
        # Add EDA plots and tables here

    # Tab 4: Charts
    with tab4:
        st.header("Charts")
        st.subheader("Rolling Annual Return Backtesting")
        return_heatmap_fig = plot_avg_return_heatmap(equities, equities_names)
        st.pyplot(return_heatmap_fig)

        st.subheader("Sharpe Ratio Backtesting")
        sharpe_heatmap_fig = plot_sharpe_ratio_heatmap(equities, equities_names)
        st.pyplot(sharpe_heatmap_fig)

        st.subheader("Fixed Income Yield vs Duration")
        yield_duration_fig = plot_yield_duration(fixed_income, fixed_income_names, 'Index Yield to Maturity', years)
        st.plotly_chart(yield_duration_fig)

        st.subheader("Fixed Income Yield vs Duration Table")
        yield_duration_table_fig = plot_yield_duration_table(fixed_income, fixed_income_names, years)
        st.pyplot(yield_duration_table_fig)



### Below this line is the Origional dashboard ### 

import streamlit as st
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def pe_ratio_plot(df, filename):
    # Assuming that the date is in the first column, last price in the second column, and P/E ratio in the fifth column
    plt.figure(figsize=(10, 4))
    ax1 = plt.gca()  # Get current axis
    # Using iloc to avoid column name issues, ensuring we use the correct columns
    sns.lineplot(x=df.iloc[:, 0], y=df.iloc[:, 1], color='red', label='Last Price', ax=ax1)
    ax1.set_ylabel('Last Price', color='red')
    ax2 = ax1.twinx()
    sns.lineplot(x=df.iloc[:, 0], y=df.iloc[:, 4], color='blue', label='P/E Ratio', ax=ax2)
    ax2.set_ylabel('P/E Ratio', color='blue')
    plt.title(filename)
    plt.legend()
    st.pyplot(plt)  # Display the plot in Streamlit


# sets the page as wide mode
#st.set_page_config(layout="wide")
# Title of the app
#st.title('Pacific Life Asset Comparison App')



# defines the asset classes used : 
asset_classes = [
    "Large Cap Growth", "Large Cap Value",
    "US Mid Cap Growth", "US Mid Cap Value",
    "US Small Cap Growth", "US Small Cap Value",
    "REITs", "Intl Equity",
    "Intl Small Cap Equity", "Emerging Market Equity",
    "US High Yield Bonds", "US TIPS Floating Rate Loan",
    "Emerging Market Bonds", "US Agg Bonds",
    "US Short-term Bonds", "Cash", "Alternatives"
]

# File uploader allows user to add multiple files with a unique key
# Using st.multiselect allows users to select multiple options from a dropdown
uploaded_files = st.sidebar.file_uploader("Upload Financial Data Here:", accept_multiple_files=True, type='csv')
selected_classes = st.sidebar.multiselect("Select Asset Classes:", asset_classes)

# Convert selected classes to lowercase for consistency in processing
selected_classes = [cls.lower() for cls in selected_classes]


#creates a dictionary to hold dataframes for selected asset class
dfs = {}
cols = st.columns(1)  # Create a single column for dataframes
# Check if dfs is not empty
if dfs:
    # Create columns for dataframes
    cols = st.columns(len(dfs))

    # Display dataframes in columns
    for idx, (filename, df) in enumerate(dfs.items()):
        with cols[idx]:
            pe_ratio_plot(df, filename)  # Call to the newly named function
else:
    st.write("No dataframes to display.")

# Create separate column for metrics
with st.container():
    st.header("Metrics")

    # Create two columns for metrics
    metric_cols = st.columns(2)

    # Display metrics in separate columns
    with metric_cols[0]:
        metric1 = 123  # Placeholder for actual metrics calculation
        st.metric(label="investment score", value=4, delta=-0.5, delta_color="inverse")

    with metric_cols[1]:
        metric2 = 456  # Placeholder for actual metrics calculation
        st.metric(label="investment score2", value=55, delta=-33, delta_color="off ")
# Check if files have been uploaded and asset classes selected
if uploaded_files and selected_classes:
    for uploaded_file in uploaded_files:
        if any(selected_class in uploaded_file.name.lower() for selected_class in selected_classes):
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure date column is datetime
            dfs[uploaded_file.name] = df

    cols = st.columns(len(dfs)+1)
    for idx, (filename, df) in enumerate(dfs.items()):
        with cols[idx]:
            pe_ratio_plot(df, filename)  # Call to the newly named function
else:
    st.write("Please select at least one asset class and upload the corresponding CSV files in the sidebar.")

# Display metrics outside of the main data processing loop
# with st.container():
#     metric_col = cols[-1]
#     st.header("Metrics")
#     metric1 = 123  # Placeholder for actual metrics calculation
#     metric2 = 456  # Placeholder for actual metrics calculation
#     st.metric(label="investment score",value=4, delta=-0.5,
#     delta_color="inverse")
#     st.metric( label="investment score2", value=55, delta=-33,
#     delta_color="off ")


## goal is to have the colors and layout by 26th 