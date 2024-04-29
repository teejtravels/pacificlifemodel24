import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf  # Install using: pip install yfinance
from datetime import datetime, timedelta
import pandas_datareader as pdr

from sklearn.ensemble import RandomForestRegressor, StackingRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

"""### Data preprocess function"""

def process_data(file_name):
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
        df.filter(like='ROC').columns[0]: 'Monthly Return(%)',
        df.filter(like='Last Price').columns[0]: 'Last Price',
        df.filter(like='P/Bk').columns[0]: 'P/B',
        df.filter(like='Div Yld').columns[0]: 'Div Yld',
        df.filter(like='P/E').columns[0]: 'P/E',
        df.filter(like='LTG EPS').columns[0]: 'LTG EPS',
        df.filter(like='Equity').columns[0]: 'ROE',
        df.filter(like='Beta').columns[0]: 'Raw Beta',
        df.filter(like='Volatility').columns[0]: 'Volatility 30D'
    }
    df.rename(columns=column_names, inplace=True)

    # Fill NaN values using forward and backward fill
    df = df.ffill()
    df = df.bfill()

    return df

"""# Scoring System Functions

### Score Calculation Functions
"""

def calculate_value(data):
    """
    Calculate the average percentile of value indicators.
    This function selects a timeframe using the 'years' variable, calculates the percentile of the first, second, and third value indicators, and then calculates the average of these three percentiles.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the value indicators.

    Returns:
    average_percentile (float): The average percentile of the three value indicators.
    """
    # Select timeframe using years variable
    if years is not None:
        cutoff_date = datetime.now() - pd.DateOffset(years=years)
        data = data.loc[data['Date'] >= cutoff_date]

    # Calculate percentile of first value indicator (P/B)
    last_pb = data['P/B'].iloc[-1]
    # Take the inverse of the percentile because lower P/B is better
    pb_percentile = 100 - stats.percentileofscore(data['P/B'], last_pb, nan_policy='omit')

    # Calculate percentile of second value indicator (Div Yld)
    last_div = data['Div Yld'].iloc[-1]
    div_percentile = stats.percentileofscore(data['Div Yld'], last_div, nan_policy='omit')

    # Calculate percentile of third value indicator (P/E)
    last_pe = data['P/E'].iloc[-1]
    # Take the inverse of the percentile because lower P/E is better
    pe_percentile = 100 - stats.percentileofscore(data['P/E'], last_pe, nan_policy='omit')

    # Calculate the average of the three percentiles
    average_percentile = (pb_percentile + div_percentile + pe_percentile) / 3

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

    # Calculate percentile of first growth indicator (LTG EPS)
    last_eps = data['LTG EPS'].iloc[-1]
    eps_percentile = stats.percentileofscore(data['LTG EPS'], last_eps, nan_policy='omit')

    # Calculate percentile of second growth indicator (ROE)
    last_roe = data['ROE'].iloc[-1]
    roe_percentile = stats.percentileofscore(data['ROE'], last_roe, nan_policy='omit')

    # Calculate the average of the two percentiles
    average_percentile = (eps_percentile + roe_percentile) / 2

    return average_percentile



def calculate_sentiment(data):
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

    # Calculate percentile of first sentiment indicator (Raw Beta)
    last_beta = data['Raw Beta'].iloc[-1]
    beta_percentile = stats.percentileofscore(data['Raw Beta'], last_beta, nan_policy='omit')

    # Calculate percentile of second sentiment indicator (Volatility 30D)
    last_volatility = data['Volatility 30D'].iloc[-1]
    volatility_percentile = stats.percentileofscore(data['Volatility 30D'], last_volatility, nan_policy='omit')

    # Calculate the average of the two percentiles
    average_percentile = (beta_percentile + volatility_percentile) / 2

    # Inverse the percentile because lower beta and volatility is considered better
    return 100 - average_percentile



def calculate_final_score(data):
    """
    Calculate the final score of an asset class
    This function calculates the final score of an asset class based on the value, growth, and sentiment scores.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the value, growth, and sentiment indicators.

    Returns:
    final_score (float): The final score of the asset class.
    """
    value_score = calculate_value(data)
    growth_score = calculate_growth(data)
    sentiment_score = calculate_sentiment(data)

    final_score = (value_score * value_weight) + (growth_score * growth_weight) + (sentiment_score * sentiment_weight)

    return final_score

"""### Display Results Functions"""

def print_result(data):
    """
    Print the final score of an asset class
    """
    value_score = calculate_value(data)
    growth_score = calculate_growth(data)
    sentiment_score = calculate_sentiment(data)
    final_score = (value_score * value_weight) + (growth_score * growth_weight) + (sentiment_score * sentiment_weight)

    result = print("Timeframe (yrs):", years, "\nValue Score (%): ", value_score, "\nGrowth Score (%) : ", growth_score, "\nSentiment Score (%) : ", sentiment_score, "\nFinal Score (%): ", final_score)

    return result


def plot_scores(data):
    """
    Plot the scores of the asset classes
    This function calculates the scores for each category and creates a bar plot to visualize the scores.

    Parameters:
    data (list): A list of DataFrames containing the value, growth, and sentiment indicators for each asset class.

    Returns:
    Bar plot of the scores for each asset class.
    """
    # Create a dictionary to store the scores for each category
    scores = {'Final': [], 'Value': [], 'Growth': [], 'Sentiment': []}

    # Calculate the scores for each category
    for df in data:
        scores['Value'].append(calculate_value(df))
        scores['Growth'].append(calculate_growth(df))
        scores['Sentiment'].append(calculate_sentiment(df))
        scores['Final'].append(calculate_final_score(df))

    # Create a dataframe from the scores dictionary
    scores_df = pd.DataFrame(scores)

    # Create a list of labels for the x-axis
    x_labels = names

    # Create a bar plot of the scores
    for i, category in enumerate(scores):
        # Create a new column for color based on whether the score is the highest or not
        scores_df['color'] = ['Highest Score' if x == scores_df[category].max() else 'Other Scores' for x in scores_df[category]]

        fig = px.bar(scores_df, y=x_labels, x=category, color='color', orientation='h', title=f'Scores of Asset Classes for {category}',
                     color_discrete_map={'Highest Score': 'blue', 'Other Scores': 'darkgray'},
                     text=scores_df[category].round(2))
        fig.show()

"""# Backtesting Functions

## Using Average Returns

#### Calculate Avg Returns Function
"""

def calculate_return(df_list, names):
    """
    Calculate the average returns for each dataset in the list.
    This function calculates the average returns for each dataset in the list over different time frames (1, 5, 10, 15, 20 years).

    Parameters:
    df_list (list): A list of DataFrames containing the returns data for each asset class.

    Returns:
    results_df (pd.DataFrame): A DataFrame containing the average returns for each dataset over different time frames.
    """
    all_returns = []  # This will store the average returns for each dataset
    time_frames = [1, 5, 10, 15, 20]

    for df in df_list:
        returns = []  # This will store the average returns for the current dataset across time frames
        latest_date = pd.to_datetime(df['Date']).max()

        for years in time_frames:
            start_date = latest_date - pd.DateOffset(years=years)
            # Filter the DataFrame based on the time frame
            filtered_df = df[(pd.to_datetime(df['Date']) > start_date) & (pd.to_datetime(df['Date']) <= latest_date)]

            # Calculate the average return for the current time frame using the 'Monthly Return(%)' column
            avg_return = round(filtered_df['Monthly Return(%)'].mean(), 2)
            returns.append(avg_return)

        # Append the calculated average returns for this dataset to the all_returns list
        all_returns.append(returns)

    # Create DataFrame for visualization
    results_df = pd.DataFrame(all_returns, index=names, columns=time_frames).transpose()

    # Reset the index to make it a regular column and rename it to 'Timeframe(yr)'
    results_df = results_df.reset_index().rename(columns={'index': 'Timeframe(yr)'})

    return results_df

"""### Line Chart"""

def plot_avg_return_line_chart(results):
    """
    Plot the average returns for each asset class over different time frames.
    This function creates a line chart to visualize the average returns for each asset class over different time frames.

    Parameters:
    results (pd.DataFrame): A DataFrame containing the average returns for each dataset over different time frames.

    Returns:
    Line chart of the average returns for each asset class over different time frames.
    """
    fig = go.Figure()

    # Add a line for each asset
    for asset in results.columns[1:]:  # Skip the first column as it's 'Timeframe(yr)'
        fig.add_trace(go.Scatter(
            x=results['Timeframe(yr)'],
            y=results[asset],
            mode='lines+markers',
            name=asset
        ))

    # Update layout
    fig.update_layout(
        title='Average Return by Retrospective Timeframe',
        xaxis_title='Retrospective Timeframe (Years)',
        yaxis_title='Average Return (%)',
        legend_title='Asset',
        xaxis=dict(type='category')  # Treat timeframes as categorical data for consistent spacing
    )

    fig.show()

"""### Heat Maps"""

def plot_returns_heatmap(results):
    """
    This function plots a heatmap of returns for different assets over various timeframes.

    Parameters:
    results (DataFrame): A DataFrame containing the returns of different assets over various timeframes.
    """
    heatmap_df = results.set_index('Timeframe(yr)').transpose()

    # Find the maximum IRR value and its position
    max_irr_value = heatmap_df.values.max()
    max_irr_pos = np.where(heatmap_df.values == max_irr_value)

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,  # Timeframes
        y=heatmap_df.index,  # Dataset names
        colorscale='RdBu',
        colorbar=dict(title='Return')
    ))

    # Highlight the best-performing asset in each timeframe with a transparent background
    for timeframe in heatmap_df.columns:
        # Find the index of the max IRR value in this timeframe
        max_irr_index = heatmap_df[timeframe].idxmax()
        max_irr_value = heatmap_df[timeframe].max()

        fig.add_annotation(
            x=timeframe,
            y=max_irr_index,
            text=f"Best: {max_irr_value:.2f}%",  # Format the IRR value to 2 decimal places
            showarrow=False,
            font=dict(
                color="black",
                size=12
            ),
            bgcolor="rgba(255, 255, 255, 0.5)"  # Use RGBA for transparent red background
        )

    title_text = 'Average Return by Retrospective Timeframe<br>' \
                 '<span style="font-size: 14px;">Initial Investment: $100</span>'

    fig.update_layout(
        title= title_text,
        xaxis_title='Retrospective Timeframe (Years)',
        yaxis_title='Dataset',
        xaxis=dict(tickmode='array', tickvals=results['Timeframe(yr)']),
        width=900,  # Set width to 600 pixels
        height=400,  # Set height to 400 pixels
    )

    fig.show()


def plot_avg_return_heatmap_with_top3(results):
    """
    This function plots a heatmap of returns for different assets over various timeframes, highlighting the top 3 performers.

    Parameters:
    results (DataFrame): A DataFrame containing the returns of different assets over various timeframes.
    """
    # Ensure 'Timeframe(yr)' is a column and not an index
    if 'Timeframe(yr)' not in results.columns:
        results = results.reset_index().rename(columns={'index': 'Timeframe(yr)'})

    heatmap_df = results.set_index('Timeframe(yr)').transpose()

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,  # Timeframes
        y=heatmap_df.index,  # Dataset names
        colorscale='RdBu',
        colorbar=dict(title='Average Return (%)')
    ))

    # Highlight the top 3 performing assets in each timeframe
    for timeframe in heatmap_df.columns:
        # Sort the assets in this timeframe by their return, keeping the top 3
        top3_assets = heatmap_df[timeframe].sort_values(ascending=False)[:3]

        # Enumerate over the top 3 to add annotations for each
        for rank, (asset, value) in enumerate(top3_assets.items(), start=1):
            fig.add_annotation(
                x=timeframe,
                y=asset,
                text=f"Top {rank}: {value:.2f}%",  # Format the value to 2 decimal places
                showarrow=False,
                font=dict(
                    color="black",
                    size=12
                ),
                bgcolor=f"rgba(255, 255, 255, {0.8 - 0.2 * (rank-1)})"  # Decrease transparency for lower ranks
            )

    title_text = 'Annual Average Return by Retrospective Timeframe<br>' \
                 '<span style="font-size: 14px;">Highlighting Top 3 Performers</span>'

    fig.update_layout(
        title=title_text,
        xaxis_title='Retrospective Timeframe (Years)',
        yaxis_title='Dataset',
        width=1200,
        height=400
    )

    fig.show()

"""## Using Sharpe Ratio

### Sharpe Ratio Results
"""

def get_treasury_bill_rate():
    # Fetch the Treasury bill rate using Yahoo Finance API
    # Here, we fetch the 3-month Treasury bill rate (change as needed)
    tbill_data = yf.download("^IRX", start=pd.to_datetime('today') - pd.Timedelta(days=1), end=pd.to_datetime('today'))
    tbill_rate = tbill_data['Close'].iloc[-1] / 100  # Convert percentage to decimal

    return tbill_rate

def calculate_sharpe_ratio(df_list, names):
    all_sharpes = []  # This will store the Sharpe ratios for each dataset
    time_frames = [1, 5, 10, 15, 20]

    risk_free_rate = get_treasury_bill_rate()

    for df in df_list:
        sharpes = []  # This will store the Sharpe ratios for the current dataset across time frames
        latest_date = pd.to_datetime(df['Date']).max()

        for years in time_frames:
            start_date = latest_date - pd.DateOffset(years=years)
            filtered_df = df[(pd.to_datetime(df['Date']) > start_date) & (pd.to_datetime(df['Date']) <= latest_date)]

            # Calculate the excess return over the risk-free rate
            returns = filtered_df.iloc[:, 9]  # Assuming returns are in the 10th column
            excess_return = returns - risk_free_rate

            # Calculate the Sharpe ratio
            sharpe_ratio = round((excess_return.mean() / returns.std()) * np.sqrt(252),2)  # Assuming 252 trading days in a year
            sharpes.append(sharpe_ratio)

        # Append the calculated Sharpe ratios for this dataset to the all_sharpes list
        all_sharpes.append(sharpes)

    # Create DataFrame for visualization
    results_df = pd.DataFrame(all_sharpes, index=names, columns=time_frames).transpose()
    results_df = results_df.reset_index().rename(columns={'index': 'Timeframe(yr)'})

    return results_df

"""### Visualization Functions"""

# Heatmap with top 3 returns
def plot_sharpe_ratio_heatmap_with_top3(results):
    # Ensure 'Timeframe(yr)' is a column and not an index
    if 'Timeframe(yr)' not in results.columns:
        results = results.reset_index().rename(columns={'index': 'Timeframe(yr)'})

    heatmap_df = results.set_index('Timeframe(yr)').transpose()

    fig = go.Figure(data=go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns,  # Timeframes
        y=heatmap_df.index,  # Dataset names
        colorscale='RdBu',
        colorbar=dict(title='Sharpe Ratio (%)')
    ))

    # Highlight the top 3 performing assets in each timeframe
    for timeframe in heatmap_df.columns:
        # Sort the assets in this timeframe by their return, keeping the top 3
        top3_assets = heatmap_df[timeframe].sort_values(ascending=False)[:3]

        # Enumerate over the top 3 to add annotations for each
        for rank, (asset, value) in enumerate(top3_assets.items(), start=1):
            fig.add_annotation(
                x=timeframe,
                y=asset,
                text=f"Top {rank}: {value:.2f}%",  # Format the value to 2 decimal places
                showarrow=False,
                font=dict(
                    color="black",
                    size=12
                ),
                bgcolor=f"rgba(255, 255, 255, {0.8 - 0.2 * (rank-1)})"  # Decrease transparency for lower ranks
            )

    title_text = 'Annual Sharpe Ratio by Retrospective Timeframe<br>' \
                 '<span style="font-size: 14px;">Highlighting Top 3 Performers</span>'

    fig.update_layout(
        title=title_text,
        xaxis_title='Retrospective Timeframe (Years)',
        yaxis_title='Dataset',
        width=1200,
        height=400
    )

    fig.show()

"""# Equities

### Import equities data
"""
large_cap_growth = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RLG.xlsx")
large_cap_value = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RLV.xlsx")
mid_cap_growth = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RDG.xlsx")
mid_cap_value = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RMV.xlsx")
small_cap_growth = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RUO.xlsx")
small_cap_value = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_RUJ.xlsx")
international_growth = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_MXEA000G.xlsx")
international_value = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_MXEA000V.xlsx")
international_small_cap = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_SBERWUU.xlsx")
emerging_markets = process_data("/workspaces/pacificlifeuci2024/data/20y_monthly_MXEF.xlsx")


# Store the dataframes in a list
equities = [large_cap_growth, large_cap_value, mid_cap_growth, mid_cap_value, small_cap_growth, small_cap_value, international_growth, international_value, international_small_cap, emerging_markets]

"""### Parameters (configure before running functions)"""

# years determines the timeframe of the data, i.e. the last 5 years, 10 years, etc.
years = 10

# set weights parameters
value_weight = 0.4
growth_weight = 0.4
sentiment_weight = 0.3

# Set names of the asset classes
names = ['Large Cap Growth', 'Large Cap Value', 'Mid Cap Growth', 'Mid Cap Value', 'Small Cap Growth', 'Small Cap Value', 'International Growth', 'International Value', 'International Small Cap', 'Emerging Markets']

"""### Scores of equity assets"""

# Plot the scores
plot_scores(equities)

"""### Backtesting on Equities"""

# Establish names of each column for returns backtest
names = ["AVG Return of Large-Cap Growth(%)", "AVG Return of Large-Cap Value(%)", "AVG Return of Mid-Cap Growth(%)", "AVG Return of Mid-Cap Value(%)", "AVG Return of Small-Cap Growth(%)", "AVG Return of Small-Cap Value(%)" , "AVG Return of International Growth(%)", "AVG Return of International Value(%)", "AVG Return of International Small Cap(%)", "AVG Return of Emerging Markets(%)"]


equities_results = calculate_return(equities, names)

display(equities_results)
display(plot_avg_return_heatmap_with_top3(equities_results))
display(plot_avg_return_line_chart(equities_results))

"""## Forecasting Next Month's Return

### Model 1: Random Forest
"""

def train_and_evaluate_rf(data, asset_name, target_column, test_size=0.2, random_state=42, n_estimators=100):
    # Prepare the features (X) and the target (y)
    X = data.drop(['Date', target_column], axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and train the model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(X_train, y_train)

    # Predict using the latest data
    latest_data = X.iloc[[-1]]  # Assuming the latest data is at the last row after sorting the Date column by ascending order
    predicted_return = rf_model.predict(latest_data)

    # Calculate and print model accuracy on test set for reference
    predictions_test = rf_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    rmse_rounded = round(rmse, 4)
    print(f'Model RMSE on test set for {asset_name}: {rmse_rounded}')

    return predicted_return[0]

def forecast_returns_rf(dataframes, asset_names, target_column):
    results = {}
    for data, asset in zip(dataframes, asset_names):
        predicted_return = train_and_evaluate_rf(data, asset, target_column)
        results[asset] = predicted_return  # Store results using the asset name as the key
        print(f'Predicted Return for {asset} Next Month: {predicted_return:.2f}%\n')

    return results

"""### Results"""

# Target column name
target_column = 'Monthly Return(%)'

# Assuming 'large_cap_growth', 'large_cap_value', etc., are DataFrame variables containing your data
dataframes = [large_cap_growth, large_cap_value, mid_cap_growth, mid_cap_value, small_cap_growth, small_cap_value]
asset_names = ['Large Cap Growth', 'Large Cap Value', 'Mid Cap Growth', 'Mid Cap Value', 'Small Cap Growth', 'Small Cap Value']

# Forecast returns for all assets
forecast_results = forecast_returns_rf(dataframes, asset_names, target_column)

"""### Model 2: Stacked Random Forest Model via SVR"""

def train_and_evaluate_svr(data, asset_name, target_column, test_size=0.2, random_state=42, n_estimators=100):
    # Prepare the features (X) and the target (y)
    X = data.drop(['Date', target_column], axis=1)
    y = data[target_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the base models
    base_models = [
        ('rf1', RandomForestRegressor(n_estimators=100, random_state=random_state)),
        ('rf2', RandomForestRegressor(n_estimators=100, random_state=random_state))
    ]

    # Define the meta-model
    meta_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='squared_error')

    # Define the stacking regressor
    stacking_model = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # Train the stacking regressor
    stacking_model.fit(X_train, y_train)

    # Predict using the latest data
    latest_data = X.iloc[[-1]]  #  the latest data is at the last row after sorting the Date column by ascending order
    predicted_return = stacking_model.predict(latest_data)

    # Calculate and print model accuracy on test set for reference
    predictions_test =  stacking_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions_test))
    rmse_rounded = round(rmse, 4)
    print(f'Model RMSE on test set for {asset_name}: {rmse_rounded}')

    return predicted_return[0]

def forecast_returns_svr(dataframes, asset_names, target_column):
    results = {}
    for data, asset in zip(dataframes, asset_names):
        predicted_return = train_and_evaluate_svr(data, asset, target_column)
        results[asset] = predicted_return  # Store results using the asset name as the key
        print(f'Predicted Return for {asset} Next Month: {predicted_return:.2f}%\n')

    return results

# Target column name
target_column = 'Monthly Return(%)'

# Assuming 'large_cap_growth', 'large_cap_value', etc., are DataFrame variables containing your data
dataframes = [large_cap_growth, large_cap_value, mid_cap_growth, mid_cap_value, small_cap_growth, small_cap_value]
asset_names = ['Large Cap Growth', 'Large Cap Value', 'Mid Cap Growth', 'Mid Cap Value', 'Small Cap Growth', 'Small Cap Value']

# Forecast returns for all assets
forecast_results = forecast_returns_svr(dataframes, asset_names, target_column)

"""# Fixed Income
## Macro- Indicators
"""

start_date = '2002-01-01'  # 20 years ago
end_date = '2024-04-17'

# Define the ticker symbol for the yield curve data
tickers = ['DGS1','DGS3','DGS10','DGS20']

yield_curve_data = pd.DataFrame()
for ticker in tickers:
    data = pdr.get_data_fred(ticker, start=start_date, end=end_date)
    data_monthly = data.resample('M').last()
    yield_curve_data[ticker] = data_monthly[ticker]

# Print the first few rows of the downloaded data
yield_curve_data.tail()

inflation = pdr.get_data_fred('CPIAUCNS', start=start_date, end=end_date)
inflation = inflation.resample('M').last()
inflation['YoY Inflation Rate (%)'] = inflation['CPIAUCNS'].pct_change(12) * 100

inflation.tail()







##Below this line is the Origional dashboard ###  above is model code### 

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
st.set_page_config(layout="wide")
# Title of the app
st.title('Pacific Life Asset Comparison App')



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