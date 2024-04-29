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