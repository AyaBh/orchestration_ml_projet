import streamlit as st
import pandas as pd
import json
import plotly.express as px

import base64

# Function to get the base64 string of an image
def get_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load predictions from the JSON file
with open('predictions.json', 'r') as f:
    predictions = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(predictions, columns=['station_name', 'bikes_available'])

# Aggregate the number of bikes available per station
df_aggregated = df.groupby('station_name', as_index=False).sum()

# Round the number of bikes available
df_aggregated['bikes_available'] = df_aggregated['bikes_available'].round().astype(int)

# Add Font Awesome for icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """,
    unsafe_allow_html=True
)

# Get the base64 string of the image
img_base64 = get_base64("background.jpg")  # Adjusted the path to match the root location

# Display the image at the beginning
st.markdown(
    f"""
    <div style="text-align: center; margin-bottom: 40px;">
        <img src="data:image/jpeg;base64,{img_base64}" style="width: 40%; border-radius: 5px;">
    </div>
    """,
    unsafe_allow_html=True
)

# Use markdown for better styling and alignment
st.markdown(
    f"""
    <div style="background-color: #f0f0f5; padding: 10px; border-radius: 5px; position: relative;">
        <!-- Bicycle Icon -->
        <div style="position: absolute; top: 10px; left: 10px; font-size: 40px; color: #007BFF;">
            <i class="fas fa-bicycle"></i>
        </div>
        <h2 style="color: #007BFF; margin-left: 90px; line-height: 1.2;font-size: 20px;">Total number of stations: {df_aggregated['station_name'].nunique()}</h2>
        <!-- Map Marker Icon -->
        <div style="position: absolute; top: 70px; left: 10px; font-size: 40px; color: #28A745;">
            <i class="fas fa-map-marker-alt"></i>
        </div>
        <h2 style="color: #28A745; margin-left: 90px; line-height: 1.2; margin-top: 0; font-size: 20px;">Total number of bikes available: {df_aggregated['bikes_available'].sum()}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# Top 10 stations with most bikes
top_10_stations = df_aggregated.nlargest(10, 'bikes_available')

# Worst 10 stations with least bikes (excluding zeros)
worst_10_stations = df_aggregated[df_aggregated['bikes_available'] > 0].sort_values(by='bikes_available').head(10)

# Add a column to combine station name and bikes available for display purposes
top_10_stations['label'] = top_10_stations['station_name'] + ' (' + top_10_stations['bikes_available'].astype(str) + ')'
worst_10_stations['label'] = worst_10_stations['station_name'] + ' (' + worst_10_stations['bikes_available'].astype(str) + ')'

# Define a color palette
color_discrete_sequence = px.colors.qualitative.Plotly

# Create columns for the pie charts
col1, col2 = st.columns(2)

with col1:
    # Donut chart for top 10 stations with most bikes
    fig_top_10 = px.pie(
        top_10_stations, values='bikes_available', names='label',
        title='Top 10 Stations with Most Bikes', hole=0.5,
        color_discrete_sequence=color_discrete_sequence
    )
    fig_top_10.update_traces(textposition='inside', textinfo='percent')

    # Apply custom styling to the Plotly chart
    fig_top_10.update_layout(
        margin=dict(t=50, b=0, l=0, r=0),  # Add top margin for space between title and chart
        font=dict(family="sans serif", size=12, color="#333333"),
        title_x=0.2,  # Center the title
        height=500,  # Increase the height of the chart
        width=500,  # Increase the width of the chart
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",  # Position legend at the bottom
            y=-0.8,  # Adjust this value as needed to move the legend closer to the chart
            xanchor="center",
            x=0.5,
            font=dict(size=10)  # Adjust legend font size
        )
    )

    st.plotly_chart(fig_top_10)

with col2:
    # Check if worst_10_stations is empty
    if worst_10_stations.empty:
        st.write("No data available for the worst 10 stations with non-zero bikes.")
    else:
        # Donut chart for worst 10 stations with least bikes
        fig_worst_10 = px.pie(
            worst_10_stations, values='bikes_available', names='label',
            title='Worst 10 Stations with Least Bikes ', hole=0.5,
            color_discrete_sequence=color_discrete_sequence
        )
        fig_worst_10.update_traces(textposition='inside', textinfo='percent')

        # Apply custom styling to the Plotly chart
        fig_worst_10.update_layout(
            margin=dict(t=50, b=0, l=0, r=0),  # Add top margin for space between title and chart
            font=dict(family="sans serif", size=12, color="#333333"),
            title_x=0.2,  # Center the title
            height=500,  # Increase the height of the chart
            width=500,  # Increase the width of the chart
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Position legend at the bottom
                y=-0.8,  # Adjust this value as needed to move the legend closer to the chart
                xanchor="center",
                x=0.5,
                font=dict(size=10)  # Adjust legend font size
            )
        )

        st.plotly_chart(fig_worst_10)

# Table filter
st.subheader("Table of Stations and Number of Bikes Available")
num_rows = st.slider('Select number of rows to display', min_value=1, max_value=len(df_aggregated), value=20)
sorted_df = df_aggregated.sort_values(by='bikes_available', ascending=False).head(num_rows).reset_index(drop=True)
st.dataframe(sorted_df, width=900, height=600)  # Increase width and height of the table for better readability