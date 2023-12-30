import streamlit as st
import pandas as pd
import time
from datetime import datetime

def load_data():
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H-%M-%S")
    df = pd.read_csv("Attendance/Attendance_"+date+".csv")
    return df

# Initial load of data
data = load_data()

# Display data initially
data_placeholder = st.empty()
data_placeholder.dataframe(data.style.highlight_max(axis=0))

# Create a button for manual refresh
if st.button('Refresh Data'):
    data = load_data()
    # Clear previous content and display updated data
    data_placeholder.empty()
    data_placeholder.dataframe(data.style.highlight_max(axis=0))

# Auto-refresh every 3 seconds
while True:
    time.sleep(3)
    data = load_data()
    # Clear previous content and display updated data
    data_placeholder.empty()
    data_placeholder.dataframe(data.style.highlight_max(axis=0))
