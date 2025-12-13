import streamlit as st
import pandas as pd
from streamlit_gsheets import GSheetsConnection
from datetime import datetime

# Establish connection
def get_conn():
    return st.connection("gsheets", type=GSheetsConnection)

def init_db():
    """
    Checks if the sheet is accessible. 
    Uses ttl=0 to ensure we don't accidentally wipe data due to stale cache.
    """
    conn = get_conn()
    try:
        # ttl=0 forces a fresh check of the real Google Sheet every time
        df = conn.read(ttl=0)
        
        # If columns are missing or file is truly empty, reset it
        if df.empty or len(df.columns) < 3:
            init_df = pd.DataFrame(columns=["timestamp", "filename", "insights"])
            conn.update(data=init_df)
    except Exception:
        # Fallback for totally new sheets
        init_df = pd.DataFrame(columns=["timestamp", "filename", "insights"])
        conn.update(data=init_df)

def save_entry(filename, insights):
    """Appends a new entry to the Google Sheet."""
    conn = get_conn()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Get current data (Fresh read)
    try:
        existing_data = conn.read(ttl=0)
    except:
        existing_data = pd.DataFrame(columns=["timestamp", "filename", "insights"])

    # 2. Create new row
    new_entry = pd.DataFrame([{
        "timestamp": timestamp,
        "filename": filename,
        "insights": insights
    }])
    
    # 3. Combine and Update
    updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    conn.update(data=updated_data)

def get_all_entries():
    """Fetches all history, newest first."""
    conn = get_conn()
    try:
        # ttl=0 ensures the sidebar updates immediately after you save
        df = conn.read(ttl=0)
        if df.empty:
            return []
        
        # Sort by timestamp descending (newest on top)
        # Convert timestamp to string just in case pandas made it a date object
        df['timestamp'] = df['timestamp'].astype(str)
        df = df.sort_values(by="timestamp", ascending=False)
        
        return list(df.itertuples(index=True, name=None)) 
    except Exception:
        return []

def delete_entry(entry_id):
    """Deletes a specific row by its Index ID."""
    conn = get_conn()
    df = conn.read(ttl=0)
    
    try:
        df = df.drop(entry_id)
        conn.update(data=df)
    except KeyError:
        st.error("Could not find that entry to delete.")