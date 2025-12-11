import sqlite3
from datetime import datetime
import json

DB_NAME = "history.db"

def init_db():
    """Creates the database table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            insights TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_entry(filename, insights):
    """Saves a new AI analysis to the DB."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO analysis_history (timestamp, filename, insights) VALUES (?, ?, ?)',
              (timestamp, filename, insights))
    conn.commit()
    conn.close()

def get_all_entries():
    """Fetches all history, newest first."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Fetch ID too so we can delete specific rows
    c.execute('SELECT id, timestamp, filename, insights FROM analysis_history ORDER BY id DESC')
    data = c.fetchall()
    conn.close()
    return data

def delete_entry(entry_id):
    """Deletes a specific entry by ID."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('DELETE FROM analysis_history WHERE id = ?', (entry_id,))
    conn.commit()
    conn.close()