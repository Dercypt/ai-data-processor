import streamlit.components.v1 as components
import streamlit as st
import time
from analyzer import analyze_dataset
from llm_service import get_ai_insights
from database import init_db, save_entry, get_all_entries, delete_entry

# 1. Initialize DB on app startup
init_db()

st.set_page_config(page_title="AI Data Processor", page_icon="📊", layout="wide")
st.title("AI Data Processor")

# --- SESSION STATE INITIALIZATION ---
# This ensures variables survive the 'rerun' command
if "generated_insight" not in st.session_state:
    st.session_state["generated_insight"] = None

# --- SIDEBAR: HISTORY & MANAGEMENT ---
st.sidebar.title("Analysis History")

# Fetch updated history
history = get_all_entries()

if not history:
    st.sidebar.text("No past analyses.")
else:
    for entry_id, timestamp, title, insights in history:
        # We display the TITLE here (which might be the filename or your custom name)
        # Using timestamp[11:16] shows just the time (HH:MM) for cleaner look
        label = f"{title} ({timestamp[11:16]})"
        
        with st.sidebar.expander(label):
            st.caption(f"Date: {timestamp[:10]}")
            st.info(insights)
            
            # The Delete Button
            if st.button("🗑️ Delete", key=f"del_{entry_id}"):
                delete_entry(entry_id)
                st.rerun()

# --- MAIN APP: UPLOAD & ANALYZE ---

file = st.file_uploader("Upload CSV", type="csv")

if file:
    # Analyze the file
    df, summary, err = analyze_dataset(file)
    
    if err:
        st.error(err)
    else:
        st.write("### Data Preview", df.head())
        
        # UI Layout: Stats on left, Controls on right
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Statistics")
            st.json(summary["stats"])
        
        with col2:
            st.write("### AI Analysis Controls")
            
            # FEATURE: Custom Title Input
            # Default value is the filename, but user can change it
            custom_title = st.text_input("Name this analysis", value=file.name)
            
            if st.button("Generate Insights"):
                with st.spinner("Consulting the AI Analyst..."):
                    # 1. Generate the insight
                    insights = get_ai_insights(summary)
                    
                    # 2. Save it to Session State (So it survives the refresh)
                    st.session_state["generated_insight"] = insights
                    
                    # 3. Save to DB (Using the CUSTOM TITLE)
                    save_entry(custom_title, insights)
                    st.success("Saved to History!")
                    
                    # 4. Refresh to update sidebar immediately
                    time.sleep(0.5)
                    st.rerun()

    # --- DISPLAY RESULTS (Outside the button) ---
    # This block runs every time the script reruns. 
    # If we have an insight in memory, we show it.
    if st.session_state["generated_insight"]:
        st.divider()
        st.write("### 🤖 Generated Insights")
        st.write(st.session_state["generated_insight"])