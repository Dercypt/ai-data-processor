import pandas as pd

def analyze_dataset(file):
    try:
        df = pd.read_csv(file)
        # Fill missing values
        df.fillna(0, inplace=True)
        
        summary = {
            "columns": list(df.columns),
            "rows": len(df),
            "stats": df.describe().to_dict()
        }
        return df, summary, None
    except Exception as e:
        return None, None, str(e)