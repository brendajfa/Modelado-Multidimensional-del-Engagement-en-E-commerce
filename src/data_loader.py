import pandas as pd

def general_cleaning(df):
    df['event_time'] = pd.to_datetime(df['event_time'])
    df["conversion"] = (df["event_type"] == "purchase").astype(int)  # Nueva columna booleana para
    df['event_day'] = df['event_time'].dt.date
    df = df[df["price"]> 0]
    df = df.drop_duplicates()
    return df
