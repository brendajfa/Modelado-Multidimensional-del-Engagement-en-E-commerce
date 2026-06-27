import config
import os
from tqdm import tqdm
import pandas as pd


def calc_click_depth(df):
  df_views = df[df["event_type"] == "view"]
  metric = df.groupby(["is_weekend", "user_session"]).size().groupby("is_weekend").mean()
  metric_weekday = metric.get(False, 0)
  metric_weekend = metric.get(True, 0)
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)

def calc_dwelltime_avg(df):
  df["event_time"] = pd.to_datetime(df["event_time"])
  session_durations = ( df.groupby(["is_weekend", "user_session"])["event_time"] .agg(lambda x: x.max() - x.min()) )
  metric = session_durations.groupby("is_weekend").mean()
  metric_weekday = metric.get(False, pd.Timedelta('0 days 00:00:00')).total_seconds()
  metric_weekend = metric.get(True, pd.Timedelta('0 days 00:00:00')).total_seconds()
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)

# def calc_active_days(df):
#   return (df.groupby("user_id")["event_day"].nunique().mean())

def calc_return_rate(df):
  metric = df.groupby(["is_weekend", "user_id"])["user_session"].nunique().groupby("is_weekend").mean()
  metric_weekday = metric.get(False, 0)
  metric_weekend = metric.get(True, 0)
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)

def calc_dwelltime_l(df):
  session_duration = (
      df.groupby("user_session")["event_time"]
        .agg(lambda x: x.max() - x.min())
        .rename("session_duration")
  )
  user_sessions = df[["user_session", "user_id"]].drop_duplicates()
  session_duration = session_duration.to_frame().merge(user_sessions, on="user_session")
  user_duration = (
      session_duration.groupby("user_id")["session_duration"].sum().mean().total_seconds()
  )
  return user_duration

def calc_dwelltime_l_time(df):
  df_weekday = df[~df["is_weekend"]]
  df_weekend = df[df["is_weekend"]]
  try:
    metric_weekday = calc_dwelltime_l(df_weekday)
  except:
    metric_weekday = 0
  metric_weekend = calc_dwelltime_l(df_weekend)
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)


def unique_users_time(df):
  metric_weekend = df[df["is_weekend"]]["user_id"].nunique()
  metric_weekday = df[~df["is_weekend"]]["user_id"].nunique()
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)

def visits_time(df):
  metric_weekend = df[df["is_weekend"]]["user_session"].nunique()
  metric_weekday = df[~df["is_weekend"]]["user_session"].nunique()
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)

def clicks_time(df):
  metric_weekend = df[(df["is_weekend"]) & (df["event_type"] == "view")].shape[0]
  metric_weekday = df[(~df["is_weekend"]) & (df["event_type"] == "view")].shape[0]
  return 0 if metric_weekday == 0 else metric_weekday / (metric_weekday + metric_weekend)


category_code_column = []
users = []
visits = []
clicks = []
clicks_depth = []
dwell_time_avg = []
active_days = []
return_rate = []
dwelltime_l = []

from tqdm import tqdm
for category_code in tqdm(config.CATEGORY_CODES):
  input_path = os.path.join(config.WORKPATH, "categories", f"{category_code}.parquet")
  df = pd.read_parquet(input_path)
  print(category_code)

  df["weekday"] = pd.to_datetime(df["event_day"]).dt.weekday
  df["is_weekend"] = df["weekday"] >= 5

  category_code_column.append(category_code)
  users.append(unique_users_time(df))
  visits.append(visits_time(df))
  clicks.append(clicks_time(df))
  clicks_depth.append(calc_click_depth(df))
  dwell_time_avg.append(calc_dwelltime_avg(df))
  return_rate.append(calc_return_rate(df))
  dwelltime_l.append(calc_dwelltime_l_time(df))

data = {
    "category_code": category_code_column,
    "users": users,
    "visits": visits,
    "clicks": clicks,

    "click_depth": clicks_depth,
    "dwell_time_avg": dwell_time_avg,

    "return_rate": return_rate,
    "dwelltime_l": dwelltime_l,
}


metrics = pd.DataFrame(data)


import pandas as pd
from src.preprocessing.base import PreprocessorBase
import os
import config


class TimeBasedPreprocessor(PreprocessorBase):
    """Preprocesador basado en análisis temporal (Weekday vs Weekend)."""
    
    def _define_reference(self):
        """Define los períodos de tiempo de referencia."""
        self.referencia = pd.DataFrame({
            "time_period": ["Weekday", "Weekend"]
        })
    
    def _get_identifier_column(self) -> str:
        return "category_code"
    
    def _get_input_path(self, category_code: str) -> str:
        return os.path.join(self.workpath, "categories", f"{category_code}.parquet")
    
    def _get_output_path(self) -> str:
        return os.path.join(self.workpath, "processed", "time_based_metrics.parquet")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara el dataframe agregando clasificación temporal."""
        df["event_time"] = pd.to_datetime(df["event_time"])
        df["event_day"] = pd.to_datetime(df["event_day"])
        df["day_of_week"] = df["event_day"].dt.dayofweek
        df["time_period"] = df["day_of_week"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")
        return df
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula métricas por período temporal."""
        df = self._prepare_dataframe(df)
        metrics = {}
        
        metrics.update(self._calc_events_by_period(df))
        metrics.update(self._calc_users_by_period(df))
        metrics.update(self._calc_sessions_by_period(df))
        metrics.update(self._calc_dwelltime_by_period(df))
        metrics.update(self._calc_conversion_by_period(df))
        
        return metrics
    
    def _calc_events_by_period(self, df: pd.DataFrame) -> dict:
        """Calcula eventos por período."""
        events = df.groupby("time_period").size().reset_index(name="events")
        events = self.referencia.merge(events, on="time_period", how="left").fillna(0)
        
        return {f"{row.time_period}_events": row.events for _, row in events.iterrows()}
    
    def _calc_users_by_period(self, df: pd.DataFrame) -> dict:
        """Calcula usuarios únicos por período."""
        users = df.groupby("time_period")["user_id"].nunique().reset_index(name="unique_users")
        users = self.referencia.merge(users, on="time_period", how="left").fillna(0)
        
        return {f"{row.time_period}_unique_users": row.unique_users for _, row in users.iterrows()}
    
    def _calc_sessions_by_period(self, df: pd.DataFrame) -> dict:
        """Calcula sesiones por período."""
        sessions = df.groupby("time_period")["user_session"].nunique().reset_index(name="sessions")
        sessions = self.referencia.merge(sessions, on="time_period", how="left").fillna(0)
        
        return {f"{row.time_period}_sessions": row.sessions for _, row in sessions.iterrows()}
    
    def _calc_dwelltime_by_period(self, df: pd.DataFrame) -> dict:
        """Calcula tiempo de permanencia por período."""
        session_duration = df.groupby(["user_session", "time_period"])["event_time"].agg(lambda x: x.max() - x.min()).reset_index(name="dwelltime")
        dwelltime = session_duration.groupby("time_period")["dwelltime"].mean().dt.total_seconds().reset_index(name="dwelltime_avg")
        dwelltime = self.referencia.merge(dwelltime, on="time_period", how="left").fillna(0)
        
        return {f"{row.time_period}_dwelltime_avg": row.dwelltime_avg for _, row in dwelltime.iterrows()}
    
    def _calc_conversion_by_period(self, df: pd.DataFrame) -> dict:
        """Calcula tasa de conversión por período."""
        purchases = len(df[df["event_type"] == "purchase"])
        views = len(df[df["event_type"] == "view"])
        conversion_rate = (purchases / views) if views > 0 else 0
        
        return {
            "conversion_rate": conversion_rate,
            "purchases": purchases
        }