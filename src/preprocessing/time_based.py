from src.preprocessing.base import PreprocessorBase
import os
import pandas as pd


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

    def _ratio_weekday_vs_weekend(self, weekday_value, weekend_value):
        total = weekday_value + weekend_value
        return 0.0 if total == 0 else weekday_value / total
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula métricas por período temporal."""
        df = self._prepare_dataframe(df)
        metrics = {}
        
        metrics.update({"users": self._calc_popularity(df, "user_id", "nunique")})
        metrics.update({"visits": self._calc_popularity(df, "user_session", "nunique")})
        metrics.update({"clicks": self._calc_popularity(df, event_type="view")})
        metrics.update({"click_depth": self._calc_click_depth(df)})
        metrics.update({"dwell_time_avg": self._calc_dwelltime_avg(df)})
        metrics.update({"return_rate": self._calc_return_rate(df)})
        metrics.update({"dwelltime_l": self._calc_dwelltime_l_time(df)})

        return metrics

    def _calc_popularity(self, df, column=None, reducer=None, event_type=None):
        if event_type is not None:
            df = df[df["event_type"] == event_type]

        if column is None:
            counts = df.groupby("is_weekend").size()
        else:
            counts = df.groupby("is_weekend")[column].agg(reducer)

        weekday = counts.get(False, 0)
        weekend = counts.get(True, 0)

        return self._ratio_weekday_vs_weekend(weekday, weekend)

    def _calc_click_depth(self, df):
        if df.empty:
            return 0.0

        clicks_per_session = df.groupby(["is_weekend", "user_session"]).size()
        metric = clicks_per_session.groupby("is_weekend").mean()

        weekday = float(metric.get(False, 0.0))
        weekend = float(metric.get(True, 0.0))
        return self._ratio_weekday_vs_weekend(weekday, weekend)

    def _calc_dwelltime_avg(self, df):
        if df.empty:
            return 0.0

        df = df.copy()
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

        session_durations = (
            df.groupby(["is_weekend", "user_session"])["event_time"]
              .agg(lambda x: x.max() - x.min())
        )
        metric = session_durations.groupby("is_weekend").mean()

        weekday = metric.get(False, pd.Timedelta(0)).total_seconds()
        weekend = metric.get(True, pd.Timedelta(0)).total_seconds()
        return self._ratio_weekday_vs_weekend(weekday, weekend)

    def _calc_return_rate(self, df):
        if df.empty:
            return 0.0

        metric = (
            df.groupby(["is_weekend", "user_id"])["user_session"]
              .nunique()
              .groupby("is_weekend")
              .mean()
        )

        weekday = float(metric.get(False, 0.0))
        weekend = float(metric.get(True, 0.0))
        return self._ratio_weekday_vs_weekend(weekday, weekend)

    def _calc_dwelltime_l(self, df):
        if df.empty:
            return 0.0

        df = df.copy()
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")

        session_duration = (
            df.groupby(["user_session", "user_id"])["event_time"]
              .agg(lambda x: x.max() - x.min())
              .groupby("user_id")
              .sum()
        )

        return float(session_duration.mean().total_seconds()) if not session_duration.empty else 0.0

    def _calc_dwelltime_l_time(self, df):
        if df.empty:
            return 0.0

        weekday = self._calc_dwelltime_l(df.loc[~df["is_weekend"]])
        weekend = self._calc_dwelltime_l(df.loc[df["is_weekend"]])
        return self._ratio_weekday_vs_weekend(weekday, weekend)
