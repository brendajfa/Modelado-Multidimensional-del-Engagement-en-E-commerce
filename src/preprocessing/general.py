from src.preprocessing.base import PreprocessorBase
import config
import pandas as pd
import os


class GeneralPreprocessor(PreprocessorBase):
    """Preprocesador para análisis general por categoría."""
    
    def _define_reference(self):
        """En el modelo general no hay grupos de referencia."""
        self.referencia = pd.DataFrame({"category_code": config.CATEGORY_CODES})
    
    def _get_identifier_column(self) -> str:
        return "category_code"
    
    def _get_input_path(self, category_code: str) -> str:
        return os.path.join(self.workpath, "categories", f"{category_code}.parquet")
    
    def _get_output_path(self) -> str:
        return os.path.join(self.workpath, "processed", "general_metrics.parquet")
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula métricas generales por categoría."""
        metrics = {}
        
        # Ejemplos de métricas generales
        metrics.update(self._calc_popularity(df))
        metrics.update(self._calc_activity(df))
        metrics.update(self._calc_loyalty(df))
        
        return metrics
    
    def _calc_popularity(self, df: pd.DataFrame) -> dict:
        """Calcula métricas de popularidad."""
        total_views = len(df[df["event_type"] == "view"])
        unique_users = df["user_id"].nunique()
        total_clicks = len(df[df["event_type"] == "click"])
        return {
            "views": total_views,
            "users": unique_users,
            "clicks": total_clicks,
        }
    
    def _calc_activity(self, df: pd.DataFrame) -> dict:
        """Calcula métricas de actividad."""
        return {
            "clicks_depth": self._calc_click_depth(df),
            "dwell_time_avg": self._calc_dwelltime_avg(df),
        }

    def _calc_click_depth(self, df: pd.DataFrame) -> float:
        df_views = df[df["event_type"] == "view"]
        views_per_session = df_views.groupby("user_session").size()
        click_depth = views_per_session.mean()
        return click_depth

    def _calc_dwelltime_avg(self, df: pd.DataFrame) -> float:
        df["event_time"] = pd.to_datetime(df["event_time"])
        session_durations = ( df.groupby("user_session")["event_time"] .agg(lambda x: x.max() - x.min()) )
        dwell_time_a = session_durations.mean().total_seconds()
        return dwell_time_a

    
    def _calc_loyalty(self, df: pd.DataFrame) -> dict:
        """Calcula métricas de lealtad."""
        return {
            "active_days": self._calc_active_days(df),
            "return_rate": self._calc_return_rate(df),
            "dwelltime_l": self._calc_dwelltime_l(df),
        }

    def _calc_active_days(self, df: pd.DataFrame) -> float:
        return (df.groupby("user_id")["event_day"].nunique().mean())

    def _calc_return_rate(self, df: pd.DataFrame) -> float:
        return (df.groupby("user_id")["user_session"].nunique().mean())

    def _calc_dwelltime_l(self, df: pd.DataFrame) -> float:
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
