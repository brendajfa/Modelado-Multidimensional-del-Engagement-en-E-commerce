import pandas as pd
from src.preprocessing.base import PreprocessorBase
import os
import src.config as config


class UserBasedPreprocessor(PreprocessorBase):
    """Preprocesador basado en segmentación de usuarios (Tourist a VIP)."""
    
    def _define_reference(self):
        """Define los grupos de usuarios de referencia."""
        self.referencia = pd.DataFrame({
            "user_group": ["Tourist", "Interested", "Average", "Active", "VIP"]
        })
    
    def _get_identifier_column(self) -> str:
        return "category_code"
    
    def _get_input_path(self, category_code: str, train_eval: str) -> str:
        return os.path.join(self.workpath, "categories", train_eval, f"{category_code}.parquet")
    
    def _get_output_path(self) -> str:
        return os.path.join(self.workpath, "processed", "user_based_metrics.parquet")
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara el dataframe agregando clasificación de usuarios."""
        df["event_time"] = pd.to_datetime(df["event_time"])
        
        # Clasificar usuarios por días visitados
        user_days = df.groupby("user_id")["event_day"].nunique().reset_index(name="days_visited")
        user_days["user_group"] = user_days["days_visited"].apply(self._classify_user)
        
        # Merge con dataframe principal
        df = df.merge(user_days[["user_id", "user_group"]], on="user_id")
        return df
    
    @staticmethod
    def _classify_user(days: int) -> str:
        """Clasifica usuarios según días visitados."""
        if days == 1:
            return "Tourist"
        elif 2 <= days <= 4:
            return "Interested"
        elif 5 <= days <= 8:
            return "Average"
        elif 9 <= days <= 15:
            return "Active"
        else:
            return "VIP"
    
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula métricas por grupo de usuario."""
        df = self._prepare_dataframe(df)
        metrics = {}

        # Ejemplos de métricas user-based        
        metrics.update(self._calc_users_visits(df))
        metrics.update(self._calc_clicks(df))

        metrics.update(self._calc_click_depth(df))
        metrics.update(self._calc_dwelltime_avg(df))

        metrics.update(self._calc_active_days(df))
        metrics.update(self._calc_return_rate(df))
        metrics.update(self._calc_dwelltime_l(df))
        
        return metrics    
    
    def _calc_users_visits(self, df: pd.DataFrame) -> dict:
        """Calcula usuarios y visitas por grupo."""
        users_visits = df.groupby("user_group").agg(
            users=("user_id", "nunique"),
            visits=("user_session", "nunique")
        ).reset_index()
        users_visits = self.referencia.merge(users_visits, on="user_group", how="left").fillna(0)
        
        result = {}
        for _, row in users_visits.iterrows():
            result[f"{row.user_group}_users"] = row.users
            result[f"{row.user_group}_visits"] = row.visits
        return result
    
    def _calc_clicks(self, df: pd.DataFrame) -> dict:
        """Calcula clics por grupo."""
        clicks = df[df["event_type"] == "view"].groupby("user_group").agg(
            clicks=("event_type", "count")
        ).reset_index()
        clicks = self.referencia.merge(clicks, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_clicks": row.clicks for _, row in clicks.iterrows()}
    
    def _calc_click_depth(self, df: pd.DataFrame) -> dict:
        """Calcula profundidad de clics por grupo."""
        df_views = df[df["event_type"] == "view"]
        click_depth = df_views.groupby(["user_session", "user_group"]).size().groupby("user_group").mean().reset_index(name="click_depth")
        click_depth = self.referencia.merge(click_depth, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_click_depth": row.click_depth for _, row in click_depth.iterrows()}
    
    def _calc_dwelltime_avg(self, df: pd.DataFrame) -> dict:
        """Calcula tiempo promedio por sesión."""
        session_durations = df.groupby(["user_session", "user_group"])["event_time"].agg(lambda x: x.max() - x.min())
        dwell_time = session_durations.groupby("user_group").mean().dt.total_seconds().reset_index(name="dwelltime_avg")
        dwell_time = self.referencia.merge(dwell_time, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_dwelltime_avg": row.dwelltime_avg for _, row in dwell_time.iterrows()}
    
    def _calc_active_days(self, df: pd.DataFrame) -> dict:
        """Calcula días activos promedio."""
        active_days = df.groupby(["user_id", "user_group"])["event_day"].nunique().groupby("user_group").mean().reset_index(name="active_days")
        active_days = self.referencia.merge(active_days, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_active_days": row.active_days for _, row in active_days.iterrows()}
    
    def _calc_return_rate(self, df: pd.DataFrame) -> dict:
        """Calcula tasa de retorno."""
        return_rate = df.groupby(["user_id", "user_group"])["user_session"].nunique().groupby("user_group").mean().reset_index(name="return_rate")
        return_rate = self.referencia.merge(return_rate, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_return_rate": row.return_rate for _, row in return_rate.iterrows()}
    
    def _calc_dwelltime_l(self, df: pd.DataFrame) -> dict:
        """Calcula tiempo total acumulado."""
        session_duration = df.groupby(["user_session", "user_group"])["event_time"].agg(lambda x: x.max() - x.min()).reset_index(name="dwelltime_l")
        user_sessions = df[["user_session", "user_id"]].drop_duplicates()
        session_duration = session_duration.merge(user_sessions, on="user_session")
        dwelltime_l = session_duration.groupby(["user_id", "user_group"])["dwelltime_l"].sum().groupby("user_group").mean().dt.total_seconds().reset_index(name="dwelltime_l")
        dwelltime_l = self.referencia.merge(dwelltime_l, on="user_group", how="left").fillna(0)
        
        return {f"{row.user_group}_dwelltime_l": row.dwelltime_l for _, row in dwelltime_l.iterrows()}