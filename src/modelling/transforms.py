import pandas as pd


class EngagementScaler:
    def __init__(self, features: list):
        self.means = None
        self.stds = None
        self.features = features
    
    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MergeUE" in df.columns:
            df["MergeUE"] = df["users"] + df["visits"] + df["clicks"]
        return df

    def fit(self, df) -> pd.DataFrame:
        # Calcula estadísticas de Noviembre
        df_ranked = df.copy()
        df_ranked[self.features] = df_ranked[self.features].rank(method='dense')
        df_ranked = self._preprocess(df_ranked)
        self.means = df_ranked.mean()
        self.stds = df_ranked.std()
        return df_ranked

    def transform(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """Aplica Z-score usando estadísticas guardadas."""
        df_normalized = df_ranked.copy()
        df_normalized = self._preprocess(df_normalized)
        df_normalized[self.features] = (df_normalized[self.features] - self.means) / self.stds
        df_normalized = df_normalized.fillna(0)
        return df_normalized
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        df_ranked = self.fit(df)
        return self.transform(df_ranked)
