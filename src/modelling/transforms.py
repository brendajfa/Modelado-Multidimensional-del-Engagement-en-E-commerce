import pandas as pd


class EngagementScaler:
    def __init__(self, features: list):
        self.means = None
        self.stds = None
        self.features = features

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if "MergeUE" in self.features:
            df["MergeUE"] = df["users"] + df["visits"] + df["clicks"]
        return df

    def rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ranked = df.copy()
        df_ranked[self.features] = df_ranked[self.features].rank(method="dense")
        df_ranked = self._preprocess(df_ranked.copy())
        return df_ranked

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df_pre = self._preprocess(df.copy())
        df_ranked = self.rank(df_pre)
        self.means = df_ranked[self.features].mean()
        self.stds = df_ranked[self.features].std()
        return df_ranked

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.means is None or self.stds is None:
            raise ValueError("Debes llamar a fit() antes de transform()")

        df_pre = self._preprocess(df.copy())
        df_ranked = self.rank(df_pre)
        df_normalized = df_ranked.copy()
        df_normalized[self.features] = (df_ranked[self.features] - self.means) / self.stds
        return df_normalized.fillna(0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_ranked = self.fit(df)
        return self.transform(df_ranked)