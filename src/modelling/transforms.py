import pandas as pd


class EngagementScaler:
    def __init__(self, features):
        self.means = None
        self.stds = None
        self.features = features

    def fit(self, df) -> pd.DataFrame:
        # Calcula estadísticas de Noviembre
        df_ranked = df.copy()
        df_ranked[self.features] = df_ranked[self.features].rank(method='dense')
        self.means = df_ranked.mean()
        self.stds = df_ranked.std()
        return df_ranked

    def transform(self, df_ranked: pd.DataFrame) -> pd.DataFrame:
        """Aplica Z-score usando estadísticas guardadas."""
        df_normalized = df_ranked.copy()
        df_normalized[self.features] = (df_normalized[self.features] - self.means) / self.stds
        df_normalized = df_normalized.fillna(0)
        return df_normalized
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        df_ranked = self.fit(df)
        return self.transform(df_ranked)


# 1. Convertir a ordinal (ranking)
df_ranked = metrics.copy()
df_ranked[cols] = df_ranked[cols].rank(method="dense")

# 2. Normalizar con Z-score
df_norm = df_ranked.copy()
df_norm[cols] = (df_norm[cols] - df_norm[cols].mean()) / df_norm[cols].std()

df_norm = df_norm.fillna(0)
matriz_corr = df_norm[cols].T.corr(method='kendall')

matriz_corr = matriz_corr.fillna(0)

# 5. TRANSFORMACIÓN A AFINIDAD POSITIVA:
# Convertir rango [-1, 1] a [3] para que SpectralClustering funcione
matriz_afinidad = (1 + matriz_corr) / 2
from sklearn.cluster import SpectralClustering

model = SpectralClustering(n_clusters=n_clusters,
                           affinity='precomputed',
                           assign_labels='kmeans',
                           n_init=50)

clusters = model.fit_predict(matriz_afinidad)