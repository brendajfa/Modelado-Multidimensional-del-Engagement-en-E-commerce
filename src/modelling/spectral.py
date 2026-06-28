import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering


class SpectralClusteringModel:
    """Pipeline de Clustering Espectral con validaciones y persistencia de centroides."""
    
    def __init__(self, n_clusters: int, n_init: int = 50, random_state: int = 42):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.random_state = random_state
        self.model = None
        self.labels = None
        self.laplacian_matrix = None
        self.centroids = None
        self.train_rank_means = None
        self.train_rank_stds = None
    
    def fit(self, affinity_matrix: np.ndarray, df_normalized: pd.DataFrame = None, 
            features: list = None) -> np.ndarray:
        """Ajusta el modelo y guarda centroides si se proporciona el dataframe normalizado."""
        self.model = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            assign_labels='kmeans',
            n_init=self.n_init,
            random_state=self.random_state,
            verbose=1
        )
        
        self.labels = self.model.fit_predict(affinity_matrix)
        
        # Guarda centroides si se proporciona el dataframe
        if df_normalized is not None and features is not None:
            self._compute_centroids(df_normalized, features)
        
        return self.labels
    
    def _compute_centroids(self, df_normalized: pd.DataFrame, features: list):
        """Calcula y guarda los centroides en el espacio normalizado."""
        self.centroids = np.array([
            df_normalized.loc[df_normalized["cluster"] == c, features].mean().values
            for c in range(self.n_clusters)
        ])
    
    def set_training_statistics(self, df_ranked: pd.DataFrame, features: list):
        """Guarda las estadísticas de ranking del conjunto de entrenamiento."""
        # Excluye la última columna si es calculada (ej: MergeUE)
        features_to_stats = features[:-1] if len(features) > 0 else features
        
        self.train_rank_means = df_ranked[features_to_stats].mean()
        self.train_rank_stds = df_ranked[features_to_stats].std()
    
    def get_laplacian(self, affinity_matrix: np.ndarray) -> np.ndarray:
        """Calcula la matriz Laplaciana para análisis adicional."""
        degrees = np.array(affinity_matrix.sum(axis=1)).flatten()
        D = np.diag(degrees)
        self.laplacian_matrix = D - affinity_matrix
        return self.laplacian_matrix
    
    def get_clusters_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Retorna DataFrame con asignaciones de clusters."""
        result = df.copy()
        result['cluster'] = self.labels
        return result.sort_values('cluster')
    
    def save_model_artifacts(self, filepath: str):
        """Guarda centroides y estadísticas en un archivo numpy."""
        artifacts = {
            'centroids': self.centroids,
            'train_rank_means': self.train_rank_means.values if self.train_rank_means is not None else None,
            'train_rank_stds': self.train_rank_stds.values if self.train_rank_stds is not None else None,
            'n_clusters': self.n_clusters,
        }
        np.save(filepath, artifacts, allow_pickle=True)
        print(f"Artefactos guardados en: {filepath}")
    
    def load_model_artifacts(self, filepath: str):
        """Carga centroides y estadísticas desde archivo."""
        artifacts = np.load(filepath, allow_pickle=True).item()
        self.centroids = artifacts['centroids']
        self.train_rank_means = artifacts['train_rank_means']
        self.train_rank_stds = artifacts['train_rank_stds']
        self.n_clusters = artifacts['n_clusters']
        print(f"Artefactos cargados desde: {filepath}")
    
    def fit_predict(self, affinity_matrix: np.ndarray) -> np.ndarray:
        """Ajusta y predice en un solo paso."""
        return self.fit(affinity_matrix)