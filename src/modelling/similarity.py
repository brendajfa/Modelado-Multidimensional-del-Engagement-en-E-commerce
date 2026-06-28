import pandas as pd
import numpy as np


class KendallTauSimilarity:
    """Calcula matriz de similitud basada en correlaciones ordinales (Kendall tau)."""
    
    def __init__(self, method: str = 'kendall'):
        self.method = method
        self.similarity_matrix = None
        self.features = None
    
    def fit(self, df_normalized: pd.DataFrame, features: list) -> pd.DataFrame:
        """Calcula la matriz de correlación Kendall tau."""
        self.features = features
        
        # Transpone para calcular correlación entre categorías (filas)
        df_transposed = df_normalized[features].T
        
        # Calcula matriz de correlación con Kendall tau
        self.similarity_matrix = df_transposed.corr(method=self.method)
        self.similarity_matrix = self.similarity_matrix.fillna(0)
        
        return self.similarity_matrix
    
    def transform(self, similarity_matrix) -> pd.DataFrame:
        """Aplica transformación a afinidad positiva [0, 1]."""
        if similarity_matrix is None:
            raise ValueError("Debe ejecutar fit() primero")
        
        affinity_matrix = (1 + similarity_matrix) / 2
        
        return affinity_matrix
    
    def fit_transform(self, df_normalized: pd.DataFrame, features: list) -> pd.DataFrame:
        """Ajusta, calcula similitud y transforma a afinidad en un solo paso."""
        similarity_matrix = self.fit(df_normalized, features)
        return self.transform(similarity_matrix)