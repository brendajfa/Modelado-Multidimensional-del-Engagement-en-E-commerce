
from abc import ABC, abstractmethod
import pandas as pd
from tqdm import tqdm
import os
import config


class EngagementScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, df):
        # Calcula estadísticas de Noviembre
        self.means = df.mean()
        self.stds = df.std()

    def transform(self, df):
        # Aplica Rank-dense y Z-score usando las estadísticas guardadas
        df_ranked = df.rank(method='dense')
        return (df_ranked - self.means) / self.stds


class PreprocessorBase(ABC):
    """Clase base abstracta para los diferentes tipos de preprocesamiento."""
    
    def __init__(self, workpath: str = config.WORKPATH):
        self.workpath = workpath
        self.referencia = None
        self.metrics_df = None
    
    @abstractmethod
    def _define_reference(self):
        """Define el DataFrame de referencia para merge operations."""
        pass
    
    @abstractmethod
    def _calculate_metrics(self, df: pd.DataFrame) -> dict:
        """Calcula las métricas específicas del preprocesador."""
        pass
    
    @abstractmethod
    def _get_input_path(self, identifier: str) -> str:
        """Retorna la ruta del archivo de entrada según el identificador."""
        pass
    
    @abstractmethod
    def _get_output_path(self) -> str:
        """Retorna la ruta del archivo de salida."""
        pass
    
    def process(self):
        """Pipeline principal de procesamiento."""
        self._define_reference()
        rows = []
        
        for identifier in tqdm(config.CATEGORY_CODES):
            input_path = self._get_input_path(identifier)
            df = pd.read_parquet(input_path)
            print(f"Procesando: {identifier}")
            
            # Diccionario base con el identificador
            row_dict = {self._get_identifier_column(): identifier}
            
            # Calcula métricas específicas
            metrics = self._calculate_metrics(df)
            row_dict.update(metrics)
            rows.append(row_dict)
        
        self.metrics_df = pd.DataFrame(rows)
        self._save_results()
        return self.metrics_df
    
    def _save_results(self):
        """Guarda los resultados en la ruta especificada."""
        output_path = self._get_output_path()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.metrics_df.to_parquet(output_path)
        print(f"Resultados guardados en: {output_path}")
    
    @abstractmethod
    def _get_identifier_column(self) -> str:
        """Retorna el nombre de la columna identificadora."""
        pass