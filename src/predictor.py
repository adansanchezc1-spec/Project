"""
Predictor Module
================

Sistema unificado que integra todo el pipeline de predicción de precios.

Author: Housing Price Prediction Team
Date: 2026-04-03
"""

import pandas as pd
import numpy as np
import logging
import joblib
from typing import List, Dict, Optional, Union
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split

from .data_cleaner import create_cleaning_pipeline
from .feature_engineering import create_feature_pipeline
from .models import EnsembleModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingPricePredictor:
    """
    Sistema completo de predicción de precios de vivienda.
    
    Integra:
    --------
    1. Data Loading & Cleaning (DataFrameLoader, DataValidator, DataCleaner)
    2. Feature Engineering (FeatureEngineer, CategoricalEncoder)
    3. Model Training (EnsembleModel con Linear, Tree, Boosting)
    4. Prediction & Evaluation
    
    Parameters
    ----------
    missing_strategy : str, default='median'
        Estrategia para valores faltantes
    outlier_method : str, default='iqr'
        Método de detección de outliers
    outlier_threshold : float, default=3.0
        Umbral para outliers
    model_weights : Dict[str, float], optional
        Pesos para ensemble
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Pipeline completo entrenado
    model_ : EnsembleModel
        Modelo ensemble
    training_metrics_ : Dict
        Métricas de entrenamiento
    
    Examples
    --------
    >>> predictor = HousingPricePredictor()
    >>> predictor.fit([df1, df2])
    >>> predictions = predictor.predict(new_data)
    >>> predictor.save('housing_model.pkl')
    """
    
    def __init__(
        self,
        missing_strategy: str = 'median',
        outlier_method: str = 'iqr',
        outlier_threshold: float = 3.0,
        model_weights: Optional[Dict[str, float]] = None,
        random_state: int = 42
    ):
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.model_weights = model_weights
        self.random_state = random_state
        
        self.pipeline_ = None
        self.model_ = None
        self.training_metrics_ = {}
    
    def _create_pipeline(self) -> Pipeline:
        """
        Crea pipeline completo de preprocessing.
        
        Returns
        -------
        Pipeline
            Pipeline con todas las etapas de transformación
        """
        # 1. Cleaning pipeline
        cleaning_steps = create_cleaning_pipeline(
            missing_strategy=self.missing_strategy,
            outlier_method=self.outlier_method,
            outlier_threshold=self.outlier_threshold
        )
        
        # 2. Feature engineering pipeline
        feature_steps = create_feature_pipeline(
            include_engineering=True,
            include_selection=False,
            include_encoding=True
        )
        
        # 3. Combinar todos los pasos
        all_steps = cleaning_steps + feature_steps
        
        pipeline = Pipeline(all_steps)
        
        return pipeline
    
    def fit(
        self,
        dataframes: Union[List[pd.DataFrame], pd.DataFrame],
        validation_split: float = 0.2,
        cross_validate: bool = True,
        cv_folds: int = 5
    ) -> 'HousingPricePredictor':
        """
        Entrena el sistema completo.
        
        Parameters
        ----------
        dataframes : List[pd.DataFrame] or pd.DataFrame
            DataFrames con datos de viviendas
        validation_split : float, default=0.2
            Fracción de datos para validación
        cross_validate : bool, default=True
            Si realizar validación cruzada
        cv_folds : int, default=5
            Número de folds para CV
        
        Returns
        -------
        self
        """
        logger.info(f"{'='*70}")
        logger.info("HOUSING PRICE PREDICTOR - Entrenamiento Completo")
        logger.info(f"{'='*70}\n")
        
        # 1. Asegurar que dataframes sea lista
        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]
        
        # 2. Crear pipeline de preprocessing
        logger.info("[STEP 1/4] Creando pipeline de preprocessing...")
        self.pipeline_ = self._create_pipeline()
        
        # 3. Aplicar preprocessing
        logger.info("\n[STEP 2/4] Aplicando preprocessing...")
        X_processed = self.pipeline_.fit_transform(dataframes)
        
        # Extraer variable objetivo
        if 'precio' not in X_processed.columns:
            raise ValueError("Columna 'precio' no encontrada después de preprocessing")
        
        y = X_processed['precio']
        X = X_processed.drop(columns=['precio'])
        
        logger.info(f"✓ Datos procesados: {len(X)} muestras, {len(X.columns)} features")
        
        # 4. Split train/validation
        logger.info(f"\n[STEP 3/4] Dividiendo datos ({int((1-validation_split)*100)}% train, {int(validation_split*100)}% val)...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.random_state
        )
        
        logger.info(f"  Train: {len(X_train)} muestras")
        logger.info(f"  Val: {len(X_val)} muestras")
        
        # 5. Entrenar modelo
        logger.info("\n[STEP 4/4] Entrenando modelo ensemble...\n")
        self.model_ = EnsembleModel(weights=self.model_weights)
        self.model_.fit(X_train, y_train)
        
        # 6. Evaluar en validation set
        logger.info(f"\n{'='*70}")
        logger.info("Evaluación en Validation Set:")
        logger.info(f"{'='*70}")
        
        val_metrics = self.model_.evaluate(X_val, y_val)
        self.training_metrics_['validation'] = val_metrics
        
        print(f"\n{self.model_.get_metrics_summary()}")
        
        # 7. Comparar modelos individuales
        logger.info(f"\n{'='*70}")
        logger.info("Comparación de Modelos en Validation:")
        logger.info(f"{'='*70}\n")
        
        comparison = self.model_.get_individual_metrics(X_val, y_val)
        print(comparison.to_string(index=False))
        
        # 8. Cross-validation (opcional)
        if cross_validate:
            logger.info(f"\n{'='*70}")
            logger.info(f"Validación Cruzada ({cv_folds}-fold):")
            logger.info(f"{'='*70}")
            
            cv_scores = cross_val_score(
                self.model_,
                X,
                y,
                cv=cv_folds,
                scoring='r2'
            )
            
            self.training_metrics_['cv_scores'] = cv_scores
            self.training_metrics_['cv_mean'] = cv_scores.mean()
            self.training_metrics_['cv_std'] = cv_scores.std()
            
            logger.info(f"R² scores: {cv_scores}")
            logger.info(f"Mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Entrenamiento completado exitosamente")
        logger.info(f"{'='*70}\n")
        
        return self
    
    def predict(
        self,
        data: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> np.ndarray:
        """
        Predice precios para nuevos datos.
        
        Parameters
        ----------
        data : pd.DataFrame or List[pd.DataFrame]
            Datos para predicción
        
        Returns
        -------
        np.ndarray
            Precios predichos
        """
        if self.model_ is None:
            raise ValueError(
                "Modelo no entrenado. Ejecute fit() primero."
            )
        
        # Asegurar formato lista
        if isinstance(data, pd.DataFrame):
            data = [data]
        
        # Aplicar preprocessing
        X_processed = self.pipeline_.transform(data)
        
        # Remover precio si existe
        if 'precio' in X_processed.columns:
            X_processed = X_processed.drop(columns=['precio'])
        
        # Predecir
        predictions = self.model_.predict(X_processed)
        
        return predictions
    
    def predict_with_details(
        self,
        data: Union[pd.DataFrame, List[pd.DataFrame]]
    ) -> pd.DataFrame:
        """
        Predice con detalles de cada modelo individual.
        
        Parameters
        ----------
        data : pd.DataFrame or List[pd.DataFrame]
            Datos para predicción
        
        Returns
        -------
        pd.DataFrame
            DataFrame con predicciones de cada modelo + ensemble
        """
        if self.model_ is None:
            raise ValueError("Modelo no entrenado. Ejecute fit() primero.")
        
        # Asegurar formato lista
        if isinstance(data, pd.DataFrame):
            data = [data]
        
        # Aplicar preprocessing
        X_processed = self.pipeline_.transform(data)
        
        # Remover precio si existe
        if 'precio' in X_processed.columns:
            X_processed = X_processed.drop(columns=['precio'])
        
        # Obtener predicciones individuales
        individual_preds = self.model_.get_individual_predictions(X_processed)
        
        # Predicción ensemble
        ensemble_pred = self.model_.predict(X_processed)
        
        # Crear DataFrame de resultados
        results = pd.DataFrame({
            'linear_prediction': individual_preds['linear'],
            'tree_prediction': individual_preds['tree'],
            'boosting_prediction': individual_preds['boosting'],
            'ensemble_prediction': ensemble_pred
        })
        
        return results
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Obtiene importancia de features del modelo.
        
        Parameters
        ----------
        top_n : int, default=10
            Número de features a mostrar
        
        Returns
        -------
        pd.DataFrame
            Features más importantes
        """
        if self.model_ is None:
            raise ValueError("Modelo no entrenado.")
        
        return self.model_.get_feature_importance(top_n=top_n)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Guarda el predictor completo.
        
        Parameters
        ----------
        filepath : str or Path
            Ruta donde guardar el modelo
        """
        if self.model_ is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        
        # Guardar todo el objeto
        joblib.dump(self, filepath)
        
        logger.info(f"✓ Modelo guardado en: {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'HousingPricePredictor':
        """
        Carga un predictor guardado.
        
        Parameters
        ----------
        filepath : str or Path
            Ruta del modelo guardado
        
        Returns
        -------
        HousingPricePredictor
            Predictor cargado
        """
        predictor = joblib.load(filepath)
        
        logger.info(f"✓ Modelo cargado desde: {filepath}")
        
        return predictor
    
    def get_pipeline_summary(self) -> str:
        """
        Retorna resumen del pipeline.
        
        Returns
        -------
        str
            Descripción del pipeline
        """
        if self.pipeline_ is None:
            return "Pipeline no creado aún."
        
        summary = "Pipeline de Preprocessing:\n"
        summary += "=" * 50 + "\n"
        
        for i, (name, transformer) in enumerate(self.pipeline_.steps, 1):
            summary += f"{i}. {name}: {type(transformer).__name__}\n"
        
        if self.model_ is not None:
            summary += "\nModelo:\n"
            summary += "=" * 50 + "\n"
            summary += f"Ensemble con pesos: {self.model_.weights_}\n"
        
        return summary
    
    def get_training_summary(self) -> str:
        """
        Retorna resumen del entrenamiento.
        
        Returns
        -------
        str
            Resumen de métricas de entrenamiento
        """
        if not self.training_metrics_:
            return "No hay métricas de entrenamiento disponibles."
        
        summary = "Resumen de Entrenamiento:\n"
        summary += "=" * 50 + "\n"
        
        if 'validation' in self.training_metrics_:
            metrics = self.training_metrics_['validation']
            summary += f"Validation Set:\n"
            summary += f"  R²: {metrics['r2']:.4f}\n"
            summary += f"  MAE: ${metrics['mae']:,.2f}\n"
            summary += f"  RMSE: ${metrics['rmse']:,.2f}\n"
            summary += f"  MAPE: {metrics['mape']:.2f}%\n"
        
        if 'cv_mean' in self.training_metrics_:
            summary += f"\nCross-Validation:\n"
            summary += f"  Mean R²: {self.training_metrics_['cv_mean']:.4f}\n"
            summary += f"  Std R²: {self.training_metrics_['cv_std']:.4f}\n"
        
        return summary


if __name__ == "__main__":
    # Ejemplo de uso completo
    print("=" * 70)
    print("HOUSING PRICE PREDICTOR - Ejemplo Completo")
    print("=" * 70)
    
    # 1. Crear datos sintéticos
    np.random.seed(42)
    n_samples = 200
    
    df1 = pd.DataFrame({
        'ubicacion': np.random.choice(['Centro', 'Norte', 'Sur'], n_samples),
        'tamano': np.random.uniform(50, 200, n_samples),
        'habitaciones': np.random.randint(1, 5, n_samples),
        'precio': np.random.uniform(100000, 500000, n_samples)
    })
    
    df2 = pd.DataFrame({
        'ubicacion': np.random.choice(['Este', 'Oeste'], 100),
        'tamano': np.random.uniform(60, 180, 100),
        'habitaciones': np.random.randint(2, 4, 100),
        'precio': np.random.uniform(120000, 450000, 100)
    })
    
    print(f"\nDataFrames de entrenamiento:")
    print(f"  DF1: {len(df1)} muestras")
    print(f"  DF2: {len(df2)} muestras")
    
    # 2. Crear y entrenar predictor
    print(f"\n{'='*70}")
    print("Iniciando entrenamiento...")
    print(f"{'='*70}\n")
    
    predictor = HousingPricePredictor(
        missing_strategy='median',
        outlier_method='iqr',
        outlier_threshold=3.0
    )
    
    predictor.fit(
        [df1, df2],
        validation_split=0.2,
        cross_validate=True,
        cv_folds=5
    )
    
    # 3. Hacer predicciones
    print(f"\n{'='*70}")
    print("Predicciones en Datos Nuevos:")
    print(f"{'='*70}\n")
    
    new_data = pd.DataFrame({
        'ubicacion': ['Centro', 'Norte', 'Este'],
        'tamano': [120, 85, 150],
        'habitaciones': [3, 2, 4],
        'precio': [0, 0, 0]  # Será ignorado
    })
    
    predictions = predictor.predict(new_data)
    detailed = predictor.predict_with_details(new_data)
    
    print("Datos de entrada:")
    print(new_data[['ubicacion', 'tamano', 'habitaciones']])
    print("\nPredicciones detalladas:")
    print(detailed)
    
    # 4. Feature importance
    print(f"\n{'='*70}")
    print("Top 10 Features Más Importantes:")
    print(f"{'='*70}\n")
    
    importance = predictor.get_feature_importance(top_n=10)
    print(importance)
    
    # 5. Guardar modelo
    print(f"\n{'='*70}")
    predictor.save('/home/claude/housing_predictor.pkl')
    
    # 6. Cargar modelo
    loaded_predictor = HousingPricePredictor.load('/home/claude/housing_predictor.pkl')
    print(f"\n{loaded_predictor.get_training_summary()}")