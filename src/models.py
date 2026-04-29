"""
Models Module
=============

Módulo con modelos de regresión y sistema ensemble para predicción de precios.

Author: Housing Price Prediction Team
Date: 2026-04-03
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousingPriceModel(BaseEstimator, RegressorMixin):
    """
    Clase base para modelos de predicción de precios.
    
    Proporciona interfaz común y utilidades para todos los modelos.
    
    Attributes
    ----------
    model_ : sklearn estimator
        Modelo subyacente ajustado
    feature_names_ : List[str]
        Nombres de features usadas en entrenamiento
    metrics_ : Dict
        Métricas de evaluación del modelo
    """
    
    def __init__(self):
        self.model_ = None
        self.feature_names_ = None
        self.metrics_ = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Debe ser implementado por subclases"""
        raise NotImplementedError
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Debe ser implementado por subclases"""
        raise NotImplementedError
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, float]:
        """
        Evalúa el modelo con múltiples métricas.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Variable objetivo verdadera
        
        Returns
        -------
        Dict[str, float]
            Diccionario con métricas: R², MAE, RMSE, MAPE
        """
        y_pred = self.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        self.metrics_ = metrics
        return metrics
    
    def get_metrics_summary(self) -> str:
        """
        Retorna resumen legible de métricas.
        
        Returns
        -------
        str
            String formateado con métricas
        """
        if not self.metrics_:
            return "No hay métricas disponibles. Ejecute evaluate() primero."
        
        return (
            f"R² Score: {self.metrics_['r2']:.4f}\n"
            f"MAE: ${self.metrics_['mae']:,.2f}\n"
            f"RMSE: ${self.metrics_['rmse']:,.2f}\n"
            f"MAPE: {self.metrics_['mape']:.2f}%"
        )


class LinearRegressionModel(HousingPriceModel):
    """
    Modelo de Regresión Lineal para precios de vivienda.
    
    Ventajas:
    - Máxima interpretabilidad (coeficientes directos)
    - Rápido entrenamiento e inferencia
    - Baseline robusto
    
    Desventajas:
    - No captura no-linealidades
    - Asume independencia de features
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Si calcular intercepto
    normalize : bool, default=False
        Si normalizar features antes de regresión
    
    Attributes
    ----------
    coefficients_ : pd.Series
        Coeficientes del modelo con nombres de features
    intercept_ : float
        Término independiente
    
    Examples
    --------
    >>> model = LinearRegressionModel()
    >>> model.fit(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> metrics = model.evaluate(X_test, y_test)
    >>> print(model.get_feature_importance())
    """
    
    def __init__(self, fit_intercept: bool = True, normalize: bool = False):
        super().__init__()
        self.fit_intercept = fit_intercept
        self.normalize = normalize
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena modelo de regresión lineal.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento
        y : pd.Series
            Precios objetivo
        
        Returns
        -------
        self
        """
        self.feature_names_ = list(X.columns)
        
        self.model_ = LinearRegression(
            fit_intercept=self.fit_intercept
        )
        
        self.model_.fit(X, y)
        
        # Guardar coeficientes con nombres
        self.coefficients_ = pd.Series(
            self.model_.coef_,
            index=self.feature_names_
        ).sort_values(ascending=False)
        
        self.intercept_ = self.model_.intercept_
        
        logger.info(
            f"✓ LinearRegression entrenado: {len(self.feature_names_)} features"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice precios.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features para predicción
        
        Returns
        -------
        np.ndarray
            Precios predichos
        """
        return self.model_.predict(X)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Retorna features más importantes (por magnitud de coeficiente).
        
        Parameters
        ----------
        top_n : int, default=10
            Número de features a retornar
        
        Returns
        -------
        pd.Series
            Top N features con sus coeficientes
        """
        return self.coefficients_.abs().nlargest(top_n)


class DecisionTreeModel(HousingPriceModel):
    """
    Modelo de Árbol de Decisión para precios de vivienda.
    
    Ventajas:
    - Captura no-linealidades
    - No requiere scaling de features
    - Interpretable (visualización del árbol)
    
    Desventajas:
    - Propenso a overfitting
    - Alta varianza
    
    Parameters
    ----------
    max_depth : int, default=10
        Profundidad máxima del árbol
    min_samples_split : int, default=20
        Mínimo de muestras para dividir nodo
    min_samples_leaf : int, default=10
        Mínimo de muestras en hoja
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Examples
    --------
    >>> model = DecisionTreeModel(max_depth=8)
    >>> model.fit(X_train, y_train)
    >>> model.get_feature_importance()
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 20,
        min_samples_leaf: int = 10,
        random_state: int = 42
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena árbol de decisión.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento
        y : pd.Series
            Precios objetivo
        
        Returns
        -------
        self
        """
        self.feature_names_ = list(X.columns)
        
        self.model_ = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
        
        self.model_.fit(X, y)
        
        logger.info(
            f"✓ DecisionTree entrenado: profundidad={self.model_.get_depth()}, "
            f"hojas={self.model_.get_n_leaves()}"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice precios"""
        return self.model_.predict(X)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Retorna importancia de features (Gini importance).
        
        Parameters
        ----------
        top_n : int, default=10
            Número de features a retornar
        
        Returns
        -------
        pd.Series
            Top N features con su importancia
        """
        importances = pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_
        ).sort_values(ascending=False)
        
        return importances.head(top_n)


class GradientBoostingModel(HousingPriceModel):
    """
    Modelo de Gradient Boosting para precios de vivienda.
    
    Ventajas:
    - Estado del arte en datos tabulares
    - Captura patrones complejos
    - Robusto a outliers
    
    Desventajas:
    - Más lento en entrenamiento
    - Requiere tuning cuidadoso
    - Menos interpretable
    
    Parameters
    ----------
    n_estimators : int, default=100
        Número de árboles
    learning_rate : float, default=0.1
        Tasa de aprendizaje
    max_depth : int, default=5
        Profundidad máxima de cada árbol
    subsample : float, default=0.8
        Fracción de muestras para entrenar cada árbol
    random_state : int, default=42
        Semilla para reproducibilidad
    
    Examples
    --------
    >>> model = GradientBoostingModel(n_estimators=150)
    >>> model.fit(X_train, y_train)
    >>> model.get_feature_importance()
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena modelo de Gradient Boosting.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento
        y : pd.Series
            Precios objetivo
        
        Returns
        -------
        self
        """
        self.feature_names_ = list(X.columns)
        
        self.model_ = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state
        )
        
        self.model_.fit(X, y)
        
        logger.info(
            f"✓ GradientBoosting entrenado: {self.n_estimators} árboles, "
            f"lr={self.learning_rate}"
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice precios"""
        return self.model_.predict(X)
    
    def get_feature_importance(self, top_n: int = 10) -> pd.Series:
        """
        Retorna importancia de features.
        
        Parameters
        ----------
        top_n : int, default=10
            Número de features a retornar
        
        Returns
        -------
        pd.Series
            Top N features con su importancia
        """
        importances = pd.Series(
            self.model_.feature_importances_,
            index=self.feature_names_
        ).sort_values(ascending=False)
        
        return importances.head(top_n)


class EnsembleModel(HousingPriceModel):
    """
    Modelo Ensemble que combina Linear, Tree y Boosting.
    
    Estrategia: Weighted average de predicciones
    - Linear Regression: 30% (interpretabilidad, baseline)
    - Decision Tree: 20% (segmentación)
    - Gradient Boosting: 50% (máxima precisión)
    
    Parameters
    ----------
    weights : Dict[str, float], optional
        Pesos personalizados para cada modelo
        Default: {'linear': 0.3, 'tree': 0.2, 'boosting': 0.5}
    linear_params : Dict, optional
        Parámetros para LinearRegression
    tree_params : Dict, optional
        Parámetros para DecisionTree
    boosting_params : Dict, optional
        Parámetros para GradientBoosting
    
    Attributes
    ----------
    models_ : Dict[str, HousingPriceModel]
        Diccionario con modelos individuales
    weights_ : Dict[str, float]
        Pesos ajustados para ensemble
    
    Examples
    --------
    >>> ensemble = EnsembleModel()
    >>> ensemble.fit(X_train, y_train)
    >>> predictions = ensemble.predict(X_test)
    >>> metrics = ensemble.evaluate(X_test, y_test)
    >>> ensemble.get_individual_metrics(X_test, y_test)
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        linear_params: Optional[Dict] = None,
        tree_params: Optional[Dict] = None,
        boosting_params: Optional[Dict] = None
    ):
        super().__init__()
        
        # Guardar parámetros (necesario para sklearn clone)
        self.weights = weights
        self.linear_params = linear_params or {}
        self.tree_params = tree_params or {}
        self.boosting_params = boosting_params or {}
        
        # Pesos por defecto (basados en ADR-001)
        self.weights_ = weights or {
            'linear': 0.3,
            'tree': 0.2,
            'boosting': 0.5
        }
        
        # Validar que sumen 1.0
        total_weight = sum(self.weights_.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Los pesos deben sumar 1.0, recibidos: {total_weight}"
            )
        
        # Crear modelos
        self.models_ = {
            'linear': LinearRegressionModel(**self.linear_params),
            'tree': DecisionTreeModel(**self.tree_params),
            'boosting': GradientBoostingModel(**self.boosting_params)
        }
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Entrena todos los modelos del ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de entrenamiento
        y : pd.Series
            Precios objetivo
        
        Returns
        -------
        self
        """
        self.feature_names_ = list(X.columns)
        
        logger.info(f"{'='*70}")
        logger.info("Entrenando Ensemble (3 modelos)...")
        logger.info(f"{'='*70}")
        
        # Entrenar cada modelo
        for name, model in self.models_.items():
            logger.info(f"\n[{name.upper()}] Iniciando entrenamiento...")
            model.fit(X, y)
            
            # Evaluar en training set
            metrics = model.evaluate(X, y)
            logger.info(f"[{name.upper()}] Métricas en training:")
            logger.info(f"  R²: {metrics['r2']:.4f}")
            logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Ensemble completamente entrenado")
        logger.info(f"Pesos: {self.weights_}")
        logger.info(f"{'='*70}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice usando weighted average de todos los modelos.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features para predicción
        
        Returns
        -------
        np.ndarray
            Precios predichos (combinación ponderada)
        """
        # Obtener predicciones de cada modelo
        predictions = {}
        for name, model in self.models_.items():
            predictions[name] = model.predict(X)
        
        # Combinar con pesos
        ensemble_pred = (
            self.weights_['linear'] * predictions['linear'] +
            self.weights_['tree'] * predictions['tree'] +
            self.weights_['boosting'] * predictions['boosting']
        )
        
        return ensemble_pred
    
    def get_individual_predictions(
        self,
        X: pd.DataFrame
    ) -> Dict[str, np.ndarray]:
        """
        Retorna predicciones de cada modelo individual.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features para predicción
        
        Returns
        -------
        Dict[str, np.ndarray]
            Diccionario con predicciones de cada modelo
        """
        predictions = {}
        for name, model in self.models_.items():
            predictions[name] = model.predict(X)
        
        return predictions
    
    def get_individual_metrics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> pd.DataFrame:
        """
        Evalúa cada modelo individual y el ensemble.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features de test
        y : pd.Series
            Precios verdaderos
        
        Returns
        -------
        pd.DataFrame
            Tabla comparativa de métricas
        """
        results = []
        
        # Evaluar modelos individuales
        for name, model in self.models_.items():
            metrics = model.evaluate(X, y)
            metrics['model'] = name
            metrics['weight'] = self.weights_[name]
            results.append(metrics)
        
        # Evaluar ensemble
        ensemble_metrics = self.evaluate(X, y)
        ensemble_metrics['model'] = 'ensemble'
        ensemble_metrics['weight'] = 1.0
        results.append(ensemble_metrics)
        
        # Crear DataFrame
        df = pd.DataFrame(results)
        df = df[['model', 'weight', 'r2', 'mae', 'rmse', 'mape']]
        
        return df
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """
        Combina importancia de features de todos los modelos.
        
        Parameters
        ----------
        top_n : int, default=10
            Número de features a mostrar
        
        Returns
        -------
        pd.DataFrame
            Tabla con importancia de cada modelo
        """
        importances = {}
        
        # Linear: usar coeficientes absolutos
        importances['linear'] = self.models_['linear'].get_feature_importance(
            top_n=len(self.feature_names_)
        )
        
        # Tree y Boosting: feature importance
        for name in ['tree', 'boosting']:
            importances[name] = self.models_[name].get_feature_importance(
                top_n=len(self.feature_names_)
            )
        
        # Combinar en DataFrame
        df = pd.DataFrame(importances).fillna(0)
        
        # Calcular importancia ponderada
        df['weighted_importance'] = (
            df['linear'] * self.weights_['linear'] +
            df['tree'] * self.weights_['tree'] +
            df['boosting'] * self.weights_['boosting']
        )
        
        return df.sort_values('weighted_importance', ascending=False).head(top_n)


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 70)
    print("MODELS - Ejemplo de Uso")
    print("=" * 70)
    
    # Crear datos sintéticos
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'tamano': np.random.uniform(50, 200, n_samples),
        'habitaciones': np.random.randint(1, 5, n_samples),
        'precio_m2': np.random.uniform(1500, 3000, n_samples)
    })
    
    # Precio = función de features + ruido
    y = pd.Series(
        X['tamano'] * X['precio_m2'] + 
        X['habitaciones'] * 10000 +
        np.random.normal(0, 5000, n_samples)
    )
    
    print(f"\nDatos sintéticos: {len(X)} muestras")
    print(X.head())
    
    # Entrenar ensemble
    print(f"\n{'='*70}")
    print("Entrenando Ensemble...")
    print(f"{'='*70}\n")
    
    ensemble = EnsembleModel()
    ensemble.fit(X, y)
    
    # Predicciones
    predictions = ensemble.predict(X)
    
    print(f"\n{'='*70}")
    print("Métricas del Ensemble:")
    print(f"{'='*70}")
    print(ensemble.get_metrics_summary())
    
    # Comparar modelos
    print(f"\n{'='*70}")
    print("Comparación de Modelos:")
    print(f"{'='*70}")
    comparison = ensemble.get_individual_metrics(X, y)
    print(comparison.to_string(index=False))