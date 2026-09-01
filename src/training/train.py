"""Entraînement d'un modèle XGBoost et enregistrement du run dans MLflow.

Le run MLflow est la seule trace qui fait foi : c'est son identifiant que la
table `modele` référence (`mlflow_run_id`) et c'est le tag du modèle enregistré
que le service d'inférence renvoie dans `PredictionOut.model_version`. Un
entraînement qui ne passe pas par ici n'est donc pas déployable.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import mlflow.xgboost
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sqlalchemy import create_engine
from xgboost import XGBRegressor

from etl.config import EtlConfig, load_config
from training.dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    read_measures,
    split_train_test,
)

DEFAULT_HISTORY_DAYS = 90
REGISTERED_MODEL_NAME = "enervision_xgboost"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelParams:
    """Hyperparamètres du modèle, journalisés tels quels dans MLflow."""

    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.9
    random_state: int = 42


def evaluate(observed: pd.Series, predicted: Sequence[float]) -> dict[str, float]:
    """Retourne les métriques de qualité comparables entre deux runs."""
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r2": float(r2_score(observed, predicted)),
    }


def fit_model(train: pd.DataFrame, params: ModelParams) -> XGBRegressor:
    """Entraîne le régresseur sur le jeu d'apprentissage."""
    model = XGBRegressor(**asdict(params))
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])
    return model


def train_site(
    config: EtlConfig,
    site_id: str,
    history_days: int = DEFAULT_HISTORY_DAYS,
    params: ModelParams | None = None,
) -> dict[str, float]:
    """Entraîne un modèle pour un site et enregistre le run dans MLflow."""
    effective_params = params or ModelParams()
    end_time = pd.Timestamp(datetime.now(UTC))
    start_time = end_time - pd.Timedelta(days=history_days)

    engine = create_engine(config.database_url)
    try:
        measures = read_measures(engine, site_id, start_time, end_time)
    finally:
        engine.dispose()

    features = build_features(measures)
    if features.empty:
        raise ValueError(
            f"Aucune mesure exploitable pour {site_id} sur"
            f" {history_days} jours."
        )
    train, test = split_train_test(features)

    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment)
    with mlflow.start_run(run_name=f"{site_id}-xgboost"):
        mlflow.log_params({"site_id": site_id, "history_days": history_days})
        mlflow.log_params(asdict(effective_params))
        model = fit_model(train, effective_params)
        metrics = evaluate(
            test[TARGET_COLUMN], model.predict(test[list(FEATURE_COLUMNS)])
        )
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )
    logger.info("site %s entraîné : %s", site_id, metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande de l'entraînement."""
    parser = argparse.ArgumentParser(description="Entraînement EnerVision.")
    parser.add_argument("--site", required=True, help="Site à entraîner.")
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Profondeur d'historique utilisée pour l'apprentissage.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée de l'entraînement en ligne de commande."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    train_site(load_config(), args.site, history_days=args.history_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
