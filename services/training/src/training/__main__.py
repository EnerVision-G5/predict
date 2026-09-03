"""Point d'entrée de l'entraînement : des partitions en entrée, un modèle en sortie.

    python -m training --feature-version v1
    python -m training --feature-version v1 --history-days 180 --site SITE001

Le service lit `features/{version}/dt=.../` et enregistre un modèle dans
MLflow. Il n'écrit aucun fichier que quelqu'un d'autre devrait aller chercher,
et ne connaît ni le collecteur, ni l'ETL, ni le service d'inférence.

Un modèle entraîné hors de ce chemin n'est pas déployable, et c'est
volontaire : le service d'inférence ne charge que ce que le registre lui
désigne. C'est aussi ce qui rend un modèle traçable — son run porte la version
des variables, la fenêtre apprise et les métriques du bloc de test.

L'entraînement produit un `challenger`, jamais un `champion`. Promouvoir est
une décision d'exploitation, prise en déplaçant l'alias dans MLflow, et le
service la suit sans être redéployé.

`--promote` fait deux choses et non une : il déplace l'alias, puis inscrit la
version dans `modele`, la table du schéma figé. Ce second geste demande la
base, que l'entraînement ne touche dans aucun autre cas — c'est pourquoi la
connexion est ouverte avant l'apprentissage et non après. Découvrir une
DATABASE_URL absente au bout d'une heure de calcul laisserait le choix entre
perdre le run et servir un modèle que rien ne référence.

`--promote-version` fait le même geste sur une version déjà enregistrée, sans
rien réapprendre :

    python -m training --promote-version 7

C'est la voie qui remet le miroir d'aplomb après un `mlflow models set-alias`,
qui déplace l'alias sans rien savoir de `modele`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from mlflow.exceptions import MlflowException
from sqlalchemy.exc import SQLAlchemyError

from predict_common.config import Config, ConfigError, load_config
from predict_common.db import DatabaseError, open_engine
from predict_common.paths import PathError, parse_date
from predict_common.schemas import feature_columns
from training import registry, tracking
from training.dataset import (
    DatasetError,
    Split,
    matrices,
    read_features,
    select,
    split_by_time,
)
from training.model import (
    ModelParams,
    best_iteration,
    evaluate,
    fit,
    residual_std,
)

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

DEFAULT_HISTORY_DAYS = 90

EXIT_OK = 0
EXIT_FAILED = 1

logger = logging.getLogger(__name__)


def tracking_settings(config: Config) -> tracking.TrackingSettings:
    """Lit où le run doit être écrit et sous quel nom enregistrer le modèle."""
    return tracking.TrackingSettings(
        tracking_uri=config.get_str("mlflow.tracking_uri"),
        experiment=config.get_str("training.experiment"),
        registered_model=config.get_str("training.registered_model"),
    )


def load_split(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
) -> Split:
    """Lit la fenêtre d'apprentissage et la découpe dans l'ordre du temps."""
    start = end - timedelta(days=history_days - 1)
    frame = read_features(config.get_str("storage.root"), version, start, end)
    if frame.empty:
        raise DatasetError(
            f"Aucune partition de variables {version} entre {start} et {end}."
            " Lancer l'ETL sur cette fenêtre avant d'entraîner."
        )
    usable = select(frame, sites, config.get_float("training.max_imputed_ratio"))
    return split_by_time(
        usable,
        valid_ratio=config.get_float("training.valid_ratio"),
        test_ratio=config.get_float("training.test_ratio"),
    )


def promote(
    engine: Engine,
    settings: tracking.TrackingSettings,
    version: str,
    run_id: str,
    trained_at: datetime,
) -> None:
    """Met la version en service, puis l'inscrit comme telle dans `modele`.

    L'alias d'abord, le miroir ensuite. C'est l'alias qui met réellement le
    modèle en service : une ligne active pour une version que le service ne
    résout pas serait un miroir qui ment, alors qu'un alias déplacé sans
    miroir est un retard visible, que le journal nomme et qu'un second
    `--promote` rattrape.
    """
    tracking.set_alias(settings.registered_model, version, tracking.PRODUCTION_ALIAS)
    registry.publish_champion(
        engine, settings.registered_model, version, run_id, trained_at
    )


def promote_registered(
    config: Config,
    engine: Engine,
    version: str,
) -> None:
    """Met en service une version déjà enregistrée, sans rien réapprendre.

    Deux usages, et le second est le plus fréquent. Servir une version qu'on
    a laissée décanter en `challenger`, et rattraper un alias déplacé à la
    main — `mlflow models set-alias` ne connaît pas `modele` et laisse le
    miroir en arrière. L'inscription étant un `ON CONFLICT DO UPDATE`, la
    rejouer sur une version déjà active ne fait rien de plus.
    """
    settings = tracking_settings(config)
    tracking.connect(settings)
    run_id, trained_at = tracking.version_identity(settings.registered_model, version)
    promote(engine, settings, version, run_id, trained_at)


def train(
    config: Config,
    version: str,
    end: date,
    history_days: int,
    sites: Sequence[str] | None,
    engine: Engine | None = None,
) -> dict[str, float]:
    """Entraîne un modèle sur la fenêtre demandée et enregistre son run.

    Un moteur passé vaut demande de promotion : `main` ne l'ouvre que sous
    `--promote`, et la base n'a aucun autre usage dans ce service.
    """
    split = load_split(config, version, end, history_days, sites)
    columns = feature_columns(
        config.get_int_list("etl.lag_hours"), config.get_int("etl.rolling_window_h")
    )
    params = ModelParams.from_mapping(config.section("training.params"))
    settings = tracking_settings(config)
    early_stopping = config.get_int("training.early_stopping_rounds")

    train_x, train_y = matrices(split.train, columns)
    valid_x, valid_y = matrices(split.valid, columns)
    test_x, test_y = matrices(split.test, columns)

    with tracking.run(settings, run_name=f"{version}-xgboost") as active:
        run_id = active.info.run_id
        trained_at = tracking.started_at(active)
        model = fit(train_x, train_y, valid_x, valid_y, params, early_stopping)
        # Une seule prédiction sur le test, relue deux fois : la refaire pour
        # l'écart-type ferait dépendre l'intervalle servi d'un second calcul
        # que rien ne garantirait identique au premier.
        predicted = model.predict(test_x)
        metrics = {
            **evaluate(test_y, predicted),
            "residual_std": residual_std(test_y, predicted),
        }
        tracking.log_params(
            {
                **params.as_dict(),
                **split.sizes,
                "feature_version": version,
                "train_window": split.window,
                "history_days": history_days,
                "sites": ",".join(sites) if sites else "toutes",
                "early_stopping_rounds": early_stopping,
                "best_iteration": best_iteration(model),
            }
        )
        tracking.log_metrics(metrics)
        registered = tracking.log_model(
            model,
            settings,
            valid_x,
            model.predict(valid_x),
            tags={
                "feature_version": version,
                "train_window": split.window,
                "sites": ",".join(sites) if sites else "toutes",
                "mae": round(metrics["mae"], 4),
                "rmse": round(metrics["rmse"], 4),
                "r2": round(metrics["r2"], 4),
                # Le service d'inférence lit ce tag pour borner sa prévision.
                # Un tag et non une métrique de run : le service résout un
                # alias, il n'a pas à remonter jusqu'au run qui l'a produit.
                "residual_std": round(metrics["residual_std"], 4),
            },
        )
    if engine is not None and registered:
        # Hors du contexte du run : promouvoir n'appartient pas à
        # l'entraînement, c'est une décision d'exploitation que la ligne de
        # commande exprime. Le run, lui, est clos dès que le modèle est écrit.
        promote(engine, settings, registered, run_id, trained_at)
    logger.info("entraînement terminé sur %s : %s", split.window, metrics)
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Analyse la ligne de commande de l'entraînement."""
    parser = argparse.ArgumentParser(
        prog="training",
        description="Entraînement du modèle de prévision EnerVision.",
    )
    parser.add_argument(
        "--feature-version",
        default=None,
        help="Version des variables lues. Défaut : etl.feature_version.",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=DEFAULT_HISTORY_DAYS,
        help="Profondeur d'historique apprise, en journées.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="Dernière journée apprise, YYYY-MM-DD. Défaut : aujourd'hui.",
    )
    parser.add_argument(
        "--site",
        action="append",
        dest="sites",
        help="Site appris. Répétable. Par défaut : tous.",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Expérience MLflow. Défaut : training.experiment.",
    )
    parser.add_argument(
        "--promote-version",
        default=None,
        help=(
            "Met en service une version DÉJÀ enregistrée, sans réapprendre :"
            " pose l'alias champion et met `modele` à jour. C'est aussi ce qui"
            " rattrape un alias déplacé à la main. Exige DATABASE_URL."
        ),
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Pose aussi l'alias champion, donc met le modèle en service, et"
            " l'inscrit active dans la table `modele`. Exige DATABASE_URL."
            " Sans cette option, la version reste challenger."
        ),
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    """Arme le journal, et met la sortie standard à l'abri de l'encodage local.

    MLflow imprime des emoji quand il rend la main ; une console Windows en
    cp1252 lève alors une UnicodeEncodeError au beau milieu d'un run qui, lui,
    s'est bien passé. On ne peut pas demander à MLflow de se taire, mais on
    peut faire en sorte qu'un caractère non représentable dégrade l'affichage
    au lieu d'interrompre le traitement.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Point d'entrée du conteneur d'entraînement."""
    _configure_logging()
    args = parse_args(argv)
    engine: Engine | None = None
    try:
        config = load_config()
        if args.experiment:
            config = _with_experiment(config, args.experiment)
        version = args.feature_version or config.get_str("etl.feature_version")
        end = parse_date(args.until) if args.until else datetime.now(UTC).date()
        if args.history_days < 1:
            raise ValueError("--history-days doit valoir au moins 1.")
        if args.promote and args.promote_version:
            raise ValueError(
                "--promote et --promote-version s'excluent : le premier met en"
                " service ce qu'il vient d'apprendre, le second une version"
                " déjà enregistrée."
            )
        # Avant l'apprentissage : une DATABASE_URL absente doit se voir en
        # quelques secondes, pas une heure plus tard, quand il ne resterait
        # qu'à choisir entre perdre le run et servir un modèle non référencé.
        if args.promote or args.promote_version:
            engine = open_engine(config.get_optional_str("database.url"))
        if args.promote_version:
            promote_registered(config, engine, args.promote_version)
        else:
            train(config, version, end, args.history_days, args.sites, engine)
    except (ConfigError, PathError, DatasetError) as exc:
        logger.error("entraînement interrompu : %s", exc)
        return EXIT_FAILED
    except DatabaseError as exc:
        logger.error("promotion impossible : %s", exc)
        return EXIT_FAILED
    except MlflowException as exc:
        # Une version inconnue du registre sort ici, et non par la trace
        # complète : c'est la faute de saisie la plus courante sous
        # --promote-version, et elle n'a rien d'un incident.
        logger.error("registre MLflow : %s", exc)
        return EXIT_FAILED
    except SQLAlchemyError as exc:
        # L'alias est posé et le modèle est servi ; seul le miroir manque. Le
        # run sort en échec pour que l'exploitant le sache, et un second
        # `--promote` sur la même version rattrape sans rien réapprendre.
        logger.error(
            "alias champion posé, mais `modele` non mise à jour : %s."
            " Relancer --promote une fois la base joignable.",
            exc,
        )
        return EXIT_FAILED
    except ValueError as exc:
        # Les erreurs de la chaîne ont leur type ; celles-ci viennent
        # d'ailleurs — un hyperparamètre refusé par XGBoost, un encodage de
        # console. Les ranger sous le même message enverrait chercher la panne
        # du mauvais côté, alors on dit d'où elle sort.
        logger.error("erreur inattendue (%s) : %s", type(exc).__name__, exc)
        return EXIT_FAILED
    finally:
        if engine is not None:
            engine.dispose()
    return EXIT_OK


def _with_experiment(config: Config, experiment: str) -> Config:
    """Retourne la configuration avec l'expérience imposée en ligne de commande."""
    values = {**config.values}
    values["training"] = {**values.get("training", {}), "experiment": experiment}
    return type(config)(values=values, env_name=config.env_name)


if __name__ == "__main__":
    raise SystemExit(main())
