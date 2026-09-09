# **********************************************************************
# * Nom     : registry.py                                              *
# * Type    : Module                                                   *
# * Sujet   : Inscription en base du modèle mis en service             *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine

from predict_common.db import MODELE_KEY, MODELE_UPDATED_COLUMNS, modele

logger = logging.getLogger(__name__)


def to_record(
    name: str,
    version: str,
    run_id: str,
    trained_at: datetime,
) -> dict[str, Any]:
    """Méthode : to_record
    Description : Compose la ligne de modele décrivant la version promue.
    """
    return {
        "nom": name,
        "version": str(version),
        "mlflow_run_id": run_id or None,
        "date_entrainement": trained_at,
        "actif": True,
    }


def build_upsert(record: dict[str, Any]) -> Any:
    """Méthode : build_upsert
    Description : Écrit la version promue, en réécrivant celle déjà inscrite.
    """
    statement = insert(modele).values([record])
    return statement.on_conflict_do_update(
        index_elements=list(MODELE_KEY),
        set_={
            name: getattr(statement.excluded, name)
            for name in MODELE_UPDATED_COLUMNS
        },
    )


def build_demotion(name: str, version: str) -> Any:
    """Méthode : build_demotion
    Description : Retire le drapeau actif à toutes les autres versions du
      modèle.
    """
    return (
        update(modele)
        .where(
            modele.c.nom == name,
            modele.c.version != str(version),
            modele.c.actif.is_(True),
        )
        .values(actif=False)
    )


def publish_champion(
    engine: Engine,
    name: str,
    version: str,
    run_id: str,
    trained_at: datetime,
) -> None:
    """Méthode : publish_champion
    Description : Inscrit la version active et désactive les précédentes, d'un
      seul tenant.
    """
    record = to_record(name, version, run_id, trained_at)
    with engine.begin() as connection:
        connection.execute(build_upsert(record))
        connection.execute(build_demotion(name, version))
    logger.info(
        "modele : %s version %s inscrite active, run %s",
        name,
        version,
        run_id or "inconnu",
    )
