"""Miroir de la promotion dans `modele`, la table du schéma figé v1.0.

Poser l'alias `champion` met un modèle en service, et c'est le seul geste qui
le fait. Sans ce miroir, l'information ne vivrait que dans le registre MLflow :
l'API EnerVision et le tableau de bord ne pourraient dire quel modèle a produit
une prévision qu'en interrogeant un service dont ce n'est pas le contrat, et
`prediction.modele_id`, qui est NOT NULL, n'aurait aucune ligne à référencer.

L'écriture n'a donc lieu qu'à la promotion, jamais à l'entraînement. Une
version restée `challenger` n'a jamais rien servi : l'inscrire ferait de
`modele` un journal des essais, alors que `prediction.modele_id` attend le
registre de ce qui a tourné.

Deux instructions, une seule transaction. La nouvelle version est posée
active, puis les autres versions du même modèle sont désactivées. Les séparer
laisserait, le temps d'un incident réseau, soit deux versions actives, soit
aucune — et un consommateur qui lit `actif` pendant cette fenêtre lirait faux
sans qu'aucune erreur ne le lui dise.

L'ordre entre les deux est ensuite celui-là et pas l'inverse, pour la même
raison qu'ailleurs dans la chaîne : à l'intérieur de la transaction, la ligne
qui remplace existe avant que celles qu'elle remplace ne s'effacent.
"""

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
    """Construit la ligne décrivant la version qui vient d'être promue.

    `actif` est vrai par construction : cette fonction ne sert qu'à la
    promotion, et une ligne écrite ici décrit toujours le modèle en service.
    """
    return {
        "nom": name,
        "version": str(version),
        "mlflow_run_id": run_id or None,
        "date_entrainement": trained_at,
        "actif": True,
    }


def build_upsert(record: dict[str, Any]) -> Any:
    """Construit l'écriture de la version promue, qui repose sans dupliquer.

    `DO UPDATE` et non `DO NOTHING` : une version rétrogradée puis reprise —
    le retour arrière est un geste courant — est déjà dans la table, avec
    `actif` à faux. L'ignorer laisserait le miroir désigner l'ancienne version
    alors que le registre en sert une autre.
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
    """Construit la désactivation des autres versions du même modèle.

    Bornée au même `nom` : deux modèles distincts ont chacun leur version en
    service, et désactiver au-delà éteindrait un modèle que personne n'a
    demandé de retirer. La version promue est exclue de la clause plutôt que
    réécrite juste après, ce qui la rend indépendante de l'ordre des deux
    instructions.
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
    """Repose dans `modele` la version que l'alias champion désigne désormais.

    Appelée après que l'alias a été posé, et jamais avant : l'alias est ce qui
    met réellement le modèle en service, et une ligne active pour une version
    que le service ne résout pas serait un miroir qui ment.
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
