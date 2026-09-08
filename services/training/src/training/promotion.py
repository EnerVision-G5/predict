# **********************************************************************
# * Nom     : promotion.py                                             *
# * Type    : Module                                                   *
# * Sujet   : Règle qui décide si un candidat mérite d'être mis en     *
# *   service                                                          *
# * Service : training                                                 *
# **********************************************************************

from __future__ import annotations

from dataclasses import dataclass

from training.arbitration import BenchResult
from training.model import DECISION_METRIC

# Marge exigée sur le champion en place, en part d'erreur.
DEFAULT_MARGIN = 0.0


@dataclass(frozen=True)
class Verdict:
    """Classe : Verdict
    Description : Décision de promotion et la raison qui l'a fondée.
    """
    accepted: bool
    reason: str


def decide(
    candidate: BenchResult,
    champion: BenchResult | None,
    naive: BenchResult,
    margin: float = DEFAULT_MARGIN,
) -> Verdict:
    """Méthode : decide
    Description : Oppose le candidat au champion et à la baseline, et tranche.
    """
    measured = candidate.error
    reference = naive.error
    if measured is None or reference is None:
        return Verdict(
            accepted=False,
            reason=(
                f"aucune mesure de {DECISION_METRIC} sur le banc : rien à"
                " comparer, la promotion serait un pari"
            ),
        )
    if measured >= reference:
        return Verdict(
            accepted=False,
            reason=(
                f"{measured:.2f} kW contre {reference:.2f} pour"
                f" {naive.name} : un modèle qui ne bat pas la persistance ne"
                " paie pas ce qu'il coûte (ADR-010)"
            ),
        )
    if champion is None:
        return Verdict(
            accepted=True,
            reason=(
                f"première mise en service : {measured:.2f} kW contre"
                f" {reference:.2f} pour {naive.name}"
            ),
        )
    if champion.window != candidate.window:
        return Verdict(
            accepted=False,
            reason=(
                f"bancs différents — candidat sur {candidate.window}, champion"
                f" sur {champion.window}. Figer training.arbitration pour les"
                " rendre comparables, ou forcer en connaissance de cause"
            ),
        )
    served = champion.error
    if served is None:
        return Verdict(
            accepted=False,
            reason=(
                "le champion ne porte aucune mesure de banc : le comparer"
                " reviendrait à ne comparer à rien"
            ),
        )
    tolerated = served * (1.0 + margin)
    if measured > tolerated:
        return Verdict(
            accepted=False,
            reason=(
                f"dégradation : {measured:.2f} kW contre {served:.2f} pour la"
                f" version servie (tolérance ×{1.0 + margin:.2f})"
            ),
        )
    return Verdict(
        accepted=True,
        reason=(
            f"{measured:.2f} kW contre {served:.2f} pour la version servie et"
            f" {reference:.2f} pour {naive.name}"
        ),
    )
