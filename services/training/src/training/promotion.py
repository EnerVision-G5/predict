"""La règle qui accepte ou refuse une mise en service.

`--promote` promouvait aveuglément : il posait l'alias `champion` sur ce qui
venait d'être appris, sans jamais regarder ce que la version en place savait
faire. Le vocabulaire du challenge était là — `challenger`, `champion` — mais
rien ne les opposait, et rien n'empêchait une semaine d'apprentissage dégradée
de remplacer un modèle meilleur qu'elle. Le seul garde-fou était l'attention
de qui tapait la commande.

Trois conditions, dans cet ordre, et l'ordre porte la décision.

**Battre la baseline naïve.** ADR-010 en fait un livrable permanent : un
modèle qui ne bat pas la recopie de la veille ne paie ni son entraînement, ni
son registre, ni sa surveillance. C'est la première question parce qu'elle
disqualifie sans qu'on ait besoin de regarder le champion — un candidat qui
échoue ici ne mérite pas d'être servi, même s'il est meilleur que ce qui l'est
déjà.

**Être comparable.** Deux mesures faites sur des bancs différents ne se
comparent pas. Le refus est alors franc plutôt que masqué par un classement
que personne ne pourrait défendre — et il se lève en figeant le banc dans
`conf/`.

**Ne pas dégrader.** La marge dit ce qu'on tolère : à zéro, le candidat doit
au moins égaler le champion.

Rien ici ne décide seul. `--force` passe outre, parce qu'un exploitant peut
avoir une raison que la règle n'a pas — un champion entraîné sur une période
aberrante, un banc qu'on sait faussé. Le passage en force est journalisé pour
ce qu'il est.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.arbitration import BenchResult
from training.model import DECISION_METRIC

DEFAULT_MARGIN = 0.0


@dataclass(frozen=True)
class Verdict:
    """La décision, et la phrase qui la justifie.

    La raison n'est pas un journal mais une valeur : elle part dans le tag du
    run et dans le code de sortie de la commande. Une décision qu'on ne peut
    pas relire six mois plus tard n'est pas une décision, c'est un souvenir.
    """

    accepted: bool
    reason: str


def decide(
    candidate: BenchResult,
    champion: BenchResult | None,
    naive: BenchResult,
    margin: float = DEFAULT_MARGIN,
) -> Verdict:
    """Dit si le candidat peut prendre la place du champion, et pourquoi.

    Le champion absent n'est pas un cas dégradé : c'est la première promotion,
    et il n'y a alors rien à ne pas dégrader. La baseline, elle, est exigée
    dans tous les cas — c'est ce qui distingue « premier » de « bon ».
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
