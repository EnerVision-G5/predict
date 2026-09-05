"""Contrôle d'accès du service d'inférence.

Le service est routé publiquement (`predict.enervision.com`) et n'exigeait
rien : ses cinq routes étaient ouvertes, dont `POST /api/v1/simulate/spike`,
qui déclenche une écriture réelle sur la source. L'API métier protège pourtant
la même opération derrière le rôle `writer` — le contrôle existait d'un côté,
et le contourner suffisait à s'en passer.

Ce module ferme cela avec une clé de service, et non avec le JWT de l'API
métier. Trois raisons :

- l'appelant n'est pas un utilisateur mais un service. Il n'a pas de session,
  pas de rôle à porter, et rien à révoquer individuellement ;
- vérifier un JWT ici demanderait de partager `JWT_SECRET` avec un second
  service, c'est-à-dire d'élargir la surface d'un secret qui ne sert
  aujourd'hui qu'à l'API ;
- le service d'inférence n'a aucune notion d'utilisateur dans son contrat :
  lui en donner une changerait ce que le contrat gelé décrit.

Le contrôle est un middleware et non une dépendance FastAPI, et c'est
délibéré : une dépendance `Security` publierait un schéma de sécurité dans la
spécification OpenAPI, que le contrat gelé de
`enervision/docs/contracts/openapi-predict.json` ne contient pas. Le job
contract-drift échouerait alors sur une PR qui n'a rien changé au contrat
métier. Déclarer la clé dans le contrat est la bonne cible ; elle demande une
PR de contrat relue par les trois consommateurs, et n'a pas à retarder la
fermeture du trou.

Ce qui reste ouvert, et pourquoi :

- `/health` est la sonde de vivacité de l'hébergeur. Elle ne consulte rien et
  ne publie qu'un horodatage ; l'orchestrateur qui l'interroge n'a pas de
  secret à porter ;
- `/openapi.json`, `/docs` et `/redoc` décrivent le contrat, qui est de toute
  façon publié dans `enervision/docs/contracts`. Le scan DAST de la CI part de
  cette spécification : la fermer rendrait le scan aveugle.

`/ready` est fermé, lui : il nomme la version du modèle servi et l'âge des
variables, ce qui renseigne un attaquant sur l'état de la chaîne.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse

# En-tête portant la clé. `X-API-Key` plutôt que `Authorization: Bearer` :
# ce n'est pas un jeton porteur d'identité, et le confondre avec le JWT de
# l'API métier inviterait à présenter l'un là où l'autre est attendu.
API_KEY_HEADER = "X-API-Key"

# Chemins servis sans clé. Comparés en préfixe pour couvrir les sous-chemins
# que FastAPI ajoute à sa documentation (`/docs/oauth2-redirect`).
PUBLIC_PREFIXES = ("/health", "/openapi.json", "/docs", "/redoc")

# Message unique. Clé absente et clé fausse répondent la même chose : les
# distinguer dirait à l'appelant s'il a trouvé le bon en-tête, ce qui est
# précisément ce qu'on ne veut pas confirmer.
UNAUTHORIZED = "Clé de service absente ou invalide."

logger = logging.getLogger(__name__)


class ServingAuthError(RuntimeError):
    """Clé de service absente de la configuration : le service ne démarre pas."""


def is_public(path: str) -> bool:
    """Dit si le chemin demandé est servi sans clé."""
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in PUBLIC_PREFIXES
    )


def check_api_key(api_key: str, auth_enabled: bool) -> str:
    """Valide la configuration d'authentification au démarrage.

    Appelée à la configuration du service pour échouer tôt et bruyamment
    plutôt qu'à la première requête. Un service qui démarre, répond, et
    accepte tout le monde est le pire des trois états : il a l'air sain.

    `auth_enabled` à faux est le mode du poste de développement et de la CI,
    où aucun secret n'est distribué. Il est journalisé en avertissement à
    chaque démarrage : un déploiement qui le porterait par inadvertance le
    dira dans ses premières lignes de journal.
    """
    if not auth_enabled:
        logger.warning(
            "SERVING_AUTH_ENABLED=false : les routes du service sont ouvertes."
            " Ce mode est celui du développement local, jamais d'un"
            " déploiement.",
        )
        return ""
    if not api_key:
        raise ServingAuthError(
            "SERVING_API_KEY est absente. Le service d'inférence relaie la"
            " simulation de pic, qui écrit sur la source : il refuse de"
            " démarrer sans clé. Générer une valeur puis la placer dans"
            " l'environnement : "
            'python -c "import secrets; print(secrets.token_urlsafe(48))"',
        )
    return api_key


def build_middleware(
    expected: Callable[[], str],
) -> Callable[[Request, Callable[[Request], Awaitable]], Awaitable]:
    """Fabrique le middleware qui exige la clé sur les routes protégées.

    La clé est passée en fonction et non en valeur : Starlette fige la pile de
    middlewares au premier démarrage, alors que la configuration n'est lue que
    dans le `lifespan`. Une valeur capturée ici serait donc toujours celle
    d'avant la configuration, c'est-à-dire vide — le contrôle serait monté et
    ne vérifierait rien.

    Une clé attendue vide désactive le contrôle : c'est ce que rend
    `check_api_key` quand l'authentification est explicitement coupée. Le cas
    « clé vide parce que personne ne l'a configurée » n'arrive jamais ici,
    puisqu'il fait échouer le démarrage en amont.
    """

    async def require_api_key(
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        secret = expected()
        if not secret or is_public(request.url.path):
            return await call_next(request)
        presented = request.headers.get(API_KEY_HEADER, "")
        # compare_digest et non `==` : une comparaison qui s'arrête au premier
        # octet différent laisse mesurer combien d'octets sont bons, et la clé
        # se retrouve octet par octet.
        if not presented or not secrets.compare_digest(presented, secret):
            logger.warning(
                "accès refusé sur %s %s", request.method, request.url.path
            )
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": UNAUTHORIZED},
            )
        return await call_next(request)

    return require_api_key
