# **********************************************************************
# * Nom     : auth.py                                                  *
# * Type    : Module                                                   *
# * Sujet   : Contrôle de la clé de service sur les routes non         *
# *   publiques                                                        *
# * Service : serving                                                  *
# **********************************************************************

from __future__ import annotations

import logging
import secrets
from collections.abc import Awaitable, Callable

from fastapi import Request, status
from fastapi.responses import JSONResponse

# En-tête portant la clé de service.
API_KEY_HEADER = "X-API-Key"

# Routes ouvertes sans clé : sonde et documentation.
PUBLIC_PREFIXES = ("/health", "/openapi.json", "/docs", "/redoc")

# Message rendu à un appel sans clé valable.
UNAUTHORIZED = "Clé de service absente ou invalide."

logger = logging.getLogger(__name__)


class ServingAuthError(RuntimeError):
    """Classe : ServingAuthError
    Description : Le service est configuré sans clé alors qu'il en exige une.
    """


def is_public(path: str) -> bool:
    """Méthode : is_public
    Description : Dit si une route est joignable sans clé de service.
    """
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in PUBLIC_PREFIXES
    )


def check_api_key(api_key: str, auth_enabled: bool) -> str:
    """Méthode : check_api_key
    Description : Vérifie la clé présentée, ou avertit que le contrôle est
      désactivé.
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
    """Méthode : build_middleware
    Description : Construit le middleware qui exige la clé sur les routes
      protégées.
    """
    async def require_api_key(
        request: Request,
        call_next: Callable[[Request], Awaitable],
    ):
        secret = expected()
        if not secret or is_public(request.url.path):
            return await call_next(request)
        presented = request.headers.get(API_KEY_HEADER, "")
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
