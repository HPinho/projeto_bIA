"""BIA-02.8: local credential verification and fail-closed institutional access.

Access control is enforced at document boundaries. Institutional sources with no
owner are legacy-unassigned and are readable only by the global administrator;
they must be explicitly classified before institutional users can read them.
"""
from __future__ import annotations

from hashlib import sha256
from hmac import compare_digest

from .institutional import INSTITUTES

ROLES = ("viewer", "researcher", "curator", "admin")


def compile_keys(config: dict) -> tuple[tuple[bytes, dict], ...]:
    if not isinstance(config, dict):
        raise ValueError("BIA_API_KEYS deve ser objeto JSON")
    identities: set[str] = set()
    profiles: list[tuple[bytes, dict]] = []
    for token, profile in config.items():
        if not isinstance(token, str) or not token or len(token) > 4096 or not isinstance(profile, dict):
            raise ValueError("Token ou perfil de autenticação inválido")
        actor, role = profile.get("actor"), profile.get("role")
        institutes = profile.get("institutes", [])
        if (not isinstance(actor, str) or not actor.strip() or len(actor) > 128
                or actor != actor.strip() or role not in ROLES or actor in identities):
            raise ValueError("Identidade duplicada ou inválida em BIA_API_KEYS")
        if not isinstance(institutes, list) or any(
                not isinstance(item, str) or item not in INSTITUTES for item in institutes) or len(set(institutes)) != len(institutes):
            raise ValueError("Afiliação institucional inválida em BIA_API_KEYS")
        identities.add(actor)
        profiles.append((sha256(token.encode('utf-8')).digest(), {
            "actor": actor, "role": role, "institutes": tuple(institutes)
        }))
    return tuple(profiles)


def authenticate(compiled: tuple[tuple[bytes, dict], ...], token: str) -> dict | None:
    digest = sha256(token.encode('utf-8')).digest()
    result = None
    for expected, profile in compiled:
        if compare_digest(digest, expected):
            result = profile
    return dict(result) if result else None


def can_read(doc, user: dict) -> bool:
    if doc.status == "restricted":
        return user["role"] == "admin"
    if doc.rights_basis != "institutional_permission":
        return True
    if user["role"] == "admin":
        return True
    return (user["role"] == "curator" and bool(doc.institute)
            and doc.institute in user.get("institutes", ()))


def upload_scope(user: dict, rights_basis: str, institute: str | None) -> str | None:
    """Returns the assigned institute or raises ValueError; never silently picks a tenant."""
    if rights_basis != "institutional_permission":
        if institute:
            raise ValueError("Instituto só pode ser informado para documentos institucionais")
        return None
    if institute not in INSTITUTES:
        raise ValueError("Documento institucional exige instituto explícito válido")
    if user["role"] != "admin" and institute not in user.get("institutes", ()):
        raise PermissionError("Afiliação não autorizada para cadastrar este documento")
    return institute
