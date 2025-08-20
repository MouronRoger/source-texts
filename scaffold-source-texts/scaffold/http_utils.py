from __future__ import annotations

"""Shared HTTP utilities for the Daphnet project."""

from typing import Dict

__all__ = ["build_default_headers"]


def _extract_email(user_agent: str) -> str | None:
    """Return email substring found in *user_agent* (best-effort)."""

    # Pattern 1 – "contact: foo@bar" inside UA
    lower = user_agent.lower()
    if "contact:" in lower:
        after = lower.split("contact:", maxsplit=1)[1].strip()
        email = after.split()[0].strip(";)")
        if "@" in email:
            return email
    # Pattern 2 – text inside the last parentheses
    if "(" in user_agent and ")" in user_agent:
        inside = user_agent.split("(")[-1].split(")")[0]
        for token in inside.replace(";", " ").split():
            if "@" in token:
                return token
    return None


def build_default_headers(user_agent: str) -> Dict[str, str]:
    """Return a default header set for all HTTP requests.

    Parameters
    ----------
    user_agent
        The User-Agent string defined in *config.yaml*.
    """

    headers: Dict[str, str] = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml,application/rdf+xml",
        "Accept-Language": "en-US,en;q=0.9,el;q=0.8",
    }
    email = _extract_email(user_agent)
    if email:
        headers["From"] = email
    return headers

