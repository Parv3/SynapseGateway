import secrets
from fastapi import Header, HTTPException, status
from typing import Optional
from app.core.config import settings

async def verify_gateway_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-Synapse-Api-Key")
) -> bool:
    """
    Validates client authentication using constant-time string comparison.
    If GATEWAY_API_KEY is unset in the environment, gateway operates in open mode.
    """
    if not settings.GATEWAY_API_KEY:
        return True

    provided_token = None
    if authorization and authorization.lower().startswith("bearer "):
        provided_token = authorization[7:].strip()
    elif x_api_key:
        provided_token = x_api_key.strip()

    if not provided_token or not secrets.compare_digest(provided_token, settings.GATEWAY_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing Gateway API Key. Provide via 'Authorization: Bearer <key>' or 'X-Synapse-Api-Key'."
        )

    return True
