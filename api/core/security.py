"""JWT authentication and role-based authorization."""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from .config import settings

security_scheme = HTTPBearer(auto_error=False)

VALID_ROLES = {"clinician", "research", "admin", "system"}


def create_access_token(user: str, role: str) -> str:
    """Creates a JWT token for the given user and role."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.JWT_EXPIRY_MINUTES)
    payload = {"sub": user, "role": role, "exp": expire}
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decodes and validates a JWT token."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> dict:
    """Extracts current user from JWT. Returns demo user in debug mode."""
    if settings.DEBUG and credentials is None:
        return {"sub": "demo@hare.med", "role": "system"}
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return decode_token(credentials.credentials)


def require_role(required_role: str):
    """Dependency factory that enforces a specific role."""
    async def role_checker(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") != required_role and user.get("role") != "system":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' required",
            )
        return user
    return role_checker
