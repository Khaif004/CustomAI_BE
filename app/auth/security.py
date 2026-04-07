from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
import logging
from app.config import get_settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], pbkdf2_sha256__default_rounds=29000, deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(data: Dict[str, Any]) -> str:
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.token_refresh_days)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    return jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def verify_token(token: str) -> Dict[str, Any]:
    """Verify JWT token - supports both internal and XSUAA tokens"""
    settings = get_settings()
    
    try:
        unverified_payload = jwt.get_unverified_claims(token)
        issuer = unverified_payload.get("iss", "")
        
        if settings.xsuaa_issuer and (settings.xsuaa_issuer in issuer or "authentication.eu10.hana.ondemand.com" in issuer):
            logger.info(f"Verifying XSUAA token from issuer: {issuer}")
            
            if not settings.xsuaa_public_key_formatted:
                logger.error("XSUAA_PUBLIC_KEY not configured in environment")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="XSUAA authentication not properly configured"
                )
            
            try:
                payload = jwt.decode(
                    token, 
                    settings.xsuaa_public_key_formatted, 
                    algorithms=["RS256"],
                    options={"verify_aud": False}
                )
                logger.info(f"XSUAA token verified for user: {payload.get('user_name', 'unknown')}")
                return payload
            except JWTError as e:
                logger.warning(f"XSUAA token verification failed: {e}")
                raise
        else:
            logger.info(f"Verifying internal token from issuer: {issuer}")
            return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
            
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        raise


async def get_current_user(request: Request) -> Dict[str, Any]:
    """FastAPI dependency to extract and validate JWT from Authorization header"""
    auth_header = request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Authorization: Bearer TOKEN",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        payload = verify_token(parts[1])

        if payload.get("type") == "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cannot use refresh token for API access",
                headers={"WWW-Authenticate": "Bearer"}
            )

        if payload.get("sub") is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials - missing subject",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user_id = payload.get("user_name") or payload.get("email") or payload.get("sub")
        logger.info(f"Authenticated user: {user_id}")

        return payload

    except JWTError as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )


class TokenData:
    """Token response data"""
    def __init__(self, access_token: str, token_type: str = "bearer",
                 refresh_token: Optional[str] = None, expires_in: int = 1800):
        self.access_token = access_token
        self.token_type = token_type
        self.refresh_token = refresh_token
        self.expires_in = expires_in

    def dict(self):
        return {"access_token": self.access_token, "refresh_token": self.refresh_token,
                "token_type": self.token_type, "expires_in": self.expires_in}
