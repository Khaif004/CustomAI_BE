"""
JWT Authentication and Security Module

Provides:
- JWT token generation and validation
- Password hashing
- Security dependencies for FastAPI
- User authentication
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# Password hashing - use pbkdf2 for compatibility
pwd_context = CryptContext(
    schemes=["pbkdf2_sha256"],
    pbkdf2_sha256__default_rounds=29000,
    deprecated="auto"
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Dict with claims to encode (user_id, username, etc.)
        expires_delta: How long token is valid for
        
    Returns:
        Encoded JWT token string
        
    Example:
        token = create_access_token(
            data={"sub": "user123", "username": "john"},
            expires_delta=timedelta(hours=24)
        )
    """
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire, "iat": datetime.utcnow()})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    logger.debug(f"Token created for {data.get('sub', 'unknown')}, expires at {expire}")
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any]
) -> str:
    """
    Create a JWT refresh token (longer expiry)
    
    Args:
        data: Dict with claims to encode
        
    Returns:
        Encoded JWT refresh token
    """
    settings = get_settings()
    to_encode = data.copy()
    
    expire = datetime.utcnow() + timedelta(days=settings.token_refresh_days)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm
    )
    
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        JWTError: If token is invalid
    """
    settings = get_settings()
    
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise


async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency to get current authenticated user
    
    Extracts Bearer token from Authorization header
    
    Usage:
        @router.get("/profile")
        async def get_profile(current_user = Depends(get_current_user)):
            return {"user": current_user["sub"]}
    
    Args:
        request: HTTP request object
        
    Returns:
        Decoded token payload with user info
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    # Get Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract bearer token
    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Use: Authorization: Bearer TOKEN",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = parts[1]
    
    try:
        payload = verify_token(token)
        
        # Check if it's a refresh token (shouldn't be used for API access)
        if payload.get("type") == "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Cannot use refresh token for API access",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


class TokenData:
    """Token response model"""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # seconds
    
    def __init__(self, access_token: str, token_type: str = "bearer", 
                 refresh_token: Optional[str] = None, expires_in: int = 1800):
        self.access_token = access_token
        self.token_type = token_type
        self.refresh_token = refresh_token
        self.expires_in = expires_in
    
    def dict(self):
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in
        }
