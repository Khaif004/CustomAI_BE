from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional
import logging

from app.auth.security import (
    hash_password, verify_password, create_access_token,
    create_refresh_token, verify_token, get_current_user, TokenData,
)
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenVerifyResponse(BaseModel):
    valid: bool
    user_id: str
    username: str
    expires_at: str


class UserInfo(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None


# In-memory user store (replace with real DB in production)
USERS_DB = {
    "developer": {
        "username": "developer",
        "password_hash": hash_password("developer123"),
        "email": "developer@example.com",
        "full_name": "Developer User"
    },
    "admin": {
        "username": "admin",
        "password_hash": hash_password("admin123"),
        "email": "admin@example.com",
        "full_name": "Admin User"
    }
}


@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(request: LoginRequest):
    """Authenticate user and return JWT tokens"""
    user = USERS_DB.get(request.username)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password")

    access_token = create_access_token(
        data={"sub": request.username, "username": request.username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = create_refresh_token(data={"sub": request.username, "username": request.username})

    logger.info(f"User logged in: {request.username}")
    return TokenResponse(access_token=access_token, refresh_token=refresh_token,
                         token_type="bearer", expires_in=settings.access_token_expire_minutes * 60)


@router.post("/refresh", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def refresh_token(request: RefreshTokenRequest):
    """Get a new access token using refresh token"""
    try:
        payload = verify_token(request.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        access_token = create_access_token(
            data={"sub": payload["sub"], "username": payload.get("username")},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        return TokenResponse(access_token=access_token, refresh_token=request.refresh_token,
                             token_type="bearer", expires_in=settings.access_token_expire_minutes * 60)
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")


@router.post("/verify", response_model=TokenVerifyResponse, status_code=status.HTTP_200_OK)
async def verify_access_token(current_user=Depends(get_current_user)):
    """Verify if a token is valid"""
    from datetime import datetime
    exp_timestamp = current_user.get("exp")
    expires_at = datetime.utcfromtimestamp(exp_timestamp).isoformat() if exp_timestamp else "unknown"
    return TokenVerifyResponse(valid=True, user_id=current_user["sub"],
                               username=current_user.get("username", "unknown"), expires_at=expires_at)


@router.get("/me", response_model=UserInfo, status_code=status.HTTP_200_OK)
async def get_current_user_info(current_user=Depends(get_current_user)):
    """Get current authenticated user info"""
    return UserInfo(user_id=current_user["sub"], username=current_user.get("username", "unknown"),
                    email=current_user.get("email"))


@router.get("/dev/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def get_dev_token(username: str = "developer"):
    """Development only - get token without password"""
    if not settings.debug:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only available in debug mode")

    if username not in USERS_DB:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"User {username} not found")

    access_token = create_access_token(
        data={"sub": username, "username": username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    refresh_token = create_refresh_token(data={"sub": username, "username": username})

    return TokenResponse(access_token=access_token, refresh_token=refresh_token,
                         token_type="bearer", expires_in=settings.access_token_expire_minutes * 60)
