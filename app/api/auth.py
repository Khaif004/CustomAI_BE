"""
Authentication API Endpoints

Provides:
- POST /api/auth/login - Get JWT tokens
- POST /api/auth/refresh - Refresh access token
- POST /api/auth/verify - Verify token validity
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from datetime import timedelta
from typing import Optional
import logging

from app.auth.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    TokenData,
)
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ==================== Request/Response Models ====================

class LoginRequest(BaseModel):
    """Login request with username and password"""
    username: str
    password: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "developer",
                "password": "your-secure-password"
            }
        }


class TokenResponse(BaseModel):
    """Token response with access and refresh tokens"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    
    class Config:
        json_schema_extra = {
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 1800
            }
        }


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
            }
        }


class TokenVerifyResponse(BaseModel):
    """Token verification response"""
    valid: bool
    user_id: str
    username: str
    expires_at: str


class UserInfo(BaseModel):
    """Current user information"""
    user_id: str
    username: str
    email: Optional[str] = None


# ==================== In-Memory User Store (Development Only) ====================
# In production, replace with real database (PostgreSQL, etc.)

USERS_DB = {
    "developer": {
        "username": "developer",
        "password_hash": hash_password("developer123"),  # Default dev password
        "email": "developer@example.com",
        "full_name": "Developer User"
    },
    "admin": {
        "username": "admin",
        "password_hash": hash_password("admin123"),  # Default admin password
        "email": "admin@example.com",
        "full_name": "Admin User"
    }
}


# ==================== Endpoints ====================

@router.post("/login", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login(request: LoginRequest):
    """
    Authenticate user and return JWT tokens
    
    This endpoint validates credentials and returns:
    - **access_token**: Use in Authorization header for API calls
    - **refresh_token**: Use to get new access token when expired
    
    Default test users (development only):
    - Username: `developer`, Password: `developer123`
    - Username: `admin`, Password: `admin123`
    
    Usage:
        ```bash
        curl -X POST http://localhost:8000/api/auth/login \\
          -H "Content-Type: application/json" \\
          -d '{"username": "developer", "password": "developer123"}'
        ```
    
    Returns:
        ```json
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer",
            "expires_in": 1800
        }
        ```
    """
    # Find user in database
    user = USERS_DB.get(request.username)
    
    if not user:
        logger.warning(f"Login attempt with non-existent username: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    if not verify_password(request.password, user["password_hash"]):
        logger.warning(f"Failed login attempt for user: {request.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": request.username, "username": request.username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    
    refresh_token = create_refresh_token(
        data={"sub": request.username, "username": request.username}
    )
    
    logger.info(f"User logged in: {request.username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )


@router.post("/refresh", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def refresh_token(request: RefreshTokenRequest):
    """
    Get a new access token using refresh token
    
    When your access token expires, use the refresh token to get a new one
    without re-entering credentials.
    
    Usage:
        ```bash
        curl -X POST http://localhost:8000/api/auth/refresh \\
          -H "Content-Type: application/json" \\
          -d '{"refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."}'
        ```
    """
    try:
        payload = verify_token(request.refresh_token)
        
        # Check if it's actually a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        # Create new access token
        access_token = create_access_token(
            data={"sub": payload["sub"], "username": payload.get("username")},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        logger.info(f"Token refreshed for user: {payload['sub']}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=request.refresh_token,  # Return same refresh token
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60
        )
        
    except Exception as e:
        logger.warning(f"Token refresh failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )


@router.post("/verify", response_model=TokenVerifyResponse, status_code=status.HTTP_200_OK)
async def verify_access_token(current_user = Depends(get_current_user)):
    """
    Verify if a token is valid
    
    Pass your token in Authorization header:
        Authorization: Bearer YOUR_TOKEN_HERE
    
    Usage:
        ```bash
        curl -X POST http://localhost:8000/api/auth/verify \\
          -H "Authorization: Bearer YOUR_TOKEN_HERE"
        ```
    """
    from datetime import datetime
    
    exp_timestamp = current_user.get("exp")
    expires_at = datetime.utcfromtimestamp(exp_timestamp).isoformat() if exp_timestamp else "unknown"
    
    return TokenVerifyResponse(
        valid=True,
        user_id=current_user["sub"],
        username=current_user.get("username", "unknown"),
        expires_at=expires_at
    )


@router.get("/me", response_model=UserInfo, status_code=status.HTTP_200_OK)
async def get_current_user_info(current_user = Depends(get_current_user)):
    """
    Get current authenticated user info
    
    Pass your token in Authorization header:
        Authorization: Bearer YOUR_TOKEN_HERE
    
    Usage:
        ```bash
        curl -X GET http://localhost:8000/api/auth/me \\
          -H "Authorization: Bearer YOUR_TOKEN_HERE"
        ```
    """
    return UserInfo(
        user_id=current_user["sub"],
        username=current_user.get("username", "unknown"),
        email=current_user.get("email")
    )


# ==================== Optional: Create Token Without Login (Development Only) ====================

@router.get("/dev/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def get_dev_token(username: str = "developer"):
    """
    ⚠️ DEVELOPMENT ONLY - Get token without password
    
    For testing purposes only! Remove in production.
    
    Usage:
        ```bash
        curl http://localhost:8000/api/auth/dev/token?username=developer
        ```
    """
    if not settings.debug:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This endpoint is only available in debug mode",
        )
    
    if username not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {username} not found",
        )
    
    access_token = create_access_token(
        data={"sub": username, "username": username},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
    )
    
    refresh_token = create_refresh_token(
        data={"sub": username, "username": username}
    )
    
    logger.warning(f"Dev token generated for user: {username}")
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60
    )
