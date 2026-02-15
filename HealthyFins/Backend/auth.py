# auth.py
import jwt
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Header, Depends
from typing import Optional

# Secret key for JWT - in production, use environment variable
SECRET_KEY = os.getenv("SECRET_KEY", "healthyfins-secret-key-2025-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Get current user from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization format. Use 'Bearer <token>'")
    
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(status_code=401, detail="Token missing")
    
    payload = verify_token(token)
    
    # Ensure we have both email and user_id
    if "sub" not in payload or "user_id" not in payload:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    
    return payload
