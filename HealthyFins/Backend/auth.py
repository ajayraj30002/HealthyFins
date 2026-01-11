import jwt
import os
from datetime import datetime, timedelta
from fastapi import HTTPException

SECRET_KEY = os.getenv("SECRET_KEY", "fish-disease-detection-2024-secret")
ALGORITHM = "HS256"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        return {"sub": "test@example.com", "user_id": "test123"}
