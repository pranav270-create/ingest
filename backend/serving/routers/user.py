from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from jose import jwt
from datetime import datetime, timedelta

router = APIRouter()

# Add this at the top with your other configurations
SECRET_KEY = "testing"
ALGORITHM = "HS256"


class User(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    status: str
    message: str
    token: Optional[str] = None


@router.post("/user/login", response_model=LoginResponse)
async def login(user: User) -> LoginResponse:
    print(user)
    if user.username == "admin@astralis.sh" and user.password == "astralis&admin":
        # Create token with expiration
        token = jwt.encode(
            {
                "sub": user.username,
                "exp": datetime.utcnow() + timedelta(days=1)  # Token expires in 1 day
            },
            SECRET_KEY,
            algorithm=ALGORITHM
        )
        return LoginResponse(
            status="success",
            message="User logged in",
            token=token
        )
    else:
        return LoginResponse(
            status="error",
            message="Invalid credentials"
        )
