import httpx
from fastapi import Cookie, HTTPException, Depends

from core.config import settings


async def get_current_user(session_id: str = Cookie()) -> dict:
    if not session_id:
        raise HTTPException(status_code=401, detail="Unauthorized")
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(settings.BACKEND_URL + "/user", cookies={"session_id": session_id})
    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Unauthorized Failed")
    json_data = response.json()
    user = json_data["data"]
    user["session_id"] = session_id
    return user


def get_admin_user(user: dict = Depends(get_current_user)):
    if not user["is_superuser"]:
        raise HTTPException(status_code=403, detail="Forbidden")
    return user
