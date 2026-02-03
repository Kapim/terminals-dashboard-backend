from __future__ import annotations

import asyncio
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field


class CommandType(str, Enum):
    RESTART = "RESTART"
    REFRESH_CONFIG = "REFRESH_CONFIG"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    token: str
    user: Dict[str, Any]


class Device(BaseModel):
    id: str
    name: str
    version: str
    online: bool
    last_seen: Optional[datetime]


class DeviceDetail(Device):
    description: str
    location: str


class CommandRequest(BaseModel):
    type: CommandType


class CommandResponse(BaseModel):
    device_id: str
    accepted: bool
    type: CommandType
    message: str


class LogItem(BaseModel):
    timestamp: float
    level: str
    message: str


class LogsResponse(BaseModel):
    items: List[LogItem]
    nextAfter: float


@dataclass
class DeviceState:
    id: str
    name: str
    version: str
    description: str
    location: str
    online: bool
    last_seen: Optional[datetime]
    logs: List[LogItem] = field(default_factory=list)


app = FastAPI(title="Cloud Terminal Dashboard API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TOKENS: set[str] = set()
DEVICES: Dict[str, DeviceState] = {}


@app.on_event("startup")
async def startup() -> None:
    seed_devices()
    asyncio.create_task(simulate_device_activity())


def seed_devices() -> None:
    if DEVICES:
        return
    now = datetime.now(timezone.utc)
    for idx in range(1, 6):
        device_id = f"term-{idx:03d}"
        DEVICES[device_id] = DeviceState(
            id=device_id,
            name=f"Terminal {idx}",
            version=f"v{1 + idx % 3}.{idx}",
            description="Self-service POS terminal",
            location=random.choice(["Prague", "Brno", "Ostrava"]),
            online=bool(idx % 2),
            last_seen=now if idx % 2 else None,
        )


def get_device_or_404(device_id: str) -> DeviceState:
    device = DEVICES.get(device_id)
    if not device:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Device not found")
    return device


def auth_dependency(authorization: Optional[str] = Header(default=None)) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    if token not in TOKENS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown token")
    return token


@app.post("/auth/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    token = f"mock-{uuid.uuid4()}"
    TOKENS.add(token)
    user = {"email": payload.email, "name": payload.email.split("@")[0]}
    return LoginResponse(token=token, user=user)


@app.get("/devices", response_model=List[Device])
async def list_devices(_: str = Depends(auth_dependency)) -> List[Device]:
    return [
        Device(
            id=device.id,
            name=device.name,
            version=device.version,
            online=device.online,
            last_seen=device.last_seen,
        )
        for device in DEVICES.values()
    ]


@app.get("/devices/{device_id}", response_model=DeviceDetail)
async def device_detail(device_id: str, _: str = Depends(auth_dependency)) -> DeviceDetail:
    device = get_device_or_404(device_id)
    return DeviceDetail(
        id=device.id,
        name=device.name,
        version=device.version,
        online=device.online,
        last_seen=device.last_seen,
        description=device.description,
        location=device.location,
    )


@app.post("/devices/{device_id}/commands", response_model=CommandResponse)
async def device_command(
    device_id: str, payload: CommandRequest, _: str = Depends(auth_dependency)
) -> CommandResponse:
    device = get_device_or_404(device_id)
    message = (
        "Restart scheduled" if payload.type == CommandType.RESTART else "Config refresh queued"
    )
    append_log(device, "INFO", f"Command received: {payload.type}")
    return CommandResponse(
        device_id=device.id,
        accepted=True,
        type=payload.type,
        message=message,
    )


@app.get("/devices/{device_id}/logs", response_model=LogsResponse)
async def device_logs(
    device_id: str,
    after: Optional[float] = None,
    _: str = Depends(auth_dependency),
) -> LogsResponse:
    device = get_device_or_404(device_id)
    filtered = [log for log in device.logs if after is None or log.timestamp > after]
    next_after = filtered[-1].timestamp if filtered else (after or time.time())
    return LogsResponse(items=filtered, nextAfter=next_after)


async def simulate_device_activity() -> None:
    while True:
        await asyncio.sleep(2)
        for device in DEVICES.values():
            if random.random() < 0.1:
                device.online = not device.online
                status = "online" if device.online else "offline"
                append_log(device, "WARN", f"Device toggled {status}")
            if device.online:
                device.last_seen = datetime.now(timezone.utc)
                if random.random() < 0.3:
                    append_log(
                        device,
                        random.choice(["INFO", "DEBUG"]),
                        random.choice(
                            [
                                "Heartbeat ok",
                                "Config applied",
                                "Metrics pushed",
                                "Session closed",
                            ]
                        ),
                    )


def append_log(device: DeviceState, level: str, message: str) -> None:
    device.logs.append(LogItem(timestamp=time.time(), level=level, message=message))
    device.logs[:] = device.logs[-200:]
