from pydantic import BaseModel
from typing import List, Optional


class ChatRequest(BaseModel):
    session_id: str
    message: str


class Citation(BaseModel):
    source: str
    page: Optional[int] = None
    content: str


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]


class UploadResponse(BaseModel):
    filenames: List[str]
    status: str
    chunks_processed: int


class TranscribeResponse(BaseModel):
    text: str
