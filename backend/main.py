import base64
from datetime import datetime, timezone
import hashlib
import os
import threading
import time
from typing import List, Optional

import anyio
import cv2
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    inspect,
    text,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from cv_pipeline.feature_extractor import VisionPipeline
from ml_engine.predictor import AttentionPredictor


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SQLITE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'learnsense.db')}"
DATABASE_URL = os.environ.get("DATABASE_URL", DEFAULT_SQLITE_URL).strip() or DEFAULT_SQLITE_URL

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=10,
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class StudySession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(64), nullable=False, index=True, default="")
    subject_name = Column(String(120), nullable=False, default="General Lecture")
    started_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    logs = relationship("MetricLog", back_populates="session", cascade="all, delete-orphan")


class MetricLog(Base):
    __tablename__ = "metric_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False, index=True)
    recorded_at = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    total_faces = Column(Integer, nullable=False, default=0)
    attentive_count = Column(Integer, nullable=False, default=0)
    distracted_count = Column(Integer, nullable=False, default=0)
    sleeping_count = Column(Integer, nullable=False, default=0)
    avg_attention_score = Column(Float, nullable=False, default=0.0)

    session = relationship("StudySession", back_populates="logs")


Base.metadata.create_all(bind=engine)


def ensure_schema():
    inspector = inspect(engine)
    if "sessions" in inspector.get_table_names():
        cols = {col["name"] for col in inspector.get_columns("sessions")}
        if "key_hash" not in cols:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE sessions ADD COLUMN key_hash VARCHAR(64) NOT NULL DEFAULT ''"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sessions_key_hash ON sessions (key_hash)"))


ensure_schema()


# --- Pydantic Schemas ---

class SessionStartRequest(BaseModel):
    subject_name: str = Field(default="General Lecture", min_length=1, max_length=120)


class SessionResponse(BaseModel):
    session_id: int
    subject_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    is_active: bool


class MetricLogCreate(BaseModel):
    session_id: int = Field(..., ge=1)
    total_faces: int = Field(..., ge=0)
    attentive_count: int = Field(..., ge=0)
    distracted_count: int = Field(..., ge=0)
    sleeping_count: int = Field(..., ge=0)
    avg_attention_score: float = Field(..., ge=0.0, le=100.0)


class FramePayload(BaseModel):
    session_id: int = Field(..., ge=1)
    image_base64: str


class MetricLogResponse(BaseModel):
    id: int
    session_id: int
    recorded_at: datetime
    total_faces: int
    attentive_count: int
    distracted_count: int
    sleeping_count: int
    avg_attention_score: float

    class Config:
        from_attributes = True


class SessionSummaryResponse(BaseModel):
    session_id: int
    subject_name: str
    is_active: bool
    started_at: datetime
    ended_at: Optional[datetime] = None
    total_samples: int
    avg_attention_score: float
    total_attentive_events: int
    total_distracted_events: int
    total_sleeping_events: int
    recent_logs: List[MetricLogResponse]


# --- FastAPI App ---

app = FastAPI(title="LearnSense API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def hash_access_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def require_workspace_key(x_train_key: Optional[str] = Header(default=None, alias="X-Train-Key")) -> str:
    cleaned = (x_train_key or "").strip()
    if not cleaned or len(cleaned) < 4 or len(cleaned) > 200:
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid X-Train-Key header (must be 4-200 characters)",
        )
    return hash_access_key(cleaned)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


_init_lock = threading.Lock()
_max_concurrent_frames = max(1, int(os.environ.get("MAX_CONCURRENT_FRAMES", "2")))
_frame_semaphore = threading.BoundedSemaphore(value=_max_concurrent_frames)
_vision_pipeline: Optional[VisionPipeline] = None
_attention_predictor: Optional[AttentionPredictor] = None


def get_vision_pipeline() -> VisionPipeline:
    global _vision_pipeline
    if _vision_pipeline is None:
        with _init_lock:
            if _vision_pipeline is None:
                _vision_pipeline = VisionPipeline(max_num_faces=5)
    return _vision_pipeline


def get_attention_predictor() -> AttentionPredictor:
    global _attention_predictor
    if _attention_predictor is None:
        with _init_lock:
            if _attention_predictor is None:
                _attention_predictor = AttentionPredictor("attention_model.joblib")
    return _attention_predictor


def _decode_base64_image(image_base64: str) -> np.ndarray:
    b64_str = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image frame")
    return frame


def _run_frame_inference_sync(image_base64: str) -> dict:
    acquired = _frame_semaphore.acquire(timeout=8.0)
    if not acquired:
        raise HTTPException(
            status_code=429,
            detail="Server is busy processing other camera frames; retry shortly.",
        )
    try:
        frame = _decode_base64_image(image_base64)
        pipeline = get_vision_pipeline()
        predictor = get_attention_predictor()

        _, faces = pipeline.extract_features(frame, timestamp=time.time())
        for face in faces:
            code, label = predictor.predict(face)
            face["state_code"] = code
            face["state_label"] = label

        return pipeline.summarize_frame(faces)
    finally:
        _frame_semaphore.release()


def _get_owned_session(db: Session, session_id: int, key_hash: str) -> StudySession:
    study_session = (
        db.query(StudySession)
        .filter(StudySession.id == session_id, StudySession.key_hash == key_hash)
        .first()
    )
    if not study_session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found for this access key")
    return study_session


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LearnSense API"}


@app.get("/api/sessions", response_model=List[SessionResponse])
def list_sessions(
    limit: int = Query(default=30, ge=1, le=100),
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(StudySession)
        .filter(StudySession.key_hash == key_hash)
        .order_by(StudySession.id.desc())
        .limit(limit)
        .all()
    )
    sessions.reverse()
    return [
        SessionResponse(
            session_id=s.id,
            subject_name=s.subject_name,
            started_at=s.started_at,
            ended_at=s.ended_at,
            is_active=s.is_active,
        )
        for s in sessions
    ]


@app.post("/api/sessions/start", response_model=SessionResponse)
def start_session(
    payload: SessionStartRequest,
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    new_session = StudySession(
        key_hash=key_hash,
        subject_name=payload.subject_name.strip() or "General Lecture",
        started_at=datetime.now(timezone.utc),
        is_active=True,
    )
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return SessionResponse(
        session_id=new_session.id,
        subject_name=new_session.subject_name,
        started_at=new_session.started_at,
        ended_at=new_session.ended_at,
        is_active=new_session.is_active,
    )


@app.post("/api/sessions/{session_id}/end", response_model=SessionResponse)
def end_session(
    session_id: int,
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    study_session = _get_owned_session(db, session_id, key_hash)
    study_session.is_active = False
    study_session.ended_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(study_session)
    return SessionResponse(
        session_id=study_session.id,
        subject_name=study_session.subject_name,
        started_at=study_session.started_at,
        ended_at=study_session.ended_at,
        is_active=study_session.is_active,
    )


@app.post("/api/metrics/log", response_model=MetricLogResponse)
def log_metrics(
    payload: MetricLogCreate,
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    _get_owned_session(db, payload.session_id, key_hash)
    entry = MetricLog(
        session_id=payload.session_id,
        recorded_at=datetime.now(timezone.utc),
        total_faces=payload.total_faces,
        attentive_count=payload.attentive_count,
        distracted_count=payload.distracted_count,
        sleeping_count=payload.sleeping_count,
        avg_attention_score=round(payload.avg_attention_score, 2),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.post("/api/metrics/frame", response_model=MetricLogResponse)
async def log_metrics_from_frame(
    payload: FramePayload = Body(...),
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    _get_owned_session(db, payload.session_id, key_hash)
    summary = await anyio.to_thread.run_sync(_run_frame_inference_sync, payload.image_base64)

    metric_row = MetricLog(
        session_id=payload.session_id,
        total_faces=summary["total_faces"],
        attentive_count=summary["attentive_count"],
        distracted_count=summary["distracted_count"],
        sleeping_count=summary["sleeping_count"],
        avg_attention_score=round(float(summary["avg_attention_score"]), 2),
    )
    db.add(metric_row)
    db.commit()
    db.refresh(metric_row)
    return metric_row


@app.get("/api/metrics/latest", response_model=List[MetricLogResponse])
def get_latest_metrics(
    limit: int = Query(default=20, ge=1, le=200),
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(MetricLog)
        .join(StudySession, MetricLog.session_id == StudySession.id)
        .filter(StudySession.key_hash == key_hash)
        .order_by(MetricLog.id.desc())
        .limit(limit)
        .all()
    )
    rows.reverse()
    return rows


@app.get("/api/sessions/{session_id}/summary", response_model=SessionSummaryResponse)
def get_session_summary(
    session_id: int,
    key_hash: str = Depends(require_workspace_key),
    db: Session = Depends(get_db),
):
    study_session = _get_owned_session(db, session_id, key_hash)
    agg = (
        db.query(
            func.count(MetricLog.id).label("total_samples"),
            func.coalesce(func.avg(MetricLog.avg_attention_score), 0.0).label("avg_score"),
            func.coalesce(func.sum(MetricLog.attentive_count), 0).label("att_sum"),
            func.coalesce(func.sum(MetricLog.distracted_count), 0).label("dis_sum"),
            func.coalesce(func.sum(MetricLog.sleeping_count), 0).label("slp_sum"),
        )
        .filter(MetricLog.session_id == session_id)
        .one()
    )

    recent = (
        db.query(MetricLog)
        .filter(MetricLog.session_id == session_id)
        .order_by(MetricLog.id.desc())
        .limit(30)
        .all()
    )
    recent.reverse()

    return SessionSummaryResponse(
        session_id=study_session.id,
        subject_name=study_session.subject_name,
        is_active=study_session.is_active,
        started_at=study_session.started_at,
        ended_at=study_session.ended_at,
        total_samples=int(agg.total_samples or 0),
        avg_attention_score=round(float(agg.avg_score or 0.0), 2),
        total_attentive_events=int(agg.att_sum or 0),
        total_distracted_events=int(agg.dis_sum or 0),
        total_sleeping_events=int(agg.slp_sum or 0),
        recent_logs=recent,
    )