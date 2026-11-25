import base64
from datetime import datetime, timezone
import os
import secrets
import sys
import time
from typing import List, Optional

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
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "cv_pipeline") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "cv_pipeline"))

from cv_pipeline.feature_extractor import VisionPipeline
from ml_engine.predictor import AttentionPredictor

# --- Database Setup (Postgres via DATABASE_URL in prod, SQLite locally) ---
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    db_path = os.path.join(BASE_DIR, "learnsense.db")
    DATABASE_URL = f"sqlite:///{db_path}"
elif DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class StudySession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    subject_name = Column(String(120), nullable=False, default="General Lecture")
    started_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    logs = relationship(
        "MetricLog", back_populates="session", cascade="all, delete-orphan"
    )


class MetricLog(Base):
    __tablename__ = "metric_logs"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(
        Integer, ForeignKey("sessions.id"), nullable=False, index=True
    )
    recorded_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    total_faces = Column(Integer, nullable=False, default=0)
    attentive_count = Column(Integer, nullable=False, default=0)
    distracted_count = Column(Integer, nullable=False, default=0)
    sleeping_count = Column(Integer, nullable=False, default=0)
    avg_attention_score = Column(Float, nullable=False, default=0.0)

    session = relationship("StudySession", back_populates="logs")


Base.metadata.create_all(bind=engine)


# --- Pydantic Schemas ---
class SessionStartRequest(BaseModel):
    subject_name: str = Field(
        default="General Lecture", min_length=1, max_length=120
    )


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


class FramePayload(BaseModel):
    session_id: int = Field(..., ge=1)
    image_base64: str = Field(..., min_length=1)


# --- FastAPI App & Security ---
app = FastAPI(title="LearnSense API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAIN_ACCESS_KEY = os.environ.get("TRAIN_ACCESS_KEY", "").strip()


def verify_train_key(
    x_train_key: Optional[str] = Header(default=None, alias="X-Train-Key"),
):
    if not TRAIN_ACCESS_KEY:
        raise HTTPException(
            status_code=503,
            detail="Server access key is not configured (set TRAIN_ACCESS_KEY).",
        )
    if not x_train_key or not secrets.compare_digest(
        x_train_key, TRAIN_ACCESS_KEY
    ):
        raise HTTPException(
            status_code=401, detail="Invalid or missing X-Train-Key"
        )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Lazy Vision + ML Pipeline ---
_vision_pipeline: Optional[VisionPipeline] = None
_attention_predictor: Optional[AttentionPredictor] = None


def get_vision_pipeline() -> VisionPipeline:
    global _vision_pipeline
    if _vision_pipeline is None:
        _vision_pipeline = VisionPipeline(max_num_faces=5)
    return _vision_pipeline


def get_attention_predictor() -> AttentionPredictor:
    global _attention_predictor
    if _attention_predictor is None:
        model_path = os.path.join(
            PROJECT_ROOT, "ml_engine", "attention_model.joblib"
        )
        if not os.path.exists(model_path):
            model_path = os.path.join(PROJECT_ROOT, "attention_model.joblib")
        _attention_predictor = AttentionPredictor(model_path)
    return _attention_predictor


def _decode_base64_image(image_base64: str) -> np.ndarray:
    b64_str = (
        image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
    )
    try:
        img_bytes = base64.b64decode(b64_str, validate=True)
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Invalid base64 image payload"
        ) from exc

    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(
            status_code=400, detail="Invalid or corrupted image frame"
        )
    return frame


# --- Routes ---
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "LearnSense API"}


@app.post(
    "/api/sessions/start",
    response_model=SessionResponse,
    dependencies=[Depends(verify_train_key)],
)
def start_session(
    payload: SessionStartRequest, db: Session = Depends(get_db)
):
    new_session = StudySession(
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


@app.post(
    "/api/sessions/{session_id}/end",
    response_model=SessionResponse,
    dependencies=[Depends(verify_train_key)],
)
def end_session(session_id: int, db: Session = Depends(get_db)):
    study_session = (
        db.query(StudySession).filter(StudySession.id == session_id).first()
    )
    if not study_session:
        raise HTTPException(status_code=404, detail="Session not found")

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


@app.post(
    "/api/metrics/log",
    response_model=MetricLogResponse,
    dependencies=[Depends(verify_train_key)],
)
def log_metrics(payload: MetricLogCreate, db: Session = Depends(get_db)):
    study_session = (
        db.query(StudySession)
        .filter(StudySession.id == payload.session_id)
        .first()
    )
    if not study_session:
        raise HTTPException(
            status_code=404, detail=f"Session {payload.session_id} not found"
        )

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


@app.post(
    "/api/metrics/frame",
    response_model=MetricLogResponse,
    dependencies=[Depends(verify_train_key)],
)
def log_metrics_from_frame(
    payload: FramePayload = Body(...), db: Session = Depends(get_db)
):
    study_session = (
        db.query(StudySession)
        .filter(StudySession.id == payload.session_id)
        .first()
    )
    if not study_session:
        raise HTTPException(
            status_code=404, detail=f"Session #{payload.session_id} not found"
        )

    frame = _decode_base64_image(payload.image_base64)
    pipeline = get_vision_pipeline()
    predictor = get_attention_predictor()

    _, faces = pipeline.extract_features(frame, timestamp=time.time())
    for face in faces:
        code, label = predictor.predict(face)
        face["state_code"] = code
        face["state_label"] = label

    summary = pipeline.summarize_frame(faces)

    entry = MetricLog(
        session_id=payload.session_id,
        recorded_at=datetime.now(timezone.utc),
        total_faces=summary["total_faces"],
        attentive_count=summary["attentive_count"],
        distracted_count=summary["distracted_count"],
        sleeping_count=summary["sleeping_count"],
        avg_attention_score=round(float(summary["avg_attention_score"]), 2),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@app.get(
    "/api/metrics/latest",
    response_model=List[MetricLogResponse],
    dependencies=[Depends(verify_train_key)],
)
def get_latest_metrics(
    limit: int = Query(default=20, ge=1, le=200),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(MetricLog).order_by(MetricLog.id.desc()).limit(limit).all()
    )
    rows.reverse()
    return rows


@app.get(
    "/api/sessions/{session_id}/summary",
    response_model=SessionSummaryResponse,
    dependencies=[Depends(verify_train_key)],
)
def get_session_summary(session_id: int, db: Session = Depends(get_db)):
    study_session = (
        db.query(StudySession).filter(StudySession.id == session_id).first()
    )
    if not study_session:
        raise HTTPException(status_code=404, detail="Session not found")

    agg = (
        db.query(
            func.count(MetricLog.id).label("total_samples"),
            func.coalesce(func.avg(MetricLog.avg_attention_score), 0.0).label(
                "avg_score"
            ),
            func.coalesce(func.sum(MetricLog.attentive_count), 0).label(
                "total_att"
            ),
            func.coalesce(func.sum(MetricLog.distracted_count), 0).label(
                "total_dis"
            ),
            func.coalesce(func.sum(MetricLog.sleeping_count), 0).label(
                "total_slp"
            ),
        )
        .filter(MetricLog.session_id == session_id)
        .one()
    )

    recent_logs = (
        db.query(MetricLog)
        .filter(MetricLog.session_id == session_id)
        .order_by(MetricLog.id.desc())
        .limit(25)
        .all()
    )
    recent_logs.reverse()

    return SessionSummaryResponse(
        session_id=study_session.id,
        subject_name=study_session.subject_name,
        is_active=study_session.is_active,
        started_at=study_session.started_at,
        ended_at=study_session.ended_at,
        total_samples=int(agg.total_samples or 0),
        avg_attention_score=round(float(agg.avg_score or 0.0), 2),
        total_attentive_events=int(agg.total_att or 0),
        total_distracted_events=int(agg.total_dis or 0),
        total_sleeping_events=int(agg.total_slp or 0),
        recent_logs=recent_logs,
    )