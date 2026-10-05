"""
API/user interface layer (FR-07, FR-10, FR-11, SEC-01..SEC-05).

* Authentication: login with password + TOTP → short-lived JWT; every endpoint requires a named "permission"
  (RBAC with deny-by-default) and every access/denial is recorded in ``audit_log``.
* Inputs are bounded with Pydantic (string length, row count, numeric ranges) and bodies over 2 MB are rejected.
* Security headers and CSP with nonce for the dashboard; API responses are not cached.
* Heavy jobs (full design) run asynchronously on a thread pool and are recorded in the ``jobs`` table.

TLS 1.3 is enforced at the deployment layer (``security.tls.ssl_context`` / ``deploy/nginx.conf``).
"""

import json
import os
import secrets
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from ..active_learning import ActiveLearningLoop, select_samples
from ..cache import Cache, make_cache
from ..database import (
    AuditRepository,
    JobRepository,
    MoleculeRepository,
    init_db,
    make_engine,
    make_session_factory,
    session_scope,
)
from ..database.models import Job, ModelVersion, User
from ..fair import record_from_candidate, to_jsonld
from ..featurization import is_valid_smiles
from ..generation import GenerationCondition
from ..md_simulation import validate_candidates
from ..nlp_interface import TargetParameters, parse_query
from ..pipeline import DesignPipeline
from ..security import AuthError, AuthService, PermissionDenied, authorize, decode_token
from ..training import ModelBundle
from .dashboard import render_dashboard
from .reports import render_report

MAX_BODY_BYTES = 2 * 1024 * 1024


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    totp_code: str = Field(default="", max_length=8)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=12, max_length=256)
    role: str = Field(default="viewer", pattern="^(admin|researcher|viewer)$")


class EnrollConfirmRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)
    code: str = Field(min_length=6, max_length=8)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class ParametersModel(BaseModel):
    scaffold_type: Optional[str] = Field(default=None, pattern="^(lipid|polymer|metal)$")
    target_tissue: Optional[str] = Field(default=None, max_length=32)
    size_range_nm: Optional[List[float]] = Field(default=None, min_length=2, max_length=2)
    max_toxicity_ic50: Optional[float] = Field(default=None, gt=0)
    min_loading_efficiency: Optional[float] = Field(default=None, ge=0, le=100)

    def to_target_parameters(self) -> TargetParameters:
        size = tuple(self.size_range_nm) if self.size_range_nm else None
        if size and not (0 <= size[0] < size[1] <= 1000):
            raise ValueError("size_range_nm must satisfy 0 ≤ min < max ≤ 1000")
        return TargetParameters(
            scaffold_type=self.scaffold_type,
            target_tissue=self.target_tissue,
            size_range_nm=size,
            max_toxicity_ic50=self.max_toxicity_ic50,
            min_loading_efficiency=self.min_loading_efficiency,
        )


class DesignRequest(BaseModel):
    query: Optional[str] = Field(default=None, max_length=1000)
    parameters: Optional[ParametersModel] = None
    n_generate: int = Field(default=1500, ge=50, le=20000)
    n_pareto: int = Field(default=15, ge=3, le=20)
    n_final: int = Field(default=5, ge=1, le=10)
    optimize_iterations: int = Field(default=120, ge=10, le=1000)
    run_md: bool = True
    explain: bool = True

    @model_validator(mode="after")
    def _need_input(self):
        # field_validator does not run for the default value None; the check must be at the model level
        if self.parameters is None and not self.query:
            raise ValueError("One of query or parameters is required")
        return self


class GenerateRequest(QueryRequest):
    n: int = Field(default=100, ge=1, le=5000)


class SmilesRequest(BaseModel):
    smiles: List[str] = Field(min_length=1, max_length=500)

    @field_validator("smiles")
    @classmethod
    def _bounded(cls, values):
        if any(len(s) > 2048 for s in values):
            raise ValueError("Each SMILES may be at most 2048 characters long")
        return values


class LabRowsRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(min_length=1, max_length=1000)


class RetrainRequest(BaseModel):
    force: bool = False


class ProposeRequest(BaseModel):
    pool_smiles: List[str] = Field(min_length=10, max_length=5000)
    n: int = Field(default=10, ge=1, le=50)
    strategy: str = Field(default="hybrid", pattern="^(hybrid|uncertainty|diverse_uncertainty|random)$")


@dataclass
class Principal:
    username: str
    role: str


@dataclass
class Settings:
    model_dir: Optional[str] = None
    database_url: Optional[str] = None
    max_workers: int = 2

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_dir=os.environ.get("IPIND_MODEL_DIR"),
            database_url=os.environ.get("IPIND_DATABASE_URL"),
            max_workers=int(os.environ.get("IPIND_MAX_WORKERS", "2")),
        )


def _json_safe(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _jsonable(value: Any) -> Any:
    """Convert numpy/tuple values to JSON-serializable types (the database JSON column uses plain json.dumps)."""
    import numpy as np

    def default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    return json.loads(json.dumps(value, default=default))


# ----------------------------------------------------------------------
def create_app(
    bundle: Optional[ModelBundle] = None,
    session_factory=None,
    settings: Optional[Settings] = None,
    encryption_key: Optional[str] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="IPIND² API", version="0.5.0", docs_url=None, redoc_url=None)

    if session_factory is None:
        engine = make_engine(settings.database_url)
        init_db(engine)
        session_factory = make_session_factory(engine)
    app.state.session_factory = session_factory
    app.state.bundle = bundle
    app.state.bundle_lock = threading.Lock()
    app.state.cache = make_cache()
    app.state.executor = ThreadPoolExecutor(max_workers=settings.max_workers)
    app.state.encryption_key = encryption_key
    app.state.metrics = {"requests_total": 0, "jobs_total": 0, "denied_total": 0}
    app.state.pipelines: Dict[int, DesignPipeline] = {}

    # --- Middleware --------------------------------------------------
    @app.middleware("http")
    async def secure(request: Request, call_next):
        app.state.metrics["requests_total"] += 1
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Request body is too large"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        if request.url.path != "/":
            response.headers["Cache-Control"] = "no-store"
        return response

    # --- Helpers -----------------------------------------------------------
    def get_bundle() -> ModelBundle:
        if app.state.bundle is None:
            if settings.model_dir and Path(settings.model_dir, "manifest.json").exists():
                with app.state.bundle_lock:
                    if app.state.bundle is None:
                        app.state.bundle = ModelBundle.load(settings.model_dir)
            else:
                raise HTTPException(503, "Model is not loaded (set IPIND_MODEL_DIR or train a model)")
        return app.state.bundle

    def pipeline() -> DesignPipeline:
        bundle_ = get_bundle()
        key = id(bundle_)
        if key not in app.state.pipelines:
            app.state.pipelines = {key: DesignPipeline(bundle_)}
        return app.state.pipelines[key]

    def audit(
        principal: Optional[Principal], action: str, resource: Optional[str] = None,
        status: str = "ok", detail: Optional[dict] = None, request: Optional[Request] = None,
    ) -> None:
        ip = request.client.host if request is not None and request.client else None
        with session_scope(session_factory) as s:
            AuditRepository(s).record(
                action, principal.username if principal else None, resource, status, detail, ip
            )

    def require(permission: str) -> Callable:
        """FastAPI dependency: reads the token, checks the permission and records the access."""

        def dependency(request: Request) -> Principal:
            header = request.headers.get("authorization", "")
            if not header.lower().startswith("bearer "):
                raise HTTPException(401, "No token provided")
            try:
                claims = decode_token(header[7:])
            except AuthError:
                raise HTTPException(401, "Token is invalid or expired")
            principal = Principal(claims["sub"], claims["role"])
            # The user may have been deactivated after the token was issued
            with session_scope(session_factory) as s:
                user = s.query(User).filter(User.username == principal.username).one_or_none()
                if user is None or not user.is_active:
                    raise HTTPException(401, "User account is not active")
                principal.role = user.role  # we read the role from the source of truth (DB), not from the token
            try:
                authorize(principal.role, permission)
            except PermissionDenied:
                app.state.metrics["denied_total"] += 1
                audit(principal, f"{request.method} {request.url.path}", status="denied",
                      detail={"permission": permission}, request=request)
                raise HTTPException(403, "Insufficient permission")
            audit(principal, f"{request.method} {request.url.path}", detail={"permission": permission}, request=request)
            return principal

        return dependency

    # --- Public -----------------------------------------------------------
    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        nonce = secrets.token_urlsafe(16)
        html = render_dashboard(nonce)
        csp = (
            f"default-src 'none'; script-src 'nonce-{nonce}'; style-src 'nonce-{nonce}'; "
            "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        return HTMLResponse(html, headers={"Content-Security-Policy": csp, "Cache-Control": "no-store"})

    @app.get("/health")
    def health() -> Dict[str, Any]:
        loaded = app.state.bundle is not None
        return {
            "status": "ok",
            "model_loaded": loaded,
            "model_version": app.state.bundle.version if loaded else None,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(principal: Principal = Depends(require("audit_read"))) -> str:
        return "".join(f"ipind2_{k} {v}\n" for k, v in app.state.metrics.items())

    # --- Authentication ---------------------------------------------------------
    @app.post("/auth/login")
    def login(body: LoginRequest, request: Request) -> Dict[str, Any]:
        error: Optional[AuthError] = None
        token = None
        with session_scope(session_factory) as s:
            try:
                token = AuthService(s, app.state.encryption_key).login(
                    body.username, body.password, body.totp_code,
                    ip_address=request.client.host if request.client else None,
                )
            except AuthError as exc:  # we catch inside the scope so the attempt counter and log are committed
                error = exc
        if error is not None:
            raise HTTPException(401, str(error))
        return {"access_token": token, "token_type": "bearer"}

    @app.post("/auth/users", status_code=201)
    def create_user(body: CreateUserRequest, principal: Principal = Depends(require("manage_users"))) -> Dict[str, Any]:
        try:
            with session_scope(session_factory) as s:
                _, uri = AuthService(s, app.state.encryption_key).create_user(body.username, body.password, body.role)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        return {"username": body.username, "role": body.role, "totp_provisioning_uri": uri}

    @app.post("/auth/enroll/confirm")
    def enroll_confirm(body: EnrollConfirmRequest) -> Dict[str, Any]:
        from ..security import verify_password

        ok = False
        with session_scope(session_factory) as s:
            user = s.query(User).filter(User.username == body.username).one_or_none()
            if user is not None and verify_password(body.password, user.password_hash):
                ok = AuthService(s, app.state.encryption_key).confirm_totp(body.username, body.code)
        if not ok:
            raise HTTPException(401, "Username, password or code is incorrect")
        return {"enrolled": True}

    @app.get("/audit")
    def audit_log(limit: int = Query(100, ge=1, le=1000), principal: Principal = Depends(require("audit_read"))):
        with session_scope(session_factory) as s:
            rows = AuditRepository(s).recent(limit)
            return [
                {"id": r.id, "timestamp": r.timestamp.isoformat(), "username": r.username, "action": r.action,
                 "status": r.status, "resource": r.resource, "ip": r.ip_address}
                for r in rows
            ]

    # --- Natural language / generation / prediction -------------------------------------
    @app.post("/nlp/parse")
    def nlp_parse(body: QueryRequest, principal: Principal = Depends(require("read"))) -> Dict[str, Any]:
        params = parse_query(body.query)
        return {**params.to_dict(), "complete": params.is_complete()}

    @app.post("/generate")
    def generate(body: GenerateRequest, principal: Principal = Depends(require("generate"))) -> Dict[str, Any]:
        bundle_ = get_bundle()
        condition = GenerationCondition.from_target_parameters(parse_query(body.query))
        structures, stats = bundle_.generator.generate(body.n, condition)
        return {"structures": [s.to_dict() for s in structures], "stats": stats.to_dict()}

    @app.post("/predict")
    def predict(body: SmilesRequest, principal: Principal = Depends(require("predict"))) -> Dict[str, Any]:
        bundle_ = get_bundle()
        valid = [s for s in body.smiles if is_valid_smiles(s)]
        results: Dict[str, Any] = {}
        if valid:
            pm, ps, pk = bundle_.physico.predict_with_uncertainty(valid)
            bm, bs, bk = bundle_.bio.predict_with_uncertainty(valid)
            for position, smiles in enumerate(valid):
                results[smiles] = {
                    "predictions": {**pm.loc[position].to_dict(), **bm.loc[position].to_dict()},
                    "uncertainty_std": {**ps.loc[position].to_dict(), **bs.loc[position].to_dict()},
                }
        return {
            "model_version": bundle_.version,
            "results": [
                {"smiles": s, **results[s]} if s in results else {"smiles": s, "error": "invalid_smiles"}
                for s in body.smiles
            ],
        }

    @app.post("/validate")
    def validate(body: SmilesRequest, principal: Principal = Depends(require("validate"))) -> Dict[str, Any]:
        if len(body.smiles) > 10:
            raise HTTPException(422, "At most 10 candidates per request")
        bad = [s for s in body.smiles if not is_valid_smiles(s)]
        if bad:
            raise HTTPException(422, f"Invalid SMILES: {bad[:3]}")
        report = validate_candidates(body.smiles, fallback_to_conformers=True)
        return {"md_complete": report.md_complete, "results": [r.to_dict() for r in report.results]}

    # --- Design (asynchronous) ------------------------------------------------
    def run_design_job(job_id: str, body: DesignRequest) -> None:
        try:
            request_value = (
                body.parameters.to_target_parameters() if body.parameters is not None else body.query
            )
            result = pipeline().design(
                request_value,
                n_generate=body.n_generate,
                n_pareto=max(body.n_pareto, body.n_final),
                n_final=body.n_final,
                optimize_iterations=body.optimize_iterations,
                run_md=body.run_md,
                explain=body.explain,
            )
            payload = result.to_dict()
            payload["pareto_candidates"] = _json_safe(result.pareto_candidates)
            payload["model_version"] = get_bundle().version
            payload = _jsonable(payload)
            outcome: Dict[str, Any] = {"status": "done", "result": payload, "error": None}
        except Exception as exc:  # a job error must not kill the thread; it is recorded in the jobs table
            outcome = {"status": "failed", "result": None, "error": f"{type(exc).__name__}: {exc}"}
        with session_scope(session_factory) as s:
            job = JobRepository(s).get(job_id)
            job.status, job.result, job.error = outcome["status"], outcome["result"], outcome["error"]
            job.finished_at = datetime.now(timezone.utc)

    @app.post("/design", status_code=202)
    def design(body: DesignRequest, principal: Principal = Depends(require("optimize"))) -> Dict[str, Any]:
        get_bundle()  # fast 503 if there is no model
        if body.parameters is not None:
            try:
                body.parameters.to_target_parameters()
            except ValueError as exc:
                raise HTTPException(422, str(exc))
        job_id = str(uuid.uuid4())
        with session_scope(session_factory) as s:
            JobRepository(s).create(job_id, "design", principal.username, body.model_dump())
        app.state.metrics["jobs_total"] += 1
        app.state.executor.submit(run_design_job, job_id, body)
        return {"job_id": job_id, "status": "queued"}

    def load_job(job_id: str, principal: Principal) -> Job:
        with session_scope(session_factory) as s:
            job = JobRepository(s).get(job_id)
            if job is None or (job.owner != principal.username and principal.role != "admin"):
                raise HTTPException(404, "Job not found")  # we do not reveal the existence/non-existence of other users' jobs
            s.expunge(job)
            return job

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str, principal: Principal = Depends(require("read"))) -> Dict[str, Any]:
        job = load_job(job_id, principal)
        return {"job_id": job.id, "status": job.status, "result": job.result, "error": job.error}

    @app.get("/reports/{job_id}", response_class=HTMLResponse)
    def report(job_id: str, principal: Principal = Depends(require("read"))) -> HTMLResponse:
        job = load_job(job_id, principal)
        if job.status != "done" or not job.result:
            raise HTTPException(409, "Job is not finished yet")
        return HTMLResponse(render_report(job.result), headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"})

    @app.get("/export/{job_id}")
    def export(job_id: str, fmt: str = Query("jsonld", pattern="^(jsonld|csv)$"),
               principal: Principal = Depends(require("read"))) -> Response:
        from ..fair import export_records

        job = load_job(job_id, principal)
        if job.status != "done" or not job.result:
            raise HTTPException(409, "Job is not finished yet")
        version = job.result.get("model_version", "unknown")
        scaffold = (job.result.get("parameters") or {}).get("scaffold_type")
        records = [record_from_candidate(c, version, scaffold) for c in job.result["final_candidates"]]
        with tempfile.TemporaryDirectory() as directory:
            path = export_records(records, str(Path(directory) / f"out.{fmt}"), fmt)
            content = path.read_bytes()
        media = "application/ld+json" if fmt == "jsonld" else "text/csv"
        return Response(content, media_type=media)

    # --- Lab + active learning ------------------------------------------
    @app.post("/lab/results")
    def lab_results(body: LabRowsRequest, principal: Principal = Depends(require("lab_ingest"))) -> Dict[str, Any]:
        from ..lab_automation.schema import ExperimentalResult

        rows: List[Dict[str, Any]] = []
        rejected: List[str] = []
        with session_scope(session_factory) as s:
            repo = MoleculeRepository(s)
            for position, raw in enumerate(body.rows):
                data = dict(raw)
                smiles = data.pop("smiles", None)
                try:
                    if smiles:
                        molecule = repo.add_molecule(str(smiles)[:2048])
                        if molecule is None:
                            raise ValueError("Invalid SMILES")
                        data["molecule_id"] = molecule.id
                    rows.append(ExperimentalResult.from_dict(data).to_dict())
                except (ValueError, TypeError) as exc:
                    rejected.append(f"Row {position + 1}: {exc}")
            accepted = repo.add_experimental_results(pd.DataFrame(rows)) if rows else 0
            pending = len(repo.unconsumed_results())
        return {"accepted": accepted, "rejected": rejected, "pending": pending}

    @app.post("/active-learning/propose")
    def al_propose(body: ProposeRequest, principal: Principal = Depends(require("active_learning"))) -> Dict[str, Any]:
        bundle_ = get_bundle()
        valid = [s for s in body.pool_smiles if is_valid_smiles(s)]
        if len(valid) < body.n:
            raise HTTPException(422, "The number of valid SMILES is less than the number requested")
        indices = select_samples(valid, [bundle_.physico, bundle_.bio], body.n, body.strategy)
        return {"strategy": body.strategy, "selected": [valid[i] for i in indices]}

    @app.post("/active-learning/retrain")
    def al_retrain(body: RetrainRequest, principal: Principal = Depends(require("active_learning"))) -> Dict[str, Any]:
        from ..data_generation.synthetic_data_generator import SyntheticDataGenerator

        bundle_ = get_bundle()
        with session_scope(session_factory) as s:
            repo = MoleculeRepository(s)
            pending = repo.unconsumed_results()
            lookup = repo.smiles_by_id()
        if pending.empty:
            return {"retrained": False, "message": "There are no new lab results"}

        replay = SyntheticDataGenerator(0).generate_dataset(300, include_pareto_labels=False)
        loop = ActiveLearningLoop(
            {"physico": bundle_.physico, "bio": bundle_.bio}, lookup, replay=replay
        )
        loop.ingest(pending.drop(columns=["_row_id"]))
        with app.state.bundle_lock:
            report_ = loop.retrain(force=body.force)
        if report_ is None:
            return {
                "retrained": False,
                "message": f"{loop.pending} results pending; at least {loop.min_batch} required (or force=true)",
            }
        with session_scope(session_factory) as s:
            MoleculeRepository(s).mark_consumed([int(i) for i in pending["_row_id"]])
            bundle_.version = f"{bundle_.version.split('+')[0]}+al{len(loop.reports)}"
            s.add(ModelVersion(name="bundle", version=bundle_.version, metrics={"n_new": report_.n_new}, trained_on="active-learning"))
        return {
            "retrained": True,
            "message": f"Model updated with {report_.n_new} new results (replay: {report_.n_replay} synthetic samples)",
            "version": bundle_.version,
        }

    return app
