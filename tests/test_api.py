"""تست‌های API (FR-07، FR-10، FR-11، SEC-01/04/05) با مدل‌های smoke و پایگاه داده حافظه‌ای."""

import copy
import json
import time

import pyotp
import pytest
from fastapi.testclient import TestClient

from ipind2.api import create_app
from ipind2.database import init_db, make_engine, make_session_factory, session_scope
from ipind2.security import AuthService

PASSWORD = "Str0ngPassword!!"


def _make_user(factory, key, username, role):
    with session_scope(factory) as s:
        svc = AuthService(s, key)
        _, uri = svc.create_user(username, PASSWORD, role)
        secret = pyotp.parse_uri(uri).secret
        svc.confirm_totp(username, pyotp.TOTP(secret).now())
    return secret


@pytest.fixture()
def env(smoke_bundle, encryption_key):
    engine = make_engine("sqlite://")
    init_db(engine)
    factory = make_session_factory(engine)
    secrets = {name: _make_user(factory, encryption_key, name, role)
               for name, role in (("admin1", "admin"), ("res1", "researcher"), ("res2", "researcher"), ("view1", "viewer"))}
    app = create_app(bundle=copy.deepcopy(smoke_bundle), session_factory=factory, encryption_key=encryption_key)
    client = TestClient(app)
    return client, secrets, factory


def _login(client, secrets, username):
    response = client.post(
        "/auth/login",
        json={"username": username, "password": PASSWORD, "totp_code": pyotp.TOTP(secrets[username]).now()},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _wait_for_job(client, headers, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}", headers=headers).json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(1.0)
    raise AssertionError("کار در زمان مجاز تمام نشد")


class TestPublicSurface:
    def test_health_is_public_and_reports_model(self, env):
        client, _, _ = env
        body = client.get("/health").json()
        assert body["status"] == "ok" and body["model_loaded"] is True

    def test_dashboard_is_self_contained_with_nonce_csp(self, env):
        client, _, _ = env
        response = client.get("/")
        assert response.status_code == 200
        csp = response.headers["content-security-policy"]
        nonce = csp.split("script-src 'nonce-")[1].split("'")[0]
        assert f'nonce="{nonce}"' in response.text
        assert "http://" not in response.text and "https://" not in response.text, "منبع خارجی نباید باشد"
        assert "innerHTML" not in response.text, "درج متن سرور فقط با textContent"
        assert "unsafe-inline" not in csp and "unsafe-eval" not in csp

    def test_nonce_changes_per_request(self, env):
        client, _, _ = env
        assert client.get("/").headers["content-security-policy"] != client.get("/").headers["content-security-policy"]

    def test_security_headers_present(self, env):
        client, _, _ = env
        headers = client.get("/health").headers
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["x-frame-options"] == "DENY"
        assert headers["cache-control"] == "no-store"

    @pytest.mark.parametrize(
        "method,path",
        [("post", "/nlp/parse"), ("post", "/generate"), ("post", "/predict"), ("post", "/design"),
         ("get", "/jobs/x"), ("post", "/lab/results"), ("get", "/audit"), ("get", "/metrics")],
    )
    def test_protected_endpoints_require_token(self, env, method, path):
        client, _, _ = env
        assert getattr(client, method)(path).status_code in (401, 422)
        if method == "get":
            assert client.get(path).status_code == 401

    def test_garbage_token_rejected(self, env):
        client, _, _ = env
        assert client.get("/audit", headers={"Authorization": "Bearer not.a.jwt"}).status_code == 401


class TestAuthFlow:
    def test_login_wrong_code_is_401_and_generic(self, env):
        client, _, _ = env
        response = client.post("/auth/login", json={"username": "res1", "password": PASSWORD, "totp_code": "000000"})
        unknown = client.post("/auth/login", json={"username": "ghost", "password": PASSWORD, "totp_code": "000000"})
        assert response.status_code == unknown.status_code == 401
        assert response.json() == unknown.json()

    def test_failed_logins_persist_and_lock_account(self, env):
        client, secrets, _ = env
        for _ in range(5):
            client.post("/auth/login", json={"username": "res2", "password": "WrongPassword123", "totp_code": "000000"})
        locked = client.post(
            "/auth/login",
            json={"username": "res2", "password": PASSWORD, "totp_code": pyotp.TOTP(secrets["res2"]).now()},
        )
        assert locked.status_code == 401, "قفل باید با وجود رول‌بک exception ماندگار باشد"

    def test_only_admin_creates_users(self, env):
        client, secrets, _ = env
        payload = {"username": "newbie", "password": PASSWORD, "role": "viewer"}
        assert client.post("/auth/users", json=payload, headers=_login(client, secrets, "res1")).status_code == 403
        created = client.post("/auth/users", json=payload, headers=_login(client, secrets, "admin1"))
        assert created.status_code == 201 and created.json()["totp_provisioning_uri"].startswith("otpauth://")

    def test_new_user_cannot_login_before_enrollment(self, env):
        client, secrets, _ = env
        admin = _login(client, secrets, "admin1")
        created = client.post("/auth/users", json={"username": "fresh", "password": PASSWORD, "role": "viewer"}, headers=admin)
        secret = pyotp.parse_uri(created.json()["totp_provisioning_uri"]).secret
        attempt = {"username": "fresh", "password": PASSWORD, "totp_code": pyotp.TOTP(secret).now()}
        assert client.post("/auth/login", json=attempt).status_code == 401
        confirm = client.post("/auth/enroll/confirm", json={"username": "fresh", "password": PASSWORD, "code": pyotp.TOTP(secret).now()})
        assert confirm.status_code == 200
        assert client.post("/auth/login", json=attempt).status_code == 200

    def test_weak_password_rejected_on_create(self, env):
        client, secrets, _ = env
        response = client.post("/auth/users", json={"username": "weak", "password": "alllowercase1234", "role": "viewer"},
                               headers=_login(client, secrets, "admin1"))
        assert response.status_code == 422

    def test_deactivated_user_token_stops_working(self, env):
        client, secrets, factory = env
        headers = _login(client, secrets, "view1")
        assert client.post("/nlp/parse", json={"query": "لیپیدی"}, headers=headers).status_code == 200
        from ipind2.database.models import User

        with session_scope(factory) as s:
            s.query(User).filter_by(username="view1").one().is_active = False
        assert client.post("/nlp/parse", json={"query": "لیپیدی"}, headers=headers).status_code == 401

    def test_role_comes_from_database_not_token(self, env):
        client, secrets, factory = env
        headers = _login(client, secrets, "res1")
        from ipind2.database.models import User

        with session_scope(factory) as s:
            s.query(User).filter_by(username="res1").one().role = "viewer"
        assert client.post("/design", json={"query": "لیپیدی"}, headers=headers).status_code == 403


class TestRBACAndAudit:
    def test_viewer_can_read_but_not_design(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "view1")
        assert client.post("/nlp/parse", json={"query": "یک نانوحامل لیپیدی"}, headers=headers).status_code == 200
        assert client.post("/design", json={"query": "لیپیدی"}, headers=headers).status_code == 403
        assert client.post("/predict", json={"smiles": ["CCO"]}, headers=headers).status_code == 403
        assert client.post("/lab/results", json={"rows": [{"molecule_id": 1}]}, headers=headers).status_code == 403

    def test_audit_is_admin_only_and_records_denials(self, env):
        client, secrets, _ = env
        viewer = _login(client, secrets, "view1")
        client.post("/design", json={"query": "لیپیدی"}, headers=viewer)
        assert client.get("/audit", headers=viewer).status_code == 403
        entries = client.get("/audit?limit=500", headers=_login(client, secrets, "admin1")).json()
        denied = [e for e in entries if e["username"] == "view1" and e["status"] == "denied"]
        assert denied and any("/design" in e["action"] for e in denied)


class TestInputValidation:
    def test_oversized_body_rejected(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        huge = "x" * (3 * 1024 * 1024)
        assert client.post("/nlp/parse", content=huge, headers={**headers, "Content-Type": "application/json"}).status_code == 413

    def test_bounds_enforced(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        assert client.post("/design", json={"query": "q", "n_generate": 10**7}, headers=headers).status_code == 422
        assert client.post("/design", json={"n_generate": 100}, headers=headers).status_code == 422  # بدون query/parameters
        assert client.post("/predict", json={"smiles": []}, headers=headers).status_code == 422
        assert client.post("/predict", json={"smiles": ["C" * 5000]}, headers=headers).status_code == 422
        assert client.post("/nlp/parse", json={"query": "x" * 5000}, headers=headers).status_code == 422

    def test_invalid_size_range_rejected(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        body = {"parameters": {"size_range_nm": [500, 100]}}
        assert client.post("/design", json=body, headers=headers).status_code == 422


class TestModelEndpoints:
    def test_nlp_parse_persian(self, env):
        client, secrets, _ = env
        body = client.post("/nlp/parse", json={"query": "نانوحامل لیپیدی برای تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر"},
                           headers=_login(client, secrets, "res1")).json()
        assert body["scaffold_type"] == "lipid" and body["target_tissue"] == "tumor"
        assert body["size_range_nm"] == [80.0, 120.0] and body["complete"] is True

    def test_predict_valid_and_invalid_smiles(self, env):
        client, secrets, _ = env
        response = client.post("/predict", json={"smiles": ["CCCCCCCCCCCCCCCC[N+](C)(C)C", "not-a-smiles(((", "OCCOCCOCCO"]},
                               headers=_login(client, secrets, "res1")).json()
        rows = response["results"]
        assert [r["smiles"] for r in rows] == ["CCCCCCCCCCCCCCCC[N+](C)(C)C", "not-a-smiles(((", "OCCOCCOCCO"]
        assert rows[1]["error"] == "invalid_smiles"
        assert rows[0]["predictions"]["phys_size_nm"] > 0 and "bio_cytotoxicity_ic50_ug_ml" in rows[0]["predictions"]
        assert rows[0]["uncertainty_std"]["phys_size_nm"] >= 0

    def test_generate_returns_valid_structures(self, env):
        from ipind2.featurization import is_valid_smiles

        client, secrets, _ = env
        body = client.post("/generate", json={"query": "نانوحامل لیپیدی", "n": 20}, headers=_login(client, secrets, "res1")).json()
        assert body["structures"] and all(is_valid_smiles(s["smiles"]) for s in body["structures"])
        assert body["stats"]["validity_rate"] == 1.0

    def test_validate_reports_fidelity_honestly(self, env):
        client, secrets, _ = env
        body = client.post("/validate", json={"smiles": ["CCCCCCCCCCCC[N+](C)(C)C"]}, headers=_login(client, secrets, "res1")).json()
        assert body["md_complete"] is False
        assert body["results"][0]["fidelity"] == "conformer_ensemble" and body["results"][0]["is_real_md"] is False

    def test_validate_rejects_bad_input(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        assert client.post("/validate", json={"smiles": ["bad((("]}, headers=headers).status_code == 422
        assert client.post("/validate", json={"smiles": ["CCO"] * 11}, headers=headers).status_code == 422


class TestDesignJob:
    def _run(self, client, headers, **overrides):
        body = {"query": "نانوحامل لیپیدی برای تومور، اندازه بین ۸۰ تا ۱۲۰ نانومتر", "n_generate": 100, "n_pareto": 6,
                "n_final": 2, "optimize_iterations": 10, "run_md": False, "explain": False}
        body.update(overrides)
        submitted = client.post("/design", json=body, headers=headers)
        assert submitted.status_code == 202, submitted.text
        return submitted.json()["job_id"]

    def test_full_job_report_and_export(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        job_id = self._run(client, headers)
        job = _wait_for_job(client, headers, job_id)
        assert job["status"] == "done", job.get("error")
        result = job["result"]
        assert 1 <= len(result["final_candidates"]) <= 2
        candidate = result["final_candidates"][0]
        assert candidate["predictions"]["phys_size_nm"] > 0 and 0 <= candidate["overall_confidence"] <= 1
        assert result["md_complete"] is False

        report = client.get(f"/reports/{job_id}", headers=headers)
        assert report.status_code == 200 and candidate["smiles"].split("(")[0] in report.text

        graph = json.loads(client.get(f"/export/{job_id}?fmt=jsonld", headers=headers).content)["@graph"]
        assert len(graph) == len(result["final_candidates"])
        provenance = {o["prov:wasGeneratedBy"]["ipind:provenance"] for node in graph for o in node["ipind:material"]}
        assert provenance == {"predicted"}, "مقادیر پیش‌بینی هرگز نباید measured برچسب بخورند"
        csv_text = client.get(f"/export/{job_id}?fmt=csv", headers=headers).text
        assert csv_text.startswith("identifier,category,field")

    def test_jobs_are_isolated_between_users_but_visible_to_admin(self, env):
        client, secrets, _ = env
        owner = _login(client, secrets, "res1")
        job_id = self._run(client, owner)
        _wait_for_job(client, owner, job_id)
        assert client.get(f"/jobs/{job_id}", headers=_login(client, secrets, "res2")).status_code == 404
        assert client.get(f"/reports/{job_id}", headers=_login(client, secrets, "res2")).status_code == 404
        assert client.get(f"/jobs/{job_id}", headers=_login(client, secrets, "admin1")).status_code == 200

    def test_report_escapes_malicious_query(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        payload = "نانوحامل لیپیدی <script>alert(1)</script>"
        job_id = self._run(client, headers, query=payload)
        _wait_for_job(client, headers, job_id)
        html = client.get(f"/reports/{job_id}", headers=headers).text
        assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html

    def test_unknown_job_is_404(self, env):
        client, secrets, _ = env
        assert client.get("/jobs/00000000-0000-0000-0000-000000000000", headers=_login(client, secrets, "res1")).status_code == 404


class TestLabFeedbackLoop:
    ROWS = [
        {"smiles": "CCCCCCCCCCCCCCCC[N+](C)(C)C", "experimental_size_nm": "101.5", "experimental_zeta_potential": "28.0",
         "experimental_loading_efficiency": "66", "experimental_date": "2026-05-01", "lab_technician": "A"},
        {"smiles": "CCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCC)COP(=O)(O)OCCN", "experimental_size_nm": "112",
         "experimental_zeta_potential": "-2.5", "experimental_cytotoxicity": "75"},
        {"smiles": "OCCOCCOCCOCCOCCS[Au]", "experimental_size_nm": "35", "experimental_zeta_potential": "-9"},
        {"smiles": "CCCCCCCCCCCCN(CCCCCCCCCCCC)CCO", "experimental_size_nm": "98", "experimental_loading_efficiency": "61"},
        {"smiles": "CCCCCCCCCCCCCCCCCCOC(=O)CCC(=O)OCCCCCCCCCCCCCCCCCC", "experimental_size_nm": "120"},
    ]

    def test_ingest_rejects_bad_rows_but_keeps_good(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        rows = self.ROWS + [{"smiles": "garbage((("}, {"experimental_size_nm": "1"}]
        body = client.post("/lab/results", json={"rows": rows}, headers=headers).json()
        assert body["accepted"] == len(self.ROWS) and body["pending"] == len(self.ROWS)
        assert len(body["rejected"]) == 2

    def test_retrain_waits_for_threshold_then_updates_and_consumes(self, env):
        client, secrets, _ = env
        headers = _login(client, secrets, "res1")
        before = client.get("/health").json()["model_version"]
        client.post("/lab/results", json={"rows": self.ROWS}, headers=headers)

        waiting = client.post("/active-learning/retrain", json={"force": False}, headers=headers).json()
        assert waiting["retrained"] is False, "کمتر از ۱۰ نتیجه ⇒ بدون به‌روزرسانی (SRS §4.6)"

        done = client.post("/active-learning/retrain", json={"force": True}, headers=headers).json()
        assert done["retrained"] is True and done["version"] == before + "+al1"
        assert client.get("/health").json()["model_version"] == done["version"]

        again = client.post("/active-learning/retrain", json={"force": True}, headers=headers).json()
        assert again["retrained"] is False, "نتایج مصرف‌شده نباید دوباره استفاده شوند"

    def test_propose_returns_valid_unique_subset(self, env, small_dataset):
        client, secrets, _ = env
        pool = small_dataset.smiles.tolist()[:60]
        body = client.post("/active-learning/propose", json={"pool_smiles": pool, "n": 8, "strategy": "hybrid"},
                           headers=_login(client, secrets, "res1")).json()
        assert len(body["selected"]) == len(set(body["selected"])) == 8 and set(body["selected"]) <= set(pool)
