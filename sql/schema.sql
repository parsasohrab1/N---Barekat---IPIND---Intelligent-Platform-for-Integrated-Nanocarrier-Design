-- IPIND² database schema
-- See docs/SRS.md section 5.2 for context.

-- Molecules table
CREATE TABLE molecules (
    id SERIAL PRIMARY KEY,
    smiles TEXT NOT NULL,
    canonical_smiles VARCHAR(2048) UNIQUE, -- deduplication key (RDKit)
    inchikey VARCHAR(27),
    molecular_weight FLOAT,
    logP FLOAT,
    tpsa FLOAT,
    num_rotatable_bonds INTEGER,
    num_h_donors INTEGER,
    num_h_acceptors INTEGER,
    scaffold_type VARCHAR(50), -- 'lipid' | 'polymer' | 'metal'
    created_at TIMESTAMP DEFAULT NOW()
);

-- Physicochemical properties table
CREATE TABLE physicochemical_properties (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES molecules(id),
    size_nm FLOAT,
    zeta_potential_mV FLOAT,
    pdi FLOAT,
    colloid_stability_hours FLOAT,
    drug_loading_efficiency FLOAT,
    drug_loading_content FLOAT,
    release_rate_constant FLOAT,
    prediction_confidence FLOAT,
    model_version VARCHAR(64)
);

-- Biological properties table
CREATE TABLE biological_properties (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES molecules(id),
    cell_line VARCHAR(50),
    cytotoxicity_ic50 FLOAT,
    cellular_uptake_efficiency FLOAT,
    serum_protein_binding FLOAT,
    circulation_half_life FLOAT,
    tumor_to_background_ratio FLOAT,
    model_version VARCHAR(64)
);

-- Lab results table (for feedback)
CREATE TABLE experimental_results (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES molecules(id),
    experimental_size_nm FLOAT,
    experimental_zeta_potential FLOAT,
    experimental_loading_efficiency FLOAT,
    experimental_cytotoxicity FLOAT,
    experimental_date DATE,
    lab_technician VARCHAR(100),
    consumed_by_training BOOLEAN DEFAULT FALSE -- consumed in an active-learning round
);

CREATE INDEX idx_molecules_inchikey ON molecules (inchikey);
CREATE INDEX idx_molecules_scaffold ON molecules (scaffold_type);
CREATE INDEX idx_physico_molecule ON physicochemical_properties (molecule_id);
CREATE INDEX idx_bio_molecule ON biological_properties (molecule_id);
CREATE INDEX idx_exp_molecule ON experimental_results (molecule_id);

-- Users and security (SEC-01, SEC-05)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(64) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,             -- bcrypt
    role VARCHAR(16) NOT NULL DEFAULT 'viewer',      -- admin | researcher | viewer
    totp_secret_encrypted TEXT,                      -- AES-256-GCM
    totp_enabled BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    failed_logins INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- User activity log (SEC-04)
CREATE TABLE audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    username VARCHAR(64),
    action VARCHAR(64) NOT NULL,
    resource VARCHAR(255),
    status VARCHAR(16) DEFAULT 'ok',
    detail JSON,
    ip_address VARCHAR(64)
);
CREATE INDEX idx_audit_ts ON audit_log (timestamp);
CREATE INDEX idx_audit_user ON audit_log (username);
CREATE INDEX idx_audit_action ON audit_log (action);

-- Asynchronous design jobs
CREATE TABLE jobs (
    id VARCHAR(36) PRIMARY KEY,
    kind VARCHAR(32) NOT NULL,
    status VARCHAR(16) DEFAULT 'queued',
    owner VARCHAR(64),
    parameters JSON,
    result JSON,
    error TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    finished_at TIMESTAMP
);
CREATE INDEX idx_jobs_status ON jobs (status);

-- Model version registry and benchmark (FR-12)
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    version VARCHAR(64) NOT NULL,
    path VARCHAR(512),
    metrics JSON,
    trained_on VARCHAR(128),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE (name, version)
);

CREATE TABLE benchmark_runs (
    id SERIAL PRIMARY KEY,
    dataset VARCHAR(64) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    n_samples INTEGER NOT NULL,
    rmse FLOAT NOT NULL,
    r2 FLOAT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW()
);
