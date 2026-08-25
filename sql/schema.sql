-- IPIND² database schema
-- See docs/SRS.md section 5.2 for context.

-- جدول مولکول‌ها
CREATE TABLE molecules (
    id SERIAL PRIMARY KEY,
    smiles TEXT NOT NULL,
    molecular_weight FLOAT,
    logP FLOAT,
    tpsa FLOAT,
    num_rotatable_bonds INTEGER,
    num_h_donors INTEGER,
    num_h_acceptors INTEGER,
    scaffold_type VARCHAR(50), -- 'lipid' | 'polymer' | 'metal'
    created_at TIMESTAMP DEFAULT NOW()
);

-- جدول ویژگی‌های فیزیکوشیمیایی
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
    prediction_confidence FLOAT
);

-- جدول ویژگی‌های زیستی
CREATE TABLE biological_properties (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES molecules(id),
    cell_line VARCHAR(50),
    cytotoxicity_ic50 FLOAT,
    cellular_uptake_efficiency FLOAT,
    serum_protein_binding FLOAT,
    circulation_half_life FLOAT,
    tumor_to_background_ratio FLOAT
);

-- جدول نتایج آزمایشگاهی (برای بازخورد)
CREATE TABLE experimental_results (
    id SERIAL PRIMARY KEY,
    molecule_id INTEGER REFERENCES molecules(id),
    experimental_size_nm FLOAT,
    experimental_zeta_potential FLOAT,
    experimental_loading_efficiency FLOAT,
    experimental_cytotoxicity FLOAT,
    experimental_date DATE,
    lab_technician VARCHAR(100)
);
