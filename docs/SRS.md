# Software Requirements Specification (SRS)
## Integrated Platform for Intelligent Design of Drug Nanocarriers (IPIND² - Intelligent Platform for Integrated Nanocarrier Design)

## 1. Introduction

### 1.1 Purpose
This document specifies the complete software requirements for the "Integrated Platform for Intelligent Design of Drug Nanocarriers". By combining deep generative models, graph neural networks, multi-objective reinforcement learning and molecular dynamics simulation, the platform reduces the nanocarrier design process from 3-5 years to 6-12 months.

### 1.2 Scope
The platform covers everything from generating a virtual structure library to property prediction, multi-objective optimization, simulation validation, and active learning with lab feedback.

### 1.3 Definitions and Acronyms

| Abbreviation | Description |
|---|---|
| VAE | Variational Autoencoder |
| GAN | Generative Adversarial Network |
| GNN | Graph Neural Network |
| RL | Reinforcement Learning |
| MD | Molecular Dynamics |
| LNP | Lipid Nanoparticle |
| PLGA | Poly(Lactic-co-Glycolic Acid) |
| SMILES | Simplified Molecular Input Line Entry System |

## 2. General System Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority | Description |
|---|---|---|---|
| FR-01 | Virtual library generation | High | Generate at least 100,000 nanocarrier structures per run using conditional VAE/GAN |
| FR-02 | Physicochemical property prediction | High | Simultaneous prediction of at least 7 properties with a multi-task GNN |
| FR-03 | Biological property prediction | High | Prediction of loading efficiency, release kinetics, toxicity, cellular uptake |
| FR-04 | Multi-objective optimization | High | Simultaneous optimization of conflicting objectives with Pareto-Guided RL |
| FR-05 | Simulation and validation | Medium | MD simulation for final confirmation of candidates |
| FR-06 | Active learning and feedback | High | Continuous model updates with lab data |
| FR-07 | User interface | Medium | Interactive dashboard for data entry and results display |
| FR-08 | Data management | High | Integrated database for storing structures, properties and results |
| FR-09 | Prediction interpretability | High | Attention layers + SHAP/LIME analysis for every prediction; confidence score and human-readable explanation for each output property |
| FR-10 | Natural-language query interface | Medium | Define target parameters in natural/structured language, complementing the form-based dashboard |
| FR-11 | Automated lab integration (Lab-in-the-loop) | Medium | API for connecting to robotic synthesis/high-throughput screening equipment, turning the feedback loop from manual data entry into automatic closed-loop feedback |
| FR-12 | Continuous internal benchmark | High | Automatic measurement of model accuracy against public reference datasets (e.g., the 622-sample LNP set, LANCE) at every model release, to prevent hidden quality degradation |
| FR-13 | Data standardization and interoperability | Medium | Data input/output per FAIR principles and the emerging MIRIBEL standard for nanoparticles, for interoperability with partners and regulators |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target value |
|---|---|---|
| NFR-01 | Size prediction accuracy | RMSE < 5 nm |
| NFR-02 | Surface charge prediction accuracy | RMSE < 2 mV |
| NFR-03 | Loading efficiency prediction accuracy | R² > 0.85 |
| NFR-04 | Library generation time | < 10 minutes for 100,000 structures |
| NFR-09 | Binding energy computation accuracy in advanced mode (quasi-quantum level, optional/complementary to MM-GBSA) | Error < 1 kcal/mol |
| NFR-10 | Initial structural library (seed library) size for warming up the generative model | At least 1 million reference structures from public databases, before synthetic generation |
| NFR-05 | Prediction time per structure | < 100 milliseconds |
| NFR-06 | Optimization time | < 1 hour for 1000 candidates |
| NFR-07 | System availability | 99.9% |
| NFR-08 | Scalability | Support for at least 1 million structures |

## 3. System Architecture

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    User Interface Layer (UI Layer)              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Dashboard   │  │ Reporter    │  │ Parameter settings       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Data Management Layer (Data Layer)          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │ Database    │  │ Cache (Redis)│ │ Cloud storage            │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Processing Layer                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Unit 1: Structure generation (Conditional VAE/GAN)       ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Unit 2: Physicochemical prediction (Multi-Task GNN)      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Unit 3: Biological prediction (Multi-Task Transformer/GNN)││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Unit 4: Multi-objective optimization (Pareto-Guided RL)  ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Unit 5: MD simulation (coupled with GROMACS/OpenMM)      ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │  Unit 6: Active learning (Uncertainty-Aware Sampling)     ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow

1. **Input**: The user enters target parameters (nanocarrier type, size range, target tissue, toxicity constraints)
2. **Generation**: The conditional VAE/GAN model generates 100,000+ new structures
3. **Prediction**: The multi-task GNN predicts 7+ properties for each structure
4. **Optimization**: The Pareto-Guided RL algorithm selects the optimal candidates
5. **Validation**: MD simulation is run for the top 10 candidates
6. **Output**: 3-5 final candidates are presented to the user
7. **Feedback**: Lab results are added to the database and the models are updated

## 4. Detailed Requirements for Each Unit

### 4.1 Molecular Structure Generation Unit (FR-01)

| Property | Value |
|---|---|
| Model type | Conditional VAE + Conditional GAN (ensemble) |
| Input | Target properties (size, charge, carrier type, target tissue) |
| Output | SMILES + molecular graph structure |
| Generated per run | ≥ 100,000 |
| Structural diversity | Coverage of at least 500 distinct molecular scaffolds |
| Valid structure rate | > 95% (validated with RDKit) |

Reference implementation: [`src/ipind2/generation`](../src/ipind2/generation)

### 4.2 Physicochemical Property Prediction Unit (FR-02)

| Property | Value |
|---|---|
| Model type | Multi-Task Graph Neural Network (MPNN + Attention) |
| Predicted properties | 1. Hydrodynamic size (nm); 2. Zeta potential (mV); 3. Polydispersity index (PDI); 4. Colloidal stability (aggregation half-life, hours); 5. Drug encapsulation efficiency (%); 6. Drug loading content (wt%); 7. Release kinetics (rate constant, k) |
| Target accuracy | RMSE < 5 nm for size, < 2 mV for zeta |

Reference implementation: [`src/ipind2/physicochemical`](../src/ipind2/physicochemical)

### 4.3 Biological Property Prediction Unit (FR-03)

| Property | Value |
|---|---|
| Model type | Multi-Task Transformer + GNN |
| Predicted properties | 1. Cytotoxicity (IC50, μg/mL) on ≥ 3 cell lines; 2. Cellular uptake efficiency (%); 3. Interaction with serum proteins (% binding); 4. Circulation half-life (hours); 5. Tumor-to-background ratio (TBR) |
| Target accuracy | R² > 0.85 for all properties |

Reference implementation: [`src/ipind2/biological`](../src/ipind2/biological)

### 4.4 Multi-Objective Optimization Unit (FR-04)

| Property | Value |
|---|---|
| Algorithm type | Pareto-Guided Reinforcement Learning (PG-RL) |
| Objective functions | Maximize loading efficiency and cellular uptake; minimize toxicity and size (within the desired range); maximize stability |
| Number of output candidates | 10-20 candidates on the Pareto front |
| Convergence criterion | Change < 1% over 100 iterations |

Reference implementation: [`src/ipind2/optimization`](../src/ipind2/optimization)

### 4.5 Molecular Dynamics Simulation Unit (FR-05)

| Property | Value |
|---|---|
| Simulation engine | GROMACS / OpenMM (selectable) |
| Force field | CHARMM36 / OPLS-AA |
| Simulation time | ≥ 100 ns per candidate |
| Extracted properties | 1. Binding free energy (MM-GBSA); 2. Radius of gyration (Rg); 3. Solvent-accessible surface area (SASA); 4. Order parameter |
| Number of simulated candidates | Top 5-10 candidates from optimization |

Reference implementation: [`src/ipind2/md_simulation`](../src/ipind2/md_simulation)

### 4.6 Active Learning and Feedback Unit (FR-06)

| Property | Value |
|---|---|
| Sampling strategy | Uncertainty-Aware Sampling + Query-by-Committee |
| Uncertainty metric | Prediction variance across ensemble models |
| Update frequency | After every 10-50 new lab data points |
| Update method | Fine-tuning with Early Stopping |

Reference implementation: [`src/ipind2/active_learning`](../src/ipind2/active_learning)

### 4.7 Interpretability Unit (FR-09)

| Property | Value |
|---|---|
| Methods | Attention layers inside GNN/Transformer + SHAP and LIME post-processing |
| Output | Importance score of each input feature per prediction + model confidence score |
| Goal | Most competing nanoparticle platforms lack formal interpretability (see [`docs/BENCHMARK.md`](BENCHMARK.md)); this unit is a competitive advantage |

Reference implementation: [`src/ipind2/interpretability`](../src/ipind2/interpretability)

### 4.8 Natural-Language Query Interface (FR-10)

| Property | Value |
|---|---|
| Input | Natural-language query or structured form to specify target parameters (nanocarrier type, target tissue, constraints) |
| Output | Conversion into structured parameters consumable by Unit 1 (structure generation) |

Reference implementation: [`src/ipind2/nlp_interface`](../src/ipind2/nlp_interface)

### 4.9 Automated Lab Integration (FR-11)

| Property | Value |
|---|---|
| Connection type | API/adapter for robotic synthesis and high-throughput screening equipment (liquid handlers, high-throughput screening) |
| Data flow | Robotic experiment results automatically enter the database and the active learning unit (4.6) |
| First version | Generic adapter with CSV/REST format; direct connection to specific equipment in later phases |

Reference implementation: [`src/ipind2/lab_automation`](../src/ipind2/lab_automation)

### 4.10 Continuous Internal Benchmark (FR-12)

| Property | Value |
|---|---|
| Reference datasets | Public LNP collections (e.g., the 622-sample LNP dataset, LANCE) |
| Schedule | Runs automatically at every model release (CI for models) |
| Output | Report comparing RMSE/R² of the new version against the previous version and against published papers |

Reference implementation: [`src/ipind2/benchmarking`](../src/ipind2/benchmarking)

A full comparison with international counterparts is given in [`docs/BENCHMARK.md`](BENCHMARK.md).

## 5. Data Requirements

### 5.1 Input Data Required for Training

| Data type | Number of samples | Format | Source |
|---|---|---|---|
| Molecular structures | ≥ 50,000 | SMILES + SDF | Public databases (PubChem, ZINC) |
| Physicochemical properties | ≥ 30,000 | CSV | Literature + specialized databases |
| Biological data (in vitro) | ≥ 10,000 | CSV | Literature + proprietary data |
| MD simulation data | ≥ 5,000 | XTC + EDR | Completed simulations |

### 5.2 Database Structure

The complete table schema is in [`sql/schema.sql`](../sql/schema.sql).

## 6. Security and Privacy Requirements

| ID | Requirement |
|---|---|
| SEC-01 | Two-factor authentication for all users |
| SEC-02 | Data encryption at rest (AES-256) |
| SEC-03 | Data encryption in transit (TLS 1.3) |
| SEC-04 | Logging of all user activities |
| SEC-05 | Role-based access (RBAC): admin, researcher, viewer |
| SEC-06 | Automatic daily database backup |

## 7. Hardware and Infrastructure Requirements

| Property | Minimum | Recommended |
|---|---|---|
| CPU | 16 cores | 32+ cores |
| RAM | 64 GB | 128+ GB |
| GPU | NVIDIA A10 (24GB) | NVIDIA A100 (40GB) × 2 |
| Storage | 2 TB SSD | 4+ TB NVMe SSD |
| Operating system | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| Bandwidth | 100 Mbps | 1+ Gbps |

## 8. Synthetic Data Generation

Developing the product with high accuracy and low error requires diverse, high-quality training data. The synthetic data generator is implemented in [`src/ipind2/data_generation/synthetic_data_generator.py`](../src/ipind2/data_generation/synthetic_data_generator.py) and its output columns are as follows:

| Column | Type | Description |
|---|---|---|
| id | int | Unique identifier |
| smiles | str | Molecular structure in SMILES format |
| scaffold_name | str | Molecular scaffold name |
| scaffold_type | str | Type: lipid/polymer/metal |
| desc_mol_weight | float | Molecular weight (Da) |
| desc_logP | float | Partition coefficient |
| desc_tpsa | float | Polar surface area (Å²) |
| desc_num_rotatable_bonds | int | Number of rotatable bonds |
| desc_num_h_donors | int | Number of hydrogen donors |
| desc_num_h_acceptors | int | Number of hydrogen acceptors |
| phys_size_nm | float | Nanoparticle size (nm) |
| phys_zeta_potential_mV | float | Zeta potential (mV) |
| phys_pdi | float | Polydispersity index |
| phys_colloidal_stability_hours | float | Colloidal stability (hours) |
| phys_drug_loading_efficiency_percent | float | Loading efficiency (%) |
| phys_drug_loading_content_percent | float | Loading content (wt%) |
| phys_release_rate_constant | float | Release rate constant |
| bio_cytotoxicity_ic50_ug_ml | float | Cytotoxicity (μg/mL) |
| bio_cellular_uptake_efficiency_percent | float | Cellular uptake efficiency (%) |
| bio_serum_protein_binding_percent | float | Serum protein binding (%) |
| bio_circulation_half_life_hours | float | Circulation half-life (hours) |
| bio_tumor_to_background_ratio | float | Tumor-to-background ratio |
| pareto_score | float | Pareto optimization score |
| is_pareto_optimal | int | Whether on the Pareto front (1/0) |
| pareto_rank | int | Rank on the Pareto front |

Run:

```bash
pip install -r requirements.txt
python -m ipind2.data_generation.synthetic_data_generator
```
