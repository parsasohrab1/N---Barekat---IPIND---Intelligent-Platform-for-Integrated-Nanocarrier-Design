"""
Synthetic Data Generator for IPIND² Platform
پلتفرم یکپارچه طراحی هوشمند نانوحامل‌های دارویی

این ماژول داده‌های سنتتیک را برای آموزش مدل‌های:
1. تولید ساختار (VAE/GAN)
2. پیش‌بینی فیزیکوشیمیایی (GNN)
3. پیش‌بینی زیستی (Multi-Task DL)
4. بهینه‌سازی چندهدفه (RL)

تولید می‌کند.
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, AllChem
from rdkit.Chem import rdMolDescriptors
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# بخش ۱: تعریف ساختارهای پایه و کتابخانه مولکولی
# ============================================================

@dataclass
class MolecularScaffold:
    """اسکلت مولکولی پایه برای نانوحامل‌ها"""
    name: str
    smiles: str
    scaffold_type: str  # 'lipid' | 'polymer' | 'metal'
    molecular_weight_range: Tuple[float, float]
    logP_range: Tuple[float, float]

# اسکلت‌های لیپیدی (۲۵ نوع)
LIPID_SCAFFOLDS = [
    MolecularScaffold("DOTAP", "CCCCCCCCCCCCCCCC[N+](C)(C)CCCCCCCCCCCCCCCC",
                      "lipid", (600, 800), (4.0, 6.0)),
    MolecularScaffold("DOPE", "CCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCCN)OC(=O)CCCCCCCCCCCCCCC",
                      "lipid", (700, 900), (3.5, 5.5)),
    MolecularScaffold("DSPC", "CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)(O)OCC[N+](C)(C)C)OC(=O)CCCCCCCCCCCCCCCCC",
                      "lipid", (750, 950), (4.5, 6.5)),
    MolecularScaffold("Cholesterol", "CC(C)CCCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C",
                      "lipid", (350, 450), (3.0, 4.5)),
    MolecularScaffold("DOTMA", "CCCCCCCCCCCCCCCC[N+](C)(C)CCOC(=O)CCCCCCCCCCCCCCC",
                      "lipid", (600, 800), (4.0, 6.0)),
    # ... ۲۰ اسکلت لیپیدی دیگر
]

# اسکلت‌های پلیمری (۱۵ نوع)
POLYMER_SCAFFOLDS = [
    MolecularScaffold("PLGA_50_50", "CCC(=O)OC(C)C(=O)OC(C)C(=O)O",
                      "polymer", (10000, 50000), (1.5, 3.0)),
    MolecularScaffold("PEG_2000", "C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)CO",
                      "polymer", (1800, 2200), (-0.5, 1.0)),
    MolecularScaffold("PEG_PLA", "CCC(=O)OC(C)C(=O)OCCOCCOCCOC",
                      "polymer", (5000, 20000), (1.0, 2.5)),
    MolecularScaffold("Chitosan", "CC1C(C(C(C(O1)OC2C(C(C(C(O2)CO)O)O)N)CO)O)N",
                      "polymer", (50000, 200000), (-0.5, 1.5)),
    # ... ۱۱ اسکلت پلیمری دیگر
]

# اسکلت‌های فلزی (۱۰ نوع)
METAL_SCAFFOLDS = [
    MolecularScaffold("Au_NP", "[Au]", "metal", (5000, 50000), (0.0, 0.5)),
    MolecularScaffold("MSN", "O=[Si]=O", "metal", (10000, 100000), (-0.5, 0.5)),
    MolecularScaffold("Fe3O4", "O=[Fe]O[Fe]=O", "metal", (8000, 60000), (0.0, 0.5)),
    # ... ۷ اسکلت فلزی دیگر
]

ALL_SCAFFOLDS = LIPID_SCAFFOLDS + POLYMER_SCAFFOLDS + METAL_SCAFFOLDS

# ============================================================
# بخش ۲: تولیدکننده داده‌های سنتتیک
# ============================================================

class SyntheticDataGenerator:
    """
    تولیدکننده داده‌های سنتتیک برای آموزش مدل‌های IPIND²

    ویژگی‌های تولیدشده:
    - ساختارهای مولکولی (SMILES)
    - ویژگی‌های فیزیکوشیمیایی (۷ ویژگی)
    - ویژگی‌های زیستی (۵ ویژگی)
    - برچسب‌های بهینه‌سازی (پارتو)
    """

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.scaler = StandardScaler()

        # توزیع‌های نویز برای شبیه‌سازی داده‌های واقعی
        self.noise_distributions = {
            'size': ('normal', 0, 3),  # nm
            'zeta': ('normal', 0, 1.5),  # mV
            'pdi': ('normal', 0, 0.02),
            'loading': ('normal', 0, 2),  # %
            'cytotoxicity': ('normal', 0, 3),  # μg/mL
        }

    def generate_molecule(
        self,
        scaffold: Optional[MolecularScaffold] = None,
        scaffold_type: Optional[str] = None
    ) -> Dict:
        """
        تولید یک مولکول با اسکلت مشخص یا تصادفی
        """
        if scaffold is None:
            if scaffold_type:
                if scaffold_type == 'lipid':
                    scaffold = random.choice(LIPID_SCAFFOLDS)
                elif scaffold_type == 'polymer':
                    scaffold = random.choice(POLYMER_SCAFFOLDS)
                elif scaffold_type == 'metal':
                    scaffold = random.choice(METAL_SCAFFOLDS)
                else:
                    scaffold = random.choice(ALL_SCAFFOLDS)
            else:
                scaffold = random.choice(ALL_SCAFFOLDS)

        # تولید SMILES با تغییرات تصادفی (جایگزینی گروه‌های عاملی)
        base_smiles = scaffold.smiles

        # افزودن تغییرات ساختاری (برای تنوع)
        modified_smiles = self._modify_smiles(base_smiles, scaffold.scaffold_type)

        # محاسبه ویژگی‌های RDKit
        mol = Chem.MolFromSmiles(modified_smiles)
        if mol is None:
            mol = Chem.MolFromSmiles(base_smiles)

        if mol is None:
            # Fallback: استفاده از مولکول ساده
            mol = Chem.MolFromSmiles("CCCCCCCC")

        # محاسبه توصیف‌گرهای مولکولی
        descriptors = self._calculate_descriptors(mol)

        # تولید ویژگی‌های فیزیکوشیمیایی با نویز
        physico_props = self._generate_physicochemical_properties(
            scaffold, descriptors
        )

        # تولید ویژگی‌های زیستی با نویز
        bio_props = self._generate_biological_properties(
            scaffold, descriptors, physico_props
        )

        return {
            'smiles': modified_smiles,
            'scaffold_name': scaffold.name,
            'scaffold_type': scaffold.scaffold_type,
            'descriptors': descriptors,
            'physicochemical': physico_props,
            'biological': bio_props
        }

    def _modify_smiles(self, smiles: str, scaffold_type: str) -> str:
        """
        ایجاد تغییرات ساختاری برای تولید تنوع
        """
        if scaffold_type == 'lipid':
            # تغییر طول زنجیره آلکیل
            if 'CCCC' in smiles:
                # افزایش یا کاهش طول زنجیره
                chain_length = random.randint(8, 20)
                modified = smiles.replace('CCCCCCCCCCCCCCCC', 'C' * chain_length)
                return modified
        elif scaffold_type == 'polymer':
            # تغییر تعداد مونومر
            if 'CCOC' in smiles or 'OC(C)C' in smiles:
                repeat = random.randint(5, 50)
                modified = smiles.replace('CCOCCO', 'CCOC' * repeat)
                return modified
        return smiles

    def _calculate_descriptors(self, mol: Chem.Mol) -> Dict:
        """
        محاسبه توصیف‌گرهای مولکولی با RDKit
        """
        try:
            return {
                'mol_weight': Descriptors.MolWt(mol),
                'logP': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'num_rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                'num_h_donors': Lipinski.NumHDonors(mol),
                'num_h_acceptors': Lipinski.NumHAcceptors(mol),
                'num_rings': Descriptors.RingCount(mol),
                'fraction_csp3': Descriptors.FractionCsp3(mol),
                'num_heavy_atoms': mol.GetNumHeavyAtoms(),
            }
        except Exception:
            return {
                'mol_weight': random.uniform(300, 800),
                'logP': random.uniform(0, 5),
                'tpsa': random.uniform(20, 150),
                'num_rotatable_bonds': random.randint(2, 20),
                'num_h_donors': random.randint(0, 5),
                'num_h_acceptors': random.randint(1, 10),
                'num_rings': random.randint(0, 5),
                'fraction_csp3': random.uniform(0.3, 0.9),
                'num_heavy_atoms': random.randint(20, 80),
            }

    def _generate_physicochemical_properties(
        self,
        scaffold: MolecularScaffold,
        descriptors: Dict
    ) -> Dict:
        """
        تولید ویژگی‌های فیزیکوشیمیایی با روابط ساختار-فعالیت شبیه‌سازی‌شده
        """
        # اندازه: تابعی از وزن مولکولی و LogP
        base_size = 50 + 0.05 * descriptors['mol_weight'] + 5 * descriptors['logP']
        size = max(10, min(500, base_size + np.random.normal(0, 3)))

        # پتانسیل زتا: تابعی از LogP و تعداد گروه‌های قطبی
        base_zeta = 30 - 5 * descriptors['logP'] + 2 * descriptors['num_h_acceptors']
        zeta = max(-50, min(50, base_zeta + np.random.normal(0, 1.5)))

        # PDI: تابعی از اندازه و تعداد زنجیره‌ها
        base_pdi = 0.05 + 0.001 * size + 0.01 * descriptors['num_rotatable_bonds']
        pdi = min(0.5, max(0.01, base_pdi + np.random.normal(0, 0.02)))

        # پایداری کلوئیدی (نیمه‌عمر تجمع)
        stability = 10 + 2 * abs(zeta) - 0.05 * size + 5 * (1 - pdi)
        stability = max(1, min(100, stability + np.random.normal(0, 2)))

        # کارایی بارگذاری دارو
        loading_efficiency = 60 + 0.3 * abs(zeta) - 0.1 * size + 5 * (1 - pdi)
        loading_efficiency = max(10, min(99, loading_efficiency + np.random.normal(0, 2)))

        # میزان بارگذاری دارو
        loading_content = 5 + 0.02 * loading_efficiency + np.random.normal(0, 0.5)
        loading_content = max(0.5, min(30, loading_content))

        # ثابت نرخ رهایش
        release_rate = 0.01 + 0.001 * size - 0.002 * abs(zeta) + 0.01 * (1 - pdi)
        release_rate = max(0.001, min(0.5, release_rate + np.random.normal(0, 0.01)))

        return {
            'size_nm': size,
            'zeta_potential_mV': zeta,
            'pdi': pdi,
            'colloidal_stability_hours': stability,
            'drug_loading_efficiency_percent': loading_efficiency,
            'drug_loading_content_percent': loading_content,
            'release_rate_constant': release_rate
        }

    def _generate_biological_properties(
        self,
        scaffold: MolecularScaffold,
        descriptors: Dict,
        physico: Dict
    ) -> Dict:
        """
        تولید ویژگی‌های زیستی با روابط ساختار-فعالیت شبیه‌سازی‌شده
        """
        # سمیت سلولی (IC50): کمتر بهتر
        base_toxicity = 50 + 5 * descriptors['logP'] - 2 * descriptors['tpsa'] / 100
        toxicity = max(5, min(200, base_toxicity + np.random.normal(0, 3)))

        # کارایی نفوذ سلولی
        uptake = 40 + 0.5 * abs(physico['zeta_potential_mV']) - 0.1 * physico['size_nm']
        uptake = max(5, min(95, uptake + np.random.normal(0, 2)))

        # اتصال به پروتئین‌های سرم
        protein_binding = 30 + 3 * descriptors['logP'] + 0.1 * physico['size_nm']
        protein_binding = max(5, min(95, protein_binding + np.random.normal(0, 2)))

        # نیمه‌عمر در گردش خون
        half_life = 1 + 0.5 * abs(physico['zeta_potential_mV']) - 0.01 * physico['size_nm']
        half_life = max(0.5, min(24, half_life + np.random.normal(0, 0.5)))

        # نسبت تجمع تومور به بافت سالم
        tbr = 0.5 + 0.01 * uptake - 0.001 * toxicity - 0.001 * physico['size_nm']
        tbr = max(0.1, min(10, tbr + np.random.normal(0, 0.1)))

        return {
            'cytotoxicity_ic50_ug_ml': toxicity,
            'cellular_uptake_efficiency_percent': uptake,
            'serum_protein_binding_percent': protein_binding,
            'circulation_half_life_hours': half_life,
            'tumor_to_background_ratio': tbr
        }

    def generate_dataset(
        self,
        n_samples: int = 100000,
        scaffold_type: Optional[str] = None,
        include_pareto_labels: bool = True
    ) -> pd.DataFrame:
        """
        تولید دیتاست کامل با تعداد نمونه مشخص
        """
        data = []

        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Generating sample {i}/{n_samples}...")

            molecule = self.generate_molecule(scaffold_type=scaffold_type)

            record = {
                'id': i,
                'smiles': molecule['smiles'],
                'scaffold_name': molecule['scaffold_name'],
                'scaffold_type': molecule['scaffold_type'],
            }

            # افزودن توصیف‌گرها
            for key, value in molecule['descriptors'].items():
                record[f'desc_{key}'] = value

            # افزودن ویژگی‌های فیزیکوشیمیایی
            for key, value in molecule['physicochemical'].items():
                record[f'phys_{key}'] = value

            # افزودن ویژگی‌های زیستی
            for key, value in molecule['biological'].items():
                record[f'bio_{key}'] = value

            data.append(record)

        df = pd.DataFrame(data)

        # تولید برچسب‌های بهینه‌سازی پارتو
        if include_pareto_labels:
            df = self._add_pareto_labels(df)

        return df

    def _add_pareto_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        افزودن برچسب‌های بهینه‌سازی پارتو به دیتاست
        """
        # محاسبه امتیاز ترکیبی (مقادیر نرمال‌شده)
        # اهداف: حداکثر loading، حداکثر uptake، حداقل toxicity، محدوده size مطلوب

        # نرمال‌سازی
        loading_norm = (df['phys_drug_loading_efficiency_percent'] - df['phys_drug_loading_efficiency_percent'].min()) / \
                       (df['phys_drug_loading_efficiency_percent'].max() - df['phys_drug_loading_efficiency_percent'].min())

        uptake_norm = (df['bio_cellular_uptake_efficiency_percent'] - df['bio_cellular_uptake_efficiency_percent'].min()) / \
                      (df['bio_cellular_uptake_efficiency_percent'].max() - df['bio_cellular_uptake_efficiency_percent'].min())

        toxicity_norm = 1 - (df['bio_cytotoxicity_ic50_ug_ml'] - df['bio_cytotoxicity_ic50_ug_ml'].min()) / \
                        (df['bio_cytotoxicity_ic50_ug_ml'].max() - df['bio_cytotoxicity_ic50_ug_ml'].min())

        # امتیاز اندازه مطلوب (۸۰-۱۲۰ نانومتر)
        size = df['phys_size_nm']
        size_score = np.exp(-((size - 100) / 30) ** 2)

        # امتیاز ترکیبی
        df['pareto_score'] = 0.3 * loading_norm + 0.3 * uptake_norm + 0.25 * toxicity_norm + 0.15 * size_score

        # برچسب Pareto-optimal (۱۰٪ بالاترین امتیازها)
        threshold = df['pareto_score'].quantile(0.90)
        df['is_pareto_optimal'] = (df['pareto_score'] >= threshold).astype(int)

        # رتبه‌بندی
        df['pareto_rank'] = df['pareto_score'].rank(ascending=False, method='dense')

        return df

    def save_dataset(self, df: pd.DataFrame, filepath: str = 'synthetic_dataset.csv'):
        """
        ذخیره دیتاست در فایل CSV
        """
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        print(f"Total samples: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"Pareto-optimal samples: {df['is_pareto_optimal'].sum()}")
        return filepath

    def generate_for_gnn(self, n_samples: int = 50000) -> Dict:
        """
        تولید داده‌های آماده برای آموزش GNN
        """
        df = self.generate_dataset(n_samples)

        # تبدیل SMILES به گراف برای GNN
        graphs = []
        for smiles in df['smiles'].values:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # استخراج ویژگی‌های گره (اتم‌ها)
                node_features = []
                for atom in mol.GetAtoms():
                    node_features.append([
                        atom.GetAtomicNum(),
                        atom.GetDegree(),
                        atom.GetTotalNumHs(),
                        atom.GetImplicitValence(),
                        atom.GetIsAromatic()
                    ])

                # استخراج ویژگی‌های یال (پیوندها)
                edge_features = []
                edge_indices = []
                for bond in mol.GetBonds():
                    edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                    edge_features.append([
                        bond.GetBondTypeAsDouble(),
                        bond.GetIsAromatic()
                    ])

                graphs.append({
                    'node_features': np.array(node_features),
                    'edge_indices': np.array(edge_indices),
                    'edge_features': np.array(edge_features),
                    'targets': {
                        'size': df.loc[df.index[0], 'phys_size_nm'],  # باید اصلاح شود
                    }
                })

        return {
            'graphs': graphs,
            'dataframe': df
        }


# ============================================================
# بخش ۴: اجرا و تولید دیتاست نهایی
# ============================================================

def main():
    """اجرای اصلی برای تولید دیتاست"""

    print("=" * 60)
    print("IPIND² Synthetic Data Generator")
    print("پلتفرم یکپارچه طراحی هوشمند نانوحامل‌های دارویی")
    print("=" * 60)

    # مقداردهی اولیه
    generator = SyntheticDataGenerator(random_seed=2024)

    # تولید دیتاست کامل
    print("\n[1] Generating full dataset (100,000 samples)...")
    df_full = generator.generate_dataset(n_samples=100000)

    # ذخیره دیتاست کامل
    generator.save_dataset(df_full, 'ipind2_dataset_full.csv')

    # تولید دیتاست اختصاصی برای هر نوع نانوحامل
    print("\n[2] Generating scaffold-specific datasets...")

    for scaffold_type in ['lipid', 'polymer', 'metal']:
        print(f"  - Generating {scaffold_type} dataset (20,000 samples)...")
        df_type = generator.generate_dataset(
            n_samples=20000,
            scaffold_type=scaffold_type
        )
        generator.save_dataset(df_type, f'ipind2_dataset_{scaffold_type}.csv')

    # تولید دیتاست کوچک برای تست
    print("\n[3] Generating test dataset (1,000 samples)...")
    df_test = generator.generate_dataset(n_samples=1000)
    generator.save_dataset(df_test, 'ipind2_dataset_test.csv')

    # آمار نهایی
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print("Total datasets generated: 5")
    print(f"Total samples: {100000 + 3*20000 + 1000:,}")
    print("\nFiles created:")
    print("  - ipind2_dataset_full.csv (100,000 samples)")
    print("  - ipind2_dataset_lipid.csv (20,000 samples)")
    print("  - ipind2_dataset_polymer.csv (20,000 samples)")
    print("  - ipind2_dataset_metal.csv (20,000 samples)")
    print("  - ipind2_dataset_test.csv (1,000 samples)")
    print("\nDataset statistics:")
    print(f"  - Features per sample: {len(df_full.columns)}")
    print(f"  - Pareto-optimal samples: {df_full['is_pareto_optimal'].sum():,}")
    print(f"  - Scaffold types: {df_full['scaffold_type'].unique()}")
    print("=" * 60)

    return df_full


if __name__ == "__main__":
    df = main()
