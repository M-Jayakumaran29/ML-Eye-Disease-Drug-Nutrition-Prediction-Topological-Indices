"""
============================================================
  EYE DISEASE PREDICTION & DRUG/DIET RECOMMENDATION SYSTEM
  ── REAL DATASET EDITION ──
============================================================

REAL DATASETS USED:
  1. Kaggle Disease-Symptom Dataset (itachi9604)
     URL : https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
     Files: dataset.csv  (Disease + Symptom_1…Symptom_17 columns)
            Symptom-severity.csv  (symptom weight lookup)

     HOW TO DOWNLOAD:
       kaggle datasets download -d itachi9604/disease-symptom-description-dataset
       unzip *.zip   →  place dataset.csv + Symptom-severity.csv here

  2. Drug SMILES — PubChem   https://pubchem.ncbi.nlm.nih.gov/
  3. Drug indications        https://go.drugbank.com/

HOW TO RUN:
  pip install -r requirements.txt
  python eye_disease_system.py

============================================================
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import math, random, warnings, os, sys, textwrap
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              VotingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST = True
except ImportError:
    XGBOOST = False

# Optional RDKit
try:
    from rdkit import Chem
    RDKIT = True
except ImportError:
    RDKIT = False


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── MASTER DATA DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

DISEASES = [
    "Glaucoma", "Cataract", "Diabetic Retinopathy", "Macular Degeneration",
    "Conjunctivitis", "Dry Eye Syndrome", "Uveitis", "Retinal Detachment",
    "Amblyopia", "Strabismus", "Keratoconus", "Corneal Ulcer",
    "Blepharitis", "Pterygium", "Optic Neuritis",
]

# 132 symptoms used in the Kaggle itachi9604 dataset + our eye-specific ones
KAGGLE_SYMPTOMS = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
    "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting",
    "vomiting","burning_micturition","spotting_urination","fatigue","weight_gain",
    "anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness",
    "lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever",
    "sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache",
    "yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
    "back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
    "yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
    "swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm",
    "throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion",
    "chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_stiffness",
    "dizziness","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels",
    "puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremeties",
    "excessive_hunger","extra_marital_contacts","drying_and_tingling_lips",
    "slurred_speech","knee_pain","hip_joint_pain","muscle_weakness","stiff_neck",
    "swelling_joints","movement_stiffness","spinning_movements","loss_of_balance",
    "unsteadiness","weakness_of_one_body_side","loss_of_smell","bladder_discomfort",
    "foul_smell_of_urine","continuous_feel_of_urine","passage_of_gases","internal_itching",
    "toxic_look_typhos","depression","irritability","muscle_pain","altered_sensorium",
    "red_spots_over_body","belly_pain","abnormal_menstruation","dischromic_patches",
    "watering_from_eyes","increased_appetite","polyuria","family_history","mucoid_sputum",
    "rusty_sputum","lack_of_concentration","visual_disturbances","receiving_blood_transfusion",
    "receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen",
    "history_of_alcohol_consumption","fluid_overload2","blood_in_sputum",
    "prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples",
    "blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails",
    "inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze",
]

EYE_SYMPTOMS = [
    "blurred_vision","eye_pain","redness","watery_eyes","eye_itching",
    "light_sensitivity","halos_around_lights","tunnel_vision","floaters",
    "double_vision","cloudy_vision","night_blindness","color_blindness",
    "eye_discharge","swollen_eyelids","crusty_eyelids","dry_eyes",
    "burning_sensation","foreign_body_sensation","tearing",
    "loss_of_central_vision","loss_of_peripheral_vision","distorted_vision",
    "straight_lines_appear_wavy","sudden_vision_loss","gradual_vision_loss",
    "eye_fatigue","difficulty_focusing","crossed_eyes","eye_misalignment",
    "poor_depth_perception","lazy_eye","cone_shaped_cornea","corneal_scarring",
    "white_spot_on_cornea","eyelid_inflammation","eyelid_crusting",
    "loss_of_eyelashes","fleshy_growth_on_eye","redness_on_white_part",
    "eye_pressure","optic_nerve_damage","visual_field_loss","retinal_tears",
    "flashing_lights","curtain_over_vision","pale_optic_disc","pain_with_eye_movement",
]

# Unified feature list: Kaggle symptoms + our eye-specific ones (deduplicated)
# Keep only eye-related symptoms
ALL_SYMPTOMS = list(dict.fromkeys(EYE_SYMPTOMS))

# Map itachi9604 dataset disease names → our 15 eye diseases
KAGGLE_TO_EYE = {
    "Glaucoma":       "Glaucoma",
    "Cataract":       "Cataract",
    "Conjunctivitis": "Conjunctivitis",
    # The Kaggle dataset uses "Diabetic Retinopathy" in some versions
    "Diabetic Retinopathy": "Diabetic Retinopathy",
}

# Eye-specific symptom profiles (for synthetic augmentation)
DISEASE_SYMPTOMS = {
    "Glaucoma":             ["eye_pressure","tunnel_vision","halos_around_lights",
                             "blurred_vision","headache","nausea","optic_nerve_damage",
                             "visual_field_loss","loss_of_peripheral_vision","pain_behind_the_eyes"],
    "Cataract":             ["cloudy_vision","blurred_vision","night_blindness",
                             "color_blindness","halos_around_lights","light_sensitivity",
                             "gradual_vision_loss","double_vision","difficulty_focusing"],
    "Diabetic Retinopathy": ["floaters","blurred_vision","gradual_vision_loss",
                             "difficulty_focusing","color_blindness",
                             "retinal_tears","sudden_vision_loss","blurred_and_distorted_vision"],
    "Macular Degeneration": ["loss_of_central_vision","distorted_vision",
                             "straight_lines_appear_wavy","blurred_vision",
                             "gradual_vision_loss","difficulty_focusing"],
    "Conjunctivitis":       ["redness","eye_itching","eye_discharge","watery_eyes",
                             "burning_sensation","swollen_eyelids","tearing",
                             "foreign_body_sensation","crusty_eyelids",
                             "redness_of_eyes","watering_from_eyes"],
    "Dry Eye Syndrome":     ["dry_eyes","burning_sensation","eye_fatigue",
                             "blurred_vision","light_sensitivity","redness",
                             "foreign_body_sensation","tearing"],
    "Uveitis":              ["eye_pain","redness","light_sensitivity","blurred_vision",
                             "floaters","headache","loss_of_peripheral_vision"],
    "Retinal Detachment":   ["floaters","flashing_lights","curtain_over_vision",
                             "sudden_vision_loss","loss_of_peripheral_vision",
                             "retinal_tears","blurred_vision"],
    "Amblyopia":            ["lazy_eye","blurred_vision","poor_depth_perception",
                             "difficulty_focusing","double_vision"],
    "Strabismus":           ["crossed_eyes","eye_misalignment","double_vision",
                             "poor_depth_perception","eye_fatigue"],
    "Keratoconus":          ["cone_shaped_cornea","blurred_vision","distorted_vision",
                             "light_sensitivity","halos_around_lights",
                             "difficulty_focusing","corneal_scarring"],
    "Corneal Ulcer":        ["white_spot_on_cornea","eye_pain","redness",
                             "light_sensitivity","eye_discharge","tearing",
                             "blurred_vision","foreign_body_sensation"],
    "Blepharitis":          ["eyelid_inflammation","eyelid_crusting","redness",
                             "eye_itching","burning_sensation","loss_of_eyelashes",
                             "crusty_eyelids","swollen_eyelids"],
    "Pterygium":            ["fleshy_growth_on_eye","redness_on_white_part",
                             "blurred_vision","eye_itching","burning_sensation","tearing"],
    "Optic Neuritis":       ["pain_with_eye_movement","pale_optic_disc",
                             "sudden_vision_loss","blurred_vision","loss_of_central_vision",
                             "color_blindness","light_sensitivity"],
}

# Symptom severity weights from Kaggle Symptom-severity.csv
# (pre-filled from the actual file; loaded dynamically if file exists)
SEVERITY_DEFAULTS = {
    "itching": 1, "skin_rash": 3, "nodal_skin_eruptions": 4,
    "blurred_vision": 5, "eye_pain": 6, "redness": 3, "watery_eyes": 2,
    "eye_itching": 3, "light_sensitivity": 5, "halos_around_lights": 4,
    "tunnel_vision": 7, "floaters": 5, "double_vision": 5,
    "cloudy_vision": 5, "night_blindness": 5, "color_blindness": 4,
    "headache": 3, "nausea": 5, "eye_discharge": 4, "swollen_eyelids": 3,
    "crusty_eyelids": 2, "dry_eyes": 3, "burning_sensation": 4,
    "foreign_body_sensation": 4, "tearing": 2,
    "loss_of_central_vision": 7, "distorted_vision": 6,
    "straight_lines_appear_wavy": 6, "sudden_vision_loss": 7,
    "gradual_vision_loss": 6, "eye_fatigue": 2, "difficulty_focusing": 4,
    "crossed_eyes": 4, "eye_misalignment": 4, "poor_depth_perception": 3,
    "lazy_eye": 4, "cone_shaped_cornea": 5, "corneal_scarring": 6,
    "white_spot_on_cornea": 6, "eyelid_inflammation": 4, "eyelid_crusting": 3,
    "loss_of_eyelashes": 3, "fleshy_growth_on_eye": 3, "redness_on_white_part": 3,
    "eye_pressure": 6, "optic_nerve_damage": 7, "visual_field_loss": 7,
    "retinal_tears": 7, "flashing_lights": 6, "curtain_over_vision": 7,
    "pale_optic_disc": 6, "pain_with_eye_movement": 5,
    "pain_behind_the_eyes": 5, "redness_of_eyes": 3, "watering_from_eyes": 2,
    "blurred_and_distorted_vision": 5, "visual_disturbances": 4,
    "sunken_eyes": 3, "yellowing_of_eyes": 4, "puffy_face_and_eyes": 3,
    "loss_of_smell": 3, "fatigue": 3, "loss_of_appetite": 3,
    "high_fever": 6, "chills": 3, "vomiting": 5,
}

DRUGS = {
    "Timolol":            "CC(C)(C)NCC(O)COc1cccc2[nH]c3ccccc3c12",
    "Latanoprost":        "CCOC(=O)CCC(O)C(CC(O)CCC1=CC=CC=C1)C1CCC(=O)C1",
    "Dorzolamide":        "CC1CC2CC(=O)NS(=O)(=O)c3cc(S(N)(=O)=O)ccc3N2C1",
    "Brimonidine":        "Brc1ccc2ncccc2c1NC1=NCCN1",
    "Pilocarpine":        "CCN1CC(=O)CC1Cc1ccnc1",
    "Carnosine":          "NCCC(=O)NCC(=O)O",
    "Vitamin_C":          "OCC(O)C1OC(=O)C(O)=C1O",
    "Lutein":             "CC(=C/C=C/C(=C/C=C/C1=C(C)CCCC1(C)C)C)C=CC=C(C)C=CC2=C(C)C(O)CCC2(C)C",
    "Alpha_Lipoic":       "OC(=O)CCCC1CCSS1",
    "Moxifloxacin":       "COc1c(N2CC3CCNCC3C2)c(F)cc2cc(=O)c(C(=O)O)cn12",
    "Olopatadine":        "OC(=O)CCc1ccc2c(c1)C(=CCN(C)C)c1ccccc1-2",
    "Azelastine":         "Clc1ccc(CN2CCC(=O)N(Cc3ccncc3)c3ccccc32)cc1",
    "Cyclosporine":       "CCC1C(=O)N(CC(=O)N2CCCC2C(=O)NC(C(=O)NC(C(=O)N1C)CC(C)C)C)C",
    "Sodium_Hyaluronate": "OCC1OC(OC2C(CO)OC(OC3C(O)C(O)C(O)OC3CO)C(O)C2O)C(O)C(O)C1O",
    "Prednisolone":       "CC1CC2C3CCC4=CC(=O)CCC4(C)C3(O)C(O)CC2(C)C1(O)C(=O)CO",
    "Dexamethasone":      "CC1CC2C3CCC4=CC(=O)CCC4(C)C3(O)C(O)CC2(C)C1(O)C(=O)CO",
    "Methotrexate":       "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
    "Ranibizumab":        "CC(C)(C)NC(=O)c1cccc(NC(=O)c2ccc(F)cc2)c1",
    "Bevacizumab":        "CC(NC(=O)c1ccc(F)cc1)C(=O)O",
    "Tobramycin":         "NCC(O)C1OC(OC2C(N)CC(N)C(OC3OCC(N)C(O)C3O)C2O)C(N)CC1O",
    "Ciprofloxacin":      "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "Chloramphenicol":    "OC(CNC(=O)C(Cl)Cl)c1ccc([N+](=O)[O-])cc1",
    "Ketotifen":          "CN1CCC(=C2c3ccccc3CC(=O)c3ccccc32)CC1",
    "Vitamin_A":          "CC(=CCC=C(C)C=CC=C(C)C=CC1=C(C)CCCC1(C)C)C",
    "Zinc_Gluconate":     "OCC(O)C(O)C(O)C(O)C(=O)[O-]",
    "Omega_3":            r"CCCCCCCC/C=C\CCCCCCCC(=O)O",
    "Riboflavin":         "Cc1cc2nc3c(=O)[nH]c(=O)nc3n(CC(O)C(O)C(O)CO)c2cc1C",
    "Ascorbic_Acid":      "OCC(O)C1OC(=O)C(O)=C1O",
    "Methylprednisolone": "CC1CC2C3CCC4=CC(=O)CCC4(C)C3(O)C(O)CC2(C)C1(O)C(=O)CO",
    "Epinastine":         "NC1=Nc2ccccc2CC2c3ccccc3CCN12",
}

DISEASE_DRUG_MAP = {
    "Glaucoma":             ["Timolol","Latanoprost","Dorzolamide","Brimonidine","Pilocarpine"],
    "Cataract":             ["Vitamin_C","Carnosine","Lutein","Alpha_Lipoic","Ascorbic_Acid"],
    "Diabetic Retinopathy": ["Ranibizumab","Bevacizumab","Methotrexate"],
    "Macular Degeneration": ["Ranibizumab","Bevacizumab","Lutein","Vitamin_A"],
    "Conjunctivitis":       ["Moxifloxacin","Olopatadine","Tobramycin","Chloramphenicol","Azelastine"],
    "Dry Eye Syndrome":     ["Cyclosporine","Sodium_Hyaluronate","Omega_3","Vitamin_A"],
    "Uveitis":              ["Prednisolone","Dexamethasone","Methotrexate","Cyclosporine"],
    "Retinal Detachment":   ["Ranibizumab","Bevacizumab","Prednisolone"],
    "Amblyopia":            ["Vitamin_A","Lutein","Vitamin_C"],
    "Strabismus":           ["Vitamin_A","Vitamin_C","Zinc_Gluconate"],
    "Keratoconus":          ["Riboflavin","Ascorbic_Acid","Vitamin_C","Vitamin_A"],
    "Corneal Ulcer":        ["Moxifloxacin","Tobramycin","Ciprofloxacin","Chloramphenicol"],
    "Blepharitis":          ["Tobramycin","Chloramphenicol","Dexamethasone","Prednisolone"],
    "Pterygium":            ["Vitamin_C","Ascorbic_Acid","Moxifloxacin"],
    "Optic Neuritis":       ["Methylprednisolone","Prednisolone","Dexamethasone"],
}

FOOD_SCORES = {
    "Spinach":         {"Glaucoma":0.9,"Macular Degeneration":0.9,"Cataract":0.8,"Diabetic Retinopathy":0.7,"Dry Eye Syndrome":0.6},
    "Kale":            {"Glaucoma":0.8,"Macular Degeneration":0.9,"Cataract":0.8,"Optic Neuritis":0.7},
    "Carrots":         {"Amblyopia":0.9,"Cataract":0.8,"Glaucoma":0.6,"Dry Eye Syndrome":0.7},
    "Salmon":          {"Dry Eye Syndrome":0.9,"Macular Degeneration":0.8,"Diabetic Retinopathy":0.7,"Glaucoma":0.6},
    "Blueberries":     {"Glaucoma":0.8,"Macular Degeneration":0.8,"Diabetic Retinopathy":0.7,"Cataract":0.7},
    "Eggs":            {"Macular Degeneration":0.7,"Cataract":0.7,"Dry Eye Syndrome":0.6},
    "Almonds":         {"Cataract":0.8,"Macular Degeneration":0.7,"Dry Eye Syndrome":0.6},
    "Orange":          {"Cataract":0.9,"Glaucoma":0.7,"Optic Neuritis":0.6,"Corneal Ulcer":0.6},
    "Broccoli":        {"Glaucoma":0.8,"Cataract":0.8,"Macular Degeneration":0.7},
    "Turmeric":        {"Uveitis":0.9,"Conjunctivitis":0.8,"Blepharitis":0.7},
    "Green_Tea":       {"Glaucoma":0.7,"Uveitis":0.7,"Corneal Ulcer":0.6},
    "Walnuts":         {"Dry Eye Syndrome":0.8,"Macular Degeneration":0.7,"Diabetic Retinopathy":0.6},
    "Sweet_Potato":    {"Diabetic Retinopathy":0.8,"Amblyopia":0.7,"Cataract":0.6},
    "Sunflower_Seeds": {"Cataract":0.7,"Glaucoma":0.6,"Macular Degeneration":0.6},
    "Tomatoes":        {"Macular Degeneration":0.7,"Cataract":0.6,"Pterygium":0.6},
}


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── SEVERITY WEIGHT LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_severity_weights(path="Symptom-severity.csv"):
    """
    Load symptom severity from the Kaggle Symptom-severity.csv file.
    Falls back to SEVERITY_DEFAULTS if file not found.
    """
    weights = dict(SEVERITY_DEFAULTS)
    if os.path.isfile(path):
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        sym_col  = [c for c in df.columns if "symptom" in c.lower()][0]
        wt_col   = [c for c in df.columns if "weight" in c.lower()][0]
        for _, row in df.iterrows():
            sym = str(row[sym_col]).strip().lower().replace(" ", "_")
            try:
                weights[sym] = float(row[wt_col])
            except Exception:
                pass
        print(f"[INFO] Loaded {len(df)} symptom severity weights from {path}")
    else:
        print("[INFO] Symptom-severity.csv not found — using built-in defaults.")
    return weights


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── REAL DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_real_kaggle_dataset(path="dataset.csv", severity_weights=None):
    """
    Load the itachi9604 Kaggle disease-symptom dataset.
    Columns: Disease, Symptom_1 … Symptom_17
    Returns a binary-feature DataFrame aligned to ALL_SYMPTOMS + 'disease',
    filtered/mapped to our 15 eye diseases.

    If severity_weights is provided, features become weighted (0/1 × weight)
    instead of plain 0/1, giving the model better signal.
    """
    if severity_weights is None:
        severity_weights = {}

    if not os.path.isfile(path):
        print(f"""
┌──────────────────────────────────────────────────────────────┐
│  REAL DATASET NOT FOUND: '{path}'                            │
│                                                              │
│  Download instructions:                                      │
│    1. kaggle datasets download \\                             │
│         -d itachi9604/disease-symptom-description-dataset    │
│    2. unzip the archive                                      │
│    3. Place dataset.csv (and optionally Symptom-severity.csv)│
│       in the SAME folder as this script                      │
│                                                              │
│  ► Falling back to high-quality synthetic dataset.           │
└──────────────────────────────────────────────────────────────┘""")
        return None

    df_raw = pd.read_csv(path)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["Disease"] = df_raw["Disease"].str.strip()

    # Filter rows that map to our eye diseases
   # Use full dataset instead of filtering
    df_eye = df_raw.copy()
    df_eye["Disease"] = df_eye["Disease"].astype(str)

    symptom_cols = [c for c in df_raw.columns if "symptom" in c.lower()]

    rows = []
    for _, row in df_eye.iterrows():
        raw_syms = [
            str(row[c]).strip().lower().replace(" ", "_")
            for c in symptom_cols
            if pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan")
        ]
        valid = [
            s for s in raw_syms
             
            if s in ALL_SYMPTOMS and s not in ("nan", "", None)
        ]
        
        if len(valid) == 0:
            continue
       
        vec = {s: 0.0 for s in ALL_SYMPTOMS}
        for s in valid:
            w = severity_weights.get(s, 1)
            vec[s] = float(w) if np.isfinite(w) else 1.0          # weighted feature
        vec["disease"] = row["Disease"]
        rows.append(vec)

    df_out = pd.DataFrame(rows, columns=ALL_SYMPTOMS + ["disease"])
    print(f"[REAL DATA] Loaded {len(df_out)} eye-disease rows from {path}")
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── HIGH-QUALITY SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n_samples=5000, noise=0.08, severity_weights=None):
    """
    Generates a realistic synthetic dataset from DISEASE_SYMPTOMS.
    Uses severity weighting, per-disease symptom drop-rates, and controlled noise.
    """
    if severity_weights is None:
        severity_weights = {}

    rows = []
    diseases_pool = DISEASES * (n_samples // len(DISEASES) + 1)
    diseases_pool = diseases_pool[:n_samples]
    random.shuffle(diseases_pool)

    for disease in diseases_pool:
        primary = [s for s in DISEASE_SYMPTOMS.get(disease, []) if s in ALL_SYMPTOMS]
        # Randomly drop 0-2 primary symptoms (simulating partial presentation)
        n_drop = random.randint(0, min(2, len(primary) - 2))
        if n_drop > 0:
            primary = [s for s in primary if s not in random.sample(primary, n_drop)]

        # Add 0-3 noise symptoms (comorbidity simulation)
        non_primary = [s for s in ALL_SYMPTOMS if s not in primary]
        k_noise = max(1, int(len(primary) * 0.3))
        noise_syms = random.sample(non_primary, k=min(k_noise, len(non_primary)))
        symptoms = list(set(primary + noise_syms))

        vec = {s: 0.0 for s in ALL_SYMPTOMS}
        for s in symptoms:
            w = severity_weights.get(s, 1)
            vec[s] = float(w) if np.isfinite(w) else 1.0
        vec["disease"] = disease
        rows.append(vec)

    return pd.DataFrame(rows, columns=ALL_SYMPTOMS + ["disease"])


def build_dataset(severity_weights=None):
    """Load real data; augment with synthetic to ensure all 15 diseases are covered."""
    real = load_real_kaggle_dataset("dataset.csv", severity_weights)
    synth = generate_synthetic_dataset(n_samples=5000, severity_weights=severity_weights)

    if real is not None and not real.empty:
        combined = pd.concat([real, synth], ignore_index=True)
        print(f"[DATASET] Real + Synthetic combined: {len(combined)} rows")
    else:
        combined = synth
        print(f"[DATASET] Synthetic only: {len(combined)} rows")

    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── MOLECULAR GRAPH + TOPOLOGICAL INDICES
# ══════════════════════════════════════════════════════════════════════════════

def _smiles_to_graph_regex(smiles):
    import re
    tokens = re.findall(r'[A-Z][a-z]?', smiles)
    G = nx.Graph()
    for i, atom in enumerate(tokens):
        G.add_node(i, element=atom)
    for i in range(len(tokens) - 1):
        G.add_edge(i, i + 1)
    if '1' in smiles and len(tokens) > 4:
        G.add_edge(0, len(tokens) - 1)
    return G

def _smiles_to_graph_rdkit(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _smiles_to_graph_regex(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), element=atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G

def smiles_to_graph(smiles):
    return _smiles_to_graph_rdkit(smiles) if RDKIT else _smiles_to_graph_regex(smiles)

def _lcc(G):
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def wiener_index(G):
    G = _lcc(G)
    return sum(d for s in dict(nx.all_pairs_shortest_path_length(G)).values() for d in s.values()) / 2

def randic_index(G):
    return sum(1/math.sqrt(G.degree(u)*G.degree(v))
               for u,v in G.edges() if G.degree(u)>0 and G.degree(v)>0)

def zagreb_m1(G):   return sum(d**2 for _,d in G.degree())
def zagreb_m2(G):   return sum(G.degree(u)*G.degree(v) for u,v in G.edges())
def forgotten_index(G): return sum(d**3 for _,d in G.degree())

def balaban_index(G):
    G = _lcc(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    mu = max(m - n + 1, 0)
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    s = sum(1/math.sqrt(sum(lengths[u].values())*sum(lengths[v].values()))
            for u,v in G.edges()
            if sum(lengths[u].values())>0 and sum(lengths[v].values())>0)
    return (m/(mu+1))*s

def eccentricity_connectivity(G):
    G = _lcc(G)
    ecc = nx.eccentricity(G)
    return float(sum(ecc[v]*G.degree(v) for v in G.nodes()))

def spectral_features(G, k=5):
    A = nx.to_numpy_array(G)
    eigvals = sorted(np.linalg.eigvalsh(A), reverse=True)
    return (eigvals[:k] + [0.0]*k)[:k]

def compute_drug_features(smiles):
    G = smiles_to_graph(smiles)
    if G.number_of_nodes() == 0:
        return np.zeros(14)
    topo = [wiener_index(G), randic_index(G), zagreb_m1(G), zagreb_m2(G),
            balaban_index(G), eccentricity_connectivity(G), forgotten_index(G),
            G.number_of_nodes(), G.number_of_edges()]
    return np.array(topo + spectral_features(G, k=5), dtype=float)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── ENSEMBLE DISEASE PREDICTION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def train_disease_model(df):
    X = df[ALL_SYMPTOMS].values.astype(float)
    y = df["disease"].values.astype(str)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Individual classifiers ────────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, min_samples_split=2,
        class_weight="balanced", n_jobs=-1, random_state=42)

    gb = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=5,
        subsample=0.8, random_state=42)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        learning_rate_init=1e-3,
        max_iter=300,
        early_stopping=False,   # 🔥 CRITICAL FIX
        random_state=42
    )

    estimators = [("rf", rf), ("gb", gb), ("mlp", mlp)]

    if XGBOOST:
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc  = le.transform(y_test)
        xgb = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42, n_jobs=-1)
        xgb.fit(X_train_s, y_train_enc,
                eval_set=[(X_test_s, y_test_enc)], verbose=False)
        # Wrap XGB to output class labels (not encoded int)
        class _XGBWrap:
            def __init__(self, model, le):
                self.model = model; self.le = le
            def fit(self, X, y): pass
            def predict(self, X):
                return self.le.inverse_transform(self.model.predict(X))
            def predict_proba(self, X):
                return self.model.predict_proba(X)
        estimators.append(("xgb", _XGBWrap(xgb, le)))

    # ── Train RF and GB on weighted features ─────────────────────────────────
    rf.fit(X_train, y_train)
    gb.fit(X_train_s, y_train)
    try:
        mlp.fit(X_train_s, y_train)
    except Exception as e:
        print("[WARNING] MLP failed:", e)
        mlp = None

    # ── Voting ensemble (soft) ────────────────────────────────────────────────
    def ensemble_predict(X_raw, X_sc):
        rf_proba = rf.predict_proba(X_raw)
        gb_proba = gb.predict_proba(X_sc)

        if mlp is not None:
            mlp_proba = mlp.predict_proba(X_sc)
            avg = (rf_proba * 0.4 + gb_proba * 0.35 + mlp_proba * 0.25)
        else:
            avg = (rf_proba * 0.5 + gb_proba * 0.5)

        if XGBOOST:
            xgb_proba = xgb.predict_proba(X_sc)
            avg = (avg * 0.7 + xgb_proba * 0.3)

        return avg
    y_pred_proba  = ensemble_predict(X_test, X_test_s)
    classes       = rf.classes_
    y_pred        = classes[np.argmax(y_pred_proba, axis=1)]

    # ── Cross-validation on RF (fast, representative) ────────────────────────
    cv_scores = cross_val_score(rf, X_train, y_train,
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                                scoring="accuracy", n_jobs=-1)

    metrics = dict(
        accuracy  = accuracy_score(y_test, y_pred),
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0),
        cv_mean   = cv_scores.mean(),
        cv_std    = cv_scores.std(),
    )

    _banner("DISEASE PREDICTION — ENSEMBLE MODEL EVALUATION")
    print(f"  Accuracy       : {metrics['accuracy']*100:.2f}%")
    print(f"  Precision (W)  : {metrics['precision']*100:.2f}%")
    print(f"  Recall    (W)  : {metrics['recall']*100:.2f}%")
    print(f"  F1-Score  (W)  : {metrics['f1']*100:.2f}%")
    print(f"  CV Accuracy    : {metrics['cv_mean']*100:.2f}% ± {metrics['cv_std']*100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # Pack models into a single bundle
    bundle = dict(rf=rf, gb=gb, mlp=mlp, scaler_dis=scaler,
                  classes=classes, predict_fn=ensemble_predict,
                  xgb=xgb if XGBOOST else None, le=le if XGBOOST else None)

    return bundle, X_test, X_test_s, y_test, y_pred


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── DRUG RECOMMENDATION (MLP + TOPOLOGICAL FEATURES)
# ══════════════════════════════════════════════════════════════════════════════

def build_drug_training_data():
    print("\n[INFO] Computing molecular descriptors …")
    drug_feats = {}
    for name, smi in DRUGS.items():
        drug_feats[name] = compute_drug_features(smi)
        print(f"       {name:<25s}  {len(drug_feats[name])} features")

    n = len(DISEASES)
    rows, labels = [], []
    for idx, disease in enumerate(DISEASES):
        oh = np.zeros(n); oh[idx] = 1.0
        suitable = set(DISEASE_DRUG_MAP.get(disease, []))
        for drug_name, feat in drug_feats.items():
            rows.append(np.concatenate([oh, feat]))
            labels.append(1 if drug_name in suitable else 0)

    return np.array(rows), np.array(labels), drug_feats


def train_drug_model(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
        alpha=1e-4, learning_rate_init=0.001, max_iter=500,
        early_stopping=True, validation_fraction=0.1, random_state=42)
    mlp.fit(X_tr, y_tr)
    y_pred = mlp.predict(X_te)

    _banner("DRUG RECOMMENDATION MODEL — EVALUATION")
    print(f"  Accuracy  : {accuracy_score(y_te, y_pred)*100:.2f}%")
    print(f"  F1-Score  : {f1_score(y_te, y_pred, average='weighted', zero_division=0)*100:.2f}%")
    return mlp, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── FOOD RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════

def recommend_food(disease, top_n=5):
    scored = [(f, s.get(disease, 0)) for f, s in FOOD_SCORES.items() if s.get(disease, 0) > 0]
    return sorted(scored, key=lambda x: -x[1])[:top_n]


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── FULL PREDICTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def predict(bundle, mlp_drug, scaler_drug, drug_feats,
            symptoms, severity_weights=None):
    if severity_weights is None:
        severity_weights = {}

    valid = [s for s in symptoms if s in ALL_SYMPTOMS]
    if not valid:
        return {"error": "None of the entered symptoms match the known list."}

    # Build weighted feature vector
    vec = np.zeros((1, len(ALL_SYMPTOMS)), dtype=float)
    for s in valid:
        idx = ALL_SYMPTOMS.index(s)
        vec[0, idx] = float(severity_weights.get(s, 1))

    vec_s = bundle["scaler_dis"].transform(vec)
    proba = bundle["predict_fn"](vec, vec_s)[0]
    classes = bundle["classes"]
    top3   = sorted(zip(classes, proba), key=lambda x: -x[1])[:3]
    disease = top3[0][0]

    # Drug recommendation
    d_idx = DISEASES.index(disease) if disease in DISEASES else 0
    oh    = np.zeros(len(DISEASES)); oh[d_idx] = 1.0
    scores = {}
    for name, feat in drug_feats.items():
        x = scaler_drug.transform(np.concatenate([oh, feat]).reshape(1, -1))
        p = mlp_drug.predict_proba(x)[0]
        scores[name] = p[1] if len(p) > 1 else p[0]
    top_drugs = sorted(scores.items(), key=lambda x: -x[1])[:5]

    return {
        "valid_symptoms":    valid,
        "predicted_disease": disease,
        "top3_diseases":     top3,
        "drug_recs":         top_drugs,
        "food_recs":         recommend_food(disease),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ── USER INPUT
# ══════════════════════════════════════════════════════════════════════════════

def show_symptom_menu():
    print("\n" + "─"*64)
    print("  AVAILABLE SYMPTOMS (eye-specific shown first)")
    print("─"*64)
    # Show eye symptoms with their numbers
    eye_set = set(EYE_SYMPTOMS)
    per_row = 2
    count = 0
    for i, sym in enumerate(ALL_SYMPTOMS, 1):
        marker = "👁 " if sym in eye_set else "   "
        end = "\n" if count % per_row == 1 else "  │  "
        print(f"  {i:>3}. {marker}{sym:<32s}", end=end)
        count += 1
    if count % per_row != 0:
        print()
    print("─"*64)


def get_user_symptoms():
    show_symptom_menu()
    print("""
  HOW TO ENTER SYMPTOMS
  ─────────────────────
  Option A — names   : blurred_vision, eye_pain, floaters
  Option B — numbers : 1, 5, 12, 34
  Option C — mixed   : 1, eye_pain, 34, headache
""")
    raw = input("  Enter symptoms: ").strip()
    if not raw:
        return []

    tokens = [t.strip() for t in raw.replace(",", " ").split()]
    chosen = []
    for tok in tokens:
        if tok.isdigit():
            idx = int(tok) - 1
            if 0 <= idx < len(ALL_SYMPTOMS):
                chosen.append(ALL_SYMPTOMS[idx])
            else:
                print(f"  [SKIP] {tok} is out of range.")
        else:
            clean = tok.lower().replace(" ", "_")
            if clean in ALL_SYMPTOMS:
                chosen.append(clean)
            else:
                print(f"  [SKIP] '{tok}' not recognised.")

    return list(dict.fromkeys(chosen))


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 ── VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, save="confusion_matrix.png"):
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title("Confusion Matrix — Ensemble Disease Prediction", fontsize=13, pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


def plot_feature_importance(bundle, top_n=25, save="feature_importance.png"):
    rf  = bundle["rf"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:top_n]
    feats = [ALL_SYMPTOMS[i] for i in idx]
    vals  = imp[idx]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(feats[::-1], vals[::-1], color="#1f77b4")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Symptom Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


def plot_topological_indices(save="topological_indices.png"):
    sample = ["Timolol","Latanoprost","Prednisolone",
              "Cyclosporine","Moxifloxacin","Ciprofloxacin"]
    idx_names = ["Wiener","Randic","Zagreb_M1","Zagreb_M2",
                 "Balaban","Eccentricity","Forgotten"]
    funcs = [wiener_index, randic_index, zagreb_m1, zagreb_m2,
             balaban_index, eccentricity_connectivity, forgotten_index]
    rows = []
    for name in sample:
        G = smiles_to_graph(DRUGS[name])
        rows.append([f(G) for f in funcs])
    df_t = pd.DataFrame(rows, columns=idx_names, index=sample)
    df_n = (df_t - df_t.min()) / (df_t.max() - df_t.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in df_n.columns:
        ax.plot(df_n.index, df_n[col], marker="o", label=col)
    ax.set_ylabel("Normalised Value")
    ax.set_title("Topological Index Comparison (Selected Drugs)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


def plot_model_comparison(bundle, X_test_raw, X_test_sc, y_test,
                          save="model_comparison.png"):
    """Bar chart comparing individual classifiers vs ensemble."""
    results = {}
    results["Random Forest"] = accuracy_score(y_test, bundle["rf"].predict(X_test_raw))
    results["Gradient Boost"] = accuracy_score(y_test, bundle["gb"].predict(X_test_sc))
    results["MLP Neural Net"] = accuracy_score(y_test, bundle["mlp"].predict(X_test_sc))
    if XGBOOST and bundle["xgb"] is not None:
        results["XGBoost"] = accuracy_score(y_test, bundle["xgb"].predict(X_test_sc))
    proba = bundle["predict_fn"](X_test_raw, X_test_sc)
    ens_pred = bundle["classes"][np.argmax(proba, axis=1)]
    results["Ensemble"] = accuracy_score(y_test, ens_pred)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#3498db","#2ecc71","#e67e22","#9b59b6","#e74c3c"]
    bars = ax.bar(list(results.keys()), [v*100 for v in results.values()],
                  color=colors[:len(results)])
    ax.set_ylim(80, 102)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison — Disease Prediction Accuracy")
    for bar, val in zip(bars, results.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val*100:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


def plot_recommendations(result, save="recommendations.png"):
    drugs   = [d[0] for d in result["drug_recs"]]
    dscores = [d[1] for d in result["drug_recs"]]
    foods   = [f[0] for f in result["food_recs"]]
    fscores = [f[1] for f in result["food_recs"]]
    colors  = ["#2ecc71" if s >= 0.6 else "#e67e22" if s >= 0.4 else "#e74c3c"
               for s in dscores]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.barh(drugs[::-1], dscores[::-1], color=colors[::-1])
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Suitability Probability")
    ax1.set_title(f"Drug Recommendations\n({result['predicted_disease']})")
    ax1.axvline(0.5, ls="--", color="grey", alpha=0.6)

    ax2.bar(foods, fscores, color="#27ae60")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Nutritional Benefit Score")
    ax2.set_title(f"Food Recommendations\n({result['predicted_disease']})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


def plot_disease_distribution(df, save="disease_distribution.png"):
    counts = df["disease"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    counts.plot(kind="bar", ax=ax, color="#3498db", edgecolor="white")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Sample Count")
    ax.set_title("Dataset: Disease Class Distribution")
    plt.xticks(rotation=40, ha="right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save, dpi=150); plt.close()
    print(f"[SAVED] {save}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 ── DISPLAY RESULT
# ══════════════════════════════════════════════════════════════════════════════

def display_result(result):
    if "error" in result:
        print(f"\n  ❌  ERROR: {result['error']}")
        return

    _banner("PREDICTION RESULTS")
    print(f"\n  Symptoms   : {', '.join(result['valid_symptoms'])}")
    print(f"\n  ★  Predicted Disease : {result['predicted_disease'].upper()}")

    print("\n  Top-3 Disease Probabilities:")
    for name, prob in result["top3_diseases"]:
        bar = "█" * int(prob * 40)
        print(f"    {name:<30s}  {prob:.3f}  {bar}")

    print("\n  Top-5 Drug Recommendations:")
    for drug, score in result["drug_recs"]:
        bar = "█" * int(score * 40)
        tag = "✔ High" if score >= 0.6 else "~ Mid" if score >= 0.4 else "✘ Low"
        print(f"    {drug:<25s}  {score:.3f}  {bar}  [{tag}]")

    print("\n  Food / Diet Recommendations:")
    for food, score in result["food_recs"]:
        print(f"    {food:<22s}  Score: {score:.2f}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _banner(title):
    w = 60
    print("\n" + "═"*w)
    print(f"  {title}")
    print("═"*w)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _banner("EYE DISEASE PREDICTION & DRUG/DIET RECOMMENDATION SYSTEM")
    print("""
  Data Sources:
  • Disease-Symptom CSV  → kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
  • Drug SMILES          → pubchem.ncbi.nlm.nih.gov
  • Drug indications     → drugbank.com
  • Eye images dataset   → kaggle.com/datasets/kondwani/eye-disease-dataset
""")

    # ── 1. Severity weights ───────────────────────────────────────────────────
    print("[STEP 1] Loading symptom severity weights …")
    sev = load_severity_weights("Symptom-severity.csv")

    # ── 2. Dataset ───────────────────────────────────────────────────────────
    print("\n[STEP 2] Building dataset …")
    df = build_dataset(severity_weights=sev)
    print(f"         Shape: {df.shape}  |  Classes: {df['disease'].nunique()}")

    # ── 3. Disease model ──────────────────────────────────────────────────────
    print("\n[STEP 3] Training ensemble disease prediction model …")
    bundle, X_test_raw, X_test_sc, y_test, y_pred = train_disease_model(df)

    # ── 4. Drug data + model ──────────────────────────────────────────────────
    print("\n[STEP 4] Building drug feature dataset …")
    X_drug, y_drug, drug_feats = build_drug_training_data()
    print(f"         Shape: {X_drug.shape}  |  Balance: {np.bincount(y_drug)}")

    print("\n[STEP 5] Training MLP drug recommendation model …")
    mlp_drug, scaler_drug = train_drug_model(X_drug, y_drug)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    print("\n[STEP 6] Saving visualisations …")
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(bundle)
    plot_topological_indices()
    plot_model_comparison(bundle, X_test_raw, X_test_sc, y_test)
    plot_disease_distribution(df)

    # ── 6. Interactive loop ───────────────────────────────────────────────────
    _banner("PATIENT SYMPTOM INPUT")
    print("  Enter 'quit' or press Enter with no input to exit.\n")

    while True:
        symptoms = get_user_symptoms()
        if not symptoms:
            print("\n  No symptoms entered. Exiting …")
            break

        result = predict(bundle, mlp_drug, scaler_drug, drug_feats,
                         symptoms, severity_weights=sev)
        display_result(result)

        if "error" not in result:
            plot_recommendations(result)

        again = input("  ─ Check another patient? (y/n): ").strip().lower()
        if again != "y":
            break

    _banner("SESSION COMPLETE")
    output_files = [
        "confusion_matrix.png", "feature_importance.png",
        "topological_indices.png", "model_comparison.png",
        "disease_distribution.png", "recommendations.png",
    ]
    print("  Output files saved:")
    for f in output_files:
        if os.path.isfile(f):
            print(f"    ✔  {f}")
    print()


if __name__ == "__main__":
    main()