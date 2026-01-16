import pandas as pd
import numpy as np
import re
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataCleaner")

def clean_text_value(x):
    """Force la conversion en string et nettoie."""
    if x is None or pd.isna(x):
        return ""
    x = str(x)
    x = x.replace('""', '"')
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def get_or_create_clean_data(raw_path: str, clean_path: str) -> pd.DataFrame:
    """
    Charge les données. Si le fichier propre n'existe pas, on le crée.
    """
    # 1. Si le fichier propre existe, on le charge
    if os.path.exists(clean_path):
        try:
            df = pd.read_csv(clean_path)
            # Vérification ultime anti-NaN
            df = df.fillna("")
            return df
        except Exception:
            logger.warning("Cache corrompu, on recharge le brut.")

    # 2. Sinon, on charge le brut
    if not os.path.exists(raw_path):
        # Création d'un DF vide de secours pour éviter le crash
        return pd.DataFrame(columns=['CompetencyID','Competency','BlockID','BlockName'])

    try:
        # On essaie de détecter le séparateur automatiquement
        with open(raw_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(1024)
            sep = ';' if sample.count(';') > sample.count(',') else ','
        
        df = pd.read_csv(raw_path, sep=sep, engine="python", on_bad_lines="skip")
    except:
        df = pd.read_csv(raw_path, engine="python", on_bad_lines="skip")

    # 3.On force les noms de colonnes si elles sont mal détectées
    if len(df.columns) == 4: 
        df.columns = ['CompetencyID','Competency','BlockID','BlockName']
    
    # On remplace tout NaN par ""
    df = df.fillna("")
    
    # On nettoie la colonne texte spécifiquement
    if 'Competency' in df.columns:
        df['Competency'] = df['Competency'].apply(clean_text_value)
        # On supprime les lignes vides
        df = df[df['Competency'] != ""]


    df.to_csv(clean_path, index=False)
    return df