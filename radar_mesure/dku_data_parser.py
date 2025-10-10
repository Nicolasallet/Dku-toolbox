import os
import re
import ast
from datetime import datetime
import pandas as pd


# =====================================================
# === Lecture du header principal =====================
# =====================================================

def parse_radar_header(filepath):
    """
    Lit le header d'un fichier radar et retourne un dictionnaire 
    avec seulement les champs essentiels : fréquence, site, angle, ID, polarisation, timestamp.
    """
    wanted_keys = {
        "Radar Frequency": "frequence",
        "Site Name": "site",
        "Radar Angle": "angle_local",
        "Measurement ID": "numero",
        "Polarization": "polarisation",
        "Timestamp": "timestamp"
    }

    header = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("#").strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key in wanted_keys:
                # Conversion spéciale pour la fréquence : "13GHz" -> 13
                if key == "Radar Frequency":
                    match = re.search(r"(\d+(?:\.\d+)?)\s*GHz", val)
                    if match:
                        val = float(match.group(1))
                # Conversion numérique pour l'angle et l'ID
                elif key in ["Radar Angle", "Measurement ID"]:
                    try:
                        val = int(val)
                    except ValueError:
                        pass
                header[wanted_keys[key]] = val
    return header

    
# =====================================================
# === Lecture du fichier de métadonnées ===============
# =====================================================

def parse_metadata_file(metadata_filepath, site, numero, timestamp):
    """
    Extrait latitude, longitude et pente depuis un fichier de métadonnées (.ods).

    Args:
        metadata_filepath : chemin du fichier metadata (.ods)
        site              : nom du site (ex: 'fortress_drift')
        numero            : numéro de mesure (int)
        timestamp         : datetime ou str au format ISO (ex: '2025-01-17T12:27:54.042166')

    Returns:
        dict contenant les colonnes du fichier correspondant à la mesure.
        Si aucun fichier trouvé : {} 
        Si fichier inexistant : message d'avertissement + {}
    """

    # --- Vérification de l'existence du fichier ---
    if not metadata_filepath or not os.path.exists(metadata_filepath):
        print(f"[WARNING] Fichier metadata introuvable : {metadata_filepath}")
        return {}

    # --- Gestion du mois ---
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)

    leg_dict = {
        11: 'fm_2024_dec/',
        12: 'fm_2024_dec/',
        1: 'fm_2025_jan/',
        2: 'fm_2025_fev/'
    }

    leg = leg_dict.get(timestamp.month)
    if leg is None:
        print(f"[WARNING] Mois {timestamp.month} non reconnu dans leg_dict")
        return {}

    # --- Lecture du fichier ---
    try:
        df = pd.read_excel(metadata_filepath, engine="odf")
    except Exception as e:
        print(f"[ERROR] Impossible de lire {metadata_filepath} : {e}")
        return {}

    # --- Filtrage des lignes correspondant à la mesure ---
    mask = df["filename"].astype(str).str.contains(site, case=False, na=False)
    mask &= df["Radar_Obs_Num"] == numero
    mask &= df["fold_of_radar"] == leg

    resultat = df[mask]

    if resultat.empty:
        print(f"[INFO] Aucune entrée trouvée pour site={site}, num={numero}, leg={leg}")
        return {}

    # --- Conversion de la ligne unique en dictionnaire ---
    row = resultat.iloc[0]
    result_dict = {col: (None if pd.isna(val) else val) for col, val in row.items()}

    return result_dict
    
# =====================================================
# === Extraction depuis le nom du fichier =============
# =====================================================

def parse_filename_info(filepath):
    """
    Extrait les infos principales depuis le nom du fichier radar.
    Exemple :
        '17GHz_fortress_drift_2_v_-30deg.txt'
    Retour :
        {
          'frequence': 17,
          'site': 'fortress_drift',
          'numero': 2,
          'polarisation': 'v',
          'angle_local': -30
        }
    """
    filename = os.path.basename(filepath)
    name = filename.rsplit('.', 1)[0]   # retire l’extension
    parts = name.split('_')

    # format attendu : <freq>GHz_<site_parts>_<num>_<pol>_<angle>deg
    if len(parts) < 5:
        raise ValueError(f"Nom de fichier inattendu : {filename}")

    freq_part = parts[0]
    angle_part = parts[-1]
    pol_part = parts[-2]
    num_part = parts[-3]
    site_parts = parts[1:-3]  # tout ce qui est entre freq et num

    try:
        numero = int(num_part)
    except ValueError:
        numero = None

    angle_match = re.match(r"([+-]?\d+)deg", angle_part)
    angle_local = int(angle_match.group(1)) if angle_match else None

    site = "_".join(site_parts) if site_parts else None
    polarisation = pol_part

    return {
        "frequence": freq_part,
        "site": site,
        "numero": numero,
        "polarisation": polarisation,
        "angle_local": angle_local
    }


# =====================================================
# === Fonction principale =============================
# =====================================================

def extract_radar_info(radar_filepath, metadata_filepath=None):
    """
    Extrait les informations essentielles depuis un fichier radar + optionnellement un fichier metadata.
    Retourne un dictionnaire :
        {
          "frequence": ...,
          "site": ...,
          "angle_local": ...,
          "numero": ...,
          "polarisation": ...,
          "timestamp": ...,
          "pente": ...,
          "lat": ...,
          "lon": ...
        }
    """

    info = {
        "frequence": None,
        "site": None,
        "angle_local": None,
        "numero": None,
        "polarisation": None,
        "timestamp": None,
        "pente": None,
        "lat": None,
        "lon": None
    }

    # --- Lecture du header ---
    header = parse_radar_header(radar_filepath)
    
    # --- Fréquence ---
    freq_val = header.get("frequence")
    if freq_val:
        info["frequence"] = freq_val

    # --- Polarisation ---
    if "polarisation" in header:
        info["polarisation"] = header["polarisation"].strip()

    # --- Timestamp ---
    if "timestamp" in header:
        info["timestamp"] = datetime.fromisoformat(header["timestamp"].strip())

    # --- Numéro de mesure ---
    if "numero" in header:
        try:
            info["numero"] = int(header["numero"])
        except ValueError:
            info["numero"] = header["numero"]


    # --- Lecture du nom de fichier (site + angle) ---

    header = parse_filename_info(radar_filepath)

    # --- site ---
    if "site" in header:
        info["site"] = header["site"].strip()

    # --- angle_local ---
    if "angle_local" in header:
        info["angle_local"] = header["angle_local"]
        
    

    # --- Lecture du fichier de métadonnées ---
    meta_info = parse_metadata_file(metadata_filepath, info['site'], info['numero'], info['timestamp'])

    
    info['lat'] = float(meta_info['lat'])
    info['lon'] = float(meta_info['lon'])
    info['pente'] = float(meta_info['pente'])
    
    return info
