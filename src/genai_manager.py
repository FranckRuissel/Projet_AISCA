import os
import json
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

class GenAIManager:
    def __init__(self, cache_file="genai_cache.json"):
        """
        Optimisations : Modèle Flash, Cache MD5, Prompts courts, Anti-Hallucination.
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.referentiel_map = self._load_referentiel_names()
        
        if not API_KEY:
            print(" Erreur : Pas de clé API dans .env")
            self.model = None
            return

        genai.configure(api_key=API_KEY)
        
        # Stratégie économique : On force Gemini 1.5 Flash
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print(">>> Modèle IA : gemini-2.5-flash (Optimisé)")
        except Exception:
            print(">>> Modèle IA : Fallback sur gemini-pro")
            self.model = genai.GenerativeModel('gemini-pro')

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _load_referentiel_names(self):
        """Charge le JSON pour savoir que 'bloc_1' = 'Architecture Data'."""
        mapping = {}
        path = os.path.join("data", "referentiel.json")
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for b in data.get('blocs_competences', []):
                        mapping[b['id_bloc']] = b['nom']
            except Exception as e:
                print(f"Erreur lecture referentiel : {e}")
        return mapping

    def _save_cache(self):
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=4)

    def _generate(self, prompt, key_prefix):
        if not self.model: return "IA non disponible."

        # Clé unique basée sur le contenu exact (MD5)
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_key = f"{key_prefix}_{prompt_hash}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            response = self.model.generate_content(prompt, generation_config={"temperature": 0.3})
            clean_text = response.text.strip()
            self.cache[cache_key] = clean_text
            self._save_cache()
            return clean_text
        except Exception as e:
            return f"Erreur API : {str(e)}"

    def generer_bio(self, user_inputs, top_metier, top_competences):
        prompt = f"""
        Rôle: Expert RH.
        Tâche: Bio professionnelle 3e personne (3 phrases max).
        Style: Corporatif, sans emojis.
        Cible: {top_metier}
        Points forts: {', '.join(top_competences)}
        Candidat: "{' '.join(user_inputs)}"
        """
        return self._generate(prompt, "BIO")

    def generer_plan_progression(self, metier_vise, scores_blocs):
        # On remplace 
        points_faibles = []
        for bloc_id, score in scores_blocs.items():
            if score < 0.6:
                nom_reel = self.referentiel_map.get(bloc_id, bloc_id)
                points_faibles.append(nom_reel)

        if not points_faibles:
            points_faibles = ["Perfectionnement technique", "Leadership"]

        prompt = f"""
        Rôle: Mentor Tech Senior.
        Cible: {metier_vise}.
        Lacunes identifiées: {', '.join(points_faibles)}.
        Tâche: Plan d'action 3 étapes concrètes pour combler ces lacunes.
        Format: Markdown. Pas d'intro/outro. Pas d'emojis.
        """
        return self._generate(prompt, "PLAN")

    def enrichir_phrase_courte(self, phrase):
        if len(phrase.split()) > 6: 
            return phrase

        prompt = f"""
        Tâche : Reformule ce mot-clé pour un CV professionnel.
        Entrée : "{phrase}"
        CONSIGNES STRICTES :
        1. N'ajoute AUCUNE compétence non mentionnée.
        2. Rends la phrase grammaticalement complète.
        3. Pas d'emojis.
        Exemple : "Python" -> "Programmation en langage Python."
        """
        return self._generate(prompt, "ENRICH")