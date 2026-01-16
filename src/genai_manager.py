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
        Gestionnaire IA optimisé : Modèle Flash, Cache MD5, Prompts professionnels.
        Respecte les contraintes : Pas d'emojis en sortie, Ton corporatif.
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Permet de traduire les IDs techniques en noms métiers pour le prompt
        self.referentiel_map = {
            "bloc_1": "Architecture Data & Modélisation",
            "bloc_2": "Ingénierie Big Data & DevOps",
            "bloc_3": "Analyse BI & Visualisation",
            "bloc_4": "Data Science & IA Avancée",
            "bloc_5": "Gouvernance & Qualité des Données"
        }
        
        if not API_KEY:
            print("[ERREUR] Pas de clé API trouvée dans le fichier .env")
            self.model = None
            return

        # Configuration de l'API Google
        genai.configure(api_key=API_KEY)
        
        # On utilise gemini-2.5-flash qui est le standard actuel pour la rapidité
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            print("[INFO] Modèle IA chargé : gemini-2.5-flash (Mode Rapide)")
        except Exception as e:
            print(f"[WARN] Erreur chargement Flash ({e}), fallback sur Pro.")
            self.model = genai.GenerativeModel('gemini-pro')

    def _load_cache(self):
        """Charge le cache depuis le disque pour économiser les appels API."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return {}
        return {}

    def _save_cache(self):
        """Sauvegarde les nouvelles réponses dans le fichier JSON."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"[ERREUR] Impossible de sauvegarder le cache : {e}")

    def _generate(self, prompt, key_prefix):
        """
        Fonction centrale de génération avec gestion du cache MD5.
        """
        if not self.model: return "Service IA indisponible (Clé API manquante)."

        # 1. Création d'une clé unique basée sur le contenu du prompt (MD5)
        # Si le prompt ne change pas, on ne rappelle pas Google 
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_key = f"{key_prefix}_{prompt_hash}"
        
        # 2. Vérification Cache
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 3. Appel API
        try:
            # Temperature 0.3 = Créativité faible pour rester factuel et pro
            response = self.model.generate_content(prompt, generation_config={"temperature": 0.3})
            clean_text = response.text.strip()
            
            # Mise en cache
            self.cache[cache_key] = clean_text
            self._save_cache()
            return clean_text
        except Exception as e:
            return f"Erreur de génération : {str(e)}"

    def generer_bio(self, user_inputs, top_metier, top_competences):
        """Génère un résumé exécutif du profil."""
        prompt = f"""
        Rôle : Expert RH spécialisé en Data.
        Tâche : Rédige une bio professionnelle courte (3 phrases max) à la 3ème personne.
        Cible Métier : {top_metier}
        Compétences Clés détectées : {', '.join(top_competences)}
        Contexte Candidat : "{' '.join(user_inputs)}"
        
        CONSIGNES DE STYLE :
        - Ton strictement corporatif et professionnel.
        - PAS d'emojis.
        - Pas de phrases introductives type "Voici la bio".
        - Mets en valeur l'expertise technique.
        """
        return self._generate(prompt, "BIO")

    def generer_plan_progression(self, metier_vise, scores_blocs):
        """Génère un plan d'action pour les blocs faibles (< 60%)."""
        
        # Identification des lacunes via le mapping
        points_faibles = []
        for bloc_id, score in scores_blocs.items():
            if score < 0.6:
                # On utilise le dictionnaire interne pour avoir un beau nom
                nom_reel = self.referentiel_map.get(bloc_id, "Compétences Techniques Générales")
                points_faibles.append(nom_reel)

        # Si le candidat est parfait partout, on propose du leadership
        if not points_faibles:
            points_faibles = ["Leadership Technique", "Architecture d'Entreprise"]

        prompt = f"""
        Rôle : Mentor Tech Senior (CTO).
        Objectif : Préparer le candidat pour le poste de {metier_vise}.
        Lacunes identifiées à combler : {', '.join(points_faibles)}.
        
        Tâche : Propose un plan d'action en 3 points concrets (Format Markdown).
        
        CONSIGNES :
        - Soyez direct et prescriptif (ex: "Apprenez X", "Pratiquez Y").
        - PAS d'emojis.
        - Pas de jargon marketing, uniquement des conseils techniques ou méthodologiques.
        - Format liste à puces.
        """
        return self._generate(prompt, "PLAN")

    def enrichir_phrase_courte(self, phrase):
        """Reformule les mots-clés isolés pour donner du contexte à SBERT."""
        # Optimisation : si la phrase est déjà longue, on ne touche à rien (économie API)
        if len(phrase.split()) > 6: 
            return phrase

        prompt = f"""
        Tâche : Transforme ce mot-clé ou cette expression courte en une phrase de compétence professionnelle pour un CV.
        Entrée : "{phrase}"
        
        RÈGLES STRICTES :
        1. N'ajoute AUCUNE technologie ou compétence qui n'est pas implicite dans le mot d'origine.
        2. La phrase doit être grammaticalement complète (Sujet/Verbe ou Action).
        3. PAS d'emojis.
        4. Restez factuel.
        
        Exemple : "Python" -> "Développement de scripts et applications en langage Python."
        Exemple : "Gestion projet" -> "Pilotage et suivi de projets techniques."
        """
        return self._generate(prompt, "ENRICH")