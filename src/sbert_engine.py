import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os

class SBERTEngine:
    def __init__(self, data_path="data"):
        
        print(">>> Chargement du modèle SBERT...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.competences_path = os.path.join(data_path, "competences.csv")
        self.metiers_path = os.path.join(data_path, "metiers.csv")
        
        self.df_competences = self._load_competences()
        self.df_metiers = self._load_metiers()
        print(">>> Encodage des compétences du référentiel...")
        self.competence_embeddings = self.model.encode(
            self.df_competences['Competency'].tolist(), 
            convert_to_tensor=True
        )
        print(">>> Moteur SBERT prêt !")

    def _load_competences(self):
        try:
            return pd.read_csv(self.competences_path)
        except FileNotFoundError:
            raise Exception(f"Erreur : Le fichier {self.competences_path} est introuvable.")

    def _load_metiers(self):
        try:
            return pd.read_csv(self.metiers_path)
        except FileNotFoundError:
            raise Exception(f"Erreur : Le fichier {self.metiers_path} est introuvable.")

    def calculate_scores(self, user_inputs):
        
        user_embeddings = self.model.encode(user_inputs, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(user_embeddings, self.competence_embeddings)
        
        max_scores_per_competency, _ = cosine_scores.max(dim=0) 
        self.df_competences['score'] = max_scores_per_competency.cpu().tolist()
        scores_par_bloc = self.df_competences.groupby('BlockID')['score'].mean().to_dict()
        
        recommandations = []
        for index, job in self.df_metiers.iterrows():
            job_title = job['Job Title']
            required_blocks = job['Required Competencies'].split('; ')
            
            total_score = 0
            valid_blocks = 0
            
            for bloc in required_blocks:
                bloc = bloc.strip() 
                if bloc in scores_par_bloc:
                    total_score += scores_par_bloc[bloc]
                    valid_blocks += 1
            
            final_job_score = total_score / valid_blocks if valid_blocks > 0 else 0
            
            recommandations.append({
                "metier": job_title,
                "score": round(final_job_score, 4),    
                "score_percent": f"{round(final_job_score * 100, 1)}%"
            })
            
        recommandations = sorted(recommandations, key=lambda x: x['score'], reverse=True)
        
        return {
            "scores_par_bloc": scores_par_bloc,
            "recommandations_metiers": recommandations
        }

if __name__ == "__main__":
    engine = SBERTEngine()
    
"""
    test_inputs = [
        "J'utilise dbt pour transformer mes données tous les jours.",
        "Je suis un expert en SQL et je fais des window functions.",
        "Je déploie mes pipelines sur Airflow via Docker",
        "Je connais un peu le Machine Learning mais pas trop."
    ]
    
    print("\n--- ANALYSE DE L'UTILISATEUR TEST ---")
    resultats = engine.calculate_scores(test_inputs)
    
    print("\nScores par Bloc :")
    for bloc, score in resultats['scores_par_bloc'].items():
        print(f"  - {bloc}: {score:.2f}")
        
    print("\nTop 3 Métiers recommandés :")
    for metier in resultats['recommandations_metiers'][:3]:
        print(f"  - {metier['metier']} : {metier['score_percent']}")
    
    """