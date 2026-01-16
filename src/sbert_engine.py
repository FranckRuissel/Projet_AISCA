import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import torch

# Importation de notre gestionnaire de données
from src.data_loader import get_or_create_clean_data

class SBERTEngine:
    def __init__(self, data_path="data"):
        print("[INFO] Initialisation du moteur SBERT...")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        self.raw_competences = os.path.join(data_path, "competences.csv")
        self.clean_competences = os.path.join(data_path, "competences_clean.csv")
        
        self.raw_metiers = os.path.join(data_path, "metiers.csv")
        self.clean_metiers = os.path.join(data_path, "metiers_clean.csv")
        
        self.cache_path = os.path.join(data_path, "embeddings_cache.pt")
        
        # Chargement et nettoyage des données
        self.df_competences = get_or_create_clean_data(self.raw_competences, self.clean_competences)
        self.df_metiers = get_or_create_clean_data(self.raw_metiers, self.clean_metiers)
        
        # Vérification critique
        if self.df_competences.empty:
            print("[ERREUR] Aucune compétence chargée. Vérifiez les fichiers CSV.")
            self.competence_embeddings = None
            return

        # Gestion du Cache (pour la rapidité)
        if os.path.exists(self.cache_path):
            print("[INFO] Chargement des embeddings depuis le cache disque...")
            try:
                self.competence_embeddings = torch.load(self.cache_path, map_location=device)
                
                # Vérification de la correspondance Cache vs CSV
                if len(self.competence_embeddings) != len(self.df_competences):
                    print("[ATTENTION] Le cache ne correspond plus aux données. Recalcul nécessaire.")
                    self._compute_and_save_embeddings()
                else:
                    print("[SUCCES] Embeddings chargés.")
            except Exception as e:
                print(f"[ERREUR] Echec lecture cache : {e}. Recalcul nécessaire.")
                self._compute_and_save_embeddings()
        else:
            print("[INFO] Premier lancement : Calcul des embeddings en cours...")
            self._compute_and_save_embeddings()

        print("[INFO] Moteur SBERT prêt et opérationnel.")

    def _compute_and_save_embeddings(self):
        """
        Calcule les vecteurs et sauvegarde le résultat pour les prochains démarrages.
        """
        raw_texts = self.df_competences['Competency'].astype(str).tolist()
        
        self.competence_embeddings = self.model.encode(
            raw_texts, 
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        torch.save(self.competence_embeddings, self.cache_path)
        print(f"[SUCCES] Embeddings sauvegardés : {self.cache_path}")

    def calculate_scores(self, user_inputs):
        """
        Analyse les entrées utilisateur et retourne les scores et recommandations.
        """
        print("\n" + "="*60)
        print("[ANALYSE] Traitement en cours...")
        
        # 1. Nettoyage des entrées
        clean_inputs = [str(i) for i in user_inputs if str(i).strip() != ""]
        
        if not clean_inputs or self.competence_embeddings is None:
            return {
                "scores_par_bloc": {},
                "recommandations_metiers": [],
                "top_competences_details": pd.DataFrame()
            }

        # 2. Vectorisation et Calcul de Similarité
        user_embeddings = self.model.encode(clean_inputs, convert_to_tensor=True)
        cosine_scores = util.cos_sim(user_embeddings, self.competence_embeddings)
        
        # Meilleur score par compétence
        max_scores_per_competency, _ = cosine_scores.max(dim=0)
        
        # Copie de travail
        df_res = self.df_competences.copy()
        df_res['score'] = max_scores_per_competency.cpu().tolist()
        
        # 3. Agrégation par Bloc (Top-K Mean)
        def get_top_k_mean(scores, k=5):
            top_scores = scores.nlargest(k)
            if len(top_scores) == 0: return 0.0
            return top_scores.mean()

        scores_par_bloc = df_res.groupby('BlockID')['score'].apply(
            lambda x: get_top_k_mean(x, k=5)
        ).to_dict()
        
        # 4. Matching Métiers
        recommandations = []
        for index, job in self.df_metiers.iterrows():
            job_title = job.get('Job Title', 'Inconnu')
            required_blocks_str = str(job.get('Required Competencies', ''))
            required_blocks = required_blocks_str.split(';')
            
            total_score = 0
            valid_blocks_count = 0
            
            for bloc in required_blocks:
                bloc = bloc.strip() 
                if bloc in scores_par_bloc:
                    score_du_bloc = scores_par_bloc[bloc]
                    
                    # Bonus Expert (>0.6)
                    if score_du_bloc > 0.6:
                        score_du_bloc *= 1.1
                        
                    total_score += score_du_bloc
                    valid_blocks_count += 1
            
            # Moyenne pondérée et normalisation
            raw_average = total_score / valid_blocks_count if valid_blocks_count > 0 else 0
            normalized_score = min(raw_average * 1.5, 1.0)
            
            recommandations.append({
                "metier": job_title,
                "score": round(normalized_score, 4),    
                "score_percent": f"{int(normalized_score * 100)}%"
            })
            
        # Tri décroissant
        recommandations = sorted(recommandations, key=lambda x: x['score'], reverse=True)
        
        # 5. Extraction des détails (Top 10)
        top_details = df_res.nlargest(10, 'score')[['Competency', 'score', 'BlockName']].copy()
        
        # --- AFFICHAGE DEBUG DANS LA CONSOLE (TEXTE SIMPLE) ---
        print("-" * 60)
        print("RESULTATS DE L'ANALYSE")
        print("-" * 60)
        
        print("SCORES PAR DOMAINE (BLOCS) :")
        for bloc, score in scores_par_bloc.items():
            print(f" - {bloc:<40} : {score:.4f}")

        print("\nTOP 3 RECOMMANDATIONS METIERS :")
        for i, job in enumerate(recommandations[:3]):
            print(f" {i+1}. {job['metier']:<40} : {job['score_percent']}")

        print("\nTOP 5 CORRESPONDANCES SEMANTIQUES (PREUVE) :")
        for i, (idx, row) in enumerate(top_details.head(5).iterrows()):
            print(f" {i+1}. Score: {row['score']:.4f} | {row['Competency'][:80]}...")
            
        print("="*60 + "\n")

        return {
            "scores_par_bloc": scores_par_bloc,
            "recommandations_metiers": recommandations,
            "top_competences_details": top_details
        }

if __name__ == "__main__":
    engine = SBERTEngine()