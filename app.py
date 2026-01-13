import streamlit as st
import pandas as pd
import plotly.express as px 
import numpy as np

# Import des modules internes
from src.sbert_engine import SBERTEngine
from src.genai_manager import GenAIManager

# Configuration de la page
st.set_page_config(page_title="AISCA - Solution RH", layout="wide")

# --- CHARGEMENT ---
@st.cache_resource
def load_engines():
    engine_sbert = SBERTEngine()
    engine_genai = GenAIManager() 
    return engine_sbert, engine_genai

try:
    sbert, genai_coach = load_engines()
    st.sidebar.info("Moteurs d'analyse prêts.")
except Exception as e:
    st.error(f"Erreur critique : {e}")
    st.stop()

# --- INTERFACE ---

st.title("AISCA : Assistant Intelligent de Carriere")
st.markdown("""
**Système d'Analyse de Compétences :**
Cet outil utilise le NLP (SBERT) et l'IA Générative pour cartographier vos compétences et recommander des parcours professionnels adaptés.
""")

st.divider()

st.header("1. Profil du Candidat")
col1, col2 = st.columns(2)

with col1:
    exp_text = st.text_area("Experience Professionnelle", height=200, 
                            placeholder="Décrivez vos projets, vos responsabilités...")
with col2:
    tech_stack = st.text_area("Compétences Techniques", height=200,
                              placeholder="Langages, Outils, Frameworks...")

if st.button("Lancer l'analyse", type="primary"):
    
    if not exp_text and not tech_stack:
        st.warning("Veuillez remplir au moins un champ.")
    else:
        with st.spinner("Traitement en cours..."):
            raw_inputs = [s for s in [exp_text, tech_stack] if s]
            final_inputs = []
            
            with st.expander("Voir le détail de l'enrichissement sémantique (Debug)", expanded=True):
                st.caption("L'IA reformule les termes courts pour aider le moteur de recherche, sans ajouter de fausses compétences.")
                for text in raw_inputs:
                    enriched = genai_coach.enrichir_phrase_courte(text)
                    final_inputs.append(enriched)
                    
                    c_a, c_b = st.columns(2)
                    with c_a:
                        st.text(f"Original : {text}")
                    with c_b:
                        if text != enriched:
                            st.success(f"Enrichi : {enriched}")
                        else:
                            st.write(f"Inchangé : {enriched}")
                    st.write("---")

            resultats = sbert.calculate_scores(final_inputs)
            
            top_job = resultats['recommandations_metiers'][0]
            scores_blocs = resultats['scores_par_bloc']
            
            st.divider()
            c1, c2 = st.columns([1, 1])
            
            with c1:
                st.subheader("Analyse Quantitative")
                st.info(f"**Profil identifié : {top_job['metier']}**")
                st.metric("Indice de correspondance", top_job['score_percent'])
                
                df_radar = pd.DataFrame(dict(
                    r=list(scores_blocs.values()),
                    theta=list(scores_blocs.keys())
                ))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0,1])
                fig.update_traces(fill='toself')
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.subheader("Analyse Qualitative (GenAI)")
                
                # Bio
                top_comp = [b for b, s in scores_blocs.items() if s > 0.6]
                bio = genai_coach.generer_bio(final_inputs, top_job['metier'], top_comp)
                st.text_area("Biographie Professionnelle :", value=bio, height=150)
                
                # Plan
                plan = genai_coach.generer_plan_progression(top_job['metier'], scores_blocs)
                with st.expander("Consulter le Plan de Progression", expanded=True):
                    st.markdown(plan)

            st.divider()
            st.caption("Autres correspondances :")
            cols = st.columns(3)
            for i, job in enumerate(resultats['recommandations_metiers'][1:4]):
                cols[i].write(f"**{job['metier']}** : {job['score_percent']}")