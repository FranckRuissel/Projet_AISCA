import streamlit as st
import pandas as pd
import plotly.express as px 
# Note : Graphviz retiré comme demandé

# Import des modules internes
from src.sbert_engine import SBERTEngine
from src.genai_manager import GenAIManager

# Configuration de la page (Mode Large & Pro)
st.set_page_config(
    page_title="AISCA - Solution RH", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CHARGEMENT DES MOTEURS ---
@st.cache_resource
def load_engines():
    engine_sbert = SBERTEngine()
    engine_genai = GenAIManager() 
    return engine_sbert, engine_genai

try:
    with st.spinner("Initialisation du système d'analyse..."):
        sbert, genai_coach = load_engines()
    st.sidebar.success("Système opérationnel")
except Exception as e:
    st.error(f"Erreur critique lors du chargement : {e}")
    st.stop()

# --- RECUPERATION DU MAPPING (BlocID -> Vrai Nom) ---
# On utilise le dataframe déjà chargé par SBERT pour faire la traduction
if not sbert.df_competences.empty:
    # Crée un dictionnaire : {'bloc_1': 'Architecture Data', 'bloc_2': 'Big Data'...}
    block_mapping = sbert.df_competences.drop_duplicates('BlockID').set_index('BlockID')['BlockName'].to_dict()
else:
    block_mapping = {}

# --- EN-TÊTE ---
st.title("AISCA : Assistant Intelligent de Carrière")
st.markdown("### Plateforme d'analyse sémantique des compétences")
st.markdown("""
Cet outil analyse la profondeur technique de votre parcours pour vous positionner sur le marché, 
au-delà des simples correspondances de mots-clés.
""")

st.divider()

# --- ZONE DE SAISIE ---
st.subheader("1. Profil du Candidat")

with st.container(border=True):
    col1, col2 = st.columns(2)

    with col1:
        exp_text = st.text_area(
            "Expérience Professionnelle", 
            height=200, 
            placeholder="Décrivez vos missions principales, les technologies utilisées et les résultats obtenus...",
            help="Soyez précis sur le contexte technique."
        )
    with col2:
        tech_stack = st.text_area(
            "Stack Technique", 
            height=200,
            placeholder="Ex: Python, Spark, Snowflake, Docker, Kubernetes, CI/CD...",
            help="Listez les langages et outils maîtrisés."
        )

# --- BOUTON D'ACTION ---
analyze_btn = st.button("Lancer l'Audit de Compétences", type="primary", use_container_width=True)

if analyze_btn:
    
    if not exp_text and not tech_stack:
        st.warning("Veuillez renseigner au moins une section pour lancer l'audit.")
    else:
        with st.spinner("Traitement analytique en cours..."):
            
            # 1. Enrichissement
            raw_inputs = [s for s in [exp_text, tech_stack] if s]
            final_inputs = []
            
            with st.expander("Voir le détail du traitement sémantique (Debug)", expanded=False):
                for text in raw_inputs:
                    enriched = genai_coach.enrichir_phrase_courte(text)
                    final_inputs.append(enriched)
                    st.text(f"Input traité : {enriched[:100]}...")

            # 2. Calcul
            resultats = sbert.calculate_scores(final_inputs)
            
            top_job = resultats['recommandations_metiers'][0]
            scores_blocs = resultats['scores_par_bloc']
            top_details = resultats.get('top_competences_details', pd.DataFrame())

            # --- RÉSULTATS ---
            st.divider()
            st.subheader("2. Résultats de l'Audit")
            
            # SECTION KPIs
            score_val = float(top_job['score_percent'].strip('%'))
            
            k1, k2, k3 = st.columns(3)
            k1.metric(label="Positionnement Métier", value=top_job['metier'], delta="Profil Dominant")
            k2.metric(label="Indice de Correspondance", value=top_job['score_percent'])
            k3.metric(label="Compétences Validées", value=f"{len(top_details)} Skills identifiés")

            st.write("") 

            # SECTION VISUALISATION
            with st.container(border=True):
                c_radar, c_bar = st.columns([1, 1])

                with c_radar:
                    st.markdown("#### Couverture par Domaine")
                    
                    # --- CORRECTION NOM DES BLOCS ---
                    # On remplace les clés (bloc_1) par les valeurs (Architecture...)
                    radar_data = {block_mapping.get(k, k): v for k, v in scores_blocs.items()}
                    
                    df_radar = pd.DataFrame(dict(
                        r=list(radar_data.values()),
                        theta=list(radar_data.keys())
                    ))
                    
                    fig_radar = px.line_polar(
                        df_radar, r='r', theta='theta', line_close=True,
                        range_r=[0, 1]
                    )
                    fig_radar.update_traces(fill='toself', line_color='#004b87')
                    
                    # --- CORRECTION DARK MODE (Fond transparent) ---
                    fig_radar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(t=30, b=30, l=40, r=40),
                        polar=dict(
                            radialaxis=dict(visible=True, tickfont=dict(size=10)),
                            angularaxis=dict(tickfont=dict(size=11))
                        )
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

                with c_bar:
                    st.markdown("#### Détail Sémantique (Top 10)")
                    if not top_details.empty:
                        fig_bar = px.bar(
                            top_details.sort_values('score', ascending=True), 
                            x='score', y='Competency', orientation='h',
                            text_auto='.1%', 
                            color='score', 
                            color_continuous_scale='Blues'
                        )
                        # --- CORRECTION DARK MODE (Fond transparent) ---
                        fig_bar.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_range=[0, 1], 
                            xaxis_title="Taux de similarité", 
                            yaxis_title=None,
                            height=400,
                            margin=dict(l=0, r=0, t=30, b=0),
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
                    else:
                        st.info("Aucune donnée détaillée disponible.")

            # --- COACHING IA ---
            st.divider()
            st.subheader("3. Recommandations Stratégiques")
            
            c_bio, c_plan = st.columns(2)
            top_comp_names = [b for b, s in scores_blocs.items() if s > 0.6]

            with c_bio:
                st.markdown("##### Résumé Exécutif")
                with st.spinner("Génération du profil..."):
                    bio = genai_coach.generer_bio(final_inputs, top_job['metier'], top_comp_names)
                st.info(bio)

            with c_plan:
                st.markdown("##### Plan d'Accélération")
                with st.spinner("Analyse des axes de progression..."):
                    plan = genai_coach.generer_plan_progression(top_job['metier'], scores_blocs)
                
                with st.container(border=True):
                    st.markdown(plan)

            # --- FOOTER ---
            st.divider()
            st.caption("Autres opportunités identifiées :")
            cols_footer = st.columns(3)
            for i, job in enumerate(resultats['recommandations_metiers'][1:4]):
                cols_footer[i].metric(label=job['metier'], value=job['score_percent'])