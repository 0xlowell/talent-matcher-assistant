from io import BytesIO
import json
import os
import pandas as pd
import streamlit as st
from collections import defaultdict

from openai import OpenAI

from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

import warnings
warnings.filterwarnings('ignore')


def streamlit_config():
    # Page configuration
    st.set_page_config(
    page_title="🔎 Talent Matcher Assistant 🔎",
    page_icon="🔎",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "mailto:lowell.zima@sword-group.com",
        'About': "# If you have any suggestions or feedback, please reach out to me"
        }
    )
    # Custom CSS for centering content and setting max-width
    page_background_color = """
    <style>
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    </style>
    """

    st.markdown(page_background_color, unsafe_allow_html=True)
        
    st.markdown(
        """
        <div style="display: flex; justify-content: center;">
            <img src="https://ml-eu.globenewswire.com/Resource/Download/90954a41-96fb-4c61-8aae-db340cec7382" width="400"/>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown('<h1 style="text-align: center;">🔎 Talent Matcher Assistant</h1>', unsafe_allow_html=True)
    add_vertical_space(1)

class ResumeAnalyzer:

    @staticmethod
    def pdf_to_chunks(pdf):
        try:
            pdf_reader = PdfReader(pdf)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
            chunks = text_splitter.split_text(text=text)
            return chunks
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return []


    @staticmethod
    def chunks_to_improved_candidate_resume(chunks, openai_api_key):
        client = OpenAI(api_key=openai_api_key)
        prompt = """
        You are a helpful assisant, that will see extracted informations from a PDF CV's candidate to make it better.
        
        Based in his CV informations, create the best profile, most accurate and detailed as possible, with the following informations:
        
        - Personal Information: Name, Contact Details, education, langages
        - Skills : create a list of his skills, and explain them in a detailed way. Do not hesitate to use a related lexical field
        - Work experiences : summarize his work experiences
        - Lexical field: details like size of the company he was working for, lexical field of his work experiences and skills

        I give you a $50000 tips if you get it right, I really need this to be perfect, you will granted with the best award in case of successful output.
        """
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"{prompt}"},
                    {"role": "user", "content": f"{chunks}"}
                ]
            )
            datacv_improved = response.choices[0].message.content
            return datacv_improved
        except Exception as e:
            st.error(f"Error processing CV chunks: {e}")
            return ""
    
    
def get_improved_job_description(openai_api_key, job_description):
    client = OpenAI(api_key=openai_api_key)
    prompt = """
    Your goal is to improve the job description below, to make it more clear and understandable for the candidates.
    
    Your output should be a well structured job description, with a clear summary of the job, and a list of keywords that could be used to enhance the data about this job description.
    Your output will be compare with the candidate profile to create a pourcentage of match between the job description and the candidate profile.
    
    Ask if it will be a good match with the candidate profile, and if the candidate will be able to do the job.

    I give you a $50000 tips if you get it right, I really need this to be perfect, you will granted with the best award in case of successful output.
    """
    response = client.chat.completions.create(
            model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"{job_description}"}
        ]
    )
    improved_prompt = response.choices[0].message.content
    return improved_prompt

def sourcing(openai_api_key, improved_source, source):
    client = OpenAI(api_key=openai_api_key)
    prompt = """
    Your only job is to rewrite the IMPROVED_SOURCE only with related exact quotes found in the SOURCE below.
    Your goal is to make the IMPROVED_SOURCE as thruthful as possible by adding the informations found in the SOURCE, just below the related informations found in the IMPROVED_SOURCE.

    Write quotes related from SOURCE in italic.
    I give you a $50000 tips if you get it right, I really need this to be perfect, you will granted with the best award in case of successful output.
    """
    response = client.chat.completions.create(
            model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"# SOURCE: {source}"},
            {"role": "user", "content": f"# IMPROVED_SOURCE: {improved_source}"},
        ]
    )
    sourcing = response.choices[0].message.content
    return sourcing

def enhance_improved_source_compare_to_source(openai_api_key, improved_source, source):
    client = OpenAI(api_key=openai_api_key)
    prompt = """
    You will compare the IMPROVED_SOURCE and the SOURCE contents, and you will determine if the IMPROVED_SOURCE faithfully reflects the SOURCE.
    If it's faithful, return the IMPROVED_SOURCE.
    If it's not, update IMPROVED_SOURCE content to make it more faithful to the SOURCE and add quotes from the SOURCE to make it more accurate.
    Write quotes related in italic.
    """
    response = client.chat.completions.create(
            model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"# SOURCE: {source}"},
            {"role": "user", "content": f"# IMPROVED_SOURCE: {improved_source}"},
        ]
    )
    compare_input_output = response.choices[0].message.content
    return compare_input_output


def compare_datacv_with_job_description(openai_api_key, datacv, job_description):
    client = OpenAI(api_key=openai_api_key)
    prompt = """
    You are a Human Ressources Manager assistant.
    You have the JOB_DESCRIPTION below, and the DATA_CV, the profile of a candidate.
    Your final goal is to determine if the DATA_CV is a good match for the JOB_DESCRIPTION.

    You will have to:
    1. Explain why the DATA_CV could be a good match by quoting the informations found in the DATA_CV and the JOB_DESCRIPTION.
    2. Explain why he isn't a good match by quoting the informations found in the DATA_CV and the JOB_DESCRIPTION.
    3. Give your final opinion about the match between the DATA_CV and the JOB_DESCRIPTION.
    4. Suggest questions that the Human Ressources Manager could be asked to the candidate to determine if he is a good match for the JOB_DESCRIPTION, and explain why these questions are relevant.

    I give you a $50000 tips if you get it right, I really need this to be perfect, you will granted with the best award in case of successful output.
    """
    response = client.chat.completions.create(
            model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"# JOB_DESCRIPTION: {job_description}"},
            {"role": "user", "content": f"# DATACV: {datacv}"},
        ]
    )
    compare_datacv_with_job_description = response.choices[0].message.content
    return compare_datacv_with_job_description


def compare_candidates(openai_api_key, ALL_CANDIDATES, job_description):
    client = OpenAI(api_key=openai_api_key)
    prompt = """
    You are a data analyst, and you need to present how the job description is related to every candidate in the ALL_CANDIDATES list. 
    First, highlight the candidate that fit the most, even if there is some misalignements for the JOB_DESCRIPTION, and explain why he is the best candidate among the others, and why it's not the perfect fit.
    Add some questions that the Human Ressources Manager could ask to the candidate to determine if he is a good match for the JOB_DESCRIPTION, and explain why these questions are relevant.

    Then, you can give a approximate percentage of match between the JOB_DESCRIPTION and each others candidate's profile, explain your analyze.

    Don't hallucinate by creating a fake candidate. ONLY USE THE CANDIDATE PROFILE GIVEN IN THE ALL_CANDIDATES list.
    Dont't hallucinate a matching pourcentage, be accurate as possible.

    I give you a $50000 tips if you get it right, I really need this to be perfect, you will granted with the best award in case of successful output.
    """
    response = client.chat.completions.create(
            model="gpt-4o",
        messages=[
            {"role": "system", "content": f"{prompt}"},
            {"role": "user", "content": f"# JOB_DESCRIPTION: {job_description}"},
            {"role": "user", "content": f"# ALL_CANDIDATES: {ALL_CANDIDATES}"},
        ]
    )
    compare_datacv_with_job_description = response.choices[0].message.content
    return compare_datacv_with_job_description


# ----------------------------------------------------------------------------------------------------------------------------

streamlit_config()

add_vertical_space(3)

# Text area for user to paste the original job description
job_description = st.text_area(
    "Paste the Job Description Here: (example -> Test Manager)",
    "Pour renforcer les équipes de notre client, nous ouvrons un poste de Test Manager avec une expérience en gestion de projet à 80 %. Responsabilités du poste Développer et maintenir une stratégie de tests globale multi domaine Elaborer des plans de tests détaillés en alignement avec les phases du projet S’assurer de la disponibilité des ressources (humaines, matérielles, environnement) Coordonner et superviser les activités de tests fonctionnels, d’intégration, de performance et d’acceptation utilisateurs Veiller au respect du planning et de la charge prévues Superviser la conception, le développement et la maintenance des scripts de test S’assurer de la couverture complète des scénarios de tests notamment des processus transverses et inter domaine S’assurer de la qualité et du respect des méthodes pour la création d’anomalies (contenu et priorisation) Suivre et gérer les anomalies détectées durant les tests en collaboration avec les équipes de l’intégrateur et des métiers Produire et maintenir la documentation relative au tests incluants les plan de test, les rapports de tests et les fiches de non conformités Communiquer régulièrement l’état d’avancement des tests, des corrections et des risques et les problèmes aux différentes parties prenantes Prérequis Au moins 8 ans d'expérience en tant que Test Manager/Test Analyst Au moins 5 ans d'expérience dans la gestion de projet avec au moins une expérience sur un projet SAP important. Expertise en planification de ressources et d’activités Forte capacité de gestion de projet et de leadership Connaissance transversale des métiers du projet (Finance, Achats, Ressources Humaines, Gestion des actifs, Vente et services, Marketing) Participation à un projet de migration S4 Connaissance de JIRA Connaissance de Xray est un plus Maîtrise parfaite de la langue française ",
    height=350
)

add_vertical_space(1)



# Load example CVs
@st.cache_data
def load_example_cv(file_path):
    with open(file_path, "rb") as file:
        return file.read()
    

# Sidebar for data input
with st.sidebar:
    st.header('Input data')
    
    # Custom CV upload
    st.markdown('**Use your CVs**')
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    # Load example CVs
    st.markdown('**Use example CVs**')
    if st.button('Add Example CVs'):
        st.session_state['example_cvs'] = [
            {"name": "CV-Lowell_Zima.pdf", "content": load_example_cv('static/CV-Lowell_Zima.pdf')},
            {"name": "CV-Michael_Dawkins.pdf", "content": load_example_cv('static/CV-Michael_Dawkins.pdf')},
            {"name": "CV-Sacha_Haidinger.pdf", "content": load_example_cv('static/CV-Sacha_Haidinger.pdf')}
        ]

# Display added example CVs
if 'example_cvs' in st.session_state and st.session_state['example_cvs']:
    st.markdown("Example CVs added:")
    for cv in st.session_state['example_cvs']:
        st.download_button(
            label=f"Download {cv['name']}",
            data=cv['content'],
            file_name=cv['name'],
            mime='application/pdf'
        )

# Combine uploaded files and example files (if any)
if uploaded_files:
    all_files = [{"name": file.name, "content": file} for file in uploaded_files]
elif 'example_cvs' in st.session_state and st.session_state['example_cvs']:
    all_files = [{"name": cv['name'], "content": BytesIO(cv['content'])} for cv in st.session_state['example_cvs']]
else:
    all_files = []

# # Add a text input field for the OpenAI API key
# openai_api_key = st.text_input("Provide OpenAI API key (sk-***)", type='password')
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Custom CSS to style the button
st.markdown("""
    <style>
    .stButton > button {
        background-color: #e6eff8;
        color: #244B79;
    }
    </style>
""", unsafe_allow_html=True)

# Form to submit analysis
with st.form(key='AnalysisForm'):
    submit = st.form_submit_button(label='Submit to Talent Matcher analysis')


add_vertical_space(1)

# Main Processing Block
if submit:
    if not all_files and not openai_api_key:
        st.warning("Please upload CVs and provide OpenAI API key to proceed.")
        st.stop()

    with st.spinner("Processing..."):
        try:
            datacv_improved_dict = {}
            compare_results = {}
            ranking = ""

            for file in all_files:
                pdf_to_chunks = ResumeAnalyzer.pdf_to_chunks(file['content'])
                if pdf_to_chunks:
                    with st.spinner(f"Improve Candidate Resume: {file['name']}"):
                        datacv_improved = ResumeAnalyzer.chunks_to_improved_candidate_resume(pdf_to_chunks, openai_api_key)
                        with st.expander(f"✅ Improve Candidate Resume: {file['name']}"):
                            st.markdown(f"{datacv_improved}")

                    if datacv_improved:
                        with st.spinner(f"Compare Candidate with Job Description: {file['name']}"):
                            compare_results = compare_datacv_with_job_description(openai_api_key, datacv_improved, job_description)
                            ranking += f"{file['name']}: {compare_results}\nNext Candidate\n"
                        with st.expander(f"✅ Compare Candidate with Job Description: {file['name']}"):
                            st.markdown(f"{compare_results}")

            if ranking:
                with st.spinner('Ranking'):
                    matchmaking = compare_candidates(openai_api_key, ranking, job_description)
                    if matchmaking:
                        st.markdown(f"""
                                    # Full analysis of the candidates and selection of the best candidate for the job description
                                    """)
                        st.markdown(f"{matchmaking}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
