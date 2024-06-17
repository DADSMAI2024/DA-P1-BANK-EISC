import streamlit as st
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, precision_score, recall_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib


@st.cache_data
def generate_random_value(x): 
  return random.uniform(0, x) 
a = generate_random_value(10) 
b = generate_random_value(20)


df= pd.read_csv("bank.csv")
st.sidebar.title("Sommaire")

# PAGE 0 # Sommaire"
pages=["Le projet","Le jeu de donnée", "Visualisations", "Préparation des données", "Modélisation", "Machine Learning", "Conclusion et Perspective"]
page=st.sidebar.radio("Aller vers", pages)
with st.sidebar:
    st.title("Auteurs")
    "[Elodie Barnay Henriet](https://www.linkedin.com/in/elodie-barnay-henriet-916a6311a/)"
    "[Irina Grankina](https://www.linkedin.com/in/irinagrankina/)"
    "[Samantha Ebrard](https://www.linkedin.com/in/samanthaebrard/)"
    "[Cédric Le Stunff](https://www.linkedin.com/in/cedric-le-stunff-profile/)"


# PAGE 0 # Edition de la page "Le projet"
if page ==  pages[0] :
  st.title("Prédiction du succès d’une campagne de Marketing d’une banque")
  st.subheader("Contexte") 
  st.markdown("Les données du dataset bank.csv sont liées aux campagnes de marketing direct d'une institution bancaire portugaise menées entre Mai 2008 et Novembre 2010. Les campagnes de marketing étaient basées sur des appels téléphoniques. Plusieurs contacts avec le même client ont eu lieu pour savoir si le dernier a, oui ou non, souscrit au produit: dépôt bancaire à terme")
  st.markdown("Note : Le dépôt à terme est un type d'investissement proposé par les banques et les institutions financières. Dans le cadre d'un dépôt à terme, un particulier dépose une certaine somme d'argent auprès de la banque pour une période déterminée, appelée durée ou échéance. L'argent est conservé par la banque pendant la durée spécifiée, au cours de laquelle il est rémunéré à un taux d'intérêt fixe.")
  st.subheader("**Objectif**")
  st.markdown("Notre but est d'analyser l'ensemble des données afin d'identifier les tendances et les facteurs qui influencent la souscription d'un compte de dépôt à terme par les clients. Nous visons à identifier les modèles qui pourront être utilisés pour optimiser les futures campagnes marketing de l'institution financière, afin d'augmenter la souscription au produit 'dépôt à terme'.")
  st.image("https://th.bing.com/th/id/R.45557d7e2852eb77e717412686be0f41?rik=jh%2b7rN%2fMIpNflg&riu=http%3a%2f%2fwww.comexplorer.com%2fhubfs%2fimage_a_la_une%2ffacteurs-cles-de-succes-marketing.jpg%23keepProtocol&ehk=VUk4ykpnfzz5QMKov79bbPgvtqF7wIssDKdhTg8CGFM%3d&risl=&pid=ImgRaw&r=0")
  st.write("")

# PAGE 01 # Edition de la page "Le jeu de donnée"
if page == pages[1] : 
  st.title("Le jeu de donnée")

  st.write("")

  st.subheader("Origines du data set")
  st.markdown("Le jeu de données 'bank.csv' est basé sur le jeu de données UCI 'Bank Marketing' dont on peut lire la description ici : http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). Créateurs: S. Moro, P. Rita, P. Cortez.")
  
  st.subheader("Les différentes variables")
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _âge_</span> : _(num)_", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _job_</span> : _(cat)_ type d'emploi (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _marital_</span> : _(cat)_ état matrimonial (divorced, married, single)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _education_</span> : _(cat)_ niveau d'étude (primary, secondary , tertiary, unknown)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _default_</span> : _(cat)_ a un défaut de crédit ? (yes, no)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _housing_</span> : _(cat)_ dispose d'un prêt au logement ? (yes, no)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _loan_</span> : _(cat)_ a un prêt personnel ?  (yes, no)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _balance_</span> : _(num)_ solde du compte en banque", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _contact_</span> : _(cat)_ type de communication du contact (cellular, telephone, unknown)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _month_</span> : _(cat)_ dernier mois de contact (jan, feb, ... dec)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _day_</span> : _(num)_ dernier jour de contact", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _duration_</span> : _(num)_ durée du dernier contact, en secondes", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _campaign_</span> : _(num)_ nombre de contacts effectués au cours de cette campagne et pour ce client", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _pdays_</span> : _num)_ nombre de jours écoulés depuis le dernier contact du client lors d'une campagne précédente (-1 signifie que le client n'a pas été contacté précédemment)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _previous_</span> : _(num)_ nombre de contacts effectués avant cette campagne et pour ce client", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _poutcome_</span> : _(cat)_ résultat de la précédente campagne marketing (failure, other, success, unknown)", unsafe_allow_html=True)
  st.write("<span style='color:white;'>.</span> <span style='color:red;'> _deposit_</span> : _(cat/target)_ le client a, oui ou non, souscrit au dépôt à terme (yes, no)", unsafe_allow_html=True)

  dataset = st.checkbox("Afficher un extract du data set")
  if dataset:
    st.dataframe(df.head())

  st.write("")

  st.subheader("Résumé des statistiques descriptives des variables quantitatives")
  st.write(df.describe())
  st.write("Nous pouvons noter que:")
  st.write("_age_ : 50% des valeurs sont entre 32 et 49 ans. Beaucoup de valeurs extrêmes : max 95.")
  st.write("_balance_ : 50% des valeurs sont entre 122 et 1708. Présence de valeurs négatives et de valeurs extrêmes : min -6 847, max 81 204.")
  st.write("_duration_ : 50% des valeurs sont entre 138 sec (2min) et 496 (8min). Présence de valeurs extrêmes : max 3 881.")
  st.write("_campaign_ :  50% des valeurs sont entre 1 et 3 contacts.Présence de valeurs extrêmes : max 63.")
  st.write("_pdays_ : 50% des valeurs sont entre - 1 et 20. La médiane est à -1 ce qui signifie que la moitié des clients n'ont jamais été contacté avant cette campagne. Présence de valeurs extrêmes : max 854.")
  st.write("_previous_ : 50% des valeurs sont entre 0 et 1. Présence de valeurs extrêmes : max 58.")

  st.write("")

# PAGE 02 # Edition de la page "Quelques Visualisations"
if page == pages[2] : 
  st.title("Visualisations")

#   col1, col2, col3 = st.columns([1, 1, 1])  

#   button1 = col1.button("Volumétrie des commandes")
#   # button2 = col2.button("Analyse des revenus")
#   # button3 = col3.button("Analyse de la variable cible")

# if button1:

# Catégories à afficher
  cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

#palette de couleur
  color_pal12 = ['#92A8D1', '#034F84', '#F7CAC9', '#F7786B', '#DEEAEE', '#B1CBBB', '#EEA29A','#F0DA86', '#53B8B2', '#BEB9DB', '#FDCCE5', '#BD7EBE']

# Création des sous-graphiques
  fig = make_subplots(rows=3, cols=3, subplot_titles=cat_columns)

  counter = 0
  for cat_column in cat_columns:
    value_counts = df[cat_column].value_counts()
    x_pos = np.arange(0, len(value_counts))
    # Mélanger les couleurs de la palette de manière aléatoire
    random_colors = color_pal12.copy()
    random.shuffle(random_colors)
    # Appliquer les couleurs mélangées aux barres de la catégorie
    colors = [random_colors[i % len(random_colors)] for i in range(len(value_counts))]

    #SANS RANDOM
    # Appliquer les couleurs de la palette aux barres de la catégorie
    #colors = [color_pal12[i % len(color_pal12)] for i in range(len(value_counts))]

    trace_x = counter // 3 + 1
    trace_y = counter % 3 + 1

    # Ajout de la barre
    fig.add_trace(go.Bar(x=x_pos,y=value_counts.values,text=value_counts.values,textposition='auto',hoverinfo='text+x',name=cat_column,marker_color=colors,opacity=0.6,marker_line_width=2,showlegend=False,),row=trace_x,col=trace_y)
    # Mise en forme de l'axe x
    fig.update_xaxes(tickvals=x_pos,ticktext=value_counts.index,row=trace_x,col=trace_y)
    # Rotation des étiquettes de l'axe x
    fig.update_xaxes(tickangle=45, row=trace_x, col=trace_y)
    counter += 1

# Mise à jour de la mise en page du graphique
    fig.update_layout(height=800,width=1000,title_text="Distribution des modalités des variables catégorielles",)
# Affichage du graphique
  st.plotly_chart(fig)
#   st.write("")
#   fig = plt.figure()
#   sns.countplot(x = 'deposit', data = df)
#   st.pyplot(fig)

##FIGURE 2
# Variables numériques à afficher
#   num_columns = ['balance', 'campaign', 'pdays', 'previous']
# # Création des sous-graphiques
#   fig2 = make_subplots(rows=2, cols=2, subplot_titles=num_columns)
# # Position du subplot
#   row = 1
#   col = 1
# # Création des boxplots pour chaque variable numérique
#   for num_column in num_columns:
#     fig2.add_trace(
#         go.Violin(
#             x=df[num_column],
#             marker_color='#B1CBBB',  # Couleur des points
#             box_visible=True,  # Afficher la boîte à l'intérieur du violon
#             meanline_visible=True, # Afficher la moyenne
#             name=num_column,
#             showlegend=False  # Supprimer les étiquettes de légende
#         ),
#         row=row,
#         col=col
#     )
#     fig2.update_yaxes(title_text=num_column, row=row, col=col)
#     col += 1
#     if col > 2:
#         row += 1
#         col = 1
# # Mise à jour de la mise en page du graphique
#   fig2.update_layout(
#     height=800,
#     width=1000,
#     title_text="Les variables numériques avec les valeurs extrêmes"
# )
# Affichage du graphique
# fig.show()
  # st.plotly_chart(fig)





# PAGE 03 # Edition de la page "Préparation des données"
if page == pages[3] : 
  st.title("Prédiction du succès d’une campagne de Marketing d’une banque")
  st.subheader("Doublons et valeurs manquantes")
  st.write("Il n'y a pas de doublons dans l'ensemble des données (_df.duplicated().sum()_)")
  st.code(df.duplicated().sum())
  st.write("Il n'y a pas de valeurs manquantes dans l'ensemble des données (_df.isna().sum()_)")
  st.code(df.isna().sum())

# PAGE 04 # Edition de la page "Modélisation"
if page == pages[4] : 
  st.write()

# PAGE 05 # Edition de la page "Machine Learning"
if page == pages[5] : 
  st.write()

# PAGE 06 # Edition de la page "Conclusion et Perspective"
if page == pages[6] : 
  st.write()
