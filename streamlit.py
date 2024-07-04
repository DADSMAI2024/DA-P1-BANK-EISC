import streamlit as st
import random
import pandas as pd
import matplotlib as plt
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
# from imblearn.over_sampling import RandomOverSampler
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
pages=["Le projet","Le jeu de données", "Visualisations", "Préparation des données", "Modélisation", "Machine Learning", "Conclusion et Perspective"]
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
  
  st.write("**age**: _(num)_")
  st.write("**job**: _(cat)_ type d'emploi (admin., blue-collar, entrepreneur, housemaid, management, retired, self-employed, services, student, technician, unemployed, unknown)")
  st.write("**marital**: _(cat)_ état matrimonial (divorced, married, single)")
  st.write("**education**: _(cat)_ niveau d'étude (primary, secondary , tertiary, unknown)")
  st.write("**default**: _(cat)_ a un défaut de crédit ? (yes, no)")
  st.write("**housing**: _(cat)_ dispose d'un prêt au logement ? (yes, no)")
  st.write("**loan**: _(cat)_ a un prêt personnel ?  (yes, no)")
  st.write("**balance**: _(num)_ solde du compte en banque")
  st.write("**contact**: _(cat)_ type de communication du contact (cellular, telephone, unknown)")
  st.write("**month**: _(cat)_ dernier mois de contact (jan, feb, ... dec)")
  st.write("**day**: _(num)_ dernier jour de contact")
  st.write("**duration**: _(num)_ durée du dernier contact, en secondes")
  st.write("**campaign**: _(num)_ nombre de contacts effectués au cours de cette campagne et pour ce client")
  st.write("**pdays**: _(num)_ nombre de jours écoulés depuis le dernier contact du client lors d'une campagne précédente (-1 signifie que le client n'a pas été contacté précédemment)")
  st.write("**previous**: _(num)_ nombre de contacts effectués avant cette campagne et pour ce client")
  st.write("**poutcome**: _(cat)_ résultat de la précédente campagne marketing (failure, other, success, unknown)")
  st.write("**deposit**: _(cat/target)_ le client a, oui ou non, souscrit au dépôt à terme (yes, no)")

  dataset = st.checkbox("Afficher un extract du data set")
  if dataset:
    st.dataframe(df.head())

  st.write("")

  st.subheader("Résumé des statistiques descriptives des variables quantitatives")
  st.write(df.describe())
  st.subheader("Constat") 
  st.write("**age** : 50% des valeurs sont entre 32 et 49 ans. Beaucoup de valeurs extrêmes : max 95.")
  st.write("**balance** : 50% des valeurs sont entre 122 et 1708. Présence de valeurs négatives et de valeurs extrêmes : min -6 847, max 81 204.")
  st.write("**duration** : 50% des valeurs sont entre 138 sec (2min) et 496 (8min). Présence de valeurs extrêmes : max 3 881.")
  st.write("**campaign** :  50% des valeurs sont entre 1 et 3 contacts.Présence de valeurs extrêmes : max 63.")
  st.write("**pdays** : 50% des valeurs sont entre - 1 et 20. La médiane est à -1 ce qui signifie que la moitié des clients n'ont jamais été contacté avant cette campagne. Présence de valeurs extrêmes : max 854.")
  st.write("**previous** : 50% des valeurs sont entre 0 et 1. Présence de valeurs extrêmes : max 58.")

  st.write("")

# PAGE 02 # Edition de la page "Visualisations"
if page == pages[2] : 
  st.title("Visualisations")

  col1, col2, col3,col4 = st.columns([1, 1, 1, 1])  

  button1 = col1.button("La variable cible : deposit")
  button2 = col2.button("Les variables catégorielles")
  button3 = col3.button("Les variables numériques")
  button4 = col4.button("Var. explicatives VS cible")

  if button1:

    count_deposit = df['deposit'].value_counts()
    color_sequence = ['#5242EA', '#FACA5E']

# pie chart
    pie_chart = go.Pie(
    labels=count_deposit.index,
    values=count_deposit.values,
    marker=dict(colors=color_sequence),
    pull=[0.05,0]
    )

# bar chart
    bar_chart = go.Bar(
    x=count_deposit.index,
    y=count_deposit.values,
    text=count_deposit.values,
    textposition='auto',
    marker=dict(color=color_sequence),
    showlegend=False
)

# figure avec deux sous-plots
    fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "domain"}, {"type": "xy"}]],
    subplot_titles=("Distribution ", "Nombre de dépôts")
    )

# Ajouter pie chart et bar chart à la figure
    fig.add_trace(pie_chart, row=1, col=1)
    fig.add_trace(bar_chart, row=1, col=2)

# Mise à jour
    fig.update_layout(
    title_text="<b>Analyse de la variable cible : dépôt à terme ou non",
    legend_title= "<b>Dépôt"
    )

# Affichage
    st.plotly_chart(fig)

    st.subheader("Constat") 
    st.markdown("La répartition entre les clients qui ont souscrit à un dépôt à terme et ceux qui ne l'ont pas fait est relativement équilibrée, avec une différence de 5,2 %.")
    st.markdown("Toutefois, il y a légèrement plus de personnes qui n'ont pas contracté de dépôt à terme (52,6 %) par rapport à celles qui l'ont fait (47,4 %).")
  


##Variables catégorielles
  if button2:
    # Catégories à afficher
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    # Couleurs à appliquer
    colors = ['#5242EA', '#FACA5E', '#56CEB2', '#28DCE0']

    # Création des sous-graphiques
    fig = make_subplots(rows=3, cols=3, subplot_titles=cat_columns)

    counter = 0
    for cat_column in cat_columns:
        value_counts = df[cat_column].value_counts()
        x_pos = np.arange(0, len(value_counts))

        trace_x = counter // 3 + 1
        trace_y = counter % 3 + 1

        # Choisir la couleur cycliquement
        color = colors[counter % len(colors)]

        # Ajout de la barre
        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=value_counts.values,
                text=value_counts.values,
                textposition='auto',
                hoverinfo='text+x',
                name=cat_column,
                marker_color=color,
                opacity=0.8,
                marker_line_color=color,
                marker_line_width=1,
                showlegend=False,
            ),
            row=trace_x,
            col=trace_y
        )

        # Mise en forme de l'axe x
        fig.update_xaxes(
            tickvals=x_pos,
            ticktext=value_counts.index,

            row=trace_x,
            col=trace_y
        )

        # Rotation des étiquettes de l'axe x
        fig.update_xaxes(tickangle=45, row=trace_x, col=trace_y)

        counter += 1

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        height=800,
        width=1000,
        title_text="<b>Distribution des modalités des variables catégorielles",
    )

    # Affichage du graphique
    st.plotly_chart(fig)

    st.subheader("Constat") 
    st.markdown("""
        1. **Profession (job)** : Les professions les plus fréquentes sont “management”,“blue-collar” (ouvriers) et “technician”.
        2. **État civil (marital)** : La majorité des clients sont “married” (mariés).
        3. **Niveau d'études (education)** : La catégorie "secondary" (enseignement secondaire) est la plus fréquente parmi ceux qui ont souscrit au produit dépôt à terme.
        4. **Défaut de paiement (default)** : Très faible part des clients en défaut de paiement.
        5. **Crédit immobilier (housing)** : plutôt équilibré entre les clients ayant un crédit immobilier ou non.
        6. **Prêt personnel (loan)** : Très faible part de clients avec un prêt personnel.
        7. **Type de contact (contact)** : Le contact par mobile est le plus fréquent.
        8. **Mois de contact (month)** : Les mois de mai, juin, juillet, et août sont les mois avec le plus de contacts pour cette campagne.      
        8. **Résultat précédente campagne (poutcome)** : Une bonne partie des résultats de la précédente campagne est inconnue.
        """)


##Variables numériques
  if button3:
    # Variables numériques à afficher
    num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

    # Couleurs à appliquer
    colors = ['#5242EA', '#FACA5E', '#56CEB2', '#28DCE0']

    # Création des sous-graphiques
    fig = make_subplots(rows=2, cols=3, subplot_titles=num_columns)

    # Position du subplot
    row = 1
    col = 1

    # Création des histogrammes pour chaque variable numérique
    for i, num_column in enumerate(num_columns):
        color = colors[i % len(colors)]  # Choisir la couleur cycliquement
        fig.add_trace(
            go.Histogram(
                x=df[num_column],
                marker_color=color,  # Couleur des barres
                opacity=0.6,
                marker_line_width=0.5,
                showlegend=False,  # Supprimer les étiquettes de légende
                name=num_column
            ),
            row=row,
            col=col
        )

        fig.update_xaxes(title_text=num_column, row=row, col=col)
        fig.update_yaxes(title_text='Count', row=row, col=col)

        col += 1
        if col > 3:
            row += 1
            col = 1

          

    # Mise à jour de la mise en page du graphique
    fig.update_layout(
        height=800,
        width=1000,
        title_text="<b>Histogrammes des variables numériques"
    )

    # Affichage du graphique
    st.plotly_chart(fig)

    st.subheader("Constat") 
    st.markdown("""
        1. **Solde moyen du compte bancaire (balance)** : Forte concentration des données autour de 0. Présence de valeurs négatives et de valeurs extrêmes..
        2. **Jour de contact (days)** : la campagne de télémarketing semble avoir lieu tous les jours du mois, avec une baisse notable en moyenne le 10 du mois et entre le 22 et le 27 du mois. Il est à noter que cette variable est lissée sur tous les mois de plusieurs années, avec l'absence de l'information année, ni celle du jour de la semaine, ne nous permettant pas de déduire de grosses tendances à partir de cette variable.
        3. **Durée du contact (duration)** : exprimée en secondes, présence de valeurs extrêmes.
        4. **Nombre de contacts de la campagne(campaign)** : présence de valeurs extrêmes.
        5. **Nombre de jours depuis le contact précédent (pdays)** : forte présence de valeurs négatives, distribution asymétrique, et nombreuses valeurs extrêmes.
        6. **Nombre de contacts précédents (previous)** : Très forte concentration autour de 0 qui signifie pas de contacts précédemment et présence de valeurs extrêmes.
        """)

    # Headmap
    # Convertir la variable cible 'deposit' en numérique
    df['deposit_num'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

    # Sélection des variables numériques
    var_num_cible = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous','deposit_num']

    # Calcul de la matrice de corrélation
    corr_matrix_cible = df[var_num_cible].corr()

    # Création du heatmap avec Plotly
    fig = px.imshow(corr_matrix_cible, text_auto=True, aspect="auto", color_continuous_scale='Plasma')

    # Mise à jour du layout
    fig.update_layout(title="<b>Heatmap des Variables Numériques avec la variable cible deposit",
                      xaxis_title="Variables",
                      yaxis_title="Variables")


    # Affichage du graphique
    st.plotly_chart(fig)














# PAGE 02 # Edition de la page "Liste déroulante"
if page == pages[3] : 
  st.title("test list déroulante")

#   col1, col2, col3 = st.columns([1, 1, 1])  

#   button1 = col1.button("Volumétrie des commandes")
#   # button2 = col2.button("Analyse des revenus")
#   # button3 = col3.button("Analyse de la variable cible")

# if button1:

# Catégories à afficher
#   cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# #palette de couleur
#   color_pal12 = ['#92A8D1', '#034F84', '#F7CAC9', '#F7786B', '#DEEAEE', '#B1CBBB', '#EEA29A','#F0DA86', '#53B8B2', '#BEB9DB', '#FDCCE5', '#BD7EBE']

# # Sélection des variables à afficher avec une liste déroulante
#  selected_columns = st.multiselect('Sélectionnez les variables à afficher', cat_columns, default=cat_columns)

# # Sélection des variables à afficher avec une liste déroulante, 'job' est sélectionné par défaut
#   # selected_columns = st.multiselect('Sélectionnez les variables à afficher', cat_columns)


# # Calcul du nombre de lignes et de colonnes nécessaires pour les sous-graphiques
#   num_cols = 3
#   num_rows = (len(selected_columns) + num_cols - 1) // num_cols  # arrondir au nombre entier supérieur

# # Création des sous-graphiques
#   fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=selected_columns)

# # Création des sous-graphiques
# #  fig = make_subplots(rows=3, cols=3, subplot_titles=selected_columns)

#   # Création des sous-graphiques
#   if len(selected_columns) == 1:
#     fig = make_subplots(rows=2, cols=1, subplot_titles=selected_columns)
#   else:
#     fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=selected_columns)

#   counter = 0
#   for cat_column in selected_columns:
#     value_counts = df[cat_column].value_counts()
#     x_pos = np.arange(0, len(value_counts))


#     # Mélanger les couleurs de la palette de manière aléatoire
#     random_colors = color_pal12.copy()
#     random.shuffle(random_colors)
#     # Appliquer les couleurs mélangées aux barres de la catégorie
#     colors = [random_colors[i % len(random_colors)] for i in range(len(value_counts))]

#     #SANS RANDOM
#     # Appliquer les couleurs de la palette aux barres de la catégorie
#     #colors = [color_pal12[i % len(color_pal12)] for i in range(len(value_counts))]

#     trace_x = counter // num_cols + 1
#     trace_y = counter % num_cols + 1

#     # Ajout de la barre
#     fig.add_trace(go.Bar(
#         x=x_pos,
#         y=value_counts.values,
#         text=value_counts.values,
#         textposition='auto',
#         hoverinfo='text+x',
#         name=cat_column,
#         marker_color=colors,
#         opacity=0.6,
#         marker_line_width=2,
#         showlegend=False,), row=trace_x, col=trace_y)


#   # Mise en forme de l'axe x
#     fig.update_xaxes(tickvals=x_pos, ticktext=value_counts.index, row=trace_x, col=trace_y)
        
#     # Rotation des étiquettes de l'axe x
#     fig.update_xaxes(tickangle=45, row=trace_x, col=trace_y)
#     counter += 1

# # Mise à jour de la mise en page du graphique
#     fig.update_layout(height=800,width=1000,title_text="Distribution des modalités des variables catégorielles",)


# # Affichage du graphique
#   st.plotly_chart(fig)

