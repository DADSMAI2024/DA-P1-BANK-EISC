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
pages=["Le projet","Le jeu de données", "Visualisations", "Pre-processing des données", "Machine Learning", "Conclusion et Perspective"]
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
  button4 = col4.button("Var. explicatives VS la variable cible")
  
  if button1:
    count_deposit = df['deposit'].value_counts()
    color_sequence = ['#FACA5E','#5242EA']

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

    st.markdown("La répartition entre les clients qui ont souscrit à un dépôt à terme et ceux qui ne l'ont pas fait est relativement équilibrée, avec une différence de 5.2 points. Toutefois, il y a légèrement plus de personnes qui n'ont pas contracté de dépôt à terme (52,6 %) par rapport à celles qui l'ont fait (47,4 %).")
 


##Variables catégorielles
  if button2:
    # Catégories à afficher
    cat_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

    #palette de couleur
    color_pal4 = ['#56CEB2', '#28DCE0','#57CF8A','#579DCF']

    # Création des sous-graphiques
    fig = make_subplots(rows=3, cols=3, subplot_titles=cat_columns)

    #Fonction d'application des couleurs
    counter = 0
    for cat_column in cat_columns:
        value_counts = df[cat_column].value_counts()
        x_pos = np.arange(0, len(value_counts))

        # Mélanger les couleurs de la palette de manière aléatoire
        random_colors = color_pal4.copy()
        random.shuffle(random_colors)
        # Appliquer les couleurs mélangées aux barres de la catégorie
        colors = [random_colors[i % len(random_colors)] for i in range(len(value_counts))]

        trace_x = counter // 3 + 1
        trace_y = counter % 3 + 1


        # Ajout de la barre
        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=value_counts.values,
                text=value_counts.values,
                textposition='auto',
                hoverinfo='text+x',
                name=cat_column,
                marker_color=colors,
                opacity=0.8,
                marker_line_color=colors,
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
    # Variables numériques à afficher
    num_columns = ['balance', 'day', 'duration', 'campaign', 'pdays', 'previous']


    # Création des sous-graphiques
    fig = make_subplots(rows=2, cols=3, subplot_titles=num_columns)

    # Position du subplot
    row = 1
    col = 1


    # Création des histogrammes pour chaque variable numérique
    for num_column in num_columns:
        fig.add_trace(
            go.Histogram(
                x=df[num_column],
                marker_color='#56CEB2',
                opacity=0.6,
                marker_line_width=0.5,
                showlegend=False,
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
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")

    # Headmap
    # Convertir la variable cible 'deposit' en numérique

    # Convertir la variable cible 'deposit' en numérique
    df['deposit_num'] = df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

    # Sélection des variables numériques
    var_num_cible = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous','deposit_num']

    # Calcul de la matrice de corrélation
    corr_matrix_cible = df[var_num_cible].corr()

    # Création du heatmap avec Plotly
    fig = px.imshow(corr_matrix_cible, text_auto=True, aspect="auto", color_continuous_scale='Turbo')

    # Mise à jour du layout
    fig.update_layout(title="<b>Heatmap des Variables Numériques avec la variable cible deposit",
                    xaxis_title="Variables",
                    yaxis_title="Variables")

    # Affichage du graphique
    st.plotly_chart(fig)



    st.subheader("Constat") 
    st.markdown("Dans ce graphique de corrélation, on note un lien entre les variables **pdays** et **previous** ; ce qui semble cohérent puisque **pdays** représente le nombre de jours depuis le dernier contact client et **previous** représente le nombre de contacts précédant cette campagne.")
    st.markdown("La variable **duration** - _durée du contact client durant la campagne_ - semble influencer la variable cible deposit. Nous étudierons plus spécifiquement cette variable exprimée en secondes, et présent des valeurs extrêmes.")
    st.markdown("Dans une très moindre mesure, les variables **pdays**, **previous** et **balance** semble légèrement influencer la variable cible deposit.")


##Variables explicatives VS variable Cible
  if button4:
    st.markdown("\n")
    st.markdown("Notre analyse se décompose en 4 grands axes :")
 
    # Création des liens vers les paragraphes
    st.markdown('''
    - [Le profil client](#le-profil-client)
    - [Le profil bancaire](#le-profil-bancaire)
    - [Analyse des contacts clients durant la campagne télémarketing](#analyse-des-contacts)
    - [Analyse de la campagne précédente et son influence sur la campagne actuelle](#analyse-de-la-campagne-precedente)
    ''', unsafe_allow_html=True)
    
    st.markdown("\n")
    st.markdown("\n")
    #DEBUT DU PROFIL CLIENT - GRAPH AGE
    # Utiliser st.markdown avec du HTML pour souligner le texte
    st.markdown("<h2 id='le-profil-client'>Le Profil Client</h2>", unsafe_allow_html=True)
    st.markdown("Nous pouvons constater ci-dessous que les clients qui ont souscrits au dépôt à terme sont en moyenne plus âgés que ceux n'ayant pas souscrit (78 ans contre 70 ans).")
    st.markdown("Le nuage de points qui suit met en exergue que ceux n'ayant pas souscrit sont plus dispersés après 60 ans. Nous constatons également la présence de nombreuses valeurs extrêmes (outliers).")

    # LE GRAPHIQUE : Le nuage de points

    # Définir les couleurs spécifiques pour chaque catégorie
    color_sequence = ['#5242EA', '#FACA5E']

    # 1er graphique : Distribution de l'âge versus dépôt
    fig1 = px.box(df, x='age', y='deposit', points='all',
                color='deposit',
                title="Distribution de l'âge versus dépôt",
                color_discrete_sequence=color_sequence,
                labels={'deposit': 'Statut de Dépôt'},
                category_orders={"deposit": ["yes", "no"]}  #"yes" est avant "no"
                )

    
    # Assemblage des graphiques
    
    fig = make_subplots(rows=1, cols=1, subplot_titles=[
        ""
    ])

    # Ajouter fig1 sans légende pour éviter les doublons
    for trace in fig1['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    

    # Mise à jour de la mise en page
    fig.update_layout(
        height=500,
        width=1500,
        title_text="<b>Analyse de l'âge en fonction de deposit",
        showlegend=True,
        bargap=0.1,
        legend=dict(
            title="Dépôt")
    )

    fig.update_xaxes(title_text='Âge du client', row=1, col=1)
    fig.update_yaxes(title_text='Deposit', row=1, col=1)

    fig.update_xaxes(title_text='Âge du client', row=1, col=2)
    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)


    # Affichage du graphique
    st.plotly_chart(fig)

    # LE GRAPHIQUE : Les âges
    color_sequence = ['#5242EA', '#FACA5E']  # bleu pour "yes" et jaune pour "no"

    # 2ème graphique : Répartition des dépôts en fonction de l'âge
    count_deposit = df.groupby(['age', 'deposit']).size().reset_index(name='count')
    fig2 = px.bar(count_deposit, x='age', y='count', color='deposit',
                barmode='group',
                title="Répartition des dépôts en fonction de l'âge",
                labels={'age': 'Âge', 'count': 'Nombre de dépôts', 'deposit': 'Statut de Dépôt'},
                category_orders={"deposit": ["yes", "no"]},  # "yes" est avant "no"
                color_discrete_sequence=color_sequence
                )

    # Affichage du graphique
    st.plotly_chart(fig2)

    st.markdown("Ci-dessus, il apparaît nettement  une plus forte proportion de dépôt à terme chez les moins de 30 ans et chez les plus de 60 ans.")
    st.markdown("Pour la suite de l'analyse, nous avons fait le choix de discrétiser la variable _'age'_ par tranches d'âge pour atténuer le rôle des valeurs extrêmes et pour afficher ensuite plusieurs graphiques par catégorie.")
    st.write("\n")
    st.write("\n")


    #SUITE DU PROFIL CLIENT - SUBPLOT
    df['age_cat'] = pd.cut(df.age, bins=[18,29,40,50,60,96], labels = ['18-29ans','30-40ans','40-50ans','50-60ans','Plus de 60 ans'])
    df['age_cat'].value_counts()
    
    # Définir les couleurs spécifiques pour chaque catégorie
    color_sequence = ['#5242EA', '#FACA5E']

    ## 1ER GRAPHIQUE AGE
    # Calculer les décomptes pour chaque tranche d'âge et deposit
    counts_age = df.groupby(['age_cat', 'deposit']).size().unstack()
    # Calculer les pourcentages
    total_counts_age = counts_age.sum(axis=1)
    percent_yes_age = (counts_age['yes'] / total_counts_age * 100).round(2)
    percent_no_age = (counts_age['no'] / total_counts_age * 100).round(2)
    # Transformer les données
    df_plot_age = pd.melt(counts_age.reset_index(), id_vars=['age_cat'], value_vars=['yes', 'no'],
                    var_name='deposit', value_name='count')
    # Ajouter les pourcentages calculés
    df_plot_age['percent'] = percent_yes_age.tolist() + percent_no_age.tolist()

    # Créer le graphique avec Plotly Express
    fig_age = px.bar(df_plot_age, x='age_cat', y='count', color='deposit', barmode='group',
                title="Répartition des dépôts en fonction de la tranche d'âge",
                labels={'age_cat': 'Age', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                category_orders={"age_cat": ['Jeune <30 ans','Jeune Adulte 30<40 ans', 'Adulte 40<50 ans', 'Senior 50<60 ans','Aînés >60 ans']},
                text=df_plot_age['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                )

    # Mettre à jour le layout
    fig_age.update_layout(yaxis_title="Nombre de dépôts",
                    legend_title_text='Statut du dépôt',
                    xaxis_tickangle=30)


    ##2EME GRAPHIQUE JOB
    # Calculer les décomptes pour chaque catégorie de job et deposit
    counts_job = df.groupby(['job', 'deposit']).size().unstack()
    #classer par ordre de valeurs
    job_order = df.groupby('job')['deposit'].count().reset_index(name='total_deposits')
    job_order = job_order.sort_values(by='total_deposits', ascending=False)['job']
    # Convertir en liste pour utilisation dans category_orders
    job_order = job_order.tolist()
    # Calculer les pourcentages
    total_counts_job = counts_job.sum(axis=1)
    percent_yes_job = (counts_job['yes'] / total_counts_job * 100).round(2)
    percent_no_job = (counts_job['no'] / total_counts_job * 100).round(2)

    # Transformer les données pour Plotly Express
    df_plot_job = pd.melt(counts_job.reset_index(), id_vars=['job'], value_vars=['yes', 'no'],
                    var_name='deposit', value_name='count')

    # Ajouter les pourcentages calculés
    df_plot_job['percent'] = percent_yes_job.tolist() + percent_no_job.tolist()

    # Créer le graphique avec Plotly Express
    fig_job = px.bar(df_plot_job, x='job', y='count', color='deposit', barmode='group',
                title="Répartition des dépôts en fonction du type d'emploi",
                labels={'job': 'Métier', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                category_orders={'job': job_order},
                color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                hover_data={'count': True}  # afficher les détails au survol
                )

    # Mettre à jour le layout
    fig_job.update_layout(yaxis_title="Nombre de dépôts",
                    legend_title_text='Statut du dépôt',
                    xaxis_tickangle=30,
                    bargap=0.1)


    ##3EME GRAPHIQUE MARITAL
    # Ordre affichage
    marital_order = ['married', 'single', 'divorced']
    # Calculer les décomptes pour chaque catégorie de marital et deposit
    counts_marital = df.groupby(['marital', 'deposit']).size().unstack()
    # Calculer les pourcentages
    total_counts_marital = counts_marital.sum(axis=1)
    percent_yes_marital = (counts_marital['yes'] / total_counts_marital * 100).round(2)
    percent_no_marital = (counts_marital['no'] / total_counts_marital * 100).round(2)
    # Transformer les données pour Plotly Express
    df_plot_marital = pd.melt(counts_marital.reset_index(), id_vars=['marital'], value_vars=['yes', 'no'],
                    var_name='deposit', value_name='count')

    # Ajouter les pourcentages calculés
    df_plot_marital['percent'] = percent_yes_marital.tolist() + percent_no_marital.tolist()

    # Créer le graphique avec Plotly Express
    fig_marital = px.bar(df_plot_marital, x='marital', y='count', color='deposit', barmode='stack',
                title="Répartition des dépôts en fonction du statut marital",
                labels={'marital': 'Statut marital', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                category_orders={'marital': marital_order},
                text=df_plot_marital['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                )

    # Mettre à jour le layout
    fig_marital.update_layout(yaxis_title="Nombre de dépôts",
                    legend_title_text='Statut du dépôt',
                    xaxis_tickangle=30)



    ##4EME GRAPHIQUE EDUCATION
    # Classer par ordre de valeurs
    education_order = df.groupby('job')['deposit'].count().reset_index(name='total_deposits')
    education_order = education_order.sort_values(by='total_deposits', ascending=False)['job']
    # Convertir en liste pour utilisation dans category_orders
    education_order = education_order.tolist()
    # Calculer les décomptes pour chaque catégorie de education et deposit
    counts_education = df.groupby(['education', 'deposit']).size().unstack()
    # Calculer les pourcentages
    total_counts_education = counts_education.sum(axis=1)
    percent_yes_education = (counts_education['yes'] / total_counts_education * 100).round(2)
    percent_no_education = (counts_education['no'] / total_counts_education * 100).round(2)
    # Transformer les données pour Plotly Express
    df_plot_education = pd.melt(counts_education.reset_index(), id_vars=['education'], value_vars=['yes', 'no'],
                    var_name='deposit', value_name='count')

    # Ajouter les pourcentages calculés
    df_plot_education['percent'] = percent_yes_education.tolist() + percent_no_education.tolist()

    # Créer le graphique avec Plotly Express
    fig_education = px.bar(df_plot_education, x='education', y='count', color='deposit', barmode='stack',
                title="Répartition des dépôts en fonction du niveau d'études ",
                labels={'education': "Niveau d'études", 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                category_orders={'education': education_order},
                text=df_plot_education['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                )

    # Mettre à jour le layout
    fig_education.update_layout(yaxis_title="Nombre de dépôts",
                    legend_title_text='Statut du dépôt',
                    xaxis_tickangle=30)


    ##CREATION SUBPLOTS
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Répartition des tranches d'âge en fonction du dépôt",
            "Répartition des jobs en fonction du dépôt",
            "Répartition statut marital en fonction du dépôt",
            "Répartition niveau d'études en fonction du dépôt"
        ),
        horizontal_spacing=0.2,  # Espace horizontal entre les subplots
        vertical_spacing=0.2     # Espace vertical entre les subplots
    )

    # Ajouter fig age
    for trace in fig_age['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    # Ajouter fig job
    for trace in fig_job['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)


    # Ajouter fig martital
    for trace in fig_marital['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=2, col=1)


    # Ajouter fig education
    for trace in fig_education['data']:
        fig.add_trace(trace, row=2, col=2)


    # Mettre à jour les axes avec les orders spécifiés
    fig.update_xaxes(categoryorder='array', categoryarray=job_order, row=1, col=2)
    fig.update_xaxes(categoryorder='array', categoryarray=marital_order, row=2, col=1)

    # Mise à jour de la mise en page
    fig.update_layout(
        height=900,
        width=1200,
        title_text="<b>Analyse du profil client selon les résultats de dépôt",
        legend_title= "Dépôt"
        )

    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

    #fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)

    fig.update_xaxes(title_text='Statut Marital', row=2, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)

    fig.update_xaxes(title_text="Niveau d'études", row=2, col=2)
    #fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=2)

    ## AFFICHER LA FIGURE
    st.plotly_chart(fig)

    st.markdown("**Âge**: Une tendance significative se dégage parmi les jeunes et les aînés à souscrire aux dépôts à terme, avec près de 60 % des moins de 30 ans et environ 82 % des plus de 60 ans ayant opté pour cette option.")
    st.markdown("**Emploi** : Alors que les managers, ouvriers, techniciens et administratifs représentent une part substantielle des clients de la banque, les retraités, étudiants, sans emploi et managers sont plus enclins à souscrire au dépôt à terme.")
    st.write("**Statut marital**: Bien que les clients mariés constituent une proportion significative de la clientèle, les célibataires montrent une plus forte propension à souscrire au dépôt, avec plus de 54 % d'entre eux ayant opté pour cette option.")
    st.write("**Niveau d'études**: Bien que la majorité des clients ait un niveau d'études secondaire, une proportion plus élevée de souscripteurs au dépôt est observée parmi ceux ayant un niveau d'études supérieur (tertiaire), atteignant 54 %. En revanche, les niveaux d'études inférieurs sont associés à des taux moindres de souscription.")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")


    #le-profil-bancaire

    #st.markdown("<h2 id='le-profil-client'>Le Profil Bancaire</h2>", unsafe_allow_html=True)
    st.markdown("<h2 id='le-profil-bancaire'>Le Profil Bancaire</h2>", unsafe_allow_html=True)
     # 1ER GRAPHIQUE DEFAULT
    counts_default = df.groupby(['default', 'deposit']).size().unstack()
    total_counts_default = counts_default.sum(axis=1)
    percent_yes_default = (counts_default['yes'] / total_counts_default * 100).round(2)
    percent_no_default = (counts_default['no'] / total_counts_default * 100).round(2)
    df_plot_default = pd.melt(counts_default.reset_index(), id_vars=['default'], value_vars=['yes', 'no'],
                            var_name='deposit', value_name='count')
    df_plot_default['percent'] = percent_yes_default.tolist() + percent_no_default.tolist()
    fig_default = px.bar(df_plot_default, x='default', y='count', color='deposit', barmode='stack',
                        title="Répartition des dépôts en fonction du défaut de paiement",
                        labels={'default': 'Défaut de paiement', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                        text=df_plot_default['percent'].apply(lambda x: f"{x:.2f}%"),
                        color_discrete_sequence=['#5242EA', '#FACA5E'],
                        hover_data={'count': True, 'percent': ':.2f%'}
                        )
    fig_default.update_layout(yaxis_title="Nombre de dépôts", legend_title_text='Statut du dépôt')

    # 2EME GRAPHIQUE LOAN
    counts_loan = df.groupby(['loan', 'deposit']).size().unstack()
    total_counts_loan = counts_loan.sum(axis=1)
    percent_yes_loan = (counts_loan['yes'] / total_counts_loan * 100).round(2)
    percent_no_loan = (counts_loan['no'] / total_counts_loan * 100).round(2)
    df_plot_loan = pd.melt(counts_loan.reset_index(), id_vars=['loan'], value_vars=['yes', 'no'],
                        var_name='deposit', value_name='count')
    df_plot_loan['percent'] = percent_yes_loan.tolist() + percent_no_loan.tolist()
    fig_loan = px.bar(df_plot_loan, x='loan', y='count', color='deposit', barmode='stack',
                    title="Répartition des dépôts en fonction du prêt personnel",
                    labels={'loan': 'Prêt personnel', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                    text=df_plot_loan['percent'].apply(lambda x: f"{x:.2f}%"),
                    color_discrete_sequence=['#5242EA', '#FACA5E'],
                    hover_data={'count': True, 'percent': ':.2f%'}
                    )
    fig_loan.update_layout(yaxis_title="Nombre de dépôts", legend_title_text='Statut du dépôt')

    # 3EME GRAPHIQUE HOUSING
    counts_housing = df.groupby(['housing', 'deposit']).size().unstack()
    total_counts_housing = counts_housing.sum(axis=1)
    percent_yes_housing = (counts_housing['yes'] / total_counts_housing * 100).round(2)
    percent_no_housing = (counts_housing['no'] / total_counts_housing * 100).round(2)
    df_plot_housing = pd.melt(counts_housing.reset_index(), id_vars=['housing'], value_vars=['yes', 'no'],
                            var_name='deposit', value_name='count')
    df_plot_housing['percent'] = percent_yes_housing.tolist() + percent_no_housing.tolist()
    fig_housing = px.bar(df_plot_housing, x='housing', y='count', color='deposit', barmode='stack',
                        title="Répartition des dépôts en fonction du Prêt immobilier",
                        labels={'housing': 'Prêt immobilier', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                        text=df_plot_housing['percent'].apply(lambda x: f"{x:.2f}%"),
                        color_discrete_sequence=['#5242EA', '#FACA5E'],
                        hover_data={'count': True, 'percent': ':.2f%'}
                        )
    fig_housing.update_layout(yaxis_title="Nombre de dépôts", legend_title_text='Statut du dépôt')

    # 4EME GRAPHIQUE BALANCE
    fig_balance = px.box(df, x='deposit', y='balance',
                        color='deposit',
                        title="Distribution du solde moyen de compte",
                        color_discrete_sequence=['#5242EA', '#FACA5E'],
                        labels={'deposit': 'Statut de Dépôt'},
                        category_orders={"deposit": ["yes", "no"]}
                        )

    # CREATION SUBPLOTS
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Défaut de paiement",
            "Prêt personnel",
            "Prêt immobilier",
            "Solde moyen de compte"
        )
    )

    # Ajouter fig default
    for trace in fig_default['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)
    # Ajouter fig loan
    for trace in fig_loan['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)
    # Ajouter fig housing
    for trace in fig_housing['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=2, col=1)
    # Ajouter fig balance
    for trace in fig_balance['data']:
        fig.add_trace(trace, row=2, col=2)

    # Mise à jour de la mise en page
    fig.update_layout(
        height=800,
        width=1200,
        title_text="<b>Analyse du profil bancaire selon les résultats de deposit</b>",
        legend_title= "Dépôt"
    )

    fig.update_xaxes(title_text='Défaut de paiement', row=1, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

    fig.update_xaxes(title_text='Prêt personnel', row=1, col=2)
    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=2)

    fig.update_xaxes(title_text='Prêt immobilier', row=2, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)

    fig.update_xaxes(title_text='Statut de Dépôt', row=2, col=2)
    fig.update_yaxes(title_text='Solde moyen de compte', row=2, col=2)

    # AFFICHER LA FIGURE
    st.plotly_chart(fig)

    st.subheader("Constat") 
    st.markdown("**Défaut de paiement (default)**: Une très faible proportion de clients est en défaut de paiement dans l'échantillon. Cependant, ceux qui sont en défaut montrent une nette réticence à souscrire au dépôt à terme : 69 % d'entre eux n'ont pas souscrit, tandis que 52 % des autres clients ont souscrit, correspondant à la moyenne générale des souscripteurs.")
    st.markdown("**Prêt personnel (loan)**: Bien que la part de clients ayant un prêt personnel soit faible, il est observé que ceux-ci sont moins enclins à souscrire au dépôt : 69 % n'ont pas souscrit, comparé à 50,4 % des autres clients, se rapprochant de la moyenne générale des souscripteurs.")
    st.markdown("**Prêt immobilier (housing)**: Une tendance marquée se dégage ici où la majorité des clients sans crédit immobilier en cours souscrivent au dépôt à terme (57 % d'entre eux), tandis que ceux ayant un crédit immobilier sont moins enclins à souscrire : 63 % d'entre eux n'ont pas souscrit.")
    st.markdown("**Solde moyen de compte (balance)**: Les données montrent une dispersion étendue avec de nombreuses valeurs extrêmes. Les clients ayant souscrit au dépôt présentent un solde médian de compte plus élevé à 733 €, comparé à 414 € pour ceux n'ayant pas souscrit.")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")

    #analyse-des-contacts

    st.markdown("<h2 id='analyse-des-contacts'>Analyse des contacts clients durant la campagne télémarketing</h2>", unsafe_allow_html=True) 
    #GRAPHIQUE CONTACT
    # Calculer les décomptes pour chaque catégorie de contact et deposit
    counts_contact = df.groupby(['contact', 'deposit']).size().unstack(fill_value=0)
    # Calculer les pourcentages
    total_counts_contact = counts_contact.sum(axis=1)
    percent_yes_contact = (counts_contact['yes'] / total_counts_contact * 100).round(2)
    percent_no_contact = (counts_contact['no'] / total_counts_contact * 100).round(2)
    # Transformer les données pour Plotly Express
    df_plot_contact = pd.melt(counts_contact.reset_index(), id_vars=['contact'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')
    # Ajouter les pourcentages calculés
    df_plot_contact['percent'] = percent_yes_contact.tolist() + percent_no_contact.tolist()
    counts_contact = df.groupby(['contact', 'deposit']).size().unstack()

    # Créer le graphique
    fig_contact = px.bar(df_plot_contact, x='contact', y='count', color='deposit', barmode='group',
                title="Mode de contact client et résultats de deposit",
                labels={'contact': 'Mode de contact client', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                color_discrete_sequence=['#5242EA', '#FACA5E'],
                hover_data={'count': True, 'percent': ':.2f%'}
                )

    # Mettre à jour le layout
    fig_contact.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt')


    ## 2 GRAPHIQUE DURATION
    fig_duration=px.box(df,
              x='duration',  # Change 'duration_minutes' to 'duration'
              y='deposit',
              color='deposit',
              color_discrete_sequence=['#5242EA', '#FACA5E'],
              title='<b><b> Influence de la durée de contact sur le résultat de la campagne')

    ## 3 GRAPHIQUE MONTH
    # Calculer le nombre total de dépôts pour chaque mois
    month_order = df.groupby('month')['deposit'].count().reset_index(name='total_deposits')
    month_order = month_order.sort_values(by='total_deposits', ascending=False)['month']

    # Convertir en liste pour utilisation dans category_orders
    month_order = month_order.tolist()

    # Création de l'histograme
    fig_month = px.histogram(df, x='month', color='deposit', barmode='group',
                      title="Répartition de deposit en fonction des mois",
                      labels={'month': 'Mois de contact', 'count': 'Nombre de dépôts', 'deposit': 'Dépôt'},
                      category_orders={"month": month_order},
                      color_discrete_sequence=['#5242EA', '#FACA5E'])

    # Mettre à jour le layout
    fig_month.update_layout(yaxis_title="Nombre de dépôts",
                      legend_title_text='Statut du dépôt',
                      xaxis_tickangle=30,
                      bargap=0.1)



    ## 4 GRAPHIQUE M CONTACT
    # Grouper par mois et agréger les décomptes
    data_month = df.groupby('month').agg(
        campaign_count=('campaign', 'sum'),
        deposit_yes_count=('deposit', lambda x: (x == 'yes').sum()),
        deposit_no_count=('deposit', lambda x: (x == 'no').sum())
    ).reset_index()
    # Ajouter une nouvelle colonne avec des valeurs manuelles
    manual_values = [4, 8, 12, 2, 1, 7, 6, 3, 5, 11, 10, 9]
    # Assigner les valeurs manuelles à la colonne 'manual_order'
    data_month['manual_order'] = manual_values
    # Tri du DataFrame par la colonne 'manual_order'
    data_month_sorted = data_month.sort_values(by='manual_order').reset_index(drop=True)
    # Création du graphique
    fig_m_contact = px.line()
    # Ajout des courbes sur le graphique
    fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['campaign_count'], mode='lines', name='Nombre de contact', line=dict(color='#034F84', dash='dash'))
    fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_yes_count'], mode='lines', name='Dépôts Yes', line=dict(color='#5242EA'))
    fig_m_contact.add_scatter(x=data_month_sorted['month'], y=data_month_sorted['deposit_no_count'], mode='lines', name='Dépôts No', line=dict(color='#FACA5E'))
    fig_m_contact.update_layout(title='Nombre de contacts et dépôts par mois')
    # Ajout des axes
    fig_m_contact.update_xaxes(title_text='Mois')
    fig_m_contact.update_yaxes(title_text='Nombre de contacts')

    ###-----------

    ##CREATION SUBPLOTS
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Type de contact (contact)",
            "Durée du contact (duration)",
            "Mois de contact (month)",
            "Nombre de contacts et dépôts par mois"
        ),
        horizontal_spacing=0.1,  # Espace horizontal entre les subplots
        vertical_spacing=0.2     # Espace vertical entre les subplots
    )

    # Ajouter fig contact
    for trace in fig_contact['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    # Ajouter fig duration
    for trace in fig_duration['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Ajouter fig month
    for trace in fig_month['data']:
        fig.add_trace(trace, row=2, col=1)

    # Ajouter fig m_contact
    for trace in fig_m_contact['data']:
        fig.add_trace(trace, row=2, col=2)


    # Mise à jour de la mise en page
    fig.update_layout(
        height=800,
        width=1600,
        title_text="<b>Analyse de la campagne : type de contact, nombres de contacts, période et durée",
        legend_title= "Dépôt"
        )

    fig.update_xaxes(title_text='Modalité de contact', row=1, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

    fig.update_xaxes(title_text='Durée de contact en minutes', row=1, col=2)
    #fig.update_yaxes(title_text='Nombres de dépôts', row=1, col=2)

    fig.update_xaxes(title_text='Mois de contact', row=2, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)

    fig.update_xaxes(title_text='Mois de contact', row=2, col=2)
    #fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=2)

    # Affichage du graphique
    st.plotly_chart(fig)

    
    st.subheader("Constat") 
    st.markdown("**Contacts précédents** : Une forte proportion des clients n'ont pas été contacté précédemment. Néanmoins, il est intéressant de noter que les clients ayant déjà été contactés avant cette campagne (lors d'une campagne précédente) sont plus enclins à souscrire au dépôt : 67% des clients contactés précédemment, ont souscrit au dépôt lors de cette campagne, et inversement ceux n'ayant pas été contactés précédemment ont été près de 60% à ne pas souscrire au dépôt durant cette campagne. Ceci indique que la multiplication des contacts sur différentes campagnes peut inciter les clients et influencer la réussite d'une campagne suivante.")
    st.markdown("**Nombre de jours depuis le dernier contact** : On peut remarquer que moins de temps a passé depuis le dernier contact chez les clients souscrivant au dépôt sur cette campagne. Avec une étendue moins large (entre 94 et 246 jours) que ceux n'ayant pas souscrit au dépôt (étendue 148 à 332 jours). En sus, on peut constater de nombreuses valeurs extrêmes, notamment chez ceux ayant souscrit au dépôt.")
    st.markdown("**Succès de la précédente campagne** : Une grande part des données sont inconnues. Il est tout de même intéressant de noter qu'un client ayant souscrit à un produit d'une campagne précédente (success), sont très enclins à souscrire au dépôt de la campagne actuelle : 91% d'entre eux ont souscrit au dépôt.")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")


    st.markdown("<h2 id='analyse-de-la-campagne-precedente'>Analyse de la campagne précédente et son influence sur la campagne actuelle</h2>", unsafe_allow_html=True)
    #GRAPHIQUE 1 CONTACT OR NO CONTACT
    # Diviser en deux groupes
    df['group'] = df['previous'].apply(lambda x: 'non contactés' if x == 0 else 'contactés')

    # Compter les valeurs de deposit pour chaque groupe
    count_df = df.groupby(['group', 'deposit']).size().reset_index(name='count')

    # Calculer les pourcentages
    total_counts = count_df.groupby('group')['count'].transform('sum')
    count_df['percentage'] = (count_df['count'] / total_counts * 100).round(2)

    # Création du bar plot avec Plotly Express
    fig_previous = px.bar(
        count_df,
        x='group',
        y='count',
        color='deposit',
        text=count_df['percentage'].astype(str) + '%',
        color_discrete_sequence=['#5242EA', '#FACA5E'],
    )



    #GRAPHIQUE 2 PDAYS
    # Filtrer les données pour exclure les valeurs de 'pdays' égales à -1
    df_filtered = df[df['pdays'] != -1]

    # Créer le box plot
    fig_pdays = px.box(df_filtered,
                x='deposit',
                y='pdays',
                color='deposit',
                color_discrete_sequence=['#5242EA', '#FACA5E'],
                )




    #GRAPHIQUE 3 POUTCOME
    # Calculer les décomptes pour chaque catégorie de poutcome et deposit
    counts_poutcome = df.groupby(['poutcome', 'deposit']).size().unstack()
    # Calculer les pourcentages
    total_counts_poutcome = counts_poutcome.sum(axis=1)
    percent_yes_poutcome = (counts_poutcome['yes'] / total_counts_poutcome * 100).round(2)
    percent_no_poutcome = (counts_poutcome['no'] / total_counts_poutcome * 100).round(2)
    # Transformer les données pour Plotly Express
    df_plot_poutcome = pd.melt(counts_poutcome.reset_index(), id_vars=['poutcome'], value_vars=['yes', 'no'],
                      var_name='deposit', value_name='count')

    # Ajouter les pourcentages calculés
    df_plot_poutcome['percent'] = percent_yes_poutcome.tolist() + percent_no_poutcome.tolist()

    # Créer le graphique avec Plotly Express
    fig_poutcome = px.bar(df_plot_poutcome, x='poutcome', y='count', color='deposit', barmode='group',
                text=df_plot_poutcome['percent'].apply(lambda x: f"{x:.2f}%"),  # ajouter le signe % aux pourcentages
                color_discrete_sequence=['#5242EA', '#FACA5E'],  # configurer les couleurs correctes
                hover_data={'count': True, 'percent': ':.2f%'}  # afficher les détails au survol
                )


    ##CREATION SUBPLOTS
    fig = make_subplots(
        rows=1, cols=3,
        #subplot_titles=(
            #"Contacts précédents ou non",
            #"Nombre de jours depuis le dernier contact",
            #"Succès de la précédente campagne",
            #"Influence du nombre de jours depuis le dernier contact"
        #),
        horizontal_spacing=0.2,  # Espace horizontal entre les subplots
        vertical_spacing=0.1     # Espace vertical entre les subplots

    )

    # Ajouter fig previous
    for trace in fig_previous['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=1)

    # Ajouter fig pdays
    for trace in fig_pdays['data']:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    # Ajouter fig poutcome
    for trace in fig_poutcome['data']:
        fig.add_trace(trace, row=1, col=3)


    # Mise à jour de la mise en page
    fig.update_layout(
        height=600,
        width=1400,
        title_text="<b>Analyses de la précédente campagne et influence sur la campagne actuelle",
        legend_title= "Dépôt"
        )

    fig.update_xaxes(title_text='Clients contactés précédemment ou non', row=1, col=1)
    fig.update_yaxes(title_text='Nombre de dépôts', row=1, col=1)

    fig.update_xaxes(title_text='Deposit', row=1, col=2)
    fig.update_yaxes(title_text='Nombre de jours depuis le dernier contact', row=1, col=2)

    fig.update_xaxes(title_text='Résultats de la campagne précédente', row=1, col=3)
    fig.update_yaxes(title_text='Nombre de dépôts', row=2, col=1)


    # Affichage du graphique
    st.plotly_chart(fig)

    st.subheader("Constat") 
    st.markdown("Une forte proportion des clients n'ont pas été contacté précédemment. Néanmoins, il est intéressant de noter que les clients ayant déjà été contactés avant cette campagne (lors d'une campagne précédente) sont plus enclins à souscrire au dépôt.")
    st.markdown("Une grande part des données sont inconnues. Il est tout de même intéressant de noter qu'un client ayant souscrit à un produit d'une campagne précédente (success), sont très enclins à souscrire au dépôt de la campagne actuelle : 91% d'entre eux ont souscrit au dépôt..")









if page == pages[3] : 
    st.title("Pre-processing des données")

    image_url = "https://www.lebigdata.fr/wp-content/uploads/2016/08/data-mining-1.jpg.webp"
    # Afficher l'image
    st.image(image_url)
    st.markdown("\n")
    st.markdown("""Avant de pouvoir appliquer les techniques d'apprentissage automatique, nous 
    devons préparer l'ensemble des données. En sachant que nous n'avons pas de 
    valeurs manquantes ou d'anomalies à traiter, nous allons procéder aux étapes suivantes :""")

    st.subheader("Pré-traitement :") 
    st.markdown("""
        - **Discrétiser** la variable **'age'** par tranches d'âge pour **atténuer le rôle des valeurs extrêmes**.
        - **Remplacer** la modalité **'unknown'** de la variable **'education'** par la modalité la plus **fréquente**.
        - **Diviser** la variable **'pdays'** (nombre de jours depuis le dernier contact - sachant 
            que pdays=-1 équivaut à previous = 0, c’est à dire pas de contact précédant cette 
            campagne) en 2 variables distinctes : **pdays_contact** (valeur no pour les 
            valeurs -1 de pdays, et valeur yes pour les autres valeurs) et la variable 
            **pdays_days** (valeur 0 pour les valeurs -1 de pdays et valeurs de pdays)""")
    st.markdown("\n")
    st.subheader("Séparation des jeux :") 
    st.markdown("""
        - **Séparer** les variables explicatives de la cible en deux DataFrames.
        - **Séparer** le jeu de données en un jeu d'entraînement (X_train,y_train) et un jeu de test (X_test, y_test) de sorte que la partie de test contient **25%** du jeu de données initial.""")

    # URL directe de l'image
    image_url = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC/main/img/preprocessing_train_test.jpg"
    # Centrer l'image avec du HTML/CSS
    st.markdown(
        f"""
        <style>
        .center {{
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>
        <img src="{image_url}" class="center">
        """,
        unsafe_allow_html=True
    )
    st.markdown("\n")
    st.markdown("\n")
    st.markdown("\n")
    st.subheader("Standardisation et encodages :") 
    st.markdown("""            
        - **LabelEncoder** de la variable cible ‘deposit’.
        - **Encodage cyclique** des variables temporelles ('month', 'day').
        - **RobustScaler** sur les variables numériques ('balance', 'duration', 'campaign','previous', ‘pdays_days’).
        - **LabelEncoder** des modalités des variables explicatives ('default', 'housing', loan').
        - **OneHotEncoder** des modalités des variables explicatives ('job', 'marital', 'contact', 'poutcome', ‘pdays_contact’).
        - **OrdinalEncoder** des modalités des variables explicatives ('age', 'education').    
                """)
    st.write("")
    st.markdown("\n")
    st.subheader("Pré-traitement des données en pipeline") 
    st.markdown("Grâce à une **pipeline**, nous avons pu générer rapidement 4 pre-processing différents, testés ensuite sur différents algorithmes de Machine Learning :")
    st.markdown("""            
        - **Un pre-processing 1** sans le feature engineering de p_days, avec l'âge discrétisé, et un encodage Robust Scaler.
        - **Un pre-processing 2** avec la division de pdays en 2 variables (pdays_contact et pdays_days), âge discrétisé, Robust Scaler sur les variables numériques.
        - **Un pre-processing 3** équivalent au précédant mais avec un Standard Scaler sur les variables numériques.
        - **Un pre-processing 4** avec l'âge sans discrétisation et Standard scaler sur les variables numériques.
    """) 
    st.markdown("\n")

    if st.checkbox("Afficher nos choix de pre-processing sous forme de df") :
        st.dataframe(pd.DataFrame({"Pre-processing":["1","2","3","4"],"Description":["sans le feature engineering de p_days, avec l'âge discrétisé, et un encodage Robust Scaler.","avec la division de pdays en pdays_contact et pdays_days, âge discrétisé, Robust Scaler","avec la division de pdays en pdays_contact et pdays_days, âge discrétisé, Standard Scaler","avec l'âge sans discrétisation et Standard scaler"]}))
      


    # URL directe de l'image
    image_url4 = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC/main/img/pipeline.jpg"
    # Centrer l'image avec du HTML/CSS
    st.image(image_url4)

    st.markdown("Les résultats des modèles se sont avérés plus performants sur la base du **pre-processing 2**. C'est celui-ci qui a été retenu pour la suite du processus.")

    # Checkbox pour afficher le contenu
    if st.checkbox("**Afficher X_train et X_test avant le pre-processing**"):
        # URL directe de l'image
        image_url2 = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC/main/img/XtrainXtest_avant.jpg"
        # Centrer l'image avec du HTML/CSS
        st.image(image_url2)


    if st.checkbox("**Afficher X_train et X_test après le pre-processing**"):
        # URL directe de l'image
        image_url3 = "https://raw.githubusercontent.com/DADSMAI2024/DA-P1-BANK-EISC/main/img/XtrainXtest_après.jpg"
        # Centrer l'image avec du HTML/CSS
        st.image(image_url3)    
