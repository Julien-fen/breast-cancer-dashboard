import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score, f1_score

import shap
import streamlit.components.v1 as components



# --- Fonction helper pour afficher les plots SHAP ---
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# --- Configuration de la page ---
st.set_page_config(layout="wide", page_title="Tableau de Bord Diagnostic du Cancer du Sein")

# =====================================================
# 0. Sélecteur de langue (Français / English)
# =====================================================
lang = st.sidebar.selectbox("Choisir la langue / Choose language", ["Français", "English"])

# --- Définition des textes selon la langue choisie ---
if lang == "Français":
    txt_title = "Tableau de Bord Interactif pour le Diagnostic du Cancer du Sein"
    txt_context_title = "Mise en Contexte"
    txt_context_desc = (
        "**Contexte du Portfolio :**\n\n"
        "Ce portfolio présente un projet de diagnostic du cancer du sein basé sur un dataset du Wisconsin. "
        "Ce dataset, issu de l'Université du Wisconsin, est largement utilisé pour évaluer les modèles de diagnostic. "
        "L'objectif est de permettre à l'utilisateur d'explorer les données, de comparer les performances de différents modèles "
        "de classification (Random Forest, Régression Logistique et SVM) et de réaliser des prédictions interactives.\n\n"
        "**Pages du Tableau de Bord :**\n"
        "1. Mise en Contexte\n"
        "2. Vue d'ensemble / EDA\n"
        "3. Comparaison des Modèles\n"
        "4. Prédiction Interactive\n"
        "5. Segmentation & Clustering\n\n"
        "**Termes Techniques :**\n"
        "- **SHAP** : Méthode pour expliquer les prédictions d’un modèle en attribuant une importance à chaque caractéristique.\n"
        "- **Matrice de Confusion** : Tableau qui compare les prédictions du modèle aux valeurs réelles afin d’identifier les erreurs de classification.\n"
        "- **Courbe ROC** : Graphique illustrant la capacité du modèle à distinguer entre classes en affichant le taux de vrais positifs vs. le taux de faux positifs.\n"
        "- **PCA** : Analyse en Composantes Principales, technique de réduction de dimension pour visualiser les tendances principales.\n"
        "- **t-SNE** : Technique non linéaire de réduction de dimension pour identifier des regroupements complexes dans les données."
    )
    # Textes pour la section EDA
    txt_EDA_desc = ("Cette section présente une analyse exploratoire des données de diagnostic du cancer du sein. "
                    "Vous trouverez des statistiques générales, la répartition des tumeurs, une matrice de corrélation, "
                    "ainsi qu'une analyse détaillée de la caractéristique 'radius_mean'.")
    txt_stats = "Statistiques Générales"
    txt_total_samples = "Nombre total d'échantillons"
    txt_malignant_rate = "Taux de Tumeurs Malignes (%)"
    txt_distribution = "Répartition des Tumeurs"
    txt_corr = "Matrice de Corrélation"
    txt_radius = "Distribution de 'radius_mean'"
    txt_radius_expl = ("La caractéristique 'radius_mean' représente la moyenne des distances entre le contour de la tumeur et son centre. "
                       "Elle est utilisée ici car elle possède une forte capacité discriminante entre tumeurs bénignes et malignes.")
    # Textes pour la section Comparaison des modèles
    txt_model_comp_title = "Comparaison des Modèles de Classification"
    txt_model_comp_desc = ("Dans cette section, nous comparons les performances des trois modèles suivants : "
                           "Random Forest, Régression Logistique et SVM. Vous pourrez visualiser leur matrice de confusion, "
                           "leur courbe ROC ainsi que l'importance des variables ou les coefficients. "
                           "Ces éléments permettent d’évaluer la capacité du modèle à distinguer entre tumeurs bénignes et malignes.")
    txt_confusion = "Matrice de Confusion"
    txt_confusion_expl = ("La matrice de confusion présente le nombre de prédictions correctes et incorrectes pour chaque classe, "
                          "permettant d’identifier où le modèle se trompe.")
    txt_ROC = "Courbe ROC"
    txt_ROC_expl = ("La courbe ROC (Receiver Operating Characteristic) montre la capacité du modèle à distinguer entre les classes "
                    "en affichant le taux de vrais positifs en fonction du taux de faux positifs pour différents seuils.")
    txt_importance = "Importance des Variables"
    txt_importance_expl = ("Ce graphique indique l'importance de chaque caractéristique dans la décision du modèle Random Forest. "
                           "Les variables avec une forte importance influencent fortement la prédiction.")
    txt_coef = "Coefficients du Modèle"
    txt_coef_expl = ("Les coefficients montrent l’influence de chaque caractéristique sur la probabilité d'appartenir à une classe. "
                     "Un coefficient positif augmente la probabilité d’une prédiction maligne, tandis qu’un coefficient négatif la réduit.")
    # Textes pour la section Prédiction Interactive
    txt_pred_title = "Prédiction Interactive et Explication"
    txt_pred_desc = ("Dans cette section, vous pouvez saisir les valeurs des caractéristiques pour obtenir une prédiction du diagnostic "
                     "(tumeur bénigne ou maligne) en choisissant l’un des trois modèles. "
                     "Pour le modèle Random Forest, un graphique SHAP vous expliquera comment chaque caractéristique influence la prédiction.")
    txt_slider_help = "Moyenne: {mean:.2f} | Écart-type: {std:.2f}"
    txt_result = "Résultats"
    txt_SHAP_title = "Explication de la Prédiction avec SHAP"
    txt_SHAP_expl = ("Le graphique SHAP ci-dessous montre comment chaque caractéristique influence la prédiction du modèle Random Forest. "
                     "Les couleurs indiquent l’effet positif ou négatif de chaque variable.")
    txt_no_shap = "L’explication SHAP n’est disponible que pour le modèle Random Forest."
    # Textes pour la section Segmentation & Clustering
    txt_seg_title = "Segmentation & Clustering"
    txt_seg_desc = ("Cette section utilise des techniques de réduction de dimension pour projeter les données dans un espace en 2D, "
                    "permettant de visualiser la répartition des échantillons selon leur diagnostic.")
    txt_PCA = "Projection PCA"
    txt_PCA_expl = ("Le graphique PCA réduit la dimensionnalité des données afin de visualiser les principales tendances et détecter des clusters potentiels.")
    txt_tSNE = "Visualisation avec t-SNE"
    txt_tSNE_expl = ("Le graphique t-SNE offre une visualisation non linéaire en 2D qui aide à identifier des regroupements d’échantillons similaires.")
    
    # Libellés spécifiques
    label_classification_report = "Rapport de Classification"
    label_slider = "Saisissez les caractéristiques"
    label_benign = "TUMEUR BÉNIGNE"
    label_predict = "Prédire"
    
    # Définition des sections pour la sidebar (en français)
    sidebar_sections = [txt_context_title, "Vue d'ensemble / EDA", "Comparaison des Modèles", "Prédiction Interactive", "Segmentation & Clustering"]

else:
    txt_title = "Breast Cancer Diagnostic Dashboard"
    txt_context_title = "Project Context"
    txt_context_desc = (
        "**Project Objective:**\n"
        "This project aims to create an interactive dashboard for breast cancer diagnosis using classification models (Random Forest, Logistic Regression, and SVM). "
        "Users can explore the data, compare model performance, and make interactive predictions.\n\n"
        "**Dashboard Pages:**\n"
        "1. Project Context\n"
        "2. Exploratory Data Analysis\n"
        "3. Model Comparison\n"
        "4. Interactive Prediction\n"
        "5. Segmentation & Clustering\n\n"
        "**Technical Terms:**\n"
        "- **SHAP (SHapley Additive exPlanations):** A method to explain model outputs by assigning an importance value to each feature.\n"
        "- **Confusion Matrix:** A table summarizing a classification model’s performance by comparing predictions with true values.\n"
        "- **ROC Curve:** A plot showing the model’s performance across different decision thresholds.\n"
        "- **PCA and t-SNE:** Dimensionality reduction techniques used for 2D visualization of high-dimensional data.\n\n"
        "**Portfolio Context:**\n"
        "This portfolio uses the Wisconsin Breast Cancer dataset, which originates from the University of Wisconsin and is widely used as a benchmark for diagnostic models."
    )
    txt_EDA_desc = ("This section presents an exploratory analysis of the breast cancer diagnosis data. "
                    "You will find general statistics, tumor distribution, a correlation matrix, and a detailed analysis of the 'radius_mean' feature.")
    txt_stats = "General Statistics"
    txt_total_samples = "Total Number of Samples"
    txt_malignant_rate = "Malignant Tumor Rate (%)"
    txt_distribution = "Tumor Distribution"
    txt_corr = "Correlation Matrix"
    txt_radius = "Distribution of 'radius_mean'"
    txt_radius_expl = ("The 'radius_mean' feature represents the average distance from the tumor's contour to its center. "
                       "It is used here because it shows strong discriminative power between benign and malignant tumors.")
    txt_model_comp_title = "Model Comparison"
    txt_model_comp_desc = ("In this section, we compare the performance of three models: Random Forest, Logistic Regression, and SVM. "
                           "You can view performance metrics, the confusion matrix, and the ROC curve to evaluate how well each model distinguishes between benign and malignant tumors.")
    txt_confusion = "Confusion Matrix"
    txt_confusion_expl = ("The confusion matrix displays the number of correct and incorrect predictions for each class, helping to identify model errors.")
    txt_ROC = "ROC Curve"
    txt_ROC_expl = ("The ROC (Receiver Operating Characteristic) curve shows the model's ability to distinguish between classes by plotting "
                    "the true positive rate against the false positive rate at various thresholds.")
    txt_importance = "Feature Importance"
    txt_importance_expl = ("This chart displays the importance of each feature in the Random Forest model's decision-making process. "
                           "Features with high importance have a strong impact on the prediction.")
    txt_coef = "Model Coefficients"
    txt_coef_expl = ("Coefficients indicate the influence of each feature on the probability of belonging to a class. "
                     "Positive coefficients increase the likelihood of a malignant prediction, while negative coefficients decrease it.")
    txt_pred_title = "Interactive Prediction and Explanation"
    txt_pred_desc = ("In this section, you can enter values for the features to obtain a diagnosis prediction (benign or malignant) using one of the three models. "
                     "For the Random Forest model, a SHAP plot will explain how each feature influences the prediction.")
    txt_slider_help = "Mean: {mean:.2f} | Std: {std:.2f}"
    txt_result = "Results"
    txt_SHAP_title = "SHAP Explanation"
    txt_SHAP_expl = ("The SHAP plot below illustrates how each feature contributes to the Random Forest model's prediction. "
                     "The colors indicate whether a feature has a positive or negative impact on the outcome.")
    txt_no_shap = "SHAP explanation is only available for the Random Forest model."
    txt_seg_title = "Segmentation & Clustering"
    txt_seg_desc = ("This section uses dimensionality reduction techniques to project the data into a 2D space, "
                    "allowing you to visualize the distribution of samples by diagnosis.")
    txt_PCA = "PCA Projection"
    txt_PCA_expl = ("The PCA chart reduces the dimensionality of the data to visualize major trends and potential clusters.")
    txt_tSNE = "t-SNE Visualization"
    txt_tSNE_expl = ("The t-SNE plot provides a non-linear 2D visualization that helps identify groups of similar samples.")
    label_classification_report = "Classification Report"
    label_slider = "Enter the features"
    label_benign = "BENIGN TUMOR"
    
    sidebar_sections = [txt_context_title, "Exploratory Data Analysis", "Model Comparison", "Interactive Prediction", "Segmentation & Clustering"]

# =====================================================
# 1. Chargement et préparation des données
# =====================================================
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("breast-cancer-wisconsin-data.csv")
    df = df.drop(columns=['id'])
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    return df

df = load_data()
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']
taux_malin = df['diagnosis'].mean() * 100

# ------------------------------------------------------------------
# Traduction des noms de caractéristiques pour la prédiction interactive
# ------------------------------------------------------------------
translation = {
    "radius_mean": "rayon_moyen" if lang == "Français" else "mean radius",
    "texture_mean": "texture_moyenne" if lang == "Français" else "mean texture",
    "perimeter_mean": "périmètre_moyen" if lang == "Français" else "mean perimeter",
    "area_mean": "surface_moyenne" if lang == "Français" else "mean area",
    "smoothness_mean": "lissité_moyenne" if lang == "Français" else "mean smoothness",
    "compactness_mean": "compacité_moyenne" if lang == "Français" else "mean compactness",
    "concavity_mean": "concavité_moyenne" if lang == "Français" else "mean concavity",
    "concave points_mean": "points concaves moyens" if lang == "Français" else "mean concave points",
    "symmetry_mean": "symétrie_moyenne" if lang == "Français" else "mean symmetry",
    "fractal_dimension_mean": "dimension fractale moyenne" if lang == "Français" else "mean fractal dimension"
}
translated_columns = {col: translation.get(col, col) for col in X.columns}
X.rename(columns=translated_columns, inplace=True)

# =====================================================
# 2. Entraînement de plusieurs modèles
# =====================================================
@st.cache_data(show_spinner=False)
def train_models(X_train_scaled, y_train):
    models = {}
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    models["Random Forest"] = rf

    lr = LogisticRegression(max_iter=10000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    models["Régression Logistique"] = lr

    svm = SVC(probability=True, random_state=42)
    svm.fit(X_train_scaled, y_train)
    models["SVM"] = svm

    return models

# Séparation et normalisation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = train_models(X_train_scaled, y_train)

def evaluate_model(model, X_test_scaled, y_test):
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, f1, cm, fpr, tpr, roc_auc, report, y_pred, y_pred_proba

# =====================================================
# 3. Navigation et affichage des pages
# =====================================================
if lang == "Français":
    sidebar_sections = [txt_context_title, "Vue d'ensemble / EDA", "Comparaison des Modèles", "Prédiction Interactive", "Segmentation & Clustering"]
else:
    sidebar_sections = [txt_context_title, "Exploratory Data Analysis", "Model Comparison", "Interactive Prediction", "Segmentation & Clustering"]

st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisir une section :" if lang=="Français" else "Choose a section :", sidebar_sections)

# ----------------------------
# Section E (0) : Mise en Contexte
# ----------------------------
if section == txt_context_title:
    st.title(txt_context_title)
    st.write(txt_context_desc)

# ----------------------------
# Section A : Vue d'ensemble / EDA
# ----------------------------
elif section == "Vue d'ensemble / EDA" or section == "Exploratory Data Analysis":
    st.title(txt_title)
    st.header("🔍 " + txt_stats)
    st.write(txt_EDA_desc)
    col1, col2 = st.columns(2)
    col1.metric(txt_total_samples, df.shape[0])
    col2.metric(txt_malignant_rate, f"{taux_malin:.2f}%")
    
    st.subheader("📌 " + txt_distribution)
    st.write("Ce graphique montre la fréquence des tumeurs bénignes et malignes dans le dataset." if lang=="Français" 
             else "This chart shows the frequency of benign and malignant tumors.")
    fig_count, ax = plt.subplots()
    sns.countplot(x=df['diagnosis'].map({0: "Bénin" if lang=="Français" else "Benign", 
                                           1: "Malin" if lang=="Français" else "Malignant"}), 
                  palette=["blue", "red"], ax=ax)
    ax.set_xlabel("Diagnostic" if lang=="Français" else "Diagnosis")
    ax.set_ylabel("Nombre" if lang=="Français" else "Count")
    st.pyplot(fig_count)
    
    st.subheader("📈 " + txt_corr)
    st.write("La matrice de corrélation indique la force des relations entre les caractéristiques." if lang=="Français" 
             else "The correlation matrix shows the strength of relationships between features.")
    fig_corr, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False, linewidths=0.5, ax=ax)
    st.pyplot(fig_corr)
    
    st.subheader(txt_radius)
    st.write(txt_radius_expl)
    fig_hist, ax = plt.subplots()
    sns.histplot(df['radius_mean'], bins=30, kde=True, ax=ax)
    ax.set_title("Distribution de 'radius_mean'" if lang=="Français" else "Distribution of 'radius_mean'")
    ax.set_xlabel("Rayon Moyen" if lang=="Français" else "Mean Radius")
    ax.set_ylabel("Fréquence" if lang=="Français" else "Frequency")
    st.pyplot(fig_hist)

# ----------------------------
# Section B : Comparaison des Modèles
# ----------------------------
elif section == "Comparaison des Modèles" or section == "Model Comparison":
    st.title(txt_model_comp_title)
    st.write(txt_model_comp_desc)
    
    model_choice = st.selectbox("Choisir un modèle :" if lang=="Français" else "Choose a model :", list(models.keys()))
    model = models[model_choice]
    
    acc, f1, cm, fpr, tpr, roc_auc, report, y_pred, y_pred_proba = evaluate_model(model, X_test_scaled, y_test)
    
    st.subheader(f"Performance du modèle : {model_choice}" if lang=="Français" else f"Model Performance: {model_choice}")
    st.write(f"**Accuracy :** {acc:.3f}")
    st.write(f"**F1-Score :** {f1:.3f}")
    st.write(f"**AUC :** {roc_auc:.3f}")
    st.write("**" + label_classification_report + " :**")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader(txt_confusion)
    st.write(txt_confusion_expl)
    fig_cm, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Bénin" if lang=="Français" else "Benign", 
                             "Malin" if lang=="Français" else "Malignant"], 
                yticklabels=["Bénin" if lang=="Français" else "Benign", 
                             "Malin" if lang=="Français" else "Malignant"], ax=ax)
    ax.set_xlabel("Prédictions" if lang=="Français" else "Predictions")
    ax.set_ylabel("Vraies Valeurs" if lang=="Français" else "True Values")
    st.pyplot(fig_cm)
    
    st.subheader(txt_ROC)
    st.write(txt_ROC_expl)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {roc_auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Aléatoire" if lang=="Français" else "Random"))
    fig_roc.update_layout(title="Courbe ROC" if lang=="Français" else "ROC Curve", 
                          xaxis_title="Taux de Faux Positifs (FPR)" if lang=="Français" else "False Positive Rate", 
                          yaxis_title="Taux de Vrais Positifs (TPR)" if lang=="Français" else "True Positive Rate")
    st.plotly_chart(fig_roc)
    
    if model_choice == "Random Forest":
        st.subheader(txt_importance)
        st.write(txt_importance_expl)
        importances = model.feature_importances_
        feature_df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False)
        fig_feat = px.bar(feature_df.head(10), x="Importance", y="Feature", orientation="h", title="Top 10 des Variables les Plus Importantes" if lang=="Français" else "Top 10 Most Important Features")
        st.plotly_chart(fig_feat)
    elif model_choice == "Régression Logistique":
        st.subheader(txt_coef)
        st.write(txt_coef_expl)
        coef = model.coef_[0]
        coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": coef}).sort_values(by="Coefficient", ascending=False)
        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="Coefficients (impact positif/négatif)" if lang=="Français" else "Model Coefficients")
        st.plotly_chart(fig_coef)
    else:
        st.info("Les coefficients ne sont pas directement interprétables pour ce modèle." if lang=="Français" else "Coefficients are not directly interpretable for this model.")

# ----------------------------
# Section C : Prédiction Interactive et Explication
# ----------------------------
elif section == "Prédiction Interactive" or section == "Interactive Prediction":
    st.title(txt_pred_title)
    st.write(txt_pred_desc)
    
    model_choice_interact = st.selectbox("Choisir un modèle pour la prédiction interactive :" if lang=="Français" else "Choose a model for interactive prediction :", list(models.keys()))
    model_pred = models[model_choice_interact]
    
    st.subheader(label_slider)
    st.write("Utilisez les curseurs ci-dessous pour définir les valeurs de chaque caractéristique. Les valeurs par défaut correspondent à la médiane de chaque variable." if lang=="Français" 
             else "Use the sliders below to set the values for each feature. The default values are the median of each variable.")
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(X.columns):
        with cols[i % 3]:
            stats = X[col].describe()
            input_data[col] = st.slider(
                label=col,
                min_value=float(stats['min']),
                max_value=float(stats['max']),
                value=float(stats['50%']),
                help=txt_slider_help.format(mean=stats['mean'], std=stats['std'])
            )
    
    if st.button("Prédire" if lang=="Français" else "Predict"):
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model_pred.predict(input_scaled)[0]
        proba = model_pred.predict_proba(input_scaled)[0][1]
        
        # Traduction du résultat affiché
        if lang == "Français":
            result_title = "🔴 TUMEUR MALIGNE" if prediction == 1 else "🟢 TUMEUR BÉNIGNE"
            result_prob = f"Probabilité de malignité : {proba:.1%}"
        else:
            result_title = "🔴 MALIGNANT TUMOR" if prediction == 1 else "🟢 BENIGN TUMOR"
            result_prob = f"Malignancy Probability: {proba:.1%}"
        
        st.subheader("🔍 " + txt_result)
        result_html = f"""
        <div style="padding:20px; border-radius:10px; 
            background: {'#ffcccc' if prediction == 1 else '#ccffcc'};
            border: 2px solid {'#ff0000' if prediction == 1 else '#00ff00'};">
            <h3 style="color:{'#ff0000' if prediction == 1 else '#00ff00'}; text-align:center;">
                {result_title}
            </h3>
            <p style="text-align:center;">{result_prob}</p>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)
        
        # Affichage de l'explication SHAP uniquement pour Random Forest
        if model_choice_interact == "Random Forest":
            st.subheader(txt_SHAP_title)
            st.write(txt_SHAP_expl)
            shap.initjs()
            try:
                explainer_rf = shap.TreeExplainer(models["Random Forest"])
                shap_values_rf = explainer_rf.shap_values(input_scaled)
                fig_shap = shap.plots.force(explainer_rf.expected_value[0], shap_values_rf[..., 0], input_df)
                st_shap(fig_shap, height=300)
            except Exception as e:
                st.error(f"Erreur de génération d'explication SHAP : {str(e)}")
                st.info("Vérifiez la dimension des données et la configuration du modèle Random Forest." if lang=="Français" else "Check the input data dimensions and Random Forest configuration.")
        else:
            st.info(txt_no_shap)
            
# ----------------------------
# Section D : Segmentation & Clustering
# ----------------------------
elif section == "Segmentation & Clustering":
    st.title(txt_seg_title)
    st.write(txt_seg_desc)
    
    st.subheader(txt_PCA)
    st.write(txt_PCA_expl)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["diagnostic"] = y.map({0: "Bénin" if lang=="Français" else "Benign", 1: "Malin" if lang=="Français" else "Malignant"})
    fig_pca = px.scatter(df_pca, x="PC1", y="PC2", color="diagnostic", title="Projection PCA des données" if lang=="Français" else "PCA Projection")
    st.plotly_chart(fig_pca)
    
    st.subheader(txt_tSNE)
    st.write(txt_tSNE_expl)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)
    df_tsne = pd.DataFrame(X_tsne, columns=["TSNE1", "TSNE2"])
    df_tsne["diagnostic"] = y.map({0: "Bénin" if lang=="Français" else "Benign", 1: "Malin" if lang=="Français" else "Malignant"})
    fig_tsne = px.scatter(df_tsne, x="TSNE1", y="TSNE2", color="diagnostic", title="Projection t-SNE des données" if lang=="Français" else "t-SNE Projection")
    st.plotly_chart(fig_tsne)
