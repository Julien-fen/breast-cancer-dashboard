# Tableau de Bord Diagnostique du Cancer du Sein

## Contexte du Projet
Ce portfolio présente un projet de diagnostic du cancer du sein basé sur le **Wisconsin Breast Cancer Dataset**.  
L'objectif est de permettre à l'utilisateur d'explorer les données, de comparer les performances des modèles et d'effectuer des prédictions interactives.  
Une attention particulière est portée à l'explicabilité des modèles grâce à l'utilisation de SHAP (SHapley Additive exPlanations).

## Structure du Portfolio
Le tableau de bord se compose des sections suivantes :
1. **Mise en Contexte** : Contexte du projet, description du dataset et définition des principaux termes techniques.
2. **Vue d'ensemble / EDA** : Analyse exploratoire des données, statistiques générales, répartition des tumeurs, matrice de corrélation et analyse détaillée de la caractéristique `radius_mean`.
3. **Comparaison des Modèles** : Comparaison des performances des modèles (Random Forest, Régression Logistique et SVM) avec matrice de confusion, courbe ROC et importance des variables ou coefficients.
4. **Prédiction Interactive** : Permet d'entrer des valeurs pour obtenir une prédiction. Pour Random Forest, un graphique SHAP explique l'influence de chaque caractéristique.
5. **Segmentation & Clustering** : Visualisation des données en 2D via PCA et t-SNE.

## Installation

1. **Cloner le dépôt**  
   ```bash
   git clone <URL-du-dépôt-GitHub>
   cd <nom-du-dossier>
