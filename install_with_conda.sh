#!/bin/bash

# Nom de l'environnement conda
ENV_NAME="recsys"

# Créer l'environnement conda avec Python 3.10
echo "Création de l'environnement conda '$ENV_NAME'..."
conda create -y -n $ENV_NAME python=3.10

# Activer l'environnement
echo "Activation de l'environnement..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Ajouter conda-forge pour des packages optimisés
echo "Configuration des canaux conda..."
conda config --add channels conda-forge
conda config --set channel_priority strict

# Installer les packages essentiels
echo "Installation des packages scientifiques et ML..."
conda install -y numpy scipy pandas matplotlib tqdm scikit-learn

# Installer LightFM
echo "Installation de LightFM..."
conda install -y -c conda-forge lightfm

# Installer optimisations MKL pour CPU
echo "Installation des optimisations MKL..."
conda install -y "libblas=*=*mkl" -c conda-forge

# Installer Jupyter et ipykernel pour VSCode
echo "Installation de Jupyter et ipykernel pour VS Code..."
conda install -y jupyter ipykernel

# Enregistrer le kernel pour Jupyter
python -m ipykernel install --user --name=$ENV_NAME --display-name="Python ($ENV_NAME)"

# Afficher le chemin Python pour configuration VSCode
PYTHON_PATH=$(which python)
echo ""
echo "======================================================"
echo "Installation complétée!"
echo "======================================================"
echo "Pour configurer VS Code:"
echo "1. Ouvrir VS Code"
echo "2. Appuyer sur F1"
echo "3. Taper 'Python: Select Interpreter'"
echo "4. Sélectionner 'Enter interpreter path...'"
echo "5. Entrer ce chemin: $PYTHON_PATH"
echo ""
echo "Dans votre terminal, activez l'environnement avec:"
echo "conda activate $ENV_NAME"
echo "======================================================"
