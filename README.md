# Skillcorner Technical Test

### Prérequis

- Python > 3.9
- Créer un environnement virtuel à partir du terminal avec le fichier "requirements.txt":
    - Création de l'environnement virtuel : `python3 -m venv .venv`
    - Activation de l'environnement virtuel : `source .venv/bin/activate`
    - Installation des librairies :`pip install -r requirements.txt`

### Fonctionnement

Après avoir activé l'environnement virtuel et installé les librairies, le script principal "video_processing.py" est utilisable à partir du terminal avec la commande suivante:

`python3 video_processing.py`

Le script "video_processing.py" prend les arguments du tableau ci-dessous en entrée et génère en sortie:
- un fichier vidéo sur lequel est appliqué le modèle de détection d'objets [YOLO](https://arxiv.org/pdf/1506.02640)
- un dossier "results" contenant un échantillon des prédictions du modèle
- un fichier log


| Argument | Type | Valeur par défaut |  Description |
| --- | --- | --- | --- |
| input_filename | `str` | cut.mp4 | nom du fichier en entrée du script | 
| output_filename | `str` | output.mp4 | nom du fichier vidéo en sortie du script |
| logs_filename | `str` | processing.log | nom du fichier log en sortie du script | 
| output_results_directory | `str` | results/ | nom du dossier contenant les prédictions du modèle | 
| nb_skipped_frames_logs | `int` | 10 | nombre d'images entre chaque log | 
| nb_preds_per_sec | `int` | 10 | nombre de prédictions par seconde | 
| output_fps | `int` | 30 | nombre d'images par sec | 
| results_saved_threshold | `int` | 10 | nombre d'images des prédictions du modèle en sortie | 
| preds_threshold | `float` | 0.5 | seuil de détection des prédictions | 
| output_width | `int` | 1280 | largeur de la vidéo en sortie | 
| output_height | `int` | 720 | hauteur de la vidéo en sortie | 
| model_verbose | `bool` | False | log par défaut du modèle | 
| model_name | `str` | yolov8n.pt | modèle PyTorch | 
| codec_name| `str` | mp4v | nom du codec | 