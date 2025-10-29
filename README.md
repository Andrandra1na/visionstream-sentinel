# VisionStream Sentinel v1.0

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-0059D6?style=for-the-badge)

Une console de supervision intelligente en temps réel qui se connecte à un flux de caméra IP, utilise un modèle YOLOv8 pour la détection d'intrusions dans une zone personnalisable, et affiche les résultats de manière performante avec OpenCV.

---

### Démonstration Vidéo

**(IMPORTANT : Enregistrez une vidéo ou un GIF de votre application en action et insérez-le ici. C'est l'élément le plus vendeur de votre projet !)**
*Sur Debian, vous pouvez utiliser des outils comme `peek` ou `kazam` pour enregistrer un GIF facilement.*

`![Demonstration GIF](link_vers_votre_gif.gif)`

---

### Fonctionnalités Clés

*   **Analyse en Temps Réel :** Traite un flux vidéo distant avec une latence minimale grâce à une lecture de flux multi-threadée.
*   **Zone d'Intrusion Interactive :** Permet à l'utilisateur de dessiner n'importe quelle forme de polygone directement sur la vidéo pour définir la zone de surveillance.
*   **Détection d'Intrusion Intelligente :** Utilise un modèle YOLOv8 pour détecter les personnes et déclenche une alerte visuelle si elles entrent dans la zone définie.
*   **Interface Performante :** Utilise l'interface graphique native d'OpenCV pour un affichage direct et rapide, sans la surcharge d'une application web.
*   **Configuration Facile :** Gère la configuration (URL du flux, etc.) via un simple fichier `.env`.

### Architecture et Décisions Techniques

Ce projet a initialement été prototypé avec un backend **FastAPI** pour diffuser les résultats sur une interface web via WebSockets et streaming MJPEG. Cependant, les tests sur du matériel contraint (CPU à faible consommation) et un réseau Wi-Fi à haute latence ont révélé des problèmes de performance significatifs.

Face à ces contraintes du monde réel, un **pivot stratégique** a été effectué :
La solution finale a été optimisée pour une performance maximale en supprimant la couche web et en créant une **application de bureau native avec OpenCV**. Cette approche élimine toute surcharge liée à l'encodage et au streaming web. Le projet démontre ainsi non seulement la maîtrise de l'IA en temps réel, mais aussi la capacité à **choisir la bonne architecture technique pour répondre à des contraintes de performance concrètes.**

### Stack Technique

*   **Langage :** Python 3.11
*   **Computer Vision :** OpenCV
*   **Détection d'Objets :** YOLOv8 (via Ultralytics)
*   **Gestion de l'Environnement :** Conda
*   **Configuration :** python-dotenv

### Installation et Utilisation

1.  **Cloner le dépôt :**
    ```bash
    git clone https://github.com/VOTRE_NOM_UTILISATEUR/visionstream-sentinel.git
    cd visionstream-sentinel
    ```

2.  **Configurer l'environnement (Conda) :**
    *   Assurez-vous d'avoir un environnement Conda avec Python 3.11.
    *   Installez les dépendances :
        ```bash
        pip install -r requirements.txt
        ```

3.  **Configurer le flux vidéo :**
    *   Renommez `.env.example` en `.env`.
    *   Éditez le fichier `.env` et remplacez l'URL par celle de votre caméra IP (fournie par l'application "IP Webcam" sur votre téléphone) :
        ```ini
        VIDEO_STREAM_URL="http://192.168.X.X:XXXX/video"
        ```

4.  **Lancer l'application :**
    *   Assurez-vous que votre téléphone diffuse la vidéo.
    *   Exécutez la commande suivante :
        ```bash
        python run_supervisor.py
        ```

5.  **Interagir avec l'application :**
    *   **Clic Gauche :** Ajouter des points pour définir la zone d'intrusion.
    *   **Clic Droit :** Finaliser la zone (minimum 3 points).
    *   **Touche 'c' :** Effacer la zone actuelle.
    *   **Touche 'q' :** Quitter l'application.