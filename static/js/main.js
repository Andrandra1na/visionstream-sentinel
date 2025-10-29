document.addEventListener("DOMContentLoaded", () => {
    const canvas = document.getElementById("videoCanvas");
    const ctx = canvas.getContext("2d");
    const statusBar = document.getElementById("status-bar");
    const statusText = document.getElementById("status-text");
    const alertOverlay = document.getElementById("alert-overlay");

    let ws;

    function connectWebSocket() {
        // Déterminer l'adresse du WebSocket
        const proto = window.location.protocol === "https:" ? "wss" : "ws";
        const host = window.location.host;
        const wsUrl = `${proto}://${host}/ws`;

        console.log(`Connexion à ${wsUrl}`);
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("Connecté au serveur WebSocket.");
            statusBar.className = "status-ok";
            statusText.textContent = "CONNECTÉ";
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            // Mettre à jour la superposition d'alerte
            if (data.status === "INTRUSION") {
                alertOverlay.classList.remove("hidden");
            } else {
                alertOverlay.classList.add("hidden");
            }

            // Dessiner l'image reçue sur le canvas
            const image = new Image();
            image.src = "data:image/jpeg;base64," + data.image_base64;
            image.onload = () => {
                // Ajuster la taille du canvas à la taille de l'image reçue
                canvas.width = image.width;
                canvas.height = image.height;
                // Dessiner l'image
                ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
            };
        };

        ws.onclose = () => {
            console.log("Déconnecté du serveur WebSocket. Tentative de reconnexion dans 3 secondes...");
            statusBar.className = "status-error";
            statusText.textContent = "DÉCONNECTÉ - RECONNEXION...";
            setTimeout(connectWebSocket, 3000); // Tenter de se reconnecter
        };

        ws.onerror = (error) => {
            console.error("Erreur WebSocket:", error);
            statusBar.className = "status-error";
            statusText.textContent = "ERREUR DE CONNEXION";
            ws.close();
        };
    }

    // Lancer la connexion initiale
    connectWebSocket();
});