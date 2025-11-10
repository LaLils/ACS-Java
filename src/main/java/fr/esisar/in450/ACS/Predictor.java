package fr.esisar.in450.ACS;

public class Predictor {
    
    private CandyTypeClassifier typeClassifier;
    private ColorDetector colorDetector;
    
    public Predictor() {
        this.typeClassifier = new CandyTypeClassifier();
        this.colorDetector = new ColorDetector();
    }
    
    /**
     * Charge le modèle de classification
     */
    public void loadModel(String modelPath) throws Exception {
        typeClassifier.loadModel(modelPath);
    }
    
    /**
     * Prédit le type et la couleur d'un bonbon
     * @param imagePath Chemin vers l'image
     * @return Tableau [type, couleur, confiance]
     */
    public String[] predict(String imagePath) throws Exception {
        System.out.println("Prédiction du type...");
        String[] typeResult = typeClassifier.predict(imagePath);
        String candyType = typeResult[0];
        String confidence = typeResult[1];
        
        System.out.println("Détection de la couleur...");
        String color = colorDetector.detectColorWithHistogram(imagePath);
        
        // Gestion des couleurs spéciales
        color = adjustColorForType(candyType, color);
        
        return new String[] {candyType, color, confidence};
    }
    
    /**
     * Ajuste la couleur en fonction du type (pour Fraise et Oeuf)
     */
    private String adjustColorForType(String type, String detectedColor) {
        if (type.equals("Fraise")) {
            return "Fraise"; // Une seule couleur possible
        } else if (type.equals("Oeuf")) {
            return "Oeuf"; // Une seule couleur possible
        }
        return detectedColor;
    }
    
    /**
     * Prédit plusieurs bonbons (version batch)
     */
    public void predictBatch(String[] imagePaths) throws Exception {
        System.out.println("=== Prédiction en lot ===\n");
        
        for (int i = 0; i < imagePaths.length; i++) {
            System.out.println("Image " + (i + 1) + "/" + imagePaths.length + ": " + imagePaths[i]);
            try {
                String[] result = predict(imagePaths[i]);
                System.out.println("  Type: " + result[0]);
                System.out.println("  Couleur: " + result[1]);
                System.out.println("  Confiance: " + result[2] + "%\n");
            } catch (Exception e) {
                System.err.println("  Erreur: " + e.getMessage() + "\n");
            }
        }
    }
}