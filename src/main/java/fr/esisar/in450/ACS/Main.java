package fr.esisar.in450.ACS;

import java.io.File;

public class Main {
    
    public static void main(String[] args) {
        try {
            System.out.println("=== Automatic Candy Selector ===\n");
            
            // Choix du mode
            if (args.length > 0 && args[0].equals("train")) {
                System.out.println("Mode: Entraînement du modèle");
                trainModel();
            } 
            else if (args.length > 0 && args[0].equals("predict")) {
                System.out.println("Mode: Prédiction");
                if (args.length < 2) {
                    System.out.println("Usage: java Main predict <chemin_image>");
                    return;
                }
                predictImage(args[1]);
            } else {
                System.out.println("Mode: Interface console");
                runConsoleInterface();
            }
            
        } 
        catch (Exception e) {
            System.err.println("Erreur: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Entraîne le modèle de classification
     */
    private static void trainModel() throws Exception {
        System.out.println("\n[1/2] Entraînement du classificateur de type...");
        CandyTypeClassifier typeClassifier = new CandyTypeClassifier();
        typeClassifier.train("src/main/train");
        typeClassifier.saveModel("models/candy_type_model.zip");
        System.out.println("✓ Modèle de type sauvegardé\n");
        
        System.out.println("[2/2] Configuration du détecteur de couleur...");
        ColorDetector colorDetector = new ColorDetector();
        System.out.println("✓ Détecteur de couleur prêt\n");
        
        System.out.println("=== Entraînement terminé ! ===");
    }
    
    /**
     * Prédit le type et la couleur d'un bonbon
     */
    private static void predictImage(String imagePath) throws Exception {
        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            System.err.println("Erreur: Image introuvable - " + imagePath);
            return;
        }
        
        System.out.println("Analyse de l'image: " + imagePath + "\n");
        
        Predictor predictor = new Predictor();
        predictor.loadModel("models/candy_type_model.zip");
        
        String[] result = predictor.predict(imagePath);
        
        System.out.println("=== Résultat ===");
        System.out.println("Type : " + result[0]);
        System.out.println("Couleur : " + result[1]);
        System.out.println("Confiance : " + result[2] + "%");
    }
    
    /**
     * Interface console simple
     */
    private static void runConsoleInterface() throws Exception {
        java.util.Scanner scanner = new java.util.Scanner(System.in);
        
        System.out.println("Que voulez-vous faire ?");
        System.out.println("1. Entraîner le modèle");
        System.out.println("2. Prédire un bonbon");
        System.out.print("\nChoix: ");
        
        int choice = scanner.nextInt();
        scanner.nextLine(); // Consommer la ligne
        
        if (choice == 1) {
            trainModel();
        } else if (choice == 2) {
            System.out.print("Chemin de l'image: ");
            String path = scanner.nextLine();
            predictImage(path);
        }
        
        scanner.close();
    }
}