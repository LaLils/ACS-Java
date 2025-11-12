package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.io.File;

/**
 * Classe principale du projet ACS (Analyse et Classification d'Images).
 * Cette classe orchestre l'ensemble du processus d'entraînement et de prédiction d'un modèle de Deep Learning (basé sur ResNet) en utilisant la bibliothèque Deeplearning4j.
 * Utilisation :
 *   - java Main< : lance le mode entraînement
 *   - java Main predict chemin/vers/image.jpg : lance le mode prédiction
 */
public class Main {
    private static final int HEIGHT = 224;	// Hauteur des images d'entrée
    private static final int WIDTH = 224;	// Largeur des images d'entrée 
    private static final int CHANNELS = 3;	// Nombre de canaux (3 pour RGB)
    private static final int NUM_CLASSES = 18;	// Nombre total de classes à prédire
    private static final int BATCH_SIZE = 4;	// Taille d’un lot d’images traité en parallèle (limité par la RAM disponible)
    private static final int N_EPOCHS = 8;	// Nombre total d’époques d’entraînement (tours complets du dataset)
    private static final String MODEL_PATH = "resnet_model.zip";	// Chemin du modèle sauvegardé
    private static final String TRAIN_PATH = "src/main/resources/dataset/train";	// Chemin vers le dossier d’entraînement
    private static final String TEST_PATH = "src/main/resources/dataset/test";	// Chemin vers le dossier de test

    /**
     * Point d’entrée du programme.
     * Selectionne le mode ENTRAINEMENT ou PREDICTION selon si @param args contient "predict"
     */
    public static void main(String[] args) {
        try {
            if (args.length > 0 && args[0].equals("predict")) {	// Mode prédiction
                if (args.length < 2) {
                    System.err.println("Usage: java Main predict <chemin_image>");
                    System.exit(1);
                }
                runPrediction(args[1]);
            } 
            else {	// Mode entraînement par défaut
                runTraining();
            }
        } 
        catch (Exception e) {
            System.err.println("ERREUR : " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * MODE ENTRAÎNEMENT du modèle:
     *   - Chargement du jeu de données d'entraînement et de test
     *   - Initialisation des labels dans la classe utilitaire
     *   - Construction du modèle ResNet
     *   - Entraînement sur plusieurs époques avec suivi du score<
     *   - Évaluation finale sur le jeu de test
     *   - Sauvegarde du modèle entraîné
     *
     * @throws Exception si une erreur survient pendant le processus
     */
    private static void runTraining() throws Exception {
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║           MODE ENTRAÎNEMENT            ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        // === 1. Chargement des datasets ===
        System.out.println("=== ÉTAPE 1 : Chargement des données ===");
        // On prépare les itérateurs d’entraînement et de test à partir des répertoires
        DataSetIterator trainIter = DataLoader.getIterator(TRAIN_PATH, HEIGHT, WIDTH, CHANNELS, BATCH_SIZE, NUM_CLASSES);
        DataSetIterator testIter = DataLoader.getIterator(TEST_PATH, HEIGHT, WIDTH, CHANNELS, BATCH_SIZE, NUM_CLASSES);

        // === 2. Initialisation des labels ===
        // Les labels (classes) sont enregistrés dans Utils pour être accessibles par la suite
        Utils.initLabels(DataLoader.getLabels());

        // === 3. Construction du modèle ===
        System.out.println("\n=== ÉTAPE 2 : Construction du modèle ===");
        ComputationGraph model = ResNetModel.buildModel();

        // === 4. Ajout d’un listener ===
        // Permet de suivre la progression et d’afficher un score intermédiaire tous les 10 lots
        model.setListeners(new ScoreIterationListener(10));

        // === 5. Entraînement du modèle ===
        System.out.println("\n=== ÉTAPE 3 : Entraînement ===");
        System.out.println("Epochs : " + N_EPOCHS);
        System.out.println("Batch size : " + BATCH_SIZE + "\n");

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            System.out.println("╔════════════════════════════════════════╗");
            System.out.println("║       EPOCH " + (epoch + 1) + "/" + N_EPOCHS + "                        ║");
            System.out.println("╚════════════════════════════════════════╝");
            
            model.fit(trainIter);	// Apprentissage sur une époque complète

            System.out.println("\n--- Epoch " + (epoch + 1) + " terminée ---");

            // Évaluation intermédiaire
            System.out.println("\n📊 Évaluation intermédiaire...");
            Evaluation eval = model.evaluate(testIter);
            System.out.println("Accuracy : " + String.format("%.2f%%", eval.accuracy() * 100));
            System.out.println("Precision : " + String.format("%.2f%%", eval.precision() * 100));
            System.out.println("Recall : " + String.format("%.2f%%", eval.recall() * 100));

            // On remet les itérateurs à zéro pour la prochaine époque
            testIter.reset();
            trainIter.reset();
            System.out.println();
        }

        // === 6. Évaluation finale sur le dataset de test complet ===
        System.out.println("\n=== ÉTAPE 4 : Évaluation finale ===");
        Evaluation finalEval = model.evaluate(testIter);
        System.out.println(finalEval.stats());

        // === 7. Sauvegarde du modèle entraîné ===
        System.out.println("\n=== ÉTAPE 5 : Sauvegarde du modèle ===");
        ResNetModel.saveModelWithLabels(model, MODEL_PATH, DataLoader.getLabels());

        // === Fin de l’entraînement ===
        System.out.println("\n╔════════════════════════════════════════╗");
        System.out.println("║    ✅ ENTRAÎNEMENT TERMINÉ !           ║");
        System.out.println("╚════════════════════════════════════════╝");
        System.out.println("\nPour tester une prédiction :");
        System.out.println("  java Main predict chemin/vers/image.jpg");
    }

    /**
     * MODE PRÉDICTION du modèle :
     * Cette méthode charge le modèle sauvegardé, puis appelle la classe {@link Predictor} pour effectuer une prédiction sur l'image passée en argument.
     *
     * @param imagePath chemin vers l’image à prédire
     * @throws Exception si l’image est introuvable ou si la prédiction échoue
     */
    private static void runPrediction(String imagePath) throws Exception {
        System.out.println("╔════════════════════════════════════════╗");
        System.out.println("║   MODE PRÉDICTION - ACS Project        ║");
        System.out.println("╚════════════════════════════════════════╝\n");

        // Vérifie que le fichier d’image existe bien
        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            throw new Exception("Image introuvable : " + imagePath);
        }

        System.out.println("Image : " + imageFile.getName());
        System.out.println();

        // Appel de la méthode de prédiction
        Predictor.predict(MODEL_PATH, imageFile);
    }
}
