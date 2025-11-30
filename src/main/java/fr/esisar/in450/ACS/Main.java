package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import java.io.File;
import java.util.Scanner;

/**
 * Classe principale du projet ACS (Analyse et Classification d'Images).
 * Cette classe orchestre l'ensemble du processus d'entraînement et de prédiction d'un modèle de Deep Learning (basé sur VGG16) en utilisant la bibliothèque Deeplearning4j.
 * Utilisation :
 *   - java Main< : lance le mode entraînement
 *   - java Main predict chemin/vers/image.jpg : lance le mode prédiction
 */
public class Main {
    private static final int HEIGHT = 224;
    private static final int WIDTH = 224;
    private static final int CHANNELS = 3;
    private static final int NUM_CLASSES = 6;
    private static final int BATCH_SIZE = 4;
    private static final int N_EPOCHS = 8;
    private static final String MODEL_TYPE_COULEUR_PATH = "modele-type-couleur.zip";
    private static final String MODEL_TYPE_PATH = "modele-type.zip";
    private static final String TRAIN_PATH = "src/main/resources/dataset/train";
    private static final String TEST_PATH = "src/main/resources/dataset/test";

    /**
     * Point d'entrée du programme.
     * Selectionne le mode ENTRAINEMENT ou PREDICTION selon si @param args contient "predict"
     */
    public static void main(String[] args) {
        Scanner scan = new Scanner(System.in);
        try {
            int choix = -1;
            while (choix != 0) {
                System.out.println("\n===== Menu ACS =====");
                System.out.println("0 : Quitter");
                System.out.println("1 : Entraîner le modèle");
                System.out.println("2 : Prédire type + couleur (modele-type-couleur.zip)");
                System.out.println("3 : Prédire type (modele-type.zip) + couleur (OpenCV)");
                System.out.print("Votre choix : ");
                
                while (!scan.hasNextInt()) {
                    System.out.print("Veuillez entrer un nombre : ");
                    scan.next();
                }
                choix = scan.nextInt();
                scan.nextLine();

                switch (choix) {
                    case 1:
                        runTraining();
                        choix = 0;
                        break;
                    case 2:
                        System.out.print("Chemin vers l'image : ");
                        String path2 = scan.nextLine();
                        runPredictionTypeCouleur(path2);
                        break;
                    case 3:
                        System.out.print("Chemin vers l'image : ");
                        String path3 = scan.nextLine();
                        runPredictionTypeWithOpenCV(path3);
                        break;
                    case 0:
                        System.out.println("Au revoir !");
                        break;
                    default:
                        System.out.println("Choix invalide.");
                }
            }
        } 
        catch (Exception e) {
            System.err.println("ERREUR : " + e.getMessage());
            e.printStackTrace();
        }
        finally {
            scan.close();
        }
    }

    /**
     * MODE ENTRAÎNEMENT du modèle:
     *   - Chargement du jeu de données d'entraînement et de test
     *   - Initialisation des labels dans la classe utilitaire
     *   - Construction du modèle VGG16
     *   - Entraînement sur plusieurs époques avec suivi du score<
     *   - Évaluation finale sur le jeu de test
     *   - Sauvegarde du modèle entraîné
     *
     * @throws Exception si une erreur survient pendant le processus
     */
    private static void runTraining() throws Exception {
        System.out.println("===== MODE ENTRAÎNEMENT=====");

        // === 1. Chargement des datasets ===
        System.out.println("=== ÉTAPE 1 : Chargement des données ===");
        DataSetIterator trainIter = DataLoader.getIterator(TRAIN_PATH, HEIGHT, WIDTH, CHANNELS, BATCH_SIZE, NUM_CLASSES);
        DataSetIterator testIter = DataLoader.getIterator(TEST_PATH, HEIGHT, WIDTH, CHANNELS, BATCH_SIZE, NUM_CLASSES);

        // === 2. Initialisation des labels ===
        Utils.initLabels(DataLoader.getLabels());

        // === 3. Construction du modèle ===
        System.out.println("\n=== ÉTAPE 2 : Construction du modèle ===");
        ComputationGraph model = VGG16Model.buildModel();

        // === 4. Ajout d'un listener ===
        model.setListeners(new ScoreIterationListener(10));

        // === 5. Entraînement du modèle ===
        System.out.println("\n=== ÉTAPE 3 : Entraînement ===");
        System.out.println("Epochs : " + N_EPOCHS);
        System.out.println("Batch size : " + BATCH_SIZE + "\n");

        for (int epoch = 0; epoch < N_EPOCHS; epoch++) {
            System.out.println("===== EPOCH " + (epoch + 1) + "/" + N_EPOCHS + "=====");
            
            model.fit(trainIter);

            System.out.println("\n--- Epoch " + (epoch + 1) + " terminée ---");

            // Évaluation intermédiaire
            System.out.println("\nÉvaluation intermédiaire...");
            Evaluation eval = model.evaluate(testIter);
            System.out.println("Accuracy : " + String.format("%.2f%%", eval.accuracy() * 100));
            System.out.println("Precision : " + String.format("%.2f%%", eval.precision() * 100));
            System.out.println("Recall : " + String.format("%.2f%%", eval.recall() * 100));

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
        VGG16Model.saveModelWithLabels(model, MODEL_TYPE_PATH, DataLoader.getLabels());

        // === Fin de l'entraînement ===
        System.out.println("===== ENTRAÎNEMENT TERMINÉ ! =====");
    }

    /**
     * MODE PRÉDICTION avec modèle type-couleur (option 2) :
     * Utilise le modèle modele-type-couleur.zip qui prédit directement type ET couleur
     *
     * @param imagePath chemin vers l'image à prédire
     * @throws Exception si l'image est introuvable ou si la prédiction échoue
     */
    private static void runPredictionTypeCouleur(String imagePath) throws Exception {
        System.out.println("===== MODE PRÉDICTION TYPE+COULEUR (modele-type-couleur.zip) =====");

        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            throw new Exception("Image introuvable : " + imagePath);
        }

        System.out.println("Image : " + imageFile.getName());
        System.out.println();

        // Prédiction avec le modèle complet type-couleur
        Predictor.predict(MODEL_TYPE_COULEUR_PATH, imageFile);
    }

    /**
     * MODE PRÉDICTION avec modèle type + OpenCV pour couleur (option 3) :
     * 1. Utilise modele-type.zip pour prédire le TYPE de bonbon
     * 2. Utilise OpenCV (ColorDetector) pour détecter la COULEUR
     *    en tenant compte des couleurs possibles pour ce type
     *
     * @param imagePath chemin vers l'image à prédire
     * @throws Exception si l'image est introuvable ou si la prédiction échoue
     */
    private static void runPredictionTypeWithOpenCV(String imagePath) throws Exception {
        System.out.println("===== MODE PRÉDICTION TYPE (modele-type.zip) + COULEUR (OpenCV) =====");

        File imageFile = new File(imagePath);
        if (!imageFile.exists()) {
            throw new Exception("Image introuvable : " + imagePath);
        }

        System.out.println("Image : " + imageFile.getName());
        System.out.println();

        // Prédiction du type avec modele-type.zip + détection couleur OpenCV
        PredictorWithColor.predictTypeAndColor(MODEL_TYPE_PATH, imageFile);
    }
}