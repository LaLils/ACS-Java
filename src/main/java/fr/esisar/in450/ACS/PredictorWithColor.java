package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.List;

/**
 * Classe de prédiction combinant :
 * 1. Un modèle VGG16 pour prédire le TYPE de bonbon
 * 2. OpenCV (ColorDetector) pour détecter la COULEUR en tenant compte du type
 */
public class PredictorWithColor {

    /**
     * Prédit le type de bonbon avec le modèle, puis détecte la couleur
     * en utilisant OpenCV avec contrainte selon le type détecté.
     *
     * @param modelPath chemin vers le modèle de type (modele-type.zip)
     * @param imageFile image à analyser
     * @throws Exception si le modèle ou l'image ne peuvent pas être chargés
     */
    public static void predictTypeAndColor(String modelPath, File imageFile) throws Exception {
        // === 1. Chargement du modèle de type ===
        System.out.println("Chargement du modèle: "+modelPath);
        ComputationGraph model = ModelSerializer.restoreComputationGraph(modelPath);

        // === 2. Chargement des labels ===
        List<String> labels = VGG16Model.loadLabels(modelPath);
        Utils.initLabels(labels);

        // === 3. Définition des paramètres d'entrée ===
        int height = 224;
        int width = 224;
        int channels = 3;

        // === 4. Chargement et prétraitement de l'image ===
        System.out.println("Prétraitement de l'image...");
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray image = loader.asMatrix(imageFile);
        ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(image);

        // === 5. Prédiction du TYPE ===
        System.out.println("Prédiction du type en cours...\n");
        INDArray output = model.outputSingle(image);
        int predicted = output.argMax(1).getInt(0);
        double confidence = output.getDouble(0, predicted);
        String predictedType = Utils.getClassName(predicted);

        System.out.println("===== RÉSULTAT TYPE =====");
        System.out.println("  Type de bonbon : " + predictedType);
        System.out.println("  Index classe   : " + predicted);
        System.out.println("  Confiance      : " + String.format("%.2f%%", confidence * 100));
        System.out.println();

        System.out.println("    Top 3 types prédits :");
        displayTopPredictions(output, labels, 3);

        // === 6. Détection de la COULEUR avec OpenCV ===
        System.out.println("===== RÉSULTAT COULEUR (OpenCV) =====");
        String couleurDetectee = ColorDetector.detectColorForType(imageFile, predictedType);

        // === 7. Affichage du résultat final ===
        System.out.println("===== RÉSULTAT FINAL =====");
        System.out.println("  Type    : " + predictedType);
        System.out.println("  Couleur : " + couleurDetectee);
        System.out.println("  Label complet : " + predictedType + "-" + couleurDetectee);
        System.out.println();

    }

    /**
     * Affiche les top N prédictions de type.
     *
     * @param output sortie du modèle (vecteur de probabilités)
     * @param labels liste des labels correspondant aux classes
     * @param topN   nombre de prédictions à afficher
     */
    private static void displayTopPredictions(INDArray output, List<String> labels, int topN) {
        INDArray outputCopy = output.dup();
        for (int i = 0; i < Math.min(topN, labels.size()); i++) {
            int idx = outputCopy.argMax(1).getInt(0);
            double conf = outputCopy.getDouble(0, idx);
            String label = labels.get(idx);
            System.out.println(String.format("    %d. %s : %.2f%%", (i + 1), label, conf * 100));
            outputCopy.putScalar(new int[] { 0, idx }, -1.0);
        }
        System.out.println();
    }
}