package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.graph.ComputationGraph;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.List;

/**
 * Classe utilitaire chargée de faire des prédictions à partir d’un modèle ResNet sauvegardé. 
 * Le {@code Predictor} effectue le pipeline complet : -
 * - Chargement du modèle depuis un fichier ZIP 
 * - Chargement et initialisation des
 * labels 
 * - Prétraitement d’une image donnée (redimensionnement + normalisation)
 * - Exécution de la prédiction via le modèle 
 * - Affichage du résultat (classe prédite + confiance + top 3)
 */
public class Predictor {

	/**
	 * Charge un modèle sauvegardé et prédit la classe d’une image donnée.
	 *
	 * @param modelPath chemin vers le modèle sauvegardé (fichier .zip)
	 * @param imageFile image à analyser
	 * @throws Exception si le modèle ou l’image ne peuvent pas être chargés
	 */
	public static void predict(String modelPath, File imageFile) throws Exception {
		// === 1. Chargement du modèle ===
        System.out.println("Chargement du modèle: "+modelPath);
		ComputationGraph model = ModelSerializer.restoreComputationGraph(modelPath);

		// === 2. Chargement des labels sauvegardés ===
		List<String> labels = VGG16Model.loadLabels(modelPath);
		Utils.initLabels(labels); // Initialisation de la classe utilitaire avec les labels

		// === 3. Définition des paramètres d’entrée ===
		int height = 224;
		int width = 224;
		int channels = 3;

		// === 4. Chargement et prétraitement de l’image ===
		System.out.println("Prétraitement de l'image...");
		NativeImageLoader loader = new NativeImageLoader(height, width, channels);
		INDArray image = loader.asMatrix(imageFile); // Convertit l’image en matrice ND4J
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); // Normalisation : échelle des pixels
																				// entre [0, 1]
		scaler.transform(image);

		// === 5. Prédiction ===
		System.out.println("Prédiction en cours...\n");
		INDArray output = model.outputSingle(image); // Sortie du modèle : vecteur de probabilités pour chaque classe
		int predicted = output.argMax(1).getInt(0); // Index de la classe ayant la plus forte probabilité
		double confidence = output.getDouble(0, predicted); // Confiance associée à cette classe (probabilité)

		// === 6. Interprétation du résultat ===
		String predictedLabel = Utils.getClassName(predicted);

		// === 7. Affichage formaté du résultat principal ===
		System.out.println("===== RÉSULTAT =====");
		System.out.println("  Type de bonbon : " + predictedLabel);
		System.out.println("  Index classe  : " + predicted);
		System.out.println("  Confiance : " + String.format("%.2f%%", confidence * 100));
		System.out.println();

		// === 8. Affichage des N meilleures classes ===
		System.out.println("    Top 5 des predictions");
		displayTopPredictions(output, labels, 5);
	}

	/**
	 * Affiche les {@code topN} meilleures prédictions à partir de la sortie du
	 * modèle. *
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
