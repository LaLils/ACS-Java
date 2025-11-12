package fr.esisar.in450.ACS;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.util.List;
import java.util.Random;

/**
 * Classe utilitaire responsable du chargement et de la préparation des jeux de données images. 
 * Le DataLoader parcourt un répertoire d’images organisé par classes (chaque sous-dossier correspondant à une étiquette), 
 * puis prépare un {@link DataSetIterator} utilisable directement par le modèle de DeepLearning.
 */
public class DataLoader {
	private static List<String> labels = null; // Liste des labels (noms des classes) détectés lors du chargement

	/**
	 * Crée un DataSetIterator pour la classification d'images (ImageRecordReader)
	 * 
	 * @param path       chemin vers le dossier (contenant des sous-dossiers par
	 *                   classe)
	 * @param height     hauteur d'entrée attendue par le modèle (ici 224)
	 * @param width      largeur d'entrée (ici 224)
	 * @param channels   3 pour RGB
	 * @param batchSize  taille de batch
	 * @param numClasses nombre total de classes (ici 18)
	 * @return DataSetIterator
	 * @throws Exception
	 */
	public static DataSetIterator getIterator(String path, int height, int width, int channels, int batchSize,
			int numClasses) throws Exception {

		// === Vérification du dossier d’entrée ===
		File parentDir = new File(path);
		if (!parentDir.exists())
			throw new IllegalArgumentException("Le dossier '" + path + "' n'existe pas.");

		// === 1. Création d’un FileSplit ===
		FileSplit fileSplit = new FileSplit(parentDir, NativeImageLoader.ALLOWED_FORMATS, new Random(123)); // FileSplit
																											// parcourt
																											// récursivement
																											// le
																											// dossier
																											// et
																											// récupère
																											// les
																											// chemins
																											// de toutes
																											// les
																											// images.

		// === 2. Génération automatique des labels ===
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		// === 3. Lecture des images ===
		ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker); // Convertit chaque
																										// image en
																										// matrice de
																										// pixel et y
																										// associe un
																										// label
		recordReader.initialize(fileSplit); // Initialise le lecteur avec le FileSplit (définit les fichiers à lire)

		// === 4. Récupération et affichage des labels détectés ===
		labels = recordReader.getLabels();
		System.out.println("✅ Classes détectées (ordre alphabétique) : " + labels);
		System.out.println("   Nombre de classes : " + labels.size());

		// === 5. Création de l’itérateur DL4J ===
		DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses); // Le
																										// RecordReaderDataSetIterator
																										// convertit
																										// chaque image
																										// + label en un
																										// DataSet ND4J.

		// === 6. Normalisation des pixels ===
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1); // Convertit les valeurs des pixels
																				// (0–255) en une plage [0,1]
		iter.setPreProcessor(scaler);

		return iter;
	}

	/**
	 * Retourne la liste des labels détectés lors du dernier chargement de dataset.
	 *
	 * @return liste ordonnée des noms de classes (labels)
	 * @throws IllegalStateException si le DataLoader n’a pas encore été initialisé
	 */
	public static List<String> getLabels() {
		if (labels == null) {
			throw new IllegalStateException("Les labels n'ont pas été initialisés. Appelez getIterator() d'abord.");
		}
		return labels;
	}
}
