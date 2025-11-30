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
 * Classe utilitaire responsable du chargement et de la préparation des jeux de données d'images.
 * 
 * Le DataLoader parcourt un répertoire d'images organisé par classes (chaque sous-dossier 
 * correspondant à une étiquette), puis prépare un DataSetIterator utilisable directement 
 * par le modèle de Deep Learning.
 */
public class DataLoader {
	private static List<String> labels = null;

	/**
	 * Crée un DataSetIterator pour la classification d'images.
	 * 
	 * Cette méthode effectue les opérations suivantes :
	 * 1. Vérifie l'existence du dossier source
	 * 2. Parcourt récursivement le dossier pour trouver toutes les images
	 * 3. Génère automatiquement les labels depuis les noms de sous-dossiers
	 * 4. Convertit chaque image en matrice de pixels
	 * 5. Normalise les valeurs des pixels entre 0 et 1
	 * 
	 * @param path       Chemin vers le dossier contenant les sous-dossiers par classe
	 * @param height     Hauteur de redimensionnement des images (224 pour VGG16)
	 * @param width      Largeur de redimensionnement des images (224 pour VGG16)
	 * @param channels   Nombre de canaux (3 pour RGB, 1 pour niveaux de gris)
	 * @param batchSize  Nombre d'images traitées simultanément (limité par la RAM)
	 * @param numClasses Nombre total de classes dans le dataset
	 * @return DataSetIterator prêt à être utilisé pour l'entraînement ou l'évaluation
	 * @throws Exception Si le dossier n'existe pas ou si la lecture échoue
	 */
	public static DataSetIterator getIterator(String path, int height, int width, int channels, int batchSize,
			int numClasses) throws Exception {

		// === ÉTAPE 1 : Vérification de l'existence du dossier d'entrée ===
		File parentDir = new File(path);
		if (!parentDir.exists())
			throw new IllegalArgumentException("Le dossier '" + path + "' n'existe pas.");

		// === ÉTAPE 2 : Création d'un FileSplit pour parcourir les fichiers ===
		FileSplit fileSplit = new FileSplit(
				parentDir,
				NativeImageLoader.ALLOWED_FORMATS,
				new Random(123));

		// === ÉTAPE 3 : Génération automatique des labels depuis les noms de dossiers ===
		ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

		// === ÉTAPE 4 : Création du lecteur d'images ===
		ImageRecordReader recordReader = new ImageRecordReader(	// Crée le lecteur qui va convertir les images
				height,
				width,
				channels,
				labelMaker);
		recordReader.initialize(fileSplit);	// Initialise le lecteur avec la liste des fichiers à traiter

		// === ÉTAPE 5 : Récupération et affichage des labels détectés ===
		labels = recordReader.getLabels();
		System.out.println("Classes détectées (ordre alphabétique) : " + labels);
		System.out.println("   Nombre de classes : " + labels.size());

		// === ÉTAPE 6 : Création de l'itérateur DL4J ===
		DataSetIterator iter = new RecordReaderDataSetIterator(	// Crée l'itérateur final
				recordReader,	// Lecteur d'images configuré
				batchSize,	// Nombre d'images par batch
				1,	// Index de la colonne contenant le label (1 pour images)
				numClasses);	// Nombre total de classes pour l'encodage one-hot

		// === ÉTAPE 7 : Normalisation des valeurs de pixels ===
		ImagePreProcessingScaler scaler = new ImagePreProcessingScaler(0, 1);
		iter.setPreProcessor(scaler);

		return iter;
	}

	/**
	 * Retourne la liste des labels détectés lors du dernier chargement de dataset.
	 * 
	 * Cette méthode permet de récupérer les noms de classes après avoir appelé getIterator().
	 * Les labels sont dans l'ordre alphabétique tel que déterminé par le système de fichiers.
	 *
	 * @return Liste ordonnée des noms de classes (labels)
	 * @throws IllegalStateException Si getIterator() n'a pas encore été appelé
	 */
	public static List<String> getLabels() {
		if (labels == null) {
			throw new IllegalStateException("Les labels n'ont pas été initialisés. Appelez getIterator() d'abord.");
		}
		return labels;
	}
}