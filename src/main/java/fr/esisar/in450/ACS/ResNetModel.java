package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearning.GraphBuilder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * Classe utilitaire responsable de la construction, sauvegarde et chargement du modèle ResNet50. 
 * Cette classe repose sur le principe du transfer learning :
 * - Charge un modèle ResNet50 pré-entraîné (ImageNet) 
 * - Adapte sa couche finale pour la classification spécifique 
 * - Sauvegarde et charge le modèle avec les labels associés
 */
public class ResNetModel {

	private static final int NUM_CLASSES = 18;	// Nombre total de classes de sortie pour notre projet 


	/**
	 * Construit un modèle basé sur ResNet50 via le transfer learning.
	 * Le modèle charge les poids pré-entraînés sur ImageNet, gèle les couches
	 * jusqu’à "activation_49", puis remplace la couche de sortie ("fc1000") par une
	 * nouvelle couche dense adaptée au nombre de classes locales.
	 *
	 * @return un {@link ComputationGraph} prêt à être entraîné
	 * @throws Exception en cas d’échec du chargement du modèle pré-entraîné
	 */
	public static ComputationGraph buildModel() throws Exception {
		System.out.println("Chargement de ResNet50 pré-entraîné...");

		// === 1. Chargement du modèle ResNet50 pré-entraîné sur ImageNet ===
		ZooModel pretrained = ResNet50.builder().build();
		ComputationGraph pretrainedNet = (ComputationGraph) pretrained.initPretrained();
		System.out.println("✅ ResNet50 chargé avec succès");
		System.out.println("\nRésumé du modèle pré-entraîné :");
		System.out.println(pretrainedNet.summary());

		// === 2. Application du transfert d’apprentissage ===
		System.out.println("\nApplication du transfer learning...");		
        GraphBuilder graphBuilder = new TransferLearning.GraphBuilder(pretrainedNet)	// On crée un nouveau "GraphBuilder" basé sur le modèle existant.
                .setFeatureExtractor("activation_49")	// Tout ce qui est avant "activation_49" est gelé (non réentraîné)
                .removeVertexKeepConnections("fc1000")	// On supprime uniquement la dernière couche fully-connected (fc1000)
                .addLayer("fc18",	// Ajout d'une nouvelle couche de sortie adaptée à nos classes
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(2048) // nombre d’entrées venant de la couche précédente (ResNet)
                                .nOut(NUM_CLASSES) // notre nombre de classes spécifiques
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX) // softmax pour classification multi-classes
                                .build(),
                        "flatten_1") // on se connecte à "flatten_1", déjà aplatie
                .setOutputs("fc18");	// On définit la nouvelle sortie du graphe
        
		// === 3. Construction du modèle final ===
		ComputationGraph model = graphBuilder.build();
		model.init();
		System.out.println("Modèle de transfer learning construit avec succès !");
		System.out.println("   - Couches gelées : jusqu'à activation_49");
		System.out.println("   - Nouvelle couche de sortie : " + NUM_CLASSES + " classes");

		return model;
	}

	/**
	 * Sauvegarde le modèle et la liste des labels associée.
	 * Le modèle est sauvegardé dans un fichier ZIP, et les labels sont enregistrés dans un fichier texte à côté
	 *
	 * @param model     modèle DL4J entraîné à sauvegarder
	 * @param modelPath chemin du fichier ZIP à créer
	 * @param labels    liste des labels (ordre utilisé par le modèle)
	 * @throws Exception si la sauvegarde échoue
	 */
	public static void saveModelWithLabels(ComputationGraph model, String modelPath, List<String> labels) throws Exception {
		ModelSerializer.writeModel(model, modelPath, true);	// Sauvegarde du graphe computationnel (architecture + poids)
		String labelPath = modelPath.replace(".zip", "_labels.txt");	// Sauvegarde des labels dans un fichier séparé
		Files.write(Paths.get(labelPath), labels);
		System.out.println("\nModèle sauvegardé : " + modelPath);
		System.out.println("Labels sauvegardés : " + labelPath);
	}

	/**
	 * Charge les labels associés à un modèle sauvegardé.
	 *
	 * @param modelPath chemin du fichier modèle (fichier .zip)
	 * @return liste des labels dans l’ordre correct
	 * @throws Exception si le fichier de labels est introuvable
	 */
	public static List<String> loadLabels(String modelPath) throws Exception {
		String labelPath = modelPath.replace(".zip", "_labels.txt");
		File labelFile = new File(labelPath);

		if (!labelFile.exists()) {
			throw new Exception("Fichier de labels introuvable : " + labelPath);
		}
		List<String> labels = Files.readAllLines(Paths.get(labelPath));
		System.out.println("Labels chargés : " + labels.size() + " classes");
		return labels;
	}

	/**
	 * Sauvegarde le modèle sans les labels (obsolète).
	 *
	 * @param model modèle à sauvegarder
	 * @param path  chemin du fichier de sortie
	 * @throws Exception si la sauvegarde échoue
	 */
	@Deprecated
	public static void saveModel(ComputationGraph model, String path) throws Exception {
		ModelSerializer.writeModel(model, path, true);
		System.out.println("Modèle sauvegardé sans les labels : " + path);
	}
}
