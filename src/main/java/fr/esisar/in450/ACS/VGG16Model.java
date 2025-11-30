package fr.esisar.in450.ACS;

import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearning.GraphBuilder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * Classe utilitaire pour construire, sauvegarder et charger un modèle VGG16
 * adapté à la classification de bonbons via transfer learning.
 * 
 * 1. Chargement du modèle VGG16 pré-entraîné
 * 2. Gel des couches convolutives
 * 3. Remplacement de la couche de sortie pour nos 6 classes
 * 4. Entraînement uniquement de la nouvelle couche de sortie
 * 
 * Architecture modifiée :
 * - Couches convolutives : GELÉES (non entraînées)
 * - fc1, fc2 : GELÉES
 * - fc18 (nouvelle) : ENTRAÎNÉE pour nos 6 classes
 */
public class VGG16Model {
    private static final int NUM_CLASSES = 6;

    /**
     * Construit un modèle VGG16 adapté via transfer learning.
     * 
     * Cette méthode effectue les opérations suivantes :
     * 1. Charge le modèle VGG16 pré-entraîné depuis le Model Zoo de DL4J
     * 2. Affiche un résumé du modèle original
     * 3. Configure le transfer learning :
     *    - Gèle toutes les couches jusqu'à fc2 (incluse)
     *    - Supprime la couche "predictions" originale (1000 classes ImageNet)
     *    - Ajoute une nouvelle couche "fc18" pour nos 6 classes
     * 4. Initialise le nouveau modèle
     * 
     * @return ComputationGraph configuré et prêt pour l'entraînement
     * @throws Exception Si le chargement du modèle échoue
     */
    public static ComputationGraph buildModel() throws Exception {
        System.out.println("Chargement de VGG16 pré-entraîné...");

        // === ÉTAPE 1 : Chargement du modèle pré-entraîné ===
        ZooModel pretrained = VGG16.builder().build();	// Crée un builder pour VGG16 depuis le Model Zoo
        ComputationGraph pretrainedNet = (ComputationGraph) pretrained.initPretrained();	// Charge les poids pré-entraînés d'ImageNet
        System.out.println("VGG16 chargé avec succès");
        System.out.println("\nRésumé du modèle pré-entraîné :");
        System.out.println(pretrainedNet.summary());

        // === ÉTAPE 2 : Application du transfer learning ===
        System.out.println("\nApplication du transfer learning...");
        GraphBuilder graphBuilder = new TransferLearning.GraphBuilder(pretrainedNet)	// Crée un builder pour modifier le graphe
                .setFeatureExtractor("fc2")	// Gèle toutes les couches jusqu'à fc2 (incluse)
                .removeVertexKeepConnections("predictions")	// Supprime la couche finale "predictions"
                .addLayer("fc18", // Ajoute une nouvelle couche nommée "fc18"
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // Configure la fonction de perte
                                .nIn(4096)	// Nombre d'entrées provenant de la couche fc2
                                .nOut(NUM_CLASSES)	// Nombre de sortie de la nouvele couche fc18
                                .weightInit(WeightInit.XAVIER)	// Initialisation des poids selon la méthode Xavier
                                .activation(Activation.SOFTMAX)	// Fonction d'activation softmax (produit des probabilités qui somment à 1)
                                .build(),	// Construit la configuration de la couche
                        "fc2")	// Spécifie que fc18 prend fc2 comme entrée
                .setOutputs("fc18");	// Définit fc18 comme couche de sortie du réseau

        // === ÉTAPE 3 : Construction et initialisation du nouveau modèle ===
        ComputationGraph model = graphBuilder.build();	// Construit le nouveau graphe avec nos modifications
        model.init();	// Initialise les paramètres du modèle (poids de la nouvelle couche fc18)
        System.out.println("Modèle de transfer learning construit avec succès !");
        System.out.println("   - Couches gelées : jusqu'à fc2");
        System.out.println("   - Nouvelle couche de sortie : " + NUM_CLASSES + " classes");

        return model;
    }

    /**
     * Sauvegarde le modèle et la liste des labels associée.
     * 
     * Cette méthode sauvegarde deux fichiers :
     * 1. Le modèle complet (architecture + poids) au format ZIP
     * 2. Les labels dans un fichier texte séparé (un label par ligne)
     * 
     * @param model Modèle entraîné à sauvegarder
     * @param modelPath Chemin où sauvegarder le modèle (doit se terminer par .zip)
     * @param labels Liste des labels dans l'ordre correspondant aux sorties du modèle
     * @throws Exception Si l'écriture des fichiers échoue
     */
    public static void saveModelWithLabels(ComputationGraph model, String modelPath, List<String> labels) throws Exception {
        ModelSerializer.writeModel(model, modelPath, true);
        String labelPath = modelPath.replace(".zip", "_labels.txt");
        Files.write(Paths.get(labelPath), labels);
        System.out.println("\nModèle sauvegardé : " + modelPath);
    }

    /**
     * Charge les labels associés à un modèle sauvegardé.
     * 
     * Cette méthode lit le fichier texte contenant les labels (un par ligne)
     * et retourne une liste dans l'ordre correspondant aux sorties du modèle.
     * 
     * Le fichier de labels doit avoir le même nom que le modèle avec le suffixe _labels.txt
     * 
     * @param modelPath Chemin vers le fichier modèle (.zip)
     * @return Liste des labels dans l'ordre d'entraînement
     * @throws Exception Si le fichier de labels est introuvable ou illisible
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
}