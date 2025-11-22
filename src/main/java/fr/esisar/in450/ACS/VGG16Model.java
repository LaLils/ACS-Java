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
 * adapté à 18 classes de bonbons via transfer learning.
 */
public class VGG16Model {

    private static final int NUM_CLASSES = 18; // Nombre de classes

    /**
     * Construit un modèle VGG16 via transfer learning :
     * - Charge VGG16 pré-entraîné sur ImageNet
     * - Gèle les couches jusqu'à fc2
     * - Remplace la dernière couche par une couche adaptée à nos classes
     */
    public static ComputationGraph buildModel() throws Exception {
        System.out.println("Chargement de VGG16 pré-entraîné...");

        // 1. Chargement du modèle pré-entraîné
        ZooModel pretrained = VGG16.builder().build();
        ComputationGraph pretrainedNet = (ComputationGraph) pretrained.initPretrained();
        System.out.println("✅ VGG16 chargé avec succès");
        System.out.println("\nRésumé du modèle pré-entraîné :");
        System.out.println(pretrainedNet.summary());

        // 2. Application du transfer learning
        System.out.println("\nApplication du transfer learning...");
        GraphBuilder graphBuilder = new TransferLearning.GraphBuilder(pretrainedNet)
                .setFeatureExtractor("fc2") // On gèle tout avant fc2
                .removeVertexKeepConnections("predictions") // On supprime la couche finale
                .addLayer("fc18",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096) // nombre d'entrées provenant de fc2
                                .nOut(NUM_CLASSES)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .build(),
                        "fc2") // connecte la nouvelle couche à fc2
                .setOutputs("fc18");

        ComputationGraph model = graphBuilder.build();
        model.init();
        System.out.println("Modèle de transfer learning construit avec succès !");
        System.out.println("   - Couches gelées : jusqu'à fc2");
        System.out.println("   - Nouvelle couche de sortie : " + NUM_CLASSES + " classes");

        return model;
    }

    /**
     * Sauvegarde le modèle et la liste des labels associée.
     */
    public static void saveModelWithLabels(ComputationGraph model, String modelPath, List<String> labels) throws Exception {
        ModelSerializer.writeModel(model, modelPath, true);
        String labelPath = modelPath.replace(".zip", "_labels.txt");
        Files.write(Paths.get(labelPath), labels);
        System.out.println("\nModèle sauvegardé : " + modelPath);
        System.out.println("Labels sauvegardés : " + labelPath);
    }

    /**
     * Charge les labels associés à un modèle sauvegardé.
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
     * Sauvegarde le modèle sans labels (obsolète).
     */
    @Deprecated
    public static void saveModel(ComputationGraph model, String path) throws Exception {
        ModelSerializer.writeModel(model, path, true);
        System.out.println("Modèle sauvegardé sans les labels : " + path);
    }
}
