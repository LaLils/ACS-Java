package fr.esisar.in450.ACS;

import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;

import java.io.File;
import java.util.Random;

public class CandyTypeClassifier {
    
    private static final int HEIGHT = 128;  // Hauteur des images
    private static final int WIDTH = 128;   // Largeur des images
    private static final int CHANNELS = 3;  // RGB (3 canaux)
    private static final int NUM_CLASSES = 6; // 6 types de bonbons
    private static final int BATCH_SIZE = 16;
    private static final int EPOCHS = 20;
    
    private MultiLayerNetwork model;
    
    /**
     * Crée l'architecture du réseau de neurones convolutif (CNN)
     */
    public void buildModel() {
        System.out.println("Construction du modèle CNN...");
        
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            // Couche 1 : Convolution + Pooling
            .layer(0, new ConvolutionLayer.Builder(5, 5)
                .nIn(CHANNELS)
                .stride(1, 1)
                .nOut(32)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // Couche 2 : Convolution + Pooling
            .layer(2, new ConvolutionLayer.Builder(5, 5)
                .stride(1, 1)
                .nOut(64)
                .activation(Activation.RELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            
            // Couche 3 : Dense (fully connected)
            .layer(4, new DenseLayer.Builder()
                .activation(Activation.RELU)
                .nOut(128)
                .build())
            
            // Couche de sortie
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(NUM_CLASSES)
                .activation(Activation.SOFTMAX)
                .build())
            
            .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
            .build();
        
        model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));
        
        System.out.println("Modèle créé avec succès !");
        System.out.println("Nombre de paramètres: " + model.numParams());
    }
    
    /**
     * Entraîne le modèle avec les images du dossier
     */
    public void train(String datasetPath) throws Exception {
        buildModel();
        
        File trainData = new File(datasetPath);
        if (!trainData.exists()) {
            throw new Exception("Dossier d'entraînement introuvable: " + datasetPath);
        }
        
        // Préparer le lecteur d'images
        FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ImageRecordReader trainRR = new ImageRecordReader(HEIGHT, WIDTH, CHANNELS, new ParentPathLabelGenerator());
        trainRR.initialize(trainSplit);
        
        // Créer l'itérateur de données
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, BATCH_SIZE, 1, NUM_CLASSES);
        
        System.out.println("\n=== Début de l'entraînement ===");
        System.out.println("Epochs: " + EPOCHS);
        System.out.println("Batch size: " + BATCH_SIZE);
        System.out.println("Images d'entraînement: ~" + trainSplit.length());
        
        // Entraînement
        for (int i = 0; i < EPOCHS; i++) {
            System.out.println("\nEpoch " + (i + 1) + "/" + EPOCHS);
            model.fit(trainIter);
            trainIter.reset();
        }
        
        System.out.println("\n=== Entraînement terminé ! ===");
    }
    
    /**
     * Sauvegarde le modèle entraîné
     */
    public void saveModel(String path) throws Exception {
        File modelFile = new File(path);
        modelFile.getParentFile().mkdirs(); // Créer le dossier si nécessaire
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("Modèle sauvegardé: " + path);
    }
    
    /**
     * Charge un modèle déjà entraîné
     */
    public void loadModel(String path) throws Exception {
        File modelFile = new File(path);
        if (!modelFile.exists()) {
            throw new Exception("Modèle introuvable: " + path);
        }
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        System.out.println("Modèle chargé: " + path);
    }
    
    /**
     * Prédit le type de bonbon
     */
    public String[] predict(String imagePath) throws Exception {
        if (model == null) {
            throw new Exception("Aucun modèle chargé !");
        }
        
        // Charger et prétraiter l'image
        NativeImageLoader loader = new NativeImageLoader(HEIGHT, WIDTH, CHANNELS);
        org.nd4j.linalg.api.ndarray.INDArray image = loader.asMatrix(new File(imagePath));
        
        // Normaliser les pixels (0-255 -> 0-1)
        image = image.div(255.0);
        
        // Prédiction
        org.nd4j.linalg.api.ndarray.INDArray output = model.output(image);
        
        // Trouver la classe avec la probabilité maximale
        int predictedClass = output.argMax(1).getInt(0);
        double confidence = output.getDouble(predictedClass) * 100;
        
        String[] classes = {"Crocodile", "Dragibus", "Fraise", "Oeuf", "Ourson", "Schtroumpf"};
        
        return new String[] {
            classes[predictedClass],
            String.format("%.2f", confidence)
        };
    }
}