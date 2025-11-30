package fr.esisar.in450.ACS;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;

import java.io.File;
import java.util.*;
import java.util.Arrays;

/**
 * Classe utilitaire pour détecter la couleur dominante d'un bonbon sur une image.
 * Utilise OpenCV pour analyser les pixels et déterminer la couleur prédominante.
 * Peut contraindre la détection selon les couleurs possibles pour un type de bonbon donné.
 */
public class ColorDetector {

    // Définition des couleurs possibles pour chaque type de bonbon
    private static final Map<String, List<String>> COULEURS_PAR_TYPE = new HashMap<>();
    
    static {
        COULEURS_PAR_TYPE.put("croco", Arrays.asList("bleu", "jaune", "orange", "rouge", "vert"));
        COULEURS_PAR_TYPE.put("dragibus", Arrays.asList("bleu", "jaune", "noir", "rose", "rouge", "vert"));
        COULEURS_PAR_TYPE.put("fraise", Arrays.asList("fraise"));
        COULEURS_PAR_TYPE.put("oeuf", Arrays.asList("oeuf"));
        COULEURS_PAR_TYPE.put("ourson", Arrays.asList("blanc", "jaune", "orange", "rouge", "vert"));
        COULEURS_PAR_TYPE.put("schtroumpf", Arrays.asList("blanc", "jaune", "rouge"));
        // Ajoutez d'autres types selon votre dataset
    }

    /**
     * Détecte la couleur d'un bonbon en prenant en compte son type.
     * Calcule les probabilités pour toutes les couleurs et choisit la plus probable
     * parmi celles autorisées pour ce type de bonbon.
     * 
     * @param imageFile fichier image à analyser
     * @param typeBonbon type de bonbon détecté (ex: "dragibus", "croco")
     * @return nom de la couleur la plus probable pour ce type
     * @throws Exception si l'image ne peut pas être chargée
     */
    public static String detectColorForType(File imageFile, String typeBonbon) throws Exception {
        // Calcul des probabilités pour toutes les couleurs
        Map<String, Double> couleurProbas = calculateColorProbabilities(imageFile);
        
        // Récupération des couleurs autorisées pour ce type
        List<String> couleursAutorisees = COULEURS_PAR_TYPE.getOrDefault(
            typeBonbon.toLowerCase(), 
            Arrays.asList("rouge", "orange", "jaune", "vert", "bleu", "violet", "rose", "blanc", "noir")
        );
        
        // Sélection de la couleur avec la plus haute probabilité parmi les autorisées
        String couleurChoisie = "indetermine";
        double maxProba = -1.0;
        
        for (String couleur : couleursAutorisees) {
            double proba = couleurProbas.getOrDefault(couleur, 0.0);
            if (proba > maxProba) {
                maxProba = proba;
                couleurChoisie = couleur;
            }
        }
        
        System.out.println("  Couleur du bonbon : " + couleurChoisie);
        System.out.println("  Confiance         : " + String.format("%.2f%%", maxProba * 100));
        System.out.println();
        
        // Affichage des probabilités
        System.out.println("\n--- Probabilités de couleurs détectées ---");
        couleurProbas.entrySet().stream()
            .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
            .forEach(entry -> System.out.println(String.format("  %s : %.2f%%", entry.getKey(), entry.getValue())));
        
        return couleurChoisie;
    }

    /**
     * Calcule les probabilités pour chaque couleur possible dans l'image.
     * Analyse la région centrale de l'image et détermine les scores pour chaque couleur.
     * 
     * @param imageFile fichier image à analyser
     * @return Map associant chaque couleur à sa probabilité (0-100)
     * @throws Exception si l'image ne peut pas être chargée
     */
    public static Map<String, Double> calculateColorProbabilities(File imageFile) throws Exception {
        // Chargement de l'image
        Mat image = opencv_imgcodecs.imread(imageFile.getAbsolutePath());
        if (image.empty()) {
            throw new Exception("Impossible de charger l'image : " + imageFile.getAbsolutePath());
        }

        // Conversion de BGR vers HSV
        Mat hsvImage = new Mat();
        opencv_imgproc.cvtColor(image, hsvImage, opencv_imgproc.COLOR_BGR2HSV);

        // Extraction de la région centrale
        int height = hsvImage.rows();
        int width = hsvImage.cols();
        int centerX = width / 2;
        int centerY = height / 2;
        int roiSize = Math.min(width, height) / 2;

        Rect roi = new Rect(
            Math.max(0, centerX - roiSize / 2),
            Math.max(0, centerY - roiSize / 2),
            Math.min(roiSize, width - (centerX - roiSize / 2)),
            Math.min(roiSize, height - (centerY - roiSize / 2))
        );
        Mat centerRegion = new Mat(hsvImage, roi);

        // Calcul de la couleur moyenne
        Scalar meanColor = opencv_core.mean(centerRegion);
        double hue = meanColor.get(0);
        double saturation = meanColor.get(1);
        double value = meanColor.get(2);

        // Calcul des scores pour chaque couleur
        Map<String, Double> scores = new HashMap<>();
        scores.put("rouge", scoreForRed(hue, saturation, value));
        scores.put("orange", scoreForOrange(hue, saturation, value));
        scores.put("jaune", scoreForYellow(hue, saturation, value));
        scores.put("vert", scoreForGreen(hue, saturation, value));
        scores.put("bleu", scoreForBlue(hue, saturation, value));
        scores.put("violet", scoreForViolet(hue, saturation, value));
        scores.put("rose", scoreForPink(hue, saturation, value));
        scores.put("blanc", scoreForWhite(hue, saturation, value));
        scores.put("noir", scoreForBlack(hue, saturation, value));

        // Libération mémoire
        image.release();
        hsvImage.release();
        centerRegion.release();

        return scores;
    }

    // Fonctions de score pour chaque couleur (basées sur la distance HSV)
    
    private static double scoreForRed(double h, double s, double v) {
        if (s < 40) return 0.0;
        double score = 0.0;
        if (h < 10 || h >= 170) {
            score = 100.0 - Math.min(Math.abs(h), Math.abs(180 - h)) * 5;
        }
        return Math.max(0, score * (s / 255.0));
    }

    private static double scoreForOrange(double h, double s, double v) {
        if (s < 40) return 0.0;
        if (h >= 10 && h < 25) {
            return (100.0 - Math.abs(h - 17) * 6) * (s / 255.0);
        }
        return 0.0;
    }

    private static double scoreForYellow(double h, double s, double v) {
        if (s < 40) return 0.0;
        if (h >= 25 && h < 40) {
            return (100.0 - Math.abs(h - 32) * 6) * (s / 255.0);
        }
        return 0.0;
    }

    private static double scoreForGreen(double h, double s, double v) {
        if (s < 40) return 0.0;
        if (h >= 40 && h < 80) {
            return (100.0 - Math.abs(h - 60) * 2) * (s / 255.0);
        }
        return 0.0;
    }

    private static double scoreForBlue(double h, double s, double v) {
        if (s < 40) return 0.0;
        if (h >= 80 && h < 130) {
            return (100.0 - Math.abs(h - 105) * 2) * (s / 255.0);
        }
        return 0.0;
    }

    private static double scoreForViolet(double h, double s, double v) {
        if (s < 40) return 0.0;
        if (h >= 130 && h < 150) {
            return (100.0 - Math.abs(h - 140) * 5) * (s / 255.0);
        }
        return 0.0;
    }

    private static double scoreForPink(double h, double s, double v) {
        if (h >= 150 && h < 170) {
            return (100.0 - Math.abs(h - 160) * 5) * (s / 255.0) * 0.8;
        }
        return 0.0;
    }

    private static double scoreForWhite(double h, double s, double v) {
        if (s < 40 && v > 200) {
            return 100.0 * (1.0 - s / 255.0) * (v / 255.0);
        }
        return 0.0;
    }

    private static double scoreForBlack(double h, double s, double v) {
        if (v < 50) {
            return 100.0 * (1.0 - v / 255.0);
        }
        return 0.0;
    }

    /**
     * Ajoute ou modifie les couleurs possibles pour un type de bonbon.
     * 
     * @param typeBonbon nom du type de bonbon
     * @param couleurs liste des couleurs possibles
     */
    public static void setCouleursForType(String typeBonbon, List<String> couleurs) {
        COULEURS_PAR_TYPE.put(typeBonbon.toLowerCase(), couleurs);
    }
}