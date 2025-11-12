package fr.esisar.in450.ACS;

import java.util.List;

/**
 * Classe utilitaire pour gérer les labels (classes de sortie) et leur parsing.
 * Cette classe gère la correspondance entre :  les indices utilisés par le modèle et les noms textuels
 * Elle permet également de parser ces labels pour séparer le type de la couleure
 */
public class Utils {

    private static List<String> labels = null;	// Liste des labels chargés depuis le dataset (dans l'ordre alphabétique)


    /**
     * Initialise la liste des labels utilisés par le modèle.
     *
     * @param datasetLabels liste des labels du dataset (non vide)
     * @throws IllegalArgumentException si la liste est nulle ou vide
     */
    public static void initLabels(List<String> datasetLabels) {
        if (datasetLabels == null || datasetLabels.isEmpty()) {
            throw new IllegalArgumentException("Les labels ne peuvent pas être null ou vides");
        }
        labels = datasetLabels;
        System.out.println("\nLabels initialisés : " + labels.size() + " classes");
        System.out.println("   Liste : " + labels);
    }

    /**
     * Retourne le nom de la classe correspondant à un index donné.
     *
     * @param index index numérique de la classe
     * @return nom de la classe correspondante, ou message d’erreur si invalide
     */
    public static String getClassName(int index) {
        if (labels == null) {
            return "ERREUR : Labels non initialisés ! Appelez Utils.initLabels() d'abord.";
        }
        if (index >= 0 && index < labels.size()) {
            return labels.get(index);
        }
        return "Index invalide : " + index;
    }

    /**
     * Analyse un label au format "type-couleur" et sépare ses deux composants.
     *
     * @param label étiquette au format "type-couleur"
     * @return tableau [type, couleur], ou ["unknown", "unknown"] si invalide
     */
    public static String[] parseLabel(String label) {
        if (label == null || !label.contains("-")) {
            return new String[]{"unknown", "unknown"};
        }
        String[] parts = label.split("-", 2);
        return parts;
    }

    /**
     * Retourne uniquement le type d’un label.
     *
     * @param label étiquette complète au format "type-couleur"
     * @return type du bonbon, ou "unknown" si non conforme
     */
    public static String getType(String label) {
        return parseLabel(label)[0];
    }

    /**
     * Retourne uniquement la couleur d’un label.
     *
     * @param label étiquette complète au format "type-couleur"
     * @return couleur du bonbon, ou "unknown" si non conforme
     */
    public static String getCouleur(String label) {
        return parseLabel(label)[1];
    }

    /**
     * Vérifie si la liste de labels a été initialisée.
     *
     * @return true si les labels sont chargés, false sinon
     */
    public static boolean areLabelsInitialized() {
        return labels != null && !labels.isEmpty();
    }

    /**
     * Retourne le nombre total de classes disponibles.
     *
     * @return nombre de labels si initialisés, sinon 0
     */
    public static int getNumClasses() {
        return labels != null ? labels.size() : 0;
    }
}
