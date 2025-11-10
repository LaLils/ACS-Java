package fr.esisar.in450.ACS;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class ColorDetector {
    
    // Définition des couleurs de référence (RGB)
    private static final Map<String, int[]> COLOR_REFERENCES = new HashMap<>();
    
    static {
        COLOR_REFERENCES.put("Rouge", new int[]{255, 0, 0});
        COLOR_REFERENCES.put("Vert", new int[]{0, 255, 0});
        COLOR_REFERENCES.put("Bleu", new int[]{0, 0, 255});
        COLOR_REFERENCES.put("Jaune", new int[]{255, 255, 0});
        COLOR_REFERENCES.put("Rose", new int[]{255, 192, 203});
        COLOR_REFERENCES.put("Noir", new int[]{0, 0, 0});
    }
    
    /**
     * Détecte la couleur dominante dans une image
     */
    public String detectDominantColor(String imagePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(imagePath));
        
        if (image == null) {
            throw new Exception("Impossible de charger l'image: " + imagePath);
        }
        
        // Calculer la couleur moyenne de l'image
        long sumR = 0, sumG = 0, sumB = 0;
        int pixelCount = 0;
        
        int width = image.getWidth();
        int height = image.getHeight();
        
        // Parcourir tous les pixels (en ignorant les bords blancs potentiels)
        int margin = Math.min(width, height) / 10; // Marge de 10%
        
        for (int y = margin; y < height - margin; y++) {
            for (int x = margin; x < width - margin; x++) {
                int rgb = image.getRGB(x, y);
                
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                // Ignorer les pixels trop blancs (fond)
                if (r > 240 && g > 240 && b > 240) {
                    continue;
                }
                
                sumR += r;
                sumG += g;
                sumB += b;
                pixelCount++;
            }
        }
        
        if (pixelCount == 0) {
            return "Indéterminé";
        }
        
        // Couleur moyenne
        int avgR = (int) (sumR / pixelCount);
        int avgG = (int) (sumG / pixelCount);
        int avgB = (int) (sumB / pixelCount);
        
        System.out.println("Couleur moyenne RGB: (" + avgR + ", " + avgG + ", " + avgB + ")");
        
        // Trouver la couleur la plus proche
        String closestColor = findClosestColor(avgR, avgG, avgB);
        
        return closestColor;
    }
    
    /**
     * Trouve la couleur de référence la plus proche
     */
    private String findClosestColor(int r, int g, int b) {
        String closestColor = "Indéterminé";
        double minDistance = Double.MAX_VALUE;
        
        for (Map.Entry<String, int[]> entry : COLOR_REFERENCES.entrySet()) {
            int[] refColor = entry.getValue();
            
            // Distance euclidienne dans l'espace RGB
            double distance = Math.sqrt(
                Math.pow(r - refColor[0], 2) +
                Math.pow(g - refColor[1], 2) +
                Math.pow(b - refColor[2], 2)
            );
            
            if (distance < minDistance) {
                minDistance = distance;
                closestColor = entry.getKey();
            }
        }
        
        return closestColor;
    }
    
    /**
     * Détection avancée avec histogramme de couleurs
     */
    public String detectColorWithHistogram(String imagePath) throws Exception {
        BufferedImage image = ImageIO.read(new File(imagePath));
        
        if (image == null) {
            throw new Exception("Impossible de charger l'image: " + imagePath);
        }
        
        // Compter les occurrences de chaque couleur proche
        Map<String, Integer> colorCounts = new HashMap<>();
        for (String color : COLOR_REFERENCES.keySet()) {
            colorCounts.put(color, 0);
        }
        
        int width = image.getWidth();
        int height = image.getHeight();
        int margin = Math.min(width, height) / 10;
        
        for (int y = margin; y < height - margin; y++) {
            for (int x = margin; x < width - margin; x++) {
                int rgb = image.getRGB(x, y);
                
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                
                // Ignorer les pixels blancs
                if (r > 240 && g > 240 && b > 240) {
                    continue;
                }
                
                // Trouver la couleur la plus proche pour ce pixel
                String closestColor = findClosestColor(r, g, b);
                colorCounts.put(closestColor, colorCounts.get(closestColor) + 1);
            }
        }
        
        // Trouver la couleur la plus fréquente
        String dominantColor = "Indéterminé";
        int maxCount = 0;
        
        for (Map.Entry<String, Integer> entry : colorCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                dominantColor = entry.getKey();
            }
        }
        
        return dominantColor;
    }
}