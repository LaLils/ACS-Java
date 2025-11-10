package fr.esisar.in450.ACS;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;

public class CandyClassifierGUI extends JFrame {
    
    private JButton trainButton;
    private JButton predictButton;
    private JButton selectImageButton;
    private JLabel imageLabel;
    private JTextArea logArea;
    private JLabel resultTypeLabel;
    private JLabel resultColorLabel;
    private JLabel resultConfidenceLabel;
    private JProgressBar progressBar;
    
    private File selectedImage;
    private CandyTypeClassifier typeClassifier;
    private ColorDetector colorDetector;
    private Predictor predictor;
    
    private static final String MODEL_PATH = "models/candy_type_model.zip";
    private static final String DATASET_PATH = "src/main/train";
    
    public CandyClassifierGUI() {
        setTitle("Automatic Candy Selector - ACS");
        setSize(900, 700);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        
        // Initialiser les composants ML
        typeClassifier = new CandyTypeClassifier();
        colorDetector = new ColorDetector();
        predictor = new Predictor();
        
        // Créer l'interface
        initComponents();
        	
        setVisible(true);
    }
    
    private void initComponents() {
        // Panel principal
        JPanel mainPanel = new JPanel(new BorderLayout(10, 10));
        mainPanel.setBorder(new EmptyBorder(10, 10, 10, 10));
        
        // === PANEL DU HAUT : Titre et actions ===
        JPanel topPanel = new JPanel(new BorderLayout());
        
        JLabel titleLabel = new JLabel("🍬 Automatic Candy Selector", SwingConstants.CENTER);
        titleLabel.setFont(new Font("Arial", Font.BOLD, 24));
        titleLabel.setBorder(new EmptyBorder(10, 0, 10, 0));
        topPanel.add(titleLabel, BorderLayout.NORTH);
        
        // Boutons d'action
        JPanel buttonPanel = new JPanel(new FlowLayout(FlowLayout.CENTER, 20, 10));
        
        trainButton = new JButton("🎓 Entraîner le modèle");
        trainButton.setFont(new Font("Arial", Font.BOLD, 14));
        trainButton.setPreferredSize(new Dimension(200, 40));
        trainButton.addActionListener(e -> trainModel());
        
        predictButton = new JButton("🔍 Prédire un bonbon");
        predictButton.setFont(new Font("Arial", Font.BOLD, 14));
        predictButton.setPreferredSize(new Dimension(200, 40));
        predictButton.setEnabled(false); // Désactivé tant que pas de modèle
        predictButton.addActionListener(e -> selectAndPredict());
        
        buttonPanel.add(trainButton);
        buttonPanel.add(predictButton);
        
        topPanel.add(buttonPanel, BorderLayout.CENTER);
        
        // Barre de progression
        progressBar = new JProgressBar();
        progressBar.setStringPainted(true);
        progressBar.setVisible(false);
        topPanel.add(progressBar, BorderLayout.SOUTH);
        
        mainPanel.add(topPanel, BorderLayout.NORTH);
        
        // === PANEL CENTRAL : Image et résultats ===
        JPanel centerPanel = new JPanel(new GridLayout(1, 2, 10, 0));
        
        // Panel gauche : Image
        JPanel imagePanel = new JPanel(new BorderLayout());
        imagePanel.setBorder(new TitledBorder("Image du bonbon"));
        
        imageLabel = new JLabel("Aucune image sélectionnée", SwingConstants.CENTER);
        imageLabel.setPreferredSize(new Dimension(400, 400));
        imageLabel.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        imageLabel.setBackground(Color.WHITE);
        imageLabel.setOpaque(true);
        
        selectImageButton = new JButton("📁 Sélectionner une image");
        selectImageButton.addActionListener(e -> selectImage());
        
        imagePanel.add(imageLabel, BorderLayout.CENTER);
        imagePanel.add(selectImageButton, BorderLayout.SOUTH);
        
        // Panel droit : Résultats
        JPanel resultsPanel = new JPanel(new GridLayout(4, 1, 0, 10));
        resultsPanel.setBorder(new TitledBorder("Résultats de la prédiction"));
        
        resultTypeLabel = createResultLabel("Type :", "?");
        resultColorLabel = createResultLabel("Couleur :", "?");
        resultConfidenceLabel = createResultLabel("Confiance :", "?");
        
        JPanel typePanel = new JPanel(new BorderLayout());
        typePanel.add(resultTypeLabel, BorderLayout.CENTER);
        
        JPanel colorPanel = new JPanel(new BorderLayout());
        colorPanel.add(resultColorLabel, BorderLayout.CENTER);
        
        JPanel confidencePanel = new JPanel(new BorderLayout());
        confidencePanel.add(resultConfidenceLabel, BorderLayout.CENTER);
        
        resultsPanel.add(new JLabel(""));
        resultsPanel.add(typePanel);
        resultsPanel.add(colorPanel);
        resultsPanel.add(confidencePanel);
        
        centerPanel.add(imagePanel);
        centerPanel.add(resultsPanel);
        
        mainPanel.add(centerPanel, BorderLayout.CENTER);
        
        // === PANEL DU BAS : Logs ===
        JPanel logPanel = new JPanel(new BorderLayout());
        logPanel.setBorder(new TitledBorder("Journal d'activité"));
        
        logArea = new JTextArea(8, 50);
        logArea.setEditable(false);
        logArea.setFont(new Font("Monospaced", Font.PLAIN, 11));
        JScrollPane scrollPane = new JScrollPane(logArea);
        
        logPanel.add(scrollPane, BorderLayout.CENTER);
        
        mainPanel.add(logPanel, BorderLayout.SOUTH);
        
        add(mainPanel);
        
        // Vérifier si un modèle existe déjà
        checkExistingModel();
    }
    
    private JLabel createResultLabel(String label, String value) {
        JLabel lbl = new JLabel(label + " " + value);
        lbl.setFont(new Font("Arial", Font.BOLD, 18));
        lbl.setBorder(new EmptyBorder(10, 20, 10, 20));
        return lbl;
    }
    
    private void log(String message) {
        SwingUtilities.invokeLater(() -> {
            logArea.append(message + "\n");
            logArea.setCaretPosition(logArea.getDocument().getLength());
        });
    }
    
    private void checkExistingModel() {
        File modelFile = new File(MODEL_PATH);
        if (modelFile.exists()) {
            log("✓ Modèle existant détecté : " + MODEL_PATH);
            predictButton.setEnabled(true);
        } else {
            log("⚠ Aucun modèle trouvé. Veuillez d'abord entraîner le modèle.");
        }
    }
    
    private void trainModel() {
        // Désactiver les boutons pendant l'entraînement
        trainButton.setEnabled(false);
        predictButton.setEnabled(false);
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        
        log("\n=== DÉBUT DE L'ENTRAÎNEMENT ===");
        
        // Entraînement dans un thread séparé pour ne pas bloquer l'interface
        SwingWorker<Void, String> worker = new SwingWorker<>() {
            @Override
            protected Void doInBackground() throws Exception {
                try {
                    publish("Vérification du dataset...");
                    File datasetDir = new File(DATASET_PATH);
                    if (!datasetDir.exists()) {
                        throw new Exception("Dataset introuvable : " + DATASET_PATH);
                    }
                    
                    publish("Entraînement du classificateur de type...");
                    typeClassifier.train(DATASET_PATH);
                    
                    publish("Sauvegarde du modèle...");
                    typeClassifier.saveModel(MODEL_PATH);
                    
                    publish("✓ Entraînement terminé avec succès !");
                    
                } catch (Exception e) {
                    publish("✗ Erreur : " + e.getMessage());
                    e.printStackTrace();
                }
                return null;
            }
            
            @Override
            protected void process(java.util.List<String> chunks) {
                for (String message : chunks) {
                    log(message);
                }
            }
            
            @Override
            protected void done() {
                trainButton.setEnabled(true);
                predictButton.setEnabled(true);
                progressBar.setVisible(false);
                progressBar.setIndeterminate(false);
                log("=== FIN DE L'ENTRAÎNEMENT ===\n");
            }
        };
        
        worker.execute();
    }
    
    private void selectImage() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                if (f.isDirectory()) return true;
                String name = f.getName().toLowerCase();
                return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
            }
            
            @Override
            public String getDescription() {
                return "Images (*.jpg, *.jpeg, *.png)";
            }
        });
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedImage = fileChooser.getSelectedFile();
            displayImage(selectedImage);
            log("Image sélectionnée : " + selectedImage.getName());
        }
    }
    
    private void displayImage(File imageFile) {
        try {
            BufferedImage img = ImageIO.read(imageFile);
            if (img != null) {
                // Redimensionner l'image pour l'affichage
                Image scaledImg = img.getScaledInstance(400, 400, Image.SCALE_SMOOTH);
                imageLabel.setIcon(new ImageIcon(scaledImg));
                imageLabel.setText("");
            }
        } catch (Exception e) {
            log("Erreur lors du chargement de l'image : " + e.getMessage());
        }
    }
    
    private void selectAndPredict() {
        selectImage();
        if (selectedImage != null) {
            predictImage();
        }
    }
    
    private void predictImage() {
        if (selectedImage == null) {
            JOptionPane.showMessageDialog(this, 
                "Veuillez d'abord sélectionner une image !", 
                "Aucune image", 
                JOptionPane.WARNING_MESSAGE);
            return;
        }
        
        predictButton.setEnabled(false);
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        
        log("\n=== PRÉDICTION EN COURS ===");
        
        SwingWorker<String[], String> worker = new SwingWorker<>() {
            @Override
            protected String[] doInBackground() throws Exception {
                try {
                    publish("Chargement du modèle...");
                    predictor.loadModel(MODEL_PATH);
                    
                    publish("Analyse de l'image...");
                    String[] result = predictor.predict(selectedImage.getAbsolutePath());
                    
                    publish("✓ Prédiction terminée !");
                    return result;
                    
                } catch (Exception e) {
                    publish("✗ Erreur : " + e.getMessage());
                    e.printStackTrace();
                    return null;
                }
            }
            
            @Override
            protected void process(java.util.List<String> chunks) {
                for (String message : chunks) {
                    log(message);
                }
            }
            
            @Override
            protected void done() {
                try {
                    String[] result = get();
                    if (result != null) {
                        // Mettre à jour les résultats
                        resultTypeLabel.setText("Type : " + result[0]);
                        resultTypeLabel.setForeground(new Color(0, 128, 0));
                        
                        resultColorLabel.setText("Couleur : " + result[1]);
                        resultColorLabel.setForeground(new Color(0, 0, 255));
                        
                        resultConfidenceLabel.setText("Confiance : " + result[2] + "%");
                        double confidence = Double.parseDouble(result[2]);
                        if (confidence > 80) {
                            resultConfidenceLabel.setForeground(new Color(0, 128, 0));
                        } else if (confidence > 60) {
                            resultConfidenceLabel.setForeground(new Color(255, 140, 0));
                        } else {
                            resultConfidenceLabel.setForeground(Color.RED);
                        }
                        
                        log("Type : " + result[0]);
                        log("Couleur : " + result[1]);
                        log("Confiance : " + result[2] + "%");
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
                
                predictButton.setEnabled(true);
                progressBar.setVisible(false);
                progressBar.setIndeterminate(false);
                log("=== FIN DE LA PRÉDICTION ===\n");
            }
        };
        
        worker.execute();
    }
    
    public static void main(String[] args) {
        // Utiliser le look and feel du système
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        // Lancer l'interface
        SwingUtilities.invokeLater(() -> new CandyClassifierGUI());
    }
}