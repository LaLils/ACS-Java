# Matrice de sélection des technologies – Projet **ACS (Automatic Candy Selector)**

## 1. Analyse par critères

### Critères primaires

| **Critères** | **Deeplearning4j** | **Weka** | **Smile** | **TensorFlow Java** | **OpenCV (JavaCV)** |
| - | - | - | - | - | - |
| Support natif des images     | 5 | 2 | 2 | 4 | 5 |
| CNN (Réseaux convolutifs)    | 5 | 0 | 1 | 5 | 0 |
| Courbe d'apprentissage        | 3 | 5 | 3 | 2 | 4 |
| Prétraitement images          | 4 | 2 | 2 | 3 | 5 |
| Documentation & exemples      | 4 | 5 | 3 | 3 | 4 |
| Performance sur petit dataset | 4 | 3 | 3 | 4 | N/A |
| **Total (sur 30)**            | **25** | **17** | **14** | **21** | **18 / 25** |

---

### Critères secondaires

| **Critères** | **Deeplearning4j** | **Weka** | **Smile** | **TensorFlow Java** | **OpenCV (JavaCV)** |
| - | - | - | - | - | - |
| Facilité d'intégration Maven  | 4 | 5 | 4 | 3 | 3 |
| Extraction de features        | 5 | 2 | 3 | 5 | 5 |
| Sauvegarde / chargement modèle| 5 | 4 | 4 | 4 | N/A |
| Visualisation résultats       | 4 | 5 | 2 | 3 | 3 |
| Gestion multi-labels          | 5 | 3 | 3 | 5 | N/A |
| Détection de couleur          | 2 | 2 | 2 | 2 | 5 |
| **Total (sur 30)**            | **25** | **21** | **18** | **22** | **16 / 20** |

---

## 2. Résumé – Score total pondéré

**Pondération :**  
- Critères primaires ×2  
- Critères secondaires ×1  

| **Bibliothèque** | **Primaires (×2)** | **Secondaires (×1)** | **Score total (max 90)** |
| - | - | - | - |
| **Deeplearning4j** | 25×2 = 50 | +25 | **75** |
| **TensorFlow Java** | 21×2 = 42 | +22 | **64** |
| **Weka** | 17×2 = 34 | +21 | **55** |
| **OpenCV (JavaCV)** | 18×2 = 36 | +16 | **52** |
| **Smile** | 14×2 = 28 | +18 | **46** |

---

## 3. Analyse par bibliothèque

### **Deeplearning4j**
- **Forces :** Excellent support des CNN/RNN, bonne performance, intégration Java native, compatibilité Maven, export/sauvegarde de modèles.  
- **Faiblesses :** Moins d’exemples récents que TensorFlow, configuration initiale plus longue.  
- **Pertinence :** Idéal pour la **classification d’images** (type de bonbon).

---

### **Weka**
- **Forces :** Interface graphique simple, documentation claire, apprentissage rapide.  
- **Faiblesses :** Peu adaptée aux images et aux réseaux neuronaux.  
- **Pertinence :** Adaptée à la **classification simple** (ex. couleur dominante).

---

### **Smile**
- **Forces :** Framework ML léger et rapide (SVM, RandomForest).  
- **Faiblesses :** Pas de support CNN ni traitement d’image natif.  
- **Pertinence :** Option secondaire pour des modèles basés sur des features extraites.

---

### **TensorFlow Java**
- **Forces :** Très performant, compatible avec CNN via modèles importés (.pb / .tflite).  
- **Faiblesses :** Documentation Java incomplète, dépend souvent d’un entraînement Python.  
- **Pertinence :** Idéal si le modèle est **pré-entraîné en Python** puis importé.

---

### **OpenCV (JavaCV)**
- **Forces :** Référence pour le **prétraitement d’images**, la **segmentation** et la **détection de couleur**.  
- **Faiblesses :** Non conçu pour le ML complet.  
- **Pertinence :** Parfait pour la **préparation visuelle avant classification**.

---

## 4. Conclusion

| **Usage** | **Librairie retenue** | **Justification** |
| - | - | - |
| Prétraitement & extraction de couleurs | **OpenCV (JavaCV)** | Gestion avancée d’images, couleurs et contours |
| Classification du type de bonbon | **Deeplearning4j** | Support CNN natif, bonne performance, intégration Java |
| Classification de la couleur dominante (fallback) | **Weka ou Smile** | Simple à implémenter, utile pour du ML classique |

---

## ✅ Choix final recommandé

### **Pipeline hybride**
> **OpenCV → extraction & prétraitement d’image → Deeplearning4j (ou Weka pour la couleur)**

---

## 5. Schéma fonctionnel
```
+-----------------------------------------+
| Input : Image bonbon (JPG/PNG) |
+-----------------------------------------+
                     │
                     v
+-----------------------------------------+
| OpenCV (JavaCV)                         |
| - Prétraitement (redimension, blur…)    |
| - Détection de couleur dominante        |
+-----------------------------------------+
                     │
                     v
+-----------------------------------------+
| Deeplearning4j                          |
| - Classification type de bonbon (CNN)   |
| - Estimation de la confiance (%)        |
+-----------------------------------------+
                     │
                     v
+-----------------------------------------+
| Output                                  |
| • Type : Ourson                         |
| • Couleur : Rouge                       |
| • Confiance : 94.2%                     |
+-----------------------------------------+
```