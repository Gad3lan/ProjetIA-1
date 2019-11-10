package perceptron;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Arrays;

import mnisttools.MnistReader;

public class ImageOnlinePerceptronMulti {
    
	public static String	path="/Users/Alan/Documents/JavaLibs/gzip/";
    public static String	labelDB=path+"emnist-byclass-train-labels.idx1-ubyte";
    public static String	imageDB=path+"emnist-byclass-train-images.idx3-ubyte";
    public static int		DIM = 785;
    public static int		classNb;
    public final static int	EPOCHMAX=40;
    public static float[][]	w;
    
    /*
     *  BinariserImage : 
     *      @param image: une image int binarisée à deux dimensions
     *      @param seuil: un entier
     *
     *  1. on convertit l'image en deux dimension dx X dy, en une image en deux dimension, 
     *  si la valeur de pixel est supérieure au seuil, on met 1, sinon 0
     *
     *		@return L'image binarisée
     */
    public static int[][] BinariserImage(int[][] image, int seuil) {
    	int[][]	binaryImage = new int[image.length][image[0].length];
    	for (int i = 0; i < image.length; i++) {
    		for (int j = 0; j < image[i].length; j++) {
    			if (image[i][j] >= seuil) {
    				binaryImage[i][j] = 1;
    			} else {
    				binaryImage[i][j] = 0;
    			}
    		}
    	}
    	return binaryImage;
    }

    /*
    *  ConvertImage : 
    *      @param image: une image binarisée à deux dimensions
    *
    *  1. on convertit l'image en deux dimension dx X dy, en un tableau unidimensionnel de tail dx.dy
    *  2. on rajoute un élément en première position du tableau qui sera à 1
    *  La taille finale renvoyée sera dx.dy + 1
    *
    *		@return L'image convertie
    */
    public static float[] ConvertImage(int[][] image) {
        float[]	convertedImage = new float[DIM];
        convertedImage[0] = 1;
        for (int i = 0; i < image.length; i++) {
        	for (int j = 0; j < image[i].length; j++) {
        		convertedImage[i*image[i].length + j + 1] = image[i][j];
        	}
        }
        return convertedImage;
    }
    
    /*
     *  checkErr : 
     *      @param label: un entier correspondant à la classe le l'image vérifiée
     *      @param probs: un tableau de float contenant les proba d'appartenance à chaque classe
     *
     *  1. on parcourt le tableau de proba
     *  2. on retourne l'indice de la plus grosse proba
     *
     *		@return 0 si il n'y a pas d'erreur, 1 sinon
     */
    public static int checkErr(int label, float[] probs) {
    	float	max = 0;
    	int		maxIdx = 0;
    	for (int i = 0; i < probs.length; i++) {
    		if (probs[i] > max) {
    			maxIdx = i;
    			max = probs[i];
    		}
    	}
    	if (maxIdx == label)
    		return 0;
    	return 1;
    }
    
    /*
     *  epoch : 
     *      @param wTemp: un tableau 2D de float contenant les poids du perceptron
     *      @param x: un tableau 2D de float contenant toutes les images de l'ensemble d'apprentissage
     *      @param label: un tableau d'entier contenant les labels des images
     *      @param eta: un float correspondant au taux d'apprentissage
     *
     *  1. on parcourt toutes les image
     *  2. on calcule la probabilité d'appartenance à chaque classe de l'image
     *  3. on met à jour les poids du perceptron
     *  4. on augmente le nombre d'erreur de 1 si le perceptron c'est trompé
     *
     *		@return Le nombre d'erreur avec l'ensemble d'apprentissage
     */
    public static int epoch(float[][] wTemp, float x[][], int[] label, float eta) {
    	float[]	y = new float[label.length];
    	int		nbErr = 0;
    	int		nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		y = OnlinePerceptronMulti.InfPerceptron(w, x[i]);
    		OnlinePerceptronMulti.updateWeights(x[i], y, OnlinePerceptronMulti.OneHot(label[i]), eta);    		
    		nbErr += checkErr(label[i], y);
    	}
    	return nbErr;
    }
    
    /*
     *  validation : 
     *      @param x: un tableau 2D de float contenant toutes les images de l'ensemble d'apprentissage
     *      @param label: un tableau d'entier contenant les labels des images
     *
     *  1. on parcourt toutes les image
     *  2. on calcule la probabilité d'appartenance à chaque classe de l'image
     *  4. on augmente le nombre d'erreur de 1 si le perceptron c'est trompé
     *
     *		@return Le nombre d'erreur avec l'ensemble de validation
     */
    public static int validation(float x[][], int[] label) {
    	float[]	y = new float[label.length];
    	int		nbErr = 0;
    	int		nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		y = OnlinePerceptronMulti.InfPerceptron(w, x[i]);
    		nbErr += checkErr(label[i], y);
    	}
    	return nbErr;
    }
    
    /*
     *  createTrainSet : 
     *      @param db: le MnistReader
     *      @param Na: un entier correspondant au nombre d'image que l'on doit selectionner
     *      @param trainData: un tableau 2D de float pour contenir les images converties
     *      @param trainLabel: un tableau d'entier pour contenir le label de chaque image
     *
     *  1. on parcourt toutes les images tant que les Na images ne sont pas séléctionnées
     *  2. on ajoute l'image et son labels aux tableaux correspondants si elle est comprise entre les
     *  classe 10 et 10+classNb (le nombre de classes que l'on utilise, 12 ou 26 en fonction des cas)
     *
     *		@return La valeur à laquelle l'algo c'est arrêté
     */
    public static int createTrainSet(MnistReader db, int Na, float[][] trainData, int[] trainLabel) {
    	int	count = 1;
    	int	i;
    	for (i = 1; count < Na; i++) {
    		int label = db.getLabel(i);
    		if (label >= 10 && label < 10 + classNb) {
    			trainData[count] = ConvertImage(BinariserImage(db.getImage(i), 100));
    			trainLabel[count] = label-10;
    			count++;
    		}
    	}
    	return i;
    }
    
    /*
     *  createValidSet : 
     *      @param db: le MnistReader
     *      @param start: le point de depart de la boucle (pour evite de prendre les mêmes valeurs que l'ensemble
     *      @param d'apprentissage
     *      @param Nv: un entier correspondant au nombre d'image que l'on doit selectionner
     *      @param trainData: un tableau 2D de float pour contenir les images converties
     *      @param trainLabel: un tableau d'entier pour contenir le label de chaque image
     *
     *  1. on parcourt toutes les images après start tant que les Nv images ne sont pas séléctionnées
     *  2. on ajoute l'image et son labels aux tableaux correspondants si elle est comprise entre les
     *  classe 10 et 10+classNb (le nombre de classes que l'on utilise, 12 ou 26 en fonction des cas)
     *
     */
    public static void createValidSet(MnistReader db, int start, int Nv, float[][] validData, int[] validLabel) {
    	int	count = 1;
    	int	i;
    	for (i = start; count < Nv; i++) {
    		int label = db.getLabel(i);
    		if (label >= 10 && label < 10 + classNb) {
    			validData[count] = ConvertImage(BinariserImage(db.getImage(i), 100));
    			validLabel[count] = label-10;
    			count++;
    		}
    	}
    }
    
    /*
     *  predictedClass : 
     *      @param y: un tableau de float contenant les probas d'appartenance à chaque classe
     *
     *  1. on parcourt le tableau de proba
     *  2. on retourne l'indice de la plus grosse proba
     *
     *		@return L'indice de la plus grosse proba
     */
    public static int predictedClass(float[] y) {
    	float	max = 0;
    	int		maxIdx = 0;
    	for (int i = 0; i < y.length; i++) {
    		if (y[i] > max) {
    			maxIdx = i;
    			max = y[i];
    		}
    	}
    	return maxIdx;
    }
    
    /*
     *  bestDataSetSize : 
     *      @param db: le MnistReader
     *
     *  1. on initialise un FileWriter pour écrire les résultats dans un fichier
     *  2. on entraine le perceptron avec la valeur de Na
     *  3. on actualise le nombre minimum d'erreurs s'il est inférieur au précédents
     *  4. on ecrit le nombre d'erreur de cette boucle dans le fichier
     *  5. on augmente la valeur de Na
     *
     *		@return Le nombre d'image necessaire pour faire le moins d'erreurs possible
     */
    public static int bestDataSetSize(MnistReader db) throws IOException {
    	classNb = 12;
    	File file = new File("bestDataSetSize.txt");
		file.createNewFile();
		FileWriter fw = new FileWriter(file);
		
    	int	bestSize = 0;
    	int	nbMinError = 10000;
    	int	Na = 1000;
		int	Nv = 1000;
    	for (int n = 0; n <= 18; n++) {
    		int nbError = 0;
    		System.out.print("Na size: " + Na + ", learning... ");
    		float[][] trainData = new float[Na][DIM];
    		int[] trainLabel = new int[Na];
    		float[][] validData = new float[Nv][DIM];
    		int[] validLabel = new int[Nv];
    		int next = createTrainSet(db, Na, trainData, trainLabel);
    		createValidSet(db, next, Nv, validData, validLabel);
    		w = OnlinePerceptronMulti.InitialiseWeights(DIM);
    		for (int i = 0; i < EPOCHMAX; i++) {
    			epoch(w, trainData, trainLabel, 0.001f);
    			nbError = validation(validData, validLabel);
    		}
    		if (nbMinError > nbError) {
    			nbMinError = nbError;
    			bestSize = Na;
    		}
			fw.write("" + Na + " " + nbError + "\n");
			System.out.println("Done");
			Na += 500;
    	}
    	fw.close();
    	return bestSize;
    }
    
    /*
     *  learn : 
     *      @param trainData: un tableau 2D de float contenant les images converties
     *      @param trainLabel: un tableau d'entier contenant le label de chaque image
     *      eta: la valeur du taux d'apprentissage
     *
     *  1. on entraine le perceptron pendant un certain nombre d'époques
     *
     */
    public static void learn(float[][] trainData, int[] trainLabel, float eta) {
    	classNb = 12;
    	
    	for (int i = 0; i < EPOCHMAX; i++) {
			epoch(w, trainData, trainLabel, eta);
    	}
    }
    
    /*
     *  confusionMatrix : 
     *      @param x: un tableau 2D de float contenant les images converties
     *      @param label: un tableau d'entier contenant le label de chaque image
     *
     *  1. on calcule les probas d'appartenance à chaque classe de l'image
     *  2. on ajoute 1 à la case de colonne correspondant à la plus grosse proba et de la ligne
     *  correspondant au label
     *  
     *  	@return La matrice de confusion
     *
     */
    public static int[][] confusionMatrix(float x[][], int[] label) {
    	classNb = 12;
    	
    	float[]	y = new float[label.length];
    	int[][]	matrix = new int[classNb][classNb];
    	int		nbPoints = y.length;
    	for (int i = 0; i < nbPoints; i++) {
    		y = OnlinePerceptronMulti.InfPerceptron(w, x[i]);
    		matrix[predictedClass(y)][label[i]] += 1;
    	}
    	return matrix;
    }
    
    /*
     *  fullAlphabet : 
     *      @param db: le MnistReader
     *      @param Na: le nombre d'images que l'on utilise pour l'ensemble d'apprentissage
     *
     *  1. on initilise un FileWriter pour ecrire les resultats dans un fichier
     *  2. on entraine le perceptron
     *  3. on note le nombre d'erreur dans le fichier à chaque epoque
     *
     */
    public static void fullAlphabet(MnistReader db, int Na) throws IOException {
    	classNb = 26;
    	File file = new File("fullAlphabetLearning.txt");
		file.createNewFile();
		FileWriter fw = new FileWriter(file);
		
		int			Nv = 1000;
		float[][]	trainData = new float[Na][DIM];
		int[]		trainLabel = new int[Na];
		float[][]	validData = new float[Nv][DIM];
		int[]		validLabel = new int[Nv];
		int			next = createTrainSet(db, Na, trainData, trainLabel);
		int[]		nbErrT = new int[EPOCHMAX];
		int[]		nbErrV = new int[EPOCHMAX];
		createValidSet(db, next, Nv, validData, validLabel);
		w = OnlinePerceptronMulti.InitialiseWeights(DIM);
		for (int i = 0; i < EPOCHMAX; i++) {
			nbErrT[i] = epoch(w, trainData, trainLabel, 0.001f);
			nbErrV[i] = validation(validData, validLabel);
			fw.write("" + i + " " + nbErrT[i]+ " " + nbErrV[i] + "\n");
		}
		fw.close();
    }
    
    public static void main(String[] args) throws IOException {
		MnistReader db = new MnistReader(labelDB, imageDB);
		// Question 1 : recherche de la meilleure taille pour l'ensemble d'apprentissage
		System.out.println("Exercise 1:");
		int Na = bestDataSetSize(db);
		System.out.println("Best data set size: " + Na);
		
		// Question 3 : creation de la matrice de confusion
		System.out.print("\nExercise 3: learning...");
		int Nv = 1000;
		float[][] trainData = new float[Na][DIM];
		int[] trainLabel = new int[Na];
		float[][] validData = new float[Nv][DIM];
		int[] validLabel = new int[Nv];
		int next = createTrainSet(db, Na, trainData, trainLabel);
		createValidSet(db, next, Nv, validData, validLabel);
		w = OnlinePerceptronMulti.InitialiseWeights(DIM);
		learn(trainData, trainLabel, 0.001f);
		System.out.println("Done");
		int[][] confMatrix = confusionMatrix(validData, validLabel);
		System.out.print(" ");
		for (int i = 0; i < classNb; i++) {
			System.out.print("   " + (char)('A'+i));
		}
		System.out.println("\n");
		for (int i = 0; i < classNb; i++) {
			System.out.print((char)('A'+i) + "");
			for (int j = 0; j < confMatrix[i].length; j++) {
				System.out.printf("%4d", confMatrix[i][j]);
			}
			System.out.println();
		}
		
		//Question 7 : apprentissage avec l'alphabet entier
		System.out.print("\nExercise 7: learning...");
		fullAlphabet(db, Na);
		System.out.println("Done");
		System.out.println("See 'fullAlphabetLearning.txt' for more information.");
    }
}

