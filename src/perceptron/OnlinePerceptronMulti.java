package perceptron;

import java.util.Random;

public class OnlinePerceptronMulti {
	
	public static int seed = 1234;
    public static Random GenRdm = new Random(seed);//Random(seed);

    /*
     *  OneHot : 
     *      @param tag: un entier representant le numéro de la classe
     *
     *  1. on initialise un tableau à 0
     *  2. on met 1 dans la case correspondant au tag
     *
     *		@return le oneHot du tag
     */
	public static int[] OneHot(int tag) {
		int[] v = new int[ImageOnlinePerceptronMulti.classNb];
		for (int i = 0; i < v.length; i++) {
			v[i] = 0;
		}
		v[tag] = 1;
		return v;
	}
	
	/*
     *  dot : 
     *      @param w: un tableau 2D de float correspondant aux poids du perceptron
     *      @param x: un tableau de float correspondant à une image convertie
     *      @param l: la classe de l'image
     *
     *  1. on fait le produit scalaire des poids de la classe l et de l'image
     *  
     *  	@return le produit scalaire des poids de la classe l et de l'image
     *
     */
	public static double dot(float[][] w, float[] x, int l) {
		double tot = 0;
		for (int i = 1; i < x.length; i++) {
			tot += x[i]*w[i][l];
		}
		return tot + w[0][l];
	}
	
	/*
     *  InfPerceptron : 
     *      @param w: un tableau 2D de float correspondant aux poids du perceptron
     *      @param x: un tableau de float correspondant à une image convertie
     *
     *  1. on calcule l'exponentielle du produit scalaire des poids de la classe i et de l'image
     *  2. on calcule la probabilité d'appartenance à chaque classe de l'image
     *  
     *  	@return Un tableau contenant les probas d'appartenir à chaque classe
     *
     */
	public static float[] InfPerceptron(float[][] w, float[] x) {
		double[]	dProbs = new double[w[0].length];
		float[]		fProbs = new float[w[0].length];
		float		tot = 0;
		for (int i = 0; i < w[0].length; i++) {
			double z = Math.exp(dot(w, x, i)); 
			dProbs[i] = z;
			tot += z;
		}
		
		for (int i = 0; i < w[0].length; i++) {
			fProbs[i] = (float) (dProbs[i]/tot);
		}
		return fProbs;
	}

	/*
     *  InitialiseWeights : 
     *      @param DIM: la dimension de l'image
     *
     *  1. on initialise les poids du perceptron avec des valeurs aléatoires
     *  
     *  	@return Un tableau 2D de float contenant les poids du perceptron
     *
     */
	public static float[][] InitialiseWeights(int DIM) {
    	float[][]	w = new float[DIM][ImageOnlinePerceptronMulti.classNb];
    	float 		alpha = 1.0f/DIM;
    	for (int i = 0; i < w.length; i++) {
    		for (int l = 0; l < w[i].length; l++) { 
    			w[i][l] = alpha*(GenRdm.nextFloat()-0.5f);
    		}	
    	}
    	return w;
    }
	
	/*
     *  updateWeights : 
     *      @param x: un tableau de float correspondant à une image convertie
     *      @param y: un tableau de float correspondant aux probas d'appertenir à une classe de l'image
     *      @param p: un tableau d'entier correspondant au oneHot de l'image
     *      @param eta: le taux d'apprentissage
     *
     *  1. on met a jour les poids du perceptron
     *
     */
	public static void updateWeights(float[] x, float[] y, int[] p, float eta) {
		for (int l = 0; l < ImageOnlinePerceptronMulti.w[0].length; l++) {
			ImageOnlinePerceptronMulti.w[0][l] -= eta*(y[l]-p[l]);
		}
		for (int i = 1; i < ImageOnlinePerceptronMulti.w.length; i++) {
			for (int l = 0; l < ImageOnlinePerceptronMulti.w[i].length; l++) {
				ImageOnlinePerceptronMulti.w[i][l] -= x[i]*eta*(y[l]-p[l]);
			}
		}
	}
}