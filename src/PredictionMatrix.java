/**
 * Created by Peter on 9/21/2017.
 * This class maintains a windowed confusion matrix (actual class v. predicted class) incrementally for a set of
 * test samples
 */

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.core.Utils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

public class PredictionMatrix {
    private Classifier model;           //Trained model
    private int numClasses;             //Number of different classes
    private HashMap classes;            //Mapping of class values to indices

    private boolean windowed = false;   //Without a sliding window it just records all test results
    private int windowLength = 0;       //Width of sliding window

    private int[] lastPrediction;
    private Queue<Integer[]> window;    //Test results for all samples in the window
    private int matrix[][];             //"Confusion matrix"

    /*These methods pertain to the construction/maintenance of the matrix----------------------*/
    //Constructor for non-windowed matrix
    public PredictionMatrix(Classifier clf, HashMap classes){
        this(clf, classes, 0);
    }

    //Constructor for windowed matrix
    public PredictionMatrix(Classifier clf, HashMap classes, int windowLength){
        this.model = clf;
        numClasses = classes.size();
        this.classes = classes;

        if (windowLength > 0) {
            this.windowLength = windowLength;
            windowed = true;
            window = new LinkedList<>();
        }

        matrix = new int[numClasses][numClasses];
    }

    //Tests a single instance and records results
    public boolean predictUpdate (Instance instance){
        int actualClass = (int) classes.get(instance.classValue());
        int predictedClass = Utils.maxIndex(model.getVotesForInstance(instance));
        Integer[] pair = {actualClass, predictedClass};

        //If windowed, whenever a new prediction is made, the oldest is removed from the matrix
        if (windowed){
            window.add(pair);
            if (window.size() > windowLength){
                Integer[] removed = window.remove();
                matrix[removed[0]][removed[1]]--;
            }
        }

        //Class updates
        matrix[actualClass][predictedClass]++;
        lastPrediction = new int[]{actualClass, predictedClass};

        //Returns whether the prediction was correct
        return actualClass == predictedClass;
    }

    public void resetMatrix(){
        matrix = new int[numClasses][numClasses];
    }

    public int[] getLastPrediction () {
        return lastPrediction;

    }

    /*These methods pertain to calculating various metrics from the matrix----------------------*/
    //Calculates the precision with respect to a specific class
    //-1 indicates an indeterminate case
    public double calcPrecision (int classInd){
        double truePositive = matrix[classInd][classInd];            //Positive and correctly predicted
        double precision;
        double predictedPositive = 0;                                //All predicted positives e.g. TP + FP

        for (int row = 0; row < numClasses; row++) {
            predictedPositive += matrix[row][classInd];
        }

        if (predictedPositive == 0){
            precision = -1;                                         //Identifies indeterminate case
        } else {
            precision = truePositive/predictedPositive;             //Otherwise calculates as normal
        }
        return precision;
    }

    //Calculates the recall with respect to a specific class
    //-1 indicates an indeterminate case
    public double calcRecall (int classInd){
        double truePositive = matrix[classInd][classInd];           //Positive and correctly predicted
        double recall;
        double classGenerated = Arrays.stream(matrix[classInd]).sum();   //All actual positives e.g. TP + FN

        if (classGenerated == 0){
            recall = -1;                                            //Identifies indeterminate case
        } else {
            recall = truePositive/classGenerated;                   //Otherwise calculates as normal
        }
        return recall;
    }

    //Calculates the fscore with respect to a specific class
    //-1 indicates an indeterminate case
    public double calcFScore (int classInd){
        double precision = calcPrecision(classInd);
        double recall = calcRecall(classInd);
        double fscore = 0;

        if (precision == -1 || recall == -1){                       //Fscore is indeterminate if either recall or
            fscore = -1;                                            //precision is indeterminate
        }else if (precision + recall != 0){
            fscore = (2 * precision * recall)/(precision + recall);
        }

        return fscore;
    }

    //Calculates the accuracy
    public double calcAccuracy (){
        double correct = 0;
        double total = 0;

        //If windowed, we know the total is just the # of window elements
        if (!windowed) {
            for (int i = 0; i < numClasses; i++) {
                correct += matrix[i][i];
                total += Arrays.stream(matrix[i]).sum();
            }
        } else {
            for (int i = 0; i < numClasses; i++) {
                correct += matrix[i][i];
            }
            total = window.size();
        }
        return correct/total;
    }

    /*These methods pertain to information retrieval from the matrix----------------------*/
    //Returns confusion matrix
    public int[][] getMatrix(){
        return matrix;
    }

    public void printMatrix(){
        System.out.print("\n     ");
        for (int i = 0; i < numClasses; i++){
            System.out.printf(" %4d ", i);
        }
        System.out.print("\n");
        for (int i = 0; i <= numClasses; i++){
            System.out.print("======");
        }
        for (int row = 0; row < numClasses; row++){
            System.out.printf("\n%2d | ", row);
            for (int col = 0; col < numClasses; col++){
                if (matrix[row][col] == 0) {
                    System.out.print("    . ");
                } else {
                    System.out.printf(" %4d ", matrix[row][col]);
                }
            }
            System.out.print("\n");
            for (int i = 0; i <= numClasses; i++){
                System.out.print(" - - -");
            }
        }
        System.out.print("\n");
    }

    public String toString (){
        return  "I need to implenment this";
    }
}
