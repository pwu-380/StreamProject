/**
 * Created by Peter on 10/30/2017.
 * This class implements the PerfSim drift detector as described in "The PerfSim Algorithm for Concept Drift Detection
 * in Imbalanced Data" (Antwi et. al, 2012)
 */

package detectors;

import core.PredictionMatrix;

public class PerfSim2 extends AbstractChangeDetectorNew{
    private int[] previousVector;               //Stores the vector to compare to
    private int size;                           //Size of the above

    //Constructor
    public PerfSim2(PredictionMatrix predictionMatrix){
        this(predictionMatrix, 0.98);           //0.98 is the default value as advised in Antwi 2012
    }

    //Constructor
    public PerfSim2(PredictionMatrix predictionMatrix, double alarmThreshold){
        super(predictionMatrix, alarmThreshold);
        int[][] confusionMatrix = predictionMatrix.getMatrix();

        int n = confusionMatrix.length;
        this.size = n * n;
        this.previousVector = reshape(confusionMatrix);
    }

    //Reshapes a square matrix into a vector
    private int[] reshape (int[][] matrix){
        int[] vector = new int[size];

        int count = 0;
        for (int i = 0; i < matrix.length; i++){
            for (int j = 0; j < matrix.length; j++){
                vector[count] = matrix[i][j];
                count++;
            }
        }

        return vector;
    }

    //Calculates the cosine similarity between two vectors
    private double calcCosSim (int[] a, int[] b){
        if (a.length != b.length){
            throw new IllegalArgumentException("Vector lengths do not match");
        }

        double nume = 0;
        double lenA = 0;
        double lenB = 0;

        for (int i = 0; i < a.length; i++){
            nume += a[i] * b[i];
            lenA += a[i] * a[i];
            lenB += b[i] * b[i];
        }

        System.out.println(nume/(Math.sqrt(lenA) * Math.sqrt(lenB)));
        return nume/(Math.sqrt(lenA) * Math.sqrt(lenB));
    }

    //Resets the alarm threshold to a different value
    public void setAlarmThreshold (double alarmThreshold){
        if (alarmThreshold > 1 || alarmThreshold < 0){
            throw new IllegalArgumentException("Alarm threshold must be set between 0 and 1");
        }
        this.alarmThreshold = alarmThreshold;
    }

    //Tests if concept drift has occured from new confusion matrix
    public int testDrift (){
        return testDrift(true);               //By default updates the compare-to vector when testing
    }

    public int testDrift (boolean update){
        int[] newVector = reshape(predictionMatrix.getMatrix());

        if (newVector.length != size){
            throw new IllegalArgumentException("New confusion matrix does not match the previous in size");
        }

        double similarity = calcCosSim(newVector, previousVector);

        if (update) {
            previousVector = newVector;
        }

        //Per parent method:
        //0 indicates drift
        //1 indicates stable
        //2 indicates warn
        return similarity < alarmThreshold ? 0 : 1;
    }
}
