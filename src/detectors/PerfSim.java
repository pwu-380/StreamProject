/**
 * Created by Peter on 10/30/2017.
 */

package detectors;

//This class implements the PerfSim drift detector as described in "The PerfSim Algorithm for Concept Drift Detection
//in Imbalanced Data" (Antwi et. al, 2012)
public class PerfSim {
    private int[] previousVector;               //Stores the vector to compare to
    private int size;                           //Size of the above
    private double alarmThreshold;              //User parameter for when a concept change is considered

    //Constructor
    public PerfSim (int[][] confusion_Matrix){
        this(confusion_Matrix, 0.98);           //0.98 is the default value as advised in Antwi 2012
    }

    //Constructor
    public PerfSim (int[][] confusion_Matrix, double alarmThreshold){
        int n = confusion_Matrix.length;
        this.size = n * n;

        this.previousVector = reshape(confusion_Matrix);
        this.alarmThreshold = alarmThreshold;
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
    public boolean testDrift (int[][] confusion_Matrix){
        return testDrift(confusion_Matrix, true);               //By default updates the compare-to vector when testing
    }

    public boolean testDrift (int[][] confusion_Matrix, boolean update){
        int[] newVector = reshape(confusion_Matrix);

        if (newVector.length != size){
            throw new IllegalArgumentException("New confusion matrix does not match the previous in size");
        }

        double similarity = calcCosSim(newVector, previousVector);

        if (update) {
            previousVector = newVector;
        }

        return similarity < alarmThreshold ? true : false;
    }
}
