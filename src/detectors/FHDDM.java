/**
 * Created by Peter on 11/18/2017.
 */

package detectors;

import java.util.LinkedList;
import java.util.Queue;
import core.PredictionMatrix;

//This class implements the FHDDM drift detector as described in "Fast Hoeffding Drift Detection Method for
// Evolving Data Streams" (Pesaranghader and Viktor, 2016)

public class FHDDM extends AbstractChangeDetectorNew{

    private int windowSize;             //Maximum sliding window width
    private Queue<Integer> window;      //Test results for all samples in the window

    private double p1;                  //Probability of correct prediction in window
    private double p1max;               //Max probability of correct prediction over time
    private double windowSum;


    public FHDDM(PredictionMatrix predictionMatrix){
        this(predictionMatrix, 0.0000001, 200);                 //Values used in P&V 2016
    }

    public FHDDM(PredictionMatrix predictionMatrix, double delta, int windowSize){
        super(predictionMatrix, Math.sqrt(Math.log(1/delta) / (2*windowSize)));
        this.windowSize = windowSize;
        resetFHDDM();
    }

    //Resets class variables
    private void resetFHDDM(){
        window = new LinkedList<>();
        windowSum = 0;
        p1max = 0;
    }

    //windowSize setter
    public void setWindowSize(int size) {
        if (size > 1) {
            windowSize = size;
        } else {
            throw new IllegalArgumentException("FHDDM: Window size must be > 1");
        }
    }

    public void setDelta(double delta){
        if (delta > 0){
            super.alarmThreshold = Math.sqrt(Math.log(1/delta) / (2*windowSize));
        } else {
            throw new IllegalArgumentException("FHDDM: delta must be > 0");
        }
    }

    public int testDrift(){
        int[] lastPrediction = predictionMatrix.getLastPrediction();
        //Per parent method:
        //0 indicates drift
        //1 indicates stable
        //2 indicates warn
        int drift = 1;
        int match =  1;         //0 if prediction does not match label, 1 otherwise

        if (lastPrediction[0] != lastPrediction[1]){
            match = 0;
            window.add(0);
        } else {
            window.add(1);
        }

        if (window.size() <= windowSize){
            //Incremental average update
            windowSum += match;
        } else {
            Integer removed = window.remove();
            if (removed != match){
                if (match == 0){
                    windowSum--;
                    p1 = windowSum/windowSize;

                    //Detects drift by Hoeffding bound
                    if (p1max - p1 >= alarmThreshold){
                        drift = 0;
                        resetFHDDM();
                    }
                } else {
                    windowSum++;
                    p1 = windowSum/windowSize;

                    if (p1 > p1max){
                        p1max = p1;
                    }
                }
            }
        }

        return drift;
    }

}
