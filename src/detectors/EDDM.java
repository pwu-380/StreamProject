/**
 * Created by Peter on 11/4/2017.
 * This class implements the EDDM drift detector as described in "Early Drift Detection Method"
 * (Baena-Garcia et. al, 2006)
 */

package detectors;

import core.PredictionMatrix;


public class EDDM extends AbstractChangeDetectorNew{
    private double pPrime;                          //Average distance between errors
    private double sPrime;                          //Std of distance between errors
    private double varPrime;                        //Var of distance between errors
    private double pMax;                            //Max mean encountered
    private double sMax;                            //Max std encountered
    private int nErrors;                            //Number of errors
    private int lastInterval;                       //Interval between errors

    public EDDM (PredictionMatrix predictionMatrix){
        //0.90 and 0.95 are the default values for alarm and warn as advised in BG 2006
        this(predictionMatrix, 0.90, 0.95);
    }

    public EDDM (PredictionMatrix predictionMatrix, double alarmThreshold){
        //0.90 and 0.95 are the default values for alarm and warn as advised in BG 2006
        super(predictionMatrix, alarmThreshold);
        pPrime = 0;
        sPrime = 0;
        varPrime = 0;
        nErrors = 0;
        lastInterval = 0;
    }

    public EDDM (PredictionMatrix predictionMatrix, double alarmThreshold, double warnThreshold){
        super(predictionMatrix, alarmThreshold, warnThreshold);
        pPrime = 0;
        sPrime = 0;
        varPrime = 0;
        nErrors = 0;
        lastInterval = 0;
    }

    //Incrementally updates the running variance, mean and std
    private void updateVarPS (int n) {
        double oldP = pPrime;

        //New average
        pPrime = oldP + (n - oldP)/nErrors;

        //New variance
        if (nErrors >= 2) {                     //(Can't calculate variance with less than 2 numbers)
            varPrime = (((nErrors - 2) / (nErrors - 1)) * varPrime) + ((1 / nErrors) * Math.pow((pPrime - oldP), 2));
        }

        //New standard deviation
        sPrime = Math.sqrt(varPrime);
    }

    private double calcDriftLevel () {
        return (pPrime + 2 * sPrime)/(pMax + 2 * sMax);
    }

    public int testDrift (){
        int[] lastPrediction = predictionMatrix.getLastPrediction();
        //Per parent method:
        //0 indicates drift
        //1 indicates stable
        //2 indicates warn
        int drift = 1;


        if (lastPrediction[0] != lastPrediction[1]){
            nErrors++;
            updateVarPS(lastInterval);
            lastInterval = 0;

            //The value of pMax + 2 * sMax corresponds with the point where the distribution of distances between
            // errors is maximum
            if ((pPrime + 2 * sPrime) > (pMax + 2 * sMax)) {
                pMax = pPrime;
                sMax = sPrime;
            }

            //Only considers drift after 30 errors have occurred per BG 2006
            if (nErrors >= 30){
                double driftLevel = calcDriftLevel();

                if (driftLevel < warnThreshold){
                    if (driftLevel < alarmThreshold){
                        drift = 0;
                        pMax = 0;
                        sMax = 0;
                    } else {
                        drift = 2;
                    }
                }
            }

        } else {
            lastInterval++;
        }

        return drift;
    }
}
