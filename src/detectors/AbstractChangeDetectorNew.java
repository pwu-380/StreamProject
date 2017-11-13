/**
 * Created by Peter on 11/12/2017.
 */

//Change detector contract
//The original MOA class was way too bulky
public abstract class AbstractChangeDetectorNew {
    protected PredictionMatrix predictionMatrix;
    protected double warnThreshold;
    protected double alarmThreshold;

    public AbstractChangeDetectorNew(PredictionMatrix predictionMatrix, double alarmThreshold){
        this.predictionMatrix = predictionMatrix;
        this.alarmThreshold = alarmThreshold;
    }

    public AbstractChangeDetectorNew(PredictionMatrix predictionMatrix, double alarmThreshold, double warnThreshold){
        this.predictionMatrix = predictionMatrix;
        this.alarmThreshold = alarmThreshold;
        this.warnThreshold = warnThreshold;
    }

    //Please use 0 to indicate drift
    //1 to indicate stable
    //2 to indicate warn
    public abstract int testDrift();

}
