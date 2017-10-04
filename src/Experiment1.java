/**
 * Created by Peter on 9/22/2017.
 * Testing buffering/unbuffering prediction performance on superminority/supermajority classes.
 * Tried setting 0 to superminority (1/5 normal proportion) and 9 to supermajority (5x normal proportion) on LED
 * data set.
 * Real life applications, may see imbalance ratios of 1:1000 or 1:5000 (e.g. fraud detection) [Krawczyk 2016]
 */

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayesMultinomial;
import moa.classifiers.trees.HoeffdingTree;
import generators.NewLEDGenerator;

import java.io.*;
import java.util.HashMap;

public class Experiment1 {

    /*User Defined Parameters-----------------------------------------*/
    private static final String RESULTS_FILE = "results/experiment1.csv";
    //Number of examples used to pre-train
    public static int PROBE_INSTANCES = 1;
    //If we want to balance class examples in pre-training
    public static boolean BALANCE_PROBE_SET = false;
    //Number of instances to produce after pre-training
    public static int STREAM_SIZE = 10000;
    //Toggle use of buffers
    public static boolean USE_BUFFERS = true;
    //Define number of elements saved in each buffer
    private static final int BUFFER_SIZE = 10;
    //Sliding window size for prequential window;
    private static final int PREQUENTIAL_WINDOW_SIZE = 50;

    //First 7 attributes correspond to LED lights, next 17 are noise attributes and last is class
    //Named 'att1', 'att2', 'att3'..., 'class'
    private static final int N_NOISE_ATTR = 17;
    private static final int N_PCT = 10;
    private static NewLEDGenerator STREAM = new NewLEDGenerator(N_NOISE_ATTR, N_PCT);

    //Define a mapping of each class to a buffer
    //The key is what is returned when we check Instance.classValue
    private static final HashMap CLASSES =
            new HashMap<Double, Integer>() {{
                put(0.0, 0);
                put(1.0, 1);
                put(2.0, 2);
                put(3.0, 3);
                put(4.0, 4);
                put(5.0, 5);
                put(6.0, 6);
                put(7.0, 7);
                put(8.0, 8);
                put(9.0, 9);
            }};

    //For NewLEDGenerator this is the distribution of classes generated
    private static final double[] CDIST = {5,5,25,5,1,5,5,5,5,5};

    //Classifier to user
    private static Classifier clf = new NaiveBayesMultinomial();

    //This is appended to the top of the results file (just to keep track of test parameters)
    public static String ANNOTATION_STRING =
            String.format("<HEADER> GENERATOR: NewLEDGenerator(%d %d)", N_NOISE_ATTR, N_PCT) +
            String.format("\tCLASS BALANCE: 0=%.1f 1=%.1f 2=%.1f 3=%.1f 4=%.1f 5=%.1f 6=%.1f 7=%.1f 8=%.1f 9=%.1f",
            CDIST[0], CDIST[1], CDIST[2], CDIST[3], CDIST[4], CDIST[5], CDIST[6], CDIST[7], CDIST[8], CDIST[9]) +
            String.format("\tCLASSIFIER: Multinomial Bayes") +
            String.format("\tBUFFERED?: %b", USE_BUFFERS) +
            String.format("\tPROBE SIZE: %d", PROBE_INSTANCES) +
            String.format("\tBUFFER_SIZE: %d", BUFFER_SIZE) +
            String.format("\tPREQUENTIAL WINDOW SIZE: %d", PREQUENTIAL_WINDOW_SIZE);

    //Define how often to calculate metrics (accuracy, precision, etc.)
    public static int CALC_METRICS_INTERVAL = 20;

    /*Program-----------------------------------------*/

    //Adds new instance to buffers, if there exists at least one element in each buffer, then train
    private static boolean handleTrainingCandidate (Classifier clf, InstanceBuffer instanceBuffer, Instance newInstance){
        double classValue = newInstance.classValue();
        boolean modelUpdated = false;

        instanceBuffer.addInstance(newInstance);

        if (instanceBuffer.existsSampleInAllClasses()){
            Instance[] samples = instanceBuffer.removeHead();
            for (int i = 0; i < samples.length; i++){
                clf.trainOnInstance(samples[i]);
            }
            modelUpdated = true;
        }

        return modelUpdated;
    }

    /*Main-----------------------------------------*/
    public static void main(String[] args) {
        //Initialize the stream
        STREAM.setClass_proportions(CDIST);
        STREAM.prepareForUse();

        //Get the stream attributes
        InstancesHeader header = STREAM.getHeader();
        int numclasses = header.numClasses();

        //Initialize the classifier
        clf.setModelContext(header);
        clf.prepareForUse();

        //Initialize buffers
        InstanceBuffer instanceBuffer = new InstanceBuffer(header, CLASSES, BUFFER_SIZE);

        //Trains classifier on a probe set
        int num_instances = 0;
        boolean trained = false;
        Instance current_inst;
        while (((current_inst = STREAM.nextInstance().getData()) != null) && (num_instances < PROBE_INSTANCES)) {
            //Uses buffering mechanism to balance probe set if desired
            if (BALANCE_PROBE_SET) {
                //Adds new instance to buffers, if there exists at least one element in each buffer, then train
                trained = handleTrainingCandidate(clf, instanceBuffer, current_inst);
                if (trained) {
                    num_instances += numclasses;
                    trained = false;
                }
             //Otherwise just train on whatever instance comes out
            } else {
                clf.trainOnInstance(current_inst);
                num_instances++;
            }
        }

        instanceBuffer.emptyBuffers();

        //Begins prequential test than train
        PredictionMatrix predictionMatrix = new PredictionMatrix(clf, CLASSES, PREQUENTIAL_WINDOW_SIZE);

        try{
            PrintWriter writer = new PrintWriter(new BufferedWriter(new FileWriter(RESULTS_FILE)));
            writer.print(ANNOTATION_STRING);
            writer.printf("\nStream Instance,Accuracy,PPV-0,TPR-0,PPV-1,TPR-1,PPV-2,TPR-2," +
            "PPV-3,TPR-3,PPV-4,TPR-4,PPV-5,TPR-5,PPV-6,TPR-6,PPV-7,TPR-7,PPV-8,TPR-8,PPV-9,TPR-9");

            num_instances = 0;
            while (((current_inst = STREAM.nextInstance().getData()) != null) && (num_instances < STREAM_SIZE)){
                //Predict and score
                predictionMatrix.predictUpdate(current_inst);
                num_instances++;

                if (USE_BUFFERS) {
                    //Adds new instance to buffers, if there exists at least one element in each buffer, then train
                    handleTrainingCandidate(clf, instanceBuffer, current_inst);
                } else {
                    //Otherwise just train on whatever instance comes out
                    clf.trainOnInstance(current_inst);
                }

                //At regular intervals, check how our classifier is performing
                if (num_instances % CALC_METRICS_INTERVAL == 0) {
                    writer.printf("\n%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                            num_instances, predictionMatrix.calcAccuracy(),
                            predictionMatrix.calcPrecision(0),predictionMatrix.calcRecall(0),
                            predictionMatrix.calcPrecision(1),predictionMatrix.calcRecall(1),
                            predictionMatrix.calcPrecision(2),predictionMatrix.calcRecall(2),
                            predictionMatrix.calcPrecision(3),predictionMatrix.calcRecall(3),
                            predictionMatrix.calcPrecision(4),predictionMatrix.calcRecall(4),
                            predictionMatrix.calcPrecision(5),predictionMatrix.calcRecall(5),
                            predictionMatrix.calcPrecision(6),predictionMatrix.calcRecall(6),
                            predictionMatrix.calcPrecision(7),predictionMatrix.calcRecall(7),
                            predictionMatrix.calcPrecision(8),predictionMatrix.calcRecall(8),
                            predictionMatrix.calcPrecision(9),predictionMatrix.calcRecall(9));
                }
            }
            writer.close();
            predictionMatrix.printMatrix();

        } catch (IOException e) {
            System.out.println("File couldn't be opened");
        }
    }
}
