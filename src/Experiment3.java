/**
 * Created by Peter on 9/22/2017.
 * Testing buffering/unbuffering prediction performance on superminority/supermajority classes.
 * Tried setting 0 to superminority (1/5 normal proportion) and 9 to supermajority (5x normal proportion) on LED
 * data set.
 * Real life applications, may see imbalance ratios of 1:1000 or 1:5000 (e.g. fraud detection) [Krawczyk 2016]
 */

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import generators.NewLEDGenerator;
import generators.NewSTAGGERGenerator;
import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayesMultinomial;
import moa.classifiers.trees.HoeffdingTree;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;

public class Experiment3 {

    /*User Defined Parameters-----------------------------------------*/
    private static final String RESULTS_FILE = "results/experiment3.csv";
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

    //Attributes 0)size 1)color 2)shape 4)class
    //Target Concept 1 : size = small AND color = red
    //Target Concept 2 : color = green OR shape = circular
    //Target Concept 3 : size = medium OR size = large
    //Only one of the target concepts is active at any time
    private static NewSTAGGERGenerator STREAM = new NewSTAGGERGenerator();

    //Define a mapping of each class to a buffer
    //The key is what is returned when we check Instance.classValue
    private static final HashMap CLASSES =
            new HashMap<Double, Integer>() {{
                put(0.0, 0);
                put(1.0, 1);
            }};

    //For NewSTAGGERGenerator this is the concept for positive instances as described above
    private static final int CONCEPT = 3;
    //Whether the generator produces equal numbers of positive and negative classes
    private static final boolean BALANCED = true;

    //Classifier to user
    private static Classifier clf = new HoeffdingTree();

    //This is appended to the top of the results file (just to keep track of test parameters)
    public static String ANNOTATION_STRING =
            String.format("<HEADER> GENERATOR: NewSTAGGERGenerator") +
            String.format("\tCONCEPT: %d ", CONCEPT) +
            String.format("\tBALANCE CLASSES?: %b ", BALANCED) +
            String.format("\tCLASSIFIER: Hoeffding Tree") +
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
        STREAM.prepareForUse();
        STREAM.setClassBalance(BALANCED);
        STREAM.setConcept(CONCEPT);

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
            writer.printf("\nStream Instance,Accuracy,PPV-Negative,TPR-Negative,PPV-Positive,TPR-Positive,");

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
                    writer.printf("\n%d,%f,%f,%f,%f,%f",
                            num_instances, predictionMatrix.calcAccuracy(),
                            predictionMatrix.calcPrecision(0),predictionMatrix.calcRecall(0),
                            predictionMatrix.calcPrecision(1),predictionMatrix.calcRecall(1));
                }
            }
            writer.close();
            predictionMatrix.printMatrix();

        } catch (IOException e) {
            System.out.println("File couldn't be opened");
        }
    }
}
