/**
 * Created by Peter on 9/22/2017.
 */

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class InstanceBuffer {

    private int bufferSize;                 //Maximum elements stored in each buffer
    private HashMap classes;                //Mapping of class values to indices
    private int numBuffers;                 //Number of buffers to maintain (= to # of classes)
    private Instances[] buffers;            //Array of buffers

    //Constructor with default buffer size
    public InstanceBuffer(InstancesHeader streamHeader, HashMap classes){
        this(streamHeader,classes, 20);
    }

    //Constructor for user set buffer size
    public InstanceBuffer(InstancesHeader streamHeader, HashMap classes, int bufferSize){
        if (bufferSize < 2){
            throw new IllegalArgumentException("Invalid buffer parameters passed");
        }

        //Initialize class variables
        this.bufferSize = bufferSize;
        this.classes = classes;
        numBuffers = streamHeader.numClasses();
        buffers = new Instances[numBuffers];

        //Initiates each buffer
        List<Attribute> streamAttributes = new ArrayList<>();
        for (int i = 0; i < streamHeader.numAttributes(); i++){
            streamAttributes.add(streamHeader.attribute(i));
        }

        for (int i = 0; i < numBuffers; i++){
            buffers[i] = new Instances("buffer" + i, streamAttributes, bufferSize);
        }
    }

    //Removes all elements from each buffer
    public void emptyBuffers(){
        for (int i = 0; i < buffers.length; i++){
            buffers[i].delete();
        }
    }

    //Checks if each class buffer has at least one example
    public boolean existsSampleInAllClasses(){
        boolean sufficientExamples = true;

        for (int i = 0; i < buffers.length; i++){
            if (buffers[i].numInstances() == 0){
                sufficientExamples = false;
                break;
            }
        }
        return sufficientExamples;
    }

    //Adds a new instance to the appropriate buffer
    public void addInstance(Instance instance){
        int buffer_ind = (int) classes.get(instance.classValue());

        //If the buffer is full, delete the oldest instance in the buffer
        if (buffers[buffer_ind].numInstances() == bufferSize){
            buffers[buffer_ind].delete(0);
        }

        buffers[buffer_ind].add(instance);
    }

    //Returns the first element from each buffer and deletes them from the buffer
    public Instance[] removeHead(){
        Instance[] head = new Instance[numBuffers];
        for (int i = 0; i < numBuffers; i++){
            try {
                head[i] = buffers[i].instance(0);
                buffers[i].delete(0);
            } catch (IndexOutOfBoundsException e){
                //If a buffer is empty, that instance in the array will end up Null
                System.out.println("Warning: Buffer " + i + " was empty [InstanceBuffer]");
            }
        }
        return head;
    }

    //Returns all the buffers
    public Instances[] getBuffers (){
        return buffers;
    }

    public String toString (){
        return "I need to implement this";
    }

}
