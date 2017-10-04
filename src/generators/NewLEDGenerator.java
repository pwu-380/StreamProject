/**
 * Created by Peter on 9/17/2017.
 */

package generators;

import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.FastVector;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

import java.util.Arrays;


//I want to be able to set LEDGenerator parameters more easily
public class NewLEDGenerator extends moa.streams.generators.LEDGenerator{
    private static int num_irrelevant_attributes = 17;
    private static int noise_percentage = 10;
    private static double[] class_gen_thresholds = new double[11];
    private static boolean proportions_changed = false;

    public NewLEDGenerator(){
    }

    public NewLEDGenerator(int num_irrelevant_attributes, int noise_percentage){
        if (num_irrelevant_attributes >= 0){
            this.num_irrelevant_attributes = num_irrelevant_attributes;
        } else {
            throw new IllegalArgumentException("Number of irrelevant attributes needs to be >= 0");
        }

        if (noise_percentage >= 0 && noise_percentage < 100){
            this.noise_percentage = noise_percentage;
        } else {
            throw new IllegalArgumentException("Noise percentage has to be an int between 0 and 100");
        }
    }

    public void setClass_proportions (double[] class_proportions){

        double sum = Arrays.stream(class_proportions).sum();

        //Parameter checking
        if (class_proportions.length != class_gen_thresholds.length - 1){
            throw new IllegalArgumentException("Proportions have not been assigned to all classes");
        } else if (sum == 0){
            throw new IllegalArgumentException("Invalid proportion entered");
        } else {
            for (int i = 0; i < class_proportions.length; i++){
                if (class_proportions[i] < 0){
                    throw new IllegalArgumentException("Invalid proportion entered");
                }
            }
        }

        //Normalize proportions
        for (int i = 0; i < class_proportions.length; i++){
            class_proportions[i] = class_proportions[i]/sum;
        }

        //Generating from a uniform distribution, these form boundaries for what class is generated
        class_gen_thresholds[0] = 0;
        class_gen_thresholds[class_proportions.length] = 1;
        for (int i = 1; i < class_proportions.length; i++){
            class_gen_thresholds[i] = class_gen_thresholds[i-1] + class_proportions[i-1];
        }

        proportions_changed = true;
    }

    @Override
    protected void prepareForUseImpl (TaskMonitor monitor, ObjectRepository repository){
        FastVector attributes = new FastVector();
        FastVector binaryLabels = new FastVector();
        binaryLabels.addElement("0");
        binaryLabels.addElement("1");
        int numAtts = 7 + num_irrelevant_attributes;

        for(int classLabels = 0; classLabels < numAtts; ++classLabels) {
            attributes.addElement(new Attribute("att" + (classLabels + 1), binaryLabels));
        }

        FastVector var8 = new FastVector();

        for(int i = 0; i < 10; ++i) {
            var8.addElement(Integer.toString(i));
        }

        attributes.addElement(new Attribute("class", var8));
        this.streamHeader = new InstancesHeader(new Instances(this.getCLICreationString(InstanceStream.class),
                attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.restart();
    }

    @Override
    public InstanceExample nextInstance(){
        InstancesHeader header = this.getHeader();
        DenseInstance inst = new DenseInstance((double)header.numAttributes());
        inst.setDataset(header);

        int selected = 0;
        if (proportions_changed){
            double n = this.instanceRandom.nextDouble();
            for (int i = 1; i < class_gen_thresholds.length; i++){
                if (n >= class_gen_thresholds[i-1] && n < class_gen_thresholds[i]) {
                    selected = i - 1;
                    break;
                }
            }

        } else {
            selected = this.instanceRandom.nextInt(10);
        }

        int i;
        for(i = 0; i < 7; ++i) {
            if(1 + this.instanceRandom.nextInt(100) <= noise_percentage) {
                inst.setValue(i, originalInstances[selected][i] == 0?1.0D:0.0D);
            } else {
                inst.setValue(i, (double)originalInstances[selected][i]);
            }
        }

        for(i = 0; i < num_irrelevant_attributes; ++i) {
            inst.setValue(i + 7, (double)this.instanceRandom.nextInt(2));
        }

        inst.setClassValue((double)selected);
        return new InstanceExample(inst);
    }

}
