package generators;

import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.core.InstanceExample;

/**
 * Created by Peter on 9/26/2017.
 */
public class NewSTAGGERGenerator extends moa.streams.generators.STAGGERGenerator {
    //True if (1) size = small & color = red, (2) color = green | shape = circle, (3) size != small
    private int conceptid = 1;
    private boolean balanceClasses = false;

    public void setConcept (int concept){
        if (concept > 0 && concept <= 3){
            conceptid = concept;
        } else {
            throw new IllegalArgumentException("Invalid concept identifier passed");
        }
    }

    public void setClassBalance (boolean balance){
        balanceClasses = balance;
    }


    @Override
    public InstanceExample nextInstance() {
        int size = 0;
        int color = 0;
        int shape = 0;
        int group = 0;
        boolean desiredClassFound = false;

        while(true) {
            while(!desiredClassFound) {
                size = this.instanceRandom.nextInt(3);
                color = this.instanceRandom.nextInt(3);
                shape = this.instanceRandom.nextInt(3);
                group = classificationFunctions[conceptid - 1].determineClass(size, color, shape);
                if(!balanceClasses) {
                    desiredClassFound = true;
                } else if(this.nextClassShouldBeZero && group == 0 || !this.nextClassShouldBeZero && group == 1) {
                    desiredClassFound = true;
                    this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                }
            }

            InstancesHeader header = this.getHeader();
            DenseInstance inst = new DenseInstance((double)header.numAttributes());
            inst.setValue(0, (double)size);
            inst.setValue(1, (double)color);
            inst.setValue(2, (double)shape);
            inst.setDataset(header);
            inst.setClassValue((double)group);
            return new InstanceExample(inst);
        }
    }


}
