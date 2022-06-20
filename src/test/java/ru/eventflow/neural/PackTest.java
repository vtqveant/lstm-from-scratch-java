package ru.eventflow.neural;

import org.junit.Test;
import ru.eventflow.neural.graph.Pack;
import ru.eventflow.neural.graph.Placeholder;

import java.util.Arrays;

import static org.junit.Assert.assertEquals;

public class PackTest {

    @Test
    public void testPack() {
        Placeholder v1 = new Placeholder(new int[]{1, 3, 1});
        Batch v1_val = Batch.ones(new int[]{1, 3, 1}).times(1.0);
        v1_val.print("v1");
        v1.setValue(v1_val);

        Placeholder v2 = new Placeholder(new int[]{1, 3, 1});
        Batch v2_val = Batch.ones(new int[]{1, 3, 1}).times(2.0);
        v2_val.print("v2");
        v2.setValue(v2_val);

        Pack pack = new Pack(new int[]{1, 3, 2}, Arrays.asList(v1, v2));
        Batch pack_val = pack.getValue();
        assertEquals(1.0, pack_val.get(new int[]{0, 0, 0}), 0.001);
        assertEquals(2.0, pack_val.get(new int[]{0, 0, 1}), 0.001);

        pack_val.print("packed vectors are columns");
    }

}
