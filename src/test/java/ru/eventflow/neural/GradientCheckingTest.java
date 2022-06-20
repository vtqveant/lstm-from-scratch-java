package ru.eventflow.neural;

import org.junit.Test;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.ReLU;
import ru.eventflow.neural.graph.Variable;

public class GradientCheckingTest {

    @Test
    public void testTransposeGradientsWithSymmetricDifferences() {
        Variable x = new Variable(new int[]{1, 2, 1});
        x.setValue(Batch.eye(new int[]{1, 2, 1}));

        Node y = new ReLU(new int[]{1, 2, 1}, x);

        x.getValue().print("x");
        y.getValue().print("y");
        x.getDualValue().print("dy/dx");

        double epsilon = 0.0001;
        Variable x_perturbed_plus = new Variable(new int[]{1, 2, 1});
        x_perturbed_plus.setValue(x.getValue().copy());
        x_perturbed_plus.getValue().set(new int[]{0, 0, 0}, x_perturbed_plus.getValue().get(new int[]{0, 0, 0}) + epsilon);
        Node y_perturbed_plus = new ReLU(new int[]{1, 2, 1}, x_perturbed_plus);
        Batch ypp = y_perturbed_plus.getValue();
        ypp.print("y+");

        Variable x_perturbed_minus = new Variable(new int[]{1, 2, 1});
        x_perturbed_minus.setValue(x.getValue().copy());
        x_perturbed_minus.getValue().set(new int[]{0, 0, 0}, x_perturbed_minus.getValue().get(new int[]{0, 0, 0}) - epsilon);
        Node y_perturbed_minus = new ReLU(new int[]{1, 2, 1}, x_perturbed_minus);
        Batch ypm = y_perturbed_minus.getValue();
        ypm.print("y-");

        Batch gradApprox = ypp.plus(ypm.times(-1d)).times(1d / 2d / epsilon);
        gradApprox.print("grad approx");
    }

}
