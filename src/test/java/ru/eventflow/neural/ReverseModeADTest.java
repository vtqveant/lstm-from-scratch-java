package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.optimization.SimpleOptimization;

import java.util.ArrayList;
import java.util.List;

public class ReverseModeADTest {

    @Test
    public void testReLU() {
        int[] shape = new int[]{2, 3, 1}; // a batch of two vectors of length three

        Batch in = new Batch(shape);
        in.put(0, new Matrix(new double[]{-1, 0, 3}, 3));
        in.put(1, new Matrix(new double[]{0, -5, 2}, 3));

        Variable x = new Variable(shape);
        ReLU y = new ReLU(shape, x);

        x.setValue(in);
        Batch relu = y.getValue();
        Batch dydx = x.getDualValue();

        relu.print("y = ReLU(x)");
        dydx.print("dy/dx");
    }

    @Test
    public void testSoftmax() {
        int[] shape = new int[]{4, 3, 1}; // a batch of two vectors of length three

        Batch in = new Batch(shape);
        in.put(0, new Matrix(new double[]{0, 0, 0}, 3));
        in.put(1, new Matrix(new double[]{-1, -1, -1}, 3));
        in.put(2, new Matrix(new double[]{-1, 0, 3}, 3));
        in.put(3, new Matrix(new double[]{0, -5, 2}, 3));

        Variable x = new Variable(shape);
        Softmax y = new Softmax(shape, x);

        x.setValue(in);
        Batch softmax = y.getValue();
        Batch dydx = x.getDualValue();

        softmax.print("y = softmax(x)");
        dydx.print("dy/dx");
    }

    /**
     * [1, 2, 3] x [1, 3, 1] -> [1, 2, 1]
     * [1, 2, 1] + [1, 2, 1] -> [1, 2, 1]
     * [1, 2, 1] -> [1, 1, 1]
     */
    @Test
    public void testMatMulGradients() {
        // network
        Variable W = new Variable(new int[]{1, 2, 3});
        Placeholder a = new Placeholder(new int[]{1, 3, 1});
        Matmul b = new Matmul(new int[]{1, 2, 1}, W, a);

        Placeholder y = new Placeholder(new int[]{1, 2, 1});
        MSELoss loss = new MSELoss(new int[]{1, 1, 1}, y, b);

        // set input and gold value
        Batch input = new Batch(new int[]{1, 3, 1});
        input.put(0, Matrix.random(3, 1));
        a.setValue(input);
        input.print("input");

        Batch gold = new Batch(new int[]{1, 2, 1});
        gold.put(0, Matrix.identity(2, 1));
        y.setValue(gold);
        gold.print("gold");

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.00001, 0.001, loss);
        optimization.fit();

        // some info
        loss.getValue().print("loss");

        W.getValue().print("W");
        W.getDualValue().print("db/dW");

        b.getValue().print("Wa = b");
        b.getDualValue().print("b_bar");
    }

    @Test
    public void testFrobeniusNorm() {
        Variable x = new Variable(new int[]{1, 3, 3});
        FrobeniusNorm y = new FrobeniusNorm(new int[]{1, 3, 3}, x);

        x.setValue(Batch.eye(new int[]{1, 3, 3}));
        Batch out = y.getValue();
        Batch dydx = x.getDualValue();

        out.print("y");
        dydx.print("dy/dx");
    }

    @Test
    public void testAverage() {
        int[] vector_shape = new int[]{1, 3, 1};
        int num_vectors = 5;

        List<Node> vs = new ArrayList<>(num_vectors);
        for (int i = 0; i < num_vectors; i++) {
            Placeholder v = new Placeholder(vector_shape);
            v.setValue(Batch.ones(vector_shape));
            vs.add(v);
        }

        Average average = new Average(vector_shape, vs);
        average.getValue().print("average");
    }
}
