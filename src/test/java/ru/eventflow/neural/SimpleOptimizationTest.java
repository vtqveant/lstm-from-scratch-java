package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Assert;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.optimization.SimpleOptimization;

import static org.junit.Assert.assertEquals;

public class SimpleOptimizationTest {

    /**
     * f = (x - 1)^2 ;
     * f' = 2 * (x - 1);
     */
    @Test
    public void testOptimizer() {
        int[] shape = new int[]{1, 1, 1};

        Placeholder c = new Placeholder(shape);
        Variable x = new Variable(shape);
        Node sum = new Sum(shape, x, c);
        Node y = new Matmul(shape, sum, sum);

        Batch in = new Batch(shape);
        in.put(0, new Matrix(new double[]{-1}, 1));
        c.setValue(in);

        SimpleOptimization optimization = new SimpleOptimization(0.001, 0.00001, y);
        optimization.fit();

        y.getValue().print("y");
    }

    /**
     * This problem is ill-conditioned, so don't expect good results
     */
    @Test
    public void testPerceptron() {
        int[] vector_shape = new int[]{1, 5, 1}; // 1 vector of size 5
        int[] matrix_shape = new int[]{1, 5, 5};

        Placeholder x = new Placeholder(vector_shape);
        Node W1 = new Variable(matrix_shape, Batch.ones(matrix_shape)); // initialize weights with random values
        Node b1 = new Variable(vector_shape, Batch.ones(vector_shape)); // bias vector
        Node h1 = new ReLU(vector_shape, new Sum(vector_shape, new Matmul(vector_shape, W1, x), b1)); // hidden layer

        Node W2 = new Variable(matrix_shape, Batch.ones(matrix_shape));
        Node b2 = new Variable(vector_shape, Batch.ones(vector_shape));
        Node y_hat = new Softmax(vector_shape, new Sum(vector_shape, new Matmul(vector_shape, W2, h1), b2)); // you don't need an activation in the final layer, since softmax is an activation

        // gold value
        Placeholder y = new Placeholder(vector_shape);
        Batch batch = new Batch(vector_shape);
        batch.put(0, new Matrix(new double[]{0.21, 0.02, 0.3, 0.01, 0.46}, 5));
        y.setValue(batch);

        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, y_hat);

        // set input
        x.setValue(Batch.ones(vector_shape));

        y.getValue().print("y");
        y_hat.getValue().print("y_hat");

        Assert.assertEquals(5, y_hat.getValue().getRows()); // nr. of vectors
        Assert.assertEquals(1, y_hat.getValue().getColumns()); // vector size

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.01, 0.001, loss);
        optimization.fit();

        // results
        System.out.println();

        loss.getValue().print("loss");
        y.getValue().print("y");
        y_hat.getValue().print("y_hat");

        W1.getValue().print("W1");
        b1.getValue().print("b1");
        W2.getValue().print("W2");
        b2.getValue().print("b2");
    }

    /**
     * This should be able to find a matrix with Frobenius norm equal to 5
     */
    @Test
    public void testFrobeniusOptimization() {
        Variable x = new Variable(new int[]{1, 3, 3});
        FrobeniusNorm y_hat = new FrobeniusNorm(new int[]{1, 1, 1}, x);

        // gold value
        Placeholder y = new Placeholder(new int[]{1, 1, 1});
        y.setValue(Batch.scalar(5));

        Node loss = new MSELoss(new int[]{1, 1, 1}, y, y_hat);

        // set input
        x.setValue(Batch.rand(new int[]{1, 3, 3}));

        SimpleOptimization optimization = new SimpleOptimization(0.0001, 0.00001, loss);
        optimization.fit();

        y.getValue().print("y");
        y_hat.getValue().print("y_hat");
        x.getValue().print("x");
    }

    /*
    @Test
    public void testAttention() {
        Variable x = new Variable(new int[]{6, 5, 1}); // six five-element column vectors to represent categorial embeddings
        Variable W = new Variable(new int[]{6, 5, 5}); // attention weight matrix

        // attention layer is a node which takes input vectors and a weight matrix and produces an attention matrix
        AttentionLayer attn = new AttentionLayer(new int[]{1, 6, 6}, W, x);

        x.setValue(Batch.rand(new int[]{6, 5, 1}));
        W.setValue(Batch.eye(new int[]{6, 5, 5}));

        attn.getValue().print("Attention");
    }
    */

    /**
     * A perceptron with one hidden layer and a softmax output, with 5-dimensional input, 2-dimensional hidden layer
     * and 3-dimensional output (thus a three classes classifier), with a sigmoid activation in the hidden layer and
     * a cross-entropy loss.
     */
    @Test
    public void testPerceptronWithSigmoidActivation() {
        Placeholder x = new Placeholder(new int[]{1, 5, 1});
        Node W1 = new Variable(new int[]{1, 2, 5}, Batch.xavier(new int[]{1, 2, 5})); // initialize weights with random values
        Node b1 = new Variable(new int[]{1, 2, 1}, Batch.zeros(new int[]{1, 2, 1})); // bias vector
        Node h1 = new Sigmoid(new int[]{1, 2, 1}, new Sum(new int[]{1, 2, 1}, new Matmul(new int[]{1, 2, 1}, W1, x), b1)); // hidden layer

        Node W2 = new Variable(new int[]{1, 3, 2}, Batch.xavier(new int[]{1, 3, 2}));
        Node b2 = new Variable(new int[]{1, 3, 1}, Batch.zeros(new int[]{1, 3, 1}));
        Node y_hat = new Softmax(new int[]{1, 3, 1}, new Sum(new int[]{1, 3, 1}, new Matmul(new int[]{1, 3, 1}, W2, h1), b2));

        // gold value
        Placeholder y = new Placeholder(new int[]{1, 3, 1});
        Batch batch = new Batch(new int[]{1, 3, 1});
        batch.put(0, new Matrix(new double[]{0.2, 0.02, 0.78}, 3));
        y.setValue(batch);

        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, y_hat);

//        new ComputationGraphVisualizer().visualize(loss);
//        ComputationGraphVisualizer.block();

        // set input
        x.setValue(Batch.ones(new int[]{1, 5, 1}));

        y.getValue().print("y");
        y_hat.getValue().print("y_hat");

        Assert.assertEquals(3, y_hat.getValue().getRows()); // nr. of vectors
        Assert.assertEquals(1, y_hat.getValue().getColumns()); // vector size

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.001, 0.00001, loss);
        optimization.fit();

        // results
        System.out.println();

        loss.getValue().print("loss");
        y.getValue().print("y");
        y_hat.getValue().print("y_hat");

        W1.getValue().print("W1");
        b1.getValue().print("b1");
        W2.getValue().print("W2");
        b2.getValue().print("b2");
    }
}
