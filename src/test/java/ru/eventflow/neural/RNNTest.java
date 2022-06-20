package ru.eventflow.neural;

import org.junit.Before;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.optimization.SimpleOptimization;

/**
 * This is a vanilla RNN where the cell is a perceptron with one hidden layer,
 * with 5-dimensional input, 2-dimensional hidden layer and 3-dimensional output,
 * with a ReLU activation in the hidden layer.
 * <p>
 * The state of the network is the activation of the hidden layer at time t.
 * Initialize the h[0] with zeros to start with a reset state.
 * <p>
 * Parameters are shared across time steps. The cross-entropy loss is averaged over all output values across the sequence.
 */
public class RNNTest {

    /**
     * input vector dimensionality
     */
    private int n = 5;

    /**
     * hidden layer activation (state vector) dimensionality
     */
    private int d = 2;

    /**
     * output vector dimensionality
     */
    private int m = 3;

    /**
     * weight from the input layer to the hidden layer
     */
    private Node W_hx;

    /**
     * bias for the hidden layer
     */
    private Node b_hx;

    /**
     * weight for the output layer
     */
    private Node W_y;

    /**
     * bias for the output layer
     */
    private Node b_y;

    /**
     * weights for the state transition
     */
    private Node W_hh;


    @Before
    public void setUp() {
        // these are my shared parameters
        W_hx = new Variable(new int[]{1, d, n}, Batch.rand(new int[]{1, d, n}));
        b_hx = new Variable(new int[]{1, d, 1}, Batch.rand(new int[]{1, d, 1}));

        W_y = new Variable(new int[]{1, m, d}, Batch.rand(new int[]{1, m, d}));
        b_y = new Variable(new int[]{1, m, 1}, Batch.rand(new int[]{1, m, 1}));

        W_hh = new Variable(new int[]{1, d, d}, Batch.eye(new int[]{1, d, d}));
    }

    @Test
    public void testBuildComputationGraph() {
        // reset cell's state
        Placeholder h0 = new Placeholder(new int[]{1, d, 1});
        h0.setValue(Batch.zeros(new int[]{1, d, 1}));

        Placeholder zero = new Placeholder(new int[]{1, m, 1});
        zero.setValue(Batch.zeros(new int[]{1, m, 1}));

        // the RNN
        Cell t0 = new Cell(h0, zero); // the y_hat_previous is currently ignored
        Cell t1 = new Cell(t0.h, t0.y_hat);

        // set input values
        t0.x.setValue(Batch.rand(new int[]{1, n, 1}));
        t1.x.setValue(Batch.rand(new int[]{1, n, 1}));

        // gold value
        Placeholder y = new Placeholder(new int[]{1, m, 1});
        y.setValue(Batch.uniformDistributionVector(m));

        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, t1.y_hat);

        // forward
        t0.y_hat.getValue().print("y_hat (t0)"); // all outputs MUST be computed during the forward pass, otherwise you'll get NPE's in the backward pass
        t1.y_hat.getValue().print("y_hat (t1)");
        loss.getValue().print("loss");

        // backward
        W_hh.getDualValue().print("dloss/dW_hh (before training)");

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.01, 0.0001, loss);
        optimization.fit();

        W_hh.getDualValue().print("dloss/dW_hh (after training)");
        W_hx.getDualValue().print("dloss/dW_hx (after training)");
        W_y.getDualValue().print("dloss/dW_y (after training)");

//        new ComputationGraphVisualizer().visualize(loss);
//        ComputationGraphVisualizer.block();
    }

    private class Cell {

        private Placeholder x;
        private Node h; // cell's state
        private Node y_hat;

        public Cell(Node h_previous, Node y_hat_previous) {
            x = new Placeholder(new int[]{1, n, 1});

            // here I add the result from the previous timestamp to the input, but set it to being equal to zero
            // this is a little hackish to make sure all outputs get computed during the forward pass
            // however, for the decoder part in a seq2seq model this is the right approach (there will be another
            // weight matrix or a gate to control how much of the previous output is added to the input)
            Placeholder zero = new Placeholder(new int[]{1, n, m});
            zero.setValue(Batch.zeros(new int[]{1, n, m}));

            Sum h1 = new Sum(new int[]{1, n, 1}, x, new Matmul(new int[]{1, n, 1}, zero, y_hat_previous));

            h = new ReLU(new int[]{1, d, 1},
                    new Sum(new int[]{1, d, 1},
                            new Matmul(new int[]{1, d, 1}, W_hh, h_previous),
                            new Sum(new int[]{1, d, 1}, new Matmul(new int[]{1, d, 1}, W_hx, h1), b_hx)
                    )
            );
            y_hat = new Softmax(new int[]{1, m, 1}, new Sum(new int[]{1, m, 1}, new Matmul(new int[]{1, m, 1}, W_y, h), b_y));
        }
    }

}
