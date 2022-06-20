package ru.eventflow.neural;

import Jama.Matrix;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.optimization.Optimization;
import ru.eventflow.neural.optimization.SimpleOptimization;
import ru.eventflow.neural.rnn.Attention;

import java.util.Arrays;

public class AttentionTest {

    @Test
    public void testAttentionLayer() {
        int[] vector_shape = new int[]{1, 3, 1}; // 1 vector of size 3

        // an input is a sequence of five 3-dim column vectors
        Placeholder[] vs = new Placeholder[5];
        for (int i = 0; i < 5; i++) {
            Placeholder p = new Placeholder(vector_shape);
            p.setValue(Batch.rand(vector_shape));
            vs[i] = p;
        }

        Variable attentionWeights = new Variable(new int[]{1, 3, 3}, Batch.xavier(new int[]{1, 3, 3}));

        // an output is a single vector
        Placeholder u = new Placeholder(vector_shape);
        u.setValue(Batch.rand(vector_shape));

        // pack the input vectors and take a dot product of the matrix with the output vector
        Pack p = new Pack(new int[]{1, 3, 5}, vs);
        Node attention = new Softmax(new int[]{1, 5, 1}, new Matmul(new int[]{1, 5, 1},
                new Transpose(new int[]{1, 5, 3}, p),
                new Matmul(vector_shape, attentionWeights, u)
        ));

        // the target distribution to make sure we can compute the derivatives and fit the model
        Batch gold = new Batch(new int[]{1, 5, 1});
        gold.put(0, new Matrix(new double[]{0.1, 0.9, 0, 0, 0}, 5));
        Placeholder y = new Placeholder(new int[]{1, 5, 1});
        y.setValue(gold);

        CrossEntropyLoss loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, attention);

        loss.getValue().print("loss");
        attention.getValue().print("distribution (attention)");
        attentionWeights.getValue().print("attention weights");
        attentionWeights.getDualValue().print("partial for attention weights");

        Optimization optimization = new SimpleOptimization(0.01, 0.00001, loss);
        optimization.fit();

        loss.getValue().print("loss");
        attention.getValue().print("distribution (attention)");
        attentionWeights.getValue().print("attention weights");
        attentionWeights.getDualValue().print("partial for attention weights");
    }

    @Test
    public void testAttentionOutput() {
        int k = 5; // input sequence length
        int d = 3; // hidden vector size
        int o = 4; // output vector size

        // an encoder hidden states is a sequence of k d-dim column vectors
        Placeholder[] encoderStates = new Placeholder[k];
        for (int i = 0; i < k; i++) {
            Placeholder p = new Placeholder(new int[]{1, d, 1});
            p.setValue(Batch.rand(new int[]{1, d, 1}));
            encoderStates[i] = p;
        }

        // a decoder hidden states is a sequence of three d-dim column vectors
        Placeholder decoderState = new Placeholder(new int[]{1, d, 1});
        decoderState.setValue(Batch.rand(new int[]{1, d, 1}));

        // two matrices of parameters (one to produce the attention distribution and another one to generate the output)
        Variable W_a = new Variable(new int[]{1, d, d}, Batch.xavier(new int[]{1, d, d}));
        Variable W_c = new Variable(new int[]{1, o, 2 * d}, Batch.xavier(new int[]{1, o, 2 * d}));

        Attention attention = new Attention(W_a, W_c, Arrays.asList(encoderStates), d, o);
        Node output = attention.buildOutputNode(decoderState);
        output.getValue().print("decoderState");

//            new ComputationGraphVisualizer().visualize(output);
//            ComputationGraphVisualizer.block();

        W_a.getValue().print("W_a");
        W_a.getDualValue().print("W_a dual");
    }

    @Test
    public void testFitAttentionWeights() {
        int k = 5; // input sequence length
        int d = 3; // hidden vector size
        int o = 4; // output vector size

        // an encoder hidden states is a sequence of k d-dim column vectors
        Placeholder[] encoderStates = new Placeholder[k];
        for (int i = 0; i < k; i++) {
            Placeholder p = new Placeholder(new int[]{1, d, 1});
            p.setValue(Batch.rand(new int[]{1, d, 1}));
            encoderStates[i] = p;
        }

        // a decoder hidden states is a sequence of three d-dim column vectors
        Placeholder decoderState = new Placeholder(new int[]{1, d, 1});
        decoderState.setValue(Batch.rand(new int[]{1, d, 1}));

        // two matrices of parameters (one to produce the attention distribution and another one to generate the output)
        Variable W_a = new Variable(new int[]{1, d, d}, Batch.xavier(new int[]{1, d, d}));
        Variable W_c = new Variable(new int[]{1, o, 2 * d}, Batch.xavier(new int[]{1, o, 2 * d}));

        Attention attention = new Attention(W_a, W_c, Arrays.asList(encoderStates), d, o);
        Node output = new Softmax(new int[]{1, o, 1}, attention.buildOutputNode(decoderState)); // applying softmax here

        // the target distribution to make sure we can compute the derivatives and fit the model
        Batch gold = new Batch(new int[]{1, o, 1});
        gold.put(0, new Matrix(new double[]{0.1, 0.95, 0.01, 0.04}, o));
        Placeholder y = new Placeholder(new int[]{1, o, 1});
        y.setValue(gold);

        CrossEntropyLoss loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, output);

        Optimization optimization = new SimpleOptimization(0.01, 0.0001, loss);
        optimization.fit();

        loss.getValue().print("loss");
        W_a.getValue().print("W_a");
        W_c.getValue().print("W_c");
    }

}
