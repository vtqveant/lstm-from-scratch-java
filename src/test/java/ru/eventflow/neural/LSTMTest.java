package ru.eventflow.neural;

import org.junit.Before;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.optimization.SimpleOptimization;
import ru.eventflow.neural.rnn.LSTM;
import ru.eventflow.neural.rnn.NetworkFactory;
import ru.eventflow.neural.rnn.Parameters;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * TODO apply truncated BPTT instead of full BPTT
 * s. https://r2rt.com/styles-of-truncated-backpropagation.html
 */
public class LSTMTest {

    private static final String sentence = "abcd abcd";
    private List<String> tokens;

    @Before
    public void setUp() {
        tokens = Arrays.asList(" ", "*", "a", "b", "c", "d");
    }

    private Node buildEmbeddingNode(Parameters parameters, int charIndex) {
        String ch = String.valueOf(sentence.charAt(charIndex));

        Placeholder embedding = new Placeholder(new int[]{1, tokens.size(), 1});
        embedding.setValue(parameters.embedding(ch));
        return embedding;
    }

    /**
     * This network should learn to produce an "a" given "abcd abcd" as input
     */
    @Test
    public void testBuildNetworkFromString() {
        int onehotSize = tokens.size();
        int embeddingSize = 5;
        int hiddenSize = 10;
        int outputSize = onehotSize;

        int length = sentence.length();

        Parameters parameters = new Parameters(embeddingSize, hiddenSize, tokens);
        NetworkFactory factory = new NetworkFactory(parameters);

        // the LSTM
        LSTM[] cells = new LSTM[length];
        cells[0] = factory.buildInitialEncoderCell(buildEmbeddingNode(parameters, 0));
        for (int i = 1; i < length; i++) {
            cells[i] = factory.buildEncoderCell(cells[i - 1], buildEmbeddingNode(parameters, i));
        }

        // output value
        Node W_y = new Variable(new int[]{1, outputSize, hiddenSize}, Batch.rand(new int[]{1, outputSize, hiddenSize}));
        Node b_y = new Variable(new int[]{1, outputSize, 1}, Batch.rand(new int[]{1, outputSize, 1}));

        // result
        Node h_final = cells[length - 1].h; // final hidden state, from which the prediction is generated
        Node y_hat = new Softmax(new int[]{1, outputSize, 1},
                new Sum(new int[]{1, outputSize, 1}, new Matmul(new int[]{1, outputSize, 1}, W_y, h_final), b_y)
        );

        // gold value
        Placeholder y = new Placeholder(new int[]{1, outputSize, 1});
        y.setValue(parameters.onehot("a"));

        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, y_hat);

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.05, 0.001, loss);
        optimization.fit();

        loss.getValue().print("final loss");

        // print final prediction
        Batch prediction = y_hat.getValue();
        prediction.print("final distribution");
        System.out.println("prediction = " + parameters.decode(prediction));

        assertEquals("a", parameters.decode(prediction));
    }

    /**
     * This test demonstrates how an embedding node is attached to the encoder cell.
     * This network most stupidly learns to produce a uniform distribution from a sequence of randomly initialized vectors
     * (strictly speaking, input vectors are not one-hot vectors)
     */
    @Test
    public void testBuildComputationGraph() {
        int onehotSize = 5;
        int embeddingSize = 4;
        int hiddenSize = 7;
        int outputSize = onehotSize;

        Parameters parameters = new Parameters(embeddingSize, hiddenSize, tokens);
        NetworkFactory factory = new NetworkFactory(parameters);

        // input values
        Placeholder emb0 = new Placeholder(new int[]{1, embeddingSize, 1});
        emb0.setValue(parameters.embedding(" "));

        Placeholder emb1 = new Placeholder(new int[]{1, embeddingSize, 1});
        emb1.setValue(parameters.embedding(" "));

        Placeholder emb2 = new Placeholder(new int[]{1, embeddingSize, 1});
        emb2.setValue(parameters.embedding(" "));

        // the LSTM
        LSTM t0 = factory.buildInitialEncoderCell(emb0);
        LSTM t1 = factory.buildEncoderCell(t0, emb1);
        LSTM t2 = factory.buildEncoderCell(t1, emb2);

        // weights for result
        Node W_y = new Variable(new int[]{1, outputSize, hiddenSize}, Batch.rand(new int[]{1, outputSize, hiddenSize}));
        Node b_y = new Variable(new int[]{1, outputSize, 1}, Batch.rand(new int[]{1, outputSize, 1}));

        // result
        Node y_hat = new Softmax(new int[]{1, outputSize, 1},
                new Sum(new int[]{1, outputSize, 1}, new Matmul(new int[]{1, outputSize, 1}, W_y, t2.h), b_y)
        );

        // gold value
        Placeholder y = new Placeholder(new int[]{1, outputSize, 1});
        y.setValue(Batch.uniformDistributionVector(outputSize));

        Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, y, y_hat);

        // train
        SimpleOptimization optimization = new SimpleOptimization(0.001, 0.00001, loss);
        optimization.fit();

        loss.getValue().print("final loss");
    }

}
