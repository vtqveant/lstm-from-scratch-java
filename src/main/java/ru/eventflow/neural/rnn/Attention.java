package ru.eventflow.neural.rnn;

import ru.eventflow.neural.Batch;
import ru.eventflow.neural.graph.*;

import java.util.ArrayList;
import java.util.List;

/**
 * The global attention mechanism from (Luong, Phang, Manning. Effective Approaches to Attention-based Neural Machine Translation. 2015)
 */
public class Attention {

    private Variable W_a;
    private Variable W_c;
    private Pack p;
    private int outputSize;
    private int hiddenSize;

    /**
     * softmax over encoder states for each of the decoder hidden states (in the order of appearance)
     */
    private List<Node> distributions = new ArrayList<>();

    /**
     * @param W_a        -- a weight matrix used to produce the attention distribution of shape [1, hiddenSize, hiddenSize]
     * @param W_c        -- a weight matrix used to generate the hidden output (h~) of shape [1, outputSize, 2 * hiddenSize]
     * @param encoder    -- the hidden states produced by the encoder LSTM
     * @param hiddenSize -- the size of the hidden state vector of the encoder LSTM
     * @param outputSize -- the size of the non-normalized output vector (must be equal to the size of the gold value)
     */
    public Attention(Variable W_a, Variable W_c, List<Node> encoder, int hiddenSize, int outputSize) {
        this.W_a = W_a;
        this.W_c = W_c;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        // pack the input vectors and take a dot product of the matrix with the output vector
        this.p = new Pack(new int[]{1, hiddenSize, encoder.size()}, encoder);
    }

    /**
     * Builds an output hidden vector (h~)
     * <p>
     * This is used to build an attention over the input sequence
     */
    public Node buildOutputNode(Node h) {
        int k = p.getColumns(); // length of encoder

        // an attention distribution -- visualize it!
        Node a = new Softmax(new int[]{1, k, 1}, new Matmul(new int[]{1, k, 1},
                new Transpose(new int[]{1, k, hiddenSize}, p),
                new Matmul(new int[]{1, hiddenSize, 1}, W_a, h)
        ));

        // for visualization
        distributions.add(a);

        // context vector for global attention -- a weighted average of source vectors
        Placeholder denominator = new Placeholder(new int[]{1, hiddenSize, 1});
        denominator.setValue(Batch.ones(new int[]{1, hiddenSize, 1}).times(1d / k));
        Node c = new Mul(new int[]{1, hiddenSize, 1}, new Matmul(new int[]{1, hiddenSize, 1}, p, a), denominator);

        // produce the output
        return new Tanh(new int[]{1, outputSize, 1},
                new Matmul(new int[]{1, outputSize, 1},
                        W_c,
                        new Concat(new int[]{1, 2 * hiddenSize, 1}, c, h)
                )
        );
    }

    /**
     * for visualization
     */
    public List<Node> getDistributions() {
        return distributions;
    }
}
