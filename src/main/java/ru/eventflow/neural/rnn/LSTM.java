package ru.eventflow.neural.rnn;

import ru.eventflow.neural.graph.*;

/**
 * An "output" of an LSTM is a vector of hidden_size!
 */
public abstract class LSTM {

    public Node x;
    public Node h; // hidden state
    public Node c; // memory

    LSTM(Parameters parameters, Parameters.Type type, Node h_previous, Node c_previous, Node x) {
        int[] h_shape = new int[]{1, parameters.hiddenSize, 1}; // memory cell shape = hidden state shape

        this.x = x; // {1, embedding_size, 1}

        // input gate
        Node i = new Sigmoid(h_shape, new Sum(h_shape,
                new Matmul(h_shape, parameters.get(type, "W_i"), x),
                new Matmul(h_shape, parameters.get(type, "U_i"), h_previous)
        ));

        // forget gate
        Node f = new Sigmoid(h_shape, new Sum(h_shape,
                new Matmul(h_shape, parameters.get(type, "W_f"), x),
                new Sum(h_shape, new Matmul(h_shape, parameters.get(type, "U_f"), h_previous), parameters.get(type, "b_f"))
        ));

        // output gate
        Node o = new Sigmoid(h_shape, new Sum(h_shape,
                new Matmul(h_shape, parameters.get(type, "W_o"), x),
                new Matmul(h_shape, parameters.get(type, "U_o"), h_previous)
        ));

        // memory
        Node c_tilde = new Tanh(h_shape, new Sum(h_shape,
                new Matmul(h_shape, parameters.get(type, "W_c"), x),
                new Matmul(h_shape, parameters.get(type, "U_c"), h_previous)
        ));

        c = new Sum(h_shape, new Mul(h_shape, f, c_previous), new Mul(h_shape, i, c_tilde));

        // hidden state = output
        h = new Mul(h_shape, o, new Tanh(h_shape, c));
    }

}
