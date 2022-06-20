package ru.eventflow.neural.rnn;

import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Placeholder;
import ru.eventflow.neural.Batch;

public class NetworkFactory {

    private final Parameters parameters;

    /**
     * Parameters must be copied, so that nodes created with a separate factory be isolated from other nodes (crucial for parallel computation)
     */
    public NetworkFactory(Parameters parameters) {
        this.parameters = parameters;
    }

    public EncoderCell buildInitialEncoderCell(Node input) {
        return new EncoderCell(parameters, h0(), c0(), input);
    }

    public EncoderCell buildEncoderCell(LSTM previous, Node input) {
        return new EncoderCell(parameters, previous.h, previous.c, input);
    }

    public DecoderCell buildInitialDecoderCell(LSTM previous, Attention attention, Node input) {
        return new DecoderCell(parameters, attention, previous.h, c0(), input);
    }

    public DecoderCell buildDecoderCell(LSTM previous, Attention attention, Node input) {
        return new DecoderCell(parameters, attention, previous.h, previous.c, input);
    }

    private Placeholder h0() {
        Placeholder h0 = new Placeholder(new int[]{1, parameters.hiddenSize, 1});
        h0.setValue(Batch.zeros(new int[]{1, parameters.hiddenSize, 1}));
        return h0;
    }

    private Placeholder c0() {
        Placeholder c0 = new Placeholder(new int[]{1, parameters.hiddenSize, 1});
        c0.setValue(Batch.zeros(new int[]{1, parameters.hiddenSize, 1}));
        return c0;
    }

    public Parameters getParameters() {
        return parameters;
    }
}