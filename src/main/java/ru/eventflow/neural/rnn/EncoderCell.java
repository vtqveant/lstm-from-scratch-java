package ru.eventflow.neural.rnn;

import ru.eventflow.neural.graph.Node;

public class EncoderCell extends LSTM {

    public EncoderCell(Parameters parameters, Node h_previous, Node c_previous, Node input) {
        super(parameters, Parameters.Type.ENCODER, h_previous, c_previous, input);
    }

}
