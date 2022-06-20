package ru.eventflow.neural.rnn;

import ru.eventflow.neural.graph.Matmul;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Softmax;
import ru.eventflow.neural.graph.Transpose;

public class DecoderCell extends LSTM {

    public final Node y_hat;
    public final Node output;

    public DecoderCell(Parameters parameters, Attention attention, Node h_previous, Node c_previous, Node input) {
        super(parameters, Parameters.Type.DECODER, h_previous, c_previous, input);

        // this is an embedding vector, y_hat simply decodes it with the same embedding parameters which were used to encode input categories
        output = new Matmul(new int[]{1, parameters.outputSize, 1},
                parameters.get(Parameters.Type.ATTENTION, "W_y"),
                attention.buildOutputNode(h)
        );

        // [1, onehotSize, outputSize] x [1, outputSize, 1]  --> [1, onehotSize, 1]
        y_hat = new Softmax(new int[]{1, parameters.onehotSize, 1},
                new Matmul(new int[]{1, parameters.onehotSize, 1},
                        new Transpose(new int[]{1, parameters.onehotSize, parameters.outputSize}, parameters.get(Parameters.Type.EMBEDDING, "e")),
                        output
                )
        );
    }

}
