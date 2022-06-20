package ru.eventflow.neural.rnn;

import ru.eventflow.neural.graph.Node;

import java.util.List;

public interface Decoder {

    List<Node> decode(NetworkFactory factory, List<LSTM> encoder, boolean verbose, boolean teacherForcing);

}
