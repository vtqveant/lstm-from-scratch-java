package ru.eventflow.neural.rnn;

import ru.eventflow.neural.Batch;
import ru.eventflow.neural.graph.Matmul;
import ru.eventflow.neural.graph.Node;
import ru.eventflow.neural.graph.Placeholder;

import java.util.ArrayList;
import java.util.List;

/**
 * Collects LSTM output (i.e. the output_size output, before it is converted to a y_hat, i.e. an embedding, not yet decoded)
 */
public class VanillaDecoder implements Decoder {

    public static final int MAX_OUTPUT_LENGTH = 2;
    private Parameters parameters;
    private List<String> teacher;

    public VanillaDecoder(Parameters parameters, List<String> teacher) {
        this.parameters = parameters;
        this.teacher = teacher;
    }

    @Override
    public List<Node> decode(NetworkFactory factory, List<LSTM> encoder, boolean verbose, boolean teacherForcing) {
        List<Node> outputs = new ArrayList<>();
        List<Node> y_hats = new ArrayList<>();

        List<Node> encoderHiddenStates = new ArrayList<>(encoder.size());
        for (LSTM cell : encoder) {
            encoderHiddenStates.add(cell.h);
        }

        // attention over input sequence
        Parameters localParameters = factory.getParameters();
        Attention attention = new Attention(
                localParameters.get(Parameters.Type.ATTENTION, "W_a"),
                localParameters.get(Parameters.Type.ATTENTION, "W_c"),
                encoderHiddenStates,
                localParameters.hiddenSize,
                localParameters.outputSize
        );

        Placeholder eos = new Placeholder(new int[]{1, parameters.onehotSize, 1});
        eos.setValue(parameters.onehot(Parameters.EOS));

        Placeholder none = new Placeholder(new int[]{1, parameters.embeddingSize, 1});
        none.setValue(Batch.zeros(new int[]{1, parameters.embeddingSize, 1}));

        StringBuilder sb = new StringBuilder();
        int outputPosition = 0;
        String s;
        DecoderCell cell = null;
        do {
            outputPosition++;

            DecoderCell previous = cell;
            if (outputPosition == 1) {
                EncoderCell lastEncoderCell = (EncoderCell) encoder.get(encoder.size() - 1);
                cell = factory.buildInitialDecoderCell(lastEncoderCell, attention, none);
            } else {
                Node input;

                if (teacherForcing) {

//                    System.out.println("DEBUG: output position = " + outputPosition);

                    // embedding of the teacher
                    if (outputPosition <= teacher.size()) {
                        String t = teacher.get(outputPosition - 2); // for pos. 2 we pass teacher value from the previous pos.
//                        System.out.println("DEBUG: teacher for pos. " + outputPosition +" = " + t);

                        Placeholder y = new Placeholder(new int[]{1, parameters.onehotSize, 1});
                        y.setValue(parameters.onehot(t));

                        input = new Matmul(new int[]{1, parameters.embeddingSize, 1}, parameters.get(Parameters.Type.EMBEDDING, "e"), y);
                    } else {
                        input = new Matmul(new int[]{1, parameters.embeddingSize, 1}, parameters.get(Parameters.Type.EMBEDDING, "e"), eos);
                    }
                } else {
                    input = previous.output;
                }

                cell = factory.buildDecoderCell(previous, attention, input);
            }


            // get the prediction -- loss is NOT computed here
            Batch distribution = cell.y_hat.getValue();
            s = parameters.decode(distribution);


            // collect outputs and y_hats
            if (!s.equals(Parameters.EOS)) {
                outputs.add(cell.output);
                y_hats.add(cell.y_hat);

                sb.append(String.format("%-6s", s));
                sb.append(" ");

                // print distribution
                for (int i = 0; i < distribution.getRows(); i++) {
                    sb.append(String.format("%.6f ", distribution.get(new int[]{0, i, 0})));
                }
                sb.append("\n");
                distribution.print(s);
            }

        } while (!s.equals(Parameters.EOS) && outputPosition < MAX_OUTPUT_LENGTH);

        if (verbose) {
            System.out.println(sb.toString());

            System.out.println("-- Attention --");
            List<Node> attentionDistributions = attention.getDistributions();
            StringBuilder sb2 = new StringBuilder();
            for (Node a : attentionDistributions) {
                double[] dist = a.getValue().get(0).getColumnPackedCopy();
                for (double d : dist) {
                    sb2.append(String.format("%.6f ", d));
                }
                sb2.append("\n");
            }
            System.out.println(sb2);
        }

//        return outputs;
        return y_hats;
    }

}
