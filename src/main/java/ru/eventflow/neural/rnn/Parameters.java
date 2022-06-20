package ru.eventflow.neural.rnn;

import Jama.Matrix;
import ru.eventflow.neural.graph.Variable;
import ru.eventflow.neural.Batch;

import java.io.Serializable;
import java.util.*;

public class Parameters implements Serializable {

    public static final String EOS = "</s>";

    /**
     * encoder input vector dimensionality equals to decoder input and output dimensionality.
     */
    int embeddingSize; // = input size

    /**
     * hidden layer activation (state vector) dimensionality
     */
    int hiddenSize;

    /**
     * encoder and decoder cells may have different I/O dimensions, but here decoder must output vectors from the same vector space as input embeddings
     * (this is one of the main ideas of the architecture)
     */
    int outputSize;

    int onehotSize;

    private final Map<String, Variable> encoderParameters = new HashMap<>();
    private final Map<String, Variable> decoderParameters = new HashMap<>();
    private final Map<String, Variable> attentionParameters = new HashMap<>();
    private Variable embeddings;

    /**
     * Tokens that embedding get mapped to
     */
    private final List<String> tokens;

    /**
     * One-hot encoding of tokens
     */
    private final Map<String, Integer> onehot;

    public Parameters(int embedding_size, int hidden_size, List<String> tokens) {
        this.embeddingSize = embedding_size;
        this.outputSize = embedding_size;
        this.hiddenSize = hidden_size;
        this.onehotSize = tokens.size();

        populate(encoderParameters);
        populate(decoderParameters);
        populateAttentionParameters();

        embeddings = new Variable("e", new int[]{1, embedding_size, onehotSize}, Batch.rand(new int[]{1, embedding_size, onehotSize}));
        this.tokens = new ArrayList<>(tokens);
        onehot = new HashMap<>(onehotSize);

        for (int i = 0; i < tokens.size(); i++) {
            onehot.put(tokens.get(i), i);
        }
    }

    public Batch onehot(String t) {
        if (!onehot.containsKey(t)) {
            System.err.println(t);
        }

        int rowIndex = onehot.get(t);
        Batch vector = new Batch(new int[]{1, onehotSize, 1});
        vector.put(new int[]{0, rowIndex, 0}, 1.0);
        return vector;
    }

    public Batch embedding(String t) {
        if (!onehot.containsKey(t)) {
            System.err.println(t);
        }

        int column = onehot.get(t);
        Matrix m = embeddings.getValue().get(0).getMatrix(0, embeddingSize - 1, column, column);
        Batch vector = new Batch(new int[]{1, embeddingSize, 1});
        vector.put(0, m);
        return vector;
    }

    public String decode(Batch distribution) {
        if (distribution.getSize() != 1 || distribution.getRows() != tokens.size() || distribution.getColumns() != 1) {
            throw new IllegalArgumentException("Invalid distribution");
        }

        double[] probabilities = distribution.get(0).getColumnPackedCopy();
        double max = 0;
        int i_max = 0;
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > max) {
                max = probabilities[i];
                i_max = i;
            }
        }
        return tokens.get(i_max);
    }

    /**
     * A copy is needed for detaching one network from the other. The values are still shared, though, as they don't change.
     */
    public Parameters copy() {
        Parameters copy = new Parameters(embeddingSize, hiddenSize, tokens);
        for (Map.Entry<String, Variable> entry : encoderParameters.entrySet()) {
            copy.encoderParameters.put(entry.getKey(), entry.getValue().copy());
        }
        for (Map.Entry<String, Variable> entry : decoderParameters.entrySet()) {
            copy.decoderParameters.put(entry.getKey(), entry.getValue().copy());
        }
        for (Map.Entry<String, Variable> entry : attentionParameters.entrySet()) {
            copy.attentionParameters.put(entry.getKey(), entry.getValue().copy());
        }
        copy.embeddings = embeddings.copy();
        return copy;
    }

    private void populateAttentionParameters() {
        attentionParameters.put("W_a", new Variable("W_a", new int[]{1, hiddenSize, hiddenSize}, Batch.xavier(new int[]{1, hiddenSize, hiddenSize})));
        attentionParameters.put("W_c", new Variable("W_c", new int[]{1, outputSize, 2 * hiddenSize}, Batch.xavier(new int[]{1, outputSize, 2 * hiddenSize})));
        attentionParameters.put("W_y", new Variable("W_y", new int[]{1, outputSize, outputSize}, Batch.xavier(new int[]{1, outputSize, outputSize})));
    }

    private void populate(Map<String, Variable> parameters) {
        // weights for the input gate
        parameters.put("W_i", new Variable("W_i", new int[]{1, hiddenSize, embeddingSize}, Batch.xavier(new int[]{1, hiddenSize, embeddingSize})));
        parameters.put("U_i", new Variable("U_i", new int[]{1, hiddenSize, hiddenSize}, Batch.xavier(new int[]{1, hiddenSize, hiddenSize})));

        // weights and bias for the forget gate
        parameters.put("W_f", new Variable("W_f", new int[]{1, hiddenSize, embeddingSize}, Batch.xavier(new int[]{1, hiddenSize, embeddingSize})));
        parameters.put("U_f", new Variable("U_f", new int[]{1, hiddenSize, hiddenSize}, Batch.xavier(new int[]{1, hiddenSize, hiddenSize})));
        parameters.put("b_f", new Variable("b_f", new int[]{1, hiddenSize, 1}, Batch.ones(new int[]{1, hiddenSize, 1}))); // (Jozefowicz et al. 2015)

        // weights for the memory computation
        parameters.put("W_c", new Variable("W_c", new int[]{1, hiddenSize, embeddingSize}, Batch.xavier(new int[]{1, hiddenSize, embeddingSize})));
        parameters.put("U_c", new Variable("U_c", new int[]{1, hiddenSize, hiddenSize}, Batch.xavier(new int[]{1, hiddenSize, hiddenSize})));

        //  weights for the output gate
        parameters.put("W_o", new Variable("W_o", new int[]{1, hiddenSize, embeddingSize}, Batch.xavier(new int[]{1, hiddenSize, embeddingSize})));
        parameters.put("U_o", new Variable("U_o", new int[]{1, hiddenSize, hiddenSize}, Batch.xavier(new int[]{1, hiddenSize, hiddenSize})));

        // weights and bias for the decoder cells
        parameters.put("W_y", new Variable("W_y", new int[]{1, outputSize, hiddenSize}, Batch.xavier(new int[]{1, outputSize, hiddenSize})));
        parameters.put("b_y", new Variable("b_y", new int[]{1, outputSize, 1}, Batch.xavier(new int[]{1, outputSize, 1})));
    }

    public Variable get(Type type, String name) {
        if (type == Type.ENCODER) {
            return encoderParameters.get(name);
        }
        if (type == Type.DECODER) {
            return decoderParameters.get(name);
        }
        if (type == Type.ATTENTION) {
            return attentionParameters.get(name);
        }
        if (type == Type.EMBEDDING) {
            return embeddings;
        }
        throw new IllegalArgumentException();
    }

    public void set(Type type, String name, Variable parameter) {
        if (type == Type.ENCODER) {
            encoderParameters.put(name, parameter);
        } else if (type == Type.DECODER) {
            decoderParameters.put(name, parameter);
        } else if (type == Type.ATTENTION) {
            attentionParameters.put(name, parameter);
        } else if (type == Type.EMBEDDING) {
            embeddings = parameter;
        } else {
            throw new IllegalArgumentException();
        }
    }

    public Collection<Variable> getAll(Type type) {
        if (type == Type.ENCODER) {
            return encoderParameters.values();
        } else if (type == Type.DECODER) {
            return decoderParameters.values();
        } else if (type == Type.ATTENTION) {
            return attentionParameters.values();
        } else if (type == Type.EMBEDDING) {
            return Collections.singletonList(embeddings);
        } else {
            throw new IllegalArgumentException();
        }
    }

    public enum Type {
        ENCODER, DECODER, ATTENTION, EMBEDDING
    }

}
