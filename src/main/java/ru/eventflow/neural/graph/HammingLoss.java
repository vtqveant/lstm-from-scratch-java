package ru.eventflow.neural.graph;

import ru.eventflow.neural.Batch;
import ru.eventflow.neural.rnn.Parameters;

/**
 * Hamming loss is the fraction of the wrong labels to the total number of labels. [...] This is a loss function, so the optimal value is zero.
 * s. https://en.wikipedia.org/wiki/Multi-label_classification
 *
 * <pre>
 *     H = average(y * (1 - y_hat) + (1 - y) * y_hat)
 * </pre>
 * is a continuous approximation of the Hamming loss.
 * <p>
 * s. https://stackoverflow.com/questions/42125472/gradient-calculation-in-hamming-loss-for-multi-label-classification
 *
 * This is well-defined for positive values in the range [0, 1], e.g. probabilities (labels are independent of each other,
 * so vector components may sum to more than one).
 */
public class HammingLoss extends BaseNode {

    /**
     * a hyper-parameter to control regularization
     */
    private double beta;
    private Node y;
    private Node y_hat;
    private int vectorSize;

    /**
     * This is needed to do regularization
     */
    private Parameters parameters;

    public HammingLoss(int[] shape, Node y, Node y_hat, double beta, Parameters parameters) {
        super(shape, y, y_hat);
        if (y.getSize() != y_hat.getSize() || y.getRows() != y_hat.getRows()) {
            throw new IllegalArgumentException("Shape mismatch");
        }
        this.y = y;
        this.y_hat = y_hat;
        this.vectorSize = y.getRows();
        this.parameters = parameters;
        this.beta = beta;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        Batch m = y.getValue();
        Batch m_hat = y_hat.getValue();

//        m.print("m");
//        m_hat.print("m_hat");

        int batchSize = y.getSize();
        Batch partial = new Batch(y.shape());

        // this regularization is meant to prevent embedding vectors to become too small to avoid getting stuck in an all-zero local minimum in the linkage layer
        double regularizationTerm;
        double regularizationDerivative;
        if (parameters != null) {
            Variable embeddingParameters = parameters.get(Parameters.Type.EMBEDDING, "e");
            double[] values = embeddingParameters.getValue().get(0).getColumnPackedCopy();
            if (values.length != 0) {
                double dot = 0.0;
                for (double d : values) {
                    dot += d * d;
                }
                regularizationTerm = dot;
                regularizationDerivative = -2 * beta / Math.pow(regularizationTerm, 1.5);
            } else {
                beta = 0;
                regularizationTerm = 1;
                regularizationDerivative = 0;
            }
        } else {
            beta = 0;
            regularizationTerm = 1;
            regularizationDerivative = 0;
        }

        // result is an average over all batch entries
        double loss = 0;
        for (int j = 0; j < batchSize; j++) {
            for (int i = 0; i < vectorSize; i++) {
                for (int k = 0; k < y.getColumns(); k++) {
                    int[] idx = new int[]{j, i, k};
                    double y_i = m.getDouble(idx);
                    double yhat_i = m_hat.getDouble(idx);

                    loss += y_i * (1 - yhat_i) + (1 - y_i) * yhat_i;

                    double dd = 2 * (1 - y_i) / batchSize + regularizationDerivative;
                    partial.put(idx, dd);
                }
            }
        }
        loss = loss / vectorSize / y.getColumns() / batchSize + beta / regularizationTerm;

        partials.put(y_hat, partial);

        value = Batch.scalar(loss);
        return value;
    }

    /**
     * short-circuiting here a little bit, since we know that the dual is scalar
     */
    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch partial = partials.get(child);
        return partial.times(dual.get(new int[]{0, 0, 0}));
    }

}
