package ru.eventflow.neural.graph;

import Jama.Matrix;
import ru.eventflow.neural.Batch;

/**
 * Wa = b
 * <p>
 * A matrix times a vector
 * <p>
 * An example for a non-square matrix
 * <p>
 * W = [1, 2, 3]
 * a = [1, 3, 1]
 * [1, 2, 3] x [1, 3, 1] -> [1, 2, 1]
 * <p>
 * db/da = [1, 3, 2]
 * db/dW = [2, 3, 2]
 * <p>
 * b_bar = [1, 2, 1]
 * <p>
 * a_bar = [1, 3, 1]
 * W_bar = [1, 2, 3]
 */
public class Matmul extends BaseNode {

    private Node matrix;
    private Node vector;

    public Matmul(int[] shape, Node matrix, Node vector) {
        super(shape, matrix, vector);
        this.matrix = matrix;
        this.vector = vector;
    }

    @Override
    public Batch getValue() {
        if (value != null) {
            return value;
        }

        // gradients
        Batch dbda = new Batch(new int[]{1, matrix.getColumns(), matrix.getRows()}); // [1, 3, 2]
        Batch dbdW = new Batch(new int[]{matrix.getRows(), matrix.getColumns(), matrix.getRows()}); // [2, 3, 2]
        for (int i = 0; i < matrix.getRows(); i++) { // for each component of an output vector...
            for (int j = 0; j < matrix.getColumns(); j++) { // ...compute a matrix of partial derivatives of this vector component w.r.t. each element of an input matrix
                dbdW.put(new int[]{i, j, i}, vector.getValue().get(new int[]{0, j, 0}));
                dbda.put(new int[]{0, j, i}, matrix.getValue().getDouble(new int[]{0, i, j}));
            }
        }
        partials.put(matrix, dbdW);
        partials.put(vector, dbda);

        value = matrix.getValue().times(vector.getValue());
        return value;
    }

    @Override
    public Batch applyChainRule(Node child) {
        Batch dual = getDualValue();
        Batch partial = partials.get(child);

        if (partial.getSize() != 1) { // partial is a rank three tensor
            Batch result = new Batch(new int[]{1, partial.getSize(), partial.getRows()});
            for (int i = 0; i < partial.getSize(); i++) {
                Matrix v_i = partial.get(i).times(dual.get(0));
                result.get(0).setMatrix(i, i, 0, partial.getRows() - 1, v_i.transpose());
            }
            return result;
        } else { // partial is a vector
            return partial.times(dual);
        }
    }

}