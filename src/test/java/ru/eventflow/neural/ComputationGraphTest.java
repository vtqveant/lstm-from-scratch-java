package ru.eventflow.neural;

import org.junit.Assert;
import org.junit.Ignore;
import org.junit.Test;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.visualization.ComputationGraphVisualizer;

import static org.junit.Assert.assertEquals;

public class ComputationGraphTest {

    private static final int[] SCALAR = new int[]{1, 1, 1};
    private static final int[] VECTOR = new int[]{1, 3, 1};
    private static final int[] MATRIX = new int[]{1, 3, 5};

    @Test
    public void testScalarVariable() {
        Variable v = new Variable(SCALAR);
        v.getValue().print("scalar variable");
        v.getDualValue().print("derivative of scalar variable (w.r.t. itself)");
    }

    @Test
    public void testScalarVariable_setValue() {
        Variable v = new Variable(SCALAR);
        v.setValue(Batch.ones(SCALAR));
        v.getValue().print("scalar variable");
        v.getDualValue().print("derivative of scalar variable (w.r.t. itself)");
    }

    @Test
    public void testVectorVariable() {
        Variable v = new Variable(VECTOR);
        v.getValue().print("vector variable");
        v.getDualValue().print("derivative of vector variable (w.r.t. itself)");
    }

    @Test
    public void testMatrixVariable() {
        Variable v = new Variable(MATRIX);
        v.getValue().print("matrix variable");
        v.getDualValue().print("derivative of matrix variable (w.r.t. itself)");
    }

    @Test
    public void testSumScalars_setValue() {
        Variable v1 = new Variable(SCALAR);
        v1.setValue(Batch.ones(SCALAR));
        System.out.println("--- v1 ---");
        v1.getValue().print("scalar variable");
        v1.getDualValue().print("derivative of scalar variable (w.r.t. itself)");

        Variable v2 = new Variable(SCALAR);
        v2.setValue(Batch.ones(SCALAR));
        System.out.println("--- v2 ---");
        v2.getValue().print("scalar variable");
        v2.getDualValue().print("derivative of scalar variable (w.r.t. itself)");

        Sum sum = new Sum(SCALAR, v1, v2);
        System.out.println("--- sum ---");
        sum.getValue().print("scalar variable");
        sum.getDualValue().print("derivative of scalar variable (w.r.t. itself)");

        System.out.println();
        System.out.println("ds/dv1");
        v1.getDualValue().print();

        System.out.println();
        System.out.println("ds/dv2");
        v2.getDualValue().print();
    }

    @Test
    public void testSumVectors_setValue() {
        Variable v1 = new Variable(VECTOR);
        v1.setValue(Batch.ones(VECTOR));
        v1.getValue().print("v1");
        v1.getDualValue().print("dv1/dv1");

        Variable v2 = new Variable(VECTOR);
        v2.setValue(Batch.ones(VECTOR));
        v2.getValue().print("v2");
        v2.getDualValue().print("dv2/dv2");

        Sum sum = new Sum(VECTOR, v1, v2);
        sum.getValue().print("sum");
        sum.getDualValue().print("ds/ds");

        v1.getDualValue().print("ds/dv1");
        v2.getDualValue().print("ds/dv2");
    }

    @Test
    public void testSumMatrices_setValue() {
        Variable v1 = new Variable(MATRIX);
        v1.setValue(Batch.ones(MATRIX));
        v1.getValue().print("v1");
        v1.getDualValue().print("dv1/dv1");

        Variable v2 = new Variable(MATRIX);
        v2.setValue(Batch.ones(MATRIX));
        v2.getValue().print("v2");
        v2.getDualValue().print("dv2/dv2");

        Sum sum = new Sum(MATRIX, v1, v2);
        sum.getValue().print("s");
        sum.getDualValue().print("ds/ds");

        v1.getDualValue().print("ds/dv1");
        v2.getDualValue().print("ds/dv2");
    }

    /**
     * This is a fully-connected layer mapping a three-element input to a two-element output
     */
    @Test
    public void testMatMul_setValue() {
        Variable W = new Variable(new int[]{1, 2, 3});
        W.setValue(Batch.ones(new int[]{1, 2, 3}));
        W.getValue().print("W");
        W.getDualValue().print("dW/dW");

        Variable a = new Variable(new int[]{1, 3, 1});
        a.setValue(Batch.ones(new int[]{1, 3, 1}));
        a.getValue().print("a");
        a.getDualValue().print("da/da");

        Matmul b = new Matmul(new int[]{1, 2, 1}, W, a);
        b.getValue().print("b");
        b.getDualValue().print("db/db");

        W.getDualValue().print("W_bar");
        a.getDualValue().print("a_bar");
    }

    @Test
    public void testForwardPropagation() {
        int[] vector_shape = new int[]{1, 3, 1}; // 1 vector of size 3
        int[] matrix_shape = new int[]{1, 3, 3};

        Placeholder x = new Placeholder(vector_shape);
        Node W1 = new Variable(matrix_shape, Batch.eye(matrix_shape));
        Node b1 = new Variable(vector_shape); // 3-element bias vector
        Node h = new ReLU(vector_shape, new Sum(vector_shape, new Matmul(vector_shape, W1, x), b1)); // hidden layer
        Node W2 = new Variable(matrix_shape, Batch.eye(matrix_shape));
        Node b2 = new Variable(vector_shape); // 3-element bias vector
        Node y = new Softmax(vector_shape, new ReLU(vector_shape, new Sum(vector_shape, new Matmul(vector_shape, W2, h), b2)));

        x.setValue(Batch.ones(vector_shape));
        y.getValue().print();

        Assert.assertEquals(1, y.getValue().getSize());
        Assert.assertEquals(3, y.getValue().getRows()); // vector size
        Assert.assertEquals(1, y.getValue().getColumns());
    }

    @Test
    public void testPack() {
        int[] vector_shape = new int[]{1, 3, 1}; // 1 vector of size 3

        Variable x1 = new Variable(vector_shape);
        x1.setValue(Batch.ones(vector_shape));

        Placeholder x2 = new Placeholder(vector_shape);
        x2.setValue(Batch.zeros(vector_shape));

        Pack pack = new Pack(new int[]{1, 3, 2}, x1, x2);
        pack.getValue().print("pack");
        pack.getDualValue().print("pack dual");

        Placeholder x3 = new Placeholder(new int[]{1, 2, 1});
        x3.setValue(Batch.ones(new int[]{1, 2, 1}));

        Matmul matmul = new Matmul(vector_shape, pack, x3);
        matmul.getValue().print("matmul");
        matmul.getDualValue().print("matmul dual");

        x1.getDualValue().print("x1 dual (backward pass)");
    }

    @Test
    public void testConcat() {
        Variable x1 = new Variable(new int[]{1, 3, 1});
        x1.setValue(Batch.ones(new int[]{1, 3, 1}));

        Placeholder x2 = new Placeholder(new int[]{1, 2, 1});
        x2.setValue(Batch.zeros(new int[]{1, 2, 1}));

        Concat concat = new Concat(new int[]{1, 5, 1}, x1, x2);

        // a more interesting derivative
        Placeholder W = new Placeholder(new int[]{1, 2, 5});
        W.setValue(Batch.rand(new int[]{1, 2, 5}));

        Matmul matmul = new Matmul(new int[]{1, 2, 1}, W, concat);
        matmul.getValue().print("matmul");
        matmul.getDualValue().print("matmul dual");

        x1.getValue().print("x1");
        x1.getDualValue().print("x1 dual (backward pass)");
    }

    @Test
    public void testTransposeVector() {
        int[] vector_shape = new int[]{1, 3, 1}; // 1 vector of size 3

        Variable x1 = new Variable(vector_shape);
        x1.setValue(Batch.ones(vector_shape));

        Transpose t = new Transpose(new int[]{1, 1, 3}, x1);
        t.getValue().print("t");

        Variable x2 = new Variable(vector_shape);
        x2.setValue(Batch.ones(vector_shape));

        Node scalar = new Matmul(new int[]{1, 1, 1}, t, x2);
        scalar.getValue().print("dot product");

        Placeholder x3 = new Placeholder(new int[]{1, 1, 1});
        x3.setValue(Batch.rand(new int[]{1, 1, 1}));

        Node y = new Matmul(new int[]{1, 1, 1}, scalar, x3);
        y.getValue().print("y");

        x1.getDualValue().print("dy/dx1");
        x2.getDualValue().print("dy/dx2");
    }

    @Test
    public void testTransposePack() {
        Variable x1 = new Variable(new int[]{1, 3, 1});
        x1.setValue(Batch.ones(new int[]{1, 3, 1}));

        Variable x2 = new Variable(new int[]{1, 3, 1});
        x2.setValue(Batch.ones(new int[]{1, 3, 1}));

        Pack pack = new Pack(new int[]{1, 3, 2}, x1, x2);

        Transpose t = new Transpose(new int[]{1, 2, 3}, pack);
        t.getValue().print("t");

        x1.getDualValue().print("dt/dx1");
        x2.getDualValue().print("dt/dx2");
    }

    @Test
    public void testDotProduct() {
        Variable x1 = new Variable(new int[]{1, 3, 1});
        x1.setValue(Batch.ones(new int[]{1, 3, 1}));

        Variable x2 = new Variable(new int[]{1, 3, 1});
        x2.setValue(Batch.ones(new int[]{1, 3, 1}));

        Node dot = new Matmul(new int[]{1, 1, 1}, new Transpose(new int[]{1, 1, 3}, x1), x2);
        dot.getValue().print("dot");

        x1.getDualValue().print("dd/dx1");
        x2.getDualValue().print("dd/dx2");
    }

    @Test
    public void testDotProductWithMatrix() {
        Variable x1 = new Variable(new int[]{1, 3, 2});
        x1.setValue(Batch.ones(new int[]{1, 3, 2}));

        Variable x2 = new Variable(new int[]{1, 3, 1});
        x2.setValue(Batch.ones(new int[]{1, 3, 1}));

        Node dot = new Matmul(new int[]{1, 2, 1}, new Transpose(new int[]{1, 2, 3}, x1), x2);
        dot.getValue().print("dot");

        x1.getDualValue().print("dd/dx1");
        x2.getDualValue().print("dd/dx2");
    }


    @Ignore
    @Test
    public void testVisualizer() {
        int[] shape = new int[]{1, 1, 3}; // 1 vector of size 3

        Placeholder x = new Placeholder(shape);
        Node W1 = new Variable(new int[]{1, 3, 3}, Batch.eye(new int[]{1, 3, 3}));
        Node b1 = new Variable(shape); // 3-element bias vector
        Node h = new ReLU(shape, new Sum(shape, new Matmul(shape, W1, x), b1)); // hidden layer
        Node W2 = new Variable(new int[]{1, 3, 3}, Batch.eye(new int[]{1, 3, 3}));
        Node b2 = new Variable(shape); // 3-element bias vector
        Node y = new Softmax(shape, new Sum(shape, new Matmul(shape, W2, h), b2));

        new ComputationGraphVisualizer().visualize(y);
        ComputationGraphVisualizer.block();
    }

}
