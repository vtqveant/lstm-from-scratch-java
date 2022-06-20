package ru.eventflow.neural.visualization;

import edu.uci.ics.jung.algorithms.layout.*;
import edu.uci.ics.jung.graph.DelegateForest;
import edu.uci.ics.jung.graph.DirectedGraph;
import edu.uci.ics.jung.graph.DirectedSparseMultigraph;
import edu.uci.ics.jung.visualization.DefaultVisualizationModel;
import edu.uci.ics.jung.visualization.GraphZoomScrollPane;
import edu.uci.ics.jung.visualization.VisualizationModel;
import edu.uci.ics.jung.visualization.VisualizationViewer;
import edu.uci.ics.jung.visualization.control.DefaultModalGraphMouse;
import edu.uci.ics.jung.visualization.control.ModalGraphMouse;
import edu.uci.ics.jung.visualization.decorators.EdgeShape;
import edu.uci.ics.jung.visualization.renderers.VertexLabelAsShapeRenderer;
import org.apache.commons.collections15.Transformer;
import ru.eventflow.neural.graph.Node;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

/**
 * JUNG graph visualization
 */
public class ComputationGraphVisualizer {

    private static int id = 0;

    public static void block() {
        BufferedReader reader = new BufferedReader(new InputStreamReader(System.in));
        try {
            reader.read();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void visualize(Node root) {
        DirectedGraph<Node, Integer> graph = buildGraph(root);
        VisualizationViewer<Node, Integer> vv = configureVisualizationViewer(graph);

        SwingUtilities.invokeLater(new Runnable(){
            public void run(){
                JFrame frame = new JFrame();
                frame.setPreferredSize(new Dimension(1200, 1000));
                frame.getContentPane().add(new GraphZoomScrollPane(vv));
                frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
                frame.pack();
                frame.setVisible(true);
            }
        });
    }

    private DirectedGraph<Node, Integer> buildGraph(Node root) {
        DirectedGraph<Node, Integer> graph = new DirectedSparseMultigraph<>();
        return buildGraph(graph, root);
    }

    private DirectedGraph<Node, Integer> buildGraph(DirectedGraph<Node, Integer> graph, Node node) {
        graph.addVertex(node);
        for (Node child : node.getChildren()) {
            if (!graph.containsVertex(child)) {
                graph.addVertex(child);
            }
            if (graph.findEdge(child, node) == null) {
                graph.addEdge(id++, child, node);
            }
            buildGraph(graph, child);
        }
        return graph;
    }

    public VisualizationViewer<Node, Integer> configureVisualizationViewer(DirectedGraph<Node, Integer> graph) {
        Layout<Node, Integer> layout = new FRLayout2<>(new DelegateForest<>(graph));
//        Layout<Node, Integer> layout = new SpringLayout2<Node, Integer>(new DelegateForest<>(graph));
//        Layout<Node, Integer> layout = new StaticLayout<Node, Integer>(new DelegateForest<>(graph));
        final VisualizationModel<Node, Integer> visualizationModel = new DefaultVisualizationModel<>(layout);
        VisualizationViewer<Node, Integer> vv = new VisualizationViewer<>(visualizationModel);

        DefaultModalGraphMouse graphMouse = new DefaultModalGraphMouse();
        vv.setGraphMouse(graphMouse);
        JComboBox modeBox = graphMouse.getModeComboBox();
        modeBox.addItemListener(graphMouse.getModeListener());
        graphMouse.setMode(ModalGraphMouse.Mode.PICKING);

        // this class will provide both label drawing and vertex shapes
        VertexLabelAsShapeRenderer<Node, Integer> vlasr = new VertexLabelAsShapeRenderer<>(vv.getRenderContext());
        HTMLTransformer htmlTransfomer = new HTMLTransformer();
        vv.getRenderContext().setVertexLabelTransformer(htmlTransfomer);
        vv.getRenderContext().setVertexShapeTransformer(vlasr);
        vv.getRenderer().setVertexLabelRenderer(vlasr);

        vv.getRenderContext().setVertexFillPaintTransformer(new PaintTransformer());
        vv.getRenderContext().setEdgeShapeTransformer(new EdgeShape.Line());

        return vv;
    }

    private class HTMLTransformer implements Transformer<Node, String> {
        @Override
        public String transform(Node node) {
            return "<html><center><font size=\"2\">" + node.toString() + "</font></html>";
        }
    }

    /**
     * Unconditionally changes the color of the node
     */
    private class PaintTransformer implements Transformer<Node, Paint> {
        @Override
        public Paint transform(Node node) {
            return Color.PINK;
        }
    }

}
