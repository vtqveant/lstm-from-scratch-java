package ru.eventflow.neural.rnn;

import org.apache.log4j.Logger;
import ru.eventflow.neural.Batch;
import ru.eventflow.neural.PersistenceUtils;
import ru.eventflow.neural.dataset.TrainingExample;
import ru.eventflow.neural.graph.*;
import ru.eventflow.neural.visualization.AttentionVisualizer;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.DoubleAdder;

/**
 * TODO implement Truncated BPTT
 *
 * Interesting points (and an embedding viz tool): https://medium.com/@rakesh.chada/understanding-neural-networks-by-embedding-hidden-representations-f256842ebf3a
 *
 * "I used binary cross entropy loss with sigmoid activation in the final layer of the neural network.
 * This way — it just outputs two probabilities for each label — thereby enabling multi-label classification."
 */
public class EncoderDecoderEngine {

    private static final Logger logger = Logger.getLogger(EncoderDecoderEngine.class);

    private static final int SNAPSHOT_FREQ = 1000;
    private final int minibatchSize;
    private final int onehotSize;
    private final int hiddenSize;
    private final int embeddingSize;
    private double learningRate; // will be scheduled, s. below
    private double gradientClippingThreashold;
    private String snapshotDirectory;
    private Parameters parameters;
    private double regularizationCoefficient;

    private ExecutorService executor;
    private AttentionVisualizer attentionVisualizer;

    private EncoderDecoderEngine(int minibatchSize, double learningRate, int embeddingSize, int hiddenSize,
                                 int numThread, double clipping, double regularizationCoefficient, List<String> tokens) {
        this.minibatchSize = minibatchSize;
        this.learningRate = learningRate;
        this.hiddenSize = hiddenSize;
        this.embeddingSize = embeddingSize;
        this.onehotSize = tokens.size();
        this.gradientClippingThreashold = clipping;
        this.regularizationCoefficient = regularizationCoefficient;
        this.executor = Executors.newFixedThreadPool(numThread);

        this.attentionVisualizer = new AttentionVisualizer(new File("/tmp/images"));
    }

    public EncoderDecoderEngine(int minibatchSize, double learningRate, int embeddingSize, int hiddenSize,
                                int numThreads, double clipping, double regularizationCoefficient, InputStream snapshot, List<String> tokens) {
        this(minibatchSize, learningRate, embeddingSize, hiddenSize, numThreads, clipping, regularizationCoefficient, tokens);

        try {
            restoreSnapshot(snapshot);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

    public EncoderDecoderEngine(int minibatchSize, double learningRate, int embeddingSize, int hiddenSize,
                                int numThreads, double clipping, double regularizationCoefficient, String snapshotDirectory, List<String> tokens) {
        this(minibatchSize, learningRate, embeddingSize, hiddenSize, numThreads, clipping, regularizationCoefficient, tokens);
        this.snapshotDirectory = snapshotDirectory;
        this.parameters = new Parameters(embeddingSize, hiddenSize, tokens);
    }

    private void restoreSnapshot(InputStream inputStream) throws IOException, ClassNotFoundException {
        Parameters parameters = PersistenceUtils.read(inputStream);
        if (parameters.hiddenSize != hiddenSize || parameters.embeddingSize != embeddingSize || parameters.onehotSize != onehotSize) {
            throw new IllegalStateException("Dimensions from the model and from the snapshot do not match.");
        }
        this.parameters = parameters;
    }

    private void saveSnapshot(File snapshotDirectory, int epoch) throws IOException {
        String name = "encdec_onehot" + parameters.onehotSize + "_i" + parameters.embeddingSize + "_h" + parameters.hiddenSize + "_epoch" + epoch;
        File file = new File(snapshotDirectory, name);
        FileOutputStream out = new FileOutputStream(file);
        PersistenceUtils.write(out, parameters);
    }

    /**
     * a new loss each time, because the decoder part can change
     */
    public void train(Set<TrainingExample> trainingSet) {
        int epoch = 0;

        ExecutorCompletionService<Void> ecs = new ExecutorCompletionService<>(executor);

        // TODO shuffle (currently I don't need it)
        List<TrainingExample> population = new ArrayList<>(trainingSet);

        // TODO stopping criterion
        // networks will be rebuilt, parameters will be reused
        while (true) {
//            boolean verbose = epoch % 10 == 0;
            boolean verbose = true;

            epoch++;
            long start_ts = System.currentTimeMillis();

            // sample a mini-batch by simply sliding over the (shuffled) data
            TrainingExample[] sample = new TrainingExample[minibatchSize];
            for (int i = 0; i < minibatchSize; i++) {
                sample[i] = population.get((i + epoch * minibatchSize) % population.size());
            }

            DoubleAdder totalMiniBatchLoss = new DoubleAdder();
            Map<String, Batch> encoderGradients = Collections.synchronizedMap(new HashMap<>());
            Map<String, Batch> decoderGradients = Collections.synchronizedMap(new HashMap<>());
            Map<String, Batch> attentionGradients = Collections.synchronizedMap(new HashMap<>());
            Map<String, Batch> embeddingGradients = Collections.synchronizedMap(new HashMap<>());

            // tasks
            List<ForwardBackwardTask> tasks = new ArrayList<>();
            for (TrainingExample example : sample) {
                tasks.add(new ForwardBackwardTask(example, totalMiniBatchLoss, encoderGradients, decoderGradients,
                        attentionGradients, embeddingGradients, verbose));
            }

            // submit and wait for completion
            try {
                for (Callable<Void> callable : tasks) {
                    ecs.submit(callable);
                }
                for (int i = 0; i < tasks.size(); i++) {
                    try {
                        ecs.take().get();
                    } catch (ExecutionException e) {
                        System.out.println("Task threw an exception");
                        logger.error(e);
                    }
                }
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

//            System.gc();

            double averageLoss = totalMiniBatchLoss.doubleValue() / minibatchSize;

            // a single update
            updateParameters(Parameters.Type.DECODER, decoderGradients);
            updateParameters(Parameters.Type.ENCODER, encoderGradients);
            updateParameters(Parameters.Type.ATTENTION, attentionGradients);
            updateParameters(Parameters.Type.EMBEDDING, embeddingGradients);

            if (verbose) {
                long end_ts = System.currentTimeMillis();
                logger.info("epoch = " + epoch + ", lr = " + learningRate + ", average loss = " + averageLoss + ", took " + (end_ts - start_ts) + " ms");
//                System.out.println("epoch = " + epoch + ", lr = " + learningRate + ", average loss = " + averageLoss + ", took " + (end_ts - start_ts) + " ms");
//                System.out.println();
            }

            // snapshots
            if (epoch % SNAPSHOT_FREQ == 0) {
                try {
                    saveSnapshot(new File(snapshotDirectory), epoch);
                    System.out.println("Snapshot persisted at epoch " + epoch);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            // schedule
            if (epoch % 10000 == 0) {
                learningRate /= 2;
                logger.info("learning_rate = " + learningRate);
            }

            // if run unattended, stop after a while
            if (epoch == 100_000) {
                break;
            }
        }
    }

    /**
     * Parameters update with gradient clipping.
     * We used a copy of parameters for computations, but we write to the original.
     */
    private void updateParameters(Parameters.Type type, Map<String, Batch> gradients) {
        for (Map.Entry<String, Batch> entry : gradients.entrySet()) {
            Batch update = entry.getValue().times(-learningRate / minibatchSize).clip(gradientClippingThreashold);

            String parameterName = entry.getKey();
            Variable previousValue = parameters.get(type, parameterName);
            Variable variable = new Variable(parameterName, update.shape(), previousValue.getValue().plus(update));
            parameters.set(type, parameterName, variable);
        }
    }

    private Node toEmbedding(String s) {
        Placeholder placeholder = new Placeholder(new int[]{1, parameters.embeddingSize, 1});
        placeholder.setValue(parameters.embedding(s));
        return placeholder;
    }

    private List<LSTM> buildEncoder(NetworkFactory factory, List<String> tokens) {
        List<LSTM> states = new ArrayList<>();
        LSTM previous = null;
        for (int i = 0; i < tokens.size(); i++) {
            LSTM cell;
            if (i == 0) {
                cell = factory.buildInitialEncoderCell(toEmbedding(tokens.get(i)));
            } else {
                cell = factory.buildEncoderCell(previous, toEmbedding(tokens.get(i)));
            }

            states.add(cell);
            previous = cell;
        }
        return states;
    }

    /**
     * Forward and backward passes for one training example
     */
    private class ForwardBackwardTask implements Callable<Void> {
        private final TrainingExample example;
        private final boolean verbose;
        private final DoubleAdder averageLoss;
        private final Map<String, Batch> encoderGradients; // total
        private final Map<String, Batch> decoderGradients; // total
        private final Map<String, Batch> attentionGradients; // total
        private final Map<String, Batch> embeddingGradients; // total

        ForwardBackwardTask(TrainingExample example,
                            DoubleAdder averageLoss,
                            Map<String, Batch> encoderGradients,
                            Map<String, Batch> decoderGradients,
                            Map<String, Batch> attentionGradients,
                            Map<String, Batch> embeddingGradients,
                            boolean verbose) {
            this.example = example;
            this.averageLoss = averageLoss;
            this.verbose = verbose;
            this.encoderGradients = encoderGradients;
            this.decoderGradients = decoderGradients;
            this.attentionGradients = attentionGradients;
            this.embeddingGradients = embeddingGradients;
        }

        @Override
        public Void call() throws Exception {
            Parameters parametersCopy = parameters.copy(); // shallow
            NetworkFactory detachedFactory = new NetworkFactory(parametersCopy);

            List<String> source = example.getSource();
            List<String> target = example.getTarget();

            List<LSTM> encoder = buildEncoder(detachedFactory, source);

            // attention is inside the decoder
            List<Node> output = new VanillaDecoder(parametersCopy, target).decode(detachedFactory, encoder, verbose, true);

            List<Node> distributions = new ArrayList<>();

            int out_seq_length = output.size();
            Pack pack = new Pack(new int[]{1, out_seq_length, out_seq_length}, distributions);

            int gold_out_seq_length = target.size();

            int maxSeqLength = Math.max(out_seq_length, gold_out_seq_length);
            int max_pad_size = maxSeqLength * maxSeqLength;

            // flatten and pad the linkage prediction
            // [1, max_pad_size, y_hat_flat_size] x [1, y_hat_flat_size, 1] -> [1, max_pad_size, 1]
            int y_hat_flat_length = out_seq_length * out_seq_length;
            Placeholder padder_y_hat = new Placeholder(new int[]{1, max_pad_size, y_hat_flat_length});

            Batch padder_batch = Batch.eye(new int[]{1, max_pad_size, y_hat_flat_length});
            // TODO this is somewhat hackish, make a proper regularization instead
            for (int i = y_hat_flat_length; i < max_pad_size; i++) {
                for (int j = 0; j < y_hat_flat_length; j++) {
                    padder_batch.put(new int[]{0, i, j}, 1.0); // the corresponding gold value will be zero, so the hamming loss will go up
                }
            }
//            padder_batch.print("padder_batch");
            padder_y_hat.setValue(padder_batch);

            Flatten y_hat_flat = new Flatten(new int[]{1, y_hat_flat_length, 1}, pack);
            Node y_hat_padded = new Matmul(new int[]{1, max_pad_size, 1}, padder_y_hat, y_hat_flat);


            // build the gold value for linkage
            Placeholder gold = new Placeholder(new int[]{1, gold_out_seq_length, gold_out_seq_length});
            Batch gold_data = new Batch(new int[]{1, gold_out_seq_length, gold_out_seq_length});

            gold.setValue(gold_data);

            // flatten and pad the gold linkage
            // [1, max_pad_size, gold_out_seq_length * gold_out_seq_length] x [1, gold_out_seq_length * gold_out_seq_length, 1] -> [1, max_pad_size, 1]
            int gold_out_seq_flat_length = gold_out_seq_length * gold_out_seq_length;
            Placeholder padder_gold = new Placeholder(new int[]{1, max_pad_size, gold_out_seq_flat_length});

            // TODO it's hackish, a proper regularization will be much better (but keep the eye padder)
            Batch padder_gold_batch = Batch.eye(new int[]{1, max_pad_size, gold_out_seq_flat_length});
            for (int i = gold_out_seq_flat_length; i < max_pad_size; i++) {
                for (int j = 0; j < gold_out_seq_flat_length; j++) {
                    padder_gold_batch.put(new int[]{0, i, j}, 1.0); // the corresponding predicted value will be zero, so the hamming loss will go up
                }
            }
//            padder_gold_batch.print("padder_gold_batch");
            padder_gold.setValue(padder_gold_batch);

            Flatten gold_flat = new Flatten(new int[]{1, gold_out_seq_flat_length, 1}, gold);
            Node gold_padded = new Matmul(new int[]{1, max_pad_size, 1}, padder_gold, gold_flat);

//            Node loss = new HammingLoss(new int[]{1, 1, 1}, gold_padded, y_hat_padded, regularizationCoefficient, copy);

            // for a cross-entropy loss I also have to turn an input vector into a probability distribution -- TODO this is so hackish
            // TODO a better approach would be to apply some activation fn. (sigmoid, say) element-wise
            Node gold_padded_distribution = new Softmax(new int[]{1, max_pad_size, 1}, gold_padded);
            Node y_hat_padded_distribution = new Softmax(new int[]{1, max_pad_size, 1}, y_hat_padded);
            Node loss = new CrossEntropyLoss(new int[]{1, 1, 1}, gold_padded_distribution, y_hat_padded_distribution);

            // forward
            averageLoss.add(loss.getValue().get(new int[]{0, 0, 0}));

//            new ComputationGraphVisualizer().visualize(loss);
//            ComputationGraphVisualizer.block();


            // backward + collect gradients (compute total)
            for (Variable variable : parametersCopy.getAll(Parameters.Type.DECODER)) {
                Batch sum = decoderGradients.getOrDefault(variable.getName(), new Batch(variable.shape()));
                decoderGradients.put(variable.getName(), sum.plus(variable.getDualValue()));
            }
            for (Variable variable : parametersCopy.getAll(Parameters.Type.ENCODER)) {
                Batch sum = encoderGradients.getOrDefault(variable.getName(), new Batch(variable.shape()));
                encoderGradients.put(variable.getName(), sum.plus(variable.getDualValue()));
            }
            for (Variable variable : parametersCopy.getAll(Parameters.Type.ATTENTION)) {
                Batch sum = attentionGradients.getOrDefault(variable.getName(), new Batch(variable.shape()));
                attentionGradients.put(variable.getName(), sum.plus(variable.getDualValue()));
            }
            for (Variable variable : parametersCopy.getAll(Parameters.Type.EMBEDDING)) {
                Batch sum = embeddingGradients.getOrDefault(variable.getName(), new Batch(variable.shape()));
                embeddingGradients.put(variable.getName(), sum.plus(variable.getDualValue()));
            }

            return null;
        }
    }

}

