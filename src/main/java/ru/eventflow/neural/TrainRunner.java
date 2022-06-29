package ru.eventflow.neural;

import org.apache.commons.cli.*;
import ru.eventflow.neural.dataset.TrainingExample;
import ru.eventflow.neural.rnn.EncoderDecoderEngine;
import ru.eventflow.neural.rnn.Parameters;

import java.io.*;
import java.util.*;

public class TrainRunner {

    public static void main(String[] args) throws IOException {
        CommandLineParser parser = new DefaultParser();

        Options options = new Options();
        options.addOption("i", "input", true, "train set in JSON format");
        options.addOption("r", "learning", true, "learning rate (double)");
        options.addOption("s", "snapshots", true, "directory for snapshots");
        options.addOption("h", "hidden", true, "hidden size");
        options.addOption("e", "embedding", true, "embedding (input) vector size, don't confuse with one-hot vector size");
        options.addOption("t", "threads", true, "number of threads");
        options.addOption("c", "clipping", true, "gradient clipping threshold");

        try {
            CommandLine line = parser.parse(options, args);
            if (line.hasOption("input") && line.hasOption("learning") &&
                    line.hasOption("snapshots") && line.hasOption("hidden") && line.hasOption("hidden")
                    && line.hasOption("threads") && line.hasOption("clipping")) {
                final File input = new File(line.getOptionValue("input"));
                if (!input.exists() && !input.isFile() && !input.canRead()) {
                    System.out.println("Unable to read the train set from a file.");
                    System.exit(-1);
                }

                final File snapshots = new File(line.getOptionValue("snapshots"));
                if (!snapshots.exists()) {
                    snapshots.mkdirs();
                }

                int embeddingSize = Integer.parseInt(line.getOptionValue("embedding"));
                int hiddenSize = Integer.parseInt(line.getOptionValue("hidden"));
                int numThreads = Integer.parseInt(line.getOptionValue("threads"));
                double learningRate = Double.parseDouble(line.getOptionValue("learning"));
                double clipping = Double.parseDouble(line.getOptionValue("clipping"));
                run(learningRate, input, embeddingSize, hiddenSize, numThreads, clipping, snapshots);

            } else {
                HelpFormatter formatter = new HelpFormatter();
                formatter.printHelp("neural.jar", options);
            }
        } catch (ParseException e) {
            System.out.println("Unexpected exception:" + e.getMessage());
        }
    }

    private static void run(double learningRate, File input, int embeddingSize, int hiddenSize, int numThreads,
                            double clipping, File snapshotsDirectory) throws IOException {

        List<String> tokens = Arrays.asList("[", "]", "{", "}", "(", ")", "<", ">", Parameters.EOS);

        List<TrainingExample> examples = new ArrayList<>();
        String line;
        BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(input)));
        while ((line = reader.readLine()) != null) {
            String[] chunks = line.split(" ");
            TrainingExample example = new TrainingExample(Arrays.asList(chunks), Arrays.asList("[", "]"));
            examples.add(example);
        }

        Set<TrainingExample> trainSet = new HashSet<>(examples);

        int minibatchSize = 10;
        double regularizationCoefficient = 0.001;

        EncoderDecoderEngine model = new EncoderDecoderEngine(minibatchSize, learningRate, embeddingSize, hiddenSize,
                numThreads, clipping, regularizationCoefficient, snapshotsDirectory.getCanonicalPath(), tokens);
        model.train(trainSet);

    }

}
