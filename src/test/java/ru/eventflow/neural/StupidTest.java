package ru.eventflow.neural;

import org.junit.Test;

public class StupidTest {

    @Test
    public void testMinibatchSelection() {
        int sampleSize = 10;
        int populationSize = 100;

        int[] sample = new int[sampleSize];

        for (int epoch = 0; epoch < 1000; epoch++) {
            System.out.println("I = " + epoch);

            for (int i = 0; i < sampleSize; i++) {
                sample[i] = (i + epoch * sampleSize) % populationSize;
                System.out.print(" " + sample[i]);
            }

            System.out.println();
        }
    }
}
