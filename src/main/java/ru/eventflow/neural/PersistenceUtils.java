package ru.eventflow.neural;

import org.nustaq.serialization.FSTObjectInput;
import org.nustaq.serialization.FSTObjectOutput;
import ru.eventflow.neural.rnn.Parameters;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * s. https://github.com/RuedigerMoeller/fast-serialization/wiki/Serialization
 */
public class PersistenceUtils {

    public static Parameters read(InputStream stream) throws IOException, ClassNotFoundException {
        FSTObjectInput in = new FSTObjectInput(stream);
        Parameters result = (Parameters) in.readObject();
        in.close(); // required !
        return result;
    }

    public static void write(OutputStream stream, Parameters object) throws IOException {
        FSTObjectOutput out = new FSTObjectOutput(stream);
        out.writeObject(object);
        out.close(); // required !
    }

}
