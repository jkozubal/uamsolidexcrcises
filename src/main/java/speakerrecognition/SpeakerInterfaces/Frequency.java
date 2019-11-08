package speakerrecognition.SpeakerInterfaces;

import speakerrecognition.impl.MyException;
import java.io.IOException;

public interface Frequency {
    public int getSamplingFrequency(String resourceSOundFilePath) throws IOException, MyException;
}
