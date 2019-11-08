package speakerrecognition.impl.SpeakerImplementations;

import speakerrecognition.impl.MyException;
import speakerrecognition.impl.WavFile;

import java.io.IOException;

public class FrequencyImp {

    public int getSamplingFrequency(String resourceSOundFilePath) throws IOException, MyException {
        WavFile wavFile = new WavFile(resourceSOundFilePath);
        wavFile.open();
        return wavFile.getFs();
    }
}
