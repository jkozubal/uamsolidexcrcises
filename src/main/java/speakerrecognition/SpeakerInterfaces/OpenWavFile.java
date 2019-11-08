package speakerrecognition.SpeakerInterfaces;

import speakerrecognition.impl.MyException;
import java.io.IOException;

public interface OpenWavFile {
    public int[] openWavFile(String resourcePath) throws IOException, MyException;
}
