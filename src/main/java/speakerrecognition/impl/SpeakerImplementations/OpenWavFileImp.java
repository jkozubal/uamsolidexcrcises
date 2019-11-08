package speakerrecognition.impl.SpeakerImplementations;


import speakerrecognition.impl.WavFile;

import java.io.IOException;

public class OpenWavFileImp {
    public int[] openWavFile(String resourcePath) throws IOException {
        WavFile wavFile = new WavFile(resourcePath);
        wavFile.open();
        return wavFile.getSamples();
    }
}
