package speakerrecognition.SpeakerInterfaces;

public interface MFCC {
    double[][] computeMFCC(int[] soundSamples, int fs);
}
