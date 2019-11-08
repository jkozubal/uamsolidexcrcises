package speakerrecognition.impl.SpeakerImplementations;

import speakerrecognition.impl.MFCC;

public class MFCCImp {
    public double[][] computeMFCC(int[] soundSamples, int fs) {
        MFCC mfcc = new MFCC(soundSamples, fs);
        double[][] speaker_mfcc = mfcc.getMFCC();
        return speaker_mfcc;
    }
}
