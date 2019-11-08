package speakerrecognition.SpeakerInterfaces;

import speakerrecognition.impl.MyException;
import speakerrecognition.impl.SpeakerModel;

public interface LogProbability {
    double getLogProbabilityOfDataUnderModel(SpeakerModel model, double[][] dataToBeTested) throws MyException;
}
