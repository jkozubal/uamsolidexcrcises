package speakerrecognition.impl.SpeakerImplementations;

import speakerrecognition.impl.MyException;
import speakerrecognition.impl.SpeakerModel;

public class LogProbabilityImp {
    public double getLogProbabilityOfDataUnderModel(SpeakerModel model, double[][] dataToBeTested) throws MyException {
        return model.getScore(dataToBeTested);
    }
}
