package speakerrecognition.SpeakerInterfaces;

import java.io.IOException;
import java.util.List;

import speakerrecognition.impl.MyException;
import speakerrecognition.impl.SpeakerModel;

public interface SpeakerRecognition {
	
	String recognize(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath) throws IOException, MyException;
	void printLogProbsForRecognition(List<SpeakerModel> speakerModels, String resourceSoundSpeechFilePath) throws IOException, MyException;

}
