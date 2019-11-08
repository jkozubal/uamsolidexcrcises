package speakerrecognition.SpeakerInterfaces;

public interface ClustersGMM {
    double[][] getMeansOfClustersFor2DdataByGMM(double[][] data, int numOfClusters);
}
