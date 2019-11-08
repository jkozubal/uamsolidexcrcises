package speakerrecognition.SpeakerInterfaces;

public interface ClustersKMeans {
    double[][] getMeansOfClustersFor2DdataByKMeans(double[][] data, int numOfClusters);
}
