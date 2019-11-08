package speakerrecognition.impl.SpeakerImplementations;

import speakerrecognition.impl.KMeans.KMeans;

public class ClustersKMeansImp {
    public double[][] getMeansOfClustersFor2DdataByKMeans(double[][] data, int numOfClusters) {
        KMeans kMeans = new KMeans(data, numOfClusters);
        kMeans.fit();
        return kMeans.get_centers();
    }
}
