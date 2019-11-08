package speakerrecognition.impl.SpeakerImplementations;

import speakerrecognition.impl.GMM.GMM;

public class ClustersGMMImp {
    public double[][] getMeansOfClustersFor2DdataByGMM(double[][] data, int numOfClusters) {
        GMM gmm = new GMM(data, numOfClusters);
        gmm.fit();
        return gmm.get_means();
    }
}
