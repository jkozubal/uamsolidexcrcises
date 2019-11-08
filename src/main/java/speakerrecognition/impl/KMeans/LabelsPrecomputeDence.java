package speakerrecognition.impl.KMeans;

import speakerrecognition.impl.Matrices;

public class LabelsPrecomputeDence {
    private double[][] data = null;
    private int[] labels = null;
    private double[][] centers = null;
    private double[] distances = null;
    private double[] x_squared_norms = null;
    private double inertia = 0;

    LabelsPrecomputeDence(double[][] X, double[] x_squared_norms, double[][] centers, double[] distances){
        this.centers = centers;
        this.x_squared_norms = x_squared_norms;
        this.distances = distances; ////////// huston, problem - k_means.py line 490, niejawne zwracanie
        this.data = X;

        int n_samples = data.length;
        int k = centers.length;
        double[][] all_distances = Matrices.euclidean_distances(centers, X, x_squared_norms);
        this.labels = new int[n_samples];
        this.labels = Matrices.addValue(this.labels, -1);
        double[] mindist = new double[n_samples];
        mindist = Matrices.addValue(mindist, Double.POSITIVE_INFINITY);

        for(int center_id=0;center_id<k;center_id++){
            double[] dist = all_distances[center_id];
            for(int i=0;i<labels.length;i++){
                if(dist[i]<mindist[i]){
                    this.labels[i] = center_id;
                }
                mindist[i]=Math.min(dist[i], mindist[i]);
            }
        }
        if(n_samples == this.distances.length)
            this.distances = mindist;
        this.inertia = Matrices.sum(mindist);
    }
    int[] getLabels() {
        return labels;
    }

    double[] getDistances() {
        return distances;
    }

    double getInertia() {
        return inertia;
    }
}
