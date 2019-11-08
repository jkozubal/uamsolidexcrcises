package speakerrecognition.impl.KMeans;

import speakerrecognition.impl.Matrices;

class LabelsInertia {
    private double[][] data = null;
    private int[] labels = null;
    private double[][] centers = null;
    private double[] distances = null;
    private double[] x_squared_norms = null;
    private double inertia = 0;

    public int[] getLabels() {
        return this.labels.clone();
    }

    public double getInertia() {
        return this.inertia;
    }

    public double[] getDistances() {
        return this.distances.clone();
    }

    LabelsInertia(double[][] X, double[] x_squared_norms, double[][] centers, double[] distances) {
        this.centers = centers;
        this.x_squared_norms = x_squared_norms;
        this.distances = distances;
        this.data = X;

        int n_samples = data.length;
        int[] labels = new int[n_samples];
        labels = Matrices.addValue(labels, -1);

        LabelsPrecomputeDence result = new LabelsPrecomputeDence(this.data, this.x_squared_norms, this.centers, this.distances);
        this.labels = result.getLabels().clone();
        this.inertia = result.getInertia();
        this.distances = result.getDistances().clone();

    }
}
