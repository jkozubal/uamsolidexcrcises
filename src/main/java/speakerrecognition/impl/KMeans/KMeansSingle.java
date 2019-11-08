package speakerrecognition.impl.KMeans;

import speakerrecognition.impl.Matrices;

import static speakerrecognition.impl.KMeans.KMeans.centers_dense;

class KMeansSingle {
    private int[] best_labels = null;
    private double[][] best_centers = null;
    private double best_inertia = Double.MAX_VALUE;
    private double[] distances = null;

    KMeansSingle(double[][] data, int n_clusters, double[] x_sq_norms, int max_iter, double tol, int numOfRows, int numOfCols){

        try{

            double[][] centers = KMeans.init_centroids(data, n_clusters, x_sq_norms, numOfRows, numOfCols);
            this.distances = new double[data.length];

            for(int i=0; i<max_iter;i++){
                double[][] centers_old = centers.clone();
                LabelsInertia labelsInertia = new LabelsInertia(data, x_sq_norms, centers, this.distances);
                int[] labels = labelsInertia.getLabels().clone();
                double inertia = labelsInertia.getInertia();
                this.distances = labelsInertia.getDistances().clone();

                centers = centers_dense(data, labels, n_clusters, distances);

                if (inertia<best_inertia){
                    this.best_labels = labels.clone();
                    this.best_centers = centers.clone();
                    this.best_inertia = inertia;
                }

                if (Matrices.squared_norm(Matrices.substractMatrixes(centers_old, centers))<=tol)
                    break;

            }
        }
        catch(Exception myEx)
        {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
    }

    public int[] get_best_labels(){
        return this.best_labels;
    }

    public double[][] get_best_centers(){
        return this.best_centers;
    }

    public double get_best_inertia(){
        return this.best_inertia;
    }
}
