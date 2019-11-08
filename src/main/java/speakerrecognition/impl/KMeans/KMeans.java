package speakerrecognition.impl.KMeans;


import speakerrecognition.impl.Matrices;
import speakerrecognition.impl.Statistics;

import java.util.Arrays;

public class KMeans {
	private int numOfClusters;
	private int numOfRows;
	private int numOfCols;
	private double[][] data;
	private int n_iter = 0;
	private int n_init = 10;
	private int max_iter = 300;
	private double tol = 0.0001;
	
	// output parameters////
	private double[][] best_cluster_centers = null;
	private int[] best_labels = null;
	private double best_inertia = Double.MAX_VALUE;
	private int n_iter_ = 0;
	/////////////////////////
	
	
	public KMeans(double[][] x, int numOfClust){
		this.numOfClusters = numOfClust;
		//int numOfClusters = this.numOfCols;
		int n_init = 10;
		int max_iter = 300;
		double tol = 0.0001;
		this.tol = tolerance(x, tol);
		this.numOfRows = x.length;
		this.numOfCols = x[0].length;
		this.data = Matrices.copy2dArray(x);
		this.best_cluster_centers = new double[numOfClust][x[0].length];
		this.best_labels = new int[x.length];
		
	}
	
	public void fit(){
		double[][] cluster_centers = null;
		int[] labels = null;
		double inertia = 0;
		//int n_iter = 0;
		//double[] result = null;
		//double[][] centers = new double[this.numOfClusters][this.numOfCols];
		
		try{
		
			////////// substracting mean //////////////
			double[] X_mean = Statistics.getMean(data);
			for(int i=0;i<this.numOfRows;i++){
				for(int j=0; j<this.numOfCols; j++){
					data[i][j] -= X_mean[j];
				}
			}
			
			
			
			////////// numpy einsum //////////////
			double[] x_squared_norms = Matrices.einsum(data);
			
			for(int i=0;i<this.n_init;i++){
				KMeansSingle kmeans_single = new KMeansSingle (this.data, this.numOfClusters, x_squared_norms, this.max_iter, this.tol, numOfRows, numOfCols);
				cluster_centers = kmeans_single.get_best_centers().clone();
				inertia = kmeans_single.get_best_inertia();
				labels = kmeans_single.get_best_labels().clone();
				
				if (inertia<this.best_inertia){
					this.best_labels = labels.clone();
					this.best_inertia = inertia;
					this.best_cluster_centers = cluster_centers.clone();
					
				}
				
			}
			
			this.best_cluster_centers = Matrices.addValue(this.best_cluster_centers, X_mean);
		}
		catch(Exception myEx)
        {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
		
		
		
		//System.out.println("kmeans end");
		//return result;
	
	}
	
	public double[][] get_centers(){
		return this.best_cluster_centers;
	}
	
	
	
	static double[][] centers_dense(double[][] X, int[] labels, int n_clusters, double[] distances){
		double[][] result = new double[n_clusters][X[0].length];
		for(int j=0;j<X[0].length;j++){
			double[] sum = new double[n_clusters];
			for(int k=0;k<n_clusters;k++){
				int samples_num = 0;
				for(int z=0;z<labels.length;z++){
					if(labels[z]==k){
						sum[k] += X[z][j];
						samples_num += 1;
					}
				}
				sum[k] /= samples_num;
				
			}
			for(int i=0;i<n_clusters;i++)
				result[i][j] = sum[i];
		}
		return result;
		
	}
	
	static double[][] init_centroids(double[][] data, int n_clusters, double[] x_sq_norms ,int numOfRows, int numOfCols){
		double centers[][] = new double[n_clusters][numOfCols];
		
		try{
			int n_local_trials = 2+(int)(Math.log(n_clusters));
			/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/int center_id = (int) Math.floor(Math.random() * numOfRows);
			for(int i=0;i<numOfCols;i++){
				centers[0][i] = data[center_id][i];
			}
			double[] closest_dist_sq =  Matrices.euclidean_distances(centers[0], data, x_sq_norms);
			double current_pot = Matrices.sum(closest_dist_sq);
			
			for(int c=1; c<n_clusters;c++){
				/*!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*/ double[] rand_vals = Matrices.genRandMatrix(current_pot, n_local_trials);
				double[] closest_dist_sq_cumsum = Matrices.cumsum(closest_dist_sq);
				int[] candidate_ids = Matrices.searchsorted(closest_dist_sq_cumsum, rand_vals);
				double[][] data_candidates = new double[n_local_trials][numOfCols];
				
				for(int z=0;z<n_local_trials;z++){
					for(int j=0;j<numOfCols;j++){
						data_candidates[z][j] = data[candidate_ids[z]][j];
					}
				}
					
				int best_candidate = -1;
				double best_pot = 99999999;
				double[] best_dist_sq = null;
	
				double[][] distance_to_candidates = Matrices.euclidean_distances(data_candidates, data, x_sq_norms);
				
				for(int trial=0;trial<n_local_trials;trial++){
					double[] new_dist_sq = Matrices.minimum(closest_dist_sq, Matrices.select_row(distance_to_candidates, trial));
					double new_pot = Matrices.sum(new_dist_sq);
					
					if(best_candidate==-1 | new_pot < best_pot){
						best_candidate = candidate_ids[trial];
						best_pot = new_pot;
						best_dist_sq = Arrays.copyOf(new_dist_sq, new_dist_sq.length);
					}
				}
				double[] center_temp = Arrays.copyOf(data[best_candidate], data[best_candidate].length);
				for(int ii=0;ii<centers[0].length;ii++){
					centers[c][ii] = center_temp[ii];
				}
				current_pot = best_pot;
				closest_dist_sq = Arrays.copyOf(best_dist_sq, best_dist_sq.length);
				//System.out.println("temp");
				
			}
		}
		catch(Exception myEx)
        {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }
		
		return centers;
		
	}
	
	
	private double tolerance(double[][] x, double tol){
		
		double temp[] = Statistics.getVariance(x);
		
		for(int i=0;i<temp.length;i++){
			temp[i] = temp[i]*tol;
		}
		return Statistics.getMean(temp);
	}

		
		public static class LabelsPrecomputeDence{

		}
}
