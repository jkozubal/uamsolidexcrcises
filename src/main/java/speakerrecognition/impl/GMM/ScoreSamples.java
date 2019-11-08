package speakerrecognition.impl.GMM;

import speakerrecognition.impl.Matrices;

class ScoreSamples{
    private double[][] data = null;
    private double[] log_likelihoods = null;
    private double[][] means = null;
    private double[][] covars = null;
    private double[] weights = null;
    /////out matrixes////
    private double[] logprob = null;
    private double[][] responsibilities = null;
    /////////////////////


    ScoreSamples(double[][] X, double[][] means, double[][] covars, double[] weights, int numOfComponents){
        this.data = X;
        this.log_likelihoods = new double[X.length];
        this.responsibilities = new double[X.length][numOfComponents];
        this.means = means;
        this.covars = covars;
        this.weights = weights;


        try{
            double[][] lpr = log_multivariate_normal_density(this.data, this.means, this.covars);
            lpr = Matrices.addValue(lpr, Matrices.makeLog(this.weights));
            this.logprob = Matrices.logsumexp(lpr);
            // gmm.py line 321
            this.responsibilities = Matrices.exp(Matrices.substractValue(lpr, logprob));
        }
        catch(Exception myEx)
        {
            //System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }

    }

    public double[] getLogprob(){
        return this.logprob;
    }

    public double[][] getResponsibilities(){
        return this.responsibilities;
    }

    private double[][] log_multivariate_normal_density(double[][] data, double[][] means, double[][] covars){
        //diagonal type
        double[][] lpr = new double[data.length][means.length];
        int n_samples = data.length;
        int n_dim = data[0].length;

        try{
            double[] sumLogCov = Matrices.sum(Matrices.makeLog(covars), 1); //np.sum(np.log(covars), 1)
            double[] sumDivMeanCov = Matrices.sum(Matrices.divideElements(Matrices.power(this.means, 2), this.covars),1); //np.sum((means ** 2) / covars, 1)
            double[][] dotXdivMeanCovT = Matrices.multiplyByValue(Matrices.multiplyByMatrix(data, Matrices.transpose(Matrices.divideElements(means, covars))), -2); //- 2 * np.dot(X, (means / covars).T)
            double[][] dotXdivOneCovT = Matrices.multiplyByMatrix(Matrices.power(data,  2), Matrices.transpose(Matrices.invertElements(covars)));


            sumLogCov = Matrices.addValue(sumLogCov,n_dim * Math.log(2*Math.PI)); //n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
            sumDivMeanCov = Matrices.addMatrixes(sumDivMeanCov, sumLogCov); // n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1)
            dotXdivOneCovT = Matrices.sum(dotXdivOneCovT, dotXdivMeanCovT); //- 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T)
            dotXdivOneCovT = Matrices.addValue(dotXdivOneCovT, sumDivMeanCov); // (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1) + np.sum((means ** 2) / covars, 1) - 2 * np.dot(X, (means / covars).T) + np.dot(X ** 2, (1.0 / covars).T))
            lpr = Matrices.multiplyByValue(dotXdivOneCovT, -0.5);
        }
        catch(Exception myEx)
        {
            System.out.println("An exception encourred: " + myEx.getMessage());
            myEx.printStackTrace();
            System.exit(1);
        }

        return lpr;
    }
}
