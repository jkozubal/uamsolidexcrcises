package speakerrecognition.impl;


public final class Statistics {
	
	public static double getMean(double[] data)
    {
        double sum = 0.0;
        for(double a : data)
            sum += a;
        return sum/data.length;
    }

	public static double[] getMean(double[][] data)
	{
		int numOfRows = data.length;
		int numOfCols = data[0].length;

	    double sum[] = new double[numOfCols];
	    for(int j=0;j<numOfCols;j++){
	    	for(int i=0;i<numOfRows;i++){
	    		//System.out.println(Double.toString(data[i][j]));
	    		sum[j] += data[i][j];
	    	}
	    	sum[j] /= numOfRows;
	    }
	    //System.out.println("sumaaa");
	    return sum;
	}

	public static double[] getVariance(double[][] data)
	{
		int numOfRows = data.length;
		int numOfCols = data[0].length;
		
	    double means[] = Statistics.getMean(data);
	    double[] temp = new double[numOfCols];
	    
	    for(int j=0;j<numOfCols;j++){
	    	for(int i=0;i<numOfRows;i++){
	    		temp[j] += Math.pow((data[i][j]-means[j]), 2);
	    	}
	    	temp[j] /= numOfRows;
	    }
	    
	    return temp;
	}

}
