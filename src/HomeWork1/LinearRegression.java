package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;

public class LinearRegression implements Classifier {

	private static final double INIT_VALUE = 1.0 / 14.0;
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha = Math.pow(3, -14);
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		// TODO: normilize the data
		trainingData.setClassIndex(m_ClassIndex);
		m_coefficients = gradientDescent(trainingData);
		
	}
	
	private void findAlpha(Instances data) throws Exception {
		
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData)
			throws Exception {
		double [] coefficients = new double[m_ClassIndex + 1];
		// Sets all values to INIT_VALUE as a guess
		for(int i = 0;i < coefficients.length;i++) {
			coefficients[i] = INIT_VALUE;
		}

		for(int i =0;i<100000; i++) { // TODO: define a stop condition
			double [] temp_coefficients = coefficients.clone();
			temp_coefficients[0] = coefficients[0] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, 0);

			for(int j=1;j<m_ClassIndex;j++) { // Updating thetas
				double fix = coefficients[j] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, j);
				System.out.println(fix);
				temp_coefficients[j] = fix;
			}
		}

		return coefficients;
	}


	private double sumOfDistances(double[] coefficients, Instances trainingData, int indexToUpdate) {
		double sum = 0;
		for(int i=0;i<trainingData.numInstances();i++) { // Sigma
			Instance dataRow = trainingData.instance(i);
			double partial_sum = 0;
			for(int j=1; j< dataRow.numAttributes(); j++) { // parentazis
				partial_sum += coefficients[j] * dataRow.value(j - 1);
			}
			partial_sum += coefficients[0] - dataRow.value(dataRow.numAttributes() - 1); // add constant and substract actual
			if(indexToUpdate != 0) {
				partial_sum *= dataRow.value(indexToUpdate - 1); // multiply by inner derviative
			}
			sum += partial_sum;
		}

		return sum;
	}

	/**
	 * Returns the prediction of a linear regression predictor with weights
	 * given by m_coefficients on a single instance.
     *
	 * @param instance
	 * @return
	 * @throws Exception
	 */
	public double regressionPrediction(Instance instance) throws Exception {
		return 0;
	}
	
	/**
	 * Calculates the total squared error over the data on a linear regression
	 * predictor with weights given by m_coefficients.
     *
	 * @param testData
	 * @return
	 * @throws Exception
	 */
	public double calculateMSE(Instances testData) throws Exception {

		return 0;
	}
    
    @Override
	public double classifyInstance(Instance arg0) throws Exception {
		// Don't change
		return 0;
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// Don't change
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// Don't change
		return null;
	}
}
