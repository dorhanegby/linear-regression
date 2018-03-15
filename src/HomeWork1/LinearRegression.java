package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.Filter;

public class LinearRegression implements Classifier {

	private static final double INIT_VALUE = 1.0 / 14.0;
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha = Math.pow(3, -4);
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {
		m_ClassIndex = trainingData.classIndex();
		Instances norm_data = normalize(trainingData);
		trainingData.setClassIndex(m_ClassIndex);
		m_coefficients = gradientDescent(norm_data);
		
	}

	private Instances normalize(Instances trainingData) throws Exception {
		Normalize normalize = new Normalize();
		normalize.setInputFormat(trainingData);
		return Filter.useFilter(trainingData, normalize);
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
		double[] coefficients = new double[m_ClassIndex + 1];
		// Sets all values to INIT_VALUE as a guess
		for (int i = 0; i < coefficients.length; i++) {
			coefficients[i] = INIT_VALUE;
		}
		double[] temp_coefficients = coefficients.clone();
		double sum = -1;
		while (sum != 0) { // TODO: define a stop condition
			sum = 0;
			temp_coefficients[0] = coefficients[0] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, 0);
			sum += Math.pow((temp_coefficients[0] - coefficients[0]), 2);
			for (int j = 1; j <= m_ClassIndex; j++) { // Updating thetas
				temp_coefficients[j] = coefficients[j] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, j);
				sum += Math.pow((temp_coefficients[0] - coefficients[0]), 2);
			}
			System.out.println(sum);
			coefficients = temp_coefficients.clone();
		}

		for (int j = 0; j < coefficients.length; j++)
			System.out.println("theta " + j + ": " + temp_coefficients[j]);

		return coefficients;
	}


	private double sumOfDistances(double[] coefficients, Instances trainingData, int indexToUpdate) {
		double sum = 0;
		for(int i=0;i<trainingData.numInstances();i++) { // Sigma
			Instance dataRow = trainingData.instance(i);
			double partial_sum = prediction(coefficients, dataRow);
			double actual = dataRow.value(dataRow.numAttributes() - 1);
			partial_sum -=  actual;
			if(indexToUpdate != 0) {
				partial_sum *= dataRow.value(indexToUpdate - 1); // multiply by inner derviative
			}
			sum += partial_sum;
		}
		return sum;
	}

	private double prediction(double[] coefficients, Instance instance) {
		double partial_sum = 0;
		for(int j=1; j< instance.numAttributes(); j++) {
			partial_sum += coefficients[j] * instance.value(j - 1);
		}
		partial_sum += coefficients[0];

		return partial_sum;
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
		return prediction(m_coefficients, instance);
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
