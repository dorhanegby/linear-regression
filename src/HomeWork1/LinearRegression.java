package HomeWork1;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Capabilities;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.Filter;

public class LinearRegression implements Classifier {

	private static final double INIT_VALUE = 1.0 / 14.0;
	private static final int MAX_ITERATIONS = 20000;
	
    private int m_ClassIndex;
	private int m_truNumAttributes;
	private double[] m_coefficients;
	private double m_alpha;
	
	//the method which runs to train the linear regression predictor, i.e.
	//finds its weights.
	@Override
	public void buildClassifier(Instances trainingData) throws Exception {

		m_ClassIndex = trainingData.classIndex();
		Instances norm_data = normalize(trainingData);
		trainingData.setClassIndex(m_ClassIndex);
		findAlpha(norm_data);
		m_coefficients = gradientDescent(norm_data, MAX_ITERATIONS, initCoefficients());
		
	}
	// TODO: Maybe we should implement it on our own
	private Instances normalize(Instances trainingData) throws Exception {
		Normalize normalize = new Normalize();
		normalize.setInputFormat(trainingData);
		return Filter.useFilter(trainingData, normalize);
	}

	private void findAlpha(Instances data) throws Exception {
		double[] errosByAlpha = new double[17];
		double error = 0;
		double prev_error = 0;
		for (int i=17;i>0;i--) {
			m_alpha = Math.pow(3, -i);
			double[] coefficients = initCoefficients();
			for(int j=1;j<200;j++) {
				coefficients = gradientDescent(data, 100, coefficients);
				if(j == 1) {
					error = calculateMSE(data, coefficients);
				}
				else {
					prev_error = error;
					error = calculateMSE(data, coefficients);
					if(prev_error < error) {
						error = prev_error;
						break;
					}
				}
			}
			errosByAlpha[i - 1] = error;
		}
		int minIndex = 16;
		double minValue = errosByAlpha[16];

		for(int i = 15; i>= 0;i--) {
			if(errosByAlpha[i] < minValue) {
				minIndex = i;
				minValue = errosByAlpha[i];
			}
		}

		m_alpha = Math.pow(3, -(minIndex + 1));
	}

	private double[] initCoefficients() {
		double[] coefficients = new double[m_ClassIndex + 1];
		for (int i = 0; i < coefficients.length; i++) {
			coefficients[i] = INIT_VALUE;
		}

		return coefficients;
	}
	
	/**
	 * An implementation of the gradient descent algorithm which should
	 * return the weights of a linear regression predictor which minimizes
	 * the average squared error.
     * 
	 * @param trainingData
	 * @throws Exception
	 */
	private double[] gradientDescent(Instances trainingData, int stopCondition, double[] start_coefficients)
			throws Exception {
		double[] coefficients = start_coefficients;
		double[] temp_coefficients = coefficients.clone();
		double error = Double.MAX_VALUE;
		double prev_error = Double.MAX_VALUE;
		int iterations = 0;
		while ((iterations < 100 || prev_error - error > 0.003) && stopCondition > 0) {
			stopCondition--;
			iterations++;
			if(iterations == 100) {
				error = calculateMSE(trainingData, coefficients);
			}
			else if(iterations % 100 == 0) {
				prev_error = error;
				error = calculateMSE(trainingData, coefficients);
			}
			temp_coefficients[0] = coefficients[0] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, 0);
			for (int j = 1; j <= m_ClassIndex; j++) { // Updating thetas
				temp_coefficients[j] = coefficients[j] - m_alpha * 1 / m_ClassIndex * sumOfDistances(coefficients, trainingData, j);
			}
			coefficients = temp_coefficients.clone();
		}

		return coefficients;
	}


	private double sumOfDistances(double[] coefficients, Instances trainingData, int indexToUpdate) {
		double sum = 0;
		for(int i=0;i<trainingData.numInstances();i++) { // Sigma
			Instance dataRow = trainingData.instance(i);
			double partial_sum = prediction(coefficients, dataRow);
			double actual = getActual(dataRow);
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
	public double calculateMSE(Instances testData, double[] coefficients) throws Exception {
		double constant = 1.0 / 2.0 * testData.numInstances();
		double sum = 0;
		for(int i=0;i<testData.numInstances();i++) {
			Instance dataRow = testData.instance(i);
			double actual = getActual(dataRow);
			sum += Math.pow(prediction(coefficients, dataRow) - actual, 2);
		}
		return constant * sum;
	}

	private double getActual(Instance instance) {
		return instance.value(instance.numAttributes() - 1);
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
