package HomeWork1;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class MainHW1 {
	
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}
		
	/**
	 * Sets the class index as the last attribute.
	 * @param fileName
	 * @return Instances data
	 * @throws IOException
	 */
	public static Instances loadData(String fileName) throws IOException{
		BufferedReader datafile = readDataFile(fileName);

		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}
	
	public static void main(String[] args) throws Exception {
		TreeSet<String> set = new TreeSet<>();
		HashMap<String, Double> results = new HashMap<>();
		HashMap<String, String> nameByIndex = new HashMap<>();
		Instances newTestData;
		// load data
		Instances trainingData = loadData("./src/Data/wind_training.txt");
		Instances testingData = loadData("./src/Data/wind_testing.txt");
		// find best alpha and build classifier with all attributes
		LinearRegression lr = new LinearRegression();
		lr.buildClassifier(trainingData);
		System.out.println("Training error with all features: " + lr.calculateMSE(trainingData));
		System.out.println("Test error with all features: " + lr.calculateMSE(testingData));

		// build classifiers with all 3 attributes combinations
		for(int i=1;i<trainingData.numAttributes();i++) {
			for (int j = 1; j < trainingData.numAttributes(); j++) {
				for (int k = 1; k < trainingData.numAttributes(); k++) {

					// Avoiding duplicates
					int[] sortedPermutation = {i, j, k};
					Arrays.sort(sortedPermutation);
					String featuresToKeep = sortedPermutation[0] + "," + sortedPermutation[1] + "," + sortedPermutation[2];
					if (i == j || i == k || j == k || set.contains(featuresToKeep)) {
						continue;
					}

					String nameOfFeatures = trainingData.attribute(i - 1).name() + "," + trainingData.attribute(j - 1).name() + "," + trainingData.attribute(k - 1).name();
					nameByIndex.put(featuresToKeep, nameOfFeatures);
					System.out.println("Testing features: " + nameOfFeatures);
					set.add(featuresToKeep);

					// Reducing data
					Instances newTrainingData = removeData(trainingData, featuresToKeep);

					// Training
					lr.buildClassifier(newTrainingData);
					double error = lr.calculateMSE(newTrainingData);
					System.out.println(featuresToKeep + " - training error: " + error);
					results.put(featuresToKeep, error);
				}
			}
		}

		double min = Double.MAX_VALUE;
		String index = "";
		for(HashMap.Entry<String, Double> entry : results.entrySet()) {
			if(entry.getValue() < min) {
				min = entry.getValue();
				index = entry.getKey();
			}
		}

		System.out.println("Training error with best features " + nameByIndex.get(index) + " with error of: "+ min);
		newTestData = removeData(testingData, index);
		System.out.println("Test error with best features " + nameByIndex.get(index) + " with error of: " + lr.calculateMSE(newTestData));
	}

	private static Instances removeData(Instances data, String featuresToKeep) throws Exception {
		Remove remove = new Remove();
		remove.setAttributeIndices(featuresToKeep + "," + data.numAttributes());
		remove.setInvertSelection(true);
		remove.setInputFormat(data);
		return Filter.useFilter(data, remove);
	}

}
