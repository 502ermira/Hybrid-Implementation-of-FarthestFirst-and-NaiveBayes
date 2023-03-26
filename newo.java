package Project4;


import weka.clusterers.ClusterEvaluation;
import weka.clusterers.FarthestFirst;
import weka.clusterers.SimpleKMeans;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class newo {
    
    public static void main(String[] args) throws Exception {
        // Load dataset
        DataSource source = new DataSource("C:\\Users\\Admin\\Desktop\\fina.arff");
        Instances data = source.getDataSet();
        
        // Cluster data using FarthestFirst
        FarthestFirst ff = new FarthestFirst();
        ff.setNumClusters(220);
        ff.buildClusterer(data);
        // Evaluate clusterer
        ClusterEvaluation eva = new ClusterEvaluation();
        eva.setClusterer(ff);
        eva.evaluateClusterer(data);

        // Print results
        System.out.println("Cluster results:");
        System.out.println(eva.clusterResultsToString());
        
        // Assign each instance to the closest cluster centroid
        Instances clusteredData = new Instances(data, data.numInstances());
        for (int i = 0; i < data.numInstances(); i++) {
            Instance inst = data.instance(i);
            int clusterIndex = ff.clusterInstance(inst);
            clusteredData.add(ff.getClusterCentroids().instance(clusterIndex));
        }
        
        // Set class attribute index
        clusteredData.setClassIndex(clusteredData.numAttributes() - 1);
        
        // Train and evaluate the NaiveBayes classifier
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(clusteredData);
        
        Evaluation eval = new Evaluation(clusteredData);
        eval.evaluateModel(nb, clusteredData);
        
        // Output evaluation results
        System.out.println(eval.toSummaryString());
        
        // Print evaluation results for each class
        System.out.println(eval.toClassDetailsString());
        
        // Print confusion matrix
        System.out.println(eval.toMatrixString());
    }
}