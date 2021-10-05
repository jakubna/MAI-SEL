Documentation/report.pdf            	Written report about the used algorithms, evaluation results and instructions on how to execute the code.
Source/decisionforest.py               Python class. Contains implementation of the Decision Forest classifier.
Source/randomforest.py 		Python class. Contains implementation of the Random Forest classifier.
Source/main.py                     	Python script. Main program used to generate experiments. 
					Example usage: python main.py -a df - d small -t 0.2
					'-a', '--algorithm' - Decision forest [df] or random forest [rf] algorithm. Possible values: [df, rf]
					'-d', '--dataset_size' - Size of the dataset. Possible values: [small, medium, large, demo]
					'-t', '--test_percentage' - Percentage of the dataset that will be used for testing. Possible values: [between 0.0 and 1.0] Default: 0.2
Data/iris.data           		Iris dataset (small dataset). https://archive.ics.uci.edu/ml/datasets/iris
Data/car.data                       	Car Evaluation dataset (medium dataset). https://archive.ics.uci.edu/ml/datasets/Car+Evaluation
Data/nursery.data             		Nursery dataset (large dataset). https://archive.ics.uci.edu/ml/datasets/Nursery
Data/human-identify.data            	Human Identification problem dataset from the lecture (it can be used for demonstration purposes and was used for the code debugging).
Output/iris_df.csv   			Output of the program for the iris dataset and Decision Forest algorithm (contains the results of all experiments ran on iris dataset using DF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)
Output/iris_rf.csv   			Output of the program for the iris dataset and Random Forest algorithm  (contains the results of all experiments ran on iris dataset using RF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)
Output/car_df.csv   			Output of the program for the car dataset and Decision Forest algorithm  (contains the results of all experiments ran on car dataset using DF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)
Output/car_rf.csv			Output of the program for the car dataset and Random Forest algorithm (contains the results of all experiments ran on car dataset using RF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)
Output/nursery_df.csv   		Output of the program for the nursery dataset and Decision Forest algorithm (contains the results of all experiments ran on nursery dataset using DF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)
Output/nursery_rf.csv			Output of the program for the nursery dataset and Random Forest algorithm (contains the results of all experiments ran on nursery dataset using RF: accuracy value for the test set, ordered list of the features used in the forest, according to its importance)