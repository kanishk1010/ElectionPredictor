import pandas as pd
import numpy as np
from random import uniform

#Data with features and target values
#Tutorial for Pandas is here - https://pandas.pydata.org/pandas-docs/stable/tutorials.html
#Helper functions are provided so you shouldn't need to learn Pandas
import time

dataset = pd.read_csv("data.csv")

#========================================== Data Helper Functions ==========================================


#Normalize values between 0 and 1
#dataset: Pandas dataframe
#categories: list of columns to normalize, e.g. ["column A", "column C"]
#Return: full dataset with normalized values
def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData


#Encode categorical values as mutliple columns (One Hot Encoding)
#dataset: Pandas dataframe
#categories: list of columns to encode, e.g. ["column A", "column C"]
#Return: full dataset with categorical columns replaced with 1 column per category
def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)


#Split data between training and testing data
#dataset: Pandas dataframe
#ratio: number [0, 1] that determines percentage of data used for training
#Return: (Training Data, Testing Data)
def trainingTestData(dataset, ratio):
    tr = int(len(dataset)*ratio)
    return dataset[:tr], dataset[tr:]


#Convenience function to extract Numpy data from dataset
#dataset: Pandas dataframe
#Return: features numpy array and corresponding labels as numpy array
def getNumpy(dataset):
    features = dataset.drop(["can_id", "can_nam","winner"], axis=1).values
    labels = dataset["winner"].astype(int).values
    return features, labels


#Convenience function to extract data from dataset (if you prefer not to use Numpy)
#dataset: Pandas dataframe
#Return: features list and corresponding labels as a list
def getPythonList(dataset):
    f, l = getNumpy(dataset)
    return f.tolist(), l.tolist()


#Calculates accuracy of your models output.
#solutions: model predictions as a list or numpy array
#real: model labels as a list or numpy array
#Return: number between 0 and 1 representing your model's accuracy
def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)
    return (predictions == labels).sum() / float(labels.size)

#===========================================================================================================

class KNN:
    def __init__(self):
        #KNN state here
        #Feel free to add methods
        self.k = 5

    def preprocess(self, dataset):
        normalizeCategories = list(dataset.columns.values)[2:5]
        encodeCategories = list(dataset.columns.values)[5:7]
        normalizedData = normalizeData(dataset, normalizeCategories)
        encodedData = encodeData(normalizedData, encodeCategories)
        return getNumpy(encodedData)

    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        self.train_features = features
        self.train_labels = labels

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        predictions = []
        for feature in features:
            euc_distance = np.sqrt(np.sum((self.train_features - feature)**2 , axis=1))
            k_nearest = np.argsort(euc_distance)[0:self.k]
            label_counter = [0,0]
            for index in k_nearest:
                label_counter[self.train_labels[index]] += 1
            predictions.append(0 if label_counter[0]>label_counter[1] else 1)
        return predictions


class Perceptron:
    def __init__(self):
        #Perceptron state here
        #Feel free to add methods
        self.weights = np.array([uniform(-0.1,0.1) for _ in range(9)])
        self.learning_rate = 0.01
        self.bias = uniform(-0.1,0.1)

    def preprocess(self, dataset):
        normalizeCategories = list(dataset.columns.values)[2:5]
        encodeCategories = list(dataset.columns.values)[5:7]
        normalizedData = normalizeData(dataset, normalizeCategories)
        encodedData = encodeData(normalizedData, encodeCategories)
        return getNumpy(encodedData)

    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        start_time = time.time()
        while time.time()-start_time <= 60:
            index = 0
            for feature in features:
                y = self.activate(feature)
                if y != labels[index]:
                    self.bias += self.learning_rate * (labels[index]-y)
                    change = self.learning_rate * (labels[index]-y) * feature
                    self.weights = np.add(self.weights, change)
                index += 1

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        predictions = []
        for feature in features:
            predictions.append(self.activate(feature))
        return predictions

    def activate(self, feature):
        activation = self.bias + np.dot(self.weights,feature)
        return 1 if activation >= 0 else 0


class MLP:
    def __init__(self):
        #Multilayer perceptron state here
        #Feel free to add methods
        np.random.seed(1)
        self.hidden_layer_weights = (2 * np.random.random((9,9))-1) /10
        self.outer_layer_weights = (2 * np.random.random((9,1))-1) /10
        self.learning_rate = 0.01
        self.hidden_layer_bias = np.array([uniform(-.1,.1) for _ in range(9)])
        self.outer_layer_bias = uniform(-.1,.1)

    def preprocess(self, dataset):
        normalizeCategories = list(dataset.columns.values)[2:5]
        encodeCategories = list(dataset.columns.values)[5:7]
        normalizedData = normalizeData(dataset, normalizeCategories)
        encodedData = encodeData(normalizedData, encodeCategories)
        return getNumpy(encodedData)


    def train(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        labels = labels.reshape((750,1))
        start_time = time.time()
        while time.time()-start_time <= 60:
            hidden_layer_output = self.sigmoid(np.dot(features, self.hidden_layer_weights) + self.hidden_layer_bias)
            outer_layer_output = self.sigmoid(np.dot(hidden_layer_output, self.outer_layer_weights) + self.outer_layer_bias)
            outer_layer_error = labels - outer_layer_output
            outer_layer_delta = outer_layer_error * self.derivative(outer_layer_output)
            hidden_layer_error = outer_layer_delta.dot(self.outer_layer_weights.T)
            hidden_layer_delta = hidden_layer_error * self.derivative(hidden_layer_output)
            self.hidden_layer_weights += features.T.dot(hidden_layer_delta) * self.learning_rate
            self.hidden_layer_bias += np.average(hidden_layer_delta, axis=0) * self.learning_rate
            self.outer_layer_weights += hidden_layer_output.T.dot(outer_layer_delta) * self.learning_rate
            self.outer_layer_bias += float(np.average(outer_layer_delta, axis=0) * self.learning_rate)

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        hidden_layer_output = self.sigmoid(np.dot(features, self.hidden_layer_weights) + self.hidden_layer_bias)
        outer_layer_output = self.sigmoid(np.dot(hidden_layer_output, self.outer_layer_weights) + self.outer_layer_bias)
        predictions = list(map(lambda x: 1 if x >= .5 else 0, list(outer_layer_output)))
        return predictions

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x))


class ID3:
    def __init__(self):
        #Decision tree state here
        #Feel free to add methods
        self.categories= ['net_ope_exp', 'net_con', 'tot_loa', 'can_off', 'can_inc_cha_ope_sea']
        pass

    def preprocess(self, dataset):
        normalizeCategories = list(dataset.columns.values)[2:5]
        normalizedData = normalizeData(dataset, normalizeCategories)
        normalizedData['net_ope_exp'] = pd.cut(normalizedData['net_ope_exp'], bins=[0, .2, .4, .6, .8, 1], include_lowest=True, labels=['0', '1', '2', '3', '4'])
        normalizedData['tot_loa'] = pd.cut(normalizedData['tot_loa'], bins=[0, .2, .4, .6, .8, 1], include_lowest=True, labels=['0', '1', '2', '3', '4'])
        normalizedData['net_con'] = pd.cut(normalizedData['net_con'], bins=[0, .2, .4, .6, .8, 1], include_lowest=True, labels=['0', '1', '2', '3', '4'])
        return getNumpy(normalizedData)

    def entropy(self, data):
        entropy = 0
        val, counts = np.unique(data, return_counts=True)
        frequency = counts.astype('float')/len(data)
        for p in frequency:
            if p != 0:
                entropy -= p * np.log2(p)
        return entropy

    def gain(self, column, labels):
        gain = self.entropy(labels)
        val, counts = np.unique(column, return_counts=True)
        freqs = counts.astype('float')/len(column)
        for p, v in zip(freqs, val):
            gain -= p * self.entropy(labels[column == v])
        return gain

    def partition(self, val):
        return {c: (val==c).nonzero()[0] for c in np.unique(val)}

    def create_tree(self, features, labels):
        #training logic here
        #input is list/array of features and labels
        if len(features)==1 or len(labels) == 0:
            return labels

        gains = []
        for i in range(5):
            gains.append(self.gain(features[:,i], labels))
        gains = np.array(gains)
        if np.all(gains == 0.0):
            return labels
        node = int(np.argmax(gains))
        sets = self.partition(features[:, node])
        tree = {}
        for k, v in sets.items():
            labels_subset = labels.take(v, axis=0)
            features_subset = features.take(v, axis=0)
            tree["%s=%s" % (self.categories[node], k)] = self.create_tree(features_subset, labels_subset)
        return tree

    def train(self, features, labels):
        self.tree = self.create_tree(features, labels)

    def get_label_from_tree(self, tree, feature):
        for k, v in tree.items():
            category, val = k.split('=')
            index = self.categories.index(category)
            if str(feature[index]) == val:
                if isinstance(v, dict):
                    return self.get_label_from_tree(v, feature)

                elif v is None:
                    return 0
                else:
                    val, counts = np.unique(v, return_counts=True)
                    return val[np.argmax(counts)]
            else:
                continue

    def predict(self, features):
        #Run model here
        #Return list/array of predictions where there is one prediction for each set of features
        predictions = []
        for feature in features:
            prediction = self.get_label_from_tree(self.tree, feature)
            predictions.append(prediction)
        return predictions

