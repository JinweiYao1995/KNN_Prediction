import numpy as np
import pandas as pd
from knn import KNN

def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)
    addup = [x + y for x, y in zip(real_labels, predicted_labels)]
    precision = np.sum(np.equal(addup,2))/np.sum(np.equal(predicted_labels,1))
    recall = np.sum(np.equal(addup,2))/np.sum(np.equal(real_labels,1))
    if precision == 0 and recall ==0:
        f1_score = 0
    else: 
        f1_score = 2*(precision*recall)/(precision+recall)
    return f1_score
    

class Distances:
    @staticmethod
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :param p: int
        :return: float
        """
        # P-value is 3 in this case
        def p_root(value, root): 
            #Function of operating the root calculation of the value
            root_value = 1 / float(root) 
            return value ** root_value
        p = 3  
        # d1 Calculate the difference of all dimension  
        d1 = [abs(x-y) for x,y in zip(point1,point2)]
        # d2 Power to the p-value exponent
        d2 = [d1[i]**p for i in range(len(d1))]
        distance = p_root(sum(d2),p)
        return distance



    @staticmethod
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        def p_root(value, root): 
            #Function of operating the root calculation of the value
            root_value = 1 / float(root) 
            return value ** root_value
        distance = p_root(sum([(a - b) ** 2 for a, b in zip(point1, point2)]),2)
        return distance

    @staticmethod
    def inner_product_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        distance = np.inner(point1,point2)
        return distance
    @staticmethod
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        dot_product = np.dot(point1,point2)
        norm_1 = np.linalg.norm(point1)
        norm_2 = np.linalg.norm(point2)
        cos_distance = 1 - dot_product/(norm_1 * norm_2)
        return cos_distance
    @staticmethod
    def gaussian_kernel_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        point1 = np.array(point1)
        point2 = np.array(point2)
        dif = point1 - point2 
        distance = -np.exp(-1.*np.inner(dif,dif)/2)
        return distance



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        self.best_scaler = None

    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """
        best_function = Distances.euclidean_distance
        best_f1_score = -1
        best_k = 0
        for name,function in distance_funcs.items():
            for k in range(1,31,2):
                model = KNN(k,function)
                model.train(x_train,y_train)
                train_f1_score = f1_score(y_train,model.predict(x_train))
                
                valid_f1_score = f1_score(y_val,model.predict(x_val))
            print('{}\tk: {}\t'.format(name, k) + 
                  'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                  'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
            
            if valid_f1_score > best_f1_score:
                best_f1_score = valid_f1_score
                best_function = function
                best_k = k
        
        self.best_k = best_k
        self.best_distance_function = best_function
        self.best_model = KNN(self.best_k,self.best_distance_function)
        raise NotImplementedError

    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [euclidean > minkowski > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """
        best_function = Distances.euclidean_distance
        best_f1_score = -1
        best_k = 0
        best_scaler = MinMaxScaler()
        for scaling_name, scaling_method in scaling_classes.items():
            scaler = scaling_method()
            x_train_scaled = scaler(x_train)
            x_val_scaled = scaler(x_val)
            for name,function in distance_funcs.items():
                for k in range(1,31,2):
                    model = KNN(k,function)
                    model.train(x_train_scaled,y_train)
                    train_f1_score = f1_score(y_train,model.predict(x_train_scaled))
                    valid_f1_score = f1_score(y_val,model.predict(x_val_scaled))
                    print('{}\tk: {}\t'.format(name, k) + 
                        'train: {train_f1_score:.5f}\t'.format(train_f1_score=train_f1_score) +
                        'valid: {valid_f1_score:.5f}'.format(valid_f1_score=valid_f1_score))
                    if valid_f1_score > best_f1_score:
                        best_f1_score = valid_f1_score
                        best_function = function
                        best_k = k
                        best_scaler = scaling_method
        
        self.best_k = best_k
        self.best_distance_function = best_function
        self.best_model = KNN(self.best_k,self.best_distance_function)
        self.best_model = best_scaler
        raise NotImplementedError

class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        norm = list()
        def normalize(vector):
            norm = np.linalg.norm(vector)
            if norm == 0: 
                return vector
            vector[:]= [float("{0:.6f}".format(i/norm)) for i in vector]
            return vector
        for x in features: 
            norm.append(normalize(x))
        return norm


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hint: Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note: You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]
    """

    def __init__(self):
        self.max = None
        self.min = None

    def __call__(self, features):
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        transpose = [list(i) for i in zip(*features)]
        if np.logical_and(self.max == None,self.min == None):
            self.min = [np.min(x) for x in transpose]
            self.max = [np.max(x) for x in transpose]

        Output = []
        for vector in features:
            num = [a-b for a,b in zip(vector,self.min)]
            denom = [a-b for a,b in zip(self.max,self.min)]
            v= [float("{0:.6f}".format(i/j)) for i, j in zip(num,denom)]
            Output.append(v)
        return Output
        raise NotImplementedError

