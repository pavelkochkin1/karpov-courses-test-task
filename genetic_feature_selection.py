import random
from shutil import ExecError
from typing import Callable, Union
from functools import partial

import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split


# Create fitness function, Individual template
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Initialization crossover, mutatuion, selection
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

class GenA:
    """Feature selector with genetic algorithm.
    
    Given an external estimator that assigns score of `chromosomes`. 
    Every `chromosome` contains boolean mask of features to take.
    Then, with the help of crosses and mutations, new populations are generated.
    At the end, one chromosome is obtained, which contains a Boolean mask for the features.

    Args:
        estimator (Callable): A supervised learning estimator with a `fit` method.
        metric (Callable): Metric to score the features selection by estimator.
        population (int, optional): Number of individuals in one population Defaults to 15.
        verbose (bool, optional): Controls verbosity of output. Defaults to False.
        random_state (int, optional): Seed of random to use. Defaults to 73.
        
    Attrs:
        support_ (ndarray): The mask of selected features.
        n_features_ (int): The number of selected features.
        features_names_ (ndarray): Names of selected features.
        self.estimator (Callable): The fitted estimator with selected features.
        
    Methods:
        fit: Fitting the selector.
        transform: Transform new data.
        fit_transform: Fitting selector and retern transformed data.
        predict: Transforme new data and make a predictions.
        predict_proba: Transforme new data and make a probability predictions.
        score: Transforme new data and score the estimator.
        
    Examples:
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.metrics import accuracy_score
    >>> X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=73)
    >>> selector = GenA(estimator=LogisticRegression, metric=accuracy_score, random_state=73)
    >>> selector = selector.fit(X, y)
    >>> selector.support_
    [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]
    """
    
    def __init__(
        self, 
        estimator: Callable,
        metric: Callable,
        population = 15,
        verbose: bool = False,
        random_state = 73,
    ):  
        if population <= 0:
            raise ValueError("Population must be >0")
        if not hasattr(estimator, "predict"):
            raise Exception("Estimator must has a `predict` method")
        
        self.estimator = estimator(random_state=random_state)
        self.verbose = verbose
        self.metric = metric
        self.population = population
        self.random_state = random_state
        self.features_names = None
        # For reproducable results
        random.seed(random_state)
        
    def fit(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        y: Union[np.ndarray, pd.Series],
        **fit_params
    ) -> Callable:
        """Fit the GenA model and then the underlying estimator on the selected features.

        Args:
            X (Union[np.ndarray, pd.DataFrame]): The training input samples.
            y (Union[np.ndarray, pd.Series]): The target values.

        Returns:
            Callable: fitted selector
        """
        
        X = self.__pd_data(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=self.random_state)
        
        # pop - start population, hof - count of best combinations we will take, stats - statistics to show
        pop, hof, stats = self.__genA_prepare(X_train, X_test, y_train, y_test)
        
        _ = algorithms.eaSimple(
            pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, halloffame=hof, stats=stats, verbose=self.verbose
        )
        
        # Set the final attrs
        # Get the best individual
        self.support_ = hof[0]
        self.n_features_ = sum(self.support_)
        self.features_names_ = np.array(X_train.loc[:, [bool(x) for x in self.support_]].columns)
        self.estimator = self.estimator.fit(X[self.features_names_], y, **fit_params)
        
        return self
    
    def transform(self, X: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """Transform new data with selected features.

        Args:
            X (Union[np.array, pd.DataFrame]): Input values

        Returns:
            np.ndarray: Values with selected features.
        """
        if self.features_names_ is None:
            raise ExecError("Can't run this method before fitting!")
        
        X = self.__pd_data(X)
        return X[self.features_names_].values
    
    def fit_transform(
        self, 
        X: Union[np.array, pd.DataFrame], 
        y: Union[np.array, pd.Series], 
        **fit_params
    ) -> np.ndarray:
        X = self.__pd_data(X)
        _ = self.fit(X, y, **fit_params)
        return self.transform(X)
    
    def predict(self, X: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """Prediction by new values.

        Args:
            X (Union[np.array, pd.DataFrame]): Input values.

        Returns:
            np.ndarray: Array with predictions.
        """
        
        return self.estimator.predict(self.transform(X))
    
    def predict_proba(self, X: Union[np.array, pd.DataFrame]) -> np.ndarray:
        """Probability predictions by new values if estimator has method `predict_proba`.

        Args:
            X (Union[np.array, pd.DataFrame]): Input alues.

        Raises:
            Exception: if estimator has not method `predict_proba`.

        Returns:
            np.ndarray: Array with predictions.
        """
        
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(self.transform(X))
        raise Exception("Estimator has not method `predict_proba`")
    
    def score(
        self, 
        X: Union[np.array, pd.DataFrame], 
        y: Union[np.array, pd.Series],
        **fit_params
    ) -> float:
        """Return the score of the underlying estimator.

        Args:
            X (Union[np.array, pd.DataFrame]): Input values.
            y (Union[np.array, pd.Series]): Target values.

        Returns:
            float: Score of the estimator with the selected features.
        """
        
        if hasattr(self.estimator, "score"):
            return self.estimator.score(self.transform(X), y)
        prediction = self.predict(X)
        return self.metric(y, prediction)
            
    @staticmethod
    def __pd_data(X: np.array) -> pd.DataFrame:
        """Returns pandas DataFrame for right work of algorithm.

        Args:
            data (np.array): Input values.

        Returns:
            pd.DataFrame
        """
        
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X)
        return X
    
    @staticmethod
    def evaluate(individual, estimator, metric, X_train, y_train, X_test, y_test) -> float:
        """Returns the score for one individual(one set of features).

        Args:
            individual (_type_)
            estimator (_type_)
            metric (_type_): Metric to score the individuals.

        Returns:
            float: score
        """
        
        sum_features = np.sum(individual)
        if sum_features == 0:
            return 0.0,
        else:
            X_train_sel = X_train.loc[:, [bool(x) for x in individual]]
            X_test_sel = X_test.loc[:, [bool(x) for x in individual]]
            clf = estimator.fit(X_train_sel, y_train)
            y_pred = clf.predict(X_test_sel)
            return metric(y_test, y_pred),
    
    def __genA_prepare(self, X_train, X_test, y_train, y_test):
        # Creating evaluation function
        evaluate = partial(
            self.evaluate,
            estimator=self.estimator,
            metric=self.metric,
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
        )
        
        # We need a mask of features like [1,0,1,0,0] then we have to random of 1 or 0
        toolbox.register("attrib_bin", random.randint, 0, 1)
        # Register the individuals in the toolbox.
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attrib_bin, n=X_train.shape[1])
        # Register the population in toolbox.
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # Register the evaluation function in toolbox.
        toolbox.register("evaluate", evaluate)
        # Sets the statistics to show
        if self.verbose:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("max", np.max)
            stats.register("min", np.min)
        else:
            stats = None
        
        # Initialize first population    
        pop = toolbox.population(n=self.population)
        hof = tools.HallOfFame(1)
        return pop, hof, stats

if __name__ == "__main__":
    import doctest
    doctest.testmod()
