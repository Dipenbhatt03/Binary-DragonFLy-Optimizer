import numpy as np
import pandas as pd
# Import PySwarms
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import fitness_function as ff
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import learning_curve
from DFO import DFO
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from plot_curve import plot_learning_curve

import warnings


def fxn():
    warnings.warn("deprecated", DeprecationWarning)



class Project:

    def __init__(self, csv="aust.csv", classifier = GaussianNB(), class_name="Gaussian"):

        self.csv_data = pd.read_csv(csv, header=None)

        if csv == "aust.csv":
            self.X = self.csv_data.iloc[:, 0:14]
            self.Y = self.csv_data.iloc[:, 14:15]
            self.dataset = "Australian"
        elif csv == "GermanData.csv":
            self.X = self.csv_data.iloc[:, 0:20]
            self.Y = self.csv_data.iloc[:, 20: 21]
            self.dataset = "German"
        elif csv == "taiwan.csv":
            self.X = self.csv_data.iloc[:, 0:24]
            self.Y = self.csv_data.iloc[:, 24: 25]
            self.dataset = "Taiwan"


        self.num_features = self.X.shape[1]
        self.classifier = classifier
        self.best_feature_Set = np.ones(self.num_features)
        self.best_feature_indices = None

        self.class_name = class_name

    def f_per_particle(self, m, alpha=0.88):

        fit_obj = ff.FitenessFunction()


        # Get the subset of the features from the binary mask

        if np.count_nonzero(m) == 0:
            X_subset = self.X
        else:
            feature_idx = np.where(np.asarray(m) == 1)[0]
            X_subset = self.X.iloc[:, feature_idx]

        P, cv_set = fit_obj.calculate_fitness(self.classifier, X_subset, self.Y)

        # Compute for the objective function
        # j = (alpha * (1.0 - P)
        #      + (1.0 - alpha) * (1 - (X_subset.shape[1] / self.num_features)))

        return (P,cv_set)

    def f(self, x):

        n_particles = x.shape[0]
        j = [self.f_per_particle(x[i])[0] for i in range(n_particles)]
        return np.array(j)

    def optimize(self):
        num_particles = 40
        iters = 40
        # if self.class_name == "KNN":
        #     num_particles = 10
        #     iters = 10
        dfo = DFO(self.f, iters = iters, num_particles = num_particles, num_features = self.num_features)
        best_pos, best_cost = dfo.optimize(5)
        self.best_feature_Set = best_pos
        self.best_feature_indices = np.where(np.asarray(self.best_feature_Set) == 1)[0]

    def train(self, plot_num, classifier):
        f1_score, y_pred = self.f_per_particle(self.best_feature_Set)
        # fpr, tpr, _ = roc_curve(self.Y,y_pred)
        # auc = roc_auc_score(self.Y, y_pred)
        #
        # plt.plot(fpr,tpr,label="{}".format(classifier) + ", auc="+str(auc))
        # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.0])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('Receiver operating characteristic(ROC) curve for {} data set'.format(self.dataset))
        # plt.legend(loc=4)


        # if np.count_nonzero(self.best_feature_Set) == 0:
        #     X_subset = self.X
        # else:
        #     feature_idx = np.where(np.asarray(self.best_feature_Set) == 1)[0]
        #     X_subset = self.X.iloc[:, feature_idx]
        # title = "Learning Curves {}".format(self.class_name)
        # cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        # plot_learning_curve(self.classifier, title, X_subset, self.Y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)


        print("For {} Data set Best Feauture set : {}, Fitness : {}%".format(self.dataset,
                                                                             self.best_feature_indices,
                                                                             f1_score * 100))


classifier = [GaussianNB(), KNeighborsClassifier(n_neighbors=5), SVC(C=2.1810,gamma=0.0423)]
class_name = ["Gaussian NB", "KNN", "SVM"]
plotn = [1, 2, 3]

for x,y,z in zip(classifier,plotn, class_name):
    s = Project(classifier = x, class_name=z)
    s.optimize()
    s.train(plot_num=y, classifier=z)
plt.show()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
