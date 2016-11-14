import numpy as np
from matplotlib import pyplot as plt

class findidentical(object):
    def create_indentical(self,X,Y):
        self.sorted_y = sorted(np.unique(Y))
        n_class = len(self.sorted_y)
        self.stringX = np.array(["".join(str(x)) for x in X])
        self.overlapmat = np.zeros([n_class,n_class])
        self.sets = [set(self.stringX[Y==y]) for y in self.sorted_y ]
        self.id_len = [len(s) for s in self.sets]
        for i in xrange(n_class):
            for j in xrange(n_class):
                self.overlapmat[i,j] = len(self.sets[i].intersection(self.sets[j]))
        N = len(Y)

        misclassified = 0
        for i in xrange(1, n_class):
            for j in xrange(i):
                print self.overlapmat[i, j]
                misclassified += self.overlapmat[i,j]

        duplicate_ratio = N/float(np.trace(self.overlapmat))
        best_accruracy = 1 - misclassified*duplicate_ratio/N

        return best_accruracy, self.overlapmat, self.sorted_y, self.id_len

    def compute_upper_bound(self, X, Y):
        self.sorted_y = sorted(np.unique(Y))
        n_class = len(self.sorted_y)
        self.stringX = np.array(["".join([str(int(xx)) for xx in x]) for x in X])

        self.stringX_dict = {}
        for s,y in zip(self.stringX, Y):
            self.stringX_dict.setdefault(s,[0]*n_class)
            self.stringX_dict[s][self.sorted_y.index(y)] += 1


        print len(self.stringX_dict)
        N = len(Y)
        best_accuracy = sum([max(self.stringX_dict[s]) for s in self.stringX_dict])/float(N)
        return best_accuracy

    def verify_matching(self,X):
        import networkx as nx
        self.sorted_y = sorted(np.unique(Y))
        n_class = len(self.sorted_y)
        self.stringX = np.array(["".join([str(int(xx)) for xx in x]) for x in X])
        Xset = list(set(self.stringX))

        for i in xrange(len(Xset)):
            for j in xrange(i+1,len(Xset)):
                print i,j
                xmati = np.array([int(e) for e in Xset[i]]).reshape(8,8)
                xmatj = np.array([int(e) for e in Xset[j]]).reshape(8,8)
                Gi = nx.from_numpy_matrix(xmati)
                Gj = nx.from_numpy_matrix(xmatj)
                if(nx.is_isomorphic(Gi,Gj)):
                    print xmati
                    nx.draw(Gi)
                    plt.show()
                    print xmatj
                    nx.draw(Gj)
                    plt.show()
                    import sys
                    sys.exit()


if __name__ == "__main__":
    d=8
    data_folder_dir = "../Extension/AdjmatData/"
    X = np.load(data_folder_dir + ("/%d/" % d) + 'X_data_image.npy')
    Y = np.load(data_folder_dir + ("/%d/" % d) + 'Y_data.npy')
    util = findidentical()
    # print util.create_indentical(X,Y)
    # print util.compute_upper_bound(X,Y)
    util.verify_matching(X)