import os
import numpy as np
from sklearn import preprocessing

class Data:
    def __init__(self, data_dir):        
        self.N = 600 # total num instances per class
        self.K_mtr = 64 # total num meta_train classes
        self.K_mte = 20 # total num meta_test classes
        
        x_mtr = np.load(os.path.join(data_dir, 'mimgnet/train_features.npy'))
        x_mte = np.load(os.path.join(data_dir, 'mimgnet/test_features.npy'))

        scaler = preprocessing.StandardScaler()
        scaler.fit(x_mtr)
        x_mtr = scaler.transform(x_mtr)
        x_mte = scaler.transform(x_mte)

        dim = x_mte.shape[-1]
        x_mte = np.reshape(x_mte, [20,600,dim])
        self.x_mte = x_mte

    def generate_test_episode(self, way, shot, query, n_episodes=1):
        generate_label = lambda way, n_samp: np.repeat(np.eye(way), n_samp, axis=0)
        n_way, n_shot, n_query = way, shot, query
        (K,x) = self.K_mte, self.x_mte

        xtr, ytr, xte, yte = [], [], [], []
        for t in range(n_episodes):
            # sample WAY classes
            classes = np.random.choice(range(K), size=n_way, replace=False)

            xtr_t = []
            xte_t = []
            for k in list(classes):
                # sample SHOT and QUERY instances
                idx = np.random.choice(range(self.N), size=n_shot+n_query, replace=False)
                x_k = x[k][idx]
                xtr_t.append(x_k[:n_shot])
                xte_t.append(x_k[n_shot:])

            xtr.append(np.concatenate(xtr_t, 0))
            xte.append(np.concatenate(xte_t, 0))
            ytr.append(generate_label(n_way, n_shot))
            yte.append(generate_label(n_way, n_query))

        xtr, ytr = np.stack(xtr, 0), np.stack(ytr, 0)
        ytr = np.argmax(ytr, -1)
        xte, yte = np.stack(xte, 0), np.stack(yte, 0)
        yte = np.argmax(yte, -1)        
        return [xtr, ytr, xte, yte]


