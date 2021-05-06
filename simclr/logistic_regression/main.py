import os
import argparse
import torch
from sklearn.linear_model import LogisticRegression
from data import Data

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Logistic Regression for SimCLR')

    # Directory Argument
    parser.add_argument('--data-dir', type=str, required=True)
    args = parser.parse_args()
    data = Data(args.data_dir)
        
    all_accuracies = []
    while(True):
        X_tr, y_tr, X_te, y_te = self.data.generate_test_episode(
            way=5,
            shot=1,
            query=15,
            n_episodes=1
        )

        if len(all_accuracies) >= 1000 or self.args.debug:
            break
        else:                
            clf = LogisticRegression().fit(X_tr[0], y_tr[0])
            y_te_pred = clf.predict(X_te[0])
            accuracy = np.mean((y_te_pred==y_te[0]).astype(float))
            all_accuracies.append(accuracy)
    print("1shot Accuracy: {0:.4f}".format(np.mean(all_accuracies)))
