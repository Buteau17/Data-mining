import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv


class LinearRegression:
    def __init__(self):
        self.closed_form_weights = None
        self.closed_form_intercept = None
        self.gradient_descent_weights = None
        self.gradient_descent_intercept = None

        self.epoch_list = []
        self.training_loss_list = []
        
    def closed_form_fit(self, X, y):
        X= np.insert(X, 0, 1, axis=1)
        X_transpose = np.transpose(X)
        self.closed_form_weights = np.dot(np.dot(np.linalg.inv(np.dot(X_transpose, X)) , X_transpose), y)
        self.closed_form_intercept = self.closed_form_weights[0]
        self.closed_form_weights = self.closed_form_weights[1:]      

    def gradient_descent_fit(self, X, y, lr, epochs):
        X= np.insert(X, 0, 1, axis=1)
        num_samples , num_features = X.shape
        self.gradient_descent_weights = np.zeros (num_features)

         #gradient descent
        for epoch in range (1, epochs):
            y_pred = np.dot (X, self.gradient_descent_weights.T)
            error = y_pred - y
            if epoch % 1000000 == 0:
                self.epoch_list.append(epoch)
                self.training_loss_list.append(self.get_mse_loss(y_pred, y))
                # print(error)
            gradient = np.dot(X.T, error) / num_samples
            # L1 regularization
            
            # gradient = np.dot(X.T, error) / num_samples + (1/num_samples) * np.abs(self.gradient_descent_weights)
            # Gradient update with L2 regularization
            # gradient = (X.T.dot(error) / num_samples) + reg_lambda * np.r_[0, self.gradient_descent_weights[1:]] 
            self.gradient_descent_weights -= lr * gradient
        self.gradient_descent_intercept = self.gradient_descent_weights[0]
        self.gradient_descent_weights = self.gradient_descent_weights[1:]    

    def get_mse_loss(self, prediction, ground_truth):
        mse= np.mean(( prediction - ground_truth ) **2 )
        return mse
      
    def closed_form_predict(self, X):
        return np.dot(X, self.closed_form_weights) + self.closed_form_intercept
   
    def gradient_descent_predict(self, X):
        return np.dot(X, self.gradient_descent_weights) + self.gradient_descent_intercept
       
    def closed_form_evaluate(self, X, y):
        return self.get_mse_loss(self.closed_form_predict(X), y)

    def gradient_descent_evaluate(self, X, y):
        return self.get_mse_loss(self.gradient_descent_predict(X), y)
        
    def plot_learning_curve(self):
        plt.plot(self.epoch_list, self.training_loss_list)
        plt.title("Training Loss")
        plt.xlabel("epoch"), plt.ylabel("MSE loss")
        plt.savefig('output.png')
        # plt.show()  
        
if __name__ == "__main__":
    # Data Preparation
    train_df = DataFrame(read_csv("train.csv"))
    # print(train_df.shape)
    date = train_df['Date'].unique()

    
    features_list = []
    target_list = []

    for i  in date :
        train_data = train_df[train_df['Date']==i]
        # print(train_data.shape)
        train_data = train_data.iloc[:, 3:].apply(pd.to_numeric, errors='coerce')
        train_data = train_data.fillna(0)

        target1 = train_data.iloc[9, -1]
        target2= train_data.iloc[9, 12]
        features1 =train_data.iloc[:, -10:-1].values.flatten()
        features2 =train_data.iloc[:,3:12].values.flatten()


        

        features_list.append(features1)
        features_list.append(features2)
        target_list.append(target1)
        target_list.append(target2)
   
    train_x = np.array(features_list)
    
    train_y = np.array(target_list)

        

    print(train_x.shape)
    print(train_y.shape)
    
    #Model Training and Evaluation
    LR = LinearRegression()

    LR.closed_form_fit(train_x, train_y)
    print("Closed-form Solution")
    print(f"Weights: {LR.closed_form_weights}, Intercept: {LR.closed_form_intercept}")

    LR.gradient_descent_fit(train_x, train_y, lr=0.000000001, epochs=30000000)
    print("Gradient Descent Solution")
    print(f"Weights: {LR.gradient_descent_weights}, Intercept: {LR.gradient_descent_intercept}")

    test_df = DataFrame(read_csv("test.csv",  header = None))

    test_date = test_df.iloc[:, 0].unique()
    # print(test_date.shape)

    test_features_list = []
    test_target_list = []

    for i in test_date:
        test_data = test_df[test_df.iloc[:, 0] == i]
        test_data = test_data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
        test_data = test_data.fillna(0)
        test_feature = test_data.values.flatten()
        
        test_features_list.append(test_feature)
      

    test_x = np.array(test_features_list , dtype=object)
    
          
    print(test_x.shape)
    gradient_descent_loss = LR.gradient_descent_evaluate(train_x.astype(float), train_y.astype(float))
    print(f'{gradient_descent_loss=}')
    gradient_descent_predict = LR.gradient_descent_predict(test_x.astype(float))
    print(gradient_descent_predict)
    print(gradient_descent_predict.shape)

    LR.plot_learning_curve()

    #Save predictions to CSV file
    with open('submission.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'answer'])
        for i, prediction in enumerate(gradient_descent_predict):
            writer.writerow([f"index_{i}", prediction])

    print("Predictions saved to submission.csv")