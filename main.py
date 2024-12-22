from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
from tqdm import tqdm

class FashionClassifier:
    def __init__(self):
        # Class labels for Fashion MNIST
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.load_data()
        
    def load_data(self):
        """Load and preprocess Fashion MNIST dataset"""
        # Load data
        (self.X_train, self.y_train), (self.X_test, self.y_test) = fashion_mnist.load_data()
        
        # Reshape and normalize data 
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1) / 255.0
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1) / 255.0
        
        # Take a subset for faster training during development
        self.X_train = self.X_train[:10000]
        self.y_train = self.y_train[:10000]
        self.X_test = self.X_test[:1000]
        self.y_test = self.y_test[:1000]

    def visualize_samples(self, num_samples=5):
        """Visualize sample images from each class"""
        plt.figure(figsize=(15, 8))
        for class_idx in range(10):
            for sample_idx in range(num_samples):
                plt.subplot(10, num_samples, class_idx * num_samples + sample_idx + 1)
                plt.imshow(self.X_train[self.y_train == class_idx][sample_idx].reshape(28, 28),
                          cmap='gray')
                plt.axis('off')
                if sample_idx == 0:
                    plt.title(self.class_names[class_idx])
        plt.tight_layout()
        plt.show()

    class SVMClassifier:
        def __init__(self, kernel='linear', C=1.0, gamma='scale'):
            self.kernel = kernel
            self.C = C
            self.gamma = gamma
            self.model = None
        
        def fit(self, X, y):
            """Train the SVM model with given kernel and parameters."""
            self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)
            self.model.fit(X, y)
        
        def predict(self, X):
            """Predict with the trained SVM model."""
            return self.model.predict(X)
        
        def evaluate_model(self, y_true, y_pred, model_name):
            """Evaluate model performance and display metrics."""
            print(f"\n{model_name} Results:")
            print("Accuracy:", accuracy_score(y_true, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=self.class_names))
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=self.class_names,
                        yticklabels=self.class_names)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def run_experiments(self):
        """Run all classification experiments including SVM."""
        # 1. KNN Classification
        print("\nRunning KNN Classification...")
        knn = self.KNNClassifier(k=3)
        knn.fit(self.X_train, self.y_train)
        knn_predictions = knn.predict(self.X_test)
        self.evaluate_model(self.y_test, knn_predictions, "KNN")

        # 2. Binary Logistic Regression (T-shirt vs. Trouser)
        print("\nRunning Binary Logistic Regression...")
        log_reg = self.LogisticRegression(learning_rate=0.1, num_iterations=100)
        binary_classes =(0, 1)
        self.lrClasses = [self.class_names[cls] for cls in binary_classes]
        log_reg.fit(self.X_train, self.y_train, binary_classes)
        # Evaluate only on test samples from the two classes
        test_mask = np.isin(self.y_test, binary_classes)
        log_reg_predictions = log_reg.predict(self.X_test[test_mask])
        self.evaluate_model(self.y_test[test_mask], log_reg_predictions, "Logistic Regression (Binary)", True)

        # 3. Softmax Regression
        print("\nRunning Softmax Regression...")
        softmax = self.SoftmaxRegression(learning_rate=0.1, num_iterations=100)
        softmax.fit(self.X_train, self.y_train)
        softmax_predictions = softmax.predict(self.X_test)
        self.evaluate_model(self.y_test, softmax_predictions, "Softmax Regression")
        
        # 4. SVM Classification with different kernels
        print("\nRunning SVM Classification with Linear Kernel...")
        svm_linear = self.SVMClassifier(kernel='linear', C=1.0, gamma='scale')
        svm_linear.fit(self.X_train, self.y_train)
        svm_linear_predictions = svm_linear.predict(self.X_test)
        self.evaluate_model(self.y_test, svm_linear_predictions, "SVM (Linear Kernel)")

        print("\nRunning SVM Classification with RBF Kernel...")
        svm_rbf = self.SVMClassifier(kernel='rbf', C=1.0, gamma='scale')
        svm_rbf.fit(self.X_train, self.y_train)
        svm_rbf_predictions = svm_rbf.predict(self.X_test)
        self.evaluate_model(self.y_test, svm_rbf_predictions, "SVM (RBF Kernel)")

        # 5. Hyperparameter tuning for SVM with GridSearchCV
        print("\nTuning SVM Hyperparameters with GridSearchCV...")
        param_grid = {'C': [0.1, 1.0, 10], 'gamma': ['scale', 'auto']}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3)
        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters from GridSearchCV:", grid_search.best_params_)
        best_svm = grid_search.best_estimator_
        best_svm_predictions = best_svm.predict(self.X_test)
        self.evaluate_model(self.y_test, best_svm_predictions, "SVM (Tuned)")

    def visualize_decision_boundaries(self, X, y, model):
        """Visualize the decision boundaries of a model."""
        X_min, X_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(X_min, X_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
        plt.title(f"Decision Boundary for {model.kernel} Kernel")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

if __name__ == "__main__":
    # Initialize and run the fashion classifier
    classifier = FashionClassifier()
    
    # Visualize sample images
    print("Visualizing sample images...")
    classifier.visualize_samples()
    
    # Run all experiments
    classifier.run_experiments()
