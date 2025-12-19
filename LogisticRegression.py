from pyexpat import features
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score

def calculateGradient(N, x, y, w):
    sum=np.zeros_like(w)
    for n in range(N):
        sum+=(y[n]*x[n])/(1+np.e**(-y[n]*(w.T@x[n])))
    #end for
    return sum

def LogisticGradientDescent(x, y, b):
    x = np.array(x)
    x = np.insert(x, 0, b.flatten(), axis=1)
    w = np.zeros(x.shape[1])
    N = len(x)
    stepsize = 0.01
    for t in range(100):
        gradient_t = (-1/N)*calculateGradient(N, x, y, w)
        moveDirection = -gradient_t
        w = w + stepsize*moveDirection
    #end for
    return w

def plotFeatures(training, test):
    plt.clf()
    #bar plot that shows the difference in training and test weights
    Labels = ["Bias", "Width", "Brick"]

    x = np.arange(len(Labels))   # [0, 1]
    width = 0.35                   # bar width
    plt.figure()
    plt.bar(x - width/2, training, width, label='Training Weights')
    plt.bar(x + width/2, test, width, label='Test Weights')
    plt.xticks(x, Labels)
    plt.ylabel("Logistic Regression Weight")
    plt.title("Bias + Learned Feature Weights (Training vs Test)")
    plt.legend()
    plt.show()

def calculateAccuracy(xTest, yTest, trainResults):
    score = xTest @ trainResults
    thresholds = np.linspace(score.min(), score.max(), 100)
    best_acc = 0
    best_thresh = 0

    #calculate best threshold
    for t in thresholds:    
        y_pred = (score > t).astype(int)
        acc = accuracy_score(yTest, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_thresh = t
    yPred = (score>best_thresh).astype(int)
    accuracy = accuracy_score(yTest, yPred)
    print(f"Model accuracy: {accuracy:.2f}")

    #plot bar graph of each prediction
    contributions = xTest * trainResults[:xTest.shape[1]]
    for i in range(len(xTest)): 
        print(f"Image {i+1}:")
        print(f"  Features: {xTest[i]}")
        print(f"  Contributions: {contributions[i]}")
        print(f"  Total score: {score[i]:.3f}")
        print(f"  Predicted: {yPred[i]}, True: {yTest[i]}")

def main():
    with open('HML Project Set.json', 'r') as f:
        data = json.load(f) 
    # Print the data
    print(data)

    training_features = []
    testFeatures = []
    for i in data['customCoordinates']:
        tags = i['extra']['tags']
        if "training" in tags:
            training_features.append(tags)
        else:
            testFeatures.append(tags)
    #end for


    training_features = [[f for f in features if f != "training"] for features in training_features]
    trainingLabels = [next(f for f in features if f.lower() in ["usa", "europe"]) for features in training_features]
    testFeatures = [[f for f in features if f != "test"] for features in testFeatures]
    testLabels = [next(f for f in features if f.lower() in ["usa", "europe"]) for features in testFeatures]
    
    print("trainingLabels", trainingLabels)

    xTrain = []
    yTrain = []

    for features in training_features:
        features_lower = [f.lower() for f in features]
        label = 1 if 'usa' in features_lower else 0
        width = 1 if 'wide' in features_lower else 0
        brick = 1 if 'brick' in features_lower else 0
        xTrain.append([width, brick])
        yTrain.append(label)
    #end for
    
    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)
    b = np.ones((xTrain.shape[0], 1))
    trainResult = LogisticGradientDescent(xTrain, yTrain, b)
    print("Learned weights:", trainResult)

    xTest = []
    yTest = []
    for features in testFeatures:
        features_lower = [f.lower() for f in features]
        label = 1 if 'usa' in features_lower else 0
        width = 1 if 'wide' in features_lower else 0
        brick = 1 if 'brick' in features_lower else 0
        xTest.append([width, brick])
        yTest.append(label)
    #end for
    xTest = np.array(xTest)
    yTest = np.array(yTest)
    bTest = np.ones((xTest.shape[0], 1))
    xTestBias = np.insert(xTest, 0, bTest.flatten(), axis=1)
    wTest = LogisticGradientDescent(xTest, yTest, bTest)
    print("Test Weights:", wTest)
    plotFeatures(trainResult, wTest)

    calculateAccuracy(xTestBias, yTest, trainResult) #calculate accuracy of test data using training results
if __name__ == "__main__":
    main()