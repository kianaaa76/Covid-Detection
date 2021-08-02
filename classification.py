from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
import random
from sklearn.metrics import accuracy_score

model = Model(inputs=model.inputs, outputs=model.layers[-3].output)

def get_data(data_dir):
    data = [] 
    for label in labels: 
        if (label == '0'):
          path = os.path.join(data_dir, 'negative')
        else:
          path = os.path.join(data_dir, 'positive')
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, int(label)])
            except Exception as e:
                print(e)
    return(data)

train = get_data('/content/drive/MyDrive/image_project_train')
test = get_data('/content/drive/MyDrive/image_project_test')
valid = get_data('/content/drive/MyDrive/image_project_validation')

x_train = []
y_train = []
x_test = []
y_test = []
x_valid = []
y_valid = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in test:
  x_test.append(feature)
  y_test.append(label)

for feature, label in valid:
  x_valid.append(feature)
  y_valid.append(label)

X = []
Y = []

for feature, label in train:
  X.append(feature)
  Y.append(label)

for feature, label in test:
  X.append(feature)
  Y.append(label)

for feature, label in valid:
  X.append(feature)
  Y.append(label)

X = model.predict(X, batch_size=1)
X = list(X)
#decision tree classifier
dt_clf = DecisionTreeClassifier(random_state=0)

#rafdom forest classifier
rf_clf = RandomForestClassifier(max_depth=2, random_state=0)

# svm classifier
svm_clf = svm.SVC()

#adaboost classifier
ada_clf = AdaBoostClassifier(n_estimators=100, random_state=0)

strtfdKFold = StratifiedKFold(n_splits=5)
kfold = strtfdKFold.split(X, Y)

accuracy_score_list = []

for k, (train, test) in enumerate(kfold):
    X_train = []
    Y_train = []
    for i in list(train):
      X_train.append(X[i])
      Y_train.append(Y[i])

    dt_clf.fit(X_train, Y_train)
    rf_clf.fit(X_train, Y_train)
    svm_clf.fit(X_train, Y_train)
    ada_clf.fit(X_train, Y_train)

    X_test = []
    Y_test = []
    for i in list(test):
      X_test.append(X[i])
      Y_test.append(Y[i])

    dt_y_pred = dt_clf.predict(X_test)
    rf_y_pred = rf_clf.predict(X_test)
    svm_y_pred = svm_clf.predict(X_test)
    ada_y_pred = ada_clf.predict(X_test)

    final_predictions = []

    for i in range(len(test)):
        predictions = [int(dt_y_pred[i]), int(rf_y_pred[i]), int(svm_y_pred[i]), int(ada_y_pred[i])]
        zero_class_count = predictions.count(0)
        one_class_count = predictions.count(1)
        if (zero_class_count>one_class_count):
            final_predictions.append(0)
        elif(one_class_count>zero_class_count):
            final_predictions.append(1)
        else:
            final_predictions.append(random.randint(0, 1))
    accuracy_score_list.append(accuracy_score(Y_test,final_predictions))
    print("Classification reporst for k=",k,": ")
    print(classification_report(Y_test, final_predictions, target_names=labels))

accuracy = sum(accuracy_score_list) / len(accuracy_score_list)
print("Average accuracy for 50 epoch: ", accuracy)




