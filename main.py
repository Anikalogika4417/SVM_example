import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets, svm, metrics

digits = datasets.load_digits()

images_and_lables = list(zip(digits.images, digits.target))

#for index, (image, lable) in enumerate(images_and_lables[:6]):
    #plt.subplot(2,3, index+1)
    #plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #plt.title('Target: %i' % lable)

#plt.show()

data = digits.images.reshape((len(digits.images), -1))


classifier = svm.SVC(gamma=0.001)

train_test_split = int(len(digits.images) * 0.75)
classifier.fit(data[:train_test_split], digits.target[:train_test_split])

expected = digits.target[train_test_split:]
prediction = classifier.predict(data[train_test_split:])

print(confusion_matrix(expected, prediction))
print(accuracy_score(expected, prediction))
