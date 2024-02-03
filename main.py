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
