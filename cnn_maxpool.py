import mnist
from conv import Conv3x3
from maxpool import MaxPool2

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8)
pool = MaxPool2()

output = conv.forward(train_images[0])
output = pool.forward(output)
print(output.shape) # (13, 13, 8)