import mnist
from Test import Conv3x3

# The mnist package handles the MNIST dataset for us!
# Learn more at https://github.com/datapythonista/mnist
train_images = mnist.train_images()
train_labels = mnist.train_labels()

conv = Conv3x3(8,)
output = conv.forward(train_images[0])
print(output.shape) # (26, 26, 8)