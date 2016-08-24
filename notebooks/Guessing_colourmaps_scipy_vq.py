from pylab import imread,imshow,figure,show,subplot
from numpy import reshape,uint8,flipud
from scipy.cluster.vq import kmeans,vq

img = imread('clearsky.jpg')

# reshaping the pixels matrix
pixel = reshape(img,(img.shape[0]*img.shape[1],3))

# performing the clustering
centroids,_ = kmeans(pixel,6) # six colors will be found
# quantization
qnt,_ = vq(pixel,centroids)

# reshaping the result of the quantization
centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))
clustered = centroids[centers_idx]

figure(1)
subplot(211)
imshow(flipud(img))
subplot(212)
imshow(flipud(clustered))
show()
