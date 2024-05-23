import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import random
import skimage

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.batch_start = 0
        self.epoch_num = 0

        labelfptr = open(self.label_path)
        self.labels = json.load(labelfptr)
        sortedlabels = dict(sorted(self.labels.items(), key = lambda x : int(x[0])))
        self.labels = list(sortedlabels.values())
        
        self.images = []
        
        for i in range(len(self.labels)):
            img = np.load(self.file_path + str(i) + ".npy")
            img = skimage.transform.resize(img,self.image_size)
            self.images.append(img)
        
        self.remainder = len(self.labels) % self.batch_size     
        

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        if(self.shuffle):
            combilist = list(zip(self.images,self.labels))
            random.shuffle(combilist)
            self.images , self.labels = zip(*combilist)

        if(self.remainder != 0):
            self.images.extend(self.images[:self.batch_size-self.remainder])
            self.labels.extend(self.labels[:self.batch_size-self.remainder])

        start = self.batch_start
        if(start >= len(self.images)):
            self.batch_start = 0
            start = 0
            self.epoch_num += 1
        end = start + self.batch_size
        self.batch_start = end

        imageres = [self.augment(img) for img in self.images[start:end]]
                
        return np.array(imageres), np.array(self.labels[start:end])
        

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function
        if(self.rotation):
            img = np.rot90(img, random.randint(0,3), axes=(0,1)) ## added case to no rot
        if(self.mirroring):
            img = np.flip(img,axis=0)
        return img

    def current_epoch(self):
        # return the current epoch number
        return self.epoch_num

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return self.class_dict[x]
    
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        xfigcount = self.batch_size // 3
        yfigcount = self.batch_size // xfigcount
        images, labels = self.next()
        
        #fig = plt.figure()
        loc ,ax = plt.subplots(xfigcount,yfigcount)
        for i in range(xfigcount):
            for j in range(yfigcount):
                ax[i,j].imshow(images[i * yfigcount + j])
                ax[i,j].set_title(self.class_name(labels[i * yfigcount + j ]))
                ax[i,j].axis('off')
        plt.show()




