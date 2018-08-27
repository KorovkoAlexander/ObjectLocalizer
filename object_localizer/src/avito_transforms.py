import random
import numpy as np

def random_crop(img, target):
    h, w,_ = img.shape
    xmin = int(target[0]*w)
    ymin = int(target[1]*h)
    xmax = int(target[2]*w)
    ymax = int(target[3]*h)
    
    i = random.randint(0, xmin)
    j = random.randint(0, ymin)
    k = random.randint(xmax, w)
    l = random.randint(ymax, h)
    
    xmin-=i
    ymin-=j
    xmax-=i
    ymax-=j
    
    img = img[j:l, i:k]
    if img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError()
    
    target = np.array([
                float(xmin)/img.shape[1],
                float(ymin)/img.shape[0],
                float(xmax)/img.shape[1],
                float(ymax)/img.shape[0]
            ])
    return img, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
    
class RandomRotate(object):
    def __call__(self, image, target):
        i = random.random()
        if i < 0.25:
            image = np.rot90(image, k = 1)
            target = np.array([
                target[1],
                1 - target[0],
                target[3],
                1 - target[2],
            ])
        elif i < 0.5:
            image = np.rot90(image, k = 2)
            target = np.array([
                1 - target[0],
                1 - target[1],
                1 - target[2],
                1 - target[3],
            ])
        elif i < 0.75:
            image = np.rot90(image, k = 3)
            target = np.array([
                1 - target[1],
                target[0],
                1 - target[3],
                target[2],
            ])
            
        target = np.array([
                min(target[0], target[2]),
                min(target[1], target[3]),
                max(target[0], target[2]),
                max(target[1], target[3]),
            ])
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'
    
class RandomCrop(object):       
    def __call__(self, image, target):
        image, target = random_crop(image, target)
        return image, target

    def __repr__(self):
        return self.__class__.__name__ + '()'
    