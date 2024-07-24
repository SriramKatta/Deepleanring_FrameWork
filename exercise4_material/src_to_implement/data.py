from torch.utils.data import Dataset
import skimage as sk
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self._transform = None
        TF = tv.transforms
        self.transform_val = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std),
                                    ])
        self.transform_train = TF.Compose([  
                                    TF.ToPILImage(),
                                    TF.ToTensor(),
                                    TF.Normalize(train_mean, train_std)
                                    ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        img = sk.io.imread(data['filename'], as_gray=True)
        img = sk.color.gray2rgb(img)
        label = np.array([data['crack'], data['inactive']])
        if self.mode == "val":
            img = self.transform_val(img)
        elif self.mode == "train":
            img = self.transform_train(img)
        else:
            raise ValueError("Mode must be 'val' or 'train")
        return img, label
