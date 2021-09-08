import cv2
import horovod.torch as hvd
import numpy as np
from PIL import Image, ImageFile
from random import shuffle
import torch
from torch.nn import parameter
from torch.utils import data
from torchvision import transforms as T
from dataset.transforms import create_train_transforms, create_val_transforms
from dataset.transforms import create_train_transforms2
from albumentations.pytorch.functional import img_to_tensor
from tools import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_infer_image(infer_path):
    image = Image.open(infer_path).convert("RGB")
    Transform = []
    Transform.append(T.Resize((256, 256)))
    Transform.append(T.CenterCrop(224))
    Transform.append(T.ToTensor())
    Transform = T.Compose(Transform)
    image = Transform(image)
    Norm_ = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    image = Norm_(image)
    image = image[None, ...]
    #image: 1x3x224x224
    return image


class ImageFolder(data.Dataset):
    def __init__(self, data_list_path, mode='train', balance=False, model_params=None, dataset_params=None):
        self.data_list = utils.read_data(data_list_path)
        self.data_num = len(self.data_list)
        self.mode = mode
        self.balance = balance
        self.num_classes = model_params['num_classes']
        self.categorys = dataset_params['categorys']
		
        if self.balance and self.mode == 'train':
            self.data = [[x for x in self.data_list if x[1] == i]
                         for i in range(self.num_classes)]
            for i in range(self.num_classes):
                print("category {}: {}".format(
                    self.categorys[i], len(self.data[i])))
        else: #with label or without label
            self.data = [self.data_list]
            print("all: %d" % len(self.data[0]))
        print("image count in {} path :{}".format(mode, self.data_num))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        result_images = []
        result_labels = []
        result_paths = []
        if self.balance and self.mode == 'train':
            for i in range(self.num_classes):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                image = self.load_sample(img_path)
                label = i
                #result_images = torch.cat(result_images, image)
                result_images.append(image)
                result_labels.append(label)
                result_paths.append(img_path)
        elif self.mode == 'infer':
            img_path = self.data[0][index][0]
            image = self.load_sample(img_path)
            result_images.append(image)
            result_paths.append(img_path)
            return torch.cat([x.unsqueeze(0) for x in result_images]), result_paths
        else:
            label = self.data[0][index][1]
            img_path = self.data[0][index][0]
            image = self.load_sample(img_path)
            result_images.append(image)
            result_labels.append(label)
            result_paths.append(img_path)
		#result_image: n x 3 x 244 x 244,  result_label: n,  result_paths: [...] len(n)
        #return torch.cat([x.unsqueeze(0) for x in result_images]),  torch.tensor(result_labels), result_paths
        return torch.stack(result_images, dim=0), torch.tensor(np.array(result_labels)), result_paths

    def transforms2(self, img_path):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.mode == 'train':
            transforms = create_train_transforms2(size=380)
        elif self.mode == 'test' or self.mode == 'valid' or self.mode == 'infer':
            transforms = create_val_transforms(size=380)
        image = transforms(image=image)["image"]
        normalize = {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
        image = img_to_tensor(image, normalize)
        return image
    
    def transforms1(self, img_path):
        image = Image.open(img_path)
        image_channels = len(image.split())
        if image_channels != 3:
            image = Image.open(img_path).convert("RGB")
        Transform = []
        if self.mode == 'train':
            Transform.append(T.Resize((256, 256)))
            Transform.append(T.RandomResizedCrop(224))
            Transform.append(T.RandomHorizontalFlip())
        elif self.mode == 'test' or self.mode == 'valid' or self.mode == 'infer':
            Transform.append(T.Resize((256, 256)))
            Transform.append(T.CenterCrop(224))

        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        image = Transform(image)
        
        Norm_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        image = Norm_(image)
        return image

    def load_sample(self, img_path):
        #image = self.transforms1(img_path)     
        image = self.transforms2(img_path)
        return image

    def __len__(self):
        return self.data_num

#@profile
def get_loader(data_list_path, batch_size, shuffle=True, num_workers=1,
               mode='train', balance=False, model_params=None, dataset_params=None, parallel_type=''):
    """Builds and returns Dataloader."""
    dataset = ImageFolder(data_list_path=data_list_path, mode=mode,
                          balance=balance, model_params=model_params, dataset_params=dataset_params)
    #Distributed_6: 使用分布式生成数据采样器，分布式训练要求每个卡分布的数据量相同
    if parallel_type == 'Distributed' or parallel_type == 'Distributed_Apex':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=sampler)
    elif parallel_type == 'Horovod':
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=hvd.size(),
                                                                    rank=hvd.rank())
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=sampler)
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers
                                  )
    return data_loader
