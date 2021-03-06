import random
import cv2
import numpy as np
import albumentations as A
from albumentations import DualTransform, ImageOnlyTransform
#from albumentations.augmentations.functional import crop
from albumentations import Compose, RandomBrightnessContrast, \
    HorizontalFlip, FancyPCA, HueSaturationValue, OneOf, ToGray, \
    ShiftScaleRotate, ImageCompression, PadIfNeeded, GaussNoise, GaussianBlur
from albumentations.pytorch.functional import img_to_tensor
from albumentations.pytorch.transforms import ToTensorV2 
import torch

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized

#各向性调整大小
class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = A.crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"


def create_train_transforms2(size=300):
    return Compose([
            A.Resize(size, size, p=1),
            A.HorizontalFlip(p = 0.5),
            # A.Transpose(),
            A.OneOf([
                A.GaussNoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
                A.GlassBlur(p=0.2),
                
            ], p=0.5),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # A.IAAPiecewiseAffine(p=0.3),
                A.PiecewiseAffine(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                # A.IAASharpen(),
                # A.IAAEmboss(),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),
            
            # 灰度
            A.OneOf([
                A.Equalize(p=0.5),
                A.ToGray(p=0.2),
            ], p=0.5),
            
            # 压缩
            A.ImageCompression(quality_lower=5, quality_upper=60, p=0.1),
            # 下雨
            A.RandomRain(p=0.1),
            A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), angle_lower=0, angle_upper=1,num_flare_circles_lower=6, num_flare_circles_upper=10,src_radius=400, src_color=(255, 255, 255), p=0.2),
            # 加小块
            A.CoarseDropout(min_holes=10,max_holes=40, min_height=4,max_height=8, min_width=4,max_width=8, fill_value=0, always_apply=False, p=0.3),
            # 反色，颜色变换
            A.OneOf([
                A.InvertImg(p=0.7),
                A.Solarize(p=0.3),
            ], p=0.3),

            # 旋转缩放
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2)
            #A.Normalize(
            #    mean=[0.485, 0.456, 0.406],
            #    std=[0.229, 0.224, 0.225],
            #),
            #ToTensorV2()
        ])

def create_train_transforms(size=300):
    return Compose([
        #压缩？？？
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        #高斯噪声
        GaussNoise(p=0.1),
        #高斯模糊
        GaussianBlur(blur_limit=3, p=0.05),
        #水平翻转
        HorizontalFlip(),
        #缩放
        OneOf([
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
            IsotropicResize(max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
        ], p=1),
        #填充
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
        #随机亮度对比，？？？， 色调饱和度值
        #OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
        #变灰
        #ToGray(p=0.2),
        #旋转
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
    ]
    )

def create_val_transforms(size=300):
    return Compose([
        #cv2.INTER_AREA用像素面积关系重采样
        #cv2.INTER_CUBIC 双三次插值
        IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
        #填充，这里是填充黑色
        PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT),
    ])



def direct_val(imgs,size):
    #img 输入为RGB顺序
    imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
    transforms = create_val_transforms(size)
    normalize = {"mean": [0.485, 0.456, 0.406],
                 "std": [0.229, 0.224, 0.225]}
    imgs = [img_to_tensor(transforms(image=each)['image'],normalize).unsqueeze(0) for each in imgs]
    imgs = torch.cat(imgs)

    return imgs


if __name__ == '__main__':
    import cv2
    from albumentations.pytorch.functional import img_to_tensor
    img_path = './frame00014.png'
    #img_path = '/home/kezhiying/data/sunzhihao/dataset/forgery_face_extract_retina/Validation/image/val_perturb_release/17/b7e4952e634f722220aabec3b4212bd8/frame00036.png'
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    print(image.shape)
    image = cv2.resize(image, (0, 0), fx=1, fy=1.2, interpolation=cv2.INTER_CUBIC)
    
    #image.reshape(162,150,3)
    #print(image_resize.shape)
    print(image.shape)
    #import matplotlib.pyplot as plt
    #plt.imshow(image)
    cv2.imwrite("./img1.png", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite("./img2.png", image) 
    #transforms = create_train_transforms(size=380)
    transforms = create_train_transforms(size=380)
    transforms2 = create_train_transforms2(size=380)
    image2 = transforms(image=image)["image"]
    image3 = transforms2(image=image)["image"]
    print(image2.shape)
    print(image3.shape)
   # print((image2 == image3).all())
    normalize = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
    #from torchvision import transforms as T
    #Norm_ = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    #image = Norm_(image)
    cv2.imwrite("./img3_1.png", image2)
    cv2.imwrite("./img3_2.png", image3)
    #image = img_to_tensor(image, normalize)
    #cv2.imwrite("./img4.png", image3)
