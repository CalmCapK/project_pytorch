import albumentations as A
# from albumentations.pytorch import ToTensor
# from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from albumentations.pytorch.transforms import ToTensorV2 


# normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
class Custom_aug():
    def __init__(self, args):
        self.trans1 =  A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),
            A.HorizontalFlip(p = 0.5),
            # A.Transpose(),
            A.OneOf([
                # A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # A.IAAPiecewiseAffine(p=0.3),
                A.PiecewiseAffine(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                # A.IAASharpen(),
                A.Sharpen(),
                # A.IAAEmboss(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        self.trans3 =  A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),
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
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        

        # random.seed(0)
        self.trans5 = A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),
            A.OneOf([
                A.ChannelDropout(p=0.05),
                A.ChannelShuffle(p=0.05),
                A.Equalize(p=0.05),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, 
                                    p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.1),
                A.ToGray(p=0.2),
                A.ToSepia(p=0.05),
            ], p=0.5),
            A.RandomBrightnessContrast(),
            A.OneOf([
                A.Blur(blur_limit=[7, 15], p=0.1),
                A.GaussianBlur(p=0.1),
                A.GlassBlur(p=0.2),
                A.Downscale(scale_min=0.25, scale_max=0.25, p=0.15),
                A.ImageCompression(quality_lower=5, quality_upper=60, p=0.1),
                A.ISONoise(color_shift=(0.05, 0.1), intensity=(0.2, 0.5), p=0.05),
                A.GaussNoise(p=0.1),
                A.RandomGamma(p=0.04),
                A.Posterize(p=0.2),
                # A.Sharpen(p=0.05),
                # A.Superpixels(p=0.02),
            ], p=0.5),
            A.OneOf([
                A.InvertImg(p=0.7),
                A.Solarize(p=0.3),
            ], p=0.1),
            A.OneOf([
                A.ElasticTransform(p=0.15),
                A.GridDistortion(p=0.2),
                # A.PiecewiseAffine(p=0.65),
            ], p=0.1),
            A.CoarseDropout(max_holes=16, max_height=10, max_width=10, min_holes=8,
                            min_height=10, min_width=10, fill_value=0, p=0.1),
            A.RandomRain(p=0.06),
            A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), angle_lower=0, angle_upper=1,
                            num_flare_circles_lower=6, num_flare_circles_upper=10,
                            src_radius=400, src_color=(255, 255, 255), p=0.08),
            # A.RandomShadow(p=0.05),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        self.trans4 =  A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),

            A.OneOf([
                A.ChannelDropout(p=0.05),
                A.ChannelShuffle(p=0.05),
                A.Equalize(p=0.05),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                    p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.1),
                A.ToGray(p=0.2),
                A.ToSepia(p=0.05),
            ], p=0.5),
            A.RandomBrightnessContrast(),
            A.OneOf([
                A.Blur(blur_limit=[7, 15], p=0.1),
                A.GaussianBlur(p=0.1),
                A.GlassBlur(p=0.2),
                A.Downscale(scale_min=0.25, scale_max=0.25, p=0.15),
                A.ImageCompression(quality_lower=5, quality_upper=60, p=0.1),
                A.ISONoise(color_shift=(0.05, 0.1), intensity=(0.2, 0.5), p=0.05),
                A.GaussNoise(p=0.1),
                A.RandomGamma(p=0.04),
                A.Posterize(p=0.2),
                A.Sharpen(p=0.05),
                A.Superpixels(p=0.02),
            ], p=0.5),
            A.OneOf([
                A.InvertImg(p=0.7),
                A.Solarize(p=0.3),
            ], p=0.1),
            A.OneOf([
                A.ElasticTransform(p=0.25),
                A.GridDistortion(p=0.3),
                A.PiecewiseAffine(p=0.45),
            ], p=0.1),
            A.CoarseDropout(max_holes=16, max_height=10, max_width=10, min_holes=8,
                            min_height=10, min_width=10, fill_value=0, p=0.1),
            A.CoarseDropout(max_holes=16, max_height=10, max_width=10, min_holes=8,
                            min_height=8, min_width=8, fill_value=0, p=0.02),
            A.RandomRain(p=0.06),

            A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), angle_lower=0, angle_upper=1,
                            num_flare_circles_lower=6, num_flare_circles_upper=10,
                            src_radius=400, src_color=(255, 255, 255), p=0.08),
            # A.RandomShadow(p=0.05),

            A.HorizontalFlip(p = 0.5),
            # A.Transpose(),
            A.OneOf([
                # A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # A.IAAPiecewiseAffine(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                # A.IAASharpen(),
                # A.IAAEmboss(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

        self.trans_raw_aug =  A.Compose([
            A.OneOf([
                A.ChannelDropout(p=0.05),
                A.ChannelShuffle(p=0.05),
                A.Equalize(p=0.05),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20,
                                    p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.1),
                A.ToGray(p=0.2),
                A.ToSepia(p=0.05),
            ], p=0.5),
            A.RandomBrightnessContrast(),
            A.OneOf([
                A.Blur(blur_limit=[7, 15], p=0.1),
                A.GaussianBlur(p=0.1),
                A.GlassBlur(p=0.2),
                A.Downscale(scale_min=0.25, scale_max=0.25, p=0.15),
                A.ImageCompression(quality_lower=5, quality_upper=60, p=0.1),
                A.ISONoise(color_shift=(0.05, 0.1), intensity=(0.2, 0.5), p=0.05),
                A.GaussNoise(p=0.1),
                A.RandomGamma(p=0.04),
                A.Posterize(p=0.2),
                A.Sharpen(p=0.05),
                A.Superpixels(p=0.02),
            ], p=0.5),
            A.OneOf([
                A.InvertImg(p=0.7),
                A.Solarize(p=0.3),
            ], p=0.1),
            A.OneOf([
                A.ElasticTransform(p=0.25),
                A.GridDistortion(p=0.3),
                A.PiecewiseAffine(p=0.45),
            ], p=0.1),
            A.CoarseDropout(max_holes=16, max_height=10, max_width=10, min_holes=8,
                            min_height=10, min_width=10, fill_value=0, p=0.1),
            A.CoarseDropout(max_holes=16, max_height=10, max_width=10, min_holes=8,
                            min_height=8, min_width=8, fill_value=0, p=0.02),
            A.RandomRain(p=0.06),

            A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9), angle_lower=0, angle_upper=1,
                            num_flare_circles_lower=6, num_flare_circles_upper=10,
                            src_radius=400, src_color=(255, 255, 255), p=0.08),
            # A.RandomShadow(p=0.05),

            A.HorizontalFlip(p = 0.5),
            # A.Transpose(),
            A.OneOf([
                # A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                # A.IAAPiecewiseAffine(p=0.3),
            ], p=0.5),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                # A.IAASharpen(),
                # A.IAAEmboss(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.5),
            A.HueSaturationValue(p=0.3),

            # A.Resize(args.input_size, args.input_size, p=1),
            # A.Normalize(
            #     mean=[0.485, 0.456, 0.406],
            #     std=[0.229, 0.224, 0.225],
            # ),
            # ToTensorV2()
        ])


        self.valid_trans = A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

        self.valid_trans_tta1 =  A.Compose([
            A.Resize(args.input_size, args.input_size, p=1),
            # A.HorizontalFlip(p = 1),
            # # A.Transpose(),
            # A.OneOf([
            #     # A.IAAAdditiveGaussianNoise(),
            #     A.GaussNoise(),
            # ], p=1),
            # A.OneOf([
            #     A.MotionBlur(p=0.2),
            #     A.MedianBlur(blur_limit=3, p=0.1),
            #     A.Blur(blur_limit=3, p=0.1),
            # ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            # A.OneOf([
            #     A.OpticalDistortion(p=0.3),
            #     A.GridDistortion(p=0.1),
            #     # A.IAAPiecewiseAffine(p=0.3),
            #     A.PiecewiseAffine(p=0.3),
            # ], p=0.5),
            # A.OneOf([
            #     A.CLAHE(clip_limit=2),
            #     # A.IAASharpen(),
            #     A.Sharpen(),
            #     # A.IAAEmboss(),
            #     A.Emboss(),
            #     A.RandomBrightnessContrast(),
            # ], p=0.2),
            A.Sharpen(p=1),
            # A.HueSaturationValue(p=0.3),
            
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2()
        ])

if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    class args():
        input_size=299
    args = args()
    image_path = "/data/fanglingfei/workspace/FFAC1/net/cat.jpg"
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    abc = Custom_aug(args)
    for i in range(0,100):
        augmented = abc.trans4(image=image_np)            
        # sample = augmented['image']
        # Convert numpy array to PIL Image
        sample = Image.fromarray(augmented['image'])
        sample.save('temp.png')

