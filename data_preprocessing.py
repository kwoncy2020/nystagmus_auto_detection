from email import iterators
import os, glob, cv2, sys, re, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from my_feature_extractor import FeatureExtractor
from my_eval_tool import Eval_tool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
FOLDER_PATH = "C:\kwoncy\eye\image_datas\\6_sixth_make_roi_with_bad_video_images\\test_with_bad_imgs"
CURRENT_DEVICE = "/device:CPU:0"
MODEL_NAME = "new2_pupil3_320x240_6912_E30_B5_R4444_S9709.h5"

## the folder must have like below condition.
## index_orig.png , index_mask.png
def load_orig_color_and_mask_gray_imgs(folder_path:str)->'list[np.ndarray, np.ndarray]':
    mask_file_list = glob.glob(f'{folder_path}/*_mask.png')
    re_mask = re.compile('([0-9]+)_mask.png')
    mask_imgs = []
    orig_imgs = []

    for mask_file in mask_file_list:
        head, tail = os.path.split(mask_file)
        num_str = re_mask.match(tail).group(1)
        pred_file = os.path.join(head, num_str + "_orig.png")

        mask_imgs.append(cv2.imread(mask_file,cv2.IMREAD_GRAYSCALE))
        orig_imgs.append(cv2.imread(pred_file,cv2.IMREAD_COLOR))

    return [np.array(orig_imgs), np.array(mask_imgs)]


def get_clahe_img(img:np.ndarray, clipLimit:float=2.0, tileGridSize:tuple=(4,4)):
    if np.ndim(img) != 2:
        raise Exception(f'from get_clahe_img: dimension failed. np.ndim-img-{np.ndim(img)} != 2')
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    
    img_clahe = clahe.apply(img)
    return img_clahe


def get_clahe_imgs(imgs:np.ndarray, clipLimit:float=2.0, tileGridSize:tuple=(4,4))->np.ndarray:
    for idx, img in enumerate(imgs):
        clahe_img = get_clahe_img(img,clipLimit,tileGridSize)
        imgs[idx] = clahe_img
    return imgs


def image_augmentation(imgs:np.ndarray,masks:np.ndarray, multiplier:int=1, flag_imshow:bool=False)->'list[np.ndarray, np.ndarray]':
    print("imgs shape", imgs.shape, "masks shape", masks.shape)
    if multiplier < 1: multiplier = 1
    if np.ndim(masks) == 3:
        masks = masks[:,:,:,np.newaxis]
    if np.ndim(imgs) != 4 or np.ndim(masks) != 4:
        raise Exception(f"from image_augmentation: the dimension of imgs({np.ndim(imgs)}) and masks({np.ndim(masks)}) must be 4")
    
    img_channel = imgs.shape[-1]
    mask_channel = masks.shape[-1]
    imgs_masks = np.concatenate([imgs,masks], axis=-1)
    # imgs_masks_flip = imgs_masks[:,:,::-1,:]
    # imgs_masks_noflip_flip = np.concatenate([imgs_masks,imgs_masks_flip],axis=0)
    
    generator = ImageDataGenerator(rescale=1/255., rotation_range=15, width_shift_range=0.05, height_shift_range=0.05, horizontal_flip=True, data_format="channels_last")
    # iterators_ = generator.flow(imgs_masks_noflip_flip)
    iterators_ = generator.flow(imgs_masks, batch_size=imgs.shape[0])
    multi = []
    for i in range(multiplier):
        augmented = iterators_.next()
        augmented = (augmented * 255).astype(np.uint8)
        multi.append(augmented)
    
    augmented = np.concatenate(multi,axis=0)
    augmented_imgs = augmented[:,:,:,:img_channel]
    augmented_masks = augmented[:,:,:,img_channel:img_channel+mask_channel]

    # comp = imgs[0:3,:,:,:].astype(np.uint8)
    # # cv2.imshow("img",np.stack(np.concatenate([augmented_imgs,comp],axis=0), axis=0))
    # cv2.imshow("img",np.concatenate([augmented_imgs[0],augmented_imgs[1], augmented_imgs[2], comp[0], comp[1], comp[2]],axis=1))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print("augmented_imgs_shape: ",augmented_imgs.shape)
    if flag_imshow:
        print("flag_imshow == True : push space if you want escape or push other key to see next image and mask")
        for i in range(len(augmented_imgs)):
            img = augmented_imgs[i]
            mask = augmented_masks[i]
            mask = np.squeeze(mask)

            img[mask==255] += np.array([0,0,60], dtype=np.uint8)

            cv2.imshow("imgs", np.concatenate([img,cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)],axis=1))
            k = cv2.waitKey(0)
            if k == 47:
                break
            else:
                continue

        cv2.destroyAllWindows()

    return [augmented_imgs, augmented_masks]


def save_augmented_images(folder_path:str, augmented_imgs:np.ndarray, augmented_masks:np.ndarray) -> None:
    head, tail = os.path.split(folder_path)
    dir_path = os.path.join(head,f"{tail}_augmented")
    if os.path.exists(dir_path):
        raise Exception(f"from save_augmented_images: the dir_path({dir_path}) already exists")
    else:
        os.mkdir(dir_path)
        for i in range(len(augmented_imgs)):
            cv2.imwrite(f"{dir_path}/{i}_orig.png", augmented_imgs[i])
            cv2.imwrite(f"{dir_path}/{i}_mask.png", augmented_masks[i])

def preprocessing(folder_path:str) -> None:
    orig_imgs, mask_imgs = load_orig_color_and_mask_gray_imgs(folder_path)
    # orig_imgs_clahe = get_clahe_imgs(orig_imgs)
    augmented_imgs, augmented_masks = image_augmentation(orig_imgs,mask_imgs,1)
    save_augmented_images(folder_path,augmented_imgs,augmented_masks)

if __name__ == '__main__':
    preprocessing(FOLDER_PATH)