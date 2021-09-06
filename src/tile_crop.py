###効率よくtile生成...lafossとかとniola_finalの2通りをする。
## 予めcropしてノイズが入らないように
## /home/u094724e/ダウンロード/byori/cam/make_tile_last2.py
import numpy as np 
import pandas as pd 
import glob
import openslide
import matplotlib.pyplot as plt
import cv2,math,sys,time,os
from tqdm import tqdm
import warnings
### utils ================
def _mask_tissue(image, kernel_size=(7, 7), gray_threshold=220):
    """Masks tissue in image. Uses gray-scaled image, as well as
    dilation kernels and 'gap filling'
    """
    # Define elliptic kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    # Convert rgb to gray scale for easier masking
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #gray = s
    
    # Now mask the gray-scaled image (capturing tissue in biopsy)
    #mask = np.where(gray < gray_threshold, 1, 0).astype(np.uint8)
    # Use dilation and findContours to fill in gaps/holes in masked tissue
    #mask = cv2.dilate(mask, kernel, iterations=1)
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, kernel)
    retval, mask = cv2.threshold(s, thresh=10, maxval=200, type=cv2.THRESH_BINARY)
    contour, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        cv2.drawContours(mask, [cnt], 0, 1, -1)
    return mask
def _pad_image(image, pad_len, pad_val):
    """Pads inputted image, accepts both 
    2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len)), pad_val)
    elif image.ndim == 3:
        return np.pad(
            image, ((pad_len, pad_len), (pad_len, pad_len), (0, 0)), pad_val)
    return None

def _transpose_image(image):
    """Inputs an image and transposes it, accepts 
    both 2-d (mask) and 3-d (rgb image) arrays
    """
    if image is None:
        return None
    elif image.ndim == 2:
        return np.transpose(image, (1, 0)).copy()
    elif image.ndim == 3:
        return np.transpose(image, (1, 0, 2)).copy()
    return None

def _get_tissue_parts_indices(tissue, min_consec_info):
    """If there are multiple tissue parts in 'tissue', 'tissue' will be 
    split. Each tissue part will be taken care of separately (later on), 
    and if the tissue part is less than min_consec_info, it's considered 
    to small and won't be returned.
    """
    split_points = np.where(np.diff(tissue) != 1)[0]+1
    tissue_parts = np.split(tissue, split_points)
    return [
        tp for tp in tissue_parts if len(tp) >= min_consec_info
    ]

def _get_tissue_subparts_coords(subtissue, patch_size, min_decimal_keep):
    """Inputs a tissue part resulting from '_get_tissue_parts_indices'.
    This tissue part is divided into N subparts and returned.
    Argument min_decimal_keep basically decides if we should reduce the
    N subparts to N-1 subparts, due to overflow.
    """
    start, end = subtissue[0], subtissue[-1]
    num_subparts = (end-start)/patch_size
    if num_subparts % 1 < min_decimal_keep and num_subparts >= 1:
        num_subparts = math.floor(num_subparts)
    else:
        num_subparts = math.ceil(num_subparts)

    excess = (num_subparts*patch_size) - (end-start)
    shift = excess // 2

    return [
        i * patch_size + start - shift 
        for i in range(num_subparts)
    ]

def _eval_and_append_xy_coords(coords,
                               image, 
                               mask, 
                               patch_size, 
                               x, y, 
                               min_patch_info,
                               transposed,
                               precompute):
    """Based on computed x and y coordinates of patch: 
    slices out patch from original image, flattens it,
    preprocesses it, and finally evaluates its mask.
    If patch contains more info than min_patch_info,
    the patch coordinates are kept, along with a value 
    'val1' that estimates how much information there 
    is in the patch. Smaller 'val1' assumes more info.
    """
    patch_1d = (
        image[y: y+patch_size, x:x+patch_size, :]
        .mean(axis=2)
        .reshape(-1)
    )
    idx_tissue = np.where(patch_1d <= 210)[0]
    idx_black = np.where(patch_1d < 5)[0]
    idx_background = np.where(patch_1d > 210)[0]

    if len(idx_tissue) > 0:
        patch_1d[idx_black] = 210
        patch_1d[idx_background] = 210
        val1 = int(patch_1d.mean())
        val2 = mask[y:y+patch_size, x:x+patch_size].mean()
        if val2 > min_patch_info:
            if precompute:
                if transposed:
                    coords = np.concatenate([
                        coords, [[val1, x-patch_size, y-patch_size]]
                    ])
                else:
                    coords = np.concatenate([
                        coords, [[val1, y-patch_size, x-patch_size]]
                    ])
            else:
                coords = np.concatenate([
                    coords, [[val1, y, x]]
                ])
               
    return coords

def compute_coords(image,
                   patch_size=256,
                   precompute=False,
                   min_patch_info=0.35,
                   min_axis_info=0.35,
                   min_consec_axis_info=0.35,
                   min_decimal_keep=0.7):

    """
    Input:
        image : 3-d np.ndarray
        patch_size : size of patches/tiles, will be of 
            size (patch_size x patch_size x 3)
        precompute : If True, only coordinates will be returned,
            these coordinates match the inputted 'original' image.
            If False, both an image and coordinates will be returned,
            the coordinates does not match the inputted image but the
            image that it is returned with.
        min_patch_info : Minimum required information in patch
            (see '_eval_and_append_xy_coords')
        min_axis_info : Minimum fraction of on-bits in x/y dimension to be 
            considered enough information. For x, this would be fraction of 
            on-bits in x-dimension of a y:y+patch_size slice. For y, this would 
            be the fraction of on-bits for the whole image in y-dimension
        min_consec_axis_info : Minimum consecutive x/y on-bits
            (see '_get_tissue_parts_indices')
        min_decimal_keep : Threshold for decimal point for removing "excessive" patch
            (see '_get_tissue_subparts_coords')
    
    Output:
        image [only if precompute is False] : similar to input image, but fits 
            to the computed coordinates
        coords : the coordinates that will be used to compute the patches later on
    """
    
    
    if type(image) != np.ndarray:
        # if image is a Tensor
        image = image.numpy()
    
    # masked tissue will be used to compute the coordinates
    mask = _mask_tissue(image)

    # initialize coordinate accumulator
    coords = np.zeros([0, 3], dtype=int)

    # pad image and mask to make sure no tissue is potentially missed out
    image = _pad_image(image, patch_size, 'maximum')
    mask = _pad_image(mask, patch_size, 'minimum')
    
    y_sum = mask.sum(axis=1)
    x_sum = mask.sum(axis=0)
    # if on bits in x_sum is greater than in y_sum, the tissue is
    # likely aligned horizontally. The algorithm works better if
    # the image is aligned vertically, thus the image will be transposed
    if len(np.where(x_sum > 0)[0]) > len(np.where(y_sum > 0)[0]):
        image = _transpose_image(image)
        mask = _transpose_image(mask)
        y_sum, _ = x_sum, y_sum
        transposed = True
    else:
        transposed = False
    
    # where y_sum is more than the minimum number of on-bits
    y_tissue = np.where(y_sum >= (patch_size*min_axis_info))[0]
    
    if len(y_tissue) < 1:
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    y_tissue_parts_indices = _get_tissue_parts_indices(
        y_tissue, patch_size*min_consec_axis_info)
    
    if len(y_tissue_parts_indices) < 1: 
        warnings.warn("Not enough tissue in image (y-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    # loop over the tissues in y-dimension
    for yidx in y_tissue_parts_indices:
        y_tissue_subparts_coords = _get_tissue_subparts_coords(
            yidx, patch_size, min_decimal_keep)
        
        for y in y_tissue_subparts_coords:
            # in y_slice, where x_slice_sum is more than the minimum number of on-bits
            x_slice_sum = mask[y:y+patch_size, :].sum(axis=0)
            x_tissue = np.where(x_slice_sum >= (patch_size*min_axis_info))[0]
            
            x_tissue_parts_indices = _get_tissue_parts_indices(
                x_tissue, patch_size*min_consec_axis_info)
            
            # loop over tissues in x-dimension (inside y_slice 'y:y+patch_size')
            for xidx in x_tissue_parts_indices:
                x_tissue_subparts_coords = _get_tissue_subparts_coords(
                    xidx, patch_size, min_decimal_keep)
                
                for x in x_tissue_subparts_coords:
                    coords = _eval_and_append_xy_coords(
                        coords, image, mask, patch_size, x, y, 
                        min_patch_info, transposed, precompute
                    )     
    
    if len(coords) < 1:
        warnings.warn("Not enough tissue in image (x-dim)", RuntimeWarning)
        if precompute: return [(0, 0, 0)]
        else: return image, [(0, 0, 0)]
    
    if precompute: return coords
    else: return image, coords


def crop_object(img, thresh=10, maxval=200, square=False,white=False):
    """
    Source: https://stackoverflow.com/questions/49577973/how-to-crop-the-biggest-object-in-image-with-python-opencv
    """
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)# convert to grayscale
    emp = np.ones_like(img).astype("uint8")*255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    #plt.imshow(s,cmap="gray")
    #plt.show()
    #kernel = np.ones((5,5),np.uint8)
    #s = cv2.morphologyEx(s, cv2.MORPH_OPEN, kernel)
    #dilation = cv2.dilate(s,kernel,iterations = 3)
    #dilation = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel)
    #plt.imshow(dilation,cmap="gray")
    #plt.show()
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(s, thresh=thresh, maxval=maxval, type=cv2.THRESH_BINARY)
    #plt.imshow(thresh_gray,cmap="gray")
    #plt.show()
    contours, hierarchy = cv2.findContours(thresh_gray,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #https://qiita.com/anyamaru/items/fd3d894966a98098376c
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx#(0,0,0,0)なのはcontoursに何も入ってないから
    emp[y:y+h, x:x+w] = np_img[y:y+h, x:x+w]
    
    #print(x,y,w,h)
    
    return emp

#### =====================


positive_1 = glob.glob("/home/u094724e/ダウンロード/byori/recurrence*/*ndpi")
negative = glob.glob("/home/u094724e/ダウンロード/byori/non*recurrence*/*ndpi")

df_pos_1 = pd.DataFrame()
df_pos_1["file_ndpi"] =positive_1
df_pos_1["label"] = 1
df_pos_1["patient"] = [i.split("/")[-1].split(".")[0].split("-")[0]+"-"+i.split("/")[-1].split(".")[0].split("-")[1] for i in positive_1]
df_pos_1["slice"] = [i.split("/")[-1].split(".")[0].split("-")[-1] for i in positive_1]
df_pos_1["file"] = [i.split("/")[-1][:-5] for i in positive_1]

df_nega = pd.DataFrame()
df_nega["file_ndpi"] = negative
df_nega["label"] = 0
df_nega["patient"] = [i.split("/")[-1].split(".")[0].split("-")[0]+"-"+i.split("/")[-1].split(".")[0].split("-")[1] for i in negative]
df_nega["slice"] = [i.split("/")[-1].split(".")[0].split("-")[-1] for i in negative]
df_nega["file"] = [i.split("/")[-1][:-5] for i in negative]

cat_df = pd.concat([df_nega,df_pos_1])
print(cat_df.head())

cat_df.to_csv("cat_df.csv",index=False)

#cat_df = cat_df.iloc[:5,:]

patch_size=256
#patch_size=128*(2**2)
df_samples = []
mini_patch = [[0,0],[128,0],[0,128],[128,128]]
#mini_patch = [[0,0],[128,0],[256,0],[384,0],[0,128],[128,128],[256,128],[384,128],[0,256],[128,256],[256,256],[384,256],[0,384],[128,384],[256,384],[384,384]]
df_Iafoss = pd.DataFrame()
files_iafoss = []
files_iafoss_png = []
for i in tqdm(range(len(cat_df))):
    df_sample = pd.DataFrame()
    file_ndpi = cat_df["file_ndpi"].values[i]
    file_ = cat_df["file"].values[i]
    slide = openslide.OpenSlide(file_ndpi)
    pil_img = slide.read_region((0,0),3,slide.level_dimensions[3])
    np_img = np.asarray(pil_img)[:,:,:3]
    emp = np.zeros_like(np_img).astype("uint8")
    np_img = crop_object(np_img, thresh=10, maxval=200)
    coords = compute_coords(np_img,patch_size=patch_size,precompute=True,min_patch_info=0.35,min_axis_info=0.35,min_consec_axis_info=0.35,min_decimal_keep=0.7)
    coords = sorted(coords, key= lambda x: x[0], reverse=False)
    np_y = []
    np_x = []
    file_pngs = []
    file_tiles = []

    for (v, y, x) in coords:
        end_point = (x, y)
        start_point = (x+patch_size, y+patch_size)
        #image = cv2.rectangle(image, start_point, end_point, 2, 14)
        tile_l = np_img[y:y+patch_size,x:x+patch_size,:]
        if tile_l.shape[0]!=patch_size or tile_l.shape[1]!=patch_size:continue
        file_tiles.append(tile_l)#512*512のtileをいれておく。
        for j,mini in enumerate(mini_patch):
            tile = tile_l[mini[0]:mini[0]+128,mini[1]:mini[1]+128,:]
            np_y.append(y+mini[0])
            np_x.append(x+mini[1])
            emp[y+mini[0]:y+mini[0]+128,x+mini[1]:x+mini[1]+128]=tile

            tile_path = f"/home/u094724e/ダウンロード/byori/tiles_crop_256_2_128_new/{file_}_y{y+mini[0]}_x{x+mini[1]}_3x.png"
            file_pngs.append(tile_path)
            try:
                cv2.imwrite(tile_path,tile)
            except Exception as e:
                print("error".tile.shape,tile_path)
                image = cv2.rectangle(np_img.astype(np.float32), start_point, end_point, 2, 14)
                #plt.imshow(image)
                #plt.show()
                #print(tile.shape,tile_l.shape,np_img.shape,x,y)
    
    """
    plt.imshow(emp)
    plt.savefig(f"tile_test/tile_test_{i}.png")
    plt.imshow(np_img)
    plt.savefig(f"tile_test/test_{i}.png")"""
    
    #if i==10:sys.exit()
    df_sample["np_y"]= np_y
    df_sample["np_x"]= np_x
    df_sample["file_png"]= file_pngs
    df_sample["file"]= file_
    df_samples.append(df_sample)



    """

    files_iafoss.append(file_)
    file_tiles = np.concatenate(file_tiles)
    iafoss_path = f"/home/u094724e/ダウンロード/byori/3x_512_tiles_iafoss/{file_}_iafoss.npy"
    files_iafoss_png.append(iafoss_path)
    np.save(iafoss_path,file_tiles)
    """



df_samples = pd.concat(df_samples)
df_all = df_samples.merge(cat_df,on="file")
df_all.to_csv("df_3x_256_2_128_tiles_cropeed_new.csv",index = False)

print(df_all.shape)
for idx in tqdm(range(len(df_all))):
    file_path = df_all["file_png"].values[idx]
    image = cv2.imread(file_path)
    image = cv2.resize(image,(128,128))
