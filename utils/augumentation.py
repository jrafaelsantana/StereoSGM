import cv2
import random
import numpy as np
import math
import torch
import time
import kornia as K

def fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img
        
def horizontal_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w*ratio
    if ratio > 0:
        img = img[:, :int(w-to_shift), :]
    if ratio < 0:
        img = img[:, int(-1*to_shift):, :]
    img = fill(img, h, w)
    return img

def vertical_shift(img, ratio=0.0):
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return img
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h*ratio
    if ratio > 0:
        img = img[:int(h-to_shift), :, :]
    if ratio < 0:
        img = img[int(-1*to_shift):, :, :]
    img = fill(img, h, w)
    return img

def brightness(img, low, high):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:,:,1] = hsv[:,:,1]*value
    hsv[:,:,1][hsv[:,:,1]>255]  = 255
    hsv[:,:,2] = hsv[:,:,2]*value 
    hsv[:,:,2][hsv[:,:,2]>255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def zoom(img, value):
    if value > 1 or value < 0:
        print('Value for zoom should be less than 1 and greater than 0')
        return img
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value*h)
    w_taken = int(value*w)
    h_start = random.randint(0, h-h_taken)
    w_start = random.randint(0, w-w_taken)
    img = img[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
    img = fill(img, h, w)
    return img

def channel_shift(img, value):
    value = int(random.uniform(-value, value))
    img = img + value
    img[:,:,:][img[:,:,:]>255]  = 255
    img[:,:,:][img[:,:,:]<0]  = 0
    img = img.astype(np.uint8)
    return img

def horizontal_flip(img):
    return cv2.flip(img, 1)

def vertical_flip(img):
    return cv2.flip(img, 0)

def rotation(img, angle):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img

'''def make_patch (src, win_size, x, y, device, scale=(1.0,1.0), phi=-0.05, trans=(0.1,0.1), hshear=0.2, brightness=0.0, contrast=1.0, flipH = 1, channel_size = 3):
    data = src.unsqueeze(0)

    _, ch, rows, cols = data.shape 
    c = math.cos(phi)
    s = math.sin(phi)       
    
    jmat = torch.FloatTensor([[1, 0, -x], [0, 1, -y], [0, 0, 1]]).to(device)
    rmat = torch.FloatTensor([[c, s, 0], [-s, c, 0], [0, 0, 1]]).to(device)
    cmat = torch.FloatTensor([[1, hshear, 0], [0, 1, 0], [0, 0, 1]]).to(device)
    smat = torch.FloatTensor([[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]).to(device)
    tmat = torch.FloatTensor([[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]]).to(device)
    jfmat = torch.FloatTensor([[1, 0, (win_size[0] - 1) / 2], [0, 1, (win_size[1] -1 ) / 2], [0, 0, 1]]).to(device)   
    
    amat = torch.mm(tmat,jmat)        
    amat = torch.mm(smat,amat)
    amat = torch.mm(cmat,amat)
    amat = torch.mm(rmat,amat)        
    amat = torch.mm(jfmat,amat)
    amat_ = amat[:2, :]
    amat_ - amat_.unsqueeze_(0)
    
    #dst = cv2.warpAffine(src, amat_, (cols,rows))
    dst = K.warp_affine(data.float(), amat_, dsize=(cols, rows)).to(device)
    dst = dst * contrast
    dst = dst + brightness
    dst = dst.squeeze()

    if channel_shift == 3:
        dst = dst[:, 0:win_size[1], 0:win_size[0]]
    else:
        dst = dst[0:win_size[1], 0:win_size[0]]
    
    if(flipH == -1):            
        fmat = torch.FloatTensor([[flipH, 0, win_size[0]-1], [0, 1, 0], [0, 0, 1]]).to(device)                
        amat = fmat
        amat_ = amat[:2, :]
        amat_ - amat_.unsqueeze_(0)
        dst = K.warp_affine(data.float(), amat_, dsize=(win_size[0],win_size[1]))
        dst = dst.squeeze()
        #dst = cv2.warpAffine(dst, amat_, (win_size[0],win_size[1]))
    
    if len(dst.shape) == 2:
        dst = dst.unsqueeze(0)

    return dst'''

def make_patch (src, win_size, x, y, device, scale=(1.0,1.0), phi=-0.05, trans=(0.1,0.1), hshear=0.2, brightness=0.0, contrast=1.0, flipH = 1, channel_size = 3):
    src = src.numpy().squeeze()

    rows, cols = src.shape

    c = math.cos(phi)
    s = math.sin(phi)       
    
    jmat = [[1, 0, -x], [0, 1, -y], [0, 0, 1]]
    rmat = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    cmat = [[1, hshear, 0], [0, 1, 0], [0, 0, 1]]
    smat = [[scale[0], 0, 0], [0, scale[1], 0], [0, 0, 1]]
    tmat = [[1, 0, trans[0]], [0, 1, trans[1]], [0, 0, 1]]
    jfmat = [[1, 0, (win_size[0] - 1) / 2], [0, 1, (win_size[1] -1 ) / 2], [0, 0, 1]]
    
    amat = np.matmul(tmat, jmat)
    amat = np.matmul(smat, amat)
    amat = np.matmul(cmat, amat)
    amat = np.matmul(rmat, amat)       
    amat = np.matmul(jfmat, amat)
    amat_ = amat[:2, :]
    
    dst = cv2.warpAffine(src, amat_, (cols,rows))
    #dst = K.warp_affine(data.float(), amat_, dsize=(cols, rows))
    dst = dst * contrast
    dst = dst + brightness

    if channel_size == 3:
        dst = dst[:, 0:win_size[1], 0:win_size[0]]
    else:
        dst = dst[0:win_size[1], 0:win_size[0]]
    
    if(flipH == -1):            
        fmat = np.array([[flipH, 0, win_size[0]-1], [0, 1, 0], [0, 0, 1]])                
        amat = fmat
        amat_ = amat[:2, :]
        #dst = K.warp_affine(data.float(), amat_, dsize=(win_size[0],win_size[1]))
        dst = cv2.warpAffine(dst, amat_, (win_size[0],win_size[1]))
    
    if len(dst.shape) == 2:
        dst = np.expand_dims(dst, axis=0)

    dst = torch.FloatTensor(dst)

    return dst

def mul32 (a,b):
    return [a[0]*b[0]+a[1]*b[3], a[0]*b[1]+a[1]*b[4], a[0]*b[2]+a[1]*b[5]+a[2], a[3]*b[0]+a[4]*b[3], a[3]*b[1]+a[4]*b[4], a[3]*b[2]+a[4]*b[5]+a[5]]

def make_patch_cv2 (src, ws, dim3, dim4, scale=(1.0,1.0), phi=-0.05, trans=(0.1,0.1), hshear=0.2, brightness=0.0, contrast=1.0):
    ch, rows, cols = src.shape
    src = src.squeeze().numpy()

    m = [1, 0, -dim4, 0, 1, -dim3]
    m = mul32([1, 0, trans[0], 0, 1, trans[1]], m) #translate
    m = mul32([scale[0], 0, 0, 0, scale[1], 0], m) #scale

    c = math.cos(phi)
    s = math.sin(phi)

    m = mul32([c, s, 0, -s, c, 0], m)   #rotate
    m = mul32([1, hshear, 0, 0, 1, 0], m)   #shear
    m = mul32([1, 0, (ws - 1) / 2, 0, 1, (ws - 1) / 2], m)
    m = np.array(m, dtype=np.float64)
    m = np.reshape(m, (-1,3))

    warp = cv2.warpAffine(src, m, (cols, rows))
    warp = warp * contrast
    warp = warp + brightness

    warp = torch.FloatTensor(warp).unsqueeze(0)

    return warp

def generate_patch (src, patch_size, x, y):
    ch, rows, cols = src.shape
    
    x = int(x)
    y = int(y)
    
    window = torch.zeros((ch, patch_size, patch_size))
    center_height = int(patch_size/2)

    pad_l = 0 
    pad_r = 0
    pad_t = 0
    pad_b = 0

    if x - center_height < 0:
        pad_l = abs(x - center_height)
    
    if x + center_height + 1 > cols:
        pad_r = abs((x + center_height + 1) - cols)

    if y - center_height < 0:
        pad_t = abs(y - center_height)

    if y + center_height + 1 > rows:
        pad_b = abs((y + center_height + 1) - rows)

    if pad_l > patch_size or pad_r > patch_size or pad_t > patch_size or pad_b > patch_size:
        return window

    crop = src[:, max(0, (y-center_height)):min(rows, (y+center_height+1)), max(0, (x-center_height)):min(cols, (x+center_height+1))]
    window[:, pad_t:patch_size-pad_b, pad_l:patch_size-pad_r] = crop

    return window
