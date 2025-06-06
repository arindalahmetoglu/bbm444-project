import cv2
import numpy as np
from PIL import Image

def preprocessing_test(warp1,warp2,mask1,mask2):
    warp1 = cv2.resize(warp1,(512,512))
    warp2 = cv2.resize(warp2, (512, 512))

    mask1 = cv2.resize(mask1, (512, 512))
    mask2 = cv2.resize(mask2, (512, 512))

    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    rightmask = mask1
    rightimg = warp1

    kernel1 = np.ones((3, 3), np.float32)
    stitchmask = cv2.bitwise_or(mask1,mask2)
    stitchmask = cv2.bitwise_not(stitchmask)
    stitchmask = cv2.dilate(stitchmask, kernel1, iterations=1)

    overlap_mask = cv2.bitwise_and(mask1, mask2)
    roi = cv2.addWeighted(warp1, 0.5, warp2, 0.5, 0)
    warp2 = cv2.bitwise_and(warp2,warp2,mask=mask2)
    warp1 = cv2.bitwise_and(warp1, warp1, mask=mask1)
    img1_masked = warp1 & cv2.bitwise_not(overlap_mask)[:,:,None]
    merged = cv2.add(warp2, img1_masked, roi)

    leftmask = cv2.bitwise_or(stitchmask,mask2)
    leftimg = cv2.inpaint(merged, stitchmask, 3, cv2.INPAINT_TELEA)

    leftimg = cv2.bitwise_and(leftimg,leftimg,mask=leftmask)

    black_mask = np.zeros_like(mask1)
    rightmask = cv2.bitwise_not(rightmask)

    kernel2 = np.ones((10, 10), np.float32)
    rightmask = cv2.dilate(rightmask, kernel2, iterations=1)
    rightmask = cv2.GaussianBlur(rightmask, (15, 15), 0)

    inputmask = cv2.hconcat([black_mask, rightmask])
    inputmask = inputmask.astype(np.uint8)
    inputmask = Image.fromarray(inputmask)

    inputimage = cv2.hconcat([leftimg, rightimg])
    inputimage = cv2.cvtColor(inputimage, cv2.COLOR_BGR2RGB)
    inputimage = Image.fromarray(inputimage)

    return  inputimage, inputmask
