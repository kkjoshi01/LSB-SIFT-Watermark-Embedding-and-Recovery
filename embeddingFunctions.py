import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

def imageKPS(image : cv2.Mat, reverse : bool = False) -> list[cv2.KeyPoint]:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=0,
        nOctaveLayers=3,
        contrastThreshold=0.02,
        edgeThreshold=15,
        sigma=1.6
    )
    kps, descriptors = sift.detectAndCompute(image, None)
    
    return sorted(kps, key=lambda k: k.response, reverse=reverse)

def KPSRanking(kps : list[cv2.KeyPoint], BORDER : tuple, imgShape : tuple) -> list[cv2.KeyPoint]:
    priorityKPs = []
    H_BORDER, W_BORDER = BORDER
    H, W = imgShape

    for kp in kps:
        x,y = kp.pt
        x,y = int(x), int(y)
        if x + W_BORDER > W or y + H_BORDER > H:
            kps.remove(kp)
            continue

        conflictingKPs = [
            conflictKP for conflictKP in priorityKPs if abs(int(conflictKP.pt[0] - x)) < W_BORDER and abs(int(conflictKP.pt[1]) - y) < H_BORDER
        ]

        if not conflictingKPs:
            priorityKPs.append(kp)
        else:
            for conflictKP in conflictingKPs:
                priorityKPs.remove(conflictKP)
            priorityKPs.append(kp)
    return priorityKPs

def generateChannel(idx : int, watermarkShape : tuple, imgShape : tuple):
    return 0
    Wmrk_H, Wmrk_W = watermarkShape
    Img_H, Img_W, _ = imgShape

    seed = Wmrk_H*31 + Wmrk_W * 17 + Img_H * 13 + Img_W * 7

    return (seed + idx) % 2


# def embedSmallWatermark(cover : cv2.Mat, watermark : np.array) -> np.array:
#     kps = imageKPS(cover)
#     watermarkedImage = cover.copy()
#     H, W, _ = cover.shape
    
#     H_BORDER = watermark.shape[0] + 2
#     W_BORDER = watermark.shape[1] + 2

#     priorityKPs = KPSRanking(kps, (H_BORDER, W_BORDER), (H, W))

#     # kpStrengths = sorted(priorityKPs, key=lambda k: k.response, reverse=True)
#     # print(kpStrengths[0].response, kpStrengths[-1].response)
#     MAX_KPS_USABLE = int(min(len(priorityKPs), np.floor(1.0 * (W * H)**0.5)) // 1)
#     usedKPs = priorityKPs[:MAX_KPS_USABLE]

#     for index in range(len(usedKPs)):
#         x, y = usedKPs[index].pt
#         x, y = int(x), int(y)
#         r = watermark.shape[0] // 2
#         c = watermark.shape[1] // 2
#         region = watermarkedImage[y-r:y+r+1, x-c:x+c+1]
#         channel = generateChannel(index, watermark.shape, cover.shape)
        
#         for row in range(len(watermark)):
#             for element in range(len(watermark[row])):
#                 value = int(region[row][element][channel]) & ~1
#                 value |= watermark[row][element]
#                 region[row][element][channel] = value

#         watermarkedImage[y-r:y+r+1, x-c:x+c+1] = region
        
#     print(f"KPS: {len(kps)}, Priority KPS: {len(usedKPs)}")
#     return watermarkedImage

# def recoverSmallWatermark(watermarkedImage : cv2.Mat, markSize : int) -> np.array:
#     kps = imageKPS(watermarkedImage)
#     recoveredWatermark = np.zeros((markSize, markSize), dtype=np.uint8)
#     H, W, _ = watermarkedImage.shape
#     H_BORDER = markSize + 2
#     W_BORDER = markSize + 2
#     priorityKPs = KPSRanking(kps, (H_BORDER, W_BORDER), (H, W))
#     # N_max = min(M, alpha * (w * h)^beta) where M = number of keypoints, alpha = 1.0, beta = 0.5, w, h = width, height of image
#     MAX_KPS_USABLE = int(min(len(priorityKPs), np.floor(1.0 * (W * H)**0.5)) // 1)
#     usedKPs = priorityKPs[:MAX_KPS_USABLE]

#     values = []
#     for index in range(len(usedKPs)):
#         x, y = usedKPs[index].pt
#         x, y = int(x), int(y)
#         r = markSize // 2
#         c = markSize // 2
#         region = watermarkedImage[y-r:y+r+1, x-c:x+c+1]

#         channel = generateChannel(index, (markSize, markSize), cover.shape)

#         mark = []
#         for row in range(len(recoveredWatermark)):
#             elements = []
#             for element in range(len(recoveredWatermark[row])):
#                 value = int(region[row][element][channel]) & 1
#                 elements.append(value)
#             mark.append(elements)
#         values.append(mark)

#     types = []
#     counters = []

#     for value in values:
#         if value not in types:
#             types.append(value)
#             counters.append(1)
#         else:
#             index = types.index(value)
#             counters[index] += 1
    
#     maxIndex = counters.index(max(counters))
#     common = types[maxIndex]
#     print(f"Common: {common}, Count: {counters[maxIndex]} out of {sum(counters)}, Percent: {counters[maxIndex] / (MAX_KPS_USABLE) * 100:.2f}%")
#     print(f"KPS: {len(kps)}, Priority KPS: {len(priorityKPs)}, MAX_KPS_USABLE: {MAX_KPS_USABLE}")
#     return np.reshape(common, (markSize, markSize)).astype(np.uint8)

def processLargeWatermark(watermark : cv2.Mat) -> np.ndarray:
    if watermark.shape[2] == 4:
        transparent_mask = watermark[:, :, 3] == 0
        watermark[transparent_mask] = [255, 255, 255, 255]
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGRA2GRAY)
    else:
        watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2GRAY)
    watermark = cv2.medianBlur(watermark, 3)
    watermark = cv2.adaptiveThreshold(watermark, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return watermark

def getBlockSize(wmrk : np.ndarray) -> int:
    H, W = wmrk.shape
    imagesize = min(H, W)
    options = [7,5,3]
    for size in options:
        if imagesize % size == 0 and imagesize % size == 0:
            return size
    return 3

def splitWatermark(watermark : np.ndarray, blockSize : int = 3) -> list[np.ndarray]:
    H, W = watermark.shape

    blocks = (
        watermark.reshape(H//blockSize, blockSize, W//blockSize, blockSize).transpose(0, 2, 1, 3).reshape(-1, blockSize, blockSize)
    )

    return blocks


# def embedLargeWatermark(cover : cv2.Mat, watermark : cv2.Mat) -> np.array:
#     kps = imageKPS(cover)
#     watermarkedImage = cover.copy()
#     watermark = processLargeWatermark(watermark)

#     priorityKPs = KPSRanking(kps, (3,3), (cover.shape[0], cover.shape[1]))

#     size = min(watermark.shape[0], watermark.shape[1])

#     scaling =  2**(math.floor(math.log2(size - 1))) if math.log2(size) % 1.0 != 0.0 else size
#     resized = cv2.resize(watermark, (scaling, scaling), interpolation=cv2.INTER_NEAREST)
#     chunks = splitWatermark(resized)

#     KPS_NEEDED = len(chunks)
#     while len(priorityKPs) < KPS_NEEDED+1:
#         scaling = 2**(math.floor(math.log2(scaling - 1)))
#         resized = cv2.resize(watermark, (scaling, scaling), interpolation=cv2.INTER_NEAREST)
#         KPS_NEEDED = len(splitWatermark(resized))
        
#     if scaling != watermark.shape[0] or watermark.shape[0] != watermark.shape[1]:
#         watermark = cv2.resize(watermark, (scaling, scaling), interpolation=cv2.INTER_NEAREST)
#         print(f"Scaled to {scaling}x{scaling}, Watermark shape: {watermark.shape}, Image shape: {cover.shape}")

#     usedKPs = priorityKPs[:KPS_NEEDED]
#     chunks = splitWatermark(resized)

#     for index in range(len(usedKPs)):
#         x, y = usedKPs[index].pt
#         x, y = int(x), int(y)

#         channel = generateChannel(index, watermark.shape, cover.shape)

#         region = watermarkedImage[y-1:y+1, x-1:x+1, channel]
#         change = region & (0xFE)
#         change = change | chunks[index]

#         watermarkedImage[y-1:y+1, x-1:x+1, channel] = change

#     print(f"KPS: {len(kps)}, Priority KPS: {len(usedKPs)}")
#     print(f"Watermarked shape: {watermarkedImage.shape}, Image shape: {cover.shape}")

#     return watermarkedImage

# def recoverLargeWatermark(image : cv2.Mat, watermarkSize : int) -> np.array:
#     kps = imageKPS(image)
#     watermarkedImage = image.copy()
    
#     scaling =  2**(math.floor(math.log2(watermarkSize - 1))) if math.log2(watermarkSize) % 1.0 != 0.0 else watermarkSize
#     KPS_NEEDED = int((scaling**2) // 4)
#     priorityKPs = KPSRanking(kps, (3,3), (image.shape[0], image.shape[1]))

#     while len(priorityKPs) < KPS_NEEDED+1:
#         scaling = 2**(math.floor(math.log2(scaling - 1)))
#         KPS_NEEDED = int((scaling ** 2) // 4)

#     usedKPs = priorityKPs[:KPS_NEEDED]
#     values = []
    
#     for index in range(KPS_NEEDED):
#         x, y = usedKPs[index].pt
#         x, y = int(x), int(y)

#         channel = generateChannel(index, (watermarkSize, watermarkSize), cover.shape)
#         region = watermarkedImage[y-1:y+1, x-1:x+1, channel]
#         change = (region & 1).astype(np.uint8)

#         values.append(change)
    
#     b = scaling // 2
#     watermark = np.array(values).reshape(b, b, 2, 2)
#     watermark = watermark.transpose(0, 2, 1, 3).reshape(b*2, b*2)
#     watermark *= 255

#     print(f"Watermark shape: {watermark.shape}, Image shape: {image.shape}")
#     print(f"KPS: {len(kps)}, Priority KPS: {len(usedKPs)}, KPS_NEEDED: {KPS_NEEDED}")
#     return watermark

# def tamperingDetectorSmall(image : cv2.Mat, watermark : np.array) -> cv2.Mat:
#     kps = imageKPS(image)
#     watermarkedImage = image.copy()
#     H, W, _ = image.shape
    
#     H_BORDER = watermark.shape[0] + 2
#     W_BORDER = watermark.shape[1] + 2

#     priorityKPs = KPSRanking(kps, (H_BORDER, W_BORDER), (H, W))

#     # kpStrengths = sorted(priorityKPs, key=lambda k: k.response, reverse=True)
#     # print(kpStrengths[0].response, kpStrengths[-1].response)
#     MAX_KPS_USABLE = int(min(len(priorityKPs), np.floor(1.0 * (W * H)**0.5)) // 1)
#     expectedKPsOnly = priorityKPs[:MAX_KPS_USABLE]

#     odds = []
#     for index in range(len(expectedKPsOnly)):
#         x,y = expectedKPsOnly[index].pt
#         x,y = int(x), int(y)
        
#         r = watermark.shape[0] // 2
#         c = watermark.shape[1] // 2

#         channel = generateChannel(index, watermark.shape, cover.shape)
#         region = watermarkedImage[y-r:y+r+1, x-c:x+c+1, channel]

#         lsb = region & 1
#         difference = lsb ^ watermark
#         if difference.any():
#             odds.append(expectedKPsOnly[index])
#             cv2.circle(watermarkedImage, (int(x), int(y)), watermark.shape[0]+3, (0, 0, 255), -1)

#     print(f"Possible: {len(odds)} Tampered Points out of {len(priorityKPs)} -> {len(odds) / len(priorityKPs) * 100:.2f}%")
#     if (len(odds) / len(priorityKPs) * 100) > 1.5:
#         print("Yes, the image is tampered")
#     else:
#         print("No, the image is not tampered")
#     return watermarkedImage

# def tamperingDetectorLarge(image : cv2.Mat, watermark : cv2.Mat) -> cv2.Mat:
#     kps = imageKPS(image)
#     priorityKPs = KPSRanking(kps, (3,3), (image.shape[0], image.shape[1]))
#     watermarkedImage = image.copy()
#     watermark = processLargeWatermark(watermark)
#     watermarkSize = min(watermark.shape[0], watermark.shape[1])

#     scaling =  2**(math.floor(math.log2(watermarkSize - 1))) if math.log2(watermarkSize) % 1.0 != 0.0 else watermarkSize
#     KPS_NEEDED = int((scaling**2) // 4)

#     while len(priorityKPs) < KPS_NEEDED+1:
#         scaling = 2**(math.floor(math.log2(scaling - 1)))
#         KPS_NEEDED = int((scaling ** 2) // 4)

#     chunks = splitWatermark(cv2.resize(watermark, (scaling, scaling), interpolation=cv2.INTER_NEAREST))
#     usedKPs = priorityKPs[:len(chunks)]

#     odds = []
#     for index in range(KPS_NEEDED):
#         x, y = usedKPs[index].pt
#         x, y = int(x), int(y)
#         channel = generateChannel(index, watermark.shape, cover.shape)
#         region = watermarkedImage[y-1:y+1, x-1:x+1, channel]
#         embedded = region & 1
#         difference = embedded ^ chunks[index]
        
#         if difference.any():
#             odds.append(usedKPs[index])
#             cv2.circle(watermarkedImage, (int(x), int(y)), 3, (0, 0, 255), -1)
    
#     print(f"Possible: {len(odds)} Tampered Points out of {KPS_NEEDED} -> {len(odds) / KPS_NEEDED * 100:.2f}%")
#     if (len(odds) / KPS_NEEDED * 100) > 15.0:
#         print("Yes, the image is tampered")
#     return watermarkedImage


# cover = cv2.imread(r'Assignment/assets/diamond.jpg', cv2.IMREAD_COLOR)

# wmrk = np.array([
#     [1, 0, 0],
#     [0, 1, 0],
#     [0, 0, 1]
# ])

# wmrk5 = np.array([
#     [0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [0, 1, 0, 1, 0],
#     [1, 0, 1, 0, 1],
#     [0, 1, 0, 1, 0]
# ])

# wmrkimage = cv2.imread(r'Assignment/assets/avis.png', cv2.IMREAD_UNCHANGED)
# watermarkedImage = embedLargeWatermark(cover, wmrkimage)
# cv2.imwrite(r'Assignment/assets/watermarkedLarge.png', watermarkedImage)
# watermarkedImage = cv2.imread(r'Assignment/assets/watermarkedLarge.png', cv2.IMREAD_UNCHANGED)
# recovered = recoverLargeWatermark(watermarkedImage, 64)
# cv2.imwrite(r'Assignment/assets/tamperedLarge.png', recovered)
# cv2.imshow('Tampered', recovered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# wmrkimage = cv2.imread(r'Assignment/assets/avis.png', cv2.IMREAD_UNCHANGED)
# tamperedLarge = tamperingDetectorLarge(watermarkedImage, wmrkimage)
# cv2.imwrite(r'Assignment/assets/tamperedLarge.png', tamperedLarge)
# cv2.imshow('Tampered', tamperedLarge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# wmrkimage = cv2.imread(r'Assignment/assets/avis.png', cv2.IMREAD_UNCHANGED)
# watermarkedImage = cv2.imread(r'Assignment/assets/watermarkedLarge.png', cv2.IMREAD_UNCHANGED)
# rotated = cv2.rotate(watermarkedImage, cv2.ROTATE_90_CLOCKWISE)
# tamperedLarge = tamperingDetectorLarge(rotated, wmrkimage)
# cv2.imwrite(r'Assignment/assets/tamperedRotatedLarge.png', tamperedLarge)
# cv2.imshow('Tampered Rotated', tamperedLarge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# wmrkimage = cv2.imread(r'Assignment/assets/avis.png', cv2.IMREAD_UNCHANGED)
# rotated = cv2.rotate(watermarkedImage, cv2.ROTATE_90_CLOCKWISE)
# tamperedLarge = tamperingDetectorLarge(rotated, wmrkimage)
# cv2.imwrite(r'Assignment/assets/tamperedRotatedLarge.png', tamperedLarge)
# cv2.imshow('Tampered Rotated', tamperedLarge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# cv2.imwrite(r'Assignment/assets/recoveredLarge.png', recovered)
# cv2.imshow('Recovered', recovered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


 
# watermarked = embedSmallWatermark(cover, wmrk)
# cv2.imwrite(r'Assignment/assets/watermarked.png', watermarked)
# watermarked = cv2.imread(r'Assignment/assets/watermarked.png', cv2.IMREAD_UNCHANGED)
# # recovered = recoverSmallWatermark(watermarked, 5)
# # cv2.imwrite(r'Assignment/assets/recoveredSmall.png', recovered)
# # print(recovered)
# tampered = tamperingDetectorSmall(watermarked, wmrk)
# cv2.imwrite(r'Assignment/assets/tampered.png', tampered)
# cv2.imshow('Tampered', tampered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# rotated = cv2.rotate(watermarked, cv2.ROTATE_90_CLOCKWISE)
# tampered = tamperingDetectorSmall(rotated, wmrk)
# cv2.imwrite(r'Assignment/assets/tamperedRotated.png', tampered)
# cv2.imshow('Tampered Rotated', tampered)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
