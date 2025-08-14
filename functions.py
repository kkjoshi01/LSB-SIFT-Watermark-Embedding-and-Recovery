import cv2
import numpy as np
import math
import random 

# SIFT and Keypoint functions
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

# Sorts KPS based on Priority of Response
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
    return sorted(priorityKPs, key=lambda k: k.response, reverse=True)

# Converts a Watermark into a binary image
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

# Gets the best fit block size for the watermark or defaults to 3
def getBlockSize(minSize : int) -> int:
    options = [7,5,3]
    for i in range(len(options)):
        if minSize % options[i] == 0:
            return options[i]
    return 3

# Breaks watermark into chunks
def splitWatermark(watermark : np.ndarray, blockSize : int = 3) -> list[np.ndarray]:
    H, W = watermark.shape

    blocks = (
        watermark.reshape(H//blockSize, blockSize, W//blockSize, blockSize).transpose(0, 2, 1, 3).reshape(-1, blockSize, blockSize)
    )

    return blocks

# Channel Seed -> Disabled, but simple Seed Generation based don H, W, Watermark Size and Index
def channelSeed(imgShape : tuple, watermarkSize : int, index : int) -> int:
    H, W, _ = imgShape
    seed = (H * 31) + (W * 17) + (watermarkSize * 7) + index
    return seed % 3

# KPS Handling for large watermarks
def largeKpsandSize(img : cv2.Mat, watermark : cv2.Mat | int) -> tuple[list[cv2.KeyPoint], int | np.ndarray, int, int]:
    kps = imageKPS(img)
    CoverDimensions = img.shape[:2]
    if type(watermark) == np.ndarray:
        blockSize = getBlockSize(min(watermark.shape[:1]))
        newSize = min(watermark.shape[:1]) - (min(watermark.shape[:1]) % blockSize)
    else:
        blockSize = getBlockSize(watermark)
        newSize = watermark - (watermark % blockSize)

    KPS_NEEDED = newSize ** 2 // (blockSize ** 2)
    kps = KPSRanking(kps, (blockSize+2, blockSize+2), CoverDimensions)
    if len(kps) < KPS_NEEDED:
        while len(kps) < KPS_NEEDED and newSize > 0:
            scaling = 2**(math.floor(math.log2(newSize - 1))) if math.log2(newSize) % 1 == 0 else 2**(math.floor(newSize))
            newSize = scaling - (scaling % blockSize)
            KPS_NEEDED = newSize ** 2 // (blockSize ** 2)

    if type(watermark) == np.ndarray:
        watermark = cv2.resize(watermark, (newSize, newSize), interpolation=cv2.INTER_NEAREST)
    else:
        watermark = newSize


    return kps, watermark, blockSize

# KPS Handling for small watermarks
def smallerKpsandSize(img : cv2.Mat, watermark : np.ndarray | int) -> tuple[list[cv2.KeyPoint], int | np.ndarray, int, int]:
    kps = imageKPS(img)
    CoverDimensions = img.shape[:2]
    if type(watermark) == np.ndarray:
        sizex, sizey = watermark.shape[0] + 2, watermark.shape[1] + 2
    else:
        sizex, sizey = watermark + 2, watermark + 2
        
    priorityKPs = KPSRanking(kps, (sizex, sizey), CoverDimensions)
    MAX_USABLE_KPS = int(min(len(priorityKPs), np.floor(1.0 * (CoverDimensions[0] * CoverDimensions[1])**0.5)) // 1)
    return priorityKPs[:MAX_USABLE_KPS]

# Shuffles or Unshuffles the blocks of the watermark (larger watermarks)
def shuffleorUnShuffleBlocks(blocks : list[np.ndarray], seed : int, unShuffle : bool = False) -> list[np.ndarray]:
    randomising = random.Random(seed)
    permutations = list(range(len(blocks)))
    randomising.shuffle(permutations)
    
    if unShuffle:
        inverse = [0] * len(permutations)
        for i, p in enumerate(permutations):
            inverse[p] = i
        return [blocks[i] for i in inverse]
    
    return [blocks[i] for i in permutations]

# Shuffles the elements of a watermark (smaller watermarks)
def shuffleorUnShuffleWatermarkSmall(watermark : np.ndarray, seed : int, unShuffle : bool = False) -> np.ndarray:
    flattened = watermark.flatten().tolist()
    permutations = list(range(len(flattened)))
    random.Random(seed).shuffle(permutations)

    if unShuffle:
        inverse = [0] * len(permutations)
        for i, p in enumerate(permutations):
            inverse[p] = i
        ordered = [flattened[i] for i in inverse]
    else:
        ordered = [flattened[i] for i in permutations]
    
    return np.array(ordered).reshape(watermark.shape).astype(np.uint8)
