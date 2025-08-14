import customtkinter as ctk
from PIL import Image
import pathlib
import cv2
import numpy as np
import threading
from collections import Counter
from functions import splitWatermark, shuffleorUnShuffleBlocks, shuffleorUnShuffleWatermarkSmall, processLargeWatermark, smallerKpsandSize, largeKpsandSize, channelSeed

# All functions that are used in support are imported from functions.py (also developed by myself)

# CustomTkinter Settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
import os

class IconButton(ctk.CTkButton):
    def __init__(self, master, text, image : Image.Image, size : tuple[int,int], command=None, **kwargs):
        icon = ctk.CTkImage(light_image=image, dark_image=image, size=size)
        super().__init__(master, text=text, image=icon, command=command, **kwargs)

# Main Application Class
class App(ctk.CTk):
    # UI Side
    def __init__(self):
        super().__init__()

        self.title("Watermark Embedding & Steganography Tool")
        self.geometry("800x600")
        self.resizable(True, True)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.titleFrame = ctk.CTkFrame(self)
        
        self.titleFrame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.titleFrame.grid_columnconfigure(0, weight=1)
        self.titleFrame.grid_rowconfigure(0, weight=1)

        self.titleLabel = ctk.CTkLabel(self.titleFrame, text="Watermark Embedding & Steganography Tool", font=("Arial", 24, "bold"), text_color="white", justify="center", anchor="center")
        self.titleLabel.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.MainFrame = ctk.CTkFrame(self)
        self.MainFrame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.MainFrame.grid_rowconfigure(0, weight=1)
        for col in range(3):
            self.MainFrame.grid_columnconfigure(col, weight=1, uniform="a")

        self.MainFrame.grid_columnconfigure(0, weight=1)   
        self.MainFrame.grid_columnconfigure(1, weight=1)   
        self.MainFrame.grid_columnconfigure(2, weight=2)   

        self.InputFrame = ctk.CTkFrame(self.MainFrame)
        self.InputFrame.grid(row=0, column=0, padx=15, pady=15, sticky="nsew")

        self.ConfigureFrame = ctk.CTkFrame(self.MainFrame)
        self.ConfigureFrame.grid(row=0, column=1, padx=15, pady=15, sticky="nsew")

        self.OutputFrame = ctk.CTkFrame(self.MainFrame)
        self.OutputFrame.grid(row=0, column=2, padx=15, pady=15, sticky="nsew")
        self.OutputFrame.grid_columnconfigure(0, weight=1)
        self.OutputFrame.grid_rowconfigure(0, weight=3)
        self.OutputFrame.grid_rowconfigure(1, weight=1)
        

        self.ImageOutputFrame = ctk.CTkFrame(self.OutputFrame)
        self.ImageOutputFrame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.ImageOutputFrameLabel = ctk.CTkLabel(self.ImageOutputFrame, text="Output", font=("Arial", 18, "italic"), text_color="white", justify="center", anchor="center")
        self.ImageOutputFrameLabel.place(in_=self.ImageOutputFrame, relx=0.0, x=5, y=10)
        self.ImageOutputFrameLabel.grid(row=0, column=0, padx=(10,5), pady=(3,2), sticky="w")

        self.ImageOutputFrame.grid_rowconfigure(0, weight=0)
        self.ImageOutputFrame.grid_rowconfigure(2, weight=0)
        self.ImageOutputFrame.grid_rowconfigure(1, weight=1)
        self.ImageOutputFrame.grid_columnconfigure(0, weight=1)

        self.ImageOutputButtonContainer = ctk.CTkFrame(self.ImageOutputFrame, fg_color="transparent")
        self.ImageOutputButtonContainer.grid(row=2, column=0, padx=(10,5), pady=(5,2), sticky="nsew")
        
        self.DisplayImageFrame = ctk.CTkScrollableFrame(self.ImageOutputFrame, width = 400, height = 500, fg_color="transparent")
        self.DisplayImageFrame.grid(row=1, column=0, padx=5, pady=(0,5), sticky="nsew")

        self.placeholder = Image.new("RGB", (400, 300), color=(50, 50, 50))
        self.placeholderSmall = Image.new("RGB", (100, 100), color=(50, 50, 50))

        self.OutputImageForLabel = ctk.CTkImage(light_image=self.placeholder, dark_image=self.placeholder, size=(400, 300))
        self.OutputImageLabel = ctk.CTkLabel(self.DisplayImageFrame, text="", image=self.OutputImageForLabel, corner_radius=5)
        self.OutputImageLabel.grid(row=0, column=0, sticky="nw")

        self.ConsoleOutputFrame = ctk.CTkFrame(self.OutputFrame)
        self.ConsoleOutputFrame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.ConsoleOutputFrame.grid_columnconfigure(0, weight=1)
        self.ConsoleOutputFrame.grid_columnconfigure(1, weight=0)

        self.ConsoleTextBox = ctk.CTkTextbox(self.ConsoleOutputFrame, width = 1, font=('Consolas', 14), corner_radius=5, state="disabled", fg_color="black")
        self.ConsoleTextBox.grid(row=0, column=0, padx=5, pady=5, sticky="nsew", ipadx=5, ipady=5)

        self.ConsoleScrollbar = ctk.CTkScrollbar(self.ConsoleOutputFrame, orientation="vertical", command=self.ConsoleTextBox.yview)
        self.ConsoleScrollbar.grid(row=0, column=1, padx=(0,5), pady=5, sticky="ns")
        self.ConsoleTextBox.configure(yscrollcommand=self.ConsoleScrollbar.set)
    

        self.ConsoleOutputFrame.grid_columnconfigure(0, weight=1)
        self.ConsoleOutputFrame.grid_rowconfigure(0, weight=1)

        # ----- Buttons ----
        resolvedPath = pathlib.Path().resolve()
        path = os.path.join(resolvedPath, "Assignment", "assets" ,"icons")
        if not os.path.exists(path):
            path = os.path.join(resolvedPath, "assets" ,"icons")
        
        
        downloadIcon = Image.open(os.path.join(path, "download.png"))
        importIcon = Image.open(os.path.join(path, "upload.png"))
        embedIcon = Image.open(os.path.join(path, "embed.png"))
        recoverIcon = Image.open(os.path.join(path, "recover.png"))
        clearIcon = Image.open(os.path.join(path, "clear.png"))
        displayIcon = Image.open(os.path.join(path, "display.png"))
        authenticateIcon = Image.open(os.path.join(path, "authenticate.png"))
        tamperIcon = Image.open(os.path.join(path, "tamper.png"))

        self.DownloadImageButton = IconButton(self.ImageOutputButtonContainer, text="Download", image=downloadIcon, size=(20,20), command=self.downloadImage, width= 350)
        self.DownloadImageButton.grid(row=0, column=0, padx=(0,5), sticky="w")

        self.DisplayImageButton = IconButton(self.ImageOutputButtonContainer, text="OpenCV", image=displayIcon, size=(20,20), command=self.openCV, width= 350)
        self.DisplayImageButton.grid(row=0, column=1, padx=(0,5), sticky="w")

        self.ClearImageButton = IconButton(self.ImageOutputButtonContainer, text="Clear", image=clearIcon, size=(20,20), command=self.clearConsoleAndImage, width= 350)
        self.ClearImageButton.grid(row=0, column=2, padx=(0,5), sticky="w")

        self.ImageOutputButtonContainer.grid_columnconfigure((0,1,2), weight=1)

        self.console("Welcome to the Watermark Embedding & Steganography Tool!")
        
        # Input Frame ------
        self.InputFrame.grid_columnconfigure(0, weight=1)
        self.InputFrame.grid_rowconfigure(0, weight=1, minsize=300)
        self.InputFrame.grid_rowconfigure(1, weight=0, minsize=10)  # fixed 10px gap
        self.InputFrame.grid_rowconfigure(2, weight=1, minsize=450)

        self.InputImageFrame = ctk.CTkFrame(self.InputFrame)
        self.InputImageFrame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.InputImageFrame.grid_rowconfigure(0, weight=1)
        self.InputImageFrame.grid_rowconfigure(1, weight=1)
        self.InputImageFrame.grid_rowconfigure(2, weight=1, minsize=50)
        self.InputImageFrame.grid_columnconfigure(0, weight=1)

        self.InputImageFrameLabel = ctk.CTkLabel(self.InputImageFrame, text="Input Image", font=("Arial", 18, "italic"), text_color="white", justify="center", anchor="center")
        self.InputImageFrameLabel.grid(row=0, column=0, padx=(10,5), pady=(3,2), sticky="w")
        self.InputImageFrameLabel.place(in_=self.InputImageFrame, relx=0.0, x=5, y=10)

        self.InputImageDisplayFrame = ctk.CTkScrollableFrame(self.InputImageFrame, width = 400, height = 500, fg_color="transparent")
        self.InputImageDisplayFrame.grid(row=1, column=0, padx=5, pady=(5,2), sticky="nsew")
        self.InputImageDisplayFrame.grid_rowconfigure(0, weight=1)
        self.InputImageDisplayFrame.grid_columnconfigure(0, weight=1)

        self.InputImage = ctk.CTkImage(light_image=self.placeholder, dark_image=self.placeholder, size=(400, 300))
        self.InputImageLabel = ctk.CTkLabel(self.InputImageDisplayFrame, text="", image=self.InputImage, corner_radius=5)
        self.InputImageLabel.grid(row=0, column=0, sticky="nsew")

        self.ImportImageButton = IconButton(self.InputImageFrame, text="Import", image=importIcon, size=(20,20), command=self.importImage, width= 350, height=50)
        self.ImportImageButton.grid(row=2, column=0, padx=(0,5), pady=(5,2), sticky="w")

        self.WatermarkInputContainer = ctk.CTkFrame(self.InputFrame)
        self.WatermarkInputContainer.grid(row=2, column=0, padx=5, pady=(2,5), sticky="nsew")

        self.WatermarkInputContainer.grid_rowconfigure(0, weight=1)
        self.WatermarkInputContainer.grid_rowconfigure(1, weight=0)
        self.WatermarkInputContainer.grid_columnconfigure(0, weight=1)

        self.WatermarkTabView = ctk.CTkTabview(self.WatermarkInputContainer, width=1)
        self.WatermarkTabView.grid(row=0, column=0, sticky="nsew")
        self.WatermarkTabView.add("Import Watermark")
        self.WatermarkTabView.add("Make Watermark")

        self.ImportWatermarkTab = self.WatermarkTabView.tab("Import Watermark")
        self.ImportWatermarkTab.grid_rowconfigure(0, weight=1)
        self.ImportWatermarkTab.grid_rowconfigure(1, weight=0)
        
        self.ImportWatermarkTab.grid_columnconfigure(0, weight=1)

        self.WatermarkDisplayFrame = ctk.CTkFrame(self.ImportWatermarkTab)
        self.WatermarkDisplayFrame.grid(row=0, column=0, padx=5, pady=(0,5), sticky="nsew")
        self.WatermarkDisplayFrame.grid_rowconfigure(0, weight=1)
        self.WatermarkDisplayFrame.grid_columnconfigure(0, weight=1)

        self.WatermarkImage = ctk.CTkImage(light_image=self.placeholderSmall, dark_image=self.placeholderSmall, size=(400, 300))
        self.WatermarkImageLabel = ctk.CTkLabel(self.WatermarkDisplayFrame, text="", image=self.WatermarkImage, corner_radius=5)
        self.WatermarkImageLabel.grid(row=0, column=0, sticky="nsew")

        self.WatermarkImportButton = IconButton(self.ImportWatermarkTab, text="Import", image=importIcon, size=(20,20), command=self.importWatermark, width= 350)
        self.WatermarkImportButton.grid(row=1, column=0, padx=(0,5), pady=(5,2), sticky="w")

        self.MakeWatermarkTab = self.WatermarkTabView.tab("Make Watermark")
        self.MakeWatermarkTab.grid_rowconfigure(0, weight=1)
        self.MakeWatermarkTab.grid_rowconfigure(1, weight=0)
        self.MakeWatermarkTab.grid_columnconfigure(0, weight=1)

        self.WatermarkGridFrame = ctk.CTkFrame(self.MakeWatermarkTab)
        self.WatermarkGridFrame.grid(row=0, column=0, padx=5, pady=(5,2), sticky="nsew")
        self.WatermarkGridFrame.grid_columnconfigure(0, weight=1)

        self.WatermarkGridSlider = ctk.CTkSlider(self.MakeWatermarkTab, from_=3, to=11, number_of_steps=4, command=self.slider)
        self.WatermarkGridSlider.set(3)
        self.WatermarkGridSlider.grid(row=1, column=0, padx=10, pady=(2,10), sticky="ew")

        self.WatermarkCheckBoxes : list[list[ctk.CTkCheckBox]] = []
        self.Image : Image.Image | cv2.Mat | None = None
        self.OutputImage : Image.Image | cv2.Mat | None = None
        self.ImageString : str | None = None
        self.WatermarkString : str | None = None

        self.CoverImage : cv2.Mat = None
        self.Watermark : cv2.Mat | np.array = None
        
        self.slider(3)

        # Configurations
        self.ConfigureFrame.grid_rowconfigure(0, weight=1)
        self.ConfigureFrame.grid_rowconfigure(1, weight=0)
        self.ConfigureFrame.grid_columnconfigure(0, weight=1, uniform="cfg")

        self.WatermarkConfiguration = ctk.CTkFrame(self.ConfigureFrame)
        self.WatermarkConfiguration.grid(row=0, column=0, padx=10, pady=(10,5), sticky="nsew")
        self.WatermarkConfiguration.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(self.WatermarkConfiguration, text="Watermark Configuration", font=("Arial", 18, "italic"), text_color="white", justify="center", anchor="center").grid(row=0, column=0, columnspan=3, pady=(0,10), sticky="w")

        ctk.CTkLabel(self.WatermarkConfiguration, text="Size: (W x H)").grid(row=1, column=0, sticky="e", padx=(0,5))
        self.WatermarkWidth = ctk.CTkEntry(self.WatermarkConfiguration, placeholder_text="Width")
        self.WatermarkHeight = ctk.CTkEntry(self.WatermarkConfiguration, placeholder_text="Height")
        self.WatermarkWidth.grid(row=1, column=1, padx=(0,5), sticky="ew")
        self.WatermarkHeight.grid(row=1, column=2, padx=(0,5), sticky="ew")

        self.useWatermarkGrid = ctk.CTkCheckBox(self.WatermarkConfiguration, text="Use Watermark Grid", command=self.usingWatermarkGrid)
        self.useWatermarkGrid.grid(row=2, column=0, columnspan=3, pady=(10,2), sticky="w")

        self.useShuffling = ctk.CTkCheckBox(self.WatermarkConfiguration, text="Shuffle Watermark", command=None)
        self.useShuffling.grid(row=3, column=0, columnspan=3, pady=(2,10), sticky="w")
        
        actions = ctk.CTkFrame(self.ConfigureFrame, fg_color="transparent")
        actions.grid(row=1, column=0, padx=10, pady=(5,10), sticky="ew")
        actions.grid_columnconfigure(0, weight=1)

        # Main Functions
        for idx, (text, command, icon) in enumerate([
            ("Embed Watermark", self.EmbedImage, embedIcon),
            ("Recover Watermark", self.recoverWatermark, recoverIcon),
            ("Authenticate", self.authenticate, authenticateIcon),
            ("Tampering Detection", self.tamperDetection, tamperIcon)
        ]):
            button = IconButton(actions, text=text, image=icon, size=(20,20), command=command, width= 350)
            button.grid(row=idx, column=0, padx=5, pady=(5,2), sticky="ew")
            actions.grid_rowconfigure(idx, weight=1)

    def importImage(self):
        dialog = ctk.filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg ")]
        )
        if dialog:
            self.ImageString = dialog
            self.Image = Image.open(dialog)
            self.InputImage = ctk.CTkImage(light_image=self.Image, dark_image=self.Image, size=(self.Image.width, self.Image.height))
            self.InputImageLabel.configure(image=self.InputImage)
            self.CoverImage = cv2.imread(dialog, cv2.IMREAD_UNCHANGED)

    def importWatermark(self):
        dialog = ctk.filedialog.askopenfilename(
            title="Select a watermark image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg ")]
        )
        if dialog:
            self.WatermarkString = dialog
            self.Watermark = Image.open(dialog)
            self.WatermarkImage = ctk.CTkImage(light_image=self.Watermark, dark_image=self.Watermark, size=(self.Watermark.width, self.Watermark.height))
            self.WatermarkImageLabel.configure(image=self.WatermarkImage)
            self.Watermark = cv2.imread(dialog, cv2.IMREAD_UNCHANGED)
            self.WatermarkHeight.delete(0, "end")
            self.WatermarkWidth.delete(0, "end")
            self.WatermarkHeight.insert(0, str(self.Watermark.shape[0]))
            self.WatermarkWidth.insert(0, str(self.Watermark.shape[1]))

    # Shuffling -> Seed Shuffling Generation
    def usingShuffing(self) -> bool:
        return self.useShuffling.get()

    # Watermark Size
    def getWatermarkSize(self) -> tuple[int, int]:
        try:
            width = int(self.WatermarkWidth.get())
            height = int(self.WatermarkHeight.get())
            return (width, height)
        except ValueError:
            self.console("Invalid Watermark size input. Please enter valid integers.")
            return (0, 0)
    
    def isUsingWatermarkGrid(self) -> bool:
        return self.useWatermarkGrid.get()
    
    def usingWatermarkGrid(self):
        if not self.isUsingWatermarkGrid():
            return
        grid = self.getGridWatermark()
        
        self.WatermarkWidth.delete(0, "end")
        self.WatermarkHeight.delete(0, "end")
        self.WatermarkHeight.insert(0, str(grid.shape[0]))
        self.WatermarkWidth.insert(0, str(grid.shape[1]))


    def slider(self, value):
        gridSize = int(value)

        for w in self.WatermarkGridFrame.winfo_children():
            w.destroy()

        self.WatermarkCheckBoxes = []
        
        for r in range(gridSize):
            self.WatermarkGridFrame.grid_rowconfigure(r, weight=1)
        
        for c in range(gridSize):
            self.WatermarkGridFrame.grid_columnconfigure(c, weight=1)
        
        for r in range(gridSize):
            row = []
            for c in range(gridSize):
                chk = ctk.CTkCheckBox(self.WatermarkGridFrame, text="", command=None)
                chk.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
                row.append(chk)
            self.WatermarkCheckBoxes.append(row)


    def openCV(self):
        if self.OutputFrame is None:
            self.console("No output image present.")
            return

        cv2.imshow("Output Image", self.OutputImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def downloadImage(self):
        if self.OutputFrame is None:
            self.console("No output image present.")
            return
        dialog = ctk.filedialog.asksaveasfilename(
            title="Save image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg ")],
            defaultextension=".png"
        )
        if dialog:
            cv2.imwrite(dialog, self.OutputImage)
            self.console(f"Image saved to {dialog}")
            
    # Message Output
    def console(self, msg : str):
        self.ConsoleTextBox.configure(state="normal")
        self.ConsoleTextBox.insert("end", msg + "\n")
        self.ConsoleTextBox.configure(state="disabled")
        self.ConsoleTextBox.see("end")

    def clearConsoleAndImage(self):
        self.ConsoleTextBox.configure(state="normal")
        self.ConsoleTextBox.delete("1.0", "end")
        self.ConsoleTextBox.configure(state="disabled")
        self.ConsoleTextBox.see("end")
        OutputImage = Image.new("RGB", (400, 300), color=(50, 50, 50))
        self.OutputImageForLabel = ctk.CTkImage(light_image=OutputImage, dark_image=OutputImage, size=(400, 300))
        self.OutputImageLabel.configure(image=self.OutputImageForLabel)
        self.OutputImage = None

    # Interactible Watermark Grid for Small Watermarks
    def getGridWatermark(self) -> np.ndarray:
        array = []
        for row in self.WatermarkCheckBoxes:
            array.append([1 if checkmark.get() else 0 for checkmark in row])
        
        return np.array(array)

    # Watermark Embedding Small Watermarks
    def embedSmallerWatermark(self, cover : cv2.Mat, watermark : np.ndarray) -> np.ndarray:
        self.console("Embedding smaller watermark")
        # KPS and Size handles SIFT, Keypoints and Sorting
        kps = smallerKpsandSize(cover, watermark)
        watermarkedImage = cover.copy()

        self.console("Shuffling watermark...")
        blockShuffled = shuffleorUnShuffleWatermarkSmall(watermark, len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1]) if self.usingShuffing() else watermark.astype(np.uint8)
        # Shuffling Blocks
        r = watermark.shape[0] // 2
        self.console("Assigning watermark to keypoints...")
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)
            # Channel Seeding, disabled due to complications
            channel = channelSeed(watermarkedImage.shape, watermark.shape[0], i) if False else 0

            # LSB and Bitwise Operations
            region = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel]
            region &= (0xFE) # Set LSB to 0
            region |= blockShuffled
            watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] = region

        return watermarkedImage

    # Larger Watermark Embedding
    def embedLargerWatermark(self, cover : cv2.Mat, watermark : cv2.Mat) -> np.ndarray:
        self.console("Embedding larger watermark")
        watermarkedImage = cover.copy()
        # Greyscaling, Transparency Handling and Adaptive Thresholding
        watermark = processLargeWatermark(watermark)
        # KPS and Size handles SIFT, Keypoints and Sorting
        kps, watermark, blockSize = largeKpsandSize(cover, watermark)
        self.console("Splitting watermark into blocks...")
        blocks = splitWatermark(watermark, blockSize)
        kps = kps[:len(blocks)]

        # Shuffling Blocks
        self.console("Shuffling watermark blocks...")
        blockShuffle = shuffleorUnShuffleBlocks(blocks, watermark.shape[0] * len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1]) if self.usingShuffing() else blocks

        r = blockSize // 2
        self.console("Assigning watermark to keypoints...")
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)
            channel = channelSeed(watermarkedImage.shape, watermark.shape[0], i) if False else 0
            
            block = blockShuffle[i]
            region = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel]
            region &= (0xFE) # Set LSB to 0
            region |= block
            
            watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] = region
        
        self.console(f"Image shape: {watermarkedImage.shape}, Watermark shape: {watermark.shape}, Block size: {blockSize}")
        self.console(f"Number of points embedded: {len(kps)}")
        self.console("Watermark embedded")

        return watermarkedImage
    # Threading to prevent tasks from freezing the UI
    def EmbedImage(self):
        threading.Thread(target=self.ThreadEmbedImage).start()

    # Output handler for Embed
    def ThreadEmbedImage(self):
        self.clearConsoleAndImage()
        if self.isUsingWatermarkGrid():
            watermark = self.getGridWatermark()
        else:
            watermark = self.Watermark
        
        if watermark is None:
            self.console("No watermark selected.")
            return
        
        if self.CoverImage is None:
            self.console("No cover image selected.")
            return
        
        if max(watermark.shape[0], watermark.shape[1]) > 11:
            img = self.embedLargerWatermark(self.CoverImage, watermark)
        else:
            img = self.embedSmallerWatermark(self.CoverImage, watermark)

        imgForLabel = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.OutputImageForLabel = ctk.CTkImage(light_image=Image.fromarray(imgForLabel), dark_image=Image.fromarray(imgForLabel), size=(img.shape[1], img.shape[0]))
        self.OutputImageLabel.configure(image=self.OutputImageForLabel)
        self.OutputImage = img

    # Recovery -> Requires Watermark Size
    def recoverLargerWatermark(self, watermarkedImage : cv2.Mat, watermarkSize : int) -> np.ndarray:
        self.console("Recovering larger watermark")
        
        kps, watermarkSize, blockSize = largeKpsandSize(watermarkedImage, watermarkSize)
        
        blocks = watermarkSize ** 2 // (blockSize ** 2)
        
        kps = kps[:blocks]

        blockShuffle = []
        r = blockSize // 2
        self.console("Extracting watermark from keypoints...")
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)
            channel = channelSeed(watermarkedImage.shape, watermarkSize, i) if False else 0

            block = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel]
            block &= 1 # Get LSB
            blockShuffle.append(block)

        self.console("Sorting watermark into image...")
        blockShuffle = shuffleorUnShuffleBlocks(blockShuffle, watermarkSize * len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1], True) if self.usingShuffing() else blockShuffle
        watermark = np.array(blockShuffle)
        watermark = watermark.reshape(watermarkSize // blockSize, watermarkSize // blockSize, blockSize, blockSize).transpose(0, 2, 1, 3)
        watermark = watermark.reshape(watermarkSize, watermarkSize)
        watermark *= 255
        self.console("Watermark recovered!")
        return np.array(watermark)
    
    # Recovery -> Requires Watermark Size
    def recoverSmallerWatermark(self, watermarkedImage : cv2.Mat, watermarkSize : int) -> np.ndarray:
        self.console("Recovering smaller watermark")
        kps = smallerKpsandSize(watermarkedImage, watermarkSize)
        watermarkedImage = watermarkedImage.copy()

        r = watermarkSize // 2
        blocks = []
        self.console("Extracting watermark from keypoints...")
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)

            channel = channelSeed(watermarkedImage.shape, watermarkSize, i) if False else 0
            # Getting the LSB and attempting to de-shuffle it when necessary
            region = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel]
            region &= 1
            block = shuffleorUnShuffleWatermarkSmall(region, len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1], True).tolist() if self.usingShuffing() else region.tolist()
            blocks.append(block)
        
        #Generating a list of matrix variations and their counts
        types = []
        counters = []
        for value in blocks:
            if value not in types:
                types.append(value)
                counters.append(1)
            else:
                index = types.index(value)
                counters[index] += 1
        self.console("Identifying Watermark...")
        mostCommon = np.argmax(counters)
        watermark = types[mostCommon]
        self.console(f"Most common watermark: {watermark}, Count: {counters[mostCommon]} out of {len(blocks)}, percentage: {counters[mostCommon] / len(blocks) * 100:.2f}%")
        return np.array(watermark, dtype=np.uint8).reshape(watermarkSize, watermarkSize) * 255

    def threadRecoverWatermark(self):
        self.clearConsoleAndImage()

        if self.CoverImage is None:
            self.console("No watermarked image selected.")
            return
        
        dimensions = self.getWatermarkSize()
        if dimensions[0] == 0 or dimensions[1] == 0:
            return
        
        if max(dimensions[0], dimensions[1]) > 11:
            watermark = self.recoverLargerWatermark(self.CoverImage, max(dimensions[0], dimensions[1]))
        else:
            watermark = self.recoverSmallerWatermark(self.CoverImage, max(dimensions[0], dimensions[1]))

        watermark = cv2.cvtColor(watermark, cv2.COLOR_GRAY2RGB)

        if max(dimensions[0], dimensions[1]) > 11:
            scale = 3
            
        else:
            scale = 150
            self.OutputImageForLabel = ctk.CTkImage(light_image=Image.fromarray(watermark), dark_image=Image.fromarray(watermark), size=(watermark.shape[1] * 150, watermark.shape[0] * 150))

        scaledW, scaledH = watermark.shape[1] * scale, watermark.shape[0] * scale
        
        watermark_Scaled = cv2.resize(watermark, (scaledW, scaledH), interpolation=cv2.INTER_NEAREST)
        self.OutputImageForLabel = ctk.CTkImage(light_image=Image.fromarray(watermark_Scaled), dark_image=Image.fromarray(watermark_Scaled), size=(scaledW, scaledH))
        
        self.OutputImageLabel.configure(image=self.OutputImageForLabel)
        self.OutputImage = watermark

    def recoverWatermark(self):
        threading.Thread(target=self.threadRecoverWatermark).start()

    # Tampering Detection Large -> Requires Watermark
    def detectTamperingLarge(self, watermarkedImage : cv2.Mat, watermark : cv2.Mat) -> np.ndarray:
        self.console("Checking for tampering in larger watermark..")
        tamperedImage = watermarkedImage.copy()
        watermark = processLargeWatermark(watermark)
        self.console("Detecting keypoints...")
        kps, watermark, blockSize = largeKpsandSize(watermarkedImage, watermark)

        actualBlocks = splitWatermark(watermark, blockSize)
        kps = kps[:len(actualBlocks)]
        # Shuffling Blocks
        actualblockShuffle = shuffleorUnShuffleBlocks(actualBlocks, watermark.shape[0] * len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1]) if self.usingShuffing() else actualBlocks
        differencesPerBlock = []
        incorrectBlocks = []
        self.console("Checking keypoints for tampering...")
        r = blockSize // 2
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)
            channel = channelSeed(watermarkedImage.shape, watermark.shape[0], i) if False else 0

            block = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] & 1
            difference = (block != actualblockShuffle[i])
            fraction = difference.mean()
            differencesPerBlock.append(fraction)
            self.console(f"Block {i}, difference: {fraction:.2f}")
            # Fractions set after careful consideration
            if fraction >= 0.23:
                cv2.circle(tamperedImage, (x, y), blockSize // 2, (0, 0, 255), 2)
                incorrectBlocks.append(i)
            elif fraction < 0.23 and fraction >= 0.05:
                cv2.circle(tamperedImage, (x, y), blockSize // 2, (255, 255, 0), 2)
                incorrectBlocks.append(i)
            else: 
                cv2.circle(tamperedImage, (x, y), blockSize // 2, (0, 255, 0), 2)

        # Outputs
        self.console(f"Number of incorrect blocks: {len(incorrectBlocks)} out of {len(kps)}, percentage: {len(incorrectBlocks) / len(kps) * 100:.2f}%")
        self.console(f"Average difference per block: {np.mean(differencesPerBlock) * 100:.2f}%")
        self.console(f"Max difference of a block: {np.max(differencesPerBlock) * 100:.2f}%")
        self.console(f"Min difference of a block: {np.min(differencesPerBlock) * 100:.2f}%")

        if len(incorrectBlocks) / len(kps) > 0.15:
            self.console("Image is tampered")
        else:
            self.console("Image is not tampered")
        return tamperedImage

    # Tampering Detection Small -> Requires Watermark
    def detectTamperingSmall(self, watermarkedImage : cv2.Mat, watermark : np.ndarray) -> np.ndarray:
        tamperedImage = watermarkedImage.copy()
        kps = smallerKpsandSize(watermarkedImage, watermark)
        r = watermark.shape[0] // 2
        differencesPerBlock = []
        incorrectBlocks = []
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)

            channel = channelSeed(watermarkedImage.shape, watermark.shape[0], i) if self.usingShuffing() else 0

            region = (watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] & 1)

            block = shuffleorUnShuffleWatermarkSmall(region, len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1], True).tolist() if self.usingShuffing() else region

            difference = (block != watermark)
            fraction = difference.mean()
            differencesPerBlock.append(fraction)
            # Highlighting the tampered blocks based on level of tampering
            if fraction >= 0.1:
                cv2.circle(tamperedImage, (x, y), r, (0, 0, 255), 2)
                incorrectBlocks.append(i)
            elif fraction < 0.1 and fraction >= 0.05:
                cv2.circle(tamperedImage, (x, y), r, (255, 255, 0), 2)
                incorrectBlocks.append(i)
            else: 
                cv2.circle(tamperedImage, (x, y), r, (0, 255, 0), 2)
        self.console(f"Number of incorrect blocks: {len(incorrectBlocks)} out of {len(kps)}, percentage: {len(incorrectBlocks) / len(kps) * 100:.2f}%")
        self.console(f"Average difference per block: {np.mean(differencesPerBlock) * 100:.2f}%")
        self.console(f"Maximum difference of a block: {np.max(differencesPerBlock) * 100:.2f}%")
        self.console(f"Minimum difference of a block block: {np.min(differencesPerBlock) * 100:.2f}%")

        if len(incorrectBlocks) / len(kps) > 0.015:
            self.console("Image is tampered")
        else:
            self.console("Image is not tampered")
        return tamperedImage

    
    def threadTamperDetection(self):
        self.clearConsoleAndImage()

        if self.CoverImage is None:
            self.console("No watermarked image selected.")
            return
        
        dimensions = self.getWatermarkSize()
        if dimensions[0] == 0 or dimensions[1] == 0:
            return
        
        if self.isUsingWatermarkGrid():
            watermark = self.getGridWatermark()
        else:
            watermark = self.Watermark
        
        if watermark is None:
            self.console(str(watermark))
            self.console("No watermark selected.")
            return
        
        if max(dimensions[0], dimensions[1]) > 11:
            tampering = self.detectTamperingLarge(self.CoverImage, watermark)
        else:
            tampering = self.detectTamperingSmall(self.CoverImage, watermark)

        tamperingforLabel = cv2.cvtColor(tampering, cv2.COLOR_BGR2RGB)

        self.OutputImageForLabel = ctk.CTkImage(light_image=Image.fromarray(tamperingforLabel), dark_image=Image.fromarray(tamperingforLabel), size=(tampering.shape[1], tampering.shape[0]))
        self.OutputImageLabel.configure(image=self.OutputImageForLabel)
        self.OutputImage = tampering

    def tamperDetection(self):
        threading.Thread(target=self.threadTamperDetection).start()

    # Verify Large Watermark requires Watermark
    def verifyLargerWatermark(self, watermarkedImage : cv2.Mat, watermark : np.ndarray) -> list[int]:
        self.console("Checking Authenticity of the image...")
        watermark = processLargeWatermark(watermark)
        self.console("Detecting keypoints...")
        kps, watermark, blockSize = largeKpsandSize(watermarkedImage, watermark)
        # Needs to use the blocks to see if the watermark is correct, similar to Tampering process
        actualBlocks = splitWatermark(watermark, blockSize)
        kps = kps[:len(actualBlocks)]
        # Shuffling Blocks
        actualblockShuffle = shuffleorUnShuffleBlocks(actualBlocks, watermark.shape[0] * len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1]) if self.usingShuffing() else actualBlocks
        differencesPerBlock = []
        incorrectBlocks = []
        self.console("Identifying watermarks...")
        r = blockSize // 2
        for i in range(len(kps)):
            x, y = kps[i].pt
            x, y = int(x), int(y)
            channel = channelSeed(watermarkedImage.shape, watermark.shape[0], i) if False else 0

            block = watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] & 1
            difference = (block != actualblockShuffle[i])
            fraction = difference.mean()
            differencesPerBlock.append(fraction)
            if fraction >= 0.23:
                differencesPerBlock.append(fraction)
                incorrectBlocks.append(i)
        
        self.console(f"Of the expected {len(kps)} KP Points to be used, {(1- len(incorrectBlocks) / len(kps)) * 100:.2f}% had the correct watermark")
        if (1 - len(incorrectBlocks) / len(kps)) < 0.65:
            self.console("No. Image has not been watermarked")
        else:
            self.console("Yes. Image has been watermarked")
            
        return [differencesPerBlock, len(kps), len(incorrectBlocks)]
    
    # Verify Smaller Watermark does not require Watermark
    def verifySmallerWatermark(self, watermarkedImage : cv2.Mat) -> list[int]:
        self.console("Checking Authenticity of the image...")
        sizes = [3,5,7,9,11]
        bestScore = 0.0
        bestSize = None
        bestBlock = None
        # Iterates through the 5 possible watermark sizes
        for size in sizes:
            self.console(f"Testing size {size}x{size}...")
            kps = smallerKpsandSize(watermarkedImage, size)
            
            r = size // 2
            blocks = []
            
            for i in range(len(kps)):
                x, y = kps[i].pt
                x, y = int(x), int(y)
                
                channel = 0
                region = (watermarkedImage[y-r:y+r+1, x-r:x+r+1, channel] & 1)
                if self.usingShuffing():
                    seed = len(kps) * watermarkedImage.shape[0] * watermarkedImage.shape[1]
                    region = shuffleorUnShuffleWatermarkSmall(region, seed, True)
                blocks.append(tuple(region.flatten()))
            # Skips if no blocks are found
            if not blocks:
                continue
            # Generates a list of matrix variations and their counts
            counting = Counter(blocks)
            block, count = counting.most_common(1)[0]
            score = count / len(blocks)
            if score > bestScore:
                bestScore = score
                bestBlock = np.array(block, dtype=np.uint8).reshape(size, size)
                bestSize = size
        # Outputs          
        if bestScore > 0.75:
            self.console(f"Yes. Image has been watermarked with size {bestSize}x{bestSize}, Percentage: {bestScore * 100:.2f}%")
        else:
            self.console("No. Image has not been watermarked")
        return [bestBlock, bestSize]
            
                
    
    def threadedVerify(self):
        self.clearConsoleAndImage()

        if self.CoverImage is None:
            self.console("No watermarked image selected.")
            return
        
        dimensions = self.getWatermarkSize()
        if dimensions[0] == 0 or dimensions[1] == 0:
            self.console("Assuming default watermark size of 3x3")
            verificationData = self.verifySmallerWatermark(self.CoverImage)
            return
        
        if self.isUsingWatermarkGrid():
            watermark = self.getGridWatermark()
        else:
            watermark = self.Watermark
        
        if watermark is None:
            self.console(str(watermark))
            self.console("No watermark selected.")
            return

        if max(dimensions[0], dimensions[1]) > 11:
            verificationData = self.verifyLargerWatermark(self.CoverImage, watermark)
        else:
            verificationData = self.verifySmallerWatermark(self.CoverImage)
                

    def authenticate(self):
        threading.Thread(target=self.threadedVerify).start()
     

if __name__ == "__main__":
    app = App()
    app.mainloop()