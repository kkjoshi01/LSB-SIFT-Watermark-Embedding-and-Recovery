# Watermark Embedding & Steganography Tool
By Keshav Joshi
> SIFT Keypoint detection based LSB Steganography and Watermarking.

## Features
- SIFT-based Keypoint selection with padding (no overlap) and prioritisation
- Multiple odd-based watermark size support (3x3, 5x5, 7x7, 9x9, ...)
- Seed-generated Watermark Shuffling and Obfuscation
- Blind (Smaller Watermark) and non-Blind (Larger Watermark) Watermark Verification

## Requirements
- Python 3.12+
- NumPy
- OpenCV-Python
- NumPy
- Customtkinter

*Libraries expected as part of Python*
- Math
- Random
- Pathlib
- Collections


Try the following command to install all the required external libraries
```ps1
pip install -r requirements.txt
```

## Files and Folders
```css
.
├── assets/
│   ├── cover images
│   ├── icons
│   ├── image watermarks
│   └── outcomes
├── functions.py
└── main.py
```
<br>

1. **main.py** - Program to run, holds the UI and all the correct interations.
2. **functions.py** - Holds all the smaller functions that work with main.py, do not run it but do not delete it.
3. **assets** 
    1. cover images - A couple of cover images for you to try, alternatively if you have any others you can store them in there.
    2. image watermarks - Image watermarks for you to use for larger watermarking.
    3. outcomes - Storage area for file outputs, also contains some pre-made challenges to decipher and try.
    4. icons - Just some icons for the aesthetic of the program.

## Notes
- Authenticity Validation for Small Watermarks does not need the watermark dimensions entered. For larger watermarks, there is no way to verify a watermark exists without having a potential answer to compare against.
- Recover Watermark only needs the watermark's dimensions (and if shuffling was activity) and it will recover the exact watermark.

## Licenses and Acknowledgements
Icons and pattern images used in this project are from [flaticon](https://www.flaticon.com) or listed sources. Flaticon images are free to use for both personal and commercial use so long as the creators are attributed.

[flaticon.com](https://www.flaticon.com)
- [64.png](https://www.flaticon.com/free-icon/decoration_11197666?related_id=11197666&origin=pack) and [128.png](https://www.flaticon.com/free-icon/decoration_11197666?related_id=11197666&origin=pack) in this project are by [Yuluck](https://www.flaticon.com/authors/yuluck)
- [authenticate.png](https://www.flaticon.com/free-icon/authenticity_10645781?term=authenticate&page=1&position=3&origin=search&related_id=10645781) is by [Canticons](https://www.flaticon.com/authors/canticons)
- [clear.png](https://www.flaticon.com/free-icon/broom_9742093?term=clear&page=1&position=1&origin=search&related_id=9742093) is by [LAFS](https://www.flaticon.com/authors/lafs)
- [download.png](https://www.flaticon.com/free-icon/download_2989976?term=download&page=1&position=3&origin=search&related_id=2989976) is by [Debi Alpa Nugraha](https://www.flaticon.com/authors/debi-alpa-nugraha)
- [embed.png](https://www.flaticon.com/free-icon/embed_11798116?term=embed&page=1&position=2&origin=search&related_id=11798116) is by [IconMarketPK](https://www.flaticon.com/authors/iconmarketpk)
- [recover.png](https://www.flaticon.com/free-icon/recover_11819057?term=recover&page=1&position=3&origin=search&related_id=11819057) is by [Grand Iconic](https://www.flaticon.com/authors/grand-iconic)
- [tamper.png](https://www.flaticon.com/free-icon/elections_5978623?term=tamper&page=1&position=1&origin=search&related_id=5978623) is by [Freepik](https://www.flaticon.com/authors/freepik)
- [upload.png](https://www.flaticon.com/free-icon/upload_3097412?term=upload&page=1&position=1&origin=search&related_id=3097412) is by [Ilham Fitrotul Hayat](https://www.flaticon.com/authors/ilham-fitrotul-hayat)
- [verify.png](https://www.flaticon.com/free-icon/check-mark_1442912?term=verify&page=1&position=15&origin=search&related_id=1442912) is by [Freepik](https://www.flaticon.com/authors/freepik)

Avis Drone Labs
- [avis.png](https://www.google.com/imgres?q=avis%20drone%20labs%20icon&imgurl=https%3A%2F%2Favatars.githubusercontent.com%2Fu%2F176430526%3Fs%3D280%26v%3D4&imgrefurl=https%3A%2F%2Fgithub.com%2Favis-Drone-Labs%2F&docid=_uBGRzsCPKm-DM&tbnid=K1rOlVkiAE69oM&vet=12ahUKEwj6guCEt7SNAxUSTkEAHQIMAVgQM3oECBEQAA..i&w=280&h=280&hcb=2&ved=2ahUKEwj6guCEt7SNAxUSTkEAHQIMAVgQM3oECBEQAA) is the logo of the University of Sheffield's Avis Drone Labs project and is available online in this format from their [Github](https://github.com/avis-Drone-Labs/).

[Emoji Island](https://emojiisland.com/products/big-smiling-iphone-emoji-image)
- smiley.jpeg is from Emoji Island and is free to use for private use only.

Dr Jefferson Alex Dos Santos, Dr Chen Chen & The University Of Sheffield
- bricks.png and diamond.jpg came from Lab 2 of COM31006 Computer Vision. These photos were provided from the Lab work. 








