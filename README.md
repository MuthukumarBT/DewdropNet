# DewdropNet - Raindrop Removal Evaluation

This repository contains the code for evaluating the performance of a raindrop removal model using various image quality metrics. The evaluation includes metrics such as PSNR, SSIM and NIQE on a set of raindrop-degraded images and their corresponding ground truth (clean) images.

## Prerequisites

Make sure you have the following libraries installed:

- Python 3.6.2
- OpenCV 4.2.0.34
- NumPy 1.19.5
- Scikit-image 0.17.2
- scipy 1.5.4

You can install the required libraries using the following command:

```bash
pip install opencv-python numpy scikit-image niqe

python evaluate.py --output_folder "path/to/output_folder" --gt_folder "path/to/gt_folder"   
```


Replace the following placeholders with your actual folder paths:

"path/to/output_folder": The folder containing the raindrop-degraded images (model outputs).  
"path/to/gt_folder": The folder containing the corresponding ground truth (clean) images.
