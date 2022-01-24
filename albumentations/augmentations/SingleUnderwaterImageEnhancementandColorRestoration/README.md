<h1>Single-Underwater-Image-Enhancement-and-Color-Restoration</h1>
<h3>This is python implementation for a comprehensive review paper "An Experimental-based Review of Image Enhancement and Image Restoration Methods for Underwater Imaging" </h3>


**ABSTRACT!**  Underwater images play a key role in ocean exploration, but often suffer from severe quality degradation due to light absorption and scattering in water medium. Although major breakthroughs have been made recently in the general area of image enhancement and restoration, the applicability of new methods for improving the quality of underwater images has not specifically been captured. In this paper, we review the image enhancement and restoration methods that tackle typical underwater image impairments, including some extreme degradations and distortions. Firstly, we introduce the key causes of quality reduction in underwater images, in terms of the underwater image formation model (IFM). Then, we reviews underwater restoration methods, considering both the IFM-free and the IFM-based approaches. Next, we present an experimental-based comparative evaluation of state-of-the-art IFM-free and IFM-based methods, considering also the prior-based parameter estimation algorithms of the IFM-based methods, using both subjective and objective analysis. Starting from this study, we pinpoint the key shortcomings of existing methods, drawing recommendations for future research in this area. Our review of underwater image enhancement and restoration provides researchers with the necessary background to appreciate challenges and opportunities in this important field.

## Already Implemented

**Underwater Image Color Restoration**
- DCP: Single Image Haze Removal Using Dark Channel Prior (2011)
- GBdehazingRCorrection: Single underwater image restoration by blue-green channels dehazing and red channel correction (2016)
- IBLA: Underwater Image Restoration Based on Image Blurriness and Light Absorption (2017)
- LowComplexityDCP: Low Complexity Underwater Image Enhancement Based on Dark Channel Prior (2011)
- MIP: Initial results in underwater single image dehazing (2010)
- NewOpticalModel: Single underwater image enhancement with a new optical model (2013)
- RoWS: Removal of water scattering (2010)
- UDCP: Transmission Estimation in Underwater Single Images (2013)
- ULAP: A Rapid Scene Depth Estimation Model Based on Underwater Light Attenuation Prior for Underwater Image Restoration (2018)

**Underwater Image Enhancement**
- CLAHE: Contrast limited adaptive histogram equalization (1994)
- Fusion-Matlab: Enhancing underwater images and videos by fusion (2012)
- GC: Gamma Correction
- HE: Image enhancement by histogram transformation (2011)
- ICM: Underwater Image Enhancement Using an Integrated Colour Model (2007)
- UCM: Enhancing the low quality images using Unsupervised Colour Correction Method (2010)
- RayleighDistribution: Underwater image quality enhancement through composition of dual-intensity images and Rayleigh-stretching (2014)
- RGHS: Shallow-Water Image Enhancement Using Relative Global Histogram Stretching Based on Adaptive Parameter Acquisition (2018)



## Install
Here is the list of libraries you need to install to execute the code:
- python = 3.6
- cv2
- numpy
- scipy
- matplotlib
- scikit-image
- natsort
- math
- datetime

## Easy Usage
1. Complete the running environment configuration;
2. Put the inputs images to corresponding folders :
  - (create 'InputImages' and 'OutputImages' folders, then put raw images to 'InputImages' folder);
3. Python main.py;
4. Find the enhanced/restored images in "OutputImages" folder.


## Citation
If our database or code proves useful for your research, please cite our review papers and some related papers.

```
@article{Review of Image Enhancement and Image Restoration Methods,
    author    = {Yan Wang, Wei Song, Giancarlo Fortino, Lizhe Qi, Wenqiang Zhang, Antonio Liotta},
    title     = {An Experimental-based Review of Image Enhancement and Image Restoration Methods for Underwater Imaging},
    journal   = {IEEE Access，DOI:10.1109/ACCESS.2019.2932130},
    year      = {2019}
}
@article{Underwater Image Enhancement Method,
    author    = {Wei Song, Yan Wang, Dongmei Huang, Antonio Liotta, Cristian Perra},
    title     = {Enhancement of Underwater Images with Statistical Model of Background Light and Optimization of Transmission Map},
    journal   = {IEEE Transactions on Broadcasting},
    year      = {2019}
}
@article{Underwater Image Restoration,PCM2018
    author    = {Wei Song, Yan Wang,  Dongmei Huang, Tjondronegoro Dian},
    title     = {A Rapid Scene Depth Estimation Model Based on Underwater Light Attenuation Prior for Underwater Image Restoration},
    journal   = {DOI: 10.1007/978-3-030-00776-8_62},
    year      = {2018}
}
@article{Underwater Image Enhancement,MMM2018
    author    = {Dongmei Huang, Yan Wang, Wei Song, Sequeira Jean, Mavromatis Sébastien},
    title     = {Shallow-Water Image Enhancement Using Relative Global Histogram Stretching Based on Adaptive Parameter Acquisition},
    journal   = {DOI: 10.1007/978-3-319-73603-7_37},
    year      = {2018}
}
```

## Contact Authors
- Yan Wang, e-mail: 19110860017@fudan.edu.cn
- Wei Song, e-mail: wsong@shou.edu.cn
