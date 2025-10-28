A very small showing of how human-faces can be compressed-sensed (does that make sense as a word?) by recognizing its sparsity 
in the universal DCT bases or a tailored basis like the SVD Eigenfaces basis made from a similar face-image dataset.

Reconstruction of the sparsest solution is the min l1-norm solution of the convex formulation of the task (solved via CVX, (thanks to Stephen Boyd and Steven Diamond)).

The reconstruction quality (as a function of the number of samples in the observation) is plot against the SSIM and PSNR figures-of-merit for both DCT and SVD.
