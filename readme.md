## Setting up environments

- ### Creating Virtual Environment
 - python3 -m venv bio
- ### Activating virtual environemnt
 - source bio/bin/activate


### Initiating git 

- git init
- git add . 


## Morisita Index


 - Binarize — Otsu threshold on grayscale PNG → binary mask B(x,y) ∈ {0,1} (bright pixel = 1)
 - Sample — draw q random quadrats of size s×s pixels, top-left corner (r_i, c_i) uniformly from valid image area
 - Count — n_i = Σ B(x,y) over each quadrat Q_i; total N = Σ n_i
 - Morisita Index:
    I_M = q · Σ[n_i(n_i − 1)] / [N(N − 1)]
    I_M = 1 → random (Poisson) I_M > 1 → clustered I_M < 1 → uniform
 - Chi-squared test (Morisita 1959, H₀: random distribution):
    χ² = I_M · (N − 1) + q − N,   df = q − 1
    p < 0.05 → statistically significant clustering or uniformity