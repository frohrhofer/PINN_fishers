# Approximating Families of Sharp Solutions to Fisher's Equation with Physics-Informed Neural Networks
Code to paper: https://doi.org/10.1016/j.cpc.2024.109422

## Abstract 
This paper employs physics-informed neural networks (PINNs) to solve Fisher's equation, a fundamental reaction-diffusion system with both simplicity and significance. The focus is on investigating Fisher's equation under conditions of large reaction rate coefficients, where solutions exhibit steep traveling waves that often present challenges for traditional numerical methods. To address these challenges, a residual weighting scheme is introduced in the network training to mitigate the difficulties associated with standard PINN approaches. Additionally, a specialized network architecture designed to capture traveling wave solutions is explored. The paper also assesses the ability of PINNs to approximate a family of solutions by generalizing across multiple reaction rate coefficients. The proposed method demonstrates high effectiveness in solving Fisher's equation with large reaction rate coefficients and shows promise for meshfree solutions of generalized reaction-diffusion systems.

## Requirements
All dependencies can be installed with
```bash
pip install -r requirements.txt

```

## Usage
Indiviudual runs can be modified by making use of the `.yaml` files in the `config` directory.

To run the code and train a single PINN instance, either run
```bash
python3 main.py

```
for utilizing the `default.yaml` configuration file, or
```bash
python3 main.py -c <path_to_config_file>

```
for individual configation files.




## Citation
```
@article{ROHRHOFER2025109422,
title = {Approximating families of sharp solutions to Fisher's equation with physics-informed neural networks},
journal = {Computer Physics Communications},
volume = {307},
pages = {109422},
year = {2025},
issn = {0010-4655},
doi = {https://doi.org/10.1016/j.cpc.2024.109422},
url = {https://www.sciencedirect.com/science/article/pii/S001046552400345X},
author = {Franz M. Rohrhofer and Stefan Posch and Clemens Gößnitzer and Bernhard C. Geiger},
keywords = {Physics-informed neural network, Reaction-diffusion system, Fisher's equation, Sharp solution, Traveling wave, Continuous parameter space},
abstract = {This paper employs physics-informed neural networks (PINNs) to solve Fisher's equation, a fundamental reaction-diffusion system with both simplicity and significance. The focus is on investigating Fisher's equation under conditions of large reaction rate coefficients, where solutions exhibit steep traveling waves that often present challenges for traditional numerical methods. To address these challenges, a residual weighting scheme is introduced in the network training to mitigate the difficulties associated with standard PINN approaches. Additionally, a specialized network architecture designed to capture traveling wave solutions is explored. The paper also assesses the ability of PINNs to approximate a family of solutions by generalizing across multiple reaction rate coefficients. The proposed method demonstrates high effectiveness in solving Fisher's equation with large reaction rate coefficients and shows promise for meshfree solutions of generalized reaction-diffusion systems.}
}
```
