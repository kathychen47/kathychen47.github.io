---
title: Alvarez Macovski Method
date: 2023-09-04
mathjax: true
categories:
  - [Computed Tomography, Phyiscs]
tags:                      
  - attenuation coefficient
disableNunjucks: true
---

## Alvarez Macovski Model

Attenuation coefficient can be represented as a function of $u(E)$

$$
\mu(E)=a_1 f_1(E)+a_2 f_2(E)+\ldots+a_n f_n(E) .
$$

Assume that only photoelectric absorption and Compton effect exist for interaction:

$$
\mu(E)=a_1 \frac{1}{E^3}+a_2 f_{\mathrm{KN}}(E)
$$

- $f_{\mathrm{KN}}(E)$ is the Klein-Nishina function：
  $$ =  > \mu (E) = {K_1}\frac{\rho }{A}{Z^3}*\frac{1}{{{E^3}}} + {K_2}\frac{\rho }{A}Z*{f_{{\text{KN}}}}(E)$$

- $\alpha=E / 510.975 \mathrm{keV}$

- $\begin{aligned}
  & a_1 \approx K_1 \frac{\rho}{A} Z^n, \quad n \approx 4 or 3 \\
  & a_2 \approx K_2 \frac{\rho}{A} Z
  \end{aligned}$

  $A$ is atomic weight, $Z$ is atomic number, $\rho$ is mass density

$$ =  > \mu (E) = {K_1}\frac{\rho }{A}{Z^3}\frac{1}{{{E^3}}} + {K_2}\frac{\rho }{A}Z{f_{{\text{KN}}}}(E)$$

- $E = \frac{{hc}}{\lambda }$

$$ =  > \mu (E) = {K_1}\frac{\rho }{A}{Z^3}\frac{{{\lambda ^3}}}{{{{(hc)}^3}}} + {K_2}\frac{\rho }{A}Z{f_{{\text{KN}}}}(E)$$

$$ =  > \mu (E) = \frac{{{K_1}}}{{A{{(hc)}^3}}}\rho {Z^3}{\lambda ^3} + \frac{{{K_2}}}{A}\rho Z{f_{{\text{KN}}}}(E)$$

- ${f_{{\text{KN}}}}(E)$ can be approximate to ${f_{{\text{KN}}}}(E) \propto {E^{ - 1}}$, then ${f_{{\text{KN}}}}(E) \propto \lambda $ ????

Thus:
$$= > \mu (E) = \frac{{{K_1}}}{{A{{(hc)}^3}}}\rho {Z^3}{\lambda ^3} + \frac{{{K_2}}}{A}\rho Z\lambda $$ ????

In tomosipo or gvxr package, attenuation coefficient is calculate by uisng:

$$\mu \left( {material,{\text{ }}E} \right){\text{ }} = {\text{ }}\mu \left( {water,{\text{ }}E} \right){\text{ }} * {\text{ }}\left( {1{\text{ }} + {\text{ }}\frac{{HU\left( {material} \right)}}{{1000}}} \right)$$

${HU\left( {material} \right)}$  and $\mu \left( {water,{\text{ }}E} \right)$ can be found by checking table.

What is the relationship between $\mu \left( {material,{\text{ }}E} \right)$ and $\mu (E)$ we calculated using Alvarez Macovski Method????

${u_\lambda } = {K_1}\rho {Z^3}{\lambda ^3} + {K_2}\rho \lambda$

if we do the projection for object with density $\rho $ and $\rho {Z^3}$, and add them together, it looks like:

$${u_\lambda } = {K_1}{\lambda ^3}(projection{\text{ }}for{\text{ }}object{\text{ }}density{\text{ }}\rho {Z^3}) + {K_2}\lambda (projection{\text{ }}for{\text{ }}object{\text{ }}density{\text{ }}\rho )$$





## calculate effective atomic number for any compound





$$u(E)/\rho = {k_1}{Z^3}{E^{ - 3}} + {k_2}{f_{KN}}(E)$$

where $E$ is the energy, $k_1$ and $k_2$ are constant, which is the parameters we want to calculate based on least square method. $Z$ is effective atomic number for compound or mixture or element. ${f_{KN}}(E)$ is Klein-Nishina function. The $u(E)/\rho$ is mass attenuation coefficient, the real value of it can be extracted based on the previous function you defined `mass_attenuation_coefficient(input_material, energy)`

define a function to fit curve for this function, and visulize it in the plot, you can take H2O as an example, use the energy ranging from 20 to 120keV



```Python
import periodictable
import re
import xraylib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from mendeleev import element
import re


def parse_formula(formula):
    """
    Parse a chemical formula and return a dictionary of elements and their weight fractions.
    For example, 'H2O' returns {'H': 0.11189879765805677, 'O': 0.8881012023419432}.
    """
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    compound = {}
    total_mass = sum(
        [int(count) * getattr(periodictable, element).mass if count else getattr(periodictable, element).mass for
         element, count in elements])

    for element, count in elements:
        count = int(count) if count else 1
        weight_fraction = (getattr(periodictable, element).mass * count) / total_mass
        compound[element] = weight_fraction

    return compound


def Z_eff(compound_or_formula_or_mixture,
          p=2.94):  # p=2.94 Exponent value commonly used for photon energies below 3 MeV
    """
    Calculate the effective atomic number using the given formula, compound dictionary, or mixture.

    Parameters:
    - compound_or_formula_or_mixture (dict, str, or dict): A dictionary with element symbols and weight fractions, a chemical formula string, or a dictionary of mixtures with weight fractions .
    For example, {'H': 0.1, 'O': 0.9}, 'H2O', or {'H2O': 0.5, 'C': 0.5}.

    Returns:
    - float: Effective atomic number (Z_eff).
    """

    if isinstance(compound_or_formula_or_mixture, str):
        compound = parse_formula(compound_or_formula_or_mixture)
    elif isinstance(compound_or_formula_or_mixture, dict):
        # Check if the keys are atomic numbers (integers)
        if all(isinstance(key, int) for key in compound_or_formula_or_mixture.keys()):
            compound = {periodictable.elements[key].symbol: value for key, value in
                        compound_or_formula_or_mixture.items()}
        else:  # Handle mixtures and compounds
            compound = {}
            for formula, fraction in compound_or_formula_or_mixture.items():
                for element, weight in parse_formula(formula).items():
                    if element in compound:
                        compound[element] += weight * fraction
                    else:
                        compound[element] = weight * fraction
    else:
        raise ValueError("Invalid input type")

    numerator = sum([w * (getattr(periodictable, symbol).number ** p) for symbol, w in compound.items()])

    Z_eff = numerator ** (1 / p)
    return Z_eff


def mass_attenuation_coefficient(input_material, energy, unit='keV'):
    """
    Calculate the mass attenuation coefficient for an element, compound, or mixture.

    Parameters:
    - input_material (str or dict): An element symbol or compound name as a string, or a dictionary where keys are compound names and values are weight fractions.
    For example, {'H2O': 0.3, 'CO2': 0.7} or "H2O".
    - energy (float): Photon energy.
    - unit (str): The unit of energy, either 'keV' or 'MeV'. Default is 'keV'.

    Returns:
    - float: Mass attenuation coefficient in cm^2/g.
    """
    # Convert energy to keV if it's in MeV
    if unit == 'MeV':
        energy *= 1000  # Convert MeV to keV
    elif unit != 'keV':
        raise ValueError("Invalid energy unit. Only 'keV' and 'MeV' are accepted.")

    # If it's an element
    if isinstance(input_material, str) and len(input_material) <= 2:
        try:
            Z = xraylib.SymbolToAtomicNumber(input_material)
            return xraylib.CS_Total(Z, energy)
        except ValueError:
            pass  # Not an element, continue to check if it's a compound

    # If it's a compound
    if isinstance(input_material, str):
        return xraylib.CS_Total_CP(input_material, energy)

    # If it's a mixture
    elif isinstance(input_material, dict):
        mac_total = 0.0
        for compound, weight_fraction in input_material.items():
            mac_compound = xraylib.CS_Total_CP(compound, energy)
            mac_total += mac_compound * weight_fraction
        return mac_total

    else:
        raise ValueError(
            "Invalid input type. Provide an element symbol, compound name as a string, or a mixture as a dictionary.")


# calculate Z/A for a compound or mixture
from mendeleev import element
import re

def get_ZA_value(input_data):
    '''
    Calculate the Z/A value for a compound, mixture, or elemental weight fractions.
    input_data: a compound formula string, a dictionary of mixtures with weight fractions, or a dictionary of elemental weight fractions.
    For example, {'H2O': 0.3, 'CO2': 0.7}, 'H2O', or {'H': 0.11189879765805677, 'O': 0.8881012023419432}

    Returns:
    - float: Z/A value.
    '''

    def parse_formula(formula):
        pattern = r'([A-Z][a-z]*)(\d*)'
        return re.findall(pattern, formula)

    def compute_ZA(compound):
        components = parse_formula(compound)
        total_Z = 0
        total_A = 0
        for el_symbol, count in components:
            el = element(el_symbol)
            count = int(count) if count else 1
            total_Z += el.atomic_number * count
            total_A += el.mass * count
        return total_Z / total_A

    def compute_ZA_from_weight_fraction(element_weights):
        total_Z = 0
        total_A = 0
        for el_symbol, weight_fraction in element_weights.items():
            el = element(el_symbol)
            total_Z += el.atomic_number * weight_fraction
            total_A += el.mass * weight_fraction
        return total_Z / total_A

    def is_valid_element(symbol):
        try:
            el = element(symbol)
            return True
        except:
            return False

    if isinstance(input_data, str):
        return compute_ZA(input_data)

    # Check if the keys in the dictionary are element symbols or compounds
    first_key = list(input_data.keys())[0]
    if len(first_key) <= 2 and is_valid_element(first_key):
        return compute_ZA_from_weight_fraction(input_data)

    total_ZA = 0
    for compound, fraction in input_data.items():
        total_ZA += compute_ZA(compound) * fraction
    return total_ZA



def klein_nishina(E):
    """
    Calculate the Klein-Nishina function for a given energy.

    Parameters:
    - E (float): Photon energy in keV.

    Returns:
    - float: Klein-Nishina function value.
    """
    alpha = E / 510.975  # Using 510.975 keV as the rest energy of an electron
    term1 = (1 + alpha) / (alpha ** 2)
    term2 = (2 * (1 + alpha)) / (1 + 2 * alpha) - (1 / alpha) * np.log(1 + 2 * alpha)
    term3 = (1 / (2 * alpha)) * np.log(1 + 2 * alpha)
    term4 = (1 + 3 * alpha) / (1 + 2 * alpha) ** 2

    f_KN = term1 * term2 + term3 - term4

    return f_KN


def log_normalize(data):
    data = np.array(data)
    # print(data)
    log_normalized_data = np.log(data + 1e-10)
    return log_normalized_data


def fit_alvarez_macovski_to_NIST(E, material="H2O"):
    print(f"effective atomic number of {material}: ", Z_eff(material))

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    E_values1 = np.linspace(1e-3, 0.1, 1000)
    mac_values = []

    for e in E_values1:
        mac = mass_attenuation_coefficient(material, e, unit='MeV')
        mac_values.append(mac)

    mac_values = np.array(mac_values)

    ax1 = axs[0]
    ax1.plot(E_values1, mac_values)
    ax1.set_yscale('log')  # Use set_yscale method
    ax1.set_ylim(1e-2, 1e4)  # Use set_ylim method
    ax1.set_yticks([1e-2, 1e0, 1e2, 1e4])  # Use set_yticks method
    ax1.set_xlabel("Energy (MeV)")  # Use set_xlabel method
    ax1.set_ylabel("Mass Attenuation Coefficient (cm$^2$/g)")  # Use set_ylabel method
    ax1.set_title(f"Mass Attenuation Coefficient (NIST) for {material} (1e-3 - 1e-1 MeV)")  # Use set_title method

    plt.tight_layout()

    E_values2 = np.linspace(1, 150, 1000)
    mac_values = []

    for e in E_values2:
        mac = mass_attenuation_coefficient(material, e, unit='keV')
        mac_values.append(mac)

    mac_values = np.array(mac_values)

    # 使用2x2布局的第二张图
    ax2 = axs[1]
    ax2.plot(E_values2, mac_values)

    # Set the y-axis to logarithmic scale
    ax2.set_yscale('log')
    # Set the y-axis range
    ax2.set_ylim(1e-2, 1e4)
    # Set specific y-axis ticks
    ax2.set_yticks([1e-2, 1e0, 1e2, 1e4])

    # Set the x-axis to logarithmic scale
    ax2.set_xscale('log')
    # Set the x-axis range
    ax2.set_xlim(1e0, 1e2)
    # Set specific x-axis ticks
    ax2.set_xticks([1e0, 1e1, 1e2])
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Mass attenuation coefficient (cm$^2$/g)")
    ax2.set_title(f"Mass attenuation coefficient (NIST) for {material} (1 - 150KeV)")

    plt.tight_layout()
    plt.show()

    # set up material and energy values
    print(f"Energy range: {E_values[0]} keV to {E_values[-1]} keV")

    # get real NIST mass coefficient values and normalize them
    mac_values = [mass_attenuation_coefficient(material, E, unit="keV") for E in E_values]
    normalized_mac_values = log_normalize(mac_values)

    # set up the fit function k1 * (Z ** 3) * (E ** -3) + k2 * klein_nishina(E))
    def fit_func(E, k1, k2, n):
        Z = Z_eff(material, p=2.94)
        return k1 * (Z ** 3) * (E ** -n) + k2 * klein_nishina(E)

    # set up the residuals function
    def residuals_normalized(params, E, y_observed_normalized):
        k1, k2, n = params
        y_predicted_normalized = log_normalize(fit_func(E, k1, k2, n))
        return y_observed_normalized - y_predicted_normalized

    initial_guess = [32, 6, 3]
    # Use the least squares method with normalized values
    params_fitted_normalized, _ = leastsq(residuals_normalized, initial_guess, args=(E_values, normalized_mac_values))

    # print the estimated values(without normalization)
    fitted_values = fit_func(E_values, *params_fitted_normalized)  # all positive

    # print the estimated values(without normalization)
    fitted_values = log_normalize(fit_func(E_values, *params_fitted_normalized))  # negative values exist

    # Print the fitted parameters
    k1, k2, n = params_fitted_normalized
    print(f"k1: {k1}, k2: {k2}, n: {n}")
    equation = f"y(E) = {k1:.4f} * (Z^3) * (E^-{n:.4f}) + {k2:.4f} * klein_nishina(E)"
    print("Fitted Model Equation:", equation)

    # Print the average relative error
    yi = normalized_mac_values
    yi_hat = log_normalize(fit_func(E_values, *params_fitted_normalized))
    average_relative_error_normalized = np.mean(np.abs((yi - yi_hat) / (yi + 1e-10)))
    print(f"Average Relative Error (Normalized): {average_relative_error_normalized}")

    # Calculate the average relative error for the original data
    mac_values = np.array(mac_values)
    average_relative_error_original = np.mean(
        np.abs(mac_values - fit_func(E_values, *params_fitted_normalized)) / (mac_values + 1e-10))
    print(f"Average Relative Error (Original Data): {average_relative_error_original}")

    

    # visulize the results with normalized values
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    ax1 = axs[0]
    ax1.plot(E_values, normalized_mac_values, 'b-', label='Normalized Real Value', linewidth=1)
    ax1.plot(E_values, log_normalize(fit_func(E_values, *params_fitted_normalized)), 'r--',
             label='Normalized Fitted Curve', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_xscale('log')
    ax1.set_xlim(1e0, 1e2)
    ax1.set_xticks([1e0, 1e1, 1e2])
    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Normalized Mass Attenuation Coefficient')
    ax1.set_title(f'Normalized Curve fitting for {material}')
    plt.tight_layout()

    # Visualize the results with original values
    ax2 = axs[1]
    ax2.plot(E_values, mac_values, 'b-', label='Real Value', linewidth=1)
    ax2.plot(E_values, fit_func(E_values, *params_fitted_normalized), 'r--', label='Fitted Curve', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-2, 1e4)  # Adjusted for normalized values
    ax2.set_yticks([1e-2, 1e-0, 1e2, 1e4])

    ax2.set_xscale('log')
    ax2.set_xlim(1e0, 1e2)
    ax2.set_xticks([1e0, 1e1, 1e2])
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Mass Attenuation Coefficient')
    ax2.set_title(f'Curve fitting for {material}')
    plt.tight_layout()
    plt.show()


    return k1, k2, n, fit_func(E,
                               *params_fitted_normalized), average_relative_error_normalized, average_relative_error_original

# # example
# # material= "H2O"
# # material= "C"
# # material= "I"
# # material= {"H": 0.114000,"C": 0.598000, "N": 0.007000, "O": 0.278000, "Na": 0.001000, "S": 0.001000, "Cl": 0.001000} # Adipose Tissue (ICRU-44)
# # material= {"H": 0.034000, "C": 0.155000, "N": 0.042000, "O": 0.435000, "Na": 0.001000, "Mg": 0.002000, "P": 0.103000, "S": 0.003000, "Ca": 0.225000} # Bone, Cortical (ICRU-44)
# material="Ti"
#
#
# E_values = np.linspace(1, 150, 10000)
# params_fitted_normalized=fit_alvarez_macovski_to_NIST(E_values, material)
# print(params_fitted_normalized)


import scipy.constants as const

# Input energy in electronvolts (eV)
energy_in_eV = 50e3  # 50 keV

# Calculate wavelength using the energy-wavelength relationship
wavelength_in_meters = const.h * const.c / energy_in_eV

# Print the result
print(f"Wavelength: {wavelength_in_meters:.2e} meters")

```











I start with the monochromatic TIE in Eqn. 3.8 using the $\int \rho d z$ and $\int \rho Z^n d z$ formulation and linearising the intensity term using the weak absorption assumption, the weak absorption assumption mathematically translates to $\log (\mu)=1-\mu$. For a strongly absorbing sample, I can linearise the problem by using a linear Newton-Raphson method. I assume power exponent for the atomic number dependency to be $n=3$ :
$$
\begin{aligned}
& 1-\tau_\lambda \\
& -\frac{R}{2 \pi} \nabla_{\perp}\left[\left(1-\tau_\lambda\right) \cdot \nabla_{\perp}\left(K_3 \lambda^2 \int \rho d z\right)\right] \\
& =S_\lambda(x, y, z+R) .
\end{aligned}
$$
Collect terms with $\lambda$ powers, and substitute $\tau_\lambda$ as the cumulative attenuation $K_1 \lambda^3 \int \rho Z^3 d z-K_2 f_{K N}(\lambda) \int \rho d z$ :
$$
\begin{aligned}
1 & -\lambda^3\left[K_1 \int \rho Z^3 d z\right]-f_{K N}(\lambda)\left[K_2 \int \rho d z\right]-\lambda^2\left[\frac{R K_3}{2 \pi} \cdot \nabla_{\perp}^2 \int \rho d z\right] \\
& +\lambda^5\left[\frac{R K_1 K_3}{2 \pi} \nabla_{\perp}\left(\int \rho Z^3 d z \cdot \nabla_{\perp} \int \rho d z\right)\right] \\
& +\lambda^2 f_{K N}(\lambda)\left[\frac{R K_2 K_3}{2 \pi} \nabla_{\perp}\left(\int \rho d z \cdot \nabla_{\perp} \int \rho d z\right)\right] \\
& =S_\lambda(x, y, z+R) .
\end{aligned}
$$
Taking the 5 th order harmonic difference between $\lambda=\lambda_1$ and $\lambda=\lambda_2$, I obtain:
$$
\begin{aligned}
& -L_2\left[K_1 \int \rho Z^3 d z\right]-K N_5\left[K_2 \int \rho d z\right]-L_3\left[\frac{R K_3}{2 \pi} \cdot \nabla_{\perp}^2 \int \rho d z\right] \\
& +K N_3\left[\frac{K_2 R K_3}{2 \pi} \nabla_{\perp}\left(\int \rho d z \cdot \nabla_{\perp} \int \rho d z\right)\right] \\
& =S_5,
\end{aligned}
$$

where $L_m=\left(\frac{1}{\lambda_1{ }^m}-\frac{1}{\lambda_2{ }^m}\right), K N_m=\left(\frac{f_{K N}\left(\lambda_1\right)}{\lambda_1{ }^m}-\frac{f_{K N}\left(\lambda_2\right)}{\lambda_2{ }^m}\right)$, and $S_m=\frac{S_{\lambda_1}(x, y, z+R)-1}{\lambda_1{ }^m}-\frac{S_{\lambda_1}(x, y, z+R)-1}{\lambda_2{ }^m}$

I use Eqn. 3.12 to find the solution of $\int \rho Z^3$ through the path of the object to be:
$$
\begin{aligned}
& -\left[K_1 \int \rho Z^3 d z\right]= \\
& \frac{S_5}{L_2}+\frac{K N_5}{L_2}\left[K_2 \int \rho d z\right]+\frac{L_3}{L_2}\left[\frac{R}{2 \pi} \cdot \nabla_{\perp}^2\left(K_3 \int \rho d z\right)\right] \\
& -\frac{K N_3}{L_2}\left[\frac{K_2 R K_3}{2 \pi} \nabla_{\perp}\left(\int \rho d z \cdot \nabla_{\perp} \int \rho d z\right)\right] .
\end{aligned}
$$
Eqn. 3.13 can be substituted into Eqn. 3.11 while assuming all the $R^2$ or higher order terms are small using the first order approximation in the propagation direction, ignoring all the $O\left(R^2\right)$ terms to obtain:
$$
\begin{aligned}
& \frac{\lambda^3}{L_2}\left[S_5+K N_5 K_2 C+\frac{L_3 R K_3}{2 \pi} \cdot \nabla_{\perp}^2 C-\frac{K N_3 K_2 R K_3}{2 \pi} \nabla_{\perp}\left(C \cdot \nabla_{\perp} C\right)\right] \\
& -f_{K N}(\lambda)\left[K_2 C\right]-\lambda^2\left[\frac{R K_3}{2 \pi} \cdot \nabla_{\perp}^2 C\right] \\
& -\lambda^5 \frac{R K_3}{L_2 2 \pi} \nabla_{\perp}\left[\left(S_5+K N_5 K_2 C\right) \cdot \nabla_{\perp} C\right] \\
& +\lambda^2 f_{K N}(\lambda)\left[\frac{R K_2 K_3}{2 \pi} \nabla_{\perp}\left(C \cdot \nabla_{\perp} C\right)\right] \\
& =S_\lambda(x, y, z+R)-1,
\end{aligned}
$$
where $C=\int \rho d z$ and $\lambda=\lambda_1$.
To simplify Eqn. 3.14, I note that:
$$
\begin{aligned}
\lambda_1{ }^3 / L_2\left\{S_5\right\}-\left[S_\lambda(x, y, z+R)-1\right] & =\frac{\lambda_1{ }^3}{L_2}\left(\frac{S_{\lambda_1}-1}{\lambda_2{ }^5}-\frac{S_{\lambda_2}-1}{\lambda_1{ }^3 \lambda_2{ }^2}\right) \\
& =\frac{\lambda_1{ }^3}{L_2 \lambda_2{ }^2} S_3
\end{aligned}
$$









refractive index $n$ is: $n=1- \delta+i\beta$

where $\delta$ and $\beta$ represent refraction and absorption respectively.

and we know equation: $phase shift =\tfrac{{ - 2\pi }}{\lambda }\delta  = {K_3}\rho \lambda$

derive this equation we can get: $\delta  =  - \frac{{{K_3}\rho {\lambda ^2}}}{{2\pi }}$

we can know the $\rho$ for any materials,  we can get K3 paramter by using fixed $\lambda$, different material to curve fit the model

we can minimize sum of square of: ${\left\| {\delta  - ( - \frac{{{K_3}\rho {\lambda ^2}}}{{2\pi }})} \right\|^2}$



```python
import periodictable
import re
import xraylib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from mendeleev import element
import re
import scipy.constants as const

def parse_formula(formula):
    """
    Parse a chemical formula and return a dictionary of elements and their weight fractions.
    For example, 'H2O' returns {'H': 0.11189879765805677, 'O': 0.8881012023419432}.
    """
    elements = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    compound = {}
    total_mass = sum(
        [int(count) * getattr(periodictable, element).mass if count else getattr(periodictable, element).mass for
         element, count in elements])

    for element, count in elements:
        count = int(count) if count else 1
        weight_fraction = (getattr(periodictable, element).mass * count) / total_mass
        compound[element] = weight_fraction

    return compound


def Z_eff(compound_or_formula_or_mixture,
          p=2.94):  # p=2.94 Exponent value commonly used for photon energies below 3 MeV
    """
    Calculate the effective atomic number using the given formula, compound dictionary, or mixture.

    Parameters:
    - compound_or_formula_or_mixture (dict, str, or dict): A dictionary with element symbols and weight fractions, a chemical formula string, or a dictionary of mixtures with weight fractions .
    For example, {'H': 0.1, 'O': 0.9}, 'H2O', or {'H2O': 0.5, 'C': 0.5}.

    Returns:
    - float: Effective atomic number (Z_eff).
    """

    if isinstance(compound_or_formula_or_mixture, str):
        compound = parse_formula(compound_or_formula_or_mixture)
    elif isinstance(compound_or_formula_or_mixture, dict):
        # Check if the keys are atomic numbers (integers)
        if all(isinstance(key, int) for key in compound_or_formula_or_mixture.keys()):
            compound = {periodictable.elements[key].symbol: value for key, value in
                        compound_or_formula_or_mixture.items()}
        else:  # Handle mixtures and compounds
            compound = {}
            for formula, fraction in compound_or_formula_or_mixture.items():
                for element, weight in parse_formula(formula).items():
                    if element in compound:
                        compound[element] += weight * fraction
                    else:
                        compound[element] = weight * fraction
    else:
        raise ValueError("Invalid input type")

    numerator = sum([w * (getattr(periodictable, symbol).number ** p) for symbol, w in compound.items()])

    Z_eff = numerator ** (1 / p)
    return Z_eff


def mass_attenuation_coefficient(input_material, energy, unit='keV'):
    """
    Calculate the mass attenuation coefficient for an element, compound, or mixture.

    Parameters:
    - input_material (str or dict): An element symbol or compound name as a string, or a dictionary where keys are compound names and values are weight fractions.
    For example, {'H2O': 0.3, 'CO2': 0.7} or "H2O".
    - energy (float): Photon energy.
    - unit (str): The unit of energy, either 'keV' or 'MeV'. Default is 'keV'.

    Returns:
    - float: Mass attenuation coefficient in cm^2/g.
    """
    # Convert energy to keV if it's in MeV
    if unit == 'MeV':
        energy *= 1000  # Convert MeV to keV
    elif unit != 'keV':
        raise ValueError("Invalid energy unit. Only 'keV' and 'MeV' are accepted.")

    # If it's an element
    if isinstance(input_material, str) and len(input_material) <= 2:
        try:
            Z = xraylib.SymbolToAtomicNumber(input_material)
            return xraylib.CS_Total(Z, energy)
        except ValueError:
            pass  # Not an element, continue to check if it's a compound

    # If it's a compound
    if isinstance(input_material, str):
        return xraylib.CS_Total_CP(input_material, energy)

    # If it's a mixture
    elif isinstance(input_material, dict):
        mac_total = 0.0
        for compound, weight_fraction in input_material.items():
            mac_compound = xraylib.CS_Total_CP(compound, energy)
            mac_total += mac_compound * weight_fraction
        return mac_total

    else:
        raise ValueError(
            "Invalid input type. Provide an element symbol, compound name as a string, or a mixture as a dictionary.")


# calculate Z/A for a compound or mixture
from mendeleev import element
import re

def get_ZA_value(input_data):
    '''
    Calculate the Z/A value for a compound, mixture, or elemental weight fractions.
    input_data: a compound formula string, a dictionary of mixtures with weight fractions, or a dictionary of elemental weight fractions.
    For example, {'H2O': 0.3, 'CO2': 0.7}, 'H2O', or {'H': 0.11189879765805677, 'O': 0.8881012023419432}

    Returns:
    - float: Z/A value.
    '''

    def parse_formula(formula):
        pattern = r'([A-Z][a-z]*)(\d*)'
        return re.findall(pattern, formula)

    def compute_ZA(compound):
        components = parse_formula(compound)
        total_Z = 0
        total_A = 0
        for el_symbol, count in components:
            el = element(el_symbol)
            count = int(count) if count else 1
            total_Z += el.atomic_number * count
            total_A += el.mass * count
        return total_Z / total_A

    def compute_ZA_from_weight_fraction(element_weights):
        total_Z = 0
        total_A = 0
        for el_symbol, weight_fraction in element_weights.items():
            el = element(el_symbol)
            total_Z += el.atomic_number * weight_fraction
            total_A += el.mass * weight_fraction
        return total_Z / total_A

    def is_valid_element(symbol):
        try:
            el = element(symbol)
            return True
        except:
            return False

    if isinstance(input_data, str):
        return compute_ZA(input_data)

    # Check if the keys in the dictionary are element symbols or compounds
    first_key = list(input_data.keys())[0]
    if len(first_key) <= 2 and is_valid_element(first_key):
        return compute_ZA_from_weight_fraction(input_data)

    total_ZA = 0
    for compound, fraction in input_data.items():
        total_ZA += compute_ZA(compound) * fraction
    return total_ZA



def klein_nishina(E):
    """
    Calculate the Klein-Nishina function for a given energy.

    Parameters:
    - E (float): Photon energy in keV.

    Returns:
    - float: Klein-Nishina function value.
    """
    alpha = E / 510.975  # Using 510.975 keV as the rest energy of an electron
    term1 = (1 + alpha) / (alpha ** 2)
    term2 = (2 * (1 + alpha)) / (1 + 2 * alpha) - (1 / alpha) * np.log(1 + 2 * alpha)
    term3 = (1 / (2 * alpha)) * np.log(1 + 2 * alpha)
    term4 = (1 + 3 * alpha) / (1 + 2 * alpha) ** 2

    f_KN = term1 * term2 + term3 - term4

    return f_KN


def log_normalize(data):
    data = np.array(data)
    # print(data)
    log_normalized_data = np.log(data + 1e-10)
    return log_normalized_data


def fit_alvarez_macovski_to_NIST(E_values,material="H2O"):

    print(f"effective atomic number of {material}: ", Z_eff(material))

    fig, axs = plt.subplots(2, 1, figsize=(6, 8))
    E_values1 = np.linspace(1e-3, 0.1, 1000)
    mac_values = []

    for e in E_values1:
        mac = mass_attenuation_coefficient(material, e, unit='MeV')
        mac_values.append(mac)

    mac_values = np.array(mac_values)

    ax1 = axs[0]
    ax1.plot(E_values1, mac_values)
    ax1.set_yscale('log')  # Use set_yscale method
    ax1.set_ylim(1e-2, 1e4)  # Use set_ylim method
    ax1.set_yticks([1e-2, 1e0, 1e2, 1e4])  # Use set_yticks method
    ax1.set_xlabel("Energy (MeV)")  # Use set_xlabel method
    ax1.set_ylabel("Mass Attenuation Coefficient (cm$^2$/g)")  # Use set_ylabel method
    ax1.set_title(f"Mass Attenuation Coefficient (NIST) for {material} (1e-3 - 1e-1 MeV)")  # Use set_title method

    plt.tight_layout()

    E_values2 = np.linspace(1, 150, 1000)
    mac_values = []

    for e in E_values2:
        mac = mass_attenuation_coefficient(material, e, unit='keV')
        mac_values.append(mac)

    mac_values = np.array(mac_values)

    # 使用2x2布局的第二张图
    ax2 = axs[1]
    ax2.plot(E_values2, mac_values)

    # Set the y-axis to logarithmic scale
    ax2.set_yscale('log')
    # Set the y-axis range
    ax2.set_ylim(1e-2, 1e4)
    # Set specific y-axis ticks
    ax2.set_yticks([1e-2, 1e0, 1e2, 1e4])

    # Set the x-axis to logarithmic scale
    ax2.set_xscale('log')
    # Set the x-axis range
    ax2.set_xlim(1e0, 1e2)
    # Set specific x-axis ticks
    ax2.set_xticks([1e0, 1e1, 1e2])
    ax2.set_xlabel("Energy (keV)")
    ax2.set_ylabel("Mass attenuation coefficient (cm$^2$/g)")
    ax2.set_title(f"Mass attenuation coefficient (NIST) for {material} (1 - 150KeV)")

    plt.tight_layout()
    plt.show()

    # set up material and energy values
    print(f"Energy range: {E_values[0]} keV to {E_values[-1]} keV")

    # get real NIST mass coefficient values and normalize them
    mac_values = [mass_attenuation_coefficient(material, E, unit="keV") for E in E_values]
    normalized_mac_values = log_normalize(mac_values)

    # set up the fit function k1 * (Z ** 3) * (E ** -3) + k2 * klein_nishina(E))
    def fit_func(E, k1, k2, n):
        Z = Z_eff(material, p=2.94)
        return k1 * (Z ** 3) * (E ** -3) + k2 * klein_nishina(E)

    # set up the residuals function
    def residuals_normalized(params, E, y_observed_normalized):
        k1, k2, n = params
        y_predicted_normalized = log_normalize(fit_func(E, k1, k2, n))
        return y_observed_normalized - y_predicted_normalized

    initial_guess = [32, 6, 3]
    # Use the least squares method with normalized values
    params_fitted_normalized, _ = leastsq(residuals_normalized, initial_guess, args=(E_values, normalized_mac_values))

    # print the estimated values(without normalization)
    fitted_values = fit_func(E_values, *params_fitted_normalized)  # all positive

    # print the estimated values(without normalization)
    fitted_values = log_normalize(fit_func(E_values, *params_fitted_normalized))  # negative values exist

    # Print the fitted parameters
    k1, k2, n = params_fitted_normalized
    print(f"k1: {k1}, k2: {k2}, n: {n}")
    equation = f"y(E) = {k1:.4f} * (Z^3) * (E^-{n:.4f}) + {k2:.4f} * klein_nishina(E)"
    print("Fitted Model Equation:", equation)

    # Print the average relative error
    yi = normalized_mac_values
    yi_hat = log_normalize(fit_func(E_values, *params_fitted_normalized))
    average_relative_error_normalized = np.mean(np.abs((yi - yi_hat) / (yi + 1e-10)))
    print(f"Average Relative Error (Normalized): {average_relative_error_normalized}")

    # Calculate the average relative error for the original data
    mac_values = np.array(mac_values)
    average_relative_error_original = np.mean(
        np.abs(mac_values - fit_func(E_values, *params_fitted_normalized)) / (mac_values + 1e-10))
    print(f"Average Relative Error (Original Data): {average_relative_error_original}")

    

    # visulize the results with normalized values
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))
    ax1 = axs[0]
    ax1.plot(E_values, normalized_mac_values, 'b-', label='Normalized NIST data', linewidth=1)
    ax1.plot(E_values, log_normalize(fit_func(E_values, *params_fitted_normalized)), 'r--',
             label='Normalized Fitted Curve', linewidth=1)
    ax1.legend(loc='upper right')
    ax1.set_xscale('log')
    ax1.set_xlim(1e0, 1e2)
    ax1.set_xticks([1e0, 1e1, 1e2])
    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Normalized Mass Attenuation Coefficient')
    ax1.set_title(f'Normalized Curve fitting for {material}')
    plt.tight_layout()

    # Visualize the results with original values
    ax2 = axs[1]
    ax2.plot(E_values, mac_values, 'b-', label='NIST data', linewidth=1)
    ax2.plot(E_values, fit_func(E_values, *params_fitted_normalized), 'r--', label='Fitted Curve', linewidth=1)
    ax2.legend(loc='upper right')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-2, 1e4)  # Adjusted for normalized values
    ax2.set_yticks([1e-2, 1e-0, 1e2, 1e4])

    ax2.set_xscale('log')
    ax2.set_xlim(1e0, 1e2)
    ax2.set_xticks([1e0, 1e1, 1e2])
    ax2.set_xlabel('Energy (keV)')
    ax2.set_ylabel('Mass Attenuation Coefficient')
    ax2.set_title(f'Curve fitting for {material}')
    plt.tight_layout()

    plt.savefig("../data/output_data/AM_model_fitted_curve/fitted_curve_for_H2O.png", dpi=2000)
    plt.show()


    return k1, k2, n, fit_func(E_values,
                               *params_fitted_normalized), average_relative_error_normalized, average_relative_error_original

# # example
# # material= "H2O"
# # material= "C"
# # material= "I"
# # material= {"H": 0.114000,"C": 0.598000, "N": 0.007000, "O": 0.278000, "Na": 0.001000, "S": 0.001000, "Cl": 0.001000} # Adipose Tissue (ICRU-44)
# # material= {"H": 0.034000, "C": 0.155000, "N": 0.042000, "O": 0.435000, "Na": 0.001000, "Mg": 0.002000, "P": 0.103000, "S": 0.003000, "Ca": 0.225000} # Bone, Cortical (ICRU-44)
# # material="Ti"
# material = {"C": 0.000124, "N": 0.755268, "O": 0.231781, "Ar": 0.012827}  # Air, Dry (near sea level)
# # material = {"H": 0.102000, "C": 0.143000, "N": 0.034000, "O": 0.708000, "Na": 0.002000, "P": 0.003000, "S": 0.003000, "Cl": 0.002000, "K": 0.003000}  # Tissue, Soft (ICRU-44)
#
# E_values = np.linspace(1, 150, 10000)
# params_fitted_normalized=fit_alvarez_macovski_to_NIST(E_values, material)
# print(params_fitted_normalized)



def convert_energy_to_wavelength(energy, energy_unit="eV", wavelength_unit="m"):
    # Convert energy to eV
    if energy_unit == "KeV":
        energy = energy * 1e3
    elif energy_unit == "meV":
        energy = energy * 1e-3
    # Assuming energy is in eV if not specified

    # Calculate wavelength in meters
    wavelength_in_meters = (const.h * const.c / 1.60217663e-19) / energy

    # Convert wavelength to desired unit
    if wavelength_unit == "nm":
        wavelength = wavelength_in_meters * 1e9
    elif wavelength_unit == "um":
        wavelength = wavelength_in_meters * 1e6
    elif wavelength_unit == "mm":
        wavelength = wavelength_in_meters * 1e3
    elif wavelength_unit == "cm":
        wavelength = wavelength_in_meters * 1e2
    elif wavelength_unit == "m":
        wavelength = wavelength_in_meters
    else:
        raise ValueError(f"Unsupported wavelength unit: {wavelength_unit}")

    print(f"Convert energy {energy} eV to Wavelength: {wavelength:.2e} {wavelength_unit}")
    return wavelength

def convert_wavelength_to_energy(wavelength, wavelength_unit="m", energy_unit="eV"):
    # Convert wavelength to meters
    if wavelength_unit == "nm":
        wavelength = wavelength * 1e-9
    elif wavelength_unit == "um":
        wavelength = wavelength * 1e-6
    elif wavelength_unit == "mm":
        wavelength = wavelength * 1e-3
    elif wavelength_unit == "cm":
        wavelength = wavelength * 1e-2
    elif wavelength_unit == "m":
        wavelength = wavelength
    else:
        raise ValueError(f"Unsupported wavelength unit: {wavelength_unit}")

    # Calculate energy in eV
    energy_in_eV = (const.h * const.c / 1.60217663e-19) / wavelength

    # Convert energy to desired unit
    if energy_unit == "KeV":
        energy = energy_in_eV * 1e-3
    elif energy_unit == "meV":
        energy = energy_in_eV * 1e3
    else:
        # Assuming energy is in eV if not specified
        energy = energy_in_eV

    print(f"Convert wavelength {wavelength:.2e} {wavelength_unit} to Energy: {energy:.2e} {energy_unit}")
    return energy

# Example
# convert_energy_to_wavelength(50, energy_unit="KeV", wavelength_unit="nm")
# convert_wavelength_to_energy(2.48e-10, wavelength_unit="m", energy_unit="KeV")




# refractive index: n=1-delta+i*beta
density=1.7 # 1.7g/cm3 convert to mg/mm3 1.7 for C
delta= 8.81800588E-07
wavelength_in_mm=convert_energy_to_wavelength(20, energy_unit="KeV", wavelength_unit="nm")

# density=1.332e-3 #for O
K3=-(2*np.pi*delta)/((wavelength_in_mm**2)*density)
K3 = "{:.3e}".format(K3)
print("K3 = ", K3, "density = ", density, "delta = ", delta, "wavelength_in_mm = ", wavelength_in_mm)
```



I want to simulate phase contrast image, using Alvarez and Macovski’s model to simulate attenuation using this equation: $u(s,E) = {k_1}{E^{ - 3}}\int {\rho {Z^3}dz}  + {k_2}{f_{KN}}(E)\int {\rho dz} $, k1 and k2 are constants, which is known values. Then I need to model phase shift using this equation: $\phi (x,y) = {K_3}\int {\rho dz} $, I want to use tomosipo to conduct this process. Here is the link for a tomosipo SIRT reconstruction example: https://aahendriksen.gitlab.io/tomosipo/intro/simple_reconstruction.html

 Can you write the code for me, tell me the process and all the details. 



Phase contrast image is: ${k_1}{E^{ - 3}}\int {\rho {Z^3}dz} + {k_2}{f_{KN}}(E)\int {\rho dz}+{K_3}\int {\rho dz} $



Using the TIE formulation [66], the monochromatic propagation after the object is formulated as:
$$
\nabla_{\perp} \cdot\left[S_\lambda(x, y, z) \cdot \nabla_{\perp} \phi_\lambda(x, y, z)\right]=-k \frac{\partial S_\lambda(x, y, z)}{\partial z} .
$$
In the near field, I take the finite difference instead of the derivative, to turn Eqn. 3.6 into:
$$
\nabla_{\perp}\left[S_\lambda\left(x, y, z_0\right) \cdot \nabla_{\perp} \phi_\lambda\left(x, y, z_0\right)\right]=-\frac{k}{R} \cdot\left[S_\lambda\left(x, y, z_0+R\right)-S_\lambda\left(x, y, z_0\right)\right] .
$$
Substitute the relations for intensity and phase shift derived in Eqn. 3.3 and 3.4 into Eqn. 3.7 to obtain:
$$
\begin{aligned}
& \exp \left(-\tau_\lambda\right) \\
& -\frac{R}{2 \pi} \nabla_{\perp}\left[\exp \left(-\tau_\lambda\right) \cdot \nabla_{\perp}\left(K_3 \lambda^2 \int \rho d z\right)\right] \\
& =S_\lambda\left(x, y, z_0+R\right) .
\end{aligned}
$$
$\tau_\lambda$已知，$R$ 已知, $${{K_3}{\lambda ^2}\int \rho  dz}$$已知，求$S_\lambda\left(x, y, z_0+R\right)$

怎样代码实现，可以以这三个参数作为输入

