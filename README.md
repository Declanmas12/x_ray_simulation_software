# AttenuX

![AttenuX Logo](AttenuX_Logo.svg)

**AttenuX** is a high-performance, interactive multi-layer X-ray attenuation and physics simulator built using Python and Streamlit. Designed for materials scientists, semiconductor physics researchers, and medical shielding engineers, AttenuX models the sequential transmission and macroscopic reduction of high-energy photon beams passing through complex heterojunction stacks (such as perovskite solar cells, thin-film electronics, or multi-material radiation barriers).

---

## 🚀 Features

*   **📐 Stateful Multi-Layer Stack Designer:** Dynamically add, re-order, or remove infinite material layers (e.g., Quartz Glass / ITO / Perovskite Absorbers) without losing your workspace configurations.
*   **🧪 Predefined & Custom Stoichiometry:** Toggle between a predefined compound database or engineer custom chemical mixtures down to individual element atom counts and densities.
*   **📊 Microscopic to Macroscopic Physics Mapping:** Instantly view cross-section interactions alongside macro-scale Beer-Lambert transmittance graphs via clean dashboard sub-tabs.
*   **🎯 Energy Spectrum Probe:** Use an interactive energy slider to observe precise transmission efficiency margins at specific keV steps.
*   **📥 Comprehensive Report Ingestion:** Download complete high-resolution data arrays as standard `.csv` spreadsheets for immediate analytical post-processing.

---

## 🔬 Theoretical Background & Physics Engine

### 1. Multi-Layer Stack Reduction
In physical systems, an X-ray beam attenuates sequentially. The remaining intensity exiting layer $i$ becomes the initial intensity input ($I_0$) for layer $i+1$. AttenuX evaluates the cumulative transmission fraction across an $N$-layer heterojunction stack by evaluating the composite linear attenuation exponent:

$$I_{\text{total}} = I_0 \times \prod_{i=1}^{N} e^{-\mu_i x_i} = I_0 e^{-\sum_{i=1}^{N} \mu_i x_i}$$

Where:
*   $\mu_i$ is the linear attenuation coefficient of the $i$-th layer ($\text{cm}^{-1}$).
*   $x_i$ is the specific physical thickness of the $i$-th layer ($\text{cm}$).

### 2. Macroscopic Cross-Section Synthesis
The total microscopic cross-section $\sigma_{\text{total}}$ represents the structural probability of a photon interacting with matter, modeled by summing four primary fundamental mechanisms:

$$\sigma_{\text{total}} = \sigma_{\text{photo}} + \sigma_{\text{compton}} + \sigma_{\text{pair}} + \sigma_{\text{triplet}}$$

AttenuX converts this microscopic cross-section ($\text{cm}^2/\text{atom}$) into the macroscopic linear attenuation profile ($\mu$) using target elemental densities and calculated effective compound weights:

$$\mu = \sigma_{\text{total}} \times 10^{-24} \times \left( \frac{\rho \cdot N_A}{M} \right)$$

Where:
*   $\rho$ is the material mass density ($\text{g/cm}^3$).
*   $N_A$ is Avogadro's constant ($6.022 \times 10^{23} \text{ atoms/mol}$).
*   $M$ is the dynamic molar mass of the compound ($\text{g/mol}$).
*   $10^{-24}$ is the conversion scale transforming cross-sections from Barns to $\text{cm}^2$.

### 3. Photon Interaction Modeling

#### A. Photoelectric Effect ($\sigma_{\text{photo}}$)
Dominating at lower keV levels, this handles inner-shell electronic absorption using empirical Effective Atomic Number ($Z_{\text{eff}}$) scaling:

$$\sigma_{\text{photo}}(E) \approx 3.0 \times 10^{12} \times \frac{Z_{\text{eff}}^{4.0}}{E^{3.5}}$$

#### B. Compton Scattering ($\sigma_{\text{compton}}$)
Models incoherent mid-energy inelastic scattering off target electrons, evaluated using the analytical **Klein-Nishina** approximation:

$$\sigma_{\text{compton}}(k) = 2\pi Z_{\text{eff}} r_e^2 \left( \frac{1+k}{k^2} \left( \frac{2(1+k)}{1+2k} - \frac{\ln(1+2k)}{k} \right) + \frac{\ln(1+2k)}{2k} - \frac{1+3k}{(1+2k)^2} \right)$$

*(where $k = E / m_e c^2$ is the normalized photon energy relative to electron rest mass).*

#### C. Pair & Triplet Production ($\sigma_{\text{pair}}$, $\sigma_{\text{triplet}}$)
Triggers at ultra-high energy boundaries ($E \ge 2m_e c^2 \approx 1.022 \text{ MeV}$ for nuclear field Pair Production, and $E \ge 4m_e c^2 \approx 2.044 \text{ MeV}$ for triplet electron field interactions), modeling complete energy-to-mass transformation profiles near atomic nuclei.

---
