import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# --- 1. System Constants & Database Ingestion ---
ALPHA = 1 / 137
M_E = 5.11E5             # Electron rest mass in eV
BARN_CONVERSION = 0.07941 
N_A = 6.022E23           # Avogadro's number

@st.cache_data
def load_material_database():
    try:
        with open("materials.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Critical Error: 'materials.json' database file not found in directory.")
        return {"elements": {}, "predefined_compounds": {}}

db = load_material_database()

# --- 2. Core Physics Calculations ---
def photoelectric_cs(energy_ev, z):
    energy_ev = np.maximum(energy_ev, 1.0)
    return (3.0E12) * ((z ** 4) / (energy_ev ** 3.5))

def compton_cs(energy_ev, z):
    k = energy_ev / M_E
    term1 = ((1 + k) / (k ** 2)) * ((2 * (1 + k) / (1 + 2 * k)) - (np.log(1 + 2 * k) / k))
    term2 = np.log(1 + 2 * k) / (2 * k)
    term3 = (1 + 3 * k) / ((1 + 2 * k) ** 2)
    return 2 * np.pi * z * BARN_CONVERSION * (term1 + term2 - term3)

def pair_production_cs(energy_ev, z):
    k = energy_ev / M_E
    rho = (2 * k - 4) / (2 + k + 2 * np.sqrt(2 * k))
    mask = k >= 2
    sigma_pair = np.zeros_like(k, dtype=float)
    if np.any(mask):
        km = k[mask]
        r = rho[mask]
        poly_term = (1 + 0.5 * r + (23 / 40) * r ** 2 + (11 / 60) * r ** 3 + (29 / 960) * r ** 4)
        sigma_pair[mask] = (z ** 2) * ALPHA * BARN_CONVERSION * (2 * np.pi / 3) * (((km - 2) / km) ** 3) * poly_term
    return sigma_pair

def triplet_production_cs(energy_ev, z):
    k = energy_ev / M_E
    a, b = -2.4674, -1.8031
    mask = k >= 4
    sigma_trip = np.zeros_like(k, dtype=float)
    if np.any(mask):
        km = k[mask]
        s1 = z * ALPHA * BARN_CONVERSION
        s2 = (28 / 9) * np.log(2 * km) - (218 / 27)
        s3 = (1 / km) * ((-4 / 3) * np.log(2 * km) ** 3 + 3 * np.log(2 * km) ** 2 - ((60 + 16 * a) / 3) * np.log(2 * km) + (123 + 12 * a + 16 * b) / 3)
        s4 = (1 / (km ** 2)) * ((8 / 3) * np.log(2 * km) ** 3 - 4 * np.log(2 * km) ** 2 + ((51 + 32 * a) / 3) * np.log(2 * km) - (123 + 32 * a + 64 * b) / 6)
        s5 = (1 / (km ** 3)) * (np.log(2 * km) ** 2 - (53 / 9) * np.log(2 * km) - (2915 - 288 * a) / 216)
        s6 = (1 / (km ** 4)) * ((-49 / 18) * np.log(2 * km) - 115 / 432) + (1 / (km ** 5)) * ((-77 / 36) * np.log(2 * km) - 10831 / 8640)
        s7 = (1 / (km ** 6)) * ((-641 / 300) * np.log(2 * km) - 64573 / 36000) + (1 / (km ** 7)) * ((-4423 / 1800) * np.log(2 * km) - 394979 / 216000)
        sigma_trip[mask] = s1 * (s2 + s3 + s4 + s5 + s6 + s7)
    return np.maximum(sigma_trip, 0)

# --- 3. Material Parsing Logic ---
def get_material_metrics(composition, density):
    total_atoms = sum(composition.values())
    if total_atoms == 0:
        return 1.0, 1.0, np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)
    
    fw = []
    atomic_numbers = []
    molar_mass = 0.0
    
    for element, count in composition.items():
        if element in db["elements"] and count > 0:
            fw.append(count)
            atomic_numbers.append(db["elements"][element]["Z"])
            molar_mass += db["elements"][element]["A"] * count
            
    if not fw:
        return 1.0, 1.0, np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000), np.zeros(1000)
        
    fw = np.array(fw)
    z = np.array(atomic_numbers)
    z_eff = np.sum((fw / np.sum(fw)) * (z ** 2.94)) ** (1 / 2.94)
    
    energies_ev = np.logspace(4, 8, 1000)
    photo = photoelectric_cs(energies_ev, z_eff)
    compton = compton_cs(energies_ev, z_eff)
    pair = pair_production_cs(energies_ev, z_eff)
    triplet = triplet_production_cs(energies_ev, z_eff)
    
    total_cs = photo + compton + pair + triplet
    number_density = (density * N_A) / molar_mass
    mu_profile = total_cs * 1e-24 * number_density
    
    return z_eff, molar_mass, mu_profile, photo, compton, pair, triplet

# --- 4. Streamlit UI Layout Configuration ---
st.set_page_config(page_title="AttenuX - Multi-Layer X-Ray Simulator", layout="wide")

# Read and display the ultra-wide vector logo banner
try:
    with open("AttenuX_Banner.svg", "r") as f:
        svg_code = f.read()
    st.logo(svg_code)
    # Instructs Streamlit to let the wide banner match the app window boundaries natively
    st.image(svg_code, use_container_width=True)
except FileNotFoundError:
    st.title("🔬 AttenuX")
    st.caption("MULTILAYER X-RAY SIMULATOR")

st.markdown("<br>", unsafe_allow_html=True) # Elegant spatial buffer

# Persistent Session State Structural Array Tracking
if "layers" not in st.session_state:
    st.session_state.layers = [
        {"name": "Quartz Glass (SiO2)", "thickness_um": 500.0, "type": "Predefined", "custom_comp": {}, "custom_rho": 2.20},
        {"name": "ITO (Indium Tin Oxide)", "thickness_um": 0.15, "type": "Predefined", "custom_comp": {}, "custom_rho": 7.14},
        {"name": "MAPI (CH3NH3PbI3 Perovskite)", "thickness_um": 0.40, "type": "Predefined", "custom_comp": {}, "custom_rho": 4.16}
    ]

# Setup High-Level Architecture Workspace Tabs (Fixes Scroll/Bottom Stack Bugs)
designer_tab, analytics_tab, stability_tab = st.tabs(["📐 1. Layer Stack Designer", 
                                       "📊 2. Physics & Attenuation Analytics", 
                                       "🛡️ 3. Layer Stability & Absorption Mapping"])

energies = np.logspace(4, 8, 1000) # 10 keV to 100 MeV

# ==========================================
#     TAB 1: LAYER ARCHITECTURE WORKSPACE
# ==========================================
with designer_tab:
    st.header("Layer Stack Architecture Designer")
    st.write("Construct and modify your semiconductor junctions or protective substrates below.")
    
    col_ctrls1, col_ctrls2, col_ctrls3 = st.columns([1, 1, 2])
    with col_ctrls1:
        if st.button("➕ Add Predefined Material", key="btn_add_predefined"):
            st.session_state.layers.append({"name": list(db["predefined_compounds"].keys())[0], "thickness_um": 100.0, "type": "Predefined", "custom_comp": {}, "custom_rho": 2.50})
            st.rerun()
    with col_ctrls2:
        if st.button("🧪 Add Custom Element Mix", key="btn_add_custom"):
            st.session_state.layers.append({"name": "Custom Element Mix", "thickness_um": 10.0, "type": "Custom", "custom_comp": {}, "custom_rho": 2.50})
            st.rerun()
    with col_ctrls3:
        if st.button("🗑️ Reset Stack Layout", key="btn_clear_stack"):
            st.session_state.layers.clear()
            st.rerun()
            
    st.markdown("---")
    
    if not st.session_state.layers:
        st.warning("Your stack is currently empty. Click an options button above to configure your first interaction layer.")
    else:
        # Loop explicitly builds inputs, keeping values persistent in session state
        for idx, layer in enumerate(st.session_state.layers):
            with st.container(border=True):
                st.markdown(f"#### Layer #{idx + 1} Parameters")
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    if layer["type"] == "Predefined":
                        # We safely pull the index mapping of the stored name to prevent drop-downs from resetting on re-renders
                        predef_keys = list(db["predefined_compounds"].keys())
                        default_idx = predef_keys.index(layer["name"]) if layer["name"] in predef_keys else 0
                        layer["name"] = st.selectbox(f"Compound Composition (L{idx+1})", predef_keys, index=default_idx, key=f"mat_{idx}")
                    else:
                        layer["name"] = st.text_input(f"Custom Label (L{idx+1})", value=layer["name"], key=f"label_{idx}")
                
                with col2:
                    layer["thickness_um"] = st.number_input(f"Layer Thickness (μm) (L{idx+1})", min_value=0.0, max_value=1000000.0, value=layer["thickness_um"], format="%.4f", key=f"thick_{idx}")
                
                with col3:
                    if layer["type"] == "Custom":
                        layer["custom_rho"] = st.number_input(f"Material Density (g/cm³) (L{idx+1})", min_value=0.1, max_value=25.0, value=layer["custom_rho"], key=f"rho_{idx}")
                    else:
                        current_density = db["predefined_compounds"][layer["name"]]["density"]
                        st.text_input(f"Density (g/cm³)", value=f"{current_density} (Database)", disabled=True, key=f"rho_disp_{idx}")
                
                with col4:
                    st.markdown("<br>", unsafe_allow_html=True) 
                    if st.button("❌ Remove", key=f"rem_{idx}"):
                        st.session_state.layers.pop(idx)
                        st.rerun()
                
                if layer["type"] == "Custom":
                    st.markdown("**Element Mix Configuration Tool:**")
                    el_cols = st.columns(len(db["elements"]))
                    for e_idx, el in enumerate(db["elements"].keys()):
                        with el_cols[e_idx]:
                            cnt = st.number_input(f"Atoms: {el}", min_value=0, max_value=50, value=layer["custom_comp"].get(el, 0), key=f"el_{idx}_{el}")
                            if cnt > 0:
                                layer["custom_comp"][el] = cnt
                            elif el in layer["custom_comp"]:
                                del layer["custom_comp"][el]

# ==========================================
#     TAB 2: PHYSICS & ANALYTICS WORKSPACE
# ==========================================
with analytics_tab:
    st.header("Simulation Analytics Workspace")
    
    # Pre-process background math numbers across arrays
    composite_mu_x = np.zeros_like(energies)
    total_photo_cs = np.zeros_like(energies)
    total_compton_cs = np.zeros_like(energies)
    total_pair_cs = np.zeros_like(energies)
    total_triplet_cs = np.zeros_like(energies)
    stack_summary_data = []
    
    for idx, layer in enumerate(st.session_state.layers):
        if layer["type"] == "Predefined":
            comp = db["predefined_compounds"][layer["name"]]["composition"]
            rho = db["predefined_compounds"][layer["name"]]["density"]
        else:
            comp = layer["custom_comp"]
            rho = layer["custom_rho"]
            
        z_eff, mol_wt, mu_profile, photo, compton, pair, triplet = get_material_metrics(comp, rho)
        thickness_cm = layer["thickness_um"] * 1e-4
        
        composite_mu_x += (mu_profile * thickness_cm)
        total_photo_cs += photo
        total_compton_cs += compton
        total_pair_cs += pair
        total_triplet_cs += triplet
        
        stack_summary_data.append({
            "Layer": idx + 1,
            "Material": layer["name"],
            "Z_eff": f"{z_eff:.2f}",
            "Molar Mass (g/mol)": f"{mol_wt:.2f}",
            "Thickness (μm)": f"{layer['thickness_um']:.3f}"
        })
        
    total_transmission = np.exp(-composite_mu_x)
    total_absorption = 1.0 - total_transmission
    total_cross_section = total_photo_cs + total_compton_cs + total_pair_cs + total_triplet_cs

    col_side, col_graphs = st.columns([1, 2])
    
    with col_side:
        st.subheader("📋 Stack Review Summary")
        if not st.session_state.layers:
            st.warning("No active materials detected in the configuration workspace.")
        else:
            st.dataframe(pd.DataFrame(stack_summary_data), width="stretch", hide_index=True)
            
        st.subheader("🎯 Specific Probe Target")
        target_energy = st.slider("Energy Probe (keV)", min_value=10, max_value=100000, value=60, step=5)
        
        close_idx = (np.abs((energies / 1e3) - target_energy)).argmin()
        val_trans = total_transmission[close_idx]
        
        st.metric(label="Beam Transmission (I/I₀)", value=f"{val_trans*100:.3f} %")
        st.progress(float(val_trans))
        
        st.subheader("📥 Export Calculations")
        df_export = pd.DataFrame({
            "Energy_keV": energies / 1e3,
            "Total_Stack_Transmission": total_transmission,
            "Total_Stack_Absorption": total_absorption,
            "Total_CrossSection_Units": total_cross_section
        })
        st.download_button(
            label="Download Data Spreadsheet (.CSV)",
            data=df_export.to_csv(index=False).encode('utf-8'),
            file_name="multilayer_xray_report.csv",
            mime="text/csv"
        )
        
    with col_graphs:
        graph_sub_tab1, graph_sub_tab2 = st.tabs(["📊 Cross-Section Profiles", "📉 X-Ray Transmittance Curve"])
        
        with graph_sub_tab1:
            st.subheader("Cross Section Contributions")
            fig1, ax1 = plt.subplots(figsize=(9, 5))
            ax1.loglog(energies / 1e3, total_photo_cs, label="Photoelectric Effect", linestyle="--", alpha=0.7)
            ax1.loglog(energies / 1e3, total_compton_cs, label="Compton Scattering", linestyle="--", alpha=0.7)
            ax1.loglog(energies / 1e3, total_pair_cs, label="Pair Production (Nuclear)", linestyle=":", color="green")
            ax1.loglog(energies / 1e3, total_triplet_cs, label="Triplet Production (Electron)", linestyle=":", color="purple")
            ax1.loglog(energies / 1e3, total_cross_section, label="Total Accumulated Cross-Section", color="black", linewidth=2.5)
            ax1.set_xlabel("Photon Energy (keV)")
            ax1.set_ylabel("Cross-Section (Barns / Arbitrary Units)")
            ax1.grid(True, which="both", alpha=0.3)
            ax1.legend()
            ax1.set_ylim(bottom=1e-3)
            st.pyplot(fig1)
            plt.close(fig1) # Explicitly closes figure to prevent cache leak crashes
            
        with graph_sub_tab2:
            st.subheader("Composite Stack Attenuation Mapping ($I / I_0$)")
            fig2, ax2 = plt.subplots(figsize=(9, 5))
            ax2.semilogx(energies / 1e3, total_transmission * 100, label="Transmission (%)", color="darkblue", linewidth=2.5)
            ax2.semilogx(energies / 1e3, total_absorption * 100, label="Absorbed/Attenuated (%)", color="crimson", linestyle="--")
            ax2.set_xlabel("Photon Energy (keV)")
            ax2.set_ylabel("Percentage (%)")
            ax2.grid(True, which="both", alpha=0.3)
            ax2.legend()
            ax2.set_ylim(-5, 105)
            st.pyplot(fig2)
            plt.close(fig2) # Explicitly closes figure to prevent cache leak crashes

# ==========================================
#     TAB 3: LAYER STABILITY & BIAS MAPPING
# ==========================================
with stability_tab:
    st.header("Layer-by-Layer Absorption & Radiation Stability Analysis")
    st.write("Evaluate which sections of your heterojunction absorb the highest concentration of photons. Layers experiencing high absorption profiles are at the greatest risk for radiation-induced defects or thermal degradation.")

    if not st.session_state.layers:
        st.warning("Please configure your material layers in Tab 1 to generate stability profiles.")
    else:
        # 1. Capture the exact probe energy from the user's workflow
        # (Re-using or overriding the target_energy slider variable)
        st.markdown(f"### Current Structural Analysis at **{target_energy} keV**")
        
        # 2. Sequential Tracking Math
        layer_names_display = []
        layer_absorptions = []
        
        # We start with 100% of the beam entering the first layer
        current_transmission_fraction = 1.0 
        
        for idx, layer in enumerate(st.session_state.layers):
            if layer["type"] == "Predefined":
                comp = db["predefined_compounds"][layer["name"]]["composition"]
                rho = db["predefined_compounds"][layer["name"]]["density"]
            else:
                comp = layer["custom_comp"]
                rho = layer["custom_rho"]
                
            z_eff, _, mu_profile, _, _, _, _ = get_material_metrics(comp, rho)
            thickness_cm = layer["thickness_um"] * 1e-4
            
            # Find closest index for our keV target
            close_idx = (np.abs((energies / 1e3) - target_energy)).argmin()
            mu_at_energy = mu_profile[close_idx]
            
            # Compute step boundaries
            intensity_entering = current_transmission_fraction
            intensity_exiting = intensity_entering * np.exp(-mu_at_energy * thickness_cm)
            
            # Absolute absorption in this slice relative to I_0
            absolute_layer_absorption = (intensity_entering - intensity_exiting) * 100
            
            layer_names_display.append(f"L{idx+1}: {layer['name']}\n({layer['thickness_um']} μm)")
            layer_absorptions.append(absolute_layer_absorption)
            
            # Cascade the remaining beam energy to the next layer down
            current_transmission_fraction = intensity_exiting

        # 3. Render the 2D Schematic Stack Map
        col_map, col_metrics = st.columns([2, 1])
        
        with col_map:
            fig3, ax3 = plt.subplots(figsize=(10, 4.5))
            
            # Draw a horizontal stack representing the device cross-section
            y_pos = np.arange(len(layer_names_display))
            
            # Color map scaling from cool (low absorption) to dangerous hot red (high absorption)
            colors = plt.cm.get_cmap('YlOrRd')(np.array(layer_absorptions) / max(max(layer_absorptions), 1.0))
            
            bars = ax3.barh(y_pos, layer_absorptions, color=colors, edgecolor='#374151', height=0.6)
            
            # Labeling tweaks
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(layer_names_display, fontsize=10, fontweight='bold', color='#ffffff')
            ax3.set_xlabel(f"Absolute Percentage of Original Beam Absorbed Inside Layer (%)", fontsize=11, color='#ffffff')
            ax3.set_title(f"2D Microstructural Radiation Load Profile ({target_energy} keV Target)", fontsize=13, fontweight='bold', color='#00f3ff')
            ax3.grid(axis='x', linestyle='--', alpha=0.2)
            ax3.set_xlim(0, max(max(layer_absorptions) * 1.15, 10))
            
            # Add value tags directly onto the bars
            for bar, val in zip(bars, layer_absorptions):
                ax3.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                         f"{val:.3f}%", 
                         va='center', ha='left', fontsize=10, fontweight='bold', color='#ffffff')
            
            # Stylize plot background to fit dark mode perfectly
            fig3.patch.set_facecolor('#0b0f19')
            ax3.set_facecolor('#111827')
            ax3.spines['bottom'].set_color('#374151')
            ax3.spines['left'].set_color('#374151')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.tick_params(colors='#ffffff')
            
            st.pyplot(fig3)
            plt.close(fig3)

        with col_metrics:
            st.subheader("⚠️ Stability Assessment")
            
            highest_absorbed_idx = np.argmax(layer_absorptions)
            highest_absorbed_val = layer_absorptions[highest_absorbed_idx]
            highest_layer_name = st.session_state.layers[highest_absorbed_idx]["name"]
            
            if highest_absorbed_val > 50.0:
                st.error(f"**Critical Stress Alert:** The primary radiation load is sinking directly into **{highest_layer_name}** ({highest_absorbed_val:.2f}% total beam loss). Watch out for operational phase degradation or secondary ion displacement traps here.")
            elif highest_absorbed_val > 5.0:
                st.warning(f"**Moderate Stress Warning:** **{highest_layer_name}** acts as the primary beam block here ({highest_absorbed_val:.2f}% absorption). Ensure encapsulants can handle localized thermal dissipation.")
            else:
                st.success("**Low Radiation Burden:** Energy dispersion is highly distributed. No single layer is bearing a critical load bottleneck at this specific keV signature.")
