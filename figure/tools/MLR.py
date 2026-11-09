import numpy as np
import matplotlib.pyplot as plt


bohr_to_A = 0.529177210903
cm_to_ha = 1.0 / 219474.63137
# E_atom = -54.58916
E_atom = -54.5892
E_diss = 2.0 * E_atom            # -109.17832 Ha

def create_MLR_function(phi, p, NS, NL):

    De   = 79845.0        # cm^-1
    C6   = 1.16e5         # cm^-1 Å^6
    R86  = 5.5            # Å^2 -> C8 = R86 * C6
    re   = 1.097679       # Å

    def y_n(r, n, re_=re):
        """y_n(r) = (r^n - r_e^n) / (r^n + r_e^n)"""
        return (r**n - re_**n) / (r**n + re_**n)

    def u_LR(r):
        """Long-range interaction term u_LR(r) = C6 / r^6 + C8 / r^8"""
        C8 = R86 * C6
        return C6 / r**6 + C8 / r**8

    def beta_of_r(r):
        """β(r) = y_p * β_inf + (1 - y_p) * Σ φ_i [y_p]^i"""
        y_p = y_n(r, p)  # p is the degree of the polynomial
        beta_inf = np.log((2.0 * De) / u_LR(re))
        
        # Calculate the polynomial terms (NS and NL define the polynomial degree at different ranges)
        powers = np.vstack([y_p**i for i in range(len(phi))])  
        poly_NL = np.dot(phi[:NL+1], powers[:NL+1])               # i=0..NL
        poly_NS = np.dot(phi[:NS+1], powers[:NS+1])               # i=0..NS
        poly = np.where(np.asarray(r) < re, poly_NS, poly_NL)

        return y_p * beta_inf + (1.0 - y_p) * poly

    def V_MLR_cm(r):
        """MLR potential in cm^-1 (relative, 0 at r_e, → De at ∞)"""
        y_p = y_n(r, p)
        beta = beta_of_r(r)
        ratio = u_LR(r) / u_LR(re)
        exp_term = np.exp(-beta * y_p)
        return De * (1.0 - ratio * exp_term)**2

    def MLR(r):
        """Absolute energy in Hartree"""
        Vcm = V_MLR_cm(r * bohr_to_A)  # Convert Bohr to Å
        return E_diss - (De - Vcm) * cm_to_ha

    return MLR


def create_EMO_function(phi, p, N):

    De   = 79845.0        # cm^-1
    re   = 1.097679       # Å
    C6   = 1.16e5         # cm^-1 Å^6
    R86  = 5.5            # Å^2
    C8   = R86 * C6       # cm^-1 Å^8

    def y_n(r, p, re_=re):
        """y_p(r) = (r^p - r_e^p) / (r^p + r_e^p)"""
        return (r**p - re_**p) / (r**p + re_**p)

    def beta_of_r(r):
        """β(r) = Σ φ_i y_p(r)^i"""
        y_p_val = y_n(r, p)  # p is the degree of the polynomial
        powers = np.vstack([y_p_val**i for i in range(len(phi))])
        return np.dot(phi[:N+1], powers[:N+1])

    def V_EMO_cm(r):
        """EMO potential in cm^-1 (relative, 0 at r_e, → De at ∞)"""
        beta = beta_of_r(r)
        exp_term = np.exp(-beta * (r-re))
        return De * (1.0 - exp_term)**2

    def EMO(r):
        """Absolute energy in Hartree"""
        Vcm = V_EMO_cm(r * bohr_to_A)  # Convert Bohr to Å
        return E_diss - (De - Vcm) * cm_to_ha

    return EMO

if __name__ == "__main__":
    #MLR_3(9)
    phi3 = np.array([
    -3.125518, -1.338651, -1.926378, -1.053483, -1.088594,
    0.253611, 0.767715, -1.929323, 1.583792, 15.210659
    ])
    MLR3 = create_MLR_function(phi=phi3,p=3,NS=9,NL=9)

    #MLR_4(6,8)
    phi4 = np.array([
    -2.34414547, -0.972469, -1.561777, -1.136, -1.3963,
    -0.819, -0.45, -3.36, 2.1
    ])
    MLR4 = create_MLR_function(phi=phi4,p=4,NS=6,NL=8)

    #EMO_2(6)
    phi_EMO = np.array([
    2.689598, 0.332343, 0.665061, 0.729573, 1.157254,
    2.824968, 12.341291
    ])
    EMO = create_EMO_function(phi_EMO, p=2, N=6)

    rs = np.linspace(0.8, 6, 8000)  
    emo_energies = EMO(rs)
    mlr3_energies = MLR3(rs)
    mlr4_energies = MLR4(rs)

    plt.figure(figsize=(7,4))
    plt.plot(rs, emo_energies-mlr3_energies, label="EMO_2(6)")
    plt.plot(rs, mlr3_energies-mlr3_energies, label="MLR_3(9)")
    plt.plot(rs, mlr4_energies-mlr3_energies, label="MLR_4(6,8)")
    plt.axhline(E_diss, ls="--", color="k", lw=1.0, label="Dissociation limit")
    plt.xlabel("r (bohr)")
    plt.ylabel("Energy (Hartree)")
    plt.title("several baseline")
    # plt.ylim(-109.7, -109.1)
    plt.xlim(0.8, 6)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
