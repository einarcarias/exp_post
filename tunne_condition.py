import numpy as np
import sympy as sp

GAMMA = 1.4


def static_temperature(T0, M):
    return T0 / (1 + ((GAMMA - 1) / 2) * M**2)


def speed_of_sound(T):
    return np.sqrt(GAMMA * T * 287)


def calc_var_terms(var, k_constant):
    return (k_constant * var) ** 2


def velocity(M, a):
    return M * a


def density(mu, vel, re):
    return mu * re / vel


def static_pressure(rho, T):
    return rho * T * 287


def visc(static_temp):  # calculating viscocity of dry air using sutherland's law
    # Dry air gas characteristics
    mu_0 = 1.7894e-5
    t_ref = 273.11
    S = 110.56  # effective temperature (sutherland constant)

    mu = mu_0 * (static_temp / t_ref) ** (3 / 2) * ((t_ref + S) / (static_temp + S))

    return mu


def find_k(symolic_eq, var):
    differential = symolic_eq.diff(var)

    return differential * (var / symolic_eq)


def main():
    re = 9.01e6  # per meter
    re_var = (7.6 / 100 * re) / re

    mach = 8.11
    mach_std = 0.76

    t_0 = 1290
    t_0_var = (3 / 100 * t_0) / t_0

    # NOTE: var is when std/mean

    t_s = static_temperature(t_0, mach)
    t_s_var = np.sqrt(calc_var_terms(t_0_var, 1) + calc_var_terms(mach_std / mach, 2))

    a = speed_of_sound(t_s)
    a_var = np.sqrt(calc_var_terms(t_s_var, 0.5))

    v = velocity(mach, a)
    v_var = np.sqrt(calc_var_terms(mach_std / mach, 1) + calc_var_terms(a_var, 1))

    mu = visc(t_s)
    mu_0_sym = sp.symbols("mu_0")
    t_ref_sym = sp.symbols("t_ref")
    S_sym = sp.symbols("S")
    t_sym = sp.symbols("t")
    mu_sym = (
        mu_0_sym
        * (t_sym / t_ref_sym) ** (3 / 2)
        * ((t_ref_sym + S_sym) / (t_sym + S_sym))
    )
    k = find_k(mu_sym, t_sym)
    # add variables to k
    k = k.subs(mu_0_sym, mu_0_sym)
    k = k.subs(t_ref_sym, t_ref_sym)
    k = k.subs(S_sym, S_sym)
    k = k.subs(t_sym, t_s)
    k = k.subs(mu_0_sym, 1.7894e-5)
    k = k.subs(t_ref_sym, 273.11)
    k = k.subs(S_sym, 110.56)
    mu_var = np.sqrt(calc_var_terms(t_s_var, float(k)))

    rho = density(mu, v, re)
    rho_var = np.sqrt(
        calc_var_terms(v_var, 1) + calc_var_terms(re_var, 1) + calc_var_terms(mu_var, 1)
    )

    p_s = static_pressure(rho, t_s)
    p_s_var = np.sqrt(calc_var_terms(rho_var, 1) + calc_var_terms(t_s_var, 1))

    p_0 = (t_0 / t_s) ** (GAMMA / (GAMMA - 1)) * p_s
    p_0_var = np.sqrt(
        calc_var_terms(t_0_var, 1)
        + calc_var_terms(t_s_var, 1)
        + calc_var_terms(p_s_var, 1)
    )

    # variance to percentage
    mach_perc = mach_std / mach * 100
    t_0_perc = t_0_var  * 100
    t_s_perc = t_s_var  * 100
    a_perc = a_var  * 100
    v_perc = v_var  * 100
    rho_perc = (rho_var ) * 100
    p_s_perc = p_s_var  * 100
    p_0_perc = p_0_var  * 100
    mu_perc = mu_var  * 100
    re_perc = re_var  * 100

    # print results with corresponding percentage
    print(f"mach: {mach:.3f} +- {mach_perc:.3f}")
    print(f"t_0: {t_0:.3f} +- {t_0_perc:.3f}")
    print(f"t_s: {t_s:.3f} +- {t_s_perc:.3f}")
    print(f"a: {a:.3f} +- {a_perc:.3f}")
    print(f"v: {v:.3f} +- {v_perc:.3f}")
    print(f"rho: {rho:.4f} +- {rho_perc:.3f}")
    print(f"p_s: {p_s:.3f} +- {p_s_perc:.3f}")
    print(f"p_0: {p_0:.3f} +- {p_0_perc:.3f}")
    print(f"re: {re} +- {re_perc:.3f}")
    print(f"mu: {mu} +- {mu_perc:.3f}")


if __name__ == "__main__":
    main()
