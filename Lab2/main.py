import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================================================================
# Лабораторна робота №2
# Побудова функцій приналежності нечітких множин
# ==============================================================================

plt.rcParams["font.size"] = 11


# ---------- допоміжні П-, S-, Z-подібні функції для апроксимації ----------

def pi_function(x, a, b, c, d):
    """П-подібна (трапецієподібна) функція приналежності."""
    result = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        if xi <= a or xi >= d:
            result[i] = 0.0
        elif a < xi <= b:
            result[i] = (xi - a) / (b - a)
        elif b < xi <= c:
            result[i] = 1.0
        elif c < xi < d:
            result[i] = (d - xi) / (d - c)
    return result


def s_function(x, a, b):
    """S-подібна функція приналежності (зростаюча)."""
    result = np.zeros_like(x, dtype=float)
    m = (a + b) / 2.0
    for i, xi in enumerate(x):
        if xi <= a:
            result[i] = 0.0
        elif xi <= m:
            result[i] = 2.0 * ((xi - a) / (b - a)) ** 2
        elif xi <= b:
            result[i] = 1.0 - 2.0 * ((xi - b) / (b - a)) ** 2
        else:
            result[i] = 1.0
    return result


def z_function(x, a, b):
    """Z-подібна функція приналежності (спадна)."""
    return 1.0 - s_function(x, a, b)


# ==============================================================================
# Завдання 1. Побудова ФП на основі експертної інформації
# ==============================================================================

def task1():
    print("=" * 60)
    print("Завдання 1: Нечітка множина «чоловік середнього зросту»")
    print("=" * 60)

    # Елементи універсальної множини (зріст від 150 до 200 см, крок 5)
    heights = np.arange(150, 205, 5)  # 150, 155, ..., 200
    n = len(heights)

    # Дані опитування 5 експертів (бінарні оцінки: 1 — середній зріст, 0 — ні)
    # Кожен рядок — відповіді одного експерта для кожного зросту
    experts = np.array([
        # 150 155 160 165 170 175 180 185 190 195 200
        [0,  0,  0,  1,  1,  1,  1,  1,  0,  0,  0],  # Експерт 1
        [0,  0,  1,  1,  1,  1,  1,  0,  0,  0,  0],  # Експерт 2
        [0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  0],  # Експерт 3
        [0,  0,  0,  1,  1,  1,  1,  0,  0,  0,  0],  # Експерт 4
        [0,  0,  1,  1,  1,  1,  1,  1,  0,  0,  0],  # Експерт 5
    ])
    k = experts.shape[0]

    # Виведення опитувальника
    print(f"\nКількість експертів: {k}")
    print(f"Елементи множини: {[int(h) for h in heights]} см\n")

    print("Таблиця опитування експертів:")
    header = f"{'Експерт':>10}"
    for h in heights:
        header += f" {h:>5}"
    print(header)
    print("-" * len(header))
    for e_idx in range(k):
        row = f"{'Експерт ' + str(e_idx + 1):>10}"
        for val in experts[e_idx]:
            row += f" {val:>5}"
        print(row)

    # Обчислення ступенів приналежності: μ(xi) = (1/k) * Σ bik
    mu = np.sum(experts, axis=0) / k
    print(f"\nСтупені приналежності μ(xi):")
    for i in range(n):
        print(f"  μ({heights[i]} см) = {mu[i]:.2f}")

    # Нормалізація (якщо множина субнормальна)
    mu_max = np.max(mu)
    if mu_max < 1.0:
        mu_norm = mu / mu_max
        print(f"\nМножина субнормальна (max μ = {mu_max:.2f}), нормалізуємо:")
        for i in range(n):
            print(f"  μ_norm({heights[i]} см) = {mu_norm[i]:.2f}")
    else:
        mu_norm = mu.copy()
        print(f"\nМножина нормальна (max μ = {mu_max:.2f}), нормалізація не потрібна.")

    # Апроксимація П-подібною функцією
    x_fine = np.linspace(heights[0], heights[-1], 300)
    try:
        popt, _ = curve_fit(pi_function, heights.astype(float), mu_norm,
                            p0=[155, 168, 182, 195], maxfev=10000)
        a, b, c, d = popt
        y_approx = pi_function(x_fine, *popt)
        print(f"\nПараметри П-подібної апроксимації:")
        print(f"  a = {a:.1f}, b = {b:.1f}, c = {c:.1f}, d = {d:.1f}")
    except Exception as e:
        print(f"\nПомилка апроксимації: {e}")
        a, b, c, d = 157, 168, 182, 193
        y_approx = pi_function(x_fine, a, b, c, d)
        popt = (a, b, c, d)

    # Побудова графіка
    plt.figure(figsize=(9, 5))
    plt.bar(heights, mu_norm, width=3, color="#5DADE2", alpha=0.6,
            edgecolor="#2E86C1", label="Експертні дані (нормалізовані)")
    plt.plot(x_fine, y_approx, color="#E74C3C", linewidth=2.5,
             label=f"П-подібна апроксимація\n(a={popt[0]:.1f}, b={popt[1]:.1f}, c={popt[2]:.1f}, d={popt[3]:.1f})")
    plt.title("Функція приналежності «чоловік середнього зросту»", fontsize=13, fontweight="bold")
    plt.xlabel("Зріст, см", fontsize=11)
    plt.ylabel("μ(x)", fontsize=11)
    plt.xticks(heights)
    plt.ylim(-0.05, 1.15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    plt.savefig("plots/task1_membership.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/task1_membership.png")

    return heights, mu_norm


# ==============================================================================
# Завдання 2. Побудова ФП на основі попарних порівнянь
# ==============================================================================

def pairwise_membership(matrix):
    """
    Обчислення ступеня приналежності через власний вектор матриці
    попарних порівнянь (метод середнього геометричного).
    ωi = n√(∏j aij) / Σi n√(∏j aij)
    """
    n = matrix.shape[0]
    # Середнє геометричне кожного рядка
    geo_means = np.prod(matrix, axis=1) ** (1.0 / n)
    # Нормалізація
    omega = geo_means / np.sum(geo_means)
    return omega


def normalize_membership(mu):
    """Нормалізація функції приналежності до [0, 1]."""
    mu_max = np.max(mu)
    if mu_max > 0:
        return mu / mu_max
    return mu


def find_support_core_boundaries(heights, mu, alpha=0.5):
    """
    Визначення носія, ядра та α-границь нечіткої множини.
    Носій: {x | μ(x) > 0}
    Ядро:  {x | μ(x) = 1}
    Границі (α-переріз): {x | μ(x) >= α}
    """
    support = [int(x) for x in heights[mu > 0]]
    core = [int(x) for x in heights[mu >= 0.999]]  # ≈ 1.0 з урахуванням числової точності
    boundaries = [int(x) for x in heights[mu >= alpha]]
    return support, core, boundaries


def task2():
    print("\n" + "=" * 60)
    print("Завдання 2: Попарні порівняння")
    print("=" * 60)

    # --- 2а: Нечітка множина «висока людина» ---
    heights_high = np.array([170, 175, 180, 185, 190, 195, 200])

    # Матриця попарних порівнянь (шкала Сааті)
    # Чим більший зріст, тим більша перевага щодо властивості «висока людина»
    A_high = np.array([
        [1,   1/3, 1/5, 1/7, 1/9, 1/9, 1/9],
        [3,   1,   1/3, 1/5, 1/7, 1/9, 1/9],
        [5,   3,   1,   1/3, 1/5, 1/7, 1/9],
        [7,   5,   3,   1,   1/3, 1/5, 1/7],
        [9,   7,   5,   3,   1,   1/3, 1/5],
        [9,   9,   7,   5,   3,   1,   1/3],
        [9,   9,   9,   7,   5,   3,   1  ],
    ])

    print("\n--- Нечітка множина «висока людина» ---")
    print(f"Елементи: {[int(h) for h in heights_high]} см")
    print("\nМатриця попарних порівнянь:")
    print_matrix(A_high, heights_high)

    omega_high = pairwise_membership(A_high)
    mu_high = normalize_membership(omega_high)

    print("\nВласний вектор (ωi):")
    for i in range(len(heights_high)):
        print(f"  ω({heights_high[i]} см) = {omega_high[i]:.4f}")

    print("\nНормалізована ФП:")
    for i in range(len(heights_high)):
        print(f"  μ({heights_high[i]} см) = {mu_high[i]:.4f}")

    # Носій, ядро, границі
    support_h, core_h, bound_h = find_support_core_boundaries(heights_high, mu_high)
    print(f"\nНосій: {support_h} см")
    print(f"Ядро:  {core_h} см")
    print(f"Границі (α=0.5): {bound_h} см")

    # Апроксимація S-подібною функцією
    x_fine = np.linspace(heights_high[0], heights_high[-1], 300)
    try:
        popt_s, _ = curve_fit(s_function, heights_high.astype(float), mu_high,
                              p0=[170, 200], maxfev=10000)
        y_s = s_function(x_fine, *popt_s)
        print(f"\nПараметри S-подібної апроксимації: a = {popt_s[0]:.1f}, b = {popt_s[1]:.1f}")
    except Exception as e:
        print(f"\nПомилка апроксимації: {e}")
        popt_s = [170, 200]
        y_s = s_function(x_fine, *popt_s)

    # Графік «висока людина»
    plt.figure(figsize=(9, 5))
    plt.stem(heights_high, mu_high, linefmt="#2E86C1", markerfmt="o",
             basefmt=" ", label="Дискретні μ (попарні порівняння)")
    plt.plot(x_fine, y_s, color="#E74C3C", linewidth=2.5,
             label=f"S-подібна апроксимація (a={popt_s[0]:.1f}, b={popt_s[1]:.1f})")
    plt.title("Функція приналежності «висока людина»", fontsize=13, fontweight="bold")
    plt.xlabel("Зріст, см", fontsize=11)
    plt.ylabel("μ(x)", fontsize=11)
    plt.xticks(heights_high)
    plt.ylim(-0.05, 1.15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/task2_high.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/task2_high.png")

    # --- 2б: Нечітка множина «низька людина» ---
    heights_low = np.array([150, 155, 160, 165, 170, 175, 180])

    # Чим менший зріст, тим більша перевага щодо властивості «низька людина»
    A_low = np.array([
        [1,   3,   5,   7,   9,   9,   9  ],
        [1/3, 1,   3,   5,   7,   9,   9  ],
        [1/5, 1/3, 1,   3,   5,   7,   9  ],
        [1/7, 1/5, 1/3, 1,   3,   5,   7  ],
        [1/9, 1/7, 1/5, 1/3, 1,   3,   5  ],
        [1/9, 1/9, 1/7, 1/5, 1/3, 1,   3  ],
        [1/9, 1/9, 1/9, 1/7, 1/5, 1/3, 1  ],
    ])

    print("\n--- Нечітка множина «низька людина» ---")
    print(f"Елементи: {[int(h) for h in heights_low]} см")
    print("\nМатриця попарних порівнянь:")
    print_matrix(A_low, heights_low)

    omega_low = pairwise_membership(A_low)
    mu_low = normalize_membership(omega_low)

    print("\nВласний вектор (ωi):")
    for i in range(len(heights_low)):
        print(f"  ω({heights_low[i]} см) = {omega_low[i]:.4f}")

    print("\nНормалізована ФП:")
    for i in range(len(heights_low)):
        print(f"  μ({heights_low[i]} см) = {mu_low[i]:.4f}")

    # Носій, ядро, границі
    support_l, core_l, bound_l = find_support_core_boundaries(heights_low, mu_low)
    print(f"\nНосій: {support_l} см")
    print(f"Ядро:  {core_l} см")
    print(f"Границі (α=0.5): {bound_l} см")

    # Апроксимація Z-подібною функцією
    x_fine_l = np.linspace(heights_low[0], heights_low[-1], 300)
    try:
        popt_z, _ = curve_fit(z_function, heights_low.astype(float), mu_low,
                              p0=[150, 180], maxfev=10000)
        y_z = z_function(x_fine_l, *popt_z)
        print(f"\nПараметри Z-подібної апроксимації: a = {popt_z[0]:.1f}, b = {popt_z[1]:.1f}")
    except Exception as e:
        print(f"\nПомилка апроксимації: {e}")
        popt_z = [150, 180]
        y_z = z_function(x_fine_l, *popt_z)

    # Графік «низька людина»
    plt.figure(figsize=(9, 5))
    plt.stem(heights_low, mu_low, linefmt="#27AE60", markerfmt="o",
             basefmt=" ", label="Дискретні μ (попарні порівняння)")
    plt.plot(x_fine_l, y_z, color="#8E44AD", linewidth=2.5,
             label=f"Z-подібна апроксимація (a={popt_z[0]:.1f}, b={popt_z[1]:.1f})")
    plt.title("Функція приналежності «низька людина»", fontsize=13, fontweight="bold")
    plt.xlabel("Зріст, см", fontsize=11)
    plt.ylabel("μ(x)", fontsize=11)
    plt.xticks(heights_low)
    plt.ylim(-0.05, 1.15)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("plots/task2_low.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/task2_low.png")

    return (heights_high, mu_high, popt_s, heights_low, mu_low, popt_z)


def print_matrix(matrix, labels):
    """Виведення матриці попарних порівнянь у зручному форматі."""
    n = len(labels)
    header = f"{'':>6}"
    for lbl in labels:
        header += f" {lbl:>7}"
    print(header)
    print("-" * len(header))
    for i in range(n):
        row = f"{labels[i]:>6}"
        for j in range(n):
            val = matrix[i, j]
            if val >= 1:
                row += f" {val:>7.1f}"
            else:
                row += f"   1/{1/val:.0f}  "
        print(row)


# ==============================================================================
# Головний блок
# ==============================================================================

def main():
    os.makedirs("plots", exist_ok=True)

    task1()
    task2()

    print("\n" + "=" * 60)
    print("Усі розрахунки та графіки успішно згенеровано.")
    print("=" * 60)


if __name__ == "__main__":
    main()
