import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Лабораторна робота №3: Операції над нечіткими множинами
# Дисципліна: М'які обчислення
# Студент: Чепара Станіслав, група ФеП-32
# ==============================================================================

# Налаштування стилю відображення графіків
plt.rcParams["font.size"] = 11
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.5


# ==============================================================================
# 1. Функції приналежності базових нечітких множин
# ==============================================================================

def mu_low(x):
    """
    Функція приналежності для нечіткої множини A = «низька людина».
    Універсум: [150, 200] см.
    - x <= 160: 1.0
    - 160 < x < 180: лінійний спад від 1.0 до 0.0
    - x >= 180: 0.0
    """
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        if val <= 160.0:
            res[i] = 1.0
        elif 160.0 < val < 180.0:
            res[i] = (180.0 - val) / 20.0
        else:
            res[i] = 0.0
    return res


def mu_high(x):
    """
    Функція приналежності для нечіткої множини B = «висока людина».
    Універсум: [150, 200] см.
    - x <= 170: 0.0
    - 170 < x < 190: лінійне зростання від 0.0 до 1.0
    - x >= 190: 1.0
    """
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        if val <= 170.0:
            res[i] = 0.0
        elif 170.0 < val < 190.0:
            res[i] = (val - 170.0) / 20.0
        else:
            res[i] = 1.0
    return res


def mu_medium(x):
    """
    Допоміжна множина C = «людина середнього зросту» (для перевірки асоціативності та дистрибутивності).
    Трикутна функція з вершиною в 175 см (діапазон [165, 185] см).
    """
    x = np.asarray(x, dtype=float)
    res = np.zeros_like(x)
    for i, val in enumerate(x):
        if 165.0 < val <= 175.0:
            res[i] = (val - 165.0) / 10.0
        elif 175.0 < val < 185.0:
            res[i] = (185.0 - val) / 10.0
        else:
            res[i] = 0.0
    return res


# ==============================================================================
# 2. Реалізація операцій над нечіткими множинами (Заде)
# ==============================================================================

def fuzzy_complement(mu):
    """Операція доповнення: not(A) = 1 - mu_A"""
    return 1.0 - mu


def fuzzy_intersection(mu1, mu2):
    """Операція перетину: A and B = min(mu1, mu2)"""
    return np.minimum(mu1, mu2)


def fuzzy_union(mu1, mu2):
    """Операція об'єднання: A or B = max(mu1, mu2)"""
    return np.maximum(mu1, mu2)


def fuzzy_difference(mu1, mu2):
    """Операція різниці: A \\ B = A ∩ not(B) = min(mu1, 1 - mu2)"""
    return np.minimum(mu1, 1.0 - mu2)


def fuzzy_symmetric_difference(mu1, mu2):
    """Операція симетричної різниці: A Δ B = (A \\ B) ∪ (B \\ A)"""
    diff_a_b = fuzzy_difference(mu1, mu2)
    diff_b_a = fuzzy_difference(mu2, mu1)
    return np.maximum(diff_a_b, diff_b_a)


def fuzzy_concentration(mu):
    """Операція концентрування (модифікатор «дуже»): CON(A) = mu^2"""
    return mu ** 2


def fuzzy_dilation(mu):
    """Операція розтягування (модифікатор «злегка»): DIL(A) = sqrt(mu)"""
    return np.sqrt(mu)


# ==============================================================================
# 3. Виконання пунктів лабораторної роботи
# ==============================================================================

def task1(x_dense, heights):
    """Пункт 1. Задати функції приналежності A та B та побудувати їх графіки."""
    print("\n" + "=" * 75)
    print("Пункт 1. Базові нечіткі множини A={низька людина} та B={висока людина}")
    print("=" * 75)

    mu_a_dense = mu_low(x_dense)
    mu_b_dense = mu_high(x_dense)

    mu_a_disc = mu_low(heights)
    mu_b_disc = mu_high(heights)

    print(f"{'Зріст, см':<12} | " + " | ".join(f"{int(h):<5}" for h in heights))
    print("-" * 85)
    print(f"{'μ_A (низька)':<12} | " + " | ".join(f"{val:<5.2f}" for val in mu_a_disc))
    print(f"{'μ_B (висока)':<12} | " + " | ".join(f"{val:<5.2f}" for val in mu_b_disc))

    plt.figure(figsize=(9, 5))
    plt.plot(x_dense, mu_a_dense, label="A (низька людина)", color="#2980B9", linewidth=2.5)
    plt.plot(x_dense, mu_b_dense, label="B (висока людина)", color="#C0392B", linewidth=2.5)
    plt.scatter(heights, mu_a_disc, color="#2980B9", s=30, zorder=5)
    plt.scatter(heights, mu_b_disc, color="#C0392B", s=30, zorder=5)

    plt.title("Базові функції приналежності на універсумі [150, 200] см", fontweight="bold")
    plt.xlabel("Зріст (x), см")
    plt.ylabel("Ступінь приналежності μ(x)")
    plt.xticks(heights)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="center")
    plt.tight_layout()
    plt.savefig("plots/task1_base_sets.png", dpi=300)
    plt.close()
    print("-> Графік збережено: plots/task1_base_sets.png")


def task2(x_dense, heights):
    """Пункт 2. Програмування операцій над нечіткими множинами."""
    print("\n" + "=" * 75)
    print("Пункт 2. Результати виконання операцій над нечіткими множинами")
    print("=" * 75)

    mu_a = mu_low(x_dense)
    mu_b = mu_high(x_dense)

    comp_a = fuzzy_complement(mu_a)
    comp_b = fuzzy_complement(mu_b)
    inter_ab = fuzzy_intersection(mu_a, mu_b)
    union_ab = fuzzy_union(mu_a, mu_b)
    diff_ab = fuzzy_difference(mu_a, mu_b)
    diff_ba = fuzzy_difference(mu_b, mu_a)
    sym_diff = fuzzy_symmetric_difference(mu_a, mu_b)
    con_a = fuzzy_concentration(mu_a)
    dil_a = fuzzy_dilation(mu_a)

    a_disc = mu_low(heights)
    b_disc = mu_high(heights)

    print("\nТаблиця значень операцій у точках дискретизації:")
    header = f"{'Зріст':<6} | {'A':<5} | {'B':<5} | {'not A':<5} | {'A ∩ B':<5} | {'A ∪ B':<5} | {'A \\ B':<5} | {'B \\ A':<5} | {'A Δ B':<5} | {'CON(A)':<6} | {'DIL(A)':<6}"
    print(header)
    print("-" * len(header))
    for i, h in enumerate(heights):
        va = a_disc[i]
        vb = b_disc[i]
        print(f"{int(h):<6} | {va:<5.2f} | {vb:<5.2f} | {fuzzy_complement(va):<5.2f} | "
              f"{fuzzy_intersection(va, vb):<5.2f} | {fuzzy_union(va, vb):<5.2f} | "
              f"{fuzzy_difference(va, vb):<5.2f} | {fuzzy_difference(vb, va):<5.2f} | "
              f"{fuzzy_symmetric_difference(va, vb):<5.2f} | {fuzzy_concentration(va):<6.2f} | {fuzzy_dilation(va):<6.2f}")

    # Зведений графік операцій
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))

    # Перетин та об'єднання
    axs[0, 0].plot(x_dense, mu_a, "--", color="#7F8C8D", label="A (низька)", alpha=0.5)
    axs[0, 0].plot(x_dense, mu_b, "--", color="#BDC3C7", label="B (висока)", alpha=0.5)
    axs[0, 0].plot(x_dense, inter_ab, color="#27AE60", linewidth=2.5, label="Перетин A ∩ B")
    axs[0, 0].plot(x_dense, union_ab, color="#8E44AD", linewidth=2, label="Об'єднання A ∪ B")
    axs[0, 0].set_title("Перетин (A ∩ B) та Об'єднання (A ∪ B)", fontweight="bold")
    axs[0, 0].set_xlabel("Зріст, см")
    axs[0, 0].set_ylabel("μ(x)")
    axs[0, 0].set_ylim(-0.05, 1.05)
    axs[0, 0].legend()

    # Доповнення
    axs[0, 1].plot(x_dense, mu_a, color="#2980B9", alpha=0.35, label="A (низька)")
    axs[0, 1].plot(x_dense, comp_a, color="#1F618D", linewidth=2.5, label="Доповнення not(A)")
    axs[0, 1].plot(x_dense, mu_b, color="#C0392B", alpha=0.35, label="B (висока)")
    axs[0, 1].plot(x_dense, comp_b, color="#922B21", linewidth=2.5, linestyle=":", label="Доповнення not(B)")
    axs[0, 1].set_title("Доповнення not(A) та not(B)", fontweight="bold")
    axs[0, 1].set_xlabel("Зріст, см")
    axs[0, 1].set_ylabel("μ(x)")
    axs[0, 1].set_ylim(-0.05, 1.05)
    axs[0, 1].legend()

    # Різниця та симетрична різниця
    axs[1, 0].plot(x_dense, diff_ab, color="#E67E22", linewidth=2, label="Різниця A \\ B")
    axs[1, 0].plot(x_dense, diff_ba, color="#D35400", linewidth=2, linestyle="--", label="Різниця B \\ A")
    axs[1, 0].plot(x_dense, sym_diff, color="#16A085", linewidth=2.5, label="Сим. різниця A Δ B")
    axs[1, 0].set_title("Різниця (A \\ B, B \\ A) та Сим. різниця (A Δ B)", fontweight="bold")
    axs[1, 0].set_xlabel("Зріст, см")
    axs[1, 0].set_ylabel("μ(x)")
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].legend()

    # Концентрування та розтягування
    axs[1, 1].plot(x_dense, mu_a, color="#2980B9", linewidth=1.5, label="A (базова)")
    axs[1, 1].plot(x_dense, con_a, color="#8E44AD", linewidth=2, linestyle="-.", label="CON(A) = A² (дуже низька)")
    axs[1, 1].plot(x_dense, dil_a, color="#27AE60", linewidth=2, label="DIL(A) = √A (злегка низька)")
    axs[1, 1].set_title("Концентрування CON(A) та Розтягування DIL(A)", fontweight="bold")
    axs[1, 1].set_xlabel("Зріст, см")
    axs[1, 1].set_ylabel("μ(x)")
    axs[1, 1].set_ylim(-0.05, 1.05)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig("plots/task2_operations.png", dpi=300)
    plt.close()
    print("-> Графік збережено: plots/task2_operations.png")


def task3(x_dense, heights):
    """Пункт 3. Моделювання висловлювань за допомогою операцій."""
    print("\n" + "=" * 75)
    print("Пункт 3. Графіки функцій приналежності для висловлювань")
    print("=" * 75)

    mu_a = mu_low(x_dense)
    mu_b = mu_high(x_dense)

    # 1. «низька і висока людина»: A ∩ B
    s1 = fuzzy_intersection(mu_a, mu_b)

    # 2. «низька або висока людина»: A ∪ B
    s2 = fuzzy_union(mu_a, mu_b)

    # 3. «не висока людина»: not(B)
    s3 = fuzzy_complement(mu_b)

    # 4. «злегка низька людина»: DIL(A)
    s4 = fuzzy_dilation(mu_a)

    # 5. «дуже висока людина»: CON(B)
    s5 = fuzzy_concentration(mu_b)

    # Дискретні контрольні точки
    a_disc = mu_low(heights)
    b_disc = mu_high(heights)
    print(f"{'Зріст':<6} | {'«низька і висока»':<18} | {'«низька або висока»':<19} | {'«не висока»':<12} | {'«злегка низька»':<16} | {'«дуже висока»':<14}")
    print("-" * 95)
    for i, h in enumerate(heights):
        va = a_disc[i]
        vb = b_disc[i]
        v_s1 = fuzzy_intersection(va, vb)
        v_s2 = fuzzy_union(va, vb)
        v_s3 = fuzzy_complement(vb)
        v_s4 = fuzzy_dilation(va)
        v_s5 = fuzzy_concentration(vb)
        print(f"{int(h):<6} | {v_s1:<18.2f} | {v_s2:<19.2f} | {v_s3:<12.2f} | {v_s4:<16.2f} | {v_s5:<14.2f}")

    # Побудова графіка
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense, s1, label="«низька і висока людина» (A ∩ B)", color="#27AE60", linewidth=2.5)
    plt.plot(x_dense, s2, label="«низька або висока людина» (A ∪ B)", color="#8E44AD", linewidth=2.5)
    plt.plot(x_dense, s3, label="«не висока людина» (not B)", color="#E67E22", linewidth=2.5, linestyle="--")
    plt.plot(x_dense, s4, label="«злегка низька людина» (DIL A)", color="#2980B9", linewidth=2.5, linestyle="-.")
    plt.plot(x_dense, s5, label="«дуже висока людина» (CON B)", color="#C0392B", linewidth=2.5, linestyle=":")

    plt.title("Функції приналежності для висловлювань над множинами A та B", fontweight="bold")
    plt.xlabel("Зріст (x), см")
    plt.ylabel("Ступінь приналежності μ(x)")
    plt.xticks(heights)
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="center left", bbox_to_anchor=(0.52, 0.45))
    plt.tight_layout()
    plt.savefig("plots/task3_linguistic.png", dpi=300)
    plt.close()
    print("-> Графік збережено: plots/task3_linguistic.png")


def task4(x_dense):
    """Пункт 4. Перевірка властивостей операцій над нечіткими множинами."""
    print("\n" + "=" * 75)
    print("Пункт 4. Перевірка властивостей операцій над нечіткими множинами")
    print("=" * 75)

    A = mu_low(x_dense)
    B = mu_high(x_dense)
    C = mu_medium(x_dense)

    results = []

    # 1. Комутативність
    # A ∩ B = B ∩ A
    left_comm_inter = fuzzy_intersection(A, B)
    right_comm_inter = fuzzy_intersection(B, A)
    diff_comm_inter = np.max(np.abs(left_comm_inter - right_comm_inter))
    results.append(("Комутативність перетину", "A ∩ B = B ∩ A", diff_comm_inter < 1e-12, diff_comm_inter))

    # A ∪ B = B ∪ A
    left_comm_union = fuzzy_union(A, B)
    right_comm_union = fuzzy_union(B, A)
    diff_comm_union = np.max(np.abs(left_comm_union - right_comm_union))
    results.append(("Комутативність об'єднання", "A ∪ B = B ∪ A", diff_comm_union < 1e-12, diff_comm_union))

    # 2. Асоціативність
    # (A ∩ B) ∩ C = A ∩ (B ∩ C)
    left_assoc_inter = fuzzy_intersection(fuzzy_intersection(A, B), C)
    right_assoc_inter = fuzzy_intersection(A, fuzzy_intersection(B, C))
    diff_assoc_inter = np.max(np.abs(left_assoc_inter - right_assoc_inter))
    results.append(("Асоціативність перетину", "(A ∩ B) ∩ C = A ∩ (B ∩ C)", diff_assoc_inter < 1e-12, diff_assoc_inter))

    # (A ∪ B) ∪ C = A ∪ (B ∪ C)
    left_assoc_union = fuzzy_union(fuzzy_union(A, B), C)
    right_assoc_union = fuzzy_union(A, fuzzy_union(B, C))
    diff_assoc_union = np.max(np.abs(left_assoc_union - right_assoc_union))
    results.append(("Асоціативність об'єднання", "(A ∪ B) ∪ C = A ∪ (B ∪ C)", diff_assoc_union < 1e-12, diff_assoc_union))

    # 3. Дистрибутивність
    # A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)
    left_dist_1 = fuzzy_intersection(A, fuzzy_union(B, C))
    right_dist_1 = fuzzy_union(fuzzy_intersection(A, B), fuzzy_intersection(A, C))
    diff_dist_1 = np.max(np.abs(left_dist_1 - right_dist_1))
    results.append(("Дистрибутивність ∩ відносно ∪", "A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C)", diff_dist_1 < 1e-12, diff_dist_1))

    # A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
    left_dist_2 = fuzzy_union(A, fuzzy_intersection(B, C))
    right_dist_2 = fuzzy_intersection(fuzzy_union(A, B), fuzzy_union(A, C))
    diff_dist_2 = np.max(np.abs(left_dist_2 - right_dist_2))
    results.append(("Дистрибутивність ∪ відносно ∩", "A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)", diff_dist_2 < 1e-12, diff_dist_2))

    # 4. Інволюція
    # not(not(A)) = A
    left_inv = fuzzy_complement(fuzzy_complement(A))
    diff_inv = np.max(np.abs(left_inv - A))
    results.append(("Інволюція (подвійне доповнення)", "not(not(A)) = A", diff_inv < 1e-12, diff_inv))

    # 5. Закони де Моргана
    # not(A ∪ B) = not(A) ∩ not(B)
    left_dm_1 = fuzzy_complement(fuzzy_union(A, B))
    right_dm_1 = fuzzy_intersection(fuzzy_complement(A), fuzzy_complement(B))
    diff_dm_1 = np.max(np.abs(left_dm_1 - right_dm_1))
    results.append(("Закон де Моргана (1)", "not(A ∪ B) = not(A) ∩ not(B)", diff_dm_1 < 1e-12, diff_dm_1))

    # not(A ∩ B) = not(A) ∪ not(B)
    left_dm_2 = fuzzy_complement(fuzzy_intersection(A, B))
    right_dm_2 = fuzzy_union(fuzzy_complement(A), fuzzy_complement(B))
    diff_dm_2 = np.max(np.abs(left_dm_2 - right_dm_2))
    results.append(("Закон де Моргана (2)", "not(A ∩ B) = not(A) ∪ not(B)", diff_dm_2 < 1e-12, diff_dm_2))

    # Друк таблиці перевірки
    print(f"\n{'Властивість':<32} | {'Рівність':<35} | {'Макс. похибка':<15} | {'Результат'}")
    print("-" * 102)
    for name, expr, passed, max_diff in results:
        status_str = "Виконується (True)" if passed else "Не виконується (False)"
        print(f"{name:<32} | {expr:<35} | {max_diff:<15.2e} | {status_str}")

    # Додатковий аналіз законів виключеного третього та суперечності
    print("\n--- Додатковий аналіз для захисту ---")
    excl_middle = fuzzy_union(A, fuzzy_complement(A))
    min_excl = np.min(excl_middle)
    print(f"Закон виключеного третього A ∪ not(A): min = {min_excl:.2f} != 1.0 (НЕ виконується для нечітких множин)")

    contradiction = fuzzy_intersection(A, fuzzy_complement(A))
    max_contr = np.max(contradiction)
    print(f"Закон суперечності A ∩ not(A): max = {max_contr:.2f} != 0.0 (НЕ виконується для нечітких множин)")


# ==============================================================================
# Точка входу
# ==============================================================================

def main():
    os.makedirs("plots", exist_ok=True)

    x_dense = np.linspace(150, 200, 501)
    heights = np.arange(150, 205, 5)

    print("Початок виконання лабораторної роботи №3...")
    task1(x_dense, heights)
    task2(x_dense, heights)
    task3(x_dense, heights)
    task4(x_dense)

    print("\n" + "=" * 75)
    print("Усі розрахунки та графіки успішно створено у папці plots/")
    print("=" * 75)


if __name__ == "__main__":
    main()
