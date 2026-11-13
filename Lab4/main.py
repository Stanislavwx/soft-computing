import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Лабораторна робота №4
# Композиція нечітких бінарних відношень
# ==============================================================================

# Налаштування стилю графіків
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "DejaVu Sans"

# 1. Множини предметної області
SPECIALTIES = [
    "Менеджер (x1)",
    "Програміст (x2)",
    "Водій (x3)",
    "Секретар (x4)"
]

CHARACTERISTICS = [
    "Гнучкість мислення (y1)",
    "Швидкість рішень (y2)",
    "Концентрація уваги (y3)",
    "Зорова пам'ять (y4)",
    "Витривалість (y5)",
    "Швидкість реакції (y6)",
    "Відповідальність (y7)"
]

CANDIDATES = [
    "Андрієнко (z1)",
    "Василенко (z2)",
    "Іваненко (z3)",
    "Дмитренко (z4)",
    "Петренко (z5)",
    "Романенко (z6)"
]


# 2. Експертні матриці нечітких відношень
def get_expert_matrices():
    """
    Повертає матриці нечітких відношень:
    - MS: профілювання спеціальностей (4 спеціальності x 7 характеристик)
    - MQ: профілювання претендентів (7 характеристик x 6 претендентів)
    """
    # Матриця вимог спеціальностей MS (4x7)
    MS = np.array([
        [0.9, 0.9, 0.6, 0.5, 0.6, 0.4, 0.8],  # Менеджер
        [0.8, 0.5, 0.9, 0.7, 0.5, 0.3, 0.7],  # Програміст
        [0.4, 0.7, 0.8, 0.7, 0.8, 0.9, 0.8],  # Водій
        [0.5, 0.5, 0.7, 0.8, 0.5, 0.5, 0.8]   # Секретар
    ])

    # Матриця якостей претендентів MQ (7x6)
    MQ = np.array([
        # z1   z2   z3   z4   z5   z6
        [0.8, 0.9, 0.4, 0.5, 0.6, 0.3],  # y1: Гнучкість мислення
        [0.5, 0.9, 0.7, 0.5, 0.6, 0.4],  # y2: Швидкість рішень
        [0.9, 0.6, 0.8, 0.7, 0.6, 0.5],  # y3: Концентрація уваги
        [0.7, 0.5, 0.7, 0.9, 0.6, 0.4],  # y4: Зорова пам'ять
        [0.5, 0.6, 0.9, 0.5, 0.6, 0.4],  # y5: Витривалість
        [0.3, 0.4, 0.9, 0.5, 0.5, 0.3],  # y6: Швидкість реакції
        [0.7, 0.8, 0.8, 0.8, 0.6, 0.4]   # y7: Відповідальність
    ])

    return MS, MQ


# 3. Операції композиції нечітких відношень
def max_min_composition(MS, MQ):
    """
    Максмінна композиція (Max-Min):
    R[i, k] = max_j ( min(MS[i, j], MQ[j, k]) )
    """
    n_specialties, n_chars = MS.shape
    _, n_candidates = MQ.shape
    R = np.zeros((n_specialties, n_candidates))

    for i in range(n_specialties):
        for k in range(n_candidates):
            # Поелементний мінімум за характеристиками j
            min_vals = np.minimum(MS[i, :], MQ[:, k])
            # Максимум серед усіх характеристик
            R[i, k] = np.max(min_vals)

    return R


def max_prod_composition(MS, MQ):
    """
    Альтернативна макс-добуткова композиція (Max-Product):
    R[i, k] = max_j ( MS[i, j] * MQ[j, k] )
    """
    n_specialties, n_chars = MS.shape
    _, n_candidates = MQ.shape
    R = np.zeros((n_specialties, n_candidates))

    for i in range(n_specialties):
        for k in range(n_candidates):
            # Алгебраїчний добуток за характеристиками j
            prod_vals = MS[i, :] * MQ[:, k]
            # Максимум серед усіх характеристик
            R[i, k] = np.max(prod_vals)

    return R


# 4. Виведення матриць у консоль
def print_matrix(title, matrix, row_labels, col_labels):
    """Форматоване виведення матриці з підписами рядків та стовпців."""
    print(f"\n{title}:")
    header = f"{'':<25}" + "".join([f"{col:>16}" for col in col_labels])
    print(header)
    print("-" * len(header))
    for i, row_name in enumerate(row_labels):
        row_str = f"{row_name:<25}"
        for val in matrix[i]:
            row_str += f"{val:>16.2f}"
        print(row_str)


# 5. Аналіз переваг і недоліків
def analyze_results(MS, MQ, R_maxmin, R_maxprod):
    """Аналіз та порівняння результатів композицій, визначення переваг і недоліків."""
    print("\n" + "=" * 75)
    print("АНАЛІЗ РЕЗУЛЬТАТІВ ТА РЕКОМЕНДАЦІЙ ДЛЯ ВИБОРУ ПРОФЕСІЙ")
    print("=" * 75)

    short_specs = ["Менеджер", "Програміст", "Водій", "Секретар"]
    short_cands = ["Андрієнко", "Василенко", "Іваненко", "Дмитренко", "Петренко", "Романенко"]

    # 5.1. Аналіз за спеціальностями
    print("\n1. Оптимальні кандидати за спеціальностями:")
    for i, spec in enumerate(short_specs):
        print(f"\n--- Спеціальність: {spec} ---")

        # Найкращі за Max-Min
        best_val_min = np.max(R_maxmin[i, :])
        best_cands_min = [short_cands[k] for k in range(len(short_cands)) if np.isclose(R_maxmin[i, k], best_val_min)]
        print(f"  [Max-Min]  Найкращий ступінь: {best_val_min:.2f} -> {', '.join(best_cands_min)}")

        # Найкращі за Max-Prod
        best_val_prod = np.max(R_maxprod[i, :])
        best_cands_prod = [short_cands[k] for k in range(len(short_cands)) if np.isclose(R_maxprod[i, k], best_val_prod)]
        print(f"  [Max-Prod] Найкращий ступінь: {best_val_prod:.2f} -> {', '.join(best_cands_prod)}")

        # Деталізація сильних та слабких сторін лідера
        lead_idx = int(np.argmax(R_maxprod[i, :]))
        lead_name = short_cands[lead_idx]
        strengths = []
        weaknesses = []
        for j, char in enumerate(CHARACTERISTICS):
            diff = MQ[j, lead_idx] - MS[i, j]
            char_short = char.split(" (")[0]
            if diff >= 0:
                strengths.append(f"{char_short} ({MQ[j, lead_idx]:.1f} >= {MS[i, j]:.1f})")
            elif diff < -0.15:
                weaknesses.append(f"{char_short} ({MQ[j, lead_idx]:.1f} < {MS[i, j]:.1f})")

        print(f"  Переваги лідера ({lead_name}): {', '.join(strengths) if strengths else 'немає явних'}")
        if weaknesses:
            print(f"  Вразливі місця ({lead_name}): {', '.join(weaknesses)}")
        else:
            print(f"  Вразливі місця ({lead_name}): відсутні критичні дефіцити")

    # 5.2. Індивідуальні рекомендації для претендентів
    print("\n2. Індивідуальні рекомендації для кожного претендента:")
    for k, cand in enumerate(short_cands):
        best_spec_min = short_specs[int(np.argmax(R_maxmin[:, k]))]
        best_spec_prod = short_specs[int(np.argmax(R_maxprod[:, k]))]
        worst_spec_prod = short_specs[int(np.argmin(R_maxprod[:, k]))]

        print(f"  • {cand:<10}: Рекомендована професія: {best_spec_prod} (оцінка {np.max(R_maxprod[:, k]):.2f}), "
              f"найменш придатна: {worst_spec_prod} ({np.min(R_maxprod[:, k]):.2f})")


# 6. Побудова графіків
def save_plots(R_maxmin, R_maxprod):
    """Створює та зберігає візуальні графіки результатів."""
    os.makedirs("plots", exist_ok=True)

    short_specs = ["Менеджер", "Програміст", "Водій", "Секретар"]
    short_cands = ["Андрієнко", "Василенко", "Іваненко", "Дмитренко", "Петренко", "Романенко"]

    # Графік 1: Теплова карта Max-Min
    plt.figure(figsize=(8, 5))
    plt.imshow(R_maxmin, cmap="YlGnBu", aspect="auto", vmin=0.3, vmax=1.0)
    plt.colorbar(label="Ступінь відповідності μ(x, z)")
    plt.xticks(range(len(short_cands)), short_cands, rotation=20)
    plt.yticks(range(len(short_specs)), short_specs)
    plt.title("Максмінна композиція нечітких відношень (Max-Min)", fontweight="bold")
    for i in range(len(short_specs)):
        for k in range(len(short_cands)):
            plt.text(k, i, f"{R_maxmin[i, k]:.2f}", ha="center", va="center",
                     color="white" if R_maxmin[i, k] > 0.75 else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/maxmin_composition.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/maxmin_composition.png")

    # Графік 2: Теплова карта Max-Product
    plt.figure(figsize=(8, 5))
    plt.imshow(R_maxprod, cmap="YlOrRd", aspect="auto", vmin=0.3, vmax=1.0)
    plt.colorbar(label="Ступінь відповідності μ(x, z)")
    plt.xticks(range(len(short_cands)), short_cands, rotation=20)
    plt.yticks(range(len(short_specs)), short_specs)
    plt.title("Макс-добуткова композиція нечітких відношень (Max-Product)", fontweight="bold")
    for i in range(len(short_specs)):
        for k in range(len(short_cands)):
            plt.text(k, i, f"{R_maxprod[i, k]:.2f}", ha="center", va="center",
                     color="white" if R_maxprod[i, k] > 0.7 else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/maxprod_composition.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/maxprod_composition.png")

    # Графік 3: Порівняльна стовпчикова діаграма для обох методів
    x = np.arange(len(short_cands))
    width = 0.2
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Підграфік Max-Min
    for i, spec in enumerate(short_specs):
        axes[0].bar(x + (i - 1.5) * width, R_maxmin[i, :], width=width, label=spec)
    axes[0].set_title("Максмінна композиція (Max-Min)", fontweight="bold")
    axes[0].set_ylabel("Ступінь відповідності μ")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(axis="y", linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Підграфік Max-Prod
    for i, spec in enumerate(short_specs):
        axes[1].bar(x + (i - 1.5) * width, R_maxprod[i, :], width=width, label=spec)
    axes[1].set_title("Макс-добуткова композиція (Max-Product)", fontweight="bold")
    axes[1].set_ylabel("Ступінь відповідності μ")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis="y", linestyle="--", alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_cands)

    plt.tight_layout()
    plt.savefig("plots/composition_comparison.png", dpi=200)
    plt.close()
    print("-> Збережено графік: plots/composition_comparison.png")


# ==============================================================================
# Головний запуск програми
# ==============================================================================
def main():
    print("=" * 75)
    print("Лабораторна робота №4: Композиція нечітких бінарних відношень")
    print("=" * 75)

    # 1. Завантаження матриць
    MS, MQ = get_expert_matrices()

    short_specs = ["Менеджер", "Програміст", "Водій", "Секретар"]
    short_chars = ["y1", "y2", "y3", "y4", "y5", "y6", "y7"]
    short_cands = ["Андрієнко", "Василенко", "Іваненко", "Дмитренко", "Петренко", "Романенко"]

    # 2. Виведення початкових даних
    print_matrix("Матриця профілювання спеціальностей MS (Вимоги посади)", MS, short_specs, short_chars)
    print_matrix("Матриця профілювання претендентів MQ (Оцінки кандидатів)", MQ, short_chars, short_cands)

    # 3. Обчислення композицій
    R_maxmin = max_min_composition(MS, MQ)
    R_maxprod = max_prod_composition(MS, MQ)

    # 4. Виведення результуючих матриць
    print_matrix("Результуюча матриця R1 = S o Q (Max-Min композиція)", R_maxmin, short_specs, short_cands)
    print_matrix("Результуюча матриця R2 = S * Q (Max-Product композиція)", R_maxprod, short_specs, short_cands)

    # 5. Аналіз переваг і недоліків
    analyze_results(MS, MQ, R_maxmin, R_maxprod)

    # 6. Збереження графіків
    save_plots(R_maxmin, R_maxprod)

    print("\n" + "=" * 75)
    print("Усі розрахунки та графіки успішно згенеровано.")
    print("=" * 75)


if __name__ == "__main__":
    main()
