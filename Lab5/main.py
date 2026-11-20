import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Лабораторна робота №5
# Аналіз ефективності комунікації на основі транзитивного замикання
# нечітких бінарних відношень
# ==============================================================================

plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "DejaVu Sans"


# 1. Вхідні дані — дві групи студентів
# Матриці Q мають бути рефлексивними (діагональ = 1) та симетричними

NAMES_1 = ["Бондаренко", "Коваленко", "Мельник", "Шевченко", "Ткаченко", "Кравченко"]
NAMES_2 = ["Лисенко", "Мороз", "Савченко", "Павленко", "Гончаренко", "Руденко"]


def get_input_data():
    """
    Повертає дві матриці нечітких відношень Q1 та Q2.
    Q1 — одногрупники (висока зв'язність).
    Q2 — студенти з різних факультетів (низька зв'язність).
    """
    # Група 1: одногрупники, добре знають один одного
    Q1 = np.array([
        [1.0, 0.8, 0.6, 0.7, 0.5, 0.3],
        [0.8, 1.0, 0.9, 0.5, 0.6, 0.4],
        [0.6, 0.9, 1.0, 0.4, 0.7, 0.5],
        [0.7, 0.5, 0.4, 1.0, 0.8, 0.6],
        [0.5, 0.6, 0.7, 0.8, 1.0, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.9, 1.0],
    ])

    # Група 2: студенти з різних факультетів, слабкі зв'язки
    Q2 = np.array([
        [1.0, 0.3, 0.1, 0.5, 0.0, 0.2],
        [0.3, 1.0, 0.4, 0.1, 0.2, 0.0],
        [0.1, 0.4, 1.0, 0.2, 0.6, 0.1],
        [0.5, 0.1, 0.2, 1.0, 0.1, 0.3],
        [0.0, 0.2, 0.6, 0.1, 1.0, 0.4],
        [0.2, 0.0, 0.1, 0.3, 0.4, 1.0],
    ])

    return Q1, Q2


# 2. Максмінна композиція двох матриць
def max_min_composition(A, B):
    """
    Обчислює max-min композицію двох матриць A і B.
    R[i,j] = max_k ( min(A[i,k], B[k,j]) )
    """
    n = A.shape[0]
    m = B.shape[1]
    R = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # Беремо мінімум між кожним елементом рядка A[i] і стовпця B[:,j]
            min_vals = np.minimum(A[i, :], B[:, j])
            # Шукаємо максимум серед цих мінімумів
            R[i, j] = np.max(min_vals)

    return R


# 3. Транзитивне замикання нечіткого відношення
def transitive_closure(Q):
    """
    Обчислює транзитивне замикання нечіткого відношення Q.

    Алгоритм:
    1. Починаємо з Q_power = Q (це Q^1)
    2. На кожному кроці обчислюємо наступну степінь: Q_power = Q_power ∘ Q
    3. Оновлюємо замикання: closure = max(closure, Q_power)
    4. Якщо замикання не змінилось — зупиняємось
    """
    n = Q.shape[0]
    closure = Q.copy()     # Починаємо із самого відношення Q
    Q_power = Q.copy()     # Поточна степінь Q^k

    for step in range(1, n):
        # Обчислюємо наступну степінь: Q^(k+1) = Q^k ∘ Q
        Q_power = max_min_composition(Q_power, Q)

        # Об'єднуємо (поелементний максимум)
        new_closure = np.maximum(closure, Q_power)

        # Перевіряємо, чи змінилось замикання
        if np.allclose(new_closure, closure):
            print(f"  Замикання стабілізувалось на кроці {step}")
            break
        closure = new_closure
    else:
        print(f"  Замикання завершено після {n - 1} кроків")

    return closure


# 4. Форматоване виведення матриці
def print_matrix(title, matrix, labels):
    """Виводить матрицю з підписами рядків та стовпців."""
    print(f"\n{title}:")
    header = f"{'':>12}" + "".join([f"{name:>12}" for name in labels])
    print(header)
    print("-" * len(header))
    for i, name in enumerate(labels):
        row = f"{name:>12}"
        for j in range(len(labels)):
            row += f"{matrix[i, j]:>12.2f}"
        print(row)


# 5. Збереження теплових карт
def save_heatmap(matrix, labels, title, filename):
    """Зберігає теплову карту матриці у файл."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="equal")
    plt.colorbar(im, ax=ax, label="Ступінь знайомства μ")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, fontweight="bold", pad=12)

    # Підписи значень у комірках
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if matrix[i, j] > 0.65 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    color=color, fontweight="bold", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"-> Збережено: {filename}")


def save_comparison(Q1, T1, Q2, T2, names1, names2):
    """Зберігає порівняльну діаграму середніх ступенів зв'язності."""
    # Обчислюємо середній ступінь зв'язності (без діагоналі)
    def avg_connectivity(matrix):
        n = matrix.shape[0]
        mask = ~np.eye(n, dtype=bool)
        return np.mean(matrix[mask])

    categories = ["Група 1\n(вихідна)", "Група 1\n(замикання)",
                   "Група 2\n(вихідна)", "Група 2\n(замикання)"]
    values = [avg_connectivity(Q1), avg_connectivity(T1),
              avg_connectivity(Q2), avg_connectivity(T2)]
    colors = ["#4a90d9", "#1a5276", "#e67e22", "#a04000"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(categories, values, color=colors, edgecolor="black", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold")

    ax.set_ylabel("Середній ступінь зв'язності μ")
    ax.set_title("Порівняння ефективності комунікації двох груп", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/comparison.png", dpi=200)
    plt.close()
    print("-> Збережено: plots/comparison.png")


# 6. Аналіз результатів
def analyze(Q, T, names, group_name):
    """Аналізує різницю між вихідним відношенням та його транзитивним замиканням."""
    print(f"\n--- Аналіз: {group_name} ---")
    n = Q.shape[0]
    diff = T - Q

    # Знаходимо пари, де зв'язність найбільше зросла
    improved = []
    for i in range(n):
        for j in range(i + 1, n):
            if diff[i, j] > 0.01:
                improved.append((names[i], names[j], Q[i, j], T[i, j], diff[i, j]))

    if improved:
        improved.sort(key=lambda x: -x[4])
        print("  Пари з найбільшим зростанням зв'язності (через посередників):")
        for name_i, name_j, old, new, delta in improved[:5]:
            print(f"    {name_i} <-> {name_j}: {old:.2f} -> {new:.2f} (+{delta:.2f})")
    else:
        print("  Відношення вже транзитивне, зростання немає.")

    # Середня зв'язність
    mask = ~np.eye(n, dtype=bool)
    avg_orig = np.mean(Q[mask])
    avg_clos = np.mean(T[mask])
    print(f"  Середня зв'язність (без діагоналі): {avg_orig:.3f} -> {avg_clos:.3f}")


# ==============================================================================
# Головний запуск програми
# ==============================================================================
def main():
    print("=" * 70)
    print("Лабораторна робота №5")
    print("Транзитивне замикання нечітких бінарних відношень")
    print("=" * 70)

    os.makedirs("plots", exist_ok=True)

    # Завантаження даних
    Q1, Q2 = get_input_data()

    # === ГРУПА 1 ===
    print("\n" + "=" * 70)
    print("ГРУПА 1: Одногрупники (висока зв'язність)")
    print("=" * 70)
    print_matrix("Вихідна матриця Q1", Q1, NAMES_1)

    print("\nОбчислення транзитивного замикання Q1...")
    T1 = transitive_closure(Q1)
    print_matrix("Транзитивне замикання Q1^", T1, NAMES_1)

    save_heatmap(Q1, NAMES_1, "Група 1 — Вихідне відношення Q",
                 "plots/group1_original.png")
    save_heatmap(T1, NAMES_1, "Група 1 — Транзитивне замикання Q̂",
                 "plots/group1_closure.png")

    # === ГРУПА 2 ===
    print("\n" + "=" * 70)
    print("ГРУПА 2: Студенти з різних факультетів (низька зв'язність)")
    print("=" * 70)
    print_matrix("Вихідна матриця Q2", Q2, NAMES_2)

    print("\nОбчислення транзитивного замикання Q2...")
    T2 = transitive_closure(Q2)
    print_matrix("Транзитивне замикання Q2^", T2, NAMES_2)

    save_heatmap(Q2, NAMES_2, "Група 2 — Вихідне відношення Q",
                 "plots/group2_original.png")
    save_heatmap(T2, NAMES_2, "Група 2 — Транзитивне замикання Q̂",
                 "plots/group2_closure.png")

    # === ПОРІВНЯННЯ ===
    print("\n" + "=" * 70)
    print("ПОРІВНЯЛЬНИЙ АНАЛІЗ")
    print("=" * 70)
    analyze(Q1, T1, NAMES_1, "Група 1 (одногрупники)")
    analyze(Q2, T2, NAMES_2, "Група 2 (різні факультети)")

    save_comparison(Q1, T1, Q2, T2, NAMES_1, NAMES_2)

    print("\n" + "=" * 70)
    print("Усі розрахунки та графіки успішно згенеровано.")
    print("=" * 70)


if __name__ == "__main__":
    main()
