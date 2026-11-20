import os
import numpy as np
import matplotlib.pyplot as plt

# Налаштування стилю для графіків
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.4
plt.rcParams["grid.linestyle"] = "--"


# ------------------------------------------------------------------------------
# 1. Базові функції належності (трикутна та трапецієподібні плечі)
# ------------------------------------------------------------------------------
def trimf(x, a, b, c):
    """
    Трикутна функція належності:
    - 0 для x <= a та x >= c;
    - лінійне зростання від a до b (досягає 1.0 в точці b);
    - лінійне спадання від b до c.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    # Висхідна гілка
    if b > a:
        idx1 = (x > a) & (x <= b)
        y[idx1] = (x[idx1] - a) / (b - a)
    elif b == a:
        y[x == b] = 1.0

    # Спадна гілка
    if c > b:
        idx2 = (x > b) & (x < c)
        y[idx2] = (c - x[idx2]) / (c - b)
    elif c == b:
        y[x == b] = 1.0

    return y


def trapmf_left(x, a, b):
    """
    Ліва плечова трапецієподібна функція:
    - 1.0 для x <= a;
    - лінійний спуск від a до b;
    - 0.0 для x >= b.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    y[x <= a] = 1.0
    idx = (x > a) & (x < b)
    y[idx] = (b - x[idx]) / (b - a)
    return y


def trapmf_right(x, a, b):
    """
    Права плечова трапецієподібна функція:
    - 0.0 для x <= a;
    - лінійний підйом від a до b;
    - 1.0 для x >= b.
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    idx = (x > a) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    y[x >= b] = 1.0
    return y


# ------------------------------------------------------------------------------
# 2. Опис лінгвістичних змінних предметної області
# ------------------------------------------------------------------------------
# Вхід: "Температура води", t ∈ [0, 100] °C
def mu_cold(t):
    """Терм: холодна (0..30 °C, плато до 10 °C)"""
    return trapmf_left(t, 10.0, 30.0)


def mu_cool(t):
    """Терм: прохолодна (20..50 °C, максимум в 35 °C)"""
    return trimf(t, 20.0, 35.0, 50.0)


def mu_warm(t):
    """Терм: тепла (35..60 °C, максимум в 50 °C)"""
    return trimf(t, 35.0, 50.0, 60.0)


def mu_not_very_hot(t):
    """Терм: не дуже гаряча (50..70 °C, максимум в 60 °C)"""
    return trimf(t, 50.0, 60.0, 70.0)


def mu_hot(t):
    """Терм: гаряча (60..100 °C, плато від 70 °C)"""
    return trapmf_right(t, 60.0, 70.0)


# Вихід: "Кут повороту крана", φ ∈ [-90, +90] градусів
def mu_big_left(phi):
    """Терм: великий кут вліво (плато до -75°, спад до -45°)"""
    return trapmf_left(phi, -75.0, -45.0)


def mu_small_left(phi):
    """Терм: невеликий кут вліво (-60°..0°, пік в -30°)"""
    return trimf(phi, -60.0, -30.0, 0.0)


def mu_zero(phi):
    """Терм: кут не змінювати (-30°..+30°, пік в 0°)"""
    return trimf(phi, -30.0, 0.0, 30.0)


def mu_small_right(phi):
    """Терм: невеликий кут вправо (0°..+60°, пік в +30°)"""
    return trimf(phi, 0.0, 30.0, 60.0)


def mu_big_right(phi):
    """Терм: великий кут вправо (підйом 45°..75°, плато від 75°)"""
    return trapmf_right(phi, 45.0, 75.0)


# ------------------------------------------------------------------------------
# 3. База продукційних правил та логічний висновок
# ------------------------------------------------------------------------------
# R1: ЯКЩО вода гаряча          ТО повернути вентиль на великий кут вправо
# R2: ЯКЩО вода не дуже гаряча  ТО повернути вентиль на невеликий кут вправо
# R3: ЯКЩО вода тепла           ТО кут повороту вентиля не змінювати
# R4: ЯКЩО вода прохолодна      ТО повернути вентиль на невеликий кут вліво
# R5: ЯКЩО вода холодна         ТО повернути вентиль на великий кут вліво

def fuzzify_temperature(t):
    """
    Фазифікація вхідного значення температури t:
    Обчислює ступені істинності умов для правил R1..R5:
    c_i = mu_i^in(t),  i = 1, 2, 3, 4, 5
    """
    c1 = float(mu_hot(t))
    c2 = float(mu_not_very_hot(t))
    c3 = float(mu_warm(t))
    c4 = float(mu_cool(t))
    c5 = float(mu_cold(t))
    return [c1, c2, c3, c4, c5]


def fuzzy_inference(c_vals, phi_grid, method="mamdani"):
    """
    Отримання нечіткого висновку:
    - Активізація консеквентів правил:
      * Метод Мамдані (min): mu'_i(phi) = min(c_i, mu_i^out(phi))
      * Метод Ларсена (prod): mu'_i(phi) = c_i * mu_i^out(phi)
    - Акумуляція результатів (max-об'єднання):
      mu'_out(phi) = max_i ( mu'_i(phi) )
    """
    # Функції належності вихідних термів для відповідних правил
    out_mfs = [
        mu_big_right(phi_grid),  # R1 -> великий кут вправо
        mu_small_right(phi_grid),  # R2 -> невеликий кут вправо
        mu_zero(phi_grid),  # R3 -> кут не змінювати
        mu_small_left(phi_grid),  # R4 -> невеликий кут вліво
        mu_big_left(phi_grid)  # R5 -> великий кут вліво
    ]

    activated_rules = []
    for c, mf in zip(c_vals, out_mfs):
        if method == "mamdani":
            # Відсікання за мінімумом (clipping)
            rule_out = np.minimum(c, mf)
        elif method == "larsen":
            # Масштабування за добутком (scaling)
            rule_out = c * mf
        else:
            raise ValueError(f"Невідомий метод активізації: {method}")
        activated_rules.append(rule_out)

    # Акумуляція висновків (операція MAX)
    aggregated = np.maximum.reduce(activated_rules)
    return activated_rules, aggregated


def defuzzify_centroid(phi_grid, aggregated_mf):
    """
    Дефазифікація методом центру ваги (Center of Gravity):
    phi* = ( sum( phi_k * mu(phi_k) ) ) / ( sum( mu(phi_k) ) )
    """
    total_area = np.sum(aggregated_mf)
    if total_area == 0:
        return 0.0
    return float(np.sum(phi_grid * aggregated_mf) / total_area)


def calculate_control_angle(t, phi_grid, method="mamdani"):
    """
    Повний цикл роботи контролера для заданої температури:
    1. Фазифікація t -> c_vals
    2. Активізація та акумуляція -> aggregated_mf
    3. Дефазифікація -> чіткий кут phi*
    """
    c_vals = fuzzify_temperature(t)
    _, aggregated_mf = fuzzy_inference(c_vals, phi_grid, method=method)
    return defuzzify_centroid(phi_grid, aggregated_mf)


def simulate_controller(t_values, phi_grid, method="mamdani"):
    """Моделювання контролера для масиву температур"""
    return np.array([calculate_control_angle(t, phi_grid, method=method) for t in t_values])


# ------------------------------------------------------------------------------
# 4. Побудова та збереження графіків
# ------------------------------------------------------------------------------
def plot_input_membership_functions(output_dir="plots"):
    """Побудова графіків функцій належності вхідної змінної 'температура води'"""
    os.makedirs(output_dir, exist_ok=True)
    t = np.linspace(0, 100, 1000)

    plt.figure(figsize=(9, 4.5))
    plt.plot(t, mu_cold(t), label="Холодна (0..30 °C)", color="#1f77b4", linewidth=2)
    plt.plot(t, mu_cool(t), label="Прохолодна (20..50 °C)", color="#17becf", linewidth=2)
    plt.plot(t, mu_warm(t), label="Тепла (35..60 °C)", color="#2ca02c", linewidth=2)
    plt.plot(t, mu_not_very_hot(t), label="Не дуже гаряча (50..70 °C)", color="#ff7f0e", linewidth=2)
    plt.plot(t, mu_hot(t), label="Гаряча (60..100 °C)", color="#d62728", linewidth=2)

    plt.title("Функції належності вхідної лінгвістичної змінної «Температура води»", fontsize=11, fontweight="bold")
    plt.xlabel("Температура води t, °C", fontsize=10)
    plt.ylabel("Ступінь належності μ(t)", fontsize=10)
    plt.xlim(0, 100)
    plt.ylim(-0.02, 1.05)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "input_membership_functions.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Збережено графік: {filepath}")


def plot_output_membership_functions(output_dir="plots"):
    """Побудова графіків функцій належності вихідної змінної 'кут повороту крана'"""
    os.makedirs(output_dir, exist_ok=True)
    phi = np.linspace(-90, 90, 1000)

    plt.figure(figsize=(9, 4.5))
    plt.plot(phi, mu_big_left(phi), label="Великий кут вліво (L2)", color="#08519c", linewidth=2)
    plt.plot(phi, mu_small_left(phi), label="Невеликий кут вліво (L1)", color="#4292c6", linewidth=2)
    plt.plot(phi, mu_zero(phi), label="Кут не змінювати (Z)", color="#238b45", linewidth=2)
    plt.plot(phi, mu_small_right(phi), label="Невеликий кут вправо (R1)", color="#fe9929", linewidth=2)
    plt.plot(phi, mu_big_right(phi), label="Великий кут вправо (R2)", color="#cb181d", linewidth=2)

    plt.title("Функції належності вихідної лінгвістичної змінної «Кут повороту крана»", fontsize=11, fontweight="bold")
    plt.xlabel("Кут повороту крана φ, градуси", fontsize=10)
    plt.ylabel("Ступінь належності μ(φ)", fontsize=10)
    plt.xlim(-90, 90)
    plt.ylim(-0.02, 1.05)
    plt.xticks(np.arange(-90, 91, 30))
    plt.yticks(np.arange(0.0, 1.1, 0.2))
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "output_membership_functions.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Збережено графік: {filepath}")


def plot_control_curves(t_values, phi_mamdani, phi_larsen, output_dir="plots"):
    """Побудова порівняльної залежності кута повороту крана від температури для Mamdani та Larsen"""
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(t_values, phi_mamdani, label="Алгоритм Мамдані (min-активізація)", color="#1f77b4", linewidth=2.2)
    plt.plot(t_values, phi_larsen, label="Алгоритм Ларсена (prod-активізація)", color="#d62728", linewidth=2.0, linestyle="--")

    # Позначення характерних температур
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.axvline(50, color="green", linestyle=":", linewidth=1, label="Комфортна температура (50 °C)")

    plt.title("Залежність кута повороту крана від температури води φ(t)", fontsize=11, fontweight="bold")
    plt.xlabel("Температура води t, °C", fontsize=10)
    plt.ylabel("Кут повороту крана φ*, градуси", fontsize=10)
    plt.xlim(0, 100)
    plt.ylim(-90, 90)
    plt.xticks(np.arange(0, 101, 10))
    plt.yticks(np.arange(-90, 91, 15))
    plt.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    filepath = os.path.join(output_dir, "control_characteristic_comparison.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Збережено графік: {filepath}")


def plot_fuzzy_inference_example(t_example=55.0, output_dir="plots"):
    """
    Візуалізація покрокового нечіткого висновку для конкретної температури (наприклад, t = 55 °C)
    для порівняння активізацій Мамдані (min) та Ларсена (prod).
    """
    os.makedirs(output_dir, exist_ok=True)
    phi_grid = np.linspace(-90, 90, 1000)
    c_vals = fuzzify_temperature(t_example)

    # Отримуємо висновок для обох методів
    rules_mam, agg_mam = fuzzy_inference(c_vals, phi_grid, method="mamdani")
    rules_lar, agg_lar = fuzzy_inference(c_vals, phi_grid, method="larsen")

    cog_mam = defuzzify_centroid(phi_grid, agg_mam)
    cog_lar = defuzzify_centroid(phi_grid, agg_lar)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)

    # 1. Графік для Мамдані
    ax = axes[0]
    ax.plot(phi_grid, agg_mam, label="Акумульована множина (MAX)", color="black", linewidth=2)
    ax.fill_between(phi_grid, 0, agg_mam, color="#1f77b4", alpha=0.3)
    ax.axvline(cog_mam, color="#d62728", linestyle="--", linewidth=2, label=f"Центр ваги φ* = {cog_mam:.1f}°")
    ax.set_title(f"Мамдані (min): t = {t_example} °C\nc1={c_vals[0]:.2f}, c2={c_vals[1]:.2f}, c3={c_vals[2]:.2f}", fontsize=10)
    ax.set_xlabel("Кут повороту крана φ, градуси")
    ax.set_ylabel("Ступінь належності μ(φ)")
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)

    # 2. Графік для Ларсена
    ax = axes[1]
    ax.plot(phi_grid, agg_lar, label="Акумульована множина (MAX)", color="black", linewidth=2)
    ax.fill_between(phi_grid, 0, agg_lar, color="#d62728", alpha=0.3)
    ax.axvline(cog_lar, color="#1f77b4", linestyle="--", linewidth=2, label=f"Центр ваги φ* = {cog_lar:.1f}°")
    ax.set_title(f"Ларсен (prod): t = {t_example} °C\nc1={c_vals[0]:.2f}, c2={c_vals[1]:.2f}, c3={c_vals[2]:.2f}", fontsize=10)
    ax.set_xlabel("Кут повороту крана φ, градуси")
    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    filepath = os.path.join(output_dir, "fuzzy_inference_example.png")
    plt.savefig(filepath, dpi=300)
    plt.close()
    print(f"Збережено графік: {filepath}")


# ------------------------------------------------------------------------------
# 5. Головна функція виконання розрахунків та тестування
# ------------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Лабораторна робота №6: Проектування нечітких систем типу Мамдані")
    print("Виконав: Чепара Станіслав (група ФеП-32)")
    print("=" * 70)

    # Дискретизація простору вихідної змінної (кут повороту крана від -90 до +90)
    phi_grid = np.linspace(-90, 90, 1001)

    # Контрольні точки температури для перевірки логіки
    test_temps = [0.0, 15.0, 25.0, 35.0, 42.0, 50.0, 55.0, 60.0, 65.0, 80.0, 100.0]

    print("\nТаблиця розрахунку для контрольних значень температури:")
    print("-" * 75)
    print(f"{'t, °C':^8} | {'Активні правила (с1..с5)':^26} | {'Mamdani φ*, °':^16} | {'Larsen φ*, °':^16}")
    print("-" * 75)

    for t in test_temps:
        c = fuzzify_temperature(t)
        phi_m = calculate_control_angle(t, phi_grid, method="mamdani")
        phi_l = calculate_control_angle(t, phi_grid, method="larsen")

        # Формуємо короткий рядок активованих правил
        active = [f"R{i+1}={c[i]:.2f}" for i in range(5) if c[i] > 0]
        active_str = ", ".join(active) if active else "Немає"

        print(f"{t:^8.1f} | {active_str:<26} | {phi_m:^16.2f} | {phi_l:^16.2f}")
    print("-" * 75)

    # Моделювання характеристики у повному діапазоні температур від 0 до 100 °C
    print("\nПроведення симуляції для діапазону температур 0..100 °C...")
    t_range = np.linspace(0, 100, 201)
    phi_mamdani = simulate_controller(t_range, phi_grid, method="mamdani")
    phi_larsen = simulate_controller(t_range, phi_grid, method="larsen")

    # Генерація та збереження графіків
    print("\nГенерація графічних результатів...")
    plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
    plot_input_membership_functions(plots_dir)
    plot_output_membership_functions(plots_dir)
    plot_control_curves(t_range, phi_mamdani, phi_larsen, plots_dir)
    plot_fuzzy_inference_example(t_example=55.0, output_dir=plots_dir)

    print("\nРозрахунки успішно завершено! Усі графіки збережено в папці 'plots/'.")


if __name__ == "__main__":
    main()
