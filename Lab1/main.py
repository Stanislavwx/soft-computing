import os
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Лабораторна робота №1: Реалізація простої багатошарової нейромережі
# ==============================================================================

def sigmoid(z):
    """Активаційна функція сигмоїд"""
    # Обмеження z для уникнення overflow при np.exp
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))

def sigmoid_derivative(a):
    """Похідна сигмоїди через її вихід a = sigmoid(z)"""
    return a * (1.0 - a)

class NeuralNetwork:
    """
    Проста двозв'язна нейромережа з одним прихованим шаром для регресії.
    """
    def __init__(self, input_size=6, hidden_size=8, output_size=1, lr=0.5, seed=42):
        # Крок 2: Ініціалізація параметрів
        np.random.seed(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        # Ваги та зсуви ініціалізуються невеликими випадковими значеннями
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def forward(self, X):
        """Крок 3: Пряме розповсюдження (Forward Propagation)"""
        # Лінійна комбінація та активація прихованого шару
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.Z1)

        # Лінійна комбінація та активація вихідного шару
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2

    def compute_loss(self, Y, A2):
        """Крок 4: Обчислення втрат (MSE Loss)"""
        m = Y.shape[0]
        loss = (1.0 / m) * np.sum((Y - A2) ** 2)
        return loss

    def backward(self, X, Y):
        """Крок 5: Зворотне розповсюдження та оновлення ваг (Backward Propagation)"""
        m = X.shape[0]

        # Похідні для вихідного шару
        dZ2 = self.A2 - Y
        dW2 = (1.0 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1.0 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Похідні для прихованого шару
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * sigmoid_derivative(self.A1)
        dW1 = (1.0 / m) * np.dot(X.T, dZ1)
        db1 = (1.0 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # Оновлення ваг та зсувів за допомогою градієнтного спуску
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, Y, epochs=2000):
        """Крок 6: Тренування нейромережі"""
        loss_history = []
        for epoch in range(epochs):
            A2 = self.forward(X)
            loss = self.compute_loss(Y, A2)
            loss_history.append(loss)
            self.backward(X, Y)
        return loss_history

    def predict(self, X):
        """Крок 7: Тестування та отримання передбачень"""
        return self.forward(X)


# ==============================================================================
# Завантаження та попередня обробка даних
# ==============================================================================

def load_and_preprocess_data(filepath="Real estate.csv"):
    # Завантаження CSV файлу з пропуском заголовку
    data = np.genfromtxt(filepath, delimiter=",", skip_header=1)

    # Ознаки: X1 (transaction date) ... X6 (longitude)
    X = data[:, 1:7]
    # Цільова змінна: Y house price of unit area
    Y = data[:, 7:8]

    # Нормалізація (Min-Max Scaling) до діапазону [0, 1] для коректної роботи сигмоїди
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    Y_min = np.min(Y, axis=0)
    Y_max = np.max(Y, axis=0)

    X_norm = (X - X_min) / (X_max - X_min)
    Y_norm = (Y - Y_min) / (Y_max - Y_min)

    # Розбиття на тренувальну (80%) та тестову (20%) вибірки
    np.random.seed(42)
    indices = np.random.permutation(len(X_norm))
    train_size = int(0.8 * len(X_norm))

    train_idx = indices[:train_size]
    test_idx = indices[train_size:]

    X_train, Y_train = X_norm[train_idx], Y_norm[train_idx]
    X_test, Y_test = X_norm[test_idx], Y_norm[test_idx]
    Y_test_real = Y[test_idx]

    return (X_train, Y_train, X_test, Y_test, Y_test_real, Y_min, Y_max)


# ==============================================================================
# Основний блок виконання та аналізу
# ==============================================================================

def main():
    print("=== Завантаження та підготовка даних ===")
    X_train, Y_train, X_test, Y_test, Y_test_real, Y_min, Y_max = load_and_preprocess_data()
    print(f"Розмір навчальної вибірки: {X_train.shape[0]} зразків")
    print(f"Розмір тестової вибірки:    {X_test.shape[0]} зразків")
    print(f"Кількість ознак:            {X_train.shape[1]}")

    # Створення папки для графіків
    os.makedirs("plots", exist_ok=True)

    # 1. Базове навчання моделі (hidden_size=8, epochs=2500)
    print("\n=== Навчання базової моделі нейромережі ===")
    base_hidden = 8
    base_epochs = 2500
    nn = NeuralNetwork(input_size=6, hidden_size=base_hidden, output_size=1, lr=0.5, seed=42)
    loss_history = nn.train(X_train, Y_train, epochs=base_epochs)

    # Тестування базової моделі
    preds_norm = nn.predict(X_test)
    preds_real = preds_norm * (Y_max - Y_min) + Y_min

    mse = np.mean((Y_test_real - preds_real) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(Y_test_real - preds_real))
    ss_tot = np.sum((Y_test_real - np.mean(Y_test_real)) ** 2)
    r2 = 1 - np.sum((Y_test_real - preds_real) ** 2) / ss_tot

    print(f"Фінальні втрати на тренуванні (MSE norm): {loss_history[-1]:.6f}")
    print(f"Метрики на тестовій вибірці:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")

    # Побудова графіка кривої навчання
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, color="#1f77b4", linewidth=2, label="Функція втрат (MSE)")
    plt.title("Крива навчання нейромережі (Loss vs Epochs)", fontsize=13, fontweight="bold")
    plt.xlabel("Епохи", fontsize=11)
    plt.ylabel("Втрати (MSE на норм. даних)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/learning_curve.png", dpi=300)
    plt.close()
    print("-> Збережено графік: plots/learning_curve.png")

    # 2. Дослідження впливу кількості нейронів у прихованому шарі
    print("\n=== Експеримент 1: Вплив кількості нейронів у прихованому шарі ===")
    neuron_counts = [2, 4, 8, 12, 16, 24, 32]
    neuron_mse = []
    neuron_r2 = []

    print(f"{'Нейрони':<10} | {'Test MSE':<10} | {'Test MAE':<10} | {'Test R²':<10}")
    print("-" * 46)
    for n in neuron_counts:
        model = NeuralNetwork(input_size=6, hidden_size=n, output_size=1, lr=0.5, seed=42)
        model.train(X_train, Y_train, epochs=2000)
        p_norm = model.predict(X_test)
        p_real = p_norm * (Y_max - Y_min) + Y_min
        cur_mse = np.mean((Y_test_real - p_real) ** 2)
        cur_mae = np.mean(np.abs(Y_test_real - p_real))
        cur_r2 = 1 - np.sum((Y_test_real - p_real) ** 2) / ss_tot
        neuron_mse.append(cur_mse)
        neuron_r2.append(cur_r2)
        print(f"{n:<10} | {cur_mse:<10.2f} | {cur_mae:<10.2f} | {cur_r2:<10.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(neuron_counts, neuron_mse, marker="o", color="#d62728", linewidth=2, label="Test MSE")
    plt.title("Залежність помилки від кількості нейронів у прихованому шарі", fontsize=13, fontweight="bold")
    plt.xlabel("Кількість нейронів у прихованому шарі", fontsize=11)
    plt.ylabel("Середньоквадратична помилка (MSE)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/neurons_comparison.png", dpi=300)
    plt.close()
    print("-> Збережено графік: plots/neurons_comparison.png")

    # 3. Дослідження впливу кількості епох навчання
    print("\n=== Експеримент 2: Вплив кількості епох навчання ===")
    epoch_counts = [200, 500, 1000, 2000, 3000, 5000]
    epoch_mse = []
    epoch_r2 = []

    print(f"{'Епохи':<10} | {'Test MSE':<10} | {'Test MAE':<10} | {'Test R²':<10}")
    print("-" * 46)
    for ep in epoch_counts:
        model = NeuralNetwork(input_size=6, hidden_size=8, output_size=1, lr=0.5, seed=42)
        model.train(X_train, Y_train, epochs=ep)
        p_norm = model.predict(X_test)
        p_real = p_norm * (Y_max - Y_min) + Y_min
        cur_mse = np.mean((Y_test_real - p_real) ** 2)
        cur_mae = np.mean(np.abs(Y_test_real - p_real))
        cur_r2 = 1 - np.sum((Y_test_real - p_real) ** 2) / ss_tot
        epoch_mse.append(cur_mse)
        epoch_r2.append(cur_r2)
        print(f"{ep:<10} | {cur_mse:<10.2f} | {cur_mae:<10.2f} | {cur_r2:<10.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_counts, epoch_mse, marker="s", color="#2ca02c", linewidth=2, label="Test MSE")
    plt.title("Залежність помилки від кількості епох навчання", fontsize=13, fontweight="bold")
    plt.xlabel("Кількість епох", fontsize=11)
    plt.ylabel("Середньоквадратична помилка (MSE)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/epochs_comparison.png", dpi=300)
    plt.close()
    print("-> Збережено графік: plots/epochs_comparison.png")

    # 4. Порівняння реальних значень та прогнозів (перші 10 зразків)
    print("\n=== Приклади передбачень на тестових даних (перші 10) ===")
    print(f"{'№':<4} | {'Фактична ціна':<15} | {'Прогнозована ціна':<18} | {'Абс. помилка':<12}")
    print("-" * 55)
    for i in range(10):
        actual = Y_test_real[i, 0]
        predicted = preds_real[i, 0]
        diff = abs(actual - predicted)
        print(f"{i+1:<4} | {actual:<15.2f} | {predicted:<18.2f} | {diff:<12.2f}")

    # Графік реальні vs прогнозовані
    plt.figure(figsize=(8, 5))
    plt.scatter(Y_test_real, preds_real, color="#9467bd", alpha=0.7, edgecolors="k", label="Зразки тесту")
    min_val = min(Y_test_real.min(), preds_real.min())
    max_val = max(Y_test_real.max(), preds_real.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ідеальний прогноз (Y = Y_pred)")
    plt.title("Фактичні та прогнозовані значення ціни нерухомості", fontsize=13, fontweight="bold")
    plt.xlabel("Фактична ціна (Y)", fontsize=11)
    plt.ylabel("Прогнозована ціна (Y_pred)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig("plots/predictions_plot.png", dpi=300)
    plt.close()
    print("-> Збережено графік: plots/predictions_plot.png")

    print("\nУсі розрахунки та графіки успішно згенеровано.")

if __name__ == "__main__":
    main()
