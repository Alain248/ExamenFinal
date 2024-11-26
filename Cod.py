import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def discretizar(state, env, num_bins=20):
    """
    Discretiza un estado continuo en un índice discreto para la tabla Q.
    """
    bins = (env.observation_space.high - env.observation_space.low) / num_bins
    discretized = (state - env.observation_space.low) // bins
    return tuple(discretized.astype(np.int32))


# Configuración del entorno
env = gym.make('MountainCar-v0')

# Parámetros de entrenamiento
num_bins = 20               # Número de divisiones por dimensión para discretización
Q = np.random.uniform(low=-1, high=1, size=(num_bins, num_bins, env.action_space.n))
episodes = 5000             # Número total de episodios de entrenamiento
alpha = 0.1                 # Tasa de aprendizaje
gamma = 0.99                # Factor de descuento
epsilon = 1.0               # Valor inicial de epsilon para exploración
epsilon_decay = 0.0005      # Factor de decaimiento de epsilon
min_epsilon = 0.01          # Valor mínimo de epsilon

# Lista para almacenar las recompensas totales por episodio
rewards_per_episode = []

for episode in range(episodes):
    state = discretizar(env.reset()[0], env, num_bins)
    done = False
    truncated = False
    total_reward = 0

    while not (done or truncated):
        # Selección de acción con epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Exploración
        else:
            action = np.argmax(Q[state])        # Explotación

        # Ejecuta la acción y observa el nuevo estado y recompensa
        next_state_continuous, reward, done, truncated, _ = env.step(action)
        next_state = discretizar(next_state_continuous, env, num_bins)

        # Actualización de la ecuación de Bellman
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        # Acumula la recompensa total del episodio
        total_reward += reward
        state = next_state

    rewards_per_episode.append(total_reward)

    # Reducir epsilon gradualmente hasta un valor mínimo (epsilon decay)
    epsilon = max(min_epsilon, epsilon - epsilon_decay)

    # Imprime el progreso cada 100 episodios
    if (episode + 1) % 100 == 0:
        print(f"Episodio {episode + 1}: Recompensa total = {total_reward}, Epsilon = {epsilon:.4f}")

env.close()

# Impresión de la tabla Q
print(f"Tabla Q: {Q}")

# Gráfica de recompensas
plt.plot(rewards_per_episode)
plt.xlabel('Episodios')
plt.ylabel('Recompensa Total')
plt.title('Recompensas por Episodio')
plt.show()

# Demostración del agente entrenado
env = gym.make('MountainCar-v0', render_mode='human')
state = discretizar(env.reset()[0], env, num_bins)
done = False
truncated = False
total_reward = 0

while not (done or truncated):
    action = np.argmax(Q[state])  # Política óptima aprendida
    next_state_continuous, reward, done, truncated, _ = env.step(action)
    state = discretizar(next_state_continuous, env, num_bins)
    total_reward += reward
    env.render()

print(f"Recompensa total del agente entrenado: {total_reward}")
env.close()


