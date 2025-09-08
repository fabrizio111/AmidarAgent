import numpy as np
import gymnasium as gym
import gymnasium
import ale_py
print("Gymnasium version:", gymnasium.__version__)
print("ALE-Py version:", ale_py.__version__)
from gymnasium.wrappers import AtariPreprocessing
from collections import defaultdict
import os
import time

# -------------------------------
# Utilidades de discretización
# -------------------------------
def downsample_and_binarize(obs84, stride=6, threshold=128):
    """
    obs84: np.ndarray uint8 de forma (84,84) (grayscale).
    stride: muestreo cada 'stride' píxeles -> (84//stride, 84//stride).
    threshold: binariza > threshold -> 1, si no -> 0.
    Devuelve un vector binario 1D y también bytes para usar como clave hash.
    """
    # Submuestreo por indexado (sin dependencias externas)
    small = obs84[::stride, ::stride]
    # Binariza
    bin_small = (small > threshold).astype(np.uint8)
    # Flatten y a bytes como clave compacta
    key_bytes = bin_small.tobytes()
    return bin_small, key_bytes

# -------------------------------
# Tabla Q basada en diccionario
# -------------------------------
class QTableDict:
    def __init__(self, n_actions, init_q=0.0):
        self.nA = n_actions
        self.init_q = init_q
        self.table = {}  # key: bytes -> np.ndarray(nA)

    def get(self, s_key):
        row = self.table.get(s_key)
        if row is None:
            row = np.full(self.nA, self.init_q, dtype=np.float32)
            self.table[s_key] = row
        return row

    def best_action_value(self, s_key):
        q = self.get(s_key)
        a = int(np.argmax(q))
        v = float(q[a])
        return a, v

    def update(self, s_key, a, target, alpha):
        q = self.get(s_key)
        q[a] += alpha * (target - q[a])

    def size(self):
        return len(self.table)

# -------------------------------
# Política epsilon-greedy
# -------------------------------
def select_action(qtab, s_key, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    a, _ = qtab.best_action_value(s_key)
    return a

# -------------------------------
# Entrenamiento
# -------------------------------
def train_amidar_qlearning(
    episodes=300,
    max_steps=20000,           # tope por episodio (con frame_skip=4 esto cubre bastante)
    gamma=0.99,
    alpha=0.1,
    eps_start=1.0,
    eps_end=0.05,
    eps_decay=0.995,           # decaimiento por episodio
    stride=6,                  # para 84x84 -> 14x14
    threshold=128,
    init_q=0.0,
    seed=42,
    render=False,
    save_path=None
):
    """
    Entrena Q-learning tabular con discretización burda de la imagen.
    Devuelve la tabla Q y el historial de recompensas.
    """
    rng = np.random.default_rng(seed)

    # Crear entorno Atari con preprocesamiento clásico (grayscale, frame-skip, vida perdida -> done opcional)
    base_env = gym.make("ALE/Amidar-v5", frameskip=1, render_mode="human")
    # AtariPreprocessing aplica frame_skip interno, grayscale y resize a 84x84
    env = AtariPreprocessing(
        base_env,
        frame_skip=4,
        grayscale_obs=True,
        scale_obs=False,       # mantiene uint8 0..255
        terminal_on_life_loss=False,
        screen_size=84
    )
    # Opcional: stack de 1 frame para mantener shape consistente (1,84,84) -> sacamos la dimensión al discretizar
    #env = FrameStack(env, num_stack=1)
    # No stack, ya tienes (84,84)

    n_actions = env.action_space.n
    qtab = QTableDict(n_actions=n_actions, init_q=init_q)

    eps = eps_start
    rewards_history = []

    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000)))
        # obs shape típicamente (1,84,84) con FrameStack; extraemos el frame 0
        frame = np.array(obs)  # (84,84)
        _, s_key = downsample_and_binarize(frame, stride=stride, threshold=threshold)

        total_reward = 0.0
        steps = 0

        while True:
            if render:
                # Render solo si se pidió, puede ser lento
                env.render()

            a = select_action(qtab, s_key, eps, n_actions, rng)
            next_obs, reward, terminated, truncated, info = env.step(int(a))

            total_reward += float(reward)
            steps += 1

            # Preparar siguiente estado
            next_frame = np.array(next_obs)
            # Si la observación tiene shape (84,84), usarla directamente
            if next_frame.shape == (84, 84):
                pass
            # Si tiene shape (1,84,84), quitar la primera dimensión
            elif next_frame.shape == (1, 84, 84):
                next_frame = next_frame[0]
            # Si es 1D de tamaño 84*84, reacomodar
            elif next_frame.ndim == 1 and next_frame.size == 84*84:
                next_frame = next_frame.reshape(84, 84)
            else:
                raise ValueError(f"Formato inesperado de next_frame: shape={next_frame.shape}")
            _, s_key_next = downsample_and_binarize(next_frame, stride=stride, threshold=threshold)

            # Q-learning target
            _, max_next = qtab.best_action_value(s_key_next)
            target = reward + (0.0 if (terminated or truncated) else gamma * max_next)

            # Actualización
            qtab.update(s_key, a, target, alpha)

            s_key = s_key_next

            if terminated or truncated or steps >= max_steps:
                break

        rewards_history.append(total_reward)

        # Decaimiento epsilon por episodio (clamp)
        eps = max(eps_end, eps * eps_decay)

        if ep % 10 == 0 or ep == 1:
            print(f"[Ep {ep:4d}] Recompensa: {total_reward:8.2f} | "
                  f"epsilon: {eps:6.3f} | Q-keys: {qtab.size():7d} | pasos: {steps}")

        # Guardado opcional cada cierto tiempo
        if save_path and (ep % 50 == 0):
            save_qtable(qtab, save_path)

    env.close()

    # Guardado final
    if save_path:
        save_qtable(qtab, save_path)

    return qtab, rewards_history


# -------------------------------
# Persistencia simple de Q-table
# -------------------------------
def save_qtable(qtab: QTableDict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convertir dict -> arrays serializables
    keys = np.array(list(qtab.table.keys()), dtype=object)
    vals = np.array(list(qtab.table.values()), dtype=object)
    np.savez_compressed(path, keys=keys, vals=vals, nA=qtab.nA, init_q=qtab.init_q)
    print(f"[INFO] Q-table guardada en: {path}.npz ({len(keys)} estados)")

def load_qtable(path: str) -> QTableDict:
    data = np.load(path + ".npz", allow_pickle=True)
    nA = int(data["nA"])
    init_q = float(data["init_q"])
    qtab = QTableDict(n_actions=nA, init_q=init_q)
    keys = data["keys"]
    vals = data["vals"]
    for k, v in zip(keys, vals):
        qtab.table[bytes(k)] = np.array(v, dtype=np.float32)
    print(f"[INFO] Q-table cargada desde: {path}.npz ({len(keys)} estados)")
    return qtab


# -------------------------------
# Ejecución
# -------------------------------
if __name__ == "__main__":
    # Hiperparámetros base (puedes ajustar)
    EPISODIOS = 300          # súbelo si quieres explorar más
    ALPHA = 0.1
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    STRIDE = 6               # 84//6 = 14 => estado de 14x14 binario
    THRESH = 128
    SAVE_PATH = "./checkpoints/amidar_qtable"

    t0 = time.time()
    q, rewards = train_amidar_qlearning(
        episodes=EPISODIOS,
        gamma=GAMMA,
        alpha=ALPHA,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        stride=STRIDE,
        threshold=THRESH,
        seed=42,
        render=False,
        save_path=SAVE_PATH
    )
    dt = time.time() - t0
    print(f"Entrenamiento terminado en {dt/60:.1f} min. Estados aprendidos: {q.size()}.")
    print(f"Recompensas últimas 10: {rewards[-10:]}")
