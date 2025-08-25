import gymnasium as gym

import gymnasium
import ale_py
print("Gymnasium version:", gymnasium.__version__)
print("ALE-Py version:", ale_py.__version__)

#Cargar el entorno del juego
# Usamos 'Amidar-v5' para la versión estándar de Amidar.
env = gym.make("ALE/Amidar-v5", render_mode="human")

#Obtener información del entorno
num_acciones = env.action_space.n
print(f"Número de acciones posibles: {num_acciones}")


# Jugaremos 5 partidas completas para ver cómo se comporta.
num_partidas = 5

for partida in range(num_partidas):
    # Reinicia el entorno al comienzo de cada partida y obtiene la primera observación.
    observacion, info = env.reset()
    puntuacion_total = 0
    terminado = False

    print(f"\n--- Empezando Partida {partida + 1} ---")

    while not terminado:

        #Aquí es donde el agente "decide" qué hacer.
        #Simplemente elige una acción válida al azar.
        accion_aleatoria = env.action_space.sample()

        #Ejecutar la acción en el entorno
        # El entorno procesa la acción y nos devuelve el resultado.
        # observacion: La nueva imagen del juego (el siguiente frame).
        # recompensa: El puntaje obtenido por esa acción (ej. pintar una sección).
        # terminado: Un valor booleano que es True si el juego terminó (perdió todas las vidas).
        # truncado: Un valor booleano que indica si la partida terminó por un límite de tiempo.
        # info: Información adicional de diagnóstico.
        observacion, recompensa, terminado, truncado, info = env.step(accion_aleatoria)
        
        if terminado or truncado:
            break
        puntuacion_total += recompensa

    print(f"Puntuación final de la partida {partida + 1}: {puntuacion_total}")

env.close()
print("\nSimulación del agente aleatorio completada.")