import gymnasium as gym
# Crea el entorno para el juego Amidar
env = gym.make("Amidar-v4")
# Reinicia el entorno para obtener la primera observación
observation, info = env.reset()
print("¡Entorno configurado exitosamente!")
# Cierra el entorno
env.close()