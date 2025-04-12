import streamlit as st
import pygame
import neat
import sys
import math
import random
from PIL import Image

# Constants
WIDTH = 800
HEIGHT = 600
CAR_SIZE_X = 60    
CAR_SIZE_Y = 60
BORDER_COLOR = (255, 255, 255, 255)  # Color To Crash on Hit
current_generation = 0  # Generation counter

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('assets/car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.position = [830, 920]  # Starting Position
        self.angle = 0
        self.speed = 0
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]  # Calculate Center
        self.alive = True  # Boolean To Check If Car is Crashed
        self.distance = 0  # Distance Driven
        self.time = 0  # Time Passed

    def draw(self, screen):
        screen.blit(self.sprite, self.position)  # Draw Sprite

    def update(self, game_map):
        if self.speed == 0:  # Set initial speed
            self.speed = 20
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.distance += self.speed
        self.time += 1
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)

def run_simulation(genomes, config):
    nets = []
    cars = []
    pygame.init()

    screen = pygame.Surface((WIDTH, HEIGHT))
    game_map = pygame.image.load('assets/map.png').convert()

    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    global current_generation
    current_generation += 1

    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate([0, 0, 0, 0, 0])
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10  # Left
            elif choice == 1:
                car.angle -= 10  # Right
            elif choice == 2:
                car.speed -= 2  # Slow Down
            else:
                car.speed += 2  # Speed Up

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break

        counter += 1
        if counter == 30 * 40:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        # Convert the Pygame Surface to a PIL Image to show in Streamlit
        img_array = pygame.surfarray.array3d(screen)
        img = Image.fromarray(img_array)
        st.image(img, caption=f'Generation: {current_generation}', use_column_width=True)

        pygame.display.flip()
        pygame.time.wait(50)

if __name__ == "__main__":
    config_path = "./config.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(run_simulation, 1000)
