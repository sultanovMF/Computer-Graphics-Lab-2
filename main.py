import pygame, sys
import numpy as np
import pandas as pd
from pygame.locals import *
from scipy.ndimage.interpolation import shift


def calculate_busier(Q_start, A, B, Q_end, step=0.1):
    points = []
    for t in np.arange(0, 1 + step, step):
        points.append(
            ((1 - t) ** 3) * Q_start + 2 * ((1 - t) ** 2) * t * A + 3 * (1 - t) * (t ** 2) * B + (t ** 3) * Q_end)

    return points


def calculate_control_points(node_points):
    shape_size = 2 * len(node_points) - 2
    matrix = np.zeros(shape=(shape_size, shape_size), dtype=int)
    right_col = np.zeros(shape=(shape_size, 2))

    matrix[0][0] = 2
    matrix[0][1] = -1
    right_col[0] = np.array(node_points[0])

    matrix[-1][-1] = 2
    matrix[-1][-2] = -1
    right_col[-1] = np.array(node_points[-1])

    first_diff_template = np.zeros(shape_size, dtype=int)
    first_diff_template[1] = 1
    first_diff_template[2] = 1

    second_diff_template = np.zeros(shape_size, dtype=int)
    second_diff_template[0] = 1
    second_diff_template[1] = -2
    second_diff_template[2] = 2
    second_diff_template[3] = -1

    index = 1
    for i in range(0, int((shape_size - 2) // 2)):
        matrix[index] = shift(first_diff_template, i * 2, cval=0)
        right_col[index] = 2 * np.array(node_points)[index]
        index += 1

    for i in range(0, int((shape_size - 2) // 2)):
        matrix[index] = shift(second_diff_template, i * 2, cval=0)
        right_col[index] = np.array([0, 0])
        index += 1
    result = np.linalg.solve(matrix, right_col)
    return result

    # matrix[-1] = shift(np.array([-1, 2], shape_size - 2))
    # print(matrix)
    # for i in range(shape_size):

    # print(matrix)
    # for i in range(1, 2 * len(node_points) - 2):


# matrix = np.array()


if __name__ == '__main__':
    scale = 0.5

    df = pd.read_csv("coordinates_of_a_serious_toad.csv", header=None, delimiter=';')
    Points = scale * df.to_numpy()
    Controls = calculate_control_points(Points)
    Controls = np.array(Controls)
    pygame.init()
    screen = pygame.display.set_mode((600, 600))

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        for i in range(1, len(Points)):
            result_points = calculate_busier(Points[i-1], Controls[2*i-2], Controls[2*i-1], Points[i], step=0.05)
            result_points = np.array(result_points).astype('int')
            for j in range(0, len(result_points)-1):
                pygame.draw.line(surface=screen, color=(255, 255, 255), start_pos=result_points[j], end_pos=result_points[j+1])

        # for j in range(0, len(result_points)):
        #     pygame.draw.line(surface=screen, color=(255, 255, 255), start_pos=result_points[j-1], end_pos=result_points[j])

        for P in Points:
            pygame.draw.circle(surface=screen, center=P, color=(255, 255, 255), radius=2)
        for P in Controls:
            pygame.draw.circle(surface=screen, center=P, color=(0, 255, 255), radius=2)

        pygame.display.update()
