import pygame, sys
import numpy as np
import pandas as pd
from pygame.locals import *
from scipy.ndimage.interpolation import shift
from scipy.spatial import distance


def calculate_bezier(Q_start, A, B, Q_end, step=0.05):
    points = []
    for t in np.arange(0, 1 + step, step):
        points.append(
            ((1 - t) ** 3) * Q_start + 3 * ((1 - t) ** 2) * t * A + 3 * (1 - t) * t ** 2 * B + t ** 3 * Q_end)

    return points


def calculate_control_points(node_points):
    n = 2 * len(node_points) - 2

    a = []  # 1 / length between points
    for i in range(0, len(node_points) - 1):
        a.append(1 / distance.euclidean(node_points[i], node_points[i + 1]))

    matrix = np.zeros(shape=(n, n))
    right_col = np.zeros(shape=(n, 2))

    matrix[0][0] = 2
    matrix[0][1] = -1
    right_col[0] = np.array(node_points[0])

    matrix[-1][-1] = 2
    matrix[-1][-2] = -1
    right_col[-1] = np.array(node_points[-1])
    for i in range(1, len(node_points) - 1):
        row = np.zeros_like(np.zeros(n))
        row[2*i - 2] = a[i-1] ** 2
        row[2*i - 1] = -2 * (a[i-1] ** 2)
        row[2*i] = 2 * (a[i] ** 2)
        row[2*i + 1] = - (a[i] ** 2)
        right_col[2*i - 1] = ((a[i] ** 2) * node_points[i]) - ((a[i-1] ** 2) * node_points[i-1])
        matrix[2*i - 1] = row

        row = np.zeros_like(np.zeros(n))
        row[2 * i - 1] = a[i-1]
        row[2 * i] = a[i]

        matrix[2*i] = row
        right_col[2*i] = node_points[i] * (a[i-1] + a[i])

    result = np.linalg.solve(matrix, right_col)
    return result


def draw_spline(surface, Q_start, A, B, Q_end, step=0.05):
    result_points = calculate_bezier(Q_start, A, B, Q_end,step=step)
    for j in range(0, len(result_points) - 1):
        pygame.draw.line(surface=surface, color=(255, 255, 255), start_pos=result_points[j],
                         end_pos=result_points[j + 1])


def draw_bezier_curve(surface, points, controls, step=0.05):
    for i in range(1, len(points)):
        draw_spline(screen, points[i - 1], controls[2 * i - 2], controls[2 * i - 1], points[i], step=step)

if __name__ == '__main__':
    WINDOW_H = 600
    WINDOW_W = 600
    N = 1
    scale = 0.5



    df = pd.read_csv("coordinates_of_a_serious_toad.csv", header=None, delimiter=';')

    Points = scale * df.to_numpy()[::N]
    Points = np.append(Points, Points[0].reshape((1,2)), axis=0)
    Controls = calculate_control_points(Points)
    Controls = np.array(Controls)
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_H, WINDOW_W))

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        draw_bezier_curve(screen, Points, Controls)

        for P in Points:
            pygame.draw.circle(surface=screen, center=P, color=(255, 255, 255), radius=2)

        # for P in Controls:
        #     pygame.draw.circle(surface=screen, center=P, color=(0, 255, 255), radius=2)

        pygame.display.update()