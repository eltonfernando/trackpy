#!/usr/bin/python
# -*- coding: utf-8 -*-
"""

    File name         : kalman2D,py
    File Description  : filtro de kalman
                      :
    Author            : Eng. Elton Fernandes dos Santos
    Date created      : 16/01/2020
    Version           : v1.0
    Python Version    : 3.6
Este filtro foi implementado baseado na equação MRUV em X Y
e modelagem de espaço de estado
Filtro de kalaman é um modelo linear por isso não é
considerado a parte de aceleração.

Como Calibar:
Auterar  em __int__
Q: quanto menor o valor de Q mais confiavel é o modelo
R: quanto menor o valor de R mais confiavel é a medição
vale o mesmo para Q2 e R2 em que nesse caso R2>R e Q>Q2

Como usar
estancia a class
FK=Kaman()
verificar se  existe dados do sensor valido
se True: #sem oclusao
result= FK.correction(medida)
se false #com oclusao
result= FK.correction(medida,0)
FK.prediction()

"""
from typing import Union, List
import numpy as np


class KalmanBox(object):
    H = np.array(
        [
            [1.0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ]
    )  # measurement function
    R = np.eye(4) * 0.001  # incerteza do sensor
    Qv = (
        np.array(
            [
                [0.0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        * 10.1
    )

    def __init__(self, bbox):
        """
        x: matriz 6x1 estado (x,y,w,h) velocidade em x e velocidade em y
        P: matriz 6x6 incerteza inicial
        u: matriz de aceleração
        F: matriz 6x6 funcão de estado
        H: matriz 4x6 função de medição
        R: matriz 4x4 incerteza de medição
        Q: matriz 6x6 incerteza do modelo
        I: matriz 6x6 identidade
        lastResult: matriz 2x1 guarda resuldado enterior
        :param point_init:list com coordenada [x,y]
        """

        # self.inicio = time()
        self.x = np.array([[bbox[0]], [0.0], [bbox[1]], [0.0], [bbox[2]], [bbox[3]]])  # initial state (location and velocity)
        self.P = np.eye(6) * 1000  # initial uncertainty

        self.I = np.eye(6)  # identity matrix
        self.prediction()

    def correction(self, Z):
        """
        y=Z-HK
        S=H*P*trans(H) +R
        K=P*trans(H)*inv(S)
        x=x+(k*y)
        P=(I-k*H)P
        :param Z: np.array[[x],[y]]
        :return:Matrix 2x1 com coordenada (x,y)
        isso pode ser alterado para retorna velocidade
        estimada pelo filtro
        """
        if Z is None:
            Z = np.dot(self.H, self.x)

        y = Z - (np.dot(self.H, self.x))
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        k = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.x = self.x + np.dot(k, y)
        self.P = np.dot((self.I - np.dot(k, self.H)), self.P)

        return np.asarray(self.x).reshape(-1)

    def prediction(self):
        """
        Estima medida futura
        x'=FX + U
        P'=FPtrans(F) + Q
        :return:
        """
        dt = 1 / 10
        # time() - self.inicio
        # self.inicio = time()
        # print(f'FPS kalman {1/dt}')
        F = np.array(
            [
                [1.0, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ]
        )
        Q = np.dot(np.dot(F, self.Qv), F.T)
        self.x = np.dot(F, self.x)  # +self.u
        self.P = np.dot(F, np.dot(self.P, F.T)) + Q

    def update(self, bbox: Union[List[int], None]) -> np.ndarray:
        result = self.correction(bbox)
        self.prediction()
        return result
