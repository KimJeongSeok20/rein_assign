import time
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 100  # 픽셀 수
HEIGHT = 7  # 그리드 월드 세로
WIDTH = 7  # 그리드 월드 가로 #수정


class Env(tk.Tk):
    def __init__(self):
        super(Env, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.texts = []
        self.n_actions = len(self.action_space)
        self.title('monte carlo')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='white',
                                height=HEIGHT * UNIT,
                                width=WIDTH * UNIT)
        # 그리드 생성
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~560 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~560 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        for i in range(WIDTH):
            for j in range(HEIGHT):
               text_id= canvas.create_text(UNIT * i + UNIT / 2+ UNIT / 3, UNIT * j + UNIT / 2 + UNIT / 3,text='0',  fill="black")
               self.texts.append(text_id)


        # 캔버스에 이미지 추가
        self.rectangle = canvas.create_image(50, 50, image=self.shapes[0])

        tri_x=[450,250,450,350,550,250,350]
        tri_y=[50,150,250,350,350,450,550]
        self.triangle1 = canvas.create_image(tri_x[0], tri_y[0], image=self.shapes[1])
        self.triangle2 = canvas.create_image(tri_x[1], tri_y[1], image=self.shapes[1])
        self.triangle3 = canvas.create_image(tri_x[2], tri_y[2], image=self.shapes[1]) # 시작 50 한칸당 100
        self.triangle4 = canvas.create_image(tri_x[3], tri_y[3], image=self.shapes[1]) 
        self.triangle5 = canvas.create_image(tri_x[4], tri_y[4], image=self.shapes[1]) 
        self.triangle6 = canvas.create_image(tri_x[5], tri_y[5], image=self.shapes[1]) 
        self.triangle7 = canvas.create_image(tri_x[6], tri_y[6], image=self.shapes[1])  #수정


        up = PhotoImage(Image.open("./1-grid-world/img/up.png").resize((13, 13)))
        right = PhotoImage(Image.open("./1-grid-world/img/right.png").resize((13, 13)))
        left = PhotoImage(Image.open("./1-grid-world/img/left.png").resize((13, 13)))
        down = PhotoImage(Image.open("./1-grid-world/img/down.png").resize((13, 13)))
        
        for i in tri_x:
            for j in tri_y:
               text_id= canvas.create_text(UNIT * i + UNIT / 2+ UNIT / 3, UNIT * j + UNIT / 2 + UNIT / 3,text='-1',  fill="black")
  

        self.circle = canvas.create_image(450, 350, image=self.shapes[2])

        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("./1-grid-world/img/rectangle.png").resize((65, 65)))
        triangle = PhotoImage(
            Image.open("./1-grid-world/img/triangle.png").resize((65, 65)))
        circle = PhotoImage(
            Image.open("./1-grid-world/img/circle.png").resize((65, 65)))

        return rectangle, triangle, circle

    @staticmethod
    def coords_to_state(coords):
        x = int((coords[0] - 50) / 100)
        y = int((coords[1] - 50) / 100)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.5)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x, UNIT / 2 - y)
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()

        if action == 0:  # 상
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # 하
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # 좌
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # 우
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        # 에이전트 이동
        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        # 에이전트(빨간 네모)를 가장 상위로 배치
        self.canvas.tag_raise(self.rectangle)

        next_state = self.canvas.coords(self.rectangle)

        # 보상 함수
        if next_state == self.canvas.coords(self.circle):
            reward = 1
            done = True
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2),
                            self.canvas.coords(self.triangle3),
                            self.canvas.coords(self.triangle4),
                            self.canvas.coords(self.triangle5),
                            self.canvas.coords(self.triangle6),
                            self.canvas.coords(self.triangle7)]: # 수정
            reward = -1
            done = True
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)

        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()

    def _draw_text(self, canvas, x, y, text):
        x_center = UNIT * x + UNIT / 2
        y_center = UNIT * y + UNIT / 2
        text_id = canvas.create_text(x_center + UNIT / 3, y_center + UNIT / 3, text=text, fill="black")
        self.texts.append(text_id)

    def display_values(self, value_table):
        # 기존 텍스트 삭제
        for text_id in self.texts:
            self.canvas.delete(text_id)
        self.texts.clear()

        # 새로운 텍스트 작성
        for i in range(WIDTH):
            for j in range(HEIGHT):
                state = str([i, j])
                value = value_table[state]
                self._draw_text(self.canvas, i, j, round(value, 4))

    def _draw_policy(self, canvas, x, y, policy):
        x_center = UNIT * x + UNIT / 2
        y_center = UNIT * y + UNIT / 2


        bottom_id = canvas.create_text(x_center, y_center + UNIT / 3, text=policy[1], fill="blue")
        top_id = canvas.create_text(x_center, y_center - UNIT / 3, text=policy[0], fill="blue")
        left_id = canvas.create_text(x_center - UNIT / 3, y_center, text=policy[2], fill="blue")
        right_id = canvas.create_text(x_center + UNIT / 3, y_center, text=policy[3], fill="blue")

        self.texts.append(bottom_id)
        self.texts.append(top_id)
        self.texts.append(left_id)
        self.texts.append(right_id)

    def display_policy(self, policy):
        lst =[[4,0],[2,1],[4,2],[3,3],[5,3],[2,4],[3,5],[4,3]]
        for i in range(WIDTH):
            for j in range(HEIGHT):
                if [i,j] not in lst:
                    state = str([i, j])
                    action_probabilities = policy[state]
                    self._draw_policy(self.canvas, i, j, action_probabilities)
