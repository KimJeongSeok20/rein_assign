import numpy as np
import random
from collections import defaultdict
from environment import Env


# 몬테카를로 에이전트 (모든 에피소드 각각의 샘플로 부터 학습)
class MCAgent:
    def __init__(self, actions):
        self.width = 7
        self.height = 7   # 수정
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.max_epsilon = 0.8 #수정
        self.min_epsilon = 0.1
        self.epsilon= self.max_epsilon

        self.samples = []
        self.value_table = defaultdict(float)
        self.policy_table=defaultdict(lambda: [0.25, 0.25, 0.25, 0.25])



    # 메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    # 모든 에피소드에서 에이전트가 방문한 상태의 큐 함수를 업데이트
    def update(self):
        if self.epsilon >= self.min_epsilon:
            self.epsilon-=0.002
        print(self.epsilon)
        G_t = 0
        visit_state = []
        done_state=True
        #for reward in reversed(self.samples):
        #    print("this is reward :",reward)
        #    state = str(reward[0])
        #    print("this is state :",state)
        #    if state not in visit_state:
        #        visit_state.append(state)
        #        G_t = reward[1] + self.discount_factor * G_t
        #        value = self.value_table[state]
        #        self.value_table[state] = (value +
        #                                   self.learning_rate * (G_t - value))
        for sample in reversed(self.samples): # 수정
            reward = sample[1]
            state = str(sample[0])
            if state not in visit_state :
                visit_state.append(state)
                G_t = reward + self.discount_factor * G_t
                value = self.value_table[state]
                self.value_table[state] = value + self.learning_rate * (G_t - value)  


    # 큐 함수에 따라서 행동을 반환
    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 랜덤 행동
            action = np.random.choice(self.actions)
        else:
            # 큐 함수에 따른 행동
            next_state = self.possible_next_state(state)
            action = self.arg_max(next_state)
        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    @staticmethod
    def arg_max(next_state,num=1):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        if num:
            return random.choice(max_index_list)
        else:
            return max_index_list

    # 가능한 다음 모든 상태들을 반환
    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]
        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]
        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]
        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state
    
    def get_policy(self):
        done_point=True
        for sample in reversed(self.samples):
                if done_point:
                    done_point=False
                    continue
                state = sample[0]
                next_state = self.possible_next_state(state)
                best_actions = self.arg_max(next_state, num=0)
                prob = 1 / len(best_actions)
                policy = [0.0] * 4
                for action in best_actions:
                    policy[action] = round(prob,2)
                self.policy_table[str(state)] = policy




# 메인 함수
if __name__ == "__main__":
    env = Env()
    agent = MCAgent(actions=list(range(env.n_actions)))

    for episode in range(1000):
        state = env.reset()
        action = agent.get_action(state)

        while True:
            env.render()

            # 다음 상태로 이동
            # 보상은 숫자이고, 완료 여부는 boolean
            next_state, reward, done = env.step(action)
            agent.save_sample(next_state, reward, done)

            # 다음 행동 받아옴
            action = agent.get_action(next_state)

            # 에피소드가 완료됐을 때, 큐 함수 업데이트
            if done:
                agent.update()
                agent.get_policy()
                agent.samples.clear()
                env.display_values(agent.value_table)
                env.display_policy(agent.policy_table)
                break
            

