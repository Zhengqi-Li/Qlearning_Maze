import random
import numpy as np

class Robot(object):

    def __init__(self, maze, alpha=0.5, gamma=0.9, epsilon0=0.5):

        self.maze = maze
        self.valid_actions = self.maze.valid_actions
        self.state = None
        self.action = None

        # Set Parameters of the Learning Robot
        self.alpha = alpha
        self.gamma = gamma

        self.epsilon0 = epsilon0
        self.epsilon = epsilon0
        self.t = 0

        self.Qtable = {}
        self.reset()

    def reset(self):
        """
        Reset the robot
        """
        self.state = self.sense_state()
        self.create_Qtable_line(self.state)

    def set_status(self, learning=False, testing=False):
        """
        Determine whether the robot is learning its q table, or
        exceuting the testing procedure.
        """
        self.learning = learning
        self.testing = testing

    def update_parameter(self):
        """
        Some of the paramters of the q learning robot can be altered,
        update these parameters when necessary.
        """
        if self.testing:
            # TODO 1. No random choice when testing
            self.epsilon = self.epsilon0
            
        else:
            # TODO 2. Update parameters when learning
            # http://bbs.bugcode.cn/t/64787
            decay = 0.95
            min_epsilon = 0.1
            self.epsilon = max(min_epsilon, self.epsilon * decay)
            self.t += 1

        return self.epsilon

    def sense_state(self):
        """
        Get the current state of the robot. In this
        """

        # TODO 3. Return robot's current state
       
        return self.maze.sense_robot()
    

    def create_Qtable_line(self, state):
        """
        Create the qtable with the current state
        """
        # TODO 4. Create qtable with current state
        # Our qtable should be a two level dict,
        # Qtable[state] ={'u':xx, 'd':xx, ...}
        # If Qtable[state] already exits, then do
        # not change it.
        self.Qtable.setdefault(state, {a : 0.0 for a in self.valid_actions})
        
#         if state not in self.Qtable:
            #http://codingpy.com/article/python-list-comprehensions-explained-visually/
#             self.Qtable[state] = {action : 0 for action in self.valid_actions}

    def choose_action(self):
        """
        Return an action according to given rules
        """
        def is_random_exploration():

            # TODO 5. Return whether do random choice
            # hint: generate a random number, and compare
            # it with epsilon
            return np.random.uniform() < self.epsilon

        if self.learning:
            if is_random_exploration():
                # TODO 6. Return random choose aciton
                return np.random.choice(self.valid_actions)
            else:
                # TODO 7. Return action with highest q value
                return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
            
        elif self.testing:
            # TODO 7. choose action with highest q value
            return max(self.Qtable[self.state], key=self.Qtable[self.state].get)
        else:
            # TODO 6. Return random choose aciton
            return np.random.choice(self.valid_actions)
            

    def update_Qtable(self, r, action, next_state):
        """
        Update the qtable according to the given rule.
        """
        if self.learning:
            
            # TODO 8. When learning, update the q table according
            # to the given rules
            # https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/2-1-general-rl/
            # 之前学习到的 Q 值，即根据当前状态和动作得到的 Q 值
            q_old = self.Qtable[self.state][action] 
            # 新学习到的 Q 值， 即根据奖励值 r，折扣率 gamma 以及具有最大 Q 值的下个状态得到的 Q 值
            q_learned = r + self.gamma * max(self.Qtable[next_state].values())
            # 将两者相减，乘上学习速率，得到当前状态和动作的 Q 值
            self.Qtable[self.state][action] += self.alpha * (q_learned - q_old)
            
#            在新学习到的Q值中，γ*maxQ 的一项目就考虑了所谓的「未来奖励」——这是强化学习中的一个巨大亮点。也就是说，我们在计算、衡量一个动作的时候，不仅考虑它当前一步获得的奖励 r，还要考虑它执行这个动作之后带来的累计奖励——这能够帮助我们更好地衡量一个动作的好坏。但是这时候机器人并没有真正地往前走，而是使用Qtable 中原有地 next_state 的值来估计这个未来奖励。其中 γ 是折扣因子，是一个(0,1)之间的值。一般我们取0.9，能够充分地对外来奖励进行考虑。如果这个值大于1，那么实际上未来奖励会发散开来（因为这是一个不断累加、迭代的过程），导致Qtable不能发散。它能够帮助终点处的正奖励“扩散”到周围，也就是说，这样机器人更能够成功地学习到通往终点的路径。
            

    def update(self):
        """
        Describle the procedure what to do when update the robot.
        Called every time in every epoch in training or testing.
        Return current action and reward.
        """
        self.state = self.sense_state() # Get the current state
        self.create_Qtable_line(self.state) # For the state, create q table line

        action = self.choose_action() # choose action for this state
        reward = self.maze.move_robot(action) # move robot for given action

        next_state = self.sense_state() # get next state
        self.create_Qtable_line(next_state) # create q table line for next state

        if self.learning and not self.testing:
            self.update_Qtable(reward, action, next_state) # update q table
            self.update_parameter() # update parameters

        return action, reward
