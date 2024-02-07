from gym_multigrid.multigrid import *

class CoveringGameEnv(MultiGridEnv):
    """
    Environment in which the agents have to collect the balls
    """

    def __init__(
        self,
        size=10,
        width=None,
        height=None,
        num_balls=[],
        agents_index = [],
        balls_index=[],
        balls_reward=[],
        zero_sum = False,
        view_size=7,
        reward_mode='LOO'
    ):
        self.num_balls = num_balls
        self.balls_index = balls_index
        self.balls_reward = balls_reward
        self.zero_sum = zero_sum
        self.reward_mode = reward_mode
        self.world = World

        agents = []
        for i in agents_index:
            agents.append(Agent(self.world, i, view_size=view_size))

        balls = []
        for number, index, reward in zip(self.num_balls, self.balls_index, self.balls_reward):
            for i in range(number):
                balls.append(Ball(self.world, index, reward))

        self.balls = balls

        super().__init__(
            grid_size=size,
            width=width,
            height=height,
            max_steps= 10000,
            # Set this to True for maximum speed
            see_through_walls=False,
            agents=agents,
            agent_view_size=view_size,
            actions_set = CoveringActions
        )



    def _gen_grid(self, width, height, pos_lst = None):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(self.world, 0, 0)
        self.grid.horz_wall(self.world, 0, height-1)
        self.grid.vert_wall(self.world, 0, 0)
        self.grid.vert_wall(self.world, width-1, 0)
        if pos_lst is None:
            for ball in self.balls:
                self.place_obj(ball)
        else:
            for ball in self.balls:
                pos = pos_lst.pop(0)
                self.grid.set(*pos, ball)
                ball.init_pos = pos
                ball.cur_pos = pos

        # Randomize the player start position and orientation
        if not pos_lst:
            for a in self.agents:
                self.place_agent(a)
        else:
            for a in self.agents:
                pos = pos_lst.pop(0)
                #self.grid.set(*pos, a)
                a.init_pos = pos
                a.pos = pos


    def _reward(self, i, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """
        for j,a in enumerate(self.agents):
            if a.index==i or a.index==0:
                rewards[j]+=reward
            if self.zero_sum:
                if a.index!=i or a.index==0:
                    rewards[j] -= reward

    def _calc_reward(self, mode='identical_interest'):
        total_reward = 0
        for ball in self.balls:
            for a in self.agents:
                if a.in_view(*ball.cur_pos):
                    total_reward += ball.reward
                    break
        if mode == 'identical_interest':
            rewards = [total_reward for _ in range(len(self.agents)+1)]
        if mode == 'LOO':
            counts = []
            rewards = []
            for ball in self.balls:
                count = 0
                for a in self.agents:
                    if a.in_view(*ball.cur_pos):
                        count += 1
                counts.append(count)
            for a in self.agents:
                reward = 0
                for (i, ball) in enumerate(self.balls):
                    if counts[i] == 1 and a.in_view(*ball.cur_pos):
                        reward += ball.reward
                rewards.append(reward)
            rewards.append(total_reward)

        if mode == 'congestion':
            counts = []
            rewards = []
            for ball in self.balls:
                count = 0
                for a in self.agents:
                    if a.in_view(*ball.cur_pos):
                        count += 1
                counts.append(count)
            for a in self.agents:
                reward = 0
                for (i, ball) in enumerate(self.balls):
                    if a.in_view(*ball.cur_pos):
                        if counts[i] == 1:
                            reward += ball.reward
                        elif counts[i] == 2:
                            reward += 3/7 * ball.reward
                rewards.append(reward)
            rewards.append(total_reward)
        return rewards




    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        #TODO: No pickup in our setting
        if fwd_cell:
            if fwd_cell.can_pickup():
                if fwd_cell.index in [0, self.agents[i].index]:
                    fwd_cell.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)
                    self._reward(i, rewards, fwd_cell.reward)

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def step(self, actions):
        obs, rewards, done, info = MultiGridEnv.step(self, actions)
        return obs, rewards, done, info


class CoveringGame4HEnv10x10N2(CoveringGameEnv):
    def __init__(self):
        super().__init__(size=10,
        view_size = 3,
        num_balls=[5],
        agents_index = [1,2,3],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=False)

class CoveringGame4HEnv10x10N3(CoveringGameEnv):
    def __init__(self):
        super().__init__(size=10,
        view_size = 3,
        num_balls=[4],
        agents_index = [1,2,3],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=False)

class CoveringGame4HEnv10x10N10(CoveringGameEnv):
    def __init__(self):
        super().__init__(size=10,
        view_size = 3,
        num_balls=[4],
        agents_index = [1 for i in range(10)],
        balls_index=[0],
        balls_reward=[1],
        zero_sum=False)


class CoveringActions:
    available=['still', 'left', 'right', 'forward']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3