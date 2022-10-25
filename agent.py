import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env import final_states

#computation class
#alpha-learning ratee
#gamma-reward decayu
#espilon-dictates e-greedy selection
class Q_algorithm:
    def __init__(self,actions,alpha=0.01,reward_decay=0.9,epsilon=0.9):
        self.actions=actions #actions taken
        self.alpha=alpha
        self.y=reward_decay
        self.e=epsilon
        self.q_table=pd.DataFrame(columns=self.actions,dtype=np.float64)
        self.final_table=pd.DataFrame(columns=self.actions,dtype=np.float64)

    def choose_action(self, observation):
        # Checking if the state exists in the table
        self.add_to_qtable(observation)
        #epsilon selection of action, np.random.uniform returns a value from the normal distribution
        if np.random.uniform() < self.e:
            #choose the best action
            state_action=self.q_table.loc[observation, :]
            state_action=state_action.reindex(np.random.permutation(state_action.index))
            action=state_action.idxmax()
        else:
            action=np.random.choice(self.actions) #exploratory step
        return action

    def learning(self,state,action,reward,next_state):
        self.add_to_qtable(next_state)#does the next stae exist, if so add it to q table
        current_stateQtable=self.q_table.loc[state,action]

        if next_state != 'goal' or next_state != 'obstacle':
            q_update=reward + self.y*self.q_table.loc[next_state,:].max()
        else:
            q_update=reward

        self.q_table.loc[state,action] += self.alpha*(q_update - current_stateQtable)
        return self.q_table.loc[state,action]

    def add_to_qtable(self,state):
        if state not in self.q_table.index:
            self.q_table=self.q_table.append(pd.Series([0]*len(self.actions),index=self.q_table.columns,name=state,))
        
    def print_table(self):
        final_route=final_states()
        print("Full Q-Table")
        print(self.q_table)


        for i in range(len(final_route)):
            state=str(final_route[i])
            #iterate thru all indices in our q table, check whether it belongs to a state, update FINAL q table with that state
            for j in range(len(self.q_table.index)):
                if self.q_table.index[j] == state:
                    self.final_table.loc[state,:]=self.q_table[state,:]

    def results(self,steps,cost):
        f,(ax1,ax2)=plt.subplots(nrows=1,ncols=2)
        ax1.plot(np.arange(len(steps)),steps,'b')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Steps')
        ax1.set_title('Steps v/s Episode')

        ax2.plot(np.arange(len(cost)),cost,'r')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cost')
        ax2.set_title('Cost v/s Episode')

        plt.tight_layout()
        
        plt.figure()
        plt.plot(np.arange(len(steps)), steps, 'b')
        plt.title('Steps v/s Episode')
        plt.xlabel('Episode')
        plt.ylabel('Steps')

        plt.figure()
        plt.plot(np.arange(len(cost)), cost, 'b')
        plt.title('Cost v/s Episode')
        plt.xlabel('Episode')
        plt.ylabel('Cost')
        plt.show()