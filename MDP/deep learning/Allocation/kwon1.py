import matplotlib.pyplot as plt
import random
import numpy as np
import math
from scipy.special import softmax
from tqdm import tqdm  # Progress Bar


model_param = {
    's': 100,
    'tau_s': 5,  # Smoothing
    'tau_d': 40,  # Grid
    'ro': [round(elem, 20) for elem in np.arange(-2, 2.1, 0.4).tolist()],  # Saftey Factor
    'step_size': 0.1,
    'max_time': 500000
}


class ProblemEnv():
    def __init__(self, model_param):
        for key, value in model_param.items():
            setattr(self, key, value)

        self.states_arr = self.StatesRange()

    def StatesRange(self):
        li = range(self.s * 20)  # *2 to cover IPs which are over the s
        return [li[i:i+int(self.s/self.tau_d)] for i in range(0, len(li), int(self.s/self.tau_d))]


class Retailers():
    _registry = []

    def __init__(self, model_param, states_arr, target_fr, lead_time, cv, T):
        self._registry.append(self)

        for key, value in model_param.items():
            setattr(self, key, value)

        self.states_arr = states_arr  # States are s1 (0~5), s2(5~10) in the paper

        self.target_fr = target_fr
        self.lead_time = lead_time
        self.cv = cv
        self.T = T
        self.Q_arr = [0] * model_param['max_time']
        self.fillrate_arr = [0] * model_param['max_time']
        self.safety_factor_arr = [0] * model_param['max_time']
        self.invt_pos = [0] * model_param['max_time']
        self.invt_pos[0] = 50

        self.case_list = {
            'sets': [],
            'fill_rates': []
        }

        _ = [0]
        while len(_) != 0:
            self.d_arr, self.sig_arr = self.Demands()
            _ = [x for x in self.d_arr if x < 1]
        #print([x for x in self.d_arr if x == 0])

    def Demands(self):
        d_arr_ = []
        sig_arr_ = []

        for _ in range(self.max_time):
            if _ == 0:
                self.mu = 50
                self.sigma = (self.mu * self.cv)
            else:
                if _ % self.T == 0:
                    self.mu += random.uniform(-1, 1)
                    self.sigma = (self.mu * self.cv)
                    if self.mu <= 0:
                        self.mu = 50
                        self.sigma = (self.mu * self.cv)
            dem = random.gauss(self.mu, self.sigma)
            while dem == 0:
                #self.mu += random.uniform(-1, 1)
                #self.sigma = (self.mu * self.cv)
                dem = random.gauss(self.mu, self.sigma)
            d_arr_.append(int(abs(dem)))
            sig_arr_.append(self.sigma)

        return d_arr_, sig_arr_

    def SetsWhereQ(self, IP):
        IP_spectrum = []
        lst = range(int(IP-self.tau_s), int(IP+self.tau_s+1))
        for i in self.states_arr:
            for _ in lst:
                if _ in i:
                    IP_spectrum.append(self.states_arr.index(i))
        return list(set(IP_spectrum))

    def AddCase(self, IP, t):  # if there is no available case at the tau_s distance of Q, a new case is added.
        for i in self.states_arr:
            if IP in i:
                IP_set_index = self.states_arr.index(i)  # in which S_x set the Q belongs to
        fill_ratesIP = []
        for safety_factor in self.ro:
            Q_alts = self.d_arr[t] + safety_factor * self.sig_arr[t]  # ??? Add IP[t-1] ???
            fill_ratesIP.append(min(Q_alts/self.d_arr[t], 1))
        self.case_list['sets'].append(IP_set_index)
        self.case_list['fill_rates'].append(fill_ratesIP)

    def Q_t(self, t, safety_factor):
        return max(int(self.d_arr[t] + safety_factor * self.sig_arr[t]), 0)

    def Eq4(self, IP, cases_in_nb_IP):
        dist_case_from_IP = []
        for _ in cases_in_nb_IP:
            dist_case_from_IP.append(abs(IP - sum(self.states_arr[_]) / len(self.states_arr[_])))  # measure distance of each case which is in neighbourhood of Q and Q
        beta_xq_roq = []  # Equation 4
        for ro_index in range(len(model_param['ro'])):
            # fl_temp = []
            numerator_e4 = []
            denamerator_e4 = []
            _ = 0
            for case_ in cases_in_nb_IP:
                case_index = self.case_list['sets'].index(case_)
                fill_rate_ = self.case_list['fill_rates'][case_index]  # Bring all fill rates of selected case
                # fl_temp.append(fill_rate_[ro_index])
                K_ = math.exp(-(dist_case_from_IP[_]**2) / (model_param['tau_s']**2))
                numerator_e4.append(K_ * fill_rate_[ro_index])
                denamerator_e4.append(K_)
                _ += 1
            beta_xq_roq.append(sum(numerator_e4)/sum(denamerator_e4))
        return beta_xq_roq  # Shape number of ros in whcih SL is stored fro each ro

    def Update_SLs(self, t):
        IP = self.invt_pos[t+self.lead_time-1]
        #self.fillrate_arr[t] = min(Q_, self.d_arr[t]) / self.d_arr[t]
        Nb_IP = self.SetsWhereQ(IP)  # Identify the neighbourhood of Q1
        cases_in_nb_IP = list(set(Nb_IP).intersection(self.case_list['sets']))  # Intersection of two lists
        if len(cases_in_nb_IP) == 0:
            print('It is weired that the set is empty')
        else:
            # Update SLs for all safety_factors or just the Ro which was selected?
            for case_ in cases_in_nb_IP:
                case_index = self.case_list['sets'].index(case_)
                which_ro = model_param['ro'].index(self.safety_factor_arr[t])
                self.case_list['fill_rates'][case_index][which_ro] = self.case_list['fill_rates'][case_index][which_ro] + model_param['step_size'] * (self.fillrate_arr[t+self.lead_time] - self.case_list['fill_rates'][case_index][which_ro])
                # for safety_factor_indx in range(len(self.ro)):
                #self.case_list['fill_rates'][case_index][safety_factor_indx] = self.case_list['fill_rates'][case_index][safety_factor_indx] + model_param['step_size'] * (self.fillrate_arr[t+self.lead_time] - self.case_list['fill_rates'][case_index][safety_factor_indx])


class Supplier():
    def __init__(self, model_param):
        for key, value in model_param.items():
            setattr(self, key, value)

        self.s = model_param['s']

    def AllocationPolicy(self, t, r1, r2):
        d1 = r1.d_arr[t]  # Same day t
        d2 = r2.d_arr[t]  # Same day t
        if t < max(r1.lead_time, r2.lead_time):
            Q1_t = d1
            Q2_t = min(d2, model_param['s']-d1)
        else:
            tetha1 = ((sum(r1.fillrate_arr[:t])/len(r1.fillrate_arr[:t])) - r1.target_fr) / \
                (abs(
                    (sum(r1.fillrate_arr[:t])/len(r1.fillrate_arr[:t])) - r1.target_fr) +
                 abs(
                    (sum(r2.fillrate_arr[:t])/len(r2.fillrate_arr[:t])) - r2.target_fr)
                 )

            tetha2 = ((sum(r2.fillrate_arr[:t])/len(r2.fillrate_arr[:t])) - r2.target_fr) / \
                (abs(
                    (sum(r1.fillrate_arr[:t])/len(r1.fillrate_arr[:t])) - r1.target_fr) +
                 abs(
                    (sum(r2.fillrate_arr[:t])/len(r2.fillrate_arr[:t])) - r2.target_fr)
                 )

            if tetha1 >= 0 and tetha2 >= 0 and (tetha1 > 0 or tetha2 > 0):
                p1 = tetha1/(tetha1+tetha2)
                p2 = tetha2/(tetha1+tetha2)
            elif tetha1 <= 0 and tetha2 <= 0 and (tetha1 < 0 or tetha2 < 0):
                p1 = abs(1/tetha1)/(abs(1/tetha1)+abs(1/tetha2))
                p2 = abs(1/tetha2)/(abs(1/tetha1)+abs(1/tetha2))
            elif tetha1 == 0 and tetha2 == 0:
                p1 = 1/2
                p2 = 1/2
            else:
                p1 = max(tetha1, 0)/(max(tetha1, 0)+max(tetha2, 0))
                p2 = max(tetha2, 0)/(max(tetha1, 0)+max(tetha2, 0))

            Q1_t = int(r1.Q_arr[t] - max((r1.Q_arr[t] + r2.Q_arr[t]) - model_param['s'], 0) * p1)
            Q2_t = int(r2.Q_arr[t] - max((r1.Q_arr[t] + r2.Q_arr[t]) - model_param['s'], 0) * p2)

        return Q1_t, Q2_t


env = ProblemEnv(model_param)
r1 = Retailers(model_param, env.states_arr, target_fr=0.95, lead_time=2, cv=0.2, T=int(random.uniform(50, 100)))
r2 = Retailers(model_param, env.states_arr, target_fr=0.95, lead_time=2, cv=0.2, T=int(random.uniform(50, 100)))

s1 = Supplier(model_param)
# Q1, Q2 = s1.AllocationPolicy(0, r1, r2)
# r1.Q_arr.append(Q1)
# r1.fillrate_arr = [Q1 /]


for t in tqdm(range(model_param['max_time'])):
    #printProgressBar(t, "Progress")
    # t = 50
    # safety_factor = 0
    # If S < Q1+Q2 --> Allocation????
    # Compute the SL for current time and update the cases.
    # t = 2
    for r in Retailers._registry:
        # if t < r.lead_time:
        if t < max(r1.lead_time, r2.lead_time):
            Q_ = r.d_arr[t]
            r.Q_arr[t] = Q_
            r.fillrate_arr[t] = min(Q_, r.d_arr[t]) / r.d_arr[t]  # ??? Calculate here or line 232???
            r.safety_factor_arr[t] = 0
            if t == 0:
                r.invt_pos[t] = r.Q_arr[t]-r.d_arr[t]
            else:
                r.invt_pos[t] = r.invt_pos[t-1] + r.Q_arr[t]-r.d_arr[t]

    # if t < max(r1.lead_time, r2.lead_time):
        # continue  # Does not execute the rest of the code and jumps to the next iteration of loop

    # Plan for t+L
    if model_param['max_time'] - t > max(r1.lead_time, r2.lead_time):
        for r in Retailers._registry:
            t_l = t + r.lead_time
            IP = r.invt_pos[t_l-1]
            Nb_IP = r1.SetsWhereQ(IP)  # Identify the neighbourhood of IP
            cases_in_nb_IP = list(set(Nb_IP).intersection(r.case_list['sets']))  # Intersection of two lists

            if len(cases_in_nb_IP) == 0:  # Add a case
                r.AddCase(IP, t_l)
                cases_in_nb_IP = list(set(Nb_IP).intersection(r.case_list['sets']))
                #r.Q_arr[t_l] = Q_
                #r.fillrate_arr[t_l] = min(Q_, r.d_arr[t_l]) / r.d_arr[t_l]
                # r.safety_factor_arr[t_l] = 0
            # else:  # Locate best action(safety-factor)
            beta_xq_roq = r.Eq4(IP, cases_in_nb_IP)  # Equation 4

            # Selecting the best action (safety_factor) based on the Softmax
            # error = [-(x-r.target_fr)**2 for x in beta_xq_roq]  # Equation 3 inside bracket as the power of e
            error = [-abs(x-r.target_fr)*50 for x in beta_xq_roq]  # Equation 3 inside bracket as the power of e
            softmax_error = softmax(error, axis=0)  # Probabilities

            # Selecting action (safety-facotr) based on the softmax probability function
            sample_ind = np.random.choice(len(softmax_error), p=softmax_error)
            satefy_factor_action_softmax = model_param['ro'][sample_ind]  # Selecting action (safety-facotr) based on the softmax probability function
            r.safety_factor_arr[t] = satefy_factor_action_softmax
            Q_action = r.Q_t(t_l, satefy_factor_action_softmax)
            #kk = Q_action
            r.Q_arr[t_l] = Q_action
            r.fillrate_arr[t_l] = min(Q_action+IP, r.d_arr[t_l]) / r.d_arr[t_l]
        if model_param['s'] < r1.Q_arr[t_l] + r2.Q_arr[t_l]:
            Q1_, Q2_ = s1.AllocationPolicy(t_l, r1, r2)
            r1.Q_arr[t_l] = Q1_
            r2.Q_arr[t_l] = Q2_
            r1.fillrate_arr[t_l] = min(Q1_+r1.invt_pos[t_l-1], r1.d_arr[t_l]) / r1.d_arr[t_l]
            r2.fillrate_arr[t_l] = min(Q2_+r2.invt_pos[t_l-1], r2.d_arr[t_l]) / r2.d_arr[t_l]
        r1.invt_pos[t_l] = r1.invt_pos[t_l-1] + r1.Q_arr[t_l] - min(r1.Q_arr[t_l]+r1.invt_pos[t_l-1], r1.d_arr[t_l])
        r2.invt_pos[t_l] = r2.invt_pos[t_l-1] + r2.Q_arr[t_l] - min(r2.Q_arr[t_l]+r2.invt_pos[t_l-1], r2.d_arr[t_l])

        for r in Retailers._registry:
            if r.invt_pos[t_l-1] >= r.d_arr[t_l]:
                r.Q_arr[t_l] = 0
                r.fillrate_arr[t_l] = 1
                r.invt_pos[t_l] = r.invt_pos[t_l-1] - r.d_arr[t_l]
    #kk = r1.Q_arr[t_l]
    # Update SLs
    if model_param['max_time'] - t > max(r1.lead_time, r2.lead_time):
        for r in Retailers._registry:
            r.Update_SLs(t)

#### ###########################################################

# Equation 2: Updating the B(s_i, p_i) in a state class 's' based on the new info on selected Q and demand
# Note that many different Qs belong to one of the 's' classes. Therefore, when new Q realises, the SL of the container 'S' is updated each time (t) that Q is selected which is in 's'


#       ############################################

plt.plot(r1.fillrate_arr)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.show()
plt.savefig('fill_rates.png')

# plt.plot(r1.safety_factor_arr)
# plt.show()


plt.plot(r1.Q_arr)
plt.savefig('Q_arr')
# plt.show()

""" kk = [0.9, 0.95, 0.6, 0.99, 0.1]
error = [-(x-0.95)**2 for x in kk]  # Equation 3 inside bracket as the power of e
softmax_error = softmax(error, axis=0)  # Probabilities
tt = [(1+x+0.5*x) for x in error]
pp = [x/sum(tt) for x in tt]

error = [-abs(x-0.95) for x in kk]  # Equation 3 inside bracket as the power of e
softmax_error = softmax(error, axis=0)

error = [-abs(x-0.95)*10 for x in kk]  # Equation 3 inside bracket as the power of e
softmax_error = softmax(error, axis=0) """
