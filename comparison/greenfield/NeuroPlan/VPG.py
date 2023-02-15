import os
import sys
import gym
import time
import torch
import numpy as np

from torch import optim

from NeuroPlan.logger import EpochLogger
from NeuroPlan.utils import combine_shape, cumulative_sum, count_params, statistics_scalar


class VPGBuffer():
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combine_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combine_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rwd_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rwd, val, logp):
        assert self.ptr < self.max_size     # buffer should have room so that you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rwd_buf[self.ptr] = rwd
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, done_flag, last_val=0, info=None, ep_ret=None):
        path_slice = slice(self.path_start_idx, self.ptr)
        if done_flag:
            ep_ret = ep_ret
        rwds = np.append(self.rwd_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # implement GAE-Lambda advantage calculation
        deltas = rwds[:-1]+self.gamma*vals[1:]-vals[:-1]
        self.adv_buf[path_slice] = cumulative_sum(deltas, self.gamma*self.lam)

        # computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = cumulative_sum(rwds, self.gamma)[:-1]
        self.path_start_idx = self.ptr
        return ep_ret

    def get(self):
        assert self.ptr == self.max_size    # buffer should be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # implement the advantage normalization trick
        adv_mean, adv_std = statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean)/adv_std
        data = dict(
            obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf, 
            adv=self.adv_buf, logp=self.logp_buf
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def policy_gradient(
    env, actor_critic, device, ac_kwargs=dict(), logger_kwargs=dict(), 
    gamma=0.99, lam=0.97, pi_lr=3e-4, v_lr=1e-3, v_iters=80,
    max_ep_len=200, steps_per_epoch=1000, epochs=100, save_freq=10, 
    seed=0, model_path=None
):
    def compute_pi_loss(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        obs, act, adv, logp_old = obs.to(device), act.to(device), adv.to(device), logp_old.to(device)

        pi, logp = ac.actor(obs, act)
        pi_loss = -(logp*adv).mean()

        approx_kl = (logp_old-logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent)
        return pi_loss, pi_info

    def compute_v_loss(data):
        obs, ret = data['obs'], data['ret']
        obs, ret = obs.to(device), ret.to(device)

        v_loss = ((ac.critic(obs)-ret)**2).mean()
        return v_loss

    def update():
        data = buf.get()

        # get loss and info values before update
        pi_loss_old, pi_info_old = compute_pi_loss(data)
        pi_loss_old = pi_loss_old.item()
        v_loss_old = compute_v_loss(data).item()

        # train policy with a single step of gradient descent
        pi_optimizer.zero_grad()
        pi_loss, pi_info = compute_pi_loss(data)
        pi_loss.backward()
        pi_optimizer.step()

        # value function learning
        for i in range(v_iters):
            v_optimizer.zero_grad()
            v_loss = compute_v_loss(data)
            v_loss.backward()
            v_optimizer.step()

        # log changes from update
        kl, ent = pi_info['kl'], pi_info_old['ent']
        logger.store(
            LossPi=pi_loss.item(), LossV=v_loss.item(), KL=kl, Entropy=ent, 
            DeltaLossPi=(pi_loss.item()-pi_loss_old), DeltaLossV=(v_loss.item()-v_loss_old)
        )
    
    
    logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    # seed = 1
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    obs_dim = env.obs_space.shape
    act_dim = env.act_space.shape

    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    ac = actor_critic(env.obs_space, env.act_space, **ac_kwargs).to(device)

    itr = -1
    if model_path: 
        # save_path = os.path.join(model_path, 'save')
        # saves = [int(x.split('.')[0][5:]) for x in os.listdir(save_path) if len(x)>8 and 'model' in x]
        # itr = '{}'.format(max(saves)) if len(saves) > 0 else ''
        # file_name = os.path.join(model_path, 'save', 'model'+itr+'.pt')
        print('\n\nLoading from {}...\n\n'.format(model_path))
        ac = torch.load(model_path)

    pi_optimizer = optim.Adam(ac.actor.parameters(), lr=pi_lr)
    v_optimizer = optim.Adam(ac.critic.parameters(), lr=v_lr)

    pi_params = count_params(ac.actor)
    v_params = count_params(ac.critic)
    logger.log('\nNumber of parameters: \t pi: {}, \t v: {}\n'.format(pi_params, v_params))
    logger.setup_saver(ac)

    start_time = time.time()
    o, mask = env.reset()
    ep_ret = 0
    ep_len = 0
    for epoch in range(int(itr)+1, epochs):
        for t in range(steps_per_epoch):
            o_tensor = torch.as_tensor(o, dtype=torch.float32).to(device)
            mask_tensor = torch.as_tensor(mask, dtype=torch.uint8).to(device)
            a, v, logp = ac.step(o_tensor, mask_tensor)

            next_o, next_mask, r, d, info = env.step(a)
            ep_ret += r
            ep_len += 1

            buf.store(o, a, r, v, logp)
            logger.store(Vals=v, TotalCosts=env.total_cost)

            o = next_o
            mask = next_mask

            timeout = ep_len==max_ep_len
            terminal = d or timeout
            epoch_ended = t==steps_per_epoch-1
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    o_end = torch.as_tensor(o, dtype=torch.float32).to(device)
                    mask_end = torch.as_tensor(mask, dtype=torch.uint8).to(device)
                    _, v, _ = ac.step(o_end, mask_end)
                else:
                    v = 0
                ep_ret = buf.finish_path(d, v, info, ep_ret)
                if terminal:
                    # save EpRet/EpLen only if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)

                if d and (env.solution['total_cost'] < env.best_solution['total_cost']): 
                    env.best_solution['path_cost'] = env.solution['path_cost']
                    env.best_solution['load_cost'] = env.solution['load_cost']
                    env.best_solution['total_cost'] = env.solution['total_cost']
                    env.best_solution['acts'] = env.solution['acts']
                    with open(os.path.join(logger.output_dir, 'solution.txt'), 'a') as f:
                        f.write('Path Cost: {}\n'.format(env.best_solution['path_cost']))
                        f.write('Load Cost: {}\n'.format(env.best_solution['load_cost']))
                        f.write('Total Cost: {}\n'.format(env.best_solution['total_cost']))
                        f.write('Actions: {}\n\n'.format(env.best_solution['acts']))
                
                o, mask = env.reset()
                ep_ret = 0
                ep_len = 0
        
        update()

        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, epoch)

        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpRet', with_min_max=True)
        logger.log_tabular('TotalCosts', with_min_max=True)
        logger.log_tabular('Vals', with_min_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

    sol = env.best_solution
    return sol