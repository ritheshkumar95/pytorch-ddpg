import torch
import numpy as np
from torch.autograd import Variable


class ReplayBuffer:

    def __init__(self, cf):
        self.buffer_size = cf.buffer_size
        self.len = 0

        # Create buffers for (s_t, a_t, r_t, s_t+1)
        self.s_t_buffer = torch.cuda.FloatTensor(
            self.buffer_size, cf.state_dim)
        self.a_t_buffer = torch.cuda.FloatTensor(
            self.buffer_size)
        self.r_t_buffer = torch.cuda.FloatTensor(
            self.buffer_size)
        self.s_tp1_buffer = torch.cuda.FloatTensor(
            self.buffer_size, cf.state_dim)

    def sample(self, count):
        batch_idxs = np.random.randint(0, self.len, count)
        batch_idxs = Variable(torch.from_numpy(batch_idxs).long()).cuda()

        s_t = Variable(self.s_t_buffer.index_select(1, batch_idxs))
        a_t = Variable(self.a_t_buffer.index_select(1, batch_idxs))
        r_t = Variable(self.r_t_buffer.index_select(1, batch_idxs))
        s_tp1 = Variable(self.s_tp1_buffer.index_select(1, batch_idxs))

        return s_t, a_t, r_t, s_tp1

    def add(self, s_t, a_t, r_t, s_tp1):
        """
        adds a particular step in the replay buffer
        """
        self.len += 1
        if self.len >= self.maxSize:
            self.len = self.maxSize - 1
            self.s_t_buffer = torch.cat(
                [self.s_t_buffer, s_t], 0)[1:]
            self.a_t_buffer = torch.cat(
                [self.a_t_buffer, a_t], 0)[1:]
            self.r_t_buffer = torch.cat(
                [self.r_t_buffer, r_t], 0)[1:]
            self.s_tp1_buffer = torch.cat(
                [self.s_tp1_buffer, s_tp1], 0)[1:]

        else:
            self.s_t_buffer[self.len - 1] = s_t
            self.a_t_buffer[self.len - 1] = a_t
            self.r_t_buffer[self.len - 1] = r_t
            self.s_tp1_buffer[self.len - 1] = s_tp1
