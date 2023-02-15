import torch
import torch.nn as nn


class Sinkhorn(nn.Module):
    """
    Sinkhorn algorithm turns the input matrix into a bi-stochastic matrix.
    Parameter: maximum iterations max_iter
               a small number for numerical stability epsilon
    Input: input matrix s
    Output: bi-stochastic matrix s
    """
    def __init__(self, max_iter=10, tau=1., epsilon=1e-4, log_forward=True, batched_operation=False):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau
        self.epsilon = epsilon
        self.log_forward = log_forward
        self.batched_operation = batched_operation # batched operation may cause instability in backward computation,
                                                   # but will boost computation.

    def forward(self, *input, dummy=False, **kwinput):
        if dummy:
            return self.forward_log_dummy(*input, **kwinput)
        else:
            return self.forward_log(*input, **kwinput)

    def forward_log_dummy(self, s, nrows=None, ncols=None, r=None, c=None, dummy_row=False, dtype=torch.float32):
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        if r is None:
            log_r = torch.zeros(batch_size, s.shape[1], device=s.device)
        else:
            log_r = torch.log(r)
        if c is None:
            log_c = torch.zeros(batch_size, s.shape[2], device=s.device)
        else:
            log_c = torch.log(c)
        for b in range(batch_size):
            log_r[b, nrows[b]:] = -float('inf')
            log_c[b, ncols[b]:] = -float('inf')

        # operations are performed on log_s
        log_s = s / self.tau

        for b in range(batch_size):
            log_s[b, nrows[b]-1, ncols[b]-1] = -float('inf')

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            log_r = torch.cat((log_r, torch.full(dummy_shape[:2], -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')

                log_r[b, ori_nrows[b]:nrows[b]] = 0

        if self.batched_operation:
            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    for b in range(batch_size):
                        log_sum[b, nrows[b]-1:, :] = 0
                    log_s = log_s - log_sum + log_r.unsqueeze(2)
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    for b in range(batch_size):
                        log_sum[b, :, ncols[b]-1:] = 0
                    log_s = log_s - log_sum + log_c.unsqueeze(1)
                    log_s[torch.isnan(log_s)] = -float('inf')

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s_b = log_s[b, row_slice, col_slice]
                log_r_b = log_r[b, row_slice]
                log_c_b = log_c[b, col_slice]

                for i in range(self.max_iter):
                    new_log_s_b = log_s_b.clone()
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                        new_log_s_b[:-1, :] = log_s_b[:-1, :] - log_sum[:-1, :] + log_r_b[:-1].unsqueeze(1)
                    else:
                        log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                        new_log_s_b[:, :-1] = log_s_b[:, :-1] - log_sum[:, :-1] + log_c_b[:-1].unsqueeze(0)

                    log_s_b = new_log_s_b

                ret_log_s[b, row_slice, col_slice] = log_s_b

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)

        # ret_log_s = torch.full((batch_size, s.shape[1], s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

        # for b in range(batch_size):
        #    row_slice = slice(0, nrows[b])
        #    col_slice = slice(0, ncols[b])
        #    log_s = s[b, row_slice, col_slice]

    def forward_log(self, s, nrows=None, ncols=None, r=None, c=None, dummy_row=False, dtype=torch.float32):
        # computing sinkhorn with row/column normalization in the log space.
        if len(s.shape) == 2:
            s = s.unsqueeze(0)
            matrix_input = True
        elif len(s.shape) == 3:
            matrix_input = False
        else:
            raise ValueError('input data shape not understood.')

        batch_size = s.shape[0]

        if nrows is None:
            nrows = [s.shape[1] for _ in range(batch_size)]
        if ncols is None:
            ncols = [s.shape[2] for _ in range(batch_size)]

        if r is None:
            log_r = torch.zeros(batch_size, s.shape[1], device=s.device)
        else:
            log_r = torch.log(r)
        if c is None:
            log_c = torch.zeros(batch_size, s.shape[2], device=s.device)
        else:
            log_c = torch.log(c)
        for b in range(batch_size):
            log_r[b, nrows[b]:] = -float('inf')
            log_c[b, ncols[b]:] = -float('inf')

        # operations are performed on log_s
        log_s = s / self.tau

        if dummy_row:
            assert s.shape[2] >= s.shape[1]
            dummy_shape = list(s.shape)
            dummy_shape[1] = s.shape[2] - s.shape[1]
            ori_nrows = nrows
            nrows = ncols
            log_s = torch.cat((log_s, torch.full(dummy_shape, -float('inf')).to(s.device)), dim=1)
            log_r = torch.cat((log_r, torch.full(dummy_shape[:2], -float('inf')).to(s.device)), dim=1)
            for b in range(batch_size):
                log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -100
                log_s[b, nrows[b]:, :] = -float('inf')
                log_s[b, :, ncols[b]:] = -float('inf')

                log_r[b, ori_nrows[b]:nrows[b]] = 0

        if self.batched_operation:
            for i in range(self.max_iter):
                if i % 2 == 0:
                    log_sum = torch.logsumexp(log_s, 2, keepdim=True)
                    log_s = log_s - log_sum + log_r.unsqueeze(2)
                    log_s[torch.isnan(log_s)] = -float('inf')
                else:
                    log_sum = torch.logsumexp(log_s, 1, keepdim=True)
                    log_s = log_s - log_sum + log_c.unsqueeze(1)
                    log_s[torch.isnan(log_s)] = -float('inf')

            if dummy_row and dummy_shape[1] > 0:
                log_s = log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                log_s.squeeze_(0)

            return torch.exp(log_s)
        else:
            ret_log_s = torch.full((batch_size, log_s.shape[1], log_s.shape[2]), -float('inf'), device=s.device, dtype=s.dtype)

            for b in range(batch_size):
                row_slice = slice(0, nrows[b])
                col_slice = slice(0, ncols[b])
                log_s_b = log_s[b, row_slice, col_slice]
                log_r_b = log_r[b, row_slice]
                log_c_b = log_c[b, col_slice]

                for i in range(self.max_iter):
                    if i % 2 == 0:
                        log_sum = torch.logsumexp(log_s_b, 1, keepdim=True)
                        log_s_b = log_s_b - log_sum + log_r_b.unsqueeze(1)
                    else:
                        log_sum = torch.logsumexp(log_s_b, 0, keepdim=True)
                        log_s_b = log_s_b - log_sum + log_c_b.unsqueeze(0)

                ret_log_s[b, row_slice, col_slice] = log_s_b

            if dummy_row:
                if dummy_shape[1] > 0:
                    ret_log_s = ret_log_s[:, :-dummy_shape[1]]
                for b in range(batch_size):
                    ret_log_s[b, ori_nrows[b]:nrows[b], :ncols[b]] = -float('inf')

            if matrix_input:
                ret_log_s.squeeze_(0)

            return torch.exp(ret_log_s)
