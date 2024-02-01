import enum
import math
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn as nn
import ipdb
class ModelMeanType(enum.Enum):
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon

class GaussianDiffusion(nn.Module):
    def __init__(self, mean_type, noise_schedule, noise_scale, noise_min, noise_max,\
            steps, device, history_num_per_term=10, beta_fixed=True):
        self.mean_type = mean_type # <ModelMeanType.START_X: 1>
        self.noise_schedule = noise_schedule # 'linear-var'
        self.noise_scale = noise_scale # 0.0005
        self.noise_min = noise_min # 0.001
        self.noise_max = noise_max # 0.005
        self.steps = steps # 10
        self.device = device # self.device

        self.history_num_per_term = history_num_per_term # 10
        self.Lt_history = th.zeros(steps, history_num_per_term, dtype=th.float64).to(device) # torch.Size([10, 10])
        self.Lt_count = th.zeros(steps, dtype=int).to(device) # torch.Size([10])

        if noise_scale != 0.:
            self.betas = th.tensor(self.get_betas(), dtype=th.float64).to(self.device) # torch.Size([10])
            if beta_fixed:
                self.betas[0] = 0.00001  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
                # The variance \beta_1 of the first step is fixed to a small constant to prevent overfitting.
            assert len(self.betas.shape) == 1, "betas must be 1-D"
            assert len(self.betas) == self.steps, "num of betas must equal to diffusion steps"
            assert (self.betas > 0).all() and (self.betas <= 1).all(), "betas out of range"
            self.calculate_for_diffusion()

        super(GaussianDiffusion, self).__init__()
    
    def get_betas(self):
        """
        Given the schedule name, create the betas for the diffusion process.
        """
        # 线性的加噪方案，DDPM的加噪方案
        if self.noise_schedule == "linear" or self.noise_schedule == "linear-var":
            start = self.noise_scale * self.noise_min
            end = self.noise_scale * self.noise_max
            if self.noise_schedule == "linear":
                return np.linspace(start, end, self.steps, dtype=np.float64)
            else:
                return betas_from_linear_variance(self.steps, np.linspace(start, end, self.steps, dtype=np.float64))
        elif self.noise_schedule == "cosine":
            return betas_for_alpha_bar(
            self.steps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )
        elif self.noise_schedule == "binomial":  # Deep Unsupervised Learning using Noneequilibrium Thermodynamics 2.4.1
            ts = np.arange(self.steps)
            betas = [1 / (self.steps - t + 1) for t in ts]
            return betas
        else:
            raise NotImplementedError(f"unknown beta schedule: {self.noise_schedule}!")
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas # α torch.Size([10])
        self.alphas_cumprod = th.cumprod(alphas, axis=0) # $\bar{\alpha}_t$
        self.alphas_cumprod_prev = th.cat([th.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)  # $\bar{\alpha}_{t-1}$ 1.0 and alpha[1:step]
        self.alphas_cumprod_next = th.cat([self.alphas_cumprod[1:], th.tensor([0.0]).to(self.device)]).to(self.device)  # $\bar{\alpha}_{t+1}$ alpha[0:step-1] and 0.0
        assert self.alphas_cumprod_prev.shape == (self.steps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = th.sqrt(self.alphas_cumprod) # torch.Size([10])
        self.sqrt_one_minus_alphas_cumprod = th.sqrt((1.0 - self.alphas_cumprod)) # torch.Size([10])
        
        self.log_one_minus_alphas_cumprod = th.log(1.0 - self.alphas_cumprod) # torch.Size([10])
        self.sqrt_recip_alphas_cumprod = th.sqrt((1.0 / self.alphas_cumprod)) # torch.Size([10])
        self.sqrt_recipm1_alphas_cumprod = th.sqrt((1.0 / self.alphas_cumprod - 1)) # torch.Size([10])

        # calculations for posterior q(x_{t-1} | x_t, x_0) ---- equation 10
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ) # torch.Size([10])

        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain.
        # 后验分布方差在扩散模型开始处为0，计算对视时需要进行截断，就是用t=1时的值替代t=0时刻的值
        self.posterior_log_variance_clipped = th.log(
            th.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]])
        ) # torch.Size([10])
        
        # 后验分布计算均值公式的两个系数，对应于论文中公式11
        self.posterior_mean_coef1 = (
            self.betas * th.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ) # torch.Size([10])
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * th.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        ) # torch.Size([10])
    
    # 从q(x_t | x_0)中采样图像
    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # model
        # x_start: torch.Size([400, 94949])
        # steps
        # sampling_noise
        assert steps <= self.steps, "Too much steps in inference."
        if steps == 0:
            x_t = x_start # ---- 
        else:
            t = th.tensor([steps - 1] * x_start.shape[0]).to(x_start.device)
            x_t = self.q_sample(x_start, t)

        indices = list(range(self.steps))[::-1] # ----

        if self.noise_scale == 0.:
            for i in indices:
                t = th.tensor([i] * x_t.shape[0]).to(x_start.device)
                x_t = model(x_t, t)
            return x_t

        # Reverse step by step
        for i in indices:
            t = th.tensor([i] * x_t.shape[0]).to(x_start.device) # ----
            out = self.p_mean_variance(model, x_t, t) # ----
            if sampling_noise:
                noise = th.randn_like(x_t)
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
                )  # no noise when t == 0
                x_t = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
            else:
                x_t = out["mean"] # ----
        return x_t # predicted x_0
    
    def training_losses(self, model, x_start, reweight=False):
        batch_size, device = x_start.size(0), x_start.device # 400, device(type='cuda', index=0)
        ts, pt = self.sample_timesteps(batch_size, device, 'importance') # torch.Size([400]), torch.Size([400])
        noise = th.randn_like(x_start) # torch.Size([400, 94949])
        if self.noise_scale != 0.:
            x_t = self.q_sample(x_start, ts, noise)
        else:
            x_t = x_start

        terms = {}
        model_output = model(x_t, ts) # torch.Size([400, 49604])
        target = {
            ModelMeanType.START_X: x_start,
            ModelMeanType.EPSILON: noise,
        }[self.mean_type]

        assert model_output.shape == target.shape == x_start.shape

        mse = mean_flat((target - model_output) ** 2)

        if reweight == True:
            if self.mean_type == ModelMeanType.START_X:
                weight = self.SNR(ts - 1) - self.SNR(ts) # torch.Size([400])
                weight = th.where((ts == 0), 1.0, weight) # torch.Size([400])
                loss = mse
            elif self.mean_type == ModelMeanType.EPSILON:
                weight = (1 - self.alphas_cumprod[ts]) / ((1-self.alphas_cumprod_prev[ts])**2 * (1-self.betas[ts]))
                weight = th.where((ts == 0), 1.0, weight)
                likelihood = mean_flat((x_start - self._predict_xstart_from_eps(x_t, ts, model_output))**2 / 2.0)
                loss = th.where((ts == 0), likelihood, mse)
        else:
            weight = th.tensor([1.0] * len(target)).to(device)

        terms["loss"] = weight * loss
        
        # update Lt_history & Lt_count
        for t, loss in zip(ts, terms["loss"]):
            if self.Lt_count[t] == self.history_num_per_term:
                Lt_history_old = self.Lt_history.clone()
                self.Lt_history[t, :-1] = Lt_history_old[t, 1:]
                self.Lt_history[t, -1] = loss.detach()
            else:
                try:
                    self.Lt_history[t, self.Lt_count[t]] = loss.detach()
                    self.Lt_count[t] += 1
                except:
                    print(t)
                    print(self.Lt_count[t])
                    print(loss)
                    raise ValueError

        terms["loss"] /= pt
        return terms

    def sample_timesteps(self, batch_size, device, method='uniform', uniform_prob=0.001):
        if method == 'importance':  # importance sampling
            if not (self.Lt_count == self.history_num_per_term).all():
                return self.sample_timesteps(batch_size, device, method='uniform')
            
            Lt_sqrt = th.sqrt(th.mean(self.Lt_history ** 2, axis=-1))
            pt_all = Lt_sqrt / th.sum(Lt_sqrt)
            pt_all *= 1- uniform_prob
            pt_all += uniform_prob / len(pt_all)

            assert pt_all.sum(-1) - 1. < 1e-5

            t = th.multinomial(pt_all, num_samples=batch_size, replacement=True)
            pt = pt_all.gather(dim=0, index=t) * len(pt_all)

            return t, pt
        
        elif method == 'uniform':  # uniform sampling
            t = th.randint(0, self.steps, (batch_size,), device=device).long()
            pt = th.ones_like(t).float()

            return t, pt
            
        else:
            raise ValueError
    
    # 从q(x_t | x_0)中采样图像
    def q_sample(self, x_start, t, noise=None):
        if noise is None: # 如果没有传入噪声
            noise = th.randn_like(x_start) # # 从标准分布中随机采样一个与x_0大小一致的噪音 torch.Size([400, 94949])
        assert noise.shape == x_start.shape
        return (
            self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise # 直接用公式9进行重参数采样得到x_t
        )
    
    # 完整对应论文中的公式10和11，计算后验分布的均值和方差
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        # _extract_into_tensor函数是把sqrt_alphas_cumprod中的第t个元素取出，与x_0相乘得到均值
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # 通过模型(Unet)，基于x_t预测x_{t-1}的均值与方差；即逆扩散过程的均值和方差，也会预测x_0
    def p_mean_variance(self, model, x, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        """        
        B, C = x.shape[:2] # # batch_size(400), channel_nums (94949)
        assert t.shape == (B, ) # 一个batch中每个图片输入都对应一个时间步t，故t的size为(batch_size,)
        # 虽然Unet输出的尺寸一样，但模型训练预测的目标不同，输出数据表示的含义不同
        model_output = model(x, t) # torch.Size([400, 94949])

        model_variance = self.posterior_variance # torch.Size([10])
        model_log_variance = self.posterior_log_variance_clipped # torch.Size([10])

        model_variance = self._extract_into_tensor(model_variance, t, x.shape) # torch.Size([400, 94949])
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape) # torch.Size([400, 94949])
        
        if self.mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, eps=model_output)
        else:
            raise NotImplementedError(self.mean_type)
        
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    # 基于论文中的公式11，将公式转换以下就能基于均值μ和x_t求x_0；参数中的xprev就是Unet模型预测的均值
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def SNR(self, t):
        """
        Compute the signal-to-noise ratio for a single timestep.
        """
        self.alphas_cumprod = self.alphas_cumprod.to(t.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    
    
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        # res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        arr = arr.to(timesteps.device) # torch.Size([10])
        res = arr[timesteps].float() # torch.Size([400])
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None] # torch.Size([400, 1])
        return res.expand(broadcast_shape) # torch.Size([400, 94949])

def betas_from_linear_variance(steps, variance, max_beta=0.999):
#         steps: 10
#         variance: np.linspace from the start(5e-7) to the end(2.5e-6) with setps(10)
#         max_beta=0.999
#     ipdb.set_trace()
    alpha_bar = 1 - variance # (10,)
    betas = []
    betas.append(1 - alpha_bar[0])
    for i in range(1, steps):
        betas.append(min(1 - alpha_bar[i] / alpha_bar[i - 1], max_beta))
    return np.array(betas) # (10,)

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))
