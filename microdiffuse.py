"""
microdiffuse.py — a single file of ~200 lines of pure Python with no dependencies
that trains and samples from a denoising diffusion probabilistic model (DDPM).
This is the complete algorithmic content of diffusion: dataset, autograd engine,
noise schedule, neural network denoiser, training loop, and sampling loop.
Everything else is just efficiency.
"""
import math, random, sys
from contextlib import contextmanager

# Autograd (inspired from micrograd)
class Value:
    __slots__ = ('data','grad','_prev','_op','_backward')
    def __init__(self, data, _prev=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other): return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other): return self * other

    def __neg__(self):
        out = Value(-self.data, (self,), 'neg')
        def _backward():
            self.grad += -1.0 * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * other**-1
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self

    def __pow__(self, other):
        # support number or Value (simple case)
        if isinstance(other, Value):
            # x**y, treat y as constant for simplicity (common usage is y=2)
            out = Value(self.data ** other.data, (self, other), 'pow')
            def _backward():
                # dx = y * x^{y-1}
                if self.data != 0:
                    self.grad += (other.data * (self.data ** (other.data - 1))) * out.grad
                # dy = x^{y} * ln(x)
                if self.data > 0:
                    other.grad += (out.data * math.log(self.data)) * out.grad
            out._backward = _backward
            return out
        else:
            out = Value(self.data ** other, (self,), 'powc')
            def _backward():
                if self.data != 0:
                    self.grad += other * (self.data ** (other - 1)) * out.grad
            out._backward = _backward
            return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        v = math.tanh(self.data)
        out = Value(v, (self,), 'tanh')
        def _backward():
            self.grad += (1 - v*v) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        v = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(v, (self,), 'sigmoid')
        def _backward():
            self.grad += v * (1 - v) * out.grad
        out._backward = _backward
        return out

    # building topo and backward pass
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for p in v._prev:
                    build(p)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    # convenience
    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

def V(x):
    return x if isinstance(x, Value) else Value(x)

# Utilities
@contextmanager
def no_grad():
    global _NO_GRAD
    old = _NO_GRAD
    _NO_GRAD = True
    try:
        yield
    finally:
        _NO_GRAD = old

_NO_GRAD = False

# Byepassing autograd
class G:
    @staticmethod
    def wrap(x):
        if _NO_GRAD:
            return x if isinstance(x, (int, float)) else float(x.data)
        return V(x)

# Dataset
n_data = 2000
# sample random points on unit circle (uniform angle)
data = [(math.cos(theta), math.sin(theta)) for theta in [2*math.pi*random.random() for _ in range(n_data)]]

# Linear noise scheduler
T = 50
beta_start, beta_end = 1e-4, 0.02
betas = [beta_start + (beta_end - beta_start) * t / (T-1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
ab = 1.0
for a in alphas:
    ab *= a
    alpha_bars.append(ab)
sqrt_alpha_bars = [math.sqrt(x) for x in alpha_bars]
sqrt_one_minus_alpha_bars = [math.sqrt(1 - x) for x in alpha_bars]
# posterior variances (tilde beta)
tilde_betas = [0.0] * T
for t in range(T):
    if t == 0:
        tilde_betas[t] = betas[t]
    else:
        tilde_betas[t] = betas[t] * (1 - alpha_bars[t-1]) / (1 - alpha_bars[t])

# Time embedding (sinusoidal as it is smoother)
def sinusoidal_embedding(t, dim=32, max_period=10000):
    emb = []
    for i in range(dim//2):
        freq = math.exp(-math.log(max_period) * i / (dim//2 - 1))
        angle = t * freq
        emb.append(math.sin(angle))
        emb.append(math.cos(angle))
    return emb

# MLP denoiser
class Linear:
    def __init__(self, nin, nout, bias=True):
        self.w = [[Value(random.gauss(0, math.sqrt(2/(nin+nout)))) for _ in range(nin)] for _ in range(nout)]
        self.b = [Value(0.0) for _ in range(nout)] if bias else [Value(0.0) for _ in range(nout)]
    def __call__(self, x_vals):
        out = []
        for i, row in enumerate(self.w):
            s = Value(0.0)
            for wi, xi in zip(row, x_vals):
                xi_wrapped = xi if isinstance(xi, Value) else Value(xi)
                s = s + wi * xi_wrapped
            s = s + self.b[i]
            out.append(s)
        return out
    def parameters(self):
        return [p for row in self.w for p in row] + self.b

class LayerNorm1D:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.scale = [Value(1.0) for _ in range(dim)]
        self.bias = [Value(0.0) for _ in range(dim)]
    def __call__(self, x):
        mean = sum(xi for xi in x) / len(x)
        var = sum((xi - mean) * (xi - mean) for xi in x) / len(x)
        std = (var + self.eps) ** 0.5
        out = []
        for i, xi in enumerate(x):
            out.append(self.scale[i] * ((xi - mean) / std) + self.bias[i])
        return out
    def parameters(self):
        return self.scale + self.bias

class Denoiser:
    def __init__(self, n_in=2, t_emb_dim=32, n_hid=128, n_out=2):
        self.t_proj = Linear(t_emb_dim, t_emb_dim)  # small projection (learned)
        self.fc1 = Linear(n_in + t_emb_dim, n_hid)
        self.ln1 = LayerNorm1D(n_hid)
        self.fc2 = Linear(n_hid, n_hid)
        self.ln2 = LayerNorm1D(n_hid)
        self.fc3 = Linear(n_hid, n_out)

    def __call__(self, x, y, t):
        # x,y are floats during training we will pass Value wrappers; t is int
        # build input list
        # time embedding -> pass through small linear and activation
        temb = sinusoidal_embedding(t, dim=len(self.t_proj.w[0]))
        # wrap temb as Values
        temb_v = [Value(v) for v in temb]
        temb_v = self.t_proj(temb_v)
        temb_v = [v.tanh() for v in temb_v]
        inp = [Value(x), Value(y)] + temb_v
        h = self.fc1(inp)
        h = self.ln1([v.tanh() for v in h])
        h = self.fc2(h)
        h = self.ln2([v.tanh() for v in h])
        out = self.fc3(h)
        # out is list of Value
        return out

    def parameters(self):
        params = []
        for comp in (self.t_proj, self.fc1, self.ln1, self.fc2, self.ln2, self.fc3):
            params += comp.parameters()
        return params

# instantiate model
t_emb_dim = 32
model = Denoiser(n_in=2, t_emb_dim=t_emb_dim, n_hid=128, n_out=2)
params = model.parameters()
print(f"num params: {len(params)}")

# Optimizer (Adam)
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.t = 0
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

opt = Adam(params, lr=3e-3)

# EMA for model weights (useful for sampling)
class EMA:
    def __init__(self, params, decay=0.999):
        self.decay = decay
        self.shadow = [p.data for p in params]
        self.params = params
    def update(self):
        for i, p in enumerate(self.params):
            self.shadow[i] = self.decay * self.shadow[i] + (1 - self.decay) * p.data
    def apply_shadow(self):
        self.backup = [p.data for p in self.params]
        for i, p in enumerate(self.params):
            p.data = self.shadow[i]
    def restore(self):
        for p, b in zip(self.params, self.backup):
            p.data = b

ema = EMA(params, decay=0.995)

# Training
random.seed(42)
num_steps = 40000 
print_every = 20

use_weighted_loss = False  # if True, weight by 1/tilde_beta_t

for step in range(num_steps):
    # sample data and timestep
    x0, y0 = data[random.randrange(len(data))]
    t = random.randrange(T)
    eps_x, eps_y = random.gauss(0,1), random.gauss(0,1)

    sab = sqrt_alpha_bars[t]
    s1ab = sqrt_one_minus_alpha_bars[t]
    xt = sab * x0 + s1ab * eps_x
    yt = sab * y0 + s1ab * eps_y

    pred = model(xt, yt, t) 
    # loss
    loss = (pred[0] - eps_x) ** 2 + (pred[1] - eps_y) ** 2
    if use_weighted_loss:
        loss = loss / (tilde_betas[t] + 1e-8)
    loss.backward()

    opt.step()
    opt.zero_grad()
    ema.update()

    if (step+1) % print_every == 0:
        print(f"step {step+1}/{num_steps} | loss {loss.data:.6f} | t={t}")
        sys.stdout.flush()

# Sampling
print('\nSampling (denoising random noise into circle points)')
ema.apply_shadow()  # use EMA weights for sampling
n_samples = 200
samples = []
with no_grad():
    for s in range(n_samples):
        x = random.gauss(0,1)
        y = random.gauss(0,1)
        # iterate t down to 0
        for t in reversed(range(T)):
            pred_vals = model(x, y, t)
            eps_px = pred_vals[0].data
            eps_py = pred_vals[1].data
            # compute mean
            coeff = betas[t] / math.sqrt(1 - alpha_bars[t])
            mean_x = (x - coeff * eps_px) / math.sqrt(alphas[t])
            mean_y = (y - coeff * eps_py) / math.sqrt(alphas[t])
            if t > 0:
                noise_x, noise_y = random.gauss(0,1), random.gauss(0,1)
                sigma = math.sqrt(tilde_betas[t])
                x = mean_x + sigma * noise_x
                y = mean_y + sigma * noise_y
            else:
                x, y = mean_x, mean_y
        samples.append((x,y))
        if (s+1) % 50 == 0:
            print(f"  generated {s+1}/{n_samples} samples")

ema.restore()

# Evaluation
# compute radii stats and angular uniformity (approx)
radii = [math.sqrt(x*x + y*y) for x,y in samples]
mean_r = sum(radii)/len(radii)
var_r = sum((r-mean_r)**2 for r in radii)/len(radii)
angles = [math.atan2(y,x) for x,y in samples]
# angular histogram
nbins = 12
hist = [0]*nbins
for a in angles:
    i = int((a + math.pi) / (2*math.pi) * nbins) % nbins
    hist[i] += 1

print(f"\nGenerated sample radius mean={mean_r:.4f} var={var_r:.6f}")
print("Angular histogram:", hist)

# Visualize the points generated wrto the actual circle
def write_svg(original, generated, fname='output_fixed.svg'):
    w, h, s = 500, 500, 180
    cx, cy = w/2, h/2
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
             f'<rect width="{w}" height="{h}" fill="#0f0f1a"/>',
             f'<line x1="{cx}" y1="0" x2="{cx}" y2="{h}" stroke="#222" stroke-width="0.5"/>',
             f'<line x1="0" y1="{cy}" x2="{w}" y2="{cy}" stroke="#222" stroke-width="0.5"/>',
             f'<circle cx="{cx}" cy="{cy}" r="{s}" fill="none" stroke="#333" stroke-width="0.5"/>']
    for x,y in original:
        lines.append(f'<circle cx="{cx + x*s:.1f}" cy="{cy - y*s:.1f}" r="1.2" fill="#4a9eff" opacity="0.25"/>')
    for x,y in generated:
        lines.append(f'<circle cx="{cx + x*s:.1f}" cy="{cy - y*s:.1f}" r="2.5" fill="#ff6b6b" opacity="0.8"/>')
    lines.append('</svg>')
    with open(fname,'w') as f:
        f.write('\n'.join(lines))
    print(f"\nwrote {fname}")

write_svg(data, samples)

