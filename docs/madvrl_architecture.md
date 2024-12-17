# MA-DVRL Architecture

## Network Components

```
                                                ┌─────────────────┐
                                                │  Value Network  │
                                                └────────┬────────┘
                                                         │
┌─────────────┐     ┌─────────────┐     ┌──────────────▼─────────────┐
│ Observation ├────►│   Encoder   ├────►│        Belief State        │
└─────────────┘     │   Network   │     │   p(z_t|o_t) ~ N(μ,σ)     │
                    └─────────────┘     └──────────────┬─────────────┘
                                                       │
┌─────────────┐     ┌─────────────┐     ┌─────────────▼─────────────┐
│  History    ├────►│  Inference  ├────►│     Policy Network        │
└─────────────┘     │   Network   │     │   π(a|z) = softmax(...)   │
                    └─────────────┘     └───────────────────────────┘
```

## Mathematical Framework

### Prior (Encoder)
```
p(z_t|o_t) = N(μ_φ(o_t), σ_φ(o_t))
```

### Posterior (Inference)
```
q(z_t|o_{1:t}, a_{1:t-1}) = N(μ_θ(h_t), σ_θ(h_t))
```

### ELBO Loss
```
L_ELBO = E_q(z)[log p(o|z)] - D_KL(q(z|o,a)||p(z|o))
```

### Policy Loss
```
L_policy = E_π_ψ[Σ_t r_t] + αH(π_ψ)
```

### Value Loss
```
L_value = E[(V_ω(s_t) - (r_t + γV_ω(s_{t+1})))²]
```

## Network Architectures

### Encoder Network
```
Input (obs_dim)
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(256) → ReLU
  ├────────────┐
  ▼            ▼
FC(256) → μ   FC(256) → σ
```

### Inference Network
```
Input (hist_dim)
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(256) → ReLU
  ├────────────┐
  ▼            ▼
FC(256) → μ   FC(256) → σ
```

### Policy Network
```
Input (512)
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(4) → Softmax
```

### Value Network
```
Input (512)
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(256) → ReLU
  │
  ▼
FC(1)
```
