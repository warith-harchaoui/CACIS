# Examples

This directory demonstrates common usage patterns for `CACISLoss`.

## Basic Usage

```python
import torch
import cacis.nn as nn

# 1. Initialize Loss (Scenario C: Default fallback)
criterion = nn.CACISLoss(alpha=2.0)

# Training loop
logits = model(x)
loss = criterion(logits, target) # Falls back to 0-1 cost if none provided
```

## Global Cost Matrix (Scenario A)

When the cost matrix is fixed for all samples (e.g., medical diagnosis costs):

```python
# Define typical costs (KxK)
C_global = torch.tensor([
    [0.0, 1.0, 5.0],
    [2.0, 0.0, 2.0],
    [10.0, 1.0, 0.0]
])

criterion = nn.CACISLoss(cost_matrix=C_global, alpha=2.0)

# Training
loss = criterion(logits, target) # Uses C_global broadcasting
```

## Per-Example Costs (Scenario B)

When costs vary per instance (e.g., dynamic pricing, context-aware risk):

```python
criterion = nn.CACISLoss(alpha=2.0)

# Inside loop
batch_costs = get_costs(batch) # (B, K, K)
loss = criterion(logits, target, cost_matrix=batch_costs)
```
