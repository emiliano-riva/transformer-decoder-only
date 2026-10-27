# Position-wise feed-forward network

After self-attention, the FFN transforms each token representation independently with shared parameters:

$$
\operatorname{FFN}(x)=W_2\,\operatorname{GELU}(W_1x+b_1)+b_2.
$$

For our tiny model, the feature dimension grows from $4\rightarrow16\rightarrow4$. Attention mixes information **between token positions**; the FFN performs a nonlinear transformation **within each token's feature vector**.

The code includes dropout after GELU and after the second linear layer, but this educational run sets `dropout=0.0`. GELU remains active during both training and evaluation; Dropout is normally deactivated in evaluation mode.

```python
self.ffn = nn.Sequential(
    nn.Linear(4, 16), nn.GELU(), nn.Dropout(0.0),
    nn.Linear(16, 4), nn.Dropout(0.0),
)
```

Next: [Output layer](08-output-layer.md).
