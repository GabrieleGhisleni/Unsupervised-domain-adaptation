# https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function
# https://discuss.pytorch.org/t/gradient-reversal-domain-adapation-not-converging-always/11327/2
# https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
# https://github.com/janfreyberg/pytorch-revgrad

class GradientReversalF(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, alpha):
    "let the input unchaged"
    ctx.save_for_backward(alpha)
    return x

  @staticmethod
  def backward(ctx, grad_output):
    "reverse the gradient by multipling -alpha"  
    alpha = ctx.saved_tensors[0]
    if ctx.needs_input_grad[0]:
      grad_output = (grad_output * (-alpha))
    return (grad_output, None)


class GradientReverse(nn.Module):
  def __init__(self, alpha, *args, **kwargs):
    "Reverse GR layer hook"
    super().__init__(*args, **kwargs)
    self.alpha = torch.tensor(alpha, requires_grad=False)
    assert alpha > 0, 'alpha must be > 0'
    print(f"The gradient will be multiplied by: {-alpha}")

  def forward(self, x):
    return GradientReversalF.apply(x, self.alpha)
