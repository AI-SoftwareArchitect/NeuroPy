class BaseModel:
    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def zero_grad(self):
        raise NotImplementedError
