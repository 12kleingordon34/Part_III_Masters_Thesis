import numpy
import matplotlib.pyplot as plt

# Generate a correlated chain
x = 0.        # Start point
alpha = 0.9   # Autocorrelation
N = 2000      # Number of samples

chain = [x]
for _ in range(N):
    x = alpha * x + numpy.random.rand()
    chain.append(x)

plt.plot(chain)

# Trim burn in
chain = numpy.array(chain[int(len(chain)/2):])


def acor(chain):
    rhos = []
    c = chain - chain.mean()
    var = c.dot(c)

    for i in range(1,len(c)):
        rho = c[:-i].dot(c[i:])/var
        if rho < 0:
            break
        rhos.append(rho)

    return 1 + 2*sum(rhos)


print("Actual autocorrelation:    %f" % ((1+alpha)/(1-alpha)))
print("Predicted autocorrelation: %f" % acor(chain))
