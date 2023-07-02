import numpy
from itertools import product
from matplotlib import pyplot
from scipy.integrate import odeint

'''Luttinger-Liquid Model (creat the Hamiltonian)'''
class LLModel():
    def __init__(self, Kinv, vecs):
        self.Kinv = Kinv
        self.vecs = vecs
        self.reset()
          
    def reset(self):
        self._ope = None
        self._spin = None
        self._N = self.Kinv.shape[-1] # number of d.o.f
        self._n = self.vecs.shape[0]  # number of charge vectors
        
    @property
    def ope(self): # calculate OPE coefficient
        if self._ope is None:
            n = self._n
            C = numpy.empty((n, n, n))
            for r, p, q in product(range(n), range(n), range(n)):
                P = self.vecs[p]
                Q = self.vecs[q]
                R = self.vecs[r]
                if (P + Q is R) or (P - Q is R) or (- P + Q is R) or (- P - Q is R):
                    C[r, p, q] = 1/2
            self._ope = C
        return self._ope
        
    def stats(self, vec1, vec2): # calculate comformal spin statistics
        return 0.5 * numpy.dot(numpy.dot(self.vecs[vec1], self.Kinv), self.vecs[vec2])
    
    @property
    def spin(self):
        if self._spin is None:
            n = self._n
            s = numpy.empty((n, n, n))
            for r, p, q in product(range(n), range(n), range(n)):
                if self.stats(p, p) + self.stats(q, q) == self.stats(r, r):
                    s[r, p, q] = 1  
            self._spin = s
        return self._spin

'''Solve perturbative RG set base on initial conditions'''
class flow():
        def __init__(self, model):
            self.model = model
            self.reset()
            
        def reset(self):
            self._U = None
            self._couplings = None           
            self._t = None
            self._resolution = None
            self._sol = None
            
        @property
        def U(self):
            if self._U is None:
                self._U = numpy.identity(self.model._N, dtype = float) # default U-matrix is identity matrix
            return self._U
    
        @property
        def couplings(self):
            if self._couplings is None:
                self._couplings = 0.2 * numpy.ones(self.model._N, dtype = float) # default i.c. for couplings is 0.2
            return self._couplings
        
        @property
        def t(self): # maxinum RG time
            if self._t is None:
                self._t = 1.0
            return self._t
        
        @property
        def resolution(self):
            if self._resolution is None:
                self._resolution = 100
            return self._resolution
        
        @property
        def g0(self): # input initial conditions
            d = []
            d = numpy.append(d, self.couplings)
            d = numpy.append(d, self.U)
            return d
                      
        @property
        def sol(self): # slove ODE set
            if self._sol is None:
                def RG(g, t):
                    n = self.model.vecs.shape[0] #number of charge vectors
                    N = self.model.vecs.shape[-1] #number of d.o.f

                    def commute(A ,B):
                        return numpy.dot(A,B) - numpy.dot(B,A)

                    y = g[:n] # dy/dt
                    U = g[n:].reshape((N, N)) # dU/dt
                    
                    def scald(U, vec):
                        return 0.5 * numpy.dot(numpy.dot(self.model.vecs[vec], numpy.linalg.inv(U)), self.model.vecs[vec])
                    
                    eq = numpy.empty(n)
                    for r, p, q in product(range(n), range(n), range(n)):
                        eq[r] = (2 - scald(U, r)) * y[r] - self.model.spin[p, q, r] * self.model.ope[p, q, r] * y[p] * y[q] # ODE of couplings
                    Mat1 = numpy.dot(self.model.Kinv, U)
                    Mat2 = sum((y[r] ** 2) * numpy.dot(self.model.Kinv, numpy.outer(self.model.vecs[r], self.model.vecs[r])) for r in range(n))
                    dU = commute(Mat1, Mat2) # ODE of the U-matrix
                    eq = numpy.append(eq, 0.5 * numpy.dot(U, dU))
                    return eq
                RGtime = numpy.linspace(0, self.t, num = self.resolution)
                self._sol = odeint(RG, self.g0, RGtime)
            return self._sol
        
        def ploty(self, ylower = None, yupper = None, vec = None): # plot couplings flow
            RGtime = numpy.linspace(0, self.t, num = self.resolution)
            if vec is None:
                for i in range(self.model.vecs.shape[0]):
                    pyplot.plot(RGtime, abs(self.sol[:,i]), linewidth=2.0)
            else:
                    pyplot.plot(RGtime, abs(self.sol[:,vec]), linewidth=2.0)                    
            pyplot.xlabel("RG time")
            pyplot.ylabel("coupling")
            pyplot.yscale('log')
            if (ylower is None) or (yupper is None):
                return pyplot.plot()
            else:
                pyplot.ylim(ylower,yupper)
                return pyplot.plot()

            def plotscld