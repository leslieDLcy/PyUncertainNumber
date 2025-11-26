import numpy
import scipy.stats as stats
import pickle
from pathlib import Path

def banana_data(n=100,d=2,seed:int=None, dist='normal'):
    def bake_banana_data(x:numpy.ndarray):
        y = numpy.asarray(x,dtype=float)
        Y1,Y2 = y[:,0]*4, y[:,0]**2+y[:,1]
        y[:,0], y[:,1] = Y1, Y2
        return y
    if dist == 'normal': dist = stats.norm(loc=0,scale=2)
    else: dist = stats.norm(loc=0,scale=2) # todo: add more distributions
    if seed is not None: x_input = dist.rvs(size=(n,d),seed=seed)  # x_input = 2*numpy.random.randn(N,2,)
    else: x_input = x_input = dist.rvs(size=(n,d))
    return bake_banana_data(x_input)

def pickle_dump(x:numpy.ndarray,filename:str=None,ext='.pickle',ordinal=True):
    dir = './data/'
    Path(dir).mkdir(parents=True,exist_ok=True)
    if filename is None: 
        if ordinal: filename = 'banana_'+str(get_next_integer(dir))
        else: filename = 'banana_'+str(numpy.random.randint(10_000))
    pickle.dump(x,open(dir+filename+ext,'wb'))
    pass

def get_next_integer(subdirectory: str):
    def next_integer(x):
        if len(x)==0: return 1
        else: return max(x)+1
    pathlist = Path(subdirectory).glob('*.pickle')
    ordinal = []
    for file in pathlist:
        index = pull_first_integer(str(file).split('_'))
        ordinal.append(int(index))
    return next_integer(ordinal)

def pull_first_integer(l:list):
    for s in l:
        s_=s.split('.')
        try: return int(s_[0])
        except: pass


def pickle_load(filename:str,ext='.pickle'):
    dir = './data/'
    Path(dir).mkdir(parents=True,exist_ok=True)
    return pickle.load(open(dir+filename+ext,'rb'))

def banana_model(x:numpy.ndarray):
    x = numpy.asarray(x,dtype=float)
    y = numpy.empty(x.shape,dtype=float)
    Y1,Y2 = x[:,0]*4, x[:,0]**2+x[:,1]
    y[:,0], y[:,1] = Y1, Y2
    return y


if __name__=='__main__':
    x = banana_data()
    # pickle_dump(x)