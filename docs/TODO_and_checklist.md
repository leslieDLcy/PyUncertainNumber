- [ ] MongoDB database 
- [ ] installation tips in readme 
- [ ] reorganise codes into UC subpackage


### refactor existing `pba` package


1. revise the `__repr__` of pba object
e.g. "Pbox: ~ norm(range=[(np.float64(-3.719), np.float64(3.719)), mean=1.0822, var=1.0822)"



2. for dist object, use 'gaussian' does not work
e.g. 'distribution_initialisation=['normal', (0,1)]' does not work
