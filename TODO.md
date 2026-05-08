remove shape bouncers using einops and move to asserts
remove shape manipulation in functions and move it to the imperative shells, functions should instead assert strict contracts. what shape should buffer keys have? like should rewards be B, 1 or B etc 

implement models and functions from https://coax.readthedocs.io/en/latest/index.html
look into possible coax like notation.

implement models and functions from RLAX 

add ability to not normalize advantages (how should this be done). should normalization be its own function and removed from advantages functions? or should a normalize flag be passed into advantages functions? (not normalization of advantages is dependant on the method use, ie EMA vs mean)