import os
import numpy as np
import random
import tensorflow as tf

#---------------------------------------------------------------------------------------------------
def reset_random_seed(random_seed):
   os.environ['PYTHONHASHrandom_seed']=str(random_seed)
   tf.random.set_seed(random_seed)
   np.random.seed(random_seed)
   random.seed(random_seed)
#---------------------------------------------------------------------------------------------------
