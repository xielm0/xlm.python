# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------
# Author:      xlm
#
# Created:     01/08/2014
# Copyright:   (c) xlm 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np

def log_likelihood(x_mat,y_vec,y_max,fvec):
        """
                 
        """
         
        log_likelihood = 0.0
        data_size = len(x_mat)  
        for i in range(data_size):
                x_vec = x_mat[i]
                y = y_vec[i]
                log_likelihood += math.log(max_ent_predict_unnormalized(x_vec,y,fvec))
                log_likelihood -= math.log(max_ent_normalizer(x_vec,y_max,fvec))
        log_likelihood /= data_size
        return log_likelihood
        
        