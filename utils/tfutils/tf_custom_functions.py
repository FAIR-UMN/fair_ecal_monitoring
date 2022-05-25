import tensorflow.keras.backend as kb

#---------------------------------------------------------------------
#                         Custom loss functions
#---------------------------------------------------------------------
def custom_log_error(y_true, y_pred):
    arr = kb.abs(y_true-y_pred)
    arr = kb.abs(kb.log(arr+1e-10))*(arr<=1.0) + (1e10+kb.abs(arr))*(arr>1.0)
    loss = kb.sum(arr)
    return loss

#---------------------------------------------------------------------
#                         Custom schedulers
#---------------------------------------------------------------------
