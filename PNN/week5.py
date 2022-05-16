import numpy as np

def heaviside(input, H_0, threshold):
    result = 0.0
    x = round(input - threshold, 5 )
    if (x < (10 ** (-9))) and (x > (10 ** (-9) * (-1))):
        result = H_0
    elif x > 0.0:
        result = 1.0
    else:
        result = 0.0

    return result

def relu(input):
    if (input < (10 ** (-9))) and (input > (10 ** (-9) * (-1))):
        return input
    elif input > 0.0:
        return input
    else:
        return 0.0

def lrelu(input,alpha):
    if (input < (10 ** (-9))) and (input > (10 ** (-9) * (-1))):
        return input
    elif input > 0.0:
        return input
    else:
        return alpha * input

def activation_function(input, alpha=None, H_0=None, threshold=None, mode=None):
    heaviside_f = np.vectorize(heaviside)
    relu_f = np.vectorize(relu)
    lrelu_f = np.vectorize(lrelu)
    if mode == "relu":
        return relu_f(input)
    elif mode == 'lrelu':
        return lrelu_f(input,alpha)
    elif mode == 'tanh':
        return np.tanh(input)
    elif mode == "heaviside":
        return heaviside_f(input, H_0, threshold)

# def get_dilation(img, dilation=None):
#     result=[]
#     rows = img.shape[0]
#     cols = img.shape[1]
#     cnt=0
#     for i in range(rows):
#         if ((i + 1) % dilation) == 1:
#             result.append([])
#             cnt+=1
#         for j in range(cols):
#             if (((i+1) % dilation) == 1) and (((j+1) % dilation) == 1):
#                 result[cnt-1].append(img[i, j])
#     return np.asarray(result)

def mask_dilation(mask, dilation=None):
    if dilation == 1:
        return mask
    result = []
    rows = (mask.shape[0] - 1) * dilation + 1
    cols = (mask.shape[1] - 1) * dilation + 1
    cnt = 0
    mask_idx = 0
    for i in range(rows):
        if ((i + 1) % dilation) == 1:
            result.append([])
            cnt+=1
            temp = mask[mask_idx].copy()
            mask_idx += 1
            temp_idx = 0
            for j in range(cols):
                if (((i + 1) % dilation) == 1) and (((j + 1) % dilation) == 1):
                    result[cnt - 1].append(temp[temp_idx])
                    temp_idx += 1
                else:
                    result[cnt - 1].append(0.0)
        else:
            result.append([0.0] * cols)
            cnt += 1
    return np.asarray(result)
def get_pooling(img, pool_size= 2, stride= 2, padding = None):
    # To store individual pools
    img = np.pad(img, padding, mode='constant')
    pools = []
    print(f"Image is: \n {img}")
    # Iterate over all row blocks (single block has `stride` rows)
    for i in np.arange(img.shape[0], step=stride):
        # Iterate over all column blocks (single block has `stride` columns)
        for j in np.arange(img.shape[0], step=stride):

            # Extract the current pool
            mat = img[i:i + pool_size, j:j + pool_size]

            # Make sure it's rectangular - has the shape identical to the pool size
            if mat.shape == (pool_size, pool_size):
                # Append to the list of pools
                pools.append(mat)

    # Return all pools as a Numpy array
    return np.array(pools)

def average_pooling(pools= None):
    num_pools = pools.shape[0]
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
    pooled = []

    for pool in pools:
        pooled.append(np.mean(pool))

    # Reshape to target shape
    return np.array(pooled).reshape(tgt_shape)


def max_pooling(pools=None):
    # Total number of pools
    num_pools = pools.shape[0]
    tgt_shape = (int(np.sqrt(num_pools)), int(np.sqrt(num_pools)))
    pooled = []

    for pool in pools:
        pooled.append(np.max(pool))

    # Reshape to target shape
    return np.array(pooled).reshape(tgt_shape)

def mask_convolution(img=None, mask=None, pools=None, stride=None, padding=None):
    """
    batch normalization for output neurons(tutarial Q5)
    """
    # Total number of pools
    num_pools = pools.shape[0]
    tgt_shape = (int(1+(img.shape[0]-mask.shape[0]+2*padding)/stride), int(1+(img.shape[1]-mask.shape[1]+2*padding)/stride))
    pooled = []

    for pool in pools:
        pooled.append(sum(sum(np.multiply(pool,mask))))

    # Reshape to target shape
    return np.array(pooled).reshape(tgt_shape)

def batch_normalization(input, beta=None, gamma=None, eta=None):
    Ex = np.mean(input, axis=0)
    Varx = np.var(input, axis=0)
    BN = beta + gamma * ((input-Ex)/np.sqrt(Varx+eta))
    return BN

def calculate_outputDimension(input_dimension, mask_dimension, pooling, stride, padding):
    """

    :param input_dimension: an array in the form of [height, width]
    :param mask_dimension:
    :param stride:
    :param padding:
    :return:
    """
    if pooling == None:
        if mask_dimension[0] == 1 and mask_dimension[1] == 1:
            print("1x1 convolution does not change dimension")
            output = [input_dimension[0], input_dimension[1], mask_dimension[3]]
        else:
            output_dim_height = 1+ ((input_dimension[0] - mask_dimension[0] + 2 * padding)/stride)
            output_dim_weight = 1+ ((input_dimension[1] - mask_dimension[1] + 2 * padding)/stride)
            output_channel = mask_dimension[3]
            output = [output_dim_height, output_dim_weight, output_channel]
    else:
        output_dim_height = 1 + ((input_dimension[0] - pooling) / stride)
        output_dim_weight = 1 + ((input_dimension[1] - pooling) / stride)
        output_channel = mask_dimension[3]
        output = [output_dim_height, output_dim_weight, output_channel]


    return output

'''
batch normalization for output neurons(tutarial Q5)
input1 = np.array([
    [1, 0.5,  0.2],
    [ -1, -0.5,  -0.2],
    [0.1, -0.1,  0],
])

input2 = np.array([
    [1, -1,  0.1],
    [0.5, -0.5,  -0.1],
    [0.2, -0.2,  0],
])

input3 = np.array([
    [0.5, -0.5,  -0.1],
    [ 0, -0.4,  0],
    [0.5, 0.5,  0.2],
])

input4 = np.array([
    [0.2, 1,  -0.2],
    [-1, -0.6,  -0.1],
    [0.1, 0,  0.1],
])

final_input_array = np.stack([input1, input2, input3, input4])
# print(final_input_array)
print(batch_normalization(final_input_array, beta=0, gamma=1, eta=0.1))
'''

'''
multiple channels and each channel corresponds to a mask
padding=0, stride=1, dilation=2:

conv_input1 = np.array([
    [0.2, 1,  0],
    [ -1, 0,  -0.1],
    [0.1, 0,  0.1]
])
conv_input2 = np.array([
    [1, 0.5,  0.2],
    [ -1, -0.5,  -0.2],
    [0.1, -0.1,  0]
])
H1 = np.array([
    [1,  -0.1],
    [1,  -0.1]
])
H2 = np.array([
    [0.5,  0.5],
    [-0.5, -0.5]
])
stride=1
padding=0

H1_after_dilation1 = mask_dilation(H1, dilation=2)
H2_after_dilation2 = mask_dilation(H2, dilation=2)

pool_result1 = get_pooling(img=conv_input1, pool_size= H1_after_dilation1.shape[0], stride=stride, padding=padding)
pool_result2 = get_pooling(img=conv_input2, pool_size= H2_after_dilation2.shape[0], stride=stride, padding=padding)

print(mask_convolution(img=conv_input1, mask = H1_after_dilation1, pools=pool_result1, stride=stride, padding=padding)+mask_convolution(img=conv_input2, mask = H2_after_dilation2, pools=pool_result2, stride=stride, padding=padding))

'''

'''
for pooling layer of a CNN
conv_input = np.array([
    [0.2, 1,  0,  0.4],
    [ -1, 0,  -0.1,  -0.1],
    [0.1, 0,  -1,  -0.5],
    [ 0.4, -0.7,  -0.5,  1]
])

pool_result = get_pooling(img=conv_input, pool_size=2, stride=2)
print(average_pooling(pools=pool_result))

pool_result = get_pooling(img=conv_input, pool_size= 2, stride=2)
print(pool_result)
print(max_pooling(pools=pool_result))
'''

'''
for activation_function
input = np.asarray([
    [1, 0.5, 0.2],
    [-1, -0.5, -0.2],
    [0.1, -0.1, 0]
], dtype=np.float32)
alpha = 0.1
threshold = 0.1
H_0 = 0.5

print(activation_function(input, mode="relu"))
print(activation_function(input, alpha=0.1, mode="lrelu"))
print(activation_function(input, mode="tanh"))
print(activation_function(input, H_0=0.5, threshold=0.1, mode="heaviside"))
'''
conv_input1 = np.array([
    [0.2, 1,  0],
    [ -1, 0,  -0.1],
    [0.1, 0,  0.1]
])
conv_input2 = np.array([
    [1, 0.5,  0.2],
    [ -1, -0.5,  -0.2],
    [0.1, -0.1,  0]
])
H1 = np.array([
    [1,  -0.1],
    [1,  -0.1]
])
H2 = np.array([
    [0.5,  0.5],
    [-0.5, -0.5]
])
stride=1
padding=0

H1_after_dilation1 = mask_dilation(H1, dilation=2)
H2_after_dilation2 = mask_dilation(H2, dilation=2)

pool_result1 = get_pooling(img=conv_input1, pool_size= H1_after_dilation1.shape[0], stride=stride, padding=padding)
pool_result2 = get_pooling(img=conv_input2, pool_size= H2_after_dilation2.shape[0], stride=stride, padding=padding)

print(mask_convolution(img=conv_input1, mask = H1_after_dilation1, pools=pool_result1, stride=stride, padding=padding)+mask_convolution(img=conv_input2, mask = H2_after_dilation2, pools=pool_result2, stride=stride, padding=padding))



