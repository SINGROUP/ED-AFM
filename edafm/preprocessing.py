
import random
import numpy as np
from PIL import Image

def add_noise(Xs, c=0.1, randomize_amplitude=False):
    '''
    Add uniform random noise to arrays. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, ...).
        c: float. Amplitude of noise. Is multiplied by (max-min) of sample.
        randomize_amplitude: Boolean. If True, noise amplitude is uniform random in [0,c]
                             for each sample in the batch.
    '''
    for X in Xs:
        sh = X.shape
        R = np.random.rand(*sh) - 0.5
        if randomize_amplitude:
            amp = np.random.uniform(0.0, 1.0, sh[0]) * c
        else:
            amp = [c] * sh[0]
        for j in range(sh[0]):
            X[j] += R[j] * amp[j]*(X[j].max()-X[j].min())

def add_gradient(Xs, c=0.3):
    '''
    Add a constant gradient plane with random direction to arrays.
    In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, x, y, z).
        c: float. Maximum range of gradient plane as a fraction of the
           range of the array values.
    '''
    assert len(set([X.shape for X in Xs])) == 1 # All same shape
    x, y = np.meshgrid(np.arange(0, Xs[0].shape[1]), np.arange(0, Xs[0].shape[2]), indexing='ij')
    for i in range(Xs[0].shape[0]):
        c_eff = c*np.random.rand()
        angle = 2*np.pi*np.random.rand()
        n = [np.cos(angle), np.sin(angle), 1]
        z = -(n[0]*x + n[1]*y)
        z -= z.mean()
        z /= np.ptp(z)
        for X in Xs:
            X[i] += z[:, :, None]*c_eff*np.ptp(X[i])

def add_norm(Xs, per_layer=True):
    '''
    Normalize arrays by subracting the mean and dividing by standard deviation. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, ...).
        per_layer: Boolean. If True, normalized separately for each element in last axis of Xs.
    '''
    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            if per_layer:
                for i in range(sh[-1]):
                    X[j,...,i] = (X[j,...,i] - np.mean(X[j,...,i])) / np.std(X[j,...,i])
            else:
                X[j] = (X[j] - np.mean(X[j])) / np.std(X[j])

def rand_shift_xy(Xs, c=0.02):
    '''
    Randomly shift z layers in x and y. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, x_dim, y_dim, z_dim).
        c: float in [0,1]. Maximum fraction of image size by which to shift.
    '''
    for X in Xs:
        sh = X.shape
        max_y_shift = np.floor(sh[1]*c).astype(int)
        max_x_shift = np.floor(sh[2]*c).astype(int)
        for j in range(sh[0]):
            for i in range(sh[3]):
                a = X[j,:,:,i]
                rand_shift_y = random.choice(np.append(np.arange(-max_y_shift,0), np.arange(1,max_y_shift+1)))   
                rand_shift_x = random.choice(np.append(np.arange(-max_x_shift,0), np.arange(1,max_x_shift+1)))  
                shift_y = abs(rand_shift_y)            
                shift_x = abs(rand_shift_x)
                tmp = np.zeros((sh[1]+2*shift_y,sh[2]+2*shift_x))
                tmp[shift_y:-shift_y,shift_x:-shift_x] = a
                tmp[:shift_y,shift_x:-shift_x] = a[shift_y:0:-1,:]
                tmp[-shift_y:,shift_x:-shift_x] = a[-2:-2-shift_y:-1,:]
                tmp[:,-shift_x:] = tmp[:,-2-shift_x:-2-2*shift_x:-1]
                tmp[:,:shift_x] = tmp[:,2*shift_x:shift_x:-1]
                X[j,:,:,i] = tmp[shift_y-rand_shift_y:shift_y-rand_shift_y+sh[1],shift_x-rand_shift_x:shift_x-rand_shift_x+sh[2] ]  

def add_cutout(Xs, n_holes=5):
    '''
    Randomly add cutouts (square patches of zeros) to images. In-place operation.
    Arguments:
        Xs: list of np.ndarray of shape (batch_size, x_dim, y_dim, z_dim).
        n_holes: int. Maximum number of cutouts to add.
    '''
    
    def get_random_eraser(input_img,p=0.2, s_l=0.001, s_h=0.01, r_1=0.1, r_2=1./0.1, v_l=0, v_h=0):
        '''        
        p : the probability that random erasing is performed
        s_l, s_h : minimum / maximum proportion of erased area against input image
        r_1, r_2 : minimum / maximum aspect ratio of erased area
        v_l, v_h : minimum / maximum value for erased area
        '''

        sh = input_img.shape
        img_h, img_w = [sh[0], sh[1]] 
        
        if np.random.uniform(0, 1) > p:
            return input_img

        while True:
            
            s = np.exp(np.random.uniform(np.log(s_l), np.log(s_h))) * img_h * img_w
            r = np.exp(np.random.uniform(np.log(r_1), np.log(r_2)))
            
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        input_img[top:top + h, left:left + w] = 0.0

        return input_img

    for X in Xs:
        sh = X.shape
        for j in range(sh[0]):
            for i in range(sh[3]):
                for attempt in range(n_holes):
                    X[j,:,:,i] = get_random_eraser(X[j,:,:,i])

def interpolate_and_crop(X, real_dim, target_res=0.125, target_multiple=8):
    '''
    Interpolate batch of AFM images to target resolution and crop to a target
    multiple of pixels in xy plane.
    Arguments:
        X: list of np.ndarray of shape (batch, x, y, z). AFM images.
        real_dim: tuple of floats (len_x, len_y): Real-space size of AFM image region
                  in x- and y-directions in angstroms.
        target_res: float. Target size for a pixel in angstroms.
        target_multiple: int. Target multiple of pixels of output image.
    Returns: list of np.ndarray of shape (batch, x_out, y_out, z). Interpolated
             and cropped AFM images.
    '''

    resized_shape = (int(real_dim[1] // target_res), int(real_dim[0] // target_res))
    for k, x in enumerate(X):

        # Interpolate for correct resolution
        X_ = np.empty((x.shape[0], resized_shape[1], resized_shape[0], x.shape[-1]))
        for i in range(x.shape[0]):
            for j in range(x.shape[-1]):
                X_[i,:,:,j] = np.array(Image.fromarray(x[i,:,:,j]).resize(resized_shape, Image.BILINEAR))
        
        # Crop for correct multiple of pixels
        dx = resized_shape[1] % target_multiple
        x_start = int(np.floor(dx / 2))
        x_end = resized_shape[1] - int(np.ceil(dx / 2))

        dy = resized_shape[0] % target_multiple
        y_start = int(np.floor(dy / 2))
        y_end = resized_shape[0] - int(np.ceil(dy / 2))

        X[k] = X_[:, x_start:x_end, y_start:y_end]

    return X

def add_rotation_reflection(X, Y, rotations=True, reflections=True, multiple=2, crop=None):
    '''
    Randomly rotate images in a batch to a different random angle (0 - 359 deg) and reflect.
    Arguments:
        X: list of np.ndarray of shape (batch, x, y, z). AFM images.
        Y: list of np.ndarray of shape (batch, x, y). Reference image descriptors.
        rotations: bool. Whether to augment with rotations.
        reflections: bool. Whether to augment with reflections. If set True each rotation would be randomly reflected or not.
        multiple: int. multiplier, how many times the batch size will increase after rotation and reflection augmentation.
        crop: None or tuple (x_size, y_size). If not None, then output batch is cropped to
              specified size in the middle of the image.
    Returns:
        X: list of np.ndarray of shape (batch, x_new, y_new, z). Rotation augmented AFM images.
        Y: list of np.ndarray of shape (batch, x_new, y_new). Rotation augmented reference image descriptors.
    '''
 
    if rotations:

        X_aug = [[] for _ in range(len(X))]
        Y_aug = [[] for _ in range(len(Y))]

        for _ in range(multiple):
            rotation = np.random.randint(1,359)
            if reflections:
                flip = np.random.randint(2)
            for k, x in enumerate(X):
                x = x.copy()
                for i in range(x.shape[0]):
                    for j in range(x.shape[-1]):
                        x[i,:,:,j] = np.array(Image.fromarray(x[i,:,:,j]).rotate(rotation, resample=Image.BICUBIC))
                if flip:
                    x = x[:,:,::-1]            
                X_aug[k].append(x)
            for k, y in enumerate(Y):
                y = y.copy()
                for i in range(y.shape[0]):
                    y[i,:,:] = np.array(Image.fromarray(y[i,:,:]).rotate(rotation, resample=Image.BICUBIC))
                if flip:
                    y = y[:,:,::-1] 
                Y_aug[k].append(y)

    X = [np.concatenate(x, axis=0) for x in X_aug]
    Y = [np.concatenate(y, axis=0) for y in Y_aug]

    if crop is not None:
        x_start = (X[0].shape[1] - crop[0]) // 2
        y_start = (X[0].shape[2] - crop[1]) // 2
        X = [x[:, x_start:x_start+crop[0], y_start:y_start+crop[1]] for x in X]
        Y = [y[:, x_start:x_start+crop[0], y_start:y_start+crop[1]] for y in Y]

    return X, Y