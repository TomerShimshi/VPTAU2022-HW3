import json
import os
from turtle import shape
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# change IDs to your IDs.
ID1 = "203200480"
ID2 = "987654321"

ID = "HW3_{0}_{1}".format(ID1, ID2)
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
IMAGE_DIR_PATH = "Images"

# SET NUMBER OF PARTICLES
N = 100

# Initial Settings
s_initial = [297,    # x center
             139,    # y center
              16,    # half width
              43,    # half height
               0,    # velocity x
               0]    # velocity y


sigma_x =5
sigma_y =1.5
sigma_vx =0
sigma_vy =0

def predict_particles(s_prior: np.ndarray) -> np.ndarray:
    """Progress the prior state with time and add noise.

    Note that we explicitly did not tell you how to add the noise.
    We allow additional manipulations to the state if you think these are necessary.

    Args:
        s_prior: np.ndarray. The prior state.
    Return:
        state_drifted: np.ndarray. The prior state after drift (applying the motion model) and adding the noise.
    """
    s_prior = s_prior.astype(float)
    state_drifted = s_prior
    """ DELETE THE LINE ABOVE AND:
    INSERT YOUR CODE HERE."""
    #first we applay the motion
    state_drifted[:2, :] = state_drifted[:2, :] + state_drifted[4:, :]

    # now we add the noise
    state_drifted[:1, : ] = state_drifted[:1, : ] + np.round(np.random.normal(0, sigma_x, size=(1, 100)))
    state_drifted[1:2, :] = state_drifted[1:2, :] + np.round(np.random.normal(0, sigma_y, size=(1, 100)))
    #not sure abot this
    state_drifted[4:5, :] =  state_drifted[4:5, :]+  np.round(np.random.normal(0, sigma_vx, size=(1, 100))) #
    state_drifted[5:6, :] = state_drifted[5:6, :] +  np.round(np.random.normal(0, sigma_vy, size=(1, 100))) #
    state_drifted = state_drifted.astype(int)
    return state_drifted


def compute_normalized_histogram(image: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Compute the normalized histogram using the state parameters.

    Args:
        image: np.ndarray. The image we want to crop the rectangle from.
        state: np.ndarray. State candidate.

    Return:
        hist: np.ndarray. histogram of quantized colors.
    """
    state = np.floor(state)
    state = state.astype(int)
    
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    
    hist = np.zeros((16, 16 , 16))
    max_higt,max_width,_ = image.shape

    x,y,hight,widthe =state[0],state[1],state[2],state[3]
    croped_image = image[ max(y- hight,1):min( y + hight,max_higt-1), max(x- widthe,1): min(x + widthe, max_width-1 )]
    # Using cv2.split() to split channels of coloured image 
    b,g,r = cv2.split(croped_image)
    b = b//16
    g = g//16
    r = r//16
    temp_range = [i for i in range(16)]
    temp_hist_b = np.histogram(b,temp_range)
    temp_hist_r= np.histogram(r,temp_range) 
    temp_hist_g= np.histogram(g,temp_range) 
    temp = croped_image.shape
    for i in range(temp[0] ):
        for j in range (temp[1]):
            temp_b = b[i,j]
            temp_r = r[i,j]
            temp_g = g[i,j]
            hist[b[i,j]][g[i,j]][r[i,j]] +=1
    



    hist = np.reshape(hist, 16 * 16 * 16)  #hist.flatten() #

    # normalize
    hist = hist/np.sum(hist)

    return hist
    '''
    x, y, width, height, x_vel, y_vel = s_initial

    temp_image = image[y - height:y + height, x - width:x + width]

    b, g, r = cv2.split(temp_image)

    b //= 16
    g //= 16
    r //= 16

    histogram = np.zeros((16, 16, 16))

    for i in range(len(temp_image)):
        for j in range(len(temp_image[0])):
            histogram[b[i, j]][g[i, j]][r[i, j]] += 1

    histogram = histogram.reshape((4096, 1))
    histogram /= np.sum(histogram)

    return histogram
    '''

def sample_particles(previous_state: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    """Sample particles from the previous state according to the cdf.

    If additional processing to the returned state is needed - feel free to do it.

    Args:
        previous_state: np.ndarray. previous state, shape: (6, N)
        cdf: np.ndarray. cummulative distribution function: (N, )

    Return:
        s_next: np.ndarray. Sampled particles. shape: (6, N)
    """
    
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    S_next = []
    for i in range(len(cdf)):
        r = np.random.uniform(0,1)
        k = np.argmax(cdf>=r)
        temp = previous_state[:,k]
        S_next.append(previous_state[:,k])
    return np.array(S_next).T


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Calculate Bhattacharyya Distance between two histograms p and q.

    Args:
        p: np.ndarray. first histogram.
        q: np.ndarray. second histogram.

    Return:
        distance: float. The Bhattacharyya Distance.
    """
    distance = 0
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    for i in range(len(p)):
        distance+=np.sqrt(p[i]*q[i])
    distance = np.exp(20*distance)

    return distance #np.exp(20 * np.sum(np.sqrt(p * q)))


def show_particles(image: np.ndarray, state: np.ndarray, W: np.ndarray, frame_index: int, ID: str,
                  frame_index_to_mean_state: dict, frame_index_to_max_state: dict,
                  ) -> tuple:
    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " + str(frame_index))
    plt.show(block=False)

    # Avg particle box
   
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    (x_avg, y_avg, w_avg, h_avg) = (0, 0, state[2][0]*2, state[3][0]*2)
    for index,partical in enumerate(state.T):
        '''
        DLETE THIS
        '''
        shuff = W[index]
        temp_x =partical[0] - w_avg/2
        temp_y = partical[1] - h_avg/2
        rect = patches.Rectangle((temp_x, temp_y), w_avg, h_avg, linewidth=1, edgecolor='y', facecolor='none')
        ax.add_patch(rect)
        x_avg += partical[0]* W[index]
        y_avg += partical[1]* W[index]
    x_avg = x_avg - w_avg/2
    y_avg -= h_avg/2
    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)

    # calculate Max particle box
    
    """ DELETE THE LINE ABOVE AND:
        INSERT YOUR CODE HERE."""
    (x_max, y_max, w_max, h_max) = (0, 0, state[2][0]*2, state[3][0]*2)
    x_max , y_max,_,_,_,_= state.T[np.argmax(W)]
    x_max = x_max - w_max/2
    y_max -= h_max/2
    rect = patches.Rectangle((x_max, y_max), w_max, h_max, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)

    fig.savefig(os.path.join(RESULTS, ID + "-" + str(frame_index) + ".png"))
    frame_index_to_mean_state[frame_index] = [float(x) for x in [x_avg, y_avg, w_avg, h_avg]]
    frame_index_to_max_state[frame_index] = [float(x) for x in [x_max, y_max, w_max, h_max]]
    return frame_index_to_mean_state, frame_index_to_max_state

def calculate_W (image,S,q):
    '''
    this method recives the image, the state, the histogram of the firs frame
    and calculates the whigets accourdingly
    '''
    W = []
    for col in S.T:
        partical_hist = compute_normalized_histogram(image=image, state= col)
        temp = bhattacharyya_distance(p= partical_hist,q= q)
        W.append(bhattacharyya_distance(p= partical_hist,q= q))
    W = np.array(W)
    W = W/ np.sum(W)
    return W

def calculate_C (W):
    '''
    this function recives the whigthes and calculTE THE CDF
    ''' 
    #c = np.zeros(len(W))
    c = [0 for i in range(len(W))]
    c[0]=W[0]
    for i in range(1,len(W)):
        c[i]+= W[i]+c[i-1]
    return np.array(c)
 

def main():
    state_at_first_frame = np.matlib.repmat(s_initial, N, 1).T
    S = predict_particles(state_at_first_frame)

    # LOAD FIRST IMAGE
    image = cv2.imread(os.path.join(IMAGE_DIR_PATH, "001.png"))

    '''
    %%%%%%%%%%%% delete THIS####################
    

    fig, ax = plt.subplots(1)
    image = image[:,:,::-1]
    plt.imshow(image)
    plt.title(ID + " - Frame mumber = " )
    x_avg, y_avg, w_avg, h_avg = s_initial[0],s_initial[1],s_initial[2]*2,s_initial[3]*2
    x_avg = x_avg - w_avg/2
    y_avg = y_avg - h_avg/2

    rect = patches.Rectangle((x_avg, y_avg), w_avg, h_avg, linewidth=1, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    plt.show(block=False)
    '''
    ###########################


    # COMPUTE NORMALIZED HISTOGRAM
    q = compute_normalized_histogram(image, s_initial)

    # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
    # YOU NEED TO FILL THIS PART WITH CODE:
    """INSERT YOUR CODE HERE."""
    W = calculate_W(image=image,S=S,q=q)
    C=calculate_C(W)
    images_processed = 1

    # MAIN TRACKING LOOP
    image_name_list = os.listdir(IMAGE_DIR_PATH)
    image_name_list.sort()
    frame_index_to_avg_state = {}
    frame_index_to_max_state = {}
    for image_name in image_name_list[1:]:

        S_prev = S

        # LOAD NEW IMAGE FRAME
        image_path = os.path.join(IMAGE_DIR_PATH, image_name)
        current_image = cv2.imread(image_path)

        # SAMPLE THE CURRENT PARTICLE FILTERS
        S_next_tag = sample_particles(S_prev, C)

        # PREDICT THE NEXT PARTICLE FILTERS (YOU MAY ADD NOISE
        S = predict_particles(S_next_tag)

        # COMPUTE NORMALIZED WEIGHTS (W) AND PREDICTOR CDFS (C)
        # YOU NEED TO FILL THIS PART WITH CODE:
        """INSERT YOUR CODE HERE."""
        W = calculate_W(image=current_image,S=S,q=q)
        C=calculate_C(W)

        # CREATE DETECTOR PLOTS
        images_processed += 1
        #$$$$ DELETE THIS CHANGE%%%%%%%%########
        #if 0 == images_processed%10:
        if 0 == images_processed%10:
            frame_index_to_avg_state, frame_index_to_max_state = show_particles(
                current_image, S, W, images_processed, ID, frame_index_to_avg_state, frame_index_to_max_state)

    with open(os.path.join(RESULTS, 'frame_index_to_avg_state.json'), 'w') as f:
        json.dump(frame_index_to_avg_state, f, indent=4)
    with open(os.path.join(RESULTS, 'frame_index_to_max_state.json'), 'w') as f:
        json.dump(frame_index_to_max_state, f, indent=4)


if __name__ == "__main__":
    main()
