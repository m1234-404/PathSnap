import numpy as np

def viterbi(emission: np.ndarray, transition: np.ndarray, log_floor: float=1e-12) -> list[int]:
    """Decode states for T observations.

    emission: (T,S) probabilities; transition: (S,S) probabilities where
    transition[i,j] = P(state_t=j | state_(t-1)=i).
    TODO: member 3 should define how candidate sets vary by observation and how
    impossible/disconnected transitions are represented. This dense baseline
    assumes a fixed state set and row-normalised transition matrix.
    """
    if emission.ndim!=2 or transition.shape!=(emission.shape[1],emission.shape[1]):
        raise ValueError("incompatible emission/transition dimensions")
    loge=np.log(np.clip(emission,log_floor,1.0))
    logt=np.log(np.clip(transition,log_floor,1.0))
    T,S=loge.shape
    dp=np.full((T,S),-np.inf); back=np.zeros((T,S),dtype=int)
    dp[0]=loge[0]-np.log(S)
    for t in range(1,T):
        scores=dp[t-1][:,None]+logt
        back[t]=np.argmax(scores,axis=0)
        dp[t]=scores[back[t],np.arange(S)]+loge[t]
    states=[int(np.argmax(dp[-1]))]
    for t in range(T-1,0,-1): states.append(int(back[t,states[-1]]))
    return states[::-1]
