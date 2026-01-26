import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def custom_clip(value, min_value, max_value):
    return max(min_value, min(value, max_value))

@jit(nopython=True)
def numba_bisect_left(a, x):
    lo = 0
    hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x: lo = mid + 1
        else: hi = mid
    return lo

@jit(nopython=True)
def mu(H_k, theta_t):
    return np.dot(H_k, theta_t[:-1])

@jit(nopython=True)
def lambda_function(theta_t):
    return theta_t[-1]

@jit(nopython=True)
def pdf(tau, u_k, theta_t, H_k, new_lam=None):
    mu_val = mu(H_k, theta_t)
    lambda_val = lambda_function(theta_t) if new_lam is None else new_lam
    diff = tau - u_k
    if diff <= 0:
        return 1e-10
    lambda_val = max(lambda_val, 1e-10)
    scaling = np.sqrt(lambda_val / (2 * np.pi * diff ** 3))
    exponent = -0.5 * lambda_val * (diff - mu_val) ** 2 / (diff * mu_val ** 2)
    exponent = custom_clip(exponent, -700, 700)
    result = scaling * np.exp(exponent)
    return max(result, 1e-30)


@jit(nopython=True)
def log_f(tau, u_k, theta_t, H_k, new_lam=None):
    return np.log(pdf(tau, u_k, theta_t, H_k, new_lam=new_lam))


@jit(nopython=True)
def local_log_likelihood(U, t,P, W, theta_t, alpha):

    N_t_W = numba_bisect_left(U, t - W) + 1
    N_t = numba_bisect_left(U, t)

    # Indices range
    indices = np.arange(N_t_W + P, N_t)

    if len(indices) == 0:
        return 0.0

    # Calculate H_k for all k in the range at once
    H_k = np.zeros((len(indices), P))
    for idx, k in enumerate(indices):
        H_k[idx, :] = U[k - P:k] - U[k - P - 1:k - 1]

    # Compute omega for all relevant k values at once
    omega = np.exp(-alpha * (t - U[indices + 1]))

    # Compute log_f for all k values at once
    log_f_values = np.zeros(len(indices))
    for idx, k in enumerate(indices):
        log_f_values[idx] = log_f(U[k + 1], U[k], theta_t, H_k[idx])

    # Compute the likelihood sum using vectorized operations
    likelihood_sum = np.sum(omega * log_f_values)

    return likelihood_sum


def estimate_theta(U, t,P, W, initial_theta, alpha):
    # Objective function to be minimized (negative log likelihood)
    def objective(theta):
        return -local_log_likelihood(U, t, P,W, theta, alpha)

    # Bounds for theta could be added if known, e.g., for physiological constraints
    result = minimize(objective, initial_theta, method='Nelder-Mead', options={'maxiter': 2000, 'maxfev': 5000})


    if result.success:
        return result.x  # Return the optimized parameters
    else:
        raise ValueError("Optimization failed:", result.message)

def extra_beat_detection(r_indices, i, H_k, p, P, W, estimated_theta, Q, eta_e, alpha, verbose=False):
    if i + Q + 1 < len(r_indices):
        p_re = log_f(r_indices[i], r_indices[i - 1], estimated_theta, H_k)
        pe = log_f(r_indices[i + 2], r_indices[i], estimated_theta, H_k)

        if verbose:
            print(f"i: {i}, pre: {p_re:.2f}, p: {p:.2f}, after: {pe:.2f}")

        if pe > p + eta_e:
            if verbose:
                print(f"Extra beat detected at index {i + 1}.")

            r_indices_new = np.delete(r_indices, i + 1)
            if verbose:
                print(f"length of array {len(r_indices_new)}.")
            accepted = evaluate_correction(r_indices, r_indices_new, i, P, Q, eta_e, estimated_theta)
            if verbose:
                print(f"Accepted")
            if accepted:
                r_indices = r_indices_new
                estimated_theta = estimate_theta(r_indices, r_indices[i], P, W, estimated_theta, alpha)

                if verbose:
                    print(f"Extra beat accepted at index {i + 1}.")

                return r_indices, estimated_theta
            else:
                return r_indices, estimated_theta
        else:
            return r_indices, estimated_theta
    return r_indices, estimated_theta


def missed_beat_detection(r_indices, i, mu1, lambda1, p, estimated_theta, eta_s, P, Q, alpha,W, verbose=False):
    if i + Q + 1 < len(r_indices):
        uk = r_indices[i]
        uk_plus_1 = r_indices[i + 1]
        initial_tau = (uk + uk_plus_1) / 2
        bounds = [(uk, uk_plus_1)]
        result = minimize(neg_log_probability, initial_tau,
                          args=(uk, uk_plus_1, mu1, lambda1, estimated_theta, r_indices, i, P), bounds=bounds)
        if result.success:
            optimal_tau = result.x[0]
            optimal_probability = -neg_log_probability(optimal_tau, uk, uk_plus_1, mu1, lambda1, estimated_theta, r_indices,
                                                       i, P)
            if optimal_probability > p + eta_s:
                if verbose:
                    print(f"Missed beat detected at index {i + 1}.")
                r_indices_new = np.insert(r_indices, i + 1, optimal_tau)
                accepted = evaluate_correction(r_indices, r_indices_new, i, P, Q, eta_s, estimated_theta)
                if accepted:
                    r_indices = r_indices_new
                    estimated_theta = estimate_theta(r_indices, r_indices[i + 2],P, W, estimated_theta, alpha)
                    return r_indices, estimated_theta
                else:
                    return r_indices, estimated_theta
            else:
                return r_indices, estimated_theta
        else:
            return r_indices, estimated_theta
    else:
        return r_indices, estimated_theta


def neg_log_probability(tau, uk, uk_plus_1, mu1, lambda1, theta_t, r_indices, i, P):
    temp_r_indices = np.insert(r_indices, i + 1, tau)
    H_k = temp_r_indices[(i - P + 1):(i + 1)] - temp_r_indices[(i - P):i]
    mu2 = mu(H_k, theta_t)
    lambda_combined = lambda1 * (mu1 + mu2) ** 3 / ((1 + theta_t[-1]) ** 2 * mu1 ** 3 + mu2 ** 3)
    return -log_f(uk_plus_1, uk, theta_t, H_k, new_lam=lambda_combined)


def compute_log_likelihood(U, k, P, Q, theta_t):
    likelihood_sum = sum(log_f(U[j + 1], U[j], theta_t, U[(j - P + 1):(j + 1)] - U[(j - P):j]) for j in range(k, k + Q))
    return likelihood_sum


def evaluate_correction(r_indices, new_r_indices, k, P, Q, eta_e, theta_t):
    original_likelihood = compute_log_likelihood(r_indices, k, P, Q, theta_t)
    corrected_likelihood = compute_log_likelihood(new_r_indices, k, P, Q, theta_t)
    return corrected_likelihood > original_likelihood + eta_e


def perfrom(r_indices, P, W, alpha, eta_e=1, eta_s=1, Q=3, eta_m=0.3, eta_t=0.4, eta_r=0.5, verbose=True):

    np.random.seed(0)
    initial_theta = np.random.rand(P + 1)
    t = r_indices[W]

    progress_bar = tqdm(total=len(r_indices) - (Q + W))

    estimated_theta = estimate_theta(r_indices, t,P, W, initial_theta, alpha)
    i = W + 1
    while i < len(r_indices) - Q:
        increment_i = True
        ut = r_indices[i]

        if i == W + 1:
            estimated_theta = estimate_theta(r_indices, ut,P, W, estimated_theta, alpha)

        H_k = r_indices[(i - P + 1):(i + 1)] - r_indices[(i - P):(i)]
        p = log_f(r_indices[i + 1], r_indices[i], estimated_theta, H_k)
        if i + 1 < len(r_indices):

            ##### extra beat detection
            r_indices, estimated_theta = extra_beat_detection(r_indices, i, H_k, p, P,W, estimated_theta, Q, eta_e, alpha,
                                                              verbose=verbose)

            ##### missed beat detection
            H_k = r_indices[(i - P + 1):(i + 1)] - r_indices[(i - P):(i)]
            p = log_f(r_indices[i + 1], r_indices[i], estimated_theta, H_k)
            mu_current = mu(H_k, estimated_theta)
            lambda_current = lambda_function(estimated_theta)
            r_indices, estimated_theta = missed_beat_detection(r_indices, i, mu_current, lambda_current, p,
                                                               estimated_theta, eta_s, P, Q, alpha,W, verbose=verbose)
        if increment_i:
            i += 1
            estimated_theta = estimate_theta(r_indices, r_indices[i],P, W, estimated_theta, alpha)
            progress_bar.update(1)
    return r_indices

################################################ Quality #########################################################

def assess_quality(r_peaks_org):
    rr_intervals = np.diff(r_peaks_org)*1000
    heart_rates = 60000 / rr_intervals

    mean_hr = np.mean(heart_rates)
    rr_intervals_in_s = rr_intervals / 1000

    if 40 < mean_hr < 180:
        rr_greater_than_3 = np.where(rr_intervals_in_s > 3)[0]
        if len(rr_greater_than_3) == 0:
            rr_ratio = max(rr_intervals_in_s) / min(rr_intervals_in_s)
            rr_ratio_lit = 2.2
            if rr_ratio < rr_ratio_lit:
                return 1
            else:
                return 0
        else:
            return 0
    else:
        return 0

def segment_quality(data, window_size, step_size):
    # Segments data into windows with a given step size
    segments = []
    for start in range(0, len(data) - window_size + 1, step_size):
        end = start + window_size
        qu = data[start:end].count(1) / len(data[start:end])
        segments.append(qu)
    return segments

def segment_rrpeaks2(peaks,t,fs, wind= 10):
    segments = []
    for start in t:
        mask = (start < peaks/fs)  & (peaks/fs < start+wind)
        segment = peaks[mask]
        segments.append(segment)
    return segments
################################################ run everything #########################################################

def process_actul(args):

    numpy_array, P, W, alpha, eta_e, eta_s, Q, verbose = args

    r_indices_ori = numpy_array
    r_indices_edit = numpy_array.copy()

    try:
        segs = segment_rrpeaks2(r_indices_ori, np.arange(r_indices_ori[0], r_indices_ori[-1], 5)[1:], 1)
        ass_original = [assess_quality(seg) for seg in segs]
        f1_original = segment_quality(ass_original, 60, 1)
        r_indices = perfrom(r_indices_edit, P=P, W=W, alpha=alpha, eta_e=eta_e, eta_s=eta_s, Q=Q, verbose=verbose)
        segs = segment_rrpeaks2(r_indices, np.arange(r_indices[0], r_indices[-1], 5)[1:], 1)
        ass = [assess_quality(seg) for seg in segs]
        f1_edited = segment_quality(ass, 60, 1)
        
        return [f1_original,f1_edited, r_indices]
    except Exception as e:
        if verbose:
            print(f"Error processing beat: {e}")
        return None
