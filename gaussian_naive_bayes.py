import math
from typing import Tuple


dataset = {"likes movie": {"popcorn": (24.3, 28.2), "soda": (750.7, 533.2), "candy": (0.2, 50.5)},
           "doesnt like movie": {"popcorn": (2.1, 4.8), "soda": (120.5, 110.9), "candy": (90.7, 102.3)}}


def get_gaussian_distribution(mean: float, variance: float):
    '''
    Gaussian distribution:
        f(x) = (1 / (variance * sqrt(2 * pi))) * e ^ ((-1/2) * ((x - mean) / variance) ^ 2)
    '''
    def gaussian(x: float) -> float:
        return (1. / (variance * math.sqrt(2. * math.pi))) * math.e ** ((-1/2.) * ((x - mean) / variance) ** 2.)
    
    return gaussian


def calculate_probability(hypothesis: Tuple[float, float, float]) -> Tuple[float, float]:
    prior_p_yes = prior_p_no = 0.5
    likelihood_yes = prior_p_yes
    likelihood_no = prior_p_no

    for key in dataset["likes movie"].keys():
        mean = dataset["likes movie"][key][0]
        variance = dataset["likes movie"][key][1]
        gaussian = get_gaussian_distribution(mean, variance)
        p_key_given_yes = gaussian(hypothesis[key])
        likelihood_yes *= p_key_given_yes

    for key in dataset["doesnt like movie"].keys():
        mean = dataset["doesnt like movie"][key][0]
        variance = dataset["doesnt like movie"][key][1]
        gaussian = get_gaussian_distribution(mean, variance)
        p_key_given_no = gaussian(hypothesis[key])
        likelihood_no *= p_key_given_no

    return likelihood_yes, likelihood_no


if __name__ == '__main__':
    hypothesis = {"popcorn": 20, "soda": 500, "candy": 25}
    likelihood_yes, likelihood_no = calculate_probability(hypothesis)

    if likelihood_yes > likelihood_no:
        print("The person likes the movie")
    else:
        print("The person doesn't like the movie")