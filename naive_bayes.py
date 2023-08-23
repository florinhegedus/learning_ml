from typing import Tuple


# (weather, day of the week, time of day, play)
tennis_dataset = [("sun", "Sunday", "morning", "yes"),
                   ("overcast", "Saturday", "lunch", "no"),
                   ("overcast", "Monday", "evening", "yes"),
                   ("rain", "Saturday", "lunch", "no"),
                   ("sun", "Saturday", "evening", "yes"),
                   ("overcast", "Wednesday", "evening", "yes"),
                   ("sun", "Friday", "evening", "no"),
                   ("sun", "Tuesday", "morning", "yes"),
                   ("rain", "Tuesday", "lunch", "yes")]


def calculate_probabilities(hypothesis: Tuple[str, str, str, str]) -> Tuple[float, float]:
    """
    Calculate the probabilities of playing and not playing tennis given the evidence using Bayes Theorem with corrected Laplace smoothing.
    """
    # Extract the evidence from the hypothesis
    weather, day, time, _ = hypothesis

    # Unique counts for each feature for smoothing
    k_weather = len(set([w for w, _, _, _ in tennis_dataset]))
    k_day = len(set([d for _, d, _, _ in tennis_dataset]))
    k_time = len(set([t for _, _, t, _ in tennis_dataset]))

    # Calculate P(play) without smoothing
    p_yes = sum([1 for _, _, _, play in tennis_dataset if play == "yes"]) / len(tennis_dataset)
    p_no = 1 - p_yes

    # Calculate P(evidence|play = "yes") with smoothing
    p_weather_given_yes = (sum([1 for w, _, _, play in tennis_dataset if play == "yes" and w == weather]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "yes"]) + k_weather)
    p_day_given_yes = (sum([1 for _, d, _, play in tennis_dataset if play == "yes" and d == day]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "yes"]) + k_day)
    p_time_given_yes = (sum([1 for _, _, t, play in tennis_dataset if play == "yes" and t == time]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "yes"]) + k_time)

    likelihood_yes = p_weather_given_yes * p_day_given_yes * p_time_given_yes

    # Calculate P(evidence|play = "no") with smoothing
    p_weather_given_no = (sum([1 for w, _, _, play in tennis_dataset if play == "no" and w == weather]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "no"]) + k_weather)
    p_day_given_no = (sum([1 for _, d, _, play in tennis_dataset if play == "no" and d == day]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "no"]) + k_day)
    p_time_given_no = (sum([1 for _, _, t, play in tennis_dataset if play == "no" and t == time]) + 1) / (sum([1 for _, _, _, play in tennis_dataset if play == "no"]) + k_time)

    likelihood_no = p_weather_given_no * p_day_given_no * p_time_given_no

    # Calculate P(evidence) without smoothing, as the sum of P(evidence|yes)*P(yes) and P(evidence|no)*P(no)
    p_evidence = (likelihood_yes * p_yes) + (likelihood_no * p_no)

    # Calculate P(play = "yes" | evidence) and P(play = "no" | evidence) using Bayes Theorem with smoothing
    p_yes_given_evidence = (p_yes * likelihood_yes) / p_evidence
    p_no_given_evidence = (p_no * likelihood_no) / p_evidence

    return p_yes_given_evidence, p_no_given_evidence


if __name__ == '__main__':
    hypothesis = ("rain", "Sunday", "lunch", None)
    probability_of_playing, probability_of_not_playing = calculate_probabilities(hypothesis)
    print(f"Probabilities of playing tennis given {hypothesis} are:")
    print(f"\tyes: {probability_of_playing}")
    print(f"\tno: {probability_of_not_playing}")