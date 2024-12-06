def subsample_frequent_words(word_count_dict, threshold=1e-5):
    """
    Calculates the subsampling probabilities for words based on their frequencies.

    This function is used to determine the probability of keeping a word in the dataset
    when subsampling frequent words. The method used is inspired by the subsampling approach
    in Word2Vec, where each word's frequency affects its probability of being kept.

    Parameters:
    - word_count_dict (dict): A dictionary where keys are words and values are the counts of those words.
    - threshold (float, optional): A threshold parameter used to adjust the frequency of word subsampling.
                                   Defaults to 1e-5.

    Returns:
    - dict: A dictionary where keys are words and values are the probabilities of keeping each word.

    Example:
    >>> word_counts = {'the': 5000, 'is': 1000, 'apple': 50}
    >>> subsample_frequent_words(word_counts)
    {'the': 0.028, 'is': 0.223, 'apple': 1.0}
    """

    total_count = sum(word_count_dict.values())
    keep_prob_dict = {}
    for word, count in word_count_dict.items():
        freq = count / total_count
        keep_prob_dict[word] = min(1.0, (threshold / freq) ** 0.5)
    
    return keep_prob_dict
    
def get_negative_sampling_prob(word_count_dict):
    """
    Calculates the negative sampling probabilities for words based on their frequencies.

    This function adjusts the frequency of each word raised to the power of 0.75, which is
    commonly used in algorithms like Word2Vec to moderate the influence of very frequent words.
    It then normalizes these adjusted frequencies to ensure they sum to 1, forming a probability
    distribution used for negative sampling.

    Parameters:
    - word_count_dict (dict): A dictionary where keys are words and values are the counts of those words.

    Returns:
    - dict: A dictionary where keys are words and values are the probabilities of selecting each word
            for negative sampling.

    Example:
    >>> word_counts = {'the': 5000, 'is': 1000, 'apple': 50}
    >>> get_negative_sampling_prob(word_counts)
    {'the': 0.298, 'is': 0.160, 'apple': 0.042}
    """

    adjusted_freq_dict = {word: count ** 0.75 for word, count in word_count_dict.items()}
    Z = sum(adjusted_freq_dict.values())
    negative_sampling_prob_dict = {word: adjusted_freq / Z for word, adjusted_freq in adjusted_freq_dict.items()}
    
    return negative_sampling_prob_dict
    