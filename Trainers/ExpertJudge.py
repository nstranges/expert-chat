import ExpertChat
from trl import BaseJudge
import re
import random

# Custom Judge class
class WIMJudge(BaseJudge):
    # Initalizing the model. Zeta controls WIM improtance
    def __init__(self, zeta=0.4, model_name='llama'):
        if model_name == 'llama':
            self.model = ExpertChat.Llama(rating=True)
        elif model_name == 'mixtral':
            self.model = ExpertChat.Mixtral(rating=True)
        else:
            raise ValueError(f"Unsupported model_name '{model_name}'. Valid options are 'llama' or 'mixtral'.")
        
        self.zeta = zeta

    def _extract_rating(self, text):
        pattern = r"\[\[(.*?)\]\]"
        return re.findall(pattern, text)

    def _extract_WIM(self, text):
        pattern = r"\[\[\[(.*?)\]\]\]"
        return re.findall(pattern, text)
        
    # Using the judge function
    def judge(self, prompts, responses, shuffle_order=False):
        if shuffle_order:
            random.shuffle(prompts)

        results = []

        # Go through the dataset
        for prompt, response in zip(prompts, responses):
            # Get rating from ExpertChat
            rating_response = self.model.rate_the_expert(single_prompt=prompt["content"], single_response=response["content"])

            # Extract info
            rating = (self._extract_rating(rating_response) - 5) / 5 # Subtract 5, divide by 5 to get -1 -> 1
            wim = self._extract_WIM(rating_response)

            # Get the cosine similarity of the outputs (-1 -> 1)
            similarity = self.model.calculate_cos_similarity(response, wim)

            # Weighted reward score function. Zeta controls weight of the similarity
            reward_score = ((1-self.zeta) * rating) + (self.zeta * similarity)

            results.append(reward_score)

        return results





