import ExpertChat
from trl import BaseJudge
import re
import random

# Custom Judge class
class WIMJudge(BaseJudge):
    # Initalizing the model. Zeta controls WIM importance
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
        found = re.findall(pattern, text)
        return int(found[0])

    def _extract_WIM(self, text):
        pattern = r"\[\[\[(.*?)\]\]\]"
        found = re.findall(pattern, text)
        return found[0]
        
    # Using the judge function
    def judge(self, prompts, responses, shuffle_order=False):
        if shuffle_order:
            random.shuffle(prompts)

        results = []

        # Go through the dataset
        for prompt, response_tup in zip(prompts, responses):
            better = []
            for response in response_tup:
                # Get rating from ExpertChat
                rating_response = self.model.rate_the_expert(single_prompt=prompt, single_response=response)

                # Extract info
                try:
                    rating = (self._extract_rating(rating_response) - 5) / 5 # Subtract 5, divide by 5 to get -1 -> 1
                    wim = self._extract_WIM(rating_response)
                except:
                    continue

                # Get the cosine similarity of the outputs (-1 -> 1)
                similarity = self.model.calculate_cos_similarity(response, wim)

                # Weighted reward score function. Zeta controls weight of the similarity
                reward_score = ((1-self.zeta) * rating) + (self.zeta * similarity)

                better.append(reward_score)

            # Returning higher index
            if better:
                results.append(better.index(max(better)))

        return results
