import ExpertChat
from trl import BaseJudge
import re
import random

# Custom Judge class
class WIMJudge(BaseJudge):
    # Initalizing the model. Zeta controls WIM importance
    def __init__(self, zeta=1.0, model_name='llama', model=None, tokenizer=None):
        if model_name == 'llama':
            self.model = ExpertChat.Llama(rating=True, model=model, tokenizer=tokenizer)
        elif model_name == 'mixtral':
            self.model = ExpertChat.Mixtral(rating=True)
        elif model_name == 'qwen':
            self.model = ExpertChat.Qwen(rating=True, model=model, tokenizer=tokenizer)
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
            print(f'Prompt: {prompt}')

            for response in response_tup:
                # Get rating from ExpertChat
                print(f'Model Response: {response}')
                rating_response = self.model.rate_the_expert(single_prompt=prompt, single_response=response)

                try:
                    rating = float(self._extract_rating(rating_response) - 5) / 5.0 # Subtract 5, divide by 5 to get -1 -> 1
                except:
                    rating = 0

                # Extract info
                try:
                    wim = self._extract_WIM(rating_response)
                    print(f'Judge Model Feedback: {wim}')

                    # Making a perfectly similar reward if there was no feedback
                    if wim == '':
                        similarity = 1
                    else:
                        # Get the cosine similarity of the outputs (-1 -> 1)
                        similarity = self.model.calculate_cos_similarity(response, wim)

                    # Printing the two rating systems 
                    print(f'Rating: {rating}')
                    print(f'Response: {similarity}')

                    # Weighted reward score function. Zeta controls weight of the similarity
                    reward_score = ((1-self.zeta) * rating) + (self.zeta * similarity)
                except:
                    reward_score = rating

                better.append(reward_score)

            # Returning higher index
            if better:
                best_idx = better.index(max(better))
                results.append(best_idx)

        return results
