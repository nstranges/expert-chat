from ExpertChat import Llama, Mixtral

# Create a conversation loop and play the conversations
def conversation_loop(total_topics, total_exchanges, saving_path):
    # Loading the models into GPU RAM
    llama = Llama()
    mixtral = Mixtral()

    # Different topics
    for i in range(total_topics):
        first = None
        second = None
        topic = ""

        # Alternate topic choosers
        if ((i+1) % 2) == 0:
            first = mixtral
            second = llama
        else:
            first = llama
            second = mixtral

        # Choose and set topic
        topic = first.create_new_topic()
        first.set_topic(topic)
        second.set_topic(topic)

        # Starting convo
        response = first.start_conversation()

        # Amount of questions/answers
        for j in range(total_exchanges): 
            # Alternate speaker
            if ((j+1) % 2) == 0:
                response = first.give_message(response)
            else:
                response = second.give_message(response)

        # Rate the other expert after the convo
        first.rate_the_expert()
        second.rate_the_expert()

        # Save convo for each
        first.save_conversation(saving_path, i)
        second.save_conversation(saving_path, i)

        # Reset convo for each
        first.reset_conversation()
        second.reset_conversation()
        

topics = 1
exchanges = 2
output_path = '/home/nstrang2/projects/def-lenck/nstrang2/Conversations/'

conversation_loop(topics, exchanges, output_path)