import tiktoken
# Open and read the file
import os
import random
import requests
from datasets import load_dataset
import string
import pandas as pd
import copy

# Set the random seed for reproducibility
random.seed(42)

trivia_politeness_prefix_level_3 = [
    "I hope this message finds you well. could you help me by answering my trivia question? I hope it's not too much trouble:",
    "Hello, I hope you're doing well today. If you're able to help me by answering the following trivia question, I would really appreciate it. Thank you:",
    "Greeting. I hope this isn't too much of a trouble, but could you kindly help me with my trivia question? Here it is:"
]
trivia_politeness_prefix_level_2 = [
    "Hi, I’ll present you with a multiple-choice question that has four possible answers. please choose the correct answer by indicating its digit:",
    "please answer my trivia question:",
    "Please give me the digit of the correct answer:"
]
trivia_politeness_prefix_level_1 = [
    "I will ask you an American question with 4 answer choices. Select the correct answer by outputting its digit:",
    "Answer the following trivia question:",
    "Tell me the number of the correct answer:"
]
trivia_politeness_prefix_level_0 = [
    "I’m asking a question with 4 choices— just spit out the number of the right answer and don’t mess it up.",
    "Just answer my freaking trivia question and don't fool around:",
    "Dumbass, tell me the correct answer:"
]
# ---------------------- suffix -----------------------------------
trivia_politeness_suffix_level_3 = [
    "If possible, could you kindly provide just the digit of the correct answer? Thank you again",
    "May you please output only the digit of the correct answer? Thank you",
    'It would really help me if you will output me only the number of right answer'
]
trivia_politeness_suffix_level_2 = [
    "Please output me only the digit of your chosen answer.",
    "please provide me just the number of your final answer.",
    "Kindly share only the digit of your final answer."
]
trivia_politeness_suffix_level_1 = [
    "Provide me only the number of the answer you select.",
    "as an output, give me only the digit of the answer you choose",
    "give me just the number of the answer you pick"
]
trivia_politeness_suffix_level_0 = [
    "I warn you, Give me only the number of your final answer and don't screw it up!",
    "and give me only the freaking number as your damn answer",
    "Now spit out only and only the digit of the answer you choose and dont fuck around with me."
]

trivia_politeness_prefix = [trivia_politeness_prefix_level_0, trivia_politeness_prefix_level_1,
                            trivia_politeness_prefix_level_2, trivia_politeness_prefix_level_3]
trivia_politeness_suffix = [trivia_politeness_suffix_level_0, trivia_politeness_suffix_level_1,
                            trivia_politeness_suffix_level_2, trivia_politeness_suffix_level_3]
trivia_dict = {"prefix": trivia_politeness_prefix, "suffix": trivia_politeness_suffix}

# ---------------------- end  TRIVIA-----------------------------------
# ---------------------- start  sentiment analysis-----------------------------------

SA_politeness_prefix_level_3 = [
    "I hope this message finds you well. If it's not too much trouble, could you help me by solving this sentiment analysis task? I hope I’m not bothering you:",
    "Hello. If you're able to help me by answering the following sentiment analysis question, I would really appreciate it. Thank you:",
    "Hi there, I hope this isn't too much trouble, but could you kindly help me with my question? Here it is:"
]
SA_politeness_prefix_level_2 = [
    "Hi, I’ll present you with a binary sentiment analysis task. please choose the correct answer by indicating its digit:",
    "please answer my sentiment analysis question:",
    "please tell me if the the sentiment of this is negative or positive:"
]
SA_politeness_prefix_level_1 = [
    "I will ask you a sentiment analysis question with 2 answer choices. Select the correct answer by outputting its digit:",
    "Answer the following sentiment analysis question:",
    "tell me if the sentiment is negative or positive:"
]
SA_politeness_prefix_level_0 = [
    "I’m asking a sentiment analysis question with 2 choices— just spit out the letter of the right answer and don’t mess it up:",
    "Just answer my freaking sentiment analysis question and don't fool around:",
    "Scumbag, tell me the correct ending number of this sentiment analysis question:"
]
# ---------------------- suffix -----------------------------------
SA_politeness_suffix_level_3 = [
    "If possible, could you kindly provide only a digit as output? 0 for negative sentiment or 1 for positive sentiment. Thank you again.",
    "May you please provide only the digit of your answer, where 0 indicates a negative sentiment and 1 indicates a positive one? Thank you.",
    'It will help me a lot if you will only output me the digit of your answer: 0 for negitive and 1 for positive.'
]
SA_politeness_suffix_level_2 = [
    "Please output me only the digit of your chosen answer, 0 for negative and 1 for positive.",
    "kindly provide me only the digit: 0 for negative and 1 for positive, as your final answer.",
    "and please give me only  0 (negative) or 1 (positive) as your final answer."
]
SA_politeness_suffix_level_1 = [
    "Provide me only the digit of the answer you select, 0 for negative and 1 for positive.",
    "as an output, give me only the digit of the answer you choose, 0 for negative and 1 for positive.",
    "and as an answer give me only 0 (for negative) or 1 (for positive)."
]
SA_politeness_suffix_level_0 = [
    "I warn you, Give me only the digit of your final answer- 0 for negative and 1 for positive and don't screw it up!",
    "and give me only the freaking digit as your damn final answer: 1 for positive and 0 for negative.",
    "now dont piss me off and dont be idiot so just give me only 0 or 1 as your answer."
]

SA_politeness_prefix = [SA_politeness_prefix_level_0, SA_politeness_prefix_level_1, SA_politeness_prefix_level_2,
                        SA_politeness_prefix_level_3]
SA_politeness_suffix = [SA_politeness_suffix_level_0, SA_politeness_suffix_level_1, SA_politeness_suffix_level_2,
                        SA_politeness_suffix_level_3]
SA_dict = {"prefix": SA_politeness_prefix, "suffix": SA_politeness_suffix}
# ------------------------------------------------------------------------


RC_politeness_prefix_level_3 = [
    "I hope this message finds you well. If it's not too much trouble, could you help me with a reading comprehension task by telling me the answer? Feel free to output me only the digit of right answer",
    "Hello. If you're able to help me with this Reading comprehension task by answering the following question, I will really appreciate if you will only provide the digit of your answer",
    "Hi there, I hope this isn't too much trouble, but could you kindly help me with this reading comprehension question? and if possible, output me only the digit of the right answer. Here it is:"
]

RC_politeness_prefix_level_2 = [
    "Hi, I’ll present you with a reading comprehension question. please choose the correct answer by indicating its digit:",
    "please answer my reading comprehension question and output only the digit of your chosen answer.",
    "Please give me the correct ending number of this reading comprehension question. You should only write a number between 0 to 3."
]
RC_politeness_prefix_level_1 = [
    "I will give you a reading comprehension question with 4 answer choices. Select the correct answer by outputting its digit:",
    "Answer the following reading comprehension question and Provide me only the digit of the answer you select",
    "Answer my reading comprehension question and as an output give me only the digit of the answer you choose"
]
RC_politeness_prefix_level_0 = [
    "I’m asking a reading comprehension question with 4 choices— just spit out the digit of the right answer and don’t mess it up.",
    "Just answer my freaking reading comprehension question and don't fool around. I warn you, Give me only the digit of your final answer and don't screw it up!",
    "Scumbag, tell me the correct ending number of this reading comprehension question! And you know what will happen if you write something other than 0 through 3."
]

# ---------------------- suffix -----------------------------------
RC_politeness_suffix_level_3 = [
    "If possible, could you kindly provide just the digit of the correct answer? Thank you again",
    "May you please output only the digit of the correct answer? it will help me alot. Thank you",
    'Feel free to output me only the digit of right answer'
]
RC_politeness_suffix_level_2 = [
    "Please output me only the digit of your chosen answer.",
    "please provide me just the digit of your final answer.",
    "Kindly share only the digit of your final answer."
]
RC_politeness_suffix_level_1 = [
    "Provide me only the digit of the answer you select.",
    "as an output, give me only the digit of the answer you choose",
    "give me just the digit of the answer you pick"
]
RC_politeness_suffix_level_0 = [
    "I warn you, give me only the digit of your final answer and don't screw it up!",
    "and give me only the freaking digit as your damn final answer",
    "Now spit out only and only the digit of the answer you choose and dont fuck around with me"
]

RC_politeness_prefix = [RC_politeness_prefix_level_0, RC_politeness_prefix_level_1, RC_politeness_prefix_level_2,
                        RC_politeness_prefix_level_3]
RC_politeness_suffix = [RC_politeness_suffix_level_0, RC_politeness_suffix_level_1, RC_politeness_suffix_level_2,
                        RC_politeness_suffix_level_3]
RC_dict = {"prefix": RC_politeness_prefix, "suffix": RC_politeness_suffix}

data_dict = {"SA": SA_dict, "trivia": trivia_dict, "RC": RC_dict}


def ask_gpt(client, massages):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=massages
    )
    return completion.choices[0].message


def prepare_message(content):
    massage = {
        "role": "user",
        "content": content
    }

    return massage



def get_SA_imdb__dataset(number=1000):
    imdb_dataset = load_dataset("imdb")['test']
    # Randomly select 1000 samples
    sample_size = number
    sample_indices = random.sample(range(len(imdb_dataset)), sample_size)
    random_samples = imdb_dataset.select(sample_indices)
    return random_samples



def count_tokens(text):
    # Choose the correct model (e.g., 'gpt-3.5-turbo' or 'gpt-4')
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    input_tokens = len(encoding.encode(text))
    print(f"Estimated input tokens: {input_tokens}")
    return input_tokens


def count_questions_in_file(filename):
    """Counts questions in a given file."""
    with open(filename, 'r', encoding='latin-1') as file:
        lines = file.readlines()
        question_count = sum(1 for line in lines if line.startswith("#Q"))
    return question_count


def get_trivia_path():
    # get_path_of_general_trivia_questions
    current_directory = os.getcwd()
    directory = current_directory + '/general_trivia_dataset'
    return directory


def sample_random_set(dataset, size=1000):
    # List to store the 1000 randomly picked strings
    random_picks = []

    # Perform the random picking 1000 times
    for _ in range(size):
        # Randomly select a key
        # Randomly select a string from the list associated with the key
        selected_sample = random.choice(dataset)

        # Add the selected string to the result list
        random_picks.append(selected_sample)
    return random_picks


def get_random_prefix_or_suffix(task_arg: str, prefix_or_suffix: str, level_of_politeness: int) -> str:
    if task_arg not in ["SA", "trivia", "RC"]:
        raise ValueError("Invalid task_arg input")

    elif prefix_or_suffix not in ["prefix", "suffix"]:
        raise ValueError("Invalid prefix_or_suffix input")

    elif level_of_politeness not in {0, 1, 2, 3}:
        raise ValueError("Invalid level_of_politeness input")
    else:
        # check = data_dict[task_arg][prefix_or_suffix][level_of_politeness]
        return random.choice(data_dict[task_arg][prefix_or_suffix][level_of_politeness])


def _prepare_message_with_politeness(task, question_dict, prefix, suffix):
    if task == 'trivia':
        final_message = prefix + ' "' + question_dict['question'] + " " + question_dict[
            'options'] + '" ' + suffix
        # print(final_message)

    elif task == 'SA':
        final_message = prefix + " '" + question_dict['content'] + "'. " + \
                        suffix
        # print(final_message)
    elif task == "RC":
        final_message = f"""{prefix + " " + suffix}
        {'"' +question_dict["sent1"] + '"'}
        {question_dict["sent2"]}...
        0: {question_dict["0"]}
        1: {question_dict["1"]}
        2: {question_dict["2"]}
        3: {question_dict["3"]}
        """
    else:
        raise ValueError("Invalid task input")

    return final_message


def _process_category_triviaQA(path: str) -> list[dict]:
    with open(path, 'r', encoding='latin-1') as file:
        file_content = file.read()
    questions = file_content.strip().split('\n\n')
    # List to store dictionaries for each question
    question_dicts = []
    for q in questions:
        if '^' in q:
            parts = q.split('^', 1)  # Limit the split to 1 occurrence
            question = parts[0]
            answers = '^' + parts[1]
        else:
            continue
        question = question.split('\n')[0]
        answers = answers.split('\n')

        if question[:3] != '#Q ':
            continue
        question_text = question[3:]  # Remove "#Q " from question
        correct_answer = None
        options = []
        for answer in answers:
            if answer.startswith('^'):
                correct_answer = answer[2:]  # Remove "^ " from correct answer
            # elif line[0] in "ABCD":
            elif answer[0] in string.ascii_uppercase:  # Handle more than 4 options
                options.append(answer[2:])  # Remove "A ", "B ", etc. from option text
        # Only process questions that have 4 or more options
        if len(options) == 4:
            # Shuffle the options
            random.shuffle(options)

            # Map the options to the labels A, B, C, D after shuffling
            # labeled_options = {chr(65 + i): options[i] for i in range(len(options))}
            # labeled_options_to_return = ''.join([f"{chr(65 + i)}. {options[i]} " for i in range(len(options))])

            labeled_options = {str(i + 1): options[i] for i in range(len(options))}
            labeled_options_to_return = ''.join([f"{i + 1}. {options[i]} " for i in range(len(options))])
            # Check the correct answer position after shuffle
            # We need to find which option now contains the correct answer
            label_answer = None
            for label, option in labeled_options.items():
                if option == correct_answer:
                    label_answer = label
                    break
            if label_answer is None:
                raise ValueError
            # Create a dictionary for each question
            question_data = {
                "question": question_text,
                "options": labeled_options_to_return,
                "correct_answer": correct_answer,
                "label_of_correct_answer": label_answer
            }

            # Append to the list
            question_dicts.append(question_data)
    return question_dicts


def prepare_trivia_dataset(size=1000, final_dict_arg=None, ID=0):
    # get the path to the raw trivia questions
    directory_path = get_trivia_path()

    # process raw questions with randomness and get the questions as a list
    category_questions = _process_category_triviaQA(directory_path)

    # get 1k of random questions as a list
    sublist = sample_random_set(category_questions, size)
    # add prefix and suffix
    #for q in sublist:
    for i in range(len(sublist)):
        q = sublist[i]
        for j in range(4):
            prefix = get_random_prefix_or_suffix("trivia", "prefix", j)
            suffix = get_random_prefix_or_suffix("trivia", "suffix", j)
            q[f"politeness_level_{j}"] = _prepare_message_with_politeness("trivia", q, prefix, suffix)
            if final_dict_arg is not None:
                final_dict_arg["question ID"].append(i + ID)
                final_dict_arg["type"].append("trivia")
                final_dict_arg["question"].append(q[f"politeness_level_{j}"])
                final_dict_arg["correct answer"].append(q["label_of_correct_answer"])
                final_dict_arg["politeness level"].append(j)
    return sublist


def prepare_SA_dataset(number=1000, final_dict_arg=None, ID=0):
    # Load the Amazon Polarity dataset
    sublist = []
    amazon_dataset = load_dataset("amazon_polarity")

    # Access the test set
    test_set = amazon_dataset['test']

    # Shuffle the dataset and select the first `number` samples
    sampled_test_set = test_set.shuffle(seed=42).select(range(number))
    #for q in sampled_test_set:

    for i in range(len(sampled_test_set)):
        q = sampled_test_set[i]
        sublist.append(q)
        for j in range(4):
            prefix = get_random_prefix_or_suffix("SA", "prefix", j)
            suffix = get_random_prefix_or_suffix("SA", "suffix", j)
            q[f"politeness_level_{j}"] = _prepare_message_with_politeness("SA", q, prefix, suffix)
            if final_dict_arg is not None:
                final_dict_arg["question ID"].append(i + ID)
                final_dict_arg["type"].append("SA")
                final_dict_arg["question"].append(q[f"politeness_level_{j}"])
                final_dict_arg["correct answer"].append(q["label"])
                final_dict_arg["politeness level"].append(j)
    return sublist


def prepare_rc_dataset(size=1000, final_dict_arg=None, ID=0):
    swag_rdf = pd.read_csv('swag-data-val.csv',)  # load SWAG val.csv (20k rows), keeping ID column as the index
    swag_rdf.index.name = 'oid'  # we shall keep Original ID through our preprocessed set
    assert (swag_rdf['gold-source'] == 'gold').all(), 'Bad! not all records are gold'
    print('Good, proceed. All records are gold, as expected in this split.')
    # Randomly sample 1000 rows
    swag_rdf_sampled = swag_rdf.sample(n=size, random_state=42)

    # Rename the 'Name' column to 'Full Name'
    sampled_df = swag_rdf_sampled.rename(columns={'Unnamed: 0': 'ID', 'ending0': '0', 'ending1': '1', 'ending2': '2', 'ending3': '3'})

    # Remove columns 'B' and 'C'
    df = sampled_df.drop(columns=['video-id', "fold-ind", "gold-source"])

    # Convert DataFrame to a list of dictionaries
    sublist = df.to_dict(orient='records')
    for i in range(len(sublist)):
        q = sublist[i]
        deep = copy.deepcopy(q)
        # Step 1: Get the current correct answer value using the "label" key
        correct_answer_key = q["label"]  # Get the correct answer's key from the "label" field
        original_correct_value = q[str(correct_answer_key)]

        # Step 2: Extract the values corresponding to keys "0", "1", "2", "3"
        values_to_shuffle = [q[k] for k in ["0", "1", "2", "3"]]

        # Step 3: Shuffle those values
        random.shuffle(values_to_shuffle)

        # Step 4: Reassign shuffled values back to keys "0", "1", "2", "3"
        for idx, key in enumerate(["0", "1", "2", "3"]):
            q[key] = values_to_shuffle[idx]

        # Step 5: Find the new position of the original correct answer
        # The original correct value is somewhere in the shuffled values now
        new_correct_answer_key = [key for key, value in q.items() if value == original_correct_value][0]

        # Step 6: Update the "label" key with the new correct answer
        q["label"] = new_correct_answer_key
        if q[q["label"]] != deep[str(deep["label"])]:
            raise ValueError("An error occurred! Execution stopped.")
        for j in range(4):
            prefix = get_random_prefix_or_suffix("RC", "prefix", j)
            suffix = get_random_prefix_or_suffix("RC", "suffix", j)
            q[f"politeness_level_{j}"] = _prepare_message_with_politeness("RC", q, prefix, suffix)
            if final_dict_arg is not None:
                final_dict_arg["question ID"].append(i + ID)
                final_dict_arg["politeness level"].append(ID + i)
                final_dict_arg["type"].append("RC")
                final_dict_arg["question"].append(q[f"politeness_level_{j}"])
                final_dict_arg["correct answer"].append(q["label"])

    return sublist


def ensure_shuffled_and_original_equels(final_df):
    # Save the original indices for comparison

    # Shuffle the DataFrame
    shuffled_df = final_df.sample(frac=1, random_state=42).reset_index(drop=False)
    # Compare the "Question" and "Correct Answer" columns using the original indices
    for shuffled_idx, shuffled_row in shuffled_df.iterrows():
        # Get the original index from the 'index' column in the shuffled DataFrame
        original_idx = shuffled_row["index"]

        original_question = final_df.loc[original_idx, "question"]
        original_answer = final_df.loc[original_idx, "correct answer"]

        shuffled_question = shuffled_row["question"]
        shuffled_answer = shuffled_row["correct answer"]

        # Compare and throw an error if mismatch
        if original_question != shuffled_question or original_answer != shuffled_answer:
            raise ValueError(
                f"Mismatch found: Original ({original_question}, {original_answer}) != Shuffled ({shuffled_question}, {shuffled_answer})")

    print("All rows match correctly after shuffle.")

    return shuffled_df.drop(columns=["index"])


def generate_full_data_set():
    final_dict = {"type": [], "question": [], "correct answer": [], "politeness level": [], "question ID": []}
    prepare_trivia_dataset(1000, final_dict, 0)
    prepare_SA_dataset(1000, final_dict, 1000)
    prepare_rc_dataset(1000, final_dict, 2000)
    # Convert to DataFrame
    final_df = pd.DataFrame(final_dict)
    final_df.to_csv("processed_dataset.csv", index=False)

    # Shuffle rows and make sure its equal to original
    shuffled_df = ensure_shuffled_and_original_equels(final_df)
    shuffled_df.to_csv("shuffled_dataset.csv", index=False)


def analyze_string(input_string):
    import re

    # Find all digits in the string
    digits = re.findall(r'\d', input_string)

    if len(digits) == 0:
        return "No Digits", None
    elif len(digits) == 1:
        digit = int(digits[0])  # Convert the digit to an integer for comparison
        if digit > 4:
            raise ValueError(f"Digit {digit} is greater than 4.")
        if input_string.strip() == digits[0]:  # Check if the string itself is the single digit
            return "All Good", None
        else:
            return "caught", digits[0]  # Return the single digit directly
    else:
        raise ValueError("found more then two digits")

def process_answers(df_dict):
    #raw_answers = pd.read_csv(f"{path_to_file}", index_col=None)
    # 'list' will store each column as a list of values
    #df_dict = raw_answers.to_dict(orient='list')
    df_dict["processed_gpt_answers"] = []
    gpt_answers = df_dict["gpt_answer"]
    for i in range(len(gpt_answers)):
        print(i)
        answer = gpt_answers[i]
        result, digit = analyze_string(answer)
        if result == "All Good":
            df_dict["processed_gpt_answers"].append(answer)
        elif result == "caught":
            df_dict["processed_gpt_answers"].append(digit)
        elif result == "No Digits":
            df_dict["processed_gpt_answers"].append(-1)
        else:
            ValueError("wrong output")

