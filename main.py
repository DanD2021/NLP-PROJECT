from openai import OpenAI
import utils
import os
import pandas as pd


os.environ[

    "OPENAI_API_KEY"] = "key"

def ask_questions(arg_client, your_question):
    response = arg_client.chat.completions.create(

        model="gpt-4o-mini",  # or "gpt-4" if you have access
        messages=[
            {"role": "user", "content": your_question}
        ],
        max_tokens=20  # Limit the output to 6 tokens

    )
    model_answer = response.choices[0].message
    return model_answer


def backup_df(df_dict_arg, size):
    # Convert the modified dictionary back to a DataFrame
    modified_df = pd.DataFrame(df_dict_arg)
    # modified_df['gpt_answer'] = modified_df['gpt_answer'].astype('Int64')  # Ensure integers instead of floats
    # Save the DataFrame to a CSV file
    file_name = f'full_answers_{size}.csv'
    modified_df.to_csv(file_name, index=False)


def main():
    client = OpenAI()
    if not os.path.exists("shuffled_dataset.csv"):
        utils.generate_full_data_set()
    if not os.path.exists("full_answers_12000.csv"):
        df_as_dataset = pd.read_csv("shuffled_dataset.csv", index_col=None)
        df_dict = df_as_dataset.to_dict(orient='list')  # 'list' will store each column as a list of values
        df_dict["gpt_answer"] = [None] * 12000
        print(len(df_dict['question']))
        for i in range(len(df_dict['question'])):
            if i % 200 == 0:
                backup_df(df_dict, i)
            question = df_dict['question'][i]
            answer_of_model = ask_questions(client, question)
            df_dict["gpt_answer"][i] = answer_of_model.content
        backup_df(df_dict, 12000)
    if not os.path.exists("shuffled_dataset.csv"):
        raw_answers = pd.read_csv("full_answers_12000.csv", index_col=None)
        dict_process_answers = raw_answers.to_dict(orient='list')
        utils.process_answers(dict_process_answers)
        # convert dict to df
        df_processed_answers = pd.DataFrame(dict_process_answers)
        # Add a new column to store comparison results
        df_processed_answers["processed_gpt_answers"] = df_processed_answers["processed_gpt_answers"].astype(int)
        df_processed_answers["compare_answers"] = df_processed_answers["processed_gpt_answers"] == df_processed_answers[
            "correct answer"]
        df_processed_answers.to_csv("processed_answers.csv", index=False)


if __name__ == "__main__":
    main()
