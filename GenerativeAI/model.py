import openai
import pandas as pd
from tqdm import tqdm
import json
import requests
from pydantic import BaseModel

# ================= GPT =================

# Using Structured Outputs
# https://platform.openai.com/docs/guides/structured-outputs/how-to-use?context=with_parse
class GPT_Response(BaseModel):
    prediction: int


def predict_gpt(df, prompt, model, pred_column):
    """
    Classify tweets using GPT-4.

    Args:
        df (dataframe): A dataframe of tweets to be classified
        prompt (str): A string containing the prompt to be passed to GPT-4

    Returns:  
    """

    discarded_indices = []
    for i in tqdm(range(0, len(df))):

        tweet = df.loc[i, "full_text"]
        #print(f"{tweet=}")


        ## Sending 1 review in each request.You may send in batches of 4 or 8 if the dataset is large
        try:
            res = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Tweet: {tweet}"},
                ],
                response_format=GPT_Response # using new response_format for Structured Outputs
            )

            response = res.choices[0].message.parsed
            print(response)

            if pred_column == "Predicted AG":

                if "0" in response:
                        df.at[i, pred_column] = 0
                elif "1" in response:
                    df.at[i, pred_column] = 1
                else:
                    df.at[i, pred_column] = -1
                    discarded_indices.append(i)

            if pred_column == "Predicted CNeg":
                if "0" in response:
                    df.at[i, pred_column] = 0
                elif "1" in response:
                    df.at[i, pred_column] = 1
                elif "2" in response:
                    df.at[i, pred_column] = 2
                elif "3" in response:
                    df.at[i, pred_column] = 3
                else:
                    df.at[i, pred_column] = -1
                    discarded_indices.append(i)

            if pred_column == "Predicted I":
                if "0" in response:
                    df.at[i, pred_column] = 0
                elif "1" in response:
                    df.at[i, pred_column] = 1
                elif "2" in response:
                    df.at[i, pred_column] = 2
                else:
                    df.at[i, pred_column] = -1
                    discarded_indices.append(i)


        except Exception as e:
            print(str(e))

    return df, discarded_indices



# ================= LM STUDIO =================

def predict_lmstudio(df, context, prompt, model, pred_column, temp, preprocessed_data=False):
    url = "http://localhost:1234/v1/chat/completions"


    discarded_indices = []

    for i in tqdm(range(0, len(df))):

        if preprocessed_data:
            tweet = df.loc[i, "full_text_processed"]
        else:
            tweet = df.loc[i, "full_text"]
        #print(f"{tweet=}")

        # Headers
        headers = {
            "Content-Type": "application/json"
        }

        
        payload = {
            "messages": [
                {"role": "assistant", "content": context},
                {"role": "assistant", "content": prompt},
                {"role": "user", "content": f"Tweet: {tweet}" }
            ],
            "temperature": temp,
            "stream": False,
            "model": model
        }
        # Send POST request
        response = requests.post(url, headers=headers, data=json.dumps(payload))

        # Check for successful response
        if response.status_code == 200:
            response_content = response.json().get("choices")[0].get("message").get("content")
            #print(f"{response_content=}")

            # Parse the structured JSON response
            try:
                response_dict = json.loads(response_content)
                prediction = response_dict.get("prediction")

                if prediction is not None:
                    df.at[i, pred_column] = int(prediction)
                else:
                    df.at[i, pred_column] = -1
                    discarded_indices.append(i)

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                df.at[i, pred_column] = -1
                discarded_indices.append(i)

        else:
            return {"error": "Request failed with status code " + str(response.status_code)}
            

    return df, discarded_indices



