import sys
print(sys.version)

import os
os.system("python -m pip install apify_client")

from apify_client import ApifyClient
import pandas as pd

# Initialize the ApifyClient with your API token
#client = ApifyClient("apify_api_pFtfizXEQ9pOBi396aPXDGoOjeN6ue0yYyAB") # jorge (dsp)
client = ApifyClient("apify_api_w76G952jzUa1ojRSJG815IbktHEEHV1nSSTr") # gatech (tfg)


# Specify the build of the actor
build = 'latest'  # or 'beta', or a specific build number


# Initialize an empty list to store the data
data = []

#months = {'6': 31}
#months = {'7': 31, '8': 31, '9': 30, '10': 31, '11': 30, '12': 31}
#months = {'1': 31, '2': 28, '3': 31, '4': 30, '5': 31, '6': 30, '7': 31, '8': 31, '9': 30, '10': 31, '11': 30, '12': 31}
#months = {'1': 31, '2': 28, '3': 31, '4': 30, '5': 31, '6': 30}
#months = {'4': 30, '5': 31, '6': 30}
#months = {'7': 31, '8': 31, '9': 30}
months = {'8': 31, '9': 30, '10': 31}

count = 0
year = 2023

for month in months:
  for day in range(1, months[month]+1):
    print("Month: {} Day: {}".format(month, day))
    end_day = day + 1
    end_month = month


    if day == 31:
      end_day = 1
      end_month = int(month)+1



    # Prepare the Actor input
    run_input = {
        "collect_user_info": True,
        "detect_language": False,
        "filter:blue_verified": False,
        "filter:has_engagement": False,
        "filter:images": False,
        "filter:media": False,
        "filter:nativeretweets": False,
        "filter:quote": False,
        "filter:replies": False,
        "filter:retweets": False,
        "filter:safe": False,
        "filter:twimg": False,
        "filter:verified": False,
        "filter:videos": False,
        "language": "es",
        "only_tweets": False,
        "queries": [
            "#seacabó OR #seacabo since:{}-{}-{} until:{}-{}-{} lang:es".format(year, month, day, year, end_month, end_day)
        ],
        "use_experimental_scraper": False,
        #"max_tweets": 500,
        "user_info": "user info and replying info",
        "max_attempts": 5
    }

    # Run the Actor and wait for it to finish
    run = client.actor("shanes/tweet-flash-plus/WehFakshFdcufQYK1").call(run_input=run_input, build=build)

    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        count += 1

        data.append(item)

    print(f"...{count} tweets stored...")


# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
df
df.to_csv('data/seacabo.csv')

