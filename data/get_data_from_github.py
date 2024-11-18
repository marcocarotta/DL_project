import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

if response.status_code == 200:
    with open("data/dataset.txt", "wb") as file:
        file.write(response.content)
    print("Dataset downloaded successfully.")
else:
    print("Failed to download the dataset.")