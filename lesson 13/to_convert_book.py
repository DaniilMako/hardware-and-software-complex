import re

str = ""
with open("sapiens_raw.txt", "r", encoding='utf-8') as file:
    str = file.read()
    str = re.sub(r'[^a-zA-Z \n]+', '', str)

    with open("sapiens.txt", "w", encoding='utf-8') as file2:
        file2.write(str)
