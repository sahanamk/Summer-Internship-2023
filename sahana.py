import requests
from bs4 import BeautifulSoup
dlithe = requests.get("https://www.dlithe.com")
soup = BeautifulSoup(dlithe.content,'html.parser')

'''for data in soup.find_all("p"):
    print(data.get_text())'''

all_links = []
links = soup.select("a")
for ahref in links:
    text = ahref.text
    text = text.strip() if text is not None else ''
    href = ahref.get('href')
    href = href.strip() if href is not None else ''
    all_links.append({"href":href, "text": text})
    
for a in all_links:
    if a["text"]=='About us':
        aboutus_href = a["href"]
        break

aboutus = requests.get(aboutus_href)
soup2 = BeautifulSoup(aboutus.content,'html.parser')

s1 = ''
for data in soup2.find_all("p"):
    s1+=data.get_text()
    
print(s1)
print("Length of s1: ",len(s1))

s1_list = s1.split('.')

words_s1_five_string = ''
for i in range(5):
    words_s1_five_string+=s1_list[i]+'.'
print(words_s1_five_string)

words_s1_five = words_s1_five_string.split(' ')
print("Length of words_s1_five: ",len(words_s1_five))

print("Number of occurences of\'DLithe\':",words_s1_five.count("DLithe"))
print("Number of occurences of\'the\':",words_s1_five.count("the"))