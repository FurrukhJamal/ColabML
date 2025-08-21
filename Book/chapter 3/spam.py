import sys 

assert(sys.version_info) >= (3, 7)

from packaging import version 
import sklearn 
assert version.parse(sklearn.__version__) >= version.parse("1.0.1")
from pathlib import Path
import tarfile
import urllib.request 

def loadData():
    spamPath = Path("datasets/spam")
    urlRoot = "https://spamassassin.apache.org/old/publiccorpus/"
    spamUrl = urlRoot + "20030228_spam.tar.bz2"
    hamUrl = urlRoot + "20030228_easy_ham.tar.bz2"

    spamPath.mkdir(parents=True, exist_ok=True)
    for dirName, tarName, url in (("easy_ham", "ham", hamUrl), ("spam", "spam", spamUrl)):
        if not (spamPath/dirName).is_dir():
            path = (spamPath / tarName).with_suffix(".tar.bz2")
            urllib.request.urlretrieve(url, path)
            tarf = tarfile.open(path)
            tarf.extractall(path = spamPath)
            tarf.close()

    return [spamPath / dirName for dirName in ("easy_ham", "spam")]

import email 
import email.policy 

def load_email(filepath):
    with open(filepath, "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
    

def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        multipart = ", ".join([get_email_structure(sub_email) for sub_email in payload])
        return f"multipart({multipart})"
    else:
        return email.get_content_type()

from collections import Counter 
def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1 
    return structures

    

if __name__ == "__main__":
    ham_dir, spam_dir = loadData()
    print(ham_dir)
    print(spam_dir)

    hamFilesNames = [f for f in sorted(ham_dir.iterdir()) if len(f.name) > 20]
    spamFilesNames = [f for f in sorted(spam_dir.iterdir()) if len(f.name) > 20]

    print(f"len(hamFilesNames) : {len(hamFilesNames)}")
    print(f"len(spamFilesNames : {len(spamFilesNames)}")

    hamEmails = [load_email(filepaths) for filepaths in hamFilesNames]
    spamEmails = [load_email(filepaths) for filepaths in spamFilesNames]

    print(hamEmails[1].get_content().strip())

    print("\n\n")
    print("SPAM EMAIL")

    print(spamEmails[1].get_content().strip())

    print(structures_counter(hamEmails).most_common())

    print(structures_counter(spamEmails).most_common())

    for header, value in spamEmails[0].items():
        print(header, ":", value)

    
    import numpy as np 
    from sklearn.model_selection import train_test_split 

    X = np.array(hamEmails + spamEmails , dtype=object)
    Y = np.array([0] * len(hamEmails) + [1] * len(spamEmails))

    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.2, random_state=42)

    import re
    from html import unescape

    def html_to_plain_text(html):
        text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
        text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
        text = re.sub('<.*?>', '', text, flags=re.M | re.S)
        text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
        return unescape(text)
    
    html_spam_emails = [email for email in xTrain[yTrain==1]
                    if get_email_structure(email) == "text/html"]
    sample_html_spam = html_spam_emails[7]
    print(sample_html_spam.get_content().strip()[:1000], "...")

    print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...") 

    def email_to_text(email):
        html = None
        for part in email.walk():
            ctype = part.get_content_type()
            if not ctype in ("text/plain", "text/html"):
                continue
            try:
                content = part.get_content()
            except: # in case of encoding issues
                content = str(part.get_payload())
            if ctype == "text/plain":
                return content
            else:
                html = content
        if html:
            return html_to_plain_text(html)
        
    
    print(email_to_text(sample_html_spam)[:100], "...")


    import nltk

    stemmer = nltk.PorterStemmer()
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute",
                "Compulsive"):
        print(word, "=>", stemmer.stem(word))


 
