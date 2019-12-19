import wget
import tarfile
import os

def download():
    wget.download("https://storage.googleapis.com/kagglesdsdata/competitions/3364/31151/fer2013.tar.gz?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1576831866&Signature=ltaqYmggNs6vnr67Zc71XoqqarHsXyx1RtdvKkut0bM5k0C13nTIAhlASHfvsQIsF7BEXWFVXp7V2PmPtqY3%2FESzp2EXwPHrXhk9ERBb4Q6V1Sn8viBq%2FFwmvX%2FS7X%2F8Rcd185I0iCxunubOCCUNAIBukwQYbdi6gyF8KzvZ4fEXHJxtD2YkWTx%2B%2BuE6P1EBBgyU1FYsmmx9Jz9JqoyOTkRvp6m3fwi65oNkI%2FlH3i7YPrDJPdvMHvYlJK194HQfHt1fuBdJBcjeInQjEh3xBdmfDJmldIF9xOQ59KyuD4uk0yaQdgXf08c1hSkPH7sSk%2F%2BUHVVWLCa04p9MCXmuDA%3D%3D&response-content-disposition=attachment%3B+filename%3Dfer2013.tar.gz", "fer2013.zip")
    tar = tarfile.open("fer2013.zip")
    tar.extractall(path='./data')
    tar.close()
    os.remove("fer2013.zip")

if __name__ == "__main__":
    download()