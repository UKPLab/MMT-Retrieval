urls = [l.strip().split("\t")[1] for l in open("validation.tsv").readlines()]

with open("validation_urls_named.txt", "w") as f:
    for i, url in enumerate(urls, start=1):
        f.write(f"{i}.jpg {url}\n")