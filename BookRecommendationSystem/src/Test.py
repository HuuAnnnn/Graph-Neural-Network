import matplotlib.pyplot as plt
from urllib.request import Request, urlopen
from PIL import Image


def read_images_from_url(url: str):
    req = Request(
        url=url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.11 (KHTML, like Gecko) "
            "Chrome/23.0.1271.64 Safari/537.11",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
            "Accept-Encoding": "none",
            "Accept-Language": "en-US,en;q=0.8",
            "Connection": "keep-alive",
        },
    )
    img = Image.open(urlopen(req))
    return img


def load_images(urls: list[str]):
    return [read_images_from_url(url) for url in urls]


def display_images_from_urls(n_cols: int = 4, urls: list[str] = []):
    n_rows: int = (len(urls) // n_cols) + 1
    plt.figure(figsize=(20, 10))
    images = load_images(urls=urls)
    for i, image in enumerate(images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.axis("off")


if __name__ == "__main__":
    display_images_from_urls(
        urls=[
            "http://images.amazon.com/images/P/0195153448",
            "http://images.amazon.com/images/P/0002005018.0",
            "http://images.amazon.com/images/P/1565120027.jpg",
            "https://images.amazon.com/images/P/8475967027.jpg",
        ]
    )


