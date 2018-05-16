import numpy as np

# Imports for visualization
import cv2
from io import StringIO
import urllib.request

# Imports for pulling metadata from imgur url
import requests
from bs4 import BeautifulSoup

# All images are returned as OpenCV images, not numpy arrays
def loadImageGrayscale(img_file):
  """Load image from file, convert to grayscale float32 numpy array"""
  img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

  # Convert to grayscale and return
  return img.convert("L")

def loadImageFromURL(url, max_size_bytes=4000000):
  """Load image from url.

  If the url has more data than max_size_bytes, fail out
  Try and update with metadata url link if an imgur link"""
  
  # If imgur try to load from metadata
  url = tryUpdateImgurURL(url)

  # Try loading image from url directly
  try:
    req = urllib.request.Request(url, headers={'User-Agent' : "TensorFlow Chessbot"})
    con = urllib.request.urlopen(req)
    # Load up to max_size_bytes of data from url
    data = con.read(max_size_bytes)
    # If there is more, image is too big, skip
    if con.read(1) != '':
      print("Skipping, url data larger than %d bytes" % max_size_bytes)
      return None, url

    # Process into OpenCV image
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    # Return OpenCV image and url used
    return img, url
  except IOError as e:
    # Return None on failure to load image from url
    return None, url

def tryUpdateImgurURL(url):
  """Try to get actual image url from imgur metadata"""
  if 'imgur' not in url: # Only attempt on urls that have imgur in it
    return url

  soup = BeautifulSoup(requests.get(url).content, "lxml")
  
  # Get metadata tags
  meta = soup.find_all('meta')
  # Get the specific tag, ex.
  # <meta content="https://i.imgur.com/bStt0Fuh.jpg" name="twitter:image"/>
  tags = list(filter(lambda tag: 'name' in tag.attrs and tag.attrs['name'] == "twitter:image", meta))
  
  if tags:
    # Replace url with metadata url
    url = tags[0]['content']
  
  return url

def loadImageFromPath(img_path):
  """Load OpenCV image from image filepath, keep as color"""
  return cv2.imread(img_path, cv2.IMREAD_COLOR)


def resizeAsNeeded(img, max_size=(2000,2000), max_fail_size=(2000,2000)):
  if type(img) == numpy.ndarray:
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

  # If image is larger than fail size, don't try resizing and give up
  if img.size[0] > max_fail_size[0] or img.size[1] > max_fail_size[1]:
    return None

  """Resize if image larger than max size"""
  if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
    print("Image too big (%d x %d)" % (img.size[0], img.size[1]))
    new_size = np.min(max_size) # px
    if img.size[0] > img.size[1]:
      # resize by width to new limit
      ratio = np.float(new_size) / img.size[0]
    else:
      # resize by height
      ratio = np.float(new_size) / img.size[1]
    print("Reducing by factor of %.2g" % (1./ratio))
    new_size = (np.array(img.size) * ratio).astype(int)
    print("New size: (%d x %d)" % (new_size[0], new_size[1]))
    img = cv2.resize(img, new_size)
  return img

def getVisualizeLink(corners, url):
  """Return online link to visualize found corners for url"""
  encoded_url = urllib.request.quote(url, safe='')
  
  return ("http://tetration.xyz/tensorflow_chessbot/overlay_chessboard.html?%d,%d,%d,%d,%s" % 
    (corners[0], corners[1], corners[2], corners[3], encoded_url))
