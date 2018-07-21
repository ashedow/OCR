from __future__ import print_function
import struct

def extractOneImage(images, index, length):
  result = []
  for i in range(index*length, index*length + length):
    result.append(images[i])
  return tuple(result)

def read_examples(file_path, how_many):
  with open(file_path, "rb") as source:
    meta = struct.unpack('>4i', source.read(16))
    magicNumber = meta[0]
    numberOfItems = meta[1]
    if how_many > numberOfItems:
      how_many = numberOfItems
    rows = meta[2]
    cols = meta[3]
    imageSize = rows*cols

    print("Images: {}".format(numberOfItems))
    print("Image size: {} x {} = {}px".format(rows, cols, imageSize))
    print("Reading...")

    structFmtStr = '>' + str(imageSize*numberOfItems) + 'B'
    allImages = struct.unpack(structFmtStr, source.read(imageSize*numberOfItems))
    print("Successfully read {} examples!".format(how_many))

    result = []
    for i in range(0, how_many):
      result.append(extractOneImage(allImages, i, imageSize))
    return result 

def read_labels(file_path, how_many):
  with open(file_path, "rb") as source:
    meta = struct.unpack('>2i', source.read(8))
    magicNumber = meta[0]
    numberOfItems = meta[1]
    if how_many > numberOfItems:
      how_many = numberOfItems
    allLabels = struct.unpack('>' + str(how_many) + 'B', source.read(how_many))
    print("Successfully read {} labels!".format(how_many))
    return allLabels

