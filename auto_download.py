#!/usr/bin/env python
"""Download example."""

import ee
import ee.mapclient

start = ee.Date('2017-01-01')
finish = ee.Date('2017-03-20')

rectangle = ee.Geometry.Polygon(
  [[-76.6,1.1],[-76.7,1.1],[-76.6,1.2],[-76.7,1.2]])


ee.Initialize()
ee.mapclient.centerMap(-76.6399, 1.1519, 14)
collection = ee.ImageCollection('COPERNICUS/S2')

filteredCollection = collection.filterBounds(rectangle).filterDate(start,finish)

first = filteredCollection.first()

# Get a download URL for an image.
path = first.getDownloadURL({
  'region': '[[-76.6,1.1],[-76.7,1.1],[-76.6,1.2],[-76.7,1.2]]',
  'scale': 10
})
print(path)
