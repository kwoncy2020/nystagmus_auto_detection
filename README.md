# Nystagmus auto detetion

first, extract eye center points from specified hospital eye video with U-Net.

second, some processing to calibrate center time series data.

third, find classification to detect nystagmus on patched(or splitted) time series data.

***
* Build custom U-net.
![build_custom_U-net](./images/2.png)

***
* Train results
![train results](./images/3.png)

***
* Segmentation to get center points
![segmentation](./images/1.png)

***
* Make video1
![make video1](./images/4.png)

***
* Analysis time-series-center-points with tool to get nystagmus area
![make video2](./images/5.png)

***
* Build another model to detect nystagmus by deeplearning model
* Make an algorithm to detect nystagmus by experts system
