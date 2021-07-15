# redrawing

Python package provinding tools for artistic interactive applications using AI

Created by ReDrawing Campinas team for the OpenCV AI 2021 Competition.

## Stages

Stages are the basic processing units of the redrawing package. They receive data objects from input channels, processes them and output data in the output channels.

Examples of stages:

Stage name | Use
--- | ---
OAK_Stage | Handles a OAK camera. Can be changed using OAK_Substages, like a Body detector or a Hand gesture detector
PCR_Viewer | Visualizer for image, depth and body pose data
CameraReceiver/IMUReceiver | Receives data from a smarthphone
UKF_IMU | UKF filter for orientation estimation using IMU data

## Data and communication

All inter stage data communication occours using Data classes.

Examples of data classes:

Data class | Use
--- | ---
BodyPose | Stores body pose data (keypoints)
Image | Stores image data
Depth_Map | Stores depth data
IMU | Stores IMU data

For exchange data with client applications, the UDP_Stage can be used, using UDP protocol with JSON converted messages. We also have client libraries for receiving and decoding data:

- [redrawing_java](https://github.com/ReDrawing/redrawing_java): client library for Java
- ReDrawing for Processing: client library for Processing, encapsulates the redrawing_java library

# Authors
- [Artemis Sanchez Moroni](https://github.com/ArtemisMoroni)
- [Cássio Gião Dezzotti](https://github.com/cassiodezotti)
- [Daniele Souza Gonçalves](https://github.com/danielegsouza)
- [Elton Cardoso do Nascimento](https://github.com/EltonCN)
- [Gabriel Tadao Kuae](https://github.com/kuta-ga)
- [Jônatas Manzolli]()
- [Marcela Medicina](https://github.com/mmedicina)
- [Pedro Victor Vieira de Paiva](https://github.com/enemy537)
- [Tiago Fernandes Tavares](https://github.com/tiagoft)
- [Thiago Danilo Silva de Lacerda](https://github.com/ThiagoDSL)