# UT volumetric classification
3D classification for ultrasonic data. Ultrasonic volumetric data has highly stretched aspect ratios, which makes it unsuitable for existing volumetric classification architectures. Two hand designed 3D-CNN architectures were tested. One with downsampling layers in the time domain then constant feature extraction layers. The other with combined, feature extraction and reduction layers. In addition, Neual Architecture Search was performed to explore the architecture search space within a ResNet based model modified for 3D.

Architecture 1:

![image](https://user-images.githubusercontent.com/71640417/226601695-47973044-1db8-4463-b72f-711bf329d651.png)

Architecture 2:

![image](https://user-images.githubusercontent.com/71640417/226602983-2a1d40fd-81b6-4c14-bfbb-1582ebc4a71a.png)


Example of volumetric experimental test data:

![image](https://user-images.githubusercontent.com/71640417/223426565-bb516ad8-1251-4585-a9f2-31ecc064950b.png)

Example of correct classification interpretability with 3D Guided-Grad CAM: 

![image](https://user-images.githubusercontent.com/71640417/223426577-e5580132-b64f-405d-a0e3-f7a5b5a3db09.png)



