from redrawing.data_interfaces.java import Data2Java
from redrawing.data_interfaces.bodypose import BodyPose

dj2 = Data2Java()
dj2.add_data_class(BodyPose)
print(dj2.createJavaClass(BodyPose))

