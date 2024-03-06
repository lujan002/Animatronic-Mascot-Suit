# Junior-Jay Animatronic Mascot 

Developing a wearable, Disney-style, animatronic suit for my university’s mascot with actuated facial features. Integrating seamless control of mouth, eyebrow, eyelid actuation mechanisms using CV facial landmark detection.


## Mechanism Sub-Systems
### Lower Beak Actuation 
![LowerBeakCAD](LowerBeakCAD.png) 
![LowerBeakInstall](LowerBeakInstall.png) 

Simple part which connects an AL extrusion to a servo which rotates open at 25 degrees to open the beak. Two servos will be mounted inside the head, around the cheek area. The AL extrusions on each side will come together to create a frame which is fastened to the foam beak.

### Eyebrow Actuation 
![InnerBrowDistance](InnerBrowDistance.png) 
![EyebrowExpressions](EyebrowExpressions.png) 

Simple 3-bar mechanism controlled by two servos with links attatched. One link acts as slider to allow for movement. Setting servos to different angles rotates the bar connecting the two links (the eyebrow itself)

I tried two computer vision algorithms to detect my expression as “normal” or “angry” depending on distance between the inner brow points.

send_expression_mediapipe.py uses google's mediapipe model
send_expression_dlib.py uses dlib model

### Eyelid Actuation 

(more to come)



