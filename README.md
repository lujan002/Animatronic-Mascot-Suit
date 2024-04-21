# Junior-Jay Animatronic Mascot 

Developing a wearable, Disney-style, animatronic suit for my university’s mascot with actuated facial features. Integrating seamless control of mouth, eyebrow, eyelid actuation mechanisms using CV facial landmark detection.


## Mechanism Sub-Systems
### Beak  
![BeakCAD](BeakCAD.png) 
![BeakInstall](BeakInstall.png) 

Goal: Open/close to simulate smile or shock

Two high torque servo motors mounted in the cheek area move a lightweight Aluminum frame that is fixed inside lower beak. Frame is made up of mini Al extrusions held together by 3D printed brackets. Beak has approximately 15 degree range of motion. 


### Eyebrows  
![EyebrowInstall](EyebrowInstall.png)
![EyebrowExpressions](EyebrowExpressions.png) 

Goal: Display nuanced angles that reflect variety of emotions

Simple 3-bar linkage controlled by two hidden servos mounted inside the head. Shafts protruding through forehead spin attatchedm links. One link acts as slider to allow for full range of movement. Setting servos to different angles rotates the eyebrow (the "third bar") connecting the two links.

Eyebrow orientation is controlled to mimic the detected emotion of wearer (See Expression Control Algorithm)

### Eyelids  
![EyelidCAD](EyelidCAD.png)
![EyelidInstall](EyelidInstall.png)

Goal: Open/close to simulate blinking

A continuous servo motor spins a belt-driven mechanism with fabric "eyelids" stitched to the belt. A 16:40 gear ratio accelerates the shaft rotation, enabling quick, realistic blinking. Limit switches positioned at each extreme of the mechanism activate when the eye reaches its fully open or closed state, precisely defining the range of motion. This is essential for the continuous servo, which operates without positional feedback.


## Expression Control Algoritm 

I tried two approaches at a computer vision algorithm which detects the wearer's expression as “netral", "angry”, "happy", "surprised", or "sad" depending on distance between the inner brow points. Having discrete facial expressions was chosen as it is easier and more robust to implement a few discrete configurations of the servo motors rather than have total control over each mechanism directly. 

The first approach uses open-source libraries which project dots upon the user's facial landmarks. The distance between these dots have some correlation depending on the facial expression. By creating an arbitrary threshold, we can classify the facial expresion between these dots. For example, a "surprised" face will have a larger distance between the dots corresponding to the eyebrows and the eyes as the eyebrows raise. Likewise, an "angry" face will have a shorter distance between the inner-most eyebrow dots as the eybrows camber inwards. 

Examples of the implementation of this method can be found in these scrips: 
send_expression_mediapipe.py uses google's mediapipe model
send_expression_dlib.py uses dlib model

The better approach I found was to use an image classification model to detect the facial emotion. The model I trained is a CNN trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) (Facial Emotion Recognition). The training process is documented in train-emotion-detection.py. I have also attatched the pre-trained model directly as model_optimal2.h5. For implementation, users can run the webcam-emotion-detection.py script, which takes model_optial2.h5 and predicts the emotion class based on the incoming webcam stream. The model performs to just above 70% accuracy on the dataset, but I have experienced better results in actual implementation. This may be due in part to the shoddy labeling job in the original FER dataset. 

![Model Performance](JJ Code\model3_performance.png) 

## Electrical System

50,000 mAh battery. 3A limit. 2 USB-A ports, 1 USB-C port. 
Raspberry Pi sends signal to servos, LED indicators over GPIO pins.

(More to come)

### Voltage Converter PCB

The servos that move the lower beak were chosen for their high torque at operating volages above 5V. The servos used here were rated for 5V - 8.4V, so a voltage converter circuit was designed to step down the input voltage to 8V. A [dummy breakout](https://www.adafruit.com/product/5807) was used to draw 12V from the [power bank](https://www.amazon.com/Charging-50000mAh-Portable-Compatible-External/dp/B0C5D1JR2K/ref=asc_df_B0C5D1JR2K/?tag=hyprod-20&linkCode=df0&hvadid=663345862487&hvpos=&hvnetw=g&hvrand=11970480634190944901&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1017537&hvtargid=pla-2188333650925&psc=1&mcid=6e21d341f4533a22af8d0ba4956a6da5), which was further stepped down to 8V by this circuit. 

The schemetic shown below incorporates an [LM317 Voltage Regulator](https://www.ti.com/product/LM317-N/part-details/LM317T/NOPB?HQS=ocb-tistore-invf-buynowlink_partpage-invf-store-snapeda-wwe) ([footprint](https://www.snapeda.com/parts/LM317T/NOPB/Texas%20Instruments/view-part/)) and a series of commonly available resistors.

![Voltage Converter Schematic](voltage_converter_schematic.png) 

The board was designed in Eagle and printed at the school shop.

![Voltage Converter Board](voltage_converter_board.png) 

### Emotion LED Indicator Panel

(more to come)
