# Junior-Jay Animatronic Mascot 

Developing a wearable, Disney-style, animatronic suit for my university’s mascot with actuated facial features. Integrating seamless control of mouth, eyebrow, eyelid actuation mechanisms using CV facial landmark detection.

This project was sponsored by the Mechanical Engineering Department at the University of Kansas, with 
the primary client being the KU mascot coach. The initiative aimed to revolutionize the university's 
mascot program by incorporating advanced, expressive features into the Big Jay and Baby Jay mascots. The primary requirements outlined by the client included the ability for the mascots to display a wide range of dynamic, real-time facial expressions, like those seen in professionally 
animated characters. 

## Mechanism Sub-Systems
### Beak  
![Beak](Beak.png)

Goal: Open/close to simulate smile, shock.

In the design of the mechanical mechanisms for our animatronic Jayhawk head, a primary focus was on ensuring robustness and minimizing weight. 

The beak assembly consisted of a lightweight aluminum frame inserted into it to provide rigidity and a point for the servo motors to drive the beak. We selected servos rated for 45 kg·cm of torque, suitable for the high-torque demands of our application. This choice was driven by the requirement to counter beak weight of 0.7 kg (6.87 N), with center of gravity centered 230 mm away (1.58 Nm Torque).

### Eyebrows  
![Eyebrows](Eyebrows.png)
![EyebrowExpressions](EyebrowExpressions.png) 

Goal: Rotate to display nuanced emotions. 

To enable the mascot to dynamically express emotions, precise control over the eyebrows was essential. Given the design constraints that the eyebrow actuation mechanisms must remain concealed, the servos were hidden inside the head. We settled on a design that used two servos and incorporated the eyebrow as a three-bar linkage. The benefit to this design was that there were two pivot locations, enabling the eyebrow to raise and lower itself without changing the angle it is oriented at. The servos were conveniently mounted to the eyelid Al frame. These servos had shaft extensions that protruded through the forehead by way of two small holes. The arms were fastened into these extended shafts, which had press-fit threaded inserts and a star pattern connection interface (figure 5) to ensure the arms would not slip. One arm kept its pivot at a fixed distance, while the other had a slider cutout which allowed for its pivot to translate back and forth. This combined setup allowed for free rotation and translation of the eyebrow. The arms are completely hidden behind the eyebrow and colored the same as the eyebrow for camouflage and redundancy.

<!-- 3-bar linkage consisting of fixed arm, slider arm, and eyebrow, uses servo rotation to pivot eyebrow. 
Servos mounted neatly on eyelid frame inside head, with shafts protruding through forehead that spin links.
Eyebrow orientation is controlled to mimic the detected emotion of wearer (See Expression Control Algorithm) -->

### Eyelids  
![EyelidInstall](Eyelids.png)

Goal: Open/close to simulate blinking

To improve the animatronics' eyelid functionality, a new eye design was developed using SLA 3D 
printing, creating a smooth surface that allows for uninterrupted eyelid movement. This 
process involved vacuum forming a PETG plastic sheet over the 3D printed model to ensure minimal 
material friction and enhance performance. For precise eyelid motion, a compact timing belt system was 
chosen instead of gears or chains due to its reliability and quiet operation, similar to those used in 3D 
printers. The frame was designed to support the eyebrow mechanics while providing enough space for 
the eyelid material to spool without increasing in diameter. Initially, a DC motor was considered for this 
system, but its torque, size, and control limitations led to the selection of a continuous servo motor, 
which offers sufficient torque and simpler control. To compensate for the servo's slower speed, a gear 
system was incorporated to increase the speed of the eyelid movement while maintaining the necessary 
torque for smooth operation. Unlike a normal servo, the continuous servo has no positional feedback. 
To work around this, two limit switches were added so that a the belt system would trip one when the 
eye was fully open, and the other when fully closed, thus shutting off the motors. 

<!-- A continuous servo motor spins a belt-driven mechanism with fabric "eyelids" stitched to the belt. 
A 16:40 gear ratio accelerates the shaft rotation, enabling quick, realistic blinking. 
Limit switches positioned at each extreme of the mechanism are tripped when the eye reaches its fully open or closed state, precisely defining the range of motion. This is essential for the continuous servo, which operates without positional feedback.
Fabric spools at top so wearer’s vision is not obscured when in open position. 
Custom thermoformed plastic coated with one-way mirror film ensures internal components cannot be seen from outside. -->

## Expression Control Algoritm 

I tried two approaches at a computer vision algorithm which detects the wearer's expression as “netral", "angry”, "happy", "surprised", or "sad" depending on distance between the inner brow points. Having discrete facial expressions was chosen as it is easier and more robust to implement a few discrete configurations of the servo motors rather than have total control over each mechanism directly. 

The first approach uses open-source libraries which project dots upon the user's facial landmarks. The distance between these dots have some correlation depending on the facial expression. By creating an arbitrary threshold, we can classify the facial expresion between these dots. For example, a "surprised" face will have a larger distance between the dots corresponding to the eyebrows and the eyes as the eyebrows raise. Likewise, an "angry" face will have a shorter distance between the inner-most eyebrow dots as the eybrows camber inwards. 

Examples of the implementation of this method can be found in these scrips: 
send_expression_mediapipe.py uses google's mediapipe model
send_expression_dlib.py uses dlib model

The better approach I found was to use an image classification model to detect the facial emotion. The model I trained is a CNN trained on the [FER-2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) (Facial Emotion Recognition). The training process is documented in train-emotion-detection.py. I have also attatched the pre-trained model directly as model_optimal2.h5. For implementation, users can run the webcam-emotion-detection.py script, which takes model_optial2.h5 and predicts the emotion class based on the incoming webcam stream. The model performs to just above 70% accuracy on the dataset, but I have experienced better results in actual implementation. This may be due in part to the shoddy labeling job in the original FER dataset. 

![Model Performance](ModelPerfomrance.png) 
![Webcam Expressions](WebcamExpressions.png)

## Electrical System

50,000mAh battery pack chosen for 2hr. lifetime, 5-20V output.  
HUSB238 draws 5V, 9V from battery, custom power bus distributes 9V power to two beak servos, 5V to other six servos. 
Power resistors protect system by limiting current flow.
Raspberry Pi runs facial detection software and sends signal to servos over GPIO pins. 
Optional LED panel installed in head indicates detected emotion to wearer.
 
There were two key considerations when designing this system. This system contains approximately fifty 
loose wires, and it quickly became apparent that wire management was crucial for easy maintenance, 
visibility inside the head, and ensuring the integrity of the sensitive wiring. All wires were strategically 
routed around the internal structure of the head. Wires were systematically braided and 
grouped by sub-system, labeled, and secured to prevent obstruction of vision, reduce tangling risks, and 
simplify maintenance procedures. A Raspberry Pi GPIO pin extension board with screw clamps was used 
and gave much needed easy accessibility when pulling and plugging wires. A power bus was designed 
and implemented that handled the distribution of power to all motors in a convenient location. Since 
these electrical components would be operating very close to a human’s face, safety was also a top 
priority. Initial testing done on the 9V beak servos under high load produced worrying results on the 
multimeter. In some cases, the current draw of an individual servo could spike to well over 1A. To 
prepare for dangerous current spikes like this, two 1Ω power resistors in series were added to the 9V rail 
on the power bus to dissipate any excess current as heat. An emergency safety switch was also added to 
shut the 9V motors off.

![Electronics](Electronics.png)
![Electrical Diagram](ElectricalDiagram.png)

### Voltage Converter PCB

The servos that move the lower beak were chosen for their high torque at operating volages above 5V. The servos used here were rated for 5V - 8.4V, so initially a voltage converter circuit was designed to step down the input voltage to 8V. Later it was realized that the servos could run on 9V directly from the power bank, making this voltage converter irrelevant. Regardless, an short summary of the work done on it will be listed. A [dummy breakout](https://www.adafruit.com/product/5807) was used to draw 12V from the [power bank](https://www.amazon.com/Charging-50000mAh-Portable-Compatible-External/dp/B0C5D1JR2K/ref=asc_df_B0C5D1JR2K/?tag=hyprod-20&linkCode=df0&hvadid=663345862487&hvpos=&hvnetw=g&hvrand=11970480634190944901&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1017537&hvtargid=pla-2188333650925&psc=1&mcid=6e21d341f4533a22af8d0ba4956a6da5), which was further stepped down to 8V by this circuit. 

The schemetic shown below incorporates an [LM317 Voltage Regulator](https://www.ti.com/product/LM317-N/part-details/LM317T/NOPB?HQS=ocb-tistore-invf-buynowlink_partpage-invf-store-snapeda-wwe) ([footprint](https://www.snapeda.com/parts/LM317T/NOPB/Texas%20Instruments/view-part/)) and a series of commonly available resistors.

![Voltage Converter Schematic](voltage_converter_schematic.png) 

The board was designed in Eagle and printed at the school shop.

![Voltage Converter Board](voltage_converter_board.png) 

### Emotion LED Indicator Panel

(more to come)

### Summary of Work 
Overall, a Junior Jay mascot was created with expressive capabilities. Junior Jay is equipped with a 
Raspberry Pi board powered by a 5-volt battery pack, and an external digital camera that acts as an 
input. Another 5-volt battery pack and 9-volt battery pack, attached to a custom power breakout board, 
power the rest of the systems of the mascot head. The jaw is equipped with two 45-kilogram servo 
motors, bracket mounts, and an aluminum frame to move the jaw. The eyelids include an aluminum 
frame and rod system that spools eyelid fabric by means of a servo motor, belts, and switches. The eyes 
are made of vacuum formed plastic with layers of film to provide a one-way mirror window. The 
eyebrows are equipped with two 7 kg micro servo motors each attached to the eyelid frame, links, and 
fabric covered 3D printed eyebrows. Lastly, 4 micro fans are scattered throughout the helmet to provide 
cooling. 

The culmination of these mechanical systems along with a Raspberry Pi Board, camera, and batteries 
allows for a mascot head that mirrors multiple facial expressions of the camera user. The eyebrows and 
jaw of junior jaw worked seamlessly and realistically, with a battery life of about 4 hours. The facial 
recognition along with the camera were able to detect and interpret facial expressions of most camera 
users with a response time of 1 second. The operation of the mascot head was comfortable and was 
adequately cooled. The one-way mirror window design of the eye’s allowed for proper vision and use. 
Batteries were stored in a backpack worn by the mascot user, which also allowed Junior Jay to be 
mobile. After testing Junior Jay, the final operating time was close to 3 hours without any issues. While 
most of the mascot head worked, the eyelids had a short life cycle along with dysfunctional code and 
were not powered on for the final exposition. 
In short, Junior Jay worked well and met all engineering objectives. Junior Jay performed facial 
expression animation for various human emotions, involved facial recognition of camera users, included 
a cooling system, was durable and comfortable enough to operate for several hours, and could be easily 
operated. 



### Future Work

While the project has made significant strides in enhancing the functionality and appeal of the university 
mascots, there remains scope for further development to fully realize the envisioned goals. Future work 
could focus on several areas to refine and expand the project's achievements: 

1.  Refinement of Mechanical Systems: Further refinement of the mechanical systems controlling 
facial expressions is essential. Continuous testing and iteration could lead to more fluid and 
natural movements, reducing any mechanical lag or stiffness currently present. 
2.  Advanced AI and Machine Learning: Implementing more sophisticated AI algorithms and 
machine learning techniques, such as training on a larger dataset or a custom dataset built to 
our specific needs, could enhance the accuracy and responsiveness of the facial recognition 
system. Also, work can be done to give more nuanced expressions prediction rather than the 
simpler discrete expression approach used. This would ensure a more precise and seamless 
translation of the performer’s expressions to the mascot. 
3.  Enhanced Cooling Technologies: Although the current cooling system provides basic relief, 
exploring advanced technologies like phase change materials or more efficient heat exchange 
mechanisms could offer better temperature regulation, increasing comfort during longer 
performances or in more extreme weather. 
4.  Integration of Interactive Technologies: Adding interactive technologies such as voice 
modulation and responsive sound systems could make the mascots even more engaging. For 
instance, allowing the mascots to respond vocally to crowd interactions or integrate sound 
effects linked to their movements could greatly enhance audience interaction. 
5.  Sustainability and Materials Science: Investigating more sustainable materials and construction 
techniques that maintain durability while reducing weight could improve the ergonomics and 
environmental impact of the mascot costumes. 
6.  Extended Wearability Studies: Conducting extended wearability studies to gather data on the 
performer's comfort and suit ergonomics over longer periods can provide insights into necessary 
adjustments and improvements. 
7.  Scalability and Transferability: Finally, considering how these enhancements can be 
standardized and transferred to other mascot programs could broaden the impact of the 
project. Developing a modular system that can be customized and adapted to different mascots 
could make these advancements more accessible to other institutions. 
8.  Foolproof electrical safety measures: A mechanical cover should be added to cover the power 
bus to protect against accidental contact and reduce the risk of short circuits. Additionally, 
wiring may be optimally routed inside the head wall to minimize exposure and vulnerability. 
Safety devices such as fuses, and micro-circuit breakers will be incorporated to prevent electrical 
overloads and potential hazards.  

These areas of future work would not only address the remaining challenges but also push the 
boundaries of what is possible in mascot design and performance, potentially setting new industry 
standards. 
 
 
