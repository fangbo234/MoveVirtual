step 1 : Object check with yolov10,person recognition is needed if this is used in sport live broadcast
step 2 : pose check with mediapipe
step 3 : depth estimate with intel/dpt-large
step 4: build scene with scene recontruction
step 5: build model with pose lanmark ,in sport live broadcast,we can build realistic model ,for example, KD,Stepen curry,LBJ
step 6: put model to scene with depth data
this is what need to do to deal one frame of video,if pose landmark data is incomplete,use pose estimate to build complete pose landmark
if pose landmark data in frame 1 is much different with model2,use some motion optimization algorithm.
