from moviepy.editor import *

startTime = 2
endTime = 8     
# loading video dsa gfg intro video  
clip = VideoFileClip("simon1.mp4")  
      
# getting only first 5 seconds  
finalClip = clip.subclip(startTime, endTime)  
   
# # cutting out some part from the clip 
# clip = clip.cutout(3, 10) 
   
# showing  clip  
# clip.ipython_display(width = 360)
finalClip.write_videofile("cutVideo.mp4")