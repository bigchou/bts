import cv2

name = '20230822_230014-00.00.22.922-00.01.09.469'
#name = 'change lane_20230822_224815-00.01.13.828-00.03.00.000'
#name = 'alotoftrafficcone_20230822_224515-00.00.50.625-00.02.30.609-00.00.15.088-00.00.20.380'

cap = cv2.VideoCapture(name+'.mp4')


fps = 30

#clip_list_origin = []
clip_list = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    #cv2.imwrite('before.png', frame)

    invalid_region = 70 # <--- 580 setting
    #invalid_region = 100
    kitti_height = 352
    ratio = 0.63
    width_new = 1216
    height, width = frame.shape[0], frame.shape[1]
    after = cv2.resize(frame, (width_new, int(height * width_new / width)))
    
    
    
    #print(after.shape)

    height_new = after.shape[0]
    # kitti_height = 352
    # invalid_region = 70
    # height_new = 684
    height_new = height_new - kitti_height - invalid_region
    # height_new = 684 - 352 - 70 = 262
    after = after[height_new:-invalid_region, :, :]
    #print(after.shape)

    #cv2.imwrite('after.png', after)
    #exit()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)
    #frame = cv2.resize(frame, (1216, 352))
    clip_list.append(after)
    #clip_list_origin.append(frame)

    #if len(clip_list) == fps*5: break
    
    #if cv2.waitKey(1) == ord('q'): break
cap.release()
#cv2.destroyAllWindows()


out = cv2.VideoWriter(name+'resize.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1216, 352))
for clip in clip_list:
    out.write(clip)
out.release()



'''
out = cv2.VideoWriter(name+'origin.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1920, 1080))
for clip in clip_list_origin:
    out.write(clip)
out.release()
'''