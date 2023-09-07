import cv2

#name = '20230822_230014-00.00.22.922-00.01.09.469'
name = 'alotoftrafficcone_20230822_224515-00.00.50.625-00.02.30.609-00.00.15.088-00.00.20.380'
cap = cv2.VideoCapture(name+'.mp4')


fps = 30

clip_list = []
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('fwerwerwerwer.png', frame)
    exit()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('frame', gray)
    frame = cv2.resize(frame, (1216, 352))
    clip_list.append(frame)

    if len(clip_list) == fps*5: break
    
    #if cv2.waitKey(1) == ord('q'): break
cap.release()
#cv2.destroyAllWindows()


out = cv2.VideoWriter(name+'resize.mp4', cv2.VideoWriter_fourcc(*'MP4V'), fps, (1216, 352))
for clip in clip_list:
    out.write(clip)
out.release()
