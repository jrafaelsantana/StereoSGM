import cv2
import sgmgpu
sgmgpu.initDisp(24,96)

for i in range(0, 1):
    path = "../../extras/dataset_generator_from_real/rect/"
    
    #full_path_r = path + str(i) + "_c0.png"
    #full_path_l = path + str(i) + "_c1.png"

    full_path_l = "imgL.png"
    full_path_r = "imgR.png"

    #print(full_path_l)

    imgL = cv2.imread(full_path_l,0)
    imgR = cv2.imread(full_path_r,0)
    #cv2.imshow("teste",imgL)
    #cv2.waitKey(0)

    disp = sgmgpu.dispCalc(imgL,imgR)
    cv2.imshow("teste",disp)
    cv2.waitKey(0)
#finish_disparity_method();
#disp = sgmgpu.dispCalc(imgL,imgR)
#cv2.imshow("teste",disp)
#cv2.waitKey(0)
#disp = sgmgpu.dispCalc(imgL,imgR)
#cv2.imshow("teste",disp)
#cv2.waitKey(0)
#disp = sgmgpu.dispCalc(imgL,imgR)
#cv2.imshow("teste",disp)
#cv2.waitKey(0)
#cv2.imshow("teste",disp)
#cv2.waitKey(0)
#input()

