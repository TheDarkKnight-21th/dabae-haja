import cv2

# ÀÌ¹ÌÁö ÆÄÀÏ °æ·Î
image_path = '/media/dongjae/6E708DD6708DA605/dobae/yolov5/runs/detect/exp/0.png'

# ÅØ½ºÆ® ÆÄÀÏ °æ·Î
result_file = '/media/dongjae/6E708DD6708DA605/dobae/yolov5/runs/detect/exp/labels/0.txt'


# ÀÌ¹ÌÁö ºÒ·¯¿À±
# â
img = cv2.imread(image_path)

print(img.shape)
# ÀÌ¹ÌÁö Å©±â
img_height, img_width = img.shape[:2]
print(img_height,img_width)
# ÅØ½ºÆ® ÆÄÀÏ ¿­±â
with open(result_file, 'r') as f:
    for line in f:
        # °á°ú¿¡¼­ Å¬·¡½º, bounding box ÁÂÇ¥, ½Å·Úµµ ÃßÃâ
        class_name, x1, y1, x2, y2, confidence = line.strip().split()

        # »ó´ëÁÂÇ¥¸¦ Àý´ëÁÂÇ¥·Î º¯È¯
        x1, y1, x2, y2 = int(float(x1) * img_width), int(float(y1) * img_height), int(float(x2) * img_width ), int(float(y2) * img_height)
        width = int(x2/2)-5
        height = int(y2/2)-5
        print(x1, y1, x2, y2)
        # bounding box¿¡ ÇØ´çÇÏ´Â ÀÌ¹ÌÁö ÀÚ¸£±â
        cropped_img = img[y1-height:y1+height,x1-width:x1+width,:]
        print(cropped_img.shape)
        image = cv2.resize(cropped_img, (250, 250), interpolation=cv2.INTER_CUBIC)

        cv2.imshow(class_name, image)
        cv2.waitKey(0)

cv2.destroyAllWindows()