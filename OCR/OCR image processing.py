import cv2
from PIL import Image
import pytesseract
# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


im_file="C:/Users/mf19-14/llama/car with text.jpg"


#im=Image.open(im_file)
#or
img=cv2.imread(im_file) 
cv2.imshow("orgininal image",img)
cv2.waitKey(0)

#image rotations
#im.rotate(90).show()
#im.save("temp/apple.png")


# Inverted Images (black to white and white to black)
inverted_image=cv2.bitwise_not(img)
cv2.imwrite(im_file,inverted_image)
cv2.imshow("Inverted image",inverted_image)
cv2.waitKey(0)


#Rescaling



#Binarization

#convert gray scale
def grayscale(image):
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_image=grayscale(img)
cv2.imwrite("temp/gray.jpg",gray_image)
gry=cv2.imread("temp/gray.jpg") 
cv2.imshow("gray scale Image ",gry)
cv2.waitKey(0)


#binary image
thresh, im_bw=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
cv2.imwrite("temp/b2_image.jpg",im_bw)
b=cv2.imread("temp/b2_image.jpg") 
cv2.imshow("binary image",b)
cv2.waitKey(0)


#remove noise 
def noise_removal(image):
    import numpy as np
    kernel=np.ones((1,1),np.uint8)
    image=cv2.dilate(image,kernel,iterations=1)
    kernel=np.ones((1,1),np.uint8)
    image=cv2.erode(image,kernel,iterations=1)
    image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
    image=cv2.medianBlur(image,3)
    return (image)

no_noise=noise_removal(im_bw)
cv2.imwrite("temp/no_noise.jpg",no_noise)
noNoise=cv2.imread("temp/no_noise.jpg") 
cv2.imshow("No Noise Image ",noNoise)
cv2.waitKey(0)



#dilation and Erosion
def thin_font(image):
    import numpy as np
    image=cv2.bitwise_not(image)
    kernel=np.ones((2,2),np.uint8)
    image=cv2.erode(image,kernel,iterations=2)
    image=cv2.bitwise_not(image)
    return (image)

eroded_image=thin_font(no_noise)
cv2.imwrite("temp/eroded_image.jpg",eroded_image)
noNoise=cv2.imread("temp/eroded_image.jpg") 
cv2.imshow("dilation and Erosion",noNoise)
cv2.waitKey(0)



#dilation
def thick_font(image):
    import numpy as np
    image=cv2.bitwise_not(image)
    kernel=np.ones((2,2),np.uint8)
    image=cv2.dilate(image,kernel,iterations=2)
    image=cv2.bitwise_not(image)
    return (image)

eroded_image=thick_font(no_noise)
cv2.imwrite("temp/dilated_image.jpg",eroded_image)
noNoise=cv2.imread("temp/dilated_image.jpg") 
cv2.imshow("Dilated: ",noNoise)
cv2.waitKey(0)


#Removing Borders
def remove_Borders(image):
  contours,heiarchy=cv2.findContours(image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cntSorted=sorted(contours,key=lambda x:cv2.contourArea(x))
  cnt=cntSorted[-1]
  x,y,w,h=cv2.boundingRect(cnt)
  crop=image[y:y+h,x:x+w]
  return (crop)

no_borders=remove_Borders(no_noise)
cv2.imwrite("temp/no_boarder.jpg",no_borders)
noBorder=cv2.imread("temp/no_boarder.jpg") 
cv2.imshow("Dilated: ",noBorder)
cv2.waitKey(0)

imag=Image.open(im_file)
ocr_result=pytesseract.image_to_string(imag)
print(ocr_result)
