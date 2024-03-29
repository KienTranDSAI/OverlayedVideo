from python_color_transfer.color_transfer import ColorTransfer
import cv2
img_arr_in = cv2.imread('first.png')
img_arr_ref = cv2.imread('second.png')

# Initialize the class
PT = ColorTransfer()

# Pdf transfer
img_arr_pdf_reg = PT.pdf_transfer(img_arr_in=img_arr_in,
                                  img_arr_ref=img_arr_ref[400:500,900:1000],
                                  regrain=True)
# Mean std transfer
img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in,
                                  img_arr_ref=img_arr_ref[400:500,900:1000])
# Lab mean transfer
img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref[400:500,900:1000])
cv2.imshow("in", img_arr_in)
# img_arr_ref[400:500,900:1000] = 0
cv2.imshow("base", img_arr_ref)
cv2.imshow("a", img_arr_pdf_reg)
cv2.imshow("b", img_arr_mt)
cv2.imshow("c", img_arr_lt)



cv2.waitKey()
cv2.DestroyAllWindows()
# Save the example results
# img_name = Path(input_image).stem
# for method, img in [('pdf-reg', img_arr_pdf_reg), ('mt', img_arr_mt),
#                    ('lt', img_arr_lt)]:
#     cv2.imwrite(f'{img_name}_{method}.jpg', img)