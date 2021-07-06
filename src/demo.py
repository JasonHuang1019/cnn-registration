from __future__ import print_function
import Registration
import matplotlib.pyplot as plt
from utils.utils import *
import cv2

# designate image path here
# IX_path = '../img/198516.jpg'
# IY_path = '../img/198517.jpg'

IX_path = '../img/a1.jpg'
IY_path = '../img/a2.jpg'


IX = cv2.imread(IX_path)
IY = cv2.imread(IY_path)

#initialize
reg = Registration.CNN()
#register
X, Y, Z = reg.register(IX, IY)
#generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)
cb = checkboard(IX, registered, 11)

plt.figure(0)
plt.subplot(131)
plt.title('reference')
plt.imshow(cv2.cvtColor(IX, cv2.COLOR_BGR2RGB))
plt.subplot(132)
plt.title('registered')
plt.imshow(cv2.cvtColor(registered, cv2.COLOR_BGR2RGB))
plt.subplot(133)
plt.title('checkboard')
plt.imshow(cv2.cvtColor(cb, cv2.COLOR_BGR2RGB))
plt.show()

# generate regsitered image using TPS
registered = tps_warp(Y, Z, IY, IX.shape)

res = np.zeros(shape=(IX.shape[0], IX.shape[1] * 2, 3), dtype=np.uint8)
res[:, :IX.shape[1], :] = IX
res[:, IX.shape[1]:, :] = registered
print("The number of matching points: %d" % len(X))
for i, pnt in enumerate(X):
    src_x = int(pnt[1])
    src_y = int(pnt[0])
    dst_x = int(Z[i][1] + IX.shape[1])
    dst_y = int(Z[i][0])

    cv2.line(res, (src_x, src_y), (dst_x, dst_y), (255, 0, 0), 2)
    cv2.circle(res, (src_x, src_y), 5, (0, 255, 0), -1)
    cv2.circle(res, (dst_x, dst_y), 5, (0, 255, 0), -1)

cv2.imwrite('res.jpg', res)






    
