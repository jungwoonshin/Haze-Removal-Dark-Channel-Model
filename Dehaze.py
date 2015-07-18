from functions import *

def deHaze(filename, patchsize):

	dest = 'MyResults0.90/'
	aerialPerspective = -0.90

	print 'Loading Image', filename
	I = readIm(filename)
	I = preprocessImage(I)
	

	J = makeDarkChannel(I, patchsize)
	# cv2.imwrite(dest + fileName + '_dark.jpg', J * 255.0)
	height, width = J.shape
	numBrightestPixels = int(math.ceil(0.001 * height*width))

	A = getA(I, J,numBrightestPixels)

	T_est = np.add(1,np.multiply(aerialPerspective, makeDarkChannel(np.divide(I,A),patchsize)))
	# cv2.imwrite(dest + fileName + '_trans.jpg', T_est * 255.0)

	dehazed = np.zeros((height, width,3))

	for c in xrange(0, 3):

		max_ndarray = np.zeros((height,width))
		max_ndarray.fill(0.1)
		maxed = np.maximum(T_est, max_ndarray)
		subtracted_val = np.subtract(I[:,:,c],A[:,:,c])
		res = np.divide(subtracted_val, maxed)
		dehazed[:,:,c] = np.add(res, A[:,:,c])

	I_out = dehazed
	print 'Writing Haze-removed Image', filename
	cv2.imwrite(dest+ filename + '_fogremoved.jpg', I_out*255.0 )

listdir = glob.glob("./Dataset/*.jpg")
for i in range(0, len(listdir)):
	fileName =  listdir[i].split('/')[2].split('.')[0]
	deHaze('Dataset/' + fileName + '.jpg',3)





