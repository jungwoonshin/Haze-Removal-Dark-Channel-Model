from functions import *
import glob

def deHaze(filename, patchsize):

	dest = 'MyResults0.95/'
	print 'Loading Image', filename
	I = readIm(filename)
	# print 'size of i', I.shape
	I = preprocessImage(I)
	# This is newer version code
	# print 'getting dark channel'
	J = makeDarkChannel(I, patchsize)
	cv2.imwrite(dest + fileName + '_dark.jpg', J * 255.0)

	print 'getting atmospheric light'
	height, width = J.shape
	numBrightestPixels = int(math.ceil(0.001 * height*width))

	# print 'numPixels', numBrightestPixels

	A = getA(I, J,numBrightestPixels)

	# print 'A shape' , A.shape
	# print A[0:8,0:8,0]
	# print A[0:8,0:8,2]

	# T_est = np.zeros((height, width))

	aerialPerspective = -0.95

	# np.array(())


	T_est = np.add(1,np.multiply(aerialPerspective, makeDarkChannel(np.divide(I,A),patchsize)))
	cv2.imwrite(dest + fileName + '_trans.jpg', T_est * 255.0)

	# print 'T_est size', T_est.shape
	# print 'T_est added\n', T_est[0:2,0:2]

	dehazed = np.zeros((height, width,3))

	# print 'dehazed.shape', dehazed.shape

	for c in xrange(0, 3):

		max_ndarray = np.zeros((height,width))
		max_ndarray.fill(0.1)

		# print 'max_ndarray.shape', max_ndarray.shape
		# print 'T_est.shape', T_est.shape

		maxed = np.maximum(T_est, max_ndarray)

		# print 'maxed.shape', maxed.shape



		subtracted_val = np.subtract(I[:,:,c],A[:,:,c])

		# print 'subtracted_val', subtracted_val
		# print 'subtracted_val.shape', subtracted_val.shape
		res = np.divide(subtracted_val, maxed)
		
		# print 'res.shape', res.shape
		# print 'A[:,:,c].shape', A[:,:,c].shape
		dehazed[:,:,c] = np.add(res, A[:,:,c])

	I_out = dehazed
	cv2.imwrite(dest+ fileName + '_fogremoved.jpg', I_out*255.0 )


	# dehazed = zeros(size(I));

	    # % Equation 16
	    # for c = 1:3
	    #     dehazed(:,:,c) = (I(:,:,c) - A(:,:,c))./(max(T, .1)) + A(:,:,c);
	    # end
	    
	#     I_out = dehazed;



def getA(I, J,numPixels):

	brightestJ = np.zeros((numPixels,3))
	x_dim, y_dim = J.shape

	# print 'x_Dim', x_dim
	# print 'y_dim', y_dim
	# print 'J[0,0]', J[0,0]
	# print 'J[0,1]', J[0,1]

	# print 'brightestJ.shape', brightestJ.shape

	min_index = np.argmin(brightestJ[:,2])
	min_element = np.amin(brightestJ[:,2])
	# print 'min_index',min_index
	# print 'min_element',min_element

	for i in xrange(0,x_dim):
		for j in xrange(0,y_dim):

			min_index = np.argmin(brightestJ[:,2])
			min_element = np.amin(brightestJ[:,2])
			if J[i,j] > min_element:
				# print 'np.array([i,j,J[i,J]])', np.array([i+1,j+1,J[i,j]])
				brightestJ[min_index,:] = np.array([i+1,j+1,J[i,j]])

	# print 'brightestJ\n', brightestJ[0:20,:]

	highestIntensity = np.zeros((1,1,3))
	# print 'hieghestIntensity size', highestIntensity.shape
	# print 'I shape', I.shape

	for i in xrange(0,numPixels):
		x = brightestJ[i,0]
		y = brightestJ[i,1]

		intensity = np.sum(I[x,y,:])
		if intensity >  np.sum(highestIntensity):
			highestIntensity[:,:,:] = I[x,y,:]

	__, __, dimI = I.shape
	# print 'dimI' , dimI
	# print 'hieghestIntensity size', highestIntensity.shape

	if dimI == 3:
		A = np.zeros((x_dim, y_dim,3))

		for a in xrange(0, 3):
			A[:, :, a]  = A[:, :, a] + highestIntensity[:,:,a]
			 # highestIntensity[:,:,a]

	else:
		A = np.zeros((x_dim,y_dim))
		A[:,:] = A[:,:] + highestIntensity[:,:]

	tmp = np.array(A[:,:,0])
	A[:,:,0] = A[:,:,2]
	A[:,:,2] = tmp


	return A


def makeDarkChannel(I, patchsize):
	height, width, channels = I.shape
	J = np.zeros((height, width))
	padsize = math.floor(patchsize/2) 
	# print 'padsize', padsize
	# print 'I size', I.shape

	# I_R = I[:,:,0]
	# I_G = I[:,:,1]
	# I_B = I[:,:,2]
	I_R = np.pad(I[:,:,2], (padsize, padsize),  'symmetric')
	I_G = np.pad(I[:,:,1], (padsize, padsize),  'symmetric')
	I_B = np.pad(I[:,:,0], (padsize, padsize),  'symmetric')
	# print 'I_R shape' , I_R.shape
	# print 'I_G shape' , I_G.shape
	# print 'I_B shape' , I_B.shape

	I = np.zeros((height+padsize*2,width+padsize*2, 3))
	# print 'I size', I.shape

	I[:,:,0] = I_R 
	I[:,:,1] = I_G
	I[:,:,2] = I_B 

	# print 'I size', I.shape 

	# print(I[0:7,0:3,0])
	# print('\n')
	# print(I[0:7,0:3,1])
	# print('\n')
	# print(I[0:7,0:3,2])
	# print(I[0,1,0])
	# print(I[1,0,0])
	# print(I)

	# print 'height', height
	# print 'width', width

	tmpPatch = np.zeros((padsize, padsize, channels))

	for i in xrange(0, height):
		minX = i
		maxX = (i+padsize*2)

		for j in xrange(0, width):
			minY = j
			maxY = (j+ padsize*2)

			tmpPatch = I[minX:maxX,minY:maxY,:]
			# print 'tmpPatch', tmpPatch
			J[i,j] = np.amin(tmpPatch[:])



	# print 'J', J[0:7,0:3]
	# print 'J.size', J.shape
	return J


def preprocessImage(I):
	max_of_I = np.amax(I)
	# print 'max of i', max_of_I

	if max_of_I > 768:
		scale = 768 / max_of_I
		I = imresize(I, scale)
	I = I / 255.0

	# make gray scales to color 
	height, width, channels = I.shape
	if channels == 2:
		# print 'now changing grayscale image into color image version'
		tmpI = np.zeros((height,width,3))
		for c in xrange(0,3):
			tmpI[:,:,c] = I
		I = tmpI

	return I


def darkChannel(im2=None, *args, **kwargs):
    
    height, width, __ = im2.shape
    # print('darkchannel height', height)
    # print('darkchannel width', width)
    # print(im2.shape)
    patchSize = 3
    padSize = math.floor(patchSize / 2.0)
    JDark = np.zeros((height,width))
    imJ = np.pad(im2, (padSize, padSize), 'constant', constant_values=(0, 0))

    for j in xrange(0, height):
      for i in xrange(0, width):
         patch = imJ[j:(j + patchSize - 1), i:(i + patchSize - 1), :]
         JDark[j, i] = np.amin(patch[:])

    return JDark


# fileName = 'IMG_8763'
# deHaze('Dataset/' + fileName + '.jpg',3)

# fileName = 'IMG_8763'
# deHaze('Dataset/' + fileName + '.jpg',3)



listdir = glob.glob("./Dataset/*.jpg")
for i in range(0, len(listdir)):
	fileName =  listdir[i].split('/')[2].split('.')[0]
	deHaze('Dataset/' + fileName + '.jpg',3)





