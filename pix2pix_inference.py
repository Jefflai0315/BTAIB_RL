import tensorflow as tf
import os
from matplotlib import pyplot as plt
from IPython import display
import cv2
import numpy as np
import random
import os

import cv2
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import json
from shapely.geometry import mapping
from shapely.affinity import scale, translate

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
# IMG_WIDTH = 256
# IMG_HEIGHT = 256
value_list = [1024]
output =[]
for i in range(3):
   
    # randomly choose value from a list 
    value = random.choice(value_list)
    print(value,'round')
    IMG_HEIGHT = value
    IMG_WIDTH = value


    def load_and_preprocess_image(file_path):
        # Read the image
        image = cv2.imread(file_path)
        
        # Determine the target size (256x256)
        # target_size = (256, 256)
        target_size = (IMG_WIDTH, IMG_HEIGHT)
        
        # Calculate padding values
        height, width, _ = image.shape
        max_dim = max(height, width)
        pad_height = max_dim - height
        pad_width = max_dim - width
        
        # Create a constant white color image
        white_color = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
        
        # Insert the original image into the center of the white image
        white_color[pad_height//2:pad_height//2+height, pad_width//2:pad_width//2+width, :] = image
        
        # Resize the padded image to 256x256
        resized_image = cv2.resize(white_color, target_size)
        
        return resized_image

    def pad_and_resize(image , target_size=(256, 256), fill_color=(255, 255, 255)):
        # Determine the size of the padding
        height, width,channel = image.shape
        print('hw',height,width)
        padding_size = abs(height - width) // 2
        # max_dim = max(height, width)
        # pad_height = max_dim - height
        # pad_width = max_dim - width
        max_dim = max(height, width)
        pad_height = max_dim - height
        pad_width = max_dim - width
        white_color = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
        white_color[pad_height//2:pad_height//2+height, pad_width//2:pad_width//2+width, :] = image

    def resize(input_image, real_image, height, width):
        input_image = tf.image.resize(input_image, [height, width],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        real_image = tf.image.resize(real_image, [height, width],
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return input_image, real_image

    # Normalizing the images to [-1, 1]
    def normalize(input_image, real_image):
        input_image = (input_image / 127.5) - 1
        real_image = (real_image / 127.5) - 1

        return input_image, real_image

    def load(image_file):
        print('image',image_file)
        
        # Read and decode an image file to a uint8 tensor
        image_f = tf.io.read_file(image_file)
        print(image_f)
        image = tf.io.decode_jpeg(image_f)

    

        w = tf.shape(image)[1]
        w = w // 2
        input_image = image[:, :w, :]
        real_image = image[:, w:, :]
        input_image = tf.image.resize(input_image,(IMG_WIDTH,IMG_HEIGHT)) #reduce
        real_image = tf.image.resize(real_image,(IMG_WIDTH,IMG_HEIGHT))
        # pad and resize to 256 256 
        # input_image = pad_and_resize(input_image)
        # real_image = pad_and_resize(real_image)

        # Convert both images to float32 tensors
        input_image = tf.cast(input_image, tf.float32)
        real_image = tf.cast(real_image, tf.float32)

        return input_image, real_image
    
    def load_image_test(image_file):
        input_image, real_image = load(image_file)
        
        input_image, real_image = resize(input_image, real_image,
                                        IMG_HEIGHT, IMG_WIDTH)
        input_image, real_image = normalize(input_image, real_image)

        return input_image, real_image

    def random_crop(input_image, real_image):
        stacked_image = tf.stack([input_image, real_image], axis=0)
        cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

        return cropped_image[0], cropped_image[1]   

    @tf.function()
    def random_jitter(input_image, real_image):
    # Resizing to 286x286
        input_image, real_image = resize(input_image, real_image, 286, 286)

        # Random cropping back to 256x256
        input_image, real_image = random_crop(input_image, real_image)

        if tf.random.uniform(()) > 0.5:
            # Random mirroring
            input_image = tf.image.flip_left_right(input_image)
            real_image = tf.image.flip_left_right(real_image)

        return input_image, real_image

    def generate_images_test(model, test_input, output_path):
        prediction = model(test_input, training=True)
        plt.figure(figsize=(30, 30))

        display_list = [prediction[0]]
        title = [ 'Predicted Image']

        for i in range(1):
            plt.subplot(1, 1, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
            plt.savefig(output_path, dpi=300,bbox_inches='tight')
        # plt.show()
        # cv2.imwrite('/Users/jefflai/SpaDS/house_diffusion/data_clavon/prediction.png', prediction[0])
        # plt.savefig('/Users/jefflai/SpaDS/house_diffusion/data_clavon/prediction.png', dpi=300,bbox_inches='tight')
        return prediction[0]

    OUTPUT_CHANNELS = 3
    def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())

        return result

    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())

        result.add(tf.keras.layers.LeakyReLU())

        return result

    def Generator():
        inputs = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3])

        down_stack = [
            downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
            downsample(128, 4),  # (batch_size, 64, 64, 128)
            downsample(256, 4),  # (batch_size, 32, 32, 256)
            downsample(512, 4),  # (batch_size, 16, 16, 512)
            downsample(512, 4),  # (batch_size, 8, 8, 512)
            downsample(512, 4),  # (batch_size, 4, 4, 512)
            downsample(512, 4),  # (batch_size, 2, 2, 512)
            downsample(512, 4),  # (batch_size, 1, 1, 512)
        ]

        up_stack = [
            upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
            upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
            upsample(512, 4),  # (batch_size, 16, 16, 1024)
            upsample(256, 4),  # (batch_size, 32, 32, 512)
            upsample(128, 4),  # (batch_size, 64, 64, 256)
            upsample(64, 4),  # (batch_size, 128, 128, 128)
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                                strides=2,
                                                padding='same',
                                                kernel_initializer=initializer,
                                                activation='tanh')  # (batch_size, 256, 256, 3)

        x = inputs

        # Downsampling through the model
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)

        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
    def Discriminator():
        initializer = tf.random_normal_initializer(0., 0.02)

        inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='input_image')
        tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, 3], name='target_image')

        x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

        down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
        down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
        down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

        zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
        conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                        kernel_initializer=initializer,
                                        use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

        batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

        leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

        zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

        last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                        kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

        return tf.keras.Model(inputs=[inp, tar], outputs=last)



# inference


    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    generator = Generator()
    discriminator = Discriminator()
    tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
    tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir ))




    try:
        test_dataset_clavon = tf.data.Dataset.list_files('./data_clavon/test/*')
    except tf.errors.InvalidArgumentError:
        # test_dataset = tf.data.Dataset.list_files(PATH + 'val/*.jpg')
        print('error')
    test_dataset_clavon = test_dataset_clavon.map(load_image_test)
    test_dataset_clavon = test_dataset_clavon.batch(BATCH_SIZE)

    for inp, tar in test_dataset_clavon .take(1):
        pred = generate_images_test(generator, inp, output_path=f'./data_clavon/prediction_{i}.png')


    plt.figure()

    # Load the image
    image = cv2.imread(f'./data_clavon/prediction_{i}.png')

    # Convert to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color mask
    # [Red color range definitions as before]
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Yellow color mask
    lower_yellow = np.array([22, 93, 0])
    upper_yellow = np.array([45, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours for red
    # [Contour finding for red as before]
    # Combine masks
    mask = mask1 + mask2

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    # Assuming the largest 2 contours is the red polygon
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:2]
    red_polygon = []
    for contour in contours:
        epsilon = 0.020 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Convert to Shapely Polygon
        polygon_points = [point[0] for point in approx]
        shapely_polygon = Polygon(polygon_points)
        red_polygon.append(shapely_polygon)
        X, Y = shapely_polygon.exterior.xy
        # Plotting
        plt.plot(X, Y)
        plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color
    


    # Find contours for yellow
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest yellow contour is the target
    yellow_contour = max(yellow_contours, key=cv2.contourArea)
    yellow_contour = np.concatenate(yellow_contour, axis = 0)
    yellow_polygon = Polygon(yellow_contour)

    X, Y = yellow_polygon.exterior.xy
    # Plotting
    plt.plot(X, Y)
    plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color


    x, y, w, h = cv2.boundingRect(yellow_contour)  # Bounding box for yellow

    yellow_bb = Polygon([(x, y), (x+w, y), (x+w, y+h), (x, y+h)]) 
    X, Y = yellow_bb.exterior.xy
    # Plotting
    plt.plot(X, Y)
    plt.fill(X, Y, alpha=0.3)  # Optional: fill the polygon with a semi-transparent color


    # Calculate the relative scale and position of the red polygon
    center_x, center_y = x + w/2, y + h/2
    relative_position = (np.mean([point[0] for point in polygon_points], axis=0) - [center_x, center_y])
    relative_scale = (len(polygon_points) / (w * h))

    # Smoothen the red polygon
    t = np.linspace(0, 1, len(polygon_points), endpoint=False)
    t_interp = np.linspace(0, 1, 500, endpoint=False)
    x, y = zip(*polygon_points)



    # Example Shapely polygons
    clavon_site = yellow_polygon
    bounding_box_site = yellow_bb
    buildings = red_polygon

    # Convert to Lat Long coordinates
    sites_info = {'clavon': {'name': 'clavon',
                            'site_boundary': Polygon(((103.76762137358979, 1.3087990670131122), (103.76695888021099, 1.3091033941901744), (103.76695888021099, 1.3091033941901782), (103.76680089609894, 1.3087280864870512), (103.766792747434, 1.3087078172631332), (103.76678557945053, 1.3086899861164933), (103.76677812454516, 1.3086684317627417), (103.76677122299097, 1.3086453681151176), (103.7667670091718, 1.308629112978794), (103.76676302875474, 1.308610442969295), (103.76675844145478, 1.308585861093), (103.76675564607066, 1.308568042468877), (103.7667533421893, 1.3085446293791023), (103.76675190226392, 1.30852567236893), (103.76675180617868, 1.3085231659641638), (103.76675217242752, 1.3085144072545394), (103.76675207742, 1.3084925885686336), (103.76675194337425, 1.3084860243757694), (103.76675179918865, 1.3084809478597594), (103.7667517106069, 1.3084758790108302), (103.7667516777277, 1.30847081910914), (103.76675170047343, 1.3084657425356094), (103.76675178278158, 1.3084603788857558), (103.76675190539342, 1.3084553122515075), (103.766759415393, 1.3084502477405577), (103.76675234045226, 1.3084448982895578), (103.76675262913469, 1.3084398468137872), (103.76675297264875, 1.30843479804762), (103.76675337164095, 1.308429760924249), (103.7667538265147, 1.3084247131985483), (103.76675436383049, 1.3084194076934133), (103.766754929798, 1.3084143621108462), (103.76675554278873, 1.3084093157891215), (103.76675623518352, 1.3084040881540013), (103.76675695186114, 1.3083991144012195), (103.76675772636361, 1.3083941199926599), (103.7667585981971, 1.3083888850082213), (103.76675947173706, 1.3083839153246697), (103.76676040667252, 1.3083789760982625), (103.76676143482773, 1.3083738029708472), (103.76676252125338, 1.3083686400989731), (103.76676359985571, 1.3083637364392524), (103.76676473694107, 1.3083588713631675), (103.76676598859025, 1.3083537327630061), (103.76676721542269, 1.3083489250335345), (103.7667685661309, 1.30834388081516), (103.76676998123311, 1.3083387781575588), (103.7667713777606, 1.3083339540246992), (103.76677288501628, 1.3083289535549412), (103.76677431495274, 1.3083243728196843), (103.76677594002432, 1.308319412844347), (103.7667776044232, 1.3083144450305626), (103.76677923565617, 1.3083097919766917), (103.76678109463339, 1.3083046375442602), (103.76678287287089, 1.30829985562966), (103.76678465129328, 1.3082952392626963), (103.76678653966118, 1.3082904938452273), (103.7667884690412, 1.3082857914938295), (103.76679050494603, 1.3082809810350113), (103.766792652022, 1.3082760417058086), (103.76679468914062, 1.3082714786919092), (103.76679669853525, 1.308267125485), (103.76679893802313, 1.308262412675837), (103.76680124155497, 1.3082576830310426), (103.76680345530123, 1.3082532600238392), (103.76680579233569, 1.3082487256194255), (103.76680831269016, 1.3082439502444518), (103.76681077662153, 1.308239388683786), (103.76681316405632, 1.3082350862344916), (103.76681367260619, 1.3082341941146542), (103.7670369584743, 1.307915310591265), (103.7670369584743, 1.3079153105912689), (103.76766580058731, 1.308298877127881), (103.76766580058731, 1.3082988771278814), (103.7676428514021, 1.30838189675668), (103.76764179810895, 1.3083863255490304), (103.76762282479181, 1.3084812464296733), (103.7676218261851, 1.3084883025243677), (103.76761371619476, 1.30858474385251), (103.76761349686825, 1.3085895323950083), (103.76761275665643, 1.3086863455250057), (103.76761291779121, 1.3086913506228472), (103.7676198824296, 1.308787898815235), (103.767645505, 1.3087927382936007), (103.76762137358979, 1.3087990670131138))),
                            'site_coverage': 0.25,
                            'building_scale': 18,
                            'postal_code': "129962",
                            "PR": 3.5,
                            "URA_GFA": 62247.2,
                            "URA_site_area": 16542.7,
                            "URA_building_height": 140,
                            "URA_dwelling_units": 640
                            },
                'clementi peaks': {'name': 'clementi peaks',
                                    'site_boundary': Polygon(((103.76881799558069, 1.3113251436959874), (103.76881140669404, 1.3113255727539448), (103.76873396113677, 1.3113315665393301), (103.76872784263504, 1.3113321102857496), (103.76865738056709, 1.3113391824336829), (103.76839126471891, 1.3113533462825622), (103.76838106854085, 1.3113540831328867), (103.76835900052754, 1.3113560993222997), (103.76833891785218, 1.3113586981033494), (103.76831699267106, 1.3113623775367729), (103.76830393468646, 1.3113649016606184), (103.76822916610874, 1.311381278147748), (103.76822326065329, 1.3113826413806544), (103.76814908496362, 1.3114006451884348), (103.76814328579329, 1.3114021211198539), (103.76806951434554, 1.3114217711079816), (103.76806369241301, 1.3114233919786933), (103.76805063357534, 1.3114271859332673), (103.76805063357534, 1.3114271859332676), (103.76793435831124, 1.3112195678874503), (103.76785570785277, 1.3110483549485854), (103.767854909063, 1.3110466321041037), (103.76783283523227, 1.3109994591068697), (103.76778345246646, 1.3108599918491781), (103.7677810570058, 1.310853493215834), (103.767775684187, 1.3108394750826222), (103.76777019997111, 1.310826231496659), (103.76776409052887, 1.3108125214049136), (103.76776114891919, 1.3108061449635922), (103.7677035103156, 1.3106853657516602), (103.7676989924413, 1.3106763377694812), (103.76768916217716, 1.3106575804661729), (103.7676792703816, 1.3106402915368032), (103.76766809354281, 1.3106223308305385), (103.76766267840111, 1.3106139769623915), (103.76760055230274, 1.3105219074519354), (103.767594802784, 1.3105107430616825), (103.767537054323, 1.3103985563959337), (103.767557039474, 1.310366533902596), (103.76751937486969, 1.3103642385678935), (103.76744440558485, 1.3102219736491236), (103.76743225026283, 1.310198905858367), (103.76743066661837, 1.3101959453563914), (103.76734150792808, 1.3100317376402688), (103.7673397693827, 1.3100285860524563), (103.76725723774882, 1.3098813090027575), (103.767004307234, 1.309318039510132), (103.767004307234, 1.3093180395101274), (103.76767692458766, 1.309016942764499), (103.76767692458766, 1.3090169427645042), (103.76769177935024, 1.30905584345123), (103.76769263097744, 1.309057957826232), (103.7677184530503, 1.3091188584835525), (103.76771935865, 1.30918791988285), (103.76797250724987, 1.3096639600470876), (103.76798035681244, 1.309681308153539), (103.76810629485455, 1.3099595184365), (103.7681429619824, 1.3100405173730216), (103.76818872560791, 1.31014162506384), (103.76824352221166, 1.3102626758022258), (103.768244409765, 1.310264537846306), (103.76830279528608, 1.3103830476364076), (103.76830295441698, 1.3103833682895258), (103.76834752882135, 1.3104725485665667), (103.76834849990372, 1.3104744128344818), (103.76839591388273, 1.31056183915754), (103.76839618181492, 1.3105623280126735), (103.76844411771513, 1.3106488769082296), (103.76844515425438, 1.3106506775505256), (103.76849661155047, 1.31073674686852), (103.76849738485257, 1.3107380064697685), (103.76855069357646, 1.3108225930541009), (103.76855129084, 1.3108235225460931), (103.76860488702482, 1.3109053421425187), (103.76860502290766, 1.3109055486933838), (103.76865875501163, 1.3109868763656687), (103.76870692933043, 1.3110702854992), (103.76875105891605, 1.3111566848241125), (103.76878968068795, 1.311244243504477), (103.76881799558069, 1.3113251436959852))),
                                    'site_coverage': 0.15,
                                    'building_scale': 12,
                                    "postal_code": "120463",
                                    "PR": 4,
                                    "URA_GFA": 144701.58,
                                    "URA_site_area": 35550,
                                    "URA_building_height": 137,
                                    "URA_dwelling_units": 1104
                                    }}

    clavon_poly_coord = sites_info.get('clavon')['site_boundary']
    x,y,w,h= clavon_poly_coord.bounds
    clavon_poly_bbx =Polygon([(x, y), (w, y), (w, h), (x, h)]) 
    # print('bound',clavon_poly_coord.bounds)
    scale_factor_x = (clavon_poly_bbx.bounds[2]-clavon_poly_bbx.bounds[0]) / (bounding_box_site.bounds[2]-bounding_box_site.bounds[0])  # Using width of bounding box
    scale_factor_y = (clavon_poly_bbx.bounds[3]-clavon_poly_bbx.bounds[1]) / (bounding_box_site.bounds[3]-bounding_box_site.bounds[1])  # Using height of bounding box

    scaled_clavon = scale(clavon_site, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))

    scaled_polygon_B = scale(bounding_box_site, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))

    translation_vector = (clavon_poly_bbx.centroid.x - scaled_polygon_B.centroid.x, 
                        clavon_poly_bbx.centroid.y - scaled_polygon_B.centroid.y)
    transformed_clavon_poly = translate(scaled_clavon, xoff=translation_vector[0], yoff=translation_vector[1])

    transformed_clavon_poly_bbx = translate(scaled_polygon_B, xoff=translation_vector[0], yoff=translation_vector[1])
    # print(transformed_clavon_poly_bbx)
    # print(clavon_poly_bbx)
    inferences={}
    buildings_output =[]
    plt.plot(transformed_clavon_poly_bbx.exterior.xy[0],transformed_clavon_poly_bbx.exterior.xy[1])
    plt.plot(transformed_clavon_poly.exterior.xy[0],transformed_clavon_poly.exterior.xy[1])
    for i in range(len(buildings)):
        # inferences[f'model_{i}'] =[]
        poly = buildings[i]
        print(poly)
            #scale and translate building 
        scale_poly = scale(poly, xfact=scale_factor_x, yfact=scale_factor_y, origin=(clavon_poly_bbx.centroid))
        translate_poly = translate(scale_poly, xoff=translation_vector[0], yoff=translation_vector[1])
        plt.plot(translate_poly.exterior.xy[0],translate_poly.exterior.xy[1])
        buildings_output.append(translate_poly)
        # inferences[f'model_{i}'].append(translate_poly)
    # plt.show()

    # Convert Shapely polygons to GeoJSON format
    # clavon_site_geojson = mapping(clavon_site)
    # bounding_box_site_geojson = mapping(transformed_clavon_poly_bbx)
    # buildings_geojson = [mapping(building) for building in buildings]



    # Prepare the data to be saved in JSON format
    data = {
        "clavon_site": transformed_clavon_poly, #clavon_site_geojson,
        "bounding_box_site": transformed_clavon_poly_bbx, #bounding_box_site_geojson,
        "buildings":  buildings_output, #buildings_geojson
    }
    output.append(data)
print(output)

# # File path
# file_path = './clavon_data.json'

# # Writing data to a JSON file
# with open(file_path, 'w') as file:
#     json.dump(data, file, indent=4)

# file_path

