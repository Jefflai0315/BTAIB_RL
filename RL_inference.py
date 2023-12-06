import os
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, DQN, A2C, DDPG, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import numpy as np
import random
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import polygon_to_Poly3DCollection
import rasterio.features
from shapely import affinity
from shapely.affinity import scale, translate
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
# np.set_printoptions(threshold=np.inf)

print('Welcome to ArchitectMind.ai RL inference!')



class SpaDesPlacement(gym.Env):
    def __init__(self, sites_info, building, grid_size=(50, 50)):
        super(SpaDesPlacement, self).__init__()
        self.max_boxes = 10
        self.box_placed = 0
        self.grid_size = grid_size
        self.sites_info = sites_info
        self.name = 'clavon'  # None
        self.site_boundary, self.site = self._generate_site(
            sites_info, name=self.name)
        self.site_coverage = self.site['site_coverage']  # Next: GFA ...
        # Next:Meters value ...
        self.building_scale = self.site['building_scale']
        self.grid = self.update_grid_with_polygon(
            self.site_boundary, init_site=True)
        self.site_pixel = np.count_nonzero(self.grid == 1)
        self.total_footprint_pixel = 0
        self.building = building  # Next: More buildings ...
        self.building_list = []
        self.action_space = spaces.Box(low=np.array(
            [-1, -1, 3, -1]), high=np.array([1, 1, 10, 1]), shape=(4,), dtype=float)
        self.observation_space = spaces.Box(
            0, 1, shape=(np.prod(grid_size),), dtype=np.float32)
        self.boxes = np.empty((1, 4), dtype=float)
        self.state = self._get_state()

    def reset(self, seed=1, **kwargs):
        self.box_placed = 0
        if kwargs.get('name') is not None:
            self.site_boundary, self.site = self._generate_site(
                self.sites_info, name=kwargs.get('name'))
        else:
            self.site_boundary, self.site = self._generate_site(
                self.sites_info, name=self.name)
        self.site_coverage = self.site['site_coverage']  # Next: GFA ...
        # Next:Meters value ...
        self.building_scale = self.site['building_scale']
        self.building_list = []
        self.total_footprint_pixel = 0
        self.boxes = np.empty((1, 4), dtype=float)
        self.grid = self.update_grid_with_polygon(
            self.site_boundary, init_site=True)
        self.site_pixel = np.count_nonzero(self.grid == 1)
        self.state = self._get_state()
        return self.state, {}

    def step(self, action):
        ''' 
        1. reward (+ve) for placing more building                                        
        2. reward (-ve) for placing building outside boundary                                                    Terminate
        3. reward (-ve) for placing building that collide with other buildings                                   Terminate
        4. reward (-ve) for placing building that violate interblock distance (short)    
        5. reward (-ve) for placing building that violate interblock distance (long)      
        6. reward (+ve) for not violating any interblock distance and collison                       
        7. reward (+ve) for placing building that is fulfiled site coverage                                       Terminate
        '''
        done = False
        is_valid = True
        x, y, height, angle = action
        x, y = ((x+1)*self.grid_size[0]/2)-0.1, ((y+1)*self.grid_size[0]/2)-0.1
        X, Y = self._resize_polygon(
            self.building, self.building_scale, (x, y), angle)
        building = Polygon(zip(X, Y))
        reward = 100  # * (len(self.boxes)) # reward for placing more building

        if self._building_outside_boundary(building):
            reward -= 400
            done = True
            is_valid = False

        else:
            reward_, is_valid = self._check_no_collision_and_interblock_distance(
                building, self.building_list)
            if reward_ < 0:
                reward = reward_
            else:
                reward += reward_
            if is_valid == False:
                done = True

        # store buildings properties for calculation of reward
        box = np.array([[x, y, height, angle]])
        self.boxes = np.append(self.boxes, box, axis=0)
        self.building_list.append(building)
        self.box_placed += 1

        # update state
        self.grid = self.update_grid_with_polygon(building)
        self.state = self._get_state()
        if self._site_coverage_covered() and is_valid:
            reward += 200
            print('site coverage covered')
            done = True

        # limit
        if self.box_placed >= 10:
            done = True

        return self.state, reward, done, None, {}

    def _site_coverage_covered(self):
        if (self.total_footprint_pixel / self.site_pixel > self.site_coverage):
            return True

    def _check_no_collision_and_interblock_distance(self, building, building_list):
        """
            Input: 
                building (shapely.Polygon): Polygon to be placed and checked for interblock distance and collision 
            Output:
                reward (int): Panalty for violation of interblock distance and collision, reward otherwise
                is_valid (bool): For Termination if building collide
        """

        bounds = building.minimum_rotated_rectangle.exterior.xy
        # get longest side of the building boundary
        x1, y1 = bounds[0][0], bounds[1][0]
        x2, y2 = bounds[0][1], bounds[1][1]
        x3, y3 = bounds[0][2], bounds[1][2]
        x4, y4 = bounds[0][3], bounds[1][3]
        w1 = math.sqrt((y2-y1)**2 + (x2-x1)**2)
        w2 = math.sqrt((y3-y2)**2 + (x3-x2)**2)

        # extend both sides of the longest sides of the building boundary by interblock distance scaled to grid size
        interblock_distance = {"facing": 30, "non_facing": 10}
        grid_to_metre_ratio = 60/self.building_scale  # To be replaced ...
        interblock_dist = interblock_distance["facing"]/grid_to_metre_ratio
        if w1 > w2:
            projection_line = LineString([(x1, y1), (x2, y2)])
            projection_line2 = LineString([(x3, y3), (x4, y4)])
        else:
            projection_line = LineString([(x2, y2), (x3, y3)])
            projection_line2 = LineString([(x4, y4), (x1, y1)])
        buffer = projection_line.buffer(
            distance=-interblock_dist, cap_style="square", single_sided=True)
        buffer2 = projection_line2.buffer(
            distance=-interblock_dist, cap_style="square", single_sided=True)

        def calculate_angle(line1, line2):
            # Get vectors representing the lines
            vector1 = np.array(line1.coords[1]) - np.array(line1.coords[0])
            vector2 = np.array(line2.coords[1]) - np.array(line2.coords[0])

            # Calculate the dot product and magnitude of the vectors
            dot_product = np.dot(vector1, vector2)
            magnitude1 = np.linalg.norm(vector1)
            magnitude2 = np.linalg.norm(vector2)

            # Calculate the cosine of the angle
            cosine_angle = dot_product / (magnitude1 * magnitude2)

            # Calculate the angle in radians and convert to degrees
            angle_in_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
            angle_in_degrees = np.degrees(angle_in_radians)

            return min(angle_in_degrees, 180 - angle_in_degrees)

        reward = 0
        is_valid = True
        # check if building intersects with other buildings or violate interblock distance
        for other_box in building_list:
            if building.intersects(other_box):
                reward -= 400
                is_valid = False
            else:
                reward += 400

            if building.distance(other_box) < interblock_distance["non_facing"]/grid_to_metre_ratio:
                reward -= 200
            else:
                reward += 400
            other_box_bounds = other_box.minimum_rotated_rectangle.exterior.xy
            # get longest side of the other building boundary
            other_building_x1, other_building_y1 = other_box_bounds[0][0], other_box_bounds[1][0]
            other_building_x2, other_building_y2 = other_box_bounds[0][1], other_box_bounds[1][1]
            other_building_x3, other_building_y3 = other_box_bounds[0][2], other_box_bounds[1][2]
            other_building_w1 = math.sqrt(
                (other_building_y2-other_building_y1)**2 + (other_building_x2-other_building_x1)**2)
            other_building_w2 = math.sqrt(
                (other_building_y3-other_building_y2)**2 + (other_building_x3-other_building_x2)**2)

            if other_building_w1 > other_building_w2:
                other_building_projection_line = LineString(
                    [(other_building_x1, other_building_y1), (other_building_x2, other_building_y2)])
            else:
                other_building_projection_line = LineString(
                    [(other_building_x2, other_building_y2), (other_building_x3, other_building_y3)])
            if calculate_angle(projection_line, other_building_projection_line) < 30:
                if buffer.intersects(other_box) or buffer2.intersects(other_box):
                    reward -= 300
                else:
                    reward += 400
        return reward, is_valid

    def _building_outside_boundary(self, building):
        # Check if the building is outside the site boundary
        if not self.site_boundary.contains(building):
            return True
        return False

    def update_grid_with_polygon(self, polygon, init_site=False):
        rasterized = rasterio.features.geometry_mask(
            [polygon],
            out_shape=self.grid_size,
            transform=rasterio.transform.from_bounds(0, 0, self.grid_size[0], self.grid_size[1], width=self.grid_size[0], height=self.grid_size[1]), invert=True)
        if init_site:
            grid = np.full(self.grid_size, -10)
            updated_grid = grid + rasterized.astype(int) * 11
            return updated_grid
        else:
            grid = np.copy(self.grid)
            updated_grid = grid + rasterized.astype(int)
            self.total_footprint_pixel = np.count_nonzero(updated_grid > 1)
            # plt.imshow(updated_grid)
            # plt.show()
            return updated_grid

    def _resize_polygon(self, poly, desired_scale=50, center=None, angle=0):
        poly = affinity.rotate(poly, (angle+1)*180, origin='centroid')
        X, Y = poly.exterior.xy
        current_width = max(X) - min(X)
        current_height = max(Y) - min(Y)
        longest_axis = max(current_width, current_height)
        scale_factor = desired_scale / longest_axis
        if center is None:
            center_x = current_width / 2
            center_y = current_height / 2
        else:
            center_x, center_y = center
        scaled_polygon_x = [(x - min(X)) * scale_factor for x in X]
        scaled_polygon_y = [(y - min(Y)) * scale_factor for y in Y]
        scaled_polygon = Polygon(list(zip(scaled_polygon_x, scaled_polygon_y)))
        x_off = scaled_polygon.centroid.x - center_x
        y_off = scaled_polygon.centroid.y - center_y
        centered_polygon_x = [x - x_off for x in scaled_polygon_x]
        centered_polygon_y = [y - y_off for y in scaled_polygon_y]
        return centered_polygon_x, centered_polygon_y

    def _get_state(self):
        flat_grid = self.grid.flatten()
        return flat_grid

    def _generate_site(self, sites_info, name):
        if name is not None and name in sites_info.keys():
            site = sites_info[name]
        else:
            site = sites_info[random.choice(list(sites_info.keys()))]
        scale = max(self.grid_size[0], self.grid_size[1])
        x, y = self._resize_polygon(
            site['site_boundary'], scale, (self.grid_size[0]/2, self.grid_size[1]/2))
        return Polygon(list(zip(x, y))), site

    def render(self, best=None, site=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if best is None:
            boxes = self.boxes[1:]
        else:
            boxes = best[1:]
        for i in range(len(boxes)):
            x = boxes[i][0]
            y = boxes[i][1]
            height = boxes[i][2]
            angle = boxes[i][3]
            X, Y = self._resize_polygon(
                self.building, self.building_scale, (x, y), angle)
            building = Polygon(zip(X, Y))
            poly3d = polygon_to_Poly3DCollection(building, height)
            ax.add_collection3d(poly3d)
        ax.plot(list(self.site_boundary.exterior.xy[0]), list(
            self.site_boundary.exterior.xy[1]), alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # Set plot limits
        ax.set_xlim([0, self.grid_size[0]])
        ax.set_ylim([0, self.grid_size[1]])
        ax.set_zlim([0, 10])
        # Display the plot
        plt.show()




# inference
building_list = [Polygon(((0.0, 0.0), (0.0, 1.1), (1.5, 1.1), (1.5, 2.0), (0.0, 2.0), (0.0, 3.0), (7.0, 3.0), (7.0, 2.0), (5.0, 2.0), (5.0, 1.1), (7.0, 1.1), (7.0, 0.0), (4.5, 0.0), (4.5, 0.5), (3.5, 0.5), (3.5, 0.0), (0.0, 0.0))),
                 #  Polygon(((0.0,0.0),(0.0,1.3),(1.5,1.3),(1.5,2.0),(0.0,2),(0.0,3),(5.0,3.0),(5.0,2.0),(3.0,2.0),(3.0,1.3),(5.0,1.3),(5.0,0.0),(0.0,0.0))),
                 ]
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

# load model
load_path = 'best_model_3_w_interblock_distance_ppo96.zip'
if os.path.exists(load_path):
    model = PPO.load(load_path)
else:
    print(f"The file {load_path} does not exist.")

assert model is not None, "Model not found"


top_models = []

env = SpaDesPlacement(sites_info, building_list[0])
best_obs = ''
best_reward = -float('inf')
best_rewards = []
best_boxes = []
episode =0

# while valid and episode <1000:
for i in range(5000):
    obs,info = env.reset(seed =1 , name = 'clavon')
    episode_reward = 0
    episode +=1
    rewards = []
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _,_ = env.step(action)
        episode_reward +=reward
        rewards.append(reward)
    
        if done :
            current_model_info = (episode_reward, obs, env.boxes, rewards)
            top_models.append(current_model_info)
            top_models.sort(key=lambda x: x[0], reverse=True)
            if len(top_models) > 3:
                top_models = top_models[:3]
            # if episode_reward > best_reward:
            #     best_reward = episode_reward
            #     best_obs = obs
            #     best_boxes = env.boxes
            #     best_rewards = rewards
            break
                
    # print('loading...')
    # valid = False
    # # check if prediction is correct
    # for box in best_boxes[1:]:
    #     X,Y = env._resize_polygon(env.building, env.building_scale, (box[0], box[1]), box[3])
    #     poly = Polygon(zip(X,Y))
    #     if env._building_outside_boundary(poly):
    #         valid = True
    #         #restart
    
            

# remove any boxes that is not inside the boundary
# print('predicted: ', best_boxes)
# print('best reward: ', best_reward)
# print('best rewards: ', best_rewards)
# boxes = best_boxes[:1]
# rewards = []
model_buildings =[]
for model in top_models:
    best_boxes = model[2] 
    best_rewards = model[3]
    best_obs = model[1]
    best_reward = model[0]
    boxes = best_boxes[:1]
    rewards = []
    buildings_poly =[]
    for i in range(1,len(best_boxes)):
        X,Y = env._resize_polygon(env.building, env.building_scale, (best_boxes[i][0], best_boxes[i][1]), best_boxes[i][3])
        poly = Polygon(zip(X,Y))
        if env._building_outside_boundary(poly):
            pass
        else:
            boxes = np.append(boxes,np.array([best_boxes[i]]),axis=0)
            rewards.append(best_rewards[i-1])
            buildings_poly.append(poly)
    model_buildings.append(buildings_poly)
    best_reward = sum(rewards)
    obs_1 = np.reshape(best_obs, (50,50))
    print('After....')
    # plt.imshow(obs_1)
    print(boxes)
    print(best_reward)
    print(rewards)
    env.render(boxes)

inferences={}
clavon_poly_coord = sites_info.get('clavon')['site_boundary']
clavon_poly_coord_centroid = clavon_poly_coord.centroid
clavon_poly_grid = env.site_boundary
# print('bound',clavon_poly_coord.bounds)
scale_factor_x = (clavon_poly_coord.bounds[2]-clavon_poly_coord.bounds[0]) / (clavon_poly_grid.bounds[2]-clavon_poly_grid.bounds[0])  # Using width of bounding box
scale_factor_y = (clavon_poly_coord.bounds[3]-clavon_poly_coord.bounds[1]) / (clavon_poly_grid.bounds[3]-clavon_poly_grid.bounds[1])  # Using height of bounding box

# Scale Polygon B
scaled_polygon_B = scale(clavon_poly_grid, xfact=-scale_factor_x, yfact=-scale_factor_y, origin=(clavon_poly_coord_centroid))

# Calculate translation vector
translation_vector = (clavon_poly_coord.centroid.x - scaled_polygon_B.centroid.x, 
                      clavon_poly_coord.centroid.y - scaled_polygon_B.centroid.y)

# Translate scaled Polygon B
transformed_clavon_poly_grid = translate(scaled_polygon_B, xoff=translation_vector[0], yoff=translation_vector[1])
# print(clavon_poly_coord)
# print(transformed_clavon_poly_grid)
plt.plot(transformed_clavon_poly_grid.exterior.xy[0],transformed_clavon_poly_grid.exterior.xy[1])
plt.plot(clavon_poly_coord.exterior.xy[0],clavon_poly_coord.exterior.xy[1])
for i in range(len(model_buildings)):
    inferences[f'model_{i}'] =[]
    for poly in model_buildings[i]:
        #scale and translate building 
        scale_poly = scale(poly, xfact=-scale_factor_x, yfact=-scale_factor_y, origin=(clavon_poly_coord_centroid))
        translate_poly = translate(scale_poly, xoff=translation_vector[0], yoff=translation_vector[1])
        plt.plot(translate_poly.exterior.xy[0],translate_poly.exterior.xy[1])
        inferences[f'model_{i}'].append(translate_poly)
plt.show()
print(inferences)

