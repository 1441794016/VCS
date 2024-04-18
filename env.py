import csv
import numpy as np
import random
from driver import Driver
from order import Order
from poi import POI


def create_POI_one_episode(episode_limit, region_n):
    """
    :param episode_limit: 时隙长度
    :param region_n: 区域的数目
    一次性随机生成所有时隙里要产生的POI
    :return: POI_every_time_slot,一个列表，列表的长度为总时隙长度加1，第i元素的也是1个列表（列表元素是POI），表示第i个时隙会产生的POI
             POI_number_every_time_slot: 一个一维numpy数组，长度是时隙长度加1，每个元素是当前时刻要产生的POI数量
    """
    POI_every_time_slot = []
    region_item = range(region_n)
    POI_number_every_timeslot = np.round(
        np.clip(np.random.normal(5, 2, episode_limit + 1), 0, 12))  # 每一步生成的数据数量

    # POI_every_time_slot_and_region = {}  # 每个time-slot每个区域要产生的POI
    # POI_number_every_timeslot_and_region = {}  # 每个time-slot每个区域要产生的POI数量
    # for i in range(region_n):
    #     POI_every_time_slot_and_region[i] = []
    #     POI_number_every_timeslot_and_region[i] = np.zeros(episode_limit + 1, )
    #     for j in range(episode_limit + 1):
    #         POI_every_time_slot_and_region[i].append([])

    for i in range(episode_limit + 1):
        temp = []
        POI_number_one_time_slot = POI_number_every_timeslot[i]  # 第i个时隙的POI产生数量

        prob = [0.06, 0.01, 0.01, 0.01, 0, 0, 0.04,
                0.05, 0.03, 0.1, 0, 0, 0.02, 0.1,
                0.1, 0.00, 0.03, 0, 0.02, 0, 0.02,
                0.03, 0.01, 0.00, 0, 0.02, 0.13, 0.04,
                0.06, 0.00, 0.02, 0, 0.05, 0, 0.04]

        region = np.random.choice(region_item, size=int(POI_number_one_time_slot),
                                  replace=True, p=prob)  # 按概率prob在一些区域生成若干个数量的POI

        # 设置均值和标准差
        mu = 6
        sigma = 2

        # 生成正态分布的随机值
        random_values = np.random.normal(mu, sigma, int(POI_number_one_time_slot))  # 生成number_of_POIs个随机值

        # 将值限制在3到12的范围内
        bounded_values = np.clip(random_values, 3, 6)

        # 四舍五入为整数
        integer_values = np.round(bounded_values) * 100  # number_of_POIs个POI的数据量
        for j in range(int(POI_number_one_time_slot)):
            temp.append(POI(integer_values[j], region[j]))
            # POI_every_time_slot_and_region[int(region[j])][i].append(POI(integer_values[j], region[j]))
            # POI_number_every_timeslot_and_region[int(region[j])][i] += 1

        POI_every_time_slot.append(temp)
    return POI_every_time_slot, POI_number_every_timeslot


def create_order_one_episode(episode_limit, order_number, order_dataset):
    """
    生成一个episode里每个时隙要产生的order
    :param episode_limit: 时隙长度
    :param order_number: 调用函数为止order的数量
    :param order_dataset: order_dataset
    :return: orders_every_time_slot 一个dict,有episode_limit + 1个元素，每个元素是一个list，list里是要产生的order
             order_number_every_time_slot 一个一维numpy数组，有episode_limit + 1个元素，是每个时隙中要产生的order数量
    """
    order_number = order_number
    orders_every_time_slot = []
    for i in range(episode_limit + 1):
        orders_every_time_slot.append([])
    orders_number_every_time_slot = np.zeros(episode_limit + 1, )

    for data in order_dataset:
        s = data[2].split(' ')
        s1 = s[1].split(':')
        time_slot = int(int((float(s1[0])) * 60 + float(s1[1])) / 3)
        if time_slot >= episode_limit:
            break

        init_location = [float(data[5]), float(data[6])]
        drop_off_location = [float(data[7]), float(data[8])]
        if init_location[0] <= -74.00 or init_location[0] >= -73.86 or \
                init_location[1] <= 40.70 or init_location[1] >= 40.80 or \
                drop_off_location[0] <= -74.00 or drop_off_location[0] >= -73.86 or \
                drop_off_location[1] <= 40.70 or drop_off_location[1] >= 40.80:
            continue
        else:
            init_region_location = int((init_location[0] + 74.00) / 0.02) + \
                                   int((40.80 - init_location[1]) / 0.02) * 7
            drop_off_region_location = int((drop_off_location[0] + 74.00) / 0.02) + \
                                       int((40.80 - drop_off_location[1]) / 0.02) * 7
            order_price = 0.02 * float(data[10])
            travel_time_cost = int(float(data[10]) / 60)

            orders_every_time_slot[time_slot].append(
                Order(order_id=order_number, init_location=init_region_location,
                      generate_time=time_slot, destination=drop_off_region_location, waiting_time=15, price=order_price,
                      travel_time=travel_time_cost)
            )

            order_number += 1
            orders_number_every_time_slot[time_slot] += 1
    return orders_every_time_slot, orders_number_every_time_slot


class Env:
    def __init__(self, args):
        """
        初始化环境
        :param args: 配置参数列表
        """
        self.args = args
        self.time_slot = 0  # 初始时隙
        self.episode_limit = args.episode_limit  # 最大时隙数量

        self.region_n = args.region_n  # 区域的数量
        self.region_graph = args.region_graph  # 区域之间的连接图，这是一个静态图

        self.agent_n = args.agent_n  # agent的数量

        self.use_gnn = args.use_gnn  # 是否使用gnn进行embedding

        # 设置状态和动作维度
        self.observation_space = 23  # 每个车的观测状态维度
        self.action_space = 11  # 每个车动作维度。包括接单，采数据，或待在原区域不进行行动
        #

        # 设置driver，order和POI
        self.drivers = []  # 一个列表来存储系统中的所有driver，元素类型是Driver。driver数量是不变的。
        self.orders = []  # 一个列表来存储系统中产生的order，元素类型是Order。
        self.POIs = []  # 一个列表来存储系统中参数的POI，元素类型是POI。

        # 记录每个区域的邻居
        self.regions_neighbor = []  # 一个列表记录每个区域的邻居
        for i in range(self.region_n):
            neighbor = []
            index = 0
            for node_index in self.region_graph.edges()[0]:
                if node_index.item() == i:
                    neighbor.append(self.region_graph.edges()[1][index].item())
                index += 1
            self.regions_neighbor.append(neighbor.copy())
        self.region_item = range(self.region_n)  # 一个保存了每个区域编号的列表
        #

        # 一些需要统计的数据, 实验中可能要比较的一些指标
        self.order_number = 0  # 已经产生的order数量
        self.POI_number = 0  # 已经参数的POI数量
        self.income = 0  # 总接单收益
        self.accepted_order_number = 0  # 被接受的order数量
        self.overdue_order_number = 0  # 过期的订单数量
        self.collected_POI_number = 0  # 被收集的POI数量
        self.overdue_POI_number = 0  # 过期订单POI数量
        self.total_data_vol = 0  # 总收集数据量
        self.total_data_utility = 0  # 总收集数据utility
        self.total_AOI = 0  # 被收集的POI的信息年龄之和
        #

        # 一些随着episode step变化的量
        self.serving_vehicle_number = 0  # 正在服务订单的车辆数量
        self.collecting_vehicle_number = 0  # 正在收集数据的车辆数量
        self.offloading_vehicle_number = 0  # 正在收集数据的车辆数量
        #

        # 加载orders数据集，根据这个数据集生成order
        self.order_dataset = []
        with open(args.order_data_path) as csvfile:
            csv_reader = csv.reader(csvfile)  # 使用csv.reader读取csvfile中的文件
            # header = next(csv_reader)        # 读取第一行每一列的标题
            for row in csv_reader:  # 将csv 文件中的数据保存到data中
                self.order_dataset.append(row)  # 选择某一列加入到data数组中
            del (self.order_dataset[0])
        #

        self.first_inti_orders_number = 100  # 环境初始化时先产生这么多个order

        # 初始化时先生成每一个时隙要生成的POI
        self.POI_every_timeslot, self.POI_number_every_timeslot = create_POI_one_episode(
            self.episode_limit, self.region_n)

        # 初始化时先生成每个时隙要生成的order
        self.orders_every_time_slot, self.orders_number_every_time_slot = create_order_one_episode(
            self.episode_limit,
            order_number=self.first_inti_orders_number,
            order_dataset=self.order_dataset)

    def init_POI_data(self):
        """
        在每个episode的第一个时隙时先初始化一些POI在地图上
        :return: no return
        """
        number_of_POIs = 40  # time slot为0时先按一个概率随机初始化一些POI在区域内
        prob = [0.08, 0.00, 0.01, 0.00, 0, 0, 0.04,
                0.06, 0.03, 0.1, 0, 0, 0.02, 0.1,
                0.1, 0.00, 0.03, 0, 0, 0, 0.03,
                0.03, 0.01, 0.00, 0, 0.02, 0.15, 0.04,
                0.06, 0.00, 0.00, 0, 0.05, 0, 0.04]  # 各个区域产生POI的概率
        region = np.random.choice(self.region_item, size=number_of_POIs, replace=True, p=prob)  # 按概率选择区域

        # 设置均值和标准差
        mu = 6
        sigma = 2

        # 生成正态分布的随机值
        random_values = np.random.normal(mu, sigma, number_of_POIs)  # 生成number_of_POIs个随机值

        # 将值限制在3到12的范围内
        bounded_values = np.clip(random_values, 3, 12)

        # 四舍五入为整数
        integer_values = np.round(bounded_values) * 100  # number_of_POIs个POI的数据量

        for i, data in enumerate(region):
            self.POIs.append(
                POI(init_data_vol=integer_values[i], init_location=data)
            )
            self.POI_number += 1

    def init_POI_data_(self):
        """
        在每个episode的第一个时隙时先初始化一些POI在地图上
        :return: no return
        """
        number_of_POIs = 40  # time slot为0时先按一个概率随机初始化一些POI在区域内
        prob = [0.01, 0.00, 0.01, 0.00, 0, 0.01, 0.0,
                0.06, 0.08, 0.1, 0.05, 0.01, 0.01, 0.1,
                0.1, 0.06, 0.03, 0.04, 0, 0, 0.03,
                0.03, 0.03, 0.04, 0, 0.02, 0.01, 0.04,
                0.04, 0.03, 0.03, 0, 0.02, 0, 0.01]  # 各个区域产生POI的概率
        region = np.random.choice(self.region_item, size=number_of_POIs, replace=True, p=prob)  # 按概率选择区域

        # 设置均值和标准差
        mu = 6
        sigma = 2

        # 生成正态分布的随机值
        random_values = np.random.normal(mu, sigma, number_of_POIs)  # 生成number_of_POIs个随机值

        # 将值限制在3到12的范围内
        bounded_values = np.clip(random_values, 3, 12)

        # 四舍五入为整数
        integer_values = np.round(bounded_values) * 100  # number_of_POIs个POI的数据量

        for i, data in enumerate(region):
            self.POIs.append(
                POI(init_data_vol=integer_values[i], init_location=data)
            )
            self.POI_number += 1

    def init_ava_POI_data(self):
        """
        在每个episode的第一个时隙时先初始化一些POI在地图上
        :return: no return
        """
        number_of_POIs = 40  # time slot为0时先按一个概率随机初始化一些POI在区域内
        prob = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                0.03, 0.03, 0.04, 0.03, 0.02, 0.02, 0.04,
                0.04, 0.04, 0.03, 0.04, 0.02, 0.02, 0.03,
                0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03,
                0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.03]  # 各个区域产生POI的概率
        region = np.random.choice(self.region_item, size=number_of_POIs, replace=True, p=prob)  # 按概率选择区域

        # 设置均值和标准差
        mu = 6
        sigma = 2

        # 生成正态分布的随机值
        random_values = np.random.normal(mu, sigma, number_of_POIs)  # 生成number_of_POIs个随机值

        # 将值限制在3到12的范围内
        bounded_values = np.clip(random_values, 3, 12)

        # 四舍五入为整数
        integer_values = np.round(bounded_values) * 100  # number_of_POIs个POI的数据量

        for i, data in enumerate(region):
            self.POIs.append(
                POI(init_data_vol=integer_values[i], init_location=data)
            )
            self.POI_number += 1

    def create_POI_data(self):
        """
        在每个时隙生成一定数量POI，修改环境的POI列表和POI数量
        :return: no return
        """
        POIs_number = self.POI_number_every_timeslot[self.time_slot]
        if POIs_number == 0:
            return

        # for i in range(self.region_n):
        #
        #     self.POIs_dict[i].extend(self.POI_every_time_slot_and_region[i][self.time_slot])

        POIs = self.POI_every_timeslot[self.time_slot]
        self.POIs.extend(POIs)

        self.POI_number += POIs_number

        # overdue_POI = 0
        # accepted_POI = 0
        # for POI_ in self.POIs:
        #     if POI_.is_overdue:
        #         overdue_POI += 1
        #     if POI_.is_collected:
        #         accepted_POI += 1
        # print("POIs len:", len(self.POIs))
        # print("over due POIs:", overdue_POI)
        # print("accepted POIs:", accepted_POI)
        # print("self.collected_POI_number:", self.collected_POI_number)
        # print("self.overdue_POI_number:", self.overdue_POI_number)

    def init_drivers(self):
        """
        初始化司机
        :return: no return
        """
        for i in range(self.agent_n):
            init_location = np.random.randint(0, self.region_n)  # 每辆车的初始位置
            init_capacity = np.random.randint(30, 40) * 100  # 每辆车的容量
            init_collect_rate = 100
            init_offload_rate = 100
            self.drivers.append(Driver(driver_id=i, init_location=init_location, init_capacity=init_capacity,
                                       init_collect_rate=init_collect_rate, init_offload_rate=init_offload_rate))

    def init_orders(self):
        """
        初始时刻先生成一些订单
        :return: no return
        """
        number_of_orders = self.first_inti_orders_number
        prob = [0.02, 0.06, 0.03, 0.02, 0, 0, 0.01,
                0.06, 0.10, 0.1, 0, 0.01, 0.02, 0.1,
                0.1, 0.01, 0.03, 0, 0.01, 0, 0.02,
                0.06, 0.01, 0.03, 0, 0.04, 0.02, 0,
                0.03, 0.02, 0.00, 0, 0.05, 0, 0.04]
        region = np.random.choice(self.region_item, size=number_of_orders, replace=True, p=prob)
        # travel_time是每个订单的用时时间
        travel_time = np.round(np.clip(np.random.normal(10, 2, number_of_orders), 2, 20))
        destination = np.random.randint(0, self.region_n, number_of_orders)
        for i, data in enumerate(region):
            self.orders.append(
                Order(order_id=self.order_number, init_location=data,
                      generate_time=self.time_slot, destination=destination[i], waiting_time=6,
                      price=20, travel_time=travel_time[i])
            )
            self.order_number += 1

    def create_orders(self):
        """
        根据order数据集产生订单
        :return: no return
        """
        if self.orders_number_every_time_slot[self.time_slot] == 0:
            return

        orders = self.orders_every_time_slot[self.time_slot]

        self.orders.extend(orders)
        self.order_number += self.orders_number_every_time_slot[self.time_slot]

        # overdue = 0
        # accepted = 0
        # for order in self.orders:
        #     if order.is_overdue:
        #         overdue += 1
        #     if order.is_matched:
        #         accepted += 1
        # print("orders len:", len(self.orders))
        # print("over due number:", overdue)
        # print("accepted:", accepted)
        # print("self.accepted_order_number:", self.accepted_order_number)
        # print("self.overdue_order_number:", self.overdue_order_number)

    def get_idle_orders(self):
        """
        返回每个区域内的待被服务的订单数量
        :return: result 是一个numpy数组，维度是(self.region_n)，包含每个区域内的待服务订单数
        """
        result = np.zeros(self.region_n)
        for order in self.orders:
            if not order.is_matched and not order.is_overdue:
                result[order.location] += 1
        return result

    def get_POI(self):
        """
        返回每个区域内没有被采集且没有过期的POI的数量
        :return:
        """
        result = np.zeros(self.region_n)
        for POI_ in self.POIs:
            if not POI_.is_overdue and not POI_.is_collected:
                result[POI_.location] += 1
        return result

    def get_idle_vehicle(self):
        """
        返回每个区域内的空闲车辆数量
        :return:
        """
        result = np.zeros(self.region_n)
        for driver in self.drivers:
            if not driver.is_serving and not driver.is_collecting and not driver.is_offloading:
                result[driver.now_location] += 1
        return result

    def get_observation(self):
        """
        返回agent的状态
        :return: 一个numpy数组，维度是(self.agent_n, observation_space)
        """
        idle_orders = self.get_idle_orders()  # 每个区域内的order数
        POIs = self.get_POI()  # 每个区域内的POI数
        idle_vehicles = self.get_idle_vehicle()  # 每个区域内的vehicle数
        obs = np.zeros((self.agent_n, self.observation_space))
        time_slot = '{:08b}'.format(self.time_slot)
        for i in range(self.agent_n):
            if self.drivers[i].is_serving:
                obs[i, 0] = 1.0
                continue
            if self.drivers[i].is_collecting:
                obs[i, 1] = 1.0
                continue
            if self.drivers[i].is_offloading:
                obs[i, 2] = 1.0
                continue

            location_ = int(self.drivers[i].now_location)
            location = '{:06b}'.format(location_)
            # obs[i, 0] = 1.0 if self.drivers[i].is_serving else 0.0  # driver i 是否正在服务订单
            # obs[i, 1] = 1.0 if self.drivers[i].is_collecting else 0.0  # driver i 是否正在收集数据
            # obs[i, 2] = 1.0 if self.drivers[i].is_offloading else 0.0  # driver i 是否正在卸载数据
            obs[i, 3] = self.drivers[i].data_vol  # driver i 已收集的数据量
            obs[i, 4] = self.drivers[i].capacity  # driver i 的容量
            obs[i, 5] = idle_orders[location_] if not self.drivers[i].is_serving else -10
            obs[i, 6] = POIs[location_] if not self.drivers[i].is_serving else -10
            obs[i, 7] = idle_vehicles[location_] if not self.drivers[i].is_serving else -10

            obs[i, 8] = int(location[0])
            obs[i, 9] = int(location[1])
            obs[i, 10] = int(location[2])
            obs[i, 11] = int(location[3])
            obs[i, 12] = int(location[4])
            obs[i, 13] = int(location[5])
            obs[i, 14] = int(time_slot[0])
            obs[i, 15] = int(time_slot[1])
            obs[i, 16] = int(time_slot[2])
            obs[i, 17] = int(time_slot[3])
            obs[i, 18] = int(time_slot[4])
            obs[i, 19] = int(time_slot[5])
            obs[i, 20] = int(time_slot[6])
            obs[i, 21] = int(time_slot[7])
            obs[i, 22] = self.time_slot
            # obs[i, 12] = i  # driver i 的编号
        return obs

    def update_driver_states(self):
        """
        更新每个driver的状态
        :return:
        """
        for driver_index in range(self.agent_n):
            if not self.drivers[driver_index].is_serving and \
                    not self.drivers[driver_index].is_collecting and \
                    not self.drivers[driver_index].is_offloading:
                # 对于不在服务订单、不在收集数据同时也不在卸载数据的车辆进行随机游走。决定留在原地的车辆会留在原地
                if self.drivers[driver_index].is_dispatched:
                    self.drivers[driver_index].complete_dispatched()
                #     self.drivers[driver_index].now_location = self.drivers[driver_index].now_location
                #     self.drivers[driver_index].stay = False
                #     self.drivers[driver_index].random_walk = False
                # elif self.drivers[driver_index].random_walk:
                #     self.drivers[driver_index].now_location = random.choice(
                #         self.regions_neighbor[self.drivers[driver_index].now_location])
                #     self.drivers[driver_index].random_walk = False
                #     self.drivers[driver_index].stay = False
            elif self.drivers[driver_index].is_collecting:
                if self.time_slot == self.drivers[driver_index].time_complete_sensing:
                    # 到了完成采集数据的时刻
                    self.drivers[driver_index].complete_collect()
                    if self.drivers[driver_index].data_vol == self.drivers[driver_index].capacity:
                        # 需要进行卸载
                        self.drivers[driver_index].offload_data(self.time_slot)
            elif self.drivers[driver_index].is_serving:
                if self.drivers[driver_index].time_arrive_order_destination == self.time_slot:
                    # 乘客到达目的地，释放乘客
                    self.drivers[driver_index].drop_off()
            elif self.drivers[driver_index].is_offloading:
                if self.drivers[driver_index].time_complete_offloading == self.time_slot:
                    # 完成offload数据
                    self.drivers[driver_index].complete_offload()

    def update_order_states(self):
        """
        更新每个订单的状态。有过期的订单或者被接受的订单我们就将他删去
        :return: no return
        """

        for order_index in range(len(self.orders)):
            if self.orders[order_index].overdue_time == self.time_slot and \
                    not self.orders[order_index].is_matched and \
                    not self.orders[order_index].is_overdue:
                self.orders[order_index].is_overdue = True
                self.overdue_order_number += 1
        # if self.time_slot % 50 == 0:
        #     self.orders = [x for x in self.orders if (not x.is_overdue and not x.is_matched)]
        # temp = [x for x in self.orders if (not x.is_overdue and not x.is_matched)]
        # self.orders = temp

        # print("time slot:", self.time_slot)
        # print("order_number:", self.order_number)
        # print("accepted_order_number:", self.accepted_order_number)
        # print("overdue_order_number:", self.overdue_order_number)
        # print("\n")

    def update_POI_states(self):
        """
        更新每个POI的状态。如果有过期的订单
        :return: no return
        """
        for POI_index in range(len(self.POIs)):
            self.POIs[POI_index].AOI += 1
            if self.POIs[POI_index].AOI >= self.POIs[POI_index].max_AOI and \
                    not self.POIs[POI_index].is_collected and \
                    not self.POIs[POI_index].is_overdue:
                self.POIs[POI_index].is_overdue = True
                self.overdue_POI_number += 1
        # self.POIs = [x for x in self.POIs if (not x.is_overdue and not x.is_collected)]

    def init_env(self):
        """
        初始化环境
        :return: no return
        """
        self.time_slot = 0
        self.order_number = 0  # 已经产生的order数量
        self.POI_number = 0  # 已经参数的POI数量
        self.income = 0  # 总接单收益
        self.accepted_order_number = 0  # 被接受的order数量
        self.overdue_order_number = 0  # 过期的订单数量
        self.collected_POI_number = 0  # 被收集的POI数量
        self.overdue_POI_number = 0  # 过期订单POI数量
        self.total_data_vol = 0  # 总收集数据量
        self.total_data_utility = 0  # 总收集数据utility
        self.total_AOI = 0  # 被收集的POI的信息年龄之和

        self.init_drivers()
        self.init_POI_data()
        self.init_orders()
        self.create_orders()
        # self.create_POI_data()

    def reset(self):
        """
        reset重新初始化环境
        :return: 环境中agent的observation
        """
        self.time_slot = 0
        self.drivers.clear()
        self.orders.clear()
        self.POIs.clear()
        self.init_env()

        # for i, POIs_ in enumerate(self.POI_every_timeslot):
        #     for j, _ in enumerate(POIs_):
        #         self.POI_every_timeslot[i][j].is_overdue = False
        #         # self.POI_every_timeslot[i][j].AOI = 1
        #         self.POI_every_timeslot[i][j].is_collected = False

        for i, orders_ in enumerate(self.orders_every_time_slot):
            for j, _ in enumerate(orders_):
                self.orders_every_time_slot[i][j].is_matched = False
                self.orders_every_time_slot[i][j].is_overdue = False
        obs = self.get_observation()
        return obs

    def take_action(self, action):
        """
        每个agent执行动作
        :param action: 要执行的动作，维度是（self.agent_n, action_space)
        :return: 执行动作产生的reward
        """
        # print("order:", self.order_number)
        # print("accepted order:", self.accepted_order_number)
        # print("POI:", self.POI_number)
        # print("accepted POI:", self.collected_POI_number)
        # print("\t")
        reward = np.zeros(self.agent_n)

        # 对order和POIs进行排序
        temp_orders = sorted(self.orders, key=lambda a: a.generate_time)
        self.orders = temp_orders
        temp_POIs = sorted(self.POIs, key=lambda a: a.AOI, reverse=True)
        self.POIs = temp_POIs

        for i, driver_action in enumerate(action):
            if driver_action == 0:
                # 0 表示匹配订单。优先和快要距离过期时间最近的订单进行匹配
                if not self.drivers[i].is_collecting and \
                        not self.drivers[i].is_serving and \
                        not self.drivers[i].is_offloading:
                    driver_location = self.drivers[i].now_location
                    for order_index in range(len(self.orders)):
                        order = self.orders[order_index]
                        if not order.is_overdue and not order.is_matched and driver_location == order.location:
                            self.orders[order_index].driver_match()
                            self.drivers[i].order_match(order, self.time_slot)
                            self.accepted_order_number += 1
                            self.income += order.price
                            reward[i] += self.args.omega * float(order.price)  # 奖励是订单价格×一个权重
                            break
            elif driver_action == 1:
                # 1 采集数据。优先采集快要过期的数据
                if not self.drivers[i].is_collecting and \
                        not self.drivers[i].is_serving and \
                        not self.drivers[i].is_offloading:
                    driver_location = self.drivers[i].now_location
                    for POI_index in range(len(self.POIs)):
                        POI_ = self.POIs[POI_index]
                        if not POI_.is_overdue and not POI_.is_collected and POI_.location == driver_location:
                            self.POIs[POI_index].be_collected()
                            data_utility = self.drivers[i].collect_data(POI_, self.time_slot)
                            self.total_data_utility += data_utility
                            self.collected_POI_number += 1
                            self.total_AOI += POI_.AOI
                            self.total_data_vol += POI_.data_vol
                            reward[i] += self.args.beta * data_utility
                            break
            else:
                # 其他表示调度到其他区域
                if not self.drivers[i].is_collecting and \
                        not self.drivers[i].is_serving and \
                        not self.drivers[i].is_offloading:
                    if driver_action - 1 > len(self.regions_neighbor[self.drivers[i].now_location]):
                        self.drivers[i].dispatched(self.drivers[i].now_location)
                    else:
                        self.drivers[i].dispatched(self.regions_neighbor[self.drivers[i].now_location][
                                                       int(driver_action - 2)])

        return reward

    def step(self, action):
        reward = self.take_action(action)
        self.time_slot += 1

        self.update_driver_states()
        self.update_POI_states()
        self.update_order_states()

        self.create_orders()
        if self.time_slot + 1 == 40 or self.time_slot + 1 == 80:
            self.init_POI_data()
        # self.create_POI_data()

        obs_next = self.get_observation()

        if self.time_slot <= self.episode_limit - 1:
            done_n = np.zeros(self.agent_n)
        else:

            done_n = np.ones(self.agent_n)

        information = None

        return obs_next, reward, done_n, information

    def close(self):
        pass