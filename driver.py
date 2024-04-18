import numpy as np


class Driver:
    def __init__(self, driver_id, init_location, init_capacity, init_collect_rate, init_offload_rate):
        """
        对一个对象实例进行初始化
        :param driver_id: driver的编号
        :param init_location: driver的初始所在的区域编号
        :param init_capacity: driver（vehicle）的最大存储量
        :param init_collect_rate: driver的收集数据的速率
        :param init_offload_rate: driver卸载数据的速率
        """
        self.driver_id = driver_id
        self.init_location = init_location  # 初始化的region
        self.now_location = self.init_location  # 现在所在的区域
        self.capacity = float(init_capacity)  # 车辆的最大存储量，到达这个数量时必须卸载数据
        self.is_serving = False  # 是否正在提供服务
        self.server_order = None  # 正在服务的订单
        self.destination = None  # 如果接受了订单，订单的目的地
        self.time_arrive_order_destination = None  # 到达订单目的地的时间
        self.collect_rate = float(init_collect_rate)  # 车的数据收集速率，假定不同车可能不同，和车的配置有关，比如配备sensor的性能
        self.offload_rate = float(init_offload_rate)  # 车向云端进行数据卸载的速率

        self.is_collecting = False  # 是否正在收集数据
        self.POI_collected = None  # 被收集的POI
        self.POI_collected_data_vol = None  # 被采集的POI的数据量
        self.data_vol = 0  # 已收集的数据量
        self.time_complete_sensing = None  # 完成数据收集的时间

        self.is_offloading = False  # 是否正在卸载数据
        self.time_complete_offloading = None  # 完成卸载数据的时间

        self.stay = False  # 是否留在原地。当车的决策是留在原地时，这个值为True
        self.random_walk = False  # 是否随机游走。当车不做出决策时，车辆会随机游走

        self.is_dispatched = False  # 是否被派遣到临近区域
        self.dispatched_destination = None  # 被调度去的目的地（临近区域）

    def dispatched(self, destination):
        self.is_dispatched = True
        self.is_serving = False
        self.is_offloading = False
        self.is_collecting = False
        self.dispatched_destination = destination

    def complete_dispatched(self):
        self.is_dispatched = False
        self.is_serving = False
        self.is_offloading = False
        self.is_collecting = False
        self.now_location = self.dispatched_destination
        self.dispatched_destination = None

    def order_match(self, order, match_time):
        """
        进行订单匹配，修改driver实例的状态
        :param order: 进行匹配的订单
        :param match_time: 订单匹配时间
        :return: no return
        """
        self.is_serving = True
        self.is_offloading = False
        self.is_collecting = False
        self.is_dispatched = False
        self.server_order = order
        self.destination = order.destination
        self.time_arrive_order_destination = max(int(order.travel_time), 2) + match_time  # 设定到达目的地至少需要两个time-slot

    def drop_off(self):
        """
        driver实例完成订单，修改状态
        :return: no return
        """
        self.now_location = self.destination
        self.is_serving = False
        self.server_order = None
        self.destination = None
        self.time_arrive_order_destination = None
        self.is_collecting = False
        self.is_offloading = False
        self.is_dispatched = False

    def collect_data(self, POI, collect_time):
        """
        开始收集数据，对POI进行数据收集
        :param POI: 被收集的POI
        :param collect_time: 开始收集数据的时间
        :return: 收集数据的奖励 即收集的数据量 / 数据的信息年龄
        """
        self.is_collecting = True
        self.POI_collected = POI
        self.POI_collected_data_vol = POI.data_vol
        self.time_complete_sensing = collect_time + int(POI.data_vol / self.collect_rate)
        return POI.data_vol / float(POI.AOI)

    def complete_collect(self):
        """
        完成POI数据的采集，修改driver的状态
        :return: no return
        """
        self.data_vol += min(self.data_vol + self.POI_collected_data_vol, self.capacity)
        self.POI_collected = None
        self.is_collecting = False
        self.is_serving = False
        self.is_offloading = False
        self.is_dispatched = False
        self.POI_collected_data_vol = None
        self.time_complete_sensing = None

    def offload_data(self, offload_time):
        """
        driver将所有的数据进行卸载
        :param offload_time: 开始卸载数据的时间
        :return: no return
        """
        self.is_offloading = True
        self.is_serving = False
        self.is_collecting = False
        self.is_dispatched = False
        self.time_complete_offloading = offload_time + max(int(self.capacity / self.offload_rate), 1)

    def complete_offload(self):
        self.is_offloading = False
        self.is_serving = False
        self.is_collecting = False
        self.is_dispatched = False
        self.data_vol = 0
        self.time_complete_offloading = None

