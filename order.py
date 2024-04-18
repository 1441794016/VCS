class Order:
    def __init__(self, order_id, init_location, generate_time, destination,
                 waiting_time=15, price=20, travel_time=3):
        """
        初始化一个order实例
        :param order_id: order的编号
        :param init_location: order的生成区域
        :param generate_time: order的生成时间
        :param destination: order的目的地
        :param waiting_time: order的可以忍受的等待时间，超过这个时间订单会过期
        :param price: order 的价格
        :param travel_time: 从产生区域到达order的目的地所需要花费的时间
        """
        self.order_id = order_id  # 订单的编号
        self.location = init_location  # 订单在哪个区域
        self.generate_time = generate_time  # 订单的生成时间
        self.destination = destination  # 订单的目的地
        self.waiting_time = waiting_time  # 订单的最大等待时间
        self.price = price  # 这笔订单的价格
        self.travel_time = travel_time  # 订单到达目的地的预期花费时间

        self.overdue_time = self.generate_time + self.waiting_time  # 订单的过期时间
        self.is_matched = False  # 订单是否已经被匹配。False表示还没有被匹配
        self.is_overdue = False  # 订单是否已经过期

    def driver_match(self):
        """
        被一个driver所匹配，修改这笔order的状态为已经被匹配的状态
        :return: no return
        """
        self.is_matched = True
        self.is_overdue = False

