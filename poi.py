class POI:
    def __init__(self, init_data_vol, init_location):
        """
        初始化POI实例
        :param init_data_vol: POI实例的可收集的数据量
        :param init_location: POI实例的产生区域编号
        """
        self.AOI = 1  # 一个POI实例产生的初始信息年龄为1
        self.max_AOI = 20  # POI的最大信息年龄；超过这个年龄数据将过期
        self.data_vol = init_data_vol  # POI的数据量
        self.location = init_location  # POI的位置区域编号
        self.is_overdue = False  # 这个POI中的数据是否过期，False表示还未过期
        self.is_collected = False  # 这个POI数据是否已经被收集过了。在更新一个区域中的POIs时我们会把已经被收集的POI实例删除，加快训练速度

    def be_collected(self):
        """
        当这个POI被采集，修改POI实例的状态
        :return: no return
        """
        self.is_collected = True
        self.is_overdue = False
