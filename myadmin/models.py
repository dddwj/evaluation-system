from django.db import models

# Create your models here.
class models_logs(models.Model):
    id = models.AutoField(primary_key=True,verbose_name='模型编号，主键')
    startMonth = models.DateTimeField(verbose_name='训练数据起始月')
    endMonth = models.DateTimeField(verbose_name='训练数据结束月')
    trainDate = models.DateTimeField(verbose_name='训练时间')
    trainer = models.CharField(max_length=20, verbose_name='训练人')
    inUseFlag = models.IntegerField(default=0, verbose_name='标记是否正在使用')
    comment = models.CharField(max_length=100, null=True,verbose_name='说明')

    objective = models.CharField(max_length=15, default='regression')
    metric = models.CharField(max_length=10, default='mse')
    learning_rate = models.DecimalField(default=0.2,decimal_places=2, max_digits=3)    # ref: https://docs.djangoproject.com/en/2.1/ref/models/fields/#decimalfield
    feature_fraction = models.DecimalField(default=0.6,decimal_places=2, max_digits=3)
    bagging_fraction = models.DecimalField(default=0.6,decimal_places=2, max_digits=3)
    max_depth = models.IntegerField(default=14)
    num_leaves = models.IntegerField(default=220)
    bagging_freq = models.IntegerField(default=5)
    min_data_in_leaf = models.IntegerField(default=10)
    min_gain_to_split = models.IntegerField(default=0)
    lambda_l1 = models.DecimalField(default=1,decimal_places=2, max_digits=5)
    lambda_l2 = models.DecimalField(default=1,decimal_places=2, max_digits=5)
    verbose = models.IntegerField(default=0)

    trainSuccess = models.IntegerField(verbose_name='是否成功训练/是否在本地有模型',default=0)

    class Meta:
        db_table = 'models_logs'     # 表名
        verbose_name = '模型训练记录'
        verbose_name_plural = verbose_name
        ordering = ['id']

