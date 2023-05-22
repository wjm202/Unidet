import os 
import json
with open('/paddle/detectron2/UniDet/datasets/oid/semi_annotations/oid_challenge_2019_val_expanded.1@1.json') as f:
        params = json.load(f)
        #加载json文件中的内容给params
        for i in range(len(params['images'])):
            params['images'][i]['id']=str(int('0x'+params['images'][i]['id'],16))
        #修改内容
        for i in range(len(params['annotations'])):
            params['annotations'][i]['image_id']=str(int('0x'+params['annotations'][i]['image_id'],16))
        print("params",params)
        #打印
        
        dict = params
        #将修改后的内容保存在dict中
        f.close()
        #关闭json读模式
with open('/paddle/detectron2/UniDet/datasets/oid/semi_annotations/oid_challenge_2019_val_expanded.1@1.json','w') as r:
#定义为写模式，名称定义为r

    json.dump(dict,r)
        #将dict写入名称为r的文件中
        