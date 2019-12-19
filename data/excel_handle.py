import xlrd
import os
import json
import re
import copy

#         key: '1',
#         Network: 'mobilenet-V1-1.0',
#         Flops: 569,
#         Params:4.24,
#         Top1:'74.37/91.87',
#         'samples/sec':4050,
#


def get_url(string):
    # findall() 查找匹配正则表达式的字符串
    list_1 = string.split('`')
    sourse = list_1[-1]
    url = re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', list_1[0])
    if len(url) > 0:
        index = list_1[0].find(url[0])
        url = list_1[0][index:]
        # print(url)
        return (url, sourse)
    else:
        return ('', sourse)


def segmentation_str(str, type=0):
    return_arr = []
    list_1 = str.split(';')
    for item in list_1:
        if type==0:
            return_arr.append(item.split(':')[1])
        else:
            return_arr.append((item.split(':')[1]).split('%')[0])
    return return_arr


def xxs_model(path='/Users/liusen/Documents/JavaScript/react-demo/my-app/data/model-zoo-imagenet_3rd.xlsx'):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 根据顺序获取sheet
    num = 0
    title_arr = ['Network', 'Params', 'Flops', 'Top1', 'Top5', 'speed']
    Classification_Image_3rd_table_data = list()
    for i in sheet.get_rows():
        num += 1
        # if num == 4:
        #     for j in i:
        #         arr.append(j)
        # print(i)#获取每一行的数据
        # print(i[0].value)
        if num == 1:
            continue
        if i[0].value != '':
            obj_item = dict()
            obj_item['Top1/Top5'] = []
            for j in range(6):
                if j != 3 and j != 4:
                    obj_item[title_arr[j]] = i[j].value
                else:
                    obj_item['Top1/Top5'].append(str(i[j].value))
            obj_item['Top1/Top5'] = '/'.join(obj_item['Top1/Top5'])

            Classification_Image_3rd_table_data.append(obj_item)
    return Classification_Image_3rd_table_data


def wzh_model(path='/Users/liusen/Documents/JavaScript/react-demo/my-app/data/网站使用版.xlsx'):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 根据顺序获取sheet
    num = 0
    rows = 0
    title_arr = ['Network', 'Params', 'Flops', 'Top1', 'Top5', 'speed']
    Classification_Image_3rd_table_data = list()
    obj_item = dict()
    for i in sheet.get_rows():
        if rows > 246:
            break

        rows += 1
        if i[1].value == '':
            continue
        num += 1
        # print(i)
        if num % 3 == 1:
            obj_item['Network'] = i[0].value
            flops_params = i[1].value
            flops_params_arr = segmentation_str(flops_params)
            obj_item['Flops'] = flops_params_arr[0]
            obj_item['Params'] = flops_params_arr[1]
            obj_item['speed'] = i[2].value
            obj_item['type'] = i[3].value
            obj_item['download'] = {}
            obj_item['download']['baidu_download'] = i[6].value
            obj_item['download']['baidu_code'] = i[7].value
        if num % 3 == 2:
            top_1_5 = i[1].value
            top_1_5_arr = segmentation_str(top_1_5, 1)
            obj_item['Top1/Top5'] = '{}/{}'.format(top_1_5_arr[0], top_1_5_arr[1])
        if num % 3 == 0:
            obj_item['source'] = []
            obj_item['source'].append(get_url(i[1].value)[0])
            obj_item['source'].append(get_url(i[1].value)[1])
            Classification_Image_3rd_table_data.append(obj_item)
            obj_item = dict()

    return Classification_Image_3rd_table_data


def wzh_model_own(path='/Users/liusen/Documents/JavaScript/react-demo/my-app/data/网站使用版.xlsx'):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 根据顺序获取sheet
    num = 0
    rows = 0
    title_arr = ['Network', 'Params', 'Flops', 'Top1', 'Top5', 'speed', 'baidu_download', 'baidu_code']
    Classification_Image_3rd_table_data = list()
    obj_item = dict()
    for i in sheet.get_rows():
        rows += 1
        if rows < 250:
            continue
        if i[1].value == '':
            continue
        num += 1
        # print(i)
        if num % 2 == 1:
            obj_item['Network'] = i[0].value
            flops_params = i[1].value
            flops_params_arr = segmentation_str(flops_params)
            obj_item['Flops'] = flops_params_arr[0]
            obj_item['Params'] = flops_params_arr[1]
            obj_item['speed'] = i[2].value
            obj_item['type'] = i[3].value
            obj_item['download'] = {}
            obj_item['download']['baidu_download'] = i[6].value
            obj_item['download']['baidu_code'] = i[7].value
        if num % 2 == 0:
            top_1_5 = i[1].value
            top_1_5_arr = segmentation_str(top_1_5, 1)
            obj_item['Top1/Top5'] = '{}/{}'.format(top_1_5_arr[0], top_1_5_arr[1])
            Classification_Image_3rd_table_data.append(obj_item)
            obj_item = dict()


    return Classification_Image_3rd_table_data


def own_charts(path='/Users/liusen/Documents/JavaScript/react-demo/my-app/data/网站使用版.xlsx'):
    book = xlrd.open_workbook(path)
    sheet = book.sheet_by_index(0)  # 根据顺序获取sheet
    num = 0
    rows = 0
    title_arr = ['Network', 'Params', 'Flops', 'Top1', 'Top5', 'speed']
    Classification_Image_3rd_table_data = list()
    obj_item = []
    last_key = ''
    obj_list_item = dict()
    for i in sheet.get_rows():
        rows += 1
        if rows == 1:
            last_item = i
        if rows < 250:
            continue
        if i[1].value == '':
            continue
        num += 1
        # print(i)
        # demo_data = {
        #             // vec
        #               top1
        #               top5
        #               flops
        #               params
        #               backbone
        # 'mobilenet': [
        #     [4050, 74.37, 91.87, 569, 4.24, 'mobilenet-V1-1.0'],
        #     [5900, 77.36, 93.57, 317, 2.59, 'mobilenet-v1-0.75'],
        #     [9000, 77.86, 93.46, 150, 1.34, 'mobilenet-v1-0.5']
        # ],
        # }
        if num % 2 == 1:
            if i[5].value != '':
                if last_key != i[5].value:
                    last_key = i[5].value
                    obj_item = [0] * 6
                    obj_list_item[last_key] = []
                flops_params = i[1].value
                flops_params_arr = segmentation_str(flops_params)
                # vec
                obj_item[0] = i[2].value
                # floaps
                obj_item[3] = float(flops_params_arr[0][:-1])
                # params
                obj_item[4] = float(flops_params_arr[1][:-1])
                obj_item[5] = i[0].value

        if num % 2 == 0 and last_item[5].value != '':
            top_1_5 = i[1].value
            top_1_5_arr = segmentation_str(top_1_5, 1)
            obj_item[1] = top_1_5_arr[0]
            obj_item[2] = top_1_5_arr[1]
            obj_list_item[last_key].append(copy.deepcopy(obj_item))
            # obj_item['Top1/Top5'] = '{}/{}'.format(top_1_5_arr[0], top_1_5_arr[1])
            # Classification_Image_3rd_table_data.append(obj_item)

        last_item = i
    return obj_list_item


if __name__ == '__main__':
    # print(xxs_model())
    print(wzh_model())
    # print(wzh_model_own())
    # print(own_charts())
