import { columns_classification, columns_detail_classification, columns_3rd_classification } from './columns_classification'
import { colums_detection ,colums_detail_detection } from './columns_detection'


const classification_title = ['Network','Params','Flops','Top1/Top5','href']
const detect_detail_title = ['Checkpoint_name','train_mem','inference_time','box_AP','mask_AP', 'href']
const detail_title_arr = [classification_title, detect_detail_title]


export {
    columns_classification, columns_detail_classification,columns_3rd_classification,
    colums_detection ,colums_detail_detection,
    detail_title_arr
}