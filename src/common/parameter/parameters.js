export let screen_width = document.documentElement.clientWidth;
// 798
export let screen_height = document.documentElement.clientHeight;

export let model_width = screen_width > 1920 ? 1920 : screen_width;
export let model_height = screen_height > 1080 ? 1080 : screen_height;

export let screen_scale_width = model_width/1920;
export let screen_scale_height = model_height/1080;


class System_param{
    // 构造
      constructor(props) {

        // 初始状态
          this.screen_width = document.documentElement.clientWidth;
          this.screen_height = document.documentElement.clientHeight;

          this.model_width = this.screen_width > 1920 ? 1920 : this.screen_width;
          this.model_height = this.screen_height > 1080 ? 1080 : this.screen_height;

          this.screen_scale_width = this.model_width/1920;
          this.screen_scale_height = this.model_height/1080;

      }

}


export let system_param = new System_param();
