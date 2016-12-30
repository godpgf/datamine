##Alpha Mine

###单位

| 单位      | 举例     |
| -------- | -------- |
| 价格      | open、close、low、high、vwap |
| 成交量    | vol      |
| 天       |          |
| 率       | returns  |
| 可比数    |          |
| 布尔      |          |

###原子alpha

| 操作符     | 参数1     | 参数2     | 参数3     | 返回值     |
| --------- | -------- | --------- | -------- | --------- |
| +、- 、mid | all       | 同参数1   | 无        | 同参数1    |
| *         | all       | all      | 无        | 可比数     |
| /         | all       | all      | 无        | 率        |
| abs       | 可比数、率  | 无       | 无        | 同参数1    |
| log       | all       | 无       | 无        | 可比数     | 
| sign      | 可比数、率  | 无       | 无        | 率        | 
| >、<、==、\|\| | all    | 同参数1  |无        | 布尔        |
| avg、delay、delta、sum | all       | 天       | 无        | 同参数1    |
| if        | 布尔       | all     | all       | 同参数2或3  |
| rank、scale、ts_min、ts_max、ts_argmax、ts_argmin | all      | 无      | 无        | 率         |
| correlation、covariance | all | 同参数1 | 天    | 率         |
| scale、signedpower | all       | 可比数    | 无        | 可比数     |
| decay_linear、ts_rank、stddev | all     | 天      | 无        | 率         |
| indneutralize | all    | 无       | 无        | 同参数1    |
| product     | 率       | 天        | 无        | 同参数1    |