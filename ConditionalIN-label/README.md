labelを用いてconditionを実装した場合(condition)と，学習データに様々なタスクを混ぜて学習させた場合(not condition)の結果の比較．
<table border="1">
  <tr>
  <td>task</td><td>paint</td><td>noise</td><td>gaussian</td><td>mosaic</td>
  </tr><tr>
  <td>original</td><td><img src="./images/original_paint.png"></td><td><img src="./images/original_noise.png"></td><td><img src="./images/original_gaussian.png"></td><td><img src="./images/original_mosaic.png"></td>
  </tr><tr>
  <td>not condition</td><td><img src="./images/not_condition_paint.png"></td><td><img src="./images/not_condition_noise.png"></td><td><img src="./images/not_condition_gaussian.png"></td><td><img src="./images/not_condition_mosaic.png"></td>
  </tr><tr>
  <td>condition</td><td><img src="./images/condition_paint.png"></td><td><img src="./images/condition_noise.png"></td><td><img src="./images/condition_gaussian.png"></td><td><img src="./images/condition_mosaic.png"></td>
  </tr>
</table>
<br>
train.ipynb,valid.ipynb:condition<br>
train2.ipynb,valid2.ipynb:not condition<br>