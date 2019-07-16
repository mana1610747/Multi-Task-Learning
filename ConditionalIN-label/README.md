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
conditionがtrain.ipynb,valid.ipynbに実装されている．<br>
not conditionがtrain2.ipynb,valid2.ipynbに実装されている．<br>
<br>
conditionでラベルを[1,1,0,0]のように複数タスクを混ぜた場合下図のような画像が出力された．<br>
<img src="./images/mix_noise.png">