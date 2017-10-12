#!/bin/bash

echo numNodes numEdges max_flow cpu_load_time gpu_load_time comute_time max_flow_time
echo

alpha=1
rho=1
iter=1000000

echo 4281081856
echo adhead.n26c10.max.bk
./main ../../../vision/adhead.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 4474560512
echo adhead.n26c100.max.bk
./main ../../../vision/adhead.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 1128079104
echo adhead.n6c10.max.bk
./main ../../../vision/adhead.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 1136908433
echo adhead.n6c100.max.bk
./main ../../../vision/adhead.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 1509191975
echo babyface.n26c10.max.bk
./main ../../../vision/babyface.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 461946278
echo babyface.n6c10.max.bk
./main ../../../vision/babyface.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 488993231
echo babyface.n6c100.max.bk
./main ../../../vision/babyface.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 726357560
echo BL06-camel-med.max.bk
./main ../../../vision/BL06-camel-med.max.bk -alpha $alpha -rho $rho -it $iter

echo 85249685
echo BL06-camel-sml.max.bk
./main ../../../vision/BL06-camel-sml.max.bk -alpha $alpha -rho $rho -it $iter

echo 648618144
echo BL06-gargoyle-med.max.bk
./main ../../../vision/BL06-gargoyle-med.max.bk -alpha $alpha -rho $rho -it $iter

echo 76022706
echo BL06-gargoyle-sml.max.bk
./main ../../../vision/BL06-gargoyle-sml.max.bk -alpha $alpha -rho $rho -it $iter

echo 715618728
echo bone.n6c10.max.bk
./main ../../../vision/bone.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 752992368
echo bone.n6c100.max.bk
./main ../../../vision/bone.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 305528832
echo bone_subx.n26c10.max.bk
./main ../../../vision/bone_subx.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 1274386165
echo bone_subx.n26c100.max.bk
./main ../../../vision/bone_subx.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 356914250
echo bone_subx.n6c10.max.bk
./main ../../../vision/bone_subx.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 375169516
echo bone_subx.n6c100.max.bk
./main ../../../vision/bone_subx.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 592950213
echo bone_subxy.n26c10.max.bk
./main ../../../vision/bone_subxy.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 174540879
echo bone_subxy.n6c10.max.bk
./main ../../../vision/bone_subxy.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 183754645
echo bone_subxy.n6c100.max.bk
./main ../../../vision/bone_subxy.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 284484135
echo bone_subxyz.n26c10.max.bk
./main ../../../vision/bone_subxyz.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 299052619
echo bone_subxyz.n26c100.max.bk
./main ../../../vision/bone_subxyz.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 84199897
echo bone_subxyz.n6c10.max.bk
./main ../../../vision/bone_subxyz.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 88812046
echo bone_subxyz.n6c100.max.bk
./main ../../../vision/bone_subxyz.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 140741321
echo bone_subxyz_subx.n26c10.max.bk
./main ../../../vision/bone_subxyz_subx.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 147978075
echo bone_subxyz_subx.n26c100.max.bk
./main ../../../vision/bone_subxyz_subx.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 41710204
echo bone_subxyz_subx.n6c10.max.bk
./main ../../../vision/bone_subxyz_subx.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 44007502
echo bone_subxyz_subx.n6c100.max.bk
./main ../../../vision/bone_subxyz_subx.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 68869834
echo bone_subxyz_subxy.n26c10.max.bk
./main ../../../vision/bone_subxyz_subxy.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 72486306
echo bone_subxyz_subxy.n26c100.max.bk
./main ../../../vision/bone_subxyz_subxy.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 20465276
echo bone_subxyz_subxy.n6c10.max.bk
./main ../../../vision/bone_subxyz_subxy.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 21637089
echo bone_subxyz_subxy.n6c100.max.bk
./main ../../../vision/bone_subxyz_subxy.n6c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 9997847
echo BVZ-sawtooth0.bk
./main ../../../vision/BVZ-sawtooth0.bk -alpha $alpha -rho $rho -it $iter

echo 9921918
echo BVZ-sawtooth10.bk
./main ../../../vision/BVZ-sawtooth10.bk -alpha $alpha -rho $rho -it $iter

echo 9940887
echo BVZ-sawtooth11.bk
./main ../../../vision/BVZ-sawtooth11.bk -alpha $alpha -rho $rho -it $iter

echo 9939655
echo BVZ-sawtooth12.bk
./main ../../../vision/BVZ-sawtooth12.bk -alpha $alpha -rho $rho -it $iter

echo 9972709
echo BVZ-sawtooth13.bk
./main ../../../vision/BVZ-sawtooth13.bk -alpha $alpha -rho $rho -it $iter

echo 9939818
echo BVZ-sawtooth14.bk
./main ../../../vision/BVZ-sawtooth14.bk -alpha $alpha -rho $rho -it $iter

echo 9946950
echo BVZ-sawtooth15.bk
./main ../../../vision/BVZ-sawtooth15.bk -alpha $alpha -rho $rho -it $iter

echo 9958901
echo BVZ-sawtooth16.bk
./main ../../../vision/BVZ-sawtooth16.bk -alpha $alpha -rho $rho -it $iter

echo 9858132
echo BVZ-sawtooth17.bk
./main ../../../vision/BVZ-sawtooth17.bk -alpha $alpha -rho $rho -it $iter

echo 9970057
echo BVZ-sawtooth18.bk
./main ../../../vision/BVZ-sawtooth18.bk -alpha $alpha -rho $rho -it $iter

echo 9989693
echo BVZ-sawtooth19.bk
./main ../../../vision/BVZ-sawtooth19.bk -alpha $alpha -rho $rho -it $iter

echo 9967649
echo BVZ-sawtooth2.bk
./main ../../../vision/BVZ-sawtooth2.bk -alpha $alpha -rho $rho -it $iter

echo 9952982
echo BVZ-sawtooth3.bk
./main ../../../vision/BVZ-sawtooth3.bk -alpha $alpha -rho $rho -it $iter

echo 9959489
echo BVZ-sawtooth4.bk
./main ../../../vision/BVZ-sawtooth4.bk -alpha $alpha -rho $rho -it $iter

echo 9953369
echo BVZ-sawtooth5.bk
./main ../../../vision/BVZ-sawtooth5.bk -alpha $alpha -rho $rho -it $iter

echo 9933596
echo BVZ-sawtooth6.bk
./main ../../../vision/BVZ-sawtooth6.bk -alpha $alpha -rho $rho -it $iter

echo 9920241
echo BVZ-sawtooth7.bk
./main ../../../vision/BVZ-sawtooth7.bk -alpha $alpha -rho $rho -it $iter

echo 9911899
echo BVZ-sawtooth8.bk
./main ../../../vision/BVZ-sawtooth8.bk -alpha $alpha -rho $rho -it $iter

echo 6493919
echo BVZ-tsukuba0.bk
./main ../../../vision/BVZ-tsukuba0.bk -alpha $alpha -rho $rho -it $iter

echo 6489706
echo BVZ-tsukuba1.bk
./main ../../../vision/BVZ-tsukuba1.bk -alpha $alpha -rho $rho -it $iter

echo 6448721
echo BVZ-tsukuba10.bk
./main ../../../vision/BVZ-tsukuba10.bk -alpha $alpha -rho $rho -it $iter

echo 6473813
echo BVZ-tsukuba11.bk
./main ../../../vision/BVZ-tsukuba11.bk -alpha $alpha -rho $rho -it $iter

echo 6480068
echo BVZ-tsukuba12.bk
./main ../../../vision/BVZ-tsukuba12.bk -alpha $alpha -rho $rho -it $iter

echo 6437777
echo BVZ-tsukuba13.bk
./main ../../../vision/BVZ-tsukuba13.bk -alpha $alpha -rho $rho -it $iter

echo 6491031
echo BVZ-tsukuba14.bk
./main ../../../vision/BVZ-tsukuba14.bk -alpha $alpha -rho $rho -it $iter

echo 6500036
echo BVZ-tsukuba15.bk
./main ../../../vision/BVZ-tsukuba15.bk -alpha $alpha -rho $rho -it $iter

echo 6476370
echo BVZ-tsukuba2.bk
./main ../../../vision/BVZ-tsukuba2.bk -alpha $alpha -rho $rho -it $iter

echo 6468465
echo BVZ-tsukuba3.bk
./main ../../../vision/BVZ-tsukuba3.bk -alpha $alpha -rho $rho -it $iter

echo 6449336
echo BVZ-tsukuba4.bk
./main ../../../vision/BVZ-tsukuba4.bk -alpha $alpha -rho $rho -it $iter

echo 6436156
echo BVZ-tsukuba5.bk
./main ../../../vision/BVZ-tsukuba5.bk -alpha $alpha -rho $rho -it $iter

echo 6442823
echo BVZ-tsukuba6.bk
./main ../../../vision/BVZ-tsukuba6.bk -alpha $alpha -rho $rho -it $iter

echo 6454338
echo BVZ-tsukuba7.bk
./main ../../../vision/BVZ-tsukuba7.bk -alpha $alpha -rho $rho -it $iter

echo 6456195
echo BVZ-tsukuba8.bk
./main ../../../vision/BVZ-tsukuba8.bk -alpha $alpha -rho $rho -it $iter

echo 6490636
echo BVZ-tsukuba9.bk
./main ../../../vision/BVZ-tsukuba9.bk -alpha $alpha -rho $rho -it $iter

echo 10026892
echo BVZ-venus0.bk
./main ../../../vision/BVZ-venus0.bk -alpha $alpha -rho $rho -it $iter

echo 622093964
echo bone_subxy.n26c100.max.bk
./main ../../../vision/bone_subxy.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 9989559
echo BVZ-sawtooth1.bk
./main ../../../vision/BVZ-sawtooth1.bk -alpha $alpha -rho $rho -it $iter

echo 9946690
echo BVZ-sawtooth9.bk
./main ../../../vision/BVZ-sawtooth9.bk -alpha $alpha -rho $rho -it $iter

echo 10020036
echo BVZ-venus1.bk
./main ../../../vision/BVZ-venus1.bk -alpha $alpha -rho $rho -it $iter

echo 9481359
echo BVZ-venus7.bk
./main ../../../vision/BVZ-venus7.bk -alpha $alpha -rho $rho -it $iter

echo 22394650
echo KZ2-sawtooth5.bk
./main ../../../vision/KZ2-sawtooth5.bk -alpha $alpha -rho $rho -it $iter

echo 9986246
echo BVZ-venus10.bk
./main ../../../vision/BVZ-venus10.bk -alpha $alpha -rho $rho -it $iter

echo 9974070
echo BVZ-venus11.bk
./main ../../../vision/BVZ-venus11.bk -alpha $alpha -rho $rho -it $iter

echo 9987470
echo BVZ-venus12.bk
./main ../../../vision/BVZ-venus12.bk -alpha $alpha -rho $rho -it $iter

echo 9979086
echo BVZ-venus13.bk
./main ../../../vision/BVZ-venus13.bk -alpha $alpha -rho $rho -it $iter

echo 9999487
echo BVZ-venus14.bk
./main ../../../vision/BVZ-venus14.bk -alpha $alpha -rho $rho -it $iter

echo 10024216
echo BVZ-venus15.bk
./main ../../../vision/BVZ-venus15.bk -alpha $alpha -rho $rho -it $iter

echo 10025126
echo BVZ-venus16.bk
./main ../../../vision/BVZ-venus16.bk -alpha $alpha -rho $rho -it $iter

echo 10029953
echo BVZ-venus17.bk
./main ../../../vision/BVZ-venus17.bk -alpha $alpha -rho $rho -it $iter

echo 10030561
echo BVZ-venus18.bk
./main ../../../vision/BVZ-venus18.bk -alpha $alpha -rho $rho -it $iter

echo 9938659
echo BVZ-venus19.bk
./main ../../../vision/BVZ-venus19.bk -alpha $alpha -rho $rho -it $iter

echo 10015137
echo BVZ-venus2.bk
./main ../../../vision/BVZ-venus2.bk -alpha $alpha -rho $rho -it $iter

echo 10055034
echo BVZ-venus20.bk
./main ../../../vision/BVZ-venus20.bk -alpha $alpha -rho $rho -it $iter

echo 10056593
echo BVZ-venus21.bk
./main ../../../vision/BVZ-venus21.bk -alpha $alpha -rho $rho -it $iter

echo 9999242
echo BVZ-venus3.bk
./main ../../../vision/BVZ-venus3.bk -alpha $alpha -rho $rho -it $iter

echo 9988103
echo BVZ-venus4.bk
./main ../../../vision/BVZ-venus4.bk -alpha $alpha -rho $rho -it $iter

echo 9977933
echo BVZ-venus5.bk
./main ../../../vision/BVZ-venus5.bk -alpha $alpha -rho $rho -it $iter

echo 9503204
echo BVZ-venus6.bk
./main ../../../vision/BVZ-venus6.bk -alpha $alpha -rho $rho -it $iter

echo 9489593
echo BVZ-venus8.bk
./main ../../../vision/BVZ-venus8.bk -alpha $alpha -rho $rho -it $iter

echo 9984599
echo BVZ-venus9.bk
./main ../../../vision/BVZ-venus9.bk -alpha $alpha -rho $rho -it $iter

echo 25911889
echo KZ2-sawtooth0.bk
./main ../../../vision/KZ2-sawtooth0.bk -alpha $alpha -rho $rho -it $iter

echo 25875109
echo KZ2-sawtooth1.bk
./main ../../../vision/KZ2-sawtooth1.bk -alpha $alpha -rho $rho -it $iter

echo 23626248
echo KZ2-sawtooth10.bk
./main ../../../vision/KZ2-sawtooth10.bk -alpha $alpha -rho $rho -it $iter

echo 25717653
echo KZ2-sawtooth11.bk
./main ../../../vision/KZ2-sawtooth11.bk -alpha $alpha -rho $rho -it $iter

echo 25544776
echo KZ2-sawtooth12.bk
./main ../../../vision/KZ2-sawtooth12.bk -alpha $alpha -rho $rho -it $iter

echo 16956263
echo KZ2-sawtooth13.bk
./main ../../../vision/KZ2-sawtooth13.bk -alpha $alpha -rho $rho -it $iter

echo 25318115
echo KZ2-sawtooth14.bk
./main ../../../vision/KZ2-sawtooth14.bk -alpha $alpha -rho $rho -it $iter

echo 24757806
echo KZ2-sawtooth15.bk
./main ../../../vision/KZ2-sawtooth15.bk -alpha $alpha -rho $rho -it $iter

echo 25588483
echo KZ2-sawtooth16.bk
./main ../../../vision/KZ2-sawtooth16.bk -alpha $alpha -rho $rho -it $iter

echo 8962725
echo KZ2-sawtooth17.bk
./main ../../../vision/KZ2-sawtooth17.bk -alpha $alpha -rho $rho -it $iter

echo 24204753
echo KZ2-sawtooth18.bk
./main ../../../vision/KZ2-sawtooth18.bk -alpha $alpha -rho $rho -it $iter

echo 25304423
echo KZ2-sawtooth19.bk
./main ../../../vision/KZ2-sawtooth19.bk -alpha $alpha -rho $rho -it $iter

echo 23706775
echo KZ2-sawtooth2.bk
./main ../../../vision/KZ2-sawtooth2.bk -alpha $alpha -rho $rho -it $iter

echo 23725598
echo KZ2-sawtooth3.bk
./main ../../../vision/KZ2-sawtooth3.bk -alpha $alpha -rho $rho -it $iter

echo 26060827
echo KZ2-sawtooth4.bk
./main ../../../vision/KZ2-sawtooth4.bk -alpha $alpha -rho $rho -it $iter

echo 25053786
echo KZ2-sawtooth6.bk
./main ../../../vision/KZ2-sawtooth6.bk -alpha $alpha -rho $rho -it $iter

echo 23822083
echo KZ2-sawtooth7.bk
./main ../../../vision/KZ2-sawtooth7.bk -alpha $alpha -rho $rho -it $iter

echo 24537630
echo KZ2-sawtooth8.bk
./main ../../../vision/KZ2-sawtooth8.bk -alpha $alpha -rho $rho -it $iter

echo 19577223
echo KZ2-sawtooth9.bk
./main ../../../vision/KZ2-sawtooth9.bk -alpha $alpha -rho $rho -it $iter

echo 26033334
echo KZ2-venus0.bk
./main ../../../vision/KZ2-venus0.bk -alpha $alpha -rho $rho -it $iter

echo 26471487
echo KZ2-venus1.bk
./main ../../../vision/KZ2-venus1.bk -alpha $alpha -rho $rho -it $iter

echo 25648996
echo KZ2-venus10.bk
./main ../../../vision/KZ2-venus10.bk -alpha $alpha -rho $rho -it $iter

echo 21550886
echo KZ2-venus11.bk
./main ../../../vision/KZ2-venus11.bk -alpha $alpha -rho $rho -it $iter

echo 25801937
echo KZ2-venus12.bk
./main ../../../vision/KZ2-venus12.bk -alpha $alpha -rho $rho -it $iter

echo 20124459
echo KZ2-venus13.bk
./main ../../../vision/KZ2-venus13.bk -alpha $alpha -rho $rho -it $iter

echo 25163879
echo KZ2-venus14.bk
./main ../../../vision/KZ2-venus14.bk -alpha $alpha -rho $rho -it $iter

echo 17945167
echo KZ2-venus15.bk
./main ../../../vision/KZ2-venus15.bk -alpha $alpha -rho $rho -it $iter

echo 26347960
echo KZ2-venus16.bk
./main ../../../vision/KZ2-venus16.bk -alpha $alpha -rho $rho -it $iter

echo 25890773
echo KZ2-venus17.bk
./main ../../../vision/KZ2-venus17.bk -alpha $alpha -rho $rho -it $iter

echo 24973538
echo KZ2-venus18.bk
./main ../../../vision/KZ2-venus18.bk -alpha $alpha -rho $rho -it $iter

echo 8990395
echo KZ2-venus19.bk
./main ../../../vision/KZ2-venus19.bk -alpha $alpha -rho $rho -it $iter

echo 26846719
echo KZ2-venus2.bk
./main ../../../vision/KZ2-venus2.bk -alpha $alpha -rho $rho -it $iter

echo 25846493
echo KZ2-venus20.bk
./main ../../../vision/KZ2-venus20.bk -alpha $alpha -rho $rho -it $iter

echo 25384213
echo KZ2-venus21.bk
./main ../../../vision/KZ2-venus21.bk -alpha $alpha -rho $rho -it $iter

echo 25963294
echo KZ2-venus3.bk
./main ../../../vision/KZ2-venus3.bk -alpha $alpha -rho $rho -it $iter

echo 25971146
echo KZ2-venus4.bk
./main ../../../vision/KZ2-venus4.bk -alpha $alpha -rho $rho -it $iter

echo 25503968
echo KZ2-venus5.bk
./main ../../../vision/KZ2-venus5.bk -alpha $alpha -rho $rho -it $iter

echo 22665917
echo KZ2-venus6.bk
./main ../../../vision/KZ2-venus6.bk -alpha $alpha -rho $rho -it $iter

echo 26444386
echo KZ2-venus7.bk
./main ../../../vision/KZ2-venus7.bk -alpha $alpha -rho $rho -it $iter

echo 26265737
echo KZ2-venus8.bk
./main ../../../vision/KZ2-venus8.bk -alpha $alpha -rho $rho -it $iter

echo 25746039
echo KZ2-venus9.bk
./main ../../../vision/KZ2-venus9.bk -alpha $alpha -rho $rho -it $iter

echo 584478759
echo LB07-bunny-med.max.bk
./main ../../../vision/LB07-bunny-med.max.bk -alpha $alpha -rho $rho -it $iter

echo 69115931
echo LB07-bunny-sml.max.bk
./main ../../../vision/LB07-bunny-sml.max.bk -alpha $alpha -rho $rho -it $iter

echo 1285293527
echo liver.n26c10.max.bk
./main ../../../vision/liver.n26c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 1243009130
echo liver.n26c100.max.bk
./main ../../../vision/liver.n26c100.max.bk -alpha $alpha -rho $rho -it $iter

echo 366993890
echo liver.n6c10.max.bk
./main ../../../vision/liver.n6c10.max.bk -alpha $alpha -rho $rho -it $iter

echo 381771728
echo liver.n6c100.max.bk
./main ../../../vision/liver.n6c100.max.bk -alpha $alpha -rho $rho -it $iter
