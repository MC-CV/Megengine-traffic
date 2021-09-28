# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import numpy as np


def py_cpu_nms(dets, thresh):
    x1 = np.ascontiguousarray(dets[:, 0])
    y1 = np.ascontiguousarray(dets[:, 1])
    x2 = np.ascontiguousarray(dets[:, 2])
    y2 = np.ascontiguousarray(dets[:, 3])

    areas = (x2 - x1) * (y2 - y1)
    order = dets[:, 4].argsort()[::-1]
    keep = list()

    while order.size > 0:
        pick_idx = order[0]
        keep.append(pick_idx)
        order = order[1:]

        xx1 = np.maximum(x1[pick_idx], x1[order])
        yy1 = np.maximum(y1[pick_idx], y1[order])
        xx2 = np.minimum(x2[pick_idx], x2[order])
        yy2 = np.minimum(y2[pick_idx], y2[order])

        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[pick_idx] + areas[order] - inter, 1e-5)

        order = order[iou <= thresh]

    return keep

def py_soft_nms(dets, thresh):
    N = len(dets[:,0])
    for i in range(N):
        maxscore = dets[i, 4]
        maxpos = i

        tx1 = dets[i,0]
        ty1 = dets[i,1]
        tx2 = dets[i,2]
        ty2 = dets[i,3]
        ts = dets[i,4]

        pos = i + 1
	# get max box
        while pos < N:
            if maxscore < dets[pos, 4]:
                maxscore = dets[pos, 4]
                maxpos = pos
            pos = pos + 1

	# add max box as a detection 
        dets[i,0] = dets[maxpos,0]
        dets[i,1] = dets[maxpos,1]
        dets[i,2] = dets[maxpos,2]
        dets[i,3] = dets[maxpos,3]
        dets[i,4] = dets[maxpos,4]

	# swap ith box with position of max box
        dets[maxpos,0] = tx1
        dets[maxpos,1] = ty1
        dets[maxpos,2] = tx2
        dets[maxpos,3] = ty2
        dets[maxpos,4] = ts

        tx1 = dets[i,0]
        ty1 = dets[i,1]
        tx2 = dets[i,2]
        ty2 = dets[i,3]
        ts = dets[i,4]

        pos = i + 1
	# NMS iterations, note that N changes if detection dets fall below threshold
        while pos < N:
            x1 = dets[pos, 0]
            y1 = dets[pos, 1]
            x2 = dets[pos, 2]
            y2 = dets[pos, 3]
            s = dets[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    
                    if ov > thresh: 
                        weight = 1 - ov
                    else:
                        weight = 1
                    
#                     weight = np.exp(-(ov * ov)/0.5)

                    dets[pos, 4] = weight*dets[pos, 4]
		    
		    # if box score falls below threshold, discard the box by swapping with last box
		    # update N
                    if dets[pos, 4] < thresh:
                        dets[pos,0] = dets[N-1, 0]
                        dets[pos,1] = dets[N-1, 1]
                        dets[pos,2] = dets[N-1, 2]
                        dets[pos,3] = dets[N-1, 3]
                        dets[pos,4] = dets[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep
