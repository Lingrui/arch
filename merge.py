#!/usr/bin/env python
import os
import simplejson as json
import cv2
import numpy as np
import picpac

DBS = [('D.db', ['m_D_0.db', 'm_D_1.db',
                 'm_D_3.db', 'm_D_4.db',
                 'm_D_5.db','m_D_6.db']),
       ('E.db', ['m_E_0.db', 'm_E_1.db',
                 'm_E_3.db','m_E_4.db',
                 'm_E_5.db','m_E_6.db'])]

RX = 0.2    # reduce to 0.2 of original size

for out_path, in_paths in DBS:
    print "Creating", out_path

    l = 1

    try:
        os.remove(out_path)
    except:
        pass

    outb = picpac.Writer(out_path)
    for in_path in in_paths:

        # iterate one database
        inb = picpac.Reader(os.path.join('db', in_path))
        cc = 0
        for _, _, _, fields in inb:
            #print pid, label, len(fields)
            image_buf, anno_buf = fields

            arr = np.fromstring(image_buf, np.uint8)
            image = cv2.imdecode(arr, cv2.CV_LOAD_IMAGE_COLOR)
            # down size image
            image2 = cv2.resize(image, None, fx=RX, fy=RX)

            #print image.shape, '=>', image2.shape

            image_buf = cv2.imencode('.png', image2)[1].tostring()

            # original annotation doesn't have labels
            anno = json.loads(anno_buf)
            for shape in anno['shapes']:
                shape['type'] = 'ellipse' 
                shape['label'] = l  # set label
                pass
            anno_buf = json.dumps(anno)
            outb.append(l, image_buf, anno_buf)
            cc += 1
            pass
        print "Loaded %d items from %s with label=%d"  % (cc, in_path, l)

        l += 1
        pass

