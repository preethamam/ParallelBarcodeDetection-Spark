"""
/usr/local/Cellar/apache-spark/2.2.0/libexec/sbin/start-master.sh

> http://localhost:8080/ for master connection

spark-submit --master MASTER src/sparksystem/main.py PATH_TO_GT DUMP_PATH PATHS_TO_IMAGESETS

e.g.
spark-submit --master spark://cdmbp.local:7077 src/sparksystem/main.py dataset/annotations/All.csv dump dataset/sampleset/*

NOTE: Images are assumed to be the undistorted image

"""
ZBAR_QUALITY_THRESHOLD = 2
from pyspark import SparkContext, SparkConf
import argparse
import numpy as np
import zbar
import cv2

import datetime
import shutil
import sys
import os

class Clock:
    """
    For rough run-timing

    """
    def __init__(self, title):
        self.title = title
    def __enter__(self):
        self.start = datetime.datetime.now()
    def __exit__(self, exc_type, exc_val, exc_tb):
        print ">> %s: %s" % (
                self.title, str(datetime.datetime.now() - self.start))

def main():
    """
    basic argparse
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('groundtruthcsv')
    parser.add_argument('dumppath')
    parser.add_argument('paths', nargs='+')
    parser.add_argument('--procs', '-p', default="*")
    args = parser.parse_args()

    """
    Spark setup

    """
    appName = 'sparkzbar'
    conf = SparkConf().setAppName(appName).setMaster('local[%s]' % args.procs)
    sc = SparkContext(conf=conf)

    """
    Mapreduce multiscale method amongst (undistorted) image files

    """
    dump_path = os.path.realpath(args.dumppath)
    paths = [os.path.realpath(p) for p in args.paths]
    paths_dist = sc.parallelize(paths)
    crop_rot_path = paths_dist.flatMap(
            lambda path: multiscale_segmentor(path, dump_path))
    segment_count = crop_rot_path.count()  #.map(lambda x: 1).reduce(lambda x, y: x+y)
    print ">> %d segments found" % segment_count

    """
    Mapreduce barcodes

    """
    all_gt_barcodes = load_ground_truth(os.path.realpath(args.groundtruthcsv))

    gt_barcodes = {}
    for filepath in paths:
        filedir, imgfilenamefull = os.path.split(filepath)
        imgfilename, ext = os.path.splitext(imgfilenamefull)
        gt_barcodes.setdefault(imgfilename, []).extend(all_gt_barcodes.get(imgfilename, []))

    detect = True
    if detect:
        # Detect barcodes form segments
        barcodes = crop_rot_path.map(lambda x: read_barcode(x[1]))

        # Reduce, sort, and count
        all_barcodes = barcodes.reduce(lambda a, b: a + b)
        sorted_barcodes = sorted(all_barcodes, key=lambda x: (x['filename'], x['cx']))
        scanned_count = len(sorted_barcodes)

        # For debugging/statistics gathering
        good, bad, all_labels = compute_accuracy(all_barcodes, gt_barcodes)
        good.sort(key=lambda x: (x['filename'], x['cx']))
        with open(os.path.join(dump_path, 'detection.log'), 'w') as f:
            perc = 0 if not len(all_labels) else scanned_count*100.0/len(all_labels)
            f.write(">> %d/%d barcodes found (%f%%)\n" % (scanned_count, len(all_labels), perc))
            f.write('\n'.join([">> {filename}\tgt: {gtlabel}\tscanned: {label}\tIoU: {iou}".format(**bc) for bc in good]))
            f.write('\n\n')
            targ = [g for g in good if g['color'].lower().strip() == 'r']
            xarg = [g for g in all_labels if g['color'].lower().strip() == 'r']
            perc = 0 if not len(xarg) else len(targ)*100.0/len(xarg)
            f.write(">> %d/%d red barcodes found (%f%%)\n" % (len(targ), len(xarg), perc))
            f.write('\n'.join([">> {filename}\tgt: {gtlabel}\tscanned: {label}\tIoU: {iou}".format(**bc) for bc in targ]))
            f.write('\n\n')
            targ = [g for g in good if g['color'].lower().strip() == 'r' and g['label'] == g['gtlabel']]
            xarg = [g for g in good if g['color'].lower().strip() == 'r']
            perc = 0 if not len(xarg) else len(targ)*100.0/len(xarg)
            f.write(">> %d/%d correct red barcodes (out of scanned red barcodes) found (%f%%)\n" % (len(targ), len(xarg), perc))
            f.write('\n'.join([">> {filename}\tgt: {gtlabel}\tscanned: {label}\tIoU: {iou}".format(**bc) for bc in targ]))
            f.write('\n\n')
            targ = [g for g in good if g['color'].lower().strip() == 'g']
            xarg = [g for g in all_labels if g['color'].lower().strip() == 'g']
            perc = 0 if not len(xarg) else len(targ)*100.0/len(xarg)
            f.write(">> %d/%d green barcodes found (%f%%)\n" % (len(targ), len(xarg), perc))
            f.write('\n'.join([">> {filename}\tgt: {gtlabel}\tscanned: {label}\tIoU: {iou}".format(**bc) for bc in targ]))
            f.write('\n\n')
            targ = [g for g in good if g['color'].lower().strip() == 'b']
            xarg = [g for g in all_labels if g['color'].lower().strip() == 'b']
            perc = 0 if not len(xarg) else len(targ)*100.0/len(xarg)
            f.write(">> %d/%d blue barcodes found (%f%%)\n" % (len(targ), len(xarg), perc))
            f.write('\n'.join([">> {filename}\tgt: {gtlabel}\tscanned: {label}\tIoU: {iou}".format(**bc) for bc in targ]))
            f.write('\n\n')

    """
    Check segmentation accuracy

    """
    segmentation_accuracy = True
    if segmentation_accuracy:
        # Group by filename and compute segmentation accuracy per file
        filename_grouped_segments = crop_rot_path.groupByKey()
        rdd_ious = filename_grouped_segments.map(
                lambda x: filename_segmentation_accuracy(
                    x[0], x[1], gt_barcodes.get(x[0], [])))
        ious = rdd_ious.reduce(lambda a, b: a + b)

        # For debugging/statistics gathering
        total = len(ious)
        good = [barcode for barcode in ious if barcode['iou'] >= 0.5]
        inaccuracies = [barcode for barcode in ious if barcode['iou'] < 0.5]
        percentage = 0 if not total else 100.0*len(good)/total
        with open(os.path.join(dump_path, 'segmentation.log'), 'w') as f:
            f.write( ">> Found %d/%d with IoU > .5 (%.2f%%)\n" % (len(good), total, percentage) )
            targ = [barcode for barcode in good if barcode['color'].strip().lower() == 'r']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'r']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d red with IoU > .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            targ = [barcode for barcode in good if barcode['color'].strip().lower() == 'g']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'g']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d green with IoU > .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            targ = [barcode for barcode in good if barcode['color'].strip().lower() == 'b']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'b']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d blue with IoU > .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            bad = [barcode for barcode in ious if barcode['iou'] < 0.5]
            f.write( ">> Inconsistencies: Found %d/%d with IoU < .5 (%.2f%%)\n" % (len(bad), total, percentage) )
            targ = [barcode for barcode in bad if barcode['color'].strip().lower() == 'r']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'r']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d red with IoU < .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            f.write( '\n'.join("   >> {fullinpath}\tid: {index}\t{label}\t{color}\t{iou}".format(**i) for i in targ) )
            f.write( '\n\n' )
            targ = [barcode for barcode in bad if barcode['color'].strip().lower() == 'g']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'g']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d green with IoU < .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            f.write( '\n'.join("   >> {fullinpath}\tid: {index}\t{label}\t{color}\t{iou}".format(**i) for i in targ) )
            f.write( '\n\n' )
            targ = [barcode for barcode in bad if barcode['color'].strip().lower() == 'b']
            xarg = [barcode for barcode in ious if barcode['color'].strip().lower() == 'b']
            percentage = 0 if not xarg else 100.0*len(targ)/len(xarg)
            f.write( "  >> Found %d/%d blue with IoU < .5 (%.2f%%)\n" % (len(targ), len(xarg), percentage) )
            f.write( '\n'.join("   >> {fullinpath}\tid: {index}\t{label}\t{color}\t{iou}".format(**i) for i in targ) )
            f.write( '\n\n' )
        for inaccuracy in inaccuracies:
            fullpath = inaccuracy['fullinpath']
            filename = inaccuracy['filename']
            index = inaccuracy['index']
            sbb = inaccuracy['sbb']
            gtbb = inaccuracy['gtbb']
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
            img = cv2.polylines(img, [np.asarray(sbb).reshape((-1,1,2)).astype(np.int32)], True, (0,0,255), 10)
            img = cv2.polylines(img, [np.asarray(gtbb).reshape((-1,1,2)).astype(np.int32)], True, (255, 255, 255), 10)
            cv2.imwrite(os.path.join(dump_path, 'BAD_%s_%d.jpg' % (filename, index)), cv2.resize(img, (0,0), fx=0.5, fy=0.5))

def merge(x):
    """
    x is 2d tuple of same type

    Returns merged value

    """
    a, b = x
    if isinstance(a, dict):
        return merge_dicts(a, b)
    else:
        return a + b

def merge_dicts(a, b):
    merged = dict(
            (k,v.__class__())
            for it in [a.iteritems(), b.iteritems()]
            for k, v in it)
    for it in [a.iteritems(), b.iteritems()]:
        for k, v in it:
            merged[k] += v
    return merged

def imbothat(img, el):
    """
    Function to mimic MATLAB's imbothat function

    Returns bottom-hat of image

    Bottom-hat defined to be the difference between input and
    the closing of the image

    """
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, el)
    return closing - img


def strel_octogon(radius):
    """
    Returns disk strel of radius radius

    https://edoras.sdsu.edu/doc/matlab/toolbox/images/strel.html#223424

    """
    sel = np.ones(((2*radius+1),(2*radius+1)), dtype=np.uint8)
    rows, cols = sel.shape
    rad2 = radius ** 2
    for i in xrange(radius/3*2):
        for j in xrange(radius/3*2-i):
            sel[i,j] = 0
            sel[i,cols-1-j] = 0
            sel[rows-1-i,j] = 0
            sel[rows-1-i,cols-1-j] = 0
    return sel


def strel_disk(radius):
    """
    Returns disk strel of radius radius

    https://edoras.sdsu.edu/doc/matlab/toolbox/images/strel.html#223367

    """
    sel = np.ones(((2*radius-1),(2*radius-1)), dtype=np.uint8)
    rows, cols = sel.shape
    rad2 = radius ** 2
    for i in xrange(rows/2):
        for j in xrange(cols/2):
            if (i+1-radius)**2 + (j+1-radius) ** 2 >= rad2:
                sel[i,j] = 0
                sel[i,cols-1-j] = 0
                sel[rows-1-i,j] = 0
                sel[rows-1-i,cols-1-j] = 0
    return sel


def strel(el_type, size):
    """
    Function to mimic MATLAB's strel function

    """
    if el_type == 'disk':
        return strel_disk(size)
    elif el_type == 'octagon':
        return strel_octogon(size)


def load_ground_truth(ground_truth_csv_path):
    """
    Returns dictionary

        [filename] => dict(cx, cy, label, color, points)

    :param ground_truth_csv_path: Path to ground truth CSV in the form of

        FILENAME,LABEL,COLOR,X1,Y1,X2,Y2,X3,Y3,X4,Y4,

    """
    with open(ground_truth_csv_path) as all_data:
        data = [
            (lambda x: (x[0], x[1], x[2],
                int(x[3]), int(x[4]), int(x[5]), int(x[6]),
                int(x[7]), int(x[8]), int(x[9]), int(x[10]),))(line.split(','))
            for line in all_data.readlines()]
    full_data = {}  # [filename] -> [(cx, cy, label, color, points)]
    for filename, label, color, x1, y1, x2, y2, x3, y3, x4, y4 in data:
        L = full_data.setdefault(filename.split(os.path.extsep)[0].strip(), [])
        cx, cy = int((x1+x2+x3+x4)*.25), int((y1+y2+y3+y4)*0.25)
        L.append({
                "cx": cx,
                "cy": cy,
                "filename": filename,
                "label": label.strip(),
                "color": color.strip(),
                "points": ((x1, y1), (x2, y2), (x3, y3), (x4, y4), )
            })
    return full_data


def bb_intersection_over_union(pointsA, pointsB):
    A = np.zeros((1944, 2592), dtype=np.uint8)
    B = np.zeros((1944, 2592), dtype=np.uint8)
    cv2.fillConvexPoly(A, np.asarray(pointsA), (1))
    cv2.fillConvexPoly(B, np.asarray(pointsB), (1))
    A = A.astype(bool)
    B = B.astype(bool)
    return float(np.sum(A & B)) / float(np.sum(A | B))


def filename_segmentation_accuracy(filename, segments, filename_ground_truth):
    """
    Prints accuracy for segments and ground truth for a particular filename

    :param filename: filename for segments and ground truth
    :param segments: list of dict (path, cx, cy, points)
    :param filename_ground_truth: list of ground truth dicts

    """
    max_iou = [(0, None),] * len(filename_ground_truth)  # [1 or 0 for each barcode if IoU > 50%]
    for si, segment in enumerate(segments):
        fullpath = segment['fullinpath']
        # Get ground truth barcodes
        for i, gt in enumerate(filename_ground_truth):
            if max_iou[i][0] < 0.5:
                # only check if we don't have a good enough segment
                iou = bb_intersection_over_union(
                        segment['points'],
                        gt['points'])
                if iou >= max_iou[i][0]:
                    max_iou[i] = (iou, gt, i, segment['points'])
    ious = [dict(gt.items() + [
        ('fullinpath', fullpath),
        ('iou', iou),
        ('index', i),
        ('gtbb', gt['points']),
        ('sbb', sbb),
        ]) for iou, gt, i, sbb in max_iou]
    return ious

def compute_accuracy(barcodes, ground_truth):
    """
    Given barcodes, return two lists
        - tuple of (filename, discovered label, GT label, IoU) for each barcode
        - missing labels: (filename, label, color)
    And total ground truth labels

    :param barcodes: list of (filename, label, cx, cy, points)
    :param ground_truth: (see function load_ground_truth()) dictionary

        [filename] => (cx, cy, label, color, points)

    NOTE: filename should not have extension

    """
    label_iou = []
    discovered = {}  # [filename] => set of labels
    num_barcodes = len(barcodes)
    for i, item in enumerate(barcodes):
        filename, label, cx, cy, points = (
                item['filename'], item['label'],
                item['cx'], item['cy'], item['points'])
        # Get ground truth barcodes
        gts = ground_truth.get(filename, [])
        gts_search = [gt['label'] for gt in gts]
        # look for label in gts_search
        if label in gts_search:
            minid = gts_search.index(label)
            mingt = gts[minid]
        else:
            if i < num_barcodes:
                # not matching with red or is incorrect, try matching later
                barcodes.append(item)
                continue
            # Scan must be incorrect
            # Find closest ground truth barcode to see how incorrect
            mindist2, mingt, minid = 1000000, None, None
            # Find ground truth barcode closest to center
            for j, gt in enumerate(gts):
                if (gt['color'].lower().strip() == 'r' and
                        gt['label'] in discovered.get(filename, [])):
                    continue
                gtcx, gtcy = gt['cx'], gt['cy']
                dist = (gtcx - cx) ** 2 + (gtcy - cy) ** 2
                if dist < mindist2:
                    mindist2, mingt, minid = dist, gt, j

        if mingt is None:
            continue

        # Tally that we discovered a barcode for this filename
        disc = discovered.setdefault(filename, set())
        disc.add(label)

        # Get IoU and append to discoveries
        gtlabel, gtcolor = mingt['label'], mingt['color']
        iou = bb_intersection_over_union(points, mingt['points'])
        label_iou.append({
            'filename': filename,
            'id': minid,
            'cx': cx,
            'cy': cy,
            'label': label,
            'gtlabel': gtlabel,
            'iou': iou,
            'color': gtcolor})

    # Check which barcodes we are missing
    missing = []
    for filename, disc in discovered.iteritems():
        for gt in ground_truth.get(filename, []):
            gtlabel = gt['label']
            if gtlabel not in disc:
                gtcolor = gt['color']
                missing.append((filename, gtlabel, gtcolor))

    # Count how many we are supposed to have
    all_labels = [label for gt in ground_truth.values() for label in gt]
    return label_iou, missing, all_labels


def multiscale_segmentor(filepath, folderpath2write):
    """
    Performs multi-Scale segmentation on filename

    Assumes that files is undistorted

    Returns list of tuples: (filename, segments),
    where segments is dict with keys:
      - filename
      - path
      - cx
      - cy
      - points

    """
    # Opening structuring element radius
    SE_Radius = range(3, 15+1, 3)
    # NOTE: range has exclusive end; want to include 15

    # Delta value to remove blob
    delsig = 0.0;

    # Create a structuring element of appropriate shape/size
    seTopBotHat = strel('octagon', 18);
    seDilate = strel('disk', 5);
    seClose  = strel('disk', 5);
    seOpenLast = strel('disk', 5);

    # Image storing lists
    IgrayArray = []
    rgbArray = []

    # Store filenames in this cell array
    imUndistorted = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if imUndistorted is None:
        return []

    rows, cols, colordepth = imUndistorted.shape

    # Initialize image open/close array
    segments = []  # (crop-rot filename, centerx)

    # (Image resize?) grayscale conversion
    Igray = cv2.cvtColor(imUndistorted, cv2.COLOR_BGR2GRAY)

    # Store grayscale image
    IgrayArray.append(Igray)

    # Morpho Ops
    tophatimg = cv2.morphologyEx(Igray, cv2.MORPH_TOPHAT, seTopBotHat)
    bothatimg = imbothat(Igray, seTopBotHat)
    enhanceimg = tophatimg + bothatimg
    dilateimg = cv2.dilate(enhanceimg, seDilate)
    closeimg = cv2.morphologyEx(dilateimg, cv2.MORPH_CLOSE, seClose)
    closeimg = closeimg.astype(np.float64)

    # Obtain the max value over imopens
    multiscaleResponseimgold = np.maximum.reduce([
            cv2.morphologyEx(closeimg, cv2.MORPH_OPEN, strel('octagon', r))
            for r in SE_Radius])

    # Opening image to remove some blobs
    # (can play with the ops like open or close)
    openCloseimg = cv2.morphologyEx(
        cv2.morphologyEx(multiscaleResponseimgold, cv2.MORPH_OPEN, seOpenLast),
        cv2.MORPH_CLOSE, seOpenLast)

    # Binarised image
    ret, BW = cv2.threshold(
            openCloseimg.astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Blob removal
    # Connected components
    connectivity = 8  # This is the default in MATLAB for 2D
    ret, CC2 = cv2.connectedComponents(BW, connectivity);
    cc_areas = []
    for component_value in xrange(1, np.max(CC2)+1):
        component = (CC2 == component_value)
        area = np.sum(component)
        component = component.astype(dtype=np.uint8) * 255
        cc_areas.append((component, area))
    cc, areas = zip(*cc_areas)
    areas = np.asarray(areas)
    mu_morph, sigma_morph = np.mean(areas), np.std(areas)

    # Remove smaller area lesser than sigma_morph
    BW3 = BW.copy()
    threshold = round(delsig * sigma_morph)
    for component, area in cc_areas:
        if area < threshold:
            BW3[component != 0] = 0

    # TODO:
    # BW3 = funct_stageI_filter(BW3)

    # Get bounding boxes of connected components
    ret, CC3 = cv2.connectedComponents(BW3, connectivity);
    allBBoxCoords = []
    allRotatedRect = []
    for component_value in xrange(1, np.max(CC2)+1):
        component = (CC3 == component_value).astype(np.uint8)
        component[component != 0] = 255
        im, cnt, hierarchy = cv2.findContours(component,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        rotated_rect = cv2.minAreaRect(cnt[0])
        box = cv2.boxPoints(rotated_rect)
        allBBoxCoords.append(box)
        allRotatedRect.append(rotated_rect)

    # Filename parts
    filedir, imgfilenamefull = os.path.split(filepath)
    imgfilename, ext = os.path.splitext(imgfilenamefull)

    write_outlined_file = False
    if write_outlined_file:
        outlined = Igray.copy()
        for boundingbox, rotatedrect in zip(allBBoxCoords, allRotatedRect):
            hcorners = np.asarray(boundingbox).reshape((-1,1,2)).astype(np.int32)
            # draw polyline on side-by-side post-homography graphic
            cv2.polylines(outlined, [hcorners], True, (10,10,255), 3)
        str_imgfilaname_cntr = '%s.jpg' % imgfilename
        save_path = os.path.join(folderpath2write, str_imgfilaname_cntr)
        cv2.imwrite(save_path, outlined)

    padding = 400
    padded_Igray = np.zeros((Igray.shape[0] + 2*padding, Igray.shape[1] + 2*padding), dtype=Igray.dtype)
    padded_Igray[padding:padding+Igray.shape[0], padding:padding+Igray.shape[1]] = Igray
    for imgCnt, (boundingbox, rotatedrect) in enumerate(zip(allBBoxCoords, allRotatedRect)):
        croppedimg = padded_Igray.copy()

        # Rotate for axis oriented crop
        center, size, angle = rotatedrect
        # Account for padding
        center = tuple(map(lambda x: x+padding, center))
        M = cv2.getRotationMatrix2D(center,angle,1.0)
        croppedimg = cv2.warpAffine(croppedimg, M, (croppedimg.shape[1], croppedimg.shape[0]))
        half_w, half_h = map(lambda x: x * 0.5, size)
        croppedimg = croppedimg[
                max(int(center[1]-half_h), 0):int(center[1]+half_h),
                max(int(center[0]-half_w), 0):int(center[0]+half_w)]

        # Get bbox information
        bx, by = zip(*boundingbox)
        bx = [int(x) for x in bx]
        by = [int(y) for y in by]
        boundingbox = zip(bx, by)

        # Save
        str_imgfilaname_cntr = '%s.%03i.jpg' % (imgfilename, imgCnt)
        save_path = os.path.join(folderpath2write, str_imgfilaname_cntr)
        cv2.imwrite(save_path, croppedimg)
        segments.append((imgfilename, {
            'path': save_path,
            'filename': imgfilename,
            'fullinpath': filepath,
            'cx': int(rotatedrect[0][0]),
            'cy': int(rotatedrect[0][1]),
            'points': boundingbox
            }))
    return segments


def read_barcode(segment):
    """
    Returns unique barcodes found in segment

    Uses both grayscale and Otis threshold images to zbar scan

    Arg segment is dict with keys:
      - filename
      - path
      - cx
      - cy
      - points

    Returns list of barcodes, dict with keys (segment + 'label' key):
      - filename
      - path
      - cx
      - cy
      - points
      - label

    """
    scanner = zbar.Scanner()
    image_file_path = segment['path']
    gray_image = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)
    ret, bw_image = cv2.threshold(
            gray_image, 0, 255,
            cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Scan both Otsu and grayscale and combine
    results = scanner.scan(bw_image) + scanner.scan(gray_image)
    # Make results unique
    unique = set()
    results = [r for r in results if r.data not in unique and not unique.add(r.data)]
    # Augment segment with label
    return [dict(segment.items() + [('label', r.data)]) for r in results]


if __name__ == '__main__':
    main()
