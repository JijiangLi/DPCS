#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import yaml
import argparse


def opencv_matrix_constructor(loader, node):

    mapping = loader.construct_mapping(node)
    return mapping


yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:opencv-matrix',
    opencv_matrix_constructor
)


def load_opencv_yaml(filename):
    lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if line.lstrip().startswith('%YAML'):
                continue
            lines.append(line)
    content = ''.join(lines)
    data = yaml.safe_load(content)
    return data

def reshape_to_matrix(data_list, rows, cols):
    if len(data_list) != rows * cols:
        raise ValueError(f" {len(data_list)} ({rows}x{cols})")
    matrix = []
    index = 0
    for r in range(rows):
        row_data = data_list[index: index + cols]
        matrix.append(row_data)
        index += cols
    return matrix


def matrix_to_yaml_list(matrix, precision=None):

    lines = []
    for row in matrix:
        if precision is not None:
            row_str = ", ".join(f"{val:.{precision}g}" for val in row)
        else:
            row_str = ", ".join(str(val) for val in row)
        lines.append(f"- {row_str}")
    return lines


def main():
    parser = argparse.ArgumentParser(
        description=" OpenCV  calibration.yaml convert to  params.yaml"
    )
    parser.add_argument("input_file", help="calibration.yaml path")
    parser.add_argument("output_file", help="params.yaml path")
    args = parser.parse_args()

    calibration = load_opencv_yaml(args.input_file)

    # 2. extract camK, camKc, prjK, prjKc, R, T
    camK = reshape_to_matrix(calibration["camK"]["data"],
                             calibration["camK"]["rows"],
                             calibration["camK"]["cols"])

    camKc = calibration["camKc"]["data"]  # 1 x 4
    prjK = reshape_to_matrix(calibration["prjK"]["data"],
                             calibration["prjK"]["rows"],
                             calibration["prjK"]["cols"])

    prjKc = calibration["prjKc"]["data"]  # 1 x 4

    R = reshape_to_matrix(calibration["R"]["data"],
                          calibration["R"]["rows"],
                          calibration["R"]["cols"])

    T = reshape_to_matrix(calibration["T"]["data"],
                          calibration["T"]["rows"],
                          calibration["T"]["cols"])
    for i in range(3):
        T[i][0] = T[i][0] * 1e-3

    prjRT = []
    for i in range(3):
        row_i = R[i] + [T[i][0]]
        prjRT.append(row_i)


    camRT = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]

    # 5. write to params.yaml
    with open(args.output_file, "w", encoding="utf-8") as f:
        # camK
        f.write("camK:\n")
        for line in matrix_to_yaml_list(camK, precision=6):
            f.write(f"{line}\n")
        f.write("\n")

        # camKc
        f.write("camKc:\n")
        camKc_str = ", ".join(str(val) for val in camKc)
        f.write(f"- {camKc_str}\n")
        f.write("\n")

        # camRT
        f.write("camRT:\n")
        for line in matrix_to_yaml_list(camRT):
            f.write(f"{line}\n")
        f.write("\n")

        # prjK
        f.write("prjK:\n")
        for line in matrix_to_yaml_list(prjK, precision=6):
            f.write(f"{line}\n")
        f.write("\n")

        # prjKc
        f.write("prjKc:\n")
        prjKc_str = ", ".join(str(val) for val in prjKc)
        f.write(f"- {prjKc_str}\n")
        f.write("\n")

        # prjRT
        f.write("prjRT:\n")
        for line in matrix_to_yaml_list(prjRT, precision=6):
            f.write(f"{line}\n")
        f.write("\n")

    print(f"conversion finished！\ninput: {args.input_file}\noutput: {args.output_file}")


if __name__ == "__main__":
    main()
