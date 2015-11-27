#!/usr/bin/env python2
# now targeting Python 2.7.5 because the CIS computers do not have numpy for Python 3
from __future__ import print_function, division
import os
import math
import time
import unittest

# tested with CIS version of numpy
import numpy
import numpy.testing

# the pypng library is in the 'libraries' folder
import libraries


class Image(object):
    """ Instances of this class represent an image. All images are represented internally as m x n x 3 numpy matrices,
    or equivalently m x n matrices of pixels of 3 channels each. For greyscale images, for every pixel the values of
    the channel are equal.

    While the image is in memory, the channels of all pixels are represented as floating point numbers between 0 and 1
    inclusive, but when writing an image back to a .png file, the channels are scaled to integers from [0, 255] i.e.
    8-bit unsigned integers.
    """

    def __init__(self, rows):
        """ Return an Image as represented by :param: rows.

        :type rows: iterable
        :param rows: an iterable holding the rows (at least one) of the image, in order, where every row has the same
        number of pixels, and each pixel has three channels (red, green and blue), and the value of each channel is
        from the real range [0, 1]
        """
        if len(rows) == 0:
            raise ValueError("There must be at least one row in the image.")
        # as all the rows are meant to be the same length
        # can look at just the length of the first row, and that should be the length of all the rows
        number_of_columns = len(rows[0])

        for row_index, row in enumerate(rows):
            assert all(len(previous_row) == number_of_columns for previous_row in rows[:row_index])
            if len(row) != number_of_columns:
                raise ValueError(
                    "Row at index {} has length {}; should be {}".format(row_index, len(row), number_of_columns))
            for column_index, pixel in enumerate(row):
                if len(pixel) != 3:
                    raise ValueError(
                        "Pixel at index {} has {} channels; should be 3".format((row_index, column_index), len(pixel)))
                assert len(pixel) == 3
                for component in pixel:
                    if not (0.0 <= component <= 1.0):
                        raise ValueError(
                            "Pixel {} at index {} has a non-[0, 1] component".format(pixel, (row_index, column_index)))
                assert all(0.0 <= component <= 1.0 for component in pixel)
        assert all(len(row) == number_of_columns for row in rows)
        self.matrix = numpy.asarray(rows, dtype=float)
        assert self.matrix.ndim == 3
        assert self.matrix.shape == (len(rows), number_of_columns, 3)

    @classmethod
    def read_from(cls, path_to_file):
        """ Create an Image instance representing the image from a .png file

        :rtype : Image
        :type path_to_file: str
        :param path_to_file: path to a .png image file (24-bit RGB or 8-bit greyscale, no alpha)
        :return: an Image object representing the image from that file
        """
        reader = libraries.png.Reader(filename=path_to_file)
        # initially an m x n image is read in as a m x 3n matrix
        # i.e. the channels of each pixel aren't packed together
        raw_rows = tuple(reader.asRGB8()[2])
        # need to build the m x n x 3 matrix
        # i.e. with the channels of each pixel `boxed` together in a tuple
        boxed_row_boxed_pixels = []
        for raw_row in raw_rows:
            # each row should be cleanly divisible by 3, as each pixel has 3 channels
            assert len(raw_row) % 3 == 0
            boxed_row = []
            red_channels = raw_row[0::3]
            green_channels = raw_row[1::3]
            blue_channels = raw_row[2::3]
            for red, green, blue in zip(red_channels, green_channels, blue_channels):
                # here is where it's assumed that all channel values are 8-bit
                # so they can be normalised to the range [0, 1] by dividing by 255
                boxed_pixel = red / 255.0, green / 255.0, blue / 255.0
                boxed_row.append(boxed_pixel)
            assert len(boxed_row) == len(raw_row) // 3
            boxed_row_boxed_pixels.append(boxed_row)
        assert len(boxed_row_boxed_pixels) == len(raw_rows)
        return cls(boxed_row_boxed_pixels)

    def write_to(self, path):
        """ Write out this Image to a RGB .png file.

        :param path: where the resulting .png file should be written to
        """
        # configure a writer
        writer = libraries.png.Writer(width=self.number_of_columns, height=self.number_of_rows, greyscale=False,
                                      bitdepth=8)
        # need to unpack the image matrix back to boxed row flat pixels format
        rows_of_flat_pixels = []
        for row_of_boxed_pixels in self.matrix:
            row_of_flat_pixels = []
            for boxed_pixel in row_of_boxed_pixels:
                assert len(boxed_pixel) == 3
                assert all(0.0 <= component <= 1.0 for component in boxed_pixel)
                for component in boxed_pixel:
                    # concomitantly convert pixel components to be in the natural range [0, 255] (i.e. 8-bit channels)
                    row_of_flat_pixels.append(int(component * 255))
            assert all(type(component) is int for component in row_of_flat_pixels)
            assert all(0 <= component <= 255 for component in row_of_flat_pixels)
            rows_of_flat_pixels.append(row_of_flat_pixels)
        assert len(rows_of_flat_pixels) == len(self.matrix)
        assert all(len(row_of_flat_pixels) == self.number_of_columns * 3 for row_of_flat_pixels in rows_of_flat_pixels)
        # pass the boxed row flat pixels to the writer
        with open(path, mode='wb') as output_filehandle:
            writer.write(output_filehandle, rows_of_flat_pixels)

    @property
    def number_of_rows(self):
        return len(self.matrix)

    @property
    def number_of_columns(self):
        return len(self.matrix[0])

    @libraries.functools.lru_cache()
    def _padded(self, pad_width, mode):
        """ Helper function to speed up implementation of neighbourhood() method - memoise padded matrices
        """
        # numpy's pad function (for a 3D matrix of rows x columns x channels) in brief
        # pad_width = [(number of rows to prepend, number of rows to append),
        # (number of cols to prepend, number of cols to append),
        #              (number of channels to prepend, number of channels to append)]
        # mode = "edge" means that the extreme-most row or column will be used to pad where applicable
        return numpy.pad(self.matrix, pad_width, mode)

    def neighbourhood(self, size, row_index, column_index):
        """ Return a square matrix representing the n x n square neighbourhood omega around some pixel p (where n = `size`
        parameter). The index of p should be specified by (row_index, column_index). If the square neighbourhood omega of p
        would go out-of-bounds of the image, the extreme-most rows and/or columns will be repeated and the image matrix
        padded with these, before returning the neighbourhood.
        """
        # PRECONDITIONS (assuming correct Python types)
        if not (0 <= row_index < self.number_of_rows):
            raise ValueError("Row index out of bounds")
        if not (0 <= column_index < self.number_of_columns):
            raise ValueError("Column index out of bounds")
        if size <= 0:
            raise ValueError("Size must be positive")
        if size % 2 == 0:
            raise ValueError("Size must be an odd number")

        # FUNCTION BODY
        p = self.matrix[row_index][column_index]
        halfsize = size // 2
        # omega is a square `size` x `size` submatrix where the central pixel is `p`
        # if the omega required is large and p is near the edge of the image matrix, it may be necessary use a padded
        # version of the image matrix
        # assuming this may be necessary, pad the image matrix by the maximum possible rows/columns that may be
        # required, i.e. halfsize rows and columns either side

        # it is not necessary to pad the matrix in most cases, but for simplicity of coding I will pad the matrix every
        # time (as it still works)

        padded_matrix = self._padded(pad_width=((halfsize, halfsize), (halfsize, halfsize), (0, 0)), mode="edge")

        # work out adjusted indices
        original_minimum_row_index = row_index - halfsize
        original_maximum_row_index = row_index + halfsize
        original_minimum_column_index = column_index - halfsize
        original_maximum_column_index = column_index + halfsize

        adjusted_minimum_row_index = original_minimum_row_index + halfsize
        adjusted_maximum_row_index = original_maximum_row_index + halfsize
        adjusted_minimum_column_index = original_minimum_column_index + halfsize
        adjusted_maximum_column_index = original_maximum_column_index + halfsize

        # slice the relevant submatrix from the padded matrix
        omega = padded_matrix[adjusted_minimum_row_index:adjusted_maximum_row_index + 1,
                adjusted_minimum_column_index:adjusted_maximum_column_index + 1]

        # POSTCONDITIONS
        if omega.ndim != 3:
            raise RuntimeError("Neighbourhood is not three-dimensional: actual dimensions are {}".format(omega.ndim))
        if omega.shape != (size, size, 3):
            raise RuntimeError(
                "Neighbourhood is not {n} x {n} x 3: actual shape is {dims}".format(n=size, dims=omega.shape))
        if not numpy.allclose(omega[halfsize][halfsize], p):
            raise RuntimeError("Central pixel of neighbourhood is not p: it is {}".format(omega[halfsize][halfsize]))
        return omega

    def apply_bilateral_filter(self, size, sigma_distance, sigma_intensity_difference):
        """ Apply a bilateral filter to this Image, according to the specified parameters.

        :param size: the bilateral filter mask will be `size` x `size` in size
        :param sigma_distance: the standard deviation value to use in the Gaussian function for the distance mask
        :param sigma_intensity_difference: the standard deviation value to use in the Gaussian function for the
        intensity difference
        """
        def gaussian_function(standard_deviation):
            """ Generate a Gaussian function.

            :param standard_deviation: the value for the parameter sigma_distance in the resulting Gaussian function
            :return: a Gaussian function with the specified standard deviation
            """
            pi = math.pi
            e = math.e
            sd = standard_deviation
            return lambda x: (1 / (sd * (2 * pi) ** (1 / 2))) * e ** -(x ** 2 / (2 * sd ** 2))

        halfsize = size // 2
        intensity_gaussian_function = gaussian_function(sigma_intensity_difference)
        distance_gaussian_function = gaussian_function(sigma_distance)

        # the distance mask is the same for every pixel so can build it once now
        distance_mask = numpy.zeros((size, size, 3))
        for y in range(-halfsize, halfsize + 1):
            for x in range(-halfsize, halfsize + 1):
                # for each position (x, y), calculate the distance from the central pixel
                distance = numpy.array([(x ** 2 + y ** 2) ** (1 / 2) for _ in range(3)])
                distance_gaussian = numpy.array([distance_gaussian_function(component) for component in distance])
                distance_mask[halfsize + y][halfsize + x] = distance_gaussian

        # calculate the response at every pixel p
        # this could be done in parallel but for simplicity I do it one pixel at a time
        for row_index in range(self.number_of_rows):
            for column_index in range(self.number_of_columns):
                p = self.matrix[row_index][column_index]
                omega = self.neighbourhood(size, row_index, column_index)

                # must build a unique intensity difference mask for each pixel
                intensity_mask = numpy.zeros((size, size, 3))
                for y in range(-halfsize, halfsize + 1):
                    for x in range(-halfsize, halfsize + 1):
                        # for each position (x, y), calculate the difference in intensity
                        intensity_difference = abs(p - omega[halfsize + y][halfsize + x])
                        intensity_difference_gaussian = numpy.array(
                            [intensity_gaussian_function(component) for component in intensity_difference])
                        intensity_mask[halfsize + y][halfsize + x] = intensity_difference_gaussian

                combined_mask = intensity_mask * distance_mask
                weighting = sum(sum(combined_mask))
                applied_to_omega = combined_mask * omega
                response = sum(sum(applied_to_omega)) / weighting
                self.matrix[row_index][column_index] = response


class ImageTest(unittest.TestCase):

    def test_read_image(self):
        Image.read_from("images/testA.png")

    def test_dimensions(self):
        test_image = Image.read_from("images/testA.png")
        self.assertEqual(test_image.matrix.shape, (512, 512, 3))
        self.assertEqual(test_image.number_of_rows, 512)
        self.assertEqual(test_image.number_of_columns, 512)

    def test_neighbourhood(self):
        m = Image([[[1 / 1, 1 / 1, 1 / 1], [1 / 2, 1 / 2, 1 / 2], [1 / 3, 1 / 3, 1 / 3]],
                   [[1 / 4, 1 / 4, 1 / 4], [1 / 5, 1 / 5, 1 / 5], [1 / 6, 1 / 6, 1 / 6]],
                   [[1 / 7, 1 / 7, 1 / 7], [1 / 8, 1 / 8, 1 / 8], [1 / 9, 1 / 9, 1 / 9]]])
        clone = m.neighbourhood(3, 1, 1)
        self.assertTrue(numpy.allclose(m.matrix, clone))
        omega_3x3_about_row_0_column_0 = Image([[[1 / 1, 1 / 1, 1 / 1], [1 / 1, 1 / 1, 1 / 1], [1 / 2, 1 / 2, 1 / 2]],
                                            [[1 / 1, 1 / 1, 1 / 1], [1 / 1, 1 / 1, 1 / 1], [1 / 2, 1 / 2, 1 / 2]],
                                            [[1 / 4, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1 / 4], [1 / 5, 1 / 5, 1 / 5]]])
        self.assertTrue(numpy.allclose(omega_3x3_about_row_0_column_0.matrix, m.neighbourhood(3, 0, 0)))

    def test_bilateral_filter(self):
        test_image = Image.read_from("images/testC.png")
        test_image.apply_bilateral_filter(3, 10, 500)
        test_image.write_to("testC_filtered.png")


def process_image(path_to_png, size, distance_sd, intensity_sd, path_to_output):
    image = Image.read_from(path_to_png)
    image.apply_bilateral_filter(size=size,
                                 sigma_distance=distance_sd,
                                 sigma_intensity_difference=intensity_sd)
    image.write_to(path_to_output)


def do_all(size):
    # size should be odd and at least 3
    for letter in "A", "B", "C":
        for sigma_distance in 0.1, 1, 10, 50, 250:
            for sigma_intensity_difference in 0.1, 1, 10, 50, 250:
                print("Image {}, size={}, distance_sd={}, intensity_sd={}".format(letter,
                                                                                  size,
                                                                                  sigma_distance,
                                                                                  sigma_intensity_difference))
                in_path = "images/test{}.png".format(letter)
                out_path = "out/{}_s{}_d{}_i{}.png".format(letter, size, sigma_distance,
                                                           sigma_intensity_difference)
                if os.path.exists(out_path):
                    print("Skipping! Image {} already exists".format(out_path))
                    continue
                else:
                    print("Processing...")
                start = time.time()
                process_image(in_path,
                              size,
                              sigma_distance,
                              sigma_intensity_difference,
                              out_path)
                time_taken = time.time() - start
                print("Time taken: {} seconds".format(time_taken))

if __name__ == "__main__":
    # I used the tutorial at https://docs.python.org/3/howto/argparse.html when writing this section
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_png", help="Path to the .png image to which the bilateral filter is to be applied",
                        type=str)
    parser.add_argument("size", help="The order of the bilateral filter matrix to be used",
                        type=int)
    parser.add_argument("distance_sd",
                        help="The sigma value for the Gaussian function used in the Euclidean distance matrix",
                        type=float)
    parser.add_argument("intensity_sd",
                        help="The sigma value for the Gaussian function used in the intensity difference matrix",
                        type=float)
    parser.add_argument("path_to_output", help="Where the resulting .png image should be written out to",
                        type=str)
    arguments = parser.parse_args()
    process_image(arguments.path_to_png,
                  arguments.size,
                  arguments.distance_sd,
                  arguments.intensity_sd,
                  arguments.path_to_output)
