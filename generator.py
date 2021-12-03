import csv
import numpy as np
import pandas as pd
import PIL.Image
from PIL import Image, ImageDraw
import cv2
import dnnlib
import dnnlib.tflib as tflib
import imageio
import matplotlib.pyplot as plt
from pathlib import Path
from pretrained_networks import load_networks
from zipfile import ZipFile
import os


class Generator():
    def __init__(self, network_pkl, direction_name, coefficient, truncation, n_levels, n_photos, type_of_preview,
                 result_dir, generator_number=1):
        self.no_generator = generator_number
        self.coefficient = coefficient  # Siła manipluacji / przemnożenie wektora
        self._truncation = truncation  # Parametr stylegan "jak różnorodne twarze"
        self.n_levels = n_levels  # liczba poziomów manipulacji 1-3
        self.n_photos = n_photos  # Ile zdjęć wygenerować
        if type(network_pkl) is str:
            self._G, self._D, self.Gs = load_networks(network_pkl)
        else:
            self.Gs = network_pkl
        self.preview_face = self.__create_coordinates(1)  # Array z koordynatami twarzy na podglądzie 1
        self.preview_3faces = self.__create_coordinates(3)  # Array z koordynatami twarzy na podglądzie 3
        self.synthesis_kwargs = {}  # Keyword arguments które przyjmuje stylegan
        self.type_of_preview = type_of_preview  # Typ podglądu, wartości: "3_faces", "manipulation" w zależności od tego które ustawienia są zmieniane
        self.dir = {"results": Path(result_dir + str(self.no_generator)),
                    "images": Path(result_dir + str(self.no_generator)) / 'images',
                    "thumbnails": Path(result_dir + str(self.no_generator)) / 'thumbnails',
                    "coordinates": Path(result_dir + str(self.no_generator)) / 'coordinates',
                    "dominance": Path("stylegan2/stylegan2directions/dominance.npy"),
                    "trustworthiness": Path("stylegan2/stylegan2directions/trustworthiness.npy")}
        self.__direction_name = direction_name.lower()  # Wybrany wymiar
        try:
            self.direction = np.load(self.dir[self.__direction_name])  # Wgrany wektor cechy
        except:
            self.direction = np.load(direction_name)

        for directory in self.dir.values():
            if directory.suffix == "": directory.mkdir(exist_ok=True, parents=True)

    @property
    def truncation(self):
        return self._truncation

    @truncation.setter
    def truncation(self, truncation):
        w_avg = self.Gs.get_var('dlatent_avg')
        f = (self.preview_face - w_avg) / self.truncation
        self.preview_face = w_avg + (f) * truncation
        f = (self.preview_3faces - w_avg) / self.truncation
        self.preview_3faces = w_avg + (f) * truncation
        self._truncation = truncation

    # w_avg + (faces_w - w_avg) * self.truncation

    @property
    def direction_name(self):
        return self.__direction_name

    @direction_name.setter
    def direction_name(self, direction_name):
        try:  # Wybrany wymiar
            self.direction = np.load(self.dir[self.__direction_name])  # Wgrany wektor cechy
            self.__direction_name = direction_name.split("/")[-1].replace(".npy", '')
        except:
            self.direction = np.load(direction_name)
            self.__direction_name = direction_name.split("/")[-1].replace(".npy", '')

    def refresh_preview(self):
        """Przełączniki co wywołać w zależności od wartości type_of_preview"""
        if self.type_of_preview == "manipulation":
            return self.__generate_preview_face_manip()
        else:
            return self.__generate_preview_3faces()

    def __create_coordinates(self, n_photos):
        all_z = np.random.randn(n_photos, *self.Gs.input_shape[1:])
        all_w = self.__map_vectors(all_z)
        return self.__truncate_vectors(all_w)

    def change_face(self):
        if self.type_of_preview == "manipulation":
            self.preview_face = self.__create_coordinates(1)
        else:
            self.preview_3faces = self.__create_coordinates(3)

    def __save_image(self, face, face_no, condition):  # Dodać kilka folderów wynikowych
        image_pil = PIL.Image.fromarray(face, 'RGB')
        image_pil.save(
            self.dir["images"] / '{}{}cond{}.png'.format(face_no, self.direction_name, condition))

        image_pil.thumbnail((256, 256))
        image_pil.save(
            self.dir["thumbnails"] / '{}{}cond{}.jpg'.format(face_no, self.direction_name, condition), format='JPEG')

    def generate(self):
        """Zapisuje wyniki"""
        minibatch_size = 8

        self.__set_synthesis_kwargs(minibatch_size)

        coeffs = [i / self.n_levels * self.coefficient if self.n_levels > 0 else i for i in
                  range(-self.n_levels, self.n_levels + 1)]

        for i in range(self.n_photos // minibatch_size + 1):  # dodajmy ładowanie w interfejsie
            all_w = self.__create_coordinates(minibatch_size)

            for k, coeff in enumerate(coeffs):
                manip_w = all_w.copy()

                for j in range(len(all_w)):
                    manip_w[j][0:8] = (manip_w[j] + coeff * self.direction)[0:8]

                manip_images = self.Gs.components.synthesis.run(manip_w, **self.synthesis_kwargs)

                for j in range(len(all_w)):
                    if i * minibatch_size + j < self.n_photos:
                        self.__save_image(manip_images[j], i * minibatch_size + j, k)

            for j, (dlatent) in enumerate(all_w):
                np.save(self.dir["coordinates"] / (str(i * minibatch_size + j) + '.npy'), dlatent[0])

        with ZipFile('face_generation_results.zip', 'w') as zipObj:
            for folderName, subfolders, filenames in os.walk(self.dir["results"]):
                for filename in filenames:
                    filePath = os.path.join(folderName, filename)
                    zipObj.write(filePath, filePath)

    def __generate_preview_face_manip(self):
        """Zwraca array ze zdjeciem, sklejonymi 3 twarzami: w środku neutralna, po bokach zmanipulowana"""
        self.__set_synthesis_kwargs(minibatch_size=3)
        all_w = self.preview_face.copy()

        all_w = np.array([all_w[0], all_w[0], all_w[0]])  # Przygotowujemy miejsca na twarze zmanipulowane

        # Przesunięcie twarzy o wektor (już rozwinięty w 18)
        all_w[0][0:8] = (all_w[0] - self.coefficient * self.direction)[0:8]
        all_w[2][0:8] = (all_w[2] + self.coefficient * self.direction)[0:8]

        all_images = self.Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)

        return np.hstack(all_images)

    def __generate_preview_3faces(self):
        """__generate_preview_face_manip tylko że używa zmiennej preview_3faces zamiast preview_face"""
        self.__set_synthesis_kwargs(minibatch_size=3)
        all_w = self.preview_3faces.copy()

        all_images = self.Gs.components.synthesis.run(all_w, **self.synthesis_kwargs)

        return np.hstack(all_images)

    def __tile_vector(self, faces_w):
        """Przyjmuje listę 512-wymierowych wektorów twarzy i rozwija je w taki które przyjmuje generator"""
        return np.array([np.tile(face, (18, 1)) for face in faces_w])

    def __generate_preview_face_face_3(self):
        """__generate_preview_face_manip tylko że używa zmiennej preview_3faces zamist preview_face"""

    def __map_vectors(self, faces_z):
        """Przyjmuje array wektorów z koordynatami twarzy w Z-space, gdzie losowane są wektory,
        zwraca array przerzucony do w-space, gdzie dzieje się manipulacja"""
        return self.Gs.components.mapping.run(faces_z, None)

    def __truncate_vectors(self, faces_w):
        """Zwraca wektory z faces_w przesunięte w kierunku uśrednionej twarzy"""
        w_avg = self.Gs.get_var('dlatent_avg')
        return w_avg + (faces_w - w_avg) * self.truncation

    def __set_synthesis_kwargs(self, minibatch_size=3):
        """Za pierwszym razem tworzy keyword arguments do gnereowania,
        następnie może być użyta do zienienia minibatch_size"""
        if len(self.synthesis_kwargs) == 0:
            Gs_syn_kwargs = dnnlib.EasyDict()
            Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                                  nchw_to_nhwc=True)
            Gs_syn_kwargs.randomize_noise = False
            self.synthesis_kwargs = Gs_syn_kwargs

        self.synthesis_kwargs.minibatch_size = minibatch_size


class Generator2(Generator):
    """Generator który działa ze styleGANem2, ale bez części graficznych"""
    def __init__(self):
        pass

class GeneratorGraficzny(Generator):
    """Działa tak jak to wyżej, czyli nie usuwamy żadnego kodu tylko przeklejamy nieważne metody tutaj"""
    def __init__(self):
        pass

class Generator3(Generator):
    """Dzaiła ze stylGANem3"""
    def __init__(self):
        self.device = torch.device('cuda:0')
        with open(network_pkl, 'rb') as fp:
            G = pickle.load(fp)['G_ema'].to(self.device)
        
    def generate(self, z_dim):
        zs = torch.randn([10000, G.mapping.z_dim], device=self.device)
        w_stds = G.mapping(zs, None).std(0)
        q = (G.mapping(torch.randn([10,G.mapping.z_dim], device=self.device), None, truncation_psi=0.7) - G.mapping.w_avg) / w_stds
        
        images = G.synthesis(q * w_stds + G.mapping.w_avg)
        
        return images
