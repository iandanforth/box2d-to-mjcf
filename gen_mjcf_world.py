import numpy as np
import dill as pickle
from bipedal_walker import BipedalWalkerHardcore

from mjcf import elements as e
from random import random, uniform
from colors import get_rgb, viridis
from Box2D.b2 import (circleShape)


def get_mjc_box(pos, size, rgba=[0.5, 0.5, 0.5, 1]):
    body = e.Body(
        pos=pos
    )
    geom = e.Geom(
        type="box",
        size=size,
        rgba=rgba
    )
    body.add_children([
        geom
    ])

    return body


def base_setup():
    #########################
    # Level 1
    mujoco = e.Mujoco(
        model="empty"
    )

    #########################
    # Level 2
    option = e.Option(
        integrator="RK4",
        timestep=0.01
    )
    asset = e.Asset()
    worldbody = e.Worldbody()

    size = e.Size(
        njmax=4000,
        nconmax=4000,
    )

    mujoco.add_children([
        option,
        size,
        asset,
        worldbody
    ])

    ######################
    # Level 3

    # Asset
    tex1 = e.Texture(
        builtin="gradient",
        height=100,
        rgb1=[0.9, 0.9, 1.0],  # Bipedal Walker Sky Purple
        rgb2=[0.9, 0.9, 1.0],
        type="skybox",
        width=100
    )
    tex2 = e.Texture(
        builtin="flat",
        height=1278,
        mark="cross",
        markrgb=[1, 1, 1],
        name="texgeom",
        random=0.01,
        rgb1=[0.8, 0.6, 0.4],
        rgb2=[0.8, 0.6, 0.4],
        type="cube",
        width=127
    )
    tex3 = e.Texture(
        builtin="checker",
        height=[100],
        name="texplane",
        rgb1=[0, 0, 0],
        rgb2=[0.8, 0.8, 0.8],
        type="2d",
        width=100
    )
    mat1 = e.Material(
        name="MatPlane",
        reflectance=0.5,
        shininess=1,
        specular=1,
        texrepeat=[60, 60],
        texture="texplane"
    )
    mat2 = e.Material(
        name="geom",
        texture="texgeom",
        texuniform=True
    )
    asset.add_children([
        tex1,
        tex2,
        tex3,
        mat1,
        mat2,
    ])

    # Worldbody
    light = e.Light(
        cutoff=100,
        diffuse=[1, 1, 1],
        dir=[-0, 0, -1.3],
        directional=True,
        exponent=1,
        pos=[0, 0, 1.3],
        specular=[.1, .1, .1]
    )
    floor_geom = e.Geom(
        conaffinity=1,
        condim=3,
        material="MatPlane",
        name="floor",
        pos=[0, 0, 0],
        rgba=[0.8, 0.9, 0.8, 1],
        size=[40, 40, 40],
        type="plane"
    )

    worldbody.add_children([
        light,
        floor_geom,
    ])

    return mujoco, worldbody


def path_to_pos_size(path):
    '''
    Take a list of vertices and convert that to a position and size
    '''
    path_len = len(path)
    if path_len not in [2, 4]:
        print("Warning: Encountered a path of unhandled length")
        print(path)
        print(len(path))
        return None, None

    if path_len == 2:
        a, b = path
        aX, aY = a
        bX, bY = b
        pos = [(aX + bX) / 2, 0, (aY + bY) / 2]

        half_width = max(abs(aX - bX), 0.1) / 2
        half_height = max(abs(aY - bY), 0.1) / 2
    else:
        a, b, c, d = path
        aX, aY = a
        bX, bY = b
        cX, cY = c
        dX, dY = d

        if aX != bX:
            print("Warning: Encountered non-rectangular shape")
            print(path)
            return None, None

        pos = [(aX + cX) / 2, 0, (aY + cY) / 2]

        half_width = max(abs(aX - cX), 0.1) / 2
        half_height = max(abs(aY - cY), 0.1) / 2

    depth = 1
    size = [half_width, depth, half_height]

    return pos, size


def main():

    # Create an empty world
    mujoco, worldbody = base_setup()

    # Add terrain
    filename = "terrain-with-boxes"
    drawlist = pickle.load(open("{}.p".format(filename), "rb"))

    # Draw walker and terrain
    mjc_boxes = []
    for obj in drawlist:
        # Collision bodies
        pos, size = path_to_pos_size(obj["path"])
        if pos:
            mjc_box = get_mjc_box(pos, size)
            mjc_boxes.append(mjc_box)

    worldbody.add_children(mjc_boxes)

    # Save as unifed XML
    model_xml = mujoco.xml()
    with open("{}.xml".format(filename), 'w') as fh:
        fh.write(model_xml)


if __name__ == "__main__":
    main()
