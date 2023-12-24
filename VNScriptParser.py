import json
import os

from PyGameTransitions import Dissolve, FadeInCharacter, FadeOutCharacter
from VNImageGenerator import VNImageGenerator


class VNScriptParser:
    def __init__(self, script_json_path=None, graphics_path="output"):
        self._graphics_path = graphics_path
        self._script_json_path = script_json_path
        self._script_lines = None
        self._script_line_index = 0

        if script_json_path is not None:
            self.parse_script(script_json_path)

    def parse_script(self, script_json_path):
        # Usage
        pipeline_path = "SDCheckpoints/aingdiffusion_v13.safetensors"
        upscaler_model_id = "stabilityai/sd-x2-latent-upscaler"
        generator = VNImageGenerator(pipeline_path, upscaler_model_id)

        self._script_json_path = script_json_path

        # load json data from file (contains a list of dictionaries)
        with open(self._script_json_path, "r") as f:
            self._script_lines = json.load(f)

        # reset the script line index
        self._script_line_index = 0

        # go thought all of the change_scene messages and check to see if there is a background image in the graphics folder
        # this is the format of the change_scene message:
        #{
        #    "type": "change_scene",
        #    "location_name": "Quaint Village Cafe",
        #    "location_description": "A charming, rustic cafe located in the heart of a quaint village. The cafe is bustling with activity, filled with the aroma of freshly brewed coffee and baked goods. The sun shines brightly outside, casting a warm, inviting glow on the scene."
        #}
        for script_line in self._script_lines:
            if script_line["type"] == "change_scene":
                # check to see if there is a background image for this scene
                location_name = script_line["location_name"]
                location_name = location_name.replace(" ", "_")
                background_image_path = f"{self._graphics_path}/{location_name}.png"
                if os.path.isfile(background_image_path):
                    # if there is a background image, add it to the message
                    script_line["image_path"] = background_image_path
                else:
                    # if there is no background image, set the path to None
                    print(f"No background image found for scene: {location_name} - Generating one now...")
                    # generate a background image for this scene
                    generator.generate_background_image(location_name, script_line["location_description"])
                    script_line["image_path"] = background_image_path

    def get_next_line(self, game):
        # get the next line from the script
        script_line = self._script_lines[self._script_line_index]

        new_text = False

        # check to see if this is a change_scene message
        if script_line["type"] == "change_scene":
            # check to see if there is a background image for this scene
            if script_line["image_path"] is not None:
                # if there is a background image, set it as the background image for the game
                game.transition_to_new_background(script_line["image_path"], Dissolve())
            else:
                print(f"No background image found for scene! {script_line['location_name']}")

        elif script_line["type"] == "add_character":
            # this is the game function that adds a character to the game:
            # load_character(self, position, character_path, expression, facing_direction, transition=None)
            # {
            #         "type": "add_character",
            #         "character_name": "Hoshi Yumeko",
            #         "expression": "neutral",
            #         "facing_direction": "right",
            #         "position": "left"
            #     }
            character_path = script_line["character_name"].replace(" ", "_")
            character_path = f"{self._graphics_path}/{character_path}"
            game.load_character(script_line["position"], character_path, script_line["expression"], script_line["facing_direction"], transition=FadeInCharacter())
        elif script_line["type"] == "remove_character":
            game.clear_character(script_line["position"], transition=FadeOutCharacter())
        elif script_line["type"] == "dialog":
            game.load_next_dialogue(script_line["text"], script_line["displayed_speaker"])
            new_text = True
        elif script_line["type"] == "event_text":
            game.load_next_dialogue(script_line["description"])
            new_text = True
        else:
            print(f"Unhandled script line type: {script_line['type']}")

        # increment the script line index
        self._script_line_index += 1

        return new_text


