import json
import os
import sys

from CharacterBuilder import CharacterBuilder
from VNImageGenerator import VNImageGenerator

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


class SceneBuilder:
    def __init__(self, chat_llm=None):
        if chat_llm is None:
            self.chat_llm = ChatOpenAI(
                model_name='gpt-4',
                temperature=0.0
            )
        else:
            self.chat_llm = chat_llm

    visual_novel_scripting_prompt = """
        As a script generator for a visual novel, your task is to create JSON-formatted messages that dictate the flow and elements of the story. Each message type has specific fields that must be included for accurate representation in the game engine. Follow these guidelines and format specifications:

        Message Types and Formats:

        Change Scene:
           - Format: {"type": "change_scene", "description": "<scene_description>"}
           - Example: {"type": "change_scene", "description": "A quaint village on a sunny day."}

        Change to CG (Computer Graphics):
           - Format: {"type": "change_cg", "description": "<CG_description>"}
           - Example: {"type": "change_cg", "description": "A dramatic close-up of the mysterious locket."}

        Add Character:
           - Format: {"type": "add_character", "character_name": "<name>", "expression": "<expression>", "facing_direction": "<left/right>", "position": "<left/middle/right>", "transition": "<transition_type>"}
           - Expressions: ["cry", "angry", "smile", "neutral", "disappointed", "blush", "scared", "laugh", "yell"]
           - Example: {"type": "add_character", "character_name": "John", "expression": "smile", "facing_direction": "left", "position": "middle", "transition": "fade_in"}

        Remove Character:
           - Format: {"type": "remove_character", "position": "<left/middle/right>", "transition": "<transition_type>"}
           - Example: {"type": "remove_character", "position": "middle", "transition": "fade_out"}

        Dialog:
           - Format: {"type": "dialog", "text": "<dialog_text>", "speaker": "<speaker_name>", "displayed_speaker": "<display_name>"}
           - Example: {"type": "dialog", "text": "I knew you'd come.", "speaker": "John", "displayed_speaker": "Mysterious Man"}

        Event Text:
           - Format: {"type": "event_text", "description": "<event_description>"}
           - Example: {"type": "event_text", "description": "The wind howls ominously, rattling the old windows."}

        Question to Player:
           - Format: {"type": "question", "text": "<question_text>", "options": ["<option1>", "<option2>", ...]}
           - Example: {"type": "question", "text": "What will you do next?", "options": ["Stay silent", "Ask about the locket"]}

        Remember to:
        - Maintain narrative coherence and logical consistency in your messages.
        - Ensure that transitions, character placements, and scene changes align with the story's flow.
        - Format the output as JSON objects as specified for each message type.
        - The whole output should be a list of JSON objects, each representing a message.

        Generate a visual novel script based on the above specifications for a given storyline or input.
        """

    def generate_scene(self, scene_prompt, character_list, background_information=None):
        # the character list is a list of dictionaries, convert it to a JSON string
        character_list_json = json.dumps(character_list)

        # if there is no background information, set it to an empty string
        if background_information is None:
            background_information = ""

        # generate the prompt using the scene prompt, character list, and background information
        prompt = f"This is medium length VN scene. The scene is set in {scene_prompt}. The characters are {character_list_json}. {background_information}"

        message = self.chat_llm(
            [
                SystemMessage(role="VisualNovelScripter", content=SceneBuilder.visual_novel_scripting_prompt),
                HumanMessage(content=prompt),
            ]
        )

        formatted_json = message.content.strip()

        # Attempt to parse the string response into a JSON object
        try:
            structured_output = json.loads(formatted_json)
        except json.JSONDecodeError:
            # Handle the case where the response is not in proper JSON format
            structured_output = "The AI's response was not in a valid JSON format. Please check the AI's output."

        return structured_output


if __name__ == "__main__":
    from util import load_secrets

    load_secrets()

    character_list = []

    # Test the CharacterBuilder class
    character_names = ["Hoshi_Yumeko", "Hikari_Yumeno", "Aiko_Tanaka"]

    #each character has a folder under the output folder, with a prompt.json file, load these to a list
    for character_name in character_names:
        with open(f"output/{character_name}/prompt.json", "r") as f:
            character_prompt = json.load(f)
        character_list.append(character_prompt)

    # Test the SceneBuilder class
    SceneBuilder = SceneBuilder()
    scene_prompt = "The scene is set in a quaint village on a sunny day. The characters are meeting at the local cafe " \
                   "to discuss their plans for the upcoming festival. "
    background_information = "The festival is a yearly event that celebrates the town's history and culture. "
    structured_output = SceneBuilder.generate_scene(scene_prompt, character_list, background_information)

    print(structured_output)

    #save the output to a file
    with open("output/scene.json", "w") as f:
        json.dump(structured_output, f, indent=4)



