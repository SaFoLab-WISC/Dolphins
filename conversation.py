"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict
import base64
from io import BytesIO
from PIL import Image

class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    CHATGLM = auto()
    CHATML = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The system prompt
    system: str
    # The image token
    image_mark: str
    # Two roles
    roles: List[str]
    # The answer token. Marking the answer starts here
    answer_mark: str
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_message(self) -> str:
        message = self.messages
        ret = ""
        for i, (role, message) in enumerate(self.messages):
            if type(message) is tuple:
                message = message[0]
            if message is not None:
                ret += message
        return ret

    def  get_interleaved_prompt(self) -> str:
        """Get the prompt for generation."""
        image_marks = [self.image_mark, ""]
        answer_marks = ["", self.answer_mark]
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret +=  role + ": " + image_marks[i % 2] + answer_marks[i % 2] + message + self.sep
                else:
                    ret +=  role + ":" + image_marks[i % 2] + answer_marks[i % 2]
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            # ret = self.system + seps[0]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += message + seps[i % 2]
                else:
                    ret += role + ": " + image_marks[i % 2] + answer_marks[i % 2]
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")



    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        image_marks = [self.image_mark, ""]
        answer_marks = ["", self.answer_mark]
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret += image_marks[i%2] + role + ": " + answer_marks[i%2] + message + self.sep
                else:
                    ret += image_marks[i%2] + role + ":" + answer_marks[i%2]
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            #ret = self.system + seps[0]
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    ret +=  image_marks[i%2] + role + ": " + answer_marks[i%2] + message + seps[i % 2]
                else:
                    ret +=  image_marks[i%2] + role + ":" + answer_marks[i%2]
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_images(self):
        images = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                msg = list(msg)
                msg, image_list = msg[0], msg[1:]
                for image in image_list:
                    if image is not None:
                        if isinstance(image, Image.Image):
                            max_len, min_len = 1280, 400
                            H, W = image.size
                            aspect_ratio = float(W) / float(H)

                            if W > max_len:
                                new_W = max_len
                                new_H = int(new_W / aspect_ratio)
                                image = image.resize((new_W, new_H))

                            buffered = BytesIO()
                            image.save(buffered, format="PNG")
                            img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                            images.append(img_b64_str)

                        elif isinstance(image, list):
                            frames = []
                            for frame in image:
                                max_len, min_len = 1280, 400
                                H, W = frame.size
                                aspect_ratio = float(W) / float(H)

                                if W > max_len:
                                    new_W = max_len
                                    new_H = int(new_W / aspect_ratio)
                                    frame = frame.resize((new_W, new_H))

                                buffered = BytesIO()
                                frame.save(buffered, format="PNG")
                                img_b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                                frames.append(img_b64_str)

                            images.append(frames)

        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                msg = list(msg)
                msg, images = msg[0], msg[1:]

                for image in images:
                    if image is not None:
                        if isinstance(image, list):
                            image = image[0]
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        H, W = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((H, W))
                        # image = image.resize((224, 224))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="user upload image" />'
                        msg = msg.replace("USER: <image> is a driving video.", img_str + "USER: ", 1).replace("<|endofchunk|>", "\n")
                # hard-coded post processing for Otter
                msg = msg.replace("USER: <image> is a driving video.", "USER: ", 1).replace("<|endofchunk|>", "\n")
                msg = msg.replace("GPT:", "\nDriveGPT:")
                ret.append([msg, None])
            else:
                if msg is not None:
                    if msg.rfind("GPT:") != -1:
                        msg = msg[msg.rfind("GPT:"):]  # remove the last GPT label
                    msg = msg.replace("GPT:", "\nDriveGPT: ")
                    msg = msg.replace("<|endofchunk|>", "")
                    msg = "DriveGPT: " + msg
                ret[-1][-1] = msg
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message


    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            image_mark=self.image_mark,
            roles=self.roles,
            answer_mark=self.answer_mark,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        image_mark="",
        roles=("Human", "Assistant"),
        answer_mark="",
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        image_mark="",
        roles=("Human", "Assistant"),
        answer_mark="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna_v1.1",
        system="A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions.",
        image_mark="",
        roles=("USER", "ASSISTANT"),
        answer_mark="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Open_Flamingo template
register_conv_template(
    Conversation(
        name="open_flamingo_v1.1",
        system="You are InstructFlamingo, a large language and vision assistant trained by ASU lab. "
        "You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
        "Follow the instructions carefully and explain your answers in detail.",
        image_mark="<image>",
        roles=("USER", "ASSISTANT"),
        answer_mark="<answer>",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="<|endofchunk|>",
    )
)



# Open_Flamingo template
register_conv_template(
    Conversation(
        name="octopus",
        system="",
        image_mark="<image>",
        roles=("USER", "GPT"),
        answer_mark="<answer>",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="<|endofchunk|>",
    )
)


# Open_Flamingo template
register_conv_template(
    Conversation(
        name="openflamingo",
        system="",
        image_mark="<image>",
        roles=("Question", "Answer"),
        answer_mark="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="<|endofchunk|>",
    )
)



# otter template
register_conv_template(
    Conversation(
        name="otter",
        system="",
        image_mark="<image>",
        roles=("User", "GPT"),
        answer_mark="<answer>",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="<|endofchunk|>",
    )
)

# drivegpt template
register_conv_template(
    Conversation(
        name="drivegpt",
        system="",
        image_mark="<image>",
        roles=("USER", "GPT"),
        answer_mark="<answer>",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="<|endofchunk|>",
    )
)


# clever_flamingo template
register_conv_template(
    Conversation(
        name="cleverflamingo",
        system="",
        image_mark="",
        roles=("### Human", "<image>\n### Assistant"),
        answer_mark="",
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="",
        sep2="<|endofchunk|>",
    )
)



if __name__ == "__main__":
    conv = get_conv_template("octopus")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)
    print(conv.get_prompt())