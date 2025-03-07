import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()
    MPT = auto()
    PLAIN = auto()
    LLAMA_2 = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages the conversation history and related settings."""

    system: str  # System message that starts the conversation, typically a prompt or instruction.
    roles: List[str]  # Roles involved in the conversation, e.g., 'user' and 'assistant'.
    messages: List[List[str]]  # A list of messages in the conversation. Each message is stored as a list where the first element is the role and the second element is the message itself.
    offset: int  # Offset used for determining the starting point in the message list when processing messages.
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE  # Separator style used to format the conversation. It defines how the messages are concatenated.
    sep: str = "###"  # Primary separator used between messages in the conversation.
    sep2: str = None  # Secondary separator, used in specific separator styles.
    version: str = "Unknown"  # Version of the conversation schema or format.
    skip_next: bool = False  # Flag to indicate whether the next message should be skipped in some processing.

    def get_prompt(self):
        """
        Generates the complete conversation prompt based on the current messages and separator style.
        This is used to build the context that the assistant uses to generate responses.
        """
        # Start with the existing messages.
        messages = self.messages

        # If the first message contains a tuple (indicating it includes a protein sequence), process it accordingly.
        if len(messages) > 0 and type(messages[0][1]) is tuple:
            messages = self.messages.copy()
            init_role, init_msg = messages[0].copy()
            init_msg = init_msg[0].strip()
            if 'mmtag' in self.version:
                # For versions with 'mmtag', adjust the first few messages.
                messages[0] = (init_role, init_msg)
                messages.insert(0, (self.roles[0], "<ProteinSequence></ProteinSequence>"))
                messages.insert(1, (self.roles[1], "Received."))
            else:
                # For other versions, prepend the protein sequence tag to the initial message.
                messages[0] = (init_role, "<protein_sequence>\n" + init_msg)

        # Handle different separator styles to concatenate the messages.
        if self.sep_style == SeparatorStyle.SINGLE:
            # Use a single separator for all messages.
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.TWO:
            # Use two separators, alternating between messages.
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
        elif self.sep_style == SeparatorStyle.MPT:
            # Use a different separator style (MPT) where role and message are concatenated differently.
            ret = self.system + self.sep
            for role, message in messages:
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += role + message + self.sep
                else:
                    ret += role
        elif self.sep_style == SeparatorStyle.LLAMA_2:
            # Special format used for LLaMA 2 model where system and instruction are wrapped.
            wrap_sys = lambda msg: f"<<SYS>>\n{msg}\n<</SYS>>\n\n" if len(msg) > 0 else msg
            wrap_inst = lambda msg: f"[INST] {msg} [/INST]"
            ret = ""

            for i, (role, message) in enumerate(messages):
                if i == 0:
                    assert message, "first message should not be none"
                    assert role == self.roles[0], "first message should come from user"
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i == 0: 
                        message = wrap_sys(self.system) + message
                    if i % 2 == 0:
                        message = wrap_inst(message)
                        ret += self.sep + message
                    else:
                        ret += " " + message + " " + self.sep2
                else:
                    ret += ""
            ret = ret.lstrip(self.sep)
        elif self.sep_style == SeparatorStyle.PLAIN:
            # Plain separator style where messages are concatenated with alternating separators.
            seps = [self.sep, self.sep2]
            ret = self.system
            for i, (role, message) in enumerate(messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    ret += message + seps[i % 2]
                else:
                    ret += ""
        else:
            # Raise an error if an invalid separator style is provided.
            raise ValueError(f"Invalid style: {self.sep_style}")

        # Return the final prompt string.
        return ret

    def append_message(self, role, message):
        """Appends a new message to the conversation with the specified role."""
        self.messages.append([role, message])

    def process_protein_sequence(self, sequence, sequence_process_mode, max_len=1000):
        """
        Processes a protein sequence according to the specified processing mode.

        :param sequence: The protein sequence to process.
        :param sequence_process_mode: The mode to process the sequence ('Truncate', 'Pad', 'Default').
        :param max_len: The maximum allowed length of the sequence. Default is 1000.
        :return: The processed protein sequence as a string.
        """
        # if sequence_process_mode == "Truncate":
        #     # Truncate the sequence to the specified maximum length.
        #     sequence = sequence[:max_len]
        # elif sequence_process_mode == "Pad":
        #     # Pad the sequence with a predefined character (e.g., 'X') to the specified length.
        #     sequence = sequence.ljust(max_len, 'X')
        # elif sequence_process_mode == "Default":
        #     # No processing, return the sequence as is.
        #     pass
        # else:
        #     # Raise an error if an invalid sequence processing mode is provided.
        #     raise ValueError(f"Invalid sequence_process_mode: {sequence_process_mode}")

        return sequence

    def get_sequences(self):
        """
        Retrieves and processes all protein sequences from the conversation based on the stored messages.

        :return: A list of processed protein sequences.
        """
        sequences = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:  # Process sequences only from specific messages, typically those sent by the user.
                if type(msg) is tuple:
                    msg, sequence, sequence_process_mode = msg
                    sequence = self.process_protein_sequence(sequence, sequence_process_mode)
                    sequences.append(sequence)
        return sequences

    def to_gradio_chatbot(self):
        """
        Converts the conversation to a format suitable for display in a Gradio chatbot interface.
        
        :return: A list of messages formatted for the Gradio chatbot.
        """
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i == 0:  # User messages
                if type(msg) is tuple:
                    msg, sequence, sequence_process_mode = msg
                    processed_sequence = self.process_protein_sequence(sequence, "Default")
                    msg = f'<div>{processed_sequence}</div>' + msg.strip()
                    ret.append([msg, None])
            elif i % 2 == 0:  # User messages
                if type(msg) is tuple:
                    msg, sequence, sequence_process_mode = msg
                    # processed_sequence = self.process_protein_sequence(sequence, "Default")
                    # msg = f'<div>{processed_sequence}</div>' + msg.strip()
                    msg = msg.strip()
                    ret.append([msg, None])
                else:
                    ret.append([msg, None])
            else:  # Assistant messages
                ret[-1][-1] = msg
        return ret

    def copy(self):
        """
        Creates a deep copy of the Conversation object, including all its messages and settings.
        
        :return: A new Conversation object with the same data.
        """
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version)

    def dict(self):
        """
        Converts the Conversation object to a dictionary format, which can be easily serialized.
        
        :return: A dictionary representation of the Conversation.
        """
        if len(self.get_sequences()) > 0:
            return {
                "system": self.system,
                "roles": self.roles,
                "messages": [[x, y[0] if type(y) is tuple else y] for x, y in self.messages],
                "offset": self.offset,
                "sep": self.sep,
                "sep2": self.sep2,
            }
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


# Conversation templates adjusted for protein sequences
conv_vicuna_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
        ("Human", "What are the key differences between renewable and non-renewable energy sources?"),
        ("Assistant",
            "Renewable energy sources are those that can be replenished naturally in a relatively "
            "short amount of time, such as solar, wind, hydro, geothermal, and biomass. "
            "Non-renewable energy sources, on the other hand, are finite and will eventually be "
            "depleted, such as coal, oil, and natural gas. Here are some key differences between "
            "renewable and non-renewable energy sources:\n"
            "1. Availability: Renewable energy sources are virtually inexhaustible, while non-renewable "
            "energy sources are finite and will eventually run out.\n"
            "2. Environmental impact: Renewable energy sources have a much lower environmental impact "
            "than non-renewable sources, which can lead to air and water pollution, greenhouse gas emissions, "
            "and other negative effects.\n"
            "3. Cost: Renewable energy sources can be more expensive to initially set up, but they typically "
            "have lower operational costs than non-renewable sources.\n"
            "4. Reliability: Renewable energy sources are often more reliable and can be used in more remote "
            "locations than non-renewable sources.\n"
            "5. Flexibility: Renewable energy sources are often more flexible and can be adapted to different "
            "situations and needs, while non-renewable sources are more rigid and inflexible.\n"
            "6. Sustainability: Renewable energy sources are more sustainable over the long term, while "
            "non-renewable sources are not, and their depletion can lead to economic and social instability.\n")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_vicuna_v1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llama_2 = Conversation(
    system="""You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_llava_llama_2 = Conversation(
    system="You are a helpful language and vision assistant. "
           "You are able to understand the visual content that the user provides, "
           "and assist the user with a variety of tasks using natural language.",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="<s>",
    sep2="</s>",
)

conv_mpt = Conversation(
    system="""system
A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.""",
    roles=("user\n", "assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="",
)

conv_llava_plain = Conversation(
    system="",
    roles=("", ""),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.PLAIN,
    sep="\n",
)

conv_llava_v0 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_llava_v0_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the protein sequence content that the user provides, and assist the user with a variety of tasks using natural language."
           "The protein sequence content will be provided with the following format: <ProteinSequence>sequence content</ProteinSequence>.",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
    version="v0_mmtag",
)

conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_llava_v1_mmtag = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
           "The assistant is able to understand the protein sequence content that the user provides, and assist the user with a variety of tasks using natural language."
           "The protein sequence content will be provided with the following format: <ProteinSequence>sequence content</ProteinSequence>.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
    version="v1_mmtag",
)

conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""system
Answer the questions.""",
    roles=("user\n", "assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="",
)

default_conversation = conv_vicuna_v1

conv_templates = {
    "default": conv_vicuna_v0,
    "v0": conv_vicuna_v0,
    "v1": conv_vicuna_v1,
    "vicuna_v1": conv_vicuna_v1,
    "llama_2": conv_llama_2,
    "mistral_instruct": conv_mistral_instruct,
    "chatml_direct": conv_chatml_direct,
    "mistral_direct": conv_chatml_direct,

    "plain": conv_llava_plain,
    "v0_plain": conv_llava_plain,
    "llava_v0": conv_llava_v0,
    "v0_mmtag": conv_llava_v0_mmtag,
    "llava_v1": conv_llava_v1,
    "v1_mmtag": conv_llava_v1_mmtag,
    "llava_llama_2": conv_llava_llama_2,

    "mpt": conv_mpt,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
