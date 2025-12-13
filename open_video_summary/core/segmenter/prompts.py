from dataclasses import dataclass, field


@dataclass
class VideoSegmenterPrompts:
    generate_subtopics: str = field(
        default="""
            @DOCUMENT
            '''{full_video_transcript}'''

            #####

            @OUTPUT

            The output should be a JSON object with the following example format
            {{
                "0": "Example topic",
                "1": "Another topic example",
                "2": ...
            }}

            #####

            Generate up to {max_subtopics} subtopics from @DOCUMENT. The response should contain only the subtopics using the @OUTPUT format specified.
        """
    )

    classify_subtopic: str = field(
        default="""
            @DOCUMENT
            '''
            {content}
            '''

            #####

            @SUBTOPICS

            {topics}

            #####

            Select one @SUBTOPICS item that best classifies the @DOCUMENT according to its textual content.
            The output should be the JSON object only with the corresponding topic key and it's text description as value.
        """
    )

    fix_full_transcription: str = field(
        default="""
            @DOCUMENT
            '''
            {content}
            '''

            #####
            @EXAMPLE

            Input: "Vamos para asno tícias de hoje. Mais antes preciso contar para vocês do presidente que recebi. Alojado action figu des My Figurs me mandou esse bosto do soro do One Piece"
            Output: "Vamos para as notícias de hoje. Mas, antes, preciso contar para vocês do presente que recebi. A loja de action figures My Figures me mandou esse busto do Zoro do One Piece."

            #####

            The text in @DOCUMENT was automatically extracted from an audio file and might contain errors, such as typos or homophones words instead of the actual ones.
            Rewrite the @DOCUMENT string fixing those errors.
            The output should only contain the fixed text, as presented in the @EXAMPLE provided.
        """
    )

    fix_segment_transcription: str = field(
        default="""
            @FULL_TRANSCRIPTION
            '''
            {full_video_transcript}
            '''

            @SEGMENT_TRANSCRIPTION
            '''
            {content}
            '''

            #####
            @EXAMPLE

            Input:
            Full transcription: "Dito isso, vamos para as notícias de hoje. Mas, antes, preciso contar para vocês do presente que recebi. A loja de action figures My Figures me mandou esse busto do Zoro do One Piece."
            Segment transcription: "Alojado e action figures My Figures me mandou esse bosto do soro do One Piece"
            
            Output:
            "A loja de action figures My Figures me mandou esse busto do Zoro do One Piece."

            #####

            The text in @SEGMENT_TRANSCRIPTION is a part of @FULL_TRANSCRIPTION. It was automatically extracted from an audio file and might contain errors.
            Rewrite the @SEGMENT_TRANSCRIPTION string fixing those errors according to the correct form present in @FULL_TRANSCRIPTION.
            The output should only contain the fixed text, as presented in the @EXAMPLE provided.
        """
    )

    get_global_topics: str = field(
        default="""
            @TOPICS_COLLECTION
            '''
            {topics_collection}
            '''

            @OUTPUT

            The output should be a JSON list object with the following example format
            ["Example topic", "Another topic example", "More topics"]

            #####

            @EXAMPLE

            Input:
            ["Streaming market consolidation", "Media industry consolidation", "Content library extension", "Expansion of franchises", "Antitrust concerns", "Impact on competing platforms"]

            Output:
            ["Streaming and media industry consolidation", "Expansion of content libraries, IPs, and franchises", "Regulatory scrutiny", "Impact on other streaming platforms"]

            #####

            The @TOPICS_COLLECTION provides a list of topics discussed on many videos on the same subject.
            Create a new list of topics mantaining the content of the initial list and avoiding topic redundance.
            The response should contain only the subtopics using the @OUTPUT format specified.
        """
    )

    def __post_init__(self) -> None:
        if (
            "{full_video_transcript}" not in self.generate_subtopics
            or "{max_subtopics}" not in self.generate_subtopics
        ):
            raise ValueError(
                "The prompt template for `generate_subtopics` must contain both '{full_video_transcript}' and '{max_subtopics}' placeholders."
            )

        if (
            "{content}" not in self.classify_subtopic
            or "{topics}" not in self.classify_subtopic
        ):
            raise ValueError(
                "The prompt template for `classify_subtopic` must contain both '{content}' and '{topics}' placeholders."
            )
