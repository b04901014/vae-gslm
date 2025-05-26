The dataset should be structured similarly with this format seperated by the deliminator `|`
For instance:
```
wavfile_name_1|text_transcript_1|phoneme_tokenizations_1
wavfile_name_2|text_transcript_2|phoneme_tokenizations_2
...
``` 

If there is no transcript provided, we will consider it an unconditional dataset.
For instance:
```
wavfile_name_1
wavfile_name_2
...
```

For flexibility, do not include the directory of the wavfile inside the metadata, the directory will be passed as a seperate argument.
