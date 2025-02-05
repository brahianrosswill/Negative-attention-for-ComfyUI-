# Negative-attention-for-ComfyUI-
Takes the difference in between the positive and negative conditioning at the cross attention.

A proof of concept which demonstrates that a negative influence can be obtained by taking the difference at the output of the cross attention.

In order to do this for now the negative conditioning is sneaked to the attention by being concatenated to the positive by using a special node.

![image](https://github.com/user-attachments/assets/c43caf96-8f43-4c1c-8813-9a70a646f3cd)

They are then split at the half before the cross attention.

An example workflow is provided.

## Example:

No modification:

![01310UI_00001_](https://github.com/user-attachments/assets/3927dd41-6c05-4f4f-92cb-50511755f6f0)

Strength at 2, rescale after ON:

![01309UI_00001_](https://github.com/user-attachments/assets/55badabe-b9e5-4cb5-a8df-93f0f320d6bb)

Strength at 2, rescale after OFF:

![example workflow](https://github.com/user-attachments/assets/e06fea9a-0d89-429a-8292-c432ae5efa05)


## Note

I haven't managed to make this work with anything but SDXL / SD1.5

I did spend two hours looking for how to patch the equivalent of the cross attention for Flux but did not find how (like the keywords for the patch or something).

Any help appreciated!
