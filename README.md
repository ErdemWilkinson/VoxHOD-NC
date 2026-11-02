# VoxHOD-NC: drone person detector with extra Turkish footage (non-commercial)

**VoxHOD-NC** is the **non-commercial** branch of
[VoxHOD](https://github.com/ErdemWilkinson/VoxHOD), a tiny int8 person
detector for drone search and rescue on ESP32-S3-class microcontrollers.

> **Türkçe:** VoxHOD-NC, VoxHOD'un ticari olmayan koludur. Ferhat Tatık'ın
> izniyle onun drone görüntüleri eklenerek eğitilir; bu yüzden yalnızca
> ticari olmayan kullanım içindir (CC BY-NC 4.0). Ticari kullanım
> gerekiyorsa, bu görüntüleri hiç görmemiş [VoxHOD](https://github.com/ErdemWilkinson/VoxHOD)
> modelini kullanın (CC BY 4.0).

## Which one do I need?

| | [VoxHOD](https://github.com/ErdemWilkinson/VoxHOD) | **VoxHOD-NC** (this repo) |
|---|---|---|
| Training data | HERIDAL only | HERIDAL **plus** drone footage by Ferhat Tatık |
| Model licence | CC BY 4.0, commercial use allowed | **CC BY-NC 4.0, non-commercial only** |
| Status | v3 (197 KB) and nano (75 KB) published | **In preparation, no weights yet** |

The footage is used with the photographer's written permission for
non-commercial use. **The footage itself is not published here or anywhere
else**, and its rights stay with Ferhat Tatık.

## Status

No model has been trained yet. Results will be added here only after they are
measured, on the same HERIDAL test set as VoxHOD and on a held-out part of the
new footage that is never trained on.

## Code

The training code is shared with VoxHOD and lives in
[ErdemWilkinson/VoxHOD](https://github.com/ErdemWilkinson/VoxHOD) (MIT).

## Thanks

Thanks to **Ferhat Tatık** ([@ferhattatik_dronoloji](https://www.instagram.com/ferhattatik_dronoloji/),
[YouTube @dronoloji](https://www.youtube.com/@dronoloji)) for the drone footage.

## Licence

Model weights: **[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)**.
See [MODEL_LICENSE.md](MODEL_LICENSE.md).

VoxHOD-NC is built for search and rescue, to help find people who need help.
It is not intended for targeting or harming people.
