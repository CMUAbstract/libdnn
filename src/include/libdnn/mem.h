#ifndef MEM_H
#define MEM_H

#define __fram __attribute__((section(".persistent")))
#define __ro_fram __attribute__((section(".rodata")))
#define __hifram __attribute__((section(".upper.persistent")))
#define __ro_hifram __attribute__((section(".upper.rodata")))
#define __known __attribute__((section(".known")))

#endif