#ifndef KERNEL_NAME
#define KERNEL_NAME func1 
#endif

.text
.align 5
.global KERNEL_NAME
.type KERNEL_NAME, %function

KERNEL_NAME:
.loop1:
    fmla v0.4s, v0.4s, v16.s[0]
    fmla v1.4s, v1.4s, v16.s[1]
    fmla v2.4s, v2.4s, v16.s[2]
    fmla v3.4s, v3.4s, v16.s[3]

    subs x0, x0, #1

    fmla v4.4s, v4.4s, v16.s[0]
    fmla v5.4s, v5.4s, v16.s[1]
    fmla v6.4s, v6.4s, v16.s[1]
    fmla v7.4s, v7.4s, v16.s[1]
    bne .loop1
    ret
