.text
.align 5
.global func2 

func2:
.loop2:
    vmla.f32 q0, q0, q0
    vmla.f32 q1, q1, q1
    vmla.f32 q2, q2, q2
    vmla.f32 q3, q3, q3

    vmla.f32 q4, q4, q4
    vmla.f32 q5, q5, q5
    vmla.f32 q6, q6, q6
    vmla.f32 q7, q7, q7

    vmla.f32 q8, q8, q8
    vmla.f32 q9, q9, q9

    subs r0, r0, #1
    bne .loop2
