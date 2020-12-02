.text
.align 5
.global func1 

func1:
.loop1:
    vmla.f32 q0, q0, d0[0]
    vmla.f32 q1, q1, d3[0]
    vmla.f32 q2, q2, d5[0]
    vmla.f32 q3, q3, d7[0]

    subs r0, r0, #1

    vmla.f32 q4, q4, d9[0]
    vmla.f32 q5, q5, d11[0]
    vmla.f32 q6, q6, d13[0]
    vmla.f32 q7, q7, d15[0]

    vmla.f32 q8, q8, d1[0]
    vmla.f32 q9, q9, d2[0]
    bne .loop1
