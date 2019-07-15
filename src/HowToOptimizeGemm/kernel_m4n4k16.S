.text
.align 5
.global kernel_sub_m4n4k16

kernel_sub_m4n4k16:
// x0: sa
// x1: sb
// x2: sc
// x3: ldc

// INIT RESULT
dup v16.4s, wzr
dup v17.4s, wzr
dup v18.4s, wzr
dup v19.4s, wzr

dup v20.4s, wzr
dup v21.4s, wzr
dup v22.4s, wzr
dup v23.4s, wzr

dup v24.4s, wzr
dup v25.4s, wzr
dup v26.4s, wzr
dup v27.4s, wzr

dup v28.4s, wzr
dup v29.4s, wzr
dup v30.4s, wzr
dup v31.4s, wzr

// preload sa and sb
prfm pldl1keep, [x0, #64]
prfm pldl1keep, [x1, #64]

ld1 {v0.16b, v1.16b, v2.16b, v3.16b}, [x0]
ld1 {v4.16b, v5.16b, v6.16b, v7.16b}, [x1]

// design:
//                  sb  | v4.16b      v5.16b      v6.16b      v7.16b
//
//    |  v0.16b          v8           v9          v10           v11
//  sa|  v1.16b          v12          v13         v14           v15
//    |  v2.16b
//    |  v3.16b
//
//                      ||
//   accumulate         ||
//                      ||
//            
//                  sb  | v4.16b      v5.16b      v6.16b      v7.16b
//
//    |  v0.16b          v16           v17          v18          v19
//  sa|  v1.16b          v20           v21          v22          v23
//    |  v2.16b          ...
//    |  v3.16b          ..                                      v31

smull   v8.8h, v0.8b, v4.8b
smull   v9.8h, v0.8b, v5.8b
smull   v10.8h, v0.8b, v6.8b
smull   v11.8h, v0.8b, v7.8b 

smull   v12.8h, v1.8b, v4.8b
smull   v13.8h, v1.8b, v5.8b
smull   v14.8h, v1.8b, v6.8b
smull   v15.8h, v1.8b, v7.8b

smlal2  v8.8h, v0.16b, v4.16b
smlal2  v9.8h, v0.16b, v5.16b
smlal2  v10.8h, v0.16b, v6.16b
smlal2  v11.8h, v0.16b, v7.16b

smlal2  v12.8h, v1.16b, v4.16b
smlal2  v13.8h, v1.16b, v5.16b
smlal2  v14.8h, v1.16b, v6.16b
smlal2  v15.8h, v1.16b, v7.16b

// proc first two lines
sadalp  v16.4s, v8.8h
sadalp  v17.4s, v9.8h
sadalp  v18.4s, v10.8h
sadalp  v19.4s, v11.8h
sadalp  v20.4s, v12.8h
sadalp  v21.4s, v13.8h
sadalp  v22.4s, v14.8h
sadalp  v23.4s, v15.8h

smull   v8.8h, v2.8b, v4.8b
smull   v9.8h, v2.8b, v5.8b
smull   v10.8h, v2.8b, v6.8b
smull   v11.8h, v2.8b, v7.8b

smull   v12.8h, v3.8b, v4.8b
smull   v13.8h, v3.8b, v5.8b
smull   v14.8h, v3.8b, v6.8b
smull   v15.8h, v3.8b, v7.8b

smlal2  v8.8h, v2.16b, v4.16b
smlal2  v9.8h, v2.16b, v5.16b
smlal2  v10.8h, v2.16b, v6.16b
smlal2  v11.8h, v2.16b, v7.16b

smlal2  v12.8h, v3.16b, v4.16b
smlal2  v13.8h, v3.16b, v5.16b
smlal2  v14.8h, v3.16b, v6.16b
smlal2  v15.8h, v3.16b, v7.16b

// proc last two lines
sadalp  v24.4s, v8.8h
sadalp  v25.4s, v9.8h
sadalp  v26.4s, v10.8h
sadalp  v27.4s, v11.8h
sadalp  v28.4s, v12.8h
sadalp  v29.4s, v13.8h
sadalp  v30.4s, v14.8h
sadalp  v31.4s, v15.8h

// add pairwise
addp    v0.4s, v16.4s, v17.4s
addp    v1.4s, v18.4s, v19.4s
addp    v2.4s, v20.4s, v21.4s
addp    v3.4s, v22.4s, v23.4s
addp    v4.4s, v24.4s, v25.4s
addp    v5.4s, v26.4s, v27.4s
addp    v6.4s, v28.4s, v29.4s
addp    v7.4s, v30.4s, v31.4s

addp    v8.4s, v0.4s, v1.4s
addp    v9.4s, v2.4s, v3.4s
addp    v10.4s, v4.4s, v5.4s
addp    v11.4s, v6.4s, v7.4s

mov x0, x2
ld1 {v12.16b}, [x2], x3
add v12.4s, v12.4s, v8.4s

ld1 {v13.16b}, [x2], x3
add v13.4s, v13.4s, v9.4s

ld1 {v14.16b}, [x2], x3
add v14.4s, v14.4s, v10.4s

ld1 {v15.16b}, [x2], x3
add v15.4s, v15.4s, v11.4s

// write C
st1 {v12.16b}, [x0], x3
st1 {v13.16b}, [x0], x3
st1 {v14.16b}, [x0], x3
st1 {v15.16b}, [x0], x3

ret
