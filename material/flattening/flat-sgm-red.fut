
def sgmscan 't [n] (op: t->t->t) (ne: t) 
             (flg : [n]bool) (arr : [n]t) : [n]t =
  let flgs_vals = 
    scan ( \ (f1, x1) (f2,x2) -> 
            let f = f1 || f2
            let v = if f2 then x2
                    else op x1 x2
            in  (f, v)
         )
         (false,ne) (zip flg arr)
  let (_, vals) = unzip flgs_vals
  in vals

def mkFlagArray 't [m]
            (aoa_shp: [m]i64)(zero: t)    --aoa_shp=[0,3,1,0,4,2,0]
            (aoa_val: [m]t)   : []t =     --aoa_val=[1,1,1,1,1,1,1]
  let shp_rot = map (\i->if i==0 then 0   --shp_rot=[0,0,3,1,0,4,2]
                         else aoa_shp[i-1]
                    ) (iota m)
  let shp_scn = scan (+) 0 shp_rot       --shp_scn=[0,0,3,4,4,8,10]
  let aoa_len = shp_scn[m-1]+aoa_shp[m-1]--aoa_len= 10
  let shp_ind = map2 (\shp ind ->        --shp_ind= 
                       if shp==0 then -1 --  [-1,0,3,-1,4,8,-1]
                       else ind          --scatter
                     ) aoa_shp shp_scn   --   [0,0,0,0,0,0,0,0,0,0]
  in scatter (replicate aoa_len zero)    --   [-1,0,3,-1,4,8,-1]
             shp_ind aoa_val             --   [1,1,1,1,1,1,1]
                                       -- F = [1,0,0,1,1,0,0,0,1,0]


def sgmScan [n] 't (op: t -> t -> t) (ne: t) (flags: [n]bool) (ass: [n]t) =
  let (_, vs) = 
    scan (\ (f1,v1) (f2,v2) ->
            (f1 || f2, if f2 then v2 else op v1 v2)      
         ) (false, ne) (zip flags ass) |> unzip
  in  vs

def sgmRed [m][n] 't (op: t -> t -> t) (ne: t) (shp: [m]i64) (flags: [n]bool) (ass: [n]t) =
    let offs  = scan (+) 0 shp
    let incsc = sgmScan op ne flags ass
    in  map2 (\i s -> if s != 0 then incsc[ offs[i]-1 ] else ne) (iota m) shp

entry test1 [m][n] (shp: [m]i64) (flags: [n]bool) (ass: [n]f32) =
  sgmRed (+) 0 shp flags ass 


entry test2 [m][n] (_shp: [m]i64) (sgmind: [n]i32) (ass: [n]f32) =
  hist (+) 0f32 m (map i64.i32 sgmind) ass 
  

entry mkdataset1 (m: i64) (n: i64) =
  let q = n / m
  let n'= m * q
  let shape = replicate m q
  let flags = mkFlagArray shape false (replicate m true)
  let fdata = map (\i -> if i % 2 == 0 then 1f32 else 3f32) (iota n')
  in  (shape, flags, fdata)

entry mkdataset2 (m: i64) (n: i64) : ([m]i64, []i32, []f32) =
  let q = n / m
  let n'= m * q
  let shape = replicate m q
  let flags = (mkFlagArray shape false (replicate m true)) :> [n']bool
  let sgmind= sgmScan (+) 0i32 flags (replicate n' 1i32)
  let fdata = iota n' 
           |> map (\i -> if i % 2 == 0 then 1 else 3)
           |> map (\x -> x-1)
  in  (shape, sgmind, fdata)

