fn sum(vals: &[i32]) -> i32 {
    let mut res = 0;
    for i in 0..vals.len() {
        res += vals[i]
    }
    res
}

fn main() {
    let arr = [10, 20, 30, 40];
    let res = sum(&arr);
    println!("sum: {}", res);
}
