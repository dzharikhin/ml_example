use std::collections::HashMap;
#[derive(Debug)]
struct Object {
    f1: i8,
    f2: i8,
    f3: i8
}
#[derive(Debug)]
struct TrainObject {
    o: Object,
    class: i8
}

trait Node {
    fn get_class(&self, obj: &Object) -> i8;
}

struct CalcNode<'r> {
    feature_name: String,
    providers: &'r HashMap<String, Box<Fn(&Object) -> i8>>,
    route_map: HashMap<i8, Box<Node + 'r>>
}

struct LeafNode {
    class: i8
}

impl<'a> Node for CalcNode<'a> {
    fn get_class(&self, obj: &Object) -> i8 {
        let target_feature_value = self.providers.get(&self.feature_name).unwrap()(&obj);
        println!("CalcNode: feature={}, value={}", self.feature_name, target_feature_value);
        return self.route_map.get(&target_feature_value).unwrap().get_class(obj);
    }
}

impl Node for LeafNode {
    fn get_class(&self, _: &Object) -> i8 {
        println!("LeafNode: class={}", self.class);
        return self.class;
    }
}


fn main() {
    let mut providers:HashMap<String, Box<Fn(&Object) -> i8>> = HashMap::new();
    providers.insert("f1".to_string(), Box::new(|x: &Object| return x.f1));
    providers.insert("f2".to_string(), Box::new(|x: &Object| return x.f2));
    providers.insert("f3".to_string(), Box::new(|x: &Object| return x.f3));

    let data = vec![
        TrainObject {o: Object { f1: 1, f2: 3, f3: 3}, class: 2},
        TrainObject {o: Object { f1: 2, f2: 3, f3: 2}, class: 1},
        TrainObject {o: Object { f1: 2, f2: 2, f3: 1}, class: 1},
        TrainObject {o: Object { f1: 1, f2: 2, f3: 1}, class: 2},
        TrainObject {o: Object { f1: 2, f2: 1, f3: 3}, class: 2},
        TrainObject {o: Object { f1: 1, f2: 1, f3: 2}, class: 2},
        TrainObject {o: Object { f1: 2, f2: 3, f3: 1}, class: 1},
        TrainObject {o: Object { f1: 1, f2: 2, f3: 2}, class: 2}
    ];
    let root = build_tree(data.iter().collect(), &providers);
    let test_object = Object { f1: 2, f2: 1, f3: 3 };
    println!("============================Learning complete, starting prediction");
    root.get_class(&test_object);
}

fn build_tree<'a>(data: Vec<&TrainObject>, providers: &'a HashMap<String, Box<Fn(&Object) -> i8>>) -> Box<Node + 'a> {
    let data_size = data.len();
    println!("Data size: {}", &data_size);
    let mut classes: Vec<i8> = data.iter().map(|obj| obj.class).collect();
    classes.sort();
    classes.dedup();
    println!("Distinct classes: {:?}", &classes);
    if classes.len() == 1 {
        println!("Only one class={} presented - returning LeafNode", classes[0]);
        return Box::new(LeafNode {class: classes[0]});
    }
    let main_gini_index = classes.iter()
        .map(|&cls| data.iter().filter(|obj| obj.class == cls).count() as f64 / data_size as f64)
        .fold(1.0, |main_gini, next_class| main_gini * next_class);
    println!("Main Gini: {}", main_gini_index);

    let feature_ginis: HashMap<_, _> = providers.clone().iter()
        .map(|(feature_key, feature_provider)| {
            println!("\nCalculating Gini for feature: {}", feature_key);
            (feature_key, calculate_gini_for_feature(feature_provider, &data, &classes))
        })
        .collect();

    for (feature_key, feature_gini) in feature_ginis.iter() {
        println!("Feature {} gini index={}", feature_key, feature_gini);
    }
    match feature_ginis.iter().fold(("".to_string(), 0.0), |(max_key, max_gini_gain), (next_feature_key, &next_feature_gini)| {
        let next_gini_gain = main_gini_index - next_feature_gini;
        if next_gini_gain > max_gini_gain {
            (next_feature_key.to_string(), next_gini_gain)
        } else {
            (max_key, max_gini_gain)
        }
    }) {
        (max_feature_key, _) => {
            println!("Best feature based on Gini is {}", &max_feature_key);
            let provider = providers.get(&max_feature_key).unwrap();
            let mut distinct_data: Vec<_> = data.clone().iter().map(|&obj| &obj.o).map(|o| provider(o)).collect();
            distinct_data.sort();
            distinct_data.dedup();
            let route_map: HashMap<_, _> = distinct_data.into_iter()
                .map(|feature_value| {
                    println!("Building route: feature={}, value={}\n", &max_feature_key, &feature_value);
                    let child_node = build_tree(data.clone().into_iter().filter(|obj| provider(&obj.o) == feature_value).collect(), providers);
                    return (feature_value, child_node);
                })
                .collect();
            println!("Returning CalcNode for feature={}", &max_feature_key);
            return Box::new(CalcNode {
                providers: &providers,
                feature_name: max_feature_key,
                route_map: route_map
            });
        }
    }
}

fn calculate_gini_for_feature(provider: &Box<Fn(&Object) -> i8>, data: &Vec<&TrainObject>, classes: &Vec<i8>) -> f64 {
    let mut distinct_target_values: Vec<_> = data.iter().map(|&obj| &obj.o).map(|o| provider(o)).collect();
    distinct_target_values.sort();
    distinct_target_values.dedup();
    println!("Feature distinct values: {:?}", &distinct_target_values);
    return distinct_target_values.into_iter()
        .map(|target_feature_value| -> f64 {
            let data_with_target_feature: Vec<_> = data.iter().filter(|obj| provider(&obj.o) == target_feature_value).collect();
            let data_with_target_feature_size = data_with_target_feature.len();
            println!("Target value: {}. data size with this value: {}, all data size: {}", target_feature_value, data_with_target_feature_size, data.len());
            return classes.iter()
                .map(|cls| {
                    let data_of_class_size = data_with_target_feature.iter().filter(|obj| &obj.class == cls).count();
                    println!("Filtered data of class {} size: {}", cls, data_of_class_size);
                    data_of_class_size as f64 / data_with_target_feature_size as f64
                })
                .fold(data_with_target_feature_size as f64 / data.len() as f64, |mult, i| mult * i);
        })
        .fold(0.0, |sum, next| {
            sum + next
        });

}
