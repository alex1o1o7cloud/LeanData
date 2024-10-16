import Mathlib

namespace NUMINAMATH_CALUDE_ice_cream_sales_l3861_386172

theorem ice_cream_sales (tuesday_sales : ℕ) (wednesday_sales : ℕ) : 
  wednesday_sales = 2 * tuesday_sales →
  tuesday_sales + wednesday_sales = 36000 →
  tuesday_sales = 12000 :=
by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l3861_386172


namespace NUMINAMATH_CALUDE_tan_negative_4095_degrees_l3861_386142

theorem tan_negative_4095_degrees : Real.tan ((-4095 : ℝ) * Real.pi / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_4095_degrees_l3861_386142


namespace NUMINAMATH_CALUDE_intersection_A_B_l3861_386119

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def B : Set ℝ := {x : ℝ | x^2 + 2*x < 0}

theorem intersection_A_B : A ∩ B = Set.Ioo (-1) 0 := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3861_386119


namespace NUMINAMATH_CALUDE_inheritance_problem_l3861_386148

theorem inheritance_problem (x y z w : ℚ) : 
  (y = 0.75 * x) →
  (z = 0.5 * x) →
  (w = 0.25 * x) →
  (y = 45) →
  (z = 2 * w) →
  (x + y + z + w = 150) :=
by sorry

end NUMINAMATH_CALUDE_inheritance_problem_l3861_386148


namespace NUMINAMATH_CALUDE_team_selection_count_l3861_386104

/-- The number of ways to select a team of 5 people from a group of 16 people -/
def select_team (total : ℕ) (team_size : ℕ) : ℕ :=
  Nat.choose total team_size

/-- The total number of students in the math club -/
def total_students : ℕ := 16

/-- The number of boys in the math club -/
def num_boys : ℕ := 7

/-- The number of girls in the math club -/
def num_girls : ℕ := 9

/-- The size of the team to be selected -/
def team_size : ℕ := 5

theorem team_selection_count :
  select_team total_students team_size = 4368 ∧
  total_students = num_boys + num_girls :=
sorry

end NUMINAMATH_CALUDE_team_selection_count_l3861_386104


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3861_386132

theorem geometric_series_sum : 
  let a : ℝ := 1
  let r : ℝ := 1/7
  let S := ∑' n, a * r^n
  S = 7/6 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3861_386132


namespace NUMINAMATH_CALUDE_expression_value_l3861_386171

theorem expression_value : -25 + 5 * (4^2 / 2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3861_386171


namespace NUMINAMATH_CALUDE_angle_inequality_l3861_386114

open Real

theorem angle_inequality (θ : Real) (h1 : 3 * π / 4 < θ) (h2 : θ < π) :
  ∀ x : Real, 0 ≤ x ∧ x ≤ 1 →
    x^2 * sin θ - x * (1 - x) + (1 - x)^2 * cos θ + 2 * x * (1 - x) * sqrt (cos θ * sin θ) > 0 :=
by sorry

end NUMINAMATH_CALUDE_angle_inequality_l3861_386114


namespace NUMINAMATH_CALUDE_amusement_park_problem_l3861_386140

/-- The number of children in the group satisfies the given conditions -/
theorem amusement_park_problem (C : ℕ) : C = 5 ↔ 
  15 + 3 * C + 16 * C = 110 := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l3861_386140


namespace NUMINAMATH_CALUDE_henry_walking_distance_l3861_386163

/-- Given a constant walking rate and duration, calculate the distance walked. -/
def distance_walked (rate : ℝ) (time : ℝ) : ℝ :=
  rate * time

/-- Theorem: Henry walks 8 miles in 2 hours at a rate of 4 miles per hour. -/
theorem henry_walking_distance :
  let rate : ℝ := 4  -- miles per hour
  let time : ℝ := 2  -- hours
  distance_walked rate time = 8 := by
  sorry

end NUMINAMATH_CALUDE_henry_walking_distance_l3861_386163


namespace NUMINAMATH_CALUDE_min_movie_audience_l3861_386180

/-- Represents the number of people in the movie theater -/
structure MovieTheater where
  adults : ℕ
  children : ℕ

/-- Conditions for the movie theater audience -/
class MovieTheaterConditions (t : MovieTheater) where
  adult_men : t.adults * 4 = t.adults * 5
  male_children : t.children * 2 = t.adults * 2
  boy_children : t.children * 1 = t.children * 5

/-- The theorem stating the minimum number of people in the movie theater -/
theorem min_movie_audience (t : MovieTheater) [MovieTheaterConditions t] :
  t.adults + t.children ≥ 55 := by
  sorry

#check min_movie_audience

end NUMINAMATH_CALUDE_min_movie_audience_l3861_386180


namespace NUMINAMATH_CALUDE_algorithm_computes_gcd_l3861_386149

/-- The algorithm described in the problem -/
def algorithm (x y : ℕ) : ℕ :=
  let rec loop (m n : ℕ) : ℕ :=
    if m / n = m / n then n
    else loop n (m % n)
  loop (max x y) (min x y)

/-- Theorem stating that the algorithm computes the GCD -/
theorem algorithm_computes_gcd (x y : ℕ) :
  algorithm x y = Nat.gcd x y := by sorry

end NUMINAMATH_CALUDE_algorithm_computes_gcd_l3861_386149


namespace NUMINAMATH_CALUDE_fan_airflow_rate_l3861_386157

/-- Proves that the airflow rate of a fan is 10 liters per second, given the specified conditions. -/
theorem fan_airflow_rate : 
  ∀ (daily_operation_minutes : ℝ) (weekly_airflow_liters : ℝ),
    daily_operation_minutes = 10 →
    weekly_airflow_liters = 42000 →
    (weekly_airflow_liters / (daily_operation_minutes * 7 * 60)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_fan_airflow_rate_l3861_386157


namespace NUMINAMATH_CALUDE_bike_cost_calculation_l3861_386120

/-- The cost of Carrie's bike --/
def bike_cost (hourly_wage : ℕ) (weekly_hours : ℕ) (weeks_per_month : ℕ) (remaining_money : ℕ) : ℕ :=
  hourly_wage * weekly_hours * weeks_per_month - remaining_money

/-- Theorem stating the cost of the bike --/
theorem bike_cost_calculation :
  bike_cost 8 35 4 720 = 400 := by
  sorry

end NUMINAMATH_CALUDE_bike_cost_calculation_l3861_386120


namespace NUMINAMATH_CALUDE_solution_set_equality_l3861_386169

open Set

/-- The solution set of the inequality |x-5|+|x+3|≥10 -/
def SolutionSet : Set ℝ := {x : ℝ | |x - 5| + |x + 3| ≥ 10}

/-- The expected result set (-∞，-4]∪[6，+∞) -/
def ExpectedSet : Set ℝ := Iic (-4) ∪ Ici 6

theorem solution_set_equality : SolutionSet = ExpectedSet := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3861_386169


namespace NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l3861_386137

theorem min_value_of_sum_of_roots (x : ℝ) : 
  Real.sqrt (x^2 + 4*x + 20) + Real.sqrt (x^2 + 2*x + 10) ≥ 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_of_roots_l3861_386137


namespace NUMINAMATH_CALUDE_largest_integer_less_than_M_div_100_l3861_386183

def factorial (n : ℕ) : ℕ := Nat.factorial n

def M : ℚ :=
  (1 / (factorial 3 * factorial 19) +
   1 / (factorial 4 * factorial 18) +
   1 / (factorial 5 * factorial 17) +
   1 / (factorial 6 * factorial 16) +
   1 / (factorial 7 * factorial 15) +
   1 / (factorial 8 * factorial 14) +
   1 / (factorial 9 * factorial 13) +
   1 / (factorial 10 * factorial 12)) * (factorial 1 * factorial 21)

theorem largest_integer_less_than_M_div_100 :
  Int.floor (M / 100) = 952 := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_M_div_100_l3861_386183


namespace NUMINAMATH_CALUDE_smaller_acute_angle_measure_l3861_386166

-- Define a right triangle with acute angles x and 4x
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x < 90 ∧ x + 4*x = 90

-- Theorem statement
theorem smaller_acute_angle_measure :
  ∃ (x : ℝ), right_triangle x ∧ x = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_acute_angle_measure_l3861_386166


namespace NUMINAMATH_CALUDE_katie_game_difference_l3861_386152

theorem katie_game_difference : 
  ∀ (katie_new_games katie_old_games friends_new_games : ℕ),
  katie_new_games = 57 →
  katie_old_games = 39 →
  friends_new_games = 34 →
  katie_new_games + katie_old_games - friends_new_games = 62 := by
sorry

end NUMINAMATH_CALUDE_katie_game_difference_l3861_386152


namespace NUMINAMATH_CALUDE_runners_meet_time_l3861_386141

/-- The time in seconds for runner P to complete one round -/
def P_time : ℕ := 252

/-- The time in seconds for runner Q to complete one round -/
def Q_time : ℕ := 198

/-- The time in seconds for runner R to complete one round -/
def R_time : ℕ := 315

/-- The time after which all runners meet at the starting point -/
def meet_time : ℕ := 13860

/-- Theorem stating that the meet time is the least common multiple of individual round times -/
theorem runners_meet_time : 
  Nat.lcm (Nat.lcm P_time Q_time) R_time = meet_time := by sorry

end NUMINAMATH_CALUDE_runners_meet_time_l3861_386141


namespace NUMINAMATH_CALUDE_function_properties_l3861_386125

/-- The function f(x) with parameters a and b -/
def f (a b x : ℝ) : ℝ := 3 * (a * x^3 + b * x^2)

/-- The derivative of f(x) with respect to x -/
def f_derivative (a b x : ℝ) : ℝ := 9 * a * x^2 + 6 * b * x

theorem function_properties :
  ∃ (a b : ℝ),
    (∀ x, f a b x ≤ f a b 1) ∧
    (f a b 1 = 3) ∧
    (f_derivative a b 1 = 0) ∧
    (a = -2) ∧
    (b = 3) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≤ 15) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = 15) ∧
    (∀ x ∈ Set.Icc (-1) 3, f a b x ≥ -81) ∧
    (∃ x ∈ Set.Icc (-1) 3, f a b x = -81) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l3861_386125


namespace NUMINAMATH_CALUDE_journey_distance_l3861_386117

theorem journey_distance (total_time : Real) (bike_speed : Real) (walk_speed : Real) 
  (h1 : total_time = 56 / 60) -- 56 minutes converted to hours
  (h2 : bike_speed = 20)
  (h3 : walk_speed = 4) :
  let total_distance := (total_time * bike_speed * walk_speed) / (1/3 * bike_speed + 2/3 * walk_speed)
  let walk_distance := 1/3 * total_distance
  walk_distance = 2.7 := by sorry

end NUMINAMATH_CALUDE_journey_distance_l3861_386117


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l3861_386128

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → n % d ≠ 0

/-- A function that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The main theorem -/
theorem smallest_prime_perimeter_triangle :
  ∃ (a b c : ℕ),
    a < b ∧ b < c ∧
    isPrime a ∧ isPrime b ∧ isPrime c ∧
    a > 5 ∧ b > 5 ∧ c > 5 ∧
    isValidTriangle a b c ∧
    isPrime (a + b + c) ∧
    (∀ (x y z : ℕ),
      x < y ∧ y < z ∧
      isPrime x ∧ isPrime y ∧ isPrime z ∧
      x > 5 ∧ y > 5 ∧ z > 5 ∧
      isValidTriangle x y z ∧
      isPrime (x + y + z) →
      a + b + c ≤ x + y + z) ∧
    a + b + c = 31 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_triangle_l3861_386128


namespace NUMINAMATH_CALUDE_infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l3861_386100

/-- An infinite grid graph -/
def InfiniteGridGraph : Type := ℤ × ℤ

/-- A coloring function for the infinite grid graph -/
def Coloring (G : Type) := G → Fin 2

/-- A valid coloring of the infinite grid graph -/
def IsValidColoring (c : Coloring InfiniteGridGraph) : Prop :=
  ∀ (x y : ℤ), (x + y) % 2 = c (x, y)

/-- The chromatic number of the infinite grid graph is at most 2 -/
theorem infinite_grid_graph_chromatic_number :
  ∃ (c : Coloring InfiniteGridGraph), IsValidColoring c :=
sorry

/-- The chromatic number of the infinite grid graph is at least 2 -/
theorem infinite_grid_graph_chromatic_number_lower_bound :
  ¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1) :=
sorry

/-- The chromatic number of the infinite grid graph is exactly 2 -/
theorem infinite_grid_graph_chromatic_number_exact : 
  (∃ (c : Coloring InfiniteGridGraph), IsValidColoring c) ∧
  (¬∃ (c : InfiniteGridGraph → Fin 1), 
    ∀ (x y : ℤ), c (x, y) ≠ c (x + 1, y) ∨ c (x, y) ≠ c (x, y + 1)) :=
sorry

end NUMINAMATH_CALUDE_infinite_grid_graph_chromatic_number_infinite_grid_graph_chromatic_number_lower_bound_infinite_grid_graph_chromatic_number_exact_l3861_386100


namespace NUMINAMATH_CALUDE_circle_ring_area_floor_l3861_386177

theorem circle_ring_area_floor :
  let r : ℝ := 30 / 3 -- radius of small circles
  let R : ℝ := 30 -- radius of large circle C
  let K : ℝ := 3 * Real.pi * r^2 -- area between large circle and six small circles
  ⌊K⌋ = 942 := by
  sorry

end NUMINAMATH_CALUDE_circle_ring_area_floor_l3861_386177


namespace NUMINAMATH_CALUDE_probability_same_color_is_31_364_l3861_386174

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3
def total_plates : ℕ := red_plates + blue_plates + green_plates

def probability_same_color : ℚ :=
  (Nat.choose red_plates 3 + Nat.choose blue_plates 3 + Nat.choose green_plates 3) /
  Nat.choose total_plates 3

theorem probability_same_color_is_31_364 :
  probability_same_color = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_31_364_l3861_386174


namespace NUMINAMATH_CALUDE_two_objects_ten_recipients_l3861_386187

/-- The number of ways to distribute two distinct objects among a given number of recipients. -/
def distributionWays (recipients : ℕ) : ℕ := recipients * recipients

/-- Theorem: The number of ways to distribute two distinct objects among ten recipients is 100. -/
theorem two_objects_ten_recipients :
  distributionWays 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_two_objects_ten_recipients_l3861_386187


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3861_386181

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_divisible : 37800 ∣ n^3) : 
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 → m ∣ n → m ≤ q ∧ q = 6 :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3861_386181


namespace NUMINAMATH_CALUDE_followers_exceed_thousand_l3861_386161

/-- 
Given that Daniel starts with 5 followers on Sunday and his followers triple each day,
this theorem proves that Saturday (6 days after Sunday) is the first day 
when Daniel has more than 1000 followers.
-/
theorem followers_exceed_thousand (k : ℕ) : 
  (∀ n < k, 5 * 3^n ≤ 1000) ∧ 5 * 3^k > 1000 → k = 6 := by
  sorry

end NUMINAMATH_CALUDE_followers_exceed_thousand_l3861_386161


namespace NUMINAMATH_CALUDE_tangent_slope_at_zero_l3861_386184

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 - 2*x

theorem tangent_slope_at_zero :
  deriv f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_zero_l3861_386184


namespace NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l3861_386186

theorem tan_fifteen_degree_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l3861_386186


namespace NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3861_386126

theorem cubic_roots_sum_cubes (p q r : ℝ) : 
  (p^3 - 2*p^2 + 7*p - 1 = 0) → 
  (q^3 - 2*q^2 + 7*q - 1 = 0) → 
  (r^3 - 2*r^2 + 7*r - 1 = 0) → 
  (p + q + r = 2) →
  (p * q * r = 1) →
  (p + q - 2)^3 + (q + r - 2)^3 + (r + p - 2)^3 = -3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_cubes_l3861_386126


namespace NUMINAMATH_CALUDE_inequality_system_solution_l3861_386162

theorem inequality_system_solution :
  let S := {x : ℝ | (2*x - 6 < 3*x) ∧ (x - 2 + (x-1)/3 ≤ 1)}
  S = {x : ℝ | -6 < x ∧ x ≤ 5/2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l3861_386162


namespace NUMINAMATH_CALUDE_pie_chart_most_suitable_l3861_386145

-- Define the characteristics of the data
structure DataCharacteristics where
  partsOfWhole : Bool
  categorical : Bool
  compareProportions : Bool

-- Define the types of statistical graphs
inductive StatisticalGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

-- Define the suitability of a graph for given data characteristics
def isSuitable (graph : StatisticalGraph) (data : DataCharacteristics) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => data.partsOfWhole ∧ data.categorical ∧ data.compareProportions
  | _ => False

-- Theorem statement
theorem pie_chart_most_suitable (data : DataCharacteristics) 
  (h1 : data.partsOfWhole = true) 
  (h2 : data.categorical = true) 
  (h3 : data.compareProportions = true) :
  ∀ (graph : StatisticalGraph), 
    isSuitable graph data → graph = StatisticalGraph.PieChart := by
  sorry

end NUMINAMATH_CALUDE_pie_chart_most_suitable_l3861_386145


namespace NUMINAMATH_CALUDE_power_expressions_l3861_386165

theorem power_expressions (m n : ℤ) (a b : ℝ) 
  (h1 : 4^m = a) (h2 : 8^n = b) : 
  (2^(2*m + 3*n) = a * b) ∧ (2^(4*m - 6*n) = a^2 / b^2) := by
  sorry

end NUMINAMATH_CALUDE_power_expressions_l3861_386165


namespace NUMINAMATH_CALUDE_dolls_ratio_l3861_386103

theorem dolls_ratio (R S G : ℕ) : 
  S = G + 2 →
  G = 50 →
  R + S + G = 258 →
  R / S = 3 := by
sorry

end NUMINAMATH_CALUDE_dolls_ratio_l3861_386103


namespace NUMINAMATH_CALUDE_teairra_clothing_count_l3861_386146

/-- The number of shirts and pants Teairra has which are neither plaid nor purple -/
def non_plaid_purple_count (total_shirts : ℕ) (total_pants : ℕ) (plaid_shirts : ℕ) (purple_pants : ℕ) : ℕ :=
  (total_shirts - plaid_shirts) + (total_pants - purple_pants)

/-- Theorem stating that Teairra has 21 items that are neither plaid nor purple -/
theorem teairra_clothing_count : non_plaid_purple_count 5 24 3 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_teairra_clothing_count_l3861_386146


namespace NUMINAMATH_CALUDE_scale_division_l3861_386156

/-- Proves that dividing a scale of length 7 feet 12 inches into 4 equal parts results in parts that are 2 feet long each. -/
theorem scale_division (scale_length_feet : ℕ) (scale_length_inches : ℕ) (num_parts : ℕ) :
  scale_length_feet = 7 →
  scale_length_inches = 12 →
  num_parts = 4 →
  (scale_length_feet * 12 + scale_length_inches) / num_parts = 24 := by
  sorry

#check scale_division

end NUMINAMATH_CALUDE_scale_division_l3861_386156


namespace NUMINAMATH_CALUDE_counterfeit_bag_identification_l3861_386199

/-- Represents a bag of coins -/
structure CoinBag where
  weight : ℕ  -- Weight of each coin in grams
  count : ℕ   -- Number of coins taken from the bag

/-- Creates a list of 10 coin bags with the specified counterfeit bag -/
def createBags (counterfeitBag : ℕ) : List CoinBag :=
  List.range 10 |>.map (fun i =>
    if i + 1 = counterfeitBag then
      { weight := 11, count := i + 1 }
    else
      { weight := 10, count := i + 1 })

/-- Calculates the total weight of coins from all bags -/
def totalWeight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.weight * bag.count) 0

/-- The main theorem to prove -/
theorem counterfeit_bag_identification
  (counterfeitBag : ℕ) (h1 : 1 ≤ counterfeitBag) (h2 : counterfeitBag ≤ 10) :
  totalWeight (createBags counterfeitBag) - 550 = counterfeitBag := by
  sorry

#check counterfeit_bag_identification

end NUMINAMATH_CALUDE_counterfeit_bag_identification_l3861_386199


namespace NUMINAMATH_CALUDE_pizza_slice_volume_l3861_386151

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 16 →
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

#check pizza_slice_volume

end NUMINAMATH_CALUDE_pizza_slice_volume_l3861_386151


namespace NUMINAMATH_CALUDE_gcd_lcm_triples_count_l3861_386197

theorem gcd_lcm_triples_count : 
  (Finset.filter 
    (fun (triple : ℕ × ℕ × ℕ) => 
      Nat.gcd (Nat.gcd triple.1 triple.2.1) triple.2.2 = 15 ∧ 
      Nat.lcm (Nat.lcm triple.1 triple.2.1) triple.2.2 = 3^15 * 5^18)
    (Finset.product (Finset.range (3^15 * 5^18 + 1)) 
      (Finset.product (Finset.range (3^15 * 5^18 + 1)) (Finset.range (3^15 * 5^18 + 1))))).card = 8568 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_triples_count_l3861_386197


namespace NUMINAMATH_CALUDE_cube_sum_and_product_theorem_l3861_386122

theorem cube_sum_and_product_theorem :
  ∃! (n : ℕ), ∃ (a b : ℕ+),
    a ^ 3 + b ^ 3 = 189 ∧
    a * b = 20 ∧
    n = 2 :=
sorry

end NUMINAMATH_CALUDE_cube_sum_and_product_theorem_l3861_386122


namespace NUMINAMATH_CALUDE_temperature_theorem_l3861_386158

def temperature_problem (temp_ny temp_miami temp_sd temp_phoenix temp_denver : ℝ) : Prop :=
  let avg_three := (temp_ny + temp_miami + temp_sd) / 3
  temp_ny = 80 ∧
  temp_miami = temp_ny + 10 ∧
  temp_sd = temp_miami + 25 ∧
  temp_phoenix = temp_sd * 1.15 ∧
  temp_denver = avg_three - 5 ∧
  (temp_ny + temp_miami + temp_sd + temp_phoenix + temp_denver) / 5 = 101.45

theorem temperature_theorem :
  ∃ temp_ny temp_miami temp_sd temp_phoenix temp_denver : ℝ,
    temperature_problem temp_ny temp_miami temp_sd temp_phoenix temp_denver :=
by
  sorry

end NUMINAMATH_CALUDE_temperature_theorem_l3861_386158


namespace NUMINAMATH_CALUDE_locus_of_tangent_circles_l3861_386123

/-- The equation of circle C1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The equation of circle C3 -/
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 25

/-- A circle is externally tangent to C1 and internally tangent to C3 -/
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (5 - r)^2)

/-- The locus equation -/
def locus_equation (a b : ℝ) : Prop :=
  40 * a^2 + 49 * b^2 - 48 * a - 64 = 0

theorem locus_of_tangent_circles :
  ∀ a b : ℝ, (∃ r : ℝ, is_tangent_to_C1_C3 a b r) ↔ locus_equation a b :=
sorry

end NUMINAMATH_CALUDE_locus_of_tangent_circles_l3861_386123


namespace NUMINAMATH_CALUDE_adult_males_in_town_l3861_386154

/-- Represents the population distribution in a small town -/
structure TownPopulation where
  total : ℕ
  ratio_children : ℕ
  ratio_adult_males : ℕ
  ratio_adult_females : ℕ

/-- Calculates the number of adult males in the town -/
def adult_males (town : TownPopulation) : ℕ :=
  let total_ratio := town.ratio_children + town.ratio_adult_males + town.ratio_adult_females
  (town.total / total_ratio) * town.ratio_adult_males

/-- Theorem stating the number of adult males in the specific town -/
theorem adult_males_in_town (town : TownPopulation) 
  (h1 : town.total = 480)
  (h2 : town.ratio_children = 1)
  (h3 : town.ratio_adult_males = 2)
  (h4 : town.ratio_adult_females = 2) :
  adult_males town = 192 := by
  sorry

end NUMINAMATH_CALUDE_adult_males_in_town_l3861_386154


namespace NUMINAMATH_CALUDE_no_prime_common_multiple_under_70_l3861_386127

theorem no_prime_common_multiple_under_70 : ¬ ∃ n : ℕ, 
  (10 ∣ n) ∧ (15 ∣ n) ∧ (n < 70) ∧ Nat.Prime n :=
by sorry

end NUMINAMATH_CALUDE_no_prime_common_multiple_under_70_l3861_386127


namespace NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3861_386118

/-- 
Given an arithmetic sequence with first term a₁ = 1 and common difference d = 3,
prove that 2011 is the 671st term of this sequence.
-/
theorem arithmetic_sequence_2011 : 
  ∀ (a : ℕ → ℕ), 
    a 1 = 1 → 
    (∀ n, a (n + 1) - a n = 3) → 
    a 671 = 2011 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_2011_l3861_386118


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3861_386188

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 34*x^2 + 225 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 34*y^2 + 225 = 0 → x ≤ y) ∧
  x = -5 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l3861_386188


namespace NUMINAMATH_CALUDE_valentinas_burger_length_l3861_386194

/-- The length of a burger shared equally between two people, given the length of one person's share. -/
def burger_length (share_length : ℝ) : ℝ := 2 * share_length

/-- Proof that Valentina's burger is 12 inches long. -/
theorem valentinas_burger_length : 
  let share_length := 6
  burger_length share_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_valentinas_burger_length_l3861_386194


namespace NUMINAMATH_CALUDE_bridget_profit_is_40_l3861_386167

/-- Calculates Bridget's profit from baking and selling bread --/
def bridget_profit (
  total_loaves : ℕ)
  (morning_price afternoon_price late_afternoon_price : ℚ)
  (production_cost fixed_cost : ℚ) : ℚ :=
  let morning_sales := total_loaves / 3
  let afternoon_sales := (total_loaves - morning_sales) / 2
  let late_afternoon_sales := total_loaves - morning_sales - afternoon_sales
  let total_revenue := 
    morning_sales * morning_price + 
    afternoon_sales * afternoon_price + 
    late_afternoon_sales * late_afternoon_price
  let total_cost := total_loaves * production_cost + fixed_cost
  total_revenue - total_cost

/-- Theorem stating that Bridget's profit is $40 given the problem conditions --/
theorem bridget_profit_is_40 :
  bridget_profit 60 3 (3/2) 1 1 10 = 40 := by
  sorry

end NUMINAMATH_CALUDE_bridget_profit_is_40_l3861_386167


namespace NUMINAMATH_CALUDE_total_pears_picked_l3861_386139

theorem total_pears_picked (sara_pears tim_pears : ℕ) 
  (h1 : sara_pears = 6) 
  (h2 : tim_pears = 5) : 
  sara_pears + tim_pears = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_pears_picked_l3861_386139


namespace NUMINAMATH_CALUDE_sum_of_legs_is_48_l3861_386135

/-- A right triangle with consecutive even whole number legs and hypotenuse 34 -/
structure RightTriangle where
  leg1 : ℕ
  leg2 : ℕ
  hypotenuse : ℕ
  is_right : leg1^2 + leg2^2 = hypotenuse^2
  consecutive_even : leg2 = leg1 + 2
  hypotenuse_34 : hypotenuse = 34

/-- The sum of the legs of the special right triangle is 48 -/
theorem sum_of_legs_is_48 (t : RightTriangle) : t.leg1 + t.leg2 = 48 := by
  sorry

#check sum_of_legs_is_48

end NUMINAMATH_CALUDE_sum_of_legs_is_48_l3861_386135


namespace NUMINAMATH_CALUDE_second_largest_is_seven_l3861_386191

def numbers : Finset ℕ := {5, 8, 4, 3, 7}

theorem second_largest_is_seven :
  ∃ (x : ℕ), x ∈ numbers ∧ x > 7 ∧ ∀ y ∈ numbers, y ≠ x → y ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_second_largest_is_seven_l3861_386191


namespace NUMINAMATH_CALUDE_min_sum_of_primes_l3861_386101

theorem min_sum_of_primes (p q : ℕ) : 
  p > 1 → q > 1 → Nat.Prime p → Nat.Prime q → 
  17 * (p + 1) = 21 * (q + 1) → 
  (∀ p' q' : ℕ, p' > 1 → q' > 1 → Nat.Prime p' → Nat.Prime q' → 
    17 * (p' + 1) = 21 * (q' + 1) → p + q ≤ p' + q') → 
  p + q = 70 := by
sorry

end NUMINAMATH_CALUDE_min_sum_of_primes_l3861_386101


namespace NUMINAMATH_CALUDE_permutation_element_selection_l3861_386138

theorem permutation_element_selection (n : ℕ) (hn : n ≥ 10) :
  (Finset.range n).card.choose 3 = Nat.choose (n - 7) 3 :=
by sorry

end NUMINAMATH_CALUDE_permutation_element_selection_l3861_386138


namespace NUMINAMATH_CALUDE_right_triangle_inequality_l3861_386105

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 = b^2 + c^2) : 
  (a + b + c) * (1/a + 1/b + 1/c) ≥ 5 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_inequality_l3861_386105


namespace NUMINAMATH_CALUDE_exists_monochromatic_triplet_l3861_386115

/-- A coloring of natural numbers using two colors. -/
def Coloring := ℕ → Bool

/-- Predicate to check if three natural numbers form a valid triplet. -/
def ValidTriplet (x y z : ℕ) : Prop :=
  x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x * y = z^2

/-- Theorem stating that for any two-color painting of natural numbers,
    there always exist three distinct natural numbers x, y, and z
    of the same color such that xy = z^2. -/
theorem exists_monochromatic_triplet (c : Coloring) :
  ∃ x y z : ℕ, ValidTriplet x y z ∧ c x = c y ∧ c y = c z :=
sorry

end NUMINAMATH_CALUDE_exists_monochromatic_triplet_l3861_386115


namespace NUMINAMATH_CALUDE_no_integer_solutions_l3861_386155

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), x^4 + y^2 = 6*y - 3 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l3861_386155


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l3861_386189

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : M ⊆ N a → a ≤ -1 :=
by sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l3861_386189


namespace NUMINAMATH_CALUDE_smallest_dual_base_representation_l3861_386168

def is_valid_representation (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 2 ∧ b > 2 ∧
  n = 2 * a + 1 ∧
  n = b + 2

theorem smallest_dual_base_representation :
  (is_valid_representation 7) ∧
  (∀ m : ℕ, m < 7 → ¬(is_valid_representation m)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_representation_l3861_386168


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3861_386121

/-- The eccentricity of a hyperbola with equation x^2/m - y^2/5 = 1 is 3/2,
    given that m > 0 and its right focus coincides with the focus of y^2 = 12x -/
theorem hyperbola_eccentricity (m : ℝ) (h1 : m > 0) : ∃ (a b c : ℝ),
  m = a^2 ∧
  b^2 = 5 ∧
  c^2 = a^2 + b^2 ∧
  c = 3 ∧
  c / a = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3861_386121


namespace NUMINAMATH_CALUDE_root_implies_h_value_l3861_386147

theorem root_implies_h_value (h : ℝ) : 
  ((-1 : ℝ)^3 + h * (-1) - 20 = 0) → h = -21 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_h_value_l3861_386147


namespace NUMINAMATH_CALUDE_scientific_notation_correct_l3861_386109

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be represented in scientific notation -/
def number : ℕ := 2700000

/-- The scientific notation representation of the number -/
def scientific_representation : ScientificNotation :=
  { coefficient := 2.7
    exponent := 6
    property := by sorry }

/-- Theorem stating that the scientific notation representation is correct -/
theorem scientific_notation_correct :
  (scientific_representation.coefficient * (10 : ℝ) ^ scientific_representation.exponent) = number := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_correct_l3861_386109


namespace NUMINAMATH_CALUDE_tangent_line_and_inequality_l3861_386164

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_and_inequality (h : Real.exp 4 > 54) :
  (∃ m : ℝ, ∀ x : ℝ, x > 0 → (2 * x + m = f x → m = -Real.exp 1)) ∧
  (∀ x : ℝ, x > 0 → -1 / Real.exp 1 ≤ f x ∧ f x < Real.exp x / (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_and_inequality_l3861_386164


namespace NUMINAMATH_CALUDE_fifth_bounce_height_l3861_386110

/-- The height of a bouncing ball after n bounces, given an initial height and a bounce factor --/
def bounce_height (initial_height : ℝ) (bounce_factor : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_factor ^ n)

/-- Theorem: The height of the fifth bounce of a ball dropped from 96 feet,
    bouncing to half its previous height each time, is 3 feet --/
theorem fifth_bounce_height :
  bounce_height 96 (1/2) 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_fifth_bounce_height_l3861_386110


namespace NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l3861_386133

-- Define the function f
def f (b c x : ℝ) : ℝ := x * abs x + b * x + c

-- Proposition 1
theorem prop_1 (b : ℝ) : 
  ∀ x, f b 0 (-x) = -(f b 0 x) := by sorry

-- Proposition 2
theorem prop_2 (c : ℝ) (h : c > 0) : 
  ∃! x, f 0 c x = 0 := by sorry

-- Proposition 3
theorem prop_3 (b c : ℝ) : 
  ∀ x, f b c (-x) = 2 * c - f b c x := by sorry

end NUMINAMATH_CALUDE_prop_1_prop_2_prop_3_l3861_386133


namespace NUMINAMATH_CALUDE_sweater_cost_l3861_386190

/-- Given shopping information, prove the cost of a sweater --/
theorem sweater_cost (initial_amount : ℕ) (tshirt_cost : ℕ) (shoes_cost : ℕ) (remaining_amount : ℕ)
  (h1 : initial_amount = 91)
  (h2 : tshirt_cost = 6)
  (h3 : shoes_cost = 11)
  (h4 : remaining_amount = 50) :
  initial_amount - remaining_amount - tshirt_cost - shoes_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_sweater_cost_l3861_386190


namespace NUMINAMATH_CALUDE_sale_price_lower_than_original_l3861_386111

theorem sale_price_lower_than_original (x : ℝ) (h : x > 0) : 
  0.75 * (1.3 * x) < x := by sorry

end NUMINAMATH_CALUDE_sale_price_lower_than_original_l3861_386111


namespace NUMINAMATH_CALUDE_green_squares_count_l3861_386144

/-- Represents a grid with colored squares -/
structure ColoredGrid where
  rows : Nat
  squares_per_row : Nat
  red_rows : Nat
  red_squares_per_row : Nat
  blue_rows : Nat

/-- Calculates the number of green squares in the grid -/
def green_squares (grid : ColoredGrid) : Nat :=
  grid.rows * grid.squares_per_row - 
  (grid.red_rows * grid.red_squares_per_row + grid.blue_rows * grid.squares_per_row)

/-- Theorem stating that the number of green squares in the given grid configuration is 66 -/
theorem green_squares_count (grid : ColoredGrid) 
  (h1 : grid.rows = 10)
  (h2 : grid.squares_per_row = 15)
  (h3 : grid.red_rows = 4)
  (h4 : grid.red_squares_per_row = 6)
  (h5 : grid.blue_rows = 4) :
  green_squares grid = 66 := by
  sorry

#eval green_squares { rows := 10, squares_per_row := 15, red_rows := 4, red_squares_per_row := 6, blue_rows := 4 }

end NUMINAMATH_CALUDE_green_squares_count_l3861_386144


namespace NUMINAMATH_CALUDE_range_of_f_l3861_386102

-- Define the function f(x) = |x| - 4
def f (x : ℝ) : ℝ := |x| - 4

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = {y : ℝ | y ≥ -4} :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3861_386102


namespace NUMINAMATH_CALUDE_cookie_arrangements_count_l3861_386130

/-- The number of distinct arrangements of letters in "COOKIE" -/
def cookieArrangements : ℕ :=
  Nat.factorial 6 / (Nat.factorial 2 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1 * Nat.factorial 1)

/-- Theorem stating that the number of distinct arrangements of letters in "COOKIE" is 360 -/
theorem cookie_arrangements_count : cookieArrangements = 360 := by
  sorry

end NUMINAMATH_CALUDE_cookie_arrangements_count_l3861_386130


namespace NUMINAMATH_CALUDE_expression_evaluation_l3861_386182

theorem expression_evaluation (x y : ℝ) (hx : x = 1) (hy : y = -2) :
  ((x - y)^2 - x*(3*x - 2*y) + (x + y)*(x - y)) / (2*x) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3861_386182


namespace NUMINAMATH_CALUDE_A_power_95_l3861_386193

def A : Matrix (Fin 3) (Fin 3) ℝ := !![0, 0, 0; 0, 0, -1; 0, 1, 0]

theorem A_power_95 : A ^ 95 = !![0, 0, 0; 0, 0, 1; 0, -1, 0] := by
  sorry

end NUMINAMATH_CALUDE_A_power_95_l3861_386193


namespace NUMINAMATH_CALUDE_max_tickets_purchasable_l3861_386196

theorem max_tickets_purchasable (ticket_price budget : ℚ) : 
  ticket_price = 18 → budget = 150 → 
  ∃ (n : ℕ), n * ticket_price ≤ budget ∧ 
  ∀ (m : ℕ), m * ticket_price ≤ budget → m ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_max_tickets_purchasable_l3861_386196


namespace NUMINAMATH_CALUDE_max_profit_l3861_386192

/-- Profit function for location A -/
def L₁ (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2

/-- Profit function for location B -/
def L₂ (x : ℝ) : ℝ := 2 * x

/-- Total number of cars sold across both locations -/
def total_cars : ℝ := 15

/-- Total profit function -/
def L (x : ℝ) : ℝ := L₁ x + L₂ (total_cars - x)

theorem max_profit :
  ∃ x : ℝ, 0 ≤ x ∧ x ≤ total_cars ∧
  ∀ y : ℝ, 0 ≤ y ∧ y ≤ total_cars → L y ≤ L x ∧ L x = 45.6 :=
sorry

end NUMINAMATH_CALUDE_max_profit_l3861_386192


namespace NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_3599_l3861_386107

theorem multiplicative_inverse_600_mod_3599 :
  ∃ (n : ℕ), n < 3599 ∧ (600 * n) % 3599 = 1 :=
by
  -- Define the right triangle
  let a : ℕ := 45
  let b : ℕ := 336
  let c : ℕ := 339
  
  -- Assert that a, b, c form a right triangle
  have right_triangle : a^2 + b^2 = c^2 := by sorry
  
  -- Define the multiplicative inverse
  let inverse : ℕ := 1200
  
  -- Prove that inverse is less than 3599
  have inverse_bound : inverse < 3599 := by sorry
  
  -- Prove that inverse is the multiplicative inverse of 600 modulo 3599
  have inverse_property : (600 * inverse) % 3599 = 1 := by sorry
  
  -- Combine the proofs
  exact ⟨inverse, inverse_bound, inverse_property⟩

#eval (600 * 1200) % 3599  -- Should output 1

end NUMINAMATH_CALUDE_multiplicative_inverse_600_mod_3599_l3861_386107


namespace NUMINAMATH_CALUDE_valid_fractions_l3861_386175

def is_valid_fraction (num den : ℕ) : Prop :=
  10 ≤ num ∧ num < 100 ∧ 10 ≤ den ∧ den < 100 ∧
  (num / 10 : ℕ) = den % 10 ∧
  (num % 10 : ℚ) / (den / 10 : ℚ) = (num : ℚ) / (den : ℚ)

theorem valid_fractions :
  {f : ℚ | ∃ (num den : ℕ), is_valid_fraction num den ∧ f = (num : ℚ) / (den : ℚ)} =
  {64/16, 98/49, 95/19, 65/26} :=
by sorry

end NUMINAMATH_CALUDE_valid_fractions_l3861_386175


namespace NUMINAMATH_CALUDE_escalator_time_l3861_386108

/-- The time taken for a person to cover the entire length of an escalator -/
theorem escalator_time (escalator_speed : ℝ) (person_speed : ℝ) (escalator_length : ℝ) 
  (h1 : escalator_speed = 20)
  (h2 : person_speed = 4)
  (h3 : escalator_length = 210) : 
  escalator_length / (escalator_speed + person_speed) = 8.75 := by
sorry

end NUMINAMATH_CALUDE_escalator_time_l3861_386108


namespace NUMINAMATH_CALUDE_lineup_theorem_l3861_386153

def total_people : ℕ := 7
def selected_people : ℕ := 5

def ways_including_A : ℕ := 1800
def ways_not_all_ABC : ℕ := 1800
def ways_ABC_adjacent : ℕ := 144

theorem lineup_theorem :
  (ways_including_A = 1800) ∧
  (ways_not_all_ABC = 1800) ∧
  (ways_ABC_adjacent = 144) :=
by sorry

end NUMINAMATH_CALUDE_lineup_theorem_l3861_386153


namespace NUMINAMATH_CALUDE_class_size_l3861_386178

theorem class_size (chinese : ℕ) (math : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : chinese = 15)
  (h2 : math = 18)
  (h3 : both = 8)
  (h4 : neither = 20) :
  chinese + math - both + neither = 45 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l3861_386178


namespace NUMINAMATH_CALUDE_congruence_mod_10_l3861_386113

theorem congruence_mod_10 : ∃ C : ℤ, (1 + C * (2^20 - 1)) % 10 = 2011 % 10 := by
  sorry

end NUMINAMATH_CALUDE_congruence_mod_10_l3861_386113


namespace NUMINAMATH_CALUDE_no_integer_solution_l3861_386143

theorem no_integer_solution : ¬∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 93^3 + b * 93^2 + c * 93 + d = 2) := by
sorry

end NUMINAMATH_CALUDE_no_integer_solution_l3861_386143


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3861_386176

theorem complex_fraction_equals_i : (Complex.I + 1) / (1 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3861_386176


namespace NUMINAMATH_CALUDE_scientific_notation_935_million_l3861_386150

theorem scientific_notation_935_million :
  (935000000 : ℝ) = 9.35 * (10 ^ 8) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_935_million_l3861_386150


namespace NUMINAMATH_CALUDE_magnitude_of_a_plus_bi_l3861_386185

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the given equation
def given_equation (a b : ℝ) : Prop :=
  a / (1 - i) = 1 - b * i

-- State the theorem
theorem magnitude_of_a_plus_bi (a b : ℝ) :
  given_equation a b → Complex.abs (a + b * i) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_a_plus_bi_l3861_386185


namespace NUMINAMATH_CALUDE_correct_calculation_l3861_386179

theorem correct_calculation : 
  (-2 - 3 = -5) ∧ 
  (-3^2 ≠ -6) ∧ 
  (1/2 / 2 ≠ 2 * 2) ∧ 
  ((-2/3)^2 ≠ 4/3) := by
sorry

end NUMINAMATH_CALUDE_correct_calculation_l3861_386179


namespace NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l3861_386195

theorem arcsin_sqrt2_over_2 : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end NUMINAMATH_CALUDE_arcsin_sqrt2_over_2_l3861_386195


namespace NUMINAMATH_CALUDE_sin_theta_value_l3861_386134

theorem sin_theta_value (θ : ℝ) (h : Real.cos (π / 4 - θ / 2) = 2 / 3) : 
  Real.sin θ = -1 / 9 := by
sorry

end NUMINAMATH_CALUDE_sin_theta_value_l3861_386134


namespace NUMINAMATH_CALUDE_x_squared_minus_x_plus_one_equals_seven_l3861_386136

theorem x_squared_minus_x_plus_one_equals_seven (x : ℝ) 
  (h : (x^2 - x)^2 - 4*(x^2 - x) - 12 = 0) : 
  x^2 - x + 1 = 7 := by
sorry

end NUMINAMATH_CALUDE_x_squared_minus_x_plus_one_equals_seven_l3861_386136


namespace NUMINAMATH_CALUDE_finite_valid_pairs_l3861_386159

/-- Represents a valid age-year pair for Dick and Jane's problem -/
structure AgePair where
  d : ℕ  -- Dick's current age
  n : ℕ  -- Years in the future
  h1 : d ≥ 35  -- Dick is at least 5 years older than Jane (who is 30)
  h2 : d + n ≥ 10 ∧ d + n ≤ 99  -- Dick's future age is a two-digit number
  h3 : 30 + n ≥ 10 ∧ 30 + n ≤ 99  -- Jane's future age is a two-digit number
  h4 : ∃ (a b : ℕ), a ≤ 9 ∧ b ≤ 9 ∧ d + n = 10 * b + a ∧ 30 + n = 10 * a + b  -- Digit interchange property
  h5 : d + n + 30 + n < 120  -- Sum of future ages is less than 120

/-- The set of all valid AgePairs -/
def validAgePairs : Set AgePair := {ap | True}

theorem finite_valid_pairs : Set.Finite validAgePairs := by
  sorry

#check finite_valid_pairs

end NUMINAMATH_CALUDE_finite_valid_pairs_l3861_386159


namespace NUMINAMATH_CALUDE_gcd_228_1995_l3861_386116

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l3861_386116


namespace NUMINAMATH_CALUDE_kathleens_allowance_increase_l3861_386198

theorem kathleens_allowance_increase (middle_school_allowance senior_year_allowance : ℚ) : 
  middle_school_allowance = 8 + 2 →
  senior_year_allowance = 2 * middle_school_allowance + 5 →
  (senior_year_allowance - middle_school_allowance) / middle_school_allowance * 100 = 150 := by
  sorry

end NUMINAMATH_CALUDE_kathleens_allowance_increase_l3861_386198


namespace NUMINAMATH_CALUDE_angle_subtraction_theorem_l3861_386170

-- Define a custom type for angle measurements in degrees, minutes, and seconds
structure AngleDMS where
  degrees : Int
  minutes : Int
  seconds : Int

-- Define the subtraction operation for AngleDMS
def AngleDMS.sub (a b : AngleDMS) : AngleDMS :=
  sorry

theorem angle_subtraction_theorem :
  let a := AngleDMS.mk 108 18 25
  let b := AngleDMS.mk 56 23 32
  let result := AngleDMS.mk 51 54 53
  a.sub b = result := by sorry

end NUMINAMATH_CALUDE_angle_subtraction_theorem_l3861_386170


namespace NUMINAMATH_CALUDE_range_of_a_l3861_386173

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ x : ℝ, x ≥ a ∧ |x - 1| ≥ 1) → 
  a ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3861_386173


namespace NUMINAMATH_CALUDE_bridge_length_l3861_386131

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 256 →
  train_speed_kmh = 72 →
  time_to_cross = 20 →
  (train_speed_kmh * 1000 / 3600 * time_to_cross) - train_length = 144 := by
  sorry

#check bridge_length

end NUMINAMATH_CALUDE_bridge_length_l3861_386131


namespace NUMINAMATH_CALUDE_fred_weekend_earnings_l3861_386112

/-- Fred's earnings over the weekend -/
def fred_earnings (initial_amount final_amount : ℕ) : ℕ :=
  final_amount - initial_amount

/-- Theorem stating Fred's earnings -/
theorem fred_weekend_earnings :
  fred_earnings 19 40 = 21 := by
  sorry

end NUMINAMATH_CALUDE_fred_weekend_earnings_l3861_386112


namespace NUMINAMATH_CALUDE_roots_difference_abs_l3861_386124

theorem roots_difference_abs (r₁ r₂ : ℝ) : 
  r₁^2 - 7*r₁ + 12 = 0 → 
  r₂^2 - 7*r₂ + 12 = 0 → 
  |r₁ - r₂| = 1 := by
sorry

end NUMINAMATH_CALUDE_roots_difference_abs_l3861_386124


namespace NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l3861_386106

/-- A quadrilateral is a polygon with four sides and four vertices. -/
structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

/-- An angle is right if it measures 90 degrees or π/2 radians. -/
def is_right_angle (q : Quadrilateral) (i : Fin 4) : Prop := sorry

/-- A rectangle is a quadrilateral with four right angles. -/
def is_rectangle (q : Quadrilateral) : Prop :=
  ∀ i : Fin 4, is_right_angle q i

/-- Theorem: If a quadrilateral has three right angles, it is a rectangle. -/
theorem three_right_angles_implies_rectangle (q : Quadrilateral) 
  (h : ∃ i j k : Fin 4, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    is_right_angle q i ∧ is_right_angle q j ∧ is_right_angle q k) : 
  is_rectangle q :=
sorry

end NUMINAMATH_CALUDE_three_right_angles_implies_rectangle_l3861_386106


namespace NUMINAMATH_CALUDE_perfect_squares_problem_l3861_386160

theorem perfect_squares_problem (m n a b c d : ℕ) :
  2000 + 100 * a + 10 * b + 9 = n^2 →
  2000 + 100 * c + 10 * d + 9 = m^2 →
  m > n →
  10 ≤ 10 * a + b →
  10 * a + b ≤ 99 →
  10 ≤ 10 * c + d →
  10 * c + d ≤ 99 →
  m + n = 100 ∧ (10 * a + b) + (10 * c + d) = 100 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_problem_l3861_386160


namespace NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3861_386129

theorem right_triangle_shorter_leg : 
  ∀ a b c : ℕ,
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 65 →           -- Hypotenuse length
  a ≤ b →            -- a is the shorter leg
  a = 16 :=          -- The shorter leg is 16
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_shorter_leg_l3861_386129
