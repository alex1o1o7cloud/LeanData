import Mathlib

namespace slope_135_implies_y_negative_four_l1360_136090

/-- Given two points A and B, if the slope of the line passing through them is 135°, then the y-coordinate of A is -4. -/
theorem slope_135_implies_y_negative_four (x_a y_a x_b y_b : ℝ) :
  x_a = 3 →
  x_b = 2 →
  y_b = -3 →
  (y_a - y_b) / (x_a - x_b) = Real.tan (135 * π / 180) →
  y_a = -4 := by
  sorry

#check slope_135_implies_y_negative_four

end slope_135_implies_y_negative_four_l1360_136090


namespace circle_rectangles_l1360_136050

/-- The number of points on the circle's circumference -/
def n : ℕ := 12

/-- The number of diameters in the circle -/
def num_diameters : ℕ := n / 2

/-- The number of rectangles that can be formed -/
def num_rectangles : ℕ := Nat.choose num_diameters 2

theorem circle_rectangles :
  num_rectangles = 15 :=
sorry

end circle_rectangles_l1360_136050


namespace theater_ticket_price_l1360_136058

theorem theater_ticket_price (adult_price : ℕ) 
  (total_attendance : ℕ) (total_revenue : ℕ) (child_attendance : ℕ) :
  total_attendance = 280 →
  total_revenue = 14000 →
  child_attendance = 80 →
  (total_attendance - child_attendance) * adult_price + child_attendance * 25 = total_revenue →
  adult_price = 60 := by
sorry

end theater_ticket_price_l1360_136058


namespace undamaged_tins_count_l1360_136033

theorem undamaged_tins_count (cases : ℕ) (tins_per_case : ℕ) (damage_percent : ℚ) : 
  cases = 15 → 
  tins_per_case = 24 → 
  damage_percent = 5 / 100 →
  cases * tins_per_case * (1 - damage_percent) = 342 := by
sorry

end undamaged_tins_count_l1360_136033


namespace symmetric_function_properties_l1360_136009

/-- A function satisfying certain symmetry properties -/
structure SymmetricFunction where
  f : ℝ → ℝ
  sym_2 : ∀ x, f (2 - x) = f (2 + x)
  sym_7 : ∀ x, f (7 - x) = f (7 + x)
  zero_at_origin : f 0 = 0

/-- The number of zeros of a function in an interval -/
def num_zeros (f : ℝ → ℝ) (a b : ℝ) : ℕ := sorry

/-- A function is periodic with period p -/
def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

/-- Main theorem about SymmetricFunction -/
theorem symmetric_function_properties (sf : SymmetricFunction) :
  num_zeros sf.f (-30) 30 ≥ 13 ∧ is_periodic sf.f 10 := by sorry

end symmetric_function_properties_l1360_136009


namespace x_fifth_minus_seven_x_equals_222_l1360_136038

theorem x_fifth_minus_seven_x_equals_222 (x : ℝ) (h : x = 3) : x^5 - 7*x = 222 := by
  sorry

end x_fifth_minus_seven_x_equals_222_l1360_136038


namespace jiangxia_is_first_largest_bidirectional_l1360_136042

structure TidalPowerPlant where
  location : String
  year_built : Nat
  is_bidirectional : Bool
  is_largest : Bool

def china_tidal_plants : Nat := 9

def jiangxia_plant : TidalPowerPlant := {
  location := "Jiangxia",
  year_built := 1980,
  is_bidirectional := true,
  is_largest := true
}

theorem jiangxia_is_first_largest_bidirectional :
  ∃ (plant : TidalPowerPlant),
    plant.year_built = 1980 ∧
    plant.is_bidirectional = true ∧
    plant.is_largest = true ∧
    plant.location = "Jiangxia" :=
by
  sorry

#check jiangxia_is_first_largest_bidirectional

end jiangxia_is_first_largest_bidirectional_l1360_136042


namespace rational_representation_condition_l1360_136015

theorem rational_representation_condition (x y : ℚ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (∀ (q : ℚ), q > 0 → ∃ (r : ℚ), r > 0 ∧ q = (r * x) / (r * y)) ↔ x * y < 0 :=
sorry

end rational_representation_condition_l1360_136015


namespace sin_alpha_value_l1360_136035

theorem sin_alpha_value (α : Real) :
  let P : Real × Real := (-2 * Real.sin (60 * π / 180), 2 * Real.cos (30 * π / 180))
  (∃ k : Real, k > 0 ∧ P = (k * Real.cos α, k * Real.sin α)) →
  Real.sin α = Real.sqrt 2 / 2 := by
sorry

end sin_alpha_value_l1360_136035


namespace division_remainder_l1360_136018

theorem division_remainder : ∃ A : ℕ, 28 = 3 * 9 + A ∧ A < 3 := by
  sorry

end division_remainder_l1360_136018


namespace solve_bus_problem_l1360_136085

def bus_problem (initial : ℕ) (stop_a_off stop_a_on : ℕ) (stop_b_off stop_b_on : ℕ) 
                 (stop_c_off stop_c_on : ℕ) (stop_d_off : ℕ) (final : ℕ) : Prop :=
  let after_a := initial - stop_a_off + stop_a_on
  let after_b := after_a - stop_b_off + stop_b_on
  let after_c := after_b - stop_c_off + stop_c_on
  let after_d := after_c - stop_d_off
  ∃ (stop_d_on : ℕ), after_d + stop_d_on = final ∧ stop_d_on = 10

theorem solve_bus_problem : 
  bus_problem 64 8 12 4 6 14 22 10 78 := by
  sorry

end solve_bus_problem_l1360_136085


namespace multiples_17_sums_l1360_136047

/-- The sum of the first n positive integers -/
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The sum of the squares of the first n positive integers -/
def sum_squares_n (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The sum of the first twelve positive multiples of 17 -/
def sum_multiples_17 : ℕ := 17 * sum_n 12

/-- The sum of the squares of the first twelve positive multiples of 17 -/
def sum_squares_multiples_17 : ℕ := 17^2 * sum_squares_n 12

theorem multiples_17_sums :
  sum_multiples_17 = 1326 ∧ sum_squares_multiples_17 = 187850 := by
  sorry

end multiples_17_sums_l1360_136047


namespace T_perimeter_is_20_l1360_136052

/-- The perimeter of a T shape formed by two 2-inch × 4-inch rectangles -/
def T_perimeter : ℝ :=
  let rectangle_width : ℝ := 2
  let rectangle_length : ℝ := 4
  let rectangle_perimeter : ℝ := 2 * (rectangle_width + rectangle_length)
  let overlap : ℝ := 2 * rectangle_width
  2 * rectangle_perimeter - overlap

/-- Theorem stating that the perimeter of the T shape is 20 inches -/
theorem T_perimeter_is_20 : T_perimeter = 20 := by
  sorry

end T_perimeter_is_20_l1360_136052


namespace sine_cosine_extreme_value_l1360_136032

open Real

theorem sine_cosine_extreme_value (a b : ℝ) (h : a < b) :
  ∃ f g : ℝ → ℝ,
    (∀ x ∈ Set.Icc a b, f x = sin x ∧ g x = cos x) ∧
    g a * g b < 0 ∧
    ¬(∃ x ∈ Set.Icc a b, ∀ y ∈ Set.Icc a b, g x ≤ g y ∨ g x ≥ g y) :=
by sorry

end sine_cosine_extreme_value_l1360_136032


namespace greatest_two_digit_product_12_proof_l1360_136030

/-- The greatest two-digit whole number whose digits have a product of 12 -/
def greatest_two_digit_product_12 : ℕ := 62

/-- Predicate to check if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Function to get the tens digit of a two-digit number -/
def tens_digit (n : ℕ) : ℕ := n / 10

/-- Function to get the ones digit of a two-digit number -/
def ones_digit (n : ℕ) : ℕ := n % 10

theorem greatest_two_digit_product_12_proof :
  (is_two_digit greatest_two_digit_product_12) ∧
  (tens_digit greatest_two_digit_product_12 * ones_digit greatest_two_digit_product_12 = 12) ∧
  (∀ m : ℕ, is_two_digit m → 
    tens_digit m * ones_digit m = 12 → 
    m ≤ greatest_two_digit_product_12) :=
by sorry

end greatest_two_digit_product_12_proof_l1360_136030


namespace intersection_of_A_and_B_l1360_136082

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define set A
def A : Set ℝ := {x | x^2 - (floor x : ℝ) = 2}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {-1, Real.sqrt 3} :=
by sorry

end intersection_of_A_and_B_l1360_136082


namespace expansion_sum_coefficients_l1360_136044

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the expression
def expansion_sum (x : ℕ) : ℕ :=
  (binomial_coeff x 1 + binomial_coeff x 2 + binomial_coeff x 3 + binomial_coeff x 4) ^ 2

-- Theorem statement
theorem expansion_sum_coefficients :
  ∃ x, expansion_sum x = 225 := by sorry

end expansion_sum_coefficients_l1360_136044


namespace vector_problem_l1360_136040

theorem vector_problem (x y : ℝ) (hx : x > 0) : 
  let a : ℝ × ℝ × ℝ := (2, 4, x)
  let b : ℝ × ℝ × ℝ := (2, y, 2)
  (2^2 + 4^2 + x^2 = (3*Real.sqrt 5)^2) →
  (2*2 + 4*y + x*2 = 0) →
  x + 2*y = -2 := by
sorry

end vector_problem_l1360_136040


namespace sports_equipment_store_problem_l1360_136064

/-- Sports equipment store problem -/
theorem sports_equipment_store_problem 
  (total_balls : ℕ) 
  (budget : ℕ) 
  (basketball_cost : ℕ) 
  (volleyball_cost : ℕ) 
  (basketball_price_ratio : ℚ) 
  (school_basketball_purchase : ℕ) 
  (school_volleyball_purchase : ℕ) 
  (school_basketball_count : ℕ) 
  (school_volleyball_count : ℕ) :
  total_balls = 200 →
  budget ≤ 5000 →
  basketball_cost = 30 →
  volleyball_cost = 24 →
  basketball_price_ratio = 3/2 →
  school_basketball_purchase = 1800 →
  school_volleyball_purchase = 1500 →
  school_volleyball_count = school_basketball_count + 10 →
  ∃ (basketball_price volleyball_price : ℕ) 
    (optimal_basketball optimal_volleyball : ℕ),
    basketball_price = 45 ∧
    volleyball_price = 30 ∧
    optimal_basketball = 33 ∧
    optimal_volleyball = 167 ∧
    optimal_basketball + optimal_volleyball = total_balls ∧
    optimal_basketball * basketball_cost + optimal_volleyball * volleyball_cost ≤ budget ∧
    ∀ (b v : ℕ), 
      b + v = total_balls →
      b * basketball_cost + v * volleyball_cost ≤ budget →
      (basketball_price - 3 - basketball_cost) * b + (volleyball_price - 2 - volleyball_cost) * v ≤
      (basketball_price - 3 - basketball_cost) * optimal_basketball + 
      (volleyball_price - 2 - volleyball_cost) * optimal_volleyball :=
by sorry

end sports_equipment_store_problem_l1360_136064


namespace village_population_l1360_136094

theorem village_population (p : ℝ) : p = 939 ↔ 0.92 * p = 1.15 * p + 216 := by
  sorry

end village_population_l1360_136094


namespace basketball_substitutions_remainder_l1360_136075

/-- The number of ways to make exactly k substitutions in a basketball game -/
def num_substitutions (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => 12 * (13 - k) * num_substitutions k

/-- The total number of ways to make substitutions in the basketball game -/
def total_substitutions : ℕ :=
  num_substitutions 0 + num_substitutions 1 + num_substitutions 2 + 
  num_substitutions 3 + num_substitutions 4

theorem basketball_substitutions_remainder :
  total_substitutions % 1000 = 953 := by
  sorry

end basketball_substitutions_remainder_l1360_136075


namespace middle_income_sample_size_l1360_136013

/-- Calculates the number of middle-income households to be sampled in a stratified sampling method. -/
theorem middle_income_sample_size 
  (total_households : ℕ) 
  (middle_income_households : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_households = 600) 
  (h2 : middle_income_households = 360) 
  (h3 : sample_size = 80) :
  (middle_income_households : ℚ) / (total_households : ℚ) * (sample_size : ℚ) = 48 := by
  sorry

end middle_income_sample_size_l1360_136013


namespace sundae_cost_calculation_l1360_136067

-- Define constants for prices and discount thresholds
def scoop_price : ℚ := 2
def topping_a_price : ℚ := 0.5
def topping_b_price : ℚ := 0.75
def topping_c_price : ℚ := 0.6
def topping_d_price : ℚ := 0.8
def topping_e_price : ℚ := 0.9

def topping_a_discount_threshold : ℕ := 3
def topping_b_discount_threshold : ℕ := 2
def topping_c_discount_threshold : ℕ := 4

def topping_a_discount : ℚ := 0.3
def topping_b_discount : ℚ := 0.4
def topping_c_discount : ℚ := 0.5

-- Define the function to calculate the total cost
def calculate_sundae_cost (scoops topping_a topping_b topping_c topping_d topping_e : ℕ) : ℚ :=
  let ice_cream_cost := scoops * scoop_price
  let topping_a_cost := topping_a * topping_a_price - (topping_a / topping_a_discount_threshold) * topping_a_discount
  let topping_b_cost := topping_b * topping_b_price - (topping_b / topping_b_discount_threshold) * topping_b_discount
  let topping_c_cost := topping_c * topping_c_price - (topping_c / topping_c_discount_threshold) * topping_c_discount
  let topping_d_cost := topping_d * topping_d_price
  let topping_e_cost := topping_e * topping_e_price
  ice_cream_cost + topping_a_cost + topping_b_cost + topping_c_cost + topping_d_cost + topping_e_cost

-- Theorem statement
theorem sundae_cost_calculation :
  calculate_sundae_cost 3 5 3 7 2 1 = 16.25 := by
  sorry

end sundae_cost_calculation_l1360_136067


namespace tricycle_count_l1360_136051

/-- The number of tricycles in a group of children -/
def num_tricycles (total_children : ℕ) (total_wheels : ℕ) : ℕ :=
  total_children - (total_wheels - 3 * total_children) / 1

/-- Theorem stating that given 10 children and 26 wheels, there are 6 tricycles -/
theorem tricycle_count : num_tricycles 10 26 = 6 := by
  sorry

end tricycle_count_l1360_136051


namespace train_speed_calculation_l1360_136045

/-- Calculate the speed of a train crossing a bridge -/
theorem train_speed_calculation (train_length bridge_length crossing_time : ℝ) 
  (h1 : train_length = 250)
  (h2 : bridge_length = 500)
  (h3 : crossing_time = 8) :
  (train_length + bridge_length) / crossing_time = 93.75 := by
  sorry

end train_speed_calculation_l1360_136045


namespace unique_square_divisible_by_three_in_range_l1360_136046

theorem unique_square_divisible_by_three_in_range : ∃! y : ℕ,
  50 < y ∧ y < 120 ∧ ∃ x : ℕ, y = x^2 ∧ y % 3 = 0 :=
by sorry

end unique_square_divisible_by_three_in_range_l1360_136046


namespace sum_of_powers_inequality_l1360_136001

theorem sum_of_powers_inequality (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 ≥ 6 ∧
  (a^6 / b^6 + a^4 / b^4 + a^2 / b^2 + b^6 / a^6 + b^4 / a^4 + b^2 / a^2 = 6 ↔ a = b) :=
by sorry

end sum_of_powers_inequality_l1360_136001


namespace fourth_number_proof_l1360_136099

theorem fourth_number_proof (x : ℝ) : 
  3 + 33 + 333 + x = 399.6 → x = 30.6 := by
sorry

end fourth_number_proof_l1360_136099


namespace log_equation_sum_l1360_136027

theorem log_equation_sum : ∃ (X Y Z : ℕ+),
  (∀ d : ℕ+, d ∣ X ∧ d ∣ Y ∧ d ∣ Z → d = 1) ∧
  (X : ℝ) * Real.log 3 / Real.log 180 + (Y : ℝ) * Real.log 5 / Real.log 180 = Z ∧
  X + Y + Z = 4 := by
  sorry

end log_equation_sum_l1360_136027


namespace inverse_proposition_is_correct_l1360_136076

/-- The original proposition -/
def original_proposition (n : ℕ) : Prop :=
  n % 10 = 5 → n % 5 = 0

/-- The inverse proposition -/
def inverse_proposition (n : ℕ) : Prop :=
  n % 5 = 0 → n % 10 = 5

/-- Theorem stating that the inverse_proposition is indeed the inverse of the original_proposition -/
theorem inverse_proposition_is_correct :
  inverse_proposition = λ n => ¬(original_proposition n) → ¬(n % 10 = 5) :=
by sorry

end inverse_proposition_is_correct_l1360_136076


namespace arrangements_remainder_l1360_136041

/-- The number of green marbles -/
def green_marbles : ℕ := 8

/-- The maximum number of red marbles that satisfies the equal neighbor condition -/
def max_red_marbles : ℕ := 23

/-- The number of possible arrangements -/
def num_arrangements : ℕ := 490314

/-- The theorem stating the remainder when the number of arrangements is divided by 1000 -/
theorem arrangements_remainder :
  num_arrangements % 1000 = 314 := by
  sorry

end arrangements_remainder_l1360_136041


namespace fibonacci_lucas_power_relation_l1360_136088

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 2
  | 1 => 1
  | (n + 2) => lucas (n + 1) + lucas n

/-- Main theorem -/
theorem fibonacci_lucas_power_relation (n p : ℕ) :
  (((lucas n : ℝ) + Real.sqrt 5 * (fib n : ℝ)) / 2) ^ p =
  ((lucas (n * p) : ℝ) + Real.sqrt 5 * (fib (n * p) : ℝ)) / 2 := by
  sorry

end fibonacci_lucas_power_relation_l1360_136088


namespace cylinder_in_sphere_volume_l1360_136026

theorem cylinder_in_sphere_volume (r h R : ℝ) (hr : r = 4) (hR : R = 7) 
  (hh : h^2 = 180) : 
  (4/3 * π * R^3 - π * r^2 * h) = (728/3) * π := by
  sorry

end cylinder_in_sphere_volume_l1360_136026


namespace local_min_implies_a_eq_4_l1360_136063

/-- The function f(x) = x^3 - ax^2 + 4x - 8 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2 + 4*x - 8

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*a*x + 4

/-- Theorem: If f(x) has a local minimum at x = 2, then a = 4 -/
theorem local_min_implies_a_eq_4 (a : ℝ) :
  (∃ δ > 0, ∀ x, |x - 2| < δ → f a x ≥ f a 2) →
  f_deriv a 2 = 0 →
  a = 4 := by sorry

end local_min_implies_a_eq_4_l1360_136063


namespace square_difference_symmetry_l1360_136071

theorem square_difference_symmetry (x y : ℝ) : (x - y)^2 = (y - x)^2 := by
  sorry

end square_difference_symmetry_l1360_136071


namespace quadratic_function_properties_l1360_136098

/-- A quadratic function with positive leading coefficient -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0

/-- The roots of f(x) - x = 0 for a quadratic function f -/
structure QuadraticRoots (f : QuadraticFunction) where
  x₁ : ℝ
  x₂ : ℝ
  root_order : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 / f.a

theorem quadratic_function_properties (f : QuadraticFunction) (roots : QuadraticRoots f) :
  (∀ x, 0 < x ∧ x < roots.x₁ → x < f.a * x^2 + f.b * x + f.c ∧ f.a * x^2 + f.b * x + f.c < roots.x₁) ∧
  roots.x₁ < roots.x₂ / 2 := by
  sorry

end quadratic_function_properties_l1360_136098


namespace difference_15x_x_squared_l1360_136060

theorem difference_15x_x_squared (x : ℕ) (h : x = 8) : 15 * x - x^2 = 56 := by
  sorry

end difference_15x_x_squared_l1360_136060


namespace sqrt_300_approximation_l1360_136017

theorem sqrt_300_approximation (ε δ : ℝ) (ε_pos : ε > 0) (δ_pos : δ > 0) 
  (h : |Real.sqrt 3 - 1.732| < δ) : 
  |Real.sqrt 300 - 17.32| < ε := by
  sorry

end sqrt_300_approximation_l1360_136017


namespace two_integers_sum_l1360_136028

theorem two_integers_sum (a b : ℕ+) : 
  a * b + a + b = 103 →
  Nat.gcd a b = 1 →
  a < 20 →
  b < 20 →
  a + b = 19 := by
sorry

end two_integers_sum_l1360_136028


namespace triangle_inequality_possible_third_side_l1360_136029

theorem triangle_inequality (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 → a + b > c → b + c > a → c + a > b → 
  ∃ (triangle : Set (ℝ × ℝ)), true := by sorry

theorem possible_third_side : ∃ (triangle : Set (ℝ × ℝ)), 
  (∃ (a b c : ℝ), a = 3 ∧ b = 7 ∧ c = 9 ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  a + b > c ∧ b + c > a ∧ c + a > b) := by sorry

end triangle_inequality_possible_third_side_l1360_136029


namespace tv_production_average_l1360_136011

/-- Proves that given the average production of 60 TVs/day for the first 25 days 
    of a 30-day month, and an overall monthly average of 58 TVs/day, 
    the average production for the last 5 days of the month is 48 TVs/day. -/
theorem tv_production_average (first_25_avg : ℕ) (total_days : ℕ) (monthly_avg : ℕ) :
  first_25_avg = 60 →
  total_days = 30 →
  monthly_avg = 58 →
  (monthly_avg * total_days - first_25_avg * 25) / 5 = 48 := by
  sorry

end tv_production_average_l1360_136011


namespace sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1360_136019

/-- The largest odd number less than 100 -/
def largest_odd_two_digit : ℕ :=
  99

/-- The smallest even number greater than or equal to 100 -/
def smallest_even_three_digit : ℕ :=
  100

/-- Theorem stating the sum of the largest odd two-digit number
    and the smallest even three-digit number -/
theorem sum_special_numbers :
  largest_odd_two_digit + smallest_even_three_digit = 199 := by
  sorry

/-- Proof that largest_odd_two_digit is indeed the largest odd number less than 100 -/
theorem largest_odd_two_digit_correct :
  largest_odd_two_digit < 100 ∧
  largest_odd_two_digit % 2 = 1 ∧
  ∀ n : ℕ, n < 100 → n % 2 = 1 → n ≤ largest_odd_two_digit := by
  sorry

/-- Proof that smallest_even_three_digit is indeed the smallest even number ≥ 100 -/
theorem smallest_even_three_digit_correct :
  smallest_even_three_digit ≥ 100 ∧
  smallest_even_three_digit % 2 = 0 ∧
  ∀ n : ℕ, n ≥ 100 → n % 2 = 0 → n ≥ smallest_even_three_digit := by
  sorry

end sum_special_numbers_largest_odd_two_digit_correct_smallest_even_three_digit_correct_l1360_136019


namespace cos_cube_decomposition_sum_of_squares_l1360_136037

open Real

theorem cos_cube_decomposition_sum_of_squares :
  (∃ b₁ b₂ b₃ : ℝ, ∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) →
  (∃ b₁ b₂ b₃ : ℝ, 
    (∀ θ : ℝ, cos θ ^ 3 = b₁ * cos θ + b₂ * cos (2 * θ) + b₃ * cos (3 * θ)) ∧
    b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 = 5 / 8) :=
by sorry

end cos_cube_decomposition_sum_of_squares_l1360_136037


namespace average_salary_non_officers_l1360_136061

/-- Prove that the average salary of non-officers is 110 Rs/month -/
theorem average_salary_non_officers (
  total_avg : ℝ) (officer_avg : ℝ) (num_officers : ℕ) (num_non_officers : ℕ)
  (h1 : total_avg = 120)
  (h2 : officer_avg = 420)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450)
  : (((total_avg * (num_officers + num_non_officers : ℝ)) - 
     (officer_avg * num_officers)) / num_non_officers) = 110 := by
  sorry

end average_salary_non_officers_l1360_136061


namespace intersection_M_N_l1360_136078

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end intersection_M_N_l1360_136078


namespace ellen_dough_balls_l1360_136049

/-- Represents the time it takes for a ball of dough to rise -/
def rise_time : ℕ := 3

/-- Represents the time it takes to bake a ball of dough -/
def bake_time : ℕ := 2

/-- Represents the total time for the entire baking process -/
def total_time : ℕ := 20

/-- Calculates the total time taken for a given number of dough balls -/
def time_for_n_balls (n : ℕ) : ℕ :=
  rise_time + bake_time + (n - 1) * rise_time

/-- The theorem stating the number of dough balls Ellen makes -/
theorem ellen_dough_balls :
  ∃ n : ℕ, n > 0 ∧ time_for_n_balls n = total_time ∧ n = 6 := by
  sorry

end ellen_dough_balls_l1360_136049


namespace max_marks_calculation_l1360_136097

theorem max_marks_calculation (passing_threshold : ℚ) (scored_marks : ℕ) (short_marks : ℕ) :
  passing_threshold = 30 / 100 →
  scored_marks = 212 →
  short_marks = 13 →
  ∃ max_marks : ℕ,
    max_marks = 750 ∧
    (scored_marks + short_marks : ℚ) / max_marks = passing_threshold :=
by sorry

end max_marks_calculation_l1360_136097


namespace triangle_angle_cosine_l1360_136080

theorem triangle_angle_cosine (A B C : Real) : 
  A + B + C = Real.pi →  -- Sum of angles in a triangle is π radians
  A + C = 2 * B →
  1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B →
  Real.cos ((A - C) / 2) = -Real.sqrt 2 / 2 := by
  sorry

end triangle_angle_cosine_l1360_136080


namespace circle_symmetry_ab_range_l1360_136074

/-- Given a circle x^2 + y^2 - 4x + 2y + 1 = 0 symmetric about the line ax - 2by - 1 = 0 (a, b ∈ ℝ),
    the range of ab is (-∞, 1/16]. -/
theorem circle_symmetry_ab_range (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ 
      a*x - 2*b*y - 1 = a*x' - 2*b*y' - 1 ∧ 
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2)) →
  a * b ≤ 1/16 := by
sorry

end circle_symmetry_ab_range_l1360_136074


namespace trapezoidal_sequence_624_l1360_136084

/-- The trapezoidal sequence -/
def trapezoidal_sequence : ℕ → ℕ
| 0 => 5
| n + 1 => trapezoidal_sequence n + (n + 4)

/-- The 624th term of the trapezoidal sequence is 196250 -/
theorem trapezoidal_sequence_624 : trapezoidal_sequence 623 = 196250 := by
  sorry

end trapezoidal_sequence_624_l1360_136084


namespace star_example_l1360_136093

-- Define the ⋆ operation
def star (a b c d : ℚ) : ℚ := a * c * (d / (2 * b))

-- Theorem statement
theorem star_example : star 5 6 9 4 = 15 := by
  sorry

end star_example_l1360_136093


namespace candies_left_l1360_136081

def initial_candies : ℕ := 88
def candies_taken : ℕ := 6

theorem candies_left : initial_candies - candies_taken = 82 := by
  sorry

end candies_left_l1360_136081


namespace number_of_valid_paths_l1360_136000

-- Define the grid dimensions
def rows : Nat := 4
def columns : Nat := 10

-- Define the total number of moves
def total_moves : Nat := rows + columns - 2

-- Define the number of unrestricted paths
def unrestricted_paths : Nat := Nat.choose total_moves (rows - 1)

-- Define the number of paths through the first forbidden segment
def forbidden_paths1 : Nat := 360

-- Define the number of paths through the second forbidden segment
def forbidden_paths2 : Nat := 420

-- Theorem statement
theorem number_of_valid_paths :
  unrestricted_paths - forbidden_paths1 - forbidden_paths2 = 221 := by
  sorry

end number_of_valid_paths_l1360_136000


namespace log_cube_of_nine_l1360_136039

-- Define a tolerance for approximation
def tolerance : ℝ := 0.000000000000002

-- Define the approximate equality
def approx_equal (a b : ℝ) : Prop := abs (a - b) < tolerance

theorem log_cube_of_nine (x y : ℝ) :
  approx_equal x 9 → (Real.log x^3 / Real.log 9 = y) → y = 3 := by
  sorry

end log_cube_of_nine_l1360_136039


namespace stratified_sample_size_l1360_136002

/-- Calculates the total sample size for a stratified sampling method given workshop productions and a known sample from one workshop. -/
theorem stratified_sample_size 
  (production_A production_B production_C : ℕ) 
  (sample_C : ℕ) : 
  production_A = 120 → 
  production_B = 80 → 
  production_C = 60 → 
  sample_C = 3 → 
  (production_A + production_B + production_C) * sample_C / production_C = 13 := by
  sorry

end stratified_sample_size_l1360_136002


namespace shooting_competition_probabilities_l1360_136014

theorem shooting_competition_probabilities 
  (p_A_not_losing : ℝ) 
  (p_B_losing : ℝ) 
  (h1 : p_A_not_losing = 0.59) 
  (h2 : p_B_losing = 0.44) : 
  ∃ (p_A_not_winning p_A_B_drawing : ℝ),
    p_A_not_winning = 0.56 ∧ 
    p_A_B_drawing = 0.15 := by
  sorry

end shooting_competition_probabilities_l1360_136014


namespace equation_solution_l1360_136055

theorem equation_solution :
  let f (n : ℚ) := (2 - n) / (n + 1) + (2 * n - 4) / (2 - n)
  ∃ (n : ℚ), f n = 1 ∧ n = -1/4 :=
by sorry

end equation_solution_l1360_136055


namespace four_touching_circles_l1360_136054

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Returns true if the circle touches the line -/
def touches (c : Circle) (l : Line) : Prop :=
  sorry

/-- The main theorem stating that there are exactly 4 circles of a given radius
    touching two given lines -/
theorem four_touching_circles 
  (r : ℝ) 
  (l₁ l₂ : Line) 
  (h_r : r > 0) 
  (h_distinct : l₁ ≠ l₂) : 
  ∃! (s : Finset Circle), 
    s.card = 4 ∧ 
    ∀ c ∈ s, c.radius = r ∧ touches c l₁ ∧ touches c l₂ :=
sorry

end four_touching_circles_l1360_136054


namespace doghouse_area_doghouse_area_value_l1360_136005

/-- The area outside a regular hexagon that can be reached by a tethered point -/
theorem doghouse_area (side_length : Real) (rope_length : Real) 
  (h1 : side_length = 2)
  (h2 : rope_length = 3) : 
  Real := by
  sorry

#check doghouse_area

theorem doghouse_area_value : 
  doghouse_area 2 3 rfl rfl = (22 / 3) * Real.pi := by
  sorry

end doghouse_area_doghouse_area_value_l1360_136005


namespace l_shaped_area_l1360_136048

/-- The area of an L-shaped region formed by subtracting three squares from a larger square -/
theorem l_shaped_area (outer_side : ℝ) (inner_side1 inner_side2 inner_side3 : ℝ) :
  outer_side = 6 ∧ 
  inner_side1 = 1 ∧ 
  inner_side2 = 2 ∧ 
  inner_side3 = 3 →
  outer_side ^ 2 - (inner_side1 ^ 2 + inner_side2 ^ 2 + inner_side3 ^ 2) = 22 :=
by sorry

end l_shaped_area_l1360_136048


namespace earth_inhabitable_fraction_l1360_136016

theorem earth_inhabitable_fraction :
  let water_free_fraction : ℚ := 1/4
  let inhabitable_land_fraction : ℚ := 1/3
  let inhabitable_fraction : ℚ := water_free_fraction * inhabitable_land_fraction
  inhabitable_fraction = 1/12 := by
sorry

end earth_inhabitable_fraction_l1360_136016


namespace f_shape_perimeter_l1360_136020

/-- The perimeter of a shape formed by two rectangles arranged in an F shape -/
def f_perimeter (h1 w1 h2 w2 overlap_h overlap_w : ℝ) : ℝ :=
  2 * (h1 + w1) + 2 * (h2 + w2) - 2 * overlap_w

/-- Theorem: The perimeter of the F shape is 18 inches -/
theorem f_shape_perimeter :
  f_perimeter 5 3 1 5 1 3 = 18 := by
  sorry

#eval f_perimeter 5 3 1 5 1 3

end f_shape_perimeter_l1360_136020


namespace calculation_proof_l1360_136091

theorem calculation_proof :
  (∃ x, x = Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1/2) * Real.sqrt 12 + Real.sqrt 24 ∧ x = 4 + Real.sqrt 6) ∧
  (∃ y, y = (Real.sqrt 20 + Real.sqrt 5) / Real.sqrt 5 - Real.sqrt 27 * Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) ∧ y = -4) :=
by sorry

end calculation_proof_l1360_136091


namespace relay_race_tables_l1360_136070

/-- The number of tables required for a relay race with given conditions -/
def num_tables (race_distance : ℕ) (distance_between_1_and_3 : ℕ) : ℕ :=
  (race_distance / (distance_between_1_and_3 / 2)) + 1

theorem relay_race_tables :
  num_tables 1200 400 = 7 :=
by sorry

end relay_race_tables_l1360_136070


namespace triangle_abc_properties_l1360_136031

theorem triangle_abc_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  b * c * Real.cos A = 4 →
  a * c * Real.sin B = 8 * Real.sin A →
  A = π / 3 ∧ 0 < Real.sin A * Real.sin B * Real.sin C ∧ 
  Real.sin A * Real.sin B * Real.sin C ≤ 3 * Real.sqrt 3 / 8 := by
sorry

end triangle_abc_properties_l1360_136031


namespace inequality_solution_l1360_136007

theorem inequality_solution (x : ℝ) : 
  (2 / (x + 2) + 5 / (x + 4) ≥ 1) ↔ (x < -4 ∨ x ≥ 5) := by sorry

end inequality_solution_l1360_136007


namespace least_subtraction_for_divisibility_problem_solution_l1360_136003

theorem least_subtraction_for_divisibility (n m : ℕ) : 
  ∃ k, k ≤ m ∧ (n - k) % m = 0 ∧ ∀ j, j < k → (n - j) % m ≠ 0 :=
by sorry

theorem problem_solution : 
  ∃ k, k ≤ 87 ∧ (13604 - k) % 87 = 0 ∧ ∀ j, j < k → (13604 - j) % 87 ≠ 0 ∧ k = 32 :=
by sorry

end least_subtraction_for_divisibility_problem_solution_l1360_136003


namespace v_closed_under_multiplication_l1360_136025

def v : Set ℕ := {n : ℕ | ∃ m : ℕ, m > 0 ∧ n = m^3}

theorem v_closed_under_multiplication :
  ∀ a b : ℕ, a ∈ v → b ∈ v → (a * b) ∈ v :=
by sorry

end v_closed_under_multiplication_l1360_136025


namespace cuboid_face_area_l1360_136083

theorem cuboid_face_area (small_face_area : ℝ) 
  (h1 : small_face_area > 0)
  (h2 : ∃ (large_face_area : ℝ), large_face_area = 4 * small_face_area)
  (h3 : 2 * small_face_area + 4 * (4 * small_face_area) = 72) :
  ∃ (large_face_area : ℝ), large_face_area = 16 := by
sorry

end cuboid_face_area_l1360_136083


namespace f_properties_l1360_136096

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

def M : Set ℝ := {x : ℝ | f x ≥ 3}

theorem f_properties :
  (M = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2}) ∧
  (∀ a ∈ M, ∀ x : ℝ, |x + a| + |x - 1/a| ≥ 5/2) := by sorry

end f_properties_l1360_136096


namespace residue_of_9_pow_2010_mod_17_l1360_136023

theorem residue_of_9_pow_2010_mod_17 : 9^2010 % 17 = 13 := by
  sorry

end residue_of_9_pow_2010_mod_17_l1360_136023


namespace marcia_blouses_l1360_136008

/-- Calculates the number of blouses Marcia can add to her wardrobe given the following conditions:
  * Marcia needs 3 skirts, 2 pairs of pants, and some blouses
  * Skirts cost $20.00 each
  * Blouses cost $15.00 each
  * Pants cost $30.00 each
  * There's a sale on pants: buy 1 pair get 1 pair 1/2 off
  * Total budget is $180.00
-/
def calculate_blouses (skirt_count : Nat) (skirt_price : Nat) (pant_count : Nat) (pant_price : Nat) (blouse_price : Nat) (total_budget : Nat) : Nat :=
  let skirt_total := skirt_count * skirt_price
  let pant_total := pant_price + (pant_price / 2)
  let remaining_budget := total_budget - skirt_total - pant_total
  remaining_budget / blouse_price

theorem marcia_blouses :
  calculate_blouses 3 20 2 30 15 180 = 5 := by
  sorry

end marcia_blouses_l1360_136008


namespace equation_solution_l1360_136004

theorem equation_solution (x : ℝ) : (10 - x)^2 = 4 * x^2 ↔ x = 10/3 ∨ x = -10 := by
  sorry

end equation_solution_l1360_136004


namespace unique_solution_quadratic_equation_l1360_136036

theorem unique_solution_quadratic_equation :
  ∃! (x y : ℝ), (4 * x^2 + 6 * x + 4) * (4 * y^2 - 12 * y + 25) = 28 ∧
                x = -3/4 ∧ y = 3/2 := by
  sorry

end unique_solution_quadratic_equation_l1360_136036


namespace farm_tax_total_l1360_136021

/-- Represents the farm tax collected from a village -/
structure FarmTax where
  /-- Total amount collected from the village -/
  total : ℝ
  /-- Amount paid by Mr. William -/
  william_paid : ℝ
  /-- Percentage of total taxable land owned by Mr. William -/
  william_percentage : ℝ
  /-- Assertion that Mr. William's percentage is 50% -/
  h_percentage : william_percentage = 50
  /-- Assertion that Mr. William paid $480 -/
  h_william_paid : william_paid = 480
  /-- The total tax is twice what Mr. William paid -/
  h_total : total = 2 * william_paid

/-- Theorem stating that the total farm tax collected is $960 -/
theorem farm_tax_total (ft : FarmTax) : ft.total = 960 := by
  sorry

end farm_tax_total_l1360_136021


namespace systematic_sample_theorem_l1360_136077

/-- Represents a systematic sample from a population --/
structure SystematicSample where
  population_size : Nat
  sample_size : Nat
  first_element : Nat
  interval : Nat

/-- Checks if a number is in the systematic sample --/
def SystematicSample.contains (s : SystematicSample) (n : Nat) : Prop :=
  ∃ k : Nat, k < s.sample_size ∧ n = s.first_element + k * s.interval

/-- The main theorem --/
theorem systematic_sample_theorem (sample : SystematicSample)
    (h_pop : sample.population_size = 56)
    (h_size : sample.sample_size = 4)
    (h_first : sample.first_element = 6)
    (h_contains_34 : sample.contains 34)
    (h_contains_48 : sample.contains 48) :
    sample.contains 20 :=
  sorry

end systematic_sample_theorem_l1360_136077


namespace monthly_fee_calculation_l1360_136043

/-- Represents the long distance phone service billing structure and usage -/
structure PhoneBill where
  monthlyFee : ℝ
  ratePerMinute : ℝ
  minutesUsed : ℕ
  totalBill : ℝ

/-- Theorem stating that given the specific conditions, the monthly fee is $2.00 -/
theorem monthly_fee_calculation (bill : PhoneBill) 
    (h1 : bill.ratePerMinute = 0.12)
    (h2 : bill.minutesUsed = 178)
    (h3 : bill.totalBill = 23.36) :
    bill.monthlyFee = 2.00 := by
  sorry

end monthly_fee_calculation_l1360_136043


namespace equal_revenue_for_all_sellers_l1360_136010

/-- Represents an apple seller with their apple count -/
structure AppleSeller :=
  (apples : ℕ)

/-- Calculates the revenue for an apple seller given the pricing scheme -/
def revenue (seller : AppleSeller) : ℕ :=
  let batches := seller.apples / 7
  let leftovers := seller.apples % 7
  batches + 3 * leftovers

/-- The list of apple sellers with their respective apple counts -/
def sellers : List AppleSeller :=
  [⟨20⟩, ⟨40⟩, ⟨60⟩, ⟨80⟩, ⟨100⟩, ⟨120⟩, ⟨140⟩]

theorem equal_revenue_for_all_sellers :
  ∀ s ∈ sellers, revenue s = 20 := by
  sorry

end equal_revenue_for_all_sellers_l1360_136010


namespace nested_radical_fifteen_l1360_136012

theorem nested_radical_fifteen (x : ℝ) : x = Real.sqrt (15 + x) → x = (1 + Real.sqrt 61) / 2 := by
  sorry

end nested_radical_fifteen_l1360_136012


namespace train_clicks_theorem_l1360_136066

/-- Represents the number of clicks heard in 30 seconds for a train accelerating from 30 to 60 mph over 5 miles --/
def train_clicks : ℕ := 40

/-- Rail length in feet --/
def rail_length : ℝ := 50

/-- Initial speed in miles per hour --/
def initial_speed : ℝ := 30

/-- Final speed in miles per hour --/
def final_speed : ℝ := 60

/-- Acceleration distance in miles --/
def acceleration_distance : ℝ := 5

/-- Time period in seconds --/
def time_period : ℝ := 30

/-- Theorem stating that the number of clicks heard in 30 seconds is approximately 40 --/
theorem train_clicks_theorem : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |train_clicks - (((initial_speed + final_speed) / 2 * 5280 / 60) / rail_length * (time_period / 60))| < ε :=
sorry

end train_clicks_theorem_l1360_136066


namespace boat_speed_l1360_136059

/-- The average speed of a boat in still water, given its travel times with and against a current. -/
theorem boat_speed (time_with_current time_against_current current_speed : ℝ) 
  (h1 : time_with_current = 2)
  (h2 : time_against_current = 2.5)
  (h3 : current_speed = 3)
  (h4 : time_with_current * (x + current_speed) = time_against_current * (x - current_speed)) : 
  x = 27 :=
by
  sorry

#check boat_speed

end boat_speed_l1360_136059


namespace lindas_savings_l1360_136034

theorem lindas_savings (savings : ℕ) : 
  (3 : ℚ) / 4 * savings + 250 = savings → savings = 1000 := by
  sorry

end lindas_savings_l1360_136034


namespace total_moving_time_l1360_136056

/-- The time (in minutes) spent filling the car for each trip. -/
def fill_time : ℕ := 15

/-- The time (in minutes) spent driving one-way for each trip. -/
def drive_time : ℕ := 30

/-- The total number of trips made. -/
def num_trips : ℕ := 6

/-- The total time spent moving, in hours. -/
def total_time : ℚ := (fill_time + drive_time) * num_trips / 60

/-- Proves that the total time spent moving is 4.5 hours. -/
theorem total_moving_time : total_time = 4.5 := by
  sorry

end total_moving_time_l1360_136056


namespace max_value_sqrt_sum_l1360_136062

theorem max_value_sqrt_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 3) :
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 →
    Real.sqrt (2 * a + 1) + Real.sqrt (2 * b + 1) + Real.sqrt (2 * c + 1) ≤
    Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1)) →
  Real.sqrt (2 * x + 1) + Real.sqrt (2 * y + 1) + Real.sqrt (2 * z + 1) = 3 * Real.sqrt 3 :=
by sorry

end max_value_sqrt_sum_l1360_136062


namespace man_speed_man_speed_specific_l1360_136072

/-- Calculates the speed of a man running opposite to a train, given the train's length, speed, and time to pass the man. -/
theorem man_speed (train_length : Real) (train_speed_kmh : Real) (pass_time : Real) : Real :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / pass_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man given specific values -/
theorem man_speed_specific : 
  man_speed 150 83.99280057595394 6 = 6.007199827245052 := by
  sorry

end man_speed_man_speed_specific_l1360_136072


namespace high_school_population_l1360_136068

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  grade10_students : ℕ
  total_sample : ℕ
  grade11_sample : ℕ
  grade12_sample : ℕ

/-- Calculates the total number of students in the high school based on stratified sampling. -/
def total_students (hs : HighSchool) : ℕ :=
  let grade10_sample := hs.total_sample - hs.grade11_sample - hs.grade12_sample
  (hs.grade10_students * hs.total_sample) / grade10_sample

/-- Theorem stating that given the specific conditions, the total number of students is 1800. -/
theorem high_school_population (hs : HighSchool)
  (h1 : hs.grade10_students = 600)
  (h2 : hs.total_sample = 45)
  (h3 : hs.grade11_sample = 20)
  (h4 : hs.grade12_sample = 10) :
  total_students hs = 1800 := by
  sorry

#eval total_students { grade10_students := 600, total_sample := 45, grade11_sample := 20, grade12_sample := 10 }

end high_school_population_l1360_136068


namespace trip_duration_proof_l1360_136024

/-- Calculates the total time spent on a trip visiting three countries. -/
def total_trip_time (first_country_stay : ℕ) : ℕ :=
  first_country_stay + 2 * first_country_stay * 2

/-- Proves that the total trip time is 10 weeks given the specified conditions. -/
theorem trip_duration_proof :
  let first_country_stay := 2
  total_trip_time first_country_stay = 10 := by
  sorry

#eval total_trip_time 2

end trip_duration_proof_l1360_136024


namespace intersection_point_count_l1360_136006

theorem intersection_point_count :
  ∃! p : ℝ × ℝ, 
    (p.1 + p.2 - 5) * (2 * p.1 - 3 * p.2 + 5) = 0 ∧ 
    (p.1 - p.2 + 1) * (3 * p.1 + 2 * p.2 - 12) = 0 :=
by sorry

end intersection_point_count_l1360_136006


namespace courtyard_area_difference_l1360_136079

/-- The difference in area between a circular courtyard and a rectangular courtyard -/
theorem courtyard_area_difference :
  let rect_length : ℝ := 60
  let rect_width : ℝ := 20
  let rect_perimeter : ℝ := 2 * (rect_length + rect_width)
  let rect_area : ℝ := rect_length * rect_width
  let circle_radius : ℝ := rect_perimeter / (2 * Real.pi)
  let circle_area : ℝ := Real.pi * circle_radius ^ 2
  circle_area - rect_area = (6400 - 1200 * Real.pi) / Real.pi := by
  sorry

end courtyard_area_difference_l1360_136079


namespace complement_intersection_theorem_l1360_136069

universe u

def U : Set (Fin 5) := {0, 1, 2, 3, 4}
def M : Set (Fin 5) := {0, 2, 3}
def N : Set (Fin 5) := {1, 3, 4}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {1, 4} := by sorry

end complement_intersection_theorem_l1360_136069


namespace equation_solution_l1360_136053

theorem equation_solution (y : ℝ) : 
  ∃ z : ℝ, 19 * (1 + y) + z = 19 * (-1 + y) - 21 ∧ z = -59 := by
  sorry

end equation_solution_l1360_136053


namespace mariela_get_well_cards_l1360_136086

theorem mariela_get_well_cards (cards_in_hospital : ℕ) (cards_at_home : ℕ) 
  (h1 : cards_in_hospital = 403)
  (h2 : cards_at_home = 287) :
  cards_in_hospital + cards_at_home = 690 := by
  sorry

end mariela_get_well_cards_l1360_136086


namespace flour_calculation_l1360_136022

/-- The number of cups of flour Mary has already put in -/
def flour_already_added : ℕ := sorry

/-- The total number of cups of flour required by the recipe -/
def total_flour_required : ℕ := 10

/-- The number of cups of flour Mary still needs to add -/
def flour_to_be_added : ℕ := 4

/-- Theorem: The number of cups of flour Mary has already put in is equal to
    the difference between the total cups of flour required and the cups of flour
    she still needs to add -/
theorem flour_calculation :
  flour_already_added = total_flour_required - flour_to_be_added :=
sorry

end flour_calculation_l1360_136022


namespace sufficient_condition_for_inequality_l1360_136057

theorem sufficient_condition_for_inequality (a : ℝ) :
  a ≥ 5 → ∀ x ∈ Set.Icc 1 2, x^2 - a ≤ 0 := by
  sorry

end sufficient_condition_for_inequality_l1360_136057


namespace distance_between_trees_441_22_l1360_136073

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 441-metre yard with 22 trees is 21 metres -/
theorem distance_between_trees_441_22 :
  distance_between_trees 441 22 = 21 := by
  sorry

end distance_between_trees_441_22_l1360_136073


namespace quadratic_is_square_of_binomial_l1360_136095

theorem quadratic_is_square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 + 80*x + d = (x + a)^2 + b^2) → d = 1600 := by
  sorry

end quadratic_is_square_of_binomial_l1360_136095


namespace probability_heart_then_club_is_13_204_l1360_136065

/-- A standard deck of cards -/
structure Deck :=
  (cards : Finset (Fin 52))
  (card_count : cards.card = 52)

/-- Suits in a deck -/
inductive Suit
| Hearts
| Clubs
| Diamonds
| Spades

/-- A card in the deck -/
structure Card :=
  (number : Fin 13)
  (suit : Suit)

/-- The probability of drawing a heart first and a club second from a standard deck -/
def probability_heart_then_club (d : Deck) : ℚ :=
  13 / 204

/-- Theorem: The probability of drawing a heart first and a club second from a standard deck is 13/204 -/
theorem probability_heart_then_club_is_13_204 (d : Deck) :
  probability_heart_then_club d = 13 / 204 := by
  sorry

end probability_heart_then_club_is_13_204_l1360_136065


namespace swap_digits_l1360_136087

theorem swap_digits (x : ℕ) (h : 9 < x ∧ x < 100) : 
  (x % 10) * 10 + (x / 10) = 10 * (x % 10) + (x / 10) := by sorry

end swap_digits_l1360_136087


namespace set_operation_proof_l1360_136092

theorem set_operation_proof (A B C : Set ℕ) : 
  A = {1, 2} → B = {1, 2, 3} → C = {2, 3, 4} → 
  (A ∩ B) ∪ C = {1, 2, 3, 4} := by
  sorry

end set_operation_proof_l1360_136092


namespace tom_read_18_books_l1360_136089

/-- The number of books Tom read in May -/
def may_books : ℕ := 2

/-- The number of books Tom read in June -/
def june_books : ℕ := 6

/-- The number of books Tom read in July -/
def july_books : ℕ := 10

/-- The total number of books Tom read -/
def total_books : ℕ := may_books + june_books + july_books

theorem tom_read_18_books : total_books = 18 := by
  sorry

end tom_read_18_books_l1360_136089
