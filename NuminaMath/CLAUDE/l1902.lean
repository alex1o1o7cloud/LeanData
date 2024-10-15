import Mathlib

namespace NUMINAMATH_CALUDE_power_expressions_l1902_190200

theorem power_expressions (m n : ℤ) (a b : ℝ) 
  (h1 : 4^m = a) (h2 : 8^n = b) : 
  (2^(2*m + 3*n) = a * b) ∧ (2^(4*m - 6*n) = a^2 / b^2) := by
  sorry

end NUMINAMATH_CALUDE_power_expressions_l1902_190200


namespace NUMINAMATH_CALUDE_smaller_acute_angle_measure_l1902_190201

-- Define a right triangle with acute angles x and 4x
def right_triangle (x : ℝ) : Prop :=
  x > 0 ∧ x < 90 ∧ x + 4*x = 90

-- Theorem statement
theorem smaller_acute_angle_measure :
  ∃ (x : ℝ), right_triangle x ∧ x = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_smaller_acute_angle_measure_l1902_190201


namespace NUMINAMATH_CALUDE_jordans_rectangle_width_l1902_190247

theorem jordans_rectangle_width (carol_length carol_width jordan_length : ℕ) 
  (jordan_width : ℕ) : 
  carol_length = 12 → 
  carol_width = 15 → 
  jordan_length = 9 → 
  carol_length * carol_width = jordan_length * jordan_width → 
  jordan_width = 20 := by
sorry

end NUMINAMATH_CALUDE_jordans_rectangle_width_l1902_190247


namespace NUMINAMATH_CALUDE_noahs_lights_l1902_190290

theorem noahs_lights (W : ℝ) 
  (h1 : W > 0)  -- Assuming W is positive
  (h2 : 2 * W + 2 * (3 * W) + 2 * (4 * W) = 96) : W = 6 := by
  sorry

end NUMINAMATH_CALUDE_noahs_lights_l1902_190290


namespace NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1902_190281

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) ≤ 2 / (1 + a*b) :=
by sorry

theorem equality_condition (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hab : a + b < 2) :
  (1 / (1 + a^2)) + (1 / (1 + b^2)) = 2 / (1 + a*b) ↔ a = b :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_equality_condition_l1902_190281


namespace NUMINAMATH_CALUDE_dance_steps_l1902_190297

theorem dance_steps (nancy_ratio : ℕ) (total_steps : ℕ) (jason_steps : ℕ) : 
  nancy_ratio = 3 →
  total_steps = 32 →
  jason_steps + nancy_ratio * jason_steps = total_steps →
  jason_steps = 8 := by
sorry

end NUMINAMATH_CALUDE_dance_steps_l1902_190297


namespace NUMINAMATH_CALUDE_min_c_value_l1902_190293

theorem min_c_value (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_order : a < b ∧ b < c)
  (h_unique : ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1012 ∧ ∃ (a' b' : ℕ), 0 < a' ∧ 0 < b' ∧ a' < b' ∧ b' < 1012 ∧
    ∃! (x y : ℝ), 2 * x + y = 2023 ∧ y = |x - a'| + |x - b'| + |x - 1012| :=
by sorry

end NUMINAMATH_CALUDE_min_c_value_l1902_190293


namespace NUMINAMATH_CALUDE_min_f_value_l1902_190233

theorem min_f_value (d e f : ℕ+) (h1 : d < e) (h2 : e < f)
  (h3 : ∃! x y : ℝ, 3 * x + y = 3005 ∧ y = |x - d| + |x - e| + |x - f|) :
  1504 ≤ f :=
sorry

end NUMINAMATH_CALUDE_min_f_value_l1902_190233


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1902_190231

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 4 + a 9 + a 11 = 32) →
  (a 6 + a 7 = 16) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l1902_190231


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l1902_190282

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 14 = 31) → 
  (d^2 - 6*d + 14 = 31) → 
  c ≥ d → 
  c + 2*d = 9 - Real.sqrt 26 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l1902_190282


namespace NUMINAMATH_CALUDE_mean_home_runs_l1902_190262

def total_players : ℕ := 6 + 4 + 3 + 1 + 1 + 1

def total_home_runs : ℕ := 6 * 6 + 7 * 4 + 8 * 3 + 10 * 1 + 11 * 1 + 12 * 1

theorem mean_home_runs : 
  (total_home_runs : ℚ) / (total_players : ℚ) = 7.5625 := by sorry

end NUMINAMATH_CALUDE_mean_home_runs_l1902_190262


namespace NUMINAMATH_CALUDE_incorrect_inequality_l1902_190271

theorem incorrect_inequality (a b : ℝ) (h : a > b) : ¬(5 - a > 5 - b) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_inequality_l1902_190271


namespace NUMINAMATH_CALUDE_value_of_expression_l1902_190214

theorem value_of_expression (x : ℝ) (h : x = 4) : (3*x + 7)^2 = 361 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l1902_190214


namespace NUMINAMATH_CALUDE_amusement_park_elementary_students_l1902_190205

theorem amusement_park_elementary_students 
  (total_women : ℕ) 
  (women_elementary : ℕ) 
  (more_men : ℕ) 
  (men_not_elementary : ℕ) 
  (h1 : total_women = 1518)
  (h2 : women_elementary = 536)
  (h3 : more_men = 525)
  (h4 : men_not_elementary = 1257) :
  women_elementary + (total_women + more_men - men_not_elementary) = 1322 :=
by
  sorry

end NUMINAMATH_CALUDE_amusement_park_elementary_students_l1902_190205


namespace NUMINAMATH_CALUDE_permutations_of_three_letter_word_is_six_l1902_190217

/-- The number of permutations of a 3-letter word with distinct letters -/
def permutations_of_three_letter_word : ℕ :=
  Nat.factorial 3

/-- Proof that the number of permutations of a 3-letter word with distinct letters is 6 -/
theorem permutations_of_three_letter_word_is_six :
  permutations_of_three_letter_word = 6 := by
  sorry

#eval permutations_of_three_letter_word

end NUMINAMATH_CALUDE_permutations_of_three_letter_word_is_six_l1902_190217


namespace NUMINAMATH_CALUDE_stuffed_animals_problem_l1902_190265

theorem stuffed_animals_problem (num_dogs : ℕ) : 
  (∃ (group_size : ℕ), group_size > 0 ∧ (14 + num_dogs) = 7 * group_size) →
  num_dogs = 7 := by
sorry

end NUMINAMATH_CALUDE_stuffed_animals_problem_l1902_190265


namespace NUMINAMATH_CALUDE_train_length_calculation_l1902_190245

/-- Proves that the length of each train is 75 meters given the specified conditions. -/
theorem train_length_calculation (v1 v2 t : ℝ) (h1 : v1 = 46) (h2 : v2 = 36) (h3 : t = 54) :
  let relative_speed := (v1 - v2) * (5 / 18)
  let distance := relative_speed * t
  let train_length := distance / 2
  train_length = 75 := by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1902_190245


namespace NUMINAMATH_CALUDE_swallow_theorem_l1902_190257

/-- The number of swallows initially on the wire -/
def initial_swallows : ℕ := 9

/-- The distance between the first and last swallow in centimeters -/
def total_distance : ℕ := 720

/-- The number of additional swallows added between each pair -/
def additional_swallows : ℕ := 3

/-- Theorem stating the distance between neighboring swallows and the total number after adding more -/
theorem swallow_theorem :
  (let gaps := initial_swallows - 1
   let distance_between := total_distance / gaps
   let new_swallows := gaps * additional_swallows
   let total_swallows := initial_swallows + new_swallows
   (distance_between = 90 ∧ total_swallows = 33)) :=
by sorry

end NUMINAMATH_CALUDE_swallow_theorem_l1902_190257


namespace NUMINAMATH_CALUDE_two_numbers_difference_l1902_190220

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff : x^2 - y^2 = 24) : 
  |x - y| = 2.4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l1902_190220


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l1902_190260

/-- Calculates the remaining candy after Debby and her sister combine their Halloween candy and eat some. -/
def remaining_candy (debby_candy : ℕ) (sister_candy : ℕ) (eaten_candy : ℕ) : ℕ :=
  debby_candy + sister_candy - eaten_candy

/-- Theorem stating that the remaining candy is correct given the initial conditions. -/
theorem halloween_candy_theorem (debby_candy sister_candy eaten_candy : ℕ) :
  remaining_candy debby_candy sister_candy eaten_candy = debby_candy + sister_candy - eaten_candy :=
by sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l1902_190260


namespace NUMINAMATH_CALUDE_work_done_on_bullet_l1902_190287

theorem work_done_on_bullet (m : Real) (v1 v2 : Real) :
  m = 0.01 →
  v1 = 500 →
  v2 = 200 →
  let K1 := (1/2) * m * v1^2
  let K2 := (1/2) * m * v2^2
  K1 - K2 = 1050 := by sorry

end NUMINAMATH_CALUDE_work_done_on_bullet_l1902_190287


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1902_190280

/-- Given a hyperbola with equation x²/4 - y²/12 = 1, its real axis has length 4 -/
theorem hyperbola_real_axis_length :
  ∀ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1 → 
  ∃ (a : ℝ), a > 0 ∧ x^2 / a^2 - y^2 / (3*a^2) = 1 ∧ 2 * a = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1902_190280


namespace NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1902_190240

/-- The speed of a train crossing a bridge -/
theorem train_speed_crossing_bridge (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 135)
  (h2 : bridge_length = 240)
  (h3 : crossing_time = 30) :
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_crossing_bridge

end NUMINAMATH_CALUDE_train_speed_crossing_bridge_l1902_190240


namespace NUMINAMATH_CALUDE_binary_101_equals_5_l1902_190234

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 5 -/
def binary_five : List Bool := [true, false, true]

/-- Theorem stating that the binary representation [1,0,1] is equal to 5 in decimal -/
theorem binary_101_equals_5 : binary_to_decimal binary_five = 5 := by
  sorry

end NUMINAMATH_CALUDE_binary_101_equals_5_l1902_190234


namespace NUMINAMATH_CALUDE_remainder_theorem_l1902_190209

theorem remainder_theorem (P D D' D'' Q Q' Q'' R R' R'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : Q' = Q'' * D'' + R'')
  (h4 : R < D)
  (h5 : R' < D')
  (h6 : R'' < D'') :
  P % (D * D' * D'') = R'' * D * D' + R' * D + R := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1902_190209


namespace NUMINAMATH_CALUDE_abs_minus_self_nonneg_l1902_190211

theorem abs_minus_self_nonneg (m : ℚ) : |m| - m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_abs_minus_self_nonneg_l1902_190211


namespace NUMINAMATH_CALUDE_reflection_composition_l1902_190246

def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  let q := (p.1, p.2 + 1)
  let r := (q.2, q.1)
  (r.1, r.2 - 1)

theorem reflection_composition (H : ℝ × ℝ) :
  H = (5, 0) →
  reflect_y_eq_x_minus_1 (reflect_x H) = (1, 4) := by
  sorry

end NUMINAMATH_CALUDE_reflection_composition_l1902_190246


namespace NUMINAMATH_CALUDE_bart_mixtape_problem_l1902_190249

/-- A mixtape with two sides -/
structure Mixtape where
  first_side_songs : ℕ
  second_side_songs : ℕ
  song_length : ℕ
  total_length : ℕ

/-- The problem statement -/
theorem bart_mixtape_problem (m : Mixtape) 
  (h1 : m.second_side_songs = 4)
  (h2 : m.song_length = 4)
  (h3 : m.total_length = 40) :
  m.first_side_songs = 6 := by
  sorry


end NUMINAMATH_CALUDE_bart_mixtape_problem_l1902_190249


namespace NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l1902_190222

/-- The amount of money made per t-shirt sold -/
def profit_per_shirt : ℕ := 98

/-- The total number of t-shirts sold during both games -/
def total_shirts_sold : ℕ := 163

/-- The number of t-shirts sold during the Arkansas game -/
def arkansas_shirts_sold : ℕ := 89

/-- The money made from selling t-shirts during the Arkansas game -/
def arkansas_game_profit : ℕ := profit_per_shirt * arkansas_shirts_sold

theorem arkansas_game_profit_calculation :
  arkansas_game_profit = 8722 := by sorry

end NUMINAMATH_CALUDE_arkansas_game_profit_calculation_l1902_190222


namespace NUMINAMATH_CALUDE_max_min_values_of_f_l1902_190255

-- Define the function f(x)
def f (x : ℝ) : ℝ := 2 * x^3 + 9 * x^2 - 2

-- Define the interval
def interval : Set ℝ := {x | -4 ≤ x ∧ x ≤ 2}

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, f x = 50) ∧
  (∀ y ∈ interval, f y ≤ 50) ∧
  (∃ x ∈ interval, f x = -2) ∧
  (∀ y ∈ interval, f y ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_max_min_values_of_f_l1902_190255


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l1902_190275

/-- Given a hyperbola, its asymptote, a parabola, and a circle, prove the value of a parameter. -/
theorem hyperbola_asymptote_intersection (a b : ℝ) (h1 : a > 0) (h2 : b > 0) : 
  (∃ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →  -- Hyperbola equation
  (∃ x₀ : ℝ, y = 2*x₀*x - x₀^2 + 1) →      -- Asymptote equation (tangent to parabola)
  (∃ x y : ℝ, x^2 + (y - a)^2 = 1) →       -- Circle equation
  (∃ x₁ y₁ x₂ y₂ : ℝ,                      -- Chord endpoints
    x₁^2 + (y₁ - a)^2 = 1 ∧ 
    x₂^2 + (y₂ - a)^2 = 1 ∧ 
    y₁ = 2*x₀*x₁ - x₀^2 + 1 ∧ 
    y₂ = 2*x₀*x₂ - x₀^2 + 1 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 2) →       -- Chord length
  a = Real.sqrt 10 / 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_intersection_l1902_190275


namespace NUMINAMATH_CALUDE_darryl_break_even_l1902_190224

/-- Calculates the number of machines needed to break even given costs and selling price -/
def machines_to_break_even (parts_cost patent_cost selling_price : ℕ) : ℕ :=
  (parts_cost + patent_cost) / selling_price

/-- Theorem: Darryl needs to sell 45 machines to break even -/
theorem darryl_break_even :
  machines_to_break_even 3600 4500 180 = 45 := by
  sorry

end NUMINAMATH_CALUDE_darryl_break_even_l1902_190224


namespace NUMINAMATH_CALUDE_cyclist_minimum_speed_l1902_190269

/-- Minimum speed for a cyclist to intercept a car -/
theorem cyclist_minimum_speed (v a b : ℝ) (hv : v > 0) (ha : a > 0) (hb : b > 0) :
  let min_speed := v * b / a
  ∀ (cyclist_speed : ℝ), cyclist_speed ≥ min_speed → 
  ∃ (t : ℝ), t > 0 ∧ 
    cyclist_speed * t = (a^2 + (v*t)^2).sqrt ∧
    cyclist_speed * t ≥ v * t :=
by sorry

end NUMINAMATH_CALUDE_cyclist_minimum_speed_l1902_190269


namespace NUMINAMATH_CALUDE_average_weight_problem_l1902_190244

theorem average_weight_problem (rachel_weight jimmy_weight adam_weight : ℝ) :
  rachel_weight = 75 ∧
  rachel_weight = jimmy_weight - 6 ∧
  rachel_weight = adam_weight + 15 →
  (rachel_weight + jimmy_weight + adam_weight) / 3 = 72 :=
by sorry

end NUMINAMATH_CALUDE_average_weight_problem_l1902_190244


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1902_190202

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ a = (k * b.1, k * b.2)

/-- The theorem states that if vectors (4,2) and (x,3) are parallel, then x = 6 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel (4, 2) (x, 3) → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1902_190202


namespace NUMINAMATH_CALUDE_polynomial_properties_l1902_190285

def polynomial_expansion (x : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : Prop :=
  (2*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

theorem polynomial_properties 
  (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) 
  (h : ∀ x, polynomial_expansion x a₀ a₁ a₂ a₃ a₄ a₅) : 
  (a₀ + a₁ + a₂ + a₃ + a₄ = -31) ∧ 
  ((a₀ + a₂ + a₄)^2 - (a₁ + a₃ + a₅)^2 = -243) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_properties_l1902_190285


namespace NUMINAMATH_CALUDE_exists_surjective_function_with_property_l1902_190294

-- Define the property of the function
def has_property (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f (x + y) - f x - f y) ∈ ({0, 1} : Set ℝ)

-- State the theorem
theorem exists_surjective_function_with_property :
  ∃ f : ℝ → ℝ, Function.Surjective f ∧ has_property f :=
sorry

end NUMINAMATH_CALUDE_exists_surjective_function_with_property_l1902_190294


namespace NUMINAMATH_CALUDE_remainder_problem_l1902_190267

theorem remainder_problem (N : ℕ) (R : ℕ) :
  (∃ q : ℕ, N = 34 * q + 2) →
  (N = 44 * 432 + R) →
  R = 0 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1902_190267


namespace NUMINAMATH_CALUDE_sine_identity_l1902_190288

theorem sine_identity (α : Real) (h : α = π / 7) :
  1 / Real.sin α = 1 / Real.sin (2 * α) + 1 / Real.sin (3 * α) := by
  sorry

end NUMINAMATH_CALUDE_sine_identity_l1902_190288


namespace NUMINAMATH_CALUDE_red_grapes_count_l1902_190254

/-- Represents the number of fruits in a fruit salad. -/
structure FruitSalad where
  green_grapes : ℕ
  red_grapes : ℕ
  raspberries : ℕ

/-- Defines the conditions for a valid fruit salad. -/
def is_valid_fruit_salad (fs : FruitSalad) : Prop :=
  fs.red_grapes = 3 * fs.green_grapes + 7 ∧
  fs.raspberries = fs.green_grapes - 5 ∧
  fs.green_grapes + fs.red_grapes + fs.raspberries = 102

/-- Theorem stating that in a valid fruit salad, there are 67 red grapes. -/
theorem red_grapes_count (fs : FruitSalad) 
  (h : is_valid_fruit_salad fs) : fs.red_grapes = 67 := by
  sorry


end NUMINAMATH_CALUDE_red_grapes_count_l1902_190254


namespace NUMINAMATH_CALUDE_remainder_18273_mod_9_l1902_190272

theorem remainder_18273_mod_9 : 18273 % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_18273_mod_9_l1902_190272


namespace NUMINAMATH_CALUDE_old_selling_price_l1902_190274

/-- Given a product with cost C, prove that if the selling price increased from 110% of C to 115% of C, 
    and the new selling price is $92.00, then the old selling price was $88.00. -/
theorem old_selling_price (C : ℝ) 
  (h1 : C + 0.15 * C = 92) 
  (h2 : C > 0) : 
  C + 0.10 * C = 88 := by
sorry

end NUMINAMATH_CALUDE_old_selling_price_l1902_190274


namespace NUMINAMATH_CALUDE_triangle_properties_l1902_190264

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π / 2 →
  0 < B ∧ B < π →
  0 < C ∧ C < π / 2 →
  A + B + C = π →
  4 * a = Real.sqrt 5 * c →
  Real.cos C = 3 / 5 →
  b = 11 →
  Real.sin A = Real.sqrt 5 / 5 ∧
  1 / 2 * a * b * Real.sin C = 22 :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1902_190264


namespace NUMINAMATH_CALUDE_max_abs_difference_l1902_190283

theorem max_abs_difference (x y : ℝ) 
  (h1 : x^2 + y^2 = 2023)
  (h2 : (x - 2) * (y - 2) = 3) :
  |x - y| ≤ 13 * Real.sqrt 13 ∧ ∃ x y : ℝ, x^2 + y^2 = 2023 ∧ (x - 2) * (y - 2) = 3 ∧ |x - y| = 13 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_max_abs_difference_l1902_190283


namespace NUMINAMATH_CALUDE_triangle_separation_l1902_190230

/-- A triangle in a 2D plane -/
structure Triangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ

/-- Check if two triangles have no common interior or boundary points -/
def no_common_points (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line separates two triangles -/
def separates (line : ℝ × ℝ → Prop) (t1 t2 : Triangle) : Prop := sorry

/-- Check if a line is a side of a triangle -/
def is_side (line : ℝ × ℝ → Prop) (t : Triangle) : Prop := sorry

/-- Main theorem: For any two triangles with no common points, 
    there exists a side of one triangle that separates them -/
theorem triangle_separation (t1 t2 : Triangle) 
  (h : no_common_points t1 t2) : 
  ∃ (line : ℝ × ℝ → Prop), 
    (is_side line t1 ∨ is_side line t2) ∧ 
    separates line t1 t2 := by sorry

end NUMINAMATH_CALUDE_triangle_separation_l1902_190230


namespace NUMINAMATH_CALUDE_monotonic_cubic_function_param_range_l1902_190253

/-- A function f(x) = -x^3 + ax^2 - x - 1 is monotonic on (-∞, +∞) if and only if a ∈ [-√3, √3] -/
theorem monotonic_cubic_function_param_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => -x^3 + a*x^2 - x - 1)) ↔ a ∈ Set.Icc (-Real.sqrt 3) (Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_monotonic_cubic_function_param_range_l1902_190253


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l1902_190289

def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7}
def M : Set ℕ := {2, 4, 7}

theorem complement_of_M_in_U :
  U \ M = {1, 3, 5, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l1902_190289


namespace NUMINAMATH_CALUDE_paint_mixture_ratio_l1902_190279

/-- Given a paint mixture with a ratio of red:yellow:white as 5:3:7,
    if 21 quarts of white paint are used, then 9 quarts of yellow paint should be used. -/
theorem paint_mixture_ratio (red yellow white : ℚ) (total : ℚ) : 
  red / total = 5 / 15 →
  yellow / total = 3 / 15 →
  white / total = 7 / 15 →
  white = 21 →
  yellow = 9 := by
sorry

end NUMINAMATH_CALUDE_paint_mixture_ratio_l1902_190279


namespace NUMINAMATH_CALUDE_square_triangles_area_bounds_l1902_190232

-- Define the unit square
def UnitSquare : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the right-angled triangles constructed outward
def OutwardTriangles (s : Set (ℝ × ℝ)) : Set (Set (ℝ × ℝ)) := sorry

-- Define the vertices A, B, C, D
def RightAngleVertices (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the incircle centers O₁, O₂, O₃, O₄
def IncircleCenters (triangles : Set (Set (ℝ × ℝ))) : Set (ℝ × ℝ) := sorry

-- Define the area of a quadrilateral
def QuadrilateralArea (vertices : Set (ℝ × ℝ)) : ℝ := sorry

theorem square_triangles_area_bounds :
  let s := UnitSquare
  let triangles := OutwardTriangles s
  let abcd := RightAngleVertices triangles
  let o₁o₂o₃o₄ := IncircleCenters triangles
  (QuadrilateralArea abcd ≤ 2) ∧ (QuadrilateralArea o₁o₂o₃o₄ ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_square_triangles_area_bounds_l1902_190232


namespace NUMINAMATH_CALUDE_boat_purchase_l1902_190203

theorem boat_purchase (a b c d : ℝ) 
  (h1 : a + b + c + d = 60)
  (h2 : a = (1/2) * (b + c + d))
  (h3 : b = (1/3) * (a + c + d))
  (h4 : c = (1/4) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : d = 13 := by
  sorry

end NUMINAMATH_CALUDE_boat_purchase_l1902_190203


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1902_190236

theorem coefficient_x_cubed_in_expansion :
  let n : ℕ := 30
  let k : ℕ := 3
  let a : ℕ := 2
  (Nat.choose n k) * a^(n - k) = 4060 * 2^27 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_in_expansion_l1902_190236


namespace NUMINAMATH_CALUDE_plates_theorem_l1902_190270

def plates_problem (flower_plates checked_plates : ℕ) : ℕ :=
  let initial_plates := flower_plates + checked_plates
  let polka_plates := 2 * checked_plates
  let total_before_smash := initial_plates + polka_plates
  total_before_smash - 1

theorem plates_theorem : 
  plates_problem 4 8 = 27 := by
  sorry

end NUMINAMATH_CALUDE_plates_theorem_l1902_190270


namespace NUMINAMATH_CALUDE_trip_savings_l1902_190273

def evening_ticket_cost : ℚ := 10
def combo_cost : ℚ := 10
def ticket_discount_percent : ℚ := 20
def combo_discount_percent : ℚ := 50

def ticket_savings : ℚ := (ticket_discount_percent / 100) * evening_ticket_cost
def combo_savings : ℚ := (combo_discount_percent / 100) * combo_cost
def total_savings : ℚ := ticket_savings + combo_savings

theorem trip_savings : total_savings = 7 := by sorry

end NUMINAMATH_CALUDE_trip_savings_l1902_190273


namespace NUMINAMATH_CALUDE_cubic_root_squared_l1902_190208

theorem cubic_root_squared (r : ℝ) : 
  r^3 - r + 3 = 0 → (r^2)^3 - 2*(r^2)^2 + r^2 - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_squared_l1902_190208


namespace NUMINAMATH_CALUDE_student_marks_theorem_l1902_190286

/-- Calculates the total marks secured by a student in an examination with the given conditions. -/
def total_marks (total_questions : ℕ) (correct_answers : ℕ) (marks_per_correct : ℕ) (marks_per_wrong : ℕ) : ℤ :=
  (correct_answers * marks_per_correct : ℤ) - ((total_questions - correct_answers) * marks_per_wrong)

/-- Theorem stating that under the given conditions, the student secures 150 marks. -/
theorem student_marks_theorem :
  total_marks 60 42 4 1 = 150 := by
  sorry

end NUMINAMATH_CALUDE_student_marks_theorem_l1902_190286


namespace NUMINAMATH_CALUDE_line_points_l1902_190239

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem line_points : 
  let p1 : Point := ⟨4, 10⟩
  let p2 : Point := ⟨-2, -8⟩
  let p3 : Point := ⟨1, 1⟩
  let p4 : Point := ⟨-1, -5⟩
  let p5 : Point := ⟨3, 7⟩
  let p6 : Point := ⟨0, -1⟩
  let p7 : Point := ⟨2, 3⟩
  collinear p1 p2 p3 ∧ 
  collinear p1 p2 p4 ∧ 
  collinear p1 p2 p5 ∧ 
  ¬collinear p1 p2 p6 ∧ 
  ¬collinear p1 p2 p7 := by sorry

end NUMINAMATH_CALUDE_line_points_l1902_190239


namespace NUMINAMATH_CALUDE_optimal_mask_purchase_l1902_190207

/-- Represents the profit function for mask sales -/
def profit_function (x : ℝ) : ℝ := -0.05 * x + 400

/-- Represents the constraints on the number of masks -/
def mask_constraints (x : ℝ) : Prop := 500 ≤ x ∧ x ≤ 1000

/-- Theorem stating the optimal purchase for maximum profit -/
theorem optimal_mask_purchase :
  ∀ x : ℝ, mask_constraints x →
  profit_function 500 ≥ profit_function x :=
sorry

end NUMINAMATH_CALUDE_optimal_mask_purchase_l1902_190207


namespace NUMINAMATH_CALUDE_square_diagonal_cut_l1902_190243

theorem square_diagonal_cut (s : ℝ) (h : s = 10) : 
  let diagonal := s * Real.sqrt 2
  ∃ (a b c : ℝ), a = s ∧ b = s ∧ c = diagonal ∧ 
    a^2 + b^2 = c^2 :=
by sorry

end NUMINAMATH_CALUDE_square_diagonal_cut_l1902_190243


namespace NUMINAMATH_CALUDE_rectangle_dimension_increase_l1902_190238

theorem rectangle_dimension_increase (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.1 * L) (h2 : L' * B' = 1.43 * (L * B)) : B' = 1.3 * B :=
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_increase_l1902_190238


namespace NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1902_190251

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with focus on the x-axis -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Represents a hyperbola with focus on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- The intersection point of the asymptotes -/
def intersectionPoint : Point := { x := 4, y := 8 }

/-- Check if a point satisfies the parabola equation -/
def satisfiesParabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- Check if a point satisfies the hyperbola equation -/
def satisfiesHyperbola (point : Point) (hyperbola : Hyperbola) : Prop :=
  (point.x^2 / hyperbola.a^2) - (point.y^2 / hyperbola.b^2) = 1

/-- The main theorem -/
theorem parabola_hyperbola_equations :
  ∃ (parabola : Parabola) (hyperbola : Hyperbola),
    satisfiesParabola intersectionPoint parabola ∧
    satisfiesHyperbola intersectionPoint hyperbola ∧
    parabola.p = 8 ∧
    hyperbola.a^2 = 16/5 ∧
    hyperbola.b^2 = 64/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_equations_l1902_190251


namespace NUMINAMATH_CALUDE_positive_numbers_l1902_190221

theorem positive_numbers (a b c : ℝ) 
  (sum_positive : a + b + c > 0)
  (sum_products_positive : a * b + b * c + c * a > 0)
  (product_positive : a * b * c > 0) :
  a > 0 ∧ b > 0 ∧ c > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_numbers_l1902_190221


namespace NUMINAMATH_CALUDE_merchant_profit_l1902_190298

theorem merchant_profit (C S : ℝ) (h : 18 * C = 16 * S) : 
  (S - C) / C * 100 = 12.5 := by
sorry

end NUMINAMATH_CALUDE_merchant_profit_l1902_190298


namespace NUMINAMATH_CALUDE_k_range_l1902_190276

open Real

theorem k_range (k : ℝ) : 
  (∀ x > 1, k * (exp (k * x) + 1) - (1 / x + 1) * log x > 0) → 
  k > 1 / exp 1 := by
  sorry

end NUMINAMATH_CALUDE_k_range_l1902_190276


namespace NUMINAMATH_CALUDE_triangle_angle_B_l1902_190277

theorem triangle_angle_B (a b : ℝ) (A : Real) (h1 : a = 4) (h2 : b = 4 * Real.sqrt 3) (h3 : A = 30 * π / 180) :
  let B := Real.arcsin ((b * Real.sin A) / a)
  B = 60 * π / 180 ∨ B = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_B_l1902_190277


namespace NUMINAMATH_CALUDE_max_sum_constraint_l1902_190210

theorem max_sum_constraint (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) :
  ∀ a b : ℝ, 3 * (a^2 + b^2) = a - b → x + y ≤ (1 : ℝ) / 3 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_constraint_l1902_190210


namespace NUMINAMATH_CALUDE_factorization_theorem_l1902_190252

variable (x : ℝ)

theorem factorization_theorem :
  (x^2 - 4*x + 3 = (x-1)*(x-3)) ∧
  (4*x^2 + 12*x - 7 = (2*x+7)*(2*x-1)) := by
  sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1902_190252


namespace NUMINAMATH_CALUDE_f_inequality_l1902_190204

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + |x|) - 1 / (1 + x^2)

theorem f_inequality (x : ℝ) : f x > f (2*x - 1) ↔ 1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1902_190204


namespace NUMINAMATH_CALUDE_valentinas_burger_length_l1902_190261

/-- The length of a burger shared equally between two people, given the length of one person's share. -/
def burger_length (share_length : ℝ) : ℝ := 2 * share_length

/-- Proof that Valentina's burger is 12 inches long. -/
theorem valentinas_burger_length : 
  let share_length := 6
  burger_length share_length = 12 := by
  sorry

end NUMINAMATH_CALUDE_valentinas_burger_length_l1902_190261


namespace NUMINAMATH_CALUDE_factorization_expr1_l1902_190248

theorem factorization_expr1 (a b : ℝ) :
  -3 * a^2 * b + 12 * a * b - 12 * b = -3 * b * (a - 2)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_expr1_l1902_190248


namespace NUMINAMATH_CALUDE_expression_evaluation_l1902_190266

theorem expression_evaluation (a b c d : ℝ) 
  (ha : a = 11) (hb : b = 13) (hc : c = 17) (hd : d = 19) :
  (a^2 * (1/b - 1/d) + b^2 * (1/d - 1/a) + c^2 * (1/a - 1/c) + d^2 * (1/c - 1/b)) /
  (a * (1/b - 1/d) + b * (1/d - 1/a) + c * (1/a - 1/c) + d * (1/c - 1/b)) = a + b + c + d :=
by sorry

#eval (11 : ℝ) + 13 + 17 + 19  -- To verify the result is indeed 60

end NUMINAMATH_CALUDE_expression_evaluation_l1902_190266


namespace NUMINAMATH_CALUDE_square_sum_and_product_l1902_190228

theorem square_sum_and_product (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) : 
  a^2 + b^2 = 5 ∧ a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_square_sum_and_product_l1902_190228


namespace NUMINAMATH_CALUDE_triangle_inequality_l1902_190229

theorem triangle_inequality (x y z : ℝ) 
  (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 1) : 
  0 ≤ y * z + z * x + x * y - 2 * x * y * z ∧ 
  y * z + z * x + x * y - 2 * x * y * z ≤ 7 / 27 := by
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1902_190229


namespace NUMINAMATH_CALUDE_comic_book_arrangements_l1902_190292

def spiderman_comics : ℕ := 6
def archie_comics : ℕ := 5
def garfield_comics : ℕ := 4
def batman_comics : ℕ := 7

def total_arrangements : ℕ := 59536691200

theorem comic_book_arrangements :
  (spiderman_comics.factorial * archie_comics.factorial * garfield_comics.factorial * batman_comics.factorial) *
  (spiderman_comics + archie_comics + garfield_comics + batman_comics - 3).factorial = total_arrangements := by
  sorry

end NUMINAMATH_CALUDE_comic_book_arrangements_l1902_190292


namespace NUMINAMATH_CALUDE_product_a4_a5_a6_l1902_190268

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem product_a4_a5_a6 (a : ℕ → ℝ) :
  geometric_sequence a →
  (∀ n : ℕ, a n > 0) →
  a 1 * a 9 = 16 →
  a 1 + a 9 = 10 →
  a 4 * a 5 * a 6 = 64 := by
sorry

end NUMINAMATH_CALUDE_product_a4_a5_a6_l1902_190268


namespace NUMINAMATH_CALUDE_max_equilateral_triangles_l1902_190250

/-- Represents a matchstick configuration --/
structure MatchstickConfig where
  num_matchsticks : ℕ
  connected_end_to_end : Bool

/-- Represents the number of equilateral triangles in a configuration --/
def num_equilateral_triangles (config : MatchstickConfig) : ℕ := sorry

/-- The theorem stating the maximum number of equilateral triangles --/
theorem max_equilateral_triangles (config : MatchstickConfig) 
  (h1 : config.num_matchsticks = 6) 
  (h2 : config.connected_end_to_end = true) : 
  ∃ (n : ℕ), n ≤ 4 ∧ ∀ (m : ℕ), num_equilateral_triangles config ≤ m → n ≤ m :=
sorry

end NUMINAMATH_CALUDE_max_equilateral_triangles_l1902_190250


namespace NUMINAMATH_CALUDE_farm_animal_count_l1902_190212

/-- Given a farm with cows and ducks, calculate the total number of animals --/
theorem farm_animal_count (total_legs : ℕ) (num_cows : ℕ) : total_legs = 42 → num_cows = 6 → ∃ (num_ducks : ℕ), num_cows + num_ducks = 15 := by
  sorry

end NUMINAMATH_CALUDE_farm_animal_count_l1902_190212


namespace NUMINAMATH_CALUDE_relationship_implies_function_l1902_190225

-- Define the relationship between x and y
def relationship (x y : ℝ) : Prop :=
  y = 2*x - 1 - Real.sqrt (y^2 - 2*x*y + 3*x - 2)

-- Define the function we want to prove
def function (x : ℝ) : Set ℝ :=
  if x ≠ 1 then {2*x - 1.5}
  else {y : ℝ | y ≤ 1}

-- Theorem statement
theorem relationship_implies_function :
  ∀ x y : ℝ, relationship x y → y ∈ function x :=
sorry

end NUMINAMATH_CALUDE_relationship_implies_function_l1902_190225


namespace NUMINAMATH_CALUDE_binomial_product_l1902_190206

theorem binomial_product (x : ℝ) : (4 * x + 3) * (x - 6) = 4 * x^2 - 21 * x - 18 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l1902_190206


namespace NUMINAMATH_CALUDE_divisible_by_56_l1902_190256

theorem divisible_by_56 (n : ℕ) 
  (h1 : ∃ k : ℕ, 3 * n + 1 = k ^ 2) 
  (h2 : ∃ m : ℕ, 4 * n + 1 = m ^ 2) : 
  56 ∣ n := by
sorry

end NUMINAMATH_CALUDE_divisible_by_56_l1902_190256


namespace NUMINAMATH_CALUDE_domain_of_g_l1902_190258

-- Define the function f with domain [0,2]
def f : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 2}

-- Define the function g(x) = f(x²)
def g (x : ℝ) : Prop := x^2 ∈ f

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x} = {x : ℝ | -Real.sqrt 2 ≤ x ∧ x ≤ Real.sqrt 2} :=
by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1902_190258


namespace NUMINAMATH_CALUDE_journey_distance_l1902_190278

theorem journey_distance (total_time : ℝ) (speed1 speed2 speed3 : ℝ) :
  total_time = 36 ∧ 
  speed1 = 21 ∧ 
  speed2 = 45 ∧ 
  speed3 = 24 → 
  ∃ (distance : ℝ),
    distance = 972 ∧
    total_time = distance / (3 * speed1) + distance / (3 * speed2) + distance / (3 * speed3) :=
by sorry

end NUMINAMATH_CALUDE_journey_distance_l1902_190278


namespace NUMINAMATH_CALUDE_supplementary_angles_ratio_l1902_190299

theorem supplementary_angles_ratio (a b : ℝ) : 
  a + b = 180 →  -- The angles are supplementary
  a / b = 5 / 4 →  -- The ratio of the angles is 5:4
  min a b = 80 :=  -- The smaller angle is 80°
by sorry

end NUMINAMATH_CALUDE_supplementary_angles_ratio_l1902_190299


namespace NUMINAMATH_CALUDE_multichoose_eq_choose_l1902_190215

/-- F_n^r represents the number of ways to choose r elements from [1, n] with repetition and disregarding order -/
def F (n : ℕ) (r : ℕ) : ℕ := sorry

/-- Theorem: The number of ways to choose r elements from [1, n] with repetition and disregarding order
    is equal to the number of ways to choose r elements from [1, n+r-1] without repetition -/
theorem multichoose_eq_choose (n : ℕ) (r : ℕ) : F n r = Nat.choose (n + r - 1) r := by sorry

end NUMINAMATH_CALUDE_multichoose_eq_choose_l1902_190215


namespace NUMINAMATH_CALUDE_cube_halving_l1902_190223

theorem cube_halving (r : ℝ) :
  let a := (2 * r) ^ 3
  let a_half := (2 * (r / 2)) ^ 3
  a_half = (1 / 8) * a := by
  sorry

end NUMINAMATH_CALUDE_cube_halving_l1902_190223


namespace NUMINAMATH_CALUDE_function_analysis_l1902_190219

/-- Given a function f and some conditions, prove its analytical expression and range -/
theorem function_analysis (f : ℝ → ℝ) (ω φ : ℝ) :
  (ω > 0) →
  (φ > 0 ∧ φ < Real.pi / 2) →
  (Real.tan φ = 2 * Real.sqrt 3) →
  (∀ x, f x = Real.sqrt 13 * Real.cos (ω * x) * Real.cos (ω * x - φ) - Real.sin (ω * x) ^ 2) →
  (∀ x, f (x + Real.pi / ω) = f x) →
  (∀ T > 0, (∀ x, f (x + T) = f x) → T ≥ Real.pi / ω) →
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (Set.Icc (1 / 13) 2 = { y | ∃ x ∈ Set.Icc (Real.pi / 12) φ, f x = y }) := by
  sorry

end NUMINAMATH_CALUDE_function_analysis_l1902_190219


namespace NUMINAMATH_CALUDE_at_least_three_functional_probability_l1902_190242

def num_lamps : ℕ := 5
def func_prob : ℝ := 0.2

theorem at_least_three_functional_probability :
  let p := func_prob
  let q := 1 - p
  let binom_prob (n k : ℕ) := (Nat.choose n k : ℝ) * p^k * q^(n-k)
  binom_prob num_lamps 3 + binom_prob num_lamps 4 + binom_prob num_lamps 5 = 0.06 := by
  sorry

end NUMINAMATH_CALUDE_at_least_three_functional_probability_l1902_190242


namespace NUMINAMATH_CALUDE_parabola_and_point_theorem_l1902_190235

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form y^2 = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

def on_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

def perpendicular (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem parabola_and_point_theorem (C : Parabola) (A B O : Point) :
  on_parabola A C →
  on_parabola B C →
  A.x = 1 →
  A.y = 2 →
  O.x = 0 →
  O.y = 0 →
  B.x ≠ 0 →
  perpendicular A O B →
  (C.p = 2 ∧ B.x = 16 ∧ B.y = -8) := by sorry

end NUMINAMATH_CALUDE_parabola_and_point_theorem_l1902_190235


namespace NUMINAMATH_CALUDE_point_on_x_axis_l1902_190291

theorem point_on_x_axis (m : ℚ) :
  (∃ x : ℚ, x = 2 - m ∧ 0 = 3 * m + 1) → m = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_point_on_x_axis_l1902_190291


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l1902_190226

/-- Represents a repeating decimal with an integer part, a non-repeating fractional part, and a repeating part -/
structure RepeatingDecimal where
  integerPart : ℤ
  nonRepeatingPart : ℚ
  repeatingPart : ℚ
  nonRepeatingPartLessThanOne : nonRepeatingPart < 1
  repeatingPartLessThanOne : repeatingPart < 1

/-- Converts a RepeatingDecimal to a rational number -/
def RepeatingDecimal.toRational (d : RepeatingDecimal) : ℚ :=
  d.integerPart + d.nonRepeatingPart + d.repeatingPart / (1 - (1/10)^(d.repeatingPart.den))

/-- Checks if a fraction is in its lowest terms -/
def isLowestTerms (n d : ℤ) : Prop :=
  Nat.gcd n.natAbs d.natAbs = 1

theorem repeating_decimal_equiv_fraction :
  let d : RepeatingDecimal := ⟨0, 4/10, 37/100, by norm_num, by norm_num⟩
  d.toRational = 433 / 990 ∧ isLowestTerms 433 990 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l1902_190226


namespace NUMINAMATH_CALUDE_inverse_sum_product_identity_l1902_190284

theorem inverse_sum_product_identity (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y*z + x*z + x*y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_inverse_sum_product_identity_l1902_190284


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l1902_190241

/-- Represents a hyperbola with semi-major axis a -/
structure Hyperbola (a : ℝ) where
  equation : ℝ → ℝ → Prop := fun x y => x^2 / a^2 - y^2 / 9 = 1
  asymptote : ℝ → ℝ → Prop := fun x y => 3 * x - 2 * y = 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Distance between two points -/
def distance (p1 p2 : Point) : ℝ := sorry

/-- Left focus of the hyperbola -/
def F1 (h : Hyperbola 2) : Point := sorry

/-- Right focus of the hyperbola -/
def F2 (h : Hyperbola 2) : Point := sorry

/-- A point P on the hyperbola -/
def P (h : Hyperbola 2) : Point := sorry

theorem hyperbola_focus_distance (h : Hyperbola 2) (p : Point) 
  (hp : h.equation p.x p.y) 
  (ha : h.asymptote 3 2) 
  (hd : distance p (F1 h) = 3) : 
  distance p (F2 h) = 7 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l1902_190241


namespace NUMINAMATH_CALUDE_fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l1902_190227

def dvd_sequence : Nat → Nat
  | 0 => 2
  | n + 1 => 2 * dvd_sequence n

theorem fifth_rack_dvds : dvd_sequence 4 = 32 :=
by
  sorry

theorem sixth_rack_dvds : dvd_sequence 5 = 64 :=
by
  sorry

theorem prove_fifth_rack (h : dvd_sequence 5 = 64) : dvd_sequence 4 = 32 :=
by
  sorry

end NUMINAMATH_CALUDE_fifth_rack_dvds_sixth_rack_dvds_prove_fifth_rack_l1902_190227


namespace NUMINAMATH_CALUDE_table_tennis_racket_sales_l1902_190259

/-- Profit function for table tennis racket sales -/
def profit_function (c : ℝ) (x : ℝ) : ℝ :=
  let y := -10 * x + 900
  y * (x - c)

/-- Problem statement for table tennis racket sales -/
theorem table_tennis_racket_sales 
  (c : ℝ) 
  (max_price : ℝ) 
  (min_profit : ℝ) 
  (h1 : c = 50) 
  (h2 : max_price = 75) 
  (h3 : min_profit = 3000) :
  ∃ (optimal_price : ℝ) (max_profit : ℝ) (price_range : Set ℝ),
    -- 1. The monthly profit function
    (∀ x, profit_function c x = -10 * x^2 + 1400 * x - 45000) ∧
    -- 2. The optimal price and maximum profit
    (optimal_price = 70 ∧ 
     max_profit = profit_function c optimal_price ∧
     max_profit = 4000 ∧
     ∀ x, profit_function c x ≤ max_profit) ∧
    -- 3. The range of acceptable selling prices
    (price_range = {x | 60 ≤ x ∧ x ≤ 75} ∧
     ∀ x ∈ price_range, 
       x ≤ max_price ∧ 
       profit_function c x ≥ min_profit) :=
by sorry

end NUMINAMATH_CALUDE_table_tennis_racket_sales_l1902_190259


namespace NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l1902_190237

theorem tan_fifteen_degree_fraction_equals_sqrt_three_over_three :
  (1 - Real.tan (15 * π / 180)) / (1 + Real.tan (15 * π / 180)) = Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_fifteen_degree_fraction_equals_sqrt_three_over_three_l1902_190237


namespace NUMINAMATH_CALUDE_x_with_three_prime_divisors_l1902_190216

theorem x_with_three_prime_divisors (x n : ℕ) 
  (h1 : x = 2^n - 32)
  (h2 : (Nat.factors x).toFinset.card = 3)
  (h3 : 2 ∈ Nat.factors x) :
  x = 2016 ∨ x = 16352 := by
  sorry

end NUMINAMATH_CALUDE_x_with_three_prime_divisors_l1902_190216


namespace NUMINAMATH_CALUDE_people_per_entrance_l1902_190218

theorem people_per_entrance 
  (total_entrances : ℕ) 
  (total_people : ℕ) 
  (h1 : total_entrances = 5) 
  (h2 : total_people = 1415) :
  total_people / total_entrances = 283 :=
by sorry

end NUMINAMATH_CALUDE_people_per_entrance_l1902_190218


namespace NUMINAMATH_CALUDE_parallel_vectors_component_l1902_190263

/-- Given two parallel vectors a and b, prove that the second component of b is 5/3. -/
theorem parallel_vectors_component (a b : ℝ × ℝ) : 
  a = (3, 5) → b.1 = 1 → (∃ (k : ℝ), a = k • b) → b.2 = 5/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_component_l1902_190263


namespace NUMINAMATH_CALUDE_swaps_theorem_l1902_190213

/-- Represents a mode of letter swapping -/
inductive SwapMode
| Adjacent : SwapMode
| Any : SwapMode

/-- Represents a string of letters -/
def Text : Type := List Char

/-- Calculate the minimum number of swaps required to transform one text into another -/
def minSwaps (original : Text) (target : Text) (mode : SwapMode) : Nat :=
  match mode with
  | SwapMode.Adjacent => sorry
  | SwapMode.Any => sorry

/-- The original text -/
def originalText : Text := ['M', 'E', 'G', 'Y', 'E', 'I', ' ', 'T', 'A', 'K', 'A', 'R', 'É', 'K', 'P', 'É', 'N', 'Z', 'T', 'Á', 'R', ' ', 'R', '.', ' ', 'T', '.']

/-- The target text -/
def targetText : Text := ['T', 'A', 'T', 'Á', 'R', ' ', 'G', 'Y', 'E', 'R', 'M', 'E', 'K', ' ', 'A', ' ', 'P', 'É', 'N', 'Z', 'T', ' ', 'K', 'É', 'R', 'I', '.']

theorem swaps_theorem :
  (minSwaps originalText targetText SwapMode.Adjacent = 85) ∧
  (minSwaps originalText targetText SwapMode.Any = 11) :=
sorry

end NUMINAMATH_CALUDE_swaps_theorem_l1902_190213


namespace NUMINAMATH_CALUDE_numerical_puzzle_solution_l1902_190295

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that checks if two digits are different -/
def differentDigits (a b : ℕ) : Prop := a ≠ b ∧ a < 10 ∧ b < 10

/-- The main theorem stating the solution to the numerical puzzle -/
theorem numerical_puzzle_solution :
  ∀ (a b : ℕ), differentDigits a b →
    isTwoDigit (10 * a + b) →
    (10 * a + b = b ^ (10 * a + b)) ↔ 
    ((a = 3 ∧ b = 2) ∨ (a = 3 ∧ b = 6) ∨ (a = 6 ∧ b = 4)) :=
by sorry

end NUMINAMATH_CALUDE_numerical_puzzle_solution_l1902_190295


namespace NUMINAMATH_CALUDE_stratified_sample_size_l1902_190296

/-- Represents the number of students in each grade --/
structure StudentCounts where
  freshman : ℕ
  sophomore : ℕ
  senior : ℕ

/-- Calculates the total number of students --/
def total_students (counts : StudentCounts) : ℕ :=
  counts.freshman + counts.sophomore + counts.senior

/-- Calculates the sample size based on the number of sampled freshmen --/
def sample_size (counts : StudentCounts) (sampled_freshmen : ℕ) : ℕ :=
  sampled_freshmen * (total_students counts) / counts.freshman

/-- Theorem stating that for the given student counts and sampled freshmen, the sample size is 30 --/
theorem stratified_sample_size 
  (counts : StudentCounts) 
  (h1 : counts.freshman = 700)
  (h2 : counts.sophomore = 500)
  (h3 : counts.senior = 300)
  (h4 : sampled_freshmen = 14) :
  sample_size counts sampled_freshmen = 30 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l1902_190296
