import Mathlib

namespace NUMINAMATH_GPT_relationship_among_abc_l222_22271

theorem relationship_among_abc 
  (f : ℝ → ℝ)
  (h_symm : ∀ x, f (x) = f (-x))
  (h_def : ∀ x, 0 < x → f x = |Real.log x / Real.log 2|)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = f (1 / 3))
  (hb : b = f (-4))
  (hc : c = f 2) :
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_abc_l222_22271


namespace NUMINAMATH_GPT_move_line_down_l222_22278

theorem move_line_down (x : ℝ) : (y = -x + 1) → (y = -x - 2) := by
  sorry

end NUMINAMATH_GPT_move_line_down_l222_22278


namespace NUMINAMATH_GPT_directrix_of_parabola_l222_22291

theorem directrix_of_parabola (x y : ℝ) : (x ^ 2 = y) → (4 * y + 1 = 0) :=
sorry

end NUMINAMATH_GPT_directrix_of_parabola_l222_22291


namespace NUMINAMATH_GPT_train_speed_l222_22258

theorem train_speed
  (train_length : ℕ)
  (man_speed_kmph : ℕ)
  (time_to_pass : ℕ)
  (speed_of_train : ℝ) :
  train_length = 180 →
  man_speed_kmph = 8 →
  time_to_pass = 4 →
  speed_of_train = 154 := 
by
  sorry

end NUMINAMATH_GPT_train_speed_l222_22258


namespace NUMINAMATH_GPT_find_y_l222_22205

def binary_op (a b c d : Int) : Int × Int := (a + d, b - c)

theorem find_y : ∃ y : Int, (binary_op 3 y 2 0) = (3, 4) ↔ y = 6 := by
  sorry

end NUMINAMATH_GPT_find_y_l222_22205


namespace NUMINAMATH_GPT_f_of_x_l222_22208

theorem f_of_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x-1) = 3*x - 1) : ∀ x : ℤ, f x = 3*x + 2 :=
by
  sorry

end NUMINAMATH_GPT_f_of_x_l222_22208


namespace NUMINAMATH_GPT_factorize_expression_l222_22299

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end NUMINAMATH_GPT_factorize_expression_l222_22299


namespace NUMINAMATH_GPT_marius_scored_3_more_than_darius_l222_22287

theorem marius_scored_3_more_than_darius 
  (D M T : ℕ) 
  (h1 : D = 10) 
  (h2 : T = D + 5) 
  (h3 : M + D + T = 38) : 
  M = D + 3 := 
by
  sorry

end NUMINAMATH_GPT_marius_scored_3_more_than_darius_l222_22287


namespace NUMINAMATH_GPT_totalCostOfCombinedSubscriptions_l222_22288

-- Define the given conditions
def packageACostPerMonth : ℝ := 10
def packageAMonths : ℝ := 6
def packageADiscount : ℝ := 0.10

def packageBCostPerMonth : ℝ := 12
def packageBMonths : ℝ := 9
def packageBDiscount : ℝ := 0.15

-- Define the total cost after discounts
def packageACostAfterDiscount : ℝ := packageACostPerMonth * packageAMonths * (1 - packageADiscount)
def packageBCostAfterDiscount : ℝ := packageBCostPerMonth * packageBMonths * (1 - packageBDiscount)

-- Statement to be proved
theorem totalCostOfCombinedSubscriptions :
  packageACostAfterDiscount + packageBCostAfterDiscount = 145.80 := by
  sorry

end NUMINAMATH_GPT_totalCostOfCombinedSubscriptions_l222_22288


namespace NUMINAMATH_GPT_total_sales_l222_22266

-- Define sales of Robyn and Lucy
def Robyn_sales : Nat := 47
def Lucy_sales : Nat := 29

-- Prove total sales
theorem total_sales : Robyn_sales + Lucy_sales = 76 :=
by
  sorry

end NUMINAMATH_GPT_total_sales_l222_22266


namespace NUMINAMATH_GPT_exp_decreasing_function_range_l222_22285

theorem exp_decreasing_function_range (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1) (h_f : a^(x+1) ≥ 1) : x ≤ -1 :=
sorry

end NUMINAMATH_GPT_exp_decreasing_function_range_l222_22285


namespace NUMINAMATH_GPT_min_sum_of_factors_l222_22222

theorem min_sum_of_factors (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 3432) :
  a + b + c ≥ 56 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l222_22222


namespace NUMINAMATH_GPT_arithmetic_mean_is_correct_l222_22249

-- Define the numbers
def num1 : ℕ := 18
def num2 : ℕ := 27
def num3 : ℕ := 45

-- Define the number of terms
def n : ℕ := 3

-- Define the sum of the numbers
def total_sum : ℕ := num1 + num2 + num3

-- Define the arithmetic mean
def arithmetic_mean : ℕ := total_sum / n

-- Theorem stating that the arithmetic mean of the numbers is 30
theorem arithmetic_mean_is_correct : arithmetic_mean = 30 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_arithmetic_mean_is_correct_l222_22249


namespace NUMINAMATH_GPT_combines_like_terms_l222_22275

theorem combines_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := 
by sorry

end NUMINAMATH_GPT_combines_like_terms_l222_22275


namespace NUMINAMATH_GPT_rectangle_area_change_l222_22276

theorem rectangle_area_change 
  (L B : ℝ) 
  (A : ℝ := L * B) 
  (L' : ℝ := 1.30 * L) 
  (B' : ℝ := 0.75 * B) 
  (A' : ℝ := L' * B') : 
  A' / A = 0.975 := 
by sorry

end NUMINAMATH_GPT_rectangle_area_change_l222_22276


namespace NUMINAMATH_GPT_cost_per_lb_of_mixture_l222_22218

def millet_weight : ℝ := 100
def millet_cost_per_lb : ℝ := 0.60
def sunflower_weight : ℝ := 25
def sunflower_cost_per_lb : ℝ := 1.10

theorem cost_per_lb_of_mixture :
  let millet_weight := 100
  let millet_cost_per_lb := 0.60
  let sunflower_weight := 25
  let sunflower_cost_per_lb := 1.10
  let millet_total_cost := millet_weight * millet_cost_per_lb
  let sunflower_total_cost := sunflower_weight * sunflower_cost_per_lb
  let total_cost := millet_total_cost + sunflower_total_cost
  let total_weight := millet_weight + sunflower_weight
  (total_cost / total_weight) = 0.70 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_lb_of_mixture_l222_22218


namespace NUMINAMATH_GPT_absentees_in_morning_session_is_three_l222_22283

theorem absentees_in_morning_session_is_three
  (registered_morning : ℕ)
  (registered_afternoon : ℕ)
  (absent_afternoon : ℕ)
  (total_students : ℕ)
  (total_registered : ℕ)
  (attended_afternoon : ℕ)
  (attended_morning : ℕ)
  (absent_morning : ℕ) :
  registered_morning = 25 →
  registered_afternoon = 24 →
  absent_afternoon = 4 →
  total_students = 42 →
  total_registered = registered_morning + registered_afternoon →
  attended_afternoon = registered_afternoon - absent_afternoon →
  attended_morning = total_students - attended_afternoon →
  absent_morning = registered_morning - attended_morning →
  absent_morning = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_absentees_in_morning_session_is_three_l222_22283


namespace NUMINAMATH_GPT_range_of_a_if_q_sufficient_but_not_necessary_for_p_l222_22214

variable {x a : ℝ}

def p (x : ℝ) : Prop := x > 1 ∨ x < -3
def q (x a : ℝ) : Prop := x > a

theorem range_of_a_if_q_sufficient_but_not_necessary_for_p :
  (∀ x, q x a → p x) ∧ ¬(∀ x, p x → q x a) → a ∈ Set.Ici 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_if_q_sufficient_but_not_necessary_for_p_l222_22214


namespace NUMINAMATH_GPT_distance_to_nearest_river_l222_22282

theorem distance_to_nearest_river (d : ℝ) (h₁ : ¬ (d ≤ 12)) (h₂ : ¬ (d ≥ 15)) (h₃ : ¬ (d ≥ 10)) :
  12 < d ∧ d < 15 :=
by 
  sorry

end NUMINAMATH_GPT_distance_to_nearest_river_l222_22282


namespace NUMINAMATH_GPT_total_weekly_water_consumption_l222_22203

-- Definitions coming from the conditions of the problem
def num_cows : Nat := 40
def water_per_cow_per_day : Nat := 80
def num_sheep : Nat := 10 * num_cows
def water_per_sheep_per_day : Nat := water_per_cow_per_day / 4
def days_in_week : Nat := 7

-- To prove statement: 
theorem total_weekly_water_consumption :
  let weekly_water_cow := water_per_cow_per_day * days_in_week
  let total_weekly_water_cows := weekly_water_cow * num_cows
  let daily_water_sheep := water_per_sheep_per_day
  let weekly_water_sheep := daily_water_sheep * days_in_week
  let total_weekly_water_sheep := weekly_water_sheep * num_sheep
  total_weekly_water_cows + total_weekly_water_sheep = 78400 := 
by
  sorry

end NUMINAMATH_GPT_total_weekly_water_consumption_l222_22203


namespace NUMINAMATH_GPT_total_students_l222_22298

theorem total_students (T : ℕ) (h1 : (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T) : 
  T = 400 :=
sorry

end NUMINAMATH_GPT_total_students_l222_22298


namespace NUMINAMATH_GPT_sheila_will_attend_picnic_l222_22239

def P_Rain : ℝ := 0.3
def P_Cloudy : ℝ := 0.4
def P_Sunny : ℝ := 0.3

def P_Attend_if_Rain : ℝ := 0.25
def P_Attend_if_Cloudy : ℝ := 0.5
def P_Attend_if_Sunny : ℝ := 0.75

def P_Attend : ℝ :=
  P_Rain * P_Attend_if_Rain +
  P_Cloudy * P_Attend_if_Cloudy +
  P_Sunny * P_Attend_if_Sunny

theorem sheila_will_attend_picnic : P_Attend = 0.5 := by
  sorry

end NUMINAMATH_GPT_sheila_will_attend_picnic_l222_22239


namespace NUMINAMATH_GPT_proj_onto_w_equals_correct_l222_22265

open Real

noncomputable def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let scalar_mul c (a : ℝ × ℝ) := (c * a.1, c * a.2)
  let w_dot_w := dot w w
  if w_dot_w = 0 then (0, 0) else scalar_mul (dot v w / w_dot_w) w

theorem proj_onto_w_equals_correct (v w : ℝ × ℝ)
  (hv : v = (2, 3))
  (hw : w = (-4, 1)) :
  proj w v = (20 / 17, -5 / 17) :=
by
  -- The proof would go here. We add sorry to skip it.
  sorry

end NUMINAMATH_GPT_proj_onto_w_equals_correct_l222_22265


namespace NUMINAMATH_GPT_primes_in_sequence_are_12_l222_22242

-- Definition of Q
def Q : Nat := (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47)

-- Set of m values
def ms : List Nat := List.range' 3 101

-- Function to check if Q + m is prime
def is_prime_minus_Q (m : Nat) : Bool := Nat.Prime (Q + m)

-- Counting primes in the sequence
def count_primes_in_sequence : Nat := (ms.filter (λ m => is_prime_minus_Q m = true)).length

theorem primes_in_sequence_are_12 :
  count_primes_in_sequence = 12 := by 
  sorry

end NUMINAMATH_GPT_primes_in_sequence_are_12_l222_22242


namespace NUMINAMATH_GPT_quadratic_properties_l222_22280

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  -- 1. The parabola opens downwards.
  (∀ x : ℝ, quadratic_function x < quadratic_function (x + 1) → false) ∧
  -- 2. The axis of symmetry is x = 1.
  (∀ x : ℝ, ∃ y : ℝ, quadratic_function x = quadratic_function y → x = y ∨ x + y = 2) ∧
  -- 3. The vertex coordinates are (1, 5).
  (quadratic_function 1 = 5) ∧
  -- 4. y decreases for x > 1.
  (∀ x : ℝ, x > 1 → quadratic_function x < quadratic_function (x - 1)) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_properties_l222_22280


namespace NUMINAMATH_GPT_max_value_of_k_l222_22213

theorem max_value_of_k (n : ℕ) (k : ℕ) (h : 3^11 = k * (2 * n + k + 1) / 2) : k = 486 :=
sorry

end NUMINAMATH_GPT_max_value_of_k_l222_22213


namespace NUMINAMATH_GPT_line_through_A_parallel_line_through_B_perpendicular_l222_22231

-- 1. Prove the equation of the line passing through point A(2, 1) and parallel to the line 2x + y - 10 = 0 is 2x + y - 5 = 0.
theorem line_through_A_parallel :
  ∃ (l : ℝ → ℝ), (∀ x, 2 * x + l x - 5 = 0) ∧ (l 2 = 1) ∧ (∃ k, ∀ x, l x = -2 * (x - 2) + k) :=
sorry

-- 2. Prove the equation of the line passing through point B(3, 2) and perpendicular to the line 4x + 5y - 8 = 0 is 5x - 4y - 7 = 0.
theorem line_through_B_perpendicular :
  ∃ (m : ℝ) (l : ℝ → ℝ), (∀ x, 5 * x - 4 * l x - 7 = 0) ∧ (l 3 = 2) ∧ (m = -7) :=
sorry

end NUMINAMATH_GPT_line_through_A_parallel_line_through_B_perpendicular_l222_22231


namespace NUMINAMATH_GPT_binom_mult_l222_22263

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_GPT_binom_mult_l222_22263


namespace NUMINAMATH_GPT_integer_triplets_satisfy_eq_l222_22261

theorem integer_triplets_satisfy_eq {x y z : ℤ} : 
  x^2 + y^2 + z^2 - x * y - y * z - z * x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) := 
by
  sorry

end NUMINAMATH_GPT_integer_triplets_satisfy_eq_l222_22261


namespace NUMINAMATH_GPT_sum_of_real_values_l222_22245

theorem sum_of_real_values (x : ℝ) (h : |3 * x + 1| = 3 * |x - 3|) : x = 4 / 3 := sorry

end NUMINAMATH_GPT_sum_of_real_values_l222_22245


namespace NUMINAMATH_GPT_set_intersection_l222_22279

def S : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def T : Set ℝ := {x | x > 1}
def result : Set ℝ := {x | x ≥ 3 ∨ (1 < x ∧ x ≤ 2)}

theorem set_intersection (x : ℝ) : x ∈ (S ∩ T) ↔ x ∈ result := by
  sorry

end NUMINAMATH_GPT_set_intersection_l222_22279


namespace NUMINAMATH_GPT_students_number_l222_22211

theorem students_number (x a o : ℕ)
  (h1 : o = 3 * a + 3)
  (h2 : a = 2 * x + 6)
  (h3 : o = 7 * x - 5) :
  x = 26 :=
by sorry

end NUMINAMATH_GPT_students_number_l222_22211


namespace NUMINAMATH_GPT_sum_of_five_primes_is_145_l222_22247

-- Condition: common difference is 12
def common_difference : ℕ := 12

-- Five prime numbers forming an arithmetic sequence with the given common difference
def a1 : ℕ := 5
def a2 : ℕ := a1 + common_difference
def a3 : ℕ := a2 + common_difference
def a4 : ℕ := a3 + common_difference
def a5 : ℕ := a4 + common_difference

-- The sum of the arithmetic sequence
def sum_of_primes : ℕ := a1 + a2 + a3 + a4 + a5

-- Prove that the sum of these five prime numbers is 145
theorem sum_of_five_primes_is_145 : sum_of_primes = 145 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_sum_of_five_primes_is_145_l222_22247


namespace NUMINAMATH_GPT_triangle_inradius_exradii_relation_l222_22260

theorem triangle_inradius_exradii_relation
  (a b c : ℝ) (S : ℝ) (r r_a r_b r_c : ℝ)
  (h_inradius : S = (1/2) * r * (a + b + c))
  (h_exradii_a : r_a = 2 * S / (b + c - a))
  (h_exradii_b : r_b = 2 * S / (c + a - b))
  (h_exradii_c : r_c = 2 * S / (a + b - c))
  (h_area : S = (1/2) * (a * r_a + b * r_b + c * r_c - a * r - b * r - c * r)) :
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  by sorry

end NUMINAMATH_GPT_triangle_inradius_exradii_relation_l222_22260


namespace NUMINAMATH_GPT_metal_beams_per_panel_l222_22255

theorem metal_beams_per_panel (panels sheets_per_panel rods_per_sheet rods_needed beams_per_panel rods_per_beam : ℕ)
    (h1 : panels = 10)
    (h2 : sheets_per_panel = 3)
    (h3 : rods_per_sheet = 10)
    (h4 : rods_needed = 380)
    (h5 : rods_per_beam = 4)
    (h6 : beams_per_panel = 2) :
    (panels * sheets_per_panel * rods_per_sheet + panels * beams_per_panel * rods_per_beam = rods_needed) :=
by
  sorry

end NUMINAMATH_GPT_metal_beams_per_panel_l222_22255


namespace NUMINAMATH_GPT_no_triangles_if_all_horizontal_removed_l222_22227

/-- 
Given a figure that consists of 40 identical toothpicks, making up a symmetric figure with 
additional rows on the top and bottom. We need to prove that removing all 40 horizontal toothpicks 
ensures there are no remaining triangles in the figure.
-/
theorem no_triangles_if_all_horizontal_removed
  (initial_toothpicks : ℕ)
  (horizontal_toothpicks_in_figure : ℕ) 
  (rows : ℕ)
  (top_row : ℕ)
  (second_row : ℕ)
  (third_row : ℕ)
  (fourth_row : ℕ)
  (bottom_row : ℕ)
  (additional_rows : ℕ)
  (triangles_for_upward : ℕ)
  (triangles_for_downward : ℕ):
  initial_toothpicks = 40 →
  horizontal_toothpicks_in_figure = top_row + second_row + third_row + fourth_row + bottom_row →
  rows = 5 →
  top_row = 5 →
  second_row = 10 →
  third_row = 10 →
  fourth_row = 10 →
  bottom_row = 5 →
  additional_rows = 2 →
  triangles_for_upward = 15 →
  triangles_for_downward = 10 →
  horizontal_toothpicks_in_figure = 40 → 
  ∀ toothpicks_removed, toothpicks_removed = 40 →
  no_triangles_remain :=
by
  intros
  sorry

end NUMINAMATH_GPT_no_triangles_if_all_horizontal_removed_l222_22227


namespace NUMINAMATH_GPT_seqAN_81_eq_640_l222_22295

-- Definitions and hypotheses
def seqAN (n : ℕ) : ℝ := sorry   -- A sequence a_n to be defined properly.

def sumSN (n : ℕ) : ℝ := sorry  -- The sum of the first n terms of a_n.

axiom condition_positivity : ∀ n : ℕ, 0 < seqAN n
axiom condition_a1 : seqAN 1 = 1
axiom condition_sum (n : ℕ) (h : 2 ≤ n) : 
  sumSN n * Real.sqrt (sumSN (n-1)) - sumSN (n-1) * Real.sqrt (sumSN n) = 
  2 * Real.sqrt (sumSN n * sumSN (n-1))

-- Proof problem: 
theorem seqAN_81_eq_640 : seqAN 81 = 640 := by sorry

end NUMINAMATH_GPT_seqAN_81_eq_640_l222_22295


namespace NUMINAMATH_GPT_points_after_perfect_games_l222_22290

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end NUMINAMATH_GPT_points_after_perfect_games_l222_22290


namespace NUMINAMATH_GPT_remainder_8927_div_11_l222_22209

theorem remainder_8927_div_11 : 8927 % 11 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_8927_div_11_l222_22209


namespace NUMINAMATH_GPT_prime_diff_of_cubes_sum_of_square_and_three_times_square_l222_22237

theorem prime_diff_of_cubes_sum_of_square_and_three_times_square 
  (p : ℕ) (a b : ℕ) (h_prime : Nat.Prime p) (h_diff : p = a^3 - b^3) :
  ∃ c d : ℤ, p = c^2 + 3 * d^2 := 
  sorry

end NUMINAMATH_GPT_prime_diff_of_cubes_sum_of_square_and_three_times_square_l222_22237


namespace NUMINAMATH_GPT_satellite_modular_units_l222_22268

variable (U N S T : ℕ)

def condition1 : Prop := N = (1/8 : ℝ) * S
def condition2 : Prop := T = 4 * S
def condition3 : Prop := U * N = 3 * S

theorem satellite_modular_units
  (h1 : condition1 N S)
  (h2 : condition2 T S)
  (h3 : condition3 U N S) :
  U = 24 :=
sorry

end NUMINAMATH_GPT_satellite_modular_units_l222_22268


namespace NUMINAMATH_GPT_molecular_weight_of_1_mole_l222_22296

theorem molecular_weight_of_1_mole (m : ℝ) (w : ℝ) (h : 7 * m = 420) : m = 60 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_1_mole_l222_22296


namespace NUMINAMATH_GPT_exponent_division_l222_22281

theorem exponent_division (h1 : 27 = 3^3) : 3^18 / 27^3 = 19683 := by
  sorry

end NUMINAMATH_GPT_exponent_division_l222_22281


namespace NUMINAMATH_GPT_value_of_difference_power_l222_22244

theorem value_of_difference_power (a b : ℝ) (h₁ : a^3 - 6 * a^2 + 15 * a = 9) 
                                  (h₂ : b^3 - 3 * b^2 + 6 * b = -1) 
                                  : (a - b)^2014 = 1 := 
by sorry

end NUMINAMATH_GPT_value_of_difference_power_l222_22244


namespace NUMINAMATH_GPT_algebra_expression_value_l222_22234

theorem algebra_expression_value (a b : ℝ)
  (h1 : |a + 2| = 0)
  (h2 : (b - 5 / 2) ^ 2 = 0) : (2 * a + 3 * b) * (2 * b - 3 * a) = 26 := by
sorry

end NUMINAMATH_GPT_algebra_expression_value_l222_22234


namespace NUMINAMATH_GPT_positive_number_sum_square_eq_210_l222_22284

theorem positive_number_sum_square_eq_210 (x : ℕ) (h1 : x^2 + x = 210) (h2 : 0 < x) (h3 : x < 15) : x = 14 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_sum_square_eq_210_l222_22284


namespace NUMINAMATH_GPT_circle_equation_exists_l222_22251

theorem circle_equation_exists :
  ∃ (x_c y_c r : ℝ), 
  x_c > 0 ∧ y_c > 0 ∧ 0 < r ∧ r < 5 ∧ (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = r^2) :=
sorry

end NUMINAMATH_GPT_circle_equation_exists_l222_22251


namespace NUMINAMATH_GPT_pages_with_same_units_digit_count_l222_22210

def same_units_digit (x : ℕ) (y : ℕ) : Prop :=
  x % 10 = y % 10

theorem pages_with_same_units_digit_count :
  ∃! (n : ℕ), n = 12 ∧ 
  ∀ x, (1 ≤ x ∧ x ≤ 61) → same_units_digit x (62 - x) → 
  (x % 10 = 2 ∨ x % 10 = 7) :=
by
  sorry

end NUMINAMATH_GPT_pages_with_same_units_digit_count_l222_22210


namespace NUMINAMATH_GPT_points_on_circle_l222_22216

theorem points_on_circle (n : ℕ) (h1 : ∃ (k : ℕ), k = (35 - 7) ∧ n = 2 * k) : n = 56 :=
sorry

end NUMINAMATH_GPT_points_on_circle_l222_22216


namespace NUMINAMATH_GPT_simple_interest_calculation_l222_22226

-- Defining the given values
def principal : ℕ := 1500
def rate : ℕ := 7
def time : ℕ := rate -- time is the same as the rate of interest

-- Define the simple interest calculation
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Proof statement
theorem simple_interest_calculation : simple_interest principal rate time = 735 := by
  sorry

end NUMINAMATH_GPT_simple_interest_calculation_l222_22226


namespace NUMINAMATH_GPT_wooden_toys_count_l222_22262

theorem wooden_toys_count :
  ∃ T : ℤ, 
    10 * 40 + 20 * T - (10 * 36 + 17 * T) = 64 ∧ T = 8 :=
by
  use 8
  sorry

end NUMINAMATH_GPT_wooden_toys_count_l222_22262


namespace NUMINAMATH_GPT_cost_of_song_book_l222_22274

-- Define the costs as constants
def cost_trumpet : ℝ := 149.16
def cost_music_tool : ℝ := 9.98
def total_spent : ℝ := 163.28

-- Define the statement to prove
theorem cost_of_song_book : total_spent - (cost_trumpet + cost_music_tool) = 4.14 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_song_book_l222_22274


namespace NUMINAMATH_GPT_domain_of_f_f_is_monotonically_increasing_l222_22297

open Real

noncomputable def f (x : ℝ) : ℝ := tan (2 * x - π / 8) + 3

theorem domain_of_f :
  ∀ x, (x ≠ 5 * π / 16 + k * π / 2) := sorry

theorem f_is_monotonically_increasing :
  ∀ x, (π / 16 < x ∧ x < 3 * π / 16 → f x < f (x + ε)) := sorry

end NUMINAMATH_GPT_domain_of_f_f_is_monotonically_increasing_l222_22297


namespace NUMINAMATH_GPT_systematic_sampling_questionnaire_B_count_l222_22264

theorem systematic_sampling_questionnaire_B_count (n : ℕ) (N : ℕ) (first_random : ℕ) (range_A_start range_A_end range_B_start range_B_end : ℕ) 
  (h1 : n = 32) (h2 : N = 960) (h3 : first_random = 9) (h4 : range_A_start = 1) (h5 : range_A_end = 460) 
  (h6 : range_B_start = 461) (h7 : range_B_end = 761) :
  ∃ count : ℕ, count = 10 := by
  sorry

end NUMINAMATH_GPT_systematic_sampling_questionnaire_B_count_l222_22264


namespace NUMINAMATH_GPT_at_least_one_divisible_by_5_l222_22228

theorem at_least_one_divisible_by_5 (k m n : ℕ) (hk : ¬ (5 ∣ k)) (hm : ¬ (5 ∣ m)) (hn : ¬ (5 ∣ n)) : 
  (5 ∣ (k^2 - m^2)) ∨ (5 ∣ (m^2 - n^2)) ∨ (5 ∣ (n^2 - k^2)) :=
by {
    sorry
}

end NUMINAMATH_GPT_at_least_one_divisible_by_5_l222_22228


namespace NUMINAMATH_GPT_problem_l222_22243

-- Define the functions f and g with their properties
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Express the given conditions in Lean
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom g_def : ∀ x : ℝ, g x = f (x - 1)
axiom f_at_2 : f 2 = 2

-- What we need to prove
theorem problem : f 2014 = 2 := 
by sorry

end NUMINAMATH_GPT_problem_l222_22243


namespace NUMINAMATH_GPT_pentagon_area_correct_l222_22202

-- Define the side lengths of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 22

-- Define the specific angle between the sides of lengths 30 and 28
def angle := 110 -- degrees

-- Define the heights used for the trapezoids and triangle calculations
def height_trapezoid1 := 10
def height_trapezoid2 := 15
def height_triangle := 8

-- Function to calculate the area of a trapezoid
def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

-- Function to calculate the area of a triangle
def triangle_area (base height : ℕ) : ℕ :=
  base * height / 2

-- Calculation of individual areas
def area_trapezoid1 := trapezoid_area side1 side2 height_trapezoid1
def area_trapezoid2 := trapezoid_area side3 side4 height_trapezoid2
def area_triangle := triangle_area side5 height_triangle

-- Total area calculation
def total_area := area_trapezoid1 + area_trapezoid2 + area_triangle

-- Expected total area
def expected_area := 738

-- Lean statement to assert the total area equals the expected value
theorem pentagon_area_correct :
  total_area = expected_area :=
by sorry

end NUMINAMATH_GPT_pentagon_area_correct_l222_22202


namespace NUMINAMATH_GPT_trig_expression_l222_22220

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end NUMINAMATH_GPT_trig_expression_l222_22220


namespace NUMINAMATH_GPT_amanda_family_painting_theorem_l222_22267

theorem amanda_family_painting_theorem
  (rooms_with_4_walls : ℕ)
  (walls_per_room_with_4_walls : ℕ)
  (rooms_with_5_walls : ℕ)
  (walls_per_room_with_5_walls : ℕ)
  (walls_per_person : ℕ)
  (total_rooms : ℕ)
  (h1 : rooms_with_4_walls = 5)
  (h2 : walls_per_room_with_4_walls = 4)
  (h3 : rooms_with_5_walls = 4)
  (h4 : walls_per_room_with_5_walls = 5)
  (h5 : walls_per_person = 8)
  (h6 : total_rooms = 9)
  : rooms_with_4_walls * walls_per_room_with_4_walls +
    rooms_with_5_walls * walls_per_room_with_5_walls =
    5 * walls_per_person :=
by
  sorry

end NUMINAMATH_GPT_amanda_family_painting_theorem_l222_22267


namespace NUMINAMATH_GPT_smallest_solution_abs_eq_20_l222_22292

theorem smallest_solution_abs_eq_20 : ∃ x : ℝ, x = -7 ∧ |4 * x + 8| = 20 ∧ (∀ y : ℝ, |4 * y + 8| = 20 → x ≤ y) :=
by
  sorry

end NUMINAMATH_GPT_smallest_solution_abs_eq_20_l222_22292


namespace NUMINAMATH_GPT_first_batch_students_l222_22229

theorem first_batch_students 
  (x : ℕ) 
  (avg1 avg2 avg3 overall_avg : ℝ) 
  (n2 n3 : ℕ) 
  (h_avg1 : avg1 = 45) 
  (h_avg2 : avg2 = 55) 
  (h_avg3 : avg3 = 65) 
  (h_n2 : n2 = 50) 
  (h_n3 : n3 = 60) 
  (h_overall_avg : overall_avg = 56.333333333333336) 
  (h_eq : overall_avg = (45 * x + 55 * 50 + 65 * 60) / (x + 50 + 60)) 
  : x = 40 :=
sorry

end NUMINAMATH_GPT_first_batch_students_l222_22229


namespace NUMINAMATH_GPT_basketball_points_total_l222_22254

variable (Tobee_points Jay_points Sean_points Remy_points Alex_points : ℕ)

def conditions := 
  Tobee_points = 4 ∧
  Jay_points = 2 * Tobee_points + 6 ∧
  Sean_points = Jay_points / 2 ∧
  Remy_points = Tobee_points + Jay_points - 3 ∧
  Alex_points = Sean_points + Remy_points + 4

theorem basketball_points_total 
  (h : conditions Tobee_points Jay_points Sean_points Remy_points Alex_points) :
  Tobee_points + Jay_points + Sean_points + Remy_points + Alex_points = 66 :=
by sorry

end NUMINAMATH_GPT_basketball_points_total_l222_22254


namespace NUMINAMATH_GPT_question1_question2_l222_22232

section

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition 1: A = {x | -1 ≤ x < 3}
def setA : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Condition 2: B = {x | 2x - 4 ≥ x - 2}
def setB : Set ℝ := {x | x ≥ 2}

-- Condition 3: C = {x | x ≥ a - 1}
def setC (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Question 1: Prove A ∩ B = {x | 2 ≤ x < 3}
theorem question1 : A = setA → B = setB → A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by intros hA hB; rw [hA, hB]; sorry

-- Question 2: If B ∪ C = C, prove a ∈ (-∞, 3]
theorem question2 : B = setB → C = setC a → (B ∪ C = C) → a ≤ 3 :=
by intros hB hC hBUC; rw [hB, hC] at hBUC; sorry

end

end NUMINAMATH_GPT_question1_question2_l222_22232


namespace NUMINAMATH_GPT_range_of_a_l222_22236

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 - a * x + 1 > 0) ↔ -2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l222_22236


namespace NUMINAMATH_GPT_leak_empties_in_24_hours_l222_22286

noncomputable def tap_rate := 1 / 6
noncomputable def combined_rate := 1 / 8
noncomputable def leak_rate := tap_rate - combined_rate
noncomputable def time_to_empty := 1 / leak_rate

theorem leak_empties_in_24_hours :
  time_to_empty = 24 := by
  sorry

end NUMINAMATH_GPT_leak_empties_in_24_hours_l222_22286


namespace NUMINAMATH_GPT_factorize_expression_l222_22272

variables (a b x : ℝ)

theorem factorize_expression :
    5 * a * (x^2 - 1) - 5 * b * (x^2 - 1) = 5 * (x + 1) * (x - 1) * (a - b) := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l222_22272


namespace NUMINAMATH_GPT_smallest_n_for_pencil_purchase_l222_22293

theorem smallest_n_for_pencil_purchase (a b c d n : ℕ)
  (h1 : 6 * a + 10 * b = n)
  (h2 : 6 * c + 10 * d = n + 2)
  (h3 : 7 * a + 12 * b > 7 * c + 12 * d)
  (h4 : 3 * (c - a) + 5 * (d - b) = 1)
  (h5 : d - b > 0) :
  n = 100 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_pencil_purchase_l222_22293


namespace NUMINAMATH_GPT_curve_not_parabola_l222_22224

theorem curve_not_parabola (k : ℝ) : ¬ ∃ (x y : ℝ), (x^2 + k * y^2 = 1) ↔ (k = -y / x) :=
by
  sorry

end NUMINAMATH_GPT_curve_not_parabola_l222_22224


namespace NUMINAMATH_GPT_compare_cube_roots_l222_22270

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem compare_cube_roots : 2 + cube_root 7 < cube_root 60 :=
sorry

end NUMINAMATH_GPT_compare_cube_roots_l222_22270


namespace NUMINAMATH_GPT_solution_exists_unique_l222_22273

theorem solution_exists_unique (x y : ℝ) : (x + y = 2 ∧ x - y = 0) ↔ (x = 1 ∧ y = 1) := 
by
  sorry

end NUMINAMATH_GPT_solution_exists_unique_l222_22273


namespace NUMINAMATH_GPT_ratio_of_spending_is_one_to_two_l222_22241

-- Definitions
def initial_amount : ℕ := 24
def doris_spent : ℕ := 6
def final_amount : ℕ := 15

-- Amount remaining after Doris spent
def remaining_after_doris : ℕ := initial_amount - doris_spent

-- Amount Martha spent
def martha_spent : ℕ := remaining_after_doris - final_amount

-- Ratio of the amounts spent
def ratio_martha_doris : ℕ × ℕ := (martha_spent, doris_spent)

-- Theorem to prove
theorem ratio_of_spending_is_one_to_two : ratio_martha_doris = (1, 2) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_ratio_of_spending_is_one_to_two_l222_22241


namespace NUMINAMATH_GPT_problem_l222_22246

def f (x : ℚ) : ℚ :=
  x⁻¹ - (x⁻¹ / (1 - x⁻¹))

theorem problem : f (f (-3)) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_l222_22246


namespace NUMINAMATH_GPT_unique_prime_value_l222_22289

theorem unique_prime_value :
  ∃! n : ℕ, n > 0 ∧ Nat.Prime (n^3 - 7 * n^2 + 17 * n - 11) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_prime_value_l222_22289


namespace NUMINAMATH_GPT_smallest_sum_of_consecutive_integers_is_square_l222_22200

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_consecutive_integers_is_square_l222_22200


namespace NUMINAMATH_GPT_min_val_z_is_7_l222_22257

noncomputable def min_val_z (x y : ℝ) (h : x + 3 * y = 2) : ℝ := 3^x + 27^y + 1

theorem min_val_z_is_7  : ∃ x y : ℝ, x + 3 * y = 2 ∧ min_val_z x y (by sorry) = 7 := sorry

end NUMINAMATH_GPT_min_val_z_is_7_l222_22257


namespace NUMINAMATH_GPT_infinitely_many_a_not_sum_of_seven_sixth_powers_l222_22207

theorem infinitely_many_a_not_sum_of_seven_sixth_powers :
  ∃ᶠ (a: ℕ) in at_top, (∀ (a_i : ℕ) (h0 : a_i > 0), a ≠ a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 ∧ a % 9 = 8) :=
sorry

end NUMINAMATH_GPT_infinitely_many_a_not_sum_of_seven_sixth_powers_l222_22207


namespace NUMINAMATH_GPT_original_numbers_l222_22201

theorem original_numbers (a b c d : ℝ) (h1 : a + b + c + d = 45)
    (h2 : ∃ x : ℝ, a + 2 = x ∧ b - 2 = x ∧ 2 * c = x ∧ d / 2 = x) : 
    a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 :=
by
  sorry

end NUMINAMATH_GPT_original_numbers_l222_22201


namespace NUMINAMATH_GPT_complex_division_l222_22230

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (2 / (1 + i)) = (1 - i) :=
by
  sorry

end NUMINAMATH_GPT_complex_division_l222_22230


namespace NUMINAMATH_GPT_arrangement_count_l222_22217

noncomputable def count_arrangements (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  sorry -- The implementation of this function is out of scope for this task

theorem arrangement_count :
  count_arrangements ({1, 2, 3, 4} : Finset ℕ) ({1, 2, 3} : Finset ℕ) = 18 :=
sorry

end NUMINAMATH_GPT_arrangement_count_l222_22217


namespace NUMINAMATH_GPT_Juanita_weekday_spending_l222_22215

/- Defining the variables and conditions in the problem -/

def Grant_spending : ℝ := 200
def Sunday_spending : ℝ := 2
def extra_spending : ℝ := 60

-- We need to prove that Juanita spends $0.50 per day from Monday through Saturday on newspapers.

theorem Juanita_weekday_spending :
  (∃ x : ℝ, 6 * 52 * x + 52 * 2 = Grant_spending + extra_spending) -> (∃ x : ℝ, x = 0.5) := by {
  sorry
}

end NUMINAMATH_GPT_Juanita_weekday_spending_l222_22215


namespace NUMINAMATH_GPT_fourth_group_trees_l222_22238

theorem fourth_group_trees (x : ℕ) :
  5 * 13 = 12 + 15 + 12 + x + 11 → x = 15 :=
by
  sorry

end NUMINAMATH_GPT_fourth_group_trees_l222_22238


namespace NUMINAMATH_GPT_departure_sequences_count_l222_22240

noncomputable def total_departure_sequences (trains: Finset ℕ) (A B : ℕ) 
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6) 
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : ℕ := 6 * 6 * 6

-- The main theorem statement: given the conditions, prove the total number of different sequences is 216
theorem departure_sequences_count (trains: Finset ℕ) (A B : ℕ)
  (h : A ∈ trains ∧ B ∈ trains ∧ trains.card = 6)
  (hAB : ∀ g1 g2 : Finset ℕ, g1 ∪ g2 = trains ∧ g1.card = 3 ∧ g2.card = 3 → ¬(A ∈ g1 ∧ B ∈ g1 ∨ A ∈ g2 ∧ B ∈ g2)) 
  : total_departure_sequences trains A B h hAB = 216 := 
by 
  sorry

end NUMINAMATH_GPT_departure_sequences_count_l222_22240


namespace NUMINAMATH_GPT_mileage_per_gallon_l222_22235

-- Definitions for the conditions
def total_distance_to_grandma (d : ℕ) : Prop := d = 100
def gallons_to_grandma (g : ℕ) : Prop := g = 5

-- The statement to be proved
theorem mileage_per_gallon :
  ∀ (d g m : ℕ), total_distance_to_grandma d → gallons_to_grandma g → m = d / g → m = 20 :=
sorry

end NUMINAMATH_GPT_mileage_per_gallon_l222_22235


namespace NUMINAMATH_GPT_jills_uncles_medicine_last_time_l222_22259

theorem jills_uncles_medicine_last_time :
  let pills := 90
  let third_of_pill_days := 3
  let days_per_full_pill := 9
  let days_per_month := 30
  let total_days := pills * days_per_full_pill
  let total_months := total_days / days_per_month
  total_months = 27 :=
by {
  sorry
}

end NUMINAMATH_GPT_jills_uncles_medicine_last_time_l222_22259


namespace NUMINAMATH_GPT_average_apples_per_hour_l222_22277

theorem average_apples_per_hour (A H : ℝ) (hA : A = 12) (hH : H = 5) : A / H = 2.4 := by
  -- sorry skips the proof
  sorry

end NUMINAMATH_GPT_average_apples_per_hour_l222_22277


namespace NUMINAMATH_GPT_count_perfect_squares_mul_36_l222_22256

theorem count_perfect_squares_mul_36 (n : ℕ) (h1 : n < 10^7) (h2 : ∃k, n = k^2) (h3 : 36 ∣ n) :
  ∃ m : ℕ, m = 263 :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_squares_mul_36_l222_22256


namespace NUMINAMATH_GPT_number_of_girls_l222_22204

variable (g b : ℕ) -- Number of girls (g) and boys (b) in the class
variable (h_ratio : g / b = 4 / 3) -- The ratio condition
variable (h_total : g + b = 63) -- The total number of students condition

theorem number_of_girls (g b : ℕ) (h_ratio : g / b = 4 / 3) (h_total : g + b = 63) :
    g = 36 :=
sorry

end NUMINAMATH_GPT_number_of_girls_l222_22204


namespace NUMINAMATH_GPT_max_possible_N_l222_22221

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end NUMINAMATH_GPT_max_possible_N_l222_22221


namespace NUMINAMATH_GPT_puzzle_pieces_l222_22252

theorem puzzle_pieces
  (total_puzzles : ℕ)
  (pieces_per_10_min : ℕ)
  (total_minutes : ℕ)
  (h1 : total_puzzles = 2)
  (h2 : pieces_per_10_min = 100)
  (h3 : total_minutes = 400) :
  ((total_minutes / 10) * pieces_per_10_min) / total_puzzles = 2000 :=
by
  sorry

end NUMINAMATH_GPT_puzzle_pieces_l222_22252


namespace NUMINAMATH_GPT_total_truck_loads_needed_l222_22212

noncomputable def truck_loads_of_material : ℝ :=
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5 -- log is the natural logarithm in Lean
  sand + dirt + cement + gravel

theorem total_truck_loads_needed : truck_loads_of_material = 1.8401374808985008 := by
  sorry

end NUMINAMATH_GPT_total_truck_loads_needed_l222_22212


namespace NUMINAMATH_GPT_equation_of_l_symmetric_point_l222_22250

/-- Define points O, A, B in the coordinate plane --/
def O := (0, 0)
def A := (2, 0)
def B := (3, 2)

/-- Define midpoint of OA --/
def midpoint_OA := ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

/-- Line l passes through midpoint_OA and B. Prove line l has equation y = x - 1 --/
theorem equation_of_l :
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

/-- Prove the symmetric point of A with respect to line l is (1, 1) --/
theorem symmetric_point :
  ∃ (a b : ℝ), (a, b) = (1, 1) ∧
                (b * (2 - 1)) / (a - 2) = -1 ∧
                b / 2 = (2 + a - 1) / 2 - 1 :=
sorry

end NUMINAMATH_GPT_equation_of_l_symmetric_point_l222_22250


namespace NUMINAMATH_GPT_proof_of_value_of_6y_plus_3_l222_22225

theorem proof_of_value_of_6y_plus_3 (y : ℤ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 :=
by
  sorry

end NUMINAMATH_GPT_proof_of_value_of_6y_plus_3_l222_22225


namespace NUMINAMATH_GPT_calculate_retail_price_l222_22248

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end NUMINAMATH_GPT_calculate_retail_price_l222_22248


namespace NUMINAMATH_GPT_volume_of_region_l222_22269

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

theorem volume_of_region (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) :
  volume_of_sphere r_large - volume_of_sphere r_small = 372 * Real.pi := by
  rw [h_small, h_large]
  sorry

end NUMINAMATH_GPT_volume_of_region_l222_22269


namespace NUMINAMATH_GPT_cars_parked_l222_22223

def front_parking_spaces : ℕ := 52
def back_parking_spaces : ℕ := 38
def filled_back_spaces : ℕ := back_parking_spaces / 2
def available_spaces : ℕ := 32
def total_parking_spaces : ℕ := front_parking_spaces + back_parking_spaces
def filled_spaces : ℕ := total_parking_spaces - available_spaces

theorem cars_parked : 
  filled_spaces = 58 := by
  sorry

end NUMINAMATH_GPT_cars_parked_l222_22223


namespace NUMINAMATH_GPT_divisibility_condition_l222_22294

theorem divisibility_condition (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1) ∣ ((a + 1)^n) ↔ (a = 1 ∧ 1 ≤ m ∧ 1 ≤ n) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n) := 
by 
  sorry

end NUMINAMATH_GPT_divisibility_condition_l222_22294


namespace NUMINAMATH_GPT_least_possible_number_l222_22206

theorem least_possible_number (k : ℕ) (n : ℕ) (r : ℕ) (h1 : k = 34 * n + r) 
  (h2 : k / 5 = r + 8) (h3 : r < 34) : k = 68 :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_least_possible_number_l222_22206


namespace NUMINAMATH_GPT_units_digit_of_sum_sequence_is_8_l222_22219

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit_sum_sequence : ℕ :=
  let term (n : ℕ) := (factorial n + n * n) % 10
  (term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7 + term 8 + term 9) % 10

theorem units_digit_of_sum_sequence_is_8 :
  units_digit_sum_sequence = 8 :=
sorry

end NUMINAMATH_GPT_units_digit_of_sum_sequence_is_8_l222_22219


namespace NUMINAMATH_GPT_zoo_ticket_sales_l222_22253

-- Define the number of total people, number of adults, and ticket prices
def total_people : ℕ := 254
def num_adults : ℕ := 51
def adult_ticket_price : ℕ := 28
def kid_ticket_price : ℕ := 12

-- Define the number of kids as the difference between total people and number of adults
def num_kids : ℕ := total_people - num_adults

-- Define the revenue from adult tickets and kid tickets
def revenue_adult_tickets : ℕ := num_adults * adult_ticket_price
def revenue_kid_tickets : ℕ := num_kids * kid_ticket_price

-- Define the total revenue
def total_revenue : ℕ := revenue_adult_tickets + revenue_kid_tickets

-- Theorem to prove the total revenue equals 3864
theorem zoo_ticket_sales : total_revenue = 3864 :=
  by {
    -- sorry allows us to skip the proof
    sorry
  }

end NUMINAMATH_GPT_zoo_ticket_sales_l222_22253


namespace NUMINAMATH_GPT_oak_trees_problem_l222_22233

theorem oak_trees_problem (c t n : ℕ) 
  (h1 : c = 9) 
  (h2 : t = 11) 
  (h3 : t = c + n) 
  : n = 2 := 
by 
  sorry

end NUMINAMATH_GPT_oak_trees_problem_l222_22233
