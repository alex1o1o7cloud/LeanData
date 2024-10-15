import Mathlib

namespace NUMINAMATH_GPT_derivative_of_y_correct_l883_88373

noncomputable def derivative_of_y (x : ℝ) : ℝ :=
  let y := (4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))) / (16 + (Real.log 4) ^ 2)
  let u := 4^x * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x))
  let v := 16 + (Real.log 4) ^ 2
  let du_dx := (4^x * Real.log 4) * (Real.log 4 * Real.sin (4 * x) - 4 * Real.cos (4 * x)) +
               (4^x) * (4 * Real.log 4 * Real.cos (4 * x) + 16 * Real.sin (4 * x))
  let dv_dx := 0
  (du_dx * v - u * dv_dx) / (v ^ 2)

theorem derivative_of_y_correct (x : ℝ) : derivative_of_y x = 4^x * Real.sin (4 * x) :=
  sorry

end NUMINAMATH_GPT_derivative_of_y_correct_l883_88373


namespace NUMINAMATH_GPT_solution_set_of_inequality_l883_88387

theorem solution_set_of_inequality (x : ℝ) (hx : x ≠ 0) :
  (x + 1) / x ≤ 3 ↔ x ∈ Set.Iio 0 ∪ Set.Ici 0.5 :=
by sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l883_88387


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l883_88311

theorem sum_of_other_endpoint_coordinates 
  (A B O : ℝ × ℝ)
  (hA : A = (6, -2)) 
  (hO : O = (3, 5)) 
  (midpoint_formula : (A.1 + B.1) / 2 = O.1 ∧ (A.2 + B.2) / 2 = O.2):
  (B.1 + B.2) = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l883_88311


namespace NUMINAMATH_GPT_tan_neg_225_is_neg_1_l883_88391

def tan_neg_225_eq_neg_1 : Prop :=
  Real.tan (-225 * Real.pi / 180) = -1

theorem tan_neg_225_is_neg_1 : tan_neg_225_eq_neg_1 :=
  by
    sorry

end NUMINAMATH_GPT_tan_neg_225_is_neg_1_l883_88391


namespace NUMINAMATH_GPT_x_equals_neg_one_l883_88392

theorem x_equals_neg_one
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : (a + b - c) / c = (a - b + c) / b ∧ (a + b - c) / c = (-a + b + c) / a)
  (x : ℝ)
  (h5 : x = (a + b) * (b + c) * (c + a) / (a * b * c))
  (h6 : x < 0) :
  x = -1 := 
sorry

end NUMINAMATH_GPT_x_equals_neg_one_l883_88392


namespace NUMINAMATH_GPT_appropriate_survey_method_l883_88340

def survey_method_suitability (method : String) (context : String) : Prop :=
  match context, method with
  | "daily floating population of our city", "sampling survey" => true
  | "security checks before passengers board an airplane", "comprehensive survey" => true
  | "killing radius of a batch of shells", "sampling survey" => true
  | "math scores of Class 1 in Grade 7 of a certain school", "census method" => true
  | _, _ => false

theorem appropriate_survey_method :
  survey_method_suitability "census method" "daily floating population of our city" = false ∧
  survey_method_suitability "comprehensive survey" "security checks before passengers board an airplane" = false ∧
  survey_method_suitability "sampling survey" "killing radius of a batch of shells" = false ∧
  survey_method_suitability "census method" "math scores of Class 1 in Grade 7 of a certain school" = true :=
by
  sorry

end NUMINAMATH_GPT_appropriate_survey_method_l883_88340


namespace NUMINAMATH_GPT_time_per_harvest_is_three_months_l883_88395

variable (area : ℕ) (trees_per_m2 : ℕ) (coconuts_per_tree : ℕ) 
variable (price_per_coconut : ℚ) (total_earning_6_months : ℚ)

theorem time_per_harvest_is_three_months 
  (h1 : area = 20) 
  (h2 : trees_per_m2 = 2) 
  (h3 : coconuts_per_tree = 6) 
  (h4 : price_per_coconut = 0.50) 
  (h5 : total_earning_6_months = 240) :
    (6 / (total_earning_6_months / (area * trees_per_m2 * coconuts_per_tree * price_per_coconut)) = 3) := 
  by 
    sorry

end NUMINAMATH_GPT_time_per_harvest_is_three_months_l883_88395


namespace NUMINAMATH_GPT_arithmetic_sequence_find_m_l883_88396

theorem arithmetic_sequence_find_m (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_find_m_l883_88396


namespace NUMINAMATH_GPT_salary_restoration_l883_88374

theorem salary_restoration (S : ℝ) : 
  let reduced_salary := 0.7 * S
  let restore_factor := 1 / 0.7
  let percentage_increase := restore_factor - 1
  percentage_increase * 100 = 42.857 :=
by
  sorry

end NUMINAMATH_GPT_salary_restoration_l883_88374


namespace NUMINAMATH_GPT_find_y_l883_88331

theorem find_y (x y : ℝ) (h1 : x = 8) (h2 : x^(3 * y) = 64) : y = 2 / 3 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_y_l883_88331


namespace NUMINAMATH_GPT_urn_contains_specific_balls_after_operations_l883_88332

def initial_red_balls : ℕ := 2
def initial_blue_balls : ℕ := 1
def total_operations : ℕ := 5
def final_red_balls : ℕ := 10
def final_blue_balls : ℕ := 6
def target_probability : ℚ := 16 / 115

noncomputable def urn_proba_result : ℚ := sorry

theorem urn_contains_specific_balls_after_operations :
  urn_proba_result = target_probability := sorry

end NUMINAMATH_GPT_urn_contains_specific_balls_after_operations_l883_88332


namespace NUMINAMATH_GPT_unique_alphabets_count_l883_88351

theorem unique_alphabets_count
  (total_alphabets : ℕ)
  (each_written_times : ℕ)
  (total_written : total_alphabets * each_written_times = 10) :
  total_alphabets = 5 := by
  -- The proof would be filled in here.
  sorry

end NUMINAMATH_GPT_unique_alphabets_count_l883_88351


namespace NUMINAMATH_GPT_find_prices_l883_88381

def price_system_of_equations (x y : ℕ) : Prop :=
  3 * x + 2 * y = 474 ∧ x - y = 8

theorem find_prices (x y : ℕ) :
  price_system_of_equations x y :=
by
  sorry

end NUMINAMATH_GPT_find_prices_l883_88381


namespace NUMINAMATH_GPT_fish_to_rice_value_l883_88366

variable (f l r : ℝ)

theorem fish_to_rice_value (h1 : 5 * f = 3 * l) (h2 : 2 * l = 7 * r) : f = 2.1 * r :=
by
  sorry

end NUMINAMATH_GPT_fish_to_rice_value_l883_88366


namespace NUMINAMATH_GPT_price_of_soda_l883_88345

-- Definitions based on the conditions given in the problem
def initial_amount := 500
def cost_rice := 2 * 20
def cost_wheat_flour := 3 * 25
def remaining_balance := 235
def total_cost := cost_rice + cost_wheat_flour

-- Definition to be proved
theorem price_of_soda : initial_amount - total_cost - remaining_balance = 150 := by
  sorry

end NUMINAMATH_GPT_price_of_soda_l883_88345


namespace NUMINAMATH_GPT_intersection_A_B_l883_88316

def setA : Set ℝ := { x | x^2 - 2*x < 3 }
def setB : Set ℝ := { x | x ≤ 2 }
def setC : Set ℝ := { x | -1 < x ∧ x ≤ 2 }

theorem intersection_A_B :
  (setA ∩ setB) = setC :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l883_88316


namespace NUMINAMATH_GPT_number_of_families_l883_88333

theorem number_of_families (x : ℕ) (h1 : x + x / 3 = 100) : x = 75 :=
sorry

end NUMINAMATH_GPT_number_of_families_l883_88333


namespace NUMINAMATH_GPT_ashley_friends_ages_correct_sum_l883_88344

noncomputable def ashley_friends_ages_sum : Prop :=
  ∃ (a b c d : ℕ), (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
                   (1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9) ∧
                   (a * b = 36) ∧ (c * d = 30) ∧ (a + b + c + d = 24)

theorem ashley_friends_ages_correct_sum : ashley_friends_ages_sum := sorry

end NUMINAMATH_GPT_ashley_friends_ages_correct_sum_l883_88344


namespace NUMINAMATH_GPT_number_of_blue_pens_minus_red_pens_is_seven_l883_88307

-- Define the problem conditions in Lean
variable (R B K T : ℕ) -- where R is red pens, B is black pens, K is blue pens, T is total pens

-- Define the hypotheses from the problem conditions
def hypotheses :=
  (R = 8) ∧ 
  (B = R + 10) ∧ 
  (T = 41) ∧ 
  (T = R + B + K)

-- Define the theorem we need to prove based on the question and the correct answer
theorem number_of_blue_pens_minus_red_pens_is_seven : 
  hypotheses R B K T → K - R = 7 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_number_of_blue_pens_minus_red_pens_is_seven_l883_88307


namespace NUMINAMATH_GPT_find_pairs_l883_88313

theorem find_pairs (a b : ℤ) (ha : a ≥ 1) (hb : b ≥ 1)
  (h1 : (a^2 + b) % (b^2 - a) = 0) 
  (h2 : (b^2 + a) % (a^2 - b) = 0) :
  (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) ∨ (a = 1 ∧ b = 2) ∨ 
  (a = 2 ∧ b = 1) ∨ (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) := 
sorry

end NUMINAMATH_GPT_find_pairs_l883_88313


namespace NUMINAMATH_GPT_second_box_probability_nth_box_probability_l883_88309

noncomputable def P_A1 : ℚ := 2 / 3
noncomputable def P_A2 : ℚ := 5 / 9
noncomputable def P_An (n : ℕ) : ℚ :=
  1 / 2 * (1 / 3) ^ n + 1 / 2

theorem second_box_probability :
  P_A2 = 5 / 9 := by
  sorry

theorem nth_box_probability (n : ℕ) :
  P_An n = 1 / 2 * (1 / 3) ^ n + 1 / 2 := by
  sorry

end NUMINAMATH_GPT_second_box_probability_nth_box_probability_l883_88309


namespace NUMINAMATH_GPT_triangle_area_l883_88325

def point := (ℚ × ℚ)

def vertex1 : point := (3, -3)
def vertex2 : point := (3, 4)
def vertex3 : point := (8, -3)

theorem triangle_area :
  let base := (vertex3.1 - vertex1.1 : ℚ)
  let height := (vertex2.2 - vertex1.2 : ℚ)
  (base * height / 2) = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_l883_88325


namespace NUMINAMATH_GPT_quadratic_minimum_val_l883_88318

theorem quadratic_minimum_val (p q x : ℝ) (hp : p > 0) (hq : q > 0) : 
  (∀ x, x^2 - 2 * p * x + 4 * q ≥ p^2 - 4 * q) := 
by
  sorry

end NUMINAMATH_GPT_quadratic_minimum_val_l883_88318


namespace NUMINAMATH_GPT_sum_real_imag_parts_l883_88327

noncomputable section

open Complex

theorem sum_real_imag_parts (z : ℂ) (h : z / (1 + 2 * i) = 2 + i) : 
  ((z + 5).re + (z + 5).im) = 0 :=
  by
  sorry

end NUMINAMATH_GPT_sum_real_imag_parts_l883_88327


namespace NUMINAMATH_GPT_equation_B_no_real_solution_l883_88341

theorem equation_B_no_real_solution : ∀ x : ℝ, |3 * x + 1| + 6 ≠ 0 := 
by 
  sorry

end NUMINAMATH_GPT_equation_B_no_real_solution_l883_88341


namespace NUMINAMATH_GPT_log2_bounds_l883_88300

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_bounds (h1 : 10^3 = 1000) (h2 : 10^4 = 10000) 
  (h3 : 2^10 = 1024) (h4 : 2^11 = 2048) (h5 : 2^12 = 4096) 
  (h6 : 2^13 = 8192) (h7 : 2^14 = 16384) :
  (3 : ℝ) / 10 < log2 10 ∧ log2 10 < (2 : ℝ) / 7 :=
by
  sorry

end NUMINAMATH_GPT_log2_bounds_l883_88300


namespace NUMINAMATH_GPT_pow_mod_equiv_l883_88359

theorem pow_mod_equiv (h : 5^500 ≡ 1 [MOD 1250]) : 5^15000 ≡ 1 [MOD 1250] := 
by 
  sorry

end NUMINAMATH_GPT_pow_mod_equiv_l883_88359


namespace NUMINAMATH_GPT_multiply_digits_correctness_l883_88394

theorem multiply_digits_correctness (a b c : ℕ) :
  (10 * a + b) * (10 * a + c) = 10 * a * (10 * a + c + b) + b * c :=
by sorry

end NUMINAMATH_GPT_multiply_digits_correctness_l883_88394


namespace NUMINAMATH_GPT_eval_expression_l883_88335

theorem eval_expression : (-3)^5 + 2^(2^3 + 5^2 - 8^2) = -242.999999999535 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l883_88335


namespace NUMINAMATH_GPT_quadratic_pos_implies_a_gt_1_l883_88385

theorem quadratic_pos_implies_a_gt_1 {a : ℝ} :
  (∀ x : ℝ, x^2 + 2 * x + a > 0) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_pos_implies_a_gt_1_l883_88385


namespace NUMINAMATH_GPT_student_knows_german_l883_88367

-- Definitions for each classmate's statement
def classmate1 (lang: String) : Prop := lang ≠ "French"
def classmate2 (lang: String) : Prop := lang = "Spanish" ∨ lang = "German"
def classmate3 (lang: String) : Prop := lang = "Spanish"

-- Conditions: at least one correct and at least one incorrect
def at_least_one_correct (lang: String) : Prop :=
  classmate1 lang ∨ classmate2 lang ∨ classmate3 lang

def at_least_one_incorrect (lang: String) : Prop :=
  ¬classmate1 lang ∨ ¬classmate2 lang ∨ ¬classmate3 lang

-- The statement to prove
theorem student_knows_german : ∀ lang : String,
  at_least_one_correct lang → at_least_one_incorrect lang → lang = "German" :=
by
  intros lang Hcorrect Hincorrect
  revert Hcorrect Hincorrect
  -- sorry stands in place of direct proof
  sorry

end NUMINAMATH_GPT_student_knows_german_l883_88367


namespace NUMINAMATH_GPT_division_of_5_parts_division_of_7_parts_division_of_8_parts_l883_88338

-- Problem 1: Primary Division of Square into 5 Equal Parts
theorem division_of_5_parts (x : ℝ) (h : x^2 = 1 / 5) : x = Real.sqrt (1 / 5) :=
sorry

-- Problem 2: Primary Division of Square into 7 Equal Parts
theorem division_of_7_parts (x : ℝ) (hx : 196 * x^3 - 294 * x^2 + 128 * x - 15 = 0) : 
  x = (7 + Real.sqrt 19) / 14 :=
sorry

-- Problem 3: Primary Division of Square into 8 Equal Parts
theorem division_of_8_parts (x : ℝ) (hx : 6 * x^2 - 6 * x + 1 = 0) : 
  x = (3 + Real.sqrt 3) / 6 :=
sorry

end NUMINAMATH_GPT_division_of_5_parts_division_of_7_parts_division_of_8_parts_l883_88338


namespace NUMINAMATH_GPT_yoque_payment_months_l883_88301

-- Define the conditions
def monthly_payment : ℝ := 15
def amount_borrowed : ℝ := 150
def total_payment : ℝ := amount_borrowed * 1.1

-- Define the proof problem
theorem yoque_payment_months :
  ∃ (n : ℕ), n * monthly_payment = total_payment :=
by 
  have monthly_payment : ℝ := 15
  have amount_borrowed : ℝ := 150
  have total_payment : ℝ := amount_borrowed * 1.1
  use 11
  sorry

end NUMINAMATH_GPT_yoque_payment_months_l883_88301


namespace NUMINAMATH_GPT_increasing_function_range_l883_88324

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 1 then
  -x^2 - a*x - 5
else
  a / x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ (-3 ≤ a ∧ a ≤ -2) :=
by
  sorry

end NUMINAMATH_GPT_increasing_function_range_l883_88324


namespace NUMINAMATH_GPT_ice_cream_melting_l883_88393

theorem ice_cream_melting :
  ∀ (r1 r2 : ℝ) (h : ℝ),
    r1 = 3 ∧ r2 = 10 →
    4 / 3 * π * r1^3 = π * r2^2 * h →
    h = 9 / 25 :=
by intros r1 r2 h hcond voldist
   sorry

end NUMINAMATH_GPT_ice_cream_melting_l883_88393


namespace NUMINAMATH_GPT_reduction_rate_equation_l883_88329

-- Define the given conditions
def original_price : ℝ := 23
def reduced_price : ℝ := 18.63
def monthly_reduction_rate (x : ℝ) : ℝ := (1 - x) ^ 2

-- Prove that the given equation holds
theorem reduction_rate_equation (x : ℝ) : 
  original_price * monthly_reduction_rate x = reduced_price :=
by
  sorry

end NUMINAMATH_GPT_reduction_rate_equation_l883_88329


namespace NUMINAMATH_GPT_ways_to_divide_week_l883_88320

-- Define the total number of seconds in a week
def total_seconds_in_week : ℕ := 604800

-- Define the math problem statement
theorem ways_to_divide_week (n m : ℕ) (h : n * m = total_seconds_in_week) (hn : 0 < n) (hm : 0 < m) : 
  (∃ (n_pairs : ℕ), n_pairs = 144) :=
sorry

end NUMINAMATH_GPT_ways_to_divide_week_l883_88320


namespace NUMINAMATH_GPT_unique_solution_for_2_pow_m_plus_1_eq_n_square_l883_88315

theorem unique_solution_for_2_pow_m_plus_1_eq_n_square (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  2 ^ m + 1 = n ^ 2 → (m = 3 ∧ n = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_unique_solution_for_2_pow_m_plus_1_eq_n_square_l883_88315


namespace NUMINAMATH_GPT_max_tiles_on_floor_l883_88342

-- Definitions based on the given conditions
def tile_length1 := 35 -- in cm
def tile_length2 := 30 -- in cm
def floor_length := 1000 -- in cm
def floor_width := 210 -- in cm

-- Lean 4 statement for the proof problem
theorem max_tiles_on_floor : 
  (max ((floor_length / tile_length1) * (floor_width / tile_length2))
       ((floor_length / tile_length2) * (floor_width / tile_length1))) = 198 := by
  sorry

end NUMINAMATH_GPT_max_tiles_on_floor_l883_88342


namespace NUMINAMATH_GPT_sum_of_consecutive_pages_l883_88386

theorem sum_of_consecutive_pages (n : ℕ) 
  (h : n * (n + 1) = 20412) : n + (n + 1) + (n + 2) = 429 := by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_pages_l883_88386


namespace NUMINAMATH_GPT_greatest_int_less_than_200_with_gcd_18_eq_9_l883_88375

theorem greatest_int_less_than_200_with_gcd_18_eq_9 :
  ∃ n, n < 200 ∧ Int.gcd n 18 = 9 ∧ ∀ m, m < 200 ∧ Int.gcd m 18 = 9 → m ≤ n :=
sorry

end NUMINAMATH_GPT_greatest_int_less_than_200_with_gcd_18_eq_9_l883_88375


namespace NUMINAMATH_GPT_tip_calculation_l883_88334

def pizza_price : ℤ := 10
def number_of_pizzas : ℤ := 4
def total_pizza_cost := pizza_price * number_of_pizzas
def bill_given : ℤ := 50
def change_received : ℤ := 5
def total_spent := bill_given - change_received
def tip_given := total_spent - total_pizza_cost

theorem tip_calculation : tip_given = 5 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_tip_calculation_l883_88334


namespace NUMINAMATH_GPT_arc_length_of_given_curve_l883_88337

open Real

noncomputable def arc_length (f : ℝ → ℝ) (a b : ℝ) :=
  ∫ x in a..b, sqrt (1 + (deriv f x)^2)

noncomputable def given_function (x : ℝ) : ℝ :=
  arccos (sqrt x) - sqrt (x - x^2) + 4

theorem arc_length_of_given_curve :
  arc_length given_function 0 (1/2) = sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_given_curve_l883_88337


namespace NUMINAMATH_GPT_cubic_solution_l883_88317

theorem cubic_solution (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by
  sorry

end NUMINAMATH_GPT_cubic_solution_l883_88317


namespace NUMINAMATH_GPT_cubes_closed_under_multiplication_l883_88383

-- Define the set of cubes of positive integers
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3

-- Define the multiplication operation on the set of cubes
def cube_mult_closed : Prop :=
  ∀ x y : ℕ, is_cube x → is_cube y → is_cube (x * y)

-- The statement we want to prove
theorem cubes_closed_under_multiplication : cube_mult_closed :=
sorry

end NUMINAMATH_GPT_cubes_closed_under_multiplication_l883_88383


namespace NUMINAMATH_GPT_value_of_k_l883_88369

theorem value_of_k (k m : ℝ)
    (h1 : m = k / 3)
    (h2 : 2 = k / (3 * m - 1)) :
    k = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l883_88369


namespace NUMINAMATH_GPT_fraction_of_number_l883_88323

theorem fraction_of_number (N : ℕ) (hN : N = 180) : 
  (6 + (1 / 2) * (1 / 3) * (1 / 5) * N) = (1 / 25) * N := 
by
  sorry

end NUMINAMATH_GPT_fraction_of_number_l883_88323


namespace NUMINAMATH_GPT_solve_for_x_l883_88306

theorem solve_for_x (x : ℝ) : 0.05 * x + 0.07 * (30 + x) = 15.4 → x = 110.8333333 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solve_for_x_l883_88306


namespace NUMINAMATH_GPT_pos_real_x_plus_inv_ge_two_l883_88321

theorem pos_real_x_plus_inv_ge_two (x : ℝ) (hx : x > 0) : x + (1 / x) ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_pos_real_x_plus_inv_ge_two_l883_88321


namespace NUMINAMATH_GPT_total_interval_length_l883_88370

noncomputable def interval_length : ℝ :=
  1 / (1 + 2^Real.pi)

theorem total_interval_length :
  ∀ x : ℝ, x < 1 ∧ Real.tan (Real.log x / Real.log 4) > 0 →
  (∃ y, interval_length = y) :=
by
  sorry

end NUMINAMATH_GPT_total_interval_length_l883_88370


namespace NUMINAMATH_GPT_number_of_ordered_triples_modulo_1000000_l883_88312

def p : ℕ := 2017
def N : ℕ := sorry -- N is the number of ordered triples (a, b, c)

theorem number_of_ordered_triples_modulo_1000000 (N : ℕ) (h : ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ p * (p - 1) ∧ 1 ≤ b ∧ b ≤ p * (p - 1) ∧ a^b - b^a = p * c → true) : 
  N % 1000000 = 2016 :=
sorry

end NUMINAMATH_GPT_number_of_ordered_triples_modulo_1000000_l883_88312


namespace NUMINAMATH_GPT_percentage_invalid_votes_l883_88371

theorem percentage_invalid_votes 
    (total_votes : ℕ)
    (candidate_A_votes : ℕ)
    (candidate_A_percentage : ℝ)
    (total_valid_percentage : ℝ) :
    total_votes = 560000 ∧
    candidate_A_votes = 357000 ∧
    candidate_A_percentage = 0.75 ∧
    total_valid_percentage = 100 - x ∧
    (0.75 * (total_valid_percentage / 100) * 560000 = 357000) →
    x = 15 :=
by
  sorry

end NUMINAMATH_GPT_percentage_invalid_votes_l883_88371


namespace NUMINAMATH_GPT_correct_parameterization_l883_88347

noncomputable def parametrize_curve (t : ℝ) : ℝ × ℝ :=
  (t, t^2)

theorem correct_parameterization : ∀ t : ℝ, ∃ x y : ℝ, parametrize_curve t = (x, y) ∧ y = x^2 :=
by
  intro t
  use t, t^2
  dsimp [parametrize_curve]
  exact ⟨rfl, rfl⟩

end NUMINAMATH_GPT_correct_parameterization_l883_88347


namespace NUMINAMATH_GPT_fouad_double_ahmed_l883_88363

/-- Proof that in 4 years, Fouad's age will be double of Ahmed's age given their current ages. -/
theorem fouad_double_ahmed (x : ℕ) (ahmed_age fouad_age : ℕ) (h1 : ahmed_age = 11) (h2 : fouad_age = 26) :
  (fouad_age + x = 2 * (ahmed_age + x)) → x = 4 :=
by
  -- This is the statement only, proof is omitted
  sorry

end NUMINAMATH_GPT_fouad_double_ahmed_l883_88363


namespace NUMINAMATH_GPT_number_of_primes_between_30_and_50_l883_88353

/-- 
  Prove that there are exactly 5 prime numbers in the range from 30 to 50. 
  These primes are 31, 37, 41, 43, and 47.
-/
theorem number_of_primes_between_30_and_50 : 
  (Finset.filter Nat.Prime (Finset.range 51)).card - 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 5 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_primes_between_30_and_50_l883_88353


namespace NUMINAMATH_GPT_range_of_a_l883_88319

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (0 < x) → (-3^x ≤ a)) ↔ (a ≥ -1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l883_88319


namespace NUMINAMATH_GPT_maximal_area_of_AMNQ_l883_88377

theorem maximal_area_of_AMNQ (s q : ℝ) (Hq1 : 0 ≤ q) (Hq2 : q ≤ s) :
  let Q := (s, q)
  ∃ M N : ℝ × ℝ, 
    (M.1 ∈ [0,s] ∧ M.2 = 0) ∧ 
    (N.1 = s ∧ N.2 ∈ [0,s]) ∧ 
    if q ≤ (2/3) * s 
    then 
      (M.1 * M.2 / 2 = (CQ/2)) 
    else 
      (N = (s, s)) :=
by sorry

end NUMINAMATH_GPT_maximal_area_of_AMNQ_l883_88377


namespace NUMINAMATH_GPT_picture_distance_l883_88384

theorem picture_distance (wall_width picture_width x y : ℝ)
  (h_wall : wall_width = 25)
  (h_picture : picture_width = 5)
  (h_relation : x = 2 * y)
  (h_total : x + picture_width + y = wall_width) :
  x = 13.34 :=
by
  sorry

end NUMINAMATH_GPT_picture_distance_l883_88384


namespace NUMINAMATH_GPT_percentage_increase_proof_l883_88304

def breakfast_calories : ℕ := 500
def shakes_total_calories : ℕ := 3 * 300
def total_daily_calories : ℕ := 3275

noncomputable def percentage_increase_in_calories (P : ℝ) : Prop :=
  let lunch_calories := breakfast_calories * (1 + P / 100)
  let dinner_calories := 2 * lunch_calories
  breakfast_calories + lunch_calories + dinner_calories + shakes_total_calories = total_daily_calories

theorem percentage_increase_proof : percentage_increase_in_calories 125 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_proof_l883_88304


namespace NUMINAMATH_GPT_min_socks_for_pairs_l883_88354

-- Definitions for conditions
def pairs_of_socks : ℕ := 4
def sizes : ℕ := 2
def colors : ℕ := 2

-- Theorem statement
theorem min_socks_for_pairs : 
  ∃ n, n = 7 ∧ 
  ∀ (socks : ℕ), socks >= pairs_of_socks → socks ≥ 7 :=
sorry

end NUMINAMATH_GPT_min_socks_for_pairs_l883_88354


namespace NUMINAMATH_GPT_max_period_of_function_l883_88349

theorem max_period_of_function (f : ℝ → ℝ) (h1 : ∀ x, f (1 + x) = f (1 - x)) (h2 : ∀ x, f (8 + x) = f (8 - x)) :
  ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ 14) ∧ T = 14 :=
sorry

end NUMINAMATH_GPT_max_period_of_function_l883_88349


namespace NUMINAMATH_GPT_power_of_six_evaluation_l883_88380

noncomputable def example_expr : ℝ := (6 : ℝ)^(1/4) / (6 : ℝ)^(1/6)

theorem power_of_six_evaluation : example_expr = (6 : ℝ)^(1/12) := 
by
  sorry

end NUMINAMATH_GPT_power_of_six_evaluation_l883_88380


namespace NUMINAMATH_GPT_ensure_mixed_tablets_l883_88330

theorem ensure_mixed_tablets (A B : ℕ) (total : ℕ) (hA : A = 10) (hB : B = 16) (htotal : total = 18) :
  ∃ (a b : ℕ), a + b = total ∧ a ≤ A ∧ b ≤ B ∧ a > 0 ∧ b > 0 :=
by
  sorry

end NUMINAMATH_GPT_ensure_mixed_tablets_l883_88330


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l883_88390

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > e) : x > 1 :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l883_88390


namespace NUMINAMATH_GPT_correctness_of_option_C_l883_88326

-- Define the conditions as hypotheses
variable (x y : ℝ)

def condA : Prop := ∀ x: ℝ, x^3 * x^5 = x^15
def condB : Prop := ∀ x y: ℝ, 2 * x + 3 * y = 5 * x * y
def condC : Prop := ∀ x y: ℝ, 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y
def condD : Prop := ∀ x: ℝ, (x - 2)^2 = x^2 - 4

-- State the proof problem is correct
theorem correctness_of_option_C (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end NUMINAMATH_GPT_correctness_of_option_C_l883_88326


namespace NUMINAMATH_GPT_sum_of_parts_l883_88368

theorem sum_of_parts (x y : ℝ) (h1 : x + y = 52) (h2 : y = 30.333333333333332) :
  10 * x + 22 * y = 884 :=
sorry

end NUMINAMATH_GPT_sum_of_parts_l883_88368


namespace NUMINAMATH_GPT_intersection_A_B_l883_88362

open Set

def set_A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k + 1}
def set_B : Set ℤ := {x | 0 < x ∧ x < 5}

theorem intersection_A_B : set_A ∩ set_B = {1, 3} := 
by 
  sorry

end NUMINAMATH_GPT_intersection_A_B_l883_88362


namespace NUMINAMATH_GPT_man_l883_88379

theorem man's_age_twice_son (S M Y : ℕ) (h1 : M = S + 26) (h2 : S = 24) (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  sorry

end NUMINAMATH_GPT_man_l883_88379


namespace NUMINAMATH_GPT_chessboard_problem_proof_l883_88372

variable (n : ℕ)

noncomputable def chessboard_problem : Prop :=
  ∀ (colors : Fin (2 * n) → Fin (2 * n) → Fin n),
  ∃ i₁ i₂ j₁ j₂,
    i₁ ≠ i₂ ∧
    j₁ ≠ j₂ ∧
    colors i₁ j₁ = colors i₁ j₂ ∧
    colors i₂ j₁ = colors i₂ j₂

/-- Given a 2n x 2n chessboard colored with n colors, there exist 2 tiles in either the same column 
or row such that if the colors of both tiles are swapped, then there exists a rectangle where all 
its four corner tiles have the same color. -/
theorem chessboard_problem_proof (n : ℕ) : chessboard_problem n :=
sorry

end NUMINAMATH_GPT_chessboard_problem_proof_l883_88372


namespace NUMINAMATH_GPT_solve_for_x_l883_88303

theorem solve_for_x (x : ℝ) : (2010 + 2 * x) ^ 2 = x ^ 2 → x = -2010 ∨ x = -670 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l883_88303


namespace NUMINAMATH_GPT_double_bed_heavier_l883_88310

-- Define the problem conditions
variable (S D B : ℝ)
variable (h1 : 5 * S = 50)
variable (h2 : 2 * S + 4 * D + 3 * B = 180)
variable (h3 : 3 * B = 60)

-- Define the goal to prove
theorem double_bed_heavier (S D B : ℝ) (h1 : 5 * S = 50) (h2 : 2 * S + 4 * D + 3 * B = 180) (h3 : 3 * B = 60) : D - S = 15 :=
by
  sorry

end NUMINAMATH_GPT_double_bed_heavier_l883_88310


namespace NUMINAMATH_GPT_min_value_fraction_sum_l883_88314

theorem min_value_fraction_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h_eq : 1 = 2 * a + b) :
  (1 / a + 1 / b) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_fraction_sum_l883_88314


namespace NUMINAMATH_GPT_john_pennies_more_than_kate_l883_88357

theorem john_pennies_more_than_kate (kate_pennies : ℕ) (john_pennies : ℕ) (h_kate : kate_pennies = 223) (h_john : john_pennies = 388) : john_pennies - kate_pennies = 165 := by
  sorry

end NUMINAMATH_GPT_john_pennies_more_than_kate_l883_88357


namespace NUMINAMATH_GPT_number_of_boys_l883_88348

theorem number_of_boys (n : ℕ)
  (initial_avg_height : ℕ)
  (incorrect_height : ℕ)
  (correct_height : ℕ)
  (actual_avg_height : ℕ)
  (h1 : initial_avg_height = 184)
  (h2 : incorrect_height = 166)
  (h3 : correct_height = 106)
  (h4 : actual_avg_height = 182)
  (h5 : initial_avg_height * n - (incorrect_height - correct_height) = actual_avg_height * n) :
  n = 30 :=
sorry

end NUMINAMATH_GPT_number_of_boys_l883_88348


namespace NUMINAMATH_GPT_coefficient_of_x_in_first_term_l883_88388

variable {a k n : ℝ} (x : ℝ)

theorem coefficient_of_x_in_first_term (h1 : (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
  (h2 : a - n + k = 7) :
  3 = 3 := 
sorry

end NUMINAMATH_GPT_coefficient_of_x_in_first_term_l883_88388


namespace NUMINAMATH_GPT_xy_value_l883_88376

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : xy = 21 :=
sorry

end NUMINAMATH_GPT_xy_value_l883_88376


namespace NUMINAMATH_GPT_admission_price_for_children_l883_88397

theorem admission_price_for_children 
  (admission_price_adult : ℕ)
  (total_persons : ℕ)
  (total_amount_dollars : ℕ)
  (children_attended : ℕ)
  (admission_price_children : ℕ)
  (h1 : admission_price_adult = 60)
  (h2 : total_persons = 280)
  (h3 : total_amount_dollars = 140)
  (h4 : children_attended = 80)
  (h5 : (total_persons - children_attended) * admission_price_adult + children_attended * admission_price_children = total_amount_dollars * 100)
  : admission_price_children = 25 := 
by 
  sorry

end NUMINAMATH_GPT_admission_price_for_children_l883_88397


namespace NUMINAMATH_GPT_melanie_total_plums_l883_88382

namespace Melanie

def initial_plums : ℝ := 7.0
def plums_given_by_sam : ℝ := 3.0

theorem melanie_total_plums : initial_plums + plums_given_by_sam = 10.0 :=
by
  sorry

end Melanie

end NUMINAMATH_GPT_melanie_total_plums_l883_88382


namespace NUMINAMATH_GPT_abs_inequality_solution_l883_88356

theorem abs_inequality_solution (x : ℝ) : 
  (|x - 1| < 1) ↔ (0 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l883_88356


namespace NUMINAMATH_GPT_typing_speed_ratio_l883_88389

variable (T M : ℝ)

-- Conditions
def condition1 : Prop := T + M = 12
def condition2 : Prop := T + 1.25 * M = 14

-- Proof statement
theorem typing_speed_ratio (h1 : condition1 T M) (h2 : condition2 T M) : M / T = 2 := by
  sorry

end NUMINAMATH_GPT_typing_speed_ratio_l883_88389


namespace NUMINAMATH_GPT_calculate_expression_l883_88350

theorem calculate_expression : (3 / 4 - 1 / 8) ^ 5 = 3125 / 32768 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l883_88350


namespace NUMINAMATH_GPT_train_speed_l883_88322

theorem train_speed (length_train : ℝ) (time_to_cross : ℝ) (length_bridge : ℝ)
  (h_train : length_train = 100) (h_time : time_to_cross = 12.499)
  (h_bridge : length_bridge = 150) : 
  ((length_train + length_bridge) / time_to_cross * 3.6) = 72 := 
by 
  sorry

end NUMINAMATH_GPT_train_speed_l883_88322


namespace NUMINAMATH_GPT_find_m_n_l883_88365

def is_prime (n : Nat) : Prop := ∀ m, m ∣ n → m = 1 ∨ m = n

theorem find_m_n (p k : ℕ) (hk : 1 < k) (hp : is_prime p) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ (m, n) ≠ (1, 1) ∧ (m^p + n^p) / 2 = (m + n) / 2 ^ k) ↔ k = p :=
sorry

end NUMINAMATH_GPT_find_m_n_l883_88365


namespace NUMINAMATH_GPT_klinker_age_l883_88343

theorem klinker_age (K D : ℕ) (h1 : D = 10) (h2 : K + 15 = 2 * (D + 15)) : K = 35 :=
by
  sorry

end NUMINAMATH_GPT_klinker_age_l883_88343


namespace NUMINAMATH_GPT_ship_length_l883_88328

theorem ship_length (E S L : ℕ) (h1 : 150 * E = L + 150 * S) (h2 : 90 * E = L - 90 * S) : 
  L = 24 :=
by
  sorry

end NUMINAMATH_GPT_ship_length_l883_88328


namespace NUMINAMATH_GPT_molecular_weight_proof_l883_88364

/-- Atomic weights in atomic mass units (amu) --/
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01
def atomic_weight_N : ℝ := 14.01
def atomic_weight_P : ℝ := 30.97

/-- Number of atoms in the compound --/
def num_Al : ℝ := 2
def num_O : ℝ := 4
def num_H : ℝ := 6
def num_N : ℝ := 3
def num_P : ℝ := 1

/-- calculating the molecular weight --/
def molecular_weight : ℝ := 
  (num_Al * atomic_weight_Al) +
  (num_O * atomic_weight_O) +
  (num_H * atomic_weight_H) +
  (num_N * atomic_weight_N) +
  (num_P * atomic_weight_P)

-- The proof statement
theorem molecular_weight_proof : molecular_weight = 197.02 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_proof_l883_88364


namespace NUMINAMATH_GPT_chocolateBarsPerBox_l883_88308

def numberOfSmallBoxes := 20
def totalChocolateBars := 500

theorem chocolateBarsPerBox : totalChocolateBars / numberOfSmallBoxes = 25 :=
by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_chocolateBarsPerBox_l883_88308


namespace NUMINAMATH_GPT_cos_pi_six_plus_alpha_l883_88358

variable (α : ℝ)

theorem cos_pi_six_plus_alpha (h : Real.sin (Real.pi / 3 - α) = 1 / 6) : 
  Real.cos (Real.pi / 6 + α) = 1 / 6 :=
sorry

end NUMINAMATH_GPT_cos_pi_six_plus_alpha_l883_88358


namespace NUMINAMATH_GPT_teacher_buys_total_21_pens_l883_88361

def num_black_pens : Nat := 7
def num_blue_pens : Nat := 9
def num_red_pens : Nat := 5
def total_pens : Nat := num_black_pens + num_blue_pens + num_red_pens

theorem teacher_buys_total_21_pens : total_pens = 21 := 
by
  unfold total_pens num_black_pens num_blue_pens num_red_pens
  rfl -- reflexivity (21 = 21)

end NUMINAMATH_GPT_teacher_buys_total_21_pens_l883_88361


namespace NUMINAMATH_GPT_number_of_pairs_l883_88302

theorem number_of_pairs (f m : ℕ) (n : ℕ) :
  n = 6 →
  (f + m ≤ n) →
  ∃! pairs : ℕ, pairs = 2 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_number_of_pairs_l883_88302


namespace NUMINAMATH_GPT_green_paint_quarts_l883_88339

theorem green_paint_quarts (x : ℕ) (h : 5 * x = 3 * 15) : x = 9 := 
sorry

end NUMINAMATH_GPT_green_paint_quarts_l883_88339


namespace NUMINAMATH_GPT_edward_chocolate_l883_88336

theorem edward_chocolate (total_chocolate : ℚ) (num_piles : ℕ) (piles_received_by_Edward : ℕ) :
  total_chocolate = 75 / 7 → num_piles = 5 → piles_received_by_Edward = 2 → 
  (total_chocolate / num_piles) * piles_received_by_Edward = 30 / 7 := 
by
  intros ht hn hp
  sorry

end NUMINAMATH_GPT_edward_chocolate_l883_88336


namespace NUMINAMATH_GPT_number_of_boxes_l883_88346

-- Definitions based on conditions
def pieces_per_box := 500
def total_pieces := 3000

-- Theorem statement, we need to prove that the number of boxes is 6
theorem number_of_boxes : total_pieces / pieces_per_box = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_boxes_l883_88346


namespace NUMINAMATH_GPT_rhombus_side_length_l883_88355

theorem rhombus_side_length (d1 d2 : ℕ) (h1 : d1 = 24) (h2 : d2 = 70) : 
  ∃ (a : ℕ), a^2 = (d1 / 2)^2 + (d2 / 2)^2 ∧ a = 37 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_side_length_l883_88355


namespace NUMINAMATH_GPT_sum_of_first_150_remainder_l883_88360

theorem sum_of_first_150_remainder :
  let n := 150
  let sum := n * (n + 1) / 2
  sum % 5600 = 125 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_150_remainder_l883_88360


namespace NUMINAMATH_GPT_solve_equation_l883_88378

def problem_statement : Prop :=
  ∃ x : ℚ, (3 - x) / (x + 2) + (3 * x - 6) / (3 - x) = 2 ∧ x = -7 / 6

theorem solve_equation : problem_statement :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_equation_l883_88378


namespace NUMINAMATH_GPT_expected_pourings_correct_l883_88398

section
  /-- Four glasses are arranged in a row: the first and third contain orange juice, 
      the second and fourth are empty. Valya can take a full glass and pour its 
      contents into one of the two empty glasses each time. -/
  def initial_state : List Bool := [true, false, true, false]
  def target_state : List Bool := [false, true, false, true]

  /-- Define a function to calculate the expected number of pourings required to 
      reach the target state from the initial state given the probabilities of 
      transitions. -/
  noncomputable def expected_number_of_pourings (init : List Bool) (target : List Bool) : ℕ :=
    if init = initial_state ∧ target = target_state then 6 else 0

  /-- Prove that the expected number of pourings required to transition from 
      the initial state [true, false, true, false] to the target state [false, true, false, true] is 6. -/
  theorem expected_pourings_correct :
    expected_number_of_pourings initial_state target_state = 6 :=
  by
    -- Proof omitted
    sorry
end

end NUMINAMATH_GPT_expected_pourings_correct_l883_88398


namespace NUMINAMATH_GPT_shopkeeper_loss_percent_l883_88399

theorem shopkeeper_loss_percent (I : ℝ) (h1 : I > 0) : 
  (0.1 * (I - 0.4 * I)) = 0.4 * (1.1 * I) :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_percent_l883_88399


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l883_88352

theorem largest_angle_of_triangle
  (a b y : ℝ)
  (h1 : a = 60)
  (h2 : b = 70)
  (h3 : a + b + y = 180) :
  max a (max b y) = b :=
by
  sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l883_88352


namespace NUMINAMATH_GPT_point_B_in_first_quadrant_l883_88305

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end NUMINAMATH_GPT_point_B_in_first_quadrant_l883_88305
