import Mathlib

namespace NUMINAMATH_GPT_triangle_area_calculation_l1512_151279

theorem triangle_area_calculation
  (A : ℕ)
  (BC : ℕ)
  (h : ℕ)
  (nine_parallel_lines : Bool)
  (equal_segments : Bool)
  (largest_area_part : ℕ)
  (largest_part_condition : largest_area_part = 38) :
  9 * (BC / 10) * (h / 10) / 2 = 10 * (BC / 2) * A / 19 :=
sorry

end NUMINAMATH_GPT_triangle_area_calculation_l1512_151279


namespace NUMINAMATH_GPT_smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l1512_151216

open Nat

theorem smallest_natur_number_with_units_digit_6_and_transf_is_four_times (n : ℕ) :
  (n % 10 = 6 ∧ ∃ m, 6 * 10 ^ (m - 1) + n / 10 = 4 * n) → n = 153846 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l1512_151216


namespace NUMINAMATH_GPT_find_leftover_amount_l1512_151237

open Nat

def octal_to_decimal (n : ℕ) : ℕ :=
  let digits := [5, 5, 5, 5]
  List.foldr (λ (d : ℕ) (acc : ℕ) => d + 8 * acc) 0 digits

def expenses_total : ℕ := 1200 + 800 + 400

theorem find_leftover_amount : 
  let initial_amount := octal_to_decimal 5555
  let final_amount := initial_amount - expenses_total
  final_amount = 525 := by
    sorry

end NUMINAMATH_GPT_find_leftover_amount_l1512_151237


namespace NUMINAMATH_GPT_largest_three_digit_congruent_to_twelve_mod_fifteen_l1512_151228

theorem largest_three_digit_congruent_to_twelve_mod_fifteen :
  ∃ n : ℕ, 100 ≤ 15 * n + 12 ∧ 15 * n + 12 < 1000 ∧ (15 * n + 12 = 987) :=
sorry

end NUMINAMATH_GPT_largest_three_digit_congruent_to_twelve_mod_fifteen_l1512_151228


namespace NUMINAMATH_GPT_quadratic_root_condition_l1512_151263

theorem quadratic_root_condition (a b : ℝ) (h : (3:ℝ)^2 + 2 * a * 3 + 3 * b = 0) : 2 * a + b = -3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l1512_151263


namespace NUMINAMATH_GPT_cost_price_of_computer_table_l1512_151200

theorem cost_price_of_computer_table (SP : ℝ) (h1 : SP = 1.15 * CP ∧ SP = 6400) : CP = 5565.22 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_computer_table_l1512_151200


namespace NUMINAMATH_GPT_correct_calculation_value_l1512_151236

theorem correct_calculation_value (x : ℕ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 :=
by
  -- The conditions are used directly in the definitions
  -- Given the condition (x * 5) + 7 = 27
  let h1 := h
  -- Solve for x and use x in the correct calculation
  sorry

end NUMINAMATH_GPT_correct_calculation_value_l1512_151236


namespace NUMINAMATH_GPT_distance_between_cities_l1512_151278

theorem distance_between_cities (x : ℝ) (h1 : x ≥ 100) (t : ℝ)
  (A_speed : ℝ := 12) (B_speed : ℝ := 0.05 * x)
  (condition_A : 7 + A_speed * t + B_speed * t = x)
  (condition_B : t = (x - 7) / (A_speed + B_speed)) :
  x = 140 :=
sorry

end NUMINAMATH_GPT_distance_between_cities_l1512_151278


namespace NUMINAMATH_GPT_remainder_1534_base12_div_by_9_l1512_151238

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem remainder_1534_base12_div_by_9 :
  (base12_to_base10 1534) % 9 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_1534_base12_div_by_9_l1512_151238


namespace NUMINAMATH_GPT_fraction_of_beans_remaining_l1512_151201

variables (J B R : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.10 * (J + B)
def condition2 : Prop := J + R = 0.60 * (J + B)

theorem fraction_of_beans_remaining (h1 : condition1 J B) (h2 : condition2 J B R) :
  R / B = 5 / 9 :=
  sorry

end NUMINAMATH_GPT_fraction_of_beans_remaining_l1512_151201


namespace NUMINAMATH_GPT_polygon_sides_arithmetic_sequence_l1512_151209

theorem polygon_sides_arithmetic_sequence 
  (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : 2 * (180 * (n - 2)) = n * (100 + 140)) :
  n = 6 :=
  sorry

end NUMINAMATH_GPT_polygon_sides_arithmetic_sequence_l1512_151209


namespace NUMINAMATH_GPT_manager_salary_l1512_151212

theorem manager_salary (average_salary_employees : ℕ)
    (employee_count : ℕ) (new_average_salary : ℕ)
    (total_salary_before : ℕ)
    (total_salary_after : ℕ)
    (M : ℕ) :
    average_salary_employees = 1500 →
    employee_count = 20 →
    new_average_salary = 1650 →
    total_salary_before = employee_count * average_salary_employees →
    total_salary_after = (employee_count + 1) * new_average_salary →
    M = total_salary_after - total_salary_before →
    M = 4650 := by
    intros h1 h2 h3 h4 h5 h6
    rw [h6]
    sorry -- The proof is not required, so we use 'sorry' here.

end NUMINAMATH_GPT_manager_salary_l1512_151212


namespace NUMINAMATH_GPT_complex_expression_l1512_151252

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_expression (i : ℂ) (h : imaginary_unit i) :
  (1 - i) ^ 2016 + (1 + i) ^ 2016 = 2 ^ 1009 :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_l1512_151252


namespace NUMINAMATH_GPT_find_unique_number_l1512_151234

theorem find_unique_number : 
  ∃ X : ℕ, 
    (X % 1000 = 376 ∨ X % 1000 = 625) ∧ 
    (X * (X - 1) % 10000 = 0) ∧ 
    (Nat.gcd X (X - 1) = 1) ∧ 
    ((X % 625 = 0) ∨ ((X - 1) % 625 = 0)) ∧ 
    ((X % 16 = 0) ∨ ((X - 1) % 16 = 0)) ∧ 
    X = 9376 :=
by sorry

end NUMINAMATH_GPT_find_unique_number_l1512_151234


namespace NUMINAMATH_GPT_min_value_a_2b_3c_l1512_151215

theorem min_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  a + 2 * b - 3 * c ≥ -2 :=
sorry

end NUMINAMATH_GPT_min_value_a_2b_3c_l1512_151215


namespace NUMINAMATH_GPT_train_cross_time_l1512_151210

/-- Given the conditions:
1. Two trains run in opposite directions and cross a man in 17 seconds and some unknown time respectively.
2. They cross each other in 22 seconds.
3. The ratio of their speeds is 1 to 1.
Prove the time it takes for the first train to cross the man. -/
theorem train_cross_time (v_1 v_2 L_1 L_2 : ℝ) (t_2 : ℝ) (h1 : t_2 = 17) (h2 : v_1 = v_2)
  (h3 : (L_1 + L_2) / (v_1 + v_2) = 22) : (L_1 / v_1) = 27 := 
by 
  -- The actual proof will go here
  sorry

end NUMINAMATH_GPT_train_cross_time_l1512_151210


namespace NUMINAMATH_GPT_moles_of_Br2_combined_l1512_151208

-- Definition of the reaction relation
def chemical_reaction (CH4 Br2 CH3Br HBr : ℕ) : Prop :=
  CH4 = 1 ∧ HBr = 1

-- Statement of the proof problem
theorem moles_of_Br2_combined (CH4 Br2 CH3Br HBr : ℕ) (h : chemical_reaction CH4 Br2 CH3Br HBr) : Br2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_Br2_combined_l1512_151208


namespace NUMINAMATH_GPT_stamps_cost_l1512_151240

theorem stamps_cost (cost_one: ℝ) (cost_three: ℝ) (h: cost_one = 0.34) (h1: cost_three = 3 * cost_one) : 
  2 * cost_one = 0.68 := 
by
  sorry

end NUMINAMATH_GPT_stamps_cost_l1512_151240


namespace NUMINAMATH_GPT_intermediate_root_exists_l1512_151218

open Polynomial

theorem intermediate_root_exists
  (a b c x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : -a * x2^2 + b * x2 + c = 0) :
  ∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) :=
sorry

end NUMINAMATH_GPT_intermediate_root_exists_l1512_151218


namespace NUMINAMATH_GPT_solution_set_inequality_l1512_151258

theorem solution_set_inequality 
  (a b : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + 3 > 0 ↔ -1 < x ∧ x < 1/2) :
  ((-1:ℝ) < x ∧ x < 2) ↔ 3 * x^2 + b * x + a < 0 :=
by 
  -- Write the proof here
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l1512_151258


namespace NUMINAMATH_GPT_probability_product_is_square_l1512_151250

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def probability_square_product : ℚ :=
  let total_outcomes   := 10 * 8
  let favorable_outcomes := 
    [(1,1), (1,4), (2,2), (4,1), (3,3), (2,8), (8,2), (5,5), (6,6), (7,7), (8,8)].length
  favorable_outcomes / total_outcomes

theorem probability_product_is_square : 
  probability_square_product = 11 / 80 :=
  sorry

end NUMINAMATH_GPT_probability_product_is_square_l1512_151250


namespace NUMINAMATH_GPT_determine_a_l1512_151280

theorem determine_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (M m : ℝ)
  (hM : M = max (a^1) (a^2))
  (hm : m = min (a^1) (a^2))
  (hM_m : M = 2 * m) :
  a = 1/2 ∨ a = 2 := 
by sorry

end NUMINAMATH_GPT_determine_a_l1512_151280


namespace NUMINAMATH_GPT_total_onions_l1512_151296

theorem total_onions (S SA F J : ℕ) (h1 : S = 4) (h2 : SA = 5) (h3 : F = 9) (h4 : J = 7) : S + SA + F + J = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_onions_l1512_151296


namespace NUMINAMATH_GPT_root_conditions_l1512_151265

theorem root_conditions (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |x1^2 - 5 * x1| = a ∧ |x2^2 - 5 * x2| = a) ↔ (a = 0 ∨ a > 25 / 4) := 
by 
  sorry

end NUMINAMATH_GPT_root_conditions_l1512_151265


namespace NUMINAMATH_GPT_tournament_chromatic_index_l1512_151275

noncomputable def chromaticIndex {n : ℕ} (k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) : ℕ :=
k

theorem tournament_chromatic_index (n k : ℕ) (h₁ : 2^(k-1) < n) (h₂ : n ≤ 2^k) :
  chromaticIndex k h₁ h₂ = k :=
by sorry

end NUMINAMATH_GPT_tournament_chromatic_index_l1512_151275


namespace NUMINAMATH_GPT_minimum_value_of_func_l1512_151256

-- Define the circle and the line constraints, and the question
namespace CircleLineProblem

def is_center_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 1 = 0

def line_divides_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, is_center_of_circle x y → a * x - b * y + 3 = 0

noncomputable def func_to_minimize (a b : ℝ) : ℝ :=
  (2 / a) + (1 / (b - 1))

theorem minimum_value_of_func :
  ∃ (a b : ℝ), a > 0 ∧ b > 1 ∧ line_divides_circle a b ∧ func_to_minimize a b = 8 :=
by
  sorry

end CircleLineProblem

end NUMINAMATH_GPT_minimum_value_of_func_l1512_151256


namespace NUMINAMATH_GPT_at_least_three_bushes_with_same_number_of_flowers_l1512_151292

-- Defining the problem using conditions as definitions.
theorem at_least_three_bushes_with_same_number_of_flowers (n : ℕ) (f : Fin n → ℕ) (h1 : n = 201)
  (h2 : ∀ (i : Fin n), 1 ≤ f i ∧ f i ≤ 100) : 
  ∃ (x : ℕ), (∃ (i1 i2 i3 : Fin n), i1 ≠ i2 ∧ i1 ≠ i3 ∧ i2 ≠ i3 ∧ f i1 = x ∧ f i2 = x ∧ f i3 = x) := 
by
  sorry

end NUMINAMATH_GPT_at_least_three_bushes_with_same_number_of_flowers_l1512_151292


namespace NUMINAMATH_GPT_yellow_ball_count_l1512_151270

def total_balls : ℕ := 500
def red_balls : ℕ := total_balls / 3
def remaining_after_red : ℕ := total_balls - red_balls
def blue_balls : ℕ := remaining_after_red / 5
def remaining_after_blue : ℕ := remaining_after_red - blue_balls
def green_balls : ℕ := remaining_after_blue / 4
def yellow_balls : ℕ := total_balls - (red_balls + blue_balls + green_balls)

theorem yellow_ball_count : yellow_balls = 201 := by
  sorry

end NUMINAMATH_GPT_yellow_ball_count_l1512_151270


namespace NUMINAMATH_GPT_shared_earnings_eq_27_l1512_151295

theorem shared_earnings_eq_27
    (shoes_pairs : ℤ) (shoes_cost : ℤ) (shirts : ℤ) (shirts_cost : ℤ)
    (h1 : shoes_pairs = 6) (h2 : shoes_cost = 3)
    (h3 : shirts = 18) (h4 : shirts_cost = 2) :
    (shoes_pairs * shoes_cost + shirts * shirts_cost) / 2 = 27 := by
  sorry

end NUMINAMATH_GPT_shared_earnings_eq_27_l1512_151295


namespace NUMINAMATH_GPT_expected_value_smallest_N_l1512_151225
noncomputable def expectedValueN : ℝ := 6.54

def barryPicksPointsInsideUnitCircle (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P n).fst^2 + (P n).snd^2 ≤ 1

def pointsIndependentAndUniform (P : ℕ → ℝ × ℝ) : Prop :=
  -- This is a placeholder representing the independent and uniform picking which 
  -- would be formally defined using probability measures in an advanced Lean library.
  sorry

theorem expected_value_smallest_N (P : ℕ → ℝ × ℝ)
  (h1 : barryPicksPointsInsideUnitCircle P)
  (h2 : pointsIndependentAndUniform P) :
  ∃ N : ℕ, N = expectedValueN :=
sorry

end NUMINAMATH_GPT_expected_value_smallest_N_l1512_151225


namespace NUMINAMATH_GPT_washing_machine_cost_l1512_151297

variable (W D : ℝ)
variable (h1 : D = W - 30)
variable (h2 : 0.90 * (W + D) = 153)

theorem washing_machine_cost :
  W = 100 := by
  sorry

end NUMINAMATH_GPT_washing_machine_cost_l1512_151297


namespace NUMINAMATH_GPT_three_digit_number_ends_same_sequence_l1512_151251

theorem three_digit_number_ends_same_sequence (N : ℕ) (a b c : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
  (h2 : N % 10 = c)
  (h3 : (N / 10) % 10 = b)
  (h4 : (N / 100) % 10 = a)
  (h5 : a ≠ 0)
  (h6 : N^2 % 1000 = N) :
  N = 127 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_ends_same_sequence_l1512_151251


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l1512_151286

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 2^x - 1 > 0)) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l1512_151286


namespace NUMINAMATH_GPT_find_first_term_geom_seq_l1512_151288

noncomputable def first_term (a r : ℝ) := a

theorem find_first_term_geom_seq 
  (a r : ℝ) 
  (h1 : a * r ^ 3 = 720) 
  (h2 : a * r ^ 6 = 5040) : 
  first_term a r = 720 / 7 := 
sorry

end NUMINAMATH_GPT_find_first_term_geom_seq_l1512_151288


namespace NUMINAMATH_GPT_triple_sum_equals_seven_l1512_151246

theorem triple_sum_equals_seven {k m n : ℕ} (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (hcoprime : Nat.gcd k m = 1 ∧ Nat.gcd k n = 1 ∧ Nat.gcd m n = 1)
  (hlog : k * Real.log 5 / Real.log 400 + m * Real.log 2 / Real.log 400 = n) :
  k + m + n = 7 := by
  sorry

end NUMINAMATH_GPT_triple_sum_equals_seven_l1512_151246


namespace NUMINAMATH_GPT_radical_conjugate_sum_l1512_151219

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end NUMINAMATH_GPT_radical_conjugate_sum_l1512_151219


namespace NUMINAMATH_GPT_journey_possibility_l1512_151282

noncomputable def possible_start_cities 
  (routes : List (String × String)) 
  (visited : List String) : List String :=
sorry

theorem journey_possibility :
  possible_start_cities 
    [("Saint Petersburg", "Tver"), 
     ("Yaroslavl", "Nizhny Novgorod"), 
     ("Moscow", "Kazan"), 
     ("Nizhny Novgorod", "Kazan"), 
     ("Moscow", "Tver"), 
     ("Moscow", "Nizhny Novgorod")]
    ["Saint Petersburg", "Tver", "Yaroslavl", "Nizhny Novgorod", "Moscow", "Kazan"] 
  = ["Saint Petersburg", "Yaroslavl"] :=
sorry

end NUMINAMATH_GPT_journey_possibility_l1512_151282


namespace NUMINAMATH_GPT_min_megabytes_for_plan_Y_more_economical_l1512_151243

theorem min_megabytes_for_plan_Y_more_economical :
  ∃ (m : ℕ), 2500 + 10 * m < 15 * m ∧ m = 501 :=
by
  sorry

end NUMINAMATH_GPT_min_megabytes_for_plan_Y_more_economical_l1512_151243


namespace NUMINAMATH_GPT_parallelogram_area_l1512_151255

theorem parallelogram_area (base height : ℝ) (h_base : base = 24) (h_height : height = 10) :
  base * height = 240 :=
by
  rw [h_base, h_height]
  norm_num

end NUMINAMATH_GPT_parallelogram_area_l1512_151255


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1512_151271

theorem solution_set_of_inequality (x : ℝ) : (∃ x, (0 ≤ x ∧ x < 1) ↔ (x-2)/(x-1) ≥ 2) :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1512_151271


namespace NUMINAMATH_GPT_perpendicular_vectors_l1512_151298

variable {t : ℝ}

def a : ℝ × ℝ := (1, -1)
def b : ℝ × ℝ := (6, -4)

theorem perpendicular_vectors (ht : a.1 * (t * a.1 + b.1) + a.2 * (t * a.2 + b.2) = 0) : t = -5 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1512_151298


namespace NUMINAMATH_GPT_line_equation_l1512_151221

theorem line_equation (m n : ℝ) (p : ℝ) (h : p = 3) :
  ∃ b : ℝ, ∀ x y : ℝ, (y = n + 21) → (x = m + 3) → y = 7 * x + b ∧ b = n - 7 * m :=
by sorry

end NUMINAMATH_GPT_line_equation_l1512_151221


namespace NUMINAMATH_GPT_find_a_l1512_151206

theorem find_a
  (a : ℝ)
  (h_perpendicular : ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 3 * x - 6 * y - 1 = 0 → true) :
  a = 4 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end NUMINAMATH_GPT_find_a_l1512_151206


namespace NUMINAMATH_GPT_find_b2_a2_minus_a1_l1512_151202

theorem find_b2_a2_minus_a1 
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (d r : ℝ)
  (h_arith_seq : a₁ = -9 + d ∧ a₂ = a₁ + d)
  (h_geo_seq : b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ (-9) * (-1) = b₁ * b₃)
  (h_d_val : a₂ - a₁ = d)
  (h_b2_val : b₂ = -1) : 
  b₂ * (a₂ - a₁) = -8 :=
sorry

end NUMINAMATH_GPT_find_b2_a2_minus_a1_l1512_151202


namespace NUMINAMATH_GPT_count_ordered_triples_lcm_l1512_151214

def lcm_of_pair (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_ordered_triples_lcm :
  (∃ (count : ℕ), count = 70 ∧
   ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) →
   lcm_of_pair a b = 1000 → lcm_of_pair b c = 2000 → lcm_of_pair c a = 2000 → count = 70) :=
sorry

end NUMINAMATH_GPT_count_ordered_triples_lcm_l1512_151214


namespace NUMINAMATH_GPT_alphanumeric_puzzle_l1512_151257

/-- Alphanumeric puzzle proof problem -/
theorem alphanumeric_puzzle
  (A B C D E F H J K L : Nat)
  (h1 : A * B = B)
  (h2 : B * C = 10 * A + C)
  (h3 : C * D = 10 * B + C)
  (h4 : D * E = 100 * C + H)
  (h5 : E * F = 10 * D + K)
  (h6 : F * H = 100 * C + J)
  (h7 : H * J = 10 * K + J)
  (h8 : J * K = E)
  (h9 : K * L = L)
  (h10 : A * L = L) :
  A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0 :=
sorry

end NUMINAMATH_GPT_alphanumeric_puzzle_l1512_151257


namespace NUMINAMATH_GPT_female_sample_count_is_correct_l1512_151273

-- Definitions based on the given conditions
def total_students : ℕ := 900
def male_students : ℕ := 500
def sample_size : ℕ := 45
def female_students : ℕ := total_students - male_students
def female_sample_size : ℕ := (female_students * sample_size) / total_students

-- The lean statement to prove
theorem female_sample_count_is_correct : female_sample_size = 20 := 
by 
  -- A placeholder to indicate the proof needs to be filled in
  sorry

end NUMINAMATH_GPT_female_sample_count_is_correct_l1512_151273


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_53_l1512_151272

theorem smallest_four_digit_multiple_of_53 : ∃ n : ℕ, (1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 ∧ n = 1007) := 
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_53_l1512_151272


namespace NUMINAMATH_GPT_can_construct_parallelogram_l1512_151224

theorem can_construct_parallelogram {a b d1 d2 : ℝ} :
  (a = 3 ∧ b = 5 ∧ (a = b ∨ (‖a + b‖ ≥ ‖d1‖ ∧ ‖a + d1‖ ≥ ‖b‖ ∧ ‖b + d1‖ ≥ ‖a‖))) ∨
  (a ≠ 3 ∨ b ≠ 5 ∨ (a ≠ b ∧ (‖a + b‖ < ‖d1‖ ∨ ‖a + d1‖ < ‖b‖ ∨ ‖b + d1‖ < ‖a‖ ∨ ‖a + d1‖ < ‖d2‖ ∨ ‖b + d1‖ < ‖d2‖ ∨ ‖a + d2‖ < ‖d1‖ ∨ ‖b + d2‖ < ‖d1‖))) ↔ 
  (a = 3 ∧ b = 5 ∧ d1 = 0) :=
sorry

end NUMINAMATH_GPT_can_construct_parallelogram_l1512_151224


namespace NUMINAMATH_GPT_law_of_cosines_l1512_151213

theorem law_of_cosines (a b c : ℝ) (A : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A ≥ 0 ∧ A ≤ π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A :=
sorry

end NUMINAMATH_GPT_law_of_cosines_l1512_151213


namespace NUMINAMATH_GPT_inequality_example_l1512_151203

theorem inequality_example (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
  sorry

end NUMINAMATH_GPT_inequality_example_l1512_151203


namespace NUMINAMATH_GPT_smallest_integer_k_l1512_151299

theorem smallest_integer_k : ∀ (k : ℕ), (64^k > 4^16) → k ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_k_l1512_151299


namespace NUMINAMATH_GPT_remaining_credit_to_be_paid_l1512_151266

-- Define conditions
def total_credit_limit := 100
def amount_paid_tuesday := 15
def amount_paid_thursday := 23

-- Define the main theorem based on the given question and its correct answer
theorem remaining_credit_to_be_paid : 
  total_credit_limit - amount_paid_tuesday - amount_paid_thursday = 62 := 
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_remaining_credit_to_be_paid_l1512_151266


namespace NUMINAMATH_GPT_race_time_comparison_l1512_151289

noncomputable def townSquare : ℝ := 3 / 4 -- distance of one lap in miles
noncomputable def laps : ℕ := 7 -- number of laps
noncomputable def totalDistance : ℝ := laps * townSquare -- total distance of the race in miles
noncomputable def thisYearTime : ℝ := 42 -- time taken by this year's winner in minutes
noncomputable def lastYearTime : ℝ := 47.25 -- time taken by last year's winner in minutes

noncomputable def thisYearPace : ℝ := thisYearTime / totalDistance -- pace of this year's winner in minutes per mile
noncomputable def lastYearPace : ℝ := lastYearTime / totalDistance -- pace of last year's winner in minutes per mile
noncomputable def timeDifference : ℝ := lastYearPace - thisYearPace -- the difference in pace

theorem race_time_comparison : timeDifference = 1 := by
  sorry

end NUMINAMATH_GPT_race_time_comparison_l1512_151289


namespace NUMINAMATH_GPT_rectangular_area_length_width_l1512_151207

open Nat

theorem rectangular_area_length_width (lengthInMeters widthInMeters : ℕ) (h1 : lengthInMeters = 500) (h2 : widthInMeters = 60) :
  (lengthInMeters * widthInMeters = 30000) ∧ ((lengthInMeters * widthInMeters) / 10000 = 3) :=
by
  sorry

end NUMINAMATH_GPT_rectangular_area_length_width_l1512_151207


namespace NUMINAMATH_GPT_cost_difference_l1512_151264

theorem cost_difference (S : ℕ) (h1 : 15 + S = 24) : 15 - S = 6 :=
by
  sorry

end NUMINAMATH_GPT_cost_difference_l1512_151264


namespace NUMINAMATH_GPT_total_songs_listened_l1512_151253

theorem total_songs_listened (vivian_daily : ℕ) (fewer_songs : ℕ) (days_in_june : ℕ) (weekend_days : ℕ) :
  vivian_daily = 10 →
  fewer_songs = 2 →
  days_in_june = 30 →
  weekend_days = 8 →
  (vivian_daily * (days_in_june - weekend_days)) + ((vivian_daily - fewer_songs) * (days_in_june - weekend_days)) = 396 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_total_songs_listened_l1512_151253


namespace NUMINAMATH_GPT_div_scaled_result_l1512_151239

theorem div_scaled_result :
  (2994 : ℝ) / 14.5 = 171 :=
by
  have cond1 : (29.94 : ℝ) / 1.45 = 17.1 := sorry
  have cond2 : (2994 : ℝ) = 100 * 29.94 := sorry
  have cond3 : (14.5 : ℝ) = 10 * 1.45 := sorry
  sorry

end NUMINAMATH_GPT_div_scaled_result_l1512_151239


namespace NUMINAMATH_GPT_point_on_graph_l1512_151268

def lies_on_graph (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  y = f x

theorem point_on_graph :
  lies_on_graph (-2) 0 (λ x => (1 / 2) * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_point_on_graph_l1512_151268


namespace NUMINAMATH_GPT_number_of_squares_or_cubes_l1512_151211

theorem number_of_squares_or_cubes (h1 : ∃ n, n = 28) (h2 : ∃ m, m = 9) (h3 : ∃ k, k = 2) : 
  ∃ t, t = 35 :=
sorry

end NUMINAMATH_GPT_number_of_squares_or_cubes_l1512_151211


namespace NUMINAMATH_GPT_inequality_bounds_l1512_151235

theorem inequality_bounds (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  1 < (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) ∧
  (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) < 4 :=
sorry

end NUMINAMATH_GPT_inequality_bounds_l1512_151235


namespace NUMINAMATH_GPT_minimum_value_of_f_l1512_151262

noncomputable def f (x : ℝ) : ℝ := sorry  -- define f such that f(x + 199) = 4x^2 + 4x + 3 for x ∈ ℝ

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 2 := by
  sorry  -- Prove that the minimum value of f(x) is 2

end NUMINAMATH_GPT_minimum_value_of_f_l1512_151262


namespace NUMINAMATH_GPT_quadratic_inequality_condition_l1512_151269

theorem quadratic_inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, ax^2 + bx + c < 0) ↔ (a < 0 ∧ b^2 - 4 * a * c < 0) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_condition_l1512_151269


namespace NUMINAMATH_GPT_onur_biking_distance_l1512_151229

-- Definitions based only on given conditions
def Onur_biking_distance_per_day (O : ℕ) := O
def Hanil_biking_distance_per_day (O : ℕ) := O + 40
def biking_days_per_week := 5
def total_distance_per_week := 2700

-- Mathematically equivalent proof problem
theorem onur_biking_distance (O : ℕ) (cond : 5 * (O + (O + 40)) = 2700) : O = 250 := by
  sorry

end NUMINAMATH_GPT_onur_biking_distance_l1512_151229


namespace NUMINAMATH_GPT_ratio_of_third_to_second_building_l1512_151291

/-
The tallest building in the world is 100 feet tall. The second tallest is half that tall, the third tallest is some 
fraction of the second tallest building's height, and the fourth is one-fifth as tall as the third. All 4 buildings 
put together are 180 feet tall. What is the ratio of the height of the third tallest building to the second tallest building?

Given H1 = 100, H2 = (1 / 2) * H1, H4 = (1 / 5) * H3, 
and H1 + H2 + H3 + H4 = 180, prove that H3 / H2 = 1 / 2.
-/

theorem ratio_of_third_to_second_building :
  ∀ (H1 H2 H3 H4 : ℝ),
  H1 = 100 →
  H2 = (1 / 2) * H1 →
  H4 = (1 / 5) * H3 →
  H1 + H2 + H3 + H4 = 180 →
  (H3 / H2) = (1 / 2) :=
by
  intros H1 H2 H3 H4 h1_eq h2_half_h1 h4_fifth_h3 total_eq
  /- proof steps go here -/
  sorry

end NUMINAMATH_GPT_ratio_of_third_to_second_building_l1512_151291


namespace NUMINAMATH_GPT_math_problem_l1512_151245

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) →
  (1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥  3 / 2)

theorem math_problem (a b c : ℝ) :
  proof_problem a b c :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1512_151245


namespace NUMINAMATH_GPT_kimberly_skittles_proof_l1512_151283

variable (SkittlesInitial : ℕ) (SkittlesBought : ℕ) (OrangesBought : ℕ)

/-- Kimberly's initial number of Skittles --/
def kimberly_initial_skittles := SkittlesInitial

/-- Skittles Kimberly buys --/
def kimberly_skittles_bought := SkittlesBought

/-- Oranges Kimbery buys (irrelevant for Skittles count) --/
def kimberly_oranges_bought := OrangesBought

/-- Total Skittles Kimberly has --/
def kimberly_total_skittles (SkittlesInitial SkittlesBought : ℕ) : ℕ :=
  SkittlesInitial + SkittlesBought

/-- Proof statement --/
theorem kimberly_skittles_proof (h1 : SkittlesInitial = 5) (h2 : SkittlesBought = 7) : 
  kimberly_total_skittles SkittlesInitial SkittlesBought = 12 :=
by
  rw [h1, h2]
  exact rfl

end NUMINAMATH_GPT_kimberly_skittles_proof_l1512_151283


namespace NUMINAMATH_GPT_blonde_hair_count_l1512_151244

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end NUMINAMATH_GPT_blonde_hair_count_l1512_151244


namespace NUMINAMATH_GPT_complex_problem_l1512_151293

open Complex

theorem complex_problem
  (α θ β : ℝ)
  (h : exp (i * (α + θ)) + exp (i * (β + θ)) = 1 / 3 + (4 / 9) * i) :
  exp (-i * (α + θ)) + exp (-i * (β + θ)) = 1 / 3 - (4 / 9) * i :=
by
  sorry

end NUMINAMATH_GPT_complex_problem_l1512_151293


namespace NUMINAMATH_GPT_wrongly_entered_mark_l1512_151290

theorem wrongly_entered_mark (x : ℝ) : 
  (∀ marks_instead_of_45 number_of_pupils (total_avg_increase : ℝ),
     marks_instead_of_45 = 45 ∧
     number_of_pupils = 44 ∧
     total_avg_increase = 0.5 →
     x = marks_instead_of_45 + total_avg_increase * number_of_pupils) →
  x = 67 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_wrongly_entered_mark_l1512_151290


namespace NUMINAMATH_GPT_greatest_integer_value_x_l1512_151260

theorem greatest_integer_value_x :
  ∀ x : ℤ, (∃ k : ℤ, x^2 + 2 * x + 9 = k * (x - 5)) ↔ x ≤ 49 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_value_x_l1512_151260


namespace NUMINAMATH_GPT_find_y_value_l1512_151226

noncomputable def y_value (y : ℝ) :=
  (3 * y)^2 + (7 * y)^2 + (1 / 2) * (3 * y) * (7 * y) = 1200

theorem find_y_value (y : ℝ) (hy : y_value y) : y = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_y_value_l1512_151226


namespace NUMINAMATH_GPT_first_new_player_weight_l1512_151233

theorem first_new_player_weight (x : ℝ) :
  (7 * 103) + x + 60 = 9 * 99 → 
  x = 110 := by
  sorry

end NUMINAMATH_GPT_first_new_player_weight_l1512_151233


namespace NUMINAMATH_GPT_price_of_each_sundae_l1512_151254

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℝ)
  (price_per_ice_cream_bar : ℝ)
  (total_cost_for_sundaes : ℝ) :
  num_ice_cream_bars = 225 →
  num_sundaes = 125 →
  total_price = 200 →
  price_per_ice_cream_bar = 0.60 →
  total_cost_for_sundaes = total_price - (num_ice_cream_bars * price_per_ice_cream_bar) →
  (total_cost_for_sundaes / num_sundaes) = 0.52 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_price_of_each_sundae_l1512_151254


namespace NUMINAMATH_GPT_relationship_y1_y2_y3_l1512_151232

variable (y1 y2 y3 : ℝ)

def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x - 5

theorem relationship_y1_y2_y3
  (h1 : quadratic_function (-4) = y1)
  (h2 : quadratic_function (-3) = y2)
  (h3 : quadratic_function (1) = y3) :
  y1 < y2 ∧ y2 < y3 :=
sorry

end NUMINAMATH_GPT_relationship_y1_y2_y3_l1512_151232


namespace NUMINAMATH_GPT_length_of_side_of_regular_tetradecagon_l1512_151249

theorem length_of_side_of_regular_tetradecagon (P : ℝ) (n : ℕ) (h₀ : n = 14) (h₁ : P = 154) : P / n = 11 := 
by
  sorry

end NUMINAMATH_GPT_length_of_side_of_regular_tetradecagon_l1512_151249


namespace NUMINAMATH_GPT_joe_height_is_82_l1512_151287

-- Given the conditions:
def Sara_height (x : ℝ) : Prop := true

def Joe_height (j : ℝ) (x : ℝ) : Prop := j = 6 + 2 * x

def combined_height (j : ℝ) (x : ℝ) : Prop := j + x = 120

-- We need to prove:
theorem joe_height_is_82 (x j : ℝ) 
  (h1 : combined_height j x)
  (h2 : Joe_height j x) :
  j = 82 := 
by 
  sorry

end NUMINAMATH_GPT_joe_height_is_82_l1512_151287


namespace NUMINAMATH_GPT_part_a_int_values_part_b_int_values_l1512_151259

-- Part (a)
theorem part_a_int_values (n : ℤ) :
  ∃ k : ℤ, (n^4 + 3) = k * (n^2 + n + 1) ↔ n = -3 ∨ n = -1 ∨ n = 0 :=
sorry

-- Part (b)
theorem part_b_int_values (n : ℤ) :
  ∃ m : ℤ, (n^3 + n + 1) = m * (n^2 - n + 1) ↔ n = 0 ∨ n = 1 :=
sorry

end NUMINAMATH_GPT_part_a_int_values_part_b_int_values_l1512_151259


namespace NUMINAMATH_GPT_count_library_books_l1512_151276

theorem count_library_books (initial_library_books : ℕ) 
  (books_given_away : ℕ) (books_added_from_source : ℕ) (books_donated : ℕ) 
  (h1 : initial_library_books = 125)
  (h2 : books_given_away = 42)
  (h3 : books_added_from_source = 68)
  (h4 : books_donated = 31) : 
  initial_library_books - books_given_away - books_donated = 52 :=
by sorry

end NUMINAMATH_GPT_count_library_books_l1512_151276


namespace NUMINAMATH_GPT_find_y_l1512_151248

theorem find_y (y : ℝ) (h : 2 * y / 3 = 12) : y = 18 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l1512_151248


namespace NUMINAMATH_GPT_prove_mutually_exclusive_and_exhaustive_events_l1512_151284

-- Definitions of conditions
def number_of_boys : ℕ := 3
def number_of_girls : ℕ := 2

-- Definitions of options
def option_A : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ ¬b3 ∧ ¬g1 ∧ g2)  -- Exactly 1 boy and exactly 2 girls
def option_B : Prop := (∃ (b1 b2 b3 : Bool), b1 ∧ b2 ∧ b3)  -- At least 1 boy and all boys
def option_C : Prop := (∃ (b1 b2 b3 : Bool) (g1 g2 : Bool), b1 ∧ b2 ∧ (b3 ∨ g1 ∨ g2))  -- At least 1 boy and at least 1 girl
def option_D : Prop := (∃ (b1 b2 : Bool) (g3 : Bool), b1 ∧ ¬b2 ∧ g3)  -- At least 1 boy and all girls

-- The proof statement showing that option_D == Mutually Exclusive and Exhaustive Events
theorem prove_mutually_exclusive_and_exhaustive_events : option_D :=
sorry

end NUMINAMATH_GPT_prove_mutually_exclusive_and_exhaustive_events_l1512_151284


namespace NUMINAMATH_GPT_annie_job_time_l1512_151267

noncomputable def annie_time : ℝ :=
  let dan_time := 15
  let dan_rate := 1 / dan_time
  let dan_hours := 6
  let fraction_done_by_dan := dan_rate * dan_hours
  let fraction_left_for_annie := 1 - fraction_done_by_dan
  let annie_work_remaining := fraction_left_for_annie
  let annie_hours := 6
  let annie_rate := annie_work_remaining / annie_hours
  let annie_time := 1 / annie_rate 
  annie_time

theorem annie_job_time :
  annie_time = 3.6 := 
sorry

end NUMINAMATH_GPT_annie_job_time_l1512_151267


namespace NUMINAMATH_GPT_shopkeeper_loss_percentages_l1512_151274

theorem shopkeeper_loss_percentages 
  (TypeA : Type) (TypeB : Type) (TypeC : Type)
  (theft_percentage_A : ℝ) (theft_percentage_B : ℝ) (theft_percentage_C : ℝ)
  (hA : theft_percentage_A = 0.20)
  (hB : theft_percentage_B = 0.25)
  (hC : theft_percentage_C = 0.30)
  :
  (theft_percentage_A = 0.20 ∧ theft_percentage_B = 0.25 ∧ theft_percentage_C = 0.30) ∧
  ((theft_percentage_A + theft_percentage_B + theft_percentage_C) / 3 = 0.25) :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_percentages_l1512_151274


namespace NUMINAMATH_GPT_gcd_765432_654321_eq_3_l1512_151217

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end NUMINAMATH_GPT_gcd_765432_654321_eq_3_l1512_151217


namespace NUMINAMATH_GPT_paving_stone_width_l1512_151241

theorem paving_stone_width 
    (length_courtyard : ℝ)
    (width_courtyard : ℝ)
    (length_paving_stone : ℝ)
    (num_paving_stones : ℕ)
    (total_area_courtyard : ℝ)
    (total_area_paving_stones : ℝ)
    (width_paving_stone : ℝ)
    (h1 : length_courtyard = 20)
    (h2 : width_courtyard = 16.5)
    (h3 : length_paving_stone = 2.5)
    (h4 : num_paving_stones = 66)
    (h5 : total_area_courtyard = length_courtyard * width_courtyard)
    (h6 : total_area_paving_stones = num_paving_stones * (length_paving_stone * width_paving_stone))
    (h7 : total_area_courtyard = total_area_paving_stones) :
    width_paving_stone = 2 :=
by
  sorry

end NUMINAMATH_GPT_paving_stone_width_l1512_151241


namespace NUMINAMATH_GPT_sins_prayers_l1512_151222

structure Sins :=
  (pride : Nat)
  (slander : Nat)
  (laziness : Nat)
  (adultery : Nat)
  (gluttony : Nat)
  (self_love : Nat)
  (jealousy : Nat)
  (malicious_gossip : Nat)

def prayer_requirements (s : Sins) : Nat × Nat × Nat :=
  ( s.pride + 2 * s.laziness + 10 * s.adultery + s.gluttony,
    2 * s.pride + 2 * s.slander + 10 * s.adultery + 3 * s.self_love + 3 * s.jealousy + 7 * s.malicious_gossip,
    7 * s.slander + 10 * s.adultery + s.self_love + 2 * s.malicious_gossip )

theorem sins_prayers (sins : Sins) :
  sins.pride = 0 ∧
  sins.slander = 1 ∧
  sins.laziness = 0 ∧
  sins.adultery = 0 ∧
  sins.gluttony = 9 ∧
  sins.self_love = 1 ∧
  sins.jealousy = 0 ∧
  sins.malicious_gossip = 2 ∧
  (sins.pride + sins.slander + sins.laziness + sins.adultery + sins.gluttony + sins.self_love + sins.jealousy + sins.malicious_gossip = 12) ∧
  prayer_requirements sins = (9, 12, 10) :=
  by
  sorry

end NUMINAMATH_GPT_sins_prayers_l1512_151222


namespace NUMINAMATH_GPT_find_a8_l1512_151261

-- Define the arithmetic sequence and the given conditions
variable {α : Type} [AddCommGroup α] [MulAction ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ}
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 5 + a 6 = 22
axiom h3 : a 3 = 7

theorem find_a8 : a 8 = 15 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_a8_l1512_151261


namespace NUMINAMATH_GPT_primes_div_order_l1512_151285

theorem primes_div_order (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : q ∣ 3^p - 2^p) : p ∣ q - 1 :=
sorry

end NUMINAMATH_GPT_primes_div_order_l1512_151285


namespace NUMINAMATH_GPT_find_a_l1512_151277

def A (x : ℝ) := (x^2 - 4 ≤ 0)
def B (x : ℝ) (a : ℝ) := (2 * x + a ≤ 0)
def C (x : ℝ) := (-2 ≤ x ∧ x ≤ 1)

theorem find_a (a : ℝ) : (∀ x : ℝ, A x → B x a → C x) → a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l1512_151277


namespace NUMINAMATH_GPT_hydrogen_atoms_count_l1512_151227

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Given conditions
def total_molecular_weight : ℝ := 88
def number_of_C_atoms : ℕ := 4
def number_of_O_atoms : ℕ := 2

theorem hydrogen_atoms_count (nh : ℕ) 
  (h_molecular_weight : total_molecular_weight = 88) 
  (h_C_atoms : number_of_C_atoms = 4) 
  (h_O_atoms : number_of_O_atoms = 2) :
  nh = 8 :=
by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_hydrogen_atoms_count_l1512_151227


namespace NUMINAMATH_GPT_find_z_l1512_151205

theorem find_z 
  (m : ℕ)
  (h1 : (1^(m+1) / 5^(m+1)) * (1^18 / z^18) = 1 / (2 * 10^35))
  (hm : m = 34) :
  z = 4 := 
sorry

end NUMINAMATH_GPT_find_z_l1512_151205


namespace NUMINAMATH_GPT_Smiths_Backery_Pies_l1512_151230

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end NUMINAMATH_GPT_Smiths_Backery_Pies_l1512_151230


namespace NUMINAMATH_GPT_area_of_region_l1512_151231

theorem area_of_region :
  ∀ (x y : ℝ), (|2 * x - 2| + |3 * y - 3| ≤ 30) → (area_of_figure = 300) :=
sorry

end NUMINAMATH_GPT_area_of_region_l1512_151231


namespace NUMINAMATH_GPT_min_rectangles_needed_l1512_151281

theorem min_rectangles_needed : ∀ (n : ℕ), n = 12 → (n * n) / (3 * 2) = 24 :=
by sorry

end NUMINAMATH_GPT_min_rectangles_needed_l1512_151281


namespace NUMINAMATH_GPT_square_of_binomial_l1512_151247

theorem square_of_binomial (c : ℝ) (h : ∃ a : ℝ, x^2 + 50 * x + c = (x + a)^2) : c = 625 :=
by
  sorry

end NUMINAMATH_GPT_square_of_binomial_l1512_151247


namespace NUMINAMATH_GPT_inequality_proof_l1512_151220

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l1512_151220


namespace NUMINAMATH_GPT_number_of_players_tournament_l1512_151204

theorem number_of_players_tournament (n : ℕ) : 
  (2 * n * (n - 1) = 272) → n = 17 :=
by
  sorry

end NUMINAMATH_GPT_number_of_players_tournament_l1512_151204


namespace NUMINAMATH_GPT_most_stable_scores_l1512_151294

-- Definitions for the variances of students A, B, and C
def s_A_2 : ℝ := 6
def s_B_2 : ℝ := 24
def s_C_2 : ℝ := 50

-- The proof that student A has the most stable math scores
theorem most_stable_scores : 
  s_A_2 < s_B_2 ∧ s_B_2 < s_C_2 → 
  ("Student A has the most stable scores" = "Student A has the most stable scores") :=
by
  intros h
  sorry

end NUMINAMATH_GPT_most_stable_scores_l1512_151294


namespace NUMINAMATH_GPT_improper_fraction_2012a_div_b_l1512_151223

theorem improper_fraction_2012a_div_b
  (a b : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) :
  2012 * a > b :=
by 
  sorry

end NUMINAMATH_GPT_improper_fraction_2012a_div_b_l1512_151223


namespace NUMINAMATH_GPT_solve_abc_l1512_151242

theorem solve_abc (a b c : ℕ) (h1 : a > b ∧ b > c) 
  (h2 : 34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0) 
  (h3 : 79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0) : 
  a = 10 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end NUMINAMATH_GPT_solve_abc_l1512_151242
