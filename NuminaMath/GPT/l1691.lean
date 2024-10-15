import Mathlib

namespace NUMINAMATH_GPT_triangle_ineq_l1691_169182

noncomputable def TriangleSidesProof (AB AC BC : ℝ) :=
  AB = AC ∧ BC = 10 ∧ 2 * AB + BC ≤ 44 → 5 < AB ∧ AB ≤ 17

-- Statement for the proof problem
theorem triangle_ineq (AB AC BC : ℝ) (h1 : AB = AC) (h2 : BC = 10) (h3 : 2 * AB + BC ≤ 44) :
  5 < AB ∧ AB ≤ 17 :=
sorry

end NUMINAMATH_GPT_triangle_ineq_l1691_169182


namespace NUMINAMATH_GPT_geometric_sequence_a3_eq_sqrt_5_l1691_169100

theorem geometric_sequence_a3_eq_sqrt_5 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * r)
  (h_a1 : a 1 = 1) (h_a5 : a 5 = 5) :
  a 3 = Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_a3_eq_sqrt_5_l1691_169100


namespace NUMINAMATH_GPT_exists_negative_number_satisfying_inequality_l1691_169149

theorem exists_negative_number_satisfying_inequality :
  ∃ x : ℝ, x < 0 ∧ (1 + x) * (1 - 9 * x) > 0 :=
sorry

end NUMINAMATH_GPT_exists_negative_number_satisfying_inequality_l1691_169149


namespace NUMINAMATH_GPT_solution_part_for_a_l1691_169172

noncomputable def find_k (k x y n : ℕ) : Prop :=
  gcd x y = 1 ∧ 
  x > 0 ∧ y > 0 ∧ 
  k % (x^2) = 0 ∧ 
  k % (y^2) = 0 ∧ 
  k / (x^2) = n ∧ 
  k / (y^2) = n + 148

theorem solution_part_for_a (k x y n : ℕ) (h : find_k k x y n) : k = 467856 :=
sorry

end NUMINAMATH_GPT_solution_part_for_a_l1691_169172


namespace NUMINAMATH_GPT_range_of_s_triangle_l1691_169136

theorem range_of_s_triangle (inequalities_form_triangle : Prop) : 
  (0 < s ∧ s ≤ 2) ∨ (s ≥ 4) ↔ inequalities_form_triangle := 
sorry

end NUMINAMATH_GPT_range_of_s_triangle_l1691_169136


namespace NUMINAMATH_GPT_determine_coordinates_of_M_l1691_169186

def point_in_fourth_quadrant (M : ℝ × ℝ) : Prop :=
  M.1 > 0 ∧ M.2 < 0

def distance_to_x_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.2| = d

def distance_to_y_axis (M : ℝ × ℝ) (d : ℝ) : Prop :=
  |M.1| = d

theorem determine_coordinates_of_M :
  ∃ M : ℝ × ℝ, point_in_fourth_quadrant M ∧ distance_to_x_axis M 3 ∧ distance_to_y_axis M 4 ∧ M = (4, -3) :=
by
  sorry

end NUMINAMATH_GPT_determine_coordinates_of_M_l1691_169186


namespace NUMINAMATH_GPT_correct_options_l1691_169178

theorem correct_options (a b : ℝ) (h : a > 0) (ha : a^2 = 4 * b) :
  ((a^2 - b^2 ≤ 4) ∧ (a^2 + 1 / b ≥ 4) ∧ (¬ (∃ x1 x2, x1 * x2 > 0 ∧ x^2 + a * x - b < 0)) ∧ 
  (∀ (x1 x2 : ℝ), |x1 - x2| = 4 → x^2 + a * x + b < 4 → 4 = 4)) :=
sorry

end NUMINAMATH_GPT_correct_options_l1691_169178


namespace NUMINAMATH_GPT_choose_socks_l1691_169105

open Nat

theorem choose_socks :
  (Nat.choose 8 4) = 70 :=
by 
  sorry

end NUMINAMATH_GPT_choose_socks_l1691_169105


namespace NUMINAMATH_GPT_alice_savings_l1691_169157

noncomputable def commission (sales : ℝ) : ℝ := 0.02 * sales
noncomputable def totalEarnings (basic_salary commission : ℝ) : ℝ := basic_salary + commission
noncomputable def savings (total_earnings : ℝ) : ℝ := 0.10 * total_earnings

theorem alice_savings (sales basic_salary : ℝ) (commission_rate savings_rate : ℝ) :
  commission_rate = 0.02 →
  savings_rate = 0.10 →
  sales = 2500 →
  basic_salary = 240 →
  savings (totalEarnings basic_salary (commission_rate * sales)) = 29 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_alice_savings_l1691_169157


namespace NUMINAMATH_GPT_intersection_point_divides_chord_l1691_169155

theorem intersection_point_divides_chord (R AB PO : ℝ)
    (hR: R = 11) (hAB: AB = 18) (hPO: PO = 7) :
    ∃ (AP PB : ℝ), (AP / PB = 2 ∨ AP / PB = 1 / 2) ∧ (AP + PB = AB) := by
  sorry

end NUMINAMATH_GPT_intersection_point_divides_chord_l1691_169155


namespace NUMINAMATH_GPT_range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l1691_169161

-- Question 1
def prop_p (x : ℝ) : Prop := (x + 1) * (x - 5) ≤ 0
def prop_q (x m : ℝ) : Prop := 1 - m ≤ x + 1 ∧ x + 1 < 1 + m ∧ m > 0
def neg_p (x : ℝ) : Prop := ¬ prop_p x
def neg_q (x m : ℝ) : Prop := ¬ prop_q x m

theorem range_m_if_neg_p_implies_neg_q : 
  (∀ x, neg_p x → neg_q x m) → 0 < m ∧ m ≤ 1 :=
by
  sorry

-- Question 2
theorem range_x_if_m_is_5_and_p_or_q_true_p_and_q_false : 
  (∀ x, (prop_p x ∨ prop_q x 5) ∧ ¬ (prop_p x ∧ prop_q x 5)) → 
  ∀ x, (x = 5 ∨ (-5 ≤ x ∧ x < -1)) :=
by
  sorry

end NUMINAMATH_GPT_range_m_if_neg_p_implies_neg_q_range_x_if_m_is_5_and_p_or_q_true_p_and_q_false_l1691_169161


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1691_169193

-- Problem 1 statement
theorem simplify_expression1 (a b : ℤ) : 2 * (2 * b - 3 * a) + 3 * (2 * a - 3 * b) = -5 * b :=
  by
  sorry

-- Problem 2 statement
theorem simplify_expression2 (a b : ℤ) : 4 * a^2 + 2 * (3 * a * b - 2 * a^2) - (7 * a * b - 1) = -a * b + 1 :=
  by
  sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l1691_169193


namespace NUMINAMATH_GPT_total_income_in_june_l1691_169160

-- Establishing the conditions
def daily_production : ℕ := 200
def days_in_june : ℕ := 30
def price_per_gallon : ℝ := 3.55

-- Defining total milk production in June as a function of daily production and days in June
def total_milk_production_in_june : ℕ :=
  daily_production * days_in_june

-- Defining total income as a function of milk production and price per gallon
def total_income (milk_production : ℕ) (price : ℝ) : ℝ :=
  milk_production * price

-- Stating the theorem that we need to prove
theorem total_income_in_june :
  total_income total_milk_production_in_june price_per_gallon = 21300 := 
sorry

end NUMINAMATH_GPT_total_income_in_june_l1691_169160


namespace NUMINAMATH_GPT_arithmetic_seq_8th_term_l1691_169133

theorem arithmetic_seq_8th_term (a d : ℤ) (h1 : a + 3 * d = 23) (h2 : a + 5 * d = 47) : a + 7 * d = 71 := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_8th_term_l1691_169133


namespace NUMINAMATH_GPT_yogurt_combinations_l1691_169175

theorem yogurt_combinations (f : ℕ) (t : ℕ) (h_f : f = 4) (h_t : t = 6) :
  (f * (t.choose 2) = 60) :=
by
  rw [h_f, h_t]
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l1691_169175


namespace NUMINAMATH_GPT_fresh_grape_weight_l1691_169140

variable (D : ℝ) (F : ℝ)

axiom dry_grape_weight : D = 66.67
axiom fresh_grape_water_content : F * 0.25 = D * 0.75

theorem fresh_grape_weight : F = 200.01 :=
by sorry

end NUMINAMATH_GPT_fresh_grape_weight_l1691_169140


namespace NUMINAMATH_GPT_complex_expression_evaluation_l1691_169164

theorem complex_expression_evaluation (z : ℂ) (h : z = 1 - I) :
  (z^2 - 2 * z) / (z - 1) = -2 * I :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_evaluation_l1691_169164


namespace NUMINAMATH_GPT_polynomial_root_sum_nonnegative_l1691_169146

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (p q : ℝ) : ℝ := x^2 + p * x + q

theorem polynomial_root_sum_nonnegative 
  (m1 m2 k1 k2 b c p q : ℝ)
  (h1 : f m1 b c = 0) (h2 : f m2 b c = 0)
  (h3 : g k1 p q = 0) (h4 : g k2 p q = 0) :
  f k1 b c + f k2 b c + g m1 p q + g m2 p q ≥ 0 := 
by
  sorry  -- Proof placeholders

end NUMINAMATH_GPT_polynomial_root_sum_nonnegative_l1691_169146


namespace NUMINAMATH_GPT_product_of_solutions_l1691_169114

theorem product_of_solutions :
  (∃ x y : ℝ, (|x^2 - 6 * x| + 5 = 41) ∧ (|y^2 - 6 * y| + 5 = 41) ∧ x ≠ y ∧ x * y = -36) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l1691_169114


namespace NUMINAMATH_GPT_John_pays_2400_per_year_l1691_169148

theorem John_pays_2400_per_year
  (hours_per_month : ℕ)
  (average_length : ℕ)
  (cost_per_song : ℕ)
  (h1 : hours_per_month = 20)
  (h2 : average_length = 3)
  (h3 : cost_per_song = 50) :
  (hours_per_month * 60 / average_length * cost_per_song * 12 = 2400) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end NUMINAMATH_GPT_John_pays_2400_per_year_l1691_169148


namespace NUMINAMATH_GPT_largest_integer_less_100_leaves_remainder_4_l1691_169106

theorem largest_integer_less_100_leaves_remainder_4 (x n : ℕ) (h1 : x = 6 * n + 4) (h2 : x < 100) : x = 94 :=
  sorry

end NUMINAMATH_GPT_largest_integer_less_100_leaves_remainder_4_l1691_169106


namespace NUMINAMATH_GPT_sum_of_square_roots_l1691_169127

theorem sum_of_square_roots : 
  (Real.sqrt 1) + (Real.sqrt (1 + 3)) + (Real.sqrt (1 + 3 + 5)) + (Real.sqrt (1 + 3 + 5 + 7)) = 10 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_square_roots_l1691_169127


namespace NUMINAMATH_GPT_minute_hand_angle_45min_l1691_169121

theorem minute_hand_angle_45min
  (duration : ℝ)
  (h1 : duration = 45) :
  (-(3 / 4) * 2 * Real.pi = - (3 * Real.pi / 2)) :=
by
  sorry

end NUMINAMATH_GPT_minute_hand_angle_45min_l1691_169121


namespace NUMINAMATH_GPT_farmer_plants_rows_per_bed_l1691_169187

theorem farmer_plants_rows_per_bed 
    (bean_seedlings : ℕ) (beans_per_row : ℕ)
    (pumpkin_seeds : ℕ) (pumpkins_per_row : ℕ)
    (radishes : ℕ) (radishes_per_row : ℕ)
    (plant_beds : ℕ)
    (h1 : bean_seedlings = 64)
    (h2 : beans_per_row = 8)
    (h3 : pumpkin_seeds = 84)
    (h4 : pumpkins_per_row = 7)
    (h5 : radishes = 48)
    (h6 : radishes_per_row = 6)
    (h7 : plant_beds = 14) : 
    (bean_seedlings / beans_per_row + pumpkin_seeds / pumpkins_per_row + radishes / radishes_per_row) / plant_beds = 2 :=
by
  sorry

end NUMINAMATH_GPT_farmer_plants_rows_per_bed_l1691_169187


namespace NUMINAMATH_GPT_winning_candidate_percentage_l1691_169166

noncomputable def votes : List ℝ := [15236.71, 20689.35, 12359.23, 30682.49, 25213.17, 18492.93]

theorem winning_candidate_percentage :
  (List.foldr max 0 votes / (List.foldr (· + ·) 0 votes) * 100) = 25.01 :=
by
  sorry

end NUMINAMATH_GPT_winning_candidate_percentage_l1691_169166


namespace NUMINAMATH_GPT_tina_savings_l1691_169128

theorem tina_savings :
  let june_savings : ℕ := 27
  let july_savings : ℕ := 14
  let august_savings : ℕ := 21
  let books_spending : ℕ := 5
  let shoes_spending : ℕ := 17
  let total_savings := june_savings + july_savings + august_savings
  let total_spending := books_spending + shoes_spending
  let remaining_money := total_savings - total_spending
  remaining_money = 40 :=
by
  sorry

end NUMINAMATH_GPT_tina_savings_l1691_169128


namespace NUMINAMATH_GPT_circle_radius_square_l1691_169168

-- Definition of the problem setup
variables {EF GH ER RF GS SH R S : ℝ}

-- Given conditions
def condition1 : ER = 23 := by sorry
def condition2 : RF = 23 := by sorry
def condition3 : GS = 31 := by sorry
def condition4 : SH = 15 := by sorry

-- Circle radius to be proven
def radius_squared : ℝ := 706

-- Lean 4 theorem statement
theorem circle_radius_square (h1 : ER = 23) (h2 : RF = 23) (h3 : GS = 31) (h4 : SH = 15) :
  (r : ℝ) ^ 2 = 706 := sorry

end NUMINAMATH_GPT_circle_radius_square_l1691_169168


namespace NUMINAMATH_GPT_nth_equation_l1691_169154

theorem nth_equation (n : ℕ) (hn: n ≥ 1) : 
  (n+1) / ((n+1)^2 - 1) - 1 / (n * (n+1) * (n+2)) = 1 / (n+1) :=
by
  sorry

end NUMINAMATH_GPT_nth_equation_l1691_169154


namespace NUMINAMATH_GPT_find_m_from_equation_l1691_169120

theorem find_m_from_equation :
  ∀ (x m : ℝ), (x^2 + 2 * x - 1 = 0) → ((x + m)^2 = 2) → m = 1 :=
by
  intros x m h1 h2
  sorry

end NUMINAMATH_GPT_find_m_from_equation_l1691_169120


namespace NUMINAMATH_GPT_avg_mark_excluded_students_l1691_169196

-- Define the given conditions
variables (n : ℕ) (A A_remaining : ℕ) (excluded_count : ℕ)
variable (T : ℕ := n * A)
variable (T_remaining : ℕ := (n - excluded_count) * A_remaining)
variable (T_excluded : ℕ := T - T_remaining)

-- Define the problem statement
theorem avg_mark_excluded_students (h1: n = 14) (h2: A = 65) (h3: A_remaining = 90) (h4: excluded_count = 5) :
   T_excluded / excluded_count = 20 :=
by
  sorry

end NUMINAMATH_GPT_avg_mark_excluded_students_l1691_169196


namespace NUMINAMATH_GPT_express_a_in_terms_of_b_l1691_169143

noncomputable def a : ℝ := Real.log 1250 / Real.log 6
noncomputable def b : ℝ := Real.log 50 / Real.log 3

theorem express_a_in_terms_of_b : a = (b + 0.6826) / 1.2619 :=
by
  sorry

end NUMINAMATH_GPT_express_a_in_terms_of_b_l1691_169143


namespace NUMINAMATH_GPT_range_of_a_for_circle_l1691_169101

theorem range_of_a_for_circle (a : ℝ) : 
  -2 < a ∧ a < 2/3 ↔ 
  ∃ (x y : ℝ), (x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1) = 0 :=
sorry

end NUMINAMATH_GPT_range_of_a_for_circle_l1691_169101


namespace NUMINAMATH_GPT_difference_between_local_and_face_value_l1691_169129

def numeral := 657903

def local_value (n : ℕ) : ℕ :=
  if n = 7 then 70000 else 0

def face_value (n : ℕ) : ℕ :=
  n

theorem difference_between_local_and_face_value :
  local_value 7 - face_value 7 = 69993 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_local_and_face_value_l1691_169129


namespace NUMINAMATH_GPT_solve_system_l1691_169165

variable {R : Type*} [CommRing R] {a b c x y z : R}

theorem solve_system (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h₁ : z + a*y + a^2*x + a^3 = 0) 
  (h₂ : z + b*y + b^2*x + b^3 = 0) 
  (h₃ : z + c*y + c^2*x + c^3 = 0) :
  x = -(a + b + c) ∧ y = (a * b + a * c + b * c) ∧ z = -(a * b * c) := 
sorry

end NUMINAMATH_GPT_solve_system_l1691_169165


namespace NUMINAMATH_GPT_capacitor_capacitance_l1691_169139

theorem capacitor_capacitance 
  (U ε Q : ℝ) 
  (hQ : Q = (U^2 * (ε - 1)^2 * C) /  (2 * ε * (ε + 1)))
  : C = (2 * ε * (ε + 1) * Q) / (U^2 * (ε - 1)^2) :=
by
  sorry

end NUMINAMATH_GPT_capacitor_capacitance_l1691_169139


namespace NUMINAMATH_GPT_sets_B_C_D_represent_same_function_l1691_169184

theorem sets_B_C_D_represent_same_function :
  (∀ x : ℝ, (2 * x = 2 * (x ^ (3 : ℝ) ^ (1 / 3)))) ∧
  (∀ x t : ℝ, (x ^ 2 + x + 3 = t ^ 2 + t + 3)) ∧
  (∀ x : ℝ, (x ^ 2 = (x ^ 4) ^ (1 / 2))) :=
by
  sorry

end NUMINAMATH_GPT_sets_B_C_D_represent_same_function_l1691_169184


namespace NUMINAMATH_GPT_frequency_of_8th_group_l1691_169194

theorem frequency_of_8th_group :
  let sample_size := 100
  let freq1 := 15
  let freq2 := 17
  let freq3 := 11
  let freq4 := 13
  let freq_5_to_7 := 0.32 * sample_size
  let total_freq_1_to_4 := freq1 + freq2 + freq3 + freq4
  let remaining_freq := sample_size - total_freq_1_to_4
  let freq8 := remaining_freq - freq_5_to_7
  (freq8 / sample_size = 0.12) :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_8th_group_l1691_169194


namespace NUMINAMATH_GPT_find_num_alligators_l1691_169107

-- We define the conditions as given in the problem
def journey_to_delta_hours : ℕ := 4
def extra_hours : ℕ := 2
def combined_time_alligators_walked : ℕ := 46

-- We define the hypothesis in terms of Lean variables
def num_alligators_traveled_with_Paul (A : ℕ) : Prop :=
  (journey_to_delta_hours + (journey_to_delta_hours + extra_hours) * A) = combined_time_alligators_walked

-- Now the theorem statement where we prove that the number of alligators (A) is 7
theorem find_num_alligators :
  ∃ A : ℕ, num_alligators_traveled_with_Paul A ∧ A = 7 :=
by
  existsi 7
  unfold num_alligators_traveled_with_Paul
  simp
  sorry -- this is where the actual proof would go

end NUMINAMATH_GPT_find_num_alligators_l1691_169107


namespace NUMINAMATH_GPT_prime_p4_minus_one_sometimes_divisible_by_48_l1691_169180

theorem prime_p4_minus_one_sometimes_divisible_by_48 (p : ℕ) (hp : Nat.Prime p) (hge : p ≥ 7) : 
  ∃ k : ℕ, k ≥ 1 ∧ 48 ∣ p^4 - 1 :=
sorry

end NUMINAMATH_GPT_prime_p4_minus_one_sometimes_divisible_by_48_l1691_169180


namespace NUMINAMATH_GPT_maximum_value_l1691_169110

noncomputable def conditions (m n t : ℝ) : Prop :=
  -- m, n, t are positive real numbers
  (0 < m) ∧ (0 < n) ∧ (0 < t) ∧
  -- Equation condition
  (m^2 - 3 * m * n + 4 * n^2 - t = 0)

noncomputable def minimum_u (m n t : ℝ) : Prop :=
  -- Minimum value condition for t / mn
  (t / (m * n) = 1)

theorem maximum_value (m n t : ℝ) (h1 : conditions m n t) (h2 : minimum_u m n t) :
  -- Proving the maximum value of m + 2n - t
  (m + 2 * n - t) = 2 :=
sorry

end NUMINAMATH_GPT_maximum_value_l1691_169110


namespace NUMINAMATH_GPT_sin_double_angle_l1691_169113

theorem sin_double_angle {θ : ℝ} (h : Real.tan θ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 := 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1691_169113


namespace NUMINAMATH_GPT_scientific_notation_example_l1691_169170

theorem scientific_notation_example :
  284000000 = 2.84 * 10^8 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_example_l1691_169170


namespace NUMINAMATH_GPT_cyrus_pages_proof_l1691_169142

def pages_remaining (total_pages: ℝ) (day1: ℝ) (day2: ℝ) (day3: ℝ) (day4: ℝ) (day5: ℝ) : ℝ :=
  total_pages - (day1 + day2 + day3 + day4 + day5)

theorem cyrus_pages_proof :
  let total_pages := 750
  let day1 := 30
  let day2 := 1.5 * day1
  let day3 := day2 / 2
  let day4 := 2.5 * day3
  let day5 := 15
  pages_remaining total_pages day1 day2 day3 day4 day5 = 581.25 :=
by 
  sorry

end NUMINAMATH_GPT_cyrus_pages_proof_l1691_169142


namespace NUMINAMATH_GPT_math_problem_l1691_169144

variables {a b c d e : ℤ}

theorem math_problem 
(h1 : a - b + c - e = 7)
(h2 : b - c + d + e = 9)
(h3 : c - d + a - e = 5)
(h4 : d - a + b + e = 1)
: a + b + c + d + e = 11 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l1691_169144


namespace NUMINAMATH_GPT_find_constant_a_l1691_169163

theorem find_constant_a (a : ℝ) : 
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ ax^2 + 2 * a * x + 1 = 9) → (a = -8 ∨ a = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_constant_a_l1691_169163


namespace NUMINAMATH_GPT_contractor_absent_days_l1691_169132

noncomputable def solve_contractor_problem : Prop :=
  ∃ (x y : ℕ), 
    x + y = 30 ∧ 
    25 * x - 750 / 100 * y = 555 ∧
    y = 6

theorem contractor_absent_days : solve_contractor_problem :=
  sorry

end NUMINAMATH_GPT_contractor_absent_days_l1691_169132


namespace NUMINAMATH_GPT_parabola_shifted_l1691_169192

-- Define the original parabola
def originalParabola (x : ℝ) : ℝ := (x + 2)^2 + 3

-- Shift the parabola by 3 units to the right
def shiftedRight (x : ℝ) : ℝ := originalParabola (x - 3)

-- Then shift the parabola 2 units down
def shiftedRightThenDown (x : ℝ) : ℝ := shiftedRight x - 2

-- The problem asks to prove that the final expression is equal to (x - 1)^2 + 1
theorem parabola_shifted (x : ℝ) : shiftedRightThenDown x = (x - 1)^2 + 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_shifted_l1691_169192


namespace NUMINAMATH_GPT_cone_volume_proof_l1691_169112

noncomputable def cone_volume (l h : ℕ) : ℝ :=
  let r := Real.sqrt (l^2 - h^2)
  1 / 3 * Real.pi * r^2 * h

theorem cone_volume_proof :
  cone_volume 13 12 = 100 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_proof_l1691_169112


namespace NUMINAMATH_GPT_katya_classmates_l1691_169190

-- Let N be the number of Katya's classmates
variable (N : ℕ)

-- Let K be the number of candies Artyom initially received
variable (K : ℕ)

-- Condition 1: After distributing some candies, Katya had 10 more candies left than Artyom
def condition_1 := K + 10

-- Condition 2: Katya gave each child, including herself, one more candy, so she gave out N + 1 candies in total
def condition_2 := N + 1

-- Condition 3: After giving out these N + 1 candies, everyone in the class has the same number of candies.
def condition_3 : Prop := (K + 1) = (condition_1 K - condition_2 N) / (N + 1)


-- Goal: Prove the number of Katya's classmates N is 9.
theorem katya_classmates : N = 9 :=
by
  -- Restate the conditions in Lean
  
  -- Apply the conditions to find that the only viable solution is N = 9
  sorry

end NUMINAMATH_GPT_katya_classmates_l1691_169190


namespace NUMINAMATH_GPT_number_of_people_in_group_l1691_169199

theorem number_of_people_in_group :
  ∃ (N : ℕ), (∀ (avg_weight : ℝ), 
  ∃ (new_person_weight : ℝ) (replaced_person_weight : ℝ),
  new_person_weight = 85 ∧ replaced_person_weight = 65 ∧
  avg_weight + 2.5 = ((N * avg_weight + (new_person_weight - replaced_person_weight)) / N) ∧ 
  N = 8) :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_in_group_l1691_169199


namespace NUMINAMATH_GPT_nonneg_reals_ineq_l1691_169179

theorem nonneg_reals_ineq 
  (a b x y : ℝ)
  (ha : 0 ≤ a) (hb : 0 ≤ b)
  (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hab : a^5 + b^5 ≤ 1)
  (hxy : x^5 + y^5 ≤ 1) :
  a^2 * x^3 + b^2 * y^3 ≤ 1 :=
sorry

end NUMINAMATH_GPT_nonneg_reals_ineq_l1691_169179


namespace NUMINAMATH_GPT_measured_percentage_weight_loss_l1691_169177

variable (W : ℝ) -- W is the starting weight.
variable (weight_loss_percent : ℝ := 0.12) -- 12% weight loss.
variable (clothes_weight_percent : ℝ := 0.03) -- 3% clothes weight addition.
variable (beverage_weight_percent : ℝ := 0.005) -- 0.5% beverage weight addition.

theorem measured_percentage_weight_loss : 
  (W - ((0.88 * W) + (clothes_weight_percent * W) + (beverage_weight_percent * W))) / W * 100 = 8.5 :=
by
  sorry

end NUMINAMATH_GPT_measured_percentage_weight_loss_l1691_169177


namespace NUMINAMATH_GPT_graph_passes_through_fixed_point_l1691_169125

theorem graph_passes_through_fixed_point (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) :
    ∃ (x y : ℝ), (x = -3) ∧ (y = -1) ∧ (y = a^(x + 3) - 2) :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_fixed_point_l1691_169125


namespace NUMINAMATH_GPT_sum_of_first_11_terms_l1691_169173

theorem sum_of_first_11_terms (a1 d : ℝ) (h : 2 * a1 + 10 * d = 8) : 
  (11 / 2) * (2 * a1 + 10 * d) = 44 := 
by sorry

end NUMINAMATH_GPT_sum_of_first_11_terms_l1691_169173


namespace NUMINAMATH_GPT_cos_half_diff_proof_l1691_169123

noncomputable def cos_half_diff (A B C : ℝ) (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) : Real :=
  Real.cos ((A - C) / 2)

theorem cos_half_diff_proof (A B C : ℝ)
  (h_triangle : A + B + C = 180)
  (h_relation : A + C = 2 * B)
  (h_equation : (1 / Real.cos A) + (1 / Real.cos C) = - (Real.sqrt 2 / Real.cos B)) :
  cos_half_diff A B C h_triangle h_relation h_equation = -Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_cos_half_diff_proof_l1691_169123


namespace NUMINAMATH_GPT_three_in_A_even_not_in_A_l1691_169169

def A : Set ℤ := {x | ∃ m n : ℤ, x = m^2 - n^2}

-- (1) Prove that 3 ∈ A
theorem three_in_A : 3 ∈ A :=
sorry

-- (2) Prove that ∀ k ∈ ℤ, 4k - 2 ∉ A
theorem even_not_in_A (k : ℤ) : (4 * k - 2) ∉ A :=
sorry

end NUMINAMATH_GPT_three_in_A_even_not_in_A_l1691_169169


namespace NUMINAMATH_GPT_average_of_all_5_numbers_is_20_l1691_169131

def average_of_all_5_numbers
  (sum_3_numbers : ℕ)
  (avg_2_numbers : ℕ) : ℕ :=
(sum_3_numbers + 2 * avg_2_numbers) / 5

theorem average_of_all_5_numbers_is_20 :
  average_of_all_5_numbers 48 26 = 20 :=
by
  unfold average_of_all_5_numbers -- Expand the definition
  -- Sum of 5 numbers is 48 (sum of 3) + (2 * 26) (sum of other 2)
  -- Total sum is 48 + 52 = 100
  -- Average is 100 / 5 = 20
  norm_num -- Check the numeric calculation
  -- sorry

end NUMINAMATH_GPT_average_of_all_5_numbers_is_20_l1691_169131


namespace NUMINAMATH_GPT_largest_4_digit_div_by_5_smallest_primes_l1691_169117

noncomputable def LCM_5_smallest_primes : ℕ := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 5 (Nat.lcm 7 11)))

theorem largest_4_digit_div_by_5_smallest_primes :
  ∃ n, 1000 ≤ n ∧ n ≤ 9999 ∧ (∀ p ∈ [2, 3, 5, 7, 11], p ∣ n) ∧ n = 9240 := by
  sorry

end NUMINAMATH_GPT_largest_4_digit_div_by_5_smallest_primes_l1691_169117


namespace NUMINAMATH_GPT_expected_winnings_is_correct_l1691_169156

noncomputable def peculiar_die_expected_winnings : ℝ :=
  (1/4) * 2 + (1/2) * 5 + (1/4) * (-10)

theorem expected_winnings_is_correct :
  peculiar_die_expected_winnings = 0.5 := by
  sorry

end NUMINAMATH_GPT_expected_winnings_is_correct_l1691_169156


namespace NUMINAMATH_GPT_find_a_l1691_169174

open Set

noncomputable def A : Set ℝ := {x | x^2 - 2 * x - 8 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a^2 - 12 = 0}

theorem find_a (a : ℝ) : (A ∪ (B a) = A) ↔ (a = -2 ∨ a ≥ 4 ∨ a < -4) := by
  sorry

end NUMINAMATH_GPT_find_a_l1691_169174


namespace NUMINAMATH_GPT_length_of_road_l1691_169135

-- Definitions based on conditions
def trees : Nat := 10
def interval : Nat := 10

-- Statement of the theorem
theorem length_of_road 
  (trees : Nat) (interval : Nat) (beginning_planting : Bool) (h_trees : trees = 10) (h_interval : interval = 10) (h_beginning : beginning_planting = true) 
  : (trees - 1) * interval = 90 := 
by 
  sorry

end NUMINAMATH_GPT_length_of_road_l1691_169135


namespace NUMINAMATH_GPT_find_number_l1691_169152

theorem find_number (n : ℝ) (h : (1 / 3) * n = 6) : n = 18 :=
sorry

end NUMINAMATH_GPT_find_number_l1691_169152


namespace NUMINAMATH_GPT_rectangle_enclosed_by_lines_l1691_169102

theorem rectangle_enclosed_by_lines : 
  ∃ (ways : ℕ), 
  (ways = (Nat.choose 5 2) * (Nat.choose 4 2)) ∧ 
  ways = 60 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_enclosed_by_lines_l1691_169102


namespace NUMINAMATH_GPT_ratio_expression_value_l1691_169197

theorem ratio_expression_value (p q s u : ℚ) (h1 : p / q = 5 / 2) (h2 : s / u = 11 / 7) : 
  (5 * p * s - 3 * q * u) / (7 * q * u - 2 * p * s) = -233 / 12 :=
by {
  -- Proof will be provided here.
  sorry
}

end NUMINAMATH_GPT_ratio_expression_value_l1691_169197


namespace NUMINAMATH_GPT_fraction_relationships_l1691_169150

variables (a b c d : ℚ)

theorem fraction_relationships (h1 : a / b = 3) (h2 : b / c = 2 / 3) (h3 : c / d = 5) :
  d / a = 1 / 10 :=
by
  sorry

end NUMINAMATH_GPT_fraction_relationships_l1691_169150


namespace NUMINAMATH_GPT_simplification_problem_l1691_169124

theorem simplification_problem (p q r : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h_sum : p + q + r = 1) :
  (1 / (q^2 + r^2 - p^2) + 1 / (p^2 + r^2 - q^2) + 1 / (p^2 + q^2 - r^2) = 3 / (1 - 2 * q * r)) :=
by
  sorry

end NUMINAMATH_GPT_simplification_problem_l1691_169124


namespace NUMINAMATH_GPT_total_population_estimate_l1691_169188

def average_population_min : ℕ := 3200
def average_population_max : ℕ := 3600
def towns : ℕ := 25

theorem total_population_estimate : 
    ∃ x : ℕ, average_population_min ≤ x ∧ x ≤ average_population_max ∧ towns * x = 85000 :=
by 
  sorry

end NUMINAMATH_GPT_total_population_estimate_l1691_169188


namespace NUMINAMATH_GPT_rahul_matches_played_l1691_169145

theorem rahul_matches_played
  (current_avg : ℕ)
  (runs_today : ℕ)
  (new_avg : ℕ)
  (m: ℕ)
  (h1 : current_avg = 51)
  (h2 : runs_today = 78)
  (h3 : new_avg = 54)
  (h4 : (51 * m + runs_today) / (m + 1) = new_avg) :
  m = 8 :=
by
  sorry

end NUMINAMATH_GPT_rahul_matches_played_l1691_169145


namespace NUMINAMATH_GPT_scientific_notation_eight_million_l1691_169162

theorem scientific_notation_eight_million :
  ∃ a n, 8000000 = a * 10 ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8 ∧ n = 6 :=
by
  use 8
  use 6
  sorry

end NUMINAMATH_GPT_scientific_notation_eight_million_l1691_169162


namespace NUMINAMATH_GPT_total_weight_of_peppers_l1691_169153

def green_peppers := 0.3333333333333333
def red_peppers := 0.4444444444444444
def yellow_peppers := 0.2222222222222222
def orange_peppers := 0.7777777777777778

theorem total_weight_of_peppers :
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 :=
by
  sorry

end NUMINAMATH_GPT_total_weight_of_peppers_l1691_169153


namespace NUMINAMATH_GPT_sum_of_integers_square_greater_272_l1691_169108

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end NUMINAMATH_GPT_sum_of_integers_square_greater_272_l1691_169108


namespace NUMINAMATH_GPT_find_x_squared_minus_y_squared_l1691_169126

variable (x y : ℝ)

theorem find_x_squared_minus_y_squared 
(h1 : y + 6 = (x - 3)^2)
(h2 : x + 6 = (y - 3)^2)
(h3 : x ≠ y) :
x^2 - y^2 = 27 := by
  sorry

end NUMINAMATH_GPT_find_x_squared_minus_y_squared_l1691_169126


namespace NUMINAMATH_GPT_parallelogram_area_l1691_169176

theorem parallelogram_area (base height : ℝ) (h_base : base = 20) (h_height : height = 16) :
  base * height = 320 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l1691_169176


namespace NUMINAMATH_GPT_find_initial_population_l1691_169103

-- Define the conditions that the population increases annually by 20%
-- and that the population after 2 years is 14400.
def initial_population (P : ℝ) : Prop :=
  1.44 * P = 14400

-- The theorem states that given the conditions, the initial population is 10000.
theorem find_initial_population (P : ℝ) (h : initial_population P) : P = 10000 :=
  sorry

end NUMINAMATH_GPT_find_initial_population_l1691_169103


namespace NUMINAMATH_GPT_fish_to_apples_l1691_169159

variable {Fish Loaf Rice Apple : Type}
variable (f : Fish → ℝ) (l : Loaf → ℝ) (r : Rice → ℝ) (a : Apple → ℝ)
variable (F : Fish) (L : Loaf) (A : Apple) (R : Rice)

-- Conditions
axiom cond1 : 4 * f F = 3 * l L
axiom cond2 : l L = 5 * r R
axiom cond3 : r R = 2 * a A

-- Proof statement
theorem fish_to_apples : f F = 7.5 * a A :=
by
  sorry

end NUMINAMATH_GPT_fish_to_apples_l1691_169159


namespace NUMINAMATH_GPT_original_cost_of_dolls_l1691_169141

theorem original_cost_of_dolls 
  (x : ℝ) -- original cost of each Russian doll
  (savings : ℝ) -- total savings of Daniel
  (h1 : savings = 15 * x) -- Daniel saves enough to buy 15 dolls at original price
  (h2 : savings = 20 * 3) -- with discounted price, he can buy 20 dolls
  : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_original_cost_of_dolls_l1691_169141


namespace NUMINAMATH_GPT_jim_miles_remaining_l1691_169158

theorem jim_miles_remaining (total_miles : ℕ) (miles_driven : ℕ) (total_miles_eq : total_miles = 1200) (miles_driven_eq : miles_driven = 384) :
  total_miles - miles_driven = 816 :=
by
  sorry

end NUMINAMATH_GPT_jim_miles_remaining_l1691_169158


namespace NUMINAMATH_GPT_unique_solution_l1691_169183

-- Given conditions in the problem:
def prime (p : ℕ) : Prop := Nat.Prime p
def is_solution (p n k : ℕ) : Prop :=
  3 ^ p + 4 ^ p = n ^ k ∧ k > 1 ∧ prime p

-- The only solution:
theorem unique_solution (p n k : ℕ) :
  is_solution p n k → (p, n, k) = (2, 5, 2) := 
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1691_169183


namespace NUMINAMATH_GPT_volume_of_convex_solid_l1691_169151

variables {m V t6 T t3 : ℝ} 

-- Definition of the distance between the two parallel planes
def distance_between_planes (m : ℝ) : Prop := m > 0

-- Areas of the two parallel faces
def area_hexagon_face (t6 : ℝ) : Prop := t6 > 0
def area_triangle_face (t3 : ℝ) : Prop := t3 > 0

-- Area of the cross-section of the solid with a plane perpendicular to the height at its midpoint
def area_cross_section (T : ℝ) : Prop := T > 0

-- Volume of the convex solid
def volume_formula_holds (V m t6 T t3 : ℝ) : Prop :=
  V = (m / 6) * (t6 + 4 * T + t3)

-- Formal statement of the problem
theorem volume_of_convex_solid
  (m t6 t3 T V : ℝ)
  (h₁ : distance_between_planes m)
  (h₂ : area_hexagon_face t6)
  (h₃ : area_triangle_face t3)
  (h₄ : area_cross_section T) :
  volume_formula_holds V m t6 T t3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_convex_solid_l1691_169151


namespace NUMINAMATH_GPT_total_carriages_l1691_169130

theorem total_carriages (Euston Norfolk Norwich FlyingScotsman : ℕ) 
  (h1 : Euston = 130)
  (h2 : Norfolk = Euston - 20)
  (h3 : Norwich = 100)
  (h4 : FlyingScotsman = Norwich + 20) :
  Euston + Norfolk + Norwich + FlyingScotsman = 460 :=
by 
  sorry

end NUMINAMATH_GPT_total_carriages_l1691_169130


namespace NUMINAMATH_GPT_determine_x_value_l1691_169116

theorem determine_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y ^ 3) (h2 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by 
  sorry

end NUMINAMATH_GPT_determine_x_value_l1691_169116


namespace NUMINAMATH_GPT_trig_identity_l1691_169104

theorem trig_identity (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1 / 2) : 
  (Real.cos a / (Real.sin a - 1)) = 1 / 2 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_trig_identity_l1691_169104


namespace NUMINAMATH_GPT_upper_limit_of_range_l1691_169137

theorem upper_limit_of_range (n : ℕ) (h : (10 + 10 * n) / 2 = 255) : 10 * n = 500 :=
by 
  sorry

end NUMINAMATH_GPT_upper_limit_of_range_l1691_169137


namespace NUMINAMATH_GPT_coloring_count_l1691_169118

theorem coloring_count (n : ℕ) (h : 0 < n) :
  ∃ (num_colorings : ℕ), num_colorings = 2 :=
sorry

end NUMINAMATH_GPT_coloring_count_l1691_169118


namespace NUMINAMATH_GPT_blue_balls_in_JarB_l1691_169115

-- Defining the conditions
def ratio_white_blue (white blue : ℕ) : Prop := white / gcd white blue = 5 ∧ blue / gcd white blue = 3

def white_balls_in_B := 15

-- Proof statement
theorem blue_balls_in_JarB :
  ∃ (blue : ℕ), ratio_white_blue 15 blue ∧ blue = 9 :=
by {
  -- Proof outline (not required, thus just using sorry)
  sorry
}


end NUMINAMATH_GPT_blue_balls_in_JarB_l1691_169115


namespace NUMINAMATH_GPT_sales_tax_is_8_percent_l1691_169119

-- Define the conditions
def total_before_tax : ℝ := 150
def total_with_tax : ℝ := 162

-- Define the relationship to find the sales tax percentage
noncomputable def sales_tax_percent (before_tax after_tax : ℝ) : ℝ :=
  ((after_tax - before_tax) / before_tax) * 100

-- State the theorem to prove the sales tax percentage is 8%
theorem sales_tax_is_8_percent :
  sales_tax_percent total_before_tax total_with_tax = 8 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_sales_tax_is_8_percent_l1691_169119


namespace NUMINAMATH_GPT_option_C_represents_same_function_l1691_169109

-- Definitions of the functions from option C
def f (x : ℝ) := x^2 - 1
def g (t : ℝ) := t^2 - 1

-- The proof statement that needs to be proven
theorem option_C_represents_same_function :
  f = g :=
sorry

end NUMINAMATH_GPT_option_C_represents_same_function_l1691_169109


namespace NUMINAMATH_GPT_min_value_x_l1691_169195

open Real 

variable (x : ℝ)

theorem min_value_x (hx_pos : 0 < x) 
    (ineq : log x ≥ 2 * log 3 + (1 / 3) * log x + 1) : 
    x ≥ 27 * exp (3 / 2) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_x_l1691_169195


namespace NUMINAMATH_GPT_three_digit_division_l1691_169122

theorem three_digit_division (abc : ℕ) (a b c : ℕ) (h1 : 100 ≤ abc ∧ abc < 1000) (h2 : abc = 100 * a + 10 * b + c) (h3 : a ≠ 0) :
  (1001 * abc) / 7 / 11 / 13 = abc :=
by
  sorry

end NUMINAMATH_GPT_three_digit_division_l1691_169122


namespace NUMINAMATH_GPT_percent_yield_hydrogen_gas_l1691_169198

theorem percent_yield_hydrogen_gas
  (moles_fe : ℝ) (moles_h2so4 : ℝ) (actual_yield_h2 : ℝ) (theoretical_yield_h2 : ℝ) :
  moles_fe = 3 →
  moles_h2so4 = 4 →
  actual_yield_h2 = 1 →
  theoretical_yield_h2 = moles_fe →
  (actual_yield_h2 / theoretical_yield_h2) * 100 = 33.33 :=
by
  intros h_moles_fe h_moles_h2so4 h_actual_yield_h2 h_theoretical_yield_h2
  sorry

end NUMINAMATH_GPT_percent_yield_hydrogen_gas_l1691_169198


namespace NUMINAMATH_GPT_area_of_circle_l1691_169138

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 + 8 * x + 10 * y = -9 → 
  ∃ a : ℝ, a = 32 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l1691_169138


namespace NUMINAMATH_GPT_reciprocal_sum_hcf_lcm_l1691_169134

variables (m n : ℕ)

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem reciprocal_sum_hcf_lcm (h₁ : HCF m n = 6) (h₂ : LCM m n = 210) (h₃ : m + n = 60) :
  (1 : ℚ) / m + (1 : ℚ) / n = 1 / 21 :=
by
  -- The proof will be inserted here.
  sorry

end NUMINAMATH_GPT_reciprocal_sum_hcf_lcm_l1691_169134


namespace NUMINAMATH_GPT_movie_ticket_notation_l1691_169147

-- Definition of movie ticket notation
def ticket_notation (row : ℕ) (seat : ℕ) : (ℕ × ℕ) :=
  (row, seat)

-- Given condition: "row 10, seat 3" is denoted as (10, 3)
def given := ticket_notation 10 3 = (10, 3)

-- Proof statement: "row 6, seat 16" is denoted as (6, 16)
theorem movie_ticket_notation : ticket_notation 6 16 = (6, 16) :=
by
  -- Proof omitted, since the theorem statement is the focus
  sorry

end NUMINAMATH_GPT_movie_ticket_notation_l1691_169147


namespace NUMINAMATH_GPT_trig_solution_l1691_169167

noncomputable def solve_trig_system (x y : ℝ) : Prop :=
  (3 * Real.cos x + 4 * Real.sin x = -1.4) ∧ 
  (13 * Real.cos x - 41 * Real.cos y = -45) ∧ 
  (13 * Real.sin x + 41 * Real.sin y = 3)

theorem trig_solution :
  solve_trig_system (112.64 * Real.pi / 180) (347.32 * Real.pi / 180) ∧ 
  solve_trig_system (239.75 * Real.pi / 180) (20.31 * Real.pi / 180) :=
by {
    repeat { sorry }
  }

end NUMINAMATH_GPT_trig_solution_l1691_169167


namespace NUMINAMATH_GPT_king_paid_after_tip_l1691_169111

-- Define the cost of the crown and the tip percentage
def cost_of_crown : ℝ := 20000
def tip_percentage : ℝ := 0.10

-- Define the total amount paid after the tip
def total_amount_paid (C : ℝ) (tip_pct : ℝ) : ℝ :=
  C + (tip_pct * C)

-- Theorem statement: The total amount paid after the tip is $22,000
theorem king_paid_after_tip : total_amount_paid cost_of_crown tip_percentage = 22000 := by
  sorry

end NUMINAMATH_GPT_king_paid_after_tip_l1691_169111


namespace NUMINAMATH_GPT_total_sticks_of_gum_in_12_brown_boxes_l1691_169189

-- Definitions based on the conditions
def packs_per_carton := 7
def sticks_per_pack := 5
def cartons_in_full_box := 6
def cartons_in_partial_box := 3
def num_brown_boxes := 12
def num_partial_boxes := 2

-- Calculation definitions
def sticks_per_carton := packs_per_carton * sticks_per_pack
def sticks_per_full_box := cartons_in_full_box * sticks_per_carton
def sticks_per_partial_box := cartons_in_partial_box * sticks_per_carton
def num_full_boxes := num_brown_boxes - num_partial_boxes

-- Final total sticks of gum
def total_sticks_of_gum := (num_full_boxes * sticks_per_full_box) + (num_partial_boxes * sticks_per_partial_box)

-- The theorem to be proved
theorem total_sticks_of_gum_in_12_brown_boxes :
  total_sticks_of_gum = 2310 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_total_sticks_of_gum_in_12_brown_boxes_l1691_169189


namespace NUMINAMATH_GPT_f_not_monotonic_l1691_169181

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-(x:ℝ)) = -f x

def is_not_monotonic (f : ℝ → ℝ) : Prop :=
  ¬ ( (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x) )

variable (f : ℝ → ℝ)

axiom periodicity : ∀ x, f (x + 3/2) = -f x 
axiom odd_shifted : is_odd_function (λ x => f (x - 3/4))

theorem f_not_monotonic : is_not_monotonic f := by
  sorry

end NUMINAMATH_GPT_f_not_monotonic_l1691_169181


namespace NUMINAMATH_GPT_parabola_focus_l1691_169171

theorem parabola_focus (x y : ℝ) : (y^2 = -8 * x) → (x, y) = (-2, 0) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_l1691_169171


namespace NUMINAMATH_GPT_farmer_animals_l1691_169185

theorem farmer_animals : 
  ∃ g s : ℕ, 
    35 * g + 40 * s = 2000 ∧ 
    g = 2 * s ∧ 
    (0 < g ∧ 0 < s) ∧ 
    g = 36 ∧ s = 18 := 
by 
  sorry

end NUMINAMATH_GPT_farmer_animals_l1691_169185


namespace NUMINAMATH_GPT_domain_shift_l1691_169191

theorem domain_shift (f : ℝ → ℝ) :
  {x : ℝ | 1 ≤ x ∧ x ≤ 2} = {x | -2 ≤ x ∧ x ≤ -1} →
  {x : ℝ | ∃ y : ℝ, x = y - 1 ∧ 1 ≤ y ∧ y ≤ 2} =
  {x : ℝ | ∃ y : ℝ, x = y + 2 ∧ -2 ≤ y ∧ y ≤ -1} :=
by
  sorry

end NUMINAMATH_GPT_domain_shift_l1691_169191
