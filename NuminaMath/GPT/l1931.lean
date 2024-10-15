import Mathlib

namespace NUMINAMATH_GPT_convert_base_8_to_base_10_l1931_193102

def to_base_10 (n : ℕ) (b : ℕ) (digits : List ℕ) : ℕ :=
  digits.foldr (λ digit acc => acc * b + digit) 0

theorem convert_base_8_to_base_10 : 
  to_base_10 10 8 [6, 4, 2] = 166 := by
  sorry

end NUMINAMATH_GPT_convert_base_8_to_base_10_l1931_193102


namespace NUMINAMATH_GPT_calculate_possible_change_l1931_193100

structure ChangeProblem where
  (change : ℕ)
  (h1 : change < 100)
  (h2 : ∃ (q : ℕ), change = 25 * q + 10 ∧ q ≤ 3)
  (h3 : ∃ (d : ℕ), change = 10 * d + 20 ∧ d ≤ 9)

theorem calculate_possible_change (p1 p2 p3 p4 : ChangeProblem) :
  p1.change + p2.change + p3.change = 180 :=
by
  sorry

end NUMINAMATH_GPT_calculate_possible_change_l1931_193100


namespace NUMINAMATH_GPT_problem_l1931_193158

def P (x : ℝ) : Prop := x^2 - 2*x + 1 > 0

theorem problem (h : ¬ ∀ x : ℝ, P x) : ∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_l1931_193158


namespace NUMINAMATH_GPT_tan_45_eq_1_l1931_193183

theorem tan_45_eq_1 : Real.tan (Real.pi / 4) = 1 := by
  have h1 : Real.sin (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  have h2 : Real.cos (Real.pi / 4) = 1 / Real.sqrt 2 := sorry
  rw [Real.tan_eq_sin_div_cos, h1, h2]
  simp

end NUMINAMATH_GPT_tan_45_eq_1_l1931_193183


namespace NUMINAMATH_GPT_angle_C_is_120_l1931_193156

theorem angle_C_is_120 (C L U A : ℝ)
  (H1 : C = L)
  (H2 : L = U)
  (H3 : A = L)
  (H4 : A + L = 180)
  (H5 : 6 * C = 720) : C = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_C_is_120_l1931_193156


namespace NUMINAMATH_GPT_find_angle_A_find_side_a_l1931_193168

variable {A B C a b c : Real}
variable {area : Real}
variable (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
variable (h2 : b = 2)
variable (h3 : area = Real.sqrt 3)
variable (h4 : area = 1 / 2 * b * c * Real.sin A)

theorem find_angle_A (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A) : A = Real.pi / 3 :=
  sorry

theorem find_side_a (h4 : area = 1 / 2 * b * c * Real.sin A) (h2 : b = 2) (h3 : area = Real.sqrt 3) : a = 2 :=
  sorry

end NUMINAMATH_GPT_find_angle_A_find_side_a_l1931_193168


namespace NUMINAMATH_GPT_find_values_of_pqr_l1931_193127

def A (p : ℝ) := {x : ℝ | x^2 + p * x - 2 = 0}
def B (q r : ℝ) := {x : ℝ | x^2 + q * x + r = 0}
def A_union_B (p q r : ℝ) := A p ∪ B q r = {-2, 1, 5}
def A_intersect_B (p q r : ℝ) := A p ∩ B q r = {-2}

theorem find_values_of_pqr (p q r : ℝ) :
  A_union_B p q r → A_intersect_B p q r → p = -1 ∧ q = -3 ∧ r = -10 :=
by
  sorry

end NUMINAMATH_GPT_find_values_of_pqr_l1931_193127


namespace NUMINAMATH_GPT_base_conversion_b_eq_3_l1931_193134

theorem base_conversion_b_eq_3 (b : ℕ) (hb : b > 0) :
  (3 * 6^1 + 5 * 6^0 = 23) →
  (1 * b^2 + 3 * b + 2 = 23) →
  b = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_base_conversion_b_eq_3_l1931_193134


namespace NUMINAMATH_GPT_percentage_exceeds_l1931_193148

theorem percentage_exceeds (x y : ℝ) (h₁ : x < y) (h₂ : y = x + 0.35 * x) : ((y - x) / x) * 100 = 35 :=
by sorry

end NUMINAMATH_GPT_percentage_exceeds_l1931_193148


namespace NUMINAMATH_GPT_incorrect_solution_among_four_l1931_193144

theorem incorrect_solution_among_four 
  (x y : ℤ) 
  (h1 : 2 * x - 3 * y = 5) 
  (h2 : 3 * x - 2 * y = 7) : 
  ¬ ((2 * (2 * x - 3 * y) - ((-3) * (3 * x - 2 * y))) = (2 * 5 - (-3) * 7)) :=
sorry

end NUMINAMATH_GPT_incorrect_solution_among_four_l1931_193144


namespace NUMINAMATH_GPT_six_divisors_third_seven_times_second_fourth_ten_more_than_third_l1931_193141

theorem six_divisors_third_seven_times_second_fourth_ten_more_than_third (n : ℕ) :
  (∀ d : ℕ, d ∣ n ↔ d ∈ [1, d2, d3, d4, d5, n]) ∧ 
  (d3 = 7 * d2) ∧ 
  (d4 = d3 + 10) → 
  n = 2891 :=
by
  sorry

end NUMINAMATH_GPT_six_divisors_third_seven_times_second_fourth_ten_more_than_third_l1931_193141


namespace NUMINAMATH_GPT_explicit_expression_for_f_l1931_193122

variable (f : ℕ → ℕ)

-- Define the condition
axiom h : ∀ x : ℕ, f (x + 1) = 3 * x + 2

-- State the theorem
theorem explicit_expression_for_f (x : ℕ) : f x = 3 * x - 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_explicit_expression_for_f_l1931_193122


namespace NUMINAMATH_GPT_tree_planting_growth_rate_l1931_193176

theorem tree_planting_growth_rate {x : ℝ} :
  400 * (1 + x) ^ 2 = 625 :=
sorry

end NUMINAMATH_GPT_tree_planting_growth_rate_l1931_193176


namespace NUMINAMATH_GPT_coefficient_x3_y7_expansion_l1931_193118

theorem coefficient_x3_y7_expansion : 
  let n := 10
  let a := (2 : ℚ) / 3
  let b := -(3 : ℚ) / 5
  let k := 3
  let binom := Nat.choose n k
  let term := binom * (a ^ k) * (b ^ (n - k))
  term = -(256 : ℚ) / 257 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_coefficient_x3_y7_expansion_l1931_193118


namespace NUMINAMATH_GPT_smallest_odd_digit_n_l1931_193160

theorem smallest_odd_digit_n {n : ℕ} (h : n > 1) : 
  (∀ d ∈ (Nat.digits 10 (9997 * n)), d % 2 = 1) → n = 3335 :=
sorry

end NUMINAMATH_GPT_smallest_odd_digit_n_l1931_193160


namespace NUMINAMATH_GPT_min_rho_squared_l1931_193135

noncomputable def rho_squared (x t : ℝ) : ℝ :=
  (x - t)^2 + (x^2 - 4 * x + 7 + t)^2

theorem min_rho_squared : 
  ∃ (x t : ℝ), x = 3/2 ∧ t = -7/8 ∧ 
  ∀ (x' t' : ℝ), rho_squared x' t' ≥ rho_squared (3/2) (-7/8) :=
by
  sorry

end NUMINAMATH_GPT_min_rho_squared_l1931_193135


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1931_193188

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}
def complementA : Set ℕ := {2, 4}

theorem complement_of_A_in_U :
  (U \ A) = complementA :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1931_193188


namespace NUMINAMATH_GPT_find_seventh_number_l1931_193165

-- Let's denote the 10 numbers as A1, A2, A3, A4, A5, A6, A7, A8, A9, A10.
variables {A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ}

-- The average of all 10 numbers is 60.
def avg_10 (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10) / 10 = 60

-- The average of the first 6 numbers is 68.
def avg_first_6 (A1 A2 A3 A4 A5 A6 : ℝ) := (A1 + A2 + A3 + A4 + A5 + A6) / 6 = 68

-- The average of the last 6 numbers is 75.
def avg_last_6 (A5 A6 A7 A8 A9 A10 : ℝ) := (A5 + A6 + A7 + A8 + A9 + A10) / 6 = 75

-- Proving that the 7th number (A7) is 192.
theorem find_seventh_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 : ℝ) 
  (h1 : avg_10 A1 A2 A3 A4 A5 A6 A7 A8 A9 A10) 
  (h2 : avg_first_6 A1 A2 A3 A4 A5 A6) 
  (h3 : avg_last_6 A5 A6 A7 A8 A9 A10) :
  A7 = 192 :=
by
  sorry

end NUMINAMATH_GPT_find_seventh_number_l1931_193165


namespace NUMINAMATH_GPT_range_of_a_l1931_193164

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1931_193164


namespace NUMINAMATH_GPT_range_of_objective_function_l1931_193166

def objective_function (x y : ℝ) : ℝ := 3 * x - y

theorem range_of_objective_function (x y : ℝ) 
  (h1 : x + 2 * y ≥ 2)
  (h2 : 2 * x + y ≤ 4)
  (h3 : 4 * x - y ≥ -1)
  : - 3 / 2 ≤ objective_function x y ∧ objective_function x y ≤ 6 := 
sorry

end NUMINAMATH_GPT_range_of_objective_function_l1931_193166


namespace NUMINAMATH_GPT_vector_difference_perpendicular_l1931_193126

/-- Proof that the vector difference a - b is perpendicular to b given specific vectors a and b -/
theorem vector_difference_perpendicular {a b : ℝ × ℝ} (h_a : a = (2, 0)) (h_b : b = (1, 1)) :
  (a - b) • b = 0 :=
by
  sorry

end NUMINAMATH_GPT_vector_difference_perpendicular_l1931_193126


namespace NUMINAMATH_GPT_solution_set_for_inequality_l1931_193107

theorem solution_set_for_inequality :
  {x : ℝ | -x^2 + 2 * x + 3 ≥ 0} = {x : ℝ | -1 ≤ x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_GPT_solution_set_for_inequality_l1931_193107


namespace NUMINAMATH_GPT_find_white_towels_l1931_193170

variable (W : ℕ) -- Let W be the number of white towels Maria bought

def green_towels : ℕ := 40
def towels_given : ℕ := 65
def towels_left : ℕ := 19

theorem find_white_towels :
  green_towels + W - towels_given = towels_left →
  W = 44 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_white_towels_l1931_193170


namespace NUMINAMATH_GPT_compound_interest_years_is_four_l1931_193185
noncomputable def compoundInterestYears (P : ℝ) (r : ℝ) (n : ℕ) (CI : ℝ) : ℕ :=
  let A := P + CI
  let factor := (1 + r / n)
  let log_A_P := Real.log (A / P)
  let log_factor := Real.log factor
  Nat.floor (log_A_P / log_factor)

theorem compound_interest_years_is_four :
  compoundInterestYears 1200 0.20 1 1288.32 = 4 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_years_is_four_l1931_193185


namespace NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l1931_193113

theorem second_smallest_packs_of_hot_dogs 
  (n : ℕ) 
  (h1 : ∃ (k : ℕ), n = 2 * k + 2)
  (h2 : 12 * n ≡ 6 [MOD 8]) : 
  n = 4 :=
by
  sorry

end NUMINAMATH_GPT_second_smallest_packs_of_hot_dogs_l1931_193113


namespace NUMINAMATH_GPT_cost_difference_l1931_193197

-- Given conditions
def first_present_cost : ℕ := 18
def third_present_cost : ℕ := first_present_cost - 11
def total_cost : ℕ := 50

-- denoting costs of the second present via variable
def second_present_cost (x : ℕ) : Prop :=
  first_present_cost + x + third_present_cost = total_cost

-- Goal statement
theorem cost_difference (x : ℕ) (h : second_present_cost x) : x - first_present_cost = 7 :=
  sorry

end NUMINAMATH_GPT_cost_difference_l1931_193197


namespace NUMINAMATH_GPT_M_subset_N_l1931_193180

open Set

noncomputable def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
noncomputable def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := 
sorry

end NUMINAMATH_GPT_M_subset_N_l1931_193180


namespace NUMINAMATH_GPT_jake_steps_per_second_l1931_193119

/-
Conditions:
1. Austin and Jake start descending from the 9th floor at the same time.
2. The stairs have 30 steps across each floor.
3. The elevator takes 1 minute (60 seconds) to reach the ground floor.
4. Jake reaches the ground floor 30 seconds after Austin.
5. Jake descends 8 floors to reach the ground floor.
-/

def floors : ℕ := 8
def steps_per_floor : ℕ := 30
def time_elevator : ℕ := 60 -- in seconds
def additional_time_jake : ℕ := 30 -- in seconds

def total_time_jake := time_elevator + additional_time_jake -- in seconds
def total_steps := floors * steps_per_floor

def steps_per_second_jake := (total_steps : ℚ) / (total_time_jake : ℚ)

theorem jake_steps_per_second :
  steps_per_second_jake = 2.67 := by
  sorry

end NUMINAMATH_GPT_jake_steps_per_second_l1931_193119


namespace NUMINAMATH_GPT_max_value_f_on_interval_l1931_193181

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  (x^2 - 4) * (x - a)

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ :=
  3 * x^2 - 2 * a * x - 4

theorem max_value_f_on_interval :
  f' (-1) (1 / 2) = 0 →
  ∃ max_f, max_f = 42 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 4, f x (1 / 2) ≤ max_f :=
by
  sorry

end NUMINAMATH_GPT_max_value_f_on_interval_l1931_193181


namespace NUMINAMATH_GPT_sequence_general_term_l1931_193114

theorem sequence_general_term (a : ℕ → ℤ) (h₀ : a 0 = 1) (hstep : ∀ n, a (n + 1) = if a n = 1 then 0 else 1) :
  ∀ n, a n = (1 + (-1)^(n + 1)) / 2 :=
sorry

end NUMINAMATH_GPT_sequence_general_term_l1931_193114


namespace NUMINAMATH_GPT_annual_increase_of_chickens_l1931_193173

theorem annual_increase_of_chickens 
  (chickens_now : ℕ)
  (chickens_after_9_years : ℕ)
  (years : ℕ)
  (chickens_now_eq : chickens_now = 550)
  (chickens_after_9_years_eq : chickens_after_9_years = 1900)
  (years_eq : years = 9)
  : ((chickens_after_9_years - chickens_now) / years) = 150 :=
by
  sorry

end NUMINAMATH_GPT_annual_increase_of_chickens_l1931_193173


namespace NUMINAMATH_GPT_B_subset_A_l1931_193124

def A (x : ℝ) : Prop := abs (2 * x - 3) > 1
def B (x : ℝ) : Prop := x^2 + x - 6 > 0

theorem B_subset_A : ∀ x, B x → A x := sorry

end NUMINAMATH_GPT_B_subset_A_l1931_193124


namespace NUMINAMATH_GPT_f_of_2_l1931_193172

-- Definition of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Given conditions
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_value : f (-2) = 11)

-- The theorem we want to prove
theorem f_of_2 : f 2 = -11 :=
by 
  sorry

end NUMINAMATH_GPT_f_of_2_l1931_193172


namespace NUMINAMATH_GPT_symmetric_graph_increasing_interval_l1931_193132

noncomputable def f : ℝ → ℝ := sorry

theorem symmetric_graph_increasing_interval :
  (∀ x : ℝ, f (-x) = -f x) → -- f is odd
  (∀ x y : ℝ, 3 ≤ x → x < y → y ≤ 7 → f x < f y) → -- f is increasing in [3,7]
  (∀ x : ℝ, 3 ≤ x → x ≤ 7 → f x ≤ 5) → -- f has a maximum value of 5 in [3,7]
  (∀ x y : ℝ, -7 ≤ x → x < y → y ≤ -3 → f x < f y) ∧ -- f is increasing in [-7,-3]
  (∀ x : ℝ, -7 ≤ x → x ≤ -3 → f x ≥ -5) -- f has a minimum value of -5 in [-7,-3]
:= sorry

end NUMINAMATH_GPT_symmetric_graph_increasing_interval_l1931_193132


namespace NUMINAMATH_GPT_real_set_x_eq_l1931_193121

theorem real_set_x_eq :
  {x : ℝ | ⌊x * ⌊x⌋⌋ = 45} = {x : ℝ | 7.5 ≤ x ∧ x < 7.6667} :=
by
  -- The proof would be provided here, but we're skipping it with sorry
  sorry

end NUMINAMATH_GPT_real_set_x_eq_l1931_193121


namespace NUMINAMATH_GPT_kai_marbles_over_200_l1931_193149

theorem kai_marbles_over_200 (marbles_on_day : Nat → Nat)
  (h_initial : marbles_on_day 0 = 4)
  (h_growth : ∀ n, marbles_on_day (n + 1) = 3 * marbles_on_day n) :
  ∃ k, marbles_on_day k > 200 ∧ k = 4 := by
  sorry

end NUMINAMATH_GPT_kai_marbles_over_200_l1931_193149


namespace NUMINAMATH_GPT_david_marks_physics_l1931_193116

def marks_english := 96
def marks_math := 95
def marks_chemistry := 97
def marks_biology := 95
def average_marks := 93
def number_of_subjects := 5

theorem david_marks_physics : 
  let total_marks := average_marks * number_of_subjects 
  let total_known_marks := marks_english + marks_math + marks_chemistry + marks_biology
  let marks_physics := total_marks - total_known_marks
  marks_physics = 82 :=
by
  sorry

end NUMINAMATH_GPT_david_marks_physics_l1931_193116


namespace NUMINAMATH_GPT_cost_price_correct_l1931_193152

open Real

-- Define the cost price of the table
def cost_price (C : ℝ) : ℝ := C

-- Define the marked price
def marked_price (C : ℝ) : ℝ := 1.30 * C

-- Define the discounted price
def discounted_price (C : ℝ) : ℝ := 0.85 * (marked_price C)

-- Define the final price after sales tax
def final_price (C : ℝ) : ℝ := 1.12 * (discounted_price C)

-- Given that the final price is 9522.84
axiom final_price_value : final_price 9522.84 = 1.2376 * 7695

-- Main theorem stating the problem to prove
theorem cost_price_correct (C : ℝ) : final_price C = 9522.84 -> C = 7695 := by
  sorry

end NUMINAMATH_GPT_cost_price_correct_l1931_193152


namespace NUMINAMATH_GPT_angle_between_hands_at_3_27_l1931_193103

noncomputable def minute_hand_angle (m : ℕ) : ℝ :=
  (m / 60.0) * 360.0

noncomputable def hour_hand_angle (h : ℕ) (m : ℕ) : ℝ :=
  ((h + m / 60.0) / 12.0) * 360.0

theorem angle_between_hands_at_3_27 : 
  minute_hand_angle 27 - hour_hand_angle 3 27 = 58.5 :=
by
  rw [minute_hand_angle, hour_hand_angle]
  simp
  sorry

end NUMINAMATH_GPT_angle_between_hands_at_3_27_l1931_193103


namespace NUMINAMATH_GPT_last_digit_product_3_2001_7_2002_13_2003_l1931_193130

def last_digit (n : ℕ) : ℕ := n % 10

theorem last_digit_product_3_2001_7_2002_13_2003 :
  last_digit (3^2001 * 7^2002 * 13^2003) = 9 :=
by
  sorry

end NUMINAMATH_GPT_last_digit_product_3_2001_7_2002_13_2003_l1931_193130


namespace NUMINAMATH_GPT_no_distinct_triple_exists_for_any_quadratic_trinomial_l1931_193174

theorem no_distinct_triple_exists_for_any_quadratic_trinomial (f : ℝ → ℝ) 
    (hf : ∃ a b c : ℝ, ∀ x, f x = a*x^2 + b*x + c) :
    ¬ ∃ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ f a = b ∧ f b = c ∧ f c = a := 
by 
  sorry

end NUMINAMATH_GPT_no_distinct_triple_exists_for_any_quadratic_trinomial_l1931_193174


namespace NUMINAMATH_GPT_goal_amount_is_correct_l1931_193192

def earnings_three_families : ℕ := 3 * 10
def earnings_fifteen_families : ℕ := 15 * 5
def total_earned : ℕ := earnings_three_families + earnings_fifteen_families
def goal_amount : ℕ := total_earned + 45

theorem goal_amount_is_correct : goal_amount = 150 :=
by
  -- We are aware of the proof steps but they are not required here
  sorry

end NUMINAMATH_GPT_goal_amount_is_correct_l1931_193192


namespace NUMINAMATH_GPT_olivia_total_earnings_l1931_193186

variable (rate : ℕ) (hours_monday : ℕ) (hours_wednesday : ℕ) (hours_friday : ℕ)

def olivia_earnings : ℕ := hours_monday * rate + hours_wednesday * rate + hours_friday * rate

theorem olivia_total_earnings :
  rate = 9 → hours_monday = 4 → hours_wednesday = 3 → hours_friday = 6 → olivia_earnings rate hours_monday hours_wednesday hours_friday = 117 :=
by
  sorry

end NUMINAMATH_GPT_olivia_total_earnings_l1931_193186


namespace NUMINAMATH_GPT_find_y_l1931_193147

theorem find_y : 
  (6 + 10 + 14 + 22) / 4 = (15 + y) / 2 → y = 11 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_y_l1931_193147


namespace NUMINAMATH_GPT_calculate_binary_expr_l1931_193138

theorem calculate_binary_expr :
  let a := 0b11001010
  let b := 0b11010
  let c := 0b100
  (a * b) / c = 0b1001110100 := by
sorry

end NUMINAMATH_GPT_calculate_binary_expr_l1931_193138


namespace NUMINAMATH_GPT_calc_molecular_weight_l1931_193182

/-- Atomic weights in g/mol -/
def atomic_weight (e : String) : Float :=
  match e with
  | "Ca"   => 40.08
  | "O"    => 16.00
  | "H"    => 1.01
  | "Al"   => 26.98
  | "S"    => 32.07
  | "K"    => 39.10
  | "N"    => 14.01
  | _      => 0.0

/-- Molecular weight calculation for specific compounds -/
def molecular_weight (compound : String) : Float :=
  match compound with
  | "Ca(OH)2"     => atomic_weight "Ca" + 2 * atomic_weight "O" + 2 * atomic_weight "H"
  | "Al2(SO4)3"   => 2 * atomic_weight "Al" + 3 * (atomic_weight "S" + 4 * atomic_weight "O")
  | "KNO3"        => atomic_weight "K" + atomic_weight "N" + 3 * atomic_weight "O"
  | _             => 0.0

/-- Given moles of different compounds, calculate the total molecular weight -/
def total_molecular_weight (moles : List (String × Float)) : Float :=
  moles.foldl (fun acc (compound, n) => acc + n * molecular_weight compound) 0.0

/-- The given problem -/
theorem calc_molecular_weight :
  total_molecular_weight [("Ca(OH)2", 4), ("Al2(SO4)3", 2), ("KNO3", 3)] = 1284.07 :=
by
  sorry

end NUMINAMATH_GPT_calc_molecular_weight_l1931_193182


namespace NUMINAMATH_GPT_chef_earns_less_than_manager_l1931_193169

noncomputable def manager_wage : ℝ := 8.50
noncomputable def dishwasher_wage : ℝ := manager_wage / 2
noncomputable def chef_wage : ℝ := dishwasher_wage * 1.22

theorem chef_earns_less_than_manager :
  manager_wage - chef_wage = 3.315 :=
by
  sorry

end NUMINAMATH_GPT_chef_earns_less_than_manager_l1931_193169


namespace NUMINAMATH_GPT_problem_1_problem_2_l1931_193153

-- Problem 1
theorem problem_1 :
  -((1 / 2) / 3) * (3 - (-3)^2) = 1 :=
by
  sorry

-- Problem 2
theorem problem_2 {x : ℝ} (h1 : x ≠ 2) (h2 : x ≠ -2) :
  (2 * x) / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1931_193153


namespace NUMINAMATH_GPT_factor_x4_minus_81_l1931_193196

theorem factor_x4_minus_81 (x : ℝ) : 
  x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
sorry

end NUMINAMATH_GPT_factor_x4_minus_81_l1931_193196


namespace NUMINAMATH_GPT_triangle_cosine_sine_inequality_l1931_193139

theorem triangle_cosine_sine_inequality (A B C : ℝ) (h : A + B + C = Real.pi) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hA_lt_pi : A < Real.pi)
  (hB_lt_pi : B < Real.pi)
  (hC_lt_pi : C < Real.pi) :
  Real.cos A * (Real.sin B + Real.sin C) ≥ -2 * Real.sqrt 6 / 9 := 
by
  sorry

end NUMINAMATH_GPT_triangle_cosine_sine_inequality_l1931_193139


namespace NUMINAMATH_GPT_smallest_n_inverse_mod_1176_l1931_193104

theorem smallest_n_inverse_mod_1176 : ∃ n : ℕ, n > 1 ∧ Nat.Coprime n 1176 ∧ (∀ m : ℕ, m > 1 ∧ Nat.Coprime m 1176 → n ≤ m) ∧ n = 5 := by
  sorry

end NUMINAMATH_GPT_smallest_n_inverse_mod_1176_l1931_193104


namespace NUMINAMATH_GPT_hyperbola_symmetric_asymptotes_l1931_193151

noncomputable def M : ℝ := 225 / 16

theorem hyperbola_symmetric_asymptotes (M_val : ℝ) :
  (∀ x y : ℝ, (x^2 / 9 - y^2 / 16 = 1 → y = x * (4 / 3) ∨ y = -x * (4 / 3))
  ∧ (y^2 / 25 - x^2 / M_val = 1 → y = x * (5 / Real.sqrt M_val) ∨ y = -x * (5 / Real.sqrt M_val)))
  → M_val = M := by
  sorry

end NUMINAMATH_GPT_hyperbola_symmetric_asymptotes_l1931_193151


namespace NUMINAMATH_GPT_solve_for_x_l1931_193129

theorem solve_for_x (x : ℚ) : (x = 70 / (8 - 3 / 4)) → (x = 280 / 29) :=
by
  intro h
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_solve_for_x_l1931_193129


namespace NUMINAMATH_GPT_compute_100a_b_l1931_193108

theorem compute_100a_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, (x + a) * (x + b) * (x + 10) = 0 ↔ x = -a ∨ x = -b ∨ x = -10)
  (h2 : a ≠ -4 ∧ b ≠ -4 ∧ 10 ≠ -4)
  (h3 : ∀ x : ℝ, (x + 2 * a) * (x + 5) * (x + 8) = 0 ↔ x = -5)
  (hb : b = 8)
  (ha : 2 * a = 5) :
  100 * a + b = 258 := 
sorry

end NUMINAMATH_GPT_compute_100a_b_l1931_193108


namespace NUMINAMATH_GPT_selling_price_l1931_193154

-- Definitions
def price_coffee_A : ℝ := 10
def price_coffee_B : ℝ := 12
def weight_coffee_A : ℝ := 240
def weight_coffee_B : ℝ := 240
def total_weight : ℝ := 480
def total_cost : ℝ := (weight_coffee_A * price_coffee_A) + (weight_coffee_B * price_coffee_B)

-- Theorem
theorem selling_price (h_total_weight : total_weight = weight_coffee_A + weight_coffee_B) :
  total_cost / total_weight = 11 :=
by
  sorry

end NUMINAMATH_GPT_selling_price_l1931_193154


namespace NUMINAMATH_GPT_product_of_consecutive_integers_plus_one_l1931_193112

theorem product_of_consecutive_integers_plus_one (n : ℤ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n * (n + 3) + 1) ^ 2 := 
sorry

end NUMINAMATH_GPT_product_of_consecutive_integers_plus_one_l1931_193112


namespace NUMINAMATH_GPT_Krishan_has_4046_l1931_193191

variable (Ram Gopal Krishan : ℕ) -- Define the variables

-- Conditions given in the problem
axiom ratio_Ram_Gopal : Ram * 17 = Gopal * 7
axiom ratio_Gopal_Krishan : Gopal * 17 = Krishan * 7
axiom Ram_value : Ram = 686

-- This is the goal to prove
theorem Krishan_has_4046 : Krishan = 4046 :=
by
  -- Here is where the proof would go
  sorry

end NUMINAMATH_GPT_Krishan_has_4046_l1931_193191


namespace NUMINAMATH_GPT_math_proof_problem_l1931_193115

noncomputable def problemStatement : Prop :=
  ∃ (α : ℝ), 5 * Real.sin (2 * α) = Real.sin (2 * Real.pi / 180) ∧ 
  (Real.tan (α + Real.pi / 180) / Real.tan (α - Real.pi / 180) = -3/2)

theorem math_proof_problem : problemStatement := 
by 
  sorry

end NUMINAMATH_GPT_math_proof_problem_l1931_193115


namespace NUMINAMATH_GPT_roots_polynomial_sum_products_l1931_193110

theorem roots_polynomial_sum_products (p q r : ℂ)
  (h : 6 * p^3 - 5 * p^2 + 13 * p - 10 = 0)
  (h' : 6 * q^3 - 5 * q^2 + 13 * q - 10 = 0)
  (h'' : 6 * r^3 - 5 * r^2 + 13 * r - 10 = 0)
  (hpq : p ≠ q) (hqr : q ≠ r) (hrp : r ≠ p) :
  p * q + q * r + r * p = 13 / 6 := 
sorry

end NUMINAMATH_GPT_roots_polynomial_sum_products_l1931_193110


namespace NUMINAMATH_GPT_max_angle_MPN_is_pi_over_2_l1931_193123

open Real

noncomputable def max_angle_MPN (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : ℝ :=
  sorry

theorem max_angle_MPN_is_pi_over_2 (θ : ℝ) (P : ℝ × ℝ) (hP : (P.1 - cos θ)^2 + (P.2 - sin θ)^2 = 1/25) : 
  max_angle_MPN θ P hP = π / 2 :=
sorry

end NUMINAMATH_GPT_max_angle_MPN_is_pi_over_2_l1931_193123


namespace NUMINAMATH_GPT_geom_seq_val_l1931_193145

noncomputable def is_geom_seq (a : ℕ → ℝ) : Prop :=
∃ q b, ∀ n, a n = b * q^n

variables (a : ℕ → ℝ)

axiom a_5_a_7 : a 5 * a 7 = 2
axiom a_2_plus_a_10 : a 2 + a 10 = 3

theorem geom_seq_val (a_geom : is_geom_seq a) :
  (a 12) / (a 4) = 2 ∨ (a 12) / (a 4) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_geom_seq_val_l1931_193145


namespace NUMINAMATH_GPT_ratio_of_hardback_books_is_two_to_one_l1931_193167

noncomputable def ratio_of_hardback_books : ℕ :=
  let sarah_paperbacks := 6
  let sarah_hardbacks := 4
  let brother_paperbacks := sarah_paperbacks / 3
  let total_books_brother := 10
  let brother_hardbacks := total_books_brother - brother_paperbacks
  brother_hardbacks / sarah_hardbacks

theorem ratio_of_hardback_books_is_two_to_one : 
  ratio_of_hardback_books = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_hardback_books_is_two_to_one_l1931_193167


namespace NUMINAMATH_GPT_parallel_or_identical_lines_l1931_193163

theorem parallel_or_identical_lines (a b c d e f : ℝ) :
  2 * b - 3 * a = 15 → 4 * d - 6 * c = 18 → (b ≠ d → a = c) :=
by
  intros h1 h2 hneq
  sorry

end NUMINAMATH_GPT_parallel_or_identical_lines_l1931_193163


namespace NUMINAMATH_GPT_system_solution_fraction_l1931_193157

theorem system_solution_fraction (x y z : ℝ) (h1 : x + (-95/9) * y + 4 * z = 0)
  (h2 : 4 * x + (-95/9) * y - 3 * z = 0) (h3 : 3 * x + 5 * y - 4 * z = 0) (hx_ne_zero : x ≠ 0) 
  (hy_ne_zero : y ≠ 0) (hz_ne_zero : z ≠ 0) : 
  (x * z) / (y ^ 2) = 20 :=
sorry

end NUMINAMATH_GPT_system_solution_fraction_l1931_193157


namespace NUMINAMATH_GPT_number_of_pickers_is_221_l1931_193146
-- Import necessary Lean and math libraries

/--
Given the conditions:
1. The number of pickers fills 100 drums of raspberries per day.
2. The number of pickers fills 221 drums of grapes per day.
3. In 77 days, the pickers would fill 17017 drums of grapes.
Prove that the number of pickers is 221.
-/
theorem number_of_pickers_is_221
  (P : ℕ)
  (d1 : P * 100 = 100 * P)
  (d2 : P * 221 = 221 * P)
  (d17 : P * 221 * 77 = 17017) : 
  P = 221 := 
sorry

end NUMINAMATH_GPT_number_of_pickers_is_221_l1931_193146


namespace NUMINAMATH_GPT_votes_ratio_l1931_193199

theorem votes_ratio (V : ℝ) 
  (counted_fraction : ℝ := 2/9) 
  (favor_fraction : ℝ := 3/4) 
  (against_fraction_remaining : ℝ := 0.7857142857142856) :
  let counted := counted_fraction * V
  let favor_counted := favor_fraction * counted
  let remaining := V - counted
  let against_remaining := against_fraction_remaining * remaining
  let against_counted := (1 - favor_fraction) * counted
  let total_against := against_counted + against_remaining
  let total_favor := favor_counted
  (total_against / total_favor) = 4 :=
by
  sorry

end NUMINAMATH_GPT_votes_ratio_l1931_193199


namespace NUMINAMATH_GPT_sqrt_sum_abs_eq_l1931_193177

theorem sqrt_sum_abs_eq (x : ℝ) :
    (Real.sqrt (x^2 + 6 * x + 9) + Real.sqrt (x^2 - 6 * x + 9)) = (|x - 3| + |x + 3|) := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_abs_eq_l1931_193177


namespace NUMINAMATH_GPT_calculate_expression_l1931_193194

theorem calculate_expression : 
  - 3 ^ 2 + (-12) * abs (-1/2) - 6 / (-1) = -9 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1931_193194


namespace NUMINAMATH_GPT_round_robin_tournament_l1931_193159

theorem round_robin_tournament (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end NUMINAMATH_GPT_round_robin_tournament_l1931_193159


namespace NUMINAMATH_GPT_smallest_k_l1931_193195

theorem smallest_k (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  ∃ k : ℕ, k = 2 * max m n + min m n - 1 ∧ 
  (∀ (persons : Finset ℕ),
    persons.card ≥ k →
    (∃ (acquainted : Finset (ℕ × ℕ)), acquainted.card = m ∧ 
      (∀ (x y : ℕ), (x, y) ∈ acquainted → (x ∈ persons ∧ y ∈ persons))) ∨
    (∃ (unacquainted : Finset (ℕ × ℕ)), unacquainted.card = n ∧ 
      (∀ (x y : ℕ), (x, y) ∈ unacquainted → (x ∈ persons ∧ y ∈ persons ∧ x ≠ y)))) :=
sorry

end NUMINAMATH_GPT_smallest_k_l1931_193195


namespace NUMINAMATH_GPT_solve_for_x_l1931_193179

theorem solve_for_x (x : ℝ) : 5 + 3.4 * x = 2.1 * x - 30 → x = -26.923 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_x_l1931_193179


namespace NUMINAMATH_GPT_number_of_sevens_l1931_193133

theorem number_of_sevens (n : ℕ) : ∃ (k : ℕ), k < n ∧ ∃ (f : ℕ → ℕ), (∀ i, f i = 7) ∧ (7 * ((77 - 7) / 7) ^ 14 - 1) / (7 + (7 + 7)/7) = 7^(f k) :=
by sorry

end NUMINAMATH_GPT_number_of_sevens_l1931_193133


namespace NUMINAMATH_GPT_complement_intersection_l1931_193106

def set_P : Set ℝ := {x | x^2 - 2 * x ≥ 0}
def set_Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

theorem complement_intersection (P Q : Set ℝ) (hP : P = set_P) (hQ : Q = set_Q) :
  (Pᶜ ∩ Q) = {x | 1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1931_193106


namespace NUMINAMATH_GPT_eval_expression_l1931_193155

theorem eval_expression {p q r s : ℝ} 
  (h : p / (30 - p) + q / (70 - q) + r / (50 - r) + s / (40 - s) = 9) :
  6 / (30 - p) + 14 / (70 - q) + 10 / (50 - r) + 8 / (40 - s) = 7.6 := 
by 
  sorry

end NUMINAMATH_GPT_eval_expression_l1931_193155


namespace NUMINAMATH_GPT_matrix_solution_l1931_193150

-- Define the 2x2 matrix N
def N : Matrix (Fin 2) (Fin 2) ℚ := 
  ![ ![30 / 7, -13 / 7], 
     ![-6 / 7, -10 / 7] ]

-- Define the vectors
def vec1 : Fin 2 → ℚ := ![2, 3]
def vec2 : Fin 2 → ℚ := ![4, -1]

-- Expected results
def result1 : Fin 2 → ℚ := ![3, -6]
def result2 : Fin 2 → ℚ := ![19, -2]

-- The proof statement
theorem matrix_solution : (N.mulVec vec1 = result1) ∧ (N.mulVec vec2 = result2) :=
  by sorry

end NUMINAMATH_GPT_matrix_solution_l1931_193150


namespace NUMINAMATH_GPT_pints_in_two_liters_nearest_tenth_l1931_193193

def liters_to_pints (liters : ℝ) : ℝ :=
  2.1 * liters

theorem pints_in_two_liters_nearest_tenth :
  liters_to_pints 2 = 4.2 :=
by
  sorry

end NUMINAMATH_GPT_pints_in_two_liters_nearest_tenth_l1931_193193


namespace NUMINAMATH_GPT_correct_proposition_l1931_193125

-- Definitions based on conditions
def p : Prop := ∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0
def not_p : Prop := ∀ x : ℝ, x^2 + 2 * x + 2016 > 0

-- Proof statement
theorem correct_proposition : p → not_p :=
by sorry

end NUMINAMATH_GPT_correct_proposition_l1931_193125


namespace NUMINAMATH_GPT_houses_per_block_correct_l1931_193120

-- Define the conditions
def total_mail_per_block : ℕ := 32
def mail_per_house : ℕ := 8

-- Define the correct answer
def houses_per_block : ℕ := 4

-- Theorem statement
theorem houses_per_block_correct (total_mail_per_block mail_per_house : ℕ) : 
  total_mail_per_block = 32 →
  mail_per_house = 8 →
  total_mail_per_block / mail_per_house = houses_per_block :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_houses_per_block_correct_l1931_193120


namespace NUMINAMATH_GPT_linda_original_savings_l1931_193161

theorem linda_original_savings :
  ∃ S : ℝ, 
    (5 / 8) * S + (1 / 4) * S = 400 ∧
    (1 / 8) * S = 600 ∧
    S = 4800 :=
by
  sorry

end NUMINAMATH_GPT_linda_original_savings_l1931_193161


namespace NUMINAMATH_GPT_sunflower_seeds_contest_l1931_193184

theorem sunflower_seeds_contest 
  (first_player_seeds : ℕ) (second_player_seeds : ℕ) (total_seeds : ℕ) 
  (third_player_seeds : ℕ) (third_more : ℕ) 
  (h1 : first_player_seeds = 78) 
  (h2 : second_player_seeds = 53) 
  (h3 : total_seeds = 214) 
  (h4 : first_player_seeds + second_player_seeds + third_player_seeds = total_seeds) 
  (h5 : third_more = third_player_seeds - second_player_seeds) : 
  third_more = 30 :=
by
  sorry

end NUMINAMATH_GPT_sunflower_seeds_contest_l1931_193184


namespace NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1931_193198

theorem arithmetic_sequence_15th_term : 
  let a₁ := 3
  let d := 4
  let n := 15
  a₁ + (n - 1) * d = 59 :=
by
  let a₁ := 3
  let d := 4
  let n := 15
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_15th_term_l1931_193198


namespace NUMINAMATH_GPT_total_selling_price_is_correct_l1931_193171

-- Define the given constants
def meters_of_cloth : ℕ := 85
def profit_per_meter : ℕ := 10
def cost_price_per_meter : ℕ := 95

-- Compute the selling price per meter
def selling_price_per_meter : ℕ := cost_price_per_meter + profit_per_meter

-- Calculate the total selling price
def total_selling_price : ℕ := selling_price_per_meter * meters_of_cloth

-- The theorem statement
theorem total_selling_price_is_correct : total_selling_price = 8925 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_is_correct_l1931_193171


namespace NUMINAMATH_GPT_count_valid_ways_l1931_193189

theorem count_valid_ways (n : ℕ) (h1 : n = 6) : 
  ∀ (library : ℕ), (1 ≤ library) → (library ≤ 5) → ∃ (checked_out : ℕ), 
  (checked_out = n - library) := 
sorry

end NUMINAMATH_GPT_count_valid_ways_l1931_193189


namespace NUMINAMATH_GPT_man_age_twice_son_age_l1931_193140

-- Definitions based on conditions
def son_age : ℕ := 20
def man_age : ℕ := son_age + 22

-- Definition of the main statement to be proven
theorem man_age_twice_son_age (Y : ℕ) : man_age + Y = 2 * (son_age + Y) → Y = 2 :=
by sorry

end NUMINAMATH_GPT_man_age_twice_son_age_l1931_193140


namespace NUMINAMATH_GPT_x_pow_y_equals_nine_l1931_193187

theorem x_pow_y_equals_nine (x y : ℝ) (h : (|x + 3| * (y - 2)^2 < 0)) : x^y = 9 :=
sorry

end NUMINAMATH_GPT_x_pow_y_equals_nine_l1931_193187


namespace NUMINAMATH_GPT_pills_per_day_l1931_193111

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end NUMINAMATH_GPT_pills_per_day_l1931_193111


namespace NUMINAMATH_GPT_Alan_eggs_count_l1931_193175

theorem Alan_eggs_count (Price_per_egg Chickens_bought Price_per_chicken Total_spent : ℕ)
  (h1 : Price_per_egg = 2) (h2 : Chickens_bought = 6) (h3 : Price_per_chicken = 8) (h4 : Total_spent = 88) :
  ∃ E : ℕ, 2 * E + Chickens_bought * Price_per_chicken = Total_spent ∧ E = 20 :=
by
  sorry

end NUMINAMATH_GPT_Alan_eggs_count_l1931_193175


namespace NUMINAMATH_GPT_evaluate_f_at_3_l1931_193178

def f (x : ℤ) : ℤ := 5 * x^3 + 3 * x^2 + 7 * x - 2

theorem evaluate_f_at_3 : f 3 = 181 := by
  sorry

end NUMINAMATH_GPT_evaluate_f_at_3_l1931_193178


namespace NUMINAMATH_GPT_trigonometric_identity_l1931_193128

variable (α : Real)

theorem trigonometric_identity :
  (Real.tan (α - Real.pi / 4) = 1 / 2) →
  ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l1931_193128


namespace NUMINAMATH_GPT_jury_concludes_you_are_not_guilty_l1931_193117

def criminal_is_a_liar : Prop := sorry -- The criminal is a liar, known.
def you_are_a_liar : Prop := sorry -- You are a liar, unknown.
def you_are_not_guilty : Prop := sorry -- You are not guilty.

theorem jury_concludes_you_are_not_guilty :
  criminal_is_a_liar → you_are_a_liar → you_are_not_guilty → "I am guilty" = "You are not guilty" :=
by
  -- Proof construct omitted as per problem requirements
  sorry

end NUMINAMATH_GPT_jury_concludes_you_are_not_guilty_l1931_193117


namespace NUMINAMATH_GPT_amount_C_l1931_193131

-- Define the variables and conditions.
variables (A B C : ℝ)
axiom h1 : A = (2 / 3) * B
axiom h2 : B = (1 / 4) * C
axiom h3 : A + B + C = 544

-- State the theorem.
theorem amount_C (A B C : ℝ) (h1 : A = (2 / 3) * B) (h2 : B = (1 / 4) * C) (h3 : A + B + C = 544) : C = 384 := 
sorry

end NUMINAMATH_GPT_amount_C_l1931_193131


namespace NUMINAMATH_GPT_juliette_and_marco_money_comparison_l1931_193190

noncomputable def euro_to_dollar (eur : ℝ) : ℝ := eur * 1.5

theorem juliette_and_marco_money_comparison :
  (600 - euro_to_dollar 350) / 600 * 100 = 12.5 := by
sorry

end NUMINAMATH_GPT_juliette_and_marco_money_comparison_l1931_193190


namespace NUMINAMATH_GPT_evalExpression_at_3_2_l1931_193101

def evalExpression (x y : ℕ) : ℕ := 3 * x^y + 4 * y^x

theorem evalExpression_at_3_2 : evalExpression 3 2 = 59 := by
  sorry

end NUMINAMATH_GPT_evalExpression_at_3_2_l1931_193101


namespace NUMINAMATH_GPT_min_digits_fraction_l1931_193109

theorem min_digits_fraction : 
  let num := 987654321
  let denom := 2^27 * 5^3
  ∃ (digits : ℕ), (10^digits * num = 987654321 * 2^27 * 5^3) ∧ digits = 27 := 
by
  sorry

end NUMINAMATH_GPT_min_digits_fraction_l1931_193109


namespace NUMINAMATH_GPT_temperature_on_tuesday_l1931_193136

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th) / 3 = 45 →
  (W + Th + F) / 3 = 50 →
  F = 53 →
  T = 38 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l1931_193136


namespace NUMINAMATH_GPT_brick_width_l1931_193162

-- Define the dimensions of the wall
def L_wall : Real := 750 -- length in cm
def W_wall : Real := 600 -- width in cm
def H_wall : Real := 22.5 -- height in cm

-- Define the dimensions of the bricks
def L_brick : Real := 25 -- length in cm
def H_brick : Real := 6 -- height in cm

-- Define the number of bricks needed
def n_bricks : Nat := 6000

-- Define the total volume of the wall
def V_wall : Real := L_wall * W_wall * H_wall

-- Define the volume of one brick
def V_brick (W : Real) : Real := L_brick * W * H_brick

-- Statement to prove
theorem brick_width : 
  ∃ W : Real, V_wall = V_brick W * (n_bricks : Real) ∧ W = 11.25 := by 
  sorry

end NUMINAMATH_GPT_brick_width_l1931_193162


namespace NUMINAMATH_GPT_correct_addition_result_l1931_193143

theorem correct_addition_result (x : ℚ) (h : x - 13/5 = 9/7) : x + 13/5 = 227/35 := 
by sorry

end NUMINAMATH_GPT_correct_addition_result_l1931_193143


namespace NUMINAMATH_GPT_steps_probability_to_point_3_3_l1931_193142

theorem steps_probability_to_point_3_3 : 
  let a := 35
  let b := 4096
  a + b = 4131 :=
by {
  sorry
}

end NUMINAMATH_GPT_steps_probability_to_point_3_3_l1931_193142


namespace NUMINAMATH_GPT_smallest_diameter_of_tablecloth_l1931_193137

theorem smallest_diameter_of_tablecloth (a : ℝ) (h : a = 1) : ∃ d : ℝ, d = Real.sqrt 2 ∧ (∀ (x : ℝ), x < d → ¬(∀ (y : ℝ), (y^2 + y^2 = x^2) → y ≤ a)) :=
by 
  sorry

end NUMINAMATH_GPT_smallest_diameter_of_tablecloth_l1931_193137


namespace NUMINAMATH_GPT_calculate_expression_l1931_193105

theorem calculate_expression : |(-5 : ℤ)| + (1 / 3 : ℝ)⁻¹ - (Real.pi - 2) ^ 0 = 7 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1931_193105
