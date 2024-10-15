import Mathlib

namespace NUMINAMATH_GPT_brown_gumdrops_after_replacement_l2070_207060

-- Definitions based on the given conditions.
def total_gumdrops (green_gumdrops : ℕ) : ℕ :=
  (green_gumdrops * 100) / 15

def blue_gumdrops (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 25 / 100

def brown_gumdrops_initial (total_gumdrops : ℕ) : ℕ :=
  total_gumdrops * 15 / 100

def brown_gumdrops_final (brown_initial : ℕ) (blue_gumdrops : ℕ) : ℕ :=
  brown_initial + blue_gumdrops / 3

-- The main theorem statement based on the proof problem.
theorem brown_gumdrops_after_replacement
  (green_gumdrops : ℕ)
  (h_green : green_gumdrops = 36)
  : brown_gumdrops_final (brown_gumdrops_initial (total_gumdrops green_gumdrops)) 
                         (blue_gumdrops (total_gumdrops green_gumdrops))
    = 56 := 
  by sorry

end NUMINAMATH_GPT_brown_gumdrops_after_replacement_l2070_207060


namespace NUMINAMATH_GPT_find_c_l2070_207095

theorem find_c (a b c : ℤ) (h1 : c ≥ 0) (h2 : ¬∃ m : ℤ, 2 * a * b = m^2)
  (h3 : ∀ n : ℕ, n > 0 → (a^n + (2 : ℤ)^n) ∣ (b^n + c)) :
  c = 0 ∨ c = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_c_l2070_207095


namespace NUMINAMATH_GPT_algorithm_must_have_sequential_structure_l2070_207025

-- Definitions for types of structures used in algorithm definitions.
inductive Structure
| Logical
| Selection
| Loop
| Sequential

-- Predicate indicating whether a given Structure is necessary for any algorithm.
def necessary (s : Structure) : Prop :=
  match s with
  | Structure.Logical => False
  | Structure.Selection => False
  | Structure.Loop => False
  | Structure.Sequential => True

-- The theorem statement to prove that the sequential structure is necessary for any algorithm.
theorem algorithm_must_have_sequential_structure :
  necessary Structure.Sequential :=
by
  sorry

end NUMINAMATH_GPT_algorithm_must_have_sequential_structure_l2070_207025


namespace NUMINAMATH_GPT_range_of_m_l2070_207019

noncomputable def p (x : ℝ) : Prop := |x - 3| ≤ 2
noncomputable def q (x m : ℝ) : Prop := (x - m + 1) * (x - m - 1) ≤ 0

theorem range_of_m {m : ℝ} (H : ∀ (x : ℝ), ¬p x → ¬q x m) :
  2 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2070_207019


namespace NUMINAMATH_GPT_range_of_m_F_x2_less_than_x2_minus_1_l2070_207013

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := 3 - 2 / x
noncomputable def T (x m : ℝ) : ℝ := Real.log x - x - 2 * m
noncomputable def F (x m : ℝ) : ℝ := x - m / x - 2 * Real.log x
noncomputable def h (t : ℝ) : ℝ := t - 2 * Real.log t - 1

-- (1)
theorem range_of_m (m : ℝ) (h_intersections : ∃ x y : ℝ, T x m = 0 ∧ T y m = 0 ∧ x ≠ y) :
  m < -1 / 2 := sorry

-- (2)
theorem F_x2_less_than_x2_minus_1 {m : ℝ} (h₀ : 0 < m ∧ m < 1) {x₁ x₂ : ℝ} (h₁ : 0 < x₁ ∧ x₁ < x₂)
  (h₂ : F x₁ m = 0 ∧ F x₂ m = 0) :
  F x₂ m < x₂ - 1 := sorry

end NUMINAMATH_GPT_range_of_m_F_x2_less_than_x2_minus_1_l2070_207013


namespace NUMINAMATH_GPT_no_such_A_exists_l2070_207041

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem no_such_A_exists :
  ¬ ∃ A : ℕ, 0 < A ∧ digit_sum A = 16 ∧ digit_sum (2 * A) = 17 :=
by 
  sorry

end NUMINAMATH_GPT_no_such_A_exists_l2070_207041


namespace NUMINAMATH_GPT_desired_alcohol_percentage_l2070_207054

def initial_volume := 6.0
def initial_percentage := 35.0 / 100.0
def added_alcohol := 1.8
def final_volume := initial_volume + added_alcohol
def initial_alcohol := initial_volume * initial_percentage
def final_alcohol := initial_alcohol + added_alcohol
def desired_percentage := (final_alcohol / final_volume) * 100.0

theorem desired_alcohol_percentage : desired_percentage = 50.0 := 
by
  -- Proof would go here, but is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_desired_alcohol_percentage_l2070_207054


namespace NUMINAMATH_GPT_s_of_1_l2070_207061

def t (x : ℚ) : ℚ := 5 * x - 10
def s (y : ℚ) : ℚ := (y^2 / (5^2)) + (5 * y / 5) + 6  -- reformulated to fit conditions

theorem s_of_1 :
  s (1 : ℚ) = 546 / 25 := by
  sorry

end NUMINAMATH_GPT_s_of_1_l2070_207061


namespace NUMINAMATH_GPT_value_of_a_l2070_207080

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * a * x^2 + 6 * x

theorem value_of_a (a : ℝ) : f' (-1) a = 4 → a = 10 / 3 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l2070_207080


namespace NUMINAMATH_GPT_gcd_six_digit_repeat_l2070_207098

theorem gcd_six_digit_repeat (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) : 
  ∀ m : ℕ, m = 1001 * n → (gcd m 1001 = 1001) :=
by
  sorry

end NUMINAMATH_GPT_gcd_six_digit_repeat_l2070_207098


namespace NUMINAMATH_GPT_balls_into_boxes_l2070_207069

theorem balls_into_boxes :
  ∃ n : ℕ, n = 56 ∧ (∀ a b c d : ℕ, a + b + c + d = 5 ∧ 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d →
    n = 4 * (b + c + d + 1)) :=
by sorry

end NUMINAMATH_GPT_balls_into_boxes_l2070_207069


namespace NUMINAMATH_GPT_paul_tickets_left_l2070_207055

theorem paul_tickets_left (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) :
  initial_tickets = 11 → spent_tickets = 3 → remaining_tickets = initial_tickets - spent_tickets → remaining_tickets = 8 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_paul_tickets_left_l2070_207055


namespace NUMINAMATH_GPT_problem_1_problem_2_l2070_207097

-- Define f as an odd function on ℝ 
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Define the main property given in the problem
def property (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, (a + b) ≠ 0 → (f a + f b) / (a + b) > 0

-- Problem 1: Prove that if a > b then f(a) > f(b)
theorem problem_1 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  ∀ a b : ℝ, a > b → f a > f b := sorry

-- Problem 2: Prove that given f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x in [0, +∞), the range of k is k < 1
theorem problem_2 (f : ℝ → ℝ) (h_odd : odd_function f) (h_property : property f) :
  (∀ x : ℝ, 0 ≤ x → f (9 ^ x - 2 * 3 ^ x) + f (2 * 9 ^ x - k) > 0) → k < 1 := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2070_207097


namespace NUMINAMATH_GPT_fill_trough_time_l2070_207077

theorem fill_trough_time 
  (old_pump_rate : ℝ := 1 / 600) 
  (new_pump_rate : ℝ := 1 / 200) : 
  1 / (old_pump_rate + new_pump_rate) = 150 := 
by 
  sorry

end NUMINAMATH_GPT_fill_trough_time_l2070_207077


namespace NUMINAMATH_GPT_initial_percentage_acidic_liquid_l2070_207068

theorem initial_percentage_acidic_liquid (P : ℝ) :
  let initial_volume := 12
  let removed_volume := 4
  let final_volume := initial_volume - removed_volume
  let desired_concentration := 60
  (P/100) * initial_volume = (desired_concentration/100) * final_volume →
  P = 40 :=
by
  intros
  sorry

end NUMINAMATH_GPT_initial_percentage_acidic_liquid_l2070_207068


namespace NUMINAMATH_GPT_Tyler_age_l2070_207064

variable (T B S : ℕ) -- Assuming ages are non-negative integers

theorem Tyler_age (h1 : T = B - 3) (h2 : T + B + S = 25) (h3 : S = B + 2) : T = 6 := by
  sorry

end NUMINAMATH_GPT_Tyler_age_l2070_207064


namespace NUMINAMATH_GPT_rainfall_on_tuesday_is_correct_l2070_207043

-- Define the total days in a week
def days_in_week : ℕ := 7

-- Define the average rainfall for the whole week
def avg_rainfall : ℝ := 3.0

-- Define the total rainfall for the week
def total_rainfall : ℝ := avg_rainfall * days_in_week

-- Define a proposition that states rainfall on Tuesday equals 10.5 cm
def rainfall_on_tuesday (T : ℝ) : Prop :=
  T = 10.5

-- Prove that the rainfall on Tuesday is 10.5 cm given the conditions
theorem rainfall_on_tuesday_is_correct : rainfall_on_tuesday (total_rainfall / 2) :=
by
  sorry

end NUMINAMATH_GPT_rainfall_on_tuesday_is_correct_l2070_207043


namespace NUMINAMATH_GPT_base8_base13_to_base10_sum_l2070_207033

-- Definitions for the base 8 and base 13 numbers
def base8_to_base10 (a b c : ℕ) : ℕ := a * 64 + b * 8 + c
def base13_to_base10 (d e f : ℕ) : ℕ := d * 169 + e * 13 + f

-- Constants for the specific numbers in the problem
def num1 := base8_to_base10 5 3 7
def num2 := base13_to_base10 4 12 5

-- The theorem to prove
theorem base8_base13_to_base10_sum : num1 + num2 = 1188 := by
  sorry

end NUMINAMATH_GPT_base8_base13_to_base10_sum_l2070_207033


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l2070_207057

-- Define the conditions and prove the required question.
theorem batsman_average_after_17th_inning (A : ℕ) (h1 : 17 * (A + 10) = 16 * A + 300) :
  (A + 10) = 140 :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l2070_207057


namespace NUMINAMATH_GPT_sum_cubes_identity_l2070_207023

theorem sum_cubes_identity (x y z : ℝ) (h1 : x + y + z = 10) (h2 : xy + yz + zx = 20) :
    x^3 + y^3 + z^3 - 3 * x * y * z = 400 := by
  sorry

end NUMINAMATH_GPT_sum_cubes_identity_l2070_207023


namespace NUMINAMATH_GPT_brady_passing_yards_proof_l2070_207079

def tom_brady_current_passing_yards 
  (record_yards : ℕ) (games_left : ℕ) (average_yards_needed : ℕ) 
  (total_yards_needed_to_break_record : ℕ :=
    record_yards + 1) : ℕ :=
  total_yards_needed_to_break_record - games_left * average_yards_needed

theorem brady_passing_yards_proof :
  tom_brady_current_passing_yards 5999 6 300 = 4200 :=
by 
  sorry

end NUMINAMATH_GPT_brady_passing_yards_proof_l2070_207079


namespace NUMINAMATH_GPT_max_k_consecutive_sum_2_times_3_pow_8_l2070_207004

theorem max_k_consecutive_sum_2_times_3_pow_8 :
  ∃ k : ℕ, 0 < k ∧ 
           (∃ n : ℕ, 2 * 3^8 = (k * (2 * n + k + 1)) / 2) ∧
           (∀ k' : ℕ, (∃ n' : ℕ, 0 < k' ∧ 2 * 3^8 = (k' * (2 * n' + k' + 1)) / 2) → k' ≤ 81) :=
sorry

end NUMINAMATH_GPT_max_k_consecutive_sum_2_times_3_pow_8_l2070_207004


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l2070_207002

variable (A : ℝ) -- Area of the triangle ABC
variable (S_heptagon : ℝ) -- Area of the heptagon ADECFGH
variable (S_overlap : ℝ) -- Overlapping area after folding

-- Given conditions
axiom ratio_condition : S_heptagon = (5 / 7) * A
axiom overlap_condition : S_overlap = 8

-- Proof statement
theorem area_of_triangle_ABC :
  A = 28 := by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l2070_207002


namespace NUMINAMATH_GPT_Uki_earnings_l2070_207029

theorem Uki_earnings (cupcake_price cookie_price biscuit_price : ℝ) 
                     (cupcake_count cookie_count biscuit_count : ℕ)
                     (days : ℕ) :
  cupcake_price = 1.50 →
  cookie_price = 2 →
  biscuit_price = 1 →
  cupcake_count = 20 →
  cookie_count = 10 →
  biscuit_count = 20 →
  days = 5 →
  (days : ℝ) * (cupcake_price * (cupcake_count : ℝ) + cookie_price * (cookie_count : ℝ) + biscuit_price * (biscuit_count : ℝ)) = 350 := 
by
  sorry

end NUMINAMATH_GPT_Uki_earnings_l2070_207029


namespace NUMINAMATH_GPT_train_time_36kmph_200m_l2070_207091

/-- How many seconds will a train 200 meters long running at the rate of 36 kmph take to pass a certain telegraph post? -/
def time_to_pass_post (length_of_train : ℕ) (speed_kmph : ℕ) : ℕ :=
  length_of_train * 3600 / (speed_kmph * 1000)

theorem train_time_36kmph_200m : time_to_pass_post 200 36 = 20 := by
  sorry

end NUMINAMATH_GPT_train_time_36kmph_200m_l2070_207091


namespace NUMINAMATH_GPT_find_k_l2070_207076

-- Define the vectors and the condition of perpendicularity
def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, -1)
def c (k : ℝ) : ℝ × ℝ := (3 + k, 1 - k)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The primary statement we aim to prove
theorem find_k : ∃ k : ℝ, dot_product a (c k) = 0 ∧ k = -5 :=
by
  exists -5
  sorry

end NUMINAMATH_GPT_find_k_l2070_207076


namespace NUMINAMATH_GPT_div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l2070_207014

-- Define the division of 246 by 73
theorem div_246_by_73 :
  246 / 73 = 3 + 27 / 73 :=
sorry

-- Define the sum calculation
theorem sum_9999_999_99_9 :
  9999 + 999 + 99 + 9 = 11106 :=
sorry

-- Define the product calculation
theorem prod_25_29_4 :
  25 * 29 * 4 = 2900 :=
sorry

end NUMINAMATH_GPT_div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l2070_207014


namespace NUMINAMATH_GPT_words_on_each_page_l2070_207090

theorem words_on_each_page (p : ℕ) (h1 : p ≤ 120) (h2 : 150 * p % 221 = 210) : p = 48 :=
sorry

end NUMINAMATH_GPT_words_on_each_page_l2070_207090


namespace NUMINAMATH_GPT_total_apples_in_stack_l2070_207020

theorem total_apples_in_stack:
  let base_layer := 6 * 9
  let layer_2 := 5 * 8
  let layer_3 := 4 * 7
  let layer_4 := 3 * 6
  let layer_5 := 2 * 5
  let layer_6 := 1 * 4
  let top_layer := 2
  base_layer + layer_2 + layer_3 + layer_4 + layer_5 + layer_6 + top_layer = 156 :=
by sorry

end NUMINAMATH_GPT_total_apples_in_stack_l2070_207020


namespace NUMINAMATH_GPT_determine_set_of_integers_for_ratio_l2070_207001

def arithmetic_sequences (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n / T n = (31 * n + 101) / (n + 3)

def ratio_is_integer (a : ℕ → ℕ) (b : ℕ → ℕ) (n : ℕ) : Prop :=
  ∃ k : ℕ, a n / b n = k

theorem determine_set_of_integers_for_ratio (a b : ℕ → ℕ) (S T : ℕ → ℕ) :
  arithmetic_sequences a b S T →
  {n : ℕ | ratio_is_integer a b n} = {1, 3} :=
sorry

end NUMINAMATH_GPT_determine_set_of_integers_for_ratio_l2070_207001


namespace NUMINAMATH_GPT_find_m_range_l2070_207092

variable {x y m : ℝ}

theorem find_m_range (h1 : x + 2 * y = m + 4) (h2 : 2 * x + y = 2 * m - 1)
    (h3 : x + y < 2) (h4 : x - y < 4) : m < 1 := by
  sorry

end NUMINAMATH_GPT_find_m_range_l2070_207092


namespace NUMINAMATH_GPT_example_theorem_l2070_207056

open Set

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

def example_problem : Prop :=
  U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4} ∧ (U \ (M ∪ N)) = {5}

theorem example_theorem (h : U = {1, 2, 3, 4, 5} ∧ M = {1, 2} ∧ N = {3, 4}) : 
    (U \ (M ∪ N)) = {5} :=
  by sorry

end NUMINAMATH_GPT_example_theorem_l2070_207056


namespace NUMINAMATH_GPT_true_for_2_and_5_l2070_207088

theorem true_for_2_and_5 (x : ℝ) : ((x - 2) * (x - 5) = 0) ↔ (x = 2 ∨ x = 5) :=
by
  sorry

end NUMINAMATH_GPT_true_for_2_and_5_l2070_207088


namespace NUMINAMATH_GPT_find_a_l2070_207070

theorem find_a 
  (a : ℝ) 
  (h1 : ∀ x : ℝ, (x - 3) ^ 2 + 5 = a * x^2 + bx + c) 
  (h2 : (3, 5) = (3, a * 3 ^ 2 + b * 3 + c))
  (h3 : (-2, -20) = (-2, a * (-2)^2 + b * (-2) + c)) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2070_207070


namespace NUMINAMATH_GPT_find_b_minus_c_l2070_207030

theorem find_b_minus_c (a b c : ℤ) (h : (x^2 + a * x - 3) * (x + 1) = x^3 + b * x^2 + c * x - 3) : b - c = 4 := by
  -- We would normally construct the proof here.
  sorry

end NUMINAMATH_GPT_find_b_minus_c_l2070_207030


namespace NUMINAMATH_GPT_contrapositive_correct_l2070_207099

-- Define the main condition: If a ≥ 1/2, then ∀ x ≥ 0, f(x) ≥ 0
def main_condition (a : ℝ) (f : ℝ → ℝ) : Prop :=
  a ≥ 1/2 → ∀ x : ℝ, x ≥ 0 → f x ≥ 0

-- Define the contrapositive statement: If ∃ x ≥ 0 such that f(x) < 0, then a < 1/2
def contrapositive (a : ℝ) (f : ℝ → ℝ) : Prop :=
  (∃ x : ℝ, x ≥ 0 ∧ f x < 0) → a < 1/2

-- Theorem to prove that the contrapositive statement is correct
theorem contrapositive_correct (a : ℝ) (f : ℝ → ℝ) :
  main_condition a f ↔ contrapositive a f :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_correct_l2070_207099


namespace NUMINAMATH_GPT_part_1_solution_set_part_2_a_range_l2070_207015

noncomputable def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

theorem part_1_solution_set (a : ℝ) (h : a = 4) :
  {x : ℝ | f x a ≥ 5} = {x : ℝ | x ≤ 0 ∨ x ≥ 5} :=
by
  sorry

theorem part_2_a_range :
  {a : ℝ | ∀ x : ℝ, f x a ≥ 4} = {a : ℝ | a ≤ -3 ∨ a ≥ 5} :=
by
  sorry

end NUMINAMATH_GPT_part_1_solution_set_part_2_a_range_l2070_207015


namespace NUMINAMATH_GPT_mps_to_kmph_conversion_l2070_207022

/-- Define the conversion factor from meters per second to kilometers per hour. -/
def mps_to_kmph : ℝ := 3.6

/-- Define the speed in meters per second. -/
def speed_mps : ℝ := 5

/-- Define the converted speed in kilometers per hour. -/
def speed_kmph : ℝ := 18

/-- Statement asserting the conversion from meters per second to kilometers per hour. -/
theorem mps_to_kmph_conversion : speed_mps * mps_to_kmph = speed_kmph := by 
  sorry

end NUMINAMATH_GPT_mps_to_kmph_conversion_l2070_207022


namespace NUMINAMATH_GPT_abscissa_of_A_is_5_l2070_207046

theorem abscissa_of_A_is_5
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A.1 = A.2 ∧ A.1 > 0)
  (hB : B = (5, 0))
  (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hC : C = ((A.1 + 5) / 2, A.2 / 2))
  (hD : D = (5 / 2, 5 / 2))
  (dot_product_eq : (B.1 - A.1, B.2 - A.2) • (D.1 - C.1, D.2 - C.2) = 0) :
  A.1 = 5 :=
sorry

end NUMINAMATH_GPT_abscissa_of_A_is_5_l2070_207046


namespace NUMINAMATH_GPT_product_three_power_l2070_207034

theorem product_three_power (w : ℕ) (hW : w = 132) (hProd : ∃ (k : ℕ), 936 * w = 2^5 * 11^2 * k) : 
  ∃ (n : ℕ), (936 * w) = (2^5 * 11^2 * (3^3 * n)) :=
by 
  sorry

end NUMINAMATH_GPT_product_three_power_l2070_207034


namespace NUMINAMATH_GPT_probability_difference_l2070_207000

theorem probability_difference (red_marbles black_marbles : ℤ) (h_red : red_marbles = 1500) (h_black : black_marbles = 1500) :
  |(22485 / 44985 : ℚ) - (22500 / 44985 : ℚ)| = 15 / 44985 := 
by {
  sorry
}

end NUMINAMATH_GPT_probability_difference_l2070_207000


namespace NUMINAMATH_GPT_conference_end_time_correct_l2070_207065

-- Define the conference conditions
def conference_start_time : ℕ := 15 * 60 -- 3:00 p.m. in minutes
def conference_duration : ℕ := 450 -- 450 minutes duration
def daylight_saving_adjustment : ℕ := 60 -- clocks set forward by one hour

-- Define the end time computation
def end_time_without_daylight_saving : ℕ := conference_start_time + conference_duration
def end_time_with_daylight_saving : ℕ := end_time_without_daylight_saving + daylight_saving_adjustment

-- Prove that the conference ended at 11:30 p.m. (11:30 p.m. in minutes is 23 * 60 + 30)
theorem conference_end_time_correct : end_time_with_daylight_saving = 23 * 60 + 30 := by
  sorry

end NUMINAMATH_GPT_conference_end_time_correct_l2070_207065


namespace NUMINAMATH_GPT_farm_horses_more_than_cows_l2070_207063

variable (x : ℤ) -- number of cows initially, must be a positive integer

def initial_horses := 6 * x
def initial_cows := x
def horses_after_transaction := initial_horses - 30
def cows_after_transaction := initial_cows + 30

-- New ratio after transaction
def new_ratio := horses_after_transaction * 1 = 4 * cows_after_transaction

-- Prove that the farm owns 315 more horses than cows after transaction
theorem farm_horses_more_than_cows :
  new_ratio → horses_after_transaction - cows_after_transaction = 315 :=
by
  sorry

end NUMINAMATH_GPT_farm_horses_more_than_cows_l2070_207063


namespace NUMINAMATH_GPT_probability_heads_9_tails_at_least_2_l2070_207048

noncomputable def probability_exactly_nine_heads : ℚ :=
  let total_outcomes := 2 ^ 12
  let successful_outcomes := Nat.choose 12 9
  successful_outcomes / total_outcomes

theorem probability_heads_9_tails_at_least_2 (n : ℕ) (h : n = 12) :
  n = 12 → probability_exactly_nine_heads = 55 / 1024 := by
  intros h
  sorry

end NUMINAMATH_GPT_probability_heads_9_tails_at_least_2_l2070_207048


namespace NUMINAMATH_GPT_sum_of_even_numbers_l2070_207027

-- Define the sequence of even numbers between 1 and 1001
def even_numbers_sequence (n : ℕ) := 2 * n

-- Conditions
def first_term := 2
def last_term := 1000
def common_difference := 2
def num_terms := 500
def sum_arithmetic_series (n : ℕ) (a l : ℕ) := n * (a + l) / 2

-- Main statement to be proved
theorem sum_of_even_numbers : 
  sum_arithmetic_series num_terms first_term last_term = 250502 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_even_numbers_l2070_207027


namespace NUMINAMATH_GPT_find_original_price_l2070_207062

-- Define the given conditions
def decreased_price : ℝ := 836
def decrease_percentage : ℝ := 0.24
def remaining_percentage : ℝ := 1 - decrease_percentage -- 76% in decimal

-- Define the original price as a variable
variable (x : ℝ)

-- State the theorem
theorem find_original_price (h : remaining_percentage * x = decreased_price) : x = 1100 :=
by
  sorry

end NUMINAMATH_GPT_find_original_price_l2070_207062


namespace NUMINAMATH_GPT_students_juice_count_l2070_207035

theorem students_juice_count (students chose_water chose_juice : ℕ) 
  (h1 : chose_water = 140) 
  (h2 : (25 : ℚ) / 100 * (students : ℚ) = chose_juice)
  (h3 : (70 : ℚ) / 100 * (students : ℚ) = chose_water) : 
  chose_juice = 50 :=
by 
  sorry

end NUMINAMATH_GPT_students_juice_count_l2070_207035


namespace NUMINAMATH_GPT_B_needs_days_l2070_207078

theorem B_needs_days (A_rate B_rate Combined_rate : ℝ) (x : ℝ) (W : ℝ) (h1: A_rate = W / 140)
(h2: B_rate = W / (3 * x)) (h3 : Combined_rate = 60 * W) (h4 : Combined_rate = A_rate + B_rate) :
 x = 140 / 25197 :=
by
  sorry

end NUMINAMATH_GPT_B_needs_days_l2070_207078


namespace NUMINAMATH_GPT_largest_fraction_l2070_207031

theorem largest_fraction :
  (∀ (a b : ℚ), a = 2 / 5 → b = 1 / 3 → a < b) ∧  
  (∀ (a c : ℚ), a = 2 / 5 → c = 7 / 15 → a < c) ∧ 
  (∀ (a d : ℚ), a = 2 / 5 → d = 5 / 12 → a < d) ∧ 
  (∀ (a e : ℚ), a = 2 / 5 → e = 3 / 8 → a < e) ∧ 
  (∀ (b c : ℚ), b = 1 / 3 → c = 7 / 15 → b < c) ∧
  (∀ (b d : ℚ), b = 1 / 3 → d = 5 / 12 → b < d) ∧ 
  (∀ (b e : ℚ), b = 1 / 3 → e = 3 / 8 → b < e) ∧ 
  (∀ (c d : ℚ), c = 7 / 15 → d = 5 / 12 → c > d) ∧
  (∀ (c e : ℚ), c = 7 / 15 → e = 3 / 8 → c > e) ∧
  (∀ (d e : ℚ), d = 5 / 12 → e = 3 / 8 → d > e) :=
sorry

end NUMINAMATH_GPT_largest_fraction_l2070_207031


namespace NUMINAMATH_GPT_relationship_abc_l2070_207082

noncomputable def a (x : ℝ) : ℝ := Real.log x
noncomputable def b (x : ℝ) : ℝ := Real.exp (Real.log x)
noncomputable def c (x : ℝ) : ℝ := Real.exp (Real.log (1 / x))

theorem relationship_abc (x : ℝ) (h : (1 / Real.exp 1) < x ∧ x < 1) : a x < b x ∧ b x < c x :=
by
  have ha : a x = Real.log x := rfl
  have hb : b x = Real.exp (Real.log x) := rfl
  have hc : c x = Real.exp (Real.log (1 / x)) := rfl
  sorry

end NUMINAMATH_GPT_relationship_abc_l2070_207082


namespace NUMINAMATH_GPT_lifting_ratio_after_gain_l2070_207074

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end NUMINAMATH_GPT_lifting_ratio_after_gain_l2070_207074


namespace NUMINAMATH_GPT_plane_through_point_contains_line_l2070_207011

-- Definitions from conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def passes_through (p : Point) (plane : Point → Prop) : Prop :=
  plane p

def contains_line (line : ℝ → Point) (plane : Point → Prop) : Prop :=
  ∀ t, plane (line t)

def line_eq (t : ℝ) : Point :=
  ⟨4 * t + 2, -6 * t - 3, 2 * t + 4⟩

def plane_eq (A B C D : ℝ) (p : Point) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

theorem plane_through_point_contains_line :
  ∃ (A B C D : ℝ), 1 < A ∧ gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  passes_through ⟨1, 2, -3⟩ (plane_eq A B C D) ∧
  contains_line line_eq (plane_eq A B C D) ∧ 
  (∃ (k : ℝ), 3 * k = A ∧ k = 1 / 3 ∧ B = k * 1 ∧ C = k * (-3) ∧ D = k * 2) :=
sorry

end NUMINAMATH_GPT_plane_through_point_contains_line_l2070_207011


namespace NUMINAMATH_GPT_upstream_distance_l2070_207039

theorem upstream_distance
  (man_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (effective_downstream_speed: ℝ)
  (stream_speed : ℝ)
  (upstream_time : ℝ)
  (upstream_distance : ℝ):
  man_speed = 7 ∧ downstream_distance = 45 ∧ downstream_time = 5 ∧ effective_downstream_speed = man_speed + stream_speed 
  ∧ effective_downstream_speed * downstream_time = downstream_distance 
  ∧ upstream_time = 5 ∧ upstream_distance = (man_speed - stream_speed) * upstream_time 
  → upstream_distance = 25 :=
by
  sorry

end NUMINAMATH_GPT_upstream_distance_l2070_207039


namespace NUMINAMATH_GPT_first_day_is_wednesday_l2070_207038

theorem first_day_is_wednesday (day22_wednesday : ∀ n, n = 22 → (n = 22 → "Wednesday" = "Wednesday")) :
  ∀ n, n = 1 → (n = 1 → "Wednesday" = "Wednesday") :=
by
  sorry

end NUMINAMATH_GPT_first_day_is_wednesday_l2070_207038


namespace NUMINAMATH_GPT_binomial_expansion_equality_l2070_207021

theorem binomial_expansion_equality (x : ℝ) : 
  (x-1)^4 - 4*x*(x-1)^3 + 6*(x^2)*(x-1)^2 - 4*(x^3)*(x-1)*x^4 = 1 := 
by 
  sorry 

end NUMINAMATH_GPT_binomial_expansion_equality_l2070_207021


namespace NUMINAMATH_GPT_bean_lands_outside_inscribed_circle_l2070_207094

theorem bean_lands_outside_inscribed_circle :
  let a := 8
  let b := 15
  let c := 17  -- hypotenuse computed as sqrt(a^2 + b^2)
  let area_triangle := (1 / 2) * a * b
  let s := (a + b + c) / 2  -- semiperimeter
  let r := area_triangle / s -- radius of the inscribed circle
  let area_incircle := π * r^2
  let probability_outside := 1 - area_incircle / area_triangle
  probability_outside = 1 - (3 * π) / 20 := 
by
  sorry

end NUMINAMATH_GPT_bean_lands_outside_inscribed_circle_l2070_207094


namespace NUMINAMATH_GPT_solve_system_l2070_207073

theorem solve_system :
  ∃ (x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ : ℚ),
  x₁ + 12 * x₂ = 15 ∧
  x₁ - 12 * x₂ + 11 * x₃ = 2 ∧
  x₁ - 11 * x₃ + 10 * x₄ = 2 ∧
  x₁ - 10 * x₄ + 9 * x₅ = 2 ∧
  x₁ - 9 * x₅ + 8 * x₆ = 2 ∧
  x₁ - 8 * x₆ + 7 * x₇ = 2 ∧
  x₁ - 7 * x₇ + 6 * x₈ = 2 ∧
  x₁ - 6 * x₈ + 5 * x₉ = 2 ∧
  x₁ - 5 * x₉ + 4 * x₁₀ = 2 ∧
  x₁ - 4 * x₁₀ + 3 * x₁₁ = 2 ∧
  x₁ - 3 * x₁₁ + 2 * x₁₂ = 2 ∧
  x₁ - 2 * x₁₂ = 2 ∧
  x₁ = 37 / 12 ∧
  x₂ = 143 / 144 ∧
  x₃ = 65 / 66 ∧
  x₄ = 39 / 40 ∧
  x₅ = 26 / 27 ∧
  x₆ = 91 / 96 ∧
  x₇ = 13 / 14 ∧
  x₈ = 65 / 72 ∧
  x₉ = 13 / 15 ∧
  x₁₀ = 13 / 16 ∧
  x₁₁ = 13 / 18 ∧
  x₁₂ = 13 / 24 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l2070_207073


namespace NUMINAMATH_GPT_factor_expression_l2070_207086

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) :=
by sorry

end NUMINAMATH_GPT_factor_expression_l2070_207086


namespace NUMINAMATH_GPT_sequence_general_formula_l2070_207084

noncomputable def sequence_formula (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) : ℝ :=
  if n = 0 then 0 else 2^(n-2)

theorem sequence_general_formula {a : ℕ → ℝ} {S : ℕ → ℝ} (hpos : ∀ n, a n > 0)
  (hSn : ∀ n, 2 * a n = S n + 0.5) : ∀ n, a n = sequence_formula a S n :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l2070_207084


namespace NUMINAMATH_GPT_impossible_event_l2070_207083

noncomputable def EventA := ∃ (ω : ℕ), ω = 0 ∨ ω = 1
noncomputable def EventB := ∃ (t : ℤ), t >= 0
noncomputable def Bag := {b : String // b = "White"}
noncomputable def EventC := ∀ (x : Bag), x.val ≠ "Red"
noncomputable def EventD := ∀ (a b : ℤ), (a > 0 ∧ b < 0) → a > b

theorem impossible_event:
  (EventA ∧ EventB ∧ EventD) →
  EventC :=
by
  sorry

end NUMINAMATH_GPT_impossible_event_l2070_207083


namespace NUMINAMATH_GPT_apple_price_equals_oranges_l2070_207024

theorem apple_price_equals_oranges (A O : ℝ) (H1 : A = 28 * O) (H2 : 45 * A + 60 * O = 1350) (H3 : 30 * A + 40 * O = 900) : A = 28 * O :=
by
  sorry

end NUMINAMATH_GPT_apple_price_equals_oranges_l2070_207024


namespace NUMINAMATH_GPT_subtracting_seven_percent_l2070_207067

theorem subtracting_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a :=
by 
  sorry

end NUMINAMATH_GPT_subtracting_seven_percent_l2070_207067


namespace NUMINAMATH_GPT_fraction_irreducible_l2070_207081

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := 
sorry

end NUMINAMATH_GPT_fraction_irreducible_l2070_207081


namespace NUMINAMATH_GPT_cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l2070_207051

theorem cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths 
  (b : ℝ)
  (h : ∀ x : ℝ, 4 * x^3 + 3 * x^2 + b * x + 27 = 0 → ∃! r : ℝ, r = x) :
  b = 3 / 4 := 
by
  sorry

end NUMINAMATH_GPT_cubic_has_exactly_one_real_solution_sum_b_eq_three_fourths_l2070_207051


namespace NUMINAMATH_GPT_problem_a_problem_b_l2070_207012

variable (α : ℝ)

theorem problem_a (hα : 0 < α ∧ α < π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = Real.tan (α / 2) :=
sorry

theorem problem_b (hα : π < α ∧ α < 2 * π) :
  Real.sqrt ((2 * Real.sin α - Real.sin (2 * α)) / (2 * Real.sin α + Real.sin (2 * α))) = -Real.tan (α / 2) :=
sorry

end NUMINAMATH_GPT_problem_a_problem_b_l2070_207012


namespace NUMINAMATH_GPT_sequence_general_term_l2070_207050

noncomputable def sequence_condition (a : ℕ → ℝ) : Prop :=
  (∀ n, 2 * a n = 3 * a (n + 1)) ∧ 
  (a 2 * a 5 = 8 / 27) ∧ 
  (∀ n, 0 < a n)

theorem sequence_general_term (a : ℕ → ℝ) (h : sequence_condition a) : 
  ∀ n, a n = (2 / 3)^(n - 2) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2070_207050


namespace NUMINAMATH_GPT_find_salary_of_january_l2070_207005

variables (J F M A May : ℝ)

theorem find_salary_of_january
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 := 
sorry

end NUMINAMATH_GPT_find_salary_of_january_l2070_207005


namespace NUMINAMATH_GPT_molecular_weight_NaClO_l2070_207032

theorem molecular_weight_NaClO :
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  Na + Cl + O = 74.44 :=
by
  let Na := 22.99
  let Cl := 35.45
  let O := 16.00
  sorry

end NUMINAMATH_GPT_molecular_weight_NaClO_l2070_207032


namespace NUMINAMATH_GPT_battery_current_l2070_207053

variable (R : ℝ) (I : ℝ)

theorem battery_current (h1 : I = 48 / R) (h2 : R = 12) : I = 4 :=
by
  sorry

end NUMINAMATH_GPT_battery_current_l2070_207053


namespace NUMINAMATH_GPT_exists_infinitely_many_triples_l2070_207028

theorem exists_infinitely_many_triples :
  ∀ n : ℕ, ∃ (a b c : ℕ), a^2 + b^2 + c^2 + 2016 = a * b * c :=
sorry

end NUMINAMATH_GPT_exists_infinitely_many_triples_l2070_207028


namespace NUMINAMATH_GPT_closest_point_on_parabola_to_line_is_l2070_207052

-- Definitions of the parabola and the line
def parabola (x : ℝ) : ℝ := 4 * x^2
def line (x : ℝ) : ℝ := 4 * x - 5

-- Prove that the point on the parabola that is closest to the line is (1/2, 1)
theorem closest_point_on_parabola_to_line_is (x y : ℝ) :
  parabola x = y ∧ (∀ (x' y' : ℝ), parabola x' = y' -> (line x - y)^2 >= (line x' - y')^2) ->
  (x, y) = (1/2, 1) :=
by
  sorry

end NUMINAMATH_GPT_closest_point_on_parabola_to_line_is_l2070_207052


namespace NUMINAMATH_GPT_exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l2070_207047

theorem exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1 (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (∃ a : ℤ, a^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end NUMINAMATH_GPT_exists_a_squared_congruent_neg1_iff_p_mod_4_eq_1_l2070_207047


namespace NUMINAMATH_GPT_pencils_per_student_l2070_207059

theorem pencils_per_student (num_students total_pencils : ℕ)
  (h1 : num_students = 4) (h2 : total_pencils = 8) : total_pencils / num_students = 2 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_pencils_per_student_l2070_207059


namespace NUMINAMATH_GPT_problem_correct_l2070_207026

noncomputable def problem := 
  1 - (1 / 2)⁻¹ * Real.sin (60 * Real.pi / 180) + abs (2^0 - Real.sqrt 3) = 0

theorem problem_correct : problem := by
  sorry

end NUMINAMATH_GPT_problem_correct_l2070_207026


namespace NUMINAMATH_GPT_distinct_positive_values_count_l2070_207071

theorem distinct_positive_values_count : 
  ∃ (n : ℕ), n = 33 ∧ ∀ (x : ℕ), 
    (20 ≤ x ∧ x ≤ 99 ∧ 20 ≤ 2 * x ∧ 2 * x < 200 ∧ 3 * x ≥ 200) 
    ↔ (67 ≤ x ∧ x < 100) :=
  sorry

end NUMINAMATH_GPT_distinct_positive_values_count_l2070_207071


namespace NUMINAMATH_GPT_part1_answer1_part1_answer2_part2_answer1_part2_answer2_l2070_207045

open Set

def A : Set ℕ := {x | 1 ≤ x ∧ x < 11}
def B : Set ℕ := {1, 2, 3, 4}
def C : Set ℕ := {3, 4, 5, 6, 7}

theorem part1_answer1 : A ∩ C = {3, 4, 5, 6, 7} :=
by
  sorry

theorem part1_answer2 : A \ B = {5, 6, 7, 8, 9, 10} :=
by
  sorry

theorem part2_answer1 : A \ (B ∪ C) = {8, 9, 10} :=
by 
  sorry

theorem part2_answer2 : A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} :=
by 
  sorry

end NUMINAMATH_GPT_part1_answer1_part1_answer2_part2_answer1_part2_answer2_l2070_207045


namespace NUMINAMATH_GPT_find_x_l2070_207093

theorem find_x (x : ℝ) (hx₀ : 0 < x) (hx₁ : x < 180) 
  (h : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : 
  x = 110 :=
sorry

end NUMINAMATH_GPT_find_x_l2070_207093


namespace NUMINAMATH_GPT_denmark_pizza_combinations_l2070_207017

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end NUMINAMATH_GPT_denmark_pizza_combinations_l2070_207017


namespace NUMINAMATH_GPT_lower_limit_of_a_l2070_207009

theorem lower_limit_of_a (a b : ℤ) (h_a : a < 26) (h_b1 : b > 14) (h_b2 : b < 31) (h_ineq : (4 : ℚ) / 3 ≤ a / b) : 
  20 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_lower_limit_of_a_l2070_207009


namespace NUMINAMATH_GPT_sqrt_49_mul_sqrt_25_l2070_207018

theorem sqrt_49_mul_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_49_mul_sqrt_25_l2070_207018


namespace NUMINAMATH_GPT_find_x_l2070_207049

def myOperation (x y : ℝ) : ℝ := 2 * x * y

theorem find_x (x : ℝ) (h : myOperation 9 (myOperation 4 x) = 720) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l2070_207049


namespace NUMINAMATH_GPT_box_volume_l2070_207016

theorem box_volume (x y z : ℕ) 
  (h1 : 2 * x + 2 * y = 26)
  (h2 : x + z = 10)
  (h3 : y + z = 7) :
  x * y * z = 80 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_l2070_207016


namespace NUMINAMATH_GPT_solve_for_a_l2070_207075

variable (a b x : ℝ)

theorem solve_for_a (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) :
  a = 3 * x := sorry

end NUMINAMATH_GPT_solve_for_a_l2070_207075


namespace NUMINAMATH_GPT_parabola_c_value_l2070_207006

theorem parabola_c_value :
  ∃ a b c : ℝ, (∀ y : ℝ, 4 = a * (3 : ℝ)^2 + b * 3 + c ∧ 2 = a * 5^2 + b * 5 + c ∧ c = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_c_value_l2070_207006


namespace NUMINAMATH_GPT_train_length_l2070_207010

theorem train_length (L V : ℝ) 
  (h1 : L = V * 110) 
  (h2 : L + 700 = V * 180) : 
  L = 1100 :=
by
  sorry

end NUMINAMATH_GPT_train_length_l2070_207010


namespace NUMINAMATH_GPT_find_constants_C_and_A_l2070_207007

theorem find_constants_C_and_A :
  ∃ (C A : ℚ), (C * x + 7 - 17)/(x^2 - 9 * x + 20) = A / (x - 4) + 2 / (x - 5) ∧ B = 7 ∧ C = 12/5 ∧ A = 2/5 := sorry

end NUMINAMATH_GPT_find_constants_C_and_A_l2070_207007


namespace NUMINAMATH_GPT_determine_k_l2070_207042

variable (x y z k : ℝ)

theorem determine_k (h1 : 7 / (x + y) = k / (x + z)) (h2 : k / (x + z) = 11 / (z - y)) : k = 18 := 
by 
  sorry

end NUMINAMATH_GPT_determine_k_l2070_207042


namespace NUMINAMATH_GPT_find_a_l2070_207044

theorem find_a (a x y : ℤ) (h1 : x = 1) (h2 : y = -2) (h3 : a * x + y = 1) : a = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2070_207044


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2070_207036

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 2 = 1)
  (h2 : a 3 + a 4 = 8)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l2070_207036


namespace NUMINAMATH_GPT_percent_between_20000_and_150000_l2070_207087

-- Define the percentages for each group of counties
def less_than_20000 := 30
def between_20000_and_150000 := 45
def more_than_150000 := 25

-- State the theorem using the above definitions
theorem percent_between_20000_and_150000 :
  between_20000_and_150000 = 45 :=
sorry -- Proof placeholder

end NUMINAMATH_GPT_percent_between_20000_and_150000_l2070_207087


namespace NUMINAMATH_GPT_tangent_lines_through_point_l2070_207072

theorem tangent_lines_through_point :
  ∃ k : ℚ, ((5  * k - 12 * (36 - k * 2) + 36 = 0) ∨ (2 = 0)) := sorry

end NUMINAMATH_GPT_tangent_lines_through_point_l2070_207072


namespace NUMINAMATH_GPT_cost_of_other_disc_l2070_207037

theorem cost_of_other_disc (x : ℝ) (total_spent : ℝ) (num_discs : ℕ) (num_850_discs : ℕ) (price_850 : ℝ) 
    (total_cost : total_spent = 93) (num_bought : num_discs = 10) (num_850 : num_850_discs = 6) (price_per_850 : price_850 = 8.50) 
    (total_cost_850 : num_850_discs * price_850 = 51) (remaining_discs_cost : total_spent - 51 = 42) (remaining_discs : num_discs - num_850_discs = 4) :
    total_spent = num_850_discs * price_850 + (num_discs - num_850_discs) * x → x = 10.50 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_other_disc_l2070_207037


namespace NUMINAMATH_GPT_solve_equation_a_solve_equation_b_l2070_207058

-- Problem a
theorem solve_equation_a (a b x : ℝ) (h₀ : x ≠ a) (h₁ : x ≠ b) (h₂ : a + b ≠ 0) (h₃ : a ≠ 0) (h₄ : b ≠ 0) (h₅ : a ≠ b):
  (x + a) / (x - a) + (x + b) / (x - b) = 2 ↔ x = (2 * a * b) / (a + b) :=
by
  sorry

-- Problem b
theorem solve_equation_b (a b c d x : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : x ≠ 0) (h₃ : c ≠ 0) (h₄ : d ≠ 0) (h₅ : ab + c ≠ 0):
  c * (d / (a * b) - (a * b) / x) + d = c^2 / x ↔ x = (a * b * c) / d :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_a_solve_equation_b_l2070_207058


namespace NUMINAMATH_GPT_tenth_term_geometric_sequence_l2070_207003

def a := 5
def r := Rat.ofInt 3 / 4
def n := 10

theorem tenth_term_geometric_sequence :
  a * r^(n-1) = Rat.ofInt 98415 / Rat.ofInt 262144 := sorry

end NUMINAMATH_GPT_tenth_term_geometric_sequence_l2070_207003


namespace NUMINAMATH_GPT_annual_interest_rate_is_correct_l2070_207085

-- Define conditions
def principal : ℝ := 900
def finalAmount : ℝ := 992.25
def compoundingPeriods : ℕ := 2
def timeYears : ℕ := 1

-- Compound interest formula
def compound_interest (P A r : ℝ) (n t : ℕ) : Prop :=
  A = P * (1 + r / n) ^ (n * t)

-- Statement to prove
theorem annual_interest_rate_is_correct :
  ∃ r : ℝ, compound_interest principal finalAmount r compoundingPeriods timeYears ∧ r = 0.10 :=
by 
  sorry

end NUMINAMATH_GPT_annual_interest_rate_is_correct_l2070_207085


namespace NUMINAMATH_GPT_complex_number_equality_l2070_207040

open Complex

theorem complex_number_equality (u v : ℂ) 
  (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
  (h2 : abs (u + v) = abs (u * v + 1)) : 
  u = 1 ∨ v = 1 :=
sorry

end NUMINAMATH_GPT_complex_number_equality_l2070_207040


namespace NUMINAMATH_GPT_arcade_playtime_l2070_207089

noncomputable def cost_per_six_minutes : ℝ := 0.50
noncomputable def total_spent : ℝ := 15
noncomputable def minutes_per_interval : ℝ := 6
noncomputable def minutes_per_hour : ℝ := 60

theorem arcade_playtime :
  (total_spent / cost_per_six_minutes) * minutes_per_interval / minutes_per_hour = 3 :=
by
  sorry

end NUMINAMATH_GPT_arcade_playtime_l2070_207089


namespace NUMINAMATH_GPT_total_weight_of_bottles_l2070_207008

variables (P G : ℕ) -- P stands for the weight of a plastic bottle, G stands for the weight of a glass bottle

-- Condition 1: The weight of 3 glass bottles is 600 grams
axiom glass_bottle_weight : 3 * G = 600

-- Condition 2: A glass bottle is 150 grams heavier than a plastic bottle
axiom glass_bottle_heavier : G = P + 150

-- The statement to prove: The total weight of 4 glass bottles and 5 plastic bottles is 1050 grams
theorem total_weight_of_bottles :
  4 * G + 5 * P = 1050 :=
sorry

end NUMINAMATH_GPT_total_weight_of_bottles_l2070_207008


namespace NUMINAMATH_GPT_negation_prop_equiv_l2070_207066

variable (a : ℝ)

theorem negation_prop_equiv :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 - 2 * a * x - 1 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 - 2 * a * x - 1 ≥ 0) :=
sorry

end NUMINAMATH_GPT_negation_prop_equiv_l2070_207066


namespace NUMINAMATH_GPT_xyz_sum_56_l2070_207096

theorem xyz_sum_56 (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y + z = 55) (h2 : y * z + x = 55) (h3 : z * x + y = 55)
  (even_cond : x % 2 = 0 ∨ y % 2 = 0 ∨ z % 2 = 0) :
  x + y + z = 56 :=
sorry

end NUMINAMATH_GPT_xyz_sum_56_l2070_207096
