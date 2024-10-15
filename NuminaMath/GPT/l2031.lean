import Mathlib

namespace NUMINAMATH_GPT_like_terms_m_eq_2_l2031_203130

theorem like_terms_m_eq_2 (m : ℕ) :
  (∀ (x y : ℝ), 3 * x^m * y^3 = 3 * x^2 * y^3) -> m = 2 :=
by
  intros _
  sorry

end NUMINAMATH_GPT_like_terms_m_eq_2_l2031_203130


namespace NUMINAMATH_GPT_maximize_S_n_l2031_203136

variable (a_1 d : ℝ)
noncomputable def S (n : ℕ) := n * a_1 + (n * (n - 1) / 2) * d

theorem maximize_S_n {n : ℕ} (h1 : S 17 > 0) (h2 : S 18 < 0) : n = 9 := sorry

end NUMINAMATH_GPT_maximize_S_n_l2031_203136


namespace NUMINAMATH_GPT_merchant_marked_price_l2031_203199

variable (L C M S : ℝ)

theorem merchant_marked_price :
  (C = 0.8 * L) → (C = 0.8 * S) → (S = 0.8 * M) → (M = 1.25 * L) :=
by
  sorry

end NUMINAMATH_GPT_merchant_marked_price_l2031_203199


namespace NUMINAMATH_GPT_sum_not_complete_residue_system_l2031_203172

theorem sum_not_complete_residue_system
  (n : ℕ) (hn : Even n)
  (a b : Fin n → Fin n)
  (ha : ∀ i : Fin n, ∃ j : Fin n, a j = i)
  (hb : ∀ i : Fin n, ∃ j : Fin n, b j = i) :
  ¬ (∀ k : Fin n, ∃ i : Fin n, a i + b i = k) :=
sorry

end NUMINAMATH_GPT_sum_not_complete_residue_system_l2031_203172


namespace NUMINAMATH_GPT_conditional_prob_correct_l2031_203148

/-- Define the events A and B as per the problem -/
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0

def event_B (x y : ℕ) : Prop := (x % 2 = 0 ∨ y % 2 = 0) ∧ x ≠ y

/-- Define the probability of event A -/
def prob_A : ℚ := 1 / 2

/-- Define the combined probability of both events A and B occurring -/
def prob_A_and_B : ℚ := 1 / 6

/-- Calculate the conditional probability P(B | A) -/
def conditional_prob : ℚ := prob_A_and_B / prob_A

theorem conditional_prob_correct : conditional_prob = 1 / 3 := by
  -- This is where you would provide the proof if required
  sorry

end NUMINAMATH_GPT_conditional_prob_correct_l2031_203148


namespace NUMINAMATH_GPT_minimum_p_l2031_203134

-- Define the problem constants and conditions
noncomputable def problem_statement :=
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧ 
    (∀ p' q' : ℕ, (0 < p' ∧ 0 < q' ∧ (2008 / 2009 < p' / (q' : ℚ)) ∧ (p' / (q' : ℚ) < 2009 / 2010)) → p ≤ p') 

-- The proof
theorem minimum_p (h : problem_statement) :
  ∃ p q : ℕ, 
    0 < p ∧ 0 < q ∧ 
    (2008 / 2009 < p / (q : ℚ)) ∧ (p / (q : ℚ) < 2009 / 2010) ∧
    p = 4017 :=
sorry

end NUMINAMATH_GPT_minimum_p_l2031_203134


namespace NUMINAMATH_GPT_tan_five_pi_over_four_l2031_203102

-- Define the question to prove
theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_l2031_203102


namespace NUMINAMATH_GPT_xiaoMing_better_performance_l2031_203184

-- Definitions based on conditions
def xiaoMing_scores : List Float := [90, 67, 90, 92, 96]
def xiaoLiang_scores : List Float := [87, 62, 90, 92, 92]

-- Definitions of average and variance calculation
def average (scores : List Float) : Float :=
  (scores.sum) / (scores.length.toFloat)

def variance (scores : List Float) : Float :=
  let avg := average scores
  (scores.map (λ x => (x - avg) ^ 2)).sum / (scores.length.toFloat)

-- Prove that Xiao Ming's performance is better than Xiao Liang's.
theorem xiaoMing_better_performance :
  average xiaoMing_scores > average xiaoLiang_scores ∧ variance xiaoMing_scores < variance xiaoLiang_scores :=
by
  sorry

end NUMINAMATH_GPT_xiaoMing_better_performance_l2031_203184


namespace NUMINAMATH_GPT_max_ratio_square_l2031_203105

variables {a b c x y : ℝ}
-- Assume a, b, c are positive real numbers
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
-- Assume the order of a, b, c: a ≥ b ≥ c
variable (h_order : a ≥ b ∧ b ≥ c)
-- Define the system of equations
variable (h_system : a^2 + y^2 = c^2 + x^2 ∧ c^2 + x^2 = (a - x)^2 + (c - y)^2)
-- Assume the constraints on x and y
variable (h_constraints : 0 ≤ x ∧ x < a ∧ 0 ≤ y ∧ y < c)

theorem max_ratio_square :
  ∃ (ρ : ℝ), ρ = (a / c) ∧ ρ^2 = 4 / 3 :=
sorry

end NUMINAMATH_GPT_max_ratio_square_l2031_203105


namespace NUMINAMATH_GPT_positive_difference_of_two_numbers_l2031_203152

theorem positive_difference_of_two_numbers :
  ∃ (x y : ℤ), (x + y = 40) ∧ (3 * y - 2 * x = 8) ∧ (|y - x| = 4) :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_two_numbers_l2031_203152


namespace NUMINAMATH_GPT_work_completion_l2031_203126

theorem work_completion (d : ℝ) :
  (9 * (1 / d) + 8 * (1 / 20) = 1) ↔ (d = 15) :=
by
  sorry

end NUMINAMATH_GPT_work_completion_l2031_203126


namespace NUMINAMATH_GPT_ratio_of_wealth_l2031_203150

theorem ratio_of_wealth (W P : ℝ) 
  (h1 : 0 < P) (h2 : 0 < W) 
  (pop_X : ℝ := 0.4 * P) 
  (wealth_X : ℝ := 0.6 * W) 
  (top50_pop_X : ℝ := 0.5 * pop_X) 
  (top50_wealth_X : ℝ := 0.8 * wealth_X) 
  (pop_Y : ℝ := 0.2 * P) 
  (wealth_Y : ℝ := 0.3 * W) 
  (avg_wealth_top50_X : ℝ := top50_wealth_X / top50_pop_X) 
  (avg_wealth_Y : ℝ := wealth_Y / pop_Y) : 
  avg_wealth_top50_X / avg_wealth_Y = 1.6 := 
by sorry

end NUMINAMATH_GPT_ratio_of_wealth_l2031_203150


namespace NUMINAMATH_GPT_integer_in_range_l2031_203171

theorem integer_in_range (x : ℤ) 
  (h1 : 0 < x) 
  (h2 : x < 7)
  (h3 : 0 < x)
  (h4 : x < 15)
  (h5 : -1 < x)
  (h6 : x < 5)
  (h7 : 0 < x)
  (h8 : x < 3)
  (h9 : x + 2 < 4) : x = 1 := 
sorry

end NUMINAMATH_GPT_integer_in_range_l2031_203171


namespace NUMINAMATH_GPT_oranges_in_each_box_l2031_203177

theorem oranges_in_each_box (total_oranges : ℝ) (boxes : ℝ) (h_total : total_oranges = 72) (h_boxes : boxes = 3.0) : total_oranges / boxes = 24 :=
by
  -- Begin proof
  sorry

end NUMINAMATH_GPT_oranges_in_each_box_l2031_203177


namespace NUMINAMATH_GPT_salt_solution_concentration_l2031_203168

theorem salt_solution_concentration :
  ∀ (C : ℝ),
  (∀ (mix_vol : ℝ) (pure_water : ℝ) (salt_solution_vol : ℝ),
    mix_vol = 1.5 →
    pure_water = 1 →
    salt_solution_vol = 0.5 →
    1.5 * 0.15 = 0.5 * (C / 100) →
    C = 45) :=
by
  intros C mix_vol pure_water salt_solution_vol h_mix h_pure h_salt h_eq
  sorry

end NUMINAMATH_GPT_salt_solution_concentration_l2031_203168


namespace NUMINAMATH_GPT_min_value_expression_l2031_203103

theorem min_value_expression (n : ℕ) (h : 0 < n) : 
  ∃ (m : ℕ), (m = n) ∧ (∀ k > 0, (k = n) -> (n / 3 + 27 / n) = 6) := 
sorry

end NUMINAMATH_GPT_min_value_expression_l2031_203103


namespace NUMINAMATH_GPT_hawks_total_points_l2031_203169

def touchdowns : ℕ := 3
def points_per_touchdown : ℕ := 7
def total_points (t : ℕ) (p : ℕ) : ℕ := t * p

theorem hawks_total_points : total_points touchdowns points_per_touchdown = 21 := 
by 
  sorry

end NUMINAMATH_GPT_hawks_total_points_l2031_203169


namespace NUMINAMATH_GPT_increasing_function_range_l2031_203176

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem increasing_function_range (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 ≤ a ∧ a < 3 :=
  sorry

end NUMINAMATH_GPT_increasing_function_range_l2031_203176


namespace NUMINAMATH_GPT_correct_operation_l2031_203135

theorem correct_operation (a b : ℝ) : (a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l2031_203135


namespace NUMINAMATH_GPT_smallest_b_for_q_ge_half_l2031_203137

open Nat

def binomial (n k : ℕ) : ℕ := if h : k ≤ n then n.choose k else 0

def q (b : ℕ) : ℚ := (binomial (32 - b) 2 + binomial (b - 1) 2) / (binomial 38 2 : ℕ)

theorem smallest_b_for_q_ge_half : ∃ (b : ℕ), b = 18 ∧ q b ≥ 1 / 2 :=
by
  -- Prove and find the smallest b such that q(b) ≥ 1/2
  sorry

end NUMINAMATH_GPT_smallest_b_for_q_ge_half_l2031_203137


namespace NUMINAMATH_GPT_age_difference_l2031_203122

variable (A B C : ℕ)

theorem age_difference (h₁ : C = A - 20) : (A + B) = (B + C) + 20 := 
sorry

end NUMINAMATH_GPT_age_difference_l2031_203122


namespace NUMINAMATH_GPT_price_of_cheaper_book_l2031_203111

theorem price_of_cheaper_book
    (total_cost : ℕ)
    (sets : ℕ)
    (price_more_expensive_book_increase : ℕ)
    (h1 : total_cost = 21000)
    (h2 : sets = 3)
    (h3 : price_more_expensive_book_increase = 300) :
  ∃ x : ℕ, 3 * ((x + (x + price_more_expensive_book_increase))) = total_cost ∧ x = 3350 :=
by
  sorry

end NUMINAMATH_GPT_price_of_cheaper_book_l2031_203111


namespace NUMINAMATH_GPT_speeding_tickets_l2031_203179

theorem speeding_tickets (p1 p2 : ℝ)
  (h1 : p1 = 16.666666666666664)
  (h2 : p2 = 40) :
  (p1 * (100 - p2) / 100 = 10) :=
by sorry

end NUMINAMATH_GPT_speeding_tickets_l2031_203179


namespace NUMINAMATH_GPT_ticket_sales_l2031_203141

-- Definitions of the conditions
theorem ticket_sales (adult_cost child_cost total_people child_count : ℕ)
  (h1 : adult_cost = 8)
  (h2 : child_cost = 1)
  (h3 : total_people = 22)
  (h4 : child_count = 18) :
  (child_count * child_cost + (total_people - child_count) * adult_cost = 50) := by
  sorry

end NUMINAMATH_GPT_ticket_sales_l2031_203141


namespace NUMINAMATH_GPT_find_hourly_rate_l2031_203131

-- Defining the conditions
def hours_worked : ℝ := 7.5
def overtime_factor : ℝ := 1.5
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 48

-- Proving the hourly rate
theorem find_hourly_rate (R : ℝ) (h : 7.5 * R + (10.5 - 7.5) * 1.5 * R = 48) : R = 4 := by
  sorry

end NUMINAMATH_GPT_find_hourly_rate_l2031_203131


namespace NUMINAMATH_GPT_Eunji_score_equals_56_l2031_203129

theorem Eunji_score_equals_56 (Minyoung_score Yuna_score : ℕ) (Eunji_score : ℕ) 
  (h1 : Minyoung_score = 55) (h2 : Yuna_score = 57)
  (h3 : Eunji_score > Minyoung_score) (h4 : Eunji_score < Yuna_score) : Eunji_score = 56 := by
  -- Given the hypothesis, it is a fact that Eunji's score is 56.
  sorry

end NUMINAMATH_GPT_Eunji_score_equals_56_l2031_203129


namespace NUMINAMATH_GPT_probability_rolls_more_ones_than_eights_l2031_203196

noncomputable def probability_more_ones_than_eights (n : ℕ) := 10246 / 32768

theorem probability_rolls_more_ones_than_eights :
  (probability_more_ones_than_eights 5) = 10246 / 32768 :=
by
  sorry

end NUMINAMATH_GPT_probability_rolls_more_ones_than_eights_l2031_203196


namespace NUMINAMATH_GPT_box_dimension_min_sum_l2031_203178

theorem box_dimension_min_sum :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 := by
  sorry

end NUMINAMATH_GPT_box_dimension_min_sum_l2031_203178


namespace NUMINAMATH_GPT_tickets_sold_correctly_l2031_203147

theorem tickets_sold_correctly :
  let total := 620
  let cost_per_ticket := 4
  let tickets_sold := 155
  total / cost_per_ticket = tickets_sold :=
by
  sorry

end NUMINAMATH_GPT_tickets_sold_correctly_l2031_203147


namespace NUMINAMATH_GPT_f_one_eq_zero_l2031_203164

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions for the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)

-- Goal: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_f_one_eq_zero_l2031_203164


namespace NUMINAMATH_GPT_shorter_leg_of_right_triangle_l2031_203100

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : c = 65) (h₃ : a ≤ b) : a = 25 :=
sorry

end NUMINAMATH_GPT_shorter_leg_of_right_triangle_l2031_203100


namespace NUMINAMATH_GPT_initial_lychees_count_l2031_203133

theorem initial_lychees_count (L : ℕ) (h1 : L / 2 = 2 * 100 * 5 / 5 * 5) : L = 500 :=
by sorry

end NUMINAMATH_GPT_initial_lychees_count_l2031_203133


namespace NUMINAMATH_GPT_dice_probability_l2031_203139

noncomputable def probability_each_number_appears_at_least_once : ℝ :=
  1 - (6 * (5/6)^10 - 15 * (4/6)^10 + 20 * (3/6)^10 - 15 * (2/6)^10 + 6 * (1/6)^10)

theorem dice_probability : probability_each_number_appears_at_least_once = 0.272 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_l2031_203139


namespace NUMINAMATH_GPT_sum_of_coefficients_l2031_203143

noncomputable def polynomial (x : ℝ) : ℝ := x^3 + 3*x^2 - 4*x - 12
noncomputable def simplified_polynomial (x : ℝ) (A B C : ℝ) : ℝ := A*x^2 + B*x + C

theorem sum_of_coefficients : 
  ∃ (A B C D : ℝ), 
    (∀ x ≠ D, simplified_polynomial x A B C = (polynomial x) / (x + 3)) ∧ 
    (A + B + C + D = -6) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2031_203143


namespace NUMINAMATH_GPT_housewife_spend_money_l2031_203175

theorem housewife_spend_money (P M: ℝ) (h1: 0.75 * P = 30) (h2: M / (0.75 * P) - M / P = 5) : 
  M = 600 :=
by
  sorry

end NUMINAMATH_GPT_housewife_spend_money_l2031_203175


namespace NUMINAMATH_GPT_calculate_expression_l2031_203157

theorem calculate_expression :
  36 + (150 / 15) + (12 ^ 2 * 5) - 300 - (270 / 9) = 436 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2031_203157


namespace NUMINAMATH_GPT_teorema_dos_bicos_white_gray_eq_angle_x_l2031_203121

-- Define the problem statement
theorem teorema_dos_bicos_white_gray_eq
    (n : ℕ)
    (AB CD : ℝ)
    (peaks : Fin n → ℝ)
    (white_angles gray_angles : Fin n → ℝ)
    (h_parallel : AB = CD)
    (h_white_angles : ∀ i, white_angles i = peaks i)
    (h_gray_angles : ∀ i, gray_angles i = peaks i):
    (Finset.univ.sum white_angles) = (Finset.univ.sum gray_angles) := sorry

theorem angle_x
    (AB CD : ℝ)
    (x : ℝ)
    (h_parallel : AB = CD):
    x = 32 := sorry

end NUMINAMATH_GPT_teorema_dos_bicos_white_gray_eq_angle_x_l2031_203121


namespace NUMINAMATH_GPT_find_a_b_l2031_203160

theorem find_a_b (a b : ℝ) (z : ℂ) (hz : z = 1 + Complex.I) 
  (h : (z^2 + a*z + b) / (z^2 - z + 1) = 1 - Complex.I) : a = -1 ∧ b = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l2031_203160


namespace NUMINAMATH_GPT_two_buttons_diff_size_color_l2031_203108

variables (box : Type) 
variable [Finite box]
variables (Big Small White Black : box → Prop)

axiom big_ex : ∃ x, Big x
axiom small_ex : ∃ x, Small x
axiom white_ex : ∃ x, White x
axiom black_ex : ∃ x, Black x
axiom size : ∀ x, Big x ∨ Small x
axiom color : ∀ x, White x ∨ Black x

theorem two_buttons_diff_size_color : 
  ∃ x y, x ≠ y ∧ (Big x ∧ Small y ∨ Small x ∧ Big y) ∧ (White x ∧ Black y ∨ Black x ∧ White y) := 
by
  sorry

end NUMINAMATH_GPT_two_buttons_diff_size_color_l2031_203108


namespace NUMINAMATH_GPT_find_x_when_y_is_20_l2031_203112

-- Definition of the problem conditions.
def constant_ratio (x y : ℝ) : Prop := ∃ k, (3 * x - 4) = k * (y + 7)

-- Main theorem statement.
theorem find_x_when_y_is_20 :
  (constant_ratio x 5 → constant_ratio 3 5) → 
  (constant_ratio x 20 → x = 5.0833) :=
  by sorry

end NUMINAMATH_GPT_find_x_when_y_is_20_l2031_203112


namespace NUMINAMATH_GPT_min_value_expression_l2031_203117

theorem min_value_expression (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : x * y * z = 1/2) :
  x^3 + 4 * x * y + 16 * y^3 + 8 * y * z + 3 * z^3 ≥ 18 :=
sorry

end NUMINAMATH_GPT_min_value_expression_l2031_203117


namespace NUMINAMATH_GPT_class_A_has_neater_scores_l2031_203197

-- Definitions for the given problem conditions
def mean_Class_A : ℝ := 120
def mean_Class_B : ℝ := 120
def variance_Class_A : ℝ := 42
def variance_Class_B : ℝ := 56

-- The theorem statement to prove Class A has neater scores
theorem class_A_has_neater_scores : (variance_Class_A < variance_Class_B) := by
  sorry

end NUMINAMATH_GPT_class_A_has_neater_scores_l2031_203197


namespace NUMINAMATH_GPT_gcd_A_C_gcd_B_C_l2031_203189

def A : ℕ := 177^5 + 30621 * 173^3 - 173^5
def B : ℕ := 173^5 + 30621 * 177^3 - 177^5
def C : ℕ := 173^4 + 30621^2 + 177^4

theorem gcd_A_C : Nat.gcd A C = 30637 := sorry

theorem gcd_B_C : Nat.gcd B C = 30637 := sorry

end NUMINAMATH_GPT_gcd_A_C_gcd_B_C_l2031_203189


namespace NUMINAMATH_GPT_sum_of_squares_of_diagonals_l2031_203188

variable (OP R : ℝ)

theorem sum_of_squares_of_diagonals (AC BD : ℝ) :
  AC^2 + BD^2 = 8 * R^2 - 4 * OP^2 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_diagonals_l2031_203188


namespace NUMINAMATH_GPT_count_correct_conclusions_l2031_203193

structure Point where
  x : ℝ
  y : ℝ

def isDoublingPoint (P Q : Point) : Prop :=
  2 * (P.x + Q.x) = P.y + Q.y

def P1 : Point := {x := 2, y := 0}

def Q1 : Point := {x := 2, y := 8}
def Q2 : Point := {x := -3, y := -2}

def onLine (P : Point) : Prop :=
  P.y = P.x + 2

def onParabola (P : Point) : Prop :=
  P.y = P.x ^ 2 - 2 * P.x - 3

def dist (P Q : Point) : ℝ :=
  (P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2

theorem count_correct_conclusions :
  (isDoublingPoint P1 Q1) ∧
  (isDoublingPoint P1 Q2) ∧
  (∃ A : Point, onLine A ∧ isDoublingPoint P1 A ∧ A = {x := -2, y := 0}) ∧
  (∃ B₁ B₂ : Point, onParabola B₁ ∧ onParabola B₂ ∧ isDoublingPoint P1 B₁ ∧ isDoublingPoint P1 B₂) ∧
  (∃ B : Point, isDoublingPoint P1 B ∧
   ∀ P : Point, isDoublingPoint P1 P → dist P1 P ≥ dist P1 B ∧
   dist P1 B = 8 * (5:ℝ)^(1/2) / 5) :=
by sorry

end NUMINAMATH_GPT_count_correct_conclusions_l2031_203193


namespace NUMINAMATH_GPT_min_diff_proof_l2031_203104

noncomputable def triangleMinDiff : ℕ :=
  let PQ := 666
  let QR := 667
  let PR := 2010 - PQ - QR
  if (PQ < QR ∧ QR < PR ∧ PQ + QR > PR ∧ PQ + PR > QR ∧ PR + QR > PQ) then QR - PQ else 0

theorem min_diff_proof :
  ∃ PQ QR PR : ℕ, PQ + QR + PR = 2010 ∧ PQ < QR ∧ QR < PR ∧ (PQ + QR > PR) ∧ (PQ + PR > QR) ∧ (PR + QR > PQ) ∧ (QR - PQ = triangleMinDiff) := sorry

end NUMINAMATH_GPT_min_diff_proof_l2031_203104


namespace NUMINAMATH_GPT_problem_I_l2031_203198

def f (x : ℝ) : ℝ := 4 - |x| - |x - 3|

theorem problem_I {x : ℝ} : f (x + 3 / 2) ≥ 0 ↔ -2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_I_l2031_203198


namespace NUMINAMATH_GPT_regular_octagon_angle_ABG_l2031_203162

-- Definition of a regular octagon
structure RegularOctagon (V : Type) :=
(vertices : Fin 8 → V)

def angleABG (O : RegularOctagon ℝ) : ℝ :=
  22.5

-- The statement: In a regular octagon ABCDEFGH, the measure of ∠ABG is 22.5°
theorem regular_octagon_angle_ABG (O : RegularOctagon ℝ) : angleABG O = 22.5 :=
  sorry

end NUMINAMATH_GPT_regular_octagon_angle_ABG_l2031_203162


namespace NUMINAMATH_GPT_selection_ways_l2031_203118

namespace CulturalPerformance

-- Define basic conditions
def num_students : ℕ := 6
def can_sing : ℕ := 3
def can_dance : ℕ := 2
def both_sing_and_dance : ℕ := 1

-- Define the proof statement
theorem selection_ways :
  ∃ (ways : ℕ), ways = 15 := by
  sorry

end CulturalPerformance

end NUMINAMATH_GPT_selection_ways_l2031_203118


namespace NUMINAMATH_GPT_Thomas_speed_greater_than_Jeremiah_l2031_203132

-- Define constants
def Thomas_passes_kilometers_per_hour := 5
def Jeremiah_passes_kilometers_per_hour := 6

-- Define speeds (in meters per hour)
def Thomas_speed := Thomas_passes_kilometers_per_hour * 1000
def Jeremiah_speed := Jeremiah_passes_kilometers_per_hour * 1000

-- Define hypothetical additional distances
def Thomas_hypothetical_additional_distance := 600 * 2
def Jeremiah_hypothetical_additional_distance := 50 * 2

-- Define effective distances traveled
def Thomas_effective_distance := Thomas_speed + Thomas_hypothetical_additional_distance
def Jeremiah_effective_distance := Jeremiah_speed + Jeremiah_hypothetical_additional_distance

-- Theorem to prove
theorem Thomas_speed_greater_than_Jeremiah : Thomas_effective_distance > Jeremiah_effective_distance := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_Thomas_speed_greater_than_Jeremiah_l2031_203132


namespace NUMINAMATH_GPT_f_leq_2x_l2031_203109

noncomputable def f : ℝ → ℝ := sorry
axiom f_nonneg {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : 0 ≤ f x
axiom f_one : f 1 = 1
axiom f_superadditive {x y : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hxy : x + y ≤ 1) : f (x + y) ≥ f x + f y

-- The theorem statement to be proved
theorem f_leq_2x {x : ℝ} (hx : 0 ≤ x ∧ x ≤ 1) : f x ≤ 2 * x := sorry

end NUMINAMATH_GPT_f_leq_2x_l2031_203109


namespace NUMINAMATH_GPT_complement_of_A_is_negatives_l2031_203124

theorem complement_of_A_is_negatives :
  let U := Set.univ (α := ℝ)
  let A := {x : ℝ | x ≥ 0}
  (U \ A) = {x : ℝ | x < 0} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_is_negatives_l2031_203124


namespace NUMINAMATH_GPT_find_a_of_min_value_of_f_l2031_203191

noncomputable def f (a x : ℝ) : ℝ := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) + 2 * a * Real.sin x + 4 * a * Real.cos x

theorem find_a_of_min_value_of_f :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≥ -6) ∧ (∃ x : ℝ, f a x = -6)) → (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_find_a_of_min_value_of_f_l2031_203191


namespace NUMINAMATH_GPT_tangent_line_at_1_l2031_203119

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the derivative f'
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

-- Define the point of tangency
def point_of_tangency : ℝ × ℝ := (1, f 1)

-- Define the slope of the tangent line at x=1
def slope_at_1 : ℝ := f' 1

-- Define the tangent line equation at x=1
def tangent_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

-- Theorem that the tangent line to f at x=1 is 2x - y + 1 = 0
theorem tangent_line_at_1 :
  tangent_line 1 (f 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_1_l2031_203119


namespace NUMINAMATH_GPT_li_bai_initial_wine_l2031_203138

theorem li_bai_initial_wine (x : ℕ) 
  (h : (((((x * 2 - 2) * 2 - 2) * 2 - 2) * 2 - 2) = 2)) : 
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_li_bai_initial_wine_l2031_203138


namespace NUMINAMATH_GPT_sum_PS_TV_l2031_203155

theorem sum_PS_TV 
  (P V : ℝ) 
  (hP : P = 3) 
  (hV : V = 33)
  (n : ℕ) 
  (hn : n = 6) 
  (Q R S T U : ℝ) 
  (hPR : P < Q ∧ Q < R ∧ R < S ∧ S < T ∧ T < U ∧ U < V)
  (h_divide : ∀ i : ℕ, i ≤ n → P + i * (V - P) / n = P + i * 5) :
  (P, V, Q, R, S, T, U) = (3, 33, 8, 13, 18, 23, 28) → (S - P) + (V - T) = 25 :=
by {
  sorry
}

end NUMINAMATH_GPT_sum_PS_TV_l2031_203155


namespace NUMINAMATH_GPT_gcd_1755_1242_l2031_203163

theorem gcd_1755_1242 : Nat.gcd 1755 1242 = 27 := 
by
  sorry

end NUMINAMATH_GPT_gcd_1755_1242_l2031_203163


namespace NUMINAMATH_GPT_value_of_Y_l2031_203120

-- Definitions for the conditions in part a)
def M := 2021 / 3
def N := M / 4
def Y := M + N

-- The theorem stating the question and its correct answer
theorem value_of_Y : Y = 843 := by
  sorry

end NUMINAMATH_GPT_value_of_Y_l2031_203120


namespace NUMINAMATH_GPT_cricket_overs_played_initially_l2031_203192

variables (x y : ℝ)

theorem cricket_overs_played_initially 
  (h1 : y = 3.2 * x)
  (h2 : 262 - y = 5.75 * 40) : 
  x = 10 := 
sorry

end NUMINAMATH_GPT_cricket_overs_played_initially_l2031_203192


namespace NUMINAMATH_GPT_arcsin_sqrt2_over_2_eq_pi_over_4_l2031_203149

theorem arcsin_sqrt2_over_2_eq_pi_over_4 :
  Real.arcsin (Real.sqrt 2 / 2) = Real.pi / 4 :=
sorry

end NUMINAMATH_GPT_arcsin_sqrt2_over_2_eq_pi_over_4_l2031_203149


namespace NUMINAMATH_GPT_example_proof_l2031_203166

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom axiom1 (x y : ℝ) : f (x - y) = f x * g y - g x * f y
axiom axiom2 (x : ℝ) : f x ≠ 0
axiom axiom3 : f 1 = f 2

theorem example_proof : g (-1) + g 1 = 1 := by
  sorry

end NUMINAMATH_GPT_example_proof_l2031_203166


namespace NUMINAMATH_GPT_max_gold_coins_l2031_203161

theorem max_gold_coins (n : ℤ) (h₁ : ∃ k : ℤ, n = 13 * k + 3) (h₂ : n < 150) : n ≤ 146 :=
by {
  sorry -- Proof not required as per instructions
}

end NUMINAMATH_GPT_max_gold_coins_l2031_203161


namespace NUMINAMATH_GPT_distance_between_P_and_F2_l2031_203185
open Real

theorem distance_between_P_and_F2 (x y c : ℝ) (h1 : c = sqrt 3)
    (h2 : x = -sqrt 3) (h3 : y = 1/2) : 
    sqrt ((sqrt 3 - x) ^ 2 + (0 - y) ^ 2) = 7 / 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_P_and_F2_l2031_203185


namespace NUMINAMATH_GPT_rectangle_side_difference_l2031_203128

theorem rectangle_side_difference (p d x y : ℝ) (h1 : 2 * x + 2 * y = p)
                                   (h2 : x^2 + y^2 = d^2)
                                   (h3 : x = 2 * y) :
    x - y = p / 6 := 
sorry

end NUMINAMATH_GPT_rectangle_side_difference_l2031_203128


namespace NUMINAMATH_GPT_solve_for_x_l2031_203153

def delta (x : ℝ) : ℝ := 5 * x + 9
def phi (x : ℝ) : ℝ := 7 * x + 6

theorem solve_for_x (x : ℝ) (h : delta (phi x) = -4) : x = -43 / 35 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2031_203153


namespace NUMINAMATH_GPT_total_amount_correct_l2031_203114

def num_2won_bills : ℕ := 8
def value_2won_bills : ℕ := 2
def num_1won_bills : ℕ := 2
def value_1won_bills : ℕ := 1

theorem total_amount_correct :
  (num_2won_bills * value_2won_bills) + (num_1won_bills * value_1won_bills) = 18 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_correct_l2031_203114


namespace NUMINAMATH_GPT_complex_number_equation_l2031_203182

theorem complex_number_equation
  (f : ℂ → ℂ)
  (z : ℂ)
  (h : f (i - z) = 2 * z - i) :
  (1 - i) * f (2 - i) = -1 + 7 * i := by
  sorry

end NUMINAMATH_GPT_complex_number_equation_l2031_203182


namespace NUMINAMATH_GPT_find_integer_k_l2031_203107

theorem find_integer_k (k : ℤ) : (∃ k : ℤ, (k = 6) ∨ (k = 2) ∨ (k = 0) ∨ (k = -4)) ↔ (∃ k : ℤ, (2 * k^2 + k - 8) % (k - 1) = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_integer_k_l2031_203107


namespace NUMINAMATH_GPT_units_digit_3968_805_l2031_203140

theorem units_digit_3968_805 : 
  (3968 ^ 805) % 10 = 8 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_units_digit_3968_805_l2031_203140


namespace NUMINAMATH_GPT_difference_thursday_tuesday_l2031_203173

-- Define the amounts given on each day
def amount_tuesday : ℕ := 8
def amount_wednesday : ℕ := 5 * amount_tuesday
def amount_thursday : ℕ := amount_wednesday + 9

-- Problem statement: prove that the difference between Thursday's and Tuesday's amount is $41
theorem difference_thursday_tuesday : amount_thursday - amount_tuesday = 41 := by
  sorry

end NUMINAMATH_GPT_difference_thursday_tuesday_l2031_203173


namespace NUMINAMATH_GPT_felicity_gas_usage_l2031_203187

variable (A F : ℕ)

theorem felicity_gas_usage
  (h1 : F = 4 * A - 5)
  (h2 : A + F = 30) :
  F = 23 := by
  sorry

end NUMINAMATH_GPT_felicity_gas_usage_l2031_203187


namespace NUMINAMATH_GPT_odd_function_f_l2031_203127

noncomputable def f : ℝ → ℝ
| x => if hx : x ≥ 0 then x * (1 - x) else x * (1 + x)

theorem odd_function_f {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = - f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x * (1 - x)) :
  ∀ x : ℝ, x ≤ 0 → f x = x * (1 + x) := by
  intro x hx
  sorry

end NUMINAMATH_GPT_odd_function_f_l2031_203127


namespace NUMINAMATH_GPT_triple_solutions_l2031_203158

theorem triple_solutions (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) ↔ a! + b! = 2 ^ c! :=
by
  sorry

end NUMINAMATH_GPT_triple_solutions_l2031_203158


namespace NUMINAMATH_GPT_abc_value_l2031_203180

theorem abc_value (a b c : ℝ) (h1 : ab = 30 * (4^(1/3))) (h2 : ac = 40 * (4^(1/3))) (h3 : bc = 24 * (4^(1/3))) :
  a * b * c = 120 :=
sorry

end NUMINAMATH_GPT_abc_value_l2031_203180


namespace NUMINAMATH_GPT_find_y_l2031_203144

open Real

variable {x y : ℝ}

theorem find_y (h1 : x * y = 25) (h2 : x / y = 36) (hx : 0 < x) (hy : 0 < y) :
  y = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l2031_203144


namespace NUMINAMATH_GPT_y_comparison_l2031_203116

theorem y_comparison :
  let y1 := (-1)^2 - 2*(-1) + 3
  let y2 := (-2)^2 - 2*(-2) + 3
  y2 > y1 := by
  sorry

end NUMINAMATH_GPT_y_comparison_l2031_203116


namespace NUMINAMATH_GPT_teams_same_matches_l2031_203165

theorem teams_same_matches (n : ℕ) (h : n = 30) : ∃ (i j : ℕ), i ≠ j ∧ ∀ (m : ℕ), m ≤ n - 1 → (some_number : ℕ) = (some_number : ℕ) :=
by {
  sorry
}

end NUMINAMATH_GPT_teams_same_matches_l2031_203165


namespace NUMINAMATH_GPT_rhombus_shorter_diagonal_l2031_203113

theorem rhombus_shorter_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 20) (h2 : area = 120) (h3 : area = (d1 * d2) / 2) : d1 = 12 :=
by 
  sorry

end NUMINAMATH_GPT_rhombus_shorter_diagonal_l2031_203113


namespace NUMINAMATH_GPT_fraction_classification_l2031_203110

theorem fraction_classification (x y : ℤ) :
  (∃ a b : ℤ, a/b = x/(x+1)) ∧ ¬(∃ a b : ℤ, a/b = x/2 + 1) ∧ ¬(∃ a b : ℤ, a/b = x/2) ∧ ¬(∃ a b : ℤ, a/b = xy/3) :=
by sorry

end NUMINAMATH_GPT_fraction_classification_l2031_203110


namespace NUMINAMATH_GPT_height_of_spruce_tree_l2031_203156

theorem height_of_spruce_tree (t : ℚ) (h1 : t = 25 / 64) :
  (∃ s : ℚ, s = 3 / (1 - t) ∧ s = 64 / 13) :=
by
  sorry

end NUMINAMATH_GPT_height_of_spruce_tree_l2031_203156


namespace NUMINAMATH_GPT_find_full_haired_dogs_l2031_203194

-- Definitions of the given conditions
def minutes_per_short_haired_dog : Nat := 10
def short_haired_dogs : Nat := 6
def total_time_minutes : Nat := 4 * 60
def twice_as_long (n : Nat) : Nat := 2 * n

-- Define the problem
def full_haired_dogs : Nat :=
  let short_haired_total_time := short_haired_dogs * minutes_per_short_haired_dog
  let remaining_time := total_time_minutes - short_haired_total_time
  remaining_time / (twice_as_long minutes_per_short_haired_dog)

-- Theorem statement
theorem find_full_haired_dogs : 
  full_haired_dogs = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_full_haired_dogs_l2031_203194


namespace NUMINAMATH_GPT_least_five_digit_congruent_6_mod_17_l2031_203154

theorem least_five_digit_congruent_6_mod_17 : ∃ n: ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 6 ∧ ∀ m: ℕ, 10000 ≤ m ∧ m < 100000 ∧ m % 17 = 6 → n ≤ m :=
sorry

end NUMINAMATH_GPT_least_five_digit_congruent_6_mod_17_l2031_203154


namespace NUMINAMATH_GPT_least_possible_value_of_p_and_q_l2031_203183

theorem least_possible_value_of_p_and_q 
  (p q : ℕ) 
  (h1 : p > 1) 
  (h2 : q > 1) 
  (h3 : 15 * (p + 1) = 29 * (q + 1)) : 
  p + q = 45 := 
sorry -- proof to be filled in

end NUMINAMATH_GPT_least_possible_value_of_p_and_q_l2031_203183


namespace NUMINAMATH_GPT_apples_in_basket_l2031_203125

theorem apples_in_basket (x : ℕ) (h1 : 22 * x = (x + 45) * 13) : 22 * x = 1430 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_basket_l2031_203125


namespace NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l2031_203142

variable (M W : ℕ)
noncomputable def M_initial := 45 - W
noncomputable def W_new := W + 9

theorem initial_ratio_of_milk_to_water :
  M_initial = 36 ∧ W = 9 →
  M_initial / (W + 9) = 2 ↔ 4 = M_initial / W := 
sorry

end NUMINAMATH_GPT_initial_ratio_of_milk_to_water_l2031_203142


namespace NUMINAMATH_GPT_distinct_flavors_count_l2031_203123

theorem distinct_flavors_count (red_candies : ℕ) (green_candies : ℕ)
  (h_red : red_candies = 0 ∨ red_candies = 1 ∨ red_candies = 2 ∨ red_candies = 3 ∨ red_candies = 4 ∨ red_candies = 5 ∨ red_candies = 6)
  (h_green : green_candies = 0 ∨ green_candies = 1 ∨ green_candies = 2 ∨ green_candies = 3 ∨ green_candies = 4 ∨ green_candies = 5) :
  ∃ unique_flavors : Finset (ℚ), unique_flavors.card = 25 :=
by
  sorry

end NUMINAMATH_GPT_distinct_flavors_count_l2031_203123


namespace NUMINAMATH_GPT_roots_of_quadratic_identity_l2031_203190

namespace RootProperties

theorem roots_of_quadratic_identity (a b : ℝ) 
(h1 : a^2 - 2*a - 1 = 0) 
(h2 : b^2 - 2*b - 1 = 0) 
(h3 : a ≠ b) 
: a^2 + b^2 = 6 := 
by sorry

end RootProperties

end NUMINAMATH_GPT_roots_of_quadratic_identity_l2031_203190


namespace NUMINAMATH_GPT_factorial_square_gt_power_l2031_203167

theorem factorial_square_gt_power {n : ℕ} (h : n > 2) : (n! * n!) > n^n :=
sorry

end NUMINAMATH_GPT_factorial_square_gt_power_l2031_203167


namespace NUMINAMATH_GPT_f_2_eq_4_l2031_203115

def f (n : ℕ) : ℕ := (List.range (n + 1)).sum + (List.range n).sum

theorem f_2_eq_4 : f 2 = 4 := by
  sorry

end NUMINAMATH_GPT_f_2_eq_4_l2031_203115


namespace NUMINAMATH_GPT_farmer_apples_after_giving_away_l2031_203195

def initial_apples : ℕ := 127
def given_away_apples : ℕ := 88
def remaining_apples : ℕ := 127 - 88

theorem farmer_apples_after_giving_away : remaining_apples = 39 := by
  sorry

end NUMINAMATH_GPT_farmer_apples_after_giving_away_l2031_203195


namespace NUMINAMATH_GPT_compute_Z_value_l2031_203186

def operation_Z (c d : ℕ) : ℤ := c^2 - 3 * c * d + d^2

theorem compute_Z_value : operation_Z 4 3 = -11 := by
  sorry

end NUMINAMATH_GPT_compute_Z_value_l2031_203186


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l2031_203151

theorem triangle_is_right_triangle
  (a b c : ℝ)
  (h : a^2 + b^2 + c^2 - 10 * a - 6 * b - 8 * c + 50 = 0) :
  a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2 ∨ c^2 = a^2 + b^2 :=
sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l2031_203151


namespace NUMINAMATH_GPT_trajectory_of_P_is_right_branch_of_hyperbola_l2031_203159

-- Definitions of the given points F1 and F2
def F1 : ℝ × ℝ := (-5, 0)
def F2 : ℝ × ℝ := (5, 0)

-- Definition of the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt

-- Definition of point P satisfying the condition
def P (x y : ℝ) : Prop :=
  abs (distance (x, y) F1 - distance (x, y) F2) = 8

-- Trajectory of point P is the right branch of the hyperbola
theorem trajectory_of_P_is_right_branch_of_hyperbola :
  ∀ (x y : ℝ), P x y → True := -- Trajectory is hyperbola (right branch)
by
  sorry

end NUMINAMATH_GPT_trajectory_of_P_is_right_branch_of_hyperbola_l2031_203159


namespace NUMINAMATH_GPT_hyperbola_focus_to_asymptote_distance_l2031_203106

theorem hyperbola_focus_to_asymptote_distance :
  ∀ (x y : ℝ), (x ^ 2 - y ^ 2 = 1) →
  ∃ c : ℝ, (c = 1) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_to_asymptote_distance_l2031_203106


namespace NUMINAMATH_GPT_melissa_total_commission_l2031_203145

def sale_price_coupe : ℝ := 30000
def sale_price_suv : ℝ := 2 * sale_price_coupe
def sale_price_luxury_sedan : ℝ := 80000

def commission_rate_coupe_and_suv : ℝ := 0.02
def commission_rate_luxury_sedan : ℝ := 0.03

def commission (rate : ℝ) (price : ℝ) : ℝ := rate * price

def total_commission : ℝ :=
  commission commission_rate_coupe_and_suv sale_price_coupe +
  commission commission_rate_coupe_and_suv sale_price_suv +
  commission commission_rate_luxury_sedan sale_price_luxury_sedan

theorem melissa_total_commission :
  total_commission = 4200 := by
  sorry

end NUMINAMATH_GPT_melissa_total_commission_l2031_203145


namespace NUMINAMATH_GPT_gain_percent_is_33_33_l2031_203101
noncomputable def gain_percent_calculation (C S : ℝ) := ((S - C) / C) * 100

theorem gain_percent_is_33_33
  (C S : ℝ)
  (h : 75 * C = 56.25 * S) :
  gain_percent_calculation C S = 33.33 := by
  sorry

end NUMINAMATH_GPT_gain_percent_is_33_33_l2031_203101


namespace NUMINAMATH_GPT_mixture_percent_chemical_a_l2031_203170

-- Defining the conditions
def solution_x : ℝ := 0.4
def solution_y : ℝ := 0.5
def percent_x_in_mixture : ℝ := 0.3
def percent_y_in_mixture : ℝ := 1.0 - percent_x_in_mixture

-- The goal is to prove that the mixture is 47% chemical a
theorem mixture_percent_chemical_a : (solution_x * percent_x_in_mixture + solution_y * percent_y_in_mixture) * 100 = 47 :=
by
  -- Calculation here
  sorry

end NUMINAMATH_GPT_mixture_percent_chemical_a_l2031_203170


namespace NUMINAMATH_GPT_cubics_of_sum_and_product_l2031_203181

theorem cubics_of_sum_and_product (x y : ℝ) (h1 : x + y = 10) (h2 : x * y = 12) : 
  x^3 + y^3 = 640 :=
by
  sorry

end NUMINAMATH_GPT_cubics_of_sum_and_product_l2031_203181


namespace NUMINAMATH_GPT_fred_baseball_cards_l2031_203146

variable (initial_cards : ℕ)
variable (bought_cards : ℕ)

theorem fred_baseball_cards (h1 : initial_cards = 5) (h2 : bought_cards = 3) : initial_cards - bought_cards = 2 := by
  sorry

end NUMINAMATH_GPT_fred_baseball_cards_l2031_203146


namespace NUMINAMATH_GPT_problem_l2031_203174

namespace MathProof

-- Definitions of A, B, and conditions
def A (x : ℤ) : Set ℤ := {0, |x|}
def B : Set ℤ := {1, 0, -1}

-- Prove x = ± 1 when A ⊆ B, 
-- A ∪ B = { -1, 0, 1 }, 
-- and complement of A in B is { -1 }
theorem problem (x : ℤ) (hx : A x ⊆ B) : 
  (x = 1 ∨ x = -1) ∧ 
  (A x ∪ B = {-1, 0, 1}) ∧ 
  (B \ (A x) = {-1}) := 
sorry 

end MathProof

end NUMINAMATH_GPT_problem_l2031_203174
