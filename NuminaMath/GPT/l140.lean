import Mathlib

namespace NUMINAMATH_GPT_harry_total_expenditure_l140_14083

theorem harry_total_expenditure :
  let pumpkin_price := 2.50
  let tomato_price := 1.50
  let chili_pepper_price := 0.90
  let pumpkin_packets := 3
  let tomato_packets := 4
  let chili_pepper_packets := 5
  (pumpkin_packets * pumpkin_price) + (tomato_packets * tomato_price) + (chili_pepper_packets * chili_pepper_price) = 18.00 :=
by
  sorry

end NUMINAMATH_GPT_harry_total_expenditure_l140_14083


namespace NUMINAMATH_GPT_find_star_1993_1935_l140_14034

axiom star (x y : ℕ) : ℕ
axiom star_idempotent (x : ℕ) : star x x = 0
axiom star_assoc (x y z : ℕ) : star x (star y z) = star x y + z

theorem find_star_1993_1935 : star 1993 1935 = 58 :=
by
  sorry

end NUMINAMATH_GPT_find_star_1993_1935_l140_14034


namespace NUMINAMATH_GPT_hannah_games_l140_14055

theorem hannah_games (total_points : ℕ) (avg_points_per_game : ℕ) (h1 : total_points = 312) (h2 : avg_points_per_game = 13) :
  total_points / avg_points_per_game = 24 :=
sorry

end NUMINAMATH_GPT_hannah_games_l140_14055


namespace NUMINAMATH_GPT_rancher_problem_l140_14020

theorem rancher_problem (s c : ℕ) (h : 30 * s + 35 * c = 1500) : (s = 1 ∧ c = 42) ∨ (s = 36 ∧ c = 12) := 
by
  sorry

end NUMINAMATH_GPT_rancher_problem_l140_14020


namespace NUMINAMATH_GPT_parity_of_E2021_E2022_E2023_l140_14057

def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_even (n : ℕ) : Prop := n % 2 = 0

def seq (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 0
  else seq (n - 2) + seq (n - 3)

theorem parity_of_E2021_E2022_E2023 :
  is_odd (seq 2021) ∧ is_even (seq 2022) ∧ is_odd (seq 2023) :=
by
  sorry

end NUMINAMATH_GPT_parity_of_E2021_E2022_E2023_l140_14057


namespace NUMINAMATH_GPT_probability_inside_octahedron_l140_14023

noncomputable def probability_of_octahedron : ℝ := 
  let cube_volume := 8
  let octahedron_volume := 4 / 3
  octahedron_volume / cube_volume

theorem probability_inside_octahedron :
  probability_of_octahedron = 1 / 6 :=
  by
    sorry

end NUMINAMATH_GPT_probability_inside_octahedron_l140_14023


namespace NUMINAMATH_GPT_find_m_l140_14032

variable (m x1 x2 : ℝ)

def quadratic_eqn (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - m * x + 2 * m - 1 = 0

def roots_condition (m x1 x2 : ℝ) : Prop :=
  x1^2 + x2^2 = 23 ∧
  x1 + x2 = m ∧
  x1 * x2 = 2 * m - 1

theorem find_m (m x1 x2 : ℝ) : 
  quadratic_eqn m → 
  roots_condition m x1 x2 → 
  m = -3 :=
by
  intro hQ hR
  sorry

end NUMINAMATH_GPT_find_m_l140_14032


namespace NUMINAMATH_GPT_equilateral_triangle_vertex_distance_l140_14093

noncomputable def distance_vertex_to_center (l r : ℝ) : ℝ :=
  Real.sqrt (r^2 + (l^2 / 4))

theorem equilateral_triangle_vertex_distance
  (l r : ℝ)
  (h1 : l > 0)
  (h2 : r > 0) :
  distance_vertex_to_center l r = Real.sqrt (r^2 + (l^2 / 4)) :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_vertex_distance_l140_14093


namespace NUMINAMATH_GPT_henry_initial_money_l140_14068

variable (x : ℤ)

theorem henry_initial_money : (x + 18 - 10 = 19) → x = 11 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_henry_initial_money_l140_14068


namespace NUMINAMATH_GPT_dean_marathon_time_l140_14024

/-- 
Micah runs 2/3 times as fast as Dean, and it takes Jake 1/3 times more time to finish the marathon
than it takes Micah. The total time the three take to complete the marathon is 23 hours.
Prove that the time it takes Dean to finish the marathon is approximately 7.67 hours.
-/
theorem dean_marathon_time (D M J : ℝ)
  (h1 : M = D * (3 / 2))
  (h2 : J = M + (1 / 3) * M)
  (h3 : D + M + J = 23) : 
  D = 23 / 3 :=
by
  sorry

end NUMINAMATH_GPT_dean_marathon_time_l140_14024


namespace NUMINAMATH_GPT_cost_price_one_metre_l140_14031

noncomputable def selling_price : ℤ := 18000
noncomputable def total_metres : ℕ := 600
noncomputable def loss_per_metre : ℤ := 5

noncomputable def total_loss : ℤ := loss_per_metre * (total_metres : ℤ) -- Note the cast to ℤ for multiplication
noncomputable def cost_price : ℤ := selling_price + total_loss
noncomputable def cost_price_per_metre : ℚ := cost_price / (total_metres : ℤ)

theorem cost_price_one_metre : cost_price_per_metre = 35 := by
  sorry

end NUMINAMATH_GPT_cost_price_one_metre_l140_14031


namespace NUMINAMATH_GPT_intersection_points_l140_14035

theorem intersection_points (a : ℝ) (h : 2 < a) :
  (∃ n : ℕ, (n = 1 ∨ n = 2) ∧ (∃ x1 x2 : ℝ, y = (a-3)*x^2 - x - 1/4 ∧ x1 ≠ x2)) :=
sorry

end NUMINAMATH_GPT_intersection_points_l140_14035


namespace NUMINAMATH_GPT_Debby_drinks_five_bottles_per_day_l140_14045

theorem Debby_drinks_five_bottles_per_day (total_bottles : ℕ) (days : ℕ) (h1 : total_bottles = 355) (h2 : days = 71) : (total_bottles / days) = 5 :=
by 
  sorry

end NUMINAMATH_GPT_Debby_drinks_five_bottles_per_day_l140_14045


namespace NUMINAMATH_GPT_sandcastle_height_difference_l140_14097

theorem sandcastle_height_difference :
  let Miki_height := 0.8333333333333334
  let Sister_height := 0.5
  Miki_height - Sister_height = 0.3333333333333334 :=
by
  sorry

end NUMINAMATH_GPT_sandcastle_height_difference_l140_14097


namespace NUMINAMATH_GPT_y_is_less_than_x_by_9444_percent_l140_14094

theorem y_is_less_than_x_by_9444_percent (x y : ℝ) (h : x = 18 * y) : (x - y) / x * 100 = 94.44 :=
by
  sorry

end NUMINAMATH_GPT_y_is_less_than_x_by_9444_percent_l140_14094


namespace NUMINAMATH_GPT_value_of_a_minus_n_plus_k_l140_14078

theorem value_of_a_minus_n_plus_k :
  ∃ (a k n : ℤ), 
    (∀ x : ℤ, (3 * x + 2) * (2 * x - 7) = a * x^2 + k * x + n) ∧ 
    (a - n + k = 3) :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_n_plus_k_l140_14078


namespace NUMINAMATH_GPT_product_increase_l140_14021

variable (x : ℤ)

theorem product_increase (h : 53 * x = 1585) : 1585 - (35 * x) = 535 :=
by sorry

end NUMINAMATH_GPT_product_increase_l140_14021


namespace NUMINAMATH_GPT_cinema_meeting_day_l140_14009

-- Define the cycles for Kolya, Seryozha, and Vanya.
def kolya_cycle : ℕ := 4
def seryozha_cycle : ℕ := 5
def vanya_cycle : ℕ := 6

-- The problem statement requiring proof.
theorem cinema_meeting_day : ∃ n : ℕ, n > 0 ∧ n % kolya_cycle = 0 ∧ n % seryozha_cycle = 0 ∧ n % vanya_cycle = 0 ∧ n = 60 := 
  sorry

end NUMINAMATH_GPT_cinema_meeting_day_l140_14009


namespace NUMINAMATH_GPT_find_AX_l140_14081

theorem find_AX (AC BC BX : ℝ) (h1 : AC = 27) (h2 : BC = 40) (h3 : BX = 36)
    (h4 : ∀ (AX : ℝ), AX = AC * BX / BC) : 
    ∃ AX, AX = 243 / 10 :=
by
  sorry

end NUMINAMATH_GPT_find_AX_l140_14081


namespace NUMINAMATH_GPT_odd_function_strictly_decreasing_l140_14066

noncomputable def f (x : ℝ) : ℝ := sorry

axiom additivity (x y : ℝ) : f (x + y) = f x + f y
axiom negative_condition (x : ℝ) (hx : x > 0) : f x < 0

theorem odd_function : ∀ x : ℝ, f (-x) = -f x :=
by sorry

theorem strictly_decreasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ > f x₂ :=
by sorry

end NUMINAMATH_GPT_odd_function_strictly_decreasing_l140_14066


namespace NUMINAMATH_GPT_relative_positions_of_P_on_AB_l140_14064

theorem relative_positions_of_P_on_AB (A B P : ℝ) : 
  A ≤ B → (A ≤ P ∧ P ≤ B ∨ P = A ∨ P = B ∨ P < A ∨ P > B) :=
by
  intro hAB
  sorry

end NUMINAMATH_GPT_relative_positions_of_P_on_AB_l140_14064


namespace NUMINAMATH_GPT_total_cost_of_selling_watermelons_l140_14062

-- Definitions of the conditions:
def watermelon_weight : ℝ := 23.0
def daily_prices : List ℝ := [2.10, 1.90, 1.80, 2.30, 2.00, 1.95, 2.20]
def discount_threshold : ℕ := 15
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05
def number_of_watermelons : ℕ := 18

-- The theorem statement:
theorem total_cost_of_selling_watermelons :
  let average_price := (daily_prices.sum / daily_prices.length)
  let total_weight := number_of_watermelons * watermelon_weight
  let initial_cost := total_weight * average_price
  let discounted_cost := if number_of_watermelons > discount_threshold then initial_cost * (1 - discount_rate) else initial_cost
  let final_cost := discounted_cost * (1 + sales_tax_rate)
  final_cost = 796.43 := by
    sorry

end NUMINAMATH_GPT_total_cost_of_selling_watermelons_l140_14062


namespace NUMINAMATH_GPT_phi_eq_pi_div_two_l140_14030

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := Real.cos (x + ϕ)

theorem phi_eq_pi_div_two (ϕ : ℝ) (h1 : 0 ≤ ϕ) (h2 : ϕ ≤ π)
  (h3 : ∀ x : ℝ, f x ϕ = -f (-x) ϕ) : ϕ = π / 2 :=
sorry

end NUMINAMATH_GPT_phi_eq_pi_div_two_l140_14030


namespace NUMINAMATH_GPT_students_left_is_6_l140_14058

-- Start of the year students
def initial_students : ℕ := 11

-- New students arrived during the year
def new_students : ℕ := 42

-- Students at the end of the year
def final_students : ℕ := 47

-- Definition to calculate the number of students who left
def students_left (initial new final : ℕ) : ℕ := (initial + new) - final

-- Statement to prove
theorem students_left_is_6 : students_left initial_students new_students final_students = 6 :=
by
  -- We skip the proof using sorry
  sorry

end NUMINAMATH_GPT_students_left_is_6_l140_14058


namespace NUMINAMATH_GPT_not_distributive_add_mul_l140_14085

-- Definition of the addition operation on pairs of real numbers
def pair_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst + b.fst, a.snd + b.snd)

-- Definition of the multiplication operation on pairs of real numbers
def pair_mul (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.fst * b.fst - a.snd * b.snd, a.fst * b.snd + a.snd * b.fst)

-- The problem statement: distributive law of addition over multiplication does not hold
theorem not_distributive_add_mul (a b c : ℝ × ℝ) :
  pair_add a (pair_mul b c) ≠ pair_mul (pair_add a b) (pair_add a c) :=
sorry

end NUMINAMATH_GPT_not_distributive_add_mul_l140_14085


namespace NUMINAMATH_GPT_satisfies_differential_equation_l140_14004

noncomputable def y (x : ℝ) : ℝ := (Real.sin x) / x

theorem satisfies_differential_equation (x : ℝ) (hx : x ≠ 0) : 
  x * (deriv (fun x => (Real.sin x) / x) x) + (Real.sin x) / x = Real.cos x := 
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_satisfies_differential_equation_l140_14004


namespace NUMINAMATH_GPT_eq_of_frac_sub_l140_14098

theorem eq_of_frac_sub (x : ℝ) (hx : x ≠ 1) : 
  (2 / (x^2 - 1) - 1 / (x - 1)) = - (1 / (x + 1)) := 
by sorry

end NUMINAMATH_GPT_eq_of_frac_sub_l140_14098


namespace NUMINAMATH_GPT_tangent_line_equation_is_correct_l140_14033

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

theorem tangent_line_equation_is_correct :
  let p : ℝ × ℝ := (0, 1)
  let f' := fun x => x * Real.exp x + Real.exp x
  let slope := f' 0
  let tangent_line := fun x y => slope * (x - p.1) - (y - p.2)
  tangent_line = (fun x y => x - y + 1) :=
by
  intros
  sorry

end NUMINAMATH_GPT_tangent_line_equation_is_correct_l140_14033


namespace NUMINAMATH_GPT_find_number_l140_14029

theorem find_number (x : ℝ) (h : x / 0.025 = 40) : x = 1 := 
by sorry

end NUMINAMATH_GPT_find_number_l140_14029


namespace NUMINAMATH_GPT_find_angle_A_l140_14048

theorem find_angle_A (BC AC : ℝ) (B : ℝ) (A : ℝ) (h_cond : BC = Real.sqrt 3 ∧ AC = 1 ∧ B = Real.pi / 6) :
  A = Real.pi / 3 ∨ A = 2 * Real.pi / 3 :=
sorry

end NUMINAMATH_GPT_find_angle_A_l140_14048


namespace NUMINAMATH_GPT_number_of_boys_l140_14010

theorem number_of_boys (x g : ℕ) 
  (h1 : x + g = 150) 
  (h2 : g = (x * 150) / 100) 
  : x = 60 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_boys_l140_14010


namespace NUMINAMATH_GPT_Fr_zero_for_all_r_l140_14095

noncomputable def F (r : ℕ) (x y z A B C : ℝ) : ℝ :=
  x^r * Real.sin (r * A) + y^r * Real.sin (r * B) + z^r * Real.sin (r * C)

theorem Fr_zero_for_all_r
  (x y z A B C : ℝ)
  (h_sum : ∃ k : ℤ, A + B + C = k * Real.pi)
  (hF1 : F 1 x y z A B C = 0)
  (hF2 : F 2 x y z A B C = 0)
  : ∀ r : ℕ, F r x y z A B C = 0 :=
sorry

end NUMINAMATH_GPT_Fr_zero_for_all_r_l140_14095


namespace NUMINAMATH_GPT_general_term_formula_l140_14090

def sequence_sums (n : ℕ) : ℕ := 2 * n^2 + n

theorem general_term_formula (a : ℕ → ℕ) (S : ℕ → ℕ) (hS : S = sequence_sums) :
  (∀ n, a n = S n - S (n-1)) → ∀ n, a n = 4 * n - 1 :=
by
  sorry

end NUMINAMATH_GPT_general_term_formula_l140_14090


namespace NUMINAMATH_GPT_tiger_catch_distance_correct_l140_14075

noncomputable def tiger_catch_distance (tiger_leaps_behind : ℕ) (tiger_leaps_per_minute : ℕ) (deer_leaps_per_minute : ℕ) (tiger_m_per_leap : ℕ) (deer_m_per_leap : ℕ) : ℕ :=
  let initial_distance := tiger_leaps_behind * tiger_m_per_leap
  let tiger_per_minute := tiger_leaps_per_minute * tiger_m_per_leap
  let deer_per_minute := deer_leaps_per_minute * deer_m_per_leap
  let gain_per_minute := tiger_per_minute - deer_per_minute
  let time_to_catch := initial_distance / gain_per_minute
  time_to_catch * tiger_per_minute

theorem tiger_catch_distance_correct :
  tiger_catch_distance 50 5 4 8 5 = 800 :=
by
  -- This is the placeholder for the proof.
  sorry

end NUMINAMATH_GPT_tiger_catch_distance_correct_l140_14075


namespace NUMINAMATH_GPT_coconut_grove_yield_l140_14071

theorem coconut_grove_yield (x : ℕ)
  (h1 : ∀ y, y = x + 3 → 60 * y = 60 * (x + 3))
  (h2 : ∀ z, z = x → 120 * z = 120 * x)
  (h3 : ∀ w, w = x - 3 → 180 * w = 180 * (x - 3))
  (avg_yield : 100 = 100)
  (total_trees : 3 * x = (x + 3) + x + (x - 3)) :
  60 * (x + 3) + 120 * x + 180 * (x - 3) = 300 * x →
  x = 6 :=
by
  sorry

end NUMINAMATH_GPT_coconut_grove_yield_l140_14071


namespace NUMINAMATH_GPT_sufficient_condition_l140_14037

theorem sufficient_condition (a b : ℝ) (h1 : a > 1) (h2 : b > 1) : ab > 1 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_l140_14037


namespace NUMINAMATH_GPT_main_theorem_l140_14015

-- Define even functions
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define odd functions
def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g x = -g (-x)

-- Given conditions
variable (f g : ℝ → ℝ)
variable (h1 : is_even_function f)
variable (h2 : is_odd_function g)
variable (h3 : ∀ x, g x = f (x - 1))

-- Theorem to prove
theorem main_theorem : f 2017 + f 2019 = 0 := sorry

end NUMINAMATH_GPT_main_theorem_l140_14015


namespace NUMINAMATH_GPT_num_right_triangles_with_incenter_origin_l140_14008

theorem num_right_triangles_with_incenter_origin (p : ℕ) (hp : Nat.Prime p) :
  let M : ℤ × ℤ := (p * 1994, 7 * p * 1994)
  let is_lattice_point (x : ℤ × ℤ) : Prop := True  -- All points considered are lattice points
  let is_right_angle_vertex (M : ℤ × ℤ) : Prop := True
  let is_incenter_origin (M : ℤ × ℤ) : Prop := True
  let num_triangles (p : ℕ) : ℕ :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  num_triangles p = if p = 2 then 18 else if p = 997 then 20 else 36 := (

  by sorry

 )

end NUMINAMATH_GPT_num_right_triangles_with_incenter_origin_l140_14008


namespace NUMINAMATH_GPT_find_angle_B_find_max_k_l140_14060

theorem find_angle_B
(A B C a b c : ℝ)
(h_angles : A + B + C = Real.pi)
(h_sides : (2 * a - c) * Real.cos B = b * Real.cos C)
(h_A_pos : 0 < A) (h_B_pos : 0 < B) (h_C_pos : 0 < C) 
(h_Alt_pos : A < Real.pi) (h_Blt_pos : B < Real.pi) 
(h_Clt_pos : C < Real.pi) :
B = Real.pi / 3 := 
sorry

theorem find_max_k
(A : ℝ)
(k : ℝ)
(m : ℝ × ℝ := (Real.sin A, Real.cos (2 * A)))
(n : ℝ × ℝ := (4 * k, 1))
(h_k_cond : 1 < k)
(h_max_dot : (m.1) * (n.1) + (m.2) * (n.2) = 5) :
k = 3 / 2 :=
sorry

end NUMINAMATH_GPT_find_angle_B_find_max_k_l140_14060


namespace NUMINAMATH_GPT_solveForN_l140_14018

-- Define the condition that sqrt(8 + n) = 9
def condition (n : ℝ) : Prop := Real.sqrt (8 + n) = 9

-- State the main theorem that given the condition, n must be 73
theorem solveForN (n : ℝ) (h : condition n) : n = 73 := by
  sorry

end NUMINAMATH_GPT_solveForN_l140_14018


namespace NUMINAMATH_GPT_rebate_percentage_l140_14069

theorem rebate_percentage (r : ℝ) (h1 : 0 ≤ r) (h2 : r ≤ 1) 
(h3 : (6650 - 6650 * r) * 1.10 = 6876.1) : r = 0.06 :=
sorry

end NUMINAMATH_GPT_rebate_percentage_l140_14069


namespace NUMINAMATH_GPT_pq_work_together_in_10_days_l140_14007

theorem pq_work_together_in_10_days 
  (p q r : ℝ)
  (hq : 1/q = 1/28)
  (hr : 1/r = 1/35)
  (hp : 1/p = 1/q + 1/r) :
  1/p + 1/q = 1/10 :=
by sorry

end NUMINAMATH_GPT_pq_work_together_in_10_days_l140_14007


namespace NUMINAMATH_GPT_games_played_l140_14036

theorem games_played (x : ℕ) (h1 : x * 26 + 42 * (20 - x) = 600) : x = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_games_played_l140_14036


namespace NUMINAMATH_GPT_time_to_fill_bucket_l140_14096

theorem time_to_fill_bucket (t : ℝ) (h : 2/3 = 2 / t) : t = 3 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_bucket_l140_14096


namespace NUMINAMATH_GPT_marks_deducted_per_wrong_answer_l140_14040

theorem marks_deducted_per_wrong_answer
  (correct_awarded : ℕ)
  (total_marks : ℕ)
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (final_marks : ℕ) :
  correct_awarded = 3 →
  total_marks = 38 →
  total_questions = 70 →
  correct_answers = 27 →
  incorrect_answers = total_questions - correct_answers →
  final_marks = total_marks →
  final_marks = correct_answers * correct_awarded - incorrect_answers * 1 →
  1 = 1
  := by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_marks_deducted_per_wrong_answer_l140_14040


namespace NUMINAMATH_GPT_fraction_combination_l140_14038

theorem fraction_combination (x y : ℝ) (h : y / x = 3 / 4) : (x + y) / x = 7 / 4 :=
by
  -- Proof steps will be inserted here (for now using sorry)
  sorry

end NUMINAMATH_GPT_fraction_combination_l140_14038


namespace NUMINAMATH_GPT_value_of_J_l140_14086

-- Given conditions
variables (Y J : ℤ)

-- Condition definitions
axiom condition1 : 150 < Y ∧ Y < 300
axiom condition2 : Y = J^2 * J^3
axiom condition3 : ∃ n : ℤ, Y = n^3

-- Goal: Value of J
theorem value_of_J : J = 3 :=
by { sorry }  -- Proof omitted

end NUMINAMATH_GPT_value_of_J_l140_14086


namespace NUMINAMATH_GPT_find_actual_price_of_good_l140_14079

theorem find_actual_price_of_good (P : ℝ) (price_after_discounts : P * 0.93 * 0.90 * 0.85 * 0.75 = 6600) :
  P = 11118.75 :=
by
  sorry

end NUMINAMATH_GPT_find_actual_price_of_good_l140_14079


namespace NUMINAMATH_GPT_max_plus_shapes_l140_14063

def cover_square (x y : ℕ) : Prop :=
  3 * x + 5 * y = 49

theorem max_plus_shapes (x y : ℕ) (h1 : cover_square x y) (h2 : x ≥ 4) : y ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_plus_shapes_l140_14063


namespace NUMINAMATH_GPT_sin_zero_necessary_not_sufficient_l140_14039

theorem sin_zero_necessary_not_sufficient:
  (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * Real.pi) → (Real.sin α = 0)) ∧
  ¬ (∀ α : ℝ, (Real.sin α = 0) → (∃ k : ℤ, α = 2 * k * Real.pi)) :=
by
  sorry

end NUMINAMATH_GPT_sin_zero_necessary_not_sufficient_l140_14039


namespace NUMINAMATH_GPT_sphere_diameter_l140_14065

theorem sphere_diameter (r : ℝ) (V : ℝ) (threeV : ℝ) (a b : ℕ) :
  (∀ (r : ℝ), r = 5 →
  V = (4 / 3) * π * r^3 →
  threeV = 3 * V →
  D = 2 * (3 * V * 3 / (4 * π))^(1 / 3) →
  D = a * b^(1 / 3) →
  a = 10 ∧ b = 3) →
  a + b = 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sphere_diameter_l140_14065


namespace NUMINAMATH_GPT_five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l140_14026

-- Definition: Number of ways to arrange n items in a row
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Question (1)
theorem five_students_in_a_row : factorial 5 = 120 :=
by sorry

-- Question (2) - Rather than performing combinatorial steps directly, we'll assume a function to calculate the specific arrangement
def specific_arrangement (students: ℕ) : ℕ :=
  if students = 5 then 24 else 0

theorem five_students_with_constraints : specific_arrangement 5 = 24 :=
by sorry

-- Question (3) - Number of ways to divide n students into k classes with at least one student in each class
def number_of_ways_to_divide (students: ℕ) (classes: ℕ) : ℕ :=
  if students = 5 ∧ classes = 3 then 150 else 0

theorem five_students_into_three_classes : number_of_ways_to_divide 5 3 = 150 :=
by sorry

end NUMINAMATH_GPT_five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l140_14026


namespace NUMINAMATH_GPT_cost_per_ice_cream_l140_14027

theorem cost_per_ice_cream (chapati_count : ℕ)
                           (rice_plate_count : ℕ)
                           (mixed_vegetable_plate_count : ℕ)
                           (ice_cream_cup_count : ℕ)
                           (cost_per_chapati : ℕ)
                           (cost_per_rice_plate : ℕ)
                           (cost_per_mixed_vegetable : ℕ)
                           (amount_paid : ℕ)
                           (total_cost_chapatis : ℕ)
                           (total_cost_rice : ℕ)
                           (total_cost_mixed_vegetable : ℕ)
                           (total_non_ice_cream_cost : ℕ)
                           (total_ice_cream_cost : ℕ)
                           (cost_per_ice_cream_cup : ℕ) :
    chapati_count = 16 →
    rice_plate_count = 5 →
    mixed_vegetable_plate_count = 7 →
    ice_cream_cup_count = 6 →
    cost_per_chapati = 6 →
    cost_per_rice_plate = 45 →
    cost_per_mixed_vegetable = 70 →
    amount_paid = 961 →
    total_cost_chapatis = chapati_count * cost_per_chapati →
    total_cost_rice = rice_plate_count * cost_per_rice_plate →
    total_cost_mixed_vegetable = mixed_vegetable_plate_count * cost_per_mixed_vegetable →
    total_non_ice_cream_cost = total_cost_chapatis + total_cost_rice + total_cost_mixed_vegetable →
    total_ice_cream_cost = amount_paid - total_non_ice_cream_cost →
    cost_per_ice_cream_cup = total_ice_cream_cost / ice_cream_cup_count →
    cost_per_ice_cream_cup = 25 :=
by
    intros; sorry

end NUMINAMATH_GPT_cost_per_ice_cream_l140_14027


namespace NUMINAMATH_GPT_increase_in_length_and_breadth_is_4_l140_14084

-- Define the variables for the original length and breadth of the room
variables (L B x : ℕ)

-- Define the original perimeter
def P_original : ℕ := 2 * (L + B)

-- Define the new perimeter after the increase
def P_new : ℕ := 2 * ((L + x) + (B + x))

-- Define the condition that the perimeter increases by 16 feet
axiom increase_perimeter : P_new L B x - P_original L B = 16

-- State the theorem that \(x = 4\)
theorem increase_in_length_and_breadth_is_4 : x = 4 :=
by
  -- Proof would be filled in here using the axioms and definitions
  sorry

end NUMINAMATH_GPT_increase_in_length_and_breadth_is_4_l140_14084


namespace NUMINAMATH_GPT_sin_inequality_l140_14080

theorem sin_inequality (d n : ℤ) (hd : d ≥ 1) (hnsq : ∀ k : ℤ, k * k ≠ d) (hn : n ≥ 1) :
  (n * Real.sqrt d + 1) * |Real.sin (n * Real.pi * Real.sqrt d)| ≥ 1 := by
  sorry

end NUMINAMATH_GPT_sin_inequality_l140_14080


namespace NUMINAMATH_GPT_goat_age_l140_14006

theorem goat_age : 26 + 42 = 68 := 
by 
  -- Since we only need the statement,
  -- we add sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_goat_age_l140_14006


namespace NUMINAMATH_GPT_non_negative_real_sum_expressions_l140_14001

theorem non_negative_real_sum_expressions (x y z : ℝ) (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) (h_sum : x + y + z = 1) :
  0 ≤ x * y + y * z + z * x - 2 * x * y * z ∧ x * y + y * z + z * x - 2 * x * y * z ≤ 7 / 27 :=
by
  sorry

end NUMINAMATH_GPT_non_negative_real_sum_expressions_l140_14001


namespace NUMINAMATH_GPT_swimming_both_days_l140_14089

theorem swimming_both_days
  (total_students swimming_today soccer_today : ℕ)
  (students_swimming_yesterday students_soccer_yesterday : ℕ)
  (soccer_today_swimming_yesterday soccer_today_soccer_yesterday : ℕ)
  (swimming_today_swimming_yesterday swimming_today_soccer_yesterday : ℕ) :
  total_students = 33 ∧
  swimming_today = 22 ∧
  soccer_today = 22 ∧
  soccer_today_swimming_yesterday = 15 ∧
  soccer_today_soccer_yesterday = 15 ∧
  swimming_today_swimming_yesterday = 15 ∧
  swimming_today_soccer_yesterday = 15 →
  ∃ (swimming_both_days : ℕ), swimming_both_days = 4 :=
by
  sorry

end NUMINAMATH_GPT_swimming_both_days_l140_14089


namespace NUMINAMATH_GPT_expected_value_of_winnings_is_5_l140_14067

namespace DiceGame

def sides : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

def winnings (roll : ℕ) : ℕ :=
  if roll % 2 = 0 then 2 * roll else 0

noncomputable def expectedValue : ℚ :=
  (winnings 2 + winnings 4 + winnings 6 + winnings 8) / 8

theorem expected_value_of_winnings_is_5 :
  expectedValue = 5 := by
  sorry

end DiceGame

end NUMINAMATH_GPT_expected_value_of_winnings_is_5_l140_14067


namespace NUMINAMATH_GPT_actual_time_is_1240pm_l140_14028

def kitchen_and_cellphone_start (t : ℕ) : Prop := t = 8 * 60  -- 8:00 AM in minutes
def kitchen_clock_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 30  -- 8:30 AM in minutes
def cellphone_after_breakfast (t : ℕ) : Prop := t = 8 * 60 + 20  -- 8:20 AM in minutes
def kitchen_clock_at_3pm (t : ℕ) : Prop := t = 15 * 60  -- 3:00 PM in minutes

theorem actual_time_is_1240pm : 
  (kitchen_and_cellphone_start 480) ∧ 
  (kitchen_clock_after_breakfast 510) ∧ 
  (cellphone_after_breakfast 500) ∧
  (kitchen_clock_at_3pm 900) → 
  real_time_at_kitchen_clock_time_3pm = 12 * 60 + 40 :=
by
  sorry

end NUMINAMATH_GPT_actual_time_is_1240pm_l140_14028


namespace NUMINAMATH_GPT_correct_operations_l140_14088

variable (x : ℚ)

def incorrect_equation := ((x - 5) * 3) / 7 = 10

theorem correct_operations :
  incorrect_equation x → (3 * x - 5) / 7 = 80 / 7 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_correct_operations_l140_14088


namespace NUMINAMATH_GPT_number_of_students_surveyed_l140_14016

noncomputable def M : ℕ := 60
noncomputable def N : ℕ := 90
noncomputable def B : ℕ := M / 3

theorem number_of_students_surveyed : M + B + N = 170 := by
  rw [M, N, B]
  norm_num
  sorry

end NUMINAMATH_GPT_number_of_students_surveyed_l140_14016


namespace NUMINAMATH_GPT_product_roots_cos_pi_by_9_cos_2pi_by_9_l140_14051

theorem product_roots_cos_pi_by_9_cos_2pi_by_9 :
  ∀ (d e : ℝ), (∀ x, x^2 + d * x + e = (x - Real.cos (π / 9)) * (x - Real.cos (2 * π / 9))) → 
    d * e = -5 / 64 :=
by
  sorry

end NUMINAMATH_GPT_product_roots_cos_pi_by_9_cos_2pi_by_9_l140_14051


namespace NUMINAMATH_GPT_basketball_game_first_half_points_l140_14012

theorem basketball_game_first_half_points (a b r d : ℕ) (H1 : a = b)
  (H2 : a * (1 + r + r^2 + r^3) = 4 * a + 6 * d + 1) 
  (H3 : 15 * a ≤ 100) (H4 : b + (b + d) + b + 2 * d + b + 3 * d < 100) : 
  (a + a * r + b + b + d) = 34 :=
by sorry

end NUMINAMATH_GPT_basketball_game_first_half_points_l140_14012


namespace NUMINAMATH_GPT_bess_throw_distance_l140_14073

-- Definitions based on the conditions
def bess_throws (x : ℝ) : ℝ := 4 * 2 * x
def holly_throws : ℝ := 5 * 8
def total_throws (x : ℝ) : ℝ := bess_throws x + holly_throws

-- Lean statement for the proof
theorem bess_throw_distance (x : ℝ) (h : total_throws x = 200) : x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_bess_throw_distance_l140_14073


namespace NUMINAMATH_GPT_max_black_cells_1000_by_1000_l140_14054

def maxBlackCells (m n : ℕ) : ℕ :=
  if m = 1 then n else if n = 1 then m else m + n - 2

theorem max_black_cells_1000_by_1000 : maxBlackCells 1000 1000 = 1998 :=
  by sorry

end NUMINAMATH_GPT_max_black_cells_1000_by_1000_l140_14054


namespace NUMINAMATH_GPT_handshake_problem_l140_14043

def combinations (n k : ℕ) : ℕ :=
  n.choose k

theorem handshake_problem : combinations 40 2 = 780 := 
by
  sorry

end NUMINAMATH_GPT_handshake_problem_l140_14043


namespace NUMINAMATH_GPT_fill_in_the_blank_with_flowchart_l140_14092

def methods_to_describe_algorithm := ["Natural language", "Flowchart", "Pseudocode"]

theorem fill_in_the_blank_with_flowchart : 
  methods_to_describe_algorithm[1] = "Flowchart" :=
sorry

end NUMINAMATH_GPT_fill_in_the_blank_with_flowchart_l140_14092


namespace NUMINAMATH_GPT_xy_product_of_sample_l140_14076

/-- Given a sample {9, 10, 11, x, y} such that the average is 10 and the standard deviation is sqrt(2), 
    prove that the product of x and y is 96. -/
theorem xy_product_of_sample (x y : ℝ) 
  (h_avg : (9 + 10 + 11 + x + y) / 5 = 10)
  (h_stddev : ( (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2 ) / 5 = 2) :
  x * y = 96 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_xy_product_of_sample_l140_14076


namespace NUMINAMATH_GPT_triathlete_average_speed_is_approx_3_5_l140_14014

noncomputable def triathlete_average_speed : ℝ :=
  let x : ℝ := 1; -- This represents the distance of biking/running segment
  let swimming_speed := 2; -- km/h
  let biking_speed := 25; -- km/h
  let running_speed := 12; -- km/h
  let swimming_distance := 2 * x; -- 2x km
  let biking_distance := x; -- x km
  let running_distance := x; -- x km
  let total_distance := swimming_distance + biking_distance + running_distance; -- 4x km
  let swimming_time := swimming_distance / swimming_speed; -- x hours
  let biking_time := biking_distance / biking_speed; -- x/25 hours
  let running_time := running_distance / running_speed; -- x/12 hours
  let total_time := swimming_time + biking_time + running_time; -- 1.12333x hours
  total_distance / total_time -- This should be the average speed

theorem triathlete_average_speed_is_approx_3_5 :
  abs (triathlete_average_speed - 3.5) < 0.1 := 
by
  sorry

end NUMINAMATH_GPT_triathlete_average_speed_is_approx_3_5_l140_14014


namespace NUMINAMATH_GPT_remainder_when_sum_divided_by_40_l140_14017

theorem remainder_when_sum_divided_by_40 (x y : ℤ) 
  (h1 : x % 80 = 75) 
  (h2 : y % 120 = 115) : 
  (x + y) % 40 = 30 := 
  sorry

end NUMINAMATH_GPT_remainder_when_sum_divided_by_40_l140_14017


namespace NUMINAMATH_GPT_mean_age_of_children_l140_14087

theorem mean_age_of_children :
  let ages := [8, 8, 12, 12, 10, 14]
  let n := ages.length
  let sum_ages := ages.foldr (· + ·) 0
  let mean_age := sum_ages / n
  mean_age = 10 + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_age_of_children_l140_14087


namespace NUMINAMATH_GPT_min_garden_cost_l140_14019

theorem min_garden_cost : 
  let flower_cost (flower : String) : Real :=
    if flower = "Asters" then 1 else
    if flower = "Begonias" then 2 else
    if flower = "Cannas" then 2 else
    if flower = "Dahlias" then 3 else
    if flower = "Easter lilies" then 2.5 else
    0
  let region_area (region : String) : Nat :=
    if region = "Bottom left" then 10 else
    if region = "Top left" then 9 else
    if region = "Bottom right" then 20 else
    if region = "Top middle" then 2 else
    if region = "Top right" then 7 else
    0
  let min_cost : Real :=
    (flower_cost "Dahlias" * region_area "Top middle") + 
    (flower_cost "Easter lilies" * region_area "Top right") + 
    (flower_cost "Cannas" * region_area "Top left") + 
    (flower_cost "Begonias" * region_area "Bottom left") + 
    (flower_cost "Asters" * region_area "Bottom right")
  min_cost = 81.5 :=
by
  sorry

end NUMINAMATH_GPT_min_garden_cost_l140_14019


namespace NUMINAMATH_GPT_soccer_team_starters_l140_14005

open Nat

-- Definitions representing the conditions
def total_players : ℕ := 18
def twins_included : ℕ := 2
def remaining_players : ℕ := total_players - twins_included
def starters_to_choose : ℕ := 7 - twins_included

-- Theorem statement to assert the solution
theorem soccer_team_starters :
  Nat.choose remaining_players starters_to_choose = 4368 :=
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_soccer_team_starters_l140_14005


namespace NUMINAMATH_GPT_arithmetic_sequence_8th_term_l140_14002

theorem arithmetic_sequence_8th_term 
  (a d : ℤ) 
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 41) : 
  a + 7 * d = 59 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_8th_term_l140_14002


namespace NUMINAMATH_GPT_positive_difference_eq_30_l140_14052

noncomputable def positive_difference_of_solutions : ℝ :=
  let x₁ : ℝ := 18
  let x₂ : ℝ := -12
  x₁ - x₂

theorem positive_difference_eq_30 (h : ∀ x, |x - 3| = 15 → (x = 18 ∨ x = -12)) :
  positive_difference_of_solutions = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_eq_30_l140_14052


namespace NUMINAMATH_GPT_intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l140_14013

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.sqrt (x + 2))) + Real.log (3 - x)
def A : Set ℝ := { x | -2 < x ∧ x < 3 }
def B (m : ℝ) : Set ℝ := { x | 1 - m < x ∧ x < 3 * m - 1 }

theorem intersection_of_A_and_B_is_B_implies_m_leq_4_over_3 (m : ℝ) 
    (h : A ∩ B m = B m) : m ≤ 4 / 3 := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_is_B_implies_m_leq_4_over_3_l140_14013


namespace NUMINAMATH_GPT_bus_stops_per_hour_l140_14070

-- Define the constants and conditions given in the problem
noncomputable def speed_without_stoppages : ℝ := 54 -- km/hr
noncomputable def speed_with_stoppages : ℝ := 45 -- km/hr

-- Theorem statement to prove the number of minutes the bus stops per hour
theorem bus_stops_per_hour : (speed_without_stoppages - speed_with_stoppages) / (speed_without_stoppages / 60) = 10 :=
by
  sorry

end NUMINAMATH_GPT_bus_stops_per_hour_l140_14070


namespace NUMINAMATH_GPT_weight_of_b_l140_14044

-- Define the weights of a, b, and c
variables (W_a W_b W_c : ℝ)

-- Define the heights of a, b, and c
variables (h_a h_b h_c : ℝ)

-- Given conditions
axiom average_weight_abc : (W_a + W_b + W_c) / 3 = 45
axiom average_weight_ab : (W_a + W_b) / 2 = 40
axiom average_weight_bc : (W_b + W_c) / 2 = 47
axiom height_condition : h_a + h_c = 2 * h_b
axiom odd_sum_weights : (W_a + W_b + W_c) % 2 = 1

-- Prove that the weight of b is 39 kg
theorem weight_of_b : W_b = 39 :=
by sorry

end NUMINAMATH_GPT_weight_of_b_l140_14044


namespace NUMINAMATH_GPT_incorrect_conclusion_l140_14082

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem incorrect_conclusion :
  ¬ (∀ x : ℝ, f ( (3 * Real.pi) / 4 - x ) + f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l140_14082


namespace NUMINAMATH_GPT_trigonometric_identity_l140_14000

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1 / 2) : 
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l140_14000


namespace NUMINAMATH_GPT_tape_needed_for_large_box_l140_14099

-- Definition of the problem conditions
def tape_per_large_box (L : ℕ) : Prop :=
  -- Each large box takes L feet of packing tape to seal
  -- Each medium box takes 2 feet of packing tape to seal
  -- Each small box takes 1 foot of packing tape to seal
  -- Each box also takes 1 foot of packing tape to stick the address label on
  -- Debbie packed two large boxes this afternoon
  -- Debbie packed eight medium boxes this afternoon
  -- Debbie packed five small boxes this afternoon
  -- Debbie used 44 feet of tape in total
  2 * L + 2 + 24 + 10 = 44

theorem tape_needed_for_large_box : ∃ L : ℕ, tape_per_large_box L ∧ L = 4 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_tape_needed_for_large_box_l140_14099


namespace NUMINAMATH_GPT_g_49_l140_14022

noncomputable def g : ℝ → ℝ := sorry

axiom g_func_eqn (x y : ℝ) : g (x^2 * y) = x * g y
axiom g_one_val : g 1 = 6

theorem g_49 : g 49 = 42 := by
  sorry

end NUMINAMATH_GPT_g_49_l140_14022


namespace NUMINAMATH_GPT_factor_expression_l140_14072

theorem factor_expression (x y : ℝ) : 5 * x * (x + 1) + 7 * (x + 1) - 2 * y * (x + 1) = (x + 1) * (5 * x + 7 - 2 * y) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l140_14072


namespace NUMINAMATH_GPT_outcome_transactions_l140_14050

-- Definition of initial property value and profit/loss percentages.
def property_value : ℝ := 15000
def profit_percentage : ℝ := 0.15
def loss_percentage : ℝ := 0.05

-- Calculate selling price after 15% profit.
def selling_price : ℝ := property_value * (1 + profit_percentage)

-- Calculate buying price after 5% loss based on the above selling price.
def buying_price : ℝ := selling_price * (1 - loss_percentage)

-- Calculate the net gain/loss.
def net_gain_or_loss : ℝ := selling_price - buying_price

-- Statement to be proved.
theorem outcome_transactions : net_gain_or_loss = 862.5 := by
  sorry

end NUMINAMATH_GPT_outcome_transactions_l140_14050


namespace NUMINAMATH_GPT_max_marks_paper_one_l140_14042

theorem max_marks_paper_one (M : ℝ) : 
  (0.42 * M = 64) → (M = 152) :=
by
  sorry

end NUMINAMATH_GPT_max_marks_paper_one_l140_14042


namespace NUMINAMATH_GPT_extremum_at_x1_l140_14046

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

theorem extremum_at_x1 (a b : ℝ) (h1 : (3*1^2 + 2*a*1 + b) = 0) (h2 : 1^3 + a*1^2 + b*1 + a^2 = 10) :
  a = 4 :=
by
  sorry

end NUMINAMATH_GPT_extremum_at_x1_l140_14046


namespace NUMINAMATH_GPT_sarah_speed_for_rest_of_trip_l140_14003

def initial_speed : ℝ := 15  -- miles per hour
def initial_time : ℝ := 1  -- hour
def total_distance : ℝ := 45  -- miles
def extra_time_if_same_speed : ℝ := 1  -- hour (late)
def arrival_early_time : ℝ := 0.5  -- hour (early)

theorem sarah_speed_for_rest_of_trip (remaining_distance remaining_time : ℝ) :
  remaining_distance = total_distance - initial_speed * initial_time →
  remaining_time = (remaining_distance / initial_speed - extra_time_if_same_speed) + arrival_early_time →
  remaining_distance / remaining_time = 20 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_sarah_speed_for_rest_of_trip_l140_14003


namespace NUMINAMATH_GPT_additional_savings_in_cents_l140_14053

/-
The book has a cover price of $30.
There are two discount methods to compare:
1. First $5 off, then 25% off.
2. First 25% off, then $5 off.
Prove that the difference in final costs (in cents) between these two discount methods is 125 cents.
-/
def book_price : ℝ := 30
def discount_cash : ℝ := 5
def discount_percentage : ℝ := 0.25

def final_price_apply_cash_first (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (price - cash_discount) * (1 - percentage_discount)

def final_price_apply_percentage_first (price : ℝ) (percentage_discount : ℝ) (cash_discount : ℝ) : ℝ :=
  (price * (1 - percentage_discount)) - cash_discount

def savings_comparison (price : ℝ) (cash_discount : ℝ) (percentage_discount : ℝ) : ℝ :=
  (final_price_apply_cash_first price cash_discount percentage_discount) - 
  (final_price_apply_percentage_first price percentage_discount cash_discount)

theorem additional_savings_in_cents : 
  savings_comparison book_price discount_cash discount_percentage * 100 = 125 :=
  by sorry

end NUMINAMATH_GPT_additional_savings_in_cents_l140_14053


namespace NUMINAMATH_GPT_probability_queen_then_club_l140_14047

-- Define the problem conditions using the definitions
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_clubs : ℕ := 13
def num_club_queens : ℕ := 1

-- Define a function that computes the probability of the given event
def probability_first_queen_second_club : ℚ :=
  let prob_first_club_queen := (num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_club_queen := (num_clubs - 1 : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_1 := prob_first_club_queen * prob_second_club_given_first_club_queen
  let prob_first_non_club_queen := (num_queens - num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_non_club_queen := (num_clubs : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_2 := prob_first_non_club_queen * prob_second_club_given_first_non_club_queen
  prob_case_1 + prob_case_2

-- The statement to be proved
theorem probability_queen_then_club : probability_first_queen_second_club = 1 / 52 := by
  sorry

end NUMINAMATH_GPT_probability_queen_then_club_l140_14047


namespace NUMINAMATH_GPT_lemonade_percentage_l140_14056

theorem lemonade_percentage (L : ℝ) : 
  (0.4 * (1 - L / 100) + 0.6 * 0.55 = 0.65) → L = 20 :=
by
  sorry

end NUMINAMATH_GPT_lemonade_percentage_l140_14056


namespace NUMINAMATH_GPT_bird_watcher_total_l140_14025

theorem bird_watcher_total
  (M : ℕ) (T : ℕ) (W : ℕ)
  (h1 : M = 70)
  (h2 : T = M / 2)
  (h3 : W = T + 8) :
  M + T + W = 148 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_bird_watcher_total_l140_14025


namespace NUMINAMATH_GPT_find_triangle_sides_l140_14061

theorem find_triangle_sides (a : Fin 7 → ℝ) (h : ∀ i, 1 < a i ∧ a i < 13) : 
  ∃ i j k, 1 ≤ i ∧ i < j ∧ j < k ∧ k ≤ 7 ∧ 
           a i + a j > a k ∧ 
           a j + a k > a i ∧ 
           a k + a i > a j :=
sorry

end NUMINAMATH_GPT_find_triangle_sides_l140_14061


namespace NUMINAMATH_GPT_speed_of_other_train_l140_14074

theorem speed_of_other_train
  (v : ℝ) -- speed of the second train
  (t : ℝ := 2.5) -- time in hours
  (distance : ℝ := 285) -- total distance
  (speed_first_train : ℝ := 50) -- speed of the first train
  (h : speed_first_train * t + v * t = distance) :
  v = 64 :=
by
  -- The proof will be assumed
  sorry

end NUMINAMATH_GPT_speed_of_other_train_l140_14074


namespace NUMINAMATH_GPT_billion_to_scientific_l140_14077
noncomputable def scientific_notation_of_billion (n : ℝ) : ℝ := n * 10^9
theorem billion_to_scientific (a : ℝ) : scientific_notation_of_billion a = 1.48056 * 10^11 :=
by sorry

end NUMINAMATH_GPT_billion_to_scientific_l140_14077


namespace NUMINAMATH_GPT_total_area_of_paths_l140_14091

theorem total_area_of_paths:
  let bed_width := 4
  let bed_height := 3
  let num_beds_width := 3
  let num_beds_height := 5
  let path_width := 2

  let total_bed_width := num_beds_width * bed_width
  let total_path_width := (num_beds_width + 1) * path_width
  let total_width := total_bed_width + total_path_width

  let total_bed_height := num_beds_height * bed_height
  let total_path_height := (num_beds_height + 1) * path_width
  let total_height := total_bed_height + total_path_height

  let total_area_greenhouse := total_width * total_height
  let total_area_beds := num_beds_width * num_beds_height * bed_width * bed_height

  let total_area_paths := total_area_greenhouse - total_area_beds

  total_area_paths = 360 :=
by sorry

end NUMINAMATH_GPT_total_area_of_paths_l140_14091


namespace NUMINAMATH_GPT_problem1_simplification_problem2_solve_fraction_l140_14049

-- Problem 1: Simplification and Calculation
theorem problem1_simplification (x : ℝ) : 
  ((12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1)) = (2 * x - 4 * x^2) :=
by sorry

-- Problem 2: Solving the Fractional Equation
theorem problem2_solve_fraction (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ -1) :
  (5 / (x^2 + x) - 1 / (x^2 - x) = 0) ↔ (x = 3 / 2) :=
by sorry

end NUMINAMATH_GPT_problem1_simplification_problem2_solve_fraction_l140_14049


namespace NUMINAMATH_GPT_kids_on_soccer_field_l140_14041

def original_kids : ℕ := 14
def joined_kids : ℕ := 22
def total_kids : ℕ := 36

theorem kids_on_soccer_field : (original_kids + joined_kids) = total_kids :=
by 
  sorry

end NUMINAMATH_GPT_kids_on_soccer_field_l140_14041


namespace NUMINAMATH_GPT_john_protest_days_l140_14059

theorem john_protest_days (days1: ℕ) (days2: ℕ) (days3: ℕ): 
  days1 = 4 → 
  days2 = (days1 + (days1 / 4)) → 
  days3 = (days2 + (days2 / 2)) → 
  (days1 + days2 + days3) = 17 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_john_protest_days_l140_14059


namespace NUMINAMATH_GPT_find_distance_CD_l140_14011

-- Define the ellipse and the required points
def ellipse (x y : ℝ) : Prop := 16 * (x-3)^2 + 4 * (y+2)^2 = 64

-- Define the center and the semi-axes lengths
noncomputable def center : (ℝ × ℝ) := (3, -2)
noncomputable def semi_major_axis_length : ℝ := 4
noncomputable def semi_minor_axis_length : ℝ := 2

-- Define the points C and D on the ellipse
def point_C (x y : ℝ) : Prop := ellipse x y ∧ (x = 3 + semi_major_axis_length ∨ x = 3 - semi_major_axis_length) ∧ y = -2
def point_D (x y : ℝ) : Prop := ellipse x y ∧ x = 3 ∧ (y = -2 + semi_minor_axis_length ∨ y = -2 - semi_minor_axis_length)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Main theorem to prove
theorem find_distance_CD : 
  ∃ C D : ℝ × ℝ, 
    (point_C C.1 C.2 ∧ point_D D.1 D.2) → 
    distance C D = 2 * Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_find_distance_CD_l140_14011
