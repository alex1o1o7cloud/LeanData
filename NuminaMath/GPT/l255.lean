import Mathlib

namespace pythagorean_triple_square_l255_255818

theorem pythagorean_triple_square (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pythagorean : a^2 + b^2 = c^2) : ∃ k : ℤ, k^2 = (c - a) * (c - b) / 2 := 
sorry

end pythagorean_triple_square_l255_255818


namespace larger_solution_of_quadratic_equation_l255_255084

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l255_255084


namespace inequality_proof_l255_255584

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
    (a^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*a - 1) +
    (b^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*b - 1) +
    (c^2 + 6) / (2*a^2 + 2*b^2 + 2*c^2 + 2*c - 1) ≤
    3 := by
  sorry

end inequality_proof_l255_255584


namespace polynomial_divisibility_p_q_l255_255100

theorem polynomial_divisibility_p_q (p' q' : ℝ) :
  (∀ x, x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0 → (x = -1 ∨ x = 2)) →
  p' = 0 ∧ q' = -9 :=
by sorry

end polynomial_divisibility_p_q_l255_255100


namespace angle_D_calculation_l255_255643

theorem angle_D_calculation (A B E C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50)
  (h4 : E = 60)
  (h5 : A + B + E = 180)
  (h6 : B + C + D = 180) :
  D = 55 :=
by
  sorry

end angle_D_calculation_l255_255643


namespace age_problem_l255_255870

theorem age_problem :
  (∃ (x y : ℕ), 
    (3 * x - 7 = 5 * (x - 7)) ∧ 
    (42 + y = 2 * (14 + y)) ∧ 
    (2 * x = 28) ∧ 
    (x = 14) ∧ 
    (3 * 14 = 42) ∧ 
    (42 - 14 = 28) ∧ 
    (y = 14)) :=
by
  sorry

end age_problem_l255_255870


namespace max_value_of_exp_diff_l255_255633

open Real

theorem max_value_of_exp_diff : ∀ x : ℝ, ∃ y : ℝ, y = 2^x - 4^x ∧ y ≤ 1/4 := sorry

end max_value_of_exp_diff_l255_255633


namespace sufficient_condition_l255_255853

theorem sufficient_condition (A B C D : Prop) (h : C → D): C → (A > B) := 
by 
  sorry

end sufficient_condition_l255_255853


namespace total_handshakes_is_790_l255_255804

-- Define the context of the two groups:
def groupA : Finset ℕ := {1, 2, ..., 30} -- Set of 30 people knowing each other
def groupB : Finset ℕ := {31, 32, ..., 50} -- Set of 20 people knowing no one

noncomputable def total_handshakes : ℕ :=
  let handshakes_between : ℕ := 30 * 20 in  -- Handshakes between Group A and Group B
  let handshakes_within_B : ℕ := Nat.choose 20 2 in -- Handshakes within Group B (choose 2 from 20)
  handshakes_between + handshakes_within_B

-- The theorem to prove
theorem total_handshakes_is_790 : total_handshakes = 790 := by
  sorry

end total_handshakes_is_790_l255_255804


namespace ribbon_used_l255_255024

def total_ribbon : ℕ := 84
def leftover_ribbon : ℕ := 38
def used_ribbon : ℕ := 46

theorem ribbon_used : total_ribbon - leftover_ribbon = used_ribbon := sorry

end ribbon_used_l255_255024


namespace profit_last_month_l255_255572

variable (gas_expenses earnings_per_lawn lawns_mowed extra_income profit : ℤ)

def toms_profit (gas_expenses earnings_per_lawn lawns_mowed extra_income : ℤ) : ℤ :=
  (lawns_mowed * earnings_per_lawn + extra_income) - gas_expenses

theorem profit_last_month :
  toms_profit 17 12 3 10 = 29 :=
by
  rw [toms_profit]
  sorry

end profit_last_month_l255_255572


namespace simplify_expression_l255_255501

theorem simplify_expression (a : ℝ) (h : a < 1 / 4) : 4 * (4 * a - 1)^2 = (1 - 4 * a) ^ 2 :=
by
  sorry

end simplify_expression_l255_255501


namespace radius_of_sphere_eq_l255_255713

theorem radius_of_sphere_eq (r : ℝ) : 
  (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2 → r = 3 :=
by
  sorry

end radius_of_sphere_eq_l255_255713


namespace parallelogram_height_same_area_l255_255045

noncomputable def rectangle_area (length width : ℕ) : ℕ := length * width

theorem parallelogram_height_same_area (length width base height : ℕ) 
  (h₁ : rectangle_area length width = base * height) 
  (h₂ : length = 12) 
  (h₃ : width = 6) 
  (h₄ : base = 12) : 
  height = 6 := 
sorry

end parallelogram_height_same_area_l255_255045


namespace max_value_expression_l255_255650

/--
Given real numbers \( x, y, z, w \) satisfying \( x + y + z + w = 1 \),
prove that the maximum value of \( M = xw + 2yw + 3xy + 3zw + 4xz + 5yz \) is \( \frac{3}{2} \).
-/
theorem max_value_expression (x y z w : ℝ) (h : x + y + z + w = 1) :
  let M := x * w + 2 * y * w + 3 * x * y + 3 * z * w + 4 * x * z + 5 * y * z
  in M ≤ 3 / 2 :=
begin
  sorry
end

end max_value_expression_l255_255650


namespace last_digit_to_appear_mod9_l255_255761

def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

def fib_mod9 (n : ℕ) : ℕ :=
  (fib n) % 9

theorem last_digit_to_appear_mod9 :
  ∃ n : ℕ, ∀ m : ℕ, m < n → fib_mod9 m ≠ 0 ∧ fib_mod9 n = 0 :=
sorry

end last_digit_to_appear_mod9_l255_255761


namespace triangle_angle_and_perimeter_l255_255673

/-
In a triangle ABC, given c * sin B = sqrt 3 * cos C,
prove that angle C equals pi / 3,
and given a + b = 6, find the minimum perimeter of triangle ABC.
-/
theorem triangle_angle_and_perimeter (A B C : ℝ) (a b c : ℝ) 
  (h1 : c * Real.sin B = Real.sqrt 3 * Real.cos C)
  (h2 : a + b = 6) :
  C = Real.pi / 3 ∧ a + b + (Real.sqrt (36 - a * b)) = 9 :=
by
  sorry

end triangle_angle_and_perimeter_l255_255673


namespace height_difference_l255_255990

-- Definitions of the terms and conditions
variables {b h : ℝ} -- base and height of Triangle B
variables {b' h' : ℝ} -- base and height of Triangle A

-- Given conditions:
-- Triangle A's base is 10% greater than Triangle B's base
def base_relation (b' : ℝ) (b : ℝ) := b' = 1.10 * b

-- The area of Triangle A is 1% less than the area of Triangle B
def area_relation (b h b' h' : ℝ) := (1 / 2) * b' * h' = (1 / 2) * b * h - 0.01 * (1 / 2) * b * h

-- Proof statement
theorem height_difference (b h b' h' : ℝ) (H_base: base_relation b' b) (H_area: area_relation b h b' h') :
  h' = 0.9 * h := 
sorry

end height_difference_l255_255990


namespace max_log_sum_l255_255926

open Real

theorem max_log_sum (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 4 * y = 40) :
  log x + log y ≤ 2 :=
sorry

end max_log_sum_l255_255926


namespace area_of_equilateral_triangle_with_inscribed_circle_l255_255718

theorem area_of_equilateral_triangle_with_inscribed_circle 
  (r : ℝ) (A : ℝ) (area_circle_eq : A = 9 * Real.pi)
  (DEF_equilateral : ∀ {a b c : ℝ}, a = b ∧ b = c): 
  ∃ area_def : ℝ, area_def = 27 * Real.sqrt 3 :=
by
  -- proof omitted
  sorry

end area_of_equilateral_triangle_with_inscribed_circle_l255_255718


namespace solve_quadratic_roots_l255_255344

theorem solve_quadratic_roots (b c : ℝ) 
  (h : {1, 2} = {x : ℝ | x^2 + b * x + c = 0}) : 
  b = -3 ∧ c = 2 :=
by
  sorry

end solve_quadratic_roots_l255_255344


namespace cricket_problem_solved_l255_255122

noncomputable def cricket_problem : Prop :=
  let run_rate_10 := 3.2
  let target := 252
  let required_rate := 5.5
  let overs_played := 10
  let total_overs := 50
  let runs_scored := run_rate_10 * overs_played
  let runs_remaining := target - runs_scored
  let overs_remaining := total_overs - overs_played
  (runs_remaining / overs_remaining = required_rate)

theorem cricket_problem_solved : cricket_problem :=
by
  sorry

end cricket_problem_solved_l255_255122


namespace expand_binomial_l255_255479

theorem expand_binomial (x : ℝ) : (x + 3) * (x - 8) = x^2 - 5 * x - 24 :=
by
  sorry

end expand_binomial_l255_255479


namespace area_of_yard_l255_255773

theorem area_of_yard (L W : ℕ) (h1 : L = 40) (h2 : L + 2 * W = 64) : L * W = 480 := by
  sorry

end area_of_yard_l255_255773


namespace ms_warren_walking_time_l255_255136

/-- 
Ms. Warren ran at 6 mph for 20 minutes. After the run, 
she walked at 2 mph for a certain amount of time. 
She ran and walked a total of 3 miles.
-/
def time_spent_walking (running_speed walking_speed : ℕ) (running_time_minutes : ℕ) (total_distance : ℕ) : ℕ := 
  let running_time_hours := running_time_minutes / 60;
  let distance_ran := running_speed * running_time_hours;
  let distance_walked := total_distance - distance_ran;
  let time_walked_hours := distance_walked / walking_speed;
  time_walked_hours * 60

theorem ms_warren_walking_time :
  time_spent_walking 6 2 20 3 = 30 :=
by
  sorry

end ms_warren_walking_time_l255_255136


namespace a_greater_than_1_and_b_less_than_1_l255_255228

theorem a_greater_than_1_and_b_less_than_1
  (a b c : ℝ) 
  (h1 : a ≥ b)
  (h2 : b ≥ c)
  (h3 : c > 0)
  (h4 : a * b * c = 1)
  (h5 : a + b + c > 1/a + 1/b + 1/c) : a > 1 ∧ b < 1 :=
by
  sorry

end a_greater_than_1_and_b_less_than_1_l255_255228


namespace second_quadrant_point_l255_255671

theorem second_quadrant_point (x : ℝ) (h1 : x < 2) (h2 : x > 1/2) : 
  (x-2 < 0) ∧ (2*x-1 > 0) ↔ (1/2 < x ∧ x < 2) :=
by
  sorry

end second_quadrant_point_l255_255671


namespace absolute_difference_rectangle_l255_255457

theorem absolute_difference_rectangle 
  (x y r k : ℝ)
  (h1 : 2 * x + 2 * y = 4 * r)
  (h2 : (x^2 + y^2) = (k * x)^2) :
  |x - y| = k * x :=
by
  sorry

end absolute_difference_rectangle_l255_255457


namespace third_candidate_votes_l255_255716

theorem third_candidate_votes (V A B W: ℕ) (hA : A = 2500) (hB : B = 15000) 
  (hW : W = (2 * V) / 3) (hV : V = W + A + B) : (V - (A + B)) = 35000 := by
  sorry

end third_candidate_votes_l255_255716


namespace equilateral_triangle_perimeter_l255_255970

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end equilateral_triangle_perimeter_l255_255970


namespace otgaday_wins_l255_255975

theorem otgaday_wins (a n : ℝ) : a * n > 0.91 * a * n := 
by
  sorry

end otgaday_wins_l255_255975


namespace distinct_collections_COMPUTATIONS_l255_255672

theorem distinct_collections_COMPUTATIONS : 
  let vowels := {O, U, A, I}
  let consonants := {C, M, P, T, T, S, N}
  let binom := Nat.choose
  (binom 4 3 * 
    (binom 6 4 + binom 6 3 * binom 2 1 + binom 6 2 * binom 2 2)) = 200 := 
by 
  unfold vowels consonants binom 
  sorry

end distinct_collections_COMPUTATIONS_l255_255672


namespace bicycle_has_four_wheels_l255_255613

variables (Car : Type) (Bicycle : Car) (FourWheeled : Car → Prop)
axiom car_four_wheels : ∀ (c : Car), FourWheeled c

theorem bicycle_has_four_wheels : FourWheeled Bicycle :=
by {
  apply car_four_wheels
}

end bicycle_has_four_wheels_l255_255613


namespace quiz_answer_keys_count_l255_255808

noncomputable def count_answer_keys : ℕ :=
  (Nat.choose 10 5) * (Nat.factorial 6)

theorem quiz_answer_keys_count :
  count_answer_keys = 181440 := 
by
  -- Proof is skipped, using sorry
  sorry

end quiz_answer_keys_count_l255_255808


namespace estimate_white_balls_l255_255807

-- Statements for conditions
variables (black_balls white_balls : ℕ)
variables (draws : ℕ := 40)
variables (black_draws : ℕ := 10)

-- Define total white draws
def white_draws := draws - black_draws

-- Ratio of black to white draws
def draw_ratio := black_draws / white_draws

-- Given condition on known draws
def black_ball_count := 4
def known_draw_ratio := 1 / 3

-- Lean 4 statement to prove the number of white balls
theorem estimate_white_balls (h : black_ball_count / white_balls = known_draw_ratio) : white_balls = 12 :=
sorry -- Proof omitted

end estimate_white_balls_l255_255807


namespace average_difference_l255_255146

theorem average_difference :
  let avg1 := (24 + 35 + 58) / 3
  let avg2 := (19 + 51 + 29) / 3
  avg1 - avg2 = 6 := by
sorry

end average_difference_l255_255146


namespace equilibrium_constant_relationship_l255_255977

def given_problem (K1 K2 : ℝ) : Prop :=
  K2 = (1 / K1)^(1 / 2)

theorem equilibrium_constant_relationship (K1 K2 : ℝ) (h : given_problem K1 K2) :
  K1 = 1 / K2^2 :=
by sorry

end equilibrium_constant_relationship_l255_255977


namespace boys_left_is_31_l255_255568

def initial_children : ℕ := 85
def girls_came_in : ℕ := 24
def final_children : ℕ := 78

noncomputable def compute_boys_left (initial : ℕ) (girls_in : ℕ) (final : ℕ) : ℕ :=
  (initial + girls_in) - final

theorem boys_left_is_31 :
  compute_boys_left initial_children girls_came_in final_children = 31 :=
by
  sorry

end boys_left_is_31_l255_255568


namespace find_multiple_l255_255846

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l255_255846


namespace find_m_l255_255797

theorem find_m (m : ℝ) (h₁: 0 < m) (h₂: ∀ p q : ℝ × ℝ, p = (m, 4) → q = (2, m) → ∃ s : ℝ, s = m^2 ∧ ((q.2 - p.2) / (q.1 - p.1)) = s) : m = 2 :=
by
  sorry

end find_m_l255_255797


namespace find_functions_l255_255481

-- Define the function f and its properties.
variable {f : ℝ → ℝ}

-- Define the condition given in the problem as a hypothesis.
def condition (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f x + f y) = y + f x ^ 2

-- State the theorem we want to prove.
theorem find_functions (hf : condition f) : (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
  sorry

end find_functions_l255_255481


namespace larger_root_of_quadratic_eq_l255_255081

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l255_255081


namespace consecutive_integers_sum_l255_255841

theorem consecutive_integers_sum (x : ℤ) (h : x * (x + 1) = 440) : x + (x + 1) = 43 :=
by sorry

end consecutive_integers_sum_l255_255841


namespace option_C_correct_l255_255169

theorem option_C_correct (a b : ℝ) : (2 * a * b^2)^2 = 4 * a^2 * b^4 := 
by 
  sorry

end option_C_correct_l255_255169


namespace original_cost_price_l255_255043

theorem original_cost_price (P : ℝ) 
  (h1 : P - 0.07 * P = 0.93 * P)
  (h2 : 0.93 * P + 0.02 * 0.93 * P = 0.9486 * P)
  (h3 : 0.9486 * P * 1.05 = 0.99603 * P)
  (h4 : 0.93 * P * 0.95 = 0.8835 * P)
  (h5 : 0.8835 * P + 0.02 * 0.8835 * P = 0.90117 * P)
  (h6 : 0.99603 * P - 5 = (0.90117 * P) * 1.10)
: P = 5 / 0.004743 :=
by
  sorry

end original_cost_price_l255_255043


namespace largest_common_term_arith_seq_l255_255611

theorem largest_common_term_arith_seq :
  ∃ a, a < 90 ∧ (∃ n : ℤ, a = 3 + 8 * n) ∧ (∃ m : ℤ, a = 5 + 9 * m) ∧ a = 59 :=
by
  sorry

end largest_common_term_arith_seq_l255_255611


namespace line_properties_l255_255710

theorem line_properties (m x_intercept : ℝ) (y_intercept point_on_line : ℝ × ℝ) :
  m = -4 → x_intercept = -3 → y_intercept = (0, -12) → point_on_line = (2, -20) → 
    (∀ x y, y = -4 * x - 12 → (y_intercept = (0, y) ∧ point_on_line = (x, y))) := 
by
  sorry

end line_properties_l255_255710


namespace value_of_k_l255_255110

theorem value_of_k (k : ℤ) : 
  (∀ x : ℤ, (x + k) * (x - 4) = x^2 - 4 * x + k * x - 4 * k ∧ 
  (k - 4) * x = 0) → k = 4 := 
by 
  sorry

end value_of_k_l255_255110


namespace sixth_year_fee_l255_255609

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l255_255609


namespace line_perp_to_plane_imp_perp_to_line_l255_255236

def Line := Type
def Plane := Type

variables (m n : Line) (α : Plane)

def is_parallel (l : Line) (p : Plane) : Prop := sorry
def is_perpendicular (l1 l2 : Line) : Prop := sorry
def is_contained (l : Line) (p : Plane) : Prop := sorry

theorem line_perp_to_plane_imp_perp_to_line :
  (is_perpendicular m α) ∧ (is_contained n α) → (is_perpendicular m n) :=
sorry

end line_perp_to_plane_imp_perp_to_line_l255_255236


namespace find_x_if_friendly_l255_255523

theorem find_x_if_friendly (x : ℚ) :
    (∃ m n : ℚ, m + n = 66 ∧ m = 7 * x ∧ n = -18) →
    x = 12 :=
by
  sorry

end find_x_if_friendly_l255_255523


namespace diamondsuit_result_l255_255132

def diam (a b : ℕ) : ℕ := a

theorem diamondsuit_result : (diam 7 (diam 4 8)) = 7 :=
by sorry

end diamondsuit_result_l255_255132


namespace invalid_votes_l255_255060

theorem invalid_votes (W L total_polls : ℕ) 
  (h1 : total_polls = 90830) 
  (h2 : L = 9 * W / 11) 
  (h3 : W = L + 9000)
  (h4 : 100 * (W + L) = 90000) : 
  total_polls - (W + L) = 830 := 
sorry

end invalid_votes_l255_255060


namespace find_quadruplets_l255_255909

theorem find_quadruplets :
  ∃ (x y z w : ℝ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
  (xyz + 1) / (x + 1) = (yzw + 1) / (y + 1) ∧
  (yzw + 1) / (y + 1) = (zwx + 1) / (z + 1) ∧
  (zwx + 1) / (z + 1) = (wxy + 1) / (w + 1) ∧
  x + y + z + w = 48 ∧
  x = 12 ∧ y = 12 ∧ z = 12 ∧ w = 12 :=
by
  sorry

end find_quadruplets_l255_255909


namespace total_amount_paid_l255_255510

def grapes_quantity := 8
def grapes_rate := 80
def mangoes_quantity := 9
def mangoes_rate := 55
def apples_quantity := 6
def apples_rate := 120
def oranges_quantity := 4
def oranges_rate := 75

theorem total_amount_paid :
  grapes_quantity * grapes_rate +
  mangoes_quantity * mangoes_rate +
  apples_quantity * apples_rate +
  oranges_quantity * oranges_rate =
  2155 := by
  sorry

end total_amount_paid_l255_255510


namespace marble_count_l255_255251

theorem marble_count (r g b : ℝ) (h1 : g + b = 9) (h2 : r + b = 7) (h3 : r + g = 5) :
  r + g + b = 10.5 :=
by sorry

end marble_count_l255_255251


namespace quadratic_distinct_real_roots_l255_255519

theorem quadratic_distinct_real_roots (m : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1^2 + m*x1 + 1 = 0 ∧ x2^2 + m*x2 + 1 = 0) ↔ m ∈ set.Ioo (-∞) (-2) ∪ set.Ioo 2 ∞ :=
by
  sorry

end quadratic_distinct_real_roots_l255_255519


namespace incorrect_line_pass_through_Q_l255_255056

theorem incorrect_line_pass_through_Q (a b : ℝ) : 
  (∀ (k : ℝ), ∃ (Q : ℝ × ℝ), Q = (0, b) ∧ y = k * x + b) →
  (¬ ∃ k : ℝ, ∀ y x, y = k * x + b ∧ x = 0)
:= 
sorry

end incorrect_line_pass_through_Q_l255_255056


namespace student_D_most_stable_l255_255772

-- Define the variances for students A, B, C, and D
def SA_squared : ℝ := 2.1
def SB_squared : ℝ := 3.5
def SC_squared : ℝ := 9
def SD_squared : ℝ := 0.7

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable :
  SD_squared < SA_squared ∧ SD_squared < SB_squared ∧ SD_squared < SC_squared := by
  sorry

end student_D_most_stable_l255_255772


namespace find_M_plus_N_l255_255518

theorem find_M_plus_N (M N : ℕ) (h1 : (3:ℚ) / 5 = M / 45) (h2 : (3:ℚ) / 5 = 60 / N) : M + N = 127 :=
sorry

end find_M_plus_N_l255_255518


namespace imaginary_part_of_quotient_l255_255558

noncomputable def imaginary_part_of_complex (z : ℂ) : ℂ := z.im

theorem imaginary_part_of_quotient :
  imaginary_part_of_complex (i / (1 - i)) = 1 / 2 :=
by sorry

end imaginary_part_of_quotient_l255_255558


namespace find_cost_price_per_meter_l255_255449

noncomputable def cost_price_per_meter
  (total_cloth : ℕ)
  (selling_price : ℕ)
  (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_cloth) / total_cloth

theorem find_cost_price_per_meter :
  cost_price_per_meter 75 4950 15 = 51 :=
by
  unfold cost_price_per_meter
  sorry

end find_cost_price_per_meter_l255_255449


namespace find_a_and_b_l255_255924

theorem find_a_and_b (a b : ℝ) 
  (h_tangent_slope : (2 * a * 2 + b = 1)) 
  (h_point_on_parabola : (a * 4 + b * 2 + 9 = -1)) : 
  a = 3 ∧ b = -11 :=
by
  sorry

end find_a_and_b_l255_255924


namespace compute_product_l255_255020

theorem compute_product (x1 y1 x2 y2 x3 y3 : ℝ) 
  (h1 : x1^3 - 3 * x1 * y1^2 = 1005) 
  (h2 : y1^3 - 3 * x1^2 * y1 = 1004)
  (h3 : x2^3 - 3 * x2 * y2^2 = 1005)
  (h4 : y2^3 - 3 * x2^2 * y2 = 1004)
  (h5 : x3^3 - 3 * x3 * y3^2 = 1005)
  (h6 : y3^3 - 3 * x3^2 * y3 = 1004) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 502 := 
sorry

end compute_product_l255_255020


namespace sum_of_x_and_y_l255_255249

theorem sum_of_x_and_y 
  (x y : ℤ)
  (h1 : x - y = 36) 
  (h2 : x = 28) : 
  x + y = 20 :=
by 
  sorry

end sum_of_x_and_y_l255_255249


namespace lunks_needed_for_12_apples_l255_255932

/-- 
  Given:
  1. 7 lunks can be traded for 4 kunks.
  2. 3 kunks will buy 5 apples.

  Prove that the number of lunks needed to purchase one dozen (12) apples is equal to 14.
-/
theorem lunks_needed_for_12_apples (L K : ℕ)
  (h1 : 7 * L = 4 * K)
  (h2 : 3 * K = 5) :
  (8 * K = 14 * L) :=
by
  sorry

end lunks_needed_for_12_apples_l255_255932


namespace angles_cos_eq_l255_255679

-- Definitions for angles of a triangle and given conditions
variables {A B C : ℝ} (ht : A + B + C = π) (hC : C > π/2)
variables {cosA cosB cosC sinA sinB sinC : ℝ}
  (hcosA : cosA = cos A) (hcosB : cosB = cos B) (hcosC : cosC = cos C)
  (hsinA : sinA = sin A) (hsinB : sinB = sin B) (hsinC : sinC = sin C)

-- Given equations
variables (eq1 : cosA ^ 2 + cosC ^ 2 + 2 * sinA * sinC * cosB = 17 / 9)
          (eq2 : cosC ^ 2 + cosB ^ 2 + 2 * sinC * sinB * cosA = 12 / 7)

-- The statement to be proved
theorem angles_cos_eq : ∃ (p q r s : ℤ), 
  p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ p + q ≠ 0 ∧ Nat.gcd (Int.toNat p + Int.toNat q) (Int.toNat s) = 1 ∧ 
  ¬ (∃ k : ℤ, k ^ 2 = r) ∧ 
  cosB ^ 2 + cosA ^ 2 + 2 * sinB * sinA * cosC = (p - q * real.sqrt r) / s ∧ 
  p + q + r + s = 220 :=
by
  sorry

end angles_cos_eq_l255_255679


namespace range_of_a_real_root_l255_255097

theorem range_of_a_real_root :
  (∀ x : ℝ, x^2 - a * x + 4 = 0 → ∃ x : ℝ, (x^2 - a * x + 4 = 0 ∧ (a ≥ 4 ∨ a ≤ -4))) ∨
  (∀ x : ℝ, x^2 + (a-2) * x + 4 = 0 → ∃ x : ℝ, (x^2 + (a-2) * x + 4 = 0 ∧ (a ≥ 6 ∨ a ≤ -2))) ∨
  (∀ x : ℝ, x^2 + 2 * a * x + a^2 + 1 = 0 → False) →
  (a ≥ 4 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_real_root_l255_255097


namespace fraction_ordering_l255_255025

theorem fraction_ordering : (4 / 17) < (6 / 25) ∧ (6 / 25) < (8 / 31) :=
by
  sorry

end fraction_ordering_l255_255025


namespace isosceles_triangle_base_angle_l255_255371

theorem isosceles_triangle_base_angle (α β γ : ℝ) 
  (h_triangle: α + β + γ = 180) 
  (h_isosceles: α = β ∨ α = γ ∨ β = γ) 
  (h_one_angle: α = 80 ∨ β = 80 ∨ γ = 80) : 
  (α = 50 ∨ β = 50 ∨ γ = 50) ∨ (α = 80 ∨ β = 80 ∨ γ = 80) :=
by 
  sorry

end isosceles_triangle_base_angle_l255_255371


namespace vertex_is_correct_l255_255653

-- Define the equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 + 10 * y + 4 * x + 9 = 0

-- The vertex of the parabola
def vertex_of_parabola : ℝ × ℝ := (4, -5)

-- The theorem stating that the given vertex satisfies the parabola equation
theorem vertex_is_correct : 
  parabola_equation vertex_of_parabola.1 vertex_of_parabola.2 :=
sorry

end vertex_is_correct_l255_255653


namespace handed_out_apples_l255_255702

def total_apples : ℤ := 96
def pies : ℤ := 9
def apples_per_pie : ℤ := 6
def apples_for_pies : ℤ := pies * apples_per_pie
def apples_handed_out : ℤ := total_apples - apples_for_pies

theorem handed_out_apples : apples_handed_out = 42 := by
  sorry

end handed_out_apples_l255_255702


namespace terminating_decimal_expansion_l255_255486

theorem terminating_decimal_expansion (a b : ℝ) :
  (13 / 200 = a / 10^b) → a = 52 ∧ b = 3 ∧ a / 10^b = 0.052 :=
by sorry

end terminating_decimal_expansion_l255_255486


namespace jose_total_caps_l255_255532

def initial_caps := 26
def additional_caps := 13
def total_caps := initial_caps + additional_caps

theorem jose_total_caps : total_caps = 39 :=
by
  sorry

end jose_total_caps_l255_255532


namespace find_complex_number_l255_255913

namespace ComplexProof

open Complex

def satisfies_conditions (z : ℂ) : Prop :=
  (z^2).im = 0 ∧ abs (z - I) = 1

theorem find_complex_number (z : ℂ) (h : satisfies_conditions z) : z = 0 ∨ z = 2 * I :=
sorry

end ComplexProof

end find_complex_number_l255_255913


namespace students_in_line_l255_255700

theorem students_in_line (T N : ℕ) (hT : T = 1) (h_btw : N = T + 4) (h_behind: ∃ k, k = 8) : T + (N - T) + 1 + 8 = 13 :=
by
  sorry

end students_in_line_l255_255700


namespace weights_less_than_90_l255_255597

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l255_255597


namespace total_wage_is_75_l255_255307

noncomputable def wages_total (man_wage : ℕ) : ℕ :=
  let men := 5
  let women := (5 : ℕ)
  let boys := 8
  (man_wage * men) + (man_wage * men) + (man_wage * men)

theorem total_wage_is_75
  (W : ℕ)
  (man_wage : ℕ := 5)
  (h1 : 5 = W) 
  (h2 : W = 8) 
  : wages_total man_wage = 75 := by
  sorry

end total_wage_is_75_l255_255307


namespace fault_line_movement_year_before_l255_255886

-- Define the total movement over two years
def total_movement : ℝ := 6.5

-- Define the movement during the past year
def past_year_movement : ℝ := 1.25

-- Define the movement the year before
def year_before_movement : ℝ := total_movement - past_year_movement

-- Prove that the fault line moved 5.25 inches the year before
theorem fault_line_movement_year_before : year_before_movement = 5.25 :=
  by  sorry

end fault_line_movement_year_before_l255_255886


namespace probability_at_least_one_consonant_l255_255869

def letters := ["k", "h", "a", "n", "t", "k", "a", "r"]
def consonants := ["k", "h", "n", "t", "r"]
def vowels := ["a", "a"]

def num_letters := 7
def num_consonants := 5
def num_vowels := 2

def probability_no_consonants : ℚ := (num_vowels / num_letters) * ((num_vowels - 1) / (num_letters - 1))

def complement_rule (p: ℚ) : ℚ := 1 - p

theorem probability_at_least_one_consonant :
  complement_rule probability_no_consonants = 20/21 :=
by
  sorry

end probability_at_least_one_consonant_l255_255869


namespace has_four_digits_l255_255149

def least_number_divisible (n: ℕ) : Prop := 
  n = 9600 ∧ 
  (∃ k1 k2 k3 k4: ℕ, n = 15 * k1 ∧ n = 25 * k2 ∧ n = 40 * k3 ∧ n = 75 * k4)

theorem has_four_digits : ∀ n: ℕ, least_number_divisible n → (Nat.digits 10 n).length = 4 :=
by
  intros n h
  sorry

end has_four_digits_l255_255149


namespace polynomials_same_type_l255_255580

-- Definitions based on the conditions
def variables_ab2 := {a, b}
def degree_ab2 := 3

-- Define the polynomial we are comparing with
def polynomial := -2 * a * b^2

-- Define the type equivalency of polynomials
def same_type (p1 p2 : Expr) : Prop :=
  (p1.variables = p2.variables) ∧ (p1.degree = p2.degree)

-- The statement to be proven
theorem polynomials_same_type : same_type polynomial ab2 :=
sorry

end polynomials_same_type_l255_255580


namespace sum_of_arithmetic_sequence_l255_255256

variables (a_n : Nat → Int) (S_n : Nat → Int)
variable (n : Nat)

-- Definitions based on given conditions:
def is_arithmetic_sequence (a_n : Nat → Int) : Prop :=
∀ n, a_n (n + 1) = a_n n + a_n 1 - a_n 0

def a_1 : Int := -2018

def arithmetic_sequence_sum (S_n : Nat → Int) (a_n : Nat → Int) (n : Nat) : Prop :=
S_n n = n * a_n 0 + (n * (n - 1) / 2 * (a_n 1 - a_n 0))

-- Given condition S_12 / 12 - S_10 / 10 = 2
def condition (S_n : Nat → Int) : Prop :=
S_n 12 / 12 - S_n 10 / 10 = 2

-- Goal: Prove S_2018 = -2018
theorem sum_of_arithmetic_sequence (a_n S_n : Nat → Int)
  (h1 : a_n 1 = -2018)
  (h2 : is_arithmetic_sequence a_n)
  (h3 : ∀ n, arithmetic_sequence_sum S_n a_n n)
  (h4 : condition S_n) :
  S_n 2018 = -2018 :=
sorry

end sum_of_arithmetic_sequence_l255_255256


namespace find_length_l255_255218

variables (w h A l : ℕ)
variable (A_eq : A = 164)
variable (w_eq : w = 4)
variable (h_eq : h = 3)

theorem find_length : 2 * l * w + 2 * l * h + 2 * w * h = A → l = 10 :=
by
  intros H
  rw [w_eq, h_eq, A_eq] at H
  linarith

end find_length_l255_255218


namespace sandwich_and_soda_cost_l255_255697

theorem sandwich_and_soda_cost:
  let sandwich_cost := 4
  let soda_cost := 1
  let num_sandwiches := 6
  let num_sodas := 10
  let total_cost := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  total_cost = 34 := 
by 
  sorry

end sandwich_and_soda_cost_l255_255697


namespace each_sibling_gets_13_pencils_l255_255963

theorem each_sibling_gets_13_pencils (colored_pencils black_pencils kept_pencils siblings : ℕ) 
  (h1 : colored_pencils = 14)
  (h2 : black_pencils = 35)
  (h3 : kept_pencils = 10)
  (h4 : siblings = 3) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by
  sorry

end each_sibling_gets_13_pencils_l255_255963


namespace max_value_X2_plus_2XY_plus_3Y2_l255_255007

theorem max_value_X2_plus_2XY_plus_3Y2 
  (x y : ℝ) 
  (h₁ : 0 < x) (h₂ : 0 < y) 
  (h₃ : x^2 - 2 * x * y + 3 * y^2 = 10) : 
  x^2 + 2 * x * y + 3 * y^2 ≤ 30 + 20 * Real.sqrt 3 :=
sorry

end max_value_X2_plus_2XY_plus_3Y2_l255_255007


namespace cost_two_cones_l255_255401

-- Definition for the cost of a single ice cream cone
def cost_one_cone : ℕ := 99

-- The theorem to prove the cost of two ice cream cones
theorem cost_two_cones : 2 * cost_one_cone = 198 := 
by 
  sorry

end cost_two_cones_l255_255401


namespace total_days_of_work_l255_255549

theorem total_days_of_work (r1 r2 r3 r4 : ℝ) (h1 : r1 = 1 / 12) (h2 : r2 = 1 / 8) (h3 : r3 = 1 / 24) (h4 : r4 = 1 / 16) : 
  (1 / (r1 + r2 + r3 + r4) = 3.2) :=
by 
  sorry

end total_days_of_work_l255_255549


namespace probability_at_least_40_cents_heads_l255_255825

noncomputable def value_of_heads (p n d q h : Bool) : Real :=
  (if p then 0.01 else 0) + (if n then 0.05 else 0) + (if d then 0.10 else 0) + (if q then 0.25 else 0) + (if h then 0.50 else 0)

theorem probability_at_least_40_cents_heads :
  let outcomes := {p : Bool, n : Bool, d : Bool, q : Bool, h : Bool}
  let favorable := (outcomes.filter $ λ (o : outcomes), value_of_heads o.p o.n o.d o.q o.h >= 0.40).size
  favorable / (outcomes.size : Real) = 19 / 32 :=
by
  sorry

end probability_at_least_40_cents_heads_l255_255825


namespace range_of_a_l255_255784

noncomputable def p (x : ℝ) : Prop := (3*x - 1)/(x - 2) ≤ 1
noncomputable def q (x a : ℝ) : Prop := x^2 - (2*a + 1)*x + a*(a + 1) < 0

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, ¬ q x a) → (¬ ∃ x : ℝ, ¬ p x) → -1/2 ≤ a ∧ a ≤ 1 :=
sorry

end range_of_a_l255_255784


namespace area_of_field_l255_255458

-- Define the variables and conditions
variables {L W : ℝ}

-- Given conditions
def length_side (L : ℝ) : Prop := L = 30
def fencing_equation (L W : ℝ) : Prop := L + 2 * W = 70

-- Prove the area of the field is 600 square feet
theorem area_of_field : length_side L → fencing_equation L W → (L * W = 600) :=
by
  intros hL hF
  rw [length_side, fencing_equation] at *
  sorry

end area_of_field_l255_255458


namespace modulus_of_z_l255_255502

noncomputable def i : ℂ := Complex.I

noncomputable def z : ℂ := sorry

theorem modulus_of_z 
  (hz : i * z = (1 - 2 * i)^2) : 
  Complex.abs z = 5 := by
  sorry

end modulus_of_z_l255_255502


namespace two_pairs_of_dice_probability_l255_255165

noncomputable def two_pairs_probability : ℚ :=
  5 / 36

theorem two_pairs_of_dice_probability :
  ∃ p : ℚ, p = two_pairs_probability := 
by 
  use 5 / 36
  sorry

end two_pairs_of_dice_probability_l255_255165


namespace factor_polynomial_l255_255766

-- Define the polynomial expression
def polynomial (x : ℝ) : ℝ := 60 * x + 45 + 9 * x ^ 2

-- Define the factored form of the polynomial
def factored_form (x : ℝ) : ℝ := 3 * (3 * x + 5) * (x + 3)

-- The statement of the problem to prove equivalence of the forms
theorem factor_polynomial : ∀ x : ℝ, polynomial x = factored_form x :=
by
  -- The actual proof is omitted and replaced by sorry
  sorry

end factor_polynomial_l255_255766


namespace correct_option_c_l255_255447

theorem correct_option_c (a : ℝ) : (-2 * a) ^ 3 = -8 * a ^ 3 :=
sorry

end correct_option_c_l255_255447


namespace white_cats_count_l255_255878

theorem white_cats_count (total_cats : ℕ) (black_cats : ℕ) (gray_cats : ℕ) (white_cats : ℕ)
  (h1 : total_cats = 15)
  (h2 : black_cats = 10)
  (h3 : gray_cats = 3)
  (h4 : total_cats = black_cats + gray_cats + white_cats) : 
  white_cats = 2 := 
  by
    -- proof or sorry here
    sorry

end white_cats_count_l255_255878


namespace distance_from_home_to_school_l255_255732

variable (t : ℕ) (D : ℕ)

-- conditions
def condition1 := 60 * (t - 10) = D
def condition2 := 50 * (t + 4) = D

-- the mathematical equivalent proof problem: proving the distance is 4200 given conditions
theorem distance_from_home_to_school :
  (∃ t, condition1 t 4200 ∧ condition2 t 4200) :=
  sorry

end distance_from_home_to_school_l255_255732


namespace solution_set_fraction_inequality_l255_255711

theorem solution_set_fraction_inequality (x : ℝ) : 
  (x + 1) / (x - 1) ≤ 0 ↔ -1 ≤ x ∧ x < 1 :=
sorry

end solution_set_fraction_inequality_l255_255711


namespace range_of_m_if_real_roots_specific_m_given_conditions_l255_255099

open Real

-- Define the quadratic equation and its conditions
def quadratic_eq (m : ℝ) (x : ℝ) : Prop := x ^ 2 - x + 2 * m - 4 = 0
def has_real_roots (m : ℝ) : Prop := ∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2

-- Proof that m ≤ 17/8 if the quadratic equation has real roots
theorem range_of_m_if_real_roots (m : ℝ) : has_real_roots m → m ≤ 17 / 8 := 
sorry

-- Define a condition on the roots
def roots_condition (x1 x2 m : ℝ) : Prop := (x1 - 3) * (x2 - 3) = m ^ 2 - 1

-- Proof of specific m when roots condition is given
theorem specific_m_given_conditions (m : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_eq m x1 ∧ quadratic_eq m x2 ∧ roots_condition x1 x2 m) → m = -1 :=
sorry

end range_of_m_if_real_roots_specific_m_given_conditions_l255_255099


namespace not_in_sequence_l255_255906

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the sequence property
def sequence_property (a b : ℕ) : Prop :=
  b = a + sum_of_digits a

-- Main theorem
theorem not_in_sequence (n : ℕ) (h : n = 793210041) : 
  ¬ (∃ a : ℕ, sequence_property a n) :=
by
  sorry

end not_in_sequence_l255_255906


namespace problem_a_lt_2b_l255_255516

theorem problem_a_lt_2b (a b : ℝ) (h : 2^a + log 2 a = 4^b + 2 * log 4 b) : a < 2 * b := 
sorry

end problem_a_lt_2b_l255_255516


namespace second_player_win_strategy_l255_255137

theorem second_player_win_strategy:
  ∃ strategy : (ℕ → ℕ) → ℕ, 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 1000 → 
    (strategy n + n = 1001) ∧
    (strategy n - n) % 13 = 0) :=
sorry

end second_player_win_strategy_l255_255137


namespace circle_equation_correct_l255_255418

-- Define the given elements: center and radius
def center : (ℝ × ℝ) := (1, -1)
def radius : ℝ := 2

-- Define the equation of the circle with the given center and radius
def circle_eqn (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = radius^2

-- Prove that the equation of the circle holds with the given center and radius
theorem circle_equation_correct : 
  ∀ x y : ℝ, circle_eqn x y ↔ (x - 1)^2 + (y + 1)^2 = 4 := 
by
  sorry

end circle_equation_correct_l255_255418


namespace derivative_f_minus_4f_at_1_l255_255957

-- Define g(x) to be f(x) - f(2x)
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f(x) - f(2 * x)

-- Given
variables {f : ℝ → ℝ}
variable (h1 : deriv (λ x, f(x) - f(2 * x)) 1 = 5)
variable (h2 : deriv (λ x, f(x) - f(2 * x)) 2 = 7)

-- Prove
theorem derivative_f_minus_4f_at_1 : deriv (λ x, f(x) - f(4 * x)) 1 = 19 :=
by
  sorry

end derivative_f_minus_4f_at_1_l255_255957


namespace non_neg_sum_sq_inequality_l255_255928

theorem non_neg_sum_sq_inequality (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
sorry

end non_neg_sum_sq_inequality_l255_255928


namespace original_denominator_is_15_l255_255318

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l255_255318


namespace initial_balloons_correct_l255_255061

-- Define the variables corresponding to the conditions given in the problem
def boy_balloon_count := 3
def girl_balloon_count := 12
def balloons_sold := boy_balloon_count + girl_balloon_count
def balloons_remaining := 21

-- State the theorem asserting the initial number of balloons
theorem initial_balloons_correct :
  balloons_sold + balloons_remaining = 36 := sorry

end initial_balloons_correct_l255_255061


namespace chris_first_day_breath_l255_255332

theorem chris_first_day_breath (x : ℕ) (h1 : x + 10 = 20) : x = 10 :=
by
  sorry

end chris_first_day_breath_l255_255332


namespace bobby_pancakes_left_l255_255329

def total_pancakes : ℕ := 21
def pancakes_eaten_by_bobby : ℕ := 5
def pancakes_eaten_by_dog : ℕ := 7

theorem bobby_pancakes_left : total_pancakes - (pancakes_eaten_by_bobby + pancakes_eaten_by_dog) = 9 :=
  by
  sorry

end bobby_pancakes_left_l255_255329


namespace smallest_number_of_seats_required_l255_255875

theorem smallest_number_of_seats_required (total_chairs : ℕ) (condition : ∀ (N : ℕ), ∀ (seating : Finset ℕ),
  seating.card = N → (∀ x ∈ seating, (x + 1) % total_chairs ∈ seating ∨ (x + total_chairs - 1) % total_chairs ∈ seating)) :
  total_chairs = 100 → ∃ N : ℕ, N = 20 :=
by
  intros
  sorry

end smallest_number_of_seats_required_l255_255875


namespace quadratic_expression_odd_quadratic_expression_not_square_l255_255004

theorem quadratic_expression_odd (n : ℕ) : 
  (n^2 + n + 1) % 2 = 1 := 
by sorry

theorem quadratic_expression_not_square (n : ℕ) : 
  ¬ ∃ (m : ℕ), m^2 = n^2 + n + 1 := 
by sorry

end quadratic_expression_odd_quadratic_expression_not_square_l255_255004


namespace min_value_function_l255_255916

theorem min_value_function (x y : ℝ) (hx : x > 1) (hy : y > 1) : 
  (∀ x y : ℝ, x > 1 ∧ y > 1 → (min ((x^2 + y) / (y^2 - 1) + (y^2 + x) / (x^2 - 1)) = 8 / 3)) := 
sorry

end min_value_function_l255_255916


namespace finite_ring_identity_l255_255813

variable {A : Type} [Ring A] [Fintype A]
variables (a b : A)

theorem finite_ring_identity (h : (ab - 1) * b = 0) : b * (ab - 1) = 0 :=
sorry

end finite_ring_identity_l255_255813


namespace sprinter_time_no_wind_l255_255607

theorem sprinter_time_no_wind :
  ∀ (x y : ℝ), (90 / (x + y) = 10) → (70 / (x - y) = 10) → x = 8 * y → 100 / x = 12.5 :=
by
  intros x y h1 h2 h3
  sorry

end sprinter_time_no_wind_l255_255607


namespace inequality_of_exponential_log_l255_255513

theorem inequality_of_exponential_log (a b : ℝ) (h : 2^a + log 2 a = 4^b + 2 * log 4 b) : a < 2 * b := 
by
  sorry

end inequality_of_exponential_log_l255_255513


namespace cryptarithm_solution_l255_255530

theorem cryptarithm_solution :
  ∃ A B C D E F G H J : ℕ,
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ J ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ J ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ J ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ J ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ J ∧
  F ≠ G ∧ F ≠ H ∧ F ≠ J ∧
  G ≠ H ∧ G ≠ J ∧
  H ≠ J ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ J < 10 ∧
  (10 * A + B) * (10 * C + A) = 100 * D + 10 * E + B ∧
  (10 * F + C) - (10 * D + G) = D ∧
  (10 * E + G) + (10 * H + J) = 100 * A + 10 * A + G ∧
  A = 1 ∧ B = 7 ∧ C = 2 ∧ D = 3 ∧ E = 5 ∧ F = 4 ∧ G = 9 ∧ H = 6 ∧ J = 0 :=
by
  sorry

end cryptarithm_solution_l255_255530


namespace suraj_average_after_17th_innings_l255_255028

theorem suraj_average_after_17th_innings (A : ℕ) :
  (16 * A + 92) / 17 = A + 4 -> A + 4 = 28 := 
by 
  sorry

end suraj_average_after_17th_innings_l255_255028


namespace scientific_notation_of_190_million_l255_255143

theorem scientific_notation_of_190_million : (190000000 : ℝ) = 1.9 * 10^8 :=
sorry

end scientific_notation_of_190_million_l255_255143


namespace kids_go_to_camp_l255_255677

variable (total_kids staying_home going_to_camp : ℕ)

theorem kids_go_to_camp (h1 : total_kids = 313473) (h2 : staying_home = 274865) (h3 : going_to_camp = total_kids - staying_home) :
  going_to_camp = 38608 :=
by
  sorry

end kids_go_to_camp_l255_255677


namespace coins_value_percentage_l255_255722

theorem coins_value_percentage :
  let penny_value := 1
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_value_cents := (1 * penny_value) + (2 * nickel_value) + (1 * dime_value) + (2 * quarter_value)
  (total_value_cents / 100) * 100 = 71 :=
by
  sorry

end coins_value_percentage_l255_255722


namespace fraction_identity_l255_255103

theorem fraction_identity (m n r t : ℚ) 
  (h₁ : m / n = 3 / 5) 
  (h₂ : r / t = 8 / 9) :
  (3 * m^2 * r - n * t^2) / (5 * n * t^2 - 9 * m^2 * r) = -1 := 
by
  sorry

end fraction_identity_l255_255103


namespace ratio_area_square_circle_eq_pi_l255_255562

theorem ratio_area_square_circle_eq_pi
  (a r : ℝ)
  (h : 4 * a = 4 * π * r) :
  (a^2 / (π * r^2)) = π := by
  sorry

end ratio_area_square_circle_eq_pi_l255_255562


namespace thalassa_population_2050_l255_255340

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050_l255_255340


namespace age_difference_is_24_l255_255333

theorem age_difference_is_24 (d f : ℕ) (h1 : d = f / 9) (h2 : f + 1 = 7 * (d + 1)) : f - d = 24 := sorry

end age_difference_is_24_l255_255333


namespace solve_inequality_l255_255637

theorem solve_inequality (x : ℝ) :
  (abs ((6 - x) / 4) < 3) ∧ (2 ≤ x) ↔ (2 ≤ x) ∧ (x < 18) := 
by
  sorry

end solve_inequality_l255_255637


namespace sequence_probability_correct_l255_255048

noncomputable def m : ℕ := 377
noncomputable def n : ℕ := 4096

theorem sequence_probability_correct :
  let m := 377
  let n := 4096
  (m.gcd n = 1) ∧ (m + n = 4473) := 
by
  -- Proof requires the given equivalent statement in Lean, so include here
  sorry

end sequence_probability_correct_l255_255048


namespace mean_of_remaining_number_is_2120_l255_255552

theorem mean_of_remaining_number_is_2120 (a1 a2 a3 a4 a5 a6 : ℕ) 
    (h1 : a1 = 1451) (h2 : a2 = 1723) (h3 : a3 = 1987) (h4 : a4 = 2056) 
    (h5 : a5 = 2191) (h6 : a6 = 2212) 
    (mean_five : (a1 + a2 + a3 + a4 + a5) = 9500):
-- Prove that the mean of the remaining number a6 is 2120
  (a6 = 2120) :=
by
  -- Placeholder for proof
  sorry

end mean_of_remaining_number_is_2120_l255_255552


namespace continuous_at_4_l255_255278

noncomputable def f (x : ℝ) := 3 * x^2 - 3

theorem continuous_at_4 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 4| < δ → |f x - f 4| < ε :=
by
  sorry

end continuous_at_4_l255_255278


namespace books_combination_l255_255605

theorem books_combination :
  (Nat.choose 15 3) = 455 := 
sorry

end books_combination_l255_255605


namespace contrapositive_statement_l255_255927

-- Conditions: x and y are real numbers
variables (x y : ℝ)

-- Contrapositive statement: If x ≠ 0 or y ≠ 0, then x^2 + y^2 ≠ 0
theorem contrapositive_statement (hx : x ≠ 0 ∨ y ≠ 0) : x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_statement_l255_255927


namespace range_of_a_l255_255094

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * x - 3 > 0 → x > a) ↔ a ≥ 1 :=
by
  sorry

end range_of_a_l255_255094


namespace find_constants_PQR_l255_255342

theorem find_constants_PQR :
  ∃ P Q R : ℝ, 
    (6 * x + 2) / ((x - 4) * (x - 2) ^ 3) = P / (x - 4) + Q / (x - 2) + R / (x - 2) ^ 3 :=
by
  use 13 / 4
  use -6.5
  use -7
  sorry

end find_constants_PQR_l255_255342


namespace fisherman_daily_earnings_l255_255421

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l255_255421


namespace factor_count_x9_minus_x_l255_255728

theorem factor_count_x9_minus_x :
  ∃ (factors : List (Polynomial ℤ)), x^9 - x = factors.prod ∧ factors.length = 5 :=
sorry

end factor_count_x9_minus_x_l255_255728


namespace find_x_l255_255521

theorem find_x (x : ℤ) (h : 7 * x - 18 = 66) : x = 12 :=
  sorry

end find_x_l255_255521


namespace terminating_decimal_expansion_l255_255485

theorem terminating_decimal_expansion (a b : ℝ) :
  (13 / 200 = a / 10^b) → a = 52 ∧ b = 3 ∧ a / 10^b = 0.052 :=
by sorry

end terminating_decimal_expansion_l255_255485


namespace calculate_gas_volumes_l255_255735

variable (gas_volume_western : ℝ) 
variable (total_gas_volume_non_western : ℝ)
variable (population_non_western : ℝ)
variable (total_gas_percentage_russia : ℝ)
variable (gas_volume_russia : ℝ)
variable (population_russia : ℝ)

theorem calculate_gas_volumes 
(h_western : gas_volume_western = 21428)
(h_non_western : total_gas_volume_non_western = 185255)
(h_population_non_western : population_non_western = 6.9)
(h_percentage_russia : total_gas_percentage_russia = 68.0)
(h_gas_volume_russia : gas_volume_russia = 30266.9)
(h_population_russia : population_russia = 0.147)
: 
  (total_gas_volume_non_western / population_non_western = 26848.55) ∧ 
  (gas_volume_russia / population_russia ≈ 302790.13) := 
  sorry

end calculate_gas_volumes_l255_255735


namespace union_A_B_l255_255226

noncomputable def A : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def B : Set ℝ := {x | x^2 - 4 = 0}

theorem union_A_B : A ∪ B = {-3, -2, 2} := by
  sorry

end union_A_B_l255_255226


namespace eccentricity_of_hyperbola_l255_255288

theorem eccentricity_of_hyperbola :
  let a_squared := (2 : ℝ)
  let b_squared := (1 : ℝ)
  let c_squared := a_squared + b_squared
  let c := Real.sqrt c_squared
  let a := Real.sqrt a_squared
  let e := c / a
  e = Real.sqrt 6 / 2 := by
  sorry

end eccentricity_of_hyperbola_l255_255288


namespace units_digit_of_x_l255_255900

theorem units_digit_of_x (p x : ℕ): 
  (p * x = 32 ^ 10) → 
  (p % 10 = 6) → 
  (x % 4 = 0) → 
  (x % 10 = 1) :=
by
  sorry

end units_digit_of_x_l255_255900


namespace percentage_increase_first_year_l255_255839

-- Assume the original price of the painting is P and the percentage increase during the first year is X
variable {P : ℝ} (X : ℝ)

-- Condition: The price decreases by 15% during the second year
def condition_decrease (price : ℝ) : ℝ := price * 0.85

-- Condition: The price at the end of the 2-year period was 93.5% of the original price
axiom condition_end_price : ∀ (P : ℝ), (P + (X/100) * P) * 0.85 = 0.935 * P

-- Proof problem: What was the percentage increase during the first year?
theorem percentage_increase_first_year : X = 10 :=
by 
  sorry

end percentage_increase_first_year_l255_255839


namespace not_obtain_other_than_given_set_l255_255666

theorem not_obtain_other_than_given_set : 
  ∀ (x : ℝ), x = 1 → 
  ∃ (n : ℕ → ℝ), (n 0 = 1) ∧ 
  (∀ k, n (k + 1) = n k + 1 ∨ n (k + 1) = -1 / n k) ∧
  (x = -2 ∨ x = 1/2 ∨ x = 5/3 ∨ x = 7) → 
  ∃ k, x = n k :=
sorry

end not_obtain_other_than_given_set_l255_255666


namespace divides_floor_factorial_div_l255_255283

theorem divides_floor_factorial_div {m n : ℕ} (h1 : 1 < m) (h2 : m < n + 2) (h3 : 3 < n) :
  (m - 1) ∣ (n! / m) :=
sorry

end divides_floor_factorial_div_l255_255283


namespace find_number_of_children_l255_255395

-- Definitions based on conditions
def decorative_spoons : Nat := 2
def new_set_large_spoons : Nat := 10
def new_set_tea_spoons : Nat := 15
def total_spoons : Nat := 39
def spoons_per_child : Nat := 3
def new_set_spoons := new_set_large_spoons + new_set_tea_spoons

-- The main statement to prove the number of children
theorem find_number_of_children (C : Nat) :
  3 * C + decorative_spoons + new_set_spoons = total_spoons → C = 4 :=
by
  -- Proof would go here
  sorry

end find_number_of_children_l255_255395


namespace cos_alpha_minus_pi_l255_255500

theorem cos_alpha_minus_pi (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi) (h3 : 3 * Real.sin (2 * α) = Real.sin α) : 
  Real.cos (α - Real.pi) = -1/6 := 
by
  sorry

end cos_alpha_minus_pi_l255_255500


namespace carlos_biked_more_than_daniel_l255_255668

-- Definitions modeled from conditions
def distance_carlos : ℕ := 108
def distance_daniel : ℕ := 90
def time_hours : ℕ := 6

-- Lean statement to prove the difference in distance
theorem carlos_biked_more_than_daniel : distance_carlos - distance_daniel = 18 := 
  by 
    sorry

end carlos_biked_more_than_daniel_l255_255668


namespace radius_increase_l255_255555

theorem radius_increase (C1 C2 : ℝ) (π : ℝ) (hC1 : C1 = 40) (hC2 : C2 = 50) (hπ : π > 0) : 
  (C2 - C1) / (2 * π) = 5 / π := 
sorry

end radius_increase_l255_255555


namespace proportional_function_y_decreases_l255_255109

theorem proportional_function_y_decreases (k : ℝ) (h₀ : k ≠ 0) (h₁ : (4 : ℝ) * k = -1) :
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → k * x₁ > k * x₂ :=
by 
  sorry

end proportional_function_y_decreases_l255_255109


namespace prairie_total_area_l255_255185

theorem prairie_total_area :
  let dust_covered := 64535
  let untouched := 522
  (dust_covered + untouched) = 65057 :=
by {
  let dust_covered := 64535
  let untouched := 522
  trivial
}

end prairie_total_area_l255_255185


namespace angles_relation_l255_255022

/-- Given angles α and β from two right-angled triangles in a 3x3 grid such that α + β = 90°,
    prove that 2α + β = 90°. -/
theorem angles_relation (α β : ℝ) (h1 : α + β = 90) : 2 * α + β = 90 := by
  sorry

end angles_relation_l255_255022


namespace value_of_a_set_of_x_l255_255098

open Real

noncomputable def f (x a : ℝ) : ℝ := sin (x + π / 6) + sin (x - π / 6) + cos x + a

theorem value_of_a : ∀ a, (∀ x, f x a ≤ 1) → a = -1 :=
sorry

theorem set_of_x (a : ℝ) (k : ℤ) : a = -1 →
  {x : ℝ | f x a = 0} = {x | ∃ k : ℤ, x = 2 * k * π ∨ x = 2 * k * π + 2 * π / 3} :=
sorry

end value_of_a_set_of_x_l255_255098


namespace simplest_form_expression_l255_255860

theorem simplest_form_expression (x y a : ℤ) : 
  (∃ (E : ℚ → Prop), (E (1/3) ∨ E (1/(x-2)) ∨ E ((x^2 * y) / (2*x)) ∨ E (2*a / 8)) → (E (1/(x-2)) ↔ E (1/(x-2)))) :=
by 
  sorry

end simplest_form_expression_l255_255860


namespace factorize_m_cubed_minus_16m_l255_255210

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l255_255210


namespace rectangle_length_increase_l255_255704

variable (L B : ℝ) -- Original length and breadth
variable (A : ℝ) -- Original area
variable (p : ℝ) -- Percentage increase in length
variable (A' : ℝ) -- New area

theorem rectangle_length_increase (hA : A = L * B) 
  (hp : L' = L + (p / 100) * L) 
  (hB' : B' = B * 0.9) 
  (hA' : A' = 1.035 * A)
  (hl' : L' = (1 + (p / 100)) * L)
  (hb_length : L' * B' = A') :
  p = 15 :=
by
  sorry

end rectangle_length_increase_l255_255704


namespace distribution_1_distribution_2_distribution_3_l255_255156

-- 1. Defining the problem statement for the first condition.
theorem distribution_1 : nat.choose 6 2 * nat.choose 4 2 * 1 = 90 := 
sorry

-- 2. Defining the problem statement for the second condition.
theorem distribution_2 : nat.choose 6 1 * nat.choose 5 2 * 1 = 60 := 
sorry

-- 3. Defining the problem statement for the third condition.
theorem distribution_3 : nat.choose 6 1 * nat.choose 5 2 * 1 * nat.factorial 3 = 360 := 
sorry

end distribution_1_distribution_2_distribution_3_l255_255156


namespace total_broken_marbles_l255_255639

theorem total_broken_marbles (marbles_set1 marbles_set2 : ℕ) 
  (percentage_broken_set1 percentage_broken_set2 : ℚ) 
  (h1 : marbles_set1 = 50) 
  (h2 : percentage_broken_set1 = 0.1) 
  (h3 : marbles_set2 = 60) 
  (h4 : percentage_broken_set2 = 0.2) : 
  (marbles_set1 * percentage_broken_set1 + marbles_set2 * percentage_broken_set2 = 17) := 
by 
  sorry

end total_broken_marbles_l255_255639


namespace find_certain_number_l255_255363

noncomputable def certain_number_is_square (n : ℕ) (x : ℕ) : Prop :=
  ∃ (y : ℕ), x * n = y * y

theorem find_certain_number : ∃ x, certain_number_is_square 3 x :=
by 
  use 1
  unfold certain_number_is_square
  use 3
  sorry

end find_certain_number_l255_255363


namespace exterior_angle_of_polygon_l255_255365

theorem exterior_angle_of_polygon (n : ℕ) (h₁ : (n - 2) * 180 = 1800) (h₂ : n > 2) :
  360 / n = 30 := by
    sorry

end exterior_angle_of_polygon_l255_255365


namespace convert_BFACE_to_decimal_l255_255468

def hex_BFACE : ℕ := 11 * 16^4 + 15 * 16^3 + 10 * 16^2 + 12 * 16^1 + 14 * 16^0

theorem convert_BFACE_to_decimal : hex_BFACE = 785102 := by
  sorry

end convert_BFACE_to_decimal_l255_255468


namespace malcolm_red_lights_bought_l255_255134

-- Define the problem's parameters and conditions
variable (R : ℕ) (B : ℕ := 3 * R) (G : ℕ := 6)
variable (initial_white_lights : ℕ := 59) (remaining_colored_lights : ℕ := 5)

-- The total number of colored lights that he still needs to replace the white lights
def total_colored_lights_needed : ℕ := initial_white_lights - remaining_colored_lights

-- Total colored lights bought so far
def total_colored_lights_bought : ℕ := R + B + G

-- The main theorem to prove that Malcolm bought 12 red lights
theorem malcolm_red_lights_bought (h : total_colored_lights_bought = total_colored_lights_needed) :
  R = 12 := by
  sorry

end malcolm_red_lights_bought_l255_255134


namespace problem_statement_l255_255086

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem problem_statement :
  (M ∩ N) = N :=
by
  sorry

end problem_statement_l255_255086


namespace probability_of_death_each_month_l255_255937

-- Defining the variables and expressions used in conditions
def p : ℝ := 0.1
def N : ℝ := 400
def surviving_after_3_months : ℝ := 291.6

-- The main theorem to be proven
theorem probability_of_death_each_month (prob : ℝ) :
  (N * (1 - prob)^3 = surviving_after_3_months) → (prob = p) :=
by
  sorry

end probability_of_death_each_month_l255_255937


namespace division_multiplication_l255_255893

theorem division_multiplication : (0.25 / 0.005) * 2 = 100 := 
by 
  sorry

end division_multiplication_l255_255893


namespace car_quotient_div_15_l255_255544

/-- On a straight, one-way, single-lane highway, cars all travel at the same speed
    and obey a modified safety rule: the distance from the back of the car ahead
    to the front of the car behind is exactly two car lengths for each 20 kilometers
    per hour of speed. A sensor by the road counts the number of cars that pass in
    one hour. Each car is 5 meters long. 
    Let N be the maximum whole number of cars that can pass the sensor in one hour.
    Prove that when N is divided by 15, the quotient is 266. -/
theorem car_quotient_div_15 
  (speed : ℕ) 
  (d : ℕ) 
  (sensor_time : ℕ) 
  (car_length : ℕ)
  (N : ℕ)
  (h1 : ∀ m, speed = 20 * m)
  (h2 : d = 2 * car_length)
  (h3 : car_length = 5)
  (h4 : sensor_time = 1)
  (h5 : N = 4000) : 
  N / 15 = 266 := 
sorry

end car_quotient_div_15_l255_255544


namespace speed_of_canoe_downstream_l255_255182

-- Definition of the problem conditions
def speed_of_canoe_in_still_water (V_c : ℝ) (V_s : ℝ) (upstream_speed : ℝ) : Prop :=
  V_c - V_s = upstream_speed

def speed_of_stream (V_s : ℝ) : Prop :=
  V_s = 4

-- The statement we want to prove
theorem speed_of_canoe_downstream (V_c V_s : ℝ) (upstream_speed : ℝ) 
  (h1 : speed_of_canoe_in_still_water V_c V_s upstream_speed)
  (h2 : speed_of_stream V_s)
  (h3 : upstream_speed = 4) :
  V_c + V_s = 12 :=
by
  sorry

end speed_of_canoe_downstream_l255_255182


namespace lateral_surface_area_ratio_l255_255013

theorem lateral_surface_area_ratio (r h : ℝ) :
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  cylinder_area / cone_area = 2 :=
by
  let cylinder_area := 2 * Real.pi * r * h
  let cone_area := (2 * Real.pi * r * h) / 2
  sorry

end lateral_surface_area_ratio_l255_255013


namespace all_three_pets_l255_255366

-- Definitions of the given conditions
def total_students : ℕ := 40
def dog_owners : ℕ := 20
def cat_owners : ℕ := 13
def other_pet_owners : ℕ := 8
def no_pets : ℕ := 7

-- Definitions from Venn diagram
def dogs_only : ℕ := 12
def cats_only : ℕ := 3
def other_pets_only : ℕ := 2

-- Intersection variables
variables (a b c d : ℕ)

-- Translated problem
theorem all_three_pets :
  dogs_only + cats_only + other_pets_only + a + b + c + d = total_students - no_pets ∧
  dogs_only + a + c + d = dog_owners ∧
  cats_only + a + b + d = cat_owners ∧
  other_pets_only + b + c + d = other_pet_owners ∧
  d = 2 :=
sorry

end all_three_pets_l255_255366


namespace calculate_two_times_square_root_squared_l255_255891

theorem calculate_two_times_square_root_squared : 2 * (Real.sqrt 50625) ^ 2 = 101250 := by
  sorry

end calculate_two_times_square_root_squared_l255_255891


namespace solve_system_l255_255630

theorem solve_system (x₁ x₂ x₃ : ℝ) (h₁ : 2 * x₁^2 / (1 + x₁^2) = x₂) (h₂ : 2 * x₂^2 / (1 + x₂^2) = x₃) (h₃ : 2 * x₃^2 / (1 + x₃^2) = x₁) :
  (x₁ = 0 ∧ x₂ = 0 ∧ x₃ = 0) ∨ (x₁ = 1 ∧ x₂ = 1 ∧ x₃ = 1) :=
sorry

end solve_system_l255_255630


namespace sum_inequality_l255_255387

theorem sum_inequality (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 1) :
  (1 / (b * (a + b))) + (1 / (c * (b + c))) + (1 / (a * (c + a))) ≥ 3 / 2 :=
sorry

end sum_inequality_l255_255387


namespace ab_value_l255_255719

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 107) : a * b = 25.3125 :=
by
  sorry

end ab_value_l255_255719


namespace proof_problem_l255_255142

def g : ℕ → ℕ := sorry
def g_inv : ℕ → ℕ := sorry

axiom g_inv_is_inverse : ∀ y, g (g_inv y) = y ∧ g_inv (g y) = y
axiom g_4_eq_6 : g 4 = 6
axiom g_6_eq_2 : g 6 = 2
axiom g_3_eq_7 : g 3 = 7

theorem proof_problem :
  g_inv (g_inv 7 + g_inv 6) = 3 :=
by
  sorry

end proof_problem_l255_255142


namespace bertha_daughters_and_granddaughters_have_no_daughters_l255_255328

def total_daughters_and_granddaughters (daughters granddaughters : Nat) : Nat :=
daughters + granddaughters

def no_daughters (bertha_daughters bertha_granddaughters : Nat) : Nat :=
bertha_daughters + bertha_granddaughters

theorem bertha_daughters_and_granddaughters_have_no_daughters :
  (bertha_daughters : Nat) →
  (daughters_with_6_daughters : Nat) →
  (granddaughters : Nat) →
  (total_daughters_and_granddaughters bertha_daughters granddaughters = 30) →
  bertha_daughters = 6 →
  granddaughters = 6 * daughters_with_6_daughters →
  no_daughters (bertha_daughters - daughters_with_6_daughters) granddaughters = 26 :=
by
  intros bertha_daughters daughters_with_6_daughters granddaughters h_total h_bertha h_granddaughters
  sorry

end bertha_daughters_and_granddaughters_have_no_daughters_l255_255328


namespace total_area_rectABCD_l255_255317

theorem total_area_rectABCD (BF CF : ℝ) (X Y : ℝ)
  (h1 : BF = 3 * CF)
  (h2 : 3 * X - Y - (X - Y) = 96)
  (h3 : X + 3 * X = 192) :
  X + 3 * X = 192 :=
by
  sorry

end total_area_rectABCD_l255_255317


namespace pages_for_thirty_dollars_l255_255125

-- Problem Statement Definitions
def costPerCopy := 4 -- cents
def pagesPerCopy := 2 -- pages
def totalCents := 3000 -- cents
def totalPages := 1500 -- pages

-- Theorem: Calculating the number of pages for a given cost.
theorem pages_for_thirty_dollars (c_per_copy : ℕ) (p_per_copy : ℕ) (t_cents : ℕ) (t_pages : ℕ) : 
  c_per_copy = 4 → p_per_copy = 2 → t_cents = 3000 → t_pages = 1500 := by
  intros h_cpc h_ppc h_tc
  sorry

end pages_for_thirty_dollars_l255_255125


namespace pink_highlighters_count_l255_255113

-- Define the necessary constants and types
def total_highlighters : ℕ := 12
def yellow_highlighters : ℕ := 2
def blue_highlighters : ℕ := 4

-- We aim to prove that the number of pink highlighters is 6
theorem pink_highlighters_count : ∃ (pink_highlighters : ℕ), 
  pink_highlighters = total_highlighters - (yellow_highlighters + blue_highlighters) ∧
  pink_highlighters = 6 :=
by
  sorry

end pink_highlighters_count_l255_255113


namespace find_C_l255_255290

noncomputable def A_annual_income : ℝ := 403200.0000000001
noncomputable def A_monthly_income : ℝ := A_annual_income / 12 -- 33600.00000000001

noncomputable def x : ℝ := A_monthly_income / 5 -- 6720.000000000002

noncomputable def C : ℝ := (2 * x) / 1.12 -- should be 12000.000000000004

theorem find_C : C = 12000.000000000004 := 
by sorry

end find_C_l255_255290


namespace new_person_weight_is_55_l255_255287

variable (W : ℝ) -- Total weight of the original 8 people
variable (new_person_weight : ℝ) -- Weight of the new person
variable (avg_increase : ℝ := 2.5) -- The average weight increase

-- Given conditions
def condition (W new_person_weight : ℝ) : Prop :=
  new_person_weight = W + (8 * avg_increase) + 35 - W

-- The proof statement
theorem new_person_weight_is_55 (W : ℝ) : (new_person_weight = 55) :=
by
  sorry

end new_person_weight_is_55_l255_255287


namespace find_k_l255_255682

noncomputable def series_sum (k : ℝ) : ℝ :=
  ∑' n, (7 * n + 2) / k^n

theorem find_k (k : ℝ) (h : k > 1) (hk : series_sum k = 5) : 
  k = (7 + Real.sqrt 14) / 5 :=
by 
  sorry

end find_k_l255_255682


namespace tobias_mowed_four_lawns_l255_255850

-- Let’s define the conditions
def shoe_cost : ℕ := 95
def allowance_per_month : ℕ := 5
def savings_months : ℕ := 3
def lawn_mowing_charge : ℕ := 15
def shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def num_driveways_shoveled : ℕ := 5

-- Total money Tobias had before buying the shoes
def total_money : ℕ := shoe_cost + change_after_purchase

-- Money saved from allowance
def money_from_allowance : ℕ := allowance_per_month * savings_months

-- Money earned from shoveling driveways
def money_from_shoveling : ℕ := shoveling_charge * num_driveways_shoveled

-- Money earned from mowing lawns
def money_from_mowing : ℕ := total_money - money_from_allowance - money_from_shoveling

-- Number of lawns mowed
def num_lawns_mowed : ℕ := money_from_mowing / lawn_mowing_charge

-- The theorem stating the number of lawns mowed is 4
theorem tobias_mowed_four_lawns : num_lawns_mowed = 4 :=
by
  sorry

end tobias_mowed_four_lawns_l255_255850


namespace each_child_ate_3_jellybeans_l255_255987

-- Define the given conditions
def total_jellybeans : ℕ := 100
def total_kids : ℕ := 24
def sick_kids : ℕ := 2
def leftover_jellybeans : ℕ := 34

-- Calculate the number of kids who attended
def attending_kids : ℕ := total_kids - sick_kids

-- Calculate the total jellybeans eaten
def total_jellybeans_eaten : ℕ := total_jellybeans - leftover_jellybeans

-- Calculate the number of jellybeans each child ate
def jellybeans_per_child : ℕ := total_jellybeans_eaten / attending_kids

theorem each_child_ate_3_jellybeans : jellybeans_per_child = 3 :=
by sorry

end each_child_ate_3_jellybeans_l255_255987


namespace john_reading_time_l255_255264

theorem john_reading_time:
  let weekday_hours_moses := 1.5
  let weekday_rate_moses := 30
  let saturday_hours_moses := 2
  let saturday_rate_moses := 40
  let pages_moses := 450
  let weekday_hours_rest := 1.5
  let weekday_rate_rest := 45
  let saturday_hours_rest := 2.5
  let saturday_rate_rest := 60
  let pages_rest := 2350
  let weekdays_per_week := 5
  let saturdays_per_week := 1
  let total_pages_per_week_moses := (weekday_hours_moses * weekday_rate_moses * weekdays_per_week) + 
                                    (saturday_hours_moses * saturday_rate_moses * saturdays_per_week)
  let total_pages_per_week_rest := (weekday_hours_rest * weekday_rate_rest * weekdays_per_week) + 
                                   (saturday_hours_rest * saturday_rate_rest * saturdays_per_week)
  let weeks_moses := (pages_moses / total_pages_per_week_moses).ceil
  let weeks_rest := (pages_rest / total_pages_per_week_rest).ceil
  let total_weeks := weeks_moses + weeks_rest
  total_weeks = 7 :=
by
  -- placeholders for the proof steps.
  sorry

end john_reading_time_l255_255264


namespace women_decreased_by_3_l255_255675

noncomputable def initial_men := 12
noncomputable def initial_women := 27

theorem women_decreased_by_3 
  (ratio_men_women : 4 / 5 = initial_men / initial_women)
  (men_after_enter : initial_men + 2 = 14)
  (women_after_leave : initial_women - 3 = 24) :
  (24 - 27 = -3) :=
by
  sorry

end women_decreased_by_3_l255_255675


namespace alpha_less_than_60_degrees_l255_255033

theorem alpha_less_than_60_degrees
  (R r : ℝ)
  (b c : ℝ)
  (α : ℝ)
  (h1 : b * c = 8 * R * r) :
  α < 60 := sorry

end alpha_less_than_60_degrees_l255_255033


namespace diophantine_eq_unique_solutions_l255_255547

theorem diophantine_eq_unique_solutions (x y : ℕ) (hx_positive : x > 0) (hy_positive : y > 0) :
  x^y = y^x + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) :=
by
  sorry

end diophantine_eq_unique_solutions_l255_255547


namespace exists_m_in_range_l255_255364

theorem exists_m_in_range :
  ∃ m : ℝ, 0 ≤ m ∧ m < 1 ∧ ∀ x : ℕ, (x > m ∧ x < 2) ↔ (x = 1) :=
by
  sorry

end exists_m_in_range_l255_255364


namespace find_multiple_l255_255843

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l255_255843


namespace inequality_proof_l255_255775

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (2 * a) + 1 / (2 * b) + 1 / (2 * c) ≥ 1 / (a + b) + 1 / (b + c) + 1 / (c + a) :=
by
  sorry

end inequality_proof_l255_255775


namespace sufficient_but_not_necessary_l255_255492

theorem sufficient_but_not_necessary (x y : ℝ) (h : x ≥ 1 ∧ y ≥ 1) : x ^ 2 + y ^ 2 ≥ 2 ∧ ∃ (x y : ℝ), x ^ 2 + y ^ 2 ≥ 2 ∧ (¬ (x ≥ 1 ∧ y ≥ 1)) :=
by
  sorry

end sufficient_but_not_necessary_l255_255492


namespace arithmetic_sequence_sum_product_l255_255564

noncomputable def a := 13 / 2
def d := 3 / 2

theorem arithmetic_sequence_sum_product (a d : ℚ) (h1 : 4 * a = 26) (h2 : a^2 - d^2 = 40) :
  (a - 3 * d, a - d, a + d, a + 3 * d) = (2, 5, 8, 11) ∨
  (a - 3 * d, a - d, a + d, a + 3 * d) = (11, 8, 5, 2) :=
  sorry

end arithmetic_sequence_sum_product_l255_255564


namespace solution_system_of_equations_solution_system_of_inequalities_l255_255955

-- Part 1: System of Equations
theorem solution_system_of_equations (x y : ℚ) :
  (3 * x + 2 * y = 13) ∧ (2 * x + 3 * y = -8) ↔ (x = 11 ∧ y = -10) :=
by
  sorry

-- Part 2: System of Inequalities
theorem solution_system_of_inequalities (y : ℚ) :
  ((5 * y - 2) / 3 - 1 > (3 * y - 5) / 2) ∧ (2 * (y - 3) ≤ 0) ↔ (-5 < y ∧ y ≤ 3) :=
by
  sorry

end solution_system_of_equations_solution_system_of_inequalities_l255_255955


namespace smallest_n_7_pow_n_eq_n_7_mod_5_l255_255334

theorem smallest_n_7_pow_n_eq_n_7_mod_5 :
  ∃ n : ℕ, 0 < n ∧ (7^n ≡ n^7 [MOD 5]) ∧ ∀ m : ℕ, 0 < m ∧ (7^m ≡ m^7 [MOD 5]) → n ≤ m :=
begin
  sorry -- proof omitted as requested
end

end smallest_n_7_pow_n_eq_n_7_mod_5_l255_255334


namespace gasoline_added_l255_255872

theorem gasoline_added (total_capacity : ℝ) (initial_fraction final_fraction : ℝ) 
(h1 : initial_fraction = 3 / 4)
(h2 : final_fraction = 9 / 10)
(h3 : total_capacity = 29.999999999999996) : 
(final_fraction * total_capacity - initial_fraction * total_capacity = 4.499999999999999) :=
by sorry

end gasoline_added_l255_255872


namespace domino_cover_grid_l255_255268

-- Definitions representing the conditions:
def isPositive (n : ℕ) : Prop := n > 0
def divides (a b : ℕ) : Prop := ∃ k, b = k * a
def canCoverWithDominos (n k : ℕ) : Prop := ∀ i j, (i < n) → (j < n) → (∃ r, i = r * k ∨ j = r * k)

-- The hypothesis: n and k are positive integers
axiom n : ℕ
axiom k : ℕ
axiom n_positive : isPositive n
axiom k_positive : isPositive k

-- The main theorem
theorem domino_cover_grid (n k : ℕ) (n_positive : isPositive n) (k_positive : isPositive k) :
  canCoverWithDominos n k ↔ divides k n := by
  sorry

end domino_cover_grid_l255_255268


namespace sector_area_half_triangle_area_l255_255810

theorem sector_area_half_triangle_area (θ : Real) (r : Real) (hθ1 : 0 < θ) (hθ2 : θ < π / 3) :
    2 * θ = Real.tan θ := by
  sorry

end sector_area_half_triangle_area_l255_255810


namespace correct_order_option_C_l255_255154

def length_unit_ordered (order : List String) : Prop :=
  order = ["kilometer", "meter", "centimeter", "millimeter"]

def option_A := ["kilometer", "meter", "millimeter", "centimeter"]
def option_B := ["meter", "kilometer", "centimeter", "millimeter"]
def option_C := ["kilometer", "meter", "centimeter", "millimeter"]

theorem correct_order_option_C : length_unit_ordered option_C := by
  sorry

end correct_order_option_C_l255_255154


namespace percentage_less_l255_255868

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l255_255868


namespace not_satisfiable_conditions_l255_255112

theorem not_satisfiable_conditions (x y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : 0 ≤ y ∧ y ≤ 9) 
    (h3 : 10 * x + y % 80 = 0) (h4 : x + y = 2) : false := 
by 
  -- The proof is omitted because we are only asked for the statement.
  sorry

end not_satisfiable_conditions_l255_255112


namespace total_wasted_time_is_10_l255_255396

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l255_255396


namespace frac_1_7_correct_l255_255424

-- Define the fraction 1/7
def frac_1_7 : ℚ := 1 / 7

-- Define the decimal approximation 0.142857142857 as a rational number
def dec_approx : ℚ := 142857142857 / 10^12

-- Define the small fractional difference
def small_diff : ℚ := 1 / (7 * 10^12)

-- The theorem to be proven
theorem frac_1_7_correct :
  frac_1_7 = dec_approx + small_diff := 
sorry

end frac_1_7_correct_l255_255424


namespace negation_of_proposition_l255_255835

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x₀ : ℝ, x₀ ≤ 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := 
sorry

end negation_of_proposition_l255_255835


namespace largest_cardinality_A_l255_255192

noncomputable def largest_cardinality (A : Type) [fintype A] [has_mul A] : ℕ :=
fintype.card A

variables {A : Type} [fintype A] [has_mul A]
variables (associative : ∀ (a b c : A), a * (b * c) = (a * b) * c)
variables (cancellation : ∀ (a b c : A), a * c = b * c → a = b)
variables (identity_exists : ∃ e : A, ∀ a : A, a * e = a)
variables (special_condition : ∀ (a b : A) (e : A), a ≠ e → b ≠ e → (a * a * a) * b = (b * b * b) * (a * a))

theorem largest_cardinality_A : largest_cardinality A = 3 :=
sorry

end largest_cardinality_A_l255_255192


namespace number_of_routes_of_duration_10_minutes_l255_255417

def M : ℕ → ℕ
| 0 := 1
| 1 := 1
| (n + 2) := M n + M (n + 1)

theorem number_of_routes_of_duration_10_minutes : M 10 = 34 :=
by {
  -- Proof will go here
  sorry
}

end number_of_routes_of_duration_10_minutes_l255_255417


namespace next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l255_255817

-- Problem: Next number after 48 in the sequence
theorem next_number_after_48 (x : ℕ) (h₁ : x % 3 = 0) (h₂ : (x + 1) = 64) : x = 63 := sorry

-- Problem: Eighth number in the sequence
theorem eighth_number_in_sequence (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 8) : n = 168 := sorry

-- Problem: 2013th number in the sequence
theorem two_thousand_thirteenth_number (n : ℕ) 
  (h₁ : ∀ k, (k + 1) % 3 = 0 → 3 * n <= (k + 1) * (k + 1) ∧ (k + 1) * (k + 1) < 3 * (n + 1))
  (h₂ : (n : ℤ) = 2013) : n = 9120399 := sorry

end next_number_after_48_eighth_number_in_sequence_two_thousand_thirteenth_number_l255_255817


namespace average_marks_correct_l255_255626

-- Define constants for the marks in each subject
def english_marks : ℕ := 76
def math_marks : ℕ := 65
def physics_marks : ℕ := 82
def chemistry_marks : ℕ := 67
def biology_marks : ℕ := 85

-- Define the total number of subjects
def num_subjects : ℕ := 5

-- Define the total marks as the sum of individual subjects
def total_marks : ℕ := english_marks + math_marks + physics_marks + chemistry_marks + biology_marks

-- Define the average marks
def average_marks : ℕ := total_marks / num_subjects

-- Prove that the average marks is as expected
theorem average_marks_correct : average_marks = 75 :=
by {
  -- skip the proof
  sorry
}

end average_marks_correct_l255_255626


namespace greater_number_is_twenty_two_l255_255986

theorem greater_number_is_twenty_two (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * (x - y) = 12) : x = 22 :=
sorry

end greater_number_is_twenty_two_l255_255986


namespace probability_of_being_admitted_expected_value_of_X_l255_255873

-- Define the data
def number_of_penalty_kicks : List Nat := [20, 30, 30, 25, 20, 25]
def number_of_goals : List Nat := [15, 17, 22, 18, 14, 14]

-- Define the probabilities
def total_penalty_kicks : Nat := number_of_penalty_kicks.sum
def total_goals : Nat := number_of_goals.sum
def P_score : Rat := total_goals / total_penalty_kicks
def P_miss : Rat := 1 - P_score

-- First statement: Probability of being admitted
theorem probability_of_being_admitted : P_score * P_score + 
                                        P_miss * (P_score * P_score) + 
                                        (P_miss * P_miss * (P_score * P_score)) + 
                                        (P_score * P_miss * (P_score * P_score)) = 20 / 27 := 
by
  sorry

-- Second statement: Expected value of X
theorem expected_value_of_X : (0 * (P_miss * P_miss * P_miss)
                               + 1 * (2 * (P_score * (P_miss * P_miss) + P_miss * (P_miss * P_score)))
                               + 2 * (P_score * P_score + P_miss * (P_score * P_score) + P_miss * P_miss * (P_score * P_score) + P_score * (P_miss * P_score))
                               + 3 * (P_score * P_miss * (P_score * P_score))) = 50 / 27 := 
by
  sorry

end probability_of_being_admitted_expected_value_of_X_l255_255873


namespace sufficient_but_not_necessary_condition_l255_255174

theorem sufficient_but_not_necessary_condition (a : ℝ) : (a^2 ≠ 4) → (a ≠ 2) ∧ ¬ ((a ≠ 2) → (a^2 ≠ 4)) :=
by {
  sorry
}

end sufficient_but_not_necessary_condition_l255_255174


namespace equation_of_perpendicular_line_l255_255419

theorem equation_of_perpendicular_line :
  ∃ (a b c : ℝ), (5, 3) ∈ {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  ∧ (a = 2 ∧ b = 1 ∧ c = -13)
  ∧ (a * 1 + b * (-2) = 0) :=
sorry

end equation_of_perpendicular_line_l255_255419


namespace number_of_best_friends_l255_255403

-- Constants and conditions
def initial_tickets : ℕ := 37
def tickets_per_friend : ℕ := 5
def tickets_left : ℕ := 2

-- Problem statement
theorem number_of_best_friends : (initial_tickets - tickets_left) / tickets_per_friend = 7 :=
by
  sorry

end number_of_best_friends_l255_255403


namespace statement_B_is_algorithm_l255_255302

def is_algorithm (statement : String) : Prop := 
  statement = "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."

def condition_A : String := "At home, it is generally the mother who cooks."
def condition_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def condition_C : String := "Cooking outdoors is called a picnic."
def condition_D : String := "Rice is necessary for cooking."

theorem statement_B_is_algorithm : is_algorithm condition_B :=
by
  sorry

end statement_B_is_algorithm_l255_255302


namespace factor_polynomial_l255_255907

def Polynomial_Factorization (x : ℝ) : Prop := 
  let P := x^2 - 6*x + 9 - 64*x^4
  P = (8*x^2 + x - 3) * (-8*x^2 + x - 3)

theorem factor_polynomial : ∀ x : ℝ, Polynomial_Factorization x :=
by 
  intro x
  unfold Polynomial_Factorization
  sorry

end factor_polynomial_l255_255907


namespace necessary_but_not_sufficient_for_p_l255_255798

variable {p q r : Prop}

theorem necessary_but_not_sufficient_for_p 
  (h₁ : p → q) (h₂ : ¬ (q → p)) 
  (h₃ : q → r) (h₄ : ¬ (r → q)) 
  : (r → p) ∧ ¬ (p → r) :=
sorry

end necessary_but_not_sufficient_for_p_l255_255798


namespace determine_g_function_l255_255014

theorem determine_g_function (t x : ℝ) (g : ℝ → ℝ) 
  (line_eq : ∀ x y : ℝ, y = 2 * x - 40) 
  (param_eq : ∀ t : ℝ, (x, 20 * t - 14) = (g t, 20 * t - 14)) :
  g t = 10 * t + 13 :=
by 
  sorry

end determine_g_function_l255_255014


namespace maximal_number_blackboard_l255_255699

/-- 
 Given some distinct positive integers on a blackboard such that the sum of any two distinct integers is a power of 2. Prove that the maximal number on the blackboard can be 
 of the form 2^k - 1 for some large k.
 -/
theorem maximal_number_blackboard (S : Finset ℕ) (h : ∀ ⦃a b⦄, a ∈ S → b ∈ S → a ≠ b → ∃ k, a + b = 2^k) : 
  ∃ k, ∀ a ∈ S, a ≤ 2^k - 1 :=
sorry

end maximal_number_blackboard_l255_255699


namespace sum_of_coordinates_of_B_l255_255277

-- Definitions
def Point := (ℝ × ℝ)
def isMidpoint (M A B : Point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

-- Given conditions
def M : Point := (4, 8)
def A : Point := (10, 4)

-- Statement to prove
theorem sum_of_coordinates_of_B (B : Point) (h : isMidpoint M A B) :
  B.1 + B.2 = 10 :=
by
  sorry

end sum_of_coordinates_of_B_l255_255277


namespace watermelon_slices_l255_255299

theorem watermelon_slices (total_seeds slices_black seeds_white seeds_per_slice num_slices : ℕ)
  (h1 : seeds_black = 20)
  (h2 : seeds_white = 20)
  (h3 : seeds_per_slice = seeds_black + seeds_white)
  (h4 : total_seeds = 1600)
  (h5 : num_slices = total_seeds / seeds_per_slice) :
  num_slices = 40 :=
by
  sorry

end watermelon_slices_l255_255299


namespace max_value_of_a_l255_255660

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l255_255660


namespace remainder_of_polynomial_l255_255899

theorem remainder_of_polynomial (x : ℤ) : 
  (x^4 - 1) * (x^2 - 1) % (x^2 + x + 1) = 3 := 
sorry

end remainder_of_polynomial_l255_255899


namespace multiply_exponents_l255_255617

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l255_255617


namespace increase_in_share_l255_255746

/-- Given a total cost of a car, earnings from a car wash, number of friends initially sharing 
    the cost, and the number of friends after Brad leaves, prove the increase in the amount each 
    remaining friend has to pay. -/
theorem increase_in_share
  (total_cost : ℕ)
  (car_wash_earnings : ℕ)
  (initial_friends : ℕ)
  (remaining_friends : ℕ)
  (initial_share : ℕ := (total_cost - car_wash_earnings) / initial_friends)
  (remaining_share : ℕ := (total_cost - car_wash_earnings) / remaining_friends) :
  total_cost = 1700 →
  car_wash_earnings = 500 →
  initial_friends = 6 →
  remaining_friends = 5 →
  (remaining_share - initial_share = 40) :=
by
  intros h1 h2 h3 h4
  have h₀ : total_cost - car_wash_earnings = 1700 - 500 := by rw [h1, h2]
  have h₁ : initial_share = (1700 - 500) / 6 := by rw [h₀, h3]
  have h₂ : remaining_share = (1700 - 500) / 5 := by rw [h₀, h4]
  have h₃ : (1700 - 500) = 1200 := by norm_num
  have h₄ : initial_share = 1200 / 6 := by rw [h₃]
  have h₅ : remaining_share = 1200 / 5 := by rw [h₃]
  have h₆ : 1200 / 6 = 200 := by norm_num
  have h₇ : 1200 / 5 = 240 := by norm_num
  rw [h₄, h₅, h₆, h₇]
  norm_num
  sorry

end increase_in_share_l255_255746


namespace cylinder_cone_surface_area_l255_255040

theorem cylinder_cone_surface_area (r h : ℝ) (π : ℝ) (l : ℝ)
    (h_relation : h = Real.sqrt 3 * r)
    (l_relation : l = 2 * r)
    (cone_lateral_surface_area : π * r * l = 2 * π * r ^ 2) :
    (2 * π * r * h) / (π * r ^ 2) = 2 * Real.sqrt 3 :=
by
    sorry

end cylinder_cone_surface_area_l255_255040


namespace angle_sum_triangle_l255_255941

theorem angle_sum_triangle (x : ℝ) 
  (h1 : 70 + 70 + x = 180) : 
  x = 40 :=
by
  sorry

end angle_sum_triangle_l255_255941


namespace solution_set_inequality_l255_255529

def custom_op (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem solution_set_inequality : {x : ℝ | custom_op x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_inequality_l255_255529


namespace sphere_volume_equals_surface_area_l255_255049

theorem sphere_volume_equals_surface_area (r : ℝ) (hr : r = 3) :
  (4 / 3) * π * r^3 = 4 * π * r^2 := by
  sorry

end sphere_volume_equals_surface_area_l255_255049


namespace scientific_notation_of_taichulight_performance_l255_255372

noncomputable def trillion := 10^12

def convert_to_scientific_notation (x : ℝ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ x * 10^n = 12.5 * trillion

theorem scientific_notation_of_taichulight_performance :
  ∃ (x : ℝ) (n : ℤ), convert_to_scientific_notation x n ∧ x = 1.25 ∧ n = 13 :=
by
  unfold convert_to_scientific_notation
  use 1.25
  use 13
  sorry

end scientific_notation_of_taichulight_performance_l255_255372


namespace measure_diff_eq_l255_255684

noncomputable def P {n : ℕ} : MeasureTheory.Measure (EuclideanSpace ℝ n) := sorry
noncomputable def F {n : ℕ} (x : Fin n → ℝ) : ℝ := P { y | ∀ i, y i ≤ x i }

noncomputable def Delta {n : ℕ} (a b : ℝ) (i : Fin n) (F : (Fin n → ℝ) → ℝ) (x : (Fin n → ℝ)) : ℝ :=
  F (fun j => if j = i then b else x j) - F (fun j => if j = i then a else x j)

theorem measure_diff_eq {n : ℕ} (a b : Fin n → ℝ) :
  Delta (a 0) (b 0) 0 (Delta (a 1) (b 1) 1 (Delta (a 2) (b 2) 2 ... F)) = 
  P { x | ∀ i, a i < x i ∧ x i ≤ b i } := sorry

end measure_diff_eq_l255_255684


namespace intersection_of_sets_l255_255931

def setA := { x : ℝ | x / (x - 1) < 0 }
def setB := { x : ℝ | 0 < x ∧ x < 3 }
def setIntersect := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_of_sets :
  ∀ x : ℝ, x ∈ setA ∧ x ∈ setB ↔ x ∈ setIntersect := 
by
  sorry

end intersection_of_sets_l255_255931


namespace bees_count_l255_255406

theorem bees_count (x : ℕ) (h1 : (1/5 : ℚ) * x + (1/3 : ℚ) * x + 
    3 * ((1/3 : ℚ) * x - (1/5 : ℚ) * x) + 1 = x) : x = 15 := 
sorry

end bees_count_l255_255406


namespace probability_of_type_I_error_l255_255859

theorem probability_of_type_I_error 
  (K_squared : ℝ)
  (alpha : ℝ)
  (critical_val : ℝ)
  (h1 : K_squared = 4.05)
  (h2 : alpha = 0.05)
  (h3 : critical_val = 3.841)
  (h4 : 4.05 > 3.841) :
  alpha = 0.05 := 
sorry

end probability_of_type_I_error_l255_255859


namespace num_different_pairs_l255_255683

theorem num_different_pairs :
  (∃ (A B : Finset ℕ), A ∪ B = {1, 2, 3, 4} ∧ A ≠ B ∧ (A, B) ≠ (B, A)) ∧
  (∃ n : ℕ, n = 81) :=
by
  -- Proof would go here, but it's skipped per instructions
  sorry

end num_different_pairs_l255_255683


namespace bruce_paid_amount_l255_255890

def kg_of_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_of_mangoes : ℕ := 10
def rate_per_kg_mangoes : ℕ := 55

def total_amount_paid : ℕ := (kg_of_grapes * rate_per_kg_grapes) + (kg_of_mangoes * rate_per_kg_mangoes)

theorem bruce_paid_amount : total_amount_paid = 1110 :=
by sorry

end bruce_paid_amount_l255_255890


namespace negation_of_exists_prop_l255_255291

theorem negation_of_exists_prop (x : ℝ) :
  (¬ ∃ (x : ℝ), (x > 0) ∧ (|x| + x >= 0)) ↔ (∀ (x : ℝ), x > 0 → |x| + x < 0) := 
sorry

end negation_of_exists_prop_l255_255291


namespace solution_set_eq_l255_255216

noncomputable def f (x : ℝ) : ℝ := x^6 + x^2
noncomputable def g (x : ℝ) : ℝ := (2*x + 3)^3 + 2*x + 3

theorem solution_set_eq : {x : ℝ | f x = g x} = {-1, 3} :=
by
  sorry

end solution_set_eq_l255_255216


namespace boys_from_school_a_not_study_science_l255_255526

theorem boys_from_school_a_not_study_science (total_boys : ℕ) (boys_from_school_a_percentage : ℝ) (science_study_percentage : ℝ)
  (total_boys_in_camp : total_boys = 250) (school_a_percent : boys_from_school_a_percentage = 0.20) 
  (science_percent : science_study_percentage = 0.30) :
  ∃ (boys_from_school_a_not_science : ℕ), boys_from_school_a_not_science = 35 :=
by
  sorry

end boys_from_school_a_not_study_science_l255_255526


namespace ratio_B_to_A_l255_255612

theorem ratio_B_to_A (A B S : ℕ) 
  (h1 : A = 2 * S)
  (h2 : A = 80)
  (h3 : B - S = 200) :
  B / A = 3 :=
by sorry

end ratio_B_to_A_l255_255612


namespace front_view_correct_l255_255074

-- Define the number of blocks in each column
def Blocks_Column_A : Nat := 3
def Blocks_Column_B : Nat := 5
def Blocks_Column_C : Nat := 2
def Blocks_Column_D : Nat := 4

-- Define the front view representation
def front_view : List Nat := [3, 5, 2, 4]

-- Statement to be proved
theorem front_view_correct :
  [Blocks_Column_A, Blocks_Column_B, Blocks_Column_C, Blocks_Column_D] = front_view :=
by
  sorry

end front_view_correct_l255_255074


namespace line_intercepts_l255_255151

theorem line_intercepts :
  (exists a b : ℝ, (forall x y : ℝ, x - 2*y - 2 = 0 ↔ (x = 2 ∨ y = -1)) ∧ a = 2 ∧ b = -1) :=
by
  sorry

end line_intercepts_l255_255151


namespace find_incorrect_statement_l255_255368

def statement_A := ∀ (P Q : Prop), (P → Q) → (¬Q → ¬P)
def statement_B := ∀ (P : Prop), ((¬P) → false) → P
def statement_C := ∀ (shape : Type), (∃ s : shape, true) → false
def statement_D := ∀ (P : ℕ → Prop), P 0 → (∀ n, P n → P (n + 1)) → ∀ n, P n
def statement_E := ∀ {α : Type} (p : Prop), (¬p ∨ p)

theorem find_incorrect_statement : statement_C :=
sorry

end find_incorrect_statement_l255_255368


namespace sine_double_angle_inequality_l255_255801

theorem sine_double_angle_inequality {α : ℝ} (hα1 : 0 < α) (hα2 : α < π / 4) : 
  Real.sin (2 * α) < 2 * Real.sin α :=
by
  sorry

end sine_double_angle_inequality_l255_255801


namespace max_value_of_expressions_l255_255634

theorem max_value_of_expressions :
  ∃ x ∈ ℝ, (∀ y ∈ ℝ, (2^y - 4^y) ≤ (2^x - 4^x)) ∧ (2^x - 4^x = 1 / 4) :=
by
  sorry

end max_value_of_expressions_l255_255634


namespace triangle_perimeter_triangle_side_c_l255_255667

theorem triangle_perimeter (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) (h2 : c = 2) : 
  a + b + c = 6 := 
sorry

theorem triangle_side_c (A B C : ℝ) (a b c : ℝ) (h1 : b * (Real.sin (A/2))^2 + a * (Real.sin (B/2))^2 = C / 2) 
(h2 : C = Real.pi / 3) (h3 : 2 * Real.sqrt 3 = (1/2) * a * b * Real.sin (Real.pi / 3)) : 
c = 2 * Real.sqrt 2 := 
sorry

end triangle_perimeter_triangle_side_c_l255_255667


namespace total_cost_is_2160_l255_255988

variables (x y z : ℝ)

-- Conditions
def cond1 : Prop := x = 0.45 * y
def cond2 : Prop := y = 0.8 * z
def cond3 : Prop := z = x + 640

-- Goal
def total_cost := x + y + z

theorem total_cost_is_2160 (x y z : ℝ) (h1 : cond1 x y) (h2 : cond2 y z) (h3 : cond3 x z) :
  total_cost x y z = 2160 :=
by
  sorry

end total_cost_is_2160_l255_255988


namespace unique_integer_sum_squares_l255_255769

theorem unique_integer_sum_squares (n : ℤ) (h : ∃ d1 d2 d3 d4 : ℕ, d1 * d2 * d3 * d4 = n ∧ n = d1*d1 + d2*d2 + d3*d3 + d4*d4) : n = 42 := 
sorry

end unique_integer_sum_squares_l255_255769


namespace percentage_increase_l255_255982

theorem percentage_increase (x : ℝ) (h : 2 * x = 540) (new_price : ℝ) (h_new_price : new_price = 351) :
  ((new_price - x) / x) * 100 = 30 := by
  sorry

end percentage_increase_l255_255982


namespace product_of_nonreal_roots_l255_255635

theorem product_of_nonreal_roots (p : Polynomial ℂ) (hp : p = Polynomial.C (-119) + Polynomial.monomial 4 (1 : ℂ) - Polynomial.monomial 3 (6 : ℂ) + Polynomial.monomial 2 (15 : ℂ) - Polynomial.monomial 1 (20 : ℂ)) :
  let nonreal_roots := {r : ℂ | Polynomial.root p r ∧ r.im ≠ 0} in
  nonreal_roots.prod (fun x => x) = 4 + complex.sqrt 103 :=
sorry

end product_of_nonreal_roots_l255_255635


namespace nina_max_digits_l255_255551

-- Define the conditions
def sam_digits (C : ℕ) := C + 6
def mina_digits := 24
def nina_digits (C : ℕ) := (7 * C) / 2

-- Define Carlos's digits and the sum condition
def carlos_digits := mina_digits / 6
def total_digits (C : ℕ) := C + sam_digits C + mina_digits + nina_digits C

-- Prove the maximum number of digits Nina could memorize
theorem nina_max_digits : ∀ C : ℕ, C = carlos_digits →
  total_digits C ≤ 100 → nina_digits C ≤ 62 :=
by
  intro C hC htotal
  sorry

end nina_max_digits_l255_255551


namespace value_of_p_l255_255504

noncomputable def term_not_containing_x (p : ℝ) : ℝ :=
  Nat.choose 6 4 * (2:ℝ)^2 / p^4

theorem value_of_p :
  (term_not_containing_x 3 = 20 / 27) :=
sorry

end value_of_p_l255_255504


namespace parallel_lines_m_l255_255929

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, 2 * x + 3 * y + 1 = 0 → 6 ≠ 0) ∧ 
  (∀ x y : ℝ, m * x + 6 * y - 5 = 0 → 6 ≠ 0) → 
  m = 4 :=
by
  intro h
  sorry

end parallel_lines_m_l255_255929


namespace ratio_of_areas_of_squares_l255_255018

theorem ratio_of_areas_of_squares (side_C side_D : ℕ) 
  (hC : side_C = 48) (hD : side_D = 60) : 
  (side_C^2 : ℚ)/(side_D^2 : ℚ) = 16/25 :=
by
  -- sorry, proof omitted
  sorry

end ratio_of_areas_of_squares_l255_255018


namespace quadratic_root_zero_l255_255085

theorem quadratic_root_zero (k : ℝ) :
    (∃ x : ℝ, x = 0 ∧ (k - 1) * x ^ 2 + 6 * x + k ^ 2 - k = 0) → k = 0 :=
by
  sorry

end quadratic_root_zero_l255_255085


namespace functional_equation_solution_l255_255768

theorem functional_equation_solution (f : ℚ → ℚ) 
  (h : ∀ x y : ℚ, f (x + f y) = f x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) :=
sorry

end functional_equation_solution_l255_255768


namespace parallel_lines_slope_condition_l255_255474

theorem parallel_lines_slope_condition (b : ℝ) (x y : ℝ) :
    (∀ (x y : ℝ), 3 * y - 3 * b = 9 * x) →
    (∀ (x y : ℝ), y + 2 = (b + 9) * x) →
    b = -6 :=
by
    sorry

end parallel_lines_slope_condition_l255_255474


namespace cos_value_given_sin_condition_l255_255346

open Real

theorem cos_value_given_sin_condition (x : ℝ) (h : sin (x + π / 12) = -1/4) : 
  cos (5 * π / 6 - 2 * x) = -7 / 8 :=
sorry -- Proof steps are omitted.

end cos_value_given_sin_condition_l255_255346


namespace tan_double_angle_l255_255354

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_derivative_def (x : ℝ) : ℝ := 3 * f x

theorem tan_double_angle (x : ℝ) (h : f_derivative_def x = Real.cos x - Real.sin x) : 
  Real.tan (2 * x) = -4 / 3 :=
by
  sorry

end tan_double_angle_l255_255354


namespace total_apples_picked_l255_255543

def apples_picked : ℕ :=
  let mike := 7
  let nancy := 3
  let keith := 6
  let olivia := 12
  let thomas := 8
  mike + nancy + keith + olivia + thomas

theorem total_apples_picked :
  apples_picked = 36 :=
by
  -- Proof would go here; 'sorry' is used to skip the proof.
  sorry

end total_apples_picked_l255_255543


namespace maximize_profit_l255_255739

theorem maximize_profit (x : ℤ) (hx : 20 ≤ x ∧ x ≤ 30) :
  (∀ y, 20 ≤ y ∧ y ≤ 30 → ((y - 20) * (30 - y)) ≤ ((25 - 20) * (30 - 25))) := 
sorry

end maximize_profit_l255_255739


namespace blake_lollipops_count_l255_255888

theorem blake_lollipops_count (lollipop_cost : ℕ) (choc_cost_per_pack : ℕ) 
  (chocolate_packs : ℕ) (total_paid : ℕ) (change_received : ℕ) 
  (total_spent : ℕ) (total_choc_cost : ℕ) (remaining_amount : ℕ) 
  (lollipop_count : ℕ) : 
  lollipop_cost = 2 →
  choc_cost_per_pack = 4 * lollipop_cost →
  chocolate_packs = 6 →
  total_paid = 6 * 10 →
  change_received = 4 →
  total_spent = total_paid - change_received →
  total_choc_cost = chocolate_packs * choc_cost_per_pack →
  remaining_amount = total_spent - total_choc_cost →
  lollipop_count = remaining_amount / lollipop_cost →
  lollipop_count = 4 := 
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end blake_lollipops_count_l255_255888


namespace total_items_l255_255330

theorem total_items (slices_of_bread bottles_of_milk cookies : ℕ) (h1 : slices_of_bread = 58)
  (h2 : bottles_of_milk = slices_of_bread - 18) (h3 : cookies = slices_of_bread + 27) :
  slices_of_bread + bottles_of_milk + cookies = 183 :=
by
  sorry

end total_items_l255_255330


namespace handshake_count_l255_255059

theorem handshake_count (n_total n_group1 n_group2 : ℕ) 
  (h_total : n_total = 40) (h_group1 : n_group1 = 25) (h_group2 : n_group2 = 15) 
  (h_sum : n_group1 + n_group2 = n_total) : 
  (15 * 39) / 2 = 292 := 
by sorry

end handshake_count_l255_255059


namespace zero_point_interval_l255_255012

noncomputable def f (x : ℝ) := 6 / x - x ^ 2

theorem zero_point_interval : ∃ x : ℝ, 1 < x ∧ x < 2 ∧ f x = 0 :=
by
  sorry

end zero_point_interval_l255_255012


namespace largest_quadrilateral_angle_l255_255834

theorem largest_quadrilateral_angle (x : ℝ)
  (h1 : 3 * x + 4 * x + 5 * x + 6 * x = 360) :
  6 * x = 120 :=
by
  sorry

end largest_quadrilateral_angle_l255_255834


namespace sugar_amount_l255_255456

noncomputable def mixed_to_improper (a : ℕ) (b c : ℕ) : ℚ :=
  a + b / c

theorem sugar_amount (a : ℚ) (h : a = mixed_to_improper 7 3 4) : 1 / 3 * a = 2 + 7 / 12 :=
by
  rw [h]
  simp
  sorry

end sugar_amount_l255_255456


namespace Ram_Shyam_weight_ratio_l255_255295

theorem Ram_Shyam_weight_ratio :
  ∃ (R S : ℝ), 
    (1.10 * R + 1.21 * S = 82.8) ∧ 
    (1.15 * (R + S) = 82.8) ∧ 
    (R / S = 1.20) :=
by {
  sorry
}

end Ram_Shyam_weight_ratio_l255_255295


namespace arithmetic_sequence_sum_l255_255373

theorem arithmetic_sequence_sum :
  ∀ (a : ℕ → ℕ), a 1 = 2 ∧ a 2 + a 3 = 13 → a 4 + a 5 + a 6 = 42 :=
by
  intro a
  intro h
  sorry

end arithmetic_sequence_sum_l255_255373


namespace negation_of_p_is_neg_p_l255_255685

-- Define the proposition p
def p : Prop := ∀ n : ℕ, 3^n ≥ n^2 + 1

-- Define the negation of p
def neg_p : Prop := ∃ n_0 : ℕ, 3^n_0 < n_0^2 + 1

-- The proof statement
theorem negation_of_p_is_neg_p : ¬p ↔ neg_p :=
by sorry

end negation_of_p_is_neg_p_l255_255685


namespace symmetric_point_y_axis_l255_255649

def M : ℝ × ℝ := (-5, 2)
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
theorem symmetric_point_y_axis :
  symmetric_point M = (5, 2) :=
by
  sorry

end symmetric_point_y_axis_l255_255649


namespace find_line_equation_l255_255362

noncomputable def line_equation (x y : ℝ) : Prop :=
  y = (Real.sqrt 3 / 3) * x - 4

theorem find_line_equation :
  ∃ (x₁ y₁ : ℝ), x₁ = Real.sqrt 3 ∧ y₁ = -3 ∧ ∀ x y, (line_equation x y ↔ 
  (y + 3 = (Real.sqrt 3 / 3) * (x - Real.sqrt 3))) :=
sorry

end find_line_equation_l255_255362


namespace solve_for_S_l255_255184

variable (D S : ℝ)
variable (h1 : D > 0)
variable (h2 : S > 0)
variable (h3 : ((0.75 * D) / 50 + (0.25 * D) / S) / D = 1 / 50)

theorem solve_for_S :
  S = 50 :=
by
  sorry

end solve_for_S_l255_255184


namespace find_constants_l255_255887

theorem find_constants (a b c : ℝ) (h_neg : a < 0) (h_amp : |a| = 3) (h_period : b > 0 ∧ (2 * π / b) = 8 * π) : 
a = -3 ∧ b = 0.5 :=
by
  sorry

end find_constants_l255_255887


namespace find_x_if_friendly_l255_255524

theorem find_x_if_friendly (x : ℚ) :
    (∃ m n : ℚ, m + n = 66 ∧ m = 7 * x ∧ n = -18) →
    x = 12 :=
by
  sorry

end find_x_if_friendly_l255_255524


namespace frog_jump_distance_l255_255426

variable (grasshopper_jump frog_jump mouse_jump : ℕ)
variable (H1 : grasshopper_jump = 19)
variable (H2 : grasshopper_jump = frog_jump + 4)
variable (H3 : mouse_jump = frog_jump - 44)

theorem frog_jump_distance : frog_jump = 15 := by
  sorry

end frog_jump_distance_l255_255426


namespace trapezoid_angles_l255_255162

-- Definition of the problem statement in Lean 4
theorem trapezoid_angles (A B C D : ℝ) (h1 : A = 60) (h2 : B = 130)
  (h3 : A + D = 180) (h4 : B + C = 180) (h_sum : A + B + C + D = 360) :
  C = 50 ∧ D = 120 :=
by
  sorry

end trapezoid_angles_l255_255162


namespace juan_stamp_cost_l255_255306

-- Defining the prices of the stamps
def price_brazil : ℝ := 0.07
def price_peru : ℝ := 0.05

-- Defining the number of stamps from the 70s and 80s
def stamps_brazil_70s : ℕ := 12
def stamps_brazil_80s : ℕ := 15
def stamps_peru_70s : ℕ := 6
def stamps_peru_80s : ℕ := 12

-- Calculating total number of stamps from the 70s and 80s
def total_stamps_brazil : ℕ := stamps_brazil_70s + stamps_brazil_80s
def total_stamps_peru : ℕ := stamps_peru_70s + stamps_peru_80s

-- Calculating total cost
def total_cost_brazil : ℝ := total_stamps_brazil * price_brazil
def total_cost_peru : ℝ := total_stamps_peru * price_peru

def total_cost : ℝ := total_cost_brazil + total_cost_peru

-- Proof statement
theorem juan_stamp_cost : total_cost = 2.79 :=
by
  sorry

end juan_stamp_cost_l255_255306


namespace initial_observations_count_l255_255147

theorem initial_observations_count (S x n : ℕ) (h1 : S = 12 * n) (h2 : S + x = 11 * (n + 1)) (h3 : x = 5) : n = 6 :=
sorry

end initial_observations_count_l255_255147


namespace original_denominator_value_l255_255320

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l255_255320


namespace compute_fraction_l255_255945

theorem compute_fraction (x y z : ℝ) (h : x ≠ y ∧ y ≠ z ∧ z ≠ x) (sum_eq : x + y + z = 12) :
  (xy + yz + zx) / (x^2 + y^2 + z^2) = (144 - (x^2 + y^2 + z^2)) / (2 * (x^2 + y^2 + z^2)) :=
by
  sorry

end compute_fraction_l255_255945


namespace eq_solutions_count_l255_255837

theorem eq_solutions_count : 
  ∃! (n : ℕ), n = 126 ∧ (∀ x y : ℕ, 2*x + 3*y = 768 → x > 0 ∧ y > 0 → ∃ t : ℤ, x = 384 + 3*t ∧ y = -2*t ∧ -127 ≤ t ∧ t <= -1) := sorry

end eq_solutions_count_l255_255837


namespace calculation_result_l255_255618

theorem calculation_result : ((55 * 45 - 37 * 43) - (3 * 221 + 1)) / 22 = 10 := by
  sorry

end calculation_result_l255_255618


namespace toys_cost_price_gain_l255_255748

theorem toys_cost_price_gain (selling_price : ℕ) (cost_price_per_toy : ℕ) (num_toys : ℕ)
    (total_cost_price : ℕ) (gain : ℕ) (x : ℕ) :
    selling_price = 21000 →
    cost_price_per_toy = 1000 →
    num_toys = 18 →
    total_cost_price = num_toys * cost_price_per_toy →
    gain = selling_price - total_cost_price →
    x = gain / cost_price_per_toy →
    x = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at *
  sorry

end toys_cost_price_gain_l255_255748


namespace cost_per_bundle_l255_255880

-- Condition: each rose costs 500 won
def rose_price := 500

-- Condition: total number of roses
def total_roses := 200

-- Condition: number of bundles
def bundles := 25

-- Question: Prove the cost per bundle
theorem cost_per_bundle (rp : ℕ) (tr : ℕ) (b : ℕ) : rp = 500 → tr = 200 → b = 25 → (rp * tr) / b = 4000 :=
by
  intros h0 h1 h2
  sorry

end cost_per_bundle_l255_255880


namespace multiply_exponents_l255_255616

theorem multiply_exponents (x : ℝ) : (x^2) * (x^3) = x^5 := 
sorry

end multiply_exponents_l255_255616


namespace quadratic_real_roots_iff_find_m_given_condition_l255_255496

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let disc := quadratic_discriminant a b c
  if disc < 0 then (0, 0)
  else ((-b + disc.sqrt) / (2 * a), (-b - disc.sqrt) / (2 * a))

theorem quadratic_real_roots_iff (m : ℝ) :
  (quadratic_discriminant 1 (-2 * (m + 1)) (m ^ 2 + 5) ≥ 0) ↔ (m ≥ 2) :=
by sorry

theorem find_m_given_condition (x1 x2 m : ℝ) (h1 : x1 + x2 = 2 * (m + 1)) (h2 : x1 * x2 = m ^ 2 + 5) (h3 : (x1 - 1) * (x2 - 1) = 28) :
  m = 6 :=
by sorry

end quadratic_real_roots_iff_find_m_given_condition_l255_255496


namespace polynomials_same_type_l255_255579

-- Definitions based on the conditions
def variables_ab2 := {a, b}
def degree_ab2 := 3

-- Define the polynomial we are comparing with
def polynomial := -2 * a * b^2

-- Define the type equivalency of polynomials
def same_type (p1 p2 : Expr) : Prop :=
  (p1.variables = p2.variables) ∧ (p1.degree = p2.degree)

-- The statement to be proven
theorem polynomials_same_type : same_type polynomial ab2 :=
sorry

end polynomials_same_type_l255_255579


namespace paula_twice_as_old_as_karl_6_years_later_l255_255308

theorem paula_twice_as_old_as_karl_6_years_later
  (P K : ℕ)
  (h1 : P - 5 = 3 * (K - 5))
  (h2 : P + K = 54) :
  P + 6 = 2 * (K + 6) :=
sorry

end paula_twice_as_old_as_karl_6_years_later_l255_255308


namespace smallest_sum_xy_l255_255497

theorem smallest_sum_xy (x y : ℕ) (hx : x ≠ y) (h : 0 < x ∧ 0 < y) (hxy : (1 : ℚ) / x + (1 : ℚ) / y = 1 / 15) :
  x + y = 64 :=
sorry

end smallest_sum_xy_l255_255497


namespace problem1_part1_problem1_part2_problem2_part1_problem2_part2_l255_255791

open Set

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | -2 < x ∧ x < 5 }
def B : Set ℝ := { x | -1 ≤ x - 1 ∧ x - 1 ≤ 2 }

theorem problem1_part1 : A ∪ B = { x | -2 < x ∧ x < 5 } := sorry
theorem problem1_part2 : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } := sorry

def B_c : Set ℝ := { x | x < 0 ∨ 3 < x }

theorem problem2_part1 : A ∪ B_c = U := sorry
theorem problem2_part2 : A ∩ B_c = { x | (-2 < x ∧ x < 0) ∨ (3 < x ∧ x < 5) } := sorry

end problem1_part1_problem1_part2_problem2_part1_problem2_part2_l255_255791


namespace initial_amount_100000_l255_255632

noncomputable def compound_interest_amount (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

noncomputable def future_value (P CI : ℝ) : ℝ :=
  P + CI

theorem initial_amount_100000
  (CI : ℝ) (P : ℝ) (r : ℝ) (n t : ℕ) 
  (h1 : CI = 8243.216)
  (h2 : r = 0.04)
  (h3 : n = 2)
  (h4 : t = 2)
  (h5 : future_value P CI = compound_interest_amount P r n t) :
  P = 100000 :=
by
  sorry

end initial_amount_100000_l255_255632


namespace find_C_and_D_l255_255767

noncomputable def C : ℚ := 15 / 8
noncomputable def D : ℚ := 17 / 8

theorem find_C_and_D (x : ℚ) (h₁ : x ≠ 9) (h₂ : x ≠ -7) :
  (4 * x - 6) / ((x - 9) * (x + 7)) = C / (x - 9) + D / (x + 7) :=
by sorry

end find_C_and_D_l255_255767


namespace find_a_l255_255780

theorem find_a (x y a : ℤ) (h1 : x = 1) (h2 : y = -1) (h3 : 2 * x - a * y = 3) : a = 1 := by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end find_a_l255_255780


namespace fraction_ordering_l255_255300

theorem fraction_ordering :
  (8 / 25 : ℚ) < 6 / 17 ∧ 6 / 17 < 10 / 27 ∧ 8 / 25 < 10 / 27 :=
by
  sorry

end fraction_ordering_l255_255300


namespace find_b_if_lines_parallel_l255_255471

theorem find_b_if_lines_parallel (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
  (∀ x y : ℝ, y + 2 = (b + 9) * x → y = (b + 9) * x - 2) →
  3 = b + 9 →
  b = -6 :=
by {
  sorry
}

end find_b_if_lines_parallel_l255_255471


namespace shaded_region_area_l255_255311

noncomputable def area_shaded_region (r_small r_large : ℝ) (A B : ℝ × ℝ) : ℝ := 
  let pi := Real.pi
  let sqrt_5 := Real.sqrt 5
  (5 * pi / 2) - (4 * sqrt_5)

theorem shaded_region_area : 
  ∀ (r_small r_large : ℝ) (A B : ℝ × ℝ), 
  r_small = 2 → 
  r_large = 3 → 
  (A = (0, 0)) → 
  (B = (4, 0)) → 
  area_shaded_region r_small r_large A B = (5 * Real.pi / 2) - (4 * Real.sqrt 5) := 
by
  intros r_small r_large A B h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end shaded_region_area_l255_255311


namespace min_dist_circle_to_line_l255_255915

noncomputable def circle_eq (x y : ℝ) := x^2 + y^2 - 2*x - 2*y

noncomputable def line_eq (x y : ℝ) := x + y - 8

theorem min_dist_circle_to_line : 
  (∀ x y : ℝ, circle_eq x y = 0 → ∃ d : ℝ, d ≥ 0 ∧ 
    (∀ x₁ y₁ : ℝ, circle_eq x₁ y₁ = 0 → ∀ x₂ y₂ : ℝ, line_eq x₂ y₂ = 0 → d ≤ dist (x₁, y₁) (x₂, y₂)) ∧ 
    d = 2 * Real.sqrt 2) :=
by
  sorry

end min_dist_circle_to_line_l255_255915


namespace surveyed_individuals_not_working_percentage_l255_255116

theorem surveyed_individuals_not_working_percentage :
  (55 / 100 * 0 + 35 / 100 * (1 / 8) + 10 / 100 * (1 / 4)) = 6.875 / 100 :=
by
  sorry

end surveyed_individuals_not_working_percentage_l255_255116


namespace m_gt_n_l255_255350

noncomputable def m : ℕ := 2015 ^ 2016
noncomputable def n : ℕ := 2016 ^ 2015

theorem m_gt_n : m > n := by
  sorry

end m_gt_n_l255_255350


namespace number_exceeds_its_part_l255_255876

theorem number_exceeds_its_part (x : ℝ) (h : x = 3/8 * x + 25) : x = 40 :=
by sorry

end number_exceeds_its_part_l255_255876


namespace mohit_discount_l255_255274

variable (SP : ℝ) -- Selling price
variable (CP : ℝ) -- Cost price
variable (discount_percentage : ℝ) -- Discount percentage

-- Conditions
axiom h1 : SP = 21000
axiom h2 : CP = 17500
axiom h3 : discount_percentage = ( (SP - (CP + 0.08 * CP)) / SP) * 100

-- Theorem to prove
theorem mohit_discount : discount_percentage = 10 :=
  sorry

end mohit_discount_l255_255274


namespace population_decrease_rate_l255_255981

theorem population_decrease_rate (r : ℕ) (h₀ : 6000 > 0) (h₁ : 4860 = 6000 * (1 - r / 100)^2) : r = 10 :=
by sorry

end population_decrease_rate_l255_255981


namespace biking_days_in_week_l255_255951

def onurDistancePerDay : ℕ := 250
def hanilDistanceMorePerDay : ℕ := 40
def weeklyDistance : ℕ := 2700

theorem biking_days_in_week : (weeklyDistance / (onurDistancePerDay + hanilDistanceMorePerDay + onurDistancePerDay)) = 5 :=
by
  sorry

end biking_days_in_week_l255_255951


namespace radius_of_two_equal_circles_eq_16_l255_255158

noncomputable def radius_of_congruent_circles : ℝ := 16

theorem radius_of_two_equal_circles_eq_16 :
  ∃ x : ℝ, 
    (∀ r1 r2 r3 : ℝ, r1 = 4 ∧ r2 = r3 ∧ r2 = x ∧ 
    ∃ line : ℝ → ℝ → Prop, 
    (line 0 r1) ∧ (line 0 r2)  ∧ 
    (line 0 r3) ∧ 
    (line r2 r3) ∧
    (line r1 r2)  ∧ (line r1 r3) ∧ (line (r1 + r2) r2) ) 
    → x = 16 := sorry

end radius_of_two_equal_circles_eq_16_l255_255158


namespace quadratic_roots_condition_l255_255507

theorem quadratic_roots_condition (k : ℝ) : 
  (∀ (r s : ℝ), r + s = -k ∧ r * s = 12 → (r + 3) + (s + 3) = k) → k = 3 := 
by 
  sorry

end quadratic_roots_condition_l255_255507


namespace find_x_l255_255522

theorem find_x (x : ℤ) (h : 7 * x - 18 = 66) : x = 12 :=
  sorry

end find_x_l255_255522


namespace pipes_fill_cistern_together_time_l255_255298

theorem pipes_fill_cistern_together_time
  (t : ℝ)
  (h1 : t * (1 / 12 + 1 / 15) + 6 * (1 / 15) = 1) : 
  t = 4 := 
by
  -- Proof is omitted here as instructed
  sorry

end pipes_fill_cistern_together_time_l255_255298


namespace sum_slope_y_intercept_l255_255258

theorem sum_slope_y_intercept (A B C F : ℝ × ℝ) (midpoint_A_C : F = ((A.1 + C.1) / 2, (A.2 + C.2) / 2)) 
  (coords_A : A = (0, 6)) (coords_B : B = (0, 0)) (coords_C : C = (8, 0)) :
  let slope : ℝ := (F.2 - B.2) / (F.1 - B.1)
  let y_intercept : ℝ := B.2
  slope + y_intercept = 3 / 4 := by
{
  -- proof steps
  sorry
}

end sum_slope_y_intercept_l255_255258


namespace accessories_cost_is_200_l255_255263

variable (c_cost a_cost : ℕ)
variable (ps_value ps_sold : ℕ)
variable (john_paid : ℕ)

-- Given Conditions
def computer_cost := 700
def accessories_cost := a_cost
def playstation_value := 400
def playstation_sold := ps_value - (ps_value * 20 / 100)
def john_paid_amount := 580

-- Theorem to be proved
theorem accessories_cost_is_200 :
  ps_value = 400 →
  ps_sold = playstation_sold →
  c_cost = 700 →
  john_paid = 580 →
  john_paid + ps_sold - c_cost = a_cost →
  a_cost = 200 :=
by
  intros
  sorry

end accessories_cost_is_200_l255_255263


namespace gcd_2_l255_255770

-- Define the two numbers obtained from the conditions.
def n : ℕ := 3589 - 23
def m : ℕ := 5273 - 41

-- State that the GCD of n and m is 2.
theorem gcd_2 : Nat.gcd n m = 2 := by
  sorry

end gcd_2_l255_255770


namespace min_n_for_positive_sum_l255_255120

theorem min_n_for_positive_sum (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum : ∀ n, S n = n * (a 0 + a (n - 1)) / 2)
  (h_cond : (a 8 : ℚ) ≠ 0 ∧ ((a 9 : ℚ) / (a 8 : ℚ) < -1))
  (h_min_val : ∃ n, ∀ m, S m ≥ S n) :
  ∃ n, S n > 0 ∧ ∀ m, m < n → S m ≤ 0 :=
sorry

end min_n_for_positive_sum_l255_255120


namespace dried_fruit_percentage_l255_255304

-- Define the percentages for Sue, Jane, and Tom's trail mixes.
structure TrailMix :=
  (nuts : ℝ)
  (dried_fruit : ℝ)

def sue : TrailMix := { nuts := 0.30, dried_fruit := 0.70 }
def jane : TrailMix := { nuts := 0.60, dried_fruit := 0.00 }  -- Note: No dried fruit
def tom : TrailMix := { nuts := 0.40, dried_fruit := 0.50 }

-- Condition: Combined mix contains 45% nuts.
def combined_nuts (sue_nuts jane_nuts tom_nuts : ℝ) : Prop :=
  0.33 * sue_nuts + 0.33 * jane_nuts + 0.33 * tom_nuts = 0.45

-- Condition: Each contributes equally to the total mixture.
def equal_contribution (sue_cont jane_cont tom_cont : ℝ) : Prop :=
  sue_cont = jane_cont ∧ jane_cont = tom_cont

-- Theorem to be proven: Combined mixture contains 40% dried fruit.
theorem dried_fruit_percentage :
  combined_nuts sue.nuts jane.nuts tom.nuts →
  equal_contribution (1 / 3) (1 / 3) (1 / 3) →
  0.33 * sue.dried_fruit + 0.33 * tom.dried_fruit = 0.40 :=
by sorry

end dried_fruit_percentage_l255_255304


namespace isosceles_triangle_angle_l255_255324

theorem isosceles_triangle_angle (x : ℕ) (h1 : 2 * x + x + x = 180) :
  x = 45 ∧ 2 * x = 90 :=
by
  have h2 : 4 * x = 180 := by linarith
  have h3 : x = 45 := by linarith
  have h4 : 2 * x = 90 := by linarith
  exact ⟨h3, h4⟩

end isosceles_triangle_angle_l255_255324


namespace circular_garden_remaining_grass_area_l255_255036

noncomputable def remaining_grass_area (diameter : ℝ) (path_width: ℝ) : ℝ :=
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let path_area := path_width * diameter
  circle_area - path_area

theorem circular_garden_remaining_grass_area :
  remaining_grass_area 10 2 = 25 * Real.pi - 20 := sorry

end circular_garden_remaining_grass_area_l255_255036


namespace find_number_subtract_four_l255_255570

theorem find_number_subtract_four (x : ℤ) (h : 35 + 3 * x = 50) : x - 4 = 1 := by
  sorry

end find_number_subtract_four_l255_255570


namespace minimum_n_for_all_columns_l255_255599

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Function to check if a given number covers all columns from 0 to 9
def covers_all_columns (n : ℕ) : Bool :=
  let columns := (List.range n).map (λ i => triangular_number i % 10)
  List.range 10 |>.all (λ c => c ∈ columns)

theorem minimum_n_for_all_columns : ∃ n, covers_all_columns n ∧ triangular_number n = 253 :=
by 
  sorry

end minimum_n_for_all_columns_l255_255599


namespace school_distance_l255_255245

theorem school_distance (T D : ℝ) (h1 : 5 * (T + 6) = 630) (h2 : 7 * (T - 30) = 630) :
  D = 630 :=
sorry

end school_distance_l255_255245


namespace range_of_f_l255_255213

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Icc (Real.pi / 2 + Real.arctan (-2)) (Real.pi / 2 + Real.arctan 2) :=
sorry

end range_of_f_l255_255213


namespace derivative_of_log_base2_inv_x_l255_255974

noncomputable def my_function (x : ℝ) : ℝ := (Real.log x⁻¹) / (Real.log 2)

theorem derivative_of_log_base2_inv_x : 
  ∀ x : ℝ, x > 0 → deriv my_function x = -1 / (x * Real.log 2) :=
by
  intros x hx
  sorry

end derivative_of_log_base2_inv_x_l255_255974


namespace find_numbers_l255_255994

theorem find_numbers
  (X Y : ℕ)
  (h1 : 10 ≤ X ∧ X < 100)
  (h2 : 10 ≤ Y ∧ Y < 100)
  (h3 : X = 2 * Y)
  (h4 : ∃ a b c d, X = 10 * a + b ∧ Y = 10 * c + d ∧ (c + d = a + b) ∧ (c = a - b ∨ d = a - b)) :
  X = 34 ∧ Y = 17 :=
sorry

end find_numbers_l255_255994


namespace quadratic_inequality_solution_l255_255229

theorem quadratic_inequality_solution (a b c : ℝ) (h1 : a < 0)
  (h2 : (∀ x, ax^2 + bx + c = 0 ↔ x = 1 ∨ x = 3)) : 
  ∀ x, cx^2 + bx + a > 0 ↔ (1/3 < x ∧ x < 1) :=
by
  sorry

end quadratic_inequality_solution_l255_255229


namespace flavored_drink_ratio_l255_255250

theorem flavored_drink_ratio :
  ∃ (F C W: ℚ), F / C = 1 / 7.5 ∧ F / W = 1 / 56.25 ∧ C/W = 6/90 ∧ F / C / 3 = ((F / W) * 2)
:= sorry

end flavored_drink_ratio_l255_255250


namespace range_of_function_l255_255842

theorem range_of_function :
  ∀ y : ℝ, (∃ x : ℝ, y = (1 / 2) ^ (x^2 + 2 * x - 1)) ↔ (0 < y ∧ y ≤ 4) :=
by
  sorry

end range_of_function_l255_255842


namespace division_problem_l255_255276

theorem division_problem (n : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h_div : divisor = 12) (h_quo : quotient = 9) (h_rem : remainder = 1) 
  (h_eq: n = divisor * quotient + remainder) : n = 109 :=
by
  sorry

end division_problem_l255_255276


namespace probability_greater_than_two_on_die_l255_255254

variable {favorable_outcomes : ℕ := 4}
variable {total_outcomes : ℕ := 6}

theorem probability_greater_than_two_on_die : (favorable_outcomes : ℚ) / total_outcomes = 2 / 3 := 
by
  sorry

end probability_greater_than_two_on_die_l255_255254


namespace units_digit_of_sum_is_7_l255_255706

noncomputable def original_num (a b c : ℕ) : ℕ := 100 * a + 10 * b + c
noncomputable def reversed_num (a b c : ℕ) : ℕ := 100 * c + 10 * b + a

theorem units_digit_of_sum_is_7 (a b c : ℕ) (h : a = 2 * c - 3) :
  (original_num a b c + reversed_num a b c) % 10 = 7 := by
  sorry

end units_digit_of_sum_is_7_l255_255706


namespace total_solar_systems_and_planets_l255_255851

theorem total_solar_systems_and_planets (planets : ℕ) (solar_systems_per_planet : ℕ) (h1 : solar_systems_per_planet = 8) (h2 : planets = 20) : (planets + planets * solar_systems_per_planet) = 180 :=
by
  -- translate conditions to definitions
  let solar_systems := planets * solar_systems_per_planet
  -- sum solar systems and planets
  let total := planets + solar_systems
  -- exact proof goal
  exact calc
    (planets + solar_systems) = planets + planets * solar_systems_per_planet : by rfl
                        ... = 20 + 20 * 8                       : by rw [h1, h2]
                        ... = 180                                : by norm_num

end total_solar_systems_and_planets_l255_255851


namespace cheryl_distance_walked_l255_255066

theorem cheryl_distance_walked :
  let s1 := 2  -- speed during the first segment in miles per hour
  let t1 := 3  -- time during the first segment in hours
  let s2 := 4  -- speed during the second segment in miles per hour
  let t2 := 2  -- time during the second segment in hours
  let s3 := 1  -- speed during the third segment in miles per hour
  let t3 := 3  -- time during the third segment in hours
  let s4 := 3  -- speed during the fourth segment in miles per hour
  let t4 := 5  -- time during the fourth segment in hours
  let d1 := s1 * t1  -- distance for the first segment
  let d2 := s2 * t2  -- distance for the second segment
  let d3 := s3 * t3  -- distance for the third segment
  let d4 := s4 * t4  -- distance for the fourth segment
  d1 + d2 + d3 + d4 = 32 :=
by
  sorry

end cheryl_distance_walked_l255_255066


namespace factorize_m_cubed_minus_16m_l255_255211

theorem factorize_m_cubed_minus_16m (m : ℝ) : m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_m_cubed_minus_16m_l255_255211


namespace range_of_a_l255_255355

def A : Set ℝ := { x | x^2 - 3 * x + 2 ≤ 0 }
def B (a : ℝ) : Set ℝ := { x | 1 / (x - 3) < a }

theorem range_of_a (a : ℝ) : A ⊆ B a ↔ a > -1/2 :=
by sorry

end range_of_a_l255_255355


namespace new_trailer_homes_added_l255_255573

theorem new_trailer_homes_added (n : ℕ) (h1 : (20 * 20 + 2 * n)/(20 + n) = 14) : n = 10 :=
by
  sorry

end new_trailer_homes_added_l255_255573


namespace valid_t_range_for_f_l255_255799

theorem valid_t_range_for_f :
  (∀ x : ℝ, |x + 1| + |x - t| ≥ 2015) ↔ t ∈ (Set.Iic (-2016) ∪ Set.Ici 2014) := 
sorry

end valid_t_range_for_f_l255_255799


namespace candies_problem_l255_255382

theorem candies_problem (emily jennifer bob : ℕ) (h1 : emily = 6) 
  (h2 : jennifer = 2 * emily) (h3 : jennifer = 3 * bob) : bob = 4 := by
  -- Lean code to skip the proof
  sorry

end candies_problem_l255_255382


namespace sum_is_945_l255_255171

def sum_of_integers_from_90_to_99 : ℕ :=
  90 + 91 + 92 + 93 + 94 + 95 + 96 + 97 + 98 + 99

theorem sum_is_945 : sum_of_integers_from_90_to_99 = 945 := 
by
  sorry

end sum_is_945_l255_255171


namespace people_per_entrance_l255_255574

theorem people_per_entrance (e p : ℕ) (h1 : e = 5) (h2 : p = 1415) : p / e = 283 := by
  sorry

end people_per_entrance_l255_255574


namespace determine_other_number_l255_255339

theorem determine_other_number (a b : ℤ) (h₁ : 3 * a + 4 * b = 161) (h₂ : a = 17 ∨ b = 17) : 
(a = 31 ∨ b = 31) :=
by
  sorry

end determine_other_number_l255_255339


namespace seashells_count_l255_255002

theorem seashells_count {s : ℕ} (h : s + 6 = 25) : s = 19 :=
by
  sorry

end seashells_count_l255_255002


namespace travel_A_to_D_l255_255760

-- Definitions for the number of roads between each pair of cities
def roads_A_to_B : ℕ := 3
def roads_A_to_C : ℕ := 1
def roads_B_to_C : ℕ := 2
def roads_B_to_D : ℕ := 1
def roads_C_to_D : ℕ := 3

-- Theorem stating the total number of ways to travel from A to D visiting each city exactly once
theorem travel_A_to_D : roads_A_to_B * roads_B_to_C * roads_C_to_D + roads_A_to_C * roads_B_to_C * roads_B_to_D = 20 :=
by
  -- Formal proof goes here
  sorry

end travel_A_to_D_l255_255760


namespace polynomial_remainder_l255_255215

theorem polynomial_remainder :
  let f := X^2023 + 1
  let g := X^6 - X^4 + X^2 - 1
  ∃ (r : Polynomial ℤ), (r = -X^3 + 1) ∧ (∃ q : Polynomial ℤ, f = q * g + r) :=
by
  sorry

end polynomial_remainder_l255_255215


namespace martin_total_waste_is_10_l255_255399

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l255_255399


namespace solution_set_for_f_l255_255490

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 + x else -x^2 + x

theorem solution_set_for_f (x : ℝ) :
  f (x^2 - x + 1) < 12 ↔ -1 < x ∧ x < 2 :=
sorry

end solution_set_for_f_l255_255490


namespace biff_hourly_earnings_l255_255758

theorem biff_hourly_earnings:
  let ticket_cost := 11
  let drinks_snacks_cost := 3
  let headphones_cost := 16
  let wifi_cost_per_hour := 2
  let bus_ride_hours := 3
  let total_non_wifi_expenses := ticket_cost + drinks_snacks_cost + headphones_cost
  let total_wifi_cost := bus_ride_hours * wifi_cost_per_hour
  let total_expenses := total_non_wifi_expenses + total_wifi_cost
  ∀ (x : ℝ), 3 * x = total_expenses → x = 12 :=
by sorry -- Proof skipped

end biff_hourly_earnings_l255_255758


namespace problem_I_problem_II_l255_255394

open Set

-- Definitions of the sets A and B, and their intersections would be needed
def A := {x : ℝ | x ≤ 1 ∨ x ≥ 2}
def B (a : ℝ) := {x : ℝ | a ≤ x ∧ x ≤ 3 * a}

-- (I) When a = 1, find A ∩ B
theorem problem_I : A ∩ (B 1) = {x : ℝ | (2 ≤ x ∧ x ≤ 3) ∨ x = 1} := by
  sorry

-- (II) When A ∩ B = B, find the range of a
theorem problem_II : {a : ℝ | a > 0 ∧ ∀ x, x ∈ B a → x ∈ A} = {a : ℝ | (0 < a ∧ a ≤ 1 / 3) ∨ a ≥ 2} := by
  sorry

end problem_I_problem_II_l255_255394


namespace incorrect_statement_proof_l255_255026

-- Define the conditions as assumptions
def inductive_reasoning_correct : Prop := ∀ (P : Prop), ¬(P → P)
def analogical_reasoning_correct : Prop := ∀ (P Q : Prop), ¬(P → Q)
def reasoning_by_plausibility_correct : Prop := ∀ (P : Prop), ¬(P → P)

-- Define the incorrect statement to be proven
def inductive_reasoning_incorrect_statement : Prop := 
  ¬ (∀ (P Q : Prop), ¬(P ↔ Q))

-- The theorem to be proven
theorem incorrect_statement_proof 
  (h1 : inductive_reasoning_correct)
  (h2 : analogical_reasoning_correct)
  (h3 : reasoning_by_plausibility_correct) : inductive_reasoning_incorrect_statement :=
sorry

end incorrect_statement_proof_l255_255026


namespace jennifer_tanks_l255_255812

theorem jennifer_tanks (initial_tanks : ℕ) (fish_per_initial_tank : ℕ) (total_fish_needed : ℕ) 
  (additional_tanks : ℕ) (fish_per_additional_tank : ℕ) 
  (initial_calculation : initial_tanks = 3) (fish_per_initial_calculation : fish_per_initial_tank = 15)
  (total_fish_calculation : total_fish_needed = 75) (additional_tanks_calculation : additional_tanks = 3) :
  initial_tanks * fish_per_initial_tank + additional_tanks * fish_per_additional_tank = total_fish_needed 
  → fish_per_additional_tank = 10 := 
by sorry

end jennifer_tanks_l255_255812


namespace same_type_as_target_l255_255577

-- Definitions of the polynomials
def optionA (a b : ℝ) : ℝ := a^2 * b
def optionB (a b : ℝ) : ℝ := -2 * a * b^2
def optionC (a b : ℝ) : ℝ := a * b
def optionD (a b c : ℝ) : ℝ := a * b^2 * c

-- Definition of the target polynomial type
def target (a b : ℝ) : ℝ := a * b^2

-- Statement: Option B is of the same type as target
theorem same_type_as_target (a b : ℝ) : optionB a b = -2 * target a b := 
sorry

end same_type_as_target_l255_255577


namespace smallest_n_modulo_5_l255_255335

theorem smallest_n_modulo_5 :
  ∃ n : ℕ, n > 0 ∧ 2^n % 5 = n^7 % 5 ∧
    ∀ m : ℕ, (m > 0 ∧ 2^m % 5 = m^7 % 5) → n ≤ m := 
begin
  use 7,
  split,
  { norm_num, },
  split,
  { norm_num, },
  { intros m h,
    cases h with hpos heq,
    sorry, -- Here would be proof that 7 is the smallest n satisfying 2^n ≡ n^7 mod 5.
  }
end

end smallest_n_modulo_5_l255_255335


namespace koala_fiber_consumption_l255_255533

theorem koala_fiber_consumption (x : ℝ) (H : 12 = 0.30 * x) : x = 40 :=
by
  sorry

end koala_fiber_consumption_l255_255533


namespace area_under_parabola_l255_255145

-- Define the function representing the parabola
def parabola (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- State the theorem about the area under the curve
theorem area_under_parabola : (∫ x in (1 : ℝ)..3, parabola x) = 4 / 3 :=
by
  -- Proof goes here
  sorry

end area_under_parabola_l255_255145


namespace bricks_required_for_courtyard_l255_255743

/-- 
A courtyard is 45 meters long and 25 meters broad needs to be paved with bricks of 
dimensions 15 cm by 7 cm. What will be the total number of bricks required?
-/
theorem bricks_required_for_courtyard 
  (courtyard_length : ℕ) (courtyard_width : ℕ)
  (brick_length : ℕ) (brick_width : ℕ)
  (H1 : courtyard_length = 4500) (H2 : courtyard_width = 2500)
  (H3 : brick_length = 15) (H4 : brick_width = 7) :
  let courtyard_area_cm : ℕ := courtyard_length * courtyard_width
  let brick_area_cm : ℕ := brick_length * brick_width
  let total_bricks : ℕ := (courtyard_area_cm + brick_area_cm - 1) / brick_area_cm
  total_bricks = 107143 := by
  sorry

end bricks_required_for_courtyard_l255_255743


namespace number_of_planes_l255_255721

theorem number_of_planes (total_wings: ℕ) (wings_per_plane: ℕ) 
  (h1: total_wings = 50) (h2: wings_per_plane = 2) : 
  total_wings / wings_per_plane = 25 := by 
  sorry

end number_of_planes_l255_255721


namespace intersection_x_axis_l255_255593

theorem intersection_x_axis (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (7, 3)) (h2 : (x2, y2) = (3, -1)) :
  ∃ x : ℝ, (x, 0) = (4, 0) :=
by sorry

end intersection_x_axis_l255_255593


namespace interest_rate_second_part_l255_255322

theorem interest_rate_second_part (P1 P2: ℝ) (total_sum : ℝ) (rate1 : ℝ) (time1 : ℝ) (time2 : ℝ) (interest_second_part: ℝ ) : 
  total_sum = 2717 → P2 = 1672 → time1 = 8 → rate1 = 3 → time2 = 3 →
  P1 + P2 = total_sum →
  P1 * rate1 * time1 / 100 = P2 * interest_second_part * time2 / 100 →
  interest_second_part = 5 :=
by
  sorry

end interest_rate_second_part_l255_255322


namespace problem_A_problem_B_problem_C_problem_D_l255_255446

theorem problem_A : 2 * Real.sqrt 3 + 3 * Real.sqrt 2 ≠ 5 * Real.sqrt 5 := by
  sorry

theorem problem_B : 3 * Real.sqrt 3 * (3 * Real.sqrt 2) ≠ 3 * Real.sqrt 6 := by
  sorry

theorem problem_C : (Real.sqrt 27 / Real.sqrt 3) = 3 := by
  sorry

theorem problem_D : 2 * Real.sqrt 2 - Real.sqrt 2 ≠ 2 := by
  sorry

end problem_A_problem_B_problem_C_problem_D_l255_255446


namespace women_more_than_men_l255_255715

theorem women_more_than_men 
(M W : ℕ) 
(h_ratio : (M:ℚ) / W = 5 / 9) 
(h_total : M + W = 14) :
W - M = 4 := 
by 
  sorry

end women_more_than_men_l255_255715


namespace refrigerator_volume_unit_l255_255567

theorem refrigerator_volume_unit (V : ℝ) (u : String) : 
  V = 150 → (u = "Liters" ∨ u = "Milliliters" ∨ u = "Cubic meters") → 
  u = "Liters" :=
by
  intro hV hu
  sorry

end refrigerator_volume_unit_l255_255567


namespace larger_solution_of_quadratic_equation_l255_255082

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l255_255082


namespace multiple_of_remainder_l255_255369

theorem multiple_of_remainder (R V D Q k : ℤ) (h1 : R = 6) (h2 : V = 86) (h3 : D = 5 * Q) 
  (h4 : D = k * R + 2) (h5 : V = D * Q + R) : k = 3 := by
  sorry

end multiple_of_remainder_l255_255369


namespace house_painting_l255_255114

theorem house_painting (n : ℕ) (h1 : n = 1000)
  (occupants : Fin n → Fin n) (perm : ∀ i, occupants i ≠ i) :
  ∃ (coloring : Fin n → Fin 3), ∀ i, coloring i ≠ coloring (occupants i) :=
by
  sorry

end house_painting_l255_255114


namespace range_of_values_abs_range_of_values_l255_255499

noncomputable def problem (x y : ℝ) : Prop :=
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1

theorem range_of_values (x y : ℝ) (h : problem x y) :
  2 ≤ (2 * x + y - 1) / x ∧ (2 * x + y - 1) / x ≤ 10 / 3 :=
sorry

theorem abs_range_of_values (x y : ℝ) (h : problem x y) :
  5 - Real.sqrt 2 ≤ abs (x + y + 1) ∧ abs (x + y + 1) ≤ 5 + Real.sqrt 2 :=
sorry

end range_of_values_abs_range_of_values_l255_255499


namespace find_k_point_verification_l255_255494

-- Definition of the linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x + 3

-- Condition that the point (2, 7) lies on the graph of the linear function
def passes_through (k : ℝ) : Prop := linear_function k 2 = 7

-- The actual proof task to verify the value of k
theorem find_k : ∃ k : ℝ, passes_through k ∧ k = 2 :=
by
  sorry

-- The condition that the point (-2, 1) is not on the graph with k = 2
def point_not_on_graph : Prop := ¬ (linear_function 2 (-2) = 1)

-- The actual proof task to verify the point (-2, 1) is not on the graph of y = 2x + 3
theorem point_verification : point_not_on_graph :=
by
  sorry

end find_k_point_verification_l255_255494


namespace find_min_value_l255_255425

noncomputable def min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :=
  (8 * a + b) / (a * b)

theorem find_min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :
  min_value a b h_a h_b h_slope = 18 :=
sorry

end find_min_value_l255_255425


namespace ratio_of_pentagon_side_to_rectangle_width_l255_255600

-- Definitions based on the conditions
def pentagon_perimeter : ℝ := 60
def rectangle_perimeter : ℝ := 60
def rectangle_length (w : ℝ) : ℝ := 2 * w

-- The statement to be proven
theorem ratio_of_pentagon_side_to_rectangle_width :
  ∀ w : ℝ, 2 * (rectangle_length w + w) = rectangle_perimeter → (pentagon_perimeter / 5) / w = 6 / 5 :=
by
  sorry

end ratio_of_pentagon_side_to_rectangle_width_l255_255600


namespace solve_trig_equation_l255_255451

open Real

def in_domain (x : ℝ) : Prop :=
  ∀ k : ℤ, x ≠ k * π / 3

theorem solve_trig_equation (x : ℝ) (h_dom : in_domain x)  :
  (cos x = 0 ∧ (∃ n : ℤ, x = (π / 2) + n * π)) ∨
  (1 - 4 * sin x ^ 2 = 0 ∧ (∃ k : ℤ, x = k * π / 6 ∨ x = k * π / 6 + π)) :=
sorry

end solve_trig_equation_l255_255451


namespace sum_of_f_l255_255000

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) / (3^x + (Real.sqrt 3))

theorem sum_of_f :
  (f (-12) + f (-11) + f (-10) + f (-9) + f (-8) + f (-7) + f (-6) + 
   f (-5) + f (-4) + f (-3) + f (-2) + f (-1) + f (0) + f (1) + f (2) + 
   f (3) + f (4) + f (5) + f (6) + f (7) + f (8) + f (9) + f (10) + 
   f (11) + f (12) + f (13)) = 13 :=
sorry

end sum_of_f_l255_255000


namespace find_first_number_l255_255914

noncomputable def x : ℕ := 7981
noncomputable def y : ℕ := 9409
noncomputable def mean_proportional : ℕ := 8665

theorem find_first_number (mean_is_correct : (mean_proportional^2 = x * y)) : x = 7981 := by
-- Given: mean_proportional^2 = x * y
-- Goal: x = 7981
  sorry

end find_first_number_l255_255914


namespace x_intercept_is_one_l255_255461

theorem x_intercept_is_one (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, -1)) (h2 : (x2, y2) = (-2, 3)) :
    ∃ x : ℝ, (0 = ((y2 - y1) / (x2 - x1)) * (x - x1) + y1) ∧ x = 1 :=
by
  sorry

end x_intercept_is_one_l255_255461


namespace rate_of_mixed_oil_l255_255106

theorem rate_of_mixed_oil (V1 V2 : ℝ) (P1 P2 : ℝ) : 
  (V1 = 10) → 
  (P1 = 50) → 
  (V2 = 5) → 
  (P2 = 67) → 
  ((V1 * P1 + V2 * P2) / (V1 + V2) = 55.67) :=
by
  intros V1_eq P1_eq V2_eq P2_eq
  rw [V1_eq, P1_eq, V2_eq, P2_eq]
  norm_num
  sorry

end rate_of_mixed_oil_l255_255106


namespace value_of_a_l255_255230

theorem value_of_a (a x : ℝ) (h : (3 * x^2 + 2 * a * x = 0) → (x^3 + a * x^2 - (4 / 3) * a = 0)) :
  a = 0 ∨ a = 3 ∨ a = -3 :=
by
  sorry

end value_of_a_l255_255230


namespace females_over_30_prefer_webstream_l255_255052

-- Define the total number of survey participants
def total_participants : ℕ := 420

-- Define the number of participants who prefer WebStream
def prefer_webstream : ℕ := 200

-- Define the number of participants who do not prefer WebStream
def not_prefer_webstream : ℕ := 220

-- Define the number of males who prefer WebStream
def males_prefer : ℕ := 80

-- Define the number of females under 30 who do not prefer WebStream
def females_under_30_not_prefer : ℕ := 90

-- Define the number of females over 30 who do not prefer WebStream
def females_over_30_not_prefer : ℕ := 70

-- Define the total number of females under 30 who do not prefer WebStream
def females_not_prefer : ℕ := females_under_30_not_prefer + females_over_30_not_prefer

-- Define the total number of participants who do not prefer WebStream
def total_not_prefer : ℕ := 220

-- Define the number of males who do not prefer WebStream
def males_not_prefer : ℕ := total_not_prefer - females_not_prefer

-- Define the number of females who prefer WebStream
def females_prefer : ℕ := prefer_webstream - males_prefer

-- Define the total number of females under 30 who prefer WebStream
def females_under_30_prefer : ℕ := total_participants - prefer_webstream - females_under_30_not_prefer

-- Define the remaining females over 30 who prefer WebStream
def females_over_30_prefer : ℕ := females_prefer - females_under_30_prefer

-- The Lean statement to prove
theorem females_over_30_prefer_webstream : females_over_30_prefer = 110 := by
  sorry

end females_over_30_prefer_webstream_l255_255052


namespace first_term_geometric_sequence_b_n_bounded_l255_255688

-- Definition: S_n = 3a_n - 5n for any n in ℕ*
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 3 * a n - 5 * n

-- The sequence a_n is given such that
-- Proving the first term a_1
theorem first_term (a : ℕ → ℝ) (h : ∀ n, S (n + 1) a = S n a + a n + 1 - 5) : 
  a 1 = 5 / 2 :=
sorry

-- Prove that {a_n + 5} is a geometric sequence with common ratio 3/2
theorem geometric_sequence (a : ℕ → ℝ) (h : ∀ n, S n a = 3 * a n - 5 * n) : 
  ∃ r, (∀ n, a (n + 1) + 5 = r * (a n + 5)) ∧ r = 3 / 2 :=
sorry

-- Prove that there exists m such that b_n < m always holds for b_n = (9n + 4) / (a_n + 5)
theorem b_n_bounded (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h1 : ∀ n, b n = (9 * ↑n + 4) / (a n + 5)) 
  (h2 : ∀ n, a n = (15 / 2) * (3 / 2)^(n-1) - 5) :
  ∃ m, ∀ n, b n < m ∧ m = 88 / 45 :=
sorry

end first_term_geometric_sequence_b_n_bounded_l255_255688


namespace polynomial_degree_le_one_l255_255248

theorem polynomial_degree_le_one {P : ℝ → ℝ} (h : ∀ x : ℝ, 2 * P x = P (x + 3) + P (x - 3)) :
  ∃ (a b : ℝ), ∀ x : ℝ, P x = a * x + b :=
sorry

end polynomial_degree_le_one_l255_255248


namespace each_sibling_gets_13_pencils_l255_255958

theorem each_sibling_gets_13_pencils (colored_pencils : ℕ) (black_pencils : ℕ) (siblings : ℕ) (kept_pencils : ℕ) 
  (hyp1 : colored_pencils = 14) (hyp2 : black_pencils = 35) (hyp3 : siblings = 3) (hyp4 : kept_pencils = 10) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end each_sibling_gets_13_pencils_l255_255958


namespace negation_of_existence_implies_universal_l255_255152

theorem negation_of_existence_implies_universal (x : ℝ) :
  (∀ x : ℝ, ¬(x^2 ≤ |x|)) ↔ (∀ x : ℝ, x^2 > |x|) :=
by 
  sorry

end negation_of_existence_implies_universal_l255_255152


namespace tate_total_years_eq_12_l255_255554

-- Definitions based on conditions
def high_school_normal_years : ℕ := 4
def high_school_years : ℕ := high_school_normal_years - 1
def college_years : ℕ := 3 * high_school_years
def total_years : ℕ := high_school_years + college_years

-- Statement to prove
theorem tate_total_years_eq_12 : total_years = 12 := by
  sorry

end tate_total_years_eq_12_l255_255554


namespace river_width_l255_255191

theorem river_width 
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) 
  (h_depth : depth = 2) 
  (h_flow_rate: flow_rate_kmph = 3) 
  (h_volume : volume_per_minute = 4500) : 
  the_width_of_the_river = 45 :=
by
  sorry 

end river_width_l255_255191


namespace arithmetic_sequence_20th_term_l255_255830

-- Definitions for the first term and common difference
def first_term : ℤ := 8
def common_difference : ℤ := -3

-- Define the general term for an arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := first_term + (n - 1) * common_difference

-- The specific property we seek to prove: the 20th term is -49
theorem arithmetic_sequence_20th_term : arithmetic_sequence 20 = -49 := by
  -- Proof is omitted, filled with sorry
  sorry

end arithmetic_sequence_20th_term_l255_255830


namespace ratio_of_volumes_total_surface_area_smaller_cube_l255_255444

-- Definitions using the conditions in (a)
def edge_length_smaller_cube := 4 -- in inches
def edge_length_larger_cube := 24 -- in inches (2 feet converted to inches)

-- Propositions based on the correct answers in (b)
theorem ratio_of_volumes : 
  (edge_length_smaller_cube ^ 3) / (edge_length_larger_cube ^ 3) = 1 / 216 := by
  sorry

theorem total_surface_area_smaller_cube : 
  6 * (edge_length_smaller_cube ^ 2) = 96 := by
  sorry

end ratio_of_volumes_total_surface_area_smaller_cube_l255_255444


namespace equilateral_triangle_perimeter_l255_255964

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l255_255964


namespace gain_percent_calculation_l255_255665

noncomputable def gain_percent (C S : ℝ) : ℝ :=
  (S - C) / C * 100

theorem gain_percent_calculation (C S : ℝ) (h : 50 * C = 46 * S) : 
  gain_percent C S = 100 / 11.5 :=
by
  sorry

end gain_percent_calculation_l255_255665


namespace neg_p_equiv_l255_255246

def p : Prop := ∃ x₀ : ℝ, x₀^2 + 1 > 3 * x₀

theorem neg_p_equiv :
  ¬ p ↔ ∀ x : ℝ, x^2 + 1 ≤ 3 * x := by
  sorry

end neg_p_equiv_l255_255246


namespace probability_of_at_least_two_hits_l255_255750

namespace Probability

def binomial_coefficient (n k : ℕ) : ℕ :=
if k > n then 0 else nat.factorial n / (nat.factorial k * nat.factorial (n - k))

noncomputable def probability_of_at_least_two_hits_in_three_shots (p : ℝ) : ℝ :=
binomial_coefficient 3 2 * (p^2) * (1 - p) + binomial_coefficient 3 3 * (p^3)

theorem probability_of_at_least_two_hits {p : ℝ} (h : p = 0.6) : 
  probability_of_at_least_two_hits_in_three_shots p = 0.648 :=
by 
  sorry

end Probability

end probability_of_at_least_two_hits_l255_255750


namespace find_k_circle_radius_l255_255638

theorem find_k_circle_radius (k : ℝ) :
  (∀ x y : ℝ, (x^2 + 8 * x + y^2 + 4 * y - k = 0) → ((x + 4)^2 + (y + 2)^2 = 7^2)) → k = 29 :=
sorry

end find_k_circle_radius_l255_255638


namespace binomial_square_correct_k_l255_255337

theorem binomial_square_correct_k (k : ℚ) : (∃ t u : ℚ, k = t^2 ∧ 28 = 2 * t * u ∧ 9 = u^2) → k = 196 / 9 :=
by
  sorry

end binomial_square_correct_k_l255_255337


namespace gavrila_travel_distance_l255_255220

-- Declaration of the problem conditions
noncomputable def smartphone_discharge_rate_video : ℝ := 1 / 3
noncomputable def smartphone_discharge_rate_tetris : ℝ := 1 / 5
def average_speed_first_half : ℝ := 80
def average_speed_second_half : ℝ := 60

-- Main theorem statement
theorem gavrila_travel_distance : ∃ d : ℝ, d = 257 := by
  -- Solve for the total travel time t
  let smartphone_discharge_time : ℝ := 15 / 4
  -- Expressing the distance formula
  let distance_1 := smartphone_discharge_time * average_speed_first_half
  let distance_2 := smartphone_discharge_time * average_speed_second_half
  let total_distance := (distance_1 / 2 + distance_2 / 2)
  -- Assert the final distance
  exact ⟨257, by decide⟩

end gavrila_travel_distance_l255_255220


namespace addition_of_two_negatives_l255_255065

theorem addition_of_two_negatives (a b : ℤ) (ha : a < 0) (hb : b < 0) : a + b < a ∧ a + b < b :=
by
  sorry

end addition_of_two_negatives_l255_255065


namespace original_number_l255_255361

theorem original_number (h : 2.04 / 1.275 = 1.6) : 204 / 12.75 = 16 := 
by
  sorry

end original_number_l255_255361


namespace quadrant_of_complex_number_l255_255789

theorem quadrant_of_complex_number
  (h : ∀ x : ℝ, 0 < x → (a^2 + a + 2)/x < 1/x^2 + 1) :
  ∃ a : ℝ, -1 < a ∧ a < 0 ∧ i^27 = -i :=
sorry

end quadrant_of_complex_number_l255_255789


namespace product_of_four_consecutive_odd_numbers_is_perfect_square_l255_255727

theorem product_of_four_consecutive_odd_numbers_is_perfect_square (n : ℤ) :
    (n + 0) * (n + 2) * (n + 4) * (n + 6) = 9 :=
sorry

end product_of_four_consecutive_odd_numbers_is_perfect_square_l255_255727


namespace marlon_goals_l255_255252

theorem marlon_goals :
  ∃ g : ℝ,
    (∀ p f : ℝ, p + f = 40 → g = 0.4 * p + 0.5 * f) → g = 20 :=
by
  sorry

end marlon_goals_l255_255252


namespace ratio_AB_PQ_f_half_func_f_l255_255121

-- Define given conditions
variables {m n : ℝ} -- Lengths of AB and PQ
variables {h : ℝ} -- Height of triangle and rectangle (both are 1)
variables {x : ℝ} -- Variable in the range [0, 1]

-- Same area and height conditions
axiom areas_equal : m / 2 = n
axiom height_equal : h = 1

-- Given the areas are equal and height is 1
theorem ratio_AB_PQ : m / n = 2 :=
by sorry -- Proof of the ratio 

-- Given the specific calculation for x = 1/2
theorem f_half (hx : x = 1 / 2) (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  f (1 / 2) = 3 / 4 :=
by sorry -- Proof of function value at 1/2

-- Prove the expression of the function f(x)
theorem func_f (f : ℝ → ℝ) (heq : ∀ x, (f x) * n = (m / 2) * (2 * x - x^2)) :
  ∀ x, 0 ≤ x → x ≤ 1 → f x = 2 * x - x^2 :=
by sorry -- Proof of the function expression


end ratio_AB_PQ_f_half_func_f_l255_255121


namespace jess_remaining_blocks_l255_255200

-- Define the number of blocks for each segment of Jess's errands
def blocks_to_post_office : Nat := 24
def blocks_to_store : Nat := 18
def blocks_to_gallery : Nat := 15
def blocks_to_library : Nat := 14
def blocks_to_work : Nat := 22
def blocks_already_walked : Nat := 9

-- Calculate the total blocks to be walked
def total_blocks : Nat :=
  blocks_to_post_office + blocks_to_store + blocks_to_gallery + blocks_to_library + blocks_to_work

-- The remaining blocks Jess needs to walk
def blocks_remaining : Nat :=
  total_blocks - blocks_already_walked

-- The statement to be proved
theorem jess_remaining_blocks : blocks_remaining = 84 :=
by
  sorry

end jess_remaining_blocks_l255_255200


namespace fisherman_daily_earnings_l255_255422

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l255_255422


namespace compute_difference_of_squares_l255_255466

theorem compute_difference_of_squares : (303^2 - 297^2) = 3600 := by
  sorry

end compute_difference_of_squares_l255_255466


namespace total_bananas_l255_255693

theorem total_bananas (bunches_8 bunches_7 : ℕ) (bananas_8 bananas_7 : ℕ) (h_bunches_8 : bunches_8 = 6) (h_bananas_8 : bananas_8 = 8) (h_bunches_7 : bunches_7 = 5) (h_bananas_7 : bananas_7 = 7) :
  bunches_8 * bananas_8 + bunches_7 * bananas_7 = 83 :=
by
  rw [h_bunches_8, h_bananas_8, h_bunches_7, h_bananas_7]
  norm_num

end total_bananas_l255_255693


namespace neither_necessary_nor_sufficient_l255_255540

def set_M : Set ℝ := {x | x > 2}
def set_P : Set ℝ := {x | x < 3}

theorem neither_necessary_nor_sufficient (x : ℝ) :
  (x ∈ set_M ∨ x ∈ set_P) ↔ (x ∉ set_M ∩ set_P) :=
sorry

end neither_necessary_nor_sufficient_l255_255540


namespace triangle_expression_simplification_l255_255088

variable (a b c : ℝ)

theorem triangle_expression_simplification (h1 : a + b > c) 
                                           (h2 : a + c > b) 
                                           (h3 : b + c > a) :
  |a - b - c| + |b - a - c| - |c - a + b| = a - b + c :=
sorry

end triangle_expression_simplification_l255_255088


namespace helga_ratio_l255_255357

variable (a b c d : ℕ)

def helga_shopping (a b c d total_shoes pairs_first_three : ℕ) : Prop :=
  a = 7 ∧
  b = a + 2 ∧
  c = 0 ∧
  a + b + c + d = total_shoes ∧
  pairs_first_three = a + b + c ∧
  total_shoes = 48 ∧
  (d : ℚ) / (pairs_first_three : ℚ) = 2

theorem helga_ratio : helga_shopping 7 9 0 32 48 16 := by
  sorry

end helga_ratio_l255_255357


namespace all_cells_equal_l255_255623

-- Define the infinite grid
def Grid := ℕ → ℕ → ℕ

-- Define the condition on the grid values
def is_min_mean_grid (g : Grid) : Prop :=
  ∀ i j : ℕ, g i j ≥ (g (i-1) j + g (i+1) j + g i (j-1) + g i (j+1)) / 4

-- Main theorem
theorem all_cells_equal (g : Grid) (h : is_min_mean_grid g) : ∃ a : ℕ, ∀ i j : ℕ, g i j = a := 
sorry

end all_cells_equal_l255_255623


namespace first_class_seat_count_l255_255460

theorem first_class_seat_count :
  let seats_first_class := 10
  let seats_business_class := 30
  let seats_economy_class := 50
  let people_economy_class := seats_economy_class / 2
  let people_business_and_first := people_economy_class
  let unoccupied_business := 8
  let people_business_class := seats_business_class - unoccupied_business
  people_business_and_first - people_business_class = 3 := by
  sorry

end first_class_seat_count_l255_255460


namespace find_a_l255_255654

noncomputable def coefficient_of_x3_in_expansion (a : ℝ) : ℝ :=
  6 * a^2 - 15 * a + 20 

theorem find_a (a : ℝ) (h : coefficient_of_x3_in_expansion a = 56) : a = 6 ∨ a = -1 :=
  sorry

end find_a_l255_255654


namespace prime_factors_of_difference_l255_255689

theorem prime_factors_of_difference (A B : ℕ) (h_neq : A ≠ B) : 
  ∃ p, Nat.Prime p ∧ p ∣ (Nat.gcd (9 * A - 9 * B + 10) (9 * B - 9 * A - 10)) :=
by
  sorry

end prime_factors_of_difference_l255_255689


namespace problem_statement_l255_255267

theorem problem_statement (p q : ℝ)
  (α β : ℝ) (h1 : α ≠ β) (h1' : α + β = -p) (h1'' : α * β = -2)
  (γ δ : ℝ) (h2 : γ ≠ δ) (h2' : γ + δ = -q) (h2'' : γ * δ = -3) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = 3 * (q ^ 2 - p ^ 2) - 2 * q + 1 := by
  sorry

end problem_statement_l255_255267


namespace minimum_strips_cover_circle_l255_255725

theorem minimum_strips_cover_circle (l R : ℝ) (hl : l > 0) (hR : R > 0) :
  ∃ (k : ℕ), (k : ℝ) * l ≥ 2 * R ∧ ((k - 1 : ℕ) : ℝ) * l < 2 * R :=
sorry

end minimum_strips_cover_circle_l255_255725


namespace find_k_values_for_intersection_l255_255375

noncomputable def intersects_at_one_point (z : ℂ) (k : ℝ) : Prop :=
  abs (z - 4) = 3 * abs (z + 4) ∧ abs z = k

theorem find_k_values_for_intersection :
  ∃ k, (∀ z : ℂ, intersects_at_one_point z k) ↔ (k = 2 ∨ k = 8) :=
begin
  sorry
end

end find_k_values_for_intersection_l255_255375


namespace all_are_truth_tellers_l255_255338

-- Define the possible states for Alice, Bob, and Carol
inductive State
| true_teller
| liar

-- Define the predicates for each person's statements
def alice_statement (B C : State) : Prop :=
  B = State.true_teller ∨ C = State.true_teller

def bob_statement (A C : State) : Prop :=
  A = State.true_teller ∧ C = State.true_teller

def carol_statement (A B : State) : Prop :=
  A = State.true_teller → B = State.true_teller

-- The theorem to be proved
theorem all_are_truth_tellers
    (A B C : State)
    (alice: A = State.true_teller → alice_statement B C)
    (bob: B = State.true_teller → bob_statement A C)
    (carol: C = State.true_teller → carol_statement A B)
    : A = State.true_teller ∧ B = State.true_teller ∧ C = State.true_teller :=
by
  sorry

end all_are_truth_tellers_l255_255338


namespace final_score_l255_255861

theorem final_score (questions_first_half questions_second_half points_per_question : ℕ) (h1 : questions_first_half = 5) (h2 : questions_second_half = 5) (h3 : points_per_question = 5) : 
  (questions_first_half + questions_second_half) * points_per_question = 50 :=
by
  sorry

end final_score_l255_255861


namespace family_trip_eggs_l255_255312

theorem family_trip_eggs (adults girls boys : ℕ)
  (eggs_per_adult : ℕ) (eggs_per_girl : ℕ) (extra_eggs_for_boy : ℕ) :
  adults = 3 →
  eggs_per_adult = 3 →
  girls = 7 →
  eggs_per_girl = 1 →
  boys = 10 →
  extra_eggs_for_boy = 1 →
  (adults * eggs_per_adult + girls * eggs_per_girl + boys * (eggs_per_girl + extra_eggs_for_boy)) = 36 :=
by
  intros
  sorry

end family_trip_eggs_l255_255312


namespace jane_reading_days_l255_255128

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l255_255128


namespace product_not_end_in_1999_l255_255294

theorem product_not_end_in_1999 (a b c d e : ℕ) (h : a + b + c + d + e = 200) : 
  ¬(a * b * c * d * e % 10000 = 1999) := 
by
  sorry

end product_not_end_in_1999_l255_255294


namespace grabbed_books_l255_255793

-- Definitions from conditions
def initial_books : ℕ := 99
def boxed_books : ℕ := 3 * 15
def room_books : ℕ := 21
def table_books : ℕ := 4
def kitchen_books : ℕ := 18
def current_books : ℕ := 23

-- Proof statement
theorem grabbed_books : (boxed_books + room_books + table_books + kitchen_books = initial_books - (23 - current_books)) → true := sorry

end grabbed_books_l255_255793


namespace inequality_x_y_z_l255_255498

-- Definitions for the variables
variables {x y z : ℝ} 
variable {n : ℕ}

-- Positive numbers and summation condition
axiom h1 : 0 < x ∧ 0 < y ∧ 0 < z
axiom h2 : x + y + z = 1

-- The theorem to be proven
theorem inequality_x_y_z (h1 : 0 < x ∧ 0 < y ∧ 0 < z) (h2 : x + y + z = 1) (hn : n > 0) : 
  x^n + y^n + z^n ≥ (1 : ℝ) / (3:ℝ)^(n-1) :=
sorry

end inequality_x_y_z_l255_255498


namespace min_value_proof_l255_255388

noncomputable def min_value_expression (α β : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin β - 7)^2 + (3 * Real.sin α + 4 * Real.cos β - 12)^2

theorem min_value_proof :
  ∃ α β : ℝ, min_value_expression α β = 48 := by
  sorry

end min_value_proof_l255_255388


namespace fraction_of_remaining_birds_left_l255_255157

theorem fraction_of_remaining_birds_left (B : ℕ) (F : ℚ) (hB : B = 60)
  (H : (1/3) * (2/3 : ℚ) * B * (1 - F) = 8) :
  F = 4/5 := 
sorry

end fraction_of_remaining_birds_left_l255_255157


namespace largest_angle_of_quadrilateral_l255_255560

open Real

theorem largest_angle_of_quadrilateral 
  (PQ QR RS : ℝ)
  (angle_RQP angle_SRQ largest_angle : ℝ)
  (h1: PQ = QR) 
  (h2: QR = RS) 
  (h3: angle_RQP = 60)
  (h4: angle_SRQ = 100)
  (h5: largest_angle = 130) : 
  largest_angle = 130 := by
  sorry

end largest_angle_of_quadrilateral_l255_255560


namespace debby_soda_bottles_l255_255067

noncomputable def total_bottles (d t : ℕ) : ℕ := d * t

theorem debby_soda_bottles :
  ∀ (d t: ℕ), d = 9 → t = 40 → total_bottles d t = 360 :=
by
  intros d t h1 h2
  sorry

end debby_soda_bottles_l255_255067


namespace additional_track_length_needed_l255_255044

theorem additional_track_length_needed
  (vertical_rise : ℝ) (initial_grade final_grade : ℝ) (initial_horizontal_length final_horizontal_length : ℝ) : 
  vertical_rise = 400 →
  initial_grade = 0.04 →
  final_grade = 0.03 →
  initial_horizontal_length = (vertical_rise / initial_grade) →
  final_horizontal_length = (vertical_rise / final_grade) →
  final_horizontal_length - initial_horizontal_length = 3333 :=
by
  intros h_vertical_rise h_initial_grade h_final_grade h_initial_horizontal_length h_final_horizontal_length
  sorry

end additional_track_length_needed_l255_255044


namespace zane_spent_more_on_cookies_l255_255428

theorem zane_spent_more_on_cookies
  (o c : ℕ) -- number of Oreos and cookies
  (pO pC : ℕ) -- price of each Oreo and cookie
  (h_ratio : 9 * o = 4 * c) -- ratio condition
  (h_price_O : pO = 2) -- price of Oreo
  (h_price_C : pC = 3) -- price of cookie
  (h_total : o + c = 65) -- total number of items
  : 3 * c - 2 * o = 95 := 
begin
  sorry
end

end zane_spent_more_on_cookies_l255_255428


namespace find_a_l255_255095

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then
  x * (x + 1)
else
  -((-x) * ((-x) + 1))

theorem find_a (a : ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_pos : ∀ x : ℝ, x >= 0 → f x = x * (x + 1)) (h_a: f a = -2) : a = -1 :=
sorry

end find_a_l255_255095


namespace cube_volume_surface_area_l255_255726

-- Define volume and surface area conditions
def volume_condition (x : ℝ) (s : ℝ) : Prop := s^3 = 3 * x
def surface_area_condition (x : ℝ) (s : ℝ) : Prop := 6 * s^2 = x

-- The main theorem statement
theorem cube_volume_surface_area (x : ℝ) (s : ℝ) :
  volume_condition x s → surface_area_condition x s → x = 5832 :=
by
  intros h_volume h_area
  sorry

end cube_volume_surface_area_l255_255726


namespace amount_of_juice_p_in_a_l255_255587

  def total_p : ℚ := 24
  def total_v : ℚ := 25
  def ratio_a : ℚ := 4 / 1
  def ratio_y : ℚ := 1 / 5

  theorem amount_of_juice_p_in_a :
    ∃ P_a : ℚ, ∃ V_a : ℚ, ∃ P_y : ℚ, ∃ V_y : ℚ,
      P_a / V_a = ratio_a ∧ P_y / V_y = ratio_y ∧
      P_a + P_y = total_p ∧ V_a + V_y = total_v ∧ P_a = 20 :=
  by
    sorry
  
end amount_of_juice_p_in_a_l255_255587


namespace store_loss_l255_255604

noncomputable def calculation (x y : ℕ) : ℤ :=
  let revenue : ℕ := 60 * 2
  let cost : ℕ := x + y
  revenue - cost

theorem store_loss (x y : ℕ) (hx : (60 - x) * 2 = x) (hy : (y - 60) * 2 = y) :
  calculation x y = -40 := by
    sorry

end store_loss_l255_255604


namespace fundraiser_goal_l255_255563

theorem fundraiser_goal (bronze_donation silver_donation gold_donation goal : ℕ)
  (bronze_families silver_families gold_family : ℕ)
  (H_bronze_amount : bronze_families * bronze_donation = 250)
  (H_silver_amount : silver_families * silver_donation = 350)
  (H_gold_amount : gold_family * gold_donation = 100)
  (H_goal : goal = 750) :
  goal - (bronze_families * bronze_donation + silver_families * silver_donation + gold_family * gold_donation) = 50 :=
by
  sorry

end fundraiser_goal_l255_255563


namespace fractional_equation_no_solution_l255_255247

theorem fractional_equation_no_solution (a : ℝ) :
  (¬ ∃ x, x ≠ 1 ∧ x ≠ 0 ∧ ((x - a) / (x - 1) - 3 / x = 1)) → (a = 1 ∨ a = -2) :=
by
  sorry

end fractional_equation_no_solution_l255_255247


namespace closure_of_A_range_of_a_l255_255687

-- Definitions for sets A and B
def A (x : ℝ) : Prop := x < -1 ∨ x > -0.5
def B (x a : ℝ) : Prop := a - 1 ≤ x ∧ x ≤ a + 1

-- 1. Closure of A
theorem closure_of_A :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ -0.5) ↔ (∀ x : ℝ, A x) :=
sorry

-- 2. Range of a when A ∪ B = ℝ
theorem range_of_a (B_condition : ∀ x : ℝ, B x a) :
  (∀ a : ℝ, -1 ≤ x ∨ x ≥ -0.5) ↔ (-1.5 ≤ a ∧ a ≤ 0) :=
sorry

end closure_of_A_range_of_a_l255_255687


namespace new_paint_intensity_l255_255822

variable (V : ℝ)  -- V is the volume of the original 50% intensity red paint.
variable (I₁ I₂ : ℝ)  -- I₁ is the intensity of the original paint, I₂ is the intensity of the replaced paint.
variable (f : ℝ)  -- f is the fraction of the original paint being replaced.

-- Assume given conditions
axiom intensity_original : I₁ = 0.5
axiom intensity_new : I₂ = 0.25
axiom fraction_replaced : f = 0.8

-- Prove that the new intensity is 30%
theorem new_paint_intensity :
  (f * I₂ + (1 - f) * I₁) = 0.3 := 
by 
  -- This is the main theorem we want to prove
  sorry

end new_paint_intensity_l255_255822


namespace pencils_in_drawer_after_operations_l255_255437

def initial_pencils : ℝ := 2
def pencils_added : ℝ := 3.5
def pencils_removed : ℝ := 1.2

theorem pencils_in_drawer_after_operations : ⌊initial_pencils + pencils_added - pencils_removed⌋ = 4 := by
  sorry

end pencils_in_drawer_after_operations_l255_255437


namespace product_of_numerator_and_denominator_l255_255166

-- Defining the repeating decimal as a fraction in lowest terms
def repeating_decimal_as_fraction_in_lowest_terms : ℚ :=
  1 / 37

-- Theorem to prove the product of the numerator and the denominator
theorem product_of_numerator_and_denominator :
  (repeating_decimal_as_fraction_in_lowest_terms.num.natAbs *
   repeating_decimal_as_fraction_in_lowest_terms.den) = 37 :=
by
  -- declaration of the needed fact and its direct consequence
  sorry

end product_of_numerator_and_denominator_l255_255166


namespace single_elimination_games_needed_l255_255805

theorem single_elimination_games_needed (n : ℕ) (n_pos : n > 0) :
  (number_of_games_needed : ℕ) = n - 1 :=
by
  sorry

end single_elimination_games_needed_l255_255805


namespace max_basketballs_l255_255571

theorem max_basketballs (x : ℕ) (h1 : 80 * x + 50 * (40 - x) ≤ 2800) : x ≤ 26 := sorry

end max_basketballs_l255_255571


namespace distinct_integer_sums_count_l255_255205

def is_special_fraction (a b : ℕ) : Prop := a + b = 20

def special_fractions : Set ℚ :=
  {q | ∃ (a b : ℕ), is_special_fraction a b ∧ q = (a : ℚ) / (b : ℚ)}

def possible_sums_of_special_fractions : Set ℚ :=
  { sum | ∃ (q1 q2 : ℚ), q1 ∈ special_fractions ∧ q2 ∈ special_fractions ∧ sum = q1 + q2 }

def integer_sums : Set ℤ :=
  { z | z ∈ possible_sums_of_special_fractions ∧ z.den = 1 }

theorem distinct_integer_sums_count : ∃ n : ℕ, n = 15 ∧ n = integer_sums.to_finset.card :=
by
  sorry

end distinct_integer_sums_count_l255_255205


namespace circumference_divided_by_diameter_l255_255874

noncomputable def radius : ℝ := 15
noncomputable def circumference : ℝ := 90
noncomputable def diameter : ℝ := 2 * radius

theorem circumference_divided_by_diameter :
  circumference / diameter = 3 := by
  sorry

end circumference_divided_by_diameter_l255_255874


namespace cars_meet_time_l255_255440

theorem cars_meet_time (t : ℝ) (highway_length : ℝ) (speed_car1 : ℝ) (speed_car2 : ℝ)
  (h1 : highway_length = 105) (h2 : speed_car1 = 15) (h3 : speed_car2 = 20) :
  15 * t + 20 * t = 105 → t = 3 := by
  sorry

end cars_meet_time_l255_255440


namespace four_nabla_seven_l255_255720

-- Define the operation ∇
def nabla (a b : ℤ) : ℚ :=
  (a + b) / (1 + a * b)

theorem four_nabla_seven :
  nabla 4 7 = 11 / 29 :=
by
  sorry

end four_nabla_seven_l255_255720


namespace each_sibling_gets_13_pencils_l255_255959

theorem each_sibling_gets_13_pencils (colored_pencils : ℕ) (black_pencils : ℕ) (siblings : ℕ) (kept_pencils : ℕ) 
  (hyp1 : colored_pencils = 14) (hyp2 : black_pencils = 35) (hyp3 : siblings = 3) (hyp4 : kept_pencils = 10) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end each_sibling_gets_13_pencils_l255_255959


namespace hydrogen_atoms_count_l255_255037

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

end hydrogen_atoms_count_l255_255037


namespace value_of_f_2012_1_l255_255089

noncomputable def f : ℝ → ℝ :=
sorry

-- Condition 1: f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(x + 3) = -f(x)
axiom periodicity_f : ∀ x : ℝ, f (x + 3) = -f x

-- Condition 3: f(x) = 2x + 3 for -3 ≤ x ≤ 0
axiom defined_f_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x = 2 * x + 3

-- Assertion to prove
theorem value_of_f_2012_1 : f 2012.1 = -1.2 :=
by sorry

end value_of_f_2012_1_l255_255089


namespace remaining_grass_area_l255_255742

theorem remaining_grass_area 
  (d : ℝ) (r : ℝ) (path_width : ℝ) (center_to_edge : ℝ) 
  (h1 : d = 16) (h2 : r = d / 2) (h3 : path_width = 4) (h4 : center_to_edge = 2)
  (h5 : center_to_edge + path_width = r) :
  let total_area := π * r^2 in
  let remaining_grass_area := total_area - (π * (r - path_width)^2) + (π * path_width * center_to_edge) in
  remaining_grass_area = 56 * π + 16 :=
by
  sorry

end remaining_grass_area_l255_255742


namespace map_scale_to_yards_l255_255186

theorem map_scale_to_yards :
  (6.25 * 500) / 3 = 1041 + 2 / 3 := 
by sorry

end map_scale_to_yards_l255_255186


namespace oranges_thrown_away_l255_255603

theorem oranges_thrown_away (initial_oranges new_oranges current_oranges : ℕ) (x : ℕ) 
  (h1 : initial_oranges = 50)
  (h2 : new_oranges = 24)
  (h3 : current_oranges = 34) : 
  initial_oranges - x + new_oranges = current_oranges → x = 40 :=
by
  intros h
  rw [h1, h2, h3] at h
  sorry

end oranges_thrown_away_l255_255603


namespace remainder_of_452867_div_9_l255_255858

theorem remainder_of_452867_div_9 : (452867 % 9) = 5 := by
  sorry

end remainder_of_452867_div_9_l255_255858


namespace plane_distance_l255_255316

theorem plane_distance :
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  total_distance_AD = 550 :=
by
  intros
  let distance_AB := 100
  let distance_BC := distance_AB + 50
  let distance_CD := 2 * distance_BC
  let total_distance_AD := distance_AB + distance_BC + distance_CD
  sorry

end plane_distance_l255_255316


namespace mixed_number_fraction_division_and_subtraction_l255_255854

theorem mixed_number_fraction_division_and_subtraction :
  ( (11 / 6) / (11 / 4) ) - (1 / 2) = 1 / 6 := 
sorry

end mixed_number_fraction_division_and_subtraction_l255_255854


namespace union_setA_setB_l255_255686

noncomputable def setA : Set ℝ := { x : ℝ | 2 / (x + 1) ≥ 1 }
noncomputable def setB : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x ∧ x < 0 }

theorem union_setA_setB : setA ∪ setB = { x : ℝ | -1 < x ∧ x ≤ 1 } :=
by
  sorry

end union_setA_setB_l255_255686


namespace ratio_apps_optimal_l255_255918

theorem ratio_apps_optimal (max_apps : ℕ) (recommended_apps : ℕ) (apps_to_delete : ℕ) (current_apps : ℕ)
  (h_max_apps : max_apps = 50)
  (h_recommended_apps : recommended_apps = 35)
  (h_apps_to_delete : apps_to_delete = 20)
  (h_current_apps : current_apps = max_apps + apps_to_delete) :
  current_apps / recommended_apps = 2 :=
by {
  sorry
}

end ratio_apps_optimal_l255_255918


namespace general_term_formula_sum_of_sequence_l255_255528

-- Define the arithmetic sequence {a_n}
def a (n : ℕ) : ℤ := n - 1

-- Conditions: a_5 = 4, a_3 + a_8 = 9
def cond1 : Prop := a 5 = 4
def cond2 : Prop := a 3 + a 8 = 9

theorem general_term_formula (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, a n = n - 1 :=
by
  -- Place holder for proof
  sorry

-- Define the sequence {b_n}
def b (n : ℕ) : ℤ := 2 * a n - 1

-- Sum of the first n terms of b_n
def S (n : ℕ) : ℤ := n * (n - 2)

theorem sum_of_sequence (h1 : cond1) (h2 : cond2) : ∀ n : ℕ, (Finset.range (n + 1)).sum b = S n :=
by
  -- Place holder for proof
  sorry

end general_term_formula_sum_of_sequence_l255_255528


namespace more_boys_than_girls_l255_255436

theorem more_boys_than_girls : 
  let girls := 28.0
  let boys := 35.0
  boys - girls = 7.0 :=
by
  sorry

end more_boys_than_girls_l255_255436


namespace initial_passengers_is_350_l255_255380

variable (N : ℕ)

def initial_passengers (N : ℕ) : Prop :=
  let after_first_train := 9 * N / 10
  let after_second_train := 27 * N / 35
  let after_third_train := 108 * N / 175
  after_third_train = 216

theorem initial_passengers_is_350 : initial_passengers 350 := 
  sorry

end initial_passengers_is_350_l255_255380


namespace part1_part2_l255_255923

variables (x y z : ℝ)
open Real

-- Part 1
theorem part1 (
  h1 : x > 0 ∧ y > 0 ∧ z > 0) 
  (h2 : x * y * z = 8) 
  (h3 : x + y < 7)
  : (x / (1 + x) + y / (1 + y) > 2 * sqrt (x * y / (x * y + 8))) := 
sorry

-- Part 2
theorem part2 (
  h1 : x > 0 ∧ y > 0 ∧ z > 0) 
  (h2 : x * y * z = 8)
  : ceil (∑ cyc, 1 / sqrt (1 + x)) = 2 := 
sorry

end part1_part2_l255_255923


namespace avg_time_stopped_per_hour_l255_255206

-- Definitions and conditions
def avgSpeedInMotion : ℝ := 75
def overallAvgSpeed : ℝ := 40

-- Statement to prove
theorem avg_time_stopped_per_hour :
  (1 - overallAvgSpeed / avgSpeedInMotion) * 60 = 28 := 
by
  sorry

end avg_time_stopped_per_hour_l255_255206


namespace probability_of_at_least_one_two_l255_255590

theorem probability_of_at_least_one_two (x y z : ℕ) (hx : 1 ≤ x ∧ x ≤ 6) (hy : 1 ≤ y ∧ y ≤ 6) (hz : 1 ≤ z ∧ z ≤ 6) (hxy : x + y = z) :
  let outcomes := [(1, 1), (1, 2), (2, 1), (1, 3), (2, 2), (3, 1), (1, 4), (2, 3), (3, 2), (4, 1), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)] in
  let valid_outcomes := (1, 1) :: (1, 2) :: (2, 1) :: (1, 3) :: (2, 2) :: (3, 1) :: (1, 4) :: (2, 3) :: (3, 2) :: (4, 1) :: (1, 5) :: (2, 4) :: (3, 3) :: (4, 2) :: (5, 1) :: [] in
  let favorable_outcomes := list.countp (λ (p : ℕ × ℕ), p.fst = 2 ∨ p.snd = 2) valid_outcomes in
  let total_outcomes := list.length valid_outcomes in
  (favorable_outcomes.to_nat / total_outcomes.to_nat : ℚ) = 8 / 15 :=
sorry

end probability_of_at_least_one_two_l255_255590


namespace latest_first_pump_time_l255_255902

theorem latest_first_pump_time 
  (V : ℝ) -- Volume of the pool
  (x y : ℝ) -- Productivity of first and second pumps respectively
  (t : ℝ) -- Time of operation of the first pump until the second pump is turned on
  (h1 : 2*x + 2*y = V/2) -- Condition from 10 AM to 12 PM
  (h2 : 5*x + 5*y = V/2) -- Condition from 12 PM to 5 PM
  (h3 : t*x + 2*x + 2*y = V/2) -- Condition for early morning until 12 PM
  (hx_pos : 0 < x) -- Assume productivity of first pump is positive
  (hy_pos : 0 < y) -- Assume productivity of second pump is positive
  : t ≥ 3 :=
by
  -- The proof goes here...
  sorry

end latest_first_pump_time_l255_255902


namespace sum_of_products_of_roots_eq_neg3_l255_255391

theorem sum_of_products_of_roots_eq_neg3 {p q r s : ℂ} 
  (h : ∀ {x : ℂ}, 4 * x^4 - 8 * x^3 + 12 * x^2 - 16 * x + 9 = 0 → (x = p ∨ x = q ∨ x = r ∨ x = s)) : 
  p * q + p * r + p * s + q * r + q * s + r * s = -3 := 
sorry

end sum_of_products_of_roots_eq_neg3_l255_255391


namespace axis_of_symmetry_of_shifted_function_l255_255821

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem axis_of_symmetry_of_shifted_function :
  (∃ x : ℝ, g x = 1 ∧ x = Real.pi / 12) :=
by
  sorry

end axis_of_symmetry_of_shifted_function_l255_255821


namespace distinct_numbers_div_sum_diff_l255_255953

theorem distinct_numbers_div_sum_diff (n : ℕ) : 
  ∃ (numbers : Fin n → ℕ), 
    ∀ i j, i ≠ j → (numbers i + numbers j) % (numbers i - numbers j) = 0 := 
by
  sorry

end distinct_numbers_div_sum_diff_l255_255953


namespace expression_divisible_by_a_square_l255_255954

theorem expression_divisible_by_a_square (n : ℕ) (a : ℤ) : 
  a^2 ∣ ((a * n - 1) * (a + 1) ^ n + 1) := 
sorry

end expression_divisible_by_a_square_l255_255954


namespace twice_joan_more_than_karl_l255_255260

-- Define the conditions
def J : ℕ := 158
def total : ℕ := 400
def K : ℕ := total - J

-- Define the theorem to be proven
theorem twice_joan_more_than_karl :
  2 * J - K = 74 := by
    -- Skip the proof steps using 'sorry'
    sorry

end twice_joan_more_than_karl_l255_255260


namespace compare_M_N_l255_255011

theorem compare_M_N (a b c : ℝ) (h1 : a > 0) (h2 : b < -2 * a) : 
  (|a - b + c| + |2 * a + b|) < (|a + b + c| + |2 * a - b|) :=
by
  sorry

end compare_M_N_l255_255011


namespace phi_value_l255_255010

theorem phi_value (phi : ℝ) (h : 0 < phi ∧ phi < π) 
  (hf : ∀ x : ℝ, 3 * Real.sin (2 * abs x - π / 3 + phi) = 3 * Real.sin (2 * x - π / 3 + phi)) 
  : φ = 5 * π / 6 :=
by 
  sorry

end phi_value_l255_255010


namespace equilateral_triangle_perimeter_l255_255968

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_l255_255968


namespace ruffy_age_difference_l255_255696

theorem ruffy_age_difference (R O : ℕ) (hR : R = 9) (hRO : R = (3/4 : ℚ) * O) :
  (R - 4) - (1 / 2 : ℚ) * (O - 4) = 1 :=
by 
  sorry

end ruffy_age_difference_l255_255696


namespace f_f_2_l255_255681

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 2 then 2 * Real.exp (x - 1) else Real.log (2^x - 1) / Real.log 3

theorem f_f_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_l255_255681


namespace sum_divisible_by_12_l255_255063

theorem sum_divisible_by_12 :
  ((2150 + 2151 + 2152 + 2153 + 2154 + 2155) % 12) = 3 := by
  sorry

end sum_divisible_by_12_l255_255063


namespace new_equation_incorrect_l255_255062

-- Definition of a function to change each digit of a number by +1 or -1 randomly.
noncomputable def modify_digit (num : ℕ) : ℕ := sorry

-- Proposition stating the original problem's condition and conclusion.
theorem new_equation_incorrect (a b : ℕ) (c := a + b) (a' b' c' : ℕ)
    (h1 : a' = modify_digit a)
    (h2 : b' = modify_digit b)
    (h3 : c' = modify_digit c) :
    a' + b' ≠ c' :=
sorry

end new_equation_incorrect_l255_255062


namespace paper_fold_ratio_l255_255050

theorem paper_fold_ratio (paper_side : ℕ) (fold_fraction : ℚ) (cut_fraction : ℚ)
  (thin_section_width thick_section_width : ℕ) (small_width large_width : ℚ)
  (P_small P_large : ℚ) (ratio : ℚ) :
  paper_side = 6 →
  fold_fraction = 1 / 3 →
  cut_fraction = 2 / 3 →
  thin_section_width = 2 →
  thick_section_width = 4 →
  small_width = 2 →
  large_width = 16 / 3 →
  P_small = 2 * (6 + small_width) →
  P_large = 2 * (6 + large_width) →
  ratio = P_small / P_large →
  ratio = 12 / 17 :=
by
  sorry

end paper_fold_ratio_l255_255050


namespace right_triangle_area_l255_255979

theorem right_triangle_area (hypotenuse : ℝ)
  (angle_deg : ℝ)
  (h_hyp : hypotenuse = 10 * Real.sqrt 2)
  (h_angle : angle_deg = 45) : 
  (1 / 2) * (hypotenuse / Real.sqrt 2)^2 = 50 := 
by 
  sorry

end right_triangle_area_l255_255979


namespace neg_prop_p_l255_255305

-- Define the function f as a real-valued function
variable (f : ℝ → ℝ)

-- Definitions for the conditions in the problem
def prop_p := ∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0

-- Theorem stating the negation of proposition p
theorem neg_prop_p : ¬prop_p f ↔ ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
by 
  sorry

end neg_prop_p_l255_255305


namespace eduardo_needs_l255_255189

variable (flour_per_24_cookies sugar_per_24_cookies : ℝ)
variable (num_cookies : ℝ)

axiom h_flour : flour_per_24_cookies = 1.5
axiom h_sugar : sugar_per_24_cookies = 0.5
axiom h_cookies : num_cookies = 120

theorem eduardo_needs (scaling_factor : ℝ) 
    (flour_needed : ℝ)
    (sugar_needed : ℝ)
    (h_scaling : scaling_factor = num_cookies / 24)
    (h_flour_needed : flour_needed = flour_per_24_cookies * scaling_factor)
    (h_sugar_needed : sugar_needed = sugar_per_24_cookies * scaling_factor) :
  flour_needed = 7.5 ∧ sugar_needed = 2.5 :=
sorry

end eduardo_needs_l255_255189


namespace tangent_line_equation_is_correct_l255_255289

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

end tangent_line_equation_is_correct_l255_255289


namespace determine_digit_X_l255_255763

theorem determine_digit_X (X : ℕ) (hX : 1 ≤ X ∧ X ≤ 9) (h : 510 / X = 10 * 4 + X + 2 * X) : X = 8 :=
sorry

end determine_digit_X_l255_255763


namespace geomSeriesSum_eq_683_l255_255465

/-- Define the first term, common ratio, and number of terms -/
def firstTerm : ℤ := -1
def commonRatio : ℤ := -2
def numTerms : ℕ := 11

/-- Function to calculate the sum of the geometric series -/
def geomSeriesSum (a r : ℤ) (n : ℕ) : ℤ :=
  a * ((r^n - 1) / (r - 1))

/-- The main theorem stating that the sum of the series equals 683 -/
theorem geomSeriesSum_eq_683 :
  geomSeriesSum firstTerm commonRatio numTerms = 683 :=
by sorry

end geomSeriesSum_eq_683_l255_255465


namespace financial_loss_example_l255_255811

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ := 
  P * (1 + r / n) ^ (n * t)

noncomputable def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := 
  P * (1 + r * t)

theorem financial_loss_example :
  let P := 10000
  let r1 := 0.06
  let r2 := 0.05
  let t := 3
  let n := 4
  let A1 := compound_interest P r1 n t
  let A2 := simple_interest P r2 t
  abs (A1 - A2 - 456.18) < 0.01 := by
    sorry

end financial_loss_example_l255_255811


namespace rectangle_area_difference_l255_255240

theorem rectangle_area_difference :
  let area (l w : ℝ) := l * w
  let combined_area (l w : ℝ) := 2 * area l w
  combined_area 11 19 - combined_area 9.5 11 = 209 :=
by
  sorry

end rectangle_area_difference_l255_255240


namespace calculate_expression_l255_255331

theorem calculate_expression : 
  3 * 995 + 4 * 996 + 5 * 997 + 6 * 998 + 7 * 999 - 4985 * 3 = 9980 :=
by
  sorry

end calculate_expression_l255_255331


namespace slices_left_for_phill_correct_l255_255409

-- Define the initial conditions about the pizza and the distribution.
def initial_pizza := 1
def slices_after_first_cut := initial_pizza * 2
def slices_after_second_cut := slices_after_first_cut * 2
def slices_after_third_cut := slices_after_second_cut * 2
def total_slices_given_to_two_friends := 2 * 2
def total_slices_given_to_three_friends := 3 * 1
def total_slices_given_out := total_slices_given_to_two_friends + total_slices_given_to_three_friends
def slices_left_for_phill := slices_after_third_cut - total_slices_given_out

-- State the theorem we need to prove.
theorem slices_left_for_phill_correct : slices_left_for_phill = 1 := by sorry

end slices_left_for_phill_correct_l255_255409


namespace mr_bird_on_time_58_mph_l255_255948

def mr_bird_travel_speed_exactly_on_time (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) : ℝ :=
  58

theorem mr_bird_on_time_58_mph (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) :
  mr_bird_travel_speed_exactly_on_time d t h₁ h₂ = 58 := 
  by
  sorry

end mr_bird_on_time_58_mph_l255_255948


namespace ratio_of_numbers_l255_255433

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 124) (h2 : y = 3 * x) : x / Nat.gcd x y = 1 ∧ y / Nat.gcd x y = 3 := by
  sorry

end ratio_of_numbers_l255_255433


namespace max_roots_poly_interval_l255_255645

noncomputable theory

open Polynomial

def max_roots_in_interval (P : Polynomial ℤ) : ℝ → ℝ → ℕ :=
  λ a b, (map (algebraMap ℤ ℝ) P).roots.countInInterval (Ioo a b)
  
theorem max_roots_poly_interval :
  ∀ P : Polynomial ℤ,
  degree P = 2022 ∧ leadingCoeff P = 1 →
  max_roots_in_interval P 0 1 ≤ 2021 :=
by sorry

end max_roots_poly_interval_l255_255645


namespace complex_circle_intersection_l255_255378

theorem complex_circle_intersection (z : ℂ) (k : ℝ) :
  (|z - 4| = 3 * |z + 4| ∧ |z| = k) →
  (k = 0.631 ∨ k = 25.369) :=
by
  sorry

end complex_circle_intersection_l255_255378


namespace card_stack_partition_l255_255434

theorem card_stack_partition (n k : ℕ) (cards : Multiset ℕ) (h1 : ∀ x ∈ cards, x ∈ Finset.range (n + 1)) (h2 : cards.sum = k * n!) :
  ∃ stacks : List (Multiset ℕ), stacks.length = k ∧ ∀ stack ∈ stacks, stack.sum = n! :=
sorry

end card_stack_partition_l255_255434


namespace shooting_prob_l255_255881

theorem shooting_prob (p q : ℚ) (h: p + q = 1) (n : ℕ) 
  (cond1: p = 2/3) 
  (cond2: q = 1 - p) 
  (cond3: n = 5) : 
  (q ^ (n-1)) = 1/81 := 
by 
  sorry

end shooting_prob_l255_255881


namespace linear_dependent_iff_38_div_3_l255_255898

theorem linear_dependent_iff_38_div_3 (k : ℚ) :
  k = 38 / 3 ↔ ∃ (α β γ : ℚ), α ≠ 0 ∨ β ≠ 0 ∨ γ ≠ 0 ∧
    α * 1 + β * 4 + γ * 7 = 0 ∧
    α * 2 + β * 5 + γ * 8 = 0 ∧
    α * 3 + β * k + γ * 9 = 0 :=
by
  sorry

end linear_dependent_iff_38_div_3_l255_255898


namespace ironman_age_l255_255021

theorem ironman_age (T C P I : ℕ) (h1 : T = 13 * C) (h2 : C = 7 * P) (h3 : I = P + 32) (h4 : T = 1456) : I = 48 := 
by
  sorry

end ironman_age_l255_255021


namespace gcd_mn_eq_one_l255_255443

def m : ℤ := 123^2 + 235^2 - 347^2
def n : ℤ := 122^2 + 234^2 - 348^2

theorem gcd_mn_eq_one : Int.gcd m n = 1 := 
by
  sorry

end gcd_mn_eq_one_l255_255443


namespace tangent_line_b_value_l255_255561

theorem tangent_line_b_value (a k b : ℝ) 
  (h_curve : ∀ x, x^3 + a * x + 1 = 3 ↔ x = 2)
  (h_derivative : k = 3 * 2^2 - 3)
  (h_tangent : 3 = k * 2 + b) : b = -15 :=
sorry

end tangent_line_b_value_l255_255561


namespace employee_salary_l255_255161

theorem employee_salary (X Y : ℝ) (h1 : X = 1.2 * Y) (h2 : X + Y = 528) : Y = 240 :=
by
  sorry

end employee_salary_l255_255161


namespace max_marks_l255_255448

theorem max_marks (M : ℕ) (h1 : M * 33 / 100 = 175 + 56) : M = 700 :=
by
  sorry

end max_marks_l255_255448


namespace bowls_initially_bought_l255_255749

theorem bowls_initially_bought 
  (x : ℕ) 
  (cost_per_bowl : ℕ := 13) 
  (revenue_per_bowl : ℕ := 17)
  (sold_bowls : ℕ := 108)
  (profit_percentage : ℝ := 23.88663967611336) 
  (approx_x : ℝ := 139) :
  (23.88663967611336 / 100) * (cost_per_bowl : ℝ) * (x : ℝ) = 
    (sold_bowls * revenue_per_bowl) - (sold_bowls * cost_per_bowl) → 
  abs ((x : ℝ) - approx_x) < 0.5 :=
by
  sorry

end bowls_initially_bought_l255_255749


namespace unshaded_squares_in_tenth_figure_l255_255511

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + d * (n - 1)

theorem unshaded_squares_in_tenth_figure :
  arithmetic_sequence 8 4 10 = 44 :=
by
  sorry

end unshaded_squares_in_tenth_figure_l255_255511


namespace p_sufficient_not_necessary_for_q_l255_255503

-- Given conditions p and q
def p_geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

def q_product_equality (a b c d : ℝ) : Prop :=
  a * d = b * c

-- Theorem statement: p implies q, but q does not imply p
theorem p_sufficient_not_necessary_for_q (a b c d : ℝ) :
  (p_geometric_sequence a b c d → q_product_equality a b c d) ∧
  (¬ (q_product_equality a b c d → p_geometric_sequence a b c d)) :=
by
  sorry

end p_sufficient_not_necessary_for_q_l255_255503


namespace ducks_and_geese_difference_l255_255199

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l255_255199


namespace marble_solid_color_percentage_l255_255740

theorem marble_solid_color_percentage (a b : ℕ) (h1 : a = 5) (h2 : b = 85) : a + b = 90 := 
by
  sorry

end marble_solid_color_percentage_l255_255740


namespace range_of_t_sum_of_squares_l255_255235

-- Define the conditions and the problem statement in Lean

variables (a b c t x : ℝ)
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (ineq1 : |x + 1| - |x - 2| ≥ |t - 1| + t)
variables (sum_pos : 2 * a + b + c = 2)

theorem range_of_t :
  (∃ x, |x + 1| - |x - 2| ≥ |t - 1| + t) → t ≤ 2 :=
sorry

theorem sum_of_squares :
  2 * a + b + c = 2 → 0 < a → 0 < b → 0 < c → a^2 + b^2 + c^2 ≥ 2 / 3 :=
sorry

end range_of_t_sum_of_squares_l255_255235


namespace angle_SRT_l255_255360

-- Define angles in degrees
def angle_P : ℝ := 50
def angle_Q : ℝ := 60
def angle_R : ℝ := 40

-- Define the problem: Prove that angle SRT is 30 degrees given the above conditions
theorem angle_SRT : 
  (angle_P = 50 ∧ angle_Q = 60 ∧ angle_R = 40) → (∃ angle_SRT : ℝ, angle_SRT = 30) :=
by
  intros h
  sorry

end angle_SRT_l255_255360


namespace snowfall_total_l255_255541

theorem snowfall_total (snowfall_wed snowfall_thu snowfall_fri : ℝ)
  (h_wed : snowfall_wed = 0.33)
  (h_thu : snowfall_thu = 0.33)
  (h_fri : snowfall_fri = 0.22) :
  snowfall_wed + snowfall_thu + snowfall_fri = 0.88 :=
by
  rw [h_wed, h_thu, h_fri]
  norm_num

end snowfall_total_l255_255541


namespace find_value_l255_255390

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity_condition : ∀ x : ℝ, f (2 + x) = f (-x)
axiom value_at_half : f (1/2) = 1/2

theorem find_value : f (2023 / 2) = 1/2 := by
  sorry

end find_value_l255_255390


namespace equilateral_triangle_perimeter_l255_255967

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_l255_255967


namespace fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l255_255877

section FoldingNumberLine

-- Part (1)
def coincides_point_3_if_minus2_2_fold (x : ℝ) : Prop :=
  x = -3

theorem fold_minus2_2_3_coincides_neg3 :
  coincides_point_3_if_minus2_2_fold 3 :=
by
  sorry

-- Part (2) ①
def coincides_point_7_if_minus1_3_fold (x : ℝ) : Prop :=
  x = -5

theorem fold_minus1_3_7_coincides_neg5 :
  coincides_point_7_if_minus1_3_fold 7 :=
by
  sorry

-- Part (2) ②
def B_position_after_folding (m : ℝ) (h : m > 0) (A B : ℝ) : Prop :=
  B = 1 + m / 2

theorem fold_distanceA_to_B_coincide (m : ℝ) (h : m > 0) (A B : ℝ) :
  B_position_after_folding m h A B :=
by
  sorry

end FoldingNumberLine

end fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l255_255877


namespace five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l255_255714

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

end five_students_in_a_row_five_students_with_constraints_five_students_into_three_classes_l255_255714


namespace tenth_term_is_513_l255_255091

def nth_term (n : ℕ) : ℕ :=
  2^(n-1) + 1

theorem tenth_term_is_513 : nth_term 10 = 513 := 
by 
  sorry

end tenth_term_is_513_l255_255091


namespace factorize_expression_l255_255341

variable {R : Type} [Ring R]
variables (a b x y : R)

theorem factorize_expression :
  8 * a * x - b * y + 4 * a * y - 2 * b * x = (4 * a - b) * (2 * x + y) :=
sorry

end factorize_expression_l255_255341


namespace bricks_lay_calculation_l255_255796

theorem bricks_lay_calculation (b c d : ℕ) (h1 : 0 < c) (h2 : 0 < d) : 
  ∃ y : ℕ, y = (b * (b + d) * (c + d))/(c * d) :=
sorry

end bricks_lay_calculation_l255_255796


namespace car_owners_without_motorcycles_l255_255939

theorem car_owners_without_motorcycles (total_adults cars motorcycles no_vehicle : ℕ) 
  (h1 : total_adults = 560) (h2 : cars = 520) (h3 : motorcycles = 80) (h4 : no_vehicle = 10) : 
  cars - (total_adults - no_vehicle - cars - motorcycles) = 470 := 
by
  sorry

end car_owners_without_motorcycles_l255_255939


namespace arith_sign_change_geo_sign_change_l255_255118

-- Definitions for sequences
def arith_sequence (a₁ d : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => arith_sequence a₁ d n + d

def geo_sequence (a₁ r : ℝ) : ℕ → ℝ
| 0 => a₁
| (n + 1) => geo_sequence a₁ r n * r

-- Problem statement
theorem arith_sign_change :
  ∀ (a₁ d : ℝ), (∃ N : ℕ, arith_sequence a₁ d N = 0) ∨ (∀ n m : ℕ, (arith_sequence a₁ d n) * (arith_sequence a₁ d m) ≥ 0) :=
sorry

theorem geo_sign_change :
  ∀ (a₁ r : ℝ), r < 0 → ∀ n : ℕ, (geo_sequence a₁ r n) * (geo_sequence a₁ r (n + 1)) < 0 :=
sorry

end arith_sign_change_geo_sign_change_l255_255118


namespace side_length_of_square_l255_255035

theorem side_length_of_square (r : ℝ) (A : ℝ) (s : ℝ) 
  (h1 : π * r^2 = 36 * π) 
  (h2 : s = 2 * r) : 
  s = 12 :=
by 
  sorry

end side_length_of_square_l255_255035


namespace exists_a_star_b_eq_a_l255_255279

variable {S : Type*} [CommSemigroup S]

def exists_element_in_S (star : S → S → S) : Prop :=
  ∃ a : S, ∀ b : S, star a b = a

theorem exists_a_star_b_eq_a
  (star : S → S → S)
  (comm : ∀ a b : S, star a b = star b a)
  (assoc : ∀ a b c : S, star (star a b) c = star a (star b c))
  (exists_a : ∃ a : S, star a a = a) :
  exists_element_in_S star := sorry

end exists_a_star_b_eq_a_l255_255279


namespace moon_temp_difference_l255_255695

def temp_difference (T_day T_night : ℤ) : ℤ := T_day - T_night

theorem moon_temp_difference :
  temp_difference 127 (-183) = 310 :=
by
  sorry

end moon_temp_difference_l255_255695


namespace martha_no_daughters_count_l255_255135

-- Definitions based on conditions
def total_people : ℕ := 40
def martha_daughters : ℕ := 8
def granddaughters_per_child (x : ℕ) : ℕ := if x = 1 then 8 else 0

-- Statement of the problem
theorem martha_no_daughters_count : 
  (total_people - martha_daughters) +
  (martha_daughters - (total_people - martha_daughters) / 8) = 36 := 
  by
    sorry

end martha_no_daughters_count_l255_255135


namespace scenario_a_scenario_b_l255_255774

-- Define the chessboard and the removal function
def is_adjacent (x1 y1 x2 y2 : ℕ) : Prop :=
  (x1 = x2 ∧ (y1 = y2 + 1 ∨ y1 + 1 = y2)) ∨ (y1 = y2 ∧ (x1 = x2 + 1 ∨ x1 + 1 = x2))

def is_square (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 8 ∧ 1 ≤ y ∧ y ≤ 8

-- Define a Hamiltonian path on the chessboard
inductive HamiltonianPath : (ℕ × ℕ) → (ℕ → (ℕ × ℕ)) → ℕ → Prop
| empty : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)), HamiltonianPath start path 0
| step : Π (start : ℕ × ℕ) (path : ℕ → (ℕ × ℕ)) (n : ℕ),
    is_adjacent (path n).1 (path n).2 (path (n+1)).1 (path (n+1)).2 →
    HamiltonianPath start path n →
    (is_square (path (n + 1)).1 (path (n + 1)).2 ∧ ¬ (∃ m < n + 1, path m = path (n + 1))) →
    HamiltonianPath start path (n + 1)

-- State the main theorems
theorem scenario_a : 
  ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 62 ∧
    (∀ n, path n ≠ (2, 2)) := sorry

theorem scenario_b :
  ¬ ∃ (path : ℕ → (ℕ × ℕ)),
    HamiltonianPath (3, 2) path 61 ∧
    (∀ n, path n ≠ (2, 2) ∧ path n ≠ (7, 7)) := sorry

end scenario_a_scenario_b_l255_255774


namespace solve_for_x_l255_255075

theorem solve_for_x (x : ℝ) (h : 3^(3 * x - 2) = (1 : ℝ) / 27) : x = -(1 : ℝ) / 3 :=
sorry

end solve_for_x_l255_255075


namespace sum_of_powers_of_i_l255_255217

theorem sum_of_powers_of_i (i : ℂ) (h : i^2 = -1) : i + i^2 + i^3 + i^4 + i^5 = i := by
  sorry

end sum_of_powers_of_i_l255_255217


namespace weight_of_A_l255_255701

variable (A B C D E : ℕ)

axiom cond1 : A + B + C = 180
axiom cond2 : A + B + C + D = 260
axiom cond3 : E = D + 3
axiom cond4 : B + C + D + E = 256

theorem weight_of_A : A = 87 :=
by
  sorry

end weight_of_A_l255_255701


namespace quad_relation_l255_255006

theorem quad_relation
  (α AI BI CI DI : ℝ)
  (h1 : AB = α * (AI / CI + BI / DI))
  (h2 : BC = α * (BI / DI + CI / AI))
  (h3 : CD = α * (CI / AI + DI / BI))
  (h4 : DA = α * (DI / BI + AI / CI)) :
  AB + CD = AD + BC := by
  sorry

end quad_relation_l255_255006


namespace max_value_of_a_l255_255234

theorem max_value_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a ≥ 0) → a ≤ 4 := 
by {
  sorry
}

end max_value_of_a_l255_255234


namespace students_in_both_clubs_l255_255155

theorem students_in_both_clubs (total_students drama_club science_club : ℕ) 
  (students_either_or_both both_clubs : ℕ) 
  (h_total_students : total_students = 250)
  (h_drama_club : drama_club = 80)
  (h_science_club : science_club = 120)
  (h_students_either_or_both : students_either_or_both = 180)
  (h_inclusion_exclusion : students_either_or_both = drama_club + science_club - both_clubs) :
  both_clubs = 20 :=
  by sorry

end students_in_both_clubs_l255_255155


namespace cab_speed_fraction_l255_255034

def usual_time := 30 -- The usual time of the journey in minutes
def delay_time := 6   -- The delay time in minutes
def usual_speed : ℝ := sorry -- Placeholder for the usual speed
def reduced_speed : ℝ := sorry -- Placeholder for the reduced speed

-- Given the conditions:
-- 1. The usual time for the cab to cover the journey is 30 minutes.
-- 2. The cab is 6 minutes late when walking at a reduced speed.
-- Prove that the fraction of the cab's usual speed it is walking at is 5/6

theorem cab_speed_fraction : (reduced_speed / usual_speed) = (5 / 6) :=
sorry

end cab_speed_fraction_l255_255034


namespace vector_t_perpendicular_l255_255658

theorem vector_t_perpendicular (t : ℝ) :
  let a := (2, 4)
  let b := (-1, 1)
  let c := (2 + t, 4 - t)
  b.1 * c.1 + b.2 * c.2 = 0 → t = 1 := by
  sorry

end vector_t_perpendicular_l255_255658


namespace domain_of_function_l255_255930

theorem domain_of_function :
  (∀ x : ℝ, (x + 1 ≥ 0) ∧ (x ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 0)) :=
sorry

end domain_of_function_l255_255930


namespace negation_of_p_l255_255934

variable (x y : ℝ)

def proposition_p := ∀ x y : ℝ, x^2 + y^2 - 1 > 0 

theorem negation_of_p : (¬ proposition_p) ↔ (∃ x y : ℝ, x^2 + y^2 - 1 ≤ 0) := by
  sorry

end negation_of_p_l255_255934


namespace number_of_triangles_number_of_rays_l255_255296

-- Definitions based on the problem conditions
def total_points : ℕ := 9
def collinear_points : ℕ := 4
def non_collinear_points : ℕ := 5

-- Lean proof problem:
theorem number_of_triangles (h1 : collinear_points = 4) (h2 : total_points = 9) :
  nat.choose total_points 3 - nat.choose collinear_points 3 = 80 :=
sorry

theorem number_of_rays (h3 : non_collinear_points = 5) (h4 : collinear_points = 4) (h5 : total_points = 9) :
  nat.perm non_collinear_points 2 + 2 * 1 + 2 * 2 + (nat.choose collinear_points 1 * nat.choose non_collinear_points 1 * nat.perm 2 2) = 66 :=
sorry

end number_of_triangles_number_of_rays_l255_255296


namespace player_weekly_earnings_l255_255984

structure Performance :=
  (points assists rebounds steals : ℕ)

def base_pay (avg_points : ℕ) : ℕ :=
  if avg_points >= 30 then 10000 else 8000

def assists_bonus (total_assists : ℕ) : ℕ :=
  if total_assists >= 20 then 5000
  else if total_assists >= 10 then 3000
  else 1000

def rebounds_bonus (total_rebounds : ℕ) : ℕ :=
  if total_rebounds >= 40 then 5000
  else if total_rebounds >= 20 then 3000
  else 1000

def steals_bonus (total_steals : ℕ) : ℕ :=
  if total_steals >= 15 then 5000
  else if total_steals >= 5 then 3000
  else 1000

def total_payment (performances : List Performance) : ℕ :=
  let total_points := performances.foldl (λ acc p => acc + p.points) 0
  let total_assists := performances.foldl (λ acc p => acc + p.assists) 0
  let total_rebounds := performances.foldl (λ acc p => acc + p.rebounds) 0
  let total_steals := performances.foldl (λ acc p => acc + p.steals) 0
  let avg_points := total_points / performances.length
  base_pay avg_points + assists_bonus total_assists + rebounds_bonus total_rebounds + steals_bonus total_steals
  
theorem player_weekly_earnings :
  let performances := [
    Performance.mk 30 5 7 3,
    Performance.mk 28 6 5 2,
    Performance.mk 32 4 9 1,
    Performance.mk 34 3 11 2,
    Performance.mk 26 2 8 3
  ]
  total_payment performances = 23000 := by 
    sorry

end player_weekly_earnings_l255_255984


namespace same_terminal_side_angles_l255_255017

theorem same_terminal_side_angles (k : ℤ) :
  ∃ (k1 k2 : ℤ), k1 * 360 - 1560 = -120 ∧ k2 * 360 - 1560 = 240 :=
by
  -- Conditions and property definitions can be added here if needed
  sorry

end same_terminal_side_angles_l255_255017


namespace distance_to_lake_l255_255610

theorem distance_to_lake 
  {d : ℝ} 
  (h1 : ¬ (d ≥ 8))
  (h2 : ¬ (d ≤ 7))
  (h3 : ¬ (d ≤ 6)) : 
  (7 < d) ∧ (d < 8) :=
by
  sorry

end distance_to_lake_l255_255610


namespace problem_l255_255517

open_locale big_operators

theorem problem (a b : ℝ) (h : 2^a + log 2 a = 4^b + 2 * log 4 b) : a < 2 * b :=
sorry

end problem_l255_255517


namespace max_b_squared_l255_255553

theorem max_b_squared (a b : ℤ) (h : (a + b) * (a + b) + a * (a + b) + b = 0) : b^2 ≤ 81 :=
sorry

end max_b_squared_l255_255553


namespace range_of_a_l255_255777

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then x^2 + 2 * a else -x

theorem range_of_a (a : ℝ) (h : a < 0) (hf : f a (1 - a) ≥ f a (1 + a)) : -2 ≤ a ∧ a ≤ -1 :=
  sorry

end range_of_a_l255_255777


namespace second_player_wins_l255_255138

theorem second_player_wins :
  ∀ (n : ℕ), n ∈ {1, 2, ..., 1000} →
  ∃ (m : ℕ), m ∈ {1, 2, ..., 1000} ∧ (n + m = 1001) →
  (n^2 - m^2) % 13 = 0 :=
by {
  intros n hn,
  use 1001 - n,
  split,
  {
    sorry -- prove that 1001 - n is in the set
  },
  {
    intro h,
    have h1: n + (1001 - n) = 1001 := by linarith,
    rw h1,
    sorry -- prove divisibility by 13
  }
}

end second_player_wins_l255_255138


namespace marbles_per_friend_l255_255678

theorem marbles_per_friend (total_marbles friends : ℕ) (h1 : total_marbles = 5504) (h2 : friends = 64) :
  total_marbles / friends = 86 :=
by {
  -- Proof will be added here
  sorry
}

end marbles_per_friend_l255_255678


namespace correct_option_exponent_equality_l255_255170

theorem correct_option_exponent_equality (a b : ℕ) : 
  (\left(2 * a * b^2\right)^2 = 4 * a^2 * b^4) :=
by
  sorry

end correct_option_exponent_equality_l255_255170


namespace sixth_year_fee_l255_255608

def first_year_fee : ℕ := 80
def yearly_increase : ℕ := 10

def membership_fee (year : ℕ) : ℕ :=
  first_year_fee + (year - 1) * yearly_increase

theorem sixth_year_fee : membership_fee 6 = 130 :=
  by sorry

end sixth_year_fee_l255_255608


namespace find_percentage_l255_255310

noncomputable def percentage (P : ℝ) : Prop :=
  (P / 100) * 1265 / 6 = 354.2

theorem find_percentage : ∃ (P : ℝ), percentage P ∧ P = 168 :=
by
  sorry

end find_percentage_l255_255310


namespace problem1_problem2_l255_255787

noncomputable def f (x : ℝ) : ℝ :=
  |x - 2| - |2 * x + 1|

theorem problem1 (x : ℝ) :
  f x ≤ 2 ↔ x ≤ -1 ∨ -1/3 ≤ x :=
sorry

theorem problem2 (a : ℝ) (b : ℝ) :
  (∀ x, |a + b| - |a - b| ≥ f x) → (a ≥ 5 / 4 ∨ a ≤ -5 / 4) :=
sorry

end problem1_problem2_l255_255787


namespace solve_ellipse_correct_m_l255_255786

noncomputable def ellipse_is_correct_m : Prop :=
  ∃ (m : ℝ), 
    (m > 6) ∧
    ((m - 2) - (10 - m) = 4) ∧
    (m = 8)

theorem solve_ellipse_correct_m : ellipse_is_correct_m :=
sorry

end solve_ellipse_correct_m_l255_255786


namespace total_distance_covered_l255_255183

theorem total_distance_covered (up_speed down_speed up_time down_time : ℕ) (H1 : up_speed = 30) (H2 : down_speed = 50) (H3 : up_time = 5) (H4 : down_time = 5) :
  (up_speed * up_time + down_speed * down_time) = 400 := 
by
  sorry

end total_distance_covered_l255_255183


namespace conic_section_eccentricity_l255_255238

theorem conic_section_eccentricity (m : ℝ) (h : 2 * 8 = m^2) :
    (∃ e : ℝ, ((e = (Real.sqrt 2) / 2) ∨ (e = Real.sqrt 3))) :=
by
  sorry

end conic_section_eccentricity_l255_255238


namespace gas_volumes_correct_l255_255736

noncomputable def west_gas_vol_per_capita : ℝ := 21428
noncomputable def non_west_gas_vol : ℝ := 185255
noncomputable def non_west_population : ℝ := 6.9
noncomputable def non_west_gas_vol_per_capita : ℝ := non_west_gas_vol / non_west_population

noncomputable def russia_gas_vol_68_percent : ℝ := 30266.9
noncomputable def russia_gas_vol : ℝ := russia_gas_vol_68_percent * 100 / 68
noncomputable def russia_population : ℝ := 0.147
noncomputable def russia_gas_vol_per_capita : ℝ := russia_gas_vol / russia_population

theorem gas_volumes_correct :
  west_gas_vol_per_capita = 21428 ∧
  non_west_gas_vol_per_capita = 26848.55 ∧
  russia_gas_vol_per_capita = 302790.13 := by
    sorry

end gas_volumes_correct_l255_255736


namespace relation_of_a_and_b_l255_255515

theorem relation_of_a_and_b (a b : ℝ) (h : 2^a + Real.log a / Real.log 2 = 4^b + 2 * Real.log b / Real.log 4) : a < 2 * b :=
sorry

end relation_of_a_and_b_l255_255515


namespace find_abc_l255_255631

theorem find_abc
  (a b c : ℝ)
  (h : ∀ x y z : ℝ, |a * x + b * y + c * z| + |b * x + c * y + a * z| + |c * x + a * y + b * z| = |x| + |y| + |z|):
  (a = 1 ∧ b = 0 ∧ c = 0) ∨ (a = -1 ∧ b = 0 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 1 ∧ c = 0) ∨ (a = 0 ∧ b = -1 ∧ c = 0) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 1) ∨ (a = 0 ∧ b = 0 ∧ c = -1) :=
sorry

end find_abc_l255_255631


namespace shift_sin_to_cos_l255_255438

open Real

theorem shift_sin_to_cos:
  ∀ x: ℝ, 3 * cos (2 * x) = 3 * sin (2 * (x + π / 6) - π / 6) :=
by 
  sorry

end shift_sin_to_cos_l255_255438


namespace Leah_lost_11_dollars_l255_255534

-- Define the conditions
def LeahEarned : ℕ := 28
def MilkshakeCost : ℕ := LeahEarned / 7
def RemainingAfterMilkshake : ℕ := LeahEarned - MilkshakeCost
def Savings : ℕ := RemainingAfterMilkshake / 2
def WalletAfterSavings : ℕ := RemainingAfterMilkshake - Savings
def WalletAfterDog : ℕ := 1

-- Define the theorem to prove Leah's loss
theorem Leah_lost_11_dollars : WalletAfterSavings - WalletAfterDog = 11 := 
by 
  sorry

end Leah_lost_11_dollars_l255_255534


namespace find_dividend_l255_255031

variable (Divisor Quotient Remainder Dividend : ℕ)
variable (h₁ : Divisor = 15)
variable (h₂ : Quotient = 8)
variable (h₃ : Remainder = 5)

theorem find_dividend : Dividend = 125 ↔ Dividend = Divisor * Quotient + Remainder := by
  sorry

end find_dividend_l255_255031


namespace mixed_doubles_teams_l255_255412

theorem mixed_doubles_teams (m n : ℕ) (h_m : m = 7) (h_n : n = 5) :
  (∃ (k : ℕ), k = 4) ∧ (m ≥ 2) ∧ (n ≥ 2) →
  ∃ (number_of_combinations : ℕ), number_of_combinations = 2 * Nat.choose 7 2 * Nat.choose 5 2 :=
by
  intros
  sorry

end mixed_doubles_teams_l255_255412


namespace increasing_function_in_interval_l255_255057

theorem increasing_function_in_interval :
  (∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = x * exp(x) ∧ (∀ y: ℝ, 0 < y → y * exp(y) > 0))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = sin x ∧ (∀ y: ℝ, 0 < y → sin y > 0)))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = x^3 - x ∧ (∀ y: ℝ, 0 < y → y^3 - y > 0)))) ∧
  (¬(∀ x: ℝ, 0 < x → (∃ (f: ℝ → ℝ), f x = log x - x ∧ (∀ y: ℝ, 0 < y → log y - y > 0)))) := by
  sorry

end increasing_function_in_interval_l255_255057


namespace geese_more_than_ducks_l255_255196

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l255_255196


namespace intersecting_lines_triangle_area_l255_255566

theorem intersecting_lines_triangle_area :
  let line1 := { p : ℝ × ℝ | p.2 = p.1 }
  let line2 := { p : ℝ × ℝ | p.1 = -6 }
  let intersection := (-6, -6)
  let base := 6
  let height := 6
  let area := (1 / 2 : ℝ) * base * height
  area = 18 := by
  sorry

end intersecting_lines_triangle_area_l255_255566


namespace percentage_less_than_l255_255865

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l255_255865


namespace integer_values_of_x_for_positive_star_l255_255707

-- Definition of the operation star
def star (a b : ℕ) : ℚ := (a^2 : ℕ) / b

-- Problem statement
theorem integer_values_of_x_for_positive_star :
  ∃ (count : ℕ), count = 9 ∧ (∀ x : ℕ, (10^2 % x = 0) → (∃ n : ℕ, star 10 x = n)) :=
sorry

end integer_values_of_x_for_positive_star_l255_255707


namespace second_solution_lemonade_is_45_l255_255882

-- Define percentages as real numbers for simplicity
def firstCarbonatedWater : ℝ := 0.80
def firstLemonade : ℝ := 0.20
def secondCarbonatedWater : ℝ := 0.55
def mixturePercentageFirst : ℝ := 0.50
def mixtureCarbonatedWater : ℝ := 0.675

-- The ones that already follow from conditions or trivial definitions:
def secondLemonade : ℝ := 1 - secondCarbonatedWater

-- Define the percentage of carbonated water in mixture, based on given conditions
def mixtureIsCorrect : Prop :=
  mixturePercentageFirst * firstCarbonatedWater + (1 - mixturePercentageFirst) * secondCarbonatedWater = mixtureCarbonatedWater

-- The theorem to prove: second solution's lemonade percentage is 45%
theorem second_solution_lemonade_is_45 :
  mixtureIsCorrect → secondLemonade = 0.45 :=
by
  sorry

end second_solution_lemonade_is_45_l255_255882


namespace domain_f_2x_minus_1_l255_255935

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f (2 * x - 1) = y) :=
by
  intro h
  sorry

end domain_f_2x_minus_1_l255_255935


namespace sum_f_sequence_l255_255224

noncomputable def f (x : ℝ) : ℝ := 1 / (4^x + 2)

theorem sum_f_sequence :
  f (1/10) + f (2/10) + f (3/10) + f (4/10) + f (5/10) + f (6/10) + f (7/10) + f (8/10) + f (9/10) = 9 / 4 :=
by {
  sorry
}

end sum_f_sequence_l255_255224


namespace compare_star_values_l255_255897

def star (A B : ℤ) : ℤ := A * B - A / B

theorem compare_star_values : star 6 (-3) < star 4 (-4) := by
  sorry

end compare_star_values_l255_255897


namespace set_intersection_l255_255356

open Set

/-- Given sets M and N as defined below, we wish to prove that their complements and intersections work as expected. -/
theorem set_intersection (R : Set ℝ)
  (M : Set ℝ := {x | x > 1})
  (N : Set ℝ := {x | abs x ≤ 2})
  (R_universal : R = univ) :
  ((compl M) ∩ N) = Icc (-2 : ℝ) (1 : ℝ) := by
  sorry

end set_intersection_l255_255356


namespace sum_powers_divisible_by_13_l255_255430

-- Statement of the problem in Lean
theorem sum_powers_divisible_by_13 (a b p : ℕ) (h1 : a = 3) (h2 : b = 2) (h3 : p = 13) :
  (a^1974 + b^1974) % p = 0 := 
by
  sorry

end sum_powers_divisible_by_13_l255_255430


namespace team_a_took_fewer_hours_l255_255992

/-- Two dogsled teams raced across a 300-mile course. 
Team A finished the course in fewer hours than Team E. 
Team A's average speed was 5 mph greater than Team E's, which was 20 mph. 
How many fewer hours did Team A take to finish the course compared to Team E? --/

theorem team_a_took_fewer_hours :
  let distance := 300
  let speed_e := 20
  let speed_a := speed_e + 5
  let time_e := distance / speed_e
  let time_a := distance / speed_a
  time_e - time_a = 3 := by
  sorry

end team_a_took_fewer_hours_l255_255992


namespace total_students_correct_l255_255432

def num_boys : ℕ := 272
def num_girls : ℕ := num_boys + 106
def total_students : ℕ := num_boys + num_girls

theorem total_students_correct : total_students = 650 :=
by
  sorry

end total_students_correct_l255_255432


namespace option_c_correct_l255_255730

theorem option_c_correct (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x < 2 → x^2 - a ≤ 0) : 4 < a :=
by
  sorry

end option_c_correct_l255_255730


namespace geometric_series_sum_l255_255464

theorem geometric_series_sum :
  let a := -1
  let r := -2
  let n := 11
  ∑ i in finset.range n, a * r^i = 683 :=
by
  let a := -1
  let r := -2
  let n := 11
  have h : ∑ i in finset.range n, a * r^i = a * (r^n - 1) / (r - 1) :=
    by sorry
  calc
    ∑ i in finset.range n, a * r^i 
    = a * (r^n - 1) / (r - 1) : by apply h
    ... = (-1) * ((-2)^11 - 1) / (-3) : by rfl
    ... = 683 : by norm_num

end geometric_series_sum_l255_255464


namespace coordinates_P_l255_255222

theorem coordinates_P 
  (P1 P2 P : ℝ × ℝ)
  (hP1 : P1 = (2, -1))
  (hP2 : P2 = (0, 5))
  (h_ext_line : ∃ t : ℝ, P = (P1.1 + t * (P2.1 - P1.1), P1.2 + t * (P2.2 - P1.2)) ∧ t ≠ 1)
  (h_distance : dist P1 P = 2 * dist P P2) :
  P = (-2, 11) := 
by
  sorry

end coordinates_P_l255_255222


namespace equilateral_triangle_perimeter_l255_255966

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l255_255966


namespace curve1_line_and_circle_curve2_two_points_l255_255703

-- Define the first condition: x(x^2 + y^2 - 4) = 0
def curve1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0

-- Define the second condition: x^2 + (x^2 + y^2 - 4)^2 = 0
def curve2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- The corresponding theorem statements
theorem curve1_line_and_circle : ∀ x y : ℝ, curve1 x y ↔ (x = 0 ∨ (x^2 + y^2 = 4)) := 
sorry 

theorem curve2_two_points : ∀ x y : ℝ, curve2 x y ↔ (x = 0 ∧ (y = 2 ∨ y = -2)) := 
sorry 

end curve1_line_and_circle_curve2_two_points_l255_255703


namespace each_sibling_gets_13_pencils_l255_255962

theorem each_sibling_gets_13_pencils (colored_pencils black_pencils kept_pencils siblings : ℕ) 
  (h1 : colored_pencils = 14)
  (h2 : black_pencils = 35)
  (h3 : kept_pencils = 10)
  (h4 : siblings = 3) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by
  sorry

end each_sibling_gets_13_pencils_l255_255962


namespace age_of_b_l255_255450

theorem age_of_b (a b : ℕ) 
(h1 : a + 10 = 2 * (b - 10)) 
(h2 : a = b + 4) : 
b = 34 := 
sorry

end age_of_b_l255_255450


namespace intersection_points_l255_255015

variables {α β : Type*} [DecidableEq α] {f : α → β} {x m : α}

theorem intersection_points (dom : α → Prop) (h : dom x → ∃! y, f x = y) : 
  (∃ y, f m = y) ∨ ¬ ∃ y, f m = y :=
by
  sorry

end intersection_points_l255_255015


namespace distance_proof_l255_255542

/-- Maxwell's walking speed in km/h. -/
def Maxwell_speed := 4

/-- Time Maxwell walks before meeting Brad in hours. -/
def Maxwell_time := 10

/-- Brad's running speed in km/h. -/
def Brad_speed := 6

/-- Time Brad runs before meeting Maxwell in hours. -/
def Brad_time := 9

/-- Distance between Maxwell and Brad's homes in km. -/
def distance_between_homes : ℕ := 94

/-- Prove the distance between their homes is 94 km given the conditions. -/
theorem distance_proof 
  (h1 : Maxwell_speed * Maxwell_time = 40)
  (h2 : Brad_speed * Brad_time = 54) :
  Maxwell_speed * Maxwell_time + Brad_speed * Brad_time = distance_between_homes := 
by 
  sorry

end distance_proof_l255_255542


namespace find_tangent_points_l255_255712

def f (x : ℝ) : ℝ := x^3 + x - 2
def tangent_parallel_to_line (x : ℝ) : Prop := deriv f x = 4

theorem find_tangent_points :
  (tangent_parallel_to_line 1 ∧ f 1 = 0) ∧ 
  (tangent_parallel_to_line (-1) ∧ f (-1) = -4) :=
by
  sorry

end find_tangent_points_l255_255712


namespace value_of_a_plus_b_l255_255243

variables (a b c d x : ℕ)

theorem value_of_a_plus_b : (b + c = 9) → (c + d = 3) → (a + d = 8) → (a + b = x) → x = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end value_of_a_plus_b_l255_255243


namespace meal_combinations_l255_255995

-- Define the sets of choices for meat, vegetables, dessert, and drinks
def meats := {'beef', 'chicken', 'pork', 'turkey'}
def vegetables := {'baked beans', 'corn', 'potatoes', 'tomatoes', 'carrots'}
def desserts := {'brownies', 'chocolate cake', 'chocolate pudding', 'ice cream', 'cheesecake'}
def drinks := {'water', 'soda', 'juice', 'tea'}

-- Define the cardinalities (sizes) of each set
def meats_card := 4
def vegetables_card := 5
def desserts_card := 5
def drinks_card := 4

-- Number of ways to choose 3 different vegetables out of 5
def choose_vegetables := nat.choose 5 3

-- Final statement to prove
theorem meal_combinations : meats_card * choose_vegetables * desserts_card * drinks_card = 800 := by
  -- The cardinality of each set is given
  have meats_card_eq : meats_card = 4 := by rfl
  have desserts_card_eq : desserts_card = 5 := by rfl
  have drinks_card_eq : drinks_card = 4 := by rfl
  -- Compute the number of ways to choose 3 vegetables out of 5
  have choose_vegetables_eq : choose_vegetables = 10 := by norm_num[choose]
  -- Now multiply all the numbers
  rw [meats_card_eq, choose_vegetables_eq, desserts_card_eq, drinks_card_eq]
  norm_num
  sorry

end meal_combinations_l255_255995


namespace find_number_l255_255309

theorem find_number (x : ℝ) : 50 + (x * 12) / (180 / 3) = 51 ↔ x = 5 := by
  sorry

end find_number_l255_255309


namespace probability_at_least_40_cents_heads_l255_255827

theorem probability_at_least_40_cents_heads :
  let coins := {50, 25, 10, 5, 1}
  3 / 8 =
    (∑ H in {x ∈ (Finset.powerset coins) | (x.sum ≥ 40)}, (1 / 2) ^ x.card) :=
sorry

end probability_at_least_40_cents_heads_l255_255827


namespace compute_expression_l255_255204

theorem compute_expression : 11 * (1 / 17) * 34 = 22 := 
sorry

end compute_expression_l255_255204


namespace similar_rect_tiling_l255_255140

-- Define the dimensions of rectangles A and B
variables {a1 a2 b1 b2 : ℝ}

-- Define the tiling condition
def similar_tiled (a1 a2 b1 b2 : ℝ) : Prop := 
  -- A placeholder for the actual definition of similar tiling
  sorry

-- The main theorem to prove
theorem similar_rect_tiling (h : similar_tiled a1 a2 b1 b2) : similar_tiled b1 b2 a1 a2 :=
sorry

end similar_rect_tiling_l255_255140


namespace ellipse_condition_l255_255556

theorem ellipse_condition (m n : ℝ) :
  (mn > 0) → (¬ (∃ x y : ℝ, (m = 1) ∧ (n = 1) ∧ (x^2)/m + (y^2)/n = 1 ∧ (x, y) ≠ (0,0))) :=
sorry

end ellipse_condition_l255_255556


namespace return_journey_time_l255_255404

-- Define the conditions
def walking_speed : ℕ := 100 -- meters per minute
def walking_time : ℕ := 36 -- minutes
def running_speed : ℕ := 3 -- meters per second

-- Define derived values from conditions
def distance_walked : ℕ := walking_speed * walking_time -- meters
def running_speed_minute : ℕ := running_speed * 60 -- meters per minute

-- Statement of the problem
theorem return_journey_time :
  (distance_walked / running_speed_minute) = 20 := by
  sorry

end return_journey_time_l255_255404


namespace find_marked_price_l255_255840

theorem find_marked_price (cp : ℝ) (d : ℝ) (p : ℝ) (x : ℝ) (h1 : cp = 80) (h2 : d = 0.3) (h3 : p = 0.05) :
  (1 - d) * x = cp * (1 + p) → x = 120 :=
by
  sorry

end find_marked_price_l255_255840


namespace floor_add_double_eq_15_4_l255_255912

theorem floor_add_double_eq_15_4 (r : ℝ) (h : (⌊r⌋ : ℝ) + 2 * r = 15.4) : r = 5.2 := 
sorry

end floor_add_double_eq_15_4_l255_255912


namespace father_l255_255314

theorem father's_age : 
  ∀ (M F : ℕ), 
  (M = (2 : ℚ) / 5 * F) → 
  (M + 10 = (1 : ℚ) / 2 * (F + 10)) → 
  F = 50 :=
by
  intros M F h1 h2
  sorry

end father_l255_255314


namespace initial_speed_is_7_l255_255454

-- Definitions based on conditions
def distance_travelled (S : ℝ) (T : ℝ) : ℝ := S * T

-- Constants from problem
def time_initial : ℝ := 6
def time_final : ℝ := 3
def speed_final : ℝ := 14

-- Theorem statement
theorem initial_speed_is_7 : ∃ S : ℝ, distance_travelled S time_initial = distance_travelled speed_final time_final ∧ S = 7 := by
  sorry

end initial_speed_is_7_l255_255454


namespace number_of_non_symmetric_letters_is_3_l255_255614

def letters_in_JUNIOR : List Char := ['J', 'U', 'N', 'I', 'O', 'R']

def axis_of_symmetry (c : Char) : Bool :=
  match c with
  | 'J' => false
  | 'U' => true
  | 'N' => false
  | 'I' => true
  | 'O' => true
  | 'R' => false
  | _   => false

def letters_with_no_symmetry : List Char :=
  letters_in_JUNIOR.filter (λ c => ¬axis_of_symmetry c)

theorem number_of_non_symmetric_letters_is_3 :
  letters_with_no_symmetry.length = 3 :=
by
  sorry

end number_of_non_symmetric_letters_is_3_l255_255614


namespace quadratic_rewrite_sum_l255_255709

theorem quadratic_rewrite_sum (a b c : ℝ) (x : ℝ) :
  -3 * x^2 + 15 * x + 75 = a * (x + b)^2 + c → (a + b + c) = 88.25 :=
sorry

end quadratic_rewrite_sum_l255_255709


namespace cos_alpha_value_l255_255493

theorem cos_alpha_value (α : ℝ) (hα : 0 < α ∧ α < π / 2) 
  (hcos : Real.cos (α + π / 3) = -2 / 3) : Real.cos α = (Real.sqrt 15 - 2) / 6 := 
  by 
  sorry

end cos_alpha_value_l255_255493


namespace heather_average_balance_l255_255055

theorem heather_average_balance :
  let balance_J := 150
  let balance_F := 250
  let balance_M := 100
  let balance_A := 200
  let balance_May := 300
  let total_balance := balance_J + balance_F + balance_M + balance_A + balance_May
  let avg_balance := total_balance / 5
  avg_balance = 200 :=
by
  sorry

end heather_average_balance_l255_255055


namespace workers_combined_time_l255_255641

theorem workers_combined_time (g_rate a_rate c_rate : ℝ)
  (hg : g_rate = 1 / 70)
  (ha : a_rate = 1 / 30)
  (hc : c_rate = 1 / 42) :
  1 / (g_rate + a_rate + c_rate) = 14 :=
by
  sorry

end workers_combined_time_l255_255641


namespace exists_k_for_any_n_l255_255919

theorem exists_k_for_any_n (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, 2 * k^2 + 2001 * k + 3 ≡ 0 [MOD 2^n] :=
sorry

end exists_k_for_any_n_l255_255919


namespace inequality_abs_l255_255788

noncomputable def f (x : ℝ) : ℝ := abs (x - 1/2) + abs (x + 1/2)

def M : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem inequality_abs (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : |a + b| < |1 + a * b| := 
by
  sorry

end inequality_abs_l255_255788


namespace area_difference_zero_l255_255956

theorem area_difference_zero
  (AG CE : ℝ)
  (s : ℝ)
  (area_square area_rectangle : ℝ)
  (h1 : AG = 2)
  (h2 : CE = 2)
  (h3 : s = 2)
  (h4 : area_square = s^2)
  (h5 : area_rectangle = 2 * 2) :
  (area_square - area_rectangle = 0) :=
by sorry

end area_difference_zero_l255_255956


namespace find_radius_of_circle_l255_255124

variable (AB BC AC R : ℝ)

-- Conditions
def is_right_triangle (ABC : Type) (AB BC : ℝ) (AC : outParam ℝ) : Prop :=
  AC = Real.sqrt (AB^2 + BC^2)

def is_tangent (O : Type) (AB BC AC R : ℝ) : Prop :=
  ∃ (P Q : ℝ), P = R ∧ Q = R ∧ P < AC ∧ Q < AC

theorem find_radius_of_circle (h1 : is_right_triangle ABC 21 28 AC) (h2 : is_tangent O 21 28 AC R) : R = 12 :=
sorry

end find_radius_of_circle_l255_255124


namespace initial_hours_per_day_l255_255949

-- Definitions capturing the conditions
def num_men_initial : ℕ := 100
def num_men_total : ℕ := 160
def portion_completed : ℚ := 1 / 3
def num_days_total : ℕ := 50
def num_days_half : ℕ := 25
def work_performed_portion : ℚ := 2 / 3
def hours_per_day_additional : ℕ := 10

-- Lean statement to prove the initial number of hours per day worked by the initial employees
theorem initial_hours_per_day (H : ℚ) :
  (num_men_initial * H * num_days_total = work_performed_portion) ∧
  (num_men_total * hours_per_day_additional * num_days_half = portion_completed) →
  H = 1.6 := 
sorry

end initial_hours_per_day_l255_255949


namespace rate_of_interest_l255_255281

-- Given conditions
def P : ℝ := 1500
def SI : ℝ := 735
def r : ℝ := 7
def t := r  -- The time period in years is equal to the rate of interest

-- The formula for simple interest and the goal
theorem rate_of_interest : SI = P * r * t / 100 ↔ r = 7 := 
by
  -- We will use the given conditions and check if they support r = 7
  sorry

end rate_of_interest_l255_255281


namespace larger_solution_of_quadratic_equation_l255_255083

open Nat

theorem larger_solution_of_quadratic_equation :
  ∃! x : ℝ, x * x - 13 * x + 36 = 0 ∧ ∀ y : ℝ, y * y - 13 * y + 36 = 0 → x ≥ y :=
by {
  sorry
}

end larger_solution_of_quadratic_equation_l255_255083


namespace sequence_property_l255_255237

noncomputable def seq (n : ℕ) : ℕ := 
if n = 0 then 1 else 
if n = 1 then 3 else 
seq (n-2) + 3 * 2^(n-2)

theorem sequence_property {n : ℕ} (h_pos : n > 0) :
(∀ n : ℕ, n > 0 → seq (n + 2) ≤ seq n + 3 * 2^n) →
(∀ n : ℕ, n > 0 → seq (n + 1) ≥ 2 * seq n + 1) →
seq n = 2^n - 1 := 
sorry

end sequence_property_l255_255237


namespace probability_at_least_40_cents_heads_l255_255828

theorem probability_at_least_40_cents_heads :
  let coins := {50, 25, 10, 5, 1}
  3 / 8 =
    (∑ H in {x ∈ (Finset.powerset coins) | (x.sum ≥ 40)}, (1 / 2) ^ x.card) :=
sorry

end probability_at_least_40_cents_heads_l255_255828


namespace runners_meet_opposite_dir_l255_255297

theorem runners_meet_opposite_dir 
  {S x y : ℝ}
  (h1 : S / x + 5 = S / y)
  (h2 : S / (x - y) = 30) :
  S / (x + y) = 6 := 
sorry

end runners_meet_opposite_dir_l255_255297


namespace gcd_f_100_f_101_l255_255537

def f (x : ℕ) : ℕ := x^2 - 2*x + 2023

theorem gcd_f_100_f_101 : Nat.gcd (f 100) (f 101) = 1 := by
  sorry

end gcd_f_100_f_101_l255_255537


namespace nathan_has_83_bananas_l255_255694

def nathan_bananas (bunches_eight bananas_eight bunches_seven bananas_seven: Nat) : Nat :=
  bunches_eight * bananas_eight + bunches_seven * bananas_seven

theorem nathan_has_83_bananas (h1 : bunches_eight = 6) (h2 : bananas_eight = 8) (h3 : bunches_seven = 5) (h4 : bananas_seven = 7) : 
  nathan_bananas bunches_eight bananas_eight bunches_seven bananas_seven = 83 := by
  sorry

end nathan_has_83_bananas_l255_255694


namespace total_spending_march_to_july_l255_255705

-- Define the conditions
def beginning_of_march_spending : ℝ := 1.2
def end_of_july_spending : ℝ := 4.8

-- State the theorem to prove
theorem total_spending_march_to_july : 
  end_of_july_spending - beginning_of_march_spending = 3.6 :=
sorry

end total_spending_march_to_july_l255_255705


namespace arithmetic_sequence_line_l255_255661

theorem arithmetic_sequence_line (A B C x y : ℝ) :
  (2 * B = A + C) → (A * 1 + B * -2 + C = 0) :=
by
  intros h
  sorry

end arithmetic_sequence_line_l255_255661


namespace inclination_angle_of_line_l255_255832

def line_equation (x y : ℝ) : Prop := x * (Real.tan (Real.pi / 3)) + y + 2 = 0

theorem inclination_angle_of_line (x y : ℝ) (h : line_equation x y) : 
  ∃ α : ℝ, α = 2 * Real.pi / 3 ∧ 0 ≤ α ∧ α < Real.pi := by
  sorry

end inclination_angle_of_line_l255_255832


namespace students_count_l255_255313

theorem students_count :
  ∃ S : ℕ, (S + 4) % 9 = 0 ∧ S = 23 :=
by
  sorry

end students_count_l255_255313


namespace complement_union_l255_255943

open Finset

def U : Finset ℕ := {0, 1, 2, 3, 4}
def M : Finset ℕ := {0, 4}
def N : Finset ℕ := {2, 4}

theorem complement_union :
  U \ (M ∪ N) = {1, 3} := by
  sorry

end complement_union_l255_255943


namespace minimum_value_l255_255272

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  (1 / x + 9 / y) = 16 :=
sorry

end minimum_value_l255_255272


namespace sheep_ratio_l255_255001

theorem sheep_ratio (S : ℕ) (h1 : 400 - S = 2 * 150) :
  S / 400 = 1 / 4 :=
by
  sorry

end sheep_ratio_l255_255001


namespace street_lights_per_side_l255_255606

theorem street_lights_per_side
  (neighborhoods : ℕ)
  (roads_per_neighborhood : ℕ)
  (total_street_lights : ℕ)
  (total_neighborhoods : neighborhoods = 10)
  (roads_in_each_neighborhood : roads_per_neighborhood = 4)
  (street_lights_in_town : total_street_lights = 20000) :
  (total_street_lights / (neighborhoods * roads_per_neighborhood * 2) = 250) :=
by
  sorry

end street_lights_per_side_l255_255606


namespace evaluate_expression_l255_255431

theorem evaluate_expression : (2014 - 2013) * (2013 - 2012) = 1 := 
by sorry

end evaluate_expression_l255_255431


namespace housing_price_equation_l255_255463

-- Initial conditions
def january_price : ℝ := 8300
def march_price : ℝ := 8700
variables (x : ℝ)

-- Lean statement of the problem
theorem housing_price_equation :
  january_price * (1 + x)^2 = march_price := 
sorry

end housing_price_equation_l255_255463


namespace geese_more_than_ducks_l255_255197

-- Define initial conditions
def initial_ducks : ℕ := 25
def initial_geese : ℕ := 2 * initial_ducks - 10
def additional_ducks : ℕ := 4
def geese_leaving : ℕ := 15 - 5

-- Calculate the number of remaining ducks
def remaining_ducks : ℕ := initial_ducks + additional_ducks

-- Calculate the number of remaining geese
def remaining_geese : ℕ := initial_geese - geese_leaving

-- Prove the final statement
theorem geese_more_than_ducks : (remaining_geese - remaining_ducks) = 1 := by
  sorry

end geese_more_than_ducks_l255_255197


namespace avg_length_remaining_wires_l255_255175

theorem avg_length_remaining_wires (N : ℕ) (avg_length : ℕ) 
    (third_wires_count : ℕ) (third_wires_avg_length : ℕ) 
    (total_length : ℕ := N * avg_length) 
    (third_wires_total_length : ℕ := third_wires_count * third_wires_avg_length) 
    (remaining_wires_count : ℕ := N - third_wires_count) 
    (remaining_wires_total_length : ℕ := total_length - third_wires_total_length) :
    N = 6 → 
    avg_length = 80 → 
    third_wires_count = 2 → 
    third_wires_avg_length = 70 → 
    remaining_wires_count = 4 → 
    remaining_wires_total_length / remaining_wires_count = 85 :=
by 
  intros hN hAvg hThirdCount hThirdAvg hRemainingCount
  sorry

end avg_length_remaining_wires_l255_255175


namespace total_wasted_time_is_10_l255_255397

-- Define the time Martin spends waiting in traffic
def waiting_time : ℕ := 2

-- Define the constant for the multiplier
def multiplier : ℕ := 4

-- Define the time spent trying to get off the freeway
def off_freeway_time : ℕ := waiting_time * multiplier

-- Define the total wasted time
def total_wasted_time : ℕ := waiting_time + off_freeway_time

-- Theorem stating that the total time wasted is 10 hours
theorem total_wasted_time_is_10 : total_wasted_time = 10 :=
by
  sorry

end total_wasted_time_is_10_l255_255397


namespace james_beats_old_record_l255_255126

def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def two_point_conversions : ℕ := 6
def points_per_two_point_conversion : ℕ := 2
def field_goals : ℕ := 8
def points_per_field_goal : ℕ := 3
def extra_points : ℕ := 20
def points_per_extra_point : ℕ := 1
def old_record : ℕ := 300

theorem james_beats_old_record :
  touchdowns_per_game * points_per_touchdown * games_in_season +
  two_point_conversions * points_per_two_point_conversion +
  field_goals * points_per_field_goal +
  extra_points * points_per_extra_point - old_record = 116 := by
  sorry -- Proof is omitted.

end james_beats_old_record_l255_255126


namespace axis_of_symmetry_values_ge_one_range_m_l255_255921

open Real

-- Definitions for vectors and the function f(x)
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Part I: Prove the equation of the axis of symmetry of f(x)
theorem axis_of_symmetry {k : ℤ} : f x = (sqrt 2 / 2) * sin (2 * x - π / 4) + 1 / 2 → 
                                    x = k * π / 2 + 3 * π / 8 := 
sorry

-- Part II: Prove the set of values x for which f(x) ≥ 1
theorem values_ge_one : (f x ≥ 1) ↔ (∃ (k : ℤ), π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) := 
sorry

-- Part III: Prove the range of m given the inequality
theorem range_m (m : ℝ) : (∀ x, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
                            m > (sqrt 3 - 5) / 4 := 
sorry

end axis_of_symmetry_values_ge_one_range_m_l255_255921


namespace product_of_roots_l255_255944

theorem product_of_roots (a b c : ℂ) (h_roots : 3 * (Polynomial.C a) * (Polynomial.C b) * (Polynomial.C c) = -7) :
  a * b * c = -7 / 3 :=
by sorry

end product_of_roots_l255_255944


namespace ducks_and_geese_difference_l255_255198

variable (d g d' l : ℕ)
variables (hd : d = 25)
variables (hg : g = 2 * d - 10)
variables (hd' : d' = d + 4)
variables (hl : l = 15 - 5)

theorem ducks_and_geese_difference :
  let geese_remain := g - l
  let ducks_remain := d'
  geese_remain - ducks_remain = 1 :=
by
  sorry

end ducks_and_geese_difference_l255_255198


namespace cosine_between_vectors_l255_255102

noncomputable def vector_cos_angle (a b : ℝ × ℝ) := 
  let dot_product := (a.1 * b.1) + (a.2 * b.2)
  let norm_a := Real.sqrt (a.1 * a.1 + a.2 * a.2)
  let norm_b := Real.sqrt (b.1 * b.1 + b.2 * b.2)
  dot_product / (norm_a * norm_b)

theorem cosine_between_vectors (t : ℝ) 
  (ht : let a := (1, t); let b := (-1, 2 * t);
        (3 * a.1 - b.1) * b.1 + (3 * a.2 - b.2) * b.2 = 0) :
  vector_cos_angle (1, t) (-1, 2 * t) = Real.sqrt 3 / 3 := 
by
  sorry

end cosine_between_vectors_l255_255102


namespace dividend_calculation_l255_255370

theorem dividend_calculation (divisor quotient remainder dividend : ℕ)
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5)
  (h4 : dividend = (divisor * quotient) + remainder)
  : dividend = 725 := 
by
  -- We skip the proof here
  sorry

end dividend_calculation_l255_255370


namespace incorrect_regression_intercept_l255_255829

theorem incorrect_regression_intercept (points : List (ℕ × ℝ)) (h_points : points = [(1, 0.5), (2, 0.8), (3, 1.0), (4, 1.2), (5, 1.5)]) :
  ¬ (∃ (a : ℝ), a = 0.26 ∧ ∀ x : ℕ, x ∈ ([1, 2, 3, 4, 5] : List ℕ) → (∃ y : ℝ, y = 0.24 * x + a)) := sorry

end incorrect_regression_intercept_l255_255829


namespace multiplier_of_reciprocal_l255_255598

theorem multiplier_of_reciprocal (x m : ℝ) (h1 : x = 7) (h2 : x - 4 = m * (1 / x)) : m = 21 :=
by
  sorry

end multiplier_of_reciprocal_l255_255598


namespace coeff_x5_in_expansion_l255_255379

theorem coeff_x5_in_expansion : 
  let p := (1 - X^3) * (1 + X)^10 in
  coeff p 5 = 207 := 
by 
  sorry

end coeff_x5_in_expansion_l255_255379


namespace problem_l255_255816

def g (x : ℕ) : ℕ := x^2 + 1
def f (x : ℕ) : ℕ := 3 * x - 2

theorem problem : f (g 3) = 28 := by
  sorry

end problem_l255_255816


namespace find_multiple_l255_255844

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l255_255844


namespace fabric_per_pair_of_pants_l255_255131

theorem fabric_per_pair_of_pants 
  (jenson_shirts_per_day : ℕ)
  (kingsley_pants_per_day : ℕ)
  (fabric_per_shirt : ℕ)
  (total_fabric_needed : ℕ)
  (days : ℕ)
  (fabric_per_pant : ℕ) :
  jenson_shirts_per_day = 3 →
  kingsley_pants_per_day = 5 →
  fabric_per_shirt = 2 →
  total_fabric_needed = 93 →
  days = 3 →
  fabric_per_pant = 5 :=
by sorry

end fabric_per_pair_of_pants_l255_255131


namespace sqrt_sum_bound_l255_255489

theorem sqrt_sum_bound (x : ℝ) (hx1 : 3 / 2 ≤ x) (hx2 : x ≤ 5) :
  2 * Real.sqrt(x + 1) + Real.sqrt(2 * x - 3) + Real.sqrt(15 - 3 * x) < 2 * Real.sqrt(19) :=
by
  sorry

end sqrt_sum_bound_l255_255489


namespace parts_of_milk_in_drink_A_l255_255588

theorem parts_of_milk_in_drink_A (x : ℝ) (h : 63 * (4 * x) / (7 * (x + 3)) = 63 * 3 / (x + 3) + 21) : x = 16.8 :=
by
  sorry

end parts_of_milk_in_drink_A_l255_255588


namespace correct_product_l255_255115

def reverse_digits (n: ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d2 * 10 + d1

theorem correct_product (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : b > 0) (h3 : reverse_digits a * b = 221) :
  a * b = 527 ∨ a * b = 923 :=
sorry

end correct_product_l255_255115


namespace fisherman_daily_earnings_l255_255423

theorem fisherman_daily_earnings :
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52 :=
by
  let red_snapper_count := 8
  let tuna_count := 14
  let red_snapper_price := 3
  let tuna_price := 2
  show red_snapper_count * red_snapper_price + tuna_count * tuna_price = 52
  sorry

end fisherman_daily_earnings_l255_255423


namespace gear_q_revolutions_per_minute_l255_255620

noncomputable def gear_p_revolutions_per_minute : ℕ := 10

noncomputable def additional_revolutions : ℕ := 15

noncomputable def calculate_q_revolutions_per_minute
  (p_rev_per_min : ℕ) (additional_rev : ℕ) : ℕ :=
  2 * (p_rev_per_min / 2 + additional_rev)

theorem gear_q_revolutions_per_minute :
  calculate_q_revolutions_per_minute gear_p_revolutions_per_minute additional_revolutions = 40 :=
by
  sorry

end gear_q_revolutions_per_minute_l255_255620


namespace slices_needed_l255_255282

def slices_per_sandwich : ℕ := 3
def number_of_sandwiches : ℕ := 5

theorem slices_needed : slices_per_sandwich * number_of_sandwiches = 15 :=
by {
  sorry
}

end slices_needed_l255_255282


namespace solve_equation_l255_255284

theorem solve_equation (x : ℝ) : (x + 3) * (x - 1) = 12 ↔ (x = -5 ∨ x = 3) := sorry

end solve_equation_l255_255284


namespace polynomial_roots_ratio_l255_255441

theorem polynomial_roots_ratio (a b c d : ℝ) (h₀ : a ≠ 0) 
    (h₁ : a * 64 + b * 16 + c * 4 + d = 0)
    (h₂ : -a + b - c + d = 0) : 
    (b + c) / a = -13 :=
by {
    sorry
}

end polynomial_roots_ratio_l255_255441


namespace total_students_in_class_l255_255016

theorem total_students_in_class (B G : ℕ) (h1 : G = 160) (h2 : 5 * G = 8 * B) : B + G = 260 :=
by
  -- Proof steps would go here
  sorry

end total_students_in_class_l255_255016


namespace total_vessels_l255_255179

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l255_255179


namespace option_b_is_same_type_l255_255575

def polynomial_same_type (p1 p2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, (p1 x y = 1 → p2 x y = 1) ∧ (p2 x y = 1 → p1 x y = 1)

def ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0

def a_squared_b (a b : ℕ) := if a = 2 ∧ b = 1 then 1 else 0
def negative_two_ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0
def ab (a b : ℕ) := if a = 1 ∧ b = 1 then 1 else 0
def ab_squared_c (a b c : ℕ) := if a = 1 ∧ b = 2 ∧ c = 1 then 1 else 0

theorem option_b_is_same_type : polynomial_same_type ab_squared negative_two_ab_squared :=
by
  sorry

end option_b_is_same_type_l255_255575


namespace exists_four_consecutive_with_square_divisors_l255_255221

theorem exists_four_consecutive_with_square_divisors :
  ∃ n : ℕ, n = 3624 ∧
  (∃ d1, d1^2 > 1 ∧ d1^2 ∣ n) ∧ 
  (∃ d2, d2^2 > 1 ∧ d2^2 ∣ (n + 1)) ∧ 
  (∃ d3, d3^2 > 1 ∧ d3^2 ∣ (n + 2)) ∧ 
  (∃ d4, d4^2 > 1 ∧ d4^2 ∣ (n + 3)) :=
sorry

end exists_four_consecutive_with_square_divisors_l255_255221


namespace base_ten_to_base_three_l255_255855

def base_three_representation (n : ℕ) : string :=
  -- Function to convert number to base 3 string
  sorry -- This would be an actual implementation of conversion to base 3

theorem base_ten_to_base_three :
  base_three_representation 172 = "20101" :=
by
  sorry -- Proof of the equivalence

end base_ten_to_base_three_l255_255855


namespace triangle_inequality_l255_255648

theorem triangle_inequality (a b c Δ : ℝ) (h_Δ: Δ = Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))) :
    a^2 + b^2 + c^2 ≥ 4 * Real.sqrt (3) * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 :=
by
  sorry

end triangle_inequality_l255_255648


namespace slower_train_pass_time_l255_255442

noncomputable def relative_speed_km_per_hr (v1 v2 : ℕ) : ℕ :=
v1 + v2

noncomputable def relative_speed_m_per_s (v_km_per_hr : ℕ) : ℝ :=
(v_km_per_hr * 5) / 18

noncomputable def time_to_pass (distance_m : ℕ) (speed_m_per_s : ℝ) : ℝ :=
distance_m / speed_m_per_s

theorem slower_train_pass_time
  (length_train1 length_train2 : ℕ)
  (speed_train1_km_per_hr speed_train2_km_per_hr : ℕ)
  (distance_to_cover : ℕ)
  (h1 : length_train1 = 800)
  (h2 : length_train2 = 600)
  (h3 : speed_train1_km_per_hr = 85)
  (h4 : speed_train2_km_per_hr = 65)
  (h5 : distance_to_cover = length_train2) :
  time_to_pass distance_to_cover (relative_speed_m_per_s (relative_speed_km_per_hr speed_train1_km_per_hr speed_train2_km_per_hr)) = 14.4 := 
sorry

end slower_train_pass_time_l255_255442


namespace average_weight_of_boys_l255_255119

theorem average_weight_of_boys
  (average_weight_girls : ℕ) 
  (average_weight_students : ℕ) 
  (h_girls : average_weight_girls = 45)
  (h_students : average_weight_students = 50) : 
  ∃ average_weight_boys : ℕ, average_weight_boys = 55 :=
by
  sorry

end average_weight_of_boys_l255_255119


namespace parakeets_per_cage_l255_255879

-- Define total number of cages
def num_cages: Nat := 6

-- Define number of parrots per cage
def parrots_per_cage: Nat := 2

-- Define total number of birds in the store
def total_birds: Nat := 54

-- Theorem statement: prove the number of parakeets per cage
theorem parakeets_per_cage : (total_birds - num_cages * parrots_per_cage) / num_cages = 7 :=
by
  sorry

end parakeets_per_cage_l255_255879


namespace length_imaginary_axis_hyperbola_l255_255150

theorem length_imaginary_axis_hyperbola : 
  ∀ (a b : ℝ), (a = 2) → (b = 1) → 
  (∀ x y : ℝ, (y^2 / a^2 - x^2 = 1) → 2 * b = 2) :=
by intros a b ha hb x y h; sorry

end length_imaginary_axis_hyperbola_l255_255150


namespace exponential_inequality_l255_255270

variables (x a b : ℝ)

theorem exponential_inequality (h1 : x > 0) (h2 : 1 < b^x) (h3 : b^x < a^x) : 1 < b ∧ b < a :=
by
   sorry

end exponential_inequality_l255_255270


namespace square_area_l255_255153

theorem square_area (x : ℝ) (s1 s2 area : ℝ) 
  (h1 : s1 = 5 * x - 21) 
  (h2 : s2 = 36 - 4 * x) 
  (hs : s1 = s2)
  (ha : area = s1 * s1) : 
  area = 113.4225 := 
by
  -- Proof goes here
  sorry

end square_area_l255_255153


namespace min_sum_of_factors_l255_255708

theorem min_sum_of_factors (a b c : ℕ) (h1 : a * b * c = 1806) (h2 : a % 2 = 0 ∨ b % 2 = 0 ∨ c % 2 = 0) : a + b + c ≥ 112 :=
sorry

end min_sum_of_factors_l255_255708


namespace swimming_pool_length_l255_255047

noncomputable def solveSwimmingPoolLength : ℕ :=
  let w_pool := 22
  let w_deck := 3
  let total_area := 728
  let total_width := w_pool + 2 * w_deck
  let L := (total_area / total_width) - 2 * w_deck
  L

theorem swimming_pool_length : solveSwimmingPoolLength = 20 := 
  by
  -- Proof goes here
  sorry

end swimming_pool_length_l255_255047


namespace find_lightest_bead_l255_255569

theorem find_lightest_bead (n : ℕ) (h : 0 < n) (H : ∀ b1 b2 b3 : ℕ, b1 + b2 + b3 = n → b1 > 0 ∧ b2 > 0 ∧ b3 > 0 → b1 ≤ 3 ∧ b2 ≤ 9 ∧ b3 ≤ 27) : n = 27 :=
sorry

end find_lightest_bead_l255_255569


namespace sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l255_255470

def legendre (n p : Nat) : Nat :=
  if p > 1 then (Nat.div n p + Nat.div n (p * p) + Nat.div n (p * p * p) + Nat.div n (p * p * p * p)) else 0

theorem sum_of_highest_powers_of_10_and_6_dividing_20_factorial :
  let highest_power_5 := legendre 20 5
  let highest_power_2 := legendre 20 2
  let highest_power_3 := legendre 20 3
  let highest_power_10 := min highest_power_2 highest_power_5
  let highest_power_6 := min highest_power_2 highest_power_3
  highest_power_10 + highest_power_6 = 12 :=
by
  sorry

end sum_of_highest_powers_of_10_and_6_dividing_20_factorial_l255_255470


namespace probability_greater_than_two_l255_255253

noncomputable def probability_of_greater_than_two : ℚ :=
  let total_outcomes := 6
  let favorable_outcomes := 4
  favorable_outcomes / total_outcomes

theorem probability_greater_than_two (total_outcomes favorable_outcomes : ℕ)
  (ht : total_outcomes = 6) (hf : favorable_outcomes = 4) :
  probability_of_greater_than_two = 2 / 3 :=
by
  rw [probability_of_greater_than_two, ht, hf]
  norm_num
  sorry

end probability_greater_than_two_l255_255253


namespace chef_bought_kilograms_of_almonds_l255_255741

def total_weight_of_nuts : ℝ := 0.52
def weight_of_pecans : ℝ := 0.38
def weight_of_almonds : ℝ := total_weight_of_nuts - weight_of_pecans

theorem chef_bought_kilograms_of_almonds : weight_of_almonds = 0.14 := by
  sorry

end chef_bought_kilograms_of_almonds_l255_255741


namespace martin_initial_spending_l255_255691

theorem martin_initial_spending :
  ∃ (x : ℝ), 
    ∀ (a b : ℝ), 
      a = x - 100 →
      b = a - 0.20 * a →
      x - b = 280 →
      x = 1000 :=
by
  sorry

end martin_initial_spending_l255_255691


namespace calculate_neg4_mul_three_div_two_l255_255203

theorem calculate_neg4_mul_three_div_two : (-4) * (3 / 2) = -6 := 
by
  sorry

end calculate_neg4_mul_three_div_two_l255_255203


namespace sum_of_three_integers_eq_57_l255_255292

theorem sum_of_three_integers_eq_57
  (a b c : ℕ) (h1: a * b * c = 7^3) (h2: a ≠ b) (h3: b ≠ c) (h4: a ≠ c) :
  a + b + c = 57 :=
sorry

end sum_of_three_integers_eq_57_l255_255292


namespace product_of_b_product_of_values_l255_255998

/-- 
If the distance between the points (3b, b+2) and (6, 3) is 3√5 units,
then the product of all possible values of b is -0.8.
-/
theorem product_of_b (b : ℝ)
  (h : (6 - 3 * b)^2 + (3 - (b + 2))^2 = (3 * Real.sqrt 5)^2) :
  b = 4 ∨ b = -0.2 := sorry

/--
The product of the values satisfying the theorem product_of_b is -0.8.
-/
theorem product_of_values : (4 : ℝ) * (-0.2) = -0.8 := 
by norm_num -- using built-in arithmetic simplification

end product_of_b_product_of_values_l255_255998


namespace total_money_l255_255863

-- Define the problem with conditions and question transformed into proof statement
theorem total_money (A B : ℕ) (h1 : 2 * A / 3 = B / 2) (h2 : B = 484) : A + B = 847 :=
by
  sorry -- Proof to be filled in

end total_money_l255_255863


namespace sum_of_exponents_of_1985_eq_40_l255_255073

theorem sum_of_exponents_of_1985_eq_40 :
  ∃ (e₀ e₁ e₂ e₃ e₄ e₅ : ℕ), 1985 = 2^e₀ + 2^e₁ + 2^e₂ + 2^e₃ + 2^e₄ + 2^e₅ 
  ∧ e₀ ≠ e₁ ∧ e₀ ≠ e₂ ∧ e₀ ≠ e₃ ∧ e₀ ≠ e₄ ∧ e₀ ≠ e₅
  ∧ e₁ ≠ e₂ ∧ e₁ ≠ e₃ ∧ e₁ ≠ e₄ ∧ e₁ ≠ e₅
  ∧ e₂ ≠ e₃ ∧ e₂ ≠ e₄ ∧ e₂ ≠ e₅
  ∧ e₃ ≠ e₄ ∧ e₃ ≠ e₅
  ∧ e₄ ≠ e₅
  ∧ e₀ + e₁ + e₂ + e₃ + e₄ + e₅ = 40 := 
by
  sorry

end sum_of_exponents_of_1985_eq_40_l255_255073


namespace totalMoney_l255_255293

noncomputable def totalAmount (x : ℝ) : ℝ := 15 * x

theorem totalMoney (x : ℝ) (h : 1.8 * x = 9) : totalAmount x = 75 :=
by sorry

end totalMoney_l255_255293


namespace larger_of_two_solutions_l255_255076

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l255_255076


namespace six_digit_number_representation_l255_255173

-- Defining that a is a two-digit number
def isTwoDigitNumber (a : ℕ) : Prop := a >= 10 ∧ a < 100

-- Defining that b is a four-digit number
def isFourDigitNumber (b : ℕ) : Prop := b >= 1000 ∧ b < 10000

-- The statement that placing a to the left of b forms the number 10000*a + b
theorem six_digit_number_representation (a b : ℕ) 
  (ha : isTwoDigitNumber a) 
  (hb : isFourDigitNumber b) : 
  (10000 * a + b) = (10^4 * a + b) :=
by
  sorry

end six_digit_number_representation_l255_255173


namespace general_formula_sum_first_20_b_l255_255225

noncomputable theory

-- Defining the geometric sequence a_n
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * 2)

-- Defining the arithmetic condition
def arithmetic_condition (a : ℕ → ℝ) : Prop :=
  let a1 := a 1 in
  let a2 := a 2 in
  let a3 := a 3 in
  2 * a2 = a1 + (a3 - 1)

-- Definition of b_n sequence
def b_sequence (a b : ℕ → ℝ) : Prop :=
  (∀ n, if n % 2 = 1 then b n = a n - 1 else b n = a n / 2)

-- The main theorem to prove the general formula for a_n
theorem general_formula (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) :
  ∀ n, a n = 2 ^ (n - 1) :=
sorry

-- The main theorem to calculate the sum of the first 20 terms of b_n
theorem sum_first_20_b (a b : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 = 2) (h3 : arithmetic_condition a) (h4 : b_sequence a b) :
  (∑ i in finset.range 20, b (i + 1)) = (2 ^ 21 - 32) / 3 :=
sorry

end general_formula_sum_first_20_b_l255_255225


namespace total_athletes_l255_255327

theorem total_athletes (g : ℕ) (p : ℕ)
  (h₁ : g = 7)
  (h₂ : p = 5)
  (h₃ : 3 * (g + p - 1) = 33) : 
  3 * (g + p - 1) = 33 :=
sorry

end total_athletes_l255_255327


namespace percentage_less_l255_255867

theorem percentage_less (P T J : ℝ) (hT : T = 0.9375 * P) (hJ : J = 0.8 * T) : (P - J) / P * 100 = 25 :=
by
  sorry

end percentage_less_l255_255867


namespace simplify_expression_l255_255698

variable (x : ℝ)

theorem simplify_expression :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) = 2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end simplify_expression_l255_255698


namespace repetend_of_five_over_eleven_l255_255636

noncomputable def repetend_of_decimal_expansion (n d : ℕ) : ℕ := sorry

theorem repetend_of_five_over_eleven : repetend_of_decimal_expansion 5 11 = 45 :=
by sorry

end repetend_of_five_over_eleven_l255_255636


namespace original_denominator_is_15_l255_255319

theorem original_denominator_is_15
  (d : ℕ)
  (h1 : (6 : ℚ) / (d + 3) = 1 / 3)
  (h2 : 3 = 3) : d = 15 := -- h2 is trivial but included according to the problem condition
by
  sorry

end original_denominator_is_15_l255_255319


namespace principal_amount_unique_l255_255581

theorem principal_amount_unique (SI R T : ℝ) (P : ℝ) : 
  SI = 4016.25 → R = 14 → T = 5 → SI = (P * R * T) / 100 → P = 5737.5 :=
by
  intro h₁ h₂ h₃ h₄
  rw [h₁, h₂, h₃] at h₄
  sorry

end principal_amount_unique_l255_255581


namespace fraction_to_terminating_decimal_l255_255483

-- Lean statement for the mathematical problem
theorem fraction_to_terminating_decimal: (13 : ℚ) / 200 = 0.26 := 
sorry

end fraction_to_terminating_decimal_l255_255483


namespace weights_less_than_90_l255_255596

variable (a b c : ℝ)
-- conditions
axiom h1 : a + b = 100
axiom h2 : a + c = 101
axiom h3 : b + c = 102

theorem weights_less_than_90 (a b c : ℝ) (h1 : a + b = 100) (h2 : a + c = 101) (h3 : b + c = 102) : a < 90 ∧ b < 90 ∧ c < 90 := 
by sorry

end weights_less_than_90_l255_255596


namespace incorrect_inequality_l255_255345

theorem incorrect_inequality (m n : ℝ) (a : ℝ) (hmn : m > n) (hm1 : m > 1) (hn1 : n > 1) (ha0 : 0 < a) (ha1 : a < 1) : 
  ¬ (a^m > a^n) :=
sorry

end incorrect_inequality_l255_255345


namespace andy_starting_problem_l255_255058

theorem andy_starting_problem (end_num problems_solved : ℕ) 
  (h_end : end_num = 125) (h_solved : problems_solved = 46) : 
  end_num - problems_solved + 1 = 80 := 
by
  sorry

end andy_starting_problem_l255_255058


namespace ratio_of_second_to_third_l255_255159

theorem ratio_of_second_to_third (A B C : ℕ) (h1 : A + B + C = 98) (h2 : A * 3 = B * 2) (h3 : B = 30) :
  B * 8 = C * 5 :=
by
  sorry

end ratio_of_second_to_third_l255_255159


namespace calculate_power_l255_255892

variable (x y : ℝ)

theorem calculate_power :
  (- (1 / 2) * x^2 * y)^3 = - (1 / 8) * x^6 * y^3 :=
sorry

end calculate_power_l255_255892


namespace martin_crayons_l255_255690

theorem martin_crayons : (8 * 7 = 56) := by
  sorry

end martin_crayons_l255_255690


namespace yadav_spends_50_percent_on_clothes_and_transport_l255_255275

variable (S : ℝ)
variable (monthly_savings : ℝ := 46800 / 12)
variable (clothes_transport_expense : ℝ := 3900)
variable (remaining_salary : ℝ := 0.40 * S)

theorem yadav_spends_50_percent_on_clothes_and_transport (h1 : remaining_salary = 2 * 3900) :
  (clothes_transport_expense / remaining_salary) * 100 = 50 :=
by
  -- skipping the proof steps
  sorry

end yadav_spends_50_percent_on_clothes_and_transport_l255_255275


namespace fraction_spent_first_week_l255_255862

theorem fraction_spent_first_week
  (S : ℝ) (F : ℝ)
  (h1 : S > 0)
  (h2 : F * S + 3 * (0.20 * S) + 0.15 * S = S) : 
  F = 0.25 := 
sorry

end fraction_spent_first_week_l255_255862


namespace emma_withdrew_amount_l255_255068

variable (W : ℝ) -- Variable representing the amount Emma withdrew

theorem emma_withdrew_amount:
  (230 - W + 2 * W = 290) →
  W = 60 :=
by
  sorry

end emma_withdrew_amount_l255_255068


namespace pencils_per_sibling_l255_255960

theorem pencils_per_sibling :
  let total_pencils := 49
  let kept_pencils := 10
  let siblings := 3
  let remaining_pencils := total_pencils - kept_pencils
  (remaining_pencils / siblings) = 13 :=
by
  -- Definitions for the variables
  let total_pencils := 49
  let kept_pencils := 10
  let siblings := 3
  let remaining_pencils := total_pencils - kept_pencils

  -- Simplification to show the desired result
  have h1 : remaining_pencils = 39 := by
    calc
      remaining_pencils = total_pencils - kept_pencils : rfl
      ... = 49 - 10 : rfl
      ... = 39 : rfl

  have h2 : (remaining_pencils / siblings) = 13 := by
    calc
      (remaining_pencils / siblings) = 39 / 3 : by rw [h1]
      ... = 13 : by norm_num

  exact h2

end pencils_per_sibling_l255_255960


namespace sum_max_min_ratio_l255_255195

def ellipse_eq (x y : ℝ) : Prop :=
  5 * x^2 + x * y + 4 * y^2 - 15 * x - 24 * y + 56 = 0

theorem sum_max_min_ratio (p q : ℝ) (x y : ℝ) :
  (∀ x y : ℝ, ellipse_eq x y → y / x = p ∨ y / x = q) → 
  p + q = 31 / 34 :=
by
  sorry

end sum_max_min_ratio_l255_255195


namespace number_of_pages_500_l255_255664

-- Define the conditions as separate constants
def cost_per_page : ℕ := 3 -- cents
def total_cents : ℕ := 1500 

-- Define the number of pages calculation
noncomputable def number_of_pages := total_cents / cost_per_page

-- Statement we want to prove
theorem number_of_pages_500 : number_of_pages = 500 :=
sorry

end number_of_pages_500_l255_255664


namespace gain_percentage_l255_255733

theorem gain_percentage (CP SP : ℕ) (h_sell : SP = 10 * CP) : 
  (10 * CP / 25 * CP) * 100 = 40 := by
  sorry

end gain_percentage_l255_255733


namespace quadratic_range_m_l255_255520

theorem quadratic_range_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
by 
  sorry

end quadratic_range_m_l255_255520


namespace not_sufficient_nor_necessary_l255_255815

theorem not_sufficient_nor_necessary (a b : ℝ) : ¬((a > b) → (a^2 > b^2)) ∧ ¬((a^2 > b^2) → (a > b)) := 
by 
  sorry

end not_sufficient_nor_necessary_l255_255815


namespace fraction_values_l255_255662

theorem fraction_values (a b c : ℚ) (h1 : a / b = 2) (h2 : b / c = 4 / 3) : c / a = 3 / 8 := 
by
  sorry

end fraction_values_l255_255662


namespace int_999_column_is_C_l255_255754

def column_of_int (n : ℕ) : String :=
  let m := n - 2
  match (m / 7 % 2, m % 7) with
  | (0, 0) => "A"
  | (0, 1) => "B"
  | (0, 2) => "C"
  | (0, 3) => "D"
  | (0, 4) => "E"
  | (0, 5) => "F"
  | (0, 6) => "G"
  | (1, 0) => "G"
  | (1, 1) => "F"
  | (1, 2) => "E"
  | (1, 3) => "D"
  | (1, 4) => "C"
  | (1, 5) => "B"
  | (1, 6) => "A"
  | _      => "Invalid"

theorem int_999_column_is_C : column_of_int 999 = "C" := by
  sorry

end int_999_column_is_C_l255_255754


namespace charlie_coins_l255_255244

variables (a c : ℕ)

axiom condition1 : c + 2 = 5 * (a - 2)
axiom condition2 : c - 2 = 4 * (a + 2)

theorem charlie_coins : c = 98 :=
by {
    sorry
}

end charlie_coins_l255_255244


namespace second_chapter_pages_l255_255177

theorem second_chapter_pages (x : ℕ) (h1 : 48 = x + 37) : x = 11 := 
sorry

end second_chapter_pages_l255_255177


namespace remainder_is_one_l255_255857

theorem remainder_is_one (dividend divisor quotient remainder : ℕ) 
  (h1 : dividend = 222) 
  (h2 : divisor = 13)
  (h3 : quotient = 17)
  (h4 : dividend = divisor * quotient + remainder) : remainder = 1 :=
sorry

end remainder_is_one_l255_255857


namespace student_knows_german_l255_255187

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

end student_knows_german_l255_255187


namespace solve_for_y_l255_255415

theorem solve_for_y (y : ℝ) (h : (↑(30 * y) + (↑(30 * y) + 17) ^ (1 / 3)) ^ (1 / 3) = 17) :
  y = 816 / 5 := 
sorry

end solve_for_y_l255_255415


namespace larger_of_two_solutions_l255_255077

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l255_255077


namespace sqrt_value_l255_255781

theorem sqrt_value (h : Real.sqrt 100.4004 = 10.02) : Real.sqrt 1.004004 = 1.002 := 
by
  sorry

end sqrt_value_l255_255781


namespace john_spends_40_dollars_l255_255384

-- Definitions based on conditions
def cost_per_loot_box : ℝ := 5
def average_value_per_loot_box : ℝ := 3.5
def average_loss : ℝ := 12

-- Prove the amount spent on loot boxes is $40
theorem john_spends_40_dollars :
  ∃ S : ℝ, (S * (cost_per_loot_box - average_value_per_loot_box) / cost_per_loot_box = average_loss) ∧ S = 40 :=
by
  sorry

end john_spends_40_dollars_l255_255384


namespace race_time_A_l255_255669

theorem race_time_A (v_A v_B : ℝ) (t_A t_B : ℝ) (hA_time_eq : v_A = 1000 / t_A)
  (hB_time_eq : v_B = 960 / t_B) (hA_beats_B_40m : 1000 / v_A = 960 / v_B)
  (hA_beats_B_8s : t_B = t_A + 8) : t_A = 200 := 
  sorry

end race_time_A_l255_255669


namespace value_of_x2_minus_y2_l255_255663

theorem value_of_x2_minus_y2 (x y : ℚ) (h1 : x + y = 9 / 17) (h2 : x - y = 1 / 19) : x^2 - y^2 = 9 / 323 :=
by
  -- the proof would go here
  sorry

end value_of_x2_minus_y2_l255_255663


namespace smallest_n_l255_255764

theorem smallest_n (n : ℕ) (h : n ≥ 2) : 
  (∃ m : ℕ, m * m = (n + 1) * (2 * n + 1) / 6) ↔ n = 337 :=
by
  sorry

end smallest_n_l255_255764


namespace trajectory_equation_l255_255491

theorem trajectory_equation (m x y : ℝ) (a b : ℝ × ℝ)
  (ha : a = (m * x, y + 1))
  (hb : b = (x, y - 1))
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) :
  m * x^2 + y^2 = 1 :=
sorry

end trajectory_equation_l255_255491


namespace max_value_of_a_l255_255659

theorem max_value_of_a (a : ℝ) : (∀ x : ℝ, x^2 - 2*x - a ≥ 0) → a ≤ -1 := 
by 
  sorry

end max_value_of_a_l255_255659


namespace prism_height_l255_255435

theorem prism_height (a h : ℝ) 
  (base_side : a = 10) 
  (total_edge_length : 3 * a + 3 * a + 3 * h = 84) : 
  h = 8 :=
by sorry

end prism_height_l255_255435


namespace length_of_crease_l255_255622

theorem length_of_crease (θ : ℝ) : 
  let B := 5
  let DM := 5 * (Real.tan θ)
  DM = 5 * (Real.tan θ) := 
by 
  sorry

end length_of_crease_l255_255622


namespace equilateral_triangle_perimeter_l255_255971

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end equilateral_triangle_perimeter_l255_255971


namespace arithmetic_sequence_l255_255680

theorem arithmetic_sequence (a : ℕ → ℝ) 
    (h : ∀ m n, |a m + a n - a (m + n)| ≤ 1 / (m + n)) :
    ∃ d, ∀ k, a k = k * d := 
sorry

end arithmetic_sequence_l255_255680


namespace sky_colors_l255_255071

theorem sky_colors (h1 : ∀ t : ℕ, t = 2) (h2 : ∀ m : ℕ, m = 60) (h3 : ∀ c : ℕ, c = 10) : 
  ∃ n : ℕ, n = 12 :=
by
  let total_duration := (2 * 60 : ℕ)
  let num_colors := total_duration / 10
  have : num_colors = 12 := by decide
  use num_colors
  assumption_needed

end sky_colors_l255_255071


namespace sale_in_2nd_month_l255_255455

-- Defining the variables for the sales in the months
def sale_in_1st_month : ℝ := 6435
def sale_in_3rd_month : ℝ := 7230
def sale_in_4th_month : ℝ := 6562
def sale_in_5th_month : ℝ := 6855
def required_sale_in_6th_month : ℝ := 5591
def required_average_sale : ℝ := 6600
def number_of_months : ℝ := 6
def total_sales_needed : ℝ := required_average_sale * number_of_months

-- Proof statement
theorem sale_in_2nd_month : sale_in_1st_month + x + sale_in_3rd_month + sale_in_4th_month + sale_in_5th_month + required_sale_in_6th_month = total_sales_needed → x = 6927 :=
by
  sorry

end sale_in_2nd_month_l255_255455


namespace h_at_2_l255_255393

noncomputable def f (x : ℝ) : ℝ := 3 * x - 4
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (3 * f x) - 3
noncomputable def h (x : ℝ) : ℝ := f (g x)

theorem h_at_2 : h 2 = 3 * Real.sqrt 6 - 13 := 
by 
  sorry -- We skip the proof steps.

end h_at_2_l255_255393


namespace roots_greater_than_one_implies_s_greater_than_zero_l255_255219

theorem roots_greater_than_one_implies_s_greater_than_zero
  (b c : ℝ)
  (h : ∃ α β : ℝ, α > 0 ∧ β > 0 ∧ (1 + α) + (1 + β) = -b ∧ (1 + α) * (1 + β) = c) :
  b + c + 1 > 0 :=
sorry

end roots_greater_than_one_implies_s_greater_than_zero_l255_255219


namespace band_row_lengths_l255_255176

theorem band_row_lengths (n : ℕ) (h1 : n = 108) (h2 : ∃ k, 10 ≤ k ∧ k ≤ 18 ∧ 108 % k = 0) : 
  (∃ count : ℕ, count = 2) :=
by 
  sorry

end band_row_lengths_l255_255176


namespace probability_p_s_multiple_of_7_l255_255991

section
variables (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 60) (h2 : 1 ≤ b ∧ b ≤ 60) (h3 : a ≠ b)

theorem probability_p_s_multiple_of_7 :
  (∃ k : ℕ, a * b + a + b = 7 * k) → (64 / 1770 : ℚ) = 32 / 885 :=
sorry
end

end probability_p_s_multiple_of_7_l255_255991


namespace product_multiplication_rule_l255_255615

theorem product_multiplication_rule (a : ℝ) : (a * a^3)^2 = a^8 := 
by  
  -- The proof will apply the rule of product multiplication here
  sorry

end product_multiplication_rule_l255_255615


namespace calculate_b_50_l255_255467

def sequence_b : ℕ → ℤ
| 0 => sorry -- This case is not used.
| 1 => 3
| (n + 2) => sequence_b (n + 1) + 3 * (n + 1) + 1

theorem calculate_b_50 : sequence_b 50 = 3727 := 
by
    sorry

end calculate_b_50_l255_255467


namespace evaluate_101_times_101_l255_255477

theorem evaluate_101_times_101 : 101 * 101 = 10201 :=
by sorry

end evaluate_101_times_101_l255_255477


namespace frequency_of_3rd_group_l255_255123

theorem frequency_of_3rd_group (m : ℕ) (h_m : m ≥ 3) (x : ℝ) (h_area_relation : ∀ k, k ≠ 3 → 4 * x = k):
  100 * x = 20 :=
by
  sorry

end frequency_of_3rd_group_l255_255123


namespace paving_cost_is_16500_l255_255559

-- Define the given conditions
def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_meter : ℝ := 800

-- Define the area calculation
def area (L W : ℝ) : ℝ := L * W

-- Define the cost calculation
def cost (A rate : ℝ) : ℝ := A * rate

-- The theorem to prove that the cost of paving the floor is 16500
theorem paving_cost_is_16500 : cost (area length width) rate_per_sq_meter = 16500 :=
by
  -- Proof is omitted here
  sorry

end paving_cost_is_16500_l255_255559


namespace glove_ratio_l255_255978

theorem glove_ratio (P : ℕ) (G : ℕ) (hf : P = 43) (hg : G = 2 * P) : G / P = 2 := by
  rw [hf, hg]
  norm_num
  sorry

end glove_ratio_l255_255978


namespace volunteer_org_percentage_change_l255_255194

theorem volunteer_org_percentage_change 
  (initial_membership : ℝ)
  (fall_increase_rate : ℝ)
  (spring_decrease_rate : ℝ) :
  (initial_membership = 100) →
  (fall_increase_rate = 0.05) →
  (spring_decrease_rate = 0.19) →
  (14.95 : ℝ) =
  ((initial_membership * (1 + fall_increase_rate)) * (1 - spring_decrease_rate)
  - initial_membership) / initial_membership * 100 := by
  sorry

end volunteer_org_percentage_change_l255_255194


namespace bridgette_has_4_birds_l255_255889

/-
Conditions:
1. Bridgette has 2 dogs.
2. Bridgette has 3 cats.
3. Bridgette has some birds.
4. She gives the dogs a bath twice a month.
5. She gives the cats a bath once a month.
6. She gives the birds a bath once every 4 months.
7. In a year, she gives a total of 96 baths.
-/

def num_birds (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ) : ℕ :=
  let yearly_dog_baths := num_dogs * dog_baths_per_month * 12
  let yearly_cat_baths := num_cats * cat_baths_per_month * 12
  let birds_baths := total_baths_per_year - (yearly_dog_baths + yearly_cat_baths)
  let baths_per_bird_per_year := 12 / bird_baths_per_4_months
  birds_baths / baths_per_bird_per_year

theorem bridgette_has_4_birds :
  ∀ (num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year : ℕ),
    num_dogs = 2 →
    num_cats = 3 →
    dog_baths_per_month = 2 →
    cat_baths_per_month = 1 →
    bird_baths_per_4_months = 4 →
    total_baths_per_year = 96 →
    num_birds num_dogs num_cats dog_baths_per_month cat_baths_per_month bird_baths_per_4_months total_baths_per_year = 4 :=
by
  intros
  sorry


end bridgette_has_4_birds_l255_255889


namespace sum_is_integer_l255_255946

theorem sum_is_integer (x y z : ℝ) (h1 : x ^ 2 = y + 2) (h2 : y ^ 2 = z + 2) (h3 : z ^ 2 = x + 2) : ∃ n : ℤ, x + y + z = n :=
  sorry

end sum_is_integer_l255_255946


namespace not_symmetric_star_l255_255896

def star (x y : ℝ) : ℝ := abs (x - 2 * y + 3)

theorem not_symmetric_star :
  ∃ x y : ℝ, star x y ≠ star y x := 
by
  sorry

end not_symmetric_star_l255_255896


namespace tan_of_cos_l255_255242

theorem tan_of_cos (α : ℝ) (h_cos : Real.cos α = -4 / 5) (h_alpha : 0 < α ∧ α < Real.pi) : 
  Real.tan α = -3 / 4 :=
sorry

end tan_of_cos_l255_255242


namespace max_clouds_crossed_by_plane_l255_255940

-- Define the conditions
def plane_region_divide (num_planes : ℕ) : ℕ :=
  num_planes + 1

-- Hypotheses/Conditions
variable (num_planes : ℕ)
variable (initial_region_clouds : ℕ)
variable (max_crosses : ℕ)

-- The primary statement to be proved
theorem max_clouds_crossed_by_plane : 
  num_planes = 10 → initial_region_clouds = 1 → max_crosses = num_planes + initial_region_clouds →
  max_crosses = 11 := 
by
  -- Placeholder for the actual proof
  intros
  sorry

end max_clouds_crossed_by_plane_l255_255940


namespace garden_fencing_l255_255190

/-- A rectangular garden has a length of 50 yards and the width is half the length.
    Prove that the total amount of fencing needed to enclose the garden is 150 yards. -/
theorem garden_fencing : 
  ∀ (length width : ℝ), 
  length = 50 ∧ width = length / 2 → 
  2 * (length + width) = 150 :=
by
  intros length width
  rintro ⟨h1, h2⟩
  sorry

end garden_fencing_l255_255190


namespace log_one_third_nine_l255_255904

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_one_third_nine : log_base (1/3) 9 = -2 := by
  sorry

end log_one_third_nine_l255_255904


namespace total_cost_is_26_30_l255_255895

open Real

-- Define the costs
def cost_snake_toy : ℝ := 11.76
def cost_cage : ℝ := 14.54

-- Define the total cost of purchases
def total_cost : ℝ := cost_snake_toy + cost_cage

-- Prove the total cost equals $26.30
theorem total_cost_is_26_30 : total_cost = 26.30 :=
by
  sorry

end total_cost_is_26_30_l255_255895


namespace complex_problem_solution_l255_255536

noncomputable def complex_problem (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) : ℂ :=
  (c^12 + d^12) / (c + d)^12

theorem complex_problem_solution (c d : ℂ) (h1 : c ≠ 0) (h2 : d ≠ 0) (h3 : c^2 - c * d + d^2 = 0) :
  complex_problem c d h1 h2 h3 = 2 / 81 := 
sorry

end complex_problem_solution_l255_255536


namespace triangle_area_proof_l255_255495

noncomputable def triangle_area (a b c C : ℝ) : ℝ := 0.5 * a * b * Real.sin C

theorem triangle_area_proof:
  ∀ (A B C a b c : ℝ),
  ¬ (C = π/2) ∧
  c = 1 ∧
  C = π/3 ∧
  Real.sin C + Real.sin (A - B) = 3 * Real.sin (2*B) →
  triangle_area a b c C = 3 * Real.sqrt 3 / 28 :=
by
  intros A B C a b c h
  sorry

end triangle_area_proof_l255_255495


namespace visible_yellow_bus_length_correct_l255_255833

noncomputable def red_bus_length : ℝ := 48
noncomputable def orange_car_length : ℝ := red_bus_length / 4
noncomputable def yellow_bus_length : ℝ := 3.5 * orange_car_length
noncomputable def green_truck_length : ℝ := 2 * orange_car_length
noncomputable def total_vehicle_length : ℝ := yellow_bus_length + green_truck_length
noncomputable def visible_yellow_bus_length : ℝ := 0.75 * yellow_bus_length

theorem visible_yellow_bus_length_correct :
  visible_yellow_bus_length = 31.5 := 
sorry

end visible_yellow_bus_length_correct_l255_255833


namespace total_salad_dressing_weight_l255_255692

noncomputable def bowl_volume := 150 -- Volume of the bowl in ml
def oil_fraction := 2 / 3 -- Fraction of the bowl that is oil
def vinegar_fraction := 1 / 3 -- Fraction of the bowl that is vinegar
def oil_density := 5 -- Density of oil (g/ml)
def vinegar_density := 4 -- Density of vinegar (g/ml)

def oil_volume := oil_fraction * bowl_volume -- Volume of oil in ml
def vinegar_volume := vinegar_fraction * bowl_volume -- Volume of vinegar in ml
def oil_weight := oil_volume * oil_density -- Weight of the oil in grams
def vinegar_weight := vinegar_volume * vinegar_density -- Weight of the vinegar in grams
def total_weight := oil_weight + vinegar_weight -- Total weight of the salad dressing in grams

theorem total_salad_dressing_weight : total_weight = 700 := by
  sorry

end total_salad_dressing_weight_l255_255692


namespace inequalities_hold_l255_255008

theorem inequalities_hold 
  (x y z a b c : ℕ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)   -- Given that x, y, z are positive integers
  (ha : a > 0) (hb : b > 0) (hc : c > 0)   -- Given that a, b, c are positive integers
  (hxa : x ≤ a) (hyb : y ≤ b) (hzc : z ≤ c) :
  x^2 * y^2 + y^2 * z^2 + z^2 * x^2 ≤ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 ∧ 
  x^3 + y^3 + z^3 ≤ a^3 + b^3 + c^3 ∧ 
  x^2 * y * z + y^2 * z * x + z^2 * x * y ≤ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by
  sorry

end inequalities_hold_l255_255008


namespace simplify_and_evaluate_expression_l255_255414

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_and_evaluate_expression :
  ((x + 1) / (x^2 + 2 * x + 1)) / (1 - (2 / (x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l255_255414


namespace suitable_for_census_l255_255731

-- Define types for each survey option.
inductive SurveyOption where
  | A : SurveyOption -- Understanding the vision of middle school students in our province
  | B : SurveyOption -- Investigating the viewership of "The Reader"
  | C : SurveyOption -- Inspecting the components of a newly developed fighter jet to ensure successful test flights
  | D : SurveyOption -- Testing the lifespan of a batch of light bulbs

-- Theorem statement asserting that Option C is the suitable one for a census.
theorem suitable_for_census : SurveyOption.C = SurveyOption.C :=
by
  exact rfl

end suitable_for_census_l255_255731


namespace determinant_scaled_l255_255920

-- Define the initial determinant condition
def init_det (x y z w : ℝ) : Prop :=
  x * w - y * z = -3

-- Define the scaled determinant
def scaled_det (x y z w : ℝ) : ℝ :=
  3 * x * (3 * w) - 3 * y * (3 * z)

-- State the theorem we want to prove
theorem determinant_scaled (x y z w : ℝ) (h : init_det x y z w) :
  scaled_det x y z w = -27 :=
by
  sorry

end determinant_scaled_l255_255920


namespace gcd_30_45_is_15_l255_255628

theorem gcd_30_45_is_15 : Nat.gcd 30 45 = 15 := by
  sorry

end gcd_30_45_is_15_l255_255628


namespace probability_at_least_40_cents_heads_l255_255826

noncomputable def value_of_heads (p n d q h : Bool) : Real :=
  (if p then 0.01 else 0) + (if n then 0.05 else 0) + (if d then 0.10 else 0) + (if q then 0.25 else 0) + (if h then 0.50 else 0)

theorem probability_at_least_40_cents_heads :
  let outcomes := {p : Bool, n : Bool, d : Bool, q : Bool, h : Bool}
  let favorable := (outcomes.filter $ λ (o : outcomes), value_of_heads o.p o.n o.d o.q o.h >= 0.40).size
  favorable / (outcomes.size : Real) = 19 / 32 :=
by
  sorry

end probability_at_least_40_cents_heads_l255_255826


namespace pete_mileage_l255_255952

def steps_per_flip : Nat := 100000
def flips : Nat := 50
def final_reading : Nat := 25000
def steps_per_mile : Nat := 2000

theorem pete_mileage :
  let total_steps := (steps_per_flip * flips) + final_reading
  let total_miles := total_steps.toFloat / steps_per_mile.toFloat
  total_miles = 2512.5 :=
by
  sorry

end pete_mileage_l255_255952


namespace unique_real_solution_l255_255009

theorem unique_real_solution :
  ∃! (a b : ℝ), 2 * (a^2 + 1) * (b^2 + 1) = (a + 1)^2 * (ab + 1) ∧ a = 1 ∧ b = 1 :=
by
  sorry

end unique_real_solution_l255_255009


namespace problem_solution_l255_255269

theorem problem_solution (u v : ℤ) (h₁ : 0 < v) (h₂ : v < u) (h₃ : u^2 + 3 * u * v = 451) : u + v = 21 :=
sorry

end problem_solution_l255_255269


namespace martin_total_waste_is_10_l255_255398

def martinWastesTrafficTime : Nat := 2
def martinWastesFreewayTime : Nat := 4 * martinWastesTrafficTime
def totalTimeWasted : Nat := martinWastesTrafficTime + martinWastesFreewayTime

theorem martin_total_waste_is_10 : totalTimeWasted = 10 := 
by 
  sorry

end martin_total_waste_is_10_l255_255398


namespace distinct_triangle_not_isosceles_l255_255111

theorem distinct_triangle_not_isosceles (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : 
  ¬(a = b ∨ b = c ∨ c = a) :=
by {
  sorry
}

end distinct_triangle_not_isosceles_l255_255111


namespace percentage_problem_l255_255105

variable (y x z : ℝ)

def A := y * x^2 + 3 * z - 6

theorem percentage_problem (h : A y x z > 0) :
  (2 * A y x z / 5) + (3 * A y x z / 10) = (70 / 100) * A y x z :=
by
  sorry

end percentage_problem_l255_255105


namespace expression_divisible_by_1961_l255_255139

theorem expression_divisible_by_1961 (n : ℕ) : 
  (5^(2*n) * 3^(4*n) - 2^(6*n)) % 1961 = 0 := by
  sorry

end expression_divisible_by_1961_l255_255139


namespace tank_holds_21_liters_l255_255747

def tank_capacity (S L : ℝ) : Prop :=
  (L = 2 * S + 3) ∧
  (L = 4) ∧
  (2 * S + 5 * L = 21)

theorem tank_holds_21_liters :
  ∃ S L : ℝ, tank_capacity S L :=
by
  use 1/2, 4
  unfold tank_capacity
  simp
  sorry

end tank_holds_21_liters_l255_255747


namespace johns_number_l255_255531

theorem johns_number (n : ℕ) (h1 : ∃ k₁ : ℤ, n = 125 * k₁) (h2 : ∃ k₂ : ℤ, n = 180 * k₂) (h3 : 1000 < n) (h4 : n < 3000) : n = 1800 :=
sorry

end johns_number_l255_255531


namespace tan_double_angle_l255_255644

theorem tan_double_angle (α : ℝ) (h : Real.tan (π - α) = 2) : Real.tan (2 * α) = 4 / 3 :=
by
  sorry

end tan_double_angle_l255_255644


namespace expenditure_proof_l255_255429

namespace OreoCookieProblem

variables (O C : ℕ) (CO CC : ℕ → ℕ) (total_items cost_difference : ℤ)

def oreo_count_eq : Prop := O = (4 * (65 : ℤ) / 13)
def cookie_count_eq : Prop := C = (9 * (65 : ℤ) / 13)
def oreo_cost (o : ℕ) : ℕ := o * 2
def cookie_cost (c : ℕ) : ℕ := c * 3
def total_item_condition : Prop := O + C = 65
def ratio_condition : Prop := 9 * O = 4 * C
def cost_difference_condition (o_cost c_cost : ℕ) : Prop := cost_difference = (c_cost - o_cost)

theorem expenditure_proof :
  (O + C = 65) →
  (9 * O = 4 * C) →
  (O = 20) →
  (C = 45) →
  cost_difference = (45 * 3 - 20 * 2) →
  cost_difference = 95 :=
by sorry

end OreoCookieProblem

end expenditure_proof_l255_255429


namespace probability_of_40_cents_l255_255823

noncomputable def num_successful_outcomes : ℕ := 16 + 3

def total_outcomes : ℕ := 2 ^ 5

def probability_success : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_40_cents : probability_success = 19 / 32 := by
  unfold probability_success num_successful_outcomes total_outcomes
  norm_num
  sorry

end probability_of_40_cents_l255_255823


namespace integer_solution_l255_255163

theorem integer_solution (x : ℤ) (h : x^2 < 3 * x) : x = 1 ∨ x = 2 :=
sorry

end integer_solution_l255_255163


namespace pharmacist_weights_exist_l255_255594

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l255_255594


namespace charles_travel_time_l255_255107

theorem charles_travel_time (D S T : ℕ) (hD : D = 6) (hS : S = 3) : T = D / S → T = 2 :=
by
  intros h
  rw [hD, hS] at h
  simp at h
  exact h

end charles_travel_time_l255_255107


namespace spade_5_7_8_l255_255771

def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spade_5_7_8 : spade 5 (spade 7 8) = -200 :=
by
  sorry

end spade_5_7_8_l255_255771


namespace field_fence_length_l255_255046

theorem field_fence_length (L : ℝ) (A : ℝ) (W : ℝ) (fencing : ℝ) (hL : L = 20) (hA : A = 210) (hW : A = L * W) : 
  fencing = 2 * W + L → fencing = 41 :=
by
  rw [hL, hA] at hW
  sorry

end field_fence_length_l255_255046


namespace find_x_l255_255392

def x_condition (x : ℤ) : Prop :=
  (120 ≤ x ∧ x ≤ 150) ∧ (x % 5 = 2) ∧ (x % 6 = 5)

theorem find_x :
  ∃ x : ℤ, x_condition x ∧ x = 137 :=
by
  sorry

end find_x_l255_255392


namespace jane_reading_days_l255_255127

theorem jane_reading_days
  (pages : ℕ)
  (half_pages : ℕ)
  (speed_first_half : ℕ)
  (speed_second_half : ℕ)
  (days_first_half : ℕ)
  (days_second_half : ℕ)
  (total_days : ℕ)
  (h1 : pages = 500)
  (h2 : half_pages = pages / 2)
  (h3 : speed_first_half = 10)
  (h4 : speed_second_half = 5)
  (h5 : days_first_half = half_pages / speed_first_half)
  (h6 : days_second_half = half_pages / speed_second_half)
  (h7 : total_days = days_first_half + days_second_half) :
  total_days = 75 :=
by
  sorry

end jane_reading_days_l255_255127


namespace minimum_value_x_plus_y_l255_255538

theorem minimum_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + y^2 = 1) : x + y = 16 :=
sorry

end minimum_value_x_plus_y_l255_255538


namespace gasoline_fraction_used_l255_255734

theorem gasoline_fraction_used
  (speed : ℕ) (gas_usage : ℕ) (initial_gallons : ℕ) (travel_time : ℕ)
  (h_speed : speed = 50) (h_gas_usage : gas_usage = 30) 
  (h_initial_gallons : initial_gallons = 15) (h_travel_time : travel_time = 5) :
  (speed * travel_time) / gas_usage / initial_gallons = 5 / 9 :=
by
  sorry

end gasoline_fraction_used_l255_255734


namespace units_digit_2016_pow_2017_add_2017_pow_2016_l255_255445

theorem units_digit_2016_pow_2017_add_2017_pow_2016 :
  (2016 ^ 2017 + 2017 ^ 2016) % 10 = 7 :=
by
  sorry

end units_digit_2016_pow_2017_add_2017_pow_2016_l255_255445


namespace line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l255_255792

-- Define the points A, B and P
def A : ℝ × ℝ := (4, 0)
def B : ℝ × ℝ := (0, 2)
def P : ℝ × ℝ := (2, 3)
def O : ℝ × ℝ := (0, 0)

-- Define the functions and theorems for the problem
theorem line_through_P_parallel_to_AB :
  ∃ k b : ℝ, ∀ x y : ℝ, ((y = k * x + b) ↔ (x + 2 * y - 8 = 0)) :=
sorry

theorem circumcircle_of_triangle_OAB :
  ∃ cx cy r : ℝ, (cx, cy) = (2, 1) ∧ r^2 = 5 ∧ ∀ x y : ℝ, ((x - cx)^2 + (y - cy)^2 = r^2) ↔ ((x - 2)^2 + (y - 1)^2 = 5) :=
sorry

end line_through_P_parallel_to_AB_circumcircle_of_triangle_OAB_l255_255792


namespace balloons_kept_by_Andrew_l255_255325

theorem balloons_kept_by_Andrew :
  let blue := 303
  let purple := 453
  let red := 165
  let yellow := 324
  let blue_kept := (2/3 : ℚ) * blue
  let purple_kept := (3/5 : ℚ) * purple
  let red_kept := (4/7 : ℚ) * red
  let yellow_kept := (1/3 : ℚ) * yellow
  let total_kept := blue_kept.floor + purple_kept.floor + red_kept.floor + yellow_kept
  total_kept = 675 := by
  sorry

end balloons_kept_by_Andrew_l255_255325


namespace original_denominator_value_l255_255321

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l255_255321


namespace constant_function_of_zero_derivative_l255_255800

theorem constant_function_of_zero_derivative
  {f : ℝ → ℝ}
  (h : ∀ x : ℝ, deriv f x = 0) :
  ∃ c : ℝ, ∀ x : ℝ, f x = c :=
sorry

end constant_function_of_zero_derivative_l255_255800


namespace sum_of_fourth_powers_l255_255202

-- Define the sum of fourth powers as per the given formula
noncomputable def sum_fourth_powers (n: ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1)) / 30

-- Define the statement to be proved
theorem sum_of_fourth_powers :
  2 * sum_fourth_powers 100 = 41006666600 :=
by sorry

end sum_of_fourth_powers_l255_255202


namespace Kvi_wins_race_l255_255993

/-- Define the frogs and their properties --/
structure Frog :=
  (name : String)
  (jump_distance_in_dm : ℕ) /-- jump distance in decimeters --/
  (jumps_per_cycle : ℕ) /-- number of jumps per cycle (unit time of reference) --/

def FrogKva : Frog := ⟨"Kva", 6, 2⟩
def FrogKvi : Frog := ⟨"Kvi", 4, 3⟩

/-- Define the conditions for the race --/
def total_distance_in_m : ℕ := 40
def total_distance_in_dm := total_distance_in_m * 10

/-- Racing function to determine winner --/
def race_winner (f1 f2 : Frog) (total_distance : ℕ) : String :=
  if (total_distance % (f1.jump_distance_in_dm * f1.jumps_per_cycle) < total_distance % (f2.jump_distance_in_dm * f2.jumps_per_cycle))
  then f1.name
  else f2.name

/-- Proving Kvi wins under the given conditions --/
theorem Kvi_wins_race :
  race_winner FrogKva FrogKvi total_distance_in_dm = "Kvi" :=
by
  sorry

end Kvi_wins_race_l255_255993


namespace tangent_line_at_point_l255_255976

theorem tangent_line_at_point (x y : ℝ) (h : y = x^2) (hx : x = 1) (hy : y = 1) : 
  2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_at_point_l255_255976


namespace option_b_is_same_type_l255_255576

def polynomial_same_type (p1 p2 : ℕ → ℕ → ℕ) : Prop :=
  ∀ x y, (p1 x y = 1 → p2 x y = 1) ∧ (p2 x y = 1 → p1 x y = 1)

def ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0

def a_squared_b (a b : ℕ) := if a = 2 ∧ b = 1 then 1 else 0
def negative_two_ab_squared (a b : ℕ) := if a = 1 ∧ b = 2 then 1 else 0
def ab (a b : ℕ) := if a = 1 ∧ b = 1 then 1 else 0
def ab_squared_c (a b c : ℕ) := if a = 1 ∧ b = 2 ∧ c = 1 then 1 else 0

theorem option_b_is_same_type : polynomial_same_type ab_squared negative_two_ab_squared :=
by
  sorry

end option_b_is_same_type_l255_255576


namespace Jakes_weight_is_198_l255_255864

variable (Jake Kendra : ℕ)

-- Conditions
variable (h1 : Jake - 8 = 2 * Kendra)
variable (h2 : Jake + Kendra = 293)

theorem Jakes_weight_is_198 : Jake = 198 :=
by
  sorry

end Jakes_weight_is_198_l255_255864


namespace infinite_series_value_l255_255894

theorem infinite_series_value :
  ∑' n : ℕ, (n^3 + 4 * n^2 + 8 * n + 8) / (3^n * (n^3 + 5)) = 1 / 2 :=
by sorry

end infinite_series_value_l255_255894


namespace percentage_pure_acid_l255_255019

theorem percentage_pure_acid (volume_pure_acid total_volume: ℝ) (h1 : volume_pure_acid = 1.4) (h2 : total_volume = 4) : 
  (volume_pure_acid / total_volume) * 100 = 35 := 
by
  -- Given metric volumes of pure acid and total solution, we need to prove the percentage 
  -- Here, we assert the conditions and conclude the result
  sorry

end percentage_pure_acid_l255_255019


namespace no_natural_pairs_exist_l255_255908

theorem no_natural_pairs_exist (n m : ℕ) : ¬(n + 1) * (2 * n + 1) = 18 * m ^ 2 :=
by
  sorry

end no_natural_pairs_exist_l255_255908


namespace flowers_lost_l255_255265

theorem flowers_lost 
  (time_per_flower : ℕ)
  (gathered_time : ℕ) 
  (additional_time : ℕ) 
  (classmates : ℕ) 
  (collected_flowers : ℕ) 
  (total_needed : ℕ)
  (lost_flowers : ℕ) 
  (H1 : time_per_flower = 10)
  (H2 : gathered_time = 120)
  (H3 : additional_time = 210)
  (H4 : classmates = 30)
  (H5 : collected_flowers = gathered_time / time_per_flower)
  (H6 : total_needed = classmates + (additional_time / time_per_flower))
  (H7 : lost_flowers = total_needed - classmates) :
lost_flowers = 3 := 
sorry

end flowers_lost_l255_255265


namespace number_of_bricks_l255_255038

noncomputable def bricklayer_one_hours : ℝ := 8
noncomputable def bricklayer_two_hours : ℝ := 12
noncomputable def reduction_rate : ℝ := 12
noncomputable def combined_hours : ℝ := 6

theorem number_of_bricks (y : ℝ) :
  ((combined_hours * ((y / bricklayer_one_hours) + (y / bricklayer_two_hours) - reduction_rate)) = y) →
  y = 288 :=
by sorry

end number_of_bricks_l255_255038


namespace zoe_total_songs_l255_255032

def initial_songs : ℕ := 15
def deleted_songs : ℕ := 8
def added_songs : ℕ := 50

theorem zoe_total_songs : initial_songs - deleted_songs + added_songs = 57 := by
  sorry

end zoe_total_songs_l255_255032


namespace milford_age_in_3_years_l255_255903

theorem milford_age_in_3_years (current_age_eustace : ℕ) (current_age_milford : ℕ) :
  (current_age_eustace = 2 * current_age_milford) → 
  (current_age_eustace + 3 = 39) → 
  current_age_milford + 3 = 21 :=
by
  intros h1 h2
  sorry

end milford_age_in_3_years_l255_255903


namespace find_ratio_l255_255656

-- Definition of the function
def f (x : ℝ) (a b: ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

-- Statement to be proved
theorem find_ratio (a b : ℝ) (h1: f 1 a b = 10) (h2 : (3 * 1^2 + 2 * a * 1 + b = 0)) : b = -a / 2 :=
by
  sorry

end find_ratio_l255_255656


namespace cricket_average_l255_255453

theorem cricket_average (x : ℝ) (h1 : 15 * x + 121 = 16 * (x + 6)) : x = 25 := by
  -- proof goes here, but we skip it with sorry
  sorry

end cricket_average_l255_255453


namespace cream_cheese_volume_l255_255400

theorem cream_cheese_volume
  (raw_spinach : ℕ)
  (spinach_reduction : ℕ)
  (eggs_volume : ℕ)
  (total_volume : ℕ)
  (cooked_spinach : ℕ)
  (cream_cheese : ℕ) :
  raw_spinach = 40 →
  spinach_reduction = 20 →
  eggs_volume = 4 →
  total_volume = 18 →
  cooked_spinach = raw_spinach * spinach_reduction / 100 →
  cream_cheese = total_volume - cooked_spinach - eggs_volume →
  cream_cheese = 6 :=
by
  intros h_raw_spinach h_spinach_reduction h_eggs_volume h_total_volume h_cooked_spinach h_cream_cheese
  sorry

end cream_cheese_volume_l255_255400


namespace tank_base_length_width_difference_l255_255751

variable (w l h : ℝ)

theorem tank_base_length_width_difference :
  (l = 5 * w) →
  (h = (1/2) * w) →
  (l * w * h = 3600) →
  (|l - w - 45.24| < 0.01) := 
by
  sorry

end tank_base_length_width_difference_l255_255751


namespace larger_of_two_solutions_l255_255078

theorem larger_of_two_solutions (x : ℝ) : 
  (x^2 - 13*x + 36 = 0 → x = 9 ∨ x = 4) →
  (∃ y z : ℝ, y ≠ z ∧ y^2 - 13*y + 36 = 0 ∧ z^2 - 13*z + 36 = 0 ∧ ∀ w, w^2 - 13*w + 36 = 0 → w = y ∨ w = z ∧ max y z = 9) :=
begin
  sorry
end

end larger_of_two_solutions_l255_255078


namespace variance_scaled_data_l255_255506

noncomputable def variance (data : List ℝ) : ℝ :=
  let n := data.length
  let mean := data.sum / n
  (data.map (λ x => (x - mean) ^ 2)).sum / n

theorem variance_scaled_data (data : List ℝ) (h_len : data.length > 0) (h_var : variance data = 4) :
  variance (data.map (λ x => 2 * x)) = 16 :=
by
  sorry

end variance_scaled_data_l255_255506


namespace find_a_l255_255779

theorem find_a (x y a : ℤ) (h₁ : x = 1) (h₂ : y = -1) (h₃ : 2 * x - a * y = 3) : a = 1 :=
sorry

end find_a_l255_255779


namespace parallel_lines_slope_condition_l255_255473

theorem parallel_lines_slope_condition (b : ℝ) (x y : ℝ) :
    (∀ (x y : ℝ), 3 * y - 3 * b = 9 * x) →
    (∀ (x y : ℝ), y + 2 = (b + 9) * x) →
    b = -6 :=
by
    sorry

end parallel_lines_slope_condition_l255_255473


namespace jordans_score_l255_255003

theorem jordans_score 
  (N : ℕ) 
  (first_19_avg : ℚ) 
  (total_avg : ℚ)
  (total_score_19 : ℚ) 
  (total_score_20 : ℚ) 
  (jordan_score : ℚ) 
  (h1 : N = 19)
  (h2 : first_19_avg = 74)
  (h3 : total_avg = 76)
  (h4 : total_score_19 = N * first_19_avg)
  (h5 : total_score_20 = (N + 1) * total_avg)
  (h6 : jordan_score = total_score_20 - total_score_19) :
  jordan_score = 114 :=
by {
  -- the proof will be filled in, but for now we use sorry
  sorry
}

end jordans_score_l255_255003


namespace price_rollback_is_correct_l255_255836

-- Define the conditions
def liters_today : ℕ := 10
def cost_per_liter_today : ℝ := 1.4
def liters_friday : ℕ := 25
def total_liters : ℕ := 35
def total_cost : ℝ := 39

-- Define the price rollback calculation
noncomputable def price_rollback : ℝ :=
  (cost_per_liter_today - (total_cost - (liters_today * cost_per_liter_today)) / liters_friday)

-- The theorem stating the rollback per liter is $0.4
theorem price_rollback_is_correct : price_rollback = 0.4 := by
  sorry

end price_rollback_is_correct_l255_255836


namespace price_after_discount_l255_255901

-- Define the original price and discount
def original_price : ℕ := 76
def discount : ℕ := 25

-- The main proof statement
theorem price_after_discount : original_price - discount = 51 := by
  sorry

end price_after_discount_l255_255901


namespace percentage_less_than_l255_255866

theorem percentage_less_than (P T J : ℝ) 
  (h1 : T = 0.9375 * P) 
  (h2 : J = 0.8 * T) 
  : (P - J) / P * 100 = 25 := 
by
  sorry

end percentage_less_than_l255_255866


namespace polynomial_integer_roots_l255_255910

theorem polynomial_integer_roots :
  ∀ x : ℤ, (x^3 - 3*x^2 - 10*x + 20 = 0) ↔ (x = -2 ∨ x = 5) :=
by
  sorry

end polynomial_integer_roots_l255_255910


namespace find_other_parallel_side_l255_255212

theorem find_other_parallel_side 
  (a b d : ℝ) 
  (area : ℝ) 
  (h_area : area = 285) 
  (h_a : a = 20) 
  (h_d : d = 15)
  : (∃ x : ℝ, area = 1/2 * (a + x) * d ∧ x = 18) :=
by
  sorry

end find_other_parallel_side_l255_255212


namespace simplify_expression_l255_255285

variable (x : ℝ)

theorem simplify_expression : 
  (3 * x + 6 - 5 * x) / 3 = (-2 / 3) * x + 2 := by
  sorry

end simplify_expression_l255_255285


namespace election_total_polled_votes_l255_255303

theorem election_total_polled_votes (V : ℝ) (invalid_votes : ℝ) (candidate_votes : ℝ) (margin : ℝ)
  (h1 : candidate_votes = 0.3 * V)
  (h2 : margin = 5000)
  (h3 : V = 0.3 * V + (0.3 * V + margin))
  (h4 : invalid_votes = 100) :
  V + invalid_votes = 12600 :=
by
  sorry

end election_total_polled_votes_l255_255303


namespace factor_expression_l255_255629

theorem factor_expression :
  (12 * x ^ 6 + 40 * x ^ 4 - 6) - (2 * x ^ 6 - 6 * x ^ 4 - 6) = 2 * x ^ 4 * (5 * x ^ 2 + 23) :=
by sorry

end factor_expression_l255_255629


namespace total_arms_collected_l255_255759

-- Define the conditions as parameters
def arms_of_starfish := 7 * 5
def arms_of_seastar := 14

-- Define the theorem to prove total arms
theorem total_arms_collected : arms_of_starfish + arms_of_seastar = 49 := by
  sorry

end total_arms_collected_l255_255759


namespace distance_between_adjacent_symmetry_axes_l255_255233

noncomputable def f (x : ℝ) : ℝ := (Real.cos (3 * x))^2 - 1/2

theorem distance_between_adjacent_symmetry_axes :
  (∃ x : ℝ, f x = f (x + π / 3)) → (∃ d : ℝ, d = π / 6) :=
by
  -- Prove the distance is π / 6 based on the properties of f(x).
  sorry

end distance_between_adjacent_symmetry_axes_l255_255233


namespace range_of_a_l255_255101

noncomputable def set_A : Set ℝ := {x | x^2 + 4 * x = 0}
noncomputable def set_B (a : ℝ) : Set ℝ := {x | x^2 + a * x + a = 0}

theorem range_of_a (a : ℝ) (h : set_A ∪ set_B a = set_A) : 0 ≤ a ∧ a < 4 := 
sorry

end range_of_a_l255_255101


namespace steven_sixth_quiz_score_l255_255416

theorem steven_sixth_quiz_score :
  ∃ x : ℕ, (75 + 80 + 85 + 90 + 100 + x) / 6 = 95 ∧ x = 140 :=
by
  sorry

end steven_sixth_quiz_score_l255_255416


namespace worker_new_wage_after_increase_l255_255054

theorem worker_new_wage_after_increase (initial_wage : ℝ) (increase_percentage : ℝ) (new_wage : ℝ) 
  (h1 : initial_wage = 34) (h2 : increase_percentage = 0.50) 
  (h3 : new_wage = initial_wage + (increase_percentage * initial_wage)) : new_wage = 51 := 
by
  sorry

end worker_new_wage_after_increase_l255_255054


namespace simplify_expression_l255_255413

variable (y : ℝ)

theorem simplify_expression : (3 * y)^3 + (4 * y) * (y^2) - 2 * y^3 = 29 * y^3 :=
by
  sorry

end simplify_expression_l255_255413


namespace total_votes_election_l255_255117

theorem total_votes_election 
  (votes_A : ℝ) 
  (valid_votes_percentage : ℝ) 
  (invalid_votes_percentage : ℝ)
  (votes_candidate_A : ℝ) 
  (total_votes : ℝ) 
  (h1 : votes_A = 0.60) 
  (h2 : invalid_votes_percentage = 0.15) 
  (h3 : votes_candidate_A = 285600) 
  (h4 : valid_votes_percentage = 0.85) 
  (h5 : total_votes = 560000) 
  : 
  ((votes_A * valid_votes_percentage * total_votes) = votes_candidate_A) 
  := 
  by sorry

end total_votes_election_l255_255117


namespace pencils_per_sibling_l255_255961

theorem pencils_per_sibling :
  let total_pencils := 49
  let kept_pencils := 10
  let siblings := 3
  let remaining_pencils := total_pencils - kept_pencils
  (remaining_pencils / siblings) = 13 :=
by
  -- Definitions for the variables
  let total_pencils := 49
  let kept_pencils := 10
  let siblings := 3
  let remaining_pencils := total_pencils - kept_pencils

  -- Simplification to show the desired result
  have h1 : remaining_pencils = 39 := by
    calc
      remaining_pencils = total_pencils - kept_pencils : rfl
      ... = 49 - 10 : rfl
      ... = 39 : rfl

  have h2 : (remaining_pencils / siblings) = 13 := by
    calc
      (remaining_pencils / siblings) = 39 / 3 : by rw [h1]
      ... = 13 : by norm_num

  exact h2

end pencils_per_sibling_l255_255961


namespace geometric_sequence_k_value_l255_255349

theorem geometric_sequence_k_value (a : ℕ → ℝ) (S : ℕ → ℝ) (a1_pos : 0 < a 1)
  (geometric_seq : ∀ n, a (n + 2) = a n * (a 3 / a 1)) (h_a1 : a 1 = 1)
  (h_a3 : a 3 = 4) (h_Sk : S k = 63) :
  k = 6 := 
by
  sorry

end geometric_sequence_k_value_l255_255349


namespace rotten_eggs_prob_l255_255462

theorem rotten_eggs_prob (T : ℕ) (P : ℝ) (R : ℕ) :
  T = 36 ∧ P = 0.0047619047619047615 ∧ P = (R / T) * ((R - 1) / (T - 1)) → R = 3 :=
by
  sorry

end rotten_eggs_prob_l255_255462


namespace set_equality_l255_255942

-- Define the universe U
def U := ℝ

-- Define the set M
def M := {x : ℝ | (x + 1) * (x - 2) ≤ 0}

-- Define the set N
def N := {x : ℝ | x > 1}

-- Define the set we want to prove is equal to the intersection of M and N
def target_set := {x : ℝ | 1 < x ∧ x ≤ 2}

theorem set_equality : target_set = M ∩ N := 
by sorry

end set_equality_l255_255942


namespace hexagon_coloring_unique_l255_255427

-- Define the coloring of the hexagon using enumeration
inductive Color
  | green
  | blue
  | orange

-- Assume we have a function that represents the coloring of the hexagons
-- with the constraints given in the problem
def is_valid_coloring (coloring : ℕ → ℕ → Color) : Prop :=
  ∀ x y : ℕ, -- For all hexagons
  (coloring x y = Color.green ∧ x = 0 ∧ y = 0) ∨ -- The labeled hexagon G is green
  (coloring x y ≠ coloring (x + 1) y ∧ -- No two hexagons with a common side have the same color
   coloring x y ≠ coloring (x - 1) y ∧ 
   coloring x y ≠ coloring x (y + 1) ∧
   coloring x y ≠ coloring x (y - 1))

-- The problem is to prove there are exactly 2 valid colorings of the hexagon grid
theorem hexagon_coloring_unique :
  ∃ (count : ℕ), count = 2 ∧
  ∀ coloring : (ℕ → ℕ → Color), is_valid_coloring coloring → count = 2 :=
by
  sorry

end hexagon_coloring_unique_l255_255427


namespace arithmetic_sequence_sum_l255_255201

theorem arithmetic_sequence_sum :
  ∃ (a l d n : ℕ), a = 71 ∧ l = 109 ∧ d = 2 ∧ n = ((l - a) / d) + 1 ∧ 
    (3 * (n * (a + l) / 2) = 5400) := sorry

end arithmetic_sequence_sum_l255_255201


namespace inscribed_circle_ratio_l255_255459

theorem inscribed_circle_ratio (a b h r : ℝ) (h_triangle : h = Real.sqrt (a^2 + b^2))
  (A : ℝ) (H1 : A = (1/2) * a * b) (s : ℝ) (H2 : s = (a + b + h) / 2) 
  (H3 : A = r * s) : (π * r / A) = (π * r) / (h + r) :=
sorry

end inscribed_circle_ratio_l255_255459


namespace interval_between_prizes_l255_255255

theorem interval_between_prizes (total_prize : ℝ) (first_place : ℝ) (interval : ℝ) :
  total_prize = 4800 ∧
  first_place = 2000 ∧
  (first_place - interval) + (first_place - 2 * interval) = total_prize - 2000 →
  interval = 400 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  rw [h1, h2] at h3
  sorry

end interval_between_prizes_l255_255255


namespace current_swans_number_l255_255905

noncomputable def swans_doubling (S : ℕ) : Prop :=
  let S_after_10_years := S * 2^5 -- Doubling every 2 years for 10 years results in multiplying by 2^5
  S_after_10_years = 480

theorem current_swans_number (S : ℕ) (h : swans_doubling S) : S = 15 := by
  sorry

end current_swans_number_l255_255905


namespace solution_set_empty_for_k_l255_255214

theorem solution_set_empty_for_k (k : ℝ) :
  (∀ x : ℝ, ¬ (kx^2 - 2 * |x - 1| + 3 * k < 0)) ↔ (1 ≤ k) :=
by
  sorry

end solution_set_empty_for_k_l255_255214


namespace complete_square_identity_l255_255624

theorem complete_square_identity (x d e : ℤ) (h : x^2 - 10 * x + 15 = 0) :
  (x + d)^2 = e → d + e = 5 :=
by
  intros hde
  sorry

end complete_square_identity_l255_255624


namespace poplar_more_than_pine_l255_255848

theorem poplar_more_than_pine (pine poplar : ℕ) (h1 : pine = 180) (h2 : poplar = 4 * pine) : poplar - pine = 540 :=
by
  -- Proof will be filled here
  sorry

end poplar_more_than_pine_l255_255848


namespace original_garden_length_l255_255831

theorem original_garden_length (x : ℝ) (area : ℝ) (reduced_length : ℝ) (width : ℝ) (length_condition : x - reduced_length = width) (area_condition : x * width = area) (given_area : area = 120) (given_reduced_length : reduced_length = 2) : x = 12 := 
by
  sorry

end original_garden_length_l255_255831


namespace fraction_to_terminating_decimal_l255_255484

-- Lean statement for the mathematical problem
theorem fraction_to_terminating_decimal: (13 : ℚ) / 200 = 0.26 := 
sorry

end fraction_to_terminating_decimal_l255_255484


namespace initial_water_amount_l255_255738

theorem initial_water_amount (W : ℝ) (h1 : 0.006 * 50 = 0.03 * W) : W = 10 :=
by
  -- Proof steps would go here
  sorry

end initial_water_amount_l255_255738


namespace find_A_l255_255535

def hash_relation (A B : ℕ) : ℕ := A^2 + B^2

theorem find_A (A : ℕ) (h1 : hash_relation A 7 = 218) : A = 13 := 
by sorry

end find_A_l255_255535


namespace farmer_crops_saved_l255_255744

noncomputable def average_corn_per_row := (10 + 14) / 2
noncomputable def average_potato_per_row := (35 + 45) / 2
noncomputable def average_wheat_per_row := (55 + 65) / 2

noncomputable def avg_reduction_corn := (40 + 60 + 25) / 3 / 100
noncomputable def avg_reduction_potato := (50 + 30 + 60) / 3 / 100
noncomputable def avg_reduction_wheat := (20 + 55 + 35) / 3 / 100

noncomputable def saved_corn_per_row := average_corn_per_row * (1 - avg_reduction_corn)
noncomputable def saved_potato_per_row := average_potato_per_row * (1 - avg_reduction_potato)
noncomputable def saved_wheat_per_row := average_wheat_per_row * (1 - avg_reduction_wheat)

def rows_corn := 30
def rows_potato := 24
def rows_wheat := 36

noncomputable def total_saved_corn := saved_corn_per_row * rows_corn
noncomputable def total_saved_potatoes := saved_potato_per_row * rows_potato
noncomputable def total_saved_wheat := saved_wheat_per_row * rows_wheat

noncomputable def total_crops_saved := total_saved_corn + total_saved_potatoes + total_saved_wheat

theorem farmer_crops_saved : total_crops_saved = 2090 := by
  sorry

end farmer_crops_saved_l255_255744


namespace second_train_length_l255_255852

theorem second_train_length
  (train1_length : ℝ)
  (train1_speed_kmph : ℝ)
  (train2_speed_kmph : ℝ)
  (time_to_clear : ℝ)
  (h1 : train1_length = 135)
  (h2 : train1_speed_kmph = 80)
  (h3 : train2_speed_kmph = 65)
  (h4 : time_to_clear = 7.447680047665153) :
  ∃ l2 : ℝ, l2 = 165 :=
by
  let train1_speed_mps := train1_speed_kmph * 1000 / 3600
  let train2_speed_mps := train2_speed_kmph * 1000 / 3600
  let total_distance := (train1_speed_mps + train2_speed_mps) * time_to_clear
  have : total_distance = 300 := by sorry
  have l2 := total_distance - train1_length
  use l2
  have : l2 = 165 := by sorry
  assumption

end second_train_length_l255_255852


namespace smaller_number_l255_255985

theorem smaller_number (x y : ℝ) (h1 : x + y = 15) (h2 : x * y = 36) : x = 3 ∨ y = 3 := by
  sorry

end smaller_number_l255_255985


namespace problems_completed_l255_255512

theorem problems_completed (p t : ℕ) (h1 : p > 15) (h2 : pt = (2 * p - 6) * (t - 3)) : p * t = 216 := 
by
  sorry

end problems_completed_l255_255512


namespace no_solution_l255_255480

theorem no_solution (x : ℝ) : ¬ (x / -4 ≥ 3 + x ∧ |2*x - 1| < 4 + 2*x) := 
by sorry

end no_solution_l255_255480


namespace cos_double_angle_from_sin_shift_l255_255104

theorem cos_double_angle_from_sin_shift (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := 
by 
  sorry

end cos_double_angle_from_sin_shift_l255_255104


namespace total_matches_played_l255_255871

theorem total_matches_played
  (avg_runs_first_20: ℕ) (num_first_20: ℕ) (avg_runs_next_10: ℕ) (num_next_10: ℕ) (overall_avg: ℕ) (total_matches: ℕ) :
  avg_runs_first_20 = 40 →
  num_first_20 = 20 →
  avg_runs_next_10 = 13 →
  num_next_10 = 10 →
  overall_avg = 31 →
  (num_first_20 + num_next_10 = total_matches) →
  total_matches = 30 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end total_matches_played_l255_255871


namespace total_marks_l255_255385

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l255_255385


namespace mean_proportional_49_64_l255_255343

theorem mean_proportional_49_64 : Real.sqrt (49 * 64) = 56 :=
by
  sorry

end mean_proportional_49_64_l255_255343


namespace no_int_solutions_x2_minus_3y2_eq_17_l255_255548

theorem no_int_solutions_x2_minus_3y2_eq_17 : 
  ∀ (x y : ℤ), (x^2 - 3 * y^2 ≠ 17) := 
by
  intros x y
  sorry

end no_int_solutions_x2_minus_3y2_eq_17_l255_255548


namespace exists_natural_2001_digits_l255_255259

theorem exists_natural_2001_digits (N : ℕ) (hN: N = 5 * 10^2000 + 1) : 
  ∃ K : ℕ, (K = N) ∧ (N^(2001) % 10^2001 = N % 10^2001) :=
by
  sorry

end exists_natural_2001_digits_l255_255259


namespace jose_marks_difference_l255_255525

theorem jose_marks_difference (M J A : ℕ) 
  (h1 : M = J - 20)
  (h2 : J + M + A = 210)
  (h3 : J = 90) : (J - A) = 40 :=
by
  sorry

end jose_marks_difference_l255_255525


namespace wilson_theorem_application_l255_255999

theorem wilson_theorem_application (h_prime : Nat.Prime 101) : 
  Nat.factorial 100 % 101 = 100 :=
by
  -- By Wilson's theorem, (p - 1)! ≡ -1 (mod p) for a prime p.
  -- Here p = 101, so (101 - 1)! ≡ -1 (mod 101).
  -- Therefore, 100! ≡ -1 (mod 101).
  -- Knowing that -1 ≡ 100 (mod 101), we can conclude that
  -- 100! ≡ 100 (mod 101).
  sorry

end wilson_theorem_application_l255_255999


namespace max_value_x_plus_y_max_value_x_plus_y_achieved_l255_255164

theorem max_value_x_plus_y (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : x + y ≤ 6 * Real.sqrt 5 :=
by
  sorry

theorem max_value_x_plus_y_achieved (x y : ℝ) (h1: x^2 + y^2 = 100) (h2: x * y = 40) : ∃ x y, x + y = 6 * Real.sqrt 5 :=
by
  sorry

end max_value_x_plus_y_max_value_x_plus_y_achieved_l255_255164


namespace initial_bottle_caps_l255_255947

theorem initial_bottle_caps 
    (x : ℝ) 
    (Nancy_bottle_caps : ℝ) 
    (Marilyn_current_bottle_caps : ℝ) 
    (h1 : Nancy_bottle_caps = 36.0)
    (h2 : Marilyn_current_bottle_caps = 87)
    (h3 : x + Nancy_bottle_caps = Marilyn_current_bottle_caps) : 
    x = 51 := 
by 
  sorry

end initial_bottle_caps_l255_255947


namespace zero_integers_in_range_such_that_expr_is_perfect_square_l255_255358

theorem zero_integers_in_range_such_that_expr_is_perfect_square :
  (∃ n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n ^ 2 + n + 2 = m ^ 2) → False :=
by sorry

end zero_integers_in_range_such_that_expr_is_perfect_square_l255_255358


namespace interest_equality_l255_255051

theorem interest_equality (total_sum : ℝ) (part1 : ℝ) (part2 : ℝ) (rate1 : ℝ) (time1 : ℝ) (rate2 : ℝ) (n : ℝ) :
  total_sum = 2730 ∧ part1 = 1050 ∧ part2 = 1680 ∧
  rate1 = 3 ∧ time1 = 8 ∧ rate2 = 5 ∧ part1 * rate1 * time1 = part2 * rate2 * n →
  n = 3 :=
by
  sorry

end interest_equality_l255_255051


namespace triangular_region_area_l255_255053

def line (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def y (x : ℝ) := x

theorem triangular_region_area : 
  ∀ (x y: ℝ),
  (y = line 1 2 x ∧ y = 3) ∨ 
  (y = line (-1) 8 x ∧ y = 3) ∨ 
  (y = line 1 2 x ∧ y = line (-1) 8 x)
  →
  ∃ (area: ℝ), area = 4.00 := 
by
  sorry

end triangular_region_area_l255_255053


namespace max_value_of_m_l255_255655

-- Define the function f(x)
def f (x : ℝ) := x^2 + 2 * x

-- Define the property of t and m such that the condition holds for all x in [1, m]
def valid_t_m (t m : ℝ) : Prop :=
  ∀ x : ℝ, 1 ≤ x ∧ x ≤ m → f (x + t) ≤ 3 * x

-- The proof statement ensuring the maximum value of m is 8
theorem max_value_of_m 
  (t : ℝ) (m : ℝ) 
  (ht : ∃ x : ℝ, valid_t_m t x ∧ x = 8) : 
  ∀ m, valid_t_m t m → m ≤ 8 :=
  sorry

end max_value_of_m_l255_255655


namespace find_multiple_l255_255845

theorem find_multiple (m : ℤ) : 38 + m * 43 = 124 → m = 2 := by
  intro h
  sorry

end find_multiple_l255_255845


namespace sally_remaining_cards_l255_255820

variable (total_cards : ℕ) (torn_cards : ℕ) (bought_cards : ℕ)

def intact_cards (total_cards : ℕ) (torn_cards : ℕ) : ℕ := total_cards - torn_cards
def remaining_cards (intact_cards : ℕ) (bought_cards : ℕ) : ℕ := intact_cards - bought_cards

theorem sally_remaining_cards :
  intact_cards 39 9 - 24 = 6 :=
by
  -- sorry for proof
  sorry

end sally_remaining_cards_l255_255820


namespace pizza_slices_left_for_Phill_l255_255407

theorem pizza_slices_left_for_Phill :
  ∀ (initial_slices : ℕ) (first_cut : ℕ) (second_cut : ℕ) (third_cut : ℕ)
    (slices_given_to_3_friends : ℕ) (slices_given_to_2_friends : ℕ) (slices_left_for_Phill : ℕ),
    initial_slices = 1 →
    first_cut = 2 →
    second_cut = 4 →
    third_cut = 8 →
    slices_given_to_3_friends = 3 →
    slices_given_to_2_friends = 4 →
    slices_left_for_Phill = third_cut - (slices_given_to_3_friends + slices_given_to_2_friends) →
    slices_left_for_Phill = 1 :=
by {
  intros,
  subst_vars,
  simp, -- Simplify the boolean equalities
  -- We assume the steps are correct, so we leave it with sorry for now
  -- The proof should be easy for the given example and conditions.
  sorry,
}

end pizza_slices_left_for_Phill_l255_255407


namespace hapok_max_coins_l255_255005

/-- The maximum number of coins Hapok can guarantee himself regardless of Glazok's actions is 46 coins. -/
theorem hapok_max_coins (total_coins : ℕ) (max_handfuls : ℕ) (coins_per_handful : ℕ) :
  total_coins = 100 ∧ max_handfuls = 9 ∧ (∀ h : ℕ, h ≤ max_handfuls) ∧ coins_per_handful ≤ total_coins →
  ∃ k : ℕ, k ≤ total_coins ∧ k = 46 :=
by {
  sorry
}

end hapok_max_coins_l255_255005


namespace equation_of_perpendicular_line_l255_255148

theorem equation_of_perpendicular_line (x y c : ℝ) (h₁ : x = -1) (h₂ : y = 2)
  (h₃ : 2 * x - 3 * y = -c) (h₄ : 3 * x + 2 * y - 7 = 0) :
  2 * x - 3 * y + 8 = 0 :=
sorry

end equation_of_perpendicular_line_l255_255148


namespace domain_of_f_l255_255723

noncomputable def f (t : ℝ) : ℝ :=  1 / ((abs (t - 1))^2 + (abs (t + 1))^2)

theorem domain_of_f : ∀ t : ℝ, (abs (t - 1))^2 + (abs (t + 1))^2 ≠ 0 :=
by
  intro t
  sorry

end domain_of_f_l255_255723


namespace harry_geckos_count_l255_255239

theorem harry_geckos_count 
  (G : ℕ)
  (iguanas : ℕ := 2)
  (snakes : ℕ := 4)
  (cost_snake : ℕ := 10)
  (cost_iguana : ℕ := 5)
  (cost_gecko : ℕ := 15)
  (annual_cost : ℕ := 1140) :
  12 * (snakes * cost_snake + iguanas * cost_iguana + G * cost_gecko) = annual_cost → 
  G = 3 := 
by 
  intros h
  sorry

end harry_geckos_count_l255_255239


namespace primes_and_one_l255_255093

-- Given conditions:
variables {a n : ℕ}
variable (ha : a > 100 ∧ a % 2 = 1)  -- a is an odd natural number greater than 100
variable (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime (a - n^2) / 4)  -- for all n ≤ √(a / 5), (a - n^2) / 4 is prime

-- Theorem: For all n > √(a / 5), (a - n^2) / 4 is either prime or 1
theorem primes_and_one {a : ℕ} (ha : a > 100 ∧ a % 2 = 1)
  (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime ((a - n^2) / 4)) :
  ∀ n > Nat.sqrt (a / 5), Prime ((a - n^2) / 4) ∨ ((a - n^2) / 4) = 1 :=
sorry

end primes_and_one_l255_255093


namespace average_age_of_team_l255_255029

variable (A : ℕ)
variable (captain_age : ℕ)
variable (wicket_keeper_age : ℕ)
variable (vice_captain_age : ℕ)

-- Conditions
def team_size := 11
def captain := 25
def wicket_keeper := captain + 3
def vice_captain := wicket_keeper - 4
def remaining_players := team_size - 3
def remaining_average := A - 1

-- Prove the average age of the whole team
theorem average_age_of_team :
  captain_age = 25 ∧
  wicket_keeper_age = captain_age + 3 ∧
  vice_captain_age = wicket_keeper_age - 4 ∧
  11 * A = (captain + wicket_keeper + vice_captain) + 8 * (A - 1) → 
  A = 23 :=
by
  sorry

end average_age_of_team_l255_255029


namespace alice_commute_distance_l255_255753

noncomputable def office_distance_commute (commute_time_regular commute_time_holiday : ℝ) (speed_increase : ℝ) : ℝ := 
  let v := commute_time_regular * ((commute_time_regular + speed_increase) / commute_time_holiday - speed_increase)
  commute_time_regular * v

theorem alice_commute_distance : 
  office_distance_commute 0.5 0.3 12 = 9 := 
sorry

end alice_commute_distance_l255_255753


namespace arithmetic_mean_correct_l255_255657

noncomputable def arithmetic_mean (n : ℕ) (h : n > 1) : ℝ :=
  let one_minus_one_div_n := 1 - (1 / n : ℝ)
  let rest_ones := (n - 1 : ℕ) • 1
  let total_sum : ℝ := rest_ones + one_minus_one_div_n
  total_sum / n

theorem arithmetic_mean_correct (n : ℕ) (h : n > 1) :
  arithmetic_mean n h = 1 - (1 / (n * n : ℝ)) := sorry

end arithmetic_mean_correct_l255_255657


namespace possible_values_of_n_l255_255838

theorem possible_values_of_n :
  let a := 1500
  let max_r2 := 562499
  let total := max_r2
  let perfect_squares := (750 : Nat)
  total - perfect_squares = 561749 := by
    sorry

end possible_values_of_n_l255_255838


namespace solve_equation_l255_255141

theorem solve_equation (x : ℝ) (h : (3*x^2 + 2*x + 1) / (x - 1) = 3*x + 1) : x = -1/2 :=
sorry

end solve_equation_l255_255141


namespace arithmetic_sequence_initial_term_l255_255782

theorem arithmetic_sequence_initial_term (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h_seq : ∀ n, a (n + 1) = a n + d)
  (h_sum : ∀ n, S n = n * (a 1 + n * d / 2))
  (h_product : a 2 * a 3 = a 4 * a 5)
  (h_sum_9 : S 9 = 27)
  (h_d_nonzero : d ≠ 0) :
  a 1 = -5 :=
sorry

end arithmetic_sequence_initial_term_l255_255782


namespace incorrect_statement_C_l255_255353

noncomputable def f (x a b c : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

theorem incorrect_statement_C (a b c : ℝ) (x0 : ℝ) (h_local_min : ∀ y, f x0 a b c ≤ f y a b c) :
  ∃ z, z < x0 ∧ ¬ (f z a b c ≤ f (z + ε) a b c) := sorry

end incorrect_statement_C_l255_255353


namespace andrew_kept_stickers_l255_255884

theorem andrew_kept_stickers :
  ∃ (b d f e g h : ℕ), b = 2000 ∧ d = (5 * b) / 100 ∧ f = d + 120 ∧ e = (d + f) / 2 ∧ g = 80 ∧ h = (e + g) / 5 ∧ (b - (d + f + e + g + h) = 1392) :=
sorry

end andrew_kept_stickers_l255_255884


namespace locus_of_tangent_circle_is_hyperbola_l255_255589

theorem locus_of_tangent_circle_is_hyperbola :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    (P.1 ^ 2 + P.2 ^ 2).sqrt = 1 + r ∧ ((P.1 - 4) ^ 2 + P.2 ^ 2).sqrt = 2 + r →
    ∃ (a b : ℝ), (P.1 - a) ^ 2 / b ^ 2 - (P.2 / a) ^ 2 / b ^ 2 = 1 :=
sorry

end locus_of_tangent_circle_is_hyperbola_l255_255589


namespace inequality_proof_l255_255488

theorem inequality_proof (x : ℝ) (h₁ : 3/2 ≤ x) (h₂ : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end inequality_proof_l255_255488


namespace uncolored_vertex_not_original_hexagon_vertex_l255_255647

theorem uncolored_vertex_not_original_hexagon_vertex
    (point_index : ℕ)
    (orig_hex_vertices : Finset ℕ) -- Assuming the vertices of the original hexagon are represented as a finite set of indices.
    (num_parts : ℕ := 1000) -- Each hexagon side is divided into 1000 parts
    (label : ℕ → Fin 3) -- A function labeling each point with 0, 1, or 2.
    (is_valid_labeling : ∀ (i j k : ℕ), label i ≠ label j ∧ label j ≠ label k ∧ label k ≠ label i) -- No duplicate labeling within a triangle.
    (is_single_uncolored : ∀ (p : ℕ), (p ∈ orig_hex_vertices ∨ ∃ (v : ℕ), v ∈ orig_hex_vertices ∧ p = v) → p ≠ point_index) -- Only one uncolored point
    : point_index ∉ orig_hex_vertices :=
by sorry

end uncolored_vertex_not_original_hexagon_vertex_l255_255647


namespace size_of_first_type_package_is_5_l255_255619

noncomputable def size_of_first_type_package (total_coffee : ℕ) (num_first_type : ℕ) (num_second_type : ℕ) (size_second_type : ℕ) : ℕ :=
  (total_coffee - num_second_type * size_second_type) / num_first_type

theorem size_of_first_type_package_is_5 :
  size_of_first_type_package 70 (4 + 2) 4 10 = 5 :=
by
  sorry

end size_of_first_type_package_is_5_l255_255619


namespace trig_identity_cos2theta_tan_minus_pi_over_4_l255_255087

variable (θ : ℝ)

-- Given condition
def tan_theta_is_2 : Prop := Real.tan θ = 2

-- Proof problem 1: Prove that cos(2θ) = -3/5
def cos2theta (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.cos (2 * θ) = -3 / 5

-- Proof problem 2: Prove that tan(θ - π/4) = 1/3
def tan_theta_minus_pi_over_4 (θ : ℝ) (h : tan_theta_is_2 θ) : Prop :=
  Real.tan (θ - Real.pi / 4) = 1 / 3

-- Main theorem statement
theorem trig_identity_cos2theta_tan_minus_pi_over_4 
  (θ : ℝ) (h : tan_theta_is_2 θ) :
  cos2theta θ h ∧ tan_theta_minus_pi_over_4 θ h :=
sorry

end trig_identity_cos2theta_tan_minus_pi_over_4_l255_255087


namespace DeansCalculatorGame_l255_255938

theorem DeansCalculatorGame (r : ℕ) (c1 c2 c3 : ℤ) (h1 : r = 45) (h2 : c1 = 1) (h3 : c2 = 0) (h4 : c3 = -2) : 
  let final1 := (c1 ^ 3)
  let final2 := (c2 ^ 2)
  let final3 := (-c3)^45
  final1 + final2 + final3 = 3 := 
by
  sorry

end DeansCalculatorGame_l255_255938


namespace Michelle_bought_14_chocolate_bars_l255_255273

-- Definitions for conditions
def sugar_per_chocolate_bar : ℕ := 10
def sugar_in_lollipop : ℕ := 37
def total_sugar_in_candy : ℕ := 177

-- Theorem to prove
theorem Michelle_bought_14_chocolate_bars :
  (total_sugar_in_candy - sugar_in_lollipop) / sugar_per_chocolate_bar = 14 :=
by
  -- Proof steps will go here, but are omitted as per the requirements.
  sorry

end Michelle_bought_14_chocolate_bars_l255_255273


namespace smallest_n_mod_equality_l255_255336

theorem smallest_n_mod_equality :
  ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 5]) ∧ (∀ m : ℕ, 0 < m ∧ m < n → (7^m ≡ m^7 [MOD 5]) = false) :=
  sorry

end smallest_n_mod_equality_l255_255336


namespace product_of_primes_sum_85_l255_255030

open Nat

theorem product_of_primes_sum_85 :
  ∃ (p q : ℕ), p.Prime ∧ q.Prime ∧ p + q = 85 ∧ p * q = 166 :=
sorry

end product_of_primes_sum_85_l255_255030


namespace sky_color_changes_l255_255070

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l255_255070


namespace sky_colors_l255_255072

theorem sky_colors (h1 : ∀ t : ℕ, t = 2) (h2 : ∀ m : ℕ, m = 60) (h3 : ∀ c : ℕ, c = 10) : 
  ∃ n : ℕ, n = 12 :=
by
  let total_duration := (2 * 60 : ℕ)
  let num_colors := total_duration / 10
  have : num_colors = 12 := by decide
  use num_colors
  assumption_needed

end sky_colors_l255_255072


namespace additional_payment_each_friend_l255_255745

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end additional_payment_each_friend_l255_255745


namespace probability_of_40_cents_l255_255824

noncomputable def num_successful_outcomes : ℕ := 16 + 3

def total_outcomes : ℕ := 2 ^ 5

def probability_success : ℚ := num_successful_outcomes / total_outcomes

theorem probability_of_40_cents : probability_success = 19 / 32 := by
  unfold probability_success num_successful_outcomes total_outcomes
  norm_num
  sorry

end probability_of_40_cents_l255_255824


namespace find_a_l255_255108

theorem find_a (a b d : ℕ) (h1 : a + b = d) (h2 : b + d = 7) (h3 : d = 4) : a = 1 := by
  sorry

end find_a_l255_255108


namespace factorize_cubic_expression_l255_255208

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l255_255208


namespace prob_even_xyz_expr_l255_255989

open Finset

-- Define the set from which numbers are chosen
def S := range 12 |>.map succ

-- Define a predicate for a multiset forming a valid choice
def valid_choice (x y z : ℕ) : Prop := x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z

-- Define the condition for the expression (x-1)(y-1)(z-1) to be even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probability calculation
def P_even : ℚ := 1 - (binomial 6 3 : ℚ) / (binomial 12 3 : ℚ)

-- The theorem to be proven
theorem prob_even_xyz_expr :
  (∃ x y z : ℕ, valid_choice x y z ∧ is_even ((x - 1) * (y - 1) * (z - 1))) →
  p_even = (10 / 11 : ℚ) :=
by
  sorry

end prob_even_xyz_expr_l255_255989


namespace complement_of_union_l255_255509

-- Define the universal set U, set M, and set N as given:
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {4, 5}

-- Define the complement of a set relative to the universal set U
def complement_U (A : Set ℕ) : Set ℕ := { x | x ∈ U ∧ x ∉ A }

-- Prove that the complement of M ∪ N with respect to U is {1, 6}
theorem complement_of_union : complement_U (M ∪ N) = {1, 6} :=
  sorry -- proof goes here

end complement_of_union_l255_255509


namespace tromino_covering_l255_255487

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

def chessboard_black_squares (n : ℕ) : ℕ := (n^2 + 1) / 2

def minimum_trominos (n : ℕ) : ℕ := (n^2 + 1) / 6

theorem tromino_covering (n : ℕ) (h_odd : is_odd n) (h_ge7 : n ≥ 7) :
  ∃ k : ℕ, chessboard_black_squares n = 3 * k ∧ (k = minimum_trominos n) :=
sorry

end tromino_covering_l255_255487


namespace similar_segments_areas_proportional_to_chords_squares_l255_255410

variables {k k₁ Δ Δ₁ r r₁ a a₁ S S₁ : ℝ}

-- Conditions given in the problem
def similar_segments (r r₁ a a₁ Δ Δ₁ k k₁ : ℝ) :=
  (Δ / Δ₁ = (a^2 / a₁^2) ∧ (Δ / Δ₁ = r^2 / r₁^2)) ∧ (k / k₁ = r^2 / r₁^2)

-- Given the areas of the segments in terms of sectors and triangles
def area_of_segment (k Δ : ℝ) := k - Δ

-- Theorem statement proving the desired relationship
theorem similar_segments_areas_proportional_to_chords_squares
  (h : similar_segments r r₁ a a₁ Δ Δ₁ k k₁) :
  (S = area_of_segment k Δ) → (S₁ = area_of_segment k₁ Δ₁) → (S / S₁ = a^2 / a₁^2) :=
by
  sorry

end similar_segments_areas_proportional_to_chords_squares_l255_255410


namespace height_table_l255_255996

variable (l w h : ℝ)

theorem height_table (h_eq1 : l + h - w = 32) (h_eq2 : w + h - l = 28) : h = 30 := by
  sorry

end height_table_l255_255996


namespace fraction_identity_l255_255064

variable (a b : ℝ)

theorem fraction_identity (h : a ≠ 0) : 
  (2 * b + a) / a + (a - 2 * b) / a = 2 := 
by
  sorry

end fraction_identity_l255_255064


namespace scientific_notation_of_0point0000025_l255_255809

theorem scientific_notation_of_0point0000025 : ∃ (a : ℝ) (n : ℤ), 0.0000025 = a * 10 ^ n ∧ a = 2.5 ∧ n = -6 :=
by {
  sorry
}

end scientific_notation_of_0point0000025_l255_255809


namespace false_propositions_l255_255729

open Classical

theorem false_propositions :
  ¬ (∀ x : ℝ, x^2 + 3 < 0) ∧ ¬ (∀ x : ℕ, x^2 > 1) ∧ (∃ x : ℤ, x^5 < 1) ∧ ¬ (∃ x : ℚ, x^2 = 3) :=
by
  sorry

end false_propositions_l255_255729


namespace nate_search_time_l255_255402

theorem nate_search_time
  (rowsG : Nat) (cars_per_rowG : Nat)
  (rowsH : Nat) (cars_per_rowH : Nat)
  (rowsI : Nat) (cars_per_rowI : Nat)
  (walk_speed : Nat) : Nat :=
  let total_cars : Nat := rowsG * cars_per_rowG + rowsH * cars_per_rowH + rowsI * cars_per_rowI
  let total_minutes : Nat := total_cars / walk_speed
  if total_cars % walk_speed == 0 then total_minutes else total_minutes + 1

/-- Given:
- rows in Section G = 18, cars per row in Section G = 12
- rows in Section H = 25, cars per row in Section H = 10
- rows in Section I = 17, cars per row in Section I = 11
- Nate's walking speed is 8 cars per minute
Prove: Nate took 82 minutes to search the parking lot
-/
example : nate_search_time 18 12 25 10 17 11 8 = 82 := by
  sorry

end nate_search_time_l255_255402


namespace fisherman_daily_earnings_l255_255420

def red_snapper_quantity : Nat := 8
def tuna_quantity : Nat := 14
def red_snapper_cost : Nat := 3
def tuna_cost : Nat := 2

theorem fisherman_daily_earnings
  (rs_qty : Nat := red_snapper_quantity)
  (t_qty : Nat := tuna_quantity)
  (rs_cost : Nat := red_snapper_cost)
  (t_cost : Nat := tuna_cost) :
  rs_qty * rs_cost + t_qty * t_cost = 52 := 
by {
  sorry
}

end fisherman_daily_earnings_l255_255420


namespace equilateral_triangle_perimeter_l255_255965

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l255_255965


namespace price_of_item_is_27_l255_255983

theorem price_of_item_is_27 (P : ℤ) (A_money B_money : ℤ) (max_A_items max_B_items : ℤ) (total_item_diff : ℤ) :
  (26 ≤ P ∧ P ≤ 33) ∧
  A_money = 200 ∧
  B_money = 400 ∧
  max_A_items = 7 ∧
  max_B_items = 14 ∧
  total_item_diff = 1 ∧
  (P ≤ A_money / max_A_items) ∧ 
  (P ≤ B_money / max_B_items) ∧ 
  ((A_money + B_money) / P = max_A_items + max_B_items + total_item_diff) → 
  P = 27 := by
  sorry

end price_of_item_is_27_l255_255983


namespace find_n_l255_255646

theorem find_n (n : ℕ) (hn_pos : 0 < n) (hn_greater_30 : 30 < n) 
  (divides : (4 * n - 1) ∣ 2002 * n) : n = 36 := 
by
  sorry

end find_n_l255_255646


namespace problem_statement_l255_255737

theorem problem_statement : (-1:ℤ) ^ 4 - (2 - (-3:ℤ) ^ 2) = 6 := by
  sorry  -- Proof will be provided separately

end problem_statement_l255_255737


namespace average_marks_l255_255625

-- Conditions
def marks_english : ℕ := 73
def marks_mathematics : ℕ := 69
def marks_physics : ℕ := 92
def marks_chemistry : ℕ := 64
def marks_biology : ℕ := 82
def number_of_subjects : ℕ := 5

-- Problem Statement
theorem average_marks :
  (marks_english + marks_mathematics + marks_physics + marks_chemistry + marks_biology) / number_of_subjects = 76 :=
by
  sorry

end average_marks_l255_255625


namespace cost_formula_correct_l255_255039

def total_cost (P : ℕ) : ℕ :=
  if P ≤ 2 then 15 else 15 + 5 * (P - 2)

theorem cost_formula_correct (P : ℕ) : 
  total_cost P = (if P ≤ 2 then 15 else 15 + 5 * (P - 2)) :=
by 
  exact rfl

end cost_formula_correct_l255_255039


namespace eval_f_at_800_l255_255271

-- Given conditions in Lean 4:
def f : ℝ → ℝ := sorry -- placeholder for the function definition
axiom func_eqn (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : f (x * y) = f x / y
axiom f_at_1000 : f 1000 = 4

-- The goal/proof statement:
theorem eval_f_at_800 : f 800 = 5 := sorry

end eval_f_at_800_l255_255271


namespace range_of_m_satisfying_obtuse_triangle_l255_255227

theorem range_of_m_satisfying_obtuse_triangle (m : ℝ) 
(h_triangle: m > 0 
  → m + (m + 1) > (m + 2) 
  ∧ m + (m + 2) > (m + 1) 
  ∧ (m + 1) + (m + 2) > m
  ∧ (m + 2) ^ 2 > m ^ 2 + (m + 1) ^ 2) : 1 < m ∧ m < 1.5 :=
by
  sorry

end range_of_m_satisfying_obtuse_triangle_l255_255227


namespace min_value_geom_seq_l255_255347

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
  0 < a 4 ∧ 0 < a 14 ∧ a 4 * a 14 = 8 ∧ 0 < a 7 ∧ 0 < a 11 ∧ a 7 * a 11 = 8

theorem min_value_geom_seq {a : ℕ → ℝ} (h : geom_seq a) :
  2 * a 7 + a 11 = 8 :=
by
  sorry

end min_value_geom_seq_l255_255347


namespace number_of_students_above_120_l255_255367

def math_scores_distribution : ℝ → ℝ := sorry  -- Define the distribution function according to N(110, 10^2)

def students_above_120 (n : ℕ) : ℕ :=
  let total_students : ℕ := 50 in
  sorry -- Implement the calculation logic here

theorem number_of_students_above_120 :
  let prob : ℝ := 0.34 in
  let total_students : ℕ := 50 in
  students_above_120 total_students = 8 :=
sorry

end number_of_students_above_120_l255_255367


namespace payment_for_work_l255_255452

theorem payment_for_work (rate_A rate_B : ℚ) (days_A days_B days_combined : ℕ) (payment_C : ℚ) :
  rate_A = 1 / days_A →
  rate_B = 1 / days_B →
  days_combined = 3 →
  payment_C = 400.0000000000002 →
  let rate_combined := rate_A + rate_B in
  let rate_ABC := 1 / days_combined in
  let rate_C := rate_ABC - rate_combined in
  let work_done_by_C := days_combined * rate_C in
  let total_work := 1 in
  let total_payment := (1 / work_done_by_C) * payment_C in
  total_payment = 3200 :=
by {
  sorry
}

end payment_for_work_l255_255452


namespace opposite_of_7_l255_255980

-- Define the concept of an opposite number for real numbers
def is_opposite (x y : ℝ) : Prop := x = -y

-- Theorem statement
theorem opposite_of_7 :
  is_opposite 7 (-7) :=
by {
  sorry
}

end opposite_of_7_l255_255980


namespace problem_statement_l255_255090

def f (x : ℝ) : ℝ := sorry

theorem problem_statement
  (cond1 : ∀ {x y w : ℝ}, x > y → f x + x ≥ w → w ≥ f y + y → ∃ (z : ℝ), z ∈ Set.Icc y x ∧ f z = w - z)
  (cond2 : ∃ (u : ℝ), 0 ∈ Set.range f ∧ ∀ a ∈ Set.range f, u ≤ a)
  (cond3 : f 0 = 1)
  (cond4 : f (-2003) ≤ 2004)
  (cond5 : ∀ x y : ℝ, f x * f y = f (x * f y + y * f x + x * y)) :
  f (-2003) = 2004 := sorry

end problem_statement_l255_255090


namespace question1_question2_l255_255922

noncomputable def f (x m : ℝ) : ℝ := (x^2 + 3) / (x - m)

theorem question1 (m : ℝ) : (∀ x : ℝ, x > m → f x m + m ≥ 0) ↔ m ∈ Set.Ici (- (2 * Real.sqrt 15) / 5) := sorry

theorem question2 (m : ℝ) : (∃ x : ℝ, x > m ∧ f x m = 6) ↔ m = 1 := sorry

end question1_question2_l255_255922


namespace relationship_between_x_t_G_D_and_x_l255_255539

-- Definitions
variables {G D : ℝ → ℝ}
variables {t : ℝ}
noncomputable def number_of_boys (x : ℝ) : ℝ := 9000 / x
noncomputable def total_population (x : ℝ) (x_t : ℝ) : Prop := x_t = 15000 / x

-- The proof problem
theorem relationship_between_x_t_G_D_and_x
  (G D : ℝ → ℝ)
  (x : ℝ) (t : ℝ) (x_t : ℝ)
  (h1 : 90 = x / 100 * number_of_boys x)
  (h2 : 0.60 * x_t = number_of_boys x)
  (h3 : 0.40 * x_t > 0)
  (h4 : true) :       -- Placeholder for some condition not used directly
  total_population x x_t :=
by
  -- Proof would go here
  sorry

end relationship_between_x_t_G_D_and_x_l255_255539


namespace total_tiles_l255_255193

theorem total_tiles (s : ℕ) (h1 : true) (h2 : true) (h3 : true) (h4 : true) (h5 : 4 * s - 4 = 100): s * s = 676 :=
by
  sorry

end total_tiles_l255_255193


namespace emily_sixth_score_l255_255765

theorem emily_sixth_score:
  ∀ (s₁ s₂ s₃ s₄ s₅ sᵣ : ℕ),
  s₁ = 88 →
  s₂ = 90 →
  s₃ = 85 →
  s₄ = 92 →
  s₅ = 97 →
  (s₁ + s₂ + s₃ + s₄ + s₅ + sᵣ) / 6 = 91 →
  sᵣ = 94 :=
by intros s₁ s₂ s₃ s₄ s₅ sᵣ h₁ h₂ h₃ h₄ h₅ h₆;
   rw [h₁, h₂, h₃, h₄, h₅] at h₆;
   sorry

end emily_sixth_score_l255_255765


namespace min_value_of_reciprocal_sum_l255_255348

variable (m n : ℝ)

theorem min_value_of_reciprocal_sum (hmn : m * n > 0) (h_line : m + n = 2) :
  (1 / m + 1 / n = 2) :=
sorry

end min_value_of_reciprocal_sum_l255_255348


namespace pizza_slices_left_l255_255408

theorem pizza_slices_left (initial_slices cuts friends total_given : ℕ) (h_slices: initial_slices = 1 * 2^cuts)
  (h_first_group: ∀ f ∈ friends, f = 2) (h_second_group: ∀ f ∉ friends, f = 1) 
  (h_friends: ∑ f in friends, f = total_given) 
  (distribution: total_given = (card friends * 2 + (card (range 3) - card friends) * 1)) 
  (h_card_friends: card friends = 2) : 
  (initial_slices - total_given = 1) :=
sorry

end pizza_slices_left_l255_255408


namespace find_second_bag_weight_l255_255476

variable (initialWeight : ℕ) (firstBagWeight : ℕ) (totalWeight : ℕ)

theorem find_second_bag_weight 
  (h1: initialWeight = 15)
  (h2: firstBagWeight = 15)
  (h3: totalWeight = 40) :
  totalWeight - (initialWeight + firstBagWeight) = 10 :=
  sorry

end find_second_bag_weight_l255_255476


namespace average_weight_of_all_boys_l255_255286

theorem average_weight_of_all_boys 
  (n₁ n₂ : ℕ) (w₁ w₂ : ℝ) 
  (h₁ : n₁ = 20) (h₂ : w₁ = 50.25) 
  (h₃ : n₂ = 8) (h₄ : w₂ = 45.15) :
  (n₁ * w₁ + n₂ * w₂) / (n₁ + n₂) = 48.79 := 
by
  sorry

end average_weight_of_all_boys_l255_255286


namespace find_m_plus_n_l255_255783

theorem find_m_plus_n (x : ℝ) (m n : ℕ) (h₁ : (1 + Real.sin x) / (Real.cos x) = 22 / 7) 
                      (h₂ : (1 + Real.cos x) / (Real.sin x) = m / n) :
                      m + n = 44 := by
  sorry

end find_m_plus_n_l255_255783


namespace find_x_squared_plus_y_squared_l255_255241

theorem find_x_squared_plus_y_squared (x y : ℝ) (h₁ : x * y = -8) (h₂ : x^2 * y + x * y^2 + 3 * x + 3 * y = 100) : x^2 + y^2 = 416 :=
sorry

end find_x_squared_plus_y_squared_l255_255241


namespace gold_coins_percentage_l255_255756

-- Definitions for conditions
def percent_beads : Float := 0.30
def percent_sculptures : Float := 0.10
def percent_silver_coins : Float := 0.30

-- Definitions derived from conditions
def percent_coins : Float := 1.0 - percent_beads - percent_sculptures
def percent_gold_coins_among_coins : Float := 1.0 - percent_silver_coins

-- Theorem statement
theorem gold_coins_percentage : percent_gold_coins_among_coins * percent_coins = 0.42 :=
by
sorry

end gold_coins_percentage_l255_255756


namespace rosie_pie_count_l255_255819

def total_apples : ℕ := 40
def initial_apples_required : ℕ := 3
def apples_per_pie : ℕ := 5

theorem rosie_pie_count : (total_apples - initial_apples_required) / apples_per_pie = 7 :=
by
  sorry

end rosie_pie_count_l255_255819


namespace same_type_as_target_l255_255578

-- Definitions of the polynomials
def optionA (a b : ℝ) : ℝ := a^2 * b
def optionB (a b : ℝ) : ℝ := -2 * a * b^2
def optionC (a b : ℝ) : ℝ := a * b
def optionD (a b c : ℝ) : ℝ := a * b^2 * c

-- Definition of the target polynomial type
def target (a b : ℝ) : ℝ := a * b^2

-- Statement: Option B is of the same type as target
theorem same_type_as_target (a b : ℝ) : optionB a b = -2 * target a b := 
sorry

end same_type_as_target_l255_255578


namespace find_angle_B_l255_255674

-- Conditions
variable (A B C a b : ℝ)
variable (h1 : a = Real.sqrt 6)
variable (h2 : b = Real.sqrt 3)
variable (h3 : b + a * (Real.sin C - Real.cos C) = 0)

-- Target
theorem find_angle_B : B = Real.pi / 6 :=
sorry

end find_angle_B_l255_255674


namespace triple_hash_100_l255_255627

def hash (N : ℝ) : ℝ :=
  0.5 * N + N

theorem triple_hash_100 : hash (hash (hash 100)) = 337.5 :=
by
  sorry

end triple_hash_100_l255_255627


namespace diet_soda_bottles_l255_255041

theorem diet_soda_bottles (r d l t : Nat) (h1 : r = 49) (h2 : l = 6) (h3 : t = 89) (h4 : t = r + d) : d = 40 :=
by
  sorry

end diet_soda_bottles_l255_255041


namespace equilateral_triangle_perimeter_l255_255972

theorem equilateral_triangle_perimeter (s : ℝ) 
  (h : (s^2 * Real.sqrt 3) / 4 = 2 * s) : 
  3 * s = 8 * Real.sqrt 3 :=
by
  -- Proof steps (omitted)
  sorry

end equilateral_triangle_perimeter_l255_255972


namespace Thursday_total_rainfall_correct_l255_255676

def Monday_rainfall : ℝ := 0.9
def Tuesday_rainfall : ℝ := Monday_rainfall - 0.7
def Wednesday_rainfall : ℝ := Tuesday_rainfall + 0.5 * Tuesday_rainfall
def additional_rain : ℝ := 0.3
def decrease_factor : ℝ := 0.2
def Thursday_rainfall_before_addition : ℝ := Wednesday_rainfall - decrease_factor * Wednesday_rainfall
def Thursday_total_rainfall : ℝ := Thursday_rainfall_before_addition + additional_rain

theorem Thursday_total_rainfall_correct :
  Thursday_total_rainfall = 0.54 :=
by
  sorry

end Thursday_total_rainfall_correct_l255_255676


namespace determine_k_for_quadratic_eq_l255_255762

theorem determine_k_for_quadratic_eq {k : ℝ} :
  (∀ r s : ℝ, 3 * r^2 + 5 * r + k = 0 ∧ 3 * s^2 + 5 * s + k = 0 →
    (|r + s| = r^2 + s^2)) ↔ k = -10/3 := by
sorry

end determine_k_for_quadratic_eq_l255_255762


namespace max_light_window_l255_255023

noncomputable def max_window_light : Prop :=
  ∃ (x : ℝ), (4 - 2 * x) / 3 * x = -2 / 3 * (x - 1) ^ 2 + 2 / 3 ∧ x = 1 ∧ (4 - 2 * x) / 3 = 2 / 3

theorem max_light_window : max_window_light :=
by
  sorry

end max_light_window_l255_255023


namespace trisha_total_distance_l255_255475

-- Define each segment of Trisha's walk in miles
def hotel_to_postcard : ℝ := 0.1111111111111111
def postcard_to_tshirt : ℝ := 0.2222222222222222
def tshirt_to_keychain : ℝ := 0.7777777777777778
def keychain_to_toy : ℝ := 0.5555555555555556
def meters_to_miles (m : ℝ) : ℝ := m * 0.000621371
def toy_to_bookstore : ℝ := meters_to_miles 400
def bookstore_to_hotel : ℝ := 0.6666666666666666

-- Sum of all distances
def total_distance : ℝ :=
  hotel_to_postcard +
  postcard_to_tshirt +
  tshirt_to_keychain +
  keychain_to_toy +
  toy_to_bookstore +
  bookstore_to_hotel

-- Proof statement
theorem trisha_total_distance : total_distance = 1.5818817333333333 := by
  sorry

end trisha_total_distance_l255_255475


namespace complete_square_transform_l255_255168

theorem complete_square_transform (x : ℝ) (h : x^2 + 8*x + 7 = 0) : (x + 4)^2 = 9 :=
by sorry

end complete_square_transform_l255_255168


namespace average_running_minutes_l255_255527

theorem average_running_minutes
  (fifth_minutes : ℕ)
  (sixth_minutes : ℕ)
  (seventh_minutes : ℕ)
  (num_fifth : ℕ)
  (num_sixth : ℕ)
  (num_seventh : ℕ)
  (fifth_minutes_condition : fifth_minutes = 8)
  (sixth_minutes_condition : sixth_minutes = 18)
  (seventh_minutes_condition : seventh_minutes = 16)
  (num_sixth_condition : num_sixth = 3 * num_fifth)
  (num_seventh_condition : num_seventh = num_fifth) :
  (fifth_minutes * num_fifth + sixth_minutes * num_sixth + seventh_minutes * num_seventh)
   / (num_fifth + num_sixth + num_seventh) = 78 / 5 := by
sorry

end average_running_minutes_l255_255527


namespace jane_spent_75_days_reading_l255_255129

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l255_255129


namespace slope_of_parallel_line_l255_255469

theorem slope_of_parallel_line (x y : ℝ) :
  (∃ (b : ℝ), 3 * x - 6 * y = 12) → ∀ (m₁ x₁ y₁ x₂ y₂ : ℝ), (y₁ = (1/2) * x₁ + b) ∧ (y₂ = (1/2) * x₂ + b) → (x₁ ≠ x₂) → m₁ = 1/2 :=
by 
  sorry

end slope_of_parallel_line_l255_255469


namespace john_spent_half_on_fruits_and_vegetables_l255_255326

theorem john_spent_half_on_fruits_and_vegetables (M : ℝ) (F : ℝ) 
  (spent_on_meat : ℝ) (spent_on_bakery : ℝ) (spent_on_candy : ℝ) :
  (M = 120) → 
  (spent_on_meat = (1 / 3) * M) → 
  (spent_on_bakery = (1 / 10) * M) → 
  (spent_on_candy = 8) → 
  (F * M + spent_on_meat + spent_on_bakery + spent_on_candy = M) → 
  (F = 1 / 2) := 
  by 
    sorry

end john_spent_half_on_fruits_and_vegetables_l255_255326


namespace remainder_sum_abc_mod8_l255_255795

theorem remainder_sum_abc_mod8 (a b c : ℕ) (h₁ : 1 ≤ a ∧ a < 8) (h₂ : 1 ≤ b ∧ b < 8) (h₃ : 1 ≤ c ∧ c < 8) 
  (h₄ : a * b * c ≡ 1 [MOD 8]) (h₅ : 4 * b * c ≡ 3 [MOD 8]) (h₆ : 5 * b ≡ 3 + b [MOD 8]) :
  (a + b + c) % 8 = 2 := 
sorry

end remainder_sum_abc_mod8_l255_255795


namespace find_b_if_lines_parallel_l255_255472

theorem find_b_if_lines_parallel (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
  (∀ x y : ℝ, y + 2 = (b + 9) * x → y = (b + 9) * x - 2) →
  3 = b + 9 →
  b = -6 :=
by {
  sorry
}

end find_b_if_lines_parallel_l255_255472


namespace total_days_spent_l255_255383

theorem total_days_spent {weeks_to_days : ℕ → ℕ} : 
  (weeks_to_days 3 + weeks_to_days 1) + 
  (weeks_to_days (weeks_to_days 3 + weeks_to_days 2) + 3) + 
  (2 * (weeks_to_days (weeks_to_days 3 + weeks_to_days 2))) + 
  (weeks_to_days 5 - weeks_to_days 1) + 
  (weeks_to_days ((weeks_to_days 5 - weeks_to_days 1) - weeks_to_days 3) + 6) + 
  (weeks_to_days (weeks_to_days 5 - weeks_to_days 1) + 4) = 230 :=
by
  sorry

end total_days_spent_l255_255383


namespace combined_stickers_leftover_l255_255550

theorem combined_stickers_leftover (r p g : ℕ) (h_r : r % 5 = 1) (h_p : p % 5 = 4) (h_g : g % 5 = 3) :
  (r + p + g) % 5 = 3 :=
by
  sorry

end combined_stickers_leftover_l255_255550


namespace total_vessels_proof_l255_255180

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l255_255180


namespace shadow_problem_l255_255883

-- Define the conditions
def cube_edge_length : ℝ := 2
def shadow_area_outside : ℝ := 147
def total_shadow_area : ℝ := shadow_area_outside + cube_edge_length^2

-- The main statement to prove
theorem shadow_problem :
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  (⌊1000 * x⌋ : ℤ) = 481 :=
by
  let x := 4 / (Real.sqrt total_shadow_area - cube_edge_length)
  have h : (⌊1000 * x⌋ : ℤ) = 481 := sorry
  exact h

end shadow_problem_l255_255883


namespace part1_inequality_l255_255778

noncomputable def f (x : ℝ) : ℝ := x - 2
noncomputable def g (x m : ℝ) : ℝ := x^2 - 2 * m * x + 4

theorem part1_inequality (m : ℝ) : (∀ x : ℝ, g x m > f x) ↔ (m ∈ Set.Ioo (-Real.sqrt 6 - (1/2)) (Real.sqrt 6 - (1/2))) :=
sorry

end part1_inequality_l255_255778


namespace sum_of_diagonals_l255_255591

noncomputable def length_AB : ℝ := 31
noncomputable def length_sides : ℝ := 81

def hexagon_inscribed_in_circle (A B C D E F : Type) : Prop :=
-- Assuming A, B, C, D, E, F are suitable points on a circle
-- Definitions to be added as per detailed proof needs
sorry

theorem sum_of_diagonals (A B C D E F : Type) :
    hexagon_inscribed_in_circle A B C D E F →
    (length_AB + length_sides + length_sides + length_sides + length_sides + length_sides = 384) := 
by
  sorry

end sum_of_diagonals_l255_255591


namespace total_vessels_proof_l255_255181

def cruise_ships : Nat := 4
def cargo_ships : Nat := cruise_ships * 2
def sailboats : Nat := cargo_ships + 6
def fishing_boats : Nat := sailboats / 7
def total_vessels : Nat := cruise_ships + cargo_ships + sailboats + fishing_boats

theorem total_vessels_proof : total_vessels = 28 := by
  sorry

end total_vessels_proof_l255_255181


namespace equilateral_triangle_perimeter_l255_255969

theorem equilateral_triangle_perimeter (s : ℝ) (h : (s^2 * Real.sqrt 3 / 4) = 2 * s) : 3 * s = 8 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_perimeter_l255_255969


namespace koala_fiber_l255_255266

theorem koala_fiber (absorption_percent: ℝ) (absorbed_fiber: ℝ) (total_fiber: ℝ) 
  (h1: absorption_percent = 0.25) 
  (h2: absorbed_fiber = 10.5) 
  (h3: absorbed_fiber = absorption_percent * total_fiber) : 
  total_fiber = 42 :=
by
  rw [h1, h2] at h3
  have h : 10.5 = 0.25 * total_fiber := h3
  sorry

end koala_fiber_l255_255266


namespace number_of_dimes_paid_l255_255405

theorem number_of_dimes_paid (cost_in_dollars : ℕ) (value_of_dime_in_cents : ℕ) (value_of_dollar_in_cents : ℕ) 
  (h_cost : cost_in_dollars = 9) (h_dime : value_of_dime_in_cents = 10) (h_dollar : value_of_dollar_in_cents = 100) : 
  (cost_in_dollars * value_of_dollar_in_cents) / value_of_dime_in_cents = 90 := by
  -- Proof to be provided here
  sorry

end number_of_dimes_paid_l255_255405


namespace acquaintances_unique_l255_255586

theorem acquaintances_unique (N : ℕ) : ∃ acquaintances : ℕ → ℕ, 
  (∀ i j k : ℕ, i < N → j < N → k < N → i ≠ j → j ≠ k → i ≠ k → 
    acquaintances i ≠ acquaintances j ∨ acquaintances j ≠ acquaintances k ∨ acquaintances i ≠ acquaintances k) :=
sorry

end acquaintances_unique_l255_255586


namespace imaginary_part_of_z_l255_255652

noncomputable def i : ℂ := Complex.I
noncomputable def z : ℂ := i / (i - 1)

theorem imaginary_part_of_z : z.im = -1 / 2 := by
  sorry

end imaginary_part_of_z_l255_255652


namespace total_batteries_correct_l255_255439

-- Definitions of the number of batteries used in each category
def batteries_flashlight : ℕ := 2
def batteries_toys : ℕ := 15
def batteries_controllers : ℕ := 2

-- The total number of batteries used by Tom
def total_batteries : ℕ := batteries_flashlight + batteries_toys + batteries_controllers

-- The proof statement that needs to be proven
theorem total_batteries_correct : total_batteries = 19 := by
  sorry

end total_batteries_correct_l255_255439


namespace find_total_tennis_balls_l255_255602

noncomputable def original_white_balls : ℕ := sorry
noncomputable def original_yellow_balls : ℕ := sorry
noncomputable def dispatched_yellow_balls : ℕ := original_yellow_balls + 20

theorem find_total_tennis_balls
  (white_balls_eq : original_white_balls = original_yellow_balls)
  (ratio_eq : original_white_balls / dispatched_yellow_balls = 8 / 13) :
  original_white_balls + original_yellow_balls = 64 := sorry

end find_total_tennis_balls_l255_255602


namespace intersection_of_circles_l255_255377

theorem intersection_of_circles (k : ℝ) :
  (∃ z : ℂ, (|z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ (k = 2 ∨ k = 14)) :=
by
  sorry

end intersection_of_circles_l255_255377


namespace total_vessels_l255_255178

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end total_vessels_l255_255178


namespace larger_root_of_quadratic_eq_l255_255080

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l255_255080


namespace smallest_norm_l255_255389

noncomputable def vectorNorm (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem smallest_norm (v : ℝ × ℝ)
  (h : vectorNorm (v.1 + 4, v.2 + 2) = 10) :
  vectorNorm v >= 10 - 2 * Real.sqrt 5 :=
by
  sorry

end smallest_norm_l255_255389


namespace part_a_part_b_part_c_part_d_l255_255411

theorem part_a : (4237 * 27925 ≠ 118275855) :=
by sorry

theorem part_b : (42971064 / 8264 ≠ 5201) :=
by sorry

theorem part_c : (1965^2 ≠ 3761225) :=
by sorry

theorem part_d : (23 ^ 5 ≠ 371293) :=
by sorry

end part_a_part_b_part_c_part_d_l255_255411


namespace min_value_proof_l255_255508

noncomputable def min_value : ℝ := 3 + 2 * Real.sqrt 2

theorem min_value_proof (m n : ℝ) (hm : 0 < m) (hn : 0 < n) (h : m + n = 1) :
  (1 / m + 2 / n) = min_value :=
sorry

end min_value_proof_l255_255508


namespace min_value_of_t_l255_255223

theorem min_value_of_t (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 1) : 
  ∃ t : ℝ, t = 3 + 2 * Real.sqrt 2 ∧ t = 1 / a + 1 / b :=
sorry

end min_value_of_t_l255_255223


namespace intersecting_circles_l255_255374

noncomputable def distance (z1 z2 : Complex) : ℝ :=
  Complex.abs (z1 - z2)

theorem intersecting_circles (k : ℝ) :
  (∀ (z : Complex), (distance z 4 = 3 * distance z (-4)) → (distance z 0 = k)) →
  (k = 13 + Real.sqrt 153 ∨ k = |13 - Real.sqrt 153|) := 
sorry

end intersecting_circles_l255_255374


namespace range_of_a_l255_255785

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ -2 ≤ a ∧ a ≤ 2 :=
by sorry

end range_of_a_l255_255785


namespace smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l255_255917

theorem smallest_N_such_that_N_and_N_squared_end_in_same_three_digits :
  ∃ N : ℕ, (N > 0) ∧ (N % 1000 = (N^2 % 1000)) ∧ (1 ≤ N / 100 % 10) ∧ (N = 376) :=
by
  sorry

end smallest_N_such_that_N_and_N_squared_end_in_same_three_digits_l255_255917


namespace cost_of_article_l255_255582

theorem cost_of_article (C : ℝ) (H1 : 350 - C = G + 0.05 * G) (H2 : 345 - C = G) : C = 245 :=
by
  sorry

end cost_of_article_l255_255582


namespace calculate_fraction_l255_255301

theorem calculate_fraction :
  (5 * 6 - 4) / 8 = 13 / 4 := 
by
  sorry

end calculate_fraction_l255_255301


namespace John_break_time_l255_255262

-- Define the constants
def John_dancing_hours : ℕ := 8

-- Define the condition for James's dancing time 
def James_dancing_time (B : ℕ) : ℕ := 
  let total_time := John_dancing_hours + B
  total_time + total_time / 3

-- State the problem as a theorem
theorem John_break_time (B : ℕ) : John_dancing_hours + James_dancing_time B = 20 → B = 1 := 
  by sorry

end John_break_time_l255_255262


namespace a3_plus_a4_l255_255352

def sum_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, S n = 3^(n + 1)

theorem a3_plus_a4 (S : ℕ → ℕ) (a : ℕ → ℕ) (h : sum_of_sequence S a) :
  a 3 + a 4 = 216 :=
sorry

end a3_plus_a4_l255_255352


namespace trigonometric_expression_identity_l255_255621

open Real

theorem trigonometric_expression_identity :
  (1 - 1 / cos (35 * (pi / 180))) * 
  (1 + 1 / sin (55 * (pi / 180))) * 
  (1 - 1 / sin (35 * (pi / 180))) * 
  (1 + 1 / cos (55 * (pi / 180))) = 1 := by
  sorry

end trigonometric_expression_identity_l255_255621


namespace standard_equation_of_circle_l255_255505

theorem standard_equation_of_circle :
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ (h - 2) / 2 = k / 1 + 3 / 2 ∧ 
  ((h - 2)^2 + (k + 3)^2 = r^2) ∧ ((h + 2)^2 + (k + 5)^2 = r^2) ∧ 
  h = -1 ∧ k = -2 ∧ r^2 = 10 :=
by
  sorry

end standard_equation_of_circle_l255_255505


namespace arithmetic_sequence_problem_l255_255257

variable {a₁ d : ℝ} (S : ℕ → ℝ)

axiom Sum_of_terms (n : ℕ) : S n = n * a₁ + (n * (n - 1) / 2) * d

theorem arithmetic_sequence_problem
  (h : S 10 = 4 * S 5) :
  (a₁ / d) = 1 / 2 :=
by
  -- definitional expansion and algebraic simplification would proceed here
  sorry

end arithmetic_sequence_problem_l255_255257


namespace third_circle_radius_l255_255557

theorem third_circle_radius (r1 r2 d : ℝ) (τ : ℝ) (h1: r1 = 1) (h2: r2 = 9) (h3: d = 17) : 
  τ = 225 / 64 :=
by
  sorry

end third_circle_radius_l255_255557


namespace mr_zander_total_payment_l255_255950

noncomputable def total_cost (cement_bags : ℕ) (price_per_bag : ℝ) (sand_lorries : ℕ) 
(tons_per_lorry : ℝ) (price_per_ton : ℝ) (discount_rate : ℝ) (tax_rate : ℝ) : ℝ :=
  let cement_cost_before_discount := cement_bags * price_per_bag
  let discount := cement_cost_before_discount * discount_rate
  let cement_cost_after_discount := cement_cost_before_discount - discount
  let sand_cost_before_tax := sand_lorries * tons_per_lorry * price_per_ton
  let tax := sand_cost_before_tax * tax_rate
  let sand_cost_after_tax := sand_cost_before_tax + tax
  cement_cost_after_discount + sand_cost_after_tax

theorem mr_zander_total_payment :
  total_cost 500 10 20 10 40 0.05 0.07 = 13310 := 
sorry

end mr_zander_total_payment_l255_255950


namespace tickets_left_l255_255757

theorem tickets_left (initial_tickets used_tickets tickets_left : ℕ) 
  (h1 : initial_tickets = 127) 
  (h2 : used_tickets = 84) : 
  tickets_left = initial_tickets - used_tickets := 
by
  sorry

end tickets_left_l255_255757


namespace indigo_restaurant_total_reviews_l255_255144

-- Define the number of 5-star reviews
def five_star_reviews : Nat := 6

-- Define the number of 4-star reviews
def four_star_reviews : Nat := 7

-- Define the number of 3-star reviews
def three_star_reviews : Nat := 4

-- Define the number of 2-star reviews
def two_star_reviews : Nat := 1

-- Define the total number of reviews
def total_reviews : Nat := five_star_reviews + four_star_reviews + three_star_reviews + two_star_reviews

-- Proof that the total number of customer reviews is 18
theorem indigo_restaurant_total_reviews : total_reviews = 18 :=
by
  -- Direct calculation
  sorry

end indigo_restaurant_total_reviews_l255_255144


namespace state_A_selection_percentage_l255_255803

theorem state_A_selection_percentage
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (x : ℕ)
  (selected_B_ratio : ℚ)
  (extra_B : ℕ)
  (h1 : candidates_A = 7900)
  (h2 : candidates_B = 7900)
  (h3 : selected_B_ratio = 0.07)
  (h4 : extra_B = 79)
  (h5 : 7900 * (7 / 100) + 79 = 7900 * (x / 100) + 79) :
  x = 7 := by
  sorry

end state_A_selection_percentage_l255_255803


namespace problem_one_problem_two_problem_three_l255_255232

noncomputable def f (x : ℝ) : ℝ := 3 - 2 * Real.log x / Real.log 2

noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := (f x + 1) * g x

noncomputable def M (x : ℝ) : ℝ :=
  if f x >= g x then g x else f x

noncomputable def condition_one : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → -6 ≤ h x ∧ h x ≤ 2

noncomputable def condition_two : Prop :=
  ∃ x, (M x = 1 ∧ 0 < x ∧ x ≤ 2) ∧ (∀ y, 0 < y ∧ y < x → M y < 1)

noncomputable def condition_three : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 8 → f (x^2) * f (Real.sqrt x) ≥ g x * -3

theorem problem_one : condition_one := sorry
theorem problem_two : condition_two := sorry
theorem problem_three : condition_three := sorry

end problem_one_problem_two_problem_three_l255_255232


namespace sub_neg_seven_eq_neg_fourteen_l255_255585

theorem sub_neg_seven_eq_neg_fourteen : (-7) - 7 = -14 := 
  by
  sorry

end sub_neg_seven_eq_neg_fourteen_l255_255585


namespace num_ways_to_select_3_colors_from_9_l255_255642

def num_ways_select_colors (n k : ℕ) : ℕ := Nat.choose n k

theorem num_ways_to_select_3_colors_from_9 : num_ways_select_colors 9 3 = 84 := by
  sorry

end num_ways_to_select_3_colors_from_9_l255_255642


namespace find_two_digit_number_l255_255188

theorem find_two_digit_number (N : ℕ) (h1 : 10 ≤ N ∧ N < 100) 
                              (h2 : N % 2 = 0) (h3 : N % 11 = 0) 
                              (h4 : ∃ k : ℕ, (N / 10) * (N % 10) = k^3) :
  N = 88 :=
by {
  sorry
}

end find_two_digit_number_l255_255188


namespace ice_cream_flavors_l255_255794

theorem ice_cream_flavors :
  ∃ n : ℕ, n = (Nat.choose 7 2) ∧ n = 21 :=
by
  use Nat.choose 7 2
  split
  · rfl
  sorry

end ice_cream_flavors_l255_255794


namespace evaluate_expression_l255_255478

theorem evaluate_expression :
  (2 * 4 * 6) * (1 / 2 + 1 / 4 + 1 / 6) = 44 :=
by
  sorry

end evaluate_expression_l255_255478


namespace quadratic_inequality_l255_255027

variables {a b c x y : ℝ}

/-- A quadratic polynomial with non-negative coefficients. -/
def p (x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem quadratic_inequality (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  (p (x * y)) ^ 2 ≤ (p (x ^ 2)) * (p (y ^ 2)) :=
by
  sorry

end quadratic_inequality_l255_255027


namespace total_cans_in_display_l255_255806

-- Definitions and conditions
def first_term : ℕ := 30
def second_term : ℕ := 27
def nth_term : ℕ := 3
def common_difference : ℕ := second_term - first_term

-- Statement of the problem
theorem total_cans_in_display : 
  ∃ (n : ℕ), nth_term = first_term + (n - 1) * common_difference ∧
  (2 * 165 = n * (first_term + nth_term)) :=
by
  sorry

end total_cans_in_display_l255_255806


namespace computer_game_cost_l255_255381

variable (ticket_cost : ℕ := 12)
variable (num_tickets : ℕ := 3)
variable (total_spent : ℕ := 102)

theorem computer_game_cost (C : ℕ) (h : C + num_tickets * ticket_cost = total_spent) : C = 66 :=
by
  -- Proof would go here
  sorry

end computer_game_cost_l255_255381


namespace union_of_M_and_N_l255_255790

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_of_M_and_N : M ∪ N = {x | -1 < x ∧ x < 3} := by
  sorry

end union_of_M_and_N_l255_255790


namespace total_broken_marbles_l255_255640

def percentage_of (percent : ℝ) (total : ℝ) : ℝ := percent * total / 100

theorem total_broken_marbles :
  let first_set_total := 50
  let second_set_total := 60
  let first_set_percent_broken := 10
  let second_set_percent_broken := 20
  let first_set_broken := percentage_of first_set_percent_broken first_set_total
  let second_set_broken := percentage_of second_set_percent_broken second_set_total
  first_set_broken + second_set_broken = 17 :=
by
  sorry

end total_broken_marbles_l255_255640


namespace transformed_point_of_function_l255_255936

theorem transformed_point_of_function (f : ℝ → ℝ) (h : f 1 = -2) : f (-1) + 1 = -1 :=
by
  sorry

end transformed_point_of_function_l255_255936


namespace polynomial_solution_l255_255482

theorem polynomial_solution (P : ℝ → ℝ) (h₀ : P 0 = 0) (h₁ : ∀ x : ℝ, P x = (1/2) * (P (x+1) + P (x-1))) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_solution_l255_255482


namespace red_ball_higher_probability_l255_255717

theorem red_ball_higher_probability : 
  let prob := λ k, 2^{-k}
  let bins := Set.Ici 1
  (∑ k in bins, ∏ ball in [red, green, blue], prob k ^ 3) = ∑ r in bins, ∑ g in bins, ∑ b in bins, if r > g ∧ r > b then ∏ ball in [r, g, b], prob ball else 0 = 2/7 :=
by
  sorry

end red_ball_higher_probability_l255_255717


namespace intersecting_circles_unique_point_l255_255376

theorem intersecting_circles_unique_point (k : ℝ) :
  (∃ z : ℂ, |z - 4| = 3 * |z + 4| ∧ |z| = k) ↔ 
  k = 4 ∨ k = 14 :=
by
  sorry

end intersecting_circles_unique_point_l255_255376


namespace divisors_log_sum_eq_l255_255847

open BigOperators

/-- Given the sum of the base-10 logarithms of the divisors of \( 10^{2n} = 4752 \), prove that \( n = 12 \). -/
theorem divisors_log_sum_eq (n : ℕ) (h : ∑ a in Finset.range (2*n + 1), ∑ b in Finset.range (2*n + 1), 
  (a * Real.log (2) / Real.log (10) + b * Real.log (5) / Real.log (10)) = 4752) : n = 12 :=
by {
  sorry
}

end divisors_log_sum_eq_l255_255847


namespace terry_age_proof_l255_255802

theorem terry_age_proof
  (nora_age : ℕ)
  (h1 : nora_age = 10)
  (terry_age_in_10_years : ℕ)
  (h2 : terry_age_in_10_years = 4 * nora_age)
  (nora_age_in_5_years : ℕ)
  (h3 : nora_age_in_5_years = nora_age + 5)
  (sam_age_in_5_years : ℕ)
  (h4 : sam_age_in_5_years = 2 * nora_age_in_5_years)
  (sam_current_age : ℕ)
  (h5 : sam_current_age = sam_age_in_5_years - 5)
  (terry_current_age : ℕ)
  (h6 : sam_current_age = terry_current_age + 6) :
  terry_current_age = 19 :=
by
  sorry

end terry_age_proof_l255_255802


namespace ratio_lcm_gcf_l255_255724

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 252) (h₂ : b = 675) : 
  let lcm_ab := Nat.lcm a b
  let gcf_ab := Nat.gcd a b
  (lcm_ab / gcf_ab) = 2100 :=
by
  sorry

end ratio_lcm_gcf_l255_255724


namespace geometric_sequence_common_ratio_l255_255933

open scoped Nat

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ)
  (h : ∀ n : ℕ, a n * a (n + 1) = (16 : ℝ) ^ n) :
  ∃ r : ℝ, (∀ n : ℕ, a n = a 0 * r ^ n) ∧ (r = 4) :=
sorry

end geometric_sequence_common_ratio_l255_255933


namespace factorize_cubic_expression_l255_255209

theorem factorize_cubic_expression (m : ℝ) : 
  m^3 - 16 * m = m * (m + 4) * (m - 4) :=
by
  sorry

end factorize_cubic_expression_l255_255209


namespace train_speed_approx_72_km_hr_l255_255752

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 132
noncomputable def crossing_time : ℝ := 14.098872090232781
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_m_s : ℝ := total_distance / crossing_time
noncomputable def conversion_factor : ℝ := 3.6
noncomputable def speed_km_hr : ℝ := speed_m_s * conversion_factor

theorem train_speed_approx_72_km_hr : abs (speed_km_hr - 72) < 0.01 :=
sorry

end train_speed_approx_72_km_hr_l255_255752


namespace pharmacist_weights_exist_l255_255595

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end pharmacist_weights_exist_l255_255595


namespace handrail_length_is_25_point_1_l255_255601

noncomputable def handrail_length (theta radius height : ℝ) : ℝ :=
  let circumference := 2 * real.pi * radius
  let width := (theta / 360) * circumference
  real.sqrt (height^2 + width^2)

theorem handrail_length_is_25_point_1 :
  handrail_length 315 4 12 = 25.1 :=
by
  sorry

end handrail_length_is_25_point_1_l255_255601


namespace abes_present_age_l255_255565

theorem abes_present_age :
  ∃ A : ℕ, A + (A - 7) = 27 ∧ A = 17 :=
by
  sorry

end abes_present_age_l255_255565


namespace largest_n_for_positive_sum_l255_255351

noncomputable def a_n (n : ℕ) : ℝ

axiom arithmetic_sequence (d : ℝ) : ∀ n : ℕ, a_n (n+1) = a_n n + d

axiom first_term_positive : a_n 1 > 0

axiom condition1 : a_n 2003 + a_n 2004 > 0

axiom condition2 : a_n 2003 * a_n 2004 < 0

theorem largest_n_for_positive_sum : ∃ n : ℕ, n = 4006 ∧ 
    let S_n := λ n, n / 2 * (2 * a_n 1 + (n - 1) * d) in
     S_n n > 0 :=
sorry

end largest_n_for_positive_sum_l255_255351


namespace rectangles_260_261_272_273_have_similar_property_l255_255997

-- Defining a rectangle as a structure with width and height
structure Rectangle where
  width : ℕ
  height : ℕ

-- Given conditions
def r1 : Rectangle := ⟨16, 10⟩
def r2 : Rectangle := ⟨23, 7⟩

-- Hypothesis function indicating the dissection trick causing apparent equality
def dissection_trick (r1 r2 : Rectangle) : Prop :=
  (r1.width * r1.height : ℕ) = (r2.width * r2.height : ℕ) + 1

-- The statement of the proof problem
theorem rectangles_260_261_272_273_have_similar_property :
  ∃ (r3 r4 : Rectangle) (r5 r6 : Rectangle),
    dissection_trick r3 r4 ∧ dissection_trick r5 r6 ∧
    r3.width * r3.height = 260 ∧ r4.width * r4.height = 261 ∧
    r5.width * r5.height = 272 ∧ r6.width * r6.height = 273 :=
  sorry

end rectangles_260_261_272_273_have_similar_property_l255_255997


namespace joan_total_cents_l255_255261

-- Conditions
def quarters : ℕ := 12
def dimes : ℕ := 8
def nickels : ℕ := 15
def pennies : ℕ := 25

def value_of_quarter : ℕ := 25
def value_of_dime : ℕ := 10
def value_of_nickel : ℕ := 5
def value_of_penny : ℕ := 1

-- The problem statement
theorem joan_total_cents : 
  (quarters * value_of_quarter + dimes * value_of_dime + nickels * value_of_nickel + pennies * value_of_penny) = 480 := 
  sorry

end joan_total_cents_l255_255261


namespace total_payment_correct_l255_255160

def payment_X (payment_Y : ℝ) : ℝ := 1.2 * payment_Y
def payment_Y : ℝ := 254.55
def total_payment (payment_X payment_Y : ℝ) : ℝ := payment_X + payment_Y

theorem total_payment_correct :
  total_payment (payment_X payment_Y) payment_Y = 560.01 :=
by
  sorry

end total_payment_correct_l255_255160


namespace total_marks_l255_255386

theorem total_marks (Keith_marks Larry_marks Danny_marks : ℕ)
  (hK : Keith_marks = 3)
  (hL : Larry_marks = 3 * Keith_marks)
  (hD : Danny_marks = Larry_marks + 5) :
  Keith_marks + Larry_marks + Danny_marks = 26 := 
by
  sorry

end total_marks_l255_255386


namespace tip_percentage_l255_255592

theorem tip_percentage (cost_of_crown : ℕ) (total_paid : ℕ) (h1 : cost_of_crown = 20000) (h2 : total_paid = 22000) :
  (total_paid - cost_of_crown) * 100 / cost_of_crown = 10 :=
by
  sorry

end tip_percentage_l255_255592


namespace percentage_difference_l255_255315

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) : (1 - y / x) * 100 = 91.67 :=
by {
  sorry
}

end percentage_difference_l255_255315


namespace canal_depth_l255_255973

theorem canal_depth (A : ℝ) (W_top : ℝ) (W_bottom : ℝ) (d : ℝ) (h: ℝ)
  (h₁ : A = 840) 
  (h₂ : W_top = 12) 
  (h₃ : W_bottom = 8)
  (h₄ : A = (1/2) * (W_top + W_bottom) * d) : 
  d = 84 :=
by 
  sorry

end canal_depth_l255_255973


namespace triplet_solution_l255_255911

theorem triplet_solution (x y n : ℕ) (hx : x > 0) (hy : y > 0) (hn : n > 0) :
  (x! + y!) / n! = 3 ^ n ↔ (x = 2 ∧ y = 1 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1) :=
sorry

end triplet_solution_l255_255911


namespace repairs_cost_correct_l255_255583

variable (C : ℝ)

def cost_of_scooter : ℝ := C
def repair_cost (C : ℝ) : ℝ := 0.10 * C
def selling_price (C : ℝ) : ℝ := 1.20 * C
def profit (C : ℝ) : ℝ := 1100
def profit_percentage (C : ℝ) : ℝ := 0.20 

theorem repairs_cost_correct (C : ℝ) (h₁ : selling_price C - cost_of_scooter C = profit C) (h₂ : profit_percentage C = 0.20) : 
  repair_cost C = 550 := by
  sorry

end repairs_cost_correct_l255_255583


namespace larger_root_of_quadratic_eq_l255_255079

theorem larger_root_of_quadratic_eq : 
  ∀ x : ℝ, x^2 - 13 * x + 36 = 0 → x = 9 ∨ x = 4 := by
  sorry

end larger_root_of_quadratic_eq_l255_255079


namespace jump_length_third_frog_l255_255849

theorem jump_length_third_frog (A B C : ℝ) 
  (h1 : A = (B + C) / 2) 
  (h2 : B = (A + C) / 2) 
  (h3 : |B - A| + |(B - C) / 2| = 60) : 
  |C - (A + B) / 2| = 30 :=
sorry

end jump_length_third_frog_l255_255849


namespace gcd_max_value_l255_255885

noncomputable def max_gcd (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else 1

theorem gcd_max_value :
  ∃ n : ℕ, n > 0 ∧ (∀ m : ℕ, m > 0 → gcd (13 * m + 4) (7 * m + 2) ≤ max_gcd m) ∧
              (∀ m : ℕ, m > 0 → max_gcd m ≤ 2) :=
by {
  sorry
}

end gcd_max_value_l255_255885


namespace parabola_standard_equation_l255_255231

variable {a : ℝ} (h : a < 0)

theorem parabola_standard_equation (h : a < 0) :
  ∃ (p : ℝ), p = -2 * a ∧ (∀ (x y : ℝ), y^2 = -2 * p * x ↔ y^2 = 4 * a * x) :=
sorry

end parabola_standard_equation_l255_255231


namespace sin_600_eq_neg_sqrt_3_div_2_l255_255172

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * (Real.pi / 180)) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l255_255172


namespace alpha_range_l255_255359

theorem alpha_range (α : ℝ) (hα : 0 < α ∧ α < 2 * Real.pi) : 
  (Real.sin α < Real.sqrt 3 / 2 ∧ Real.cos α > 1 / 2) ↔ 
  (0 < α ∧ α < Real.pi / 3 ∨ 5 * Real.pi / 3 < α ∧ α < 2 * Real.pi) := 
sorry

end alpha_range_l255_255359


namespace arithmetic_sequence_l255_255925

variable (p q : ℕ) -- Assuming natural numbers for simplicity, but can be generalized.

def a (n : ℕ) : ℕ := p * n + q

theorem arithmetic_sequence:
  ∀ n : ℕ, n ≥ 1 → (a n - a (n-1) = p) := by
  -- proof steps would go here
  sorry

end arithmetic_sequence_l255_255925


namespace find_integer_l255_255167

theorem find_integer (n : ℕ) (h1 : 0 < n) (h2 : 200 % n = 2) (h3 : 398 % n = 2) : n = 6 :=
sorry

end find_integer_l255_255167


namespace least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l255_255856

noncomputable def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else n.digits 10

noncomputable def product_of_digits (n : ℕ) : ℕ :=
  (digits n).foldl (λ x y => x * y) 1

def is_multiple_of (m n : ℕ) : Prop :=
  ∃ k, n = m * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_multiple_of 45 n ∧ is_multiple_of 9 (product_of_digits n)

theorem least_positive_multiple_of_45_with_product_of_digits_multiple_of_9 : 
  ∀ n, satisfies_conditions n → 495 ≤ n :=
by
  sorry

end least_positive_multiple_of_45_with_product_of_digits_multiple_of_9_l255_255856


namespace patrol_streets_in_one_hour_l255_255545

-- Definitions of the given conditions
def streets_patrolled_by_A := 36
def hours_by_A := 4
def rate_A := streets_patrolled_by_A / hours_by_A

def streets_patrolled_by_B := 55
def hours_by_B := 5
def rate_B := streets_patrolled_by_B / hours_by_B

def streets_patrolled_by_C := 42
def hours_by_C := 6
def rate_C := streets_patrolled_by_C / hours_by_C

-- Proof statement 
theorem patrol_streets_in_one_hour : rate_A + rate_B + rate_C = 27 := by
  sorry

end patrol_streets_in_one_hour_l255_255545


namespace jane_spent_75_days_reading_l255_255130

def pages : ℕ := 500
def speed_first_half : ℕ := 10
def speed_second_half : ℕ := 5

def book_reading_days (p s1 s2 : ℕ) : ℕ :=
  let half_pages := p / 2
  let days_first_half := half_pages / s1
  let days_second_half := half_pages / s2
  days_first_half + days_second_half

theorem jane_spent_75_days_reading :
  book_reading_days pages speed_first_half speed_second_half = 75 :=
by
  sorry

end jane_spent_75_days_reading_l255_255130


namespace sum_arithmetic_sequence_l255_255092

variable (a : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := (n + 1) / 2 * (2 * a 0 + n * (a 1 - a 0))

theorem sum_arithmetic_sequence (h_arith : arithmetic_sequence a) (h_condition : a 3 + a 4 + a 5 + a 6 = 18) :
  S a 9 = 45 :=
sorry

end sum_arithmetic_sequence_l255_255092


namespace verify_n_l255_255670

noncomputable def find_n (n : ℕ) : Prop :=
  let widget_rate1 := 3                             -- Widgets per worker-hour from the first condition
  let whoosit_rate1 := 2                            -- Whoosits per worker-hour from the first condition
  let widget_rate3 := 1                             -- Widgets per worker-hour from the third condition
  let minutes_per_widget := 1                       -- Arbitrary unit time for one widget
  let minutes_per_whoosit := 2                      -- 2 times unit time for one whoosit based on problem statement
  let whoosit_rate3 := 2 / 3                        -- Whoosits per worker-hour from the third condition
  let widget_rate2 := 540 / (90 * 3 : ℕ)            -- Widgets per hour in the second condition
  let whoosit_rate2 := n / (90 * 3 : ℕ)             -- Whoosits per hour in the second condition
  widget_rate2 = 2 ∧ whoosit_rate2 = 4 / 3 ∧
  (minutes_per_widget < minutes_per_whoosit) ∧
  (whoosit_rate2 = (4 / 3 : ℚ) ↔ n = 360)

theorem verify_n : find_n 360 :=
by sorry

end verify_n_l255_255670


namespace sky_color_changes_l255_255069

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l255_255069


namespace g_neg_two_is_zero_l255_255096

theorem g_neg_two_is_zero {f g : ℤ → ℤ} 
  (h_odd: ∀ x: ℤ, f (-x) + (-x) = -(f x + x)) 
  (hf_two: f 2 = 1) 
  (hg_def: ∀ x: ℤ, g x = f x + 1):
  g (-2) = 0 := 
sorry

end g_neg_two_is_zero_l255_255096


namespace coins_fit_in_new_box_l255_255755

-- Definitions
def diameters_bound (d : ℕ) : Prop :=
  d ≤ 10

def box_fits (length width : ℕ) (fits : Prop) : Prop :=
  fits

-- Conditions
axiom coins_diameter_bound : ∀ (d : ℕ), diameters_bound d
axiom original_box_fits : box_fits 30 70 True

-- Proof statement
theorem coins_fit_in_new_box : box_fits 40 60 True :=
sorry

end coins_fit_in_new_box_l255_255755


namespace hyperbola_equation_l255_255776

-- Lean 4 statement
theorem hyperbola_equation (a b : ℝ) (hpos_a : a > 0) (hpos_b : b > 0)
    (length_imag_axis : 2 * b = 2)
    (asymptote : ∃ (k : ℝ), ∀ x : ℝ, y = k * x ↔ y = (1 / 2) * x) :
  (x y : ℝ) → (x^2 / a^2) - (y^2 / b^2) = 1 ↔ (x^2 / 4) - (y^2 / 1) = 1 :=
by 
  intros
  sorry

end hyperbola_equation_l255_255776


namespace problem_l255_255546

theorem problem (n : ℕ) (h : n ∣ (2^n - 2)) : (2^n - 1) ∣ (2^(2^n - 1) - 2) :=
by
  sorry

end problem_l255_255546


namespace edward_final_money_l255_255207

theorem edward_final_money 
  (spring_earnings : ℕ)
  (summer_earnings : ℕ)
  (supplies_cost : ℕ)
  (h_spring : spring_earnings = 2)
  (h_summer : summer_earnings = 27)
  (h_supplies : supplies_cost = 5)
  : spring_earnings + summer_earnings - supplies_cost = 24 := 
sorry

end edward_final_money_l255_255207


namespace broccoli_difference_l255_255042

theorem broccoli_difference (A : ℕ) (s : ℕ) (s' : ℕ)
  (h1 : A = 1600)
  (h2 : s = Nat.sqrt A)
  (h3 : s' < s)
  (h4 : (s')^2 < A)
  (h5 : A - (s')^2 = 79) :
  (1600 - (s')^2) = 79 :=
by
  sorry

end broccoli_difference_l255_255042


namespace jiaqi_grade_is_95_3_l255_255280

def extracurricular_score : ℝ := 96
def mid_term_score : ℝ := 92
def final_exam_score : ℝ := 97

def extracurricular_weight : ℝ := 0.2
def mid_term_weight : ℝ := 0.3
def final_exam_weight : ℝ := 0.5

def total_grade : ℝ :=
  extracurricular_score * extracurricular_weight +
  mid_term_score * mid_term_weight +
  final_exam_score * final_exam_weight

theorem jiaqi_grade_is_95_3 : total_grade = 95.3 :=
by
  simp [total_grade, extracurricular_score, mid_term_score, final_exam_score,
    extracurricular_weight, mid_term_weight, final_exam_weight]
  sorry

end jiaqi_grade_is_95_3_l255_255280


namespace problem_statement_l255_255514

open Real

theorem problem_statement (a b : ℝ) (h : 2^a + log 2 a = 4^b + 2 * log 4 b) : a < 2 * b := sorry

end problem_statement_l255_255514


namespace window_area_l255_255323

def meter_to_feet : ℝ := 3.28084
def length_in_meters : ℝ := 2
def width_in_feet : ℝ := 15

def length_in_feet := length_in_meters * meter_to_feet
def area_in_square_feet := length_in_feet * width_in_feet

theorem window_area : area_in_square_feet = 98.4252 := 
by
  sorry

end window_area_l255_255323


namespace fraction_of_subsets_l255_255814

theorem fraction_of_subsets (S T : ℕ) (hS : S = 2^10) (hT : T = Nat.choose 10 3) :
    (T:ℚ) / (S:ℚ) = 15 / 128 :=
by sorry

end fraction_of_subsets_l255_255814


namespace calculate_expression_l255_255651

theorem calculate_expression (f : ℕ → ℝ) (h1 : ∀ a b, f (a + b) = f a * f b) (h2 : f 1 = 2) : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) = 6 := 
sorry

end calculate_expression_l255_255651


namespace intersection_A_B_l255_255133

noncomputable def domain_ln_1_minus_x : Set ℝ := {x : ℝ | x < 1}
def range_x_squared : Set ℝ := {y : ℝ | 0 ≤ y}
def intersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

theorem intersection_A_B :
  (domain_ln_1_minus_x ∩ range_x_squared) = intersection :=
by sorry

end intersection_A_B_l255_255133
