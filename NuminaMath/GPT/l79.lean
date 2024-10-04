import Mathlib

namespace largest_of_three_numbers_l79_79472

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l79_79472


namespace min_value_2a_3b_6c_l79_79902

theorem min_value_2a_3b_6c (a b c : ℝ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_c : 0 < c) (habc : a * b * c = 27) :
  2 * a + 3 * b + 6 * c ≥ 27 :=
sorry

end min_value_2a_3b_6c_l79_79902


namespace solve_for_x_l79_79128

theorem solve_for_x (x y : ℝ) (h : x / (x - 1) = (y^2 + 3 * y - 5) / (y^2 + 3 * y - 7)) :
  x = (y^2 + 3 * y - 5) / 2 :=
by 
  sorry

end solve_for_x_l79_79128


namespace BANANA_arrangements_l79_79523

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l79_79523


namespace point_on_line_l79_79867

theorem point_on_line (m : ℝ) (P : ℝ × ℝ) (line_eq : ℝ × ℝ → Prop) (h : P = (2, m)) 
  (h_line : line_eq = fun P => 3 * P.1 + P.2 = 2) : 
  3 * 2 + m = 2 → m = -4 :=
by
  intro h1
  linarith

end point_on_line_l79_79867


namespace no_equilateral_triangle_OAB_exists_l79_79752

theorem no_equilateral_triangle_OAB_exists :
  ∀ (A B : ℝ × ℝ), 
  ((∃ a : ℝ, A = (a, (3 / 2) ^ a)) ∧ B.1 > 0 ∧ B.2 = 0) → 
  ¬ (∃ k : ℝ, k = (A.2 / A.1) ∧ k > (3 ^ (1 / 2)) / 3) := 
by 
  intro A B h
  sorry

end no_equilateral_triangle_OAB_exists_l79_79752


namespace common_factor_of_polynomial_l79_79926

variables (x y m n : ℝ)

theorem common_factor_of_polynomial :
  ∃ (k : ℝ), (2 * (m - n)) = k ∧ (4 * x * (m - n) + 2 * y * (m - n)^2) = k * (2 * x * (m - n)) :=
sorry

end common_factor_of_polynomial_l79_79926


namespace xiao_ming_proposition_false_l79_79815

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m * m ≤ n → m = 1 ∨ m = n → m ∣ n

def check_xiao_ming_proposition : Prop :=
  ∃ n : ℕ, ∃ (k : ℕ), k < n → ∃ (p q : ℕ), p = q → n^2 - n + 11 = p * q ∧ p > 1 ∧ q > 1

theorem xiao_ming_proposition_false : ¬ (∀ n: ℕ, is_prime (n^2 - n + 11)) :=
by
  sorry

end xiao_ming_proposition_false_l79_79815


namespace independent_dependent_variables_max_acceptance_at_13_acceptance_increasing_decreasing_l79_79355

-- Definitions of the given variables and conditions
def time := ℝ
def acceptance := ℝ

def acceptance_ability (x : time) : acceptance :=
  match x with
  | 2 := 47.8
  | 5 := 53.5
  | 7 := 56.3
  | 10 := 59
  | 12 := 59.8
  | 13 := 59.9
  | 14 := 59.8
  | 17 := 58.3
  | 20 := 55
  | _ := 0

-- Statement 1: Independent and dependent variables
theorem independent_dependent_variables :
  ∃ (x : time → acceptance), 
  ∃ (y : time → acceptance), 
  (∀ t : time, 0 ≤ t ∧ t ≤ 30 → 
    (x t = t) ∧ (y t = acceptance_ability t)) := sorry

-- Statement 2: Maximum acceptance ability at x = 13
theorem max_acceptance_at_13 :
  ∃ t : time, (t = 13) ∧ (acceptance_ability t = 59.9) := sorry

-- Statement 3: Increasing and decreasing ranges of acceptance ability
theorem acceptance_increasing_decreasing :
  (∀ t : time, 0 < t ∧ t < 13 → acceptance_ability t < acceptance_ability (t + 1)) ∧ 
  (∀ t : time, 13 < t ∧ t < 20 → acceptance_ability t > acceptance_ability (t + 1)) := sorry

end independent_dependent_variables_max_acceptance_at_13_acceptance_increasing_decreasing_l79_79355


namespace pool_filling_times_l79_79493

theorem pool_filling_times:
  ∃ (x y z u : ℕ),
    (1/x + 1/y = 1/70) ∧
    (1/x + 1/z = 1/84) ∧
    (1/y + 1/z = 1/140) ∧
    (1/u = 1/x + 1/y + 1/z) ∧
    (x = 105) ∧
    (y = 210) ∧
    (z = 420) ∧
    (u = 60) := 
  sorry

end pool_filling_times_l79_79493


namespace line_parallel_through_point_l79_79989

theorem line_parallel_through_point (P : ℝ × ℝ) (a b c : ℝ) (ha : a = 3) (hb : b = -4) (hc : c = 6) (hP : P = (4, -1)) :
  ∃ d : ℝ, (d = -16) ∧ (∀ x y : ℝ, a * x + b * y + d = 0 ↔ 3 * x - 4 * y - 16 = 0) :=
by
  sorry

end line_parallel_through_point_l79_79989


namespace proposition_D_l79_79159

variable {A B C : Set α} (h1 : ∀ a (ha : a ∈ A), ∃ b ∈ B, a = b)
variable {A B C : Set α} (h2 : ∀ c (hc : c ∈ C), ∃ b ∈ B, b = c) 

theorem proposition_D (A B C : Set α) (h : A ∩ B = B ∪ C) : C ⊆ B :=
by 
  sorry

end proposition_D_l79_79159


namespace value_of_a_l79_79268

open Set

theorem value_of_a (a : ℝ) (h : {1, 2} ∪ {x | x^2 - a * x + a - 1 = 0} = {1, 2}) : a = 3 :=
by
  sorry

end value_of_a_l79_79268


namespace arithmetic_geometric_sequence_l79_79595

theorem arithmetic_geometric_sequence :
  ∀ (a : ℕ → ℕ) (b : ℕ → ℕ),
    (a 1 + a 2 = 10) →
    (a 4 - a 3 = 2) →
    (b 2 = a 3) →
    (b 3 = a 7) →
    a 15 = b 4 :=
by
  intros a b h1 h2 h3 h4
  sorry

end arithmetic_geometric_sequence_l79_79595


namespace ellie_oil_needs_l79_79571

def oil_per_wheel : ℕ := 10
def number_of_wheels : ℕ := 2
def oil_for_rest : ℕ := 5
def total_oil_needed : ℕ := oil_per_wheel * number_of_wheels + oil_for_rest

theorem ellie_oil_needs : total_oil_needed = 25 := by
  sorry

end ellie_oil_needs_l79_79571


namespace chad_ice_cost_l79_79839

theorem chad_ice_cost
  (n : ℕ) -- Number of people
  (p : ℕ) -- Pounds of ice per person
  (c : ℝ) -- Cost per 10 pound bag of ice
  (h1 : n = 20) 
  (h2 : p = 3)
  (h3 : c = 4.5) :
  (3 * 20 / 10) * 4.5 = 27 :=
by
  sorry

end chad_ice_cost_l79_79839


namespace shortest_handspan_is_Doyoon_l79_79784

def Sangwon_handspan_cm : ℝ := 19.8
def Doyoon_handspan_cm : ℝ := 18.9
def Changhyeok_handspan_cm : ℝ := 19.3

theorem shortest_handspan_is_Doyoon :
  Doyoon_handspan_cm < Sangwon_handspan_cm ∧ Doyoon_handspan_cm < Changhyeok_handspan_cm :=
by
  sorry

end shortest_handspan_is_Doyoon_l79_79784


namespace max_balls_of_clay_l79_79941

theorem max_balls_of_clay (radius cube_side_length : ℝ) (V_cube : ℝ) (V_ball : ℝ) (num_balls : ℕ) :
  radius = 3 ->
  cube_side_length = 10 ->
  V_cube = cube_side_length ^ 3 ->
  V_ball = (4 / 3) * π * radius ^ 3 ->
  num_balls = ⌊ V_cube / V_ball ⌋ ->
  num_balls = 8 :=
by
  sorry

end max_balls_of_clay_l79_79941


namespace union_of_A_and_B_l79_79590

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l79_79590


namespace prime_factor_difference_duodecimal_l79_79092

theorem prime_factor_difference_duodecimal (A B : ℕ) (hA : 0 ≤ A ∧ A ≤ 11) (hB : 0 ≤ B ∧ B ≤ 11) (h : A ≠ B) : 
  ∃ k : ℤ, (12 * A + B - (12 * B + A)) = 11 * k := 
by sorry

end prime_factor_difference_duodecimal_l79_79092


namespace inequality_proof_l79_79289

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l79_79289


namespace total_hike_time_l79_79629

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l79_79629


namespace initial_cheerleaders_count_l79_79650

theorem initial_cheerleaders_count (C : ℕ) 
  (initial_football_players : ℕ := 13) 
  (quit_football_players : ℕ := 10) 
  (quit_cheerleaders : ℕ := 4) 
  (remaining_people : ℕ := 15) 
  (initial_total : ℕ := initial_football_players + C) 
  (final_total : ℕ := (initial_football_players - quit_football_players) + (C - quit_cheerleaders)) :
  remaining_people = final_total → C = 16 :=
by intros h; sorry

end initial_cheerleaders_count_l79_79650


namespace shekar_biology_marks_l79_79914

variable (M S SS E A : ℕ)

theorem shekar_biology_marks (hM : M = 76) (hS : S = 65) (hSS : SS = 82) (hE : E = 67) (hA : A = 77) :
  let total_marks := M + S + SS + E
  let total_average_marks := A * 5
  let biology_marks := total_average_marks - total_marks
  biology_marks = 95 :=
by
  sorry

end shekar_biology_marks_l79_79914


namespace find_y_l79_79610

open Classical

theorem find_y (a b c x y : ℚ)
  (h1 : a / b = 5 / 4)
  (h2 : b / c = 3 / x)
  (h3 : a / c = y / 4) :
  y = 15 / x :=
sorry

end find_y_l79_79610


namespace Ben_cards_left_l79_79503

def BenInitialBasketballCards : ℕ := 4 * 10
def BenInitialBaseballCards : ℕ := 5 * 8
def BenTotalInitialCards : ℕ := BenInitialBasketballCards + BenInitialBaseballCards
def BenGivenCards : ℕ := 58
def BenRemainingCards : ℕ := BenTotalInitialCards - BenGivenCards

theorem Ben_cards_left : BenRemainingCards = 22 :=
by 
  -- The proof will be placed here.
  sorry

end Ben_cards_left_l79_79503


namespace add_fifteen_sub_fifteen_l79_79455

theorem add_fifteen (n : ℕ) (m : ℕ) : n + m = 195 :=
by {
  sorry  -- placeholder for the actual proof
}

theorem sub_fifteen (n : ℕ) (m : ℕ) : n - m = 165 :=
by {
  sorry  -- placeholder for the actual proof
}

-- Let's instantiate these theorems with the specific values from the problem:
noncomputable def verify_addition : 180 + 15 = 195 :=
by exact add_fifteen 180 15

noncomputable def verify_subtraction : 180 - 15 = 165 :=
by exact sub_fifteen 180 15

end add_fifteen_sub_fifteen_l79_79455


namespace original_list_length_l79_79675

variable (n m : ℕ)   -- number of integers and the mean respectively
variable (l : List ℤ) -- the original list of integers

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Condition 1: Appending 25 increases mean by 3
def condition1 (l : List ℤ) : Prop :=
  mean (25 :: l) = mean l + 3

-- Condition 2: Appending -4 to the enlarged list decreases the mean by 1.5
def condition2 (l : List ℤ) : Prop :=
  mean (-4 :: 25 :: l) = mean (25 :: l) - 1.5

theorem original_list_length (l : List ℤ) (h1 : condition1 l) (h2 : condition2 l) : l.length = 4 := by
  sorry

end original_list_length_l79_79675


namespace chemical_x_added_l79_79487

theorem chemical_x_added (initial_volume : ℝ) (initial_percentage : ℝ) (final_percentage : ℝ) : 
  initial_volume = 80 → initial_percentage = 0.2 → final_percentage = 0.36 → 
  ∃ (a : ℝ), 0.20 * initial_volume + a = 0.36 * (initial_volume + a) ∧ a = 20 :=
by
  intros h1 h2 h3
  use 20
  sorry

end chemical_x_added_l79_79487


namespace find_P_and_Q_l79_79403

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l79_79403


namespace distance_travelled_l79_79341

def actual_speed : ℝ := 50
def additional_speed : ℝ := 25
def time_difference : ℝ := 0.5

theorem distance_travelled (D : ℝ) : 0.5 = (D / actual_speed) - (D / (actual_speed + additional_speed)) → D = 75 :=
by sorry

end distance_travelled_l79_79341


namespace complete_triangles_l79_79042

noncomputable def possible_placements_count : Nat :=
  sorry

theorem complete_triangles {a b c : Nat} :
  (1 + 2 + 4 + 10 + a + b + c) = 23 →
  ∃ (count : Nat), count = 4 := 
by
  sorry

end complete_triangles_l79_79042


namespace sandy_hourly_wage_l79_79442

theorem sandy_hourly_wage (x : ℝ)
    (h1 : 10 * x + 6 * x + 14 * x = 450) : x = 15 :=
by
    sorry

end sandy_hourly_wage_l79_79442


namespace vector_dot_product_l79_79024

variables (a b : ℝ × ℝ)
variables (h1 : a = (1, 2))
variables (h2 : a - (1 / 5) • b = (-2, 1))

theorem vector_dot_product : (a.1 * b.1 + a.2 * b.2) = 25 :=
by
  sorry

end vector_dot_product_l79_79024


namespace coprime_n_minus_2_n_squared_minus_n_minus_1_l79_79212

theorem coprime_n_minus_2_n_squared_minus_n_minus_1 (n : ℕ) : n - 2 ∣ n^2 - n - 1 → False :=
by
-- proof omitted as per instructions
sorry

end coprime_n_minus_2_n_squared_minus_n_minus_1_l79_79212


namespace solution_of_equations_l79_79743

variables (x y z w : ℤ)

def system_of_equations :=
  x + y + z + w = 20 ∧
  y + 2 * z - 3 * w = 28 ∧
  x - 2 * y + z = 36 ∧
  -7 * x - y + 5 * z + 3 * w = 84

theorem solution_of_equations (x y z w : ℤ) :
  system_of_equations x y z w → (x, y, z, w) = (4, -6, 20, 2) :=
by sorry

end solution_of_equations_l79_79743


namespace find_m_value_l79_79855

theorem find_m_value (x y m : ℝ) 
  (h1 : 2 * x + y = 5) 
  (h2 : x - 2 * y = m)
  (h3 : 2 * x - 3 * y = 1) : 
  m = 0 := 
sorry

end find_m_value_l79_79855


namespace price_per_glass_second_day_l79_79908

-- Given conditions
variables {O P : ℝ}
axiom condition1 : 0.82 * 2 * O = P * 3 * O

-- Problem statement
theorem price_per_glass_second_day : 
  P = 0.55 :=
by
  -- This is where the actual proof would go
  sorry

end price_per_glass_second_day_l79_79908


namespace find_smallest_k_satisfying_cos_square_l79_79103

theorem find_smallest_k_satisfying_cos_square (k : ℕ) (h : ∃ n : ℕ, k^2 = 180 * n - 64):
  k = 48 ∨ k = 53 :=
by sorry

end find_smallest_k_satisfying_cos_square_l79_79103


namespace sum_of_squares_of_medians_l79_79910

-- Define the components of the triangle
variables (a b c : ℝ)

-- Define the medians of the triangle
variables (s_a s_b s_c : ℝ)

-- State the theorem
theorem sum_of_squares_of_medians (h1 : s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2)) : 
  s_a^2 + s_b^2 + s_c^2 = (3 / 4) * (a^2 + b^2 + c^2) :=
by {
  -- The proof goes here
  sorry
}

end sum_of_squares_of_medians_l79_79910


namespace janice_overtime_shifts_l79_79891

theorem janice_overtime_shifts (x : ℕ) (h1 : 5 * 30 + 15 * x = 195) : x = 3 :=
by
  -- leaving the proof unfinished, as asked
  sorry

end janice_overtime_shifts_l79_79891


namespace polynomial_division_remainder_zero_l79_79577

theorem polynomial_division_remainder_zero (x : ℂ) (hx : x^5 + x^4 + x^3 + x^2 + x + 1 = 0)
  : (x^55 + x^44 + x^33 + x^22 + x^11 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 := by
  sorry

end polynomial_division_remainder_zero_l79_79577


namespace expression_is_integer_if_k_eq_2_l79_79252

def binom (n k : ℕ) := n.factorial / (k.factorial * (n-k).factorial)

theorem expression_is_integer_if_k_eq_2 
  (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : k = 2) : 
  ∃ (m : ℕ), m = (n - 3 * k + 2) * binom n k / (k + 2) := sorry

end expression_is_integer_if_k_eq_2_l79_79252


namespace factor_polynomial_l79_79621

theorem factor_polynomial (x y z : ℂ) : 
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) = 
  (x - y) * (y - z) * (z - x) * (-(x * y + x * z + y * z)) := by
  sorry

end factor_polynomial_l79_79621


namespace candies_of_different_flavors_l79_79162

theorem candies_of_different_flavors (total_treats chewing_gums chocolate_bars : ℕ) (h1 : total_treats = 155) (h2 : chewing_gums = 60) (h3 : chocolate_bars = 55) :
  total_treats - (chewing_gums + chocolate_bars) = 40 := 
by 
  sorry

end candies_of_different_flavors_l79_79162


namespace possible_values_for_a_l79_79122

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) - x - 1

theorem possible_values_for_a (a : ℝ) (h: a ≠ 0) : 
  (∀ x : ℝ, f a x ≥ 0) ↔ a = 1 :=
by
  sorry

end possible_values_for_a_l79_79122


namespace difference_length_width_l79_79338

-- Definition of variables and conditions
variables (L W : ℝ)
def hall_width_half_length : Prop := W = (1/2) * L
def hall_area_578 : Prop := L * W = 578

-- Theorem to prove the desired result
theorem difference_length_width (h1 : hall_width_half_length L W) (h2 : hall_area_578 L W) : L - W = 17 :=
sorry

end difference_length_width_l79_79338


namespace total_students_l79_79883

-- Given conditions
variable (A B : ℕ)
noncomputable def M_A := 80 * A
noncomputable def M_B := 70 * B

axiom classA_condition1 : M_A - 160 = 90 * (A - 8)
axiom classB_condition1 : M_B - 180 = 85 * (B - 6)

-- Required proof in Lean 4 statement
theorem total_students : A + B = 78 :=
by
  sorry

end total_students_l79_79883


namespace find_a_plus_b_l79_79236

theorem find_a_plus_b (a b : ℝ)
  (h1 : ab^2 = 0)
  (h2 : 2 * a^2 * b = 0)
  (h3 : a^3 + b^2 = 0)
  (h4 : ab = 1) : a + b = -2 :=
sorry

end find_a_plus_b_l79_79236


namespace proof_problem_solution_l79_79291

noncomputable def proof_problem (a b c d : ℝ) : Prop :=
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ (a * b + b * c + c * d + d * a = 1) →
  (a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1 / 3)

theorem proof_problem_solution (a b c d : ℝ) : proof_problem a b c d :=
  sorry

end proof_problem_solution_l79_79291


namespace arrangement_of_BANANA_l79_79544

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l79_79544


namespace saving_percentage_l79_79693

variable (I S : Real)

-- Conditions
def cond1 : Prop := S = 0.3 * I -- Man saves 30% of his income

def cond2 : Prop := let income_next_year := 1.3 * I
                    let savings_next_year := 2 * S
                    let expenditure_first_year := I - S
                    let expenditure_second_year := income_next_year - savings_next_year
                    expenditure_first_year + expenditure_second_year = 2 * expenditure_first_year

-- Question
theorem saving_percentage :
  cond1 I S →
  cond2 I S →
  S = 0.3 * I :=
by
  intros
  sorry

end saving_percentage_l79_79693


namespace incorrect_statement_d_l79_79755

-- Definitions from the problem:
variables (x y : ℝ)
variables (b a : ℝ)
variables (x_bar y_bar : ℝ)

-- Linear regression equation:
def linear_regression (x y : ℝ) (b a : ℝ) : Prop :=
  y = b * x + a

-- Properties given in the problem:
axiom pass_through_point : ∀ (x_bar y_bar : ℝ), ∃ b a, y_bar = b * x_bar + a
axiom avg_increase : ∀ (b a : ℝ), y = b * (x + 1) + a → y = b * x + a + b
axiom possible_at_origin : ∀ (b a : ℝ), ∃ y, y = a

-- The statement D which is incorrect:
theorem incorrect_statement_d : ¬ (∀ (b a : ℝ), ∀ y, x = 0 → y = a) :=
sorry

end incorrect_statement_d_l79_79755


namespace num_integers_between_sqrt_range_l79_79461

theorem num_integers_between_sqrt_range :
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.card = 15 :=
by sorry

end num_integers_between_sqrt_range_l79_79461


namespace total_combinations_l79_79834

/-- Tim's rearrangement choices for the week -/
def monday_choices : Nat := 1
def tuesday_choices : Nat := 2
def wednesday_choices : Nat := 3
def thursday_choices : Nat := 2
def friday_choices : Nat := 1

theorem total_combinations :
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 12 :=
by
  sorry

end total_combinations_l79_79834


namespace cost_of_50_snacks_l79_79150

-- Definitions based on conditions
def travel_time_to_work : ℕ := 2 -- hours
def cost_of_snack : ℕ := 10 * (2 * travel_time_to_work) -- Ten times the round trip time

-- The theorem to prove
theorem cost_of_50_snacks : (50 * cost_of_snack) = 2000 := by
  sorry

end cost_of_50_snacks_l79_79150


namespace find_y_l79_79108

theorem find_y (y : ℝ) (a b : ℝ × ℝ) (h_a : a = (4, 2)) (h_b : b = (6, y)) (h_parallel : 4 * y - 2 * 6 = 0) :
  y = 3 :=
sorry

end find_y_l79_79108


namespace ratio_of_investments_l79_79183

-- Define the conditions
def ratio_of_profits (p q : ℝ) : Prop := 7/12 = (p * 5) / (q * 12)

-- Define the problem: given the conditions, prove the ratio of investments is 7/5
theorem ratio_of_investments (P Q : ℝ) (h : ratio_of_profits P Q) : P / Q = 7 / 5 :=
by
  sorry

end ratio_of_investments_l79_79183


namespace cylinder_volume_increase_l79_79131

variable (r h : ℝ)

theorem cylinder_volume_increase :
  (π * (4 * r) ^ 2 * (2 * h)) = 32 * (π * r ^ 2 * h) :=
by
  sorry

end cylinder_volume_increase_l79_79131


namespace polynomial_equivalence_l79_79754

-- Define the polynomial 'A' according to the conditions provided
def polynomial_A (x : ℝ) : ℝ := x^2 - 2*x

-- Define the given equation with polynomial A
def given_equation (x : ℝ) (A : ℝ) : Prop :=
  (x / (x + 2)) = (A / (x^2 - 4))

-- Prove that for the given equation, the polynomial 'A' is 'x^2 - 2x'
theorem polynomial_equivalence (x : ℝ) : given_equation x (polynomial_A x) :=
  by
    sorry -- Proof is skipped

end polynomial_equivalence_l79_79754


namespace composition_func_n_l79_79861

variable (a b x : ℝ) (n : ℕ)
hypothesis h1 : a ≠ 1

noncomputable def f (x : ℝ) : ℝ := a * x / (1 + b * x)

theorem composition_func_n (h1 : a ≠ 1) : 
  (f^[n] x) = a^n * x / (1 + (a^n - 1) / (a - 1) * b * x) := 
sorry

end composition_func_n_l79_79861


namespace cherry_sodas_in_cooler_l79_79689

theorem cherry_sodas_in_cooler (C : ℕ) (h1 : (C + 2 * C = 24)) : C = 8 :=
sorry

end cherry_sodas_in_cooler_l79_79689


namespace no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l79_79301

theorem no_perfect_squares_in_ap (n x : ℤ) : ¬(3 * n + 2 = x^2) :=
sorry

theorem infinitely_many_perfect_cubes_in_ap : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^3 :=
sorry

theorem no_terms_of_form_x_pow_2m (n x : ℤ) (m : ℕ) : 3 * n + 2 ≠ x^(2 * m) :=
sorry

theorem infinitely_many_terms_of_form_x_pow_2m_plus_1 (m : ℕ) : ∃ᶠ n in Filter.atTop, ∃ x : ℤ, 3 * n + 2 = x^(2 * m + 1) :=
sorry

end no_perfect_squares_in_ap_infinitely_many_perfect_cubes_in_ap_no_terms_of_form_x_pow_2m_infinitely_many_terms_of_form_x_pow_2m_plus_1_l79_79301


namespace susan_strawberries_l79_79649

def strawberries_picked (total_in_basket : ℕ) (handful_size : ℕ) (eats_per_handful : ℕ) : ℕ :=
  let strawberries_per_handful := handful_size - eats_per_handful
  (total_in_basket / strawberries_per_handful) * handful_size

theorem susan_strawberries : strawberries_picked 60 5 1 = 75 := by
  sorry

end susan_strawberries_l79_79649


namespace total_distance_covered_l79_79970

theorem total_distance_covered :
  let speed1 := 40 -- miles per hour
  let speed2 := 50 -- miles per hour
  let speed3 := 30 -- miles per hour
  let time1 := 1.5 -- hours
  let time2 := 1 -- hour
  let time3 := 2.25 -- hours
  let distance1 := speed1 * time1 -- distance covered in the first part of the trip
  let distance2 := speed2 * time2 -- distance covered in the second part of the trip
  let distance3 := speed3 * time3 -- distance covered in the third part of the trip
  distance1 + distance2 + distance3 = 177.5 := 
by
  sorry

end total_distance_covered_l79_79970


namespace concert_revenue_l79_79791

-- Define the prices and attendees
def adult_price := 26
def teenager_price := 18
def children_price := adult_price / 2
def num_adults := 183
def num_teenagers := 75
def num_children := 28

-- Calculate total revenue
def total_revenue := num_adults * adult_price + num_teenagers * teenager_price + num_children * children_price

-- The goal is to prove that total_revenue equals 6472
theorem concert_revenue : total_revenue = 6472 :=
by
  sorry

end concert_revenue_l79_79791


namespace find_value_of_f_at_1_l79_79601

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_value_of_f_at_1 (f : ℝ → ℝ)
  (h : ∀ x : ℝ, 2 * f x - f (- x) = 3 * x + 1) : f 1 = 2 :=
by
  sorry

end find_value_of_f_at_1_l79_79601


namespace time_for_tom_to_finish_wall_l79_79481

theorem time_for_tom_to_finish_wall (avery_rate tom_rate : ℝ) (combined_duration : ℝ) (remaining_wall : ℝ) :
  avery_rate = 1 / 2 ∧ tom_rate = 1 / 4 ∧ combined_duration = 1 ∧ remaining_wall = 1 / 4 →
  (remaining_wall / tom_rate) = 1 :=
by
  intros h
  -- Definitions from conditions
  let avery_rate := 1 / 2
  let tom_rate := 1 / 4
  let combined_duration := 1
  let remaining_wall := 1 / 4
  -- Question to be proven
  sorry

end time_for_tom_to_finish_wall_l79_79481


namespace fraction_d_can_be_zero_l79_79334

theorem fraction_d_can_be_zero :
  ∃ x : ℝ, (x + 1) / (x - 1) = 0 :=
by {
  sorry
}

end fraction_d_can_be_zero_l79_79334


namespace evaluate_b3_l79_79373

variable (b1 q : ℤ)
variable (b1_cond : b1 = 5 ∨ b1 = -5)
variable (q_cond : q = 3 ∨ q = -3)
def b3 : ℤ := b1 * q^2

theorem evaluate_b3 (h : b1^2 * (1 + q^2 + q^4) = 2275) : b3 = 45 ∨ b3 = -45 :=
by sorry

end evaluate_b3_l79_79373


namespace cube_vertices_count_l79_79489

-- Defining the conditions of the problem
def num_faces : ℕ := 6
def num_edges : ℕ := 12
def euler_formula (V E F : ℕ) : Prop := V - E + F = 2

-- Stating the proof problem
theorem cube_vertices_count : ∃ V : ℕ, euler_formula V num_edges num_faces ∧ V = 8 :=
by
  sorry

end cube_vertices_count_l79_79489


namespace inequality_proof_l79_79288

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l79_79288


namespace sum_of_two_numbers_l79_79660

theorem sum_of_two_numbers :
  ∃ x y : ℝ, (x * y = 9375 ∧ y / x = 15) ∧ (x + y = 400) :=
by
  sorry

end sum_of_two_numbers_l79_79660


namespace quotient_of_0_009_div_0_3_is_0_03_l79_79955

-- Statement:
theorem quotient_of_0_009_div_0_3_is_0_03 (x : ℝ) (h : x = 0.3) : 0.009 / x = 0.03 :=
by
  sorry

end quotient_of_0_009_div_0_3_is_0_03_l79_79955


namespace algebra_expr_solution_l79_79723

theorem algebra_expr_solution (a b : ℝ) (h : 2 * a - b = 5) : 2 * b - 4 * a + 8 = -2 :=
by
  sorry

end algebra_expr_solution_l79_79723


namespace solve_for_x_l79_79169

variable {x : ℝ}

theorem solve_for_x (h : (4 * x ^ 2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : x = 1 :=
sorry

end solve_for_x_l79_79169


namespace find_angle_between_altitude_and_median_l79_79884

noncomputable def angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : ℝ :=
  Real.arctan ((a^2 - b^2) / (4 * S))

theorem find_angle_between_altitude_and_median 
  (a b S : ℝ) (h1 : a > b) (h2 : S > 0) : 
  angle_between_altitude_and_median a b S h1 h2 = 
    Real.arctan ((a^2 - b^2) / (4 * S)) := 
  sorry

end find_angle_between_altitude_and_median_l79_79884


namespace original_faculty_members_l79_79206

theorem original_faculty_members
  (x : ℝ) (h : 0.87 * x = 195) : x = 224 := sorry

end original_faculty_members_l79_79206


namespace smallest_natural_b_for_root_exists_l79_79993

-- Define the problem's conditions
def quadratic_eqn (b : ℕ) := ∀ x : ℝ, x^2 + (b : ℝ) * x + 25 = 0

def discriminant (a b c : ℝ) := b^2 - 4 * a * c

-- Define the main problem statement
theorem smallest_natural_b_for_root_exists :
  ∃ b : ℕ, (discriminant 1 b 25 ≥ 0) ∧ (∀ b' : ℕ, b' < b → discriminant 1 b' 25 < 0) ∧ b = 10 :=
by
  sorry

end smallest_natural_b_for_root_exists_l79_79993


namespace tan_difference_of_angle_l79_79264

noncomputable def point_on_terminal_side (θ : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ (2, 3) = (k * Real.cos θ, k * Real.sin θ)

theorem tan_difference_of_angle (θ : ℝ) (hθ : point_on_terminal_side θ) :
  Real.tan (θ - Real.pi / 4) = 1 / 5 :=
sorry

end tan_difference_of_angle_l79_79264


namespace range_of_b_l79_79409

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := 3^x + b

theorem range_of_b (b : ℝ) :
  (∀ x : ℝ, f x b ≥ 0) ↔ b ≤ -1 :=
by sorry

end range_of_b_l79_79409


namespace probability_part_not_scrap_l79_79182

noncomputable def probability_not_scrap : Prop :=
  let p_scrap_first := 0.01
  let p_scrap_second := 0.02
  let p_not_scrap_first := 1 - p_scrap_first
  let p_not_scrap_second := 1 - p_scrap_second
  let p_not_scrap := p_not_scrap_first * p_not_scrap_second
  p_not_scrap = 0.9702

theorem probability_part_not_scrap : probability_not_scrap :=
by simp [probability_not_scrap] ; sorry

end probability_part_not_scrap_l79_79182


namespace bus_seat_problem_l79_79134

theorem bus_seat_problem 
  (left_seats : ℕ) 
  (right_seats := left_seats - 3) 
  (left_capacity := 3 * left_seats)
  (right_capacity := 3 * right_seats)
  (back_seat_capacity := 12)
  (total_capacity := left_capacity + right_capacity + back_seat_capacity)
  (h1 : total_capacity = 93) 
  : left_seats = 15 := 
by 
  sorry

end bus_seat_problem_l79_79134


namespace relationship_y1_y2_y3_l79_79739

theorem relationship_y1_y2_y3 
  (y_1 y_2 y_3 : ℝ)
  (h1 : y_1 = (-2)^2 + 2*(-2) + 2)
  (h2 : y_2 = (-1)^2 + 2*(-1) + 2)
  (h3 : y_3 = 2^2 + 2*2 + 2) :
  y_2 < y_1 ∧ y_1 < y_3 := 
sorry

end relationship_y1_y2_y3_l79_79739


namespace find_number_l79_79213

theorem find_number (x : ℝ) (h : 1345 - x / 20.04 = 1295) : x = 1002 :=
sorry

end find_number_l79_79213


namespace range_of_c_l79_79615

theorem range_of_c :
  (∃ (c : ℝ), ∀ (x y : ℝ), (x^2 + y^2 = 4) → ((12 * x - 5 * y + c) / 13 = 1))
  → (c > -13 ∧ c < 13) := 
sorry

end range_of_c_l79_79615


namespace mart_income_more_than_tim_l79_79292

variable (J : ℝ) -- Let's denote Juan's income as J
def T : ℝ := J - 0.40 * J -- Tim's income is 40 percent less than Juan's income
def M : ℝ := 0.78 * J -- Mart's income is 78 percent of Juan's income

theorem mart_income_more_than_tim : (M - T) / T * 100 = 30 := by
  sorry

end mart_income_more_than_tim_l79_79292


namespace square_carpet_side_length_l79_79497

theorem square_carpet_side_length (area : ℝ) (h : area = 10) :
  ∃ s : ℝ, s * s = area ∧ 3 < s ∧ s < 4 :=
by
  sorry

end square_carpet_side_length_l79_79497


namespace none_of_these_l79_79888

def table : List (ℕ × ℕ) := [(1, 5), (2, 15), (3, 33), (4, 61), (5, 101)]

def formula_A (x : ℕ) : ℕ := 2 * x^3 + 3 * x^2 - x + 1
def formula_B (x : ℕ) : ℕ := 3 * x^3 + x^2 + x + 1
def formula_C (x : ℕ) : ℕ := 2 * x^3 + x^2 + x + 1
def formula_D (x : ℕ) : ℕ := 2 * x^3 + x^2 + x - 1

theorem none_of_these :
  ¬ (∀ (x y : ℕ), (x, y) ∈ table → (y = formula_A x ∨ y = formula_B x ∨ y = formula_C x ∨ y = formula_D x)) :=
by {
  sorry
}

end none_of_these_l79_79888


namespace represent_259BC_as_neg259_l79_79032

def year_AD (n: ℤ) : ℤ := n

def year_BC (n: ℕ) : ℤ := -(n : ℤ)

theorem represent_259BC_as_neg259 : year_BC 259 = -259 := 
by 
  rw [year_BC]
  norm_num

end represent_259BC_as_neg259_l79_79032


namespace sample_size_community_A_l79_79369

variable (A B C H : ℕ) (total_families : ℕ) (sampling_ratio : ℚ)

def low_income_families_A := 360
def low_income_families_B := 270
def low_income_families_C := 180
def housing_units := 90

theorem sample_size_community_A (h1 : total_families = low_income_families_A + low_income_families_B + low_income_families_C)
  (h2 : sampling_ratio = housing_units / total_families) : 
  low_income_families_A * sampling_ratio = 40 :=
by
  rw [←h1, ←h2]
  sorry

end sample_size_community_A_l79_79369


namespace books_at_end_l79_79812

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l79_79812


namespace feathers_already_have_l79_79977

-- Given conditions
def total_feathers : Nat := 900
def feathers_still_needed : Nat := 513

-- Prove that the number of feathers Charlie already has is 387
theorem feathers_already_have : (total_feathers - feathers_still_needed) = 387 := by
  sorry

end feathers_already_have_l79_79977


namespace union_of_A_and_B_l79_79588

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l79_79588


namespace circles_chord_length_l79_79979

theorem circles_chord_length (r1 r2 r3 : ℕ) (m n p : ℕ) (h1 : r1 = 4) (h2 : r2 = 10) (h3 : r3 = 14)
(h4 : gcd m p = 1) (h5 : ¬ (∃ (k : ℕ), k^2 ∣ n)) : m + n + p = 19 :=
by
  sorry

end circles_chord_length_l79_79979


namespace arithmetic_sequence_sum_proof_l79_79286

theorem arithmetic_sequence_sum_proof
  (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h1 : S 17 = 170)
  (h2 : a 2000 = 2001)
  (h3 : ∀ n, S n = n * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2)
  (h4 : ∀ n, a (n + 1) = a n + (a 2 - a 1)) :
  S 2008 = 2019044 :=
  sorry

end arithmetic_sequence_sum_proof_l79_79286


namespace carbon_copies_after_folding_l79_79205

def initial_sheets : ℕ := 6
def initial_carbons (sheets : ℕ) : ℕ := sheets - 1
def final_copies (sheets : ℕ) : ℕ := sheets - 1

theorem carbon_copies_after_folding :
  (final_copies initial_sheets) =
  initial_carbons initial_sheets :=
by {
    -- sorry is a placeholder for the proof
    sorry
}

end carbon_copies_after_folding_l79_79205


namespace change_received_proof_l79_79346

-- Define the costs and amounts
def regular_ticket_cost : ℕ := 9
def children_ticket_discount : ℕ := 2
def amount_given : ℕ := 2 * 20

-- Define the number of people
def number_of_adults : ℕ := 2
def number_of_children : ℕ := 3

-- Define the costs calculations
def child_ticket_cost := regular_ticket_cost - children_ticket_discount
def total_adults_cost := number_of_adults * regular_ticket_cost
def total_children_cost := number_of_children * child_ticket_cost
def total_cost := total_adults_cost + total_children_cost
def change_received := amount_given - total_cost

-- Lean statement to prove the change received
theorem change_received_proof : change_received = 1 := by
  sorry

end change_received_proof_l79_79346


namespace max_average_growth_rate_l79_79790

theorem max_average_growth_rate 
  (P1 P2 : ℝ) (M : ℝ)
  (h1 : P1 + P2 = M) : 
  (1 + (M / 2))^2 ≥ (1 + P1) * (1 + P2) := 
by
  -- AM-GM Inequality application and other mathematical steps go here.
  sorry

end max_average_growth_rate_l79_79790


namespace extra_bananas_each_child_l79_79778

theorem extra_bananas_each_child (total_children absent_children planned_bananas_per_child : ℕ) 
    (h1 : total_children = 660) (h2 : absent_children = 330) (h3 : planned_bananas_per_child = 2) : (1320 / (total_children - absent_children)) - planned_bananas_per_child = 2 := by
  sorry

end extra_bananas_each_child_l79_79778


namespace polynomial_factor_l79_79095

theorem polynomial_factor (a b : ℝ) :
  (∃ (c d : ℝ), a = 4 * c ∧ b = -3 * c + 4 * d ∧ 40 = 2 * c - 3 * d + 18 ∧ -20 = 2 * d - 9 ∧ 9 = 9) →
  a = 11 ∧ b = -121 / 4 :=
by
  sorry

end polynomial_factor_l79_79095


namespace count_divisibles_in_range_l79_79599

theorem count_divisibles_in_range :
  let lower_bound := (2:ℤ)^10
  let upper_bound := (2:ℤ)^18
  let divisor := (2:ℤ)^9 
  (upper_bound - lower_bound) / divisor + 1 = 511 :=
by 
  sorry

end count_divisibles_in_range_l79_79599


namespace num_seating_arrangements_l79_79719

/-- Define the notion of people and seats with corresponding numbers. --/
set_option pp.portableStrings true
namespace SeatingArrangement

/-- Define the total number of people and seats --/
def numPeople := 5

/-- Define a function that counts the number of permutations with at most two fixed points --/
def countSeatingArrangementsWithAtMostTwoFixedPoints (n : ℕ) : ℕ :=
  let total := n! -- Total number of permutations
  let numThreeFixedPoints := Nat.choose n 3 -- Number of ways to choose 3 fixed points
  let numAllFixedPoints := 1 -- Only 1 way to have all fixed points
  total - numThreeFixedPoints - numAllFixedPoints -- Total - Three fixed points - All fixed points

theorem num_seating_arrangements (n : ℕ) (h : n = 5) :
  countSeatingArrangementsWithAtMostTwoFixedPoints n = 109 :=
by
  rw [h]
  unfold countSeatingArrangementsWithAtMostTwoFixedPoints
  norm_num
  sorry -- Proof elided

end num_seating_arrangements_l79_79719


namespace relationship_between_M_n_and_N_n_plus_2_l79_79954

theorem relationship_between_M_n_and_N_n_plus_2 (n : ℕ) (h : 2 ≤ n) :
  let M_n := (n * (n + 1)) / 2 + 1
  let N_n_plus_2 := n + 3
  M_n < N_n_plus_2 :=
by
  sorry

end relationship_between_M_n_and_N_n_plus_2_l79_79954


namespace evaluate_expression_l79_79337

theorem evaluate_expression : 7899665 - 12 * 3 * 2 = 7899593 :=
by
  -- This proof is skipped.
  sorry

end evaluate_expression_l79_79337


namespace calculate_angle_C_l79_79138

variable (A B C : ℝ)

theorem calculate_angle_C (h1 : A = C - 40) (h2 : B = 2 * A) (h3 : A + B + C = 180) :
  C = 75 :=
by
  sorry

end calculate_angle_C_l79_79138


namespace banana_arrangements_l79_79530

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l79_79530


namespace sum_of_multiples_is_even_l79_79305

theorem sum_of_multiples_is_even (a b : ℤ) (h1 : ∃ m : ℤ, a = 4 * m) (h2 : ∃ n : ℤ, b = 6 * n) : Even (a + b) :=
sorry

end sum_of_multiples_is_even_l79_79305


namespace multiple_of_pumpkins_l79_79233

theorem multiple_of_pumpkins (M S : ℕ) (hM : M = 14) (hS : S = 54) (h : S = x * M + 12) : x = 3 := sorry

end multiple_of_pumpkins_l79_79233


namespace fish_tank_problem_l79_79422

def number_of_fish_in_first_tank
  (F : ℕ)          -- Let F represent the number of fish in the first tank
  (twoF : ℕ)       -- Let twoF represent twice the number of fish in the first tank
  (total : ℕ) :    -- Let total represent the total number of fish
  Prop :=
  (2 * F = twoF)  -- The other two tanks each have twice as many fish as the first
  ∧ (F + twoF + twoF = total)  -- The sum of the fish in all three tanks equals the total number of fish

theorem fish_tank_problem
  (F : ℕ)
  (H : number_of_fish_in_first_tank F (2 * F) 100) : F = 20 :=
by
  sorry

end fish_tank_problem_l79_79422


namespace perm_banana_l79_79561

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l79_79561


namespace acute_angle_89_l79_79357

def is_acute_angle (angle : ℝ) : Prop := angle > 0 ∧ angle < 90

theorem acute_angle_89 :
  is_acute_angle 89 :=
by {
  -- proof details would go here, since only the statement is required
  sorry
}

end acute_angle_89_l79_79357


namespace binom_1300_2_eq_844350_l79_79363

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l79_79363


namespace special_op_eight_four_l79_79456

def special_op (a b : ℕ) : ℕ := 2 * a + a / b

theorem special_op_eight_four : special_op 8 4 = 18 := by
  sorry

end special_op_eight_four_l79_79456


namespace possible_distance_between_houses_l79_79893

variable (d : ℝ)

theorem possible_distance_between_houses (h_d1 : 1 ≤ d) (h_d2 : d ≤ 5) : 1 ≤ d ∧ d ≤ 5 :=
by
  exact ⟨h_d1, h_d2⟩

end possible_distance_between_houses_l79_79893


namespace jon_original_number_l79_79147

theorem jon_original_number :
  ∃ y : ℤ, (5 * (3 * y + 6) - 8 = 142) ∧ (y = 8) :=
sorry

end jon_original_number_l79_79147


namespace factor_expression_l79_79717

theorem factor_expression (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = 
    ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) :=
by
  sorry

end factor_expression_l79_79717


namespace cube_probability_l79_79716

noncomputable def probability_of_three_vertical_faces_same_color : ℚ :=
  73 / 243

theorem cube_probability :
  let colors := {0, 1, 2} -- Represent Red, Blue, Yellow
  let faces := Finset.range 6 -- Represent the 6 faces of the cube
  let total_arrangements := colors.card ^ faces.card
  let favorable_arrangements := 219
  (favorable_arrangements : ℚ) / total_arrangements = probability_of_three_vertical_faces_same_color :=
by
  sorry

end cube_probability_l79_79716


namespace range_of_m_l79_79997

theorem range_of_m (m : ℝ) (x : ℝ) :
  (¬ (|1 - (x - 1) / 3| ≤ 2) → ¬ (x^2 - 2 * x + (1 - m^2) ≤ 0)) → 
  (|m| ≥ 9) :=
by
  sorry

end range_of_m_l79_79997


namespace no_integer_b_for_four_integer_solutions_l79_79982

theorem no_integer_b_for_four_integer_solutions :
  ∀ (b : ℤ), ¬ ∃ x1 x2 x3 x4 : ℤ, 
    (x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4) ∧
    (∀ x : ℤ, (x^2 + b*x + 1 ≤ 0) ↔ (x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4)) :=
by sorry

end no_integer_b_for_four_integer_solutions_l79_79982


namespace tv_height_l79_79033

theorem tv_height (H : ℝ) : 
  672 / (24 * H) = (1152 / (48 * 32)) + 1 → 
  H = 16 := 
by
  have h_area_first_TV : 24 * H ≠ 0 := sorry
  have h_new_condition: 1152 / (48 * 32) + 1 = 1.75 := sorry
  have h_cost_condition: 672 / (24 * H) = 1.75 := sorry
  sorry

end tv_height_l79_79033


namespace sqrt_product_simplification_l79_79916

theorem sqrt_product_simplification : (ℝ) : 
  (Real.sqrt 18) * (Real.sqrt 72) = 12 * (Real.sqrt 2) :=
sorry

end sqrt_product_simplification_l79_79916


namespace text_messages_relationship_l79_79149

theorem text_messages_relationship (l x : ℕ) (h_l : l = 111) (h_combined : l + x = 283) : x = l + 61 :=
by sorry

end text_messages_relationship_l79_79149


namespace painter_completes_at_9pm_l79_79075

noncomputable def mural_completion_time (start_time : Nat) (fraction_completed_time : Nat)
    (fraction_completed : ℚ) : Nat :=
  let fraction_per_hour := fraction_completed / fraction_completed_time
  start_time + Nat.ceil (1 / fraction_per_hour)

theorem painter_completes_at_9pm :
  mural_completion_time 9 3 (1/4) = 21 := by
  sorry

end painter_completes_at_9pm_l79_79075


namespace simplify_expression_correct_l79_79168

def simplify_expression (y : ℝ) : ℝ :=
  (3 * y - 2) * (5 * y ^ 12 + 3 * y ^ 11 + 5 * y ^ 9 + y ^ 8)

theorem simplify_expression_correct (y : ℝ) :
  simplify_expression y = 15 * y ^ 13 - y ^ 12 + 6 * y ^ 11 + 5 * y ^ 10 - 7 * y ^ 9 - 2 * y ^ 8 :=
by
  sorry

end simplify_expression_correct_l79_79168


namespace evaluate_at_10_l79_79389

def f (x : ℝ) : ℝ := x^3 - 2 * x^2 - 5 * x + 6

theorem evaluate_at_10 : f 10 = 756 := by
  -- the proof is omitted
  sorry

end evaluate_at_10_l79_79389


namespace inconsistent_coordinates_l79_79419

theorem inconsistent_coordinates
  (m n : ℝ) 
  (h1 : m - (5/2)*n + 1 = 0) 
  (h2 : (m + 1/2) - (5/2)*(n + 1) + 1 = 0) :
  false :=
by
  sorry

end inconsistent_coordinates_l79_79419


namespace marble_244_is_white_l79_79494

noncomputable def color_of_marble (n : ℕ) : String :=
  let cycle := ["white", "white", "white", "white", "gray", "gray", "gray", "gray", "gray", "black", "black", "black"]
  cycle.get! (n % 12)

theorem marble_244_is_white : color_of_marble 244 = "white" :=
by
  sorry

end marble_244_is_white_l79_79494


namespace determine_fraction_l79_79312

noncomputable def q (x : ℝ) : ℝ := (x + 3) * (x - 2)

noncomputable def p (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem determine_fraction (a b : ℝ) (h : a + b = 1 / 4) :
  (p a b (-1)) / (q (-1)) = (a - b) / 4 :=
by
  sorry

end determine_fraction_l79_79312


namespace fraction_evaluation_l79_79372

theorem fraction_evaluation : 
  (1/4 - 1/6) / (1/3 - 1/5) = 5/8 := by
  sorry

end fraction_evaluation_l79_79372


namespace orensano_subset_count_l79_79697

def is_orensano (T : Finset ℤ) : Prop :=
  ∃ a b c, a < b ∧ b < c ∧ a ∈ T ∧ c ∈ T ∧ b ∉ T

theorem orensano_subset_count :
  let S := Finset.range 2020 in
  (Finset.powerset S).filter is_orensano).card = 2^2019 - 2039191 :=
begin
  sorry,
end

end orensano_subset_count_l79_79697


namespace g_h_of_2_eq_869_l79_79604

-- Define the functions g and h
def g (x : ℝ) : ℝ := 3 * x^2 + 2
def h (x : ℝ) : ℝ := -2 * x^3 - 1

-- State the theorem we need to prove
theorem g_h_of_2_eq_869 : g (h 2) = 869 := by
  sorry

end g_h_of_2_eq_869_l79_79604


namespace smallest_ninequality_l79_79578

theorem smallest_ninequality 
  (n : ℕ) 
  (h : ∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≤ 2 ^ (1 - n)) : 
  n = 2 := 
by
  sorry

end smallest_ninequality_l79_79578


namespace cherry_sodas_in_cooler_l79_79690

theorem cherry_sodas_in_cooler (C : ℕ) (h1 : (C + 2 * C = 24)) : C = 8 :=
sorry

end cherry_sodas_in_cooler_l79_79690


namespace prime_factors_of_expression_l79_79099

theorem prime_factors_of_expression
  (p : ℕ) (prime_p : Nat.Prime p) :
  (∀ x y : ℕ, 0 < x → 0 < y → p ∣ ((x + y)^19 - x^19 - y^19)) ↔ (p = 2 ∨ p = 3 ∨ p = 7 ∨ p = 19) :=
by
  sorry

end prime_factors_of_expression_l79_79099


namespace problem_I_ellipse_equation_problem_II_range_of_t_l79_79113

-- Problem (I) Proof statement
theorem problem_I_ellipse_equation (a b c : ℝ) (e : ℝ) (h1 : a > b)
  (h2 : b > 0) (h3 : e = 1/2)
  (h4 : a^2 = b^2 + c^2) (h5 : 1/2 * (2 * c) * b = sqrt 3) :
  (∃ (a b : ℝ), a = 2 ∧ b = sqrt 3 ∧ (a ^ 2 = 4 ∧ b ^ 2 = 3) ∧ 
  ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ 
  (x^2 / 4 + y^2 / 3 = 1)) :=
sorry

-- Problem (II) Proof statement
theorem problem_II_range_of_t (k t : ℝ) (h : t = k / (4 * k^2 + 3))
  (hx1 : ∀ k > 0, 4 * k + 3 / k ≥ 4 * (sqrt 3))
  (hx2 : ∀ k < 0, 4 * k + 3 / k ≤ -4 * (sqrt 3)) :
  t ∈ set.Icc (- sqrt 3 / 12) (sqrt 3 / 12) :=
sorry

end problem_I_ellipse_equation_problem_II_range_of_t_l79_79113


namespace beam_reflection_problem_l79_79486

theorem beam_reflection_problem
  (A B D C : Point)
  (angle_CDA : ℝ)
  (total_path_length_max : ℝ)
  (equal_angle_reflections : ∀ (k : ℕ), angle_CDA * k ≤ 90)
  (path_length_constraint : ∀ (n : ℕ) (d : ℝ), 2 * n * d ≤ total_path_length_max)
  : angle_CDA = 5 ∧ total_path_length_max = 100 → ∃ (n : ℕ), n = 10 :=
sorry

end beam_reflection_problem_l79_79486


namespace arithmetic_sequence_nth_term_l79_79617

-- Definitions from conditions
def a₁ : ℕ → ℤ := -60
def a₁₇  : ℕ → ℤ := -12

-- Lean statement

theorem arithmetic_sequence_nth_term (n : ℕ) :
  (∃ d, d = 3 ∧ ∀ n : ℕ, a₁₇ = a₁ + 16 * d ∧ (∀ n : ℕ, a_n = -60 + 3 * (n - 1))) ∧
  sum_abs_first_30_terms : (∃ S, S = 765 ∧ ∑ i in range 1..31, abs (a_n(i)) = 765) :=
sorry

end arithmetic_sequence_nth_term_l79_79617


namespace yeast_cells_at_2_20_pm_l79_79285

noncomputable def yeast_population (initial : Nat) (rate : Nat) (intervals : Nat) : Nat :=
  initial * rate ^ intervals

theorem yeast_cells_at_2_20_pm :
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5 -- 20 minutes / 4 minutes per interval
  yeast_population initial_population triple_rate intervals = 7290 :=
by
  let initial_population := 30
  let triple_rate := 3
  let intervals := 5
  show yeast_population initial_population triple_rate intervals = 7290
  sorry

end yeast_cells_at_2_20_pm_l79_79285


namespace pizza_consumption_order_l79_79412

noncomputable def amount_eaten (fraction: ℚ) (total: ℚ) := fraction * total

theorem pizza_consumption_order :
  let total := 1
  let samuel := (1 / 6 : ℚ)
  let teresa := (2 / 5 : ℚ)
  let uma := (1 / 4 : ℚ)
  let victor := total - (samuel + teresa + uma)
  let samuel_eaten := amount_eaten samuel 60
  let teresa_eaten := amount_eaten teresa 60
  let uma_eaten := amount_eaten uma 60
  let victor_eaten := amount_eaten victor 60
  (teresa_eaten > uma_eaten) 
  ∧ (uma_eaten > victor_eaten) 
  ∧ (victor_eaten > samuel_eaten) := 
by
  sorry

end pizza_consumption_order_l79_79412


namespace amy_hours_per_week_school_year_l79_79973

variable (hours_per_week_summer : ℕ)
variable (weeks_summer : ℕ)
variable (earnings_summer : ℕ)
variable (additional_earnings_needed : ℕ)
variable (weeks_school_year : ℕ)
variable (hourly_wage : ℝ := earnings_summer / (hours_per_week_summer * weeks_summer))

theorem amy_hours_per_week_school_year :
  hours_per_week_school_year = (additional_earnings_needed / hourly_wage) / weeks_school_year :=
by 
  -- Using the hourly wage and total income needed, calculate the hours.
  let total_hours_needed := additional_earnings_needed / hourly_wage
  have h1 : hours_per_week_school_year = total_hours_needed / weeks_school_year := sorry
  exact h1

end amy_hours_per_week_school_year_l79_79973


namespace banana_arrangement_count_l79_79552

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l79_79552


namespace arithmetic_mean_difference_l79_79058

theorem arithmetic_mean_difference (p q r : ℝ)
  (h1 : (p + q) / 2 = 10)
  (h2 : (q + r) / 2 = 22) :
  r - p = 24 :=
by
  sorry

end arithmetic_mean_difference_l79_79058


namespace average_consecutive_pairs_is_correct_l79_79670

/-- Definition of the set S and nCr function -/
def S : Finset ℕ := Finset.range (33 - 5 + 1) + 5
def nCr (n r : ℕ) : ℕ := (Finset.powersetLen r (Finset.range n)).card

/-- Definitions of the specific combinatorial calculations -/
def omega : ℕ := nCr 29 4
def single_pair : ℕ := 4 * nCr 26 2
def two_pairs : ℕ := 3 * nCr 27 1
def three_pairs : ℕ := 26
def average_consecutive_pairs : ℚ := (single_pair + two_pairs + three_pairs) / omega

/-- The proof problem itself -/
theorem average_consecutive_pairs_is_correct :
  average_consecutive_pairs = 0.0648 := by
  sorry

end average_consecutive_pairs_is_correct_l79_79670


namespace distinct_arrangements_of_BANANA_l79_79558

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l79_79558


namespace jackson_earned_on_monday_l79_79420

-- Definitions
def goal := 1000
def tuesday_earnings := 40
def avg_rate := 10
def houses := 88
def days_remaining := 3
def total_collected_remaining_days := days_remaining * (houses / 4) * avg_rate

-- The proof problem statement
theorem jackson_earned_on_monday (m : ℕ) :
  m + tuesday_earnings + total_collected_remaining_days = goal → m = 300 :=
by
  -- We will eventually provide the proof here
  sorry

end jackson_earned_on_monday_l79_79420


namespace cube_volume_of_surface_area_l79_79343

theorem cube_volume_of_surface_area (S : ℝ) (V : ℝ) (a : ℝ) (h1 : S = 150) (h2 : S = 6 * a^2) (h3 : V = a^3) : V = 125 := by
  sorry

end cube_volume_of_surface_area_l79_79343


namespace intersection_M_N_l79_79433

open Set

noncomputable def M : Set ℕ := {x | x < 6}
noncomputable def N : Set ℕ := {x | x^2 - 11 * x + 18 < 0}

theorem intersection_M_N : M ∩ N = {3, 4, 5} := by
  sorry

end intersection_M_N_l79_79433


namespace fixed_point_coordinates_l79_79485

noncomputable def fixed_point (A : Real × Real) : Prop :=
∀ (k : Real), ∃ (x y : Real), A = (x, y) ∧ (3 + k) * x + (1 - 2 * k) * y + 1 + 5 * k = 0

theorem fixed_point_coordinates :
  fixed_point (-1, 2) :=
by
  sorry

end fixed_point_coordinates_l79_79485


namespace probability_A_not_winning_l79_79667

theorem probability_A_not_winning 
  (prob_draw : ℚ := 1/2)
  (prob_B_wins : ℚ := 1/3) : 
  (prob_draw + prob_B_wins) = 5 / 6 := 
by
  sorry

end probability_A_not_winning_l79_79667


namespace rectangular_plot_area_l79_79453

theorem rectangular_plot_area (breadth length : ℕ) (h1 : breadth = 14) (h2 : length = 3 * breadth) : (length * breadth) = 588 := 
by 
  -- imports, noncomputable keyword, and placeholder proof for compilation
  sorry

end rectangular_plot_area_l79_79453


namespace division_multiplication_identity_l79_79704

theorem division_multiplication_identity :
  24 / (-6) * (3 / 2) / (- (4 / 3)) = 9 / 2 := 
by 
  sorry

end division_multiplication_identity_l79_79704


namespace ratio_of_larger_to_smaller_l79_79795

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l79_79795


namespace evaluate_expression_l79_79098

theorem evaluate_expression (x z : ℤ) (hx : x = 4) (hz : z = -2) : 
  z * (z - 4 * x) = 36 := by
  sorry

end evaluate_expression_l79_79098


namespace find_y1_y2_over_y0_l79_79157

variable (p : ℝ) (h : 0 < p)
variable (y1 y2 y0 : ℝ)
variable (k : ℝ) (hk : k ≠ 0) 
variable (A B : ℝ × ℝ) 
variable (Focus : ℝ × ℝ := ((p / 2), 0))
variable (P O : ℝ × ℝ) (hO : O = (0, 0)) (hy0_O : P ≠ O)
variable (concyclic : circline.concyclic ({P, A, B, O} : finset (ℝ × ℝ)))
variable (coords : A.2 = y1 ∧ B.2 = y2 ∧ P.2 = y0)

theorem find_y1_y2_over_y0 :
  (\frac{y1 + y2}{y0}) = 4 :=
sorry

end find_y1_y2_over_y0_l79_79157


namespace average_salary_proof_l79_79447

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l79_79447


namespace initial_number_of_girls_l79_79579

theorem initial_number_of_girls (b g : ℤ) 
  (h1 : b = 3 * (g - 20)) 
  (h2 : 3 * (b - 30) = g - 20) : 
  g = 31 :=
by
  sorry

end initial_number_of_girls_l79_79579


namespace profit_percentage_l79_79974

theorem profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) 
  (h1 : cost_price = 66.5) (h2 : marked_price = 87.5) (h3 : discount_rate = 0.05) : 
  (100 * ((marked_price * (1 - discount_rate) - cost_price) / cost_price)) = 25 :=
by
  sorry

end profit_percentage_l79_79974


namespace order_DABC_l79_79232

-- Definitions of the variables given in the problem
def A : ℕ := 77^7
def B : ℕ := 7^77
def C : ℕ := 7^7^7
def D : ℕ := Nat.factorial 7

-- The theorem stating the required ascending order
theorem order_DABC : D < A ∧ A < B ∧ B < C :=
by sorry

end order_DABC_l79_79232


namespace arrangement_of_BANANA_l79_79546

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l79_79546


namespace banana_permutations_l79_79543

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l79_79543


namespace BANANA_arrangements_l79_79519

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l79_79519


namespace max_angle_OAB_l79_79164

/-- Let OA = a, OB = b, and OM = x on the right angle XOY, where a < b. 
    The value of x which maximizes the angle ∠AMB is sqrt(ab). -/
theorem max_angle_OAB (a b x : ℝ) (h : a < b) (h1 : x = Real.sqrt (a * b)) :
  x = Real.sqrt (a * b) :=
sorry

end max_angle_OAB_l79_79164


namespace BANANA_arrangements_l79_79522

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l79_79522


namespace problem_statement_l79_79008

theorem problem_statement (x y : ℝ) (h : |x + 1| + |y + 2 * x| = 0) : (x + y) ^ 2004 = 1 := by
  sorry

end problem_statement_l79_79008


namespace quadrilateral_inequality_l79_79767

theorem quadrilateral_inequality
  (AB AC BD CD: ℝ)
  (h1 : AB + BD ≤ AC + CD)
  (h2 : AB + CD < AC + BD) :
  AB < AC := by
  sorry

end quadrilateral_inequality_l79_79767


namespace projection_correct_l79_79576

open Real

def vec1 : Vector ℝ := ⟨[5, -3, 2]⟩
def dir : Vector ℝ := ⟨[4, -3, 2]⟩

noncomputable def projection : Vector ℝ :=
  let dot_uv := vec1.dot_product dir
  let dot_vv := dir.dot_product dir
  let scalar := dot_uv / dot_vv
  ⟨dir.1.map (λ x => x * scalar)⟩

theorem projection_correct : projection = ⟨[132 / 29, -99 / 29, 66 / 29]⟩ := by
  sorry

end projection_correct_l79_79576


namespace train_length_calculation_l79_79701

noncomputable def train_length (speed_km_hr : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_km_hr * 1000 / 3600) * time_sec

theorem train_length_calculation :
  train_length 250 6 = 416.67 :=
by
  sorry

end train_length_calculation_l79_79701


namespace base_number_mod_100_l79_79194

theorem base_number_mod_100 (base : ℕ) (h : base ^ 8 % 100 = 1) : base = 1 := 
sorry

end base_number_mod_100_l79_79194


namespace number_of_rectangles_in_5x5_grid_l79_79000

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l79_79000


namespace hiking_time_l79_79626

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l79_79626


namespace seq_a8_value_l79_79499

theorem seq_a8_value 
  (a : ℕ → ℕ) 
  (h1 : ∀ n : ℕ, a n < a (n + 1)) 
  (h2 : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n) 
  (a7_eq : a 7 = 120) 
  : a 8 = 194 :=
sorry

end seq_a8_value_l79_79499


namespace gcd_78_36_l79_79669

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := 
by
  sorry

end gcd_78_36_l79_79669


namespace complex_division_example_l79_79308

theorem complex_division_example : (2 : ℂ) / (I * (3 - I)) = (1 - 3 * I) / 5 := 
by {
  sorry
}

end complex_division_example_l79_79308


namespace evaluate_expression_l79_79933

theorem evaluate_expression : 6 + 4 / 2 = 8 :=
by
  sorry

end evaluate_expression_l79_79933


namespace sequence_geometric_sum_bn_l79_79112

theorem sequence_geometric (a : ℕ → ℕ) (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) : 
  (∀ n, a n = 2^n) :=
by sorry

theorem sum_bn (a : ℕ → ℕ) (b : ℕ → ℕ) (S : ℕ → ℕ)
  (h_recurrence : ∀ n ≥ 2, (a n)^2 = (a (n - 1)) * (a (n + 1)))
  (h_init1 : a 2 + 2 * a 1 = 4) (h_init2 : (a 3)^2 = a 5) 
  (h_gen : ∀ n, a n = 2^n) (h_bn : ∀ n, b n = n * a n) :
  (∀ n, S n = (n-1) * 2^(n+1) + 2) :=
by sorry

end sequence_geometric_sum_bn_l79_79112


namespace pyarelal_loss_l79_79057

theorem pyarelal_loss (total_loss : ℝ) (P : ℝ) (Ashok_capital : ℝ) (ratio_Ashok_Pyarelal : ℝ) :
  total_loss = 670 →
  Ashok_capital = P / 9 →
  ratio_Ashok_Pyarelal = 1 / 9 →
  Pyarelal_loss = 603 :=
by
  intro total_loss_eq Ashok_capital_eq ratio_eq
  sorry

end pyarelal_loss_l79_79057


namespace smallest_circle_area_l79_79055

/-- The smallest possible area of a circle passing through two given points in the coordinate plane. -/
theorem smallest_circle_area (P Q : ℝ × ℝ) (hP : P = (-3, -2)) (hQ : Q = (2, 4)) : 
  ∃ (A : ℝ), A = (61 * Real.pi) / 4 :=
by
  sorry

end smallest_circle_area_l79_79055


namespace baby_polar_bear_playing_hours_l79_79415

-- Define the conditions
def total_hours_in_a_day : ℕ := 24
def total_central_angle : ℕ := 360
def angle_sleeping : ℕ := 130
def angle_eating : ℕ := 110

-- Main theorem statement
theorem baby_polar_bear_playing_hours :
  let angle_playing := total_central_angle - angle_sleeping - angle_eating
  let fraction_playing := angle_playing / total_central_angle
  let hours_playing := fraction_playing * total_hours_in_a_day
  hours_playing = 8 := by
  sorry

end baby_polar_bear_playing_hours_l79_79415


namespace condition_A_condition_B_condition_C_condition_D_correct_answer_l79_79310

theorem condition_A : ∀ x : ℝ, x^2 + 2 * x - 1 ≠ x * (x + 2) - 1 := sorry

theorem condition_B : ∀ a b : ℝ, (a + b)^2 = a^2 + 2 * a * b + b^2 := sorry

theorem condition_C : ∀ x y : ℝ, x^2 - 4 * y^2 = (x + 2 * y) * (x - 2 * y) := sorry

theorem condition_D : ∀ a b : ℝ, a^2 - a * b - a ≠ a * (a - b) := sorry

theorem correct_answer : ∀ x y : ℝ, (x^2 - 4 * y^2) = (x + 2 * y) * (x - 2 * y) := 
  by 
    exact condition_C

end condition_A_condition_B_condition_C_condition_D_correct_answer_l79_79310


namespace coplanar_lines_l79_79331

def vector3 := ℝ × ℝ × ℝ

def vec1 : vector3 := (2, -1, 3)
def vec2 (k : ℝ) : vector3 := (3 * k, 1, 2)
def pointVec : vector3 := (-3, 2, -3)

def det3x3 (a b c d e f g h i : ℝ) : ℝ :=
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

theorem coplanar_lines (k : ℝ) : det3x3 2 (-1) 3 (3 * k) 1 2 (-3) 2 (-3) = 0 → k = -29 / 9 :=
  sorry

end coplanar_lines_l79_79331


namespace hiking_time_l79_79627

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l79_79627


namespace correct_time_after_2011_minutes_l79_79200

def time_2011_minutes_after_midnight : String :=
  "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM"

theorem correct_time_after_2011_minutes :
  time_2011_minutes_after_midnight = "2011 minutes after midnight on January 1, 2011 is January 2 at 9:31AM" :=
sorry

end correct_time_after_2011_minutes_l79_79200


namespace travel_time_l79_79220

theorem travel_time (distance speed : ℕ) (h_distance : distance = 810) (h_speed : speed = 162) :
  distance / speed = 5 :=
by
  sorry

end travel_time_l79_79220


namespace binom_12_11_eq_12_l79_79510

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_eq_12_l79_79510


namespace elizabeth_net_profit_l79_79097

-- Define the conditions
def cost_per_bag : ℝ := 3.00
def bags_produced : ℕ := 20
def selling_price_per_bag : ℝ := 6.00
def bags_sold_full_price : ℕ := 15
def discount_percentage : ℝ := 0.25

-- Define the net profit computation
def net_profit : ℝ :=
  let revenue_full_price := bags_sold_full_price * selling_price_per_bag
  let remaining_bags := bags_produced - bags_sold_full_price
  let discounted_price_per_bag := selling_price_per_bag * (1 - discount_percentage)
  let revenue_discounted := remaining_bags * discounted_price_per_bag
  let total_revenue := revenue_full_price + revenue_discounted
  let total_cost := bags_produced * cost_per_bag
  total_revenue - total_cost

theorem elizabeth_net_profit : net_profit = 52.50 := by
  sorry

end elizabeth_net_profit_l79_79097


namespace simplify_sqrt_product_l79_79917

theorem simplify_sqrt_product :
  sqrt 18 * sqrt 72 = 36 :=
sorry

end simplify_sqrt_product_l79_79917


namespace binom_identity1_binom_identity2_l79_79037

variable (n k : ℕ)

theorem binom_identity1 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) + (Nat.choose n (k + 1)) = (Nat.choose (n + 1) (k + 1)) :=
sorry

theorem binom_identity2 (hn : n > 0) (hk : k > 0) :
  (Nat.choose n k) = (n * Nat.choose (n - 1) (k - 1)) / k :=
sorry

end binom_identity1_binom_identity2_l79_79037


namespace problem_statement_l79_79859

theorem problem_statement (k x₁ x₂ : ℝ) (hx₁x₂ : x₁ < x₂)
  (h_eq : ∀ x : ℝ, x^2 - (k - 3) * x + (k + 4) = 0) 
  (P : ℝ) (hP : P ≠ 0) 
  (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2) 
  (hacute : ∀ A B : ℝ, A = x₁ ∧ B = x₂ ∧ A < 0 ∧ B > 0) :
  k < -4 ∧ α ≠ β ∧ α < β := 
sorry

end problem_statement_l79_79859


namespace solve_xyz_eq_x_plus_y_l79_79919

theorem solve_xyz_eq_x_plus_y (x y z : ℕ) (h1 : x * y * z = x + y) (h2 : x ≤ y) : (x = 2 ∧ y = 2 ∧ z = 1) ∨ (x = 1 ∧ y = 1 ∧ z = 2) :=
by {
    sorry -- The actual proof goes here
}

end solve_xyz_eq_x_plus_y_l79_79919


namespace flour_needed_l79_79774

-- Definitions
def cups_per_loaf := 2.5
def loaves := 2

-- Statement we want to prove
theorem flour_needed {cups_per_loaf loaves : ℝ} (h : cups_per_loaf = 2.5) (l : loaves = 2) : 
  cups_per_loaf * loaves = 5 :=
sorry

end flour_needed_l79_79774


namespace range_of_x_l79_79992

variable {x p : ℝ}

theorem range_of_x (H : 0 ≤ p ∧ p ≤ 4) : 
  (x^2 + p * x > 4 * x + p - 3) ↔ (x ∈ Set.Iio (-1) ∪ Set.Ioi 3) := 
by
  sorry

end range_of_x_l79_79992


namespace books_from_first_shop_l79_79036

theorem books_from_first_shop (x : ℕ) (h : (2080 : ℚ) / (x + 50) = 18.08695652173913) : x = 65 :=
by
  -- proof steps
  sorry

end books_from_first_shop_l79_79036


namespace find_second_number_l79_79322

def sum_of_three (a b c : ℚ) : Prop :=
  a + b + c = 120

def ratio_first_to_second (a b : ℚ) : Prop :=
  a / b = 3 / 4

def ratio_second_to_third (b c : ℚ) : Prop :=
  b / c = 3 / 5

theorem find_second_number (a b c : ℚ) 
  (h_sum : sum_of_three a b c)
  (h_ratio_ab : ratio_first_to_second a b)
  (h_ratio_bc : ratio_second_to_third b c) : 
  b = 1440 / 41 := 
sorry

end find_second_number_l79_79322


namespace randy_biscuits_left_l79_79641

-- Define the function biscuits_left
def biscuits_left (initial: ℚ) (father_gift: ℚ) (mother_gift: ℚ) (brother_eat_percent: ℚ) : ℚ :=
  let total_before_eat := initial + father_gift + mother_gift
  let brother_ate := brother_eat_percent * total_before_eat
  total_before_eat - brother_ate

-- Given conditions
def initial_biscuits : ℚ := 32
def father_gift : ℚ := 2 / 3
def mother_gift : ℚ := 15
def brother_eat_percent : ℚ := 0.3

-- Correct answer as an approximation since we're dealing with real-world numbers
def approx (x y : ℚ) := abs (x - y) < 0.01

-- The proof problem statement in Lean 4
theorem randy_biscuits_left :
  approx (biscuits_left initial_biscuits father_gift mother_gift brother_eat_percent) 33.37 :=
by
  sorry

end randy_biscuits_left_l79_79641


namespace polynomial_coefficients_sum_l79_79256

theorem polynomial_coefficients_sum (a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (2*x - 3)^5 = a_0 + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 →
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 160 :=
by
  sorry

end polynomial_coefficients_sum_l79_79256


namespace parallelogram_circumference_l79_79665

-- Defining the conditions
def isParallelogram (a b : ℕ) := a = 18 ∧ b = 12

-- The theorem statement to prove
theorem parallelogram_circumference (a b : ℕ) (h : isParallelogram a b) : 2 * (a + b) = 60 :=
  by
  -- Extract the conditions from hypothesis
  cases h with
  | intro hab' hab'' =>
    sorry

end parallelogram_circumference_l79_79665


namespace unique_solution_l79_79248

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l79_79248


namespace probability_two_same_color_l79_79802

-- Definitions based on conditions
def total_pairs := 14
def blue_pairs := 8
def red_pairs := 4
def green_pairs := 2

def total_socks := 2 * total_pairs
def blue_socks := 2 * blue_pairs
def red_socks := 2 * red_pairs
def green_socks := 2 * green_pairs

-- Statement to prove: the probability that two randomly picked socks are of the same color
theorem probability_two_same_color : 
  ( (blue_socks * (blue_socks - 1) + red_socks * (red_socks - 1) + green_socks * (green_socks - 1)) / (total_socks * (total_socks - 1)) = (77 / 189 : ℚ) ) :=
begin
  sorry
end

end probability_two_same_color_l79_79802


namespace max_colors_l79_79459

theorem max_colors (n : ℕ) (color : ℕ → ℕ → ℕ)
  (h_color_property : ∀ i j : ℕ, i < 2^n → j < 2^n → color i j = color j ((i + j) % 2^n)) :
  ∃ (c : ℕ), c ≤ 2^n ∧ (∀ i j : ℕ, i < 2^n → j < 2^n → color i j < c) :=
sorry

end max_colors_l79_79459


namespace martin_total_distance_l79_79570

noncomputable def calculate_distance_traveled : ℕ :=
  let segment1 := 70 * 3 -- 210 km
  let segment2 := 80 * 4 -- 320 km
  let segment3 := 65 * 3 -- 195 km
  let segment4 := 50 * 2 -- 100 km
  let segment5 := 90 * 4 -- 360 km
  segment1 + segment2 + segment3 + segment4 + segment5

theorem martin_total_distance : calculate_distance_traveled = 1185 :=
by
  sorry

end martin_total_distance_l79_79570


namespace distinct_arrangements_of_BANANA_l79_79557

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l79_79557


namespace tom_trip_cost_l79_79936

-- Definitions of hourly rates
def rate_6AM_to_10AM := 10
def rate_10AM_to_2PM := 12
def rate_2PM_to_6PM := 15
def rate_6PM_to_10PM := 20

-- Definitions of trip start times and durations
def first_trip_start := 8
def second_trip_start := 14
def third_trip_start := 20

-- Function to calculate the cost for each trip segment
def cost (start_hour : Nat) (duration : Nat) : Nat :=
  if start_hour >= 6 ∧ start_hour < 10 then duration * rate_6AM_to_10AM
  else if start_hour >= 10 ∧ start_hour < 14 then duration * rate_10AM_to_2PM
  else if start_hour >= 14 ∧ start_hour < 18 then duration * rate_2PM_to_6PM
  else if start_hour >= 18 ∧ start_hour < 22 then duration * rate_6PM_to_10PM
  else 0

-- Function to calculate the total trip cost
def total_cost : Nat :=
  cost first_trip_start 2 + cost (first_trip_start + 2) 2 +
  cost second_trip_start 4 +
  cost third_trip_start 4

-- Proof statement
theorem tom_trip_cost : total_cost = 184 := by
  -- The detailed steps of the proof would go here. Replaced with 'sorry' presently to indicate incomplete proof.
  sorry

end tom_trip_cost_l79_79936


namespace find_value_of_expression_l79_79792

theorem find_value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 4 * y + 13) : (12 * y - 5)^2 = 161 :=
sorry

end find_value_of_expression_l79_79792


namespace BANANA_arrangements_l79_79517

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l79_79517


namespace product_of_two_numbers_l79_79788

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := 
by 
  sorry

end product_of_two_numbers_l79_79788


namespace total_profit_is_27_l79_79437

noncomputable def total_profit : ℕ :=
  let natasha_money := 60
  let carla_money := natasha_money / 3
  let cosima_money := carla_money / 2
  let sergio_money := 3 * cosima_money / 2

  let natasha_spent := 4 * 15
  let carla_spent := 6 * 10
  let cosima_spent := 5 * 8
  let sergio_spent := 3 * 12

  let natasha_profit := natasha_spent * 10 / 100
  let carla_profit := carla_spent * 15 / 100
  let cosima_profit := cosima_spent * 12 / 100
  let sergio_profit := sergio_spent * 20 / 100

  natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_is_27 : total_profit = 27 := by
  sorry

end total_profit_is_27_l79_79437


namespace unique_solution_l79_79247

theorem unique_solution (x : ℝ) : (2:ℝ)^x + (3:ℝ)^x + (6:ℝ)^x = (7:ℝ)^x ↔ x = 2 :=
by
  sorry

end unique_solution_l79_79247


namespace solution_is_63_l79_79939

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)
def last_digit (n : ℕ) : ℕ := n % 10
def rhyming_primes_around (r : ℕ) : Prop :=
  r >= 1 ∧ r <= 100 ∧
  ¬ is_prime r ∧
  ∃ ps : List ℕ, (∀ p ∈ ps, is_prime p ∧ last_digit p = last_digit r) ∧
  (∀ q : ℕ, is_prime q ∧ last_digit q = last_digit r → q ∈ ps) ∧
  List.length ps = 4

theorem solution_is_63 : ∃ r : ℕ, rhyming_primes_around r ∧ r = 63 :=
by sorry

end solution_is_63_l79_79939


namespace relationship_l79_79909

-- Definitions for the points on the inverse proportion function
def on_inverse_proportion (x : ℝ) (y : ℝ) : Prop :=
  y = -6 / x

-- Given conditions
def A (y1 : ℝ) : Prop :=
  on_inverse_proportion (-3) y1

def B (y2 : ℝ) : Prop :=
  on_inverse_proportion (-1) y2

def C (y3 : ℝ) : Prop :=
  on_inverse_proportion (2) y3

-- The theorem that expresses the relationship
theorem relationship (y1 y2 y3 : ℝ) (hA : A y1) (hB : B y2) (hC : C y3) : y3 < y1 ∧ y1 < y2 :=
by
  -- skeleton of proof
  sorry

end relationship_l79_79909


namespace count_perfect_cubes_l79_79394

theorem count_perfect_cubes (a b : ℤ) (h₁ : 100 < a) (h₂ : b < 1000) : 
  ∃ n m : ℤ, (n^3 > 100 ∧ m^3 < 1000) ∧ m - n + 1 = 5 := 
by
  -- The proof is omitted and replaced with sorry.
  sorry

end count_perfect_cubes_l79_79394


namespace total_feathers_needed_l79_79105

theorem total_feathers_needed
  (animals_first_group : ℕ := 934)
  (feathers_first_group : ℕ := 7)
  (animals_second_group : ℕ := 425)
  (colored_feathers_second_group : ℕ := 7)
  (golden_feathers_second_group : ℕ := 5)
  (animals_third_group : ℕ := 289)
  (colored_feathers_third_group : ℕ := 4)
  (golden_feathers_third_group : ℕ := 10) :
  (animals_first_group * feathers_first_group) +
  (animals_second_group * (colored_feathers_second_group + golden_feathers_second_group)) +
  (animals_third_group * (colored_feathers_third_group + golden_feathers_third_group)) = 15684 := by
  sorry

end total_feathers_needed_l79_79105


namespace exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l79_79071

noncomputable def quadratic_sequence (n : ℕ) (a : ℕ → ℤ) :=
  ∀i : ℕ, 1 ≤ i ∧ i ≤ n → abs (a i - a (i - 1)) = i * i

theorem exists_quadratic_sequence_for_any_b_c (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ quadratic_sequence n a := by
  sorry

theorem smallest_n_for_quadratic_sequence_0_to_2021 :
  ∃ n : ℕ, 0 < n ∧ ∀ (a : ℕ → ℤ), a 0 = 0 → a n = 2021 → quadratic_sequence n a := by
  sorry

end exists_quadratic_sequence_for_any_b_c_smallest_n_for_quadratic_sequence_0_to_2021_l79_79071


namespace inequality_holds_l79_79300

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (y * z)) + (y^3 / (z * x)) + (z^3 / (x * y)) ≥ x + y + z :=
by
  sorry

end inequality_holds_l79_79300


namespace permutations_of_BANANA_l79_79525

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l79_79525


namespace product_of_0_5_and_0_8_l79_79376

theorem product_of_0_5_and_0_8 : (0.5 * 0.8) = 0.4 := by
  sorry

end product_of_0_5_and_0_8_l79_79376


namespace remaining_money_l79_79826

def initial_amount : Float := 499.9999999999999

def spent_on_clothes (initial : Float) : Float :=
  (1/3) * initial

def remaining_after_clothes (initial : Float) : Float :=
  initial - spent_on_clothes initial

def spent_on_food (remaining_clothes : Float) : Float :=
  (1/5) * remaining_clothes

def remaining_after_food (remaining_clothes : Float) : Float :=
  remaining_clothes - spent_on_food remaining_clothes

def spent_on_travel (remaining_food : Float) : Float :=
  (1/4) * remaining_food

def remaining_after_travel (remaining_food : Float) : Float :=
  remaining_food - spent_on_travel remaining_food

theorem remaining_money :
  remaining_after_travel (remaining_after_food (remaining_after_clothes initial_amount)) = 199.99 :=
by
  sorry

end remaining_money_l79_79826


namespace days_of_harvest_l79_79871

-- Conditions
def ripeOrangesPerDay : ℕ := 82
def totalRipeOranges : ℕ := 2050

-- Problem statement: Prove the number of days of harvest
theorem days_of_harvest : (totalRipeOranges / ripeOrangesPerDay) = 25 :=
by
  sorry

end days_of_harvest_l79_79871


namespace tan_half_angle_eq_neg2_l79_79430

-- Given conditions
variable (a : ℝ) (ha : a ∈ Set.Ioo (π / 2) π) (hcos : Real.cos a = -3 / 5)

-- Theorem statement
theorem tan_half_angle_eq_neg2 : Real.tan (a / 2) = -2 :=
sorry

end tan_half_angle_eq_neg2_l79_79430


namespace repeating_decimal_rational_representation_l79_79490

theorem repeating_decimal_rational_representation :
  (0.12512512512512514 : ℝ) = (125 / 999 : ℝ) :=
sorry

end repeating_decimal_rational_representation_l79_79490


namespace total_amount_paid_l79_79638

-- Definitions based on the conditions in step a)
def ring_cost : ℕ := 24
def ring_quantity : ℕ := 2

-- Statement to prove that the total cost is $48.
theorem total_amount_paid : ring_quantity * ring_cost = 48 := 
by
  sorry

end total_amount_paid_l79_79638


namespace solve_equation_l79_79920

theorem solve_equation (x y z : ℤ) (h : 19 * (x + y) + z = 19 * (-x + y) - 21) (hx : x = 1) : z = -59 := by
  sorry

end solve_equation_l79_79920


namespace add_least_number_l79_79044

theorem add_least_number (n : ℕ) (h1 : n = 1789) (h2 : ∃ k : ℕ, 5 * k = n + 11) (h3 : ∃ j : ℕ, 6 * j = n + 11) (h4 : ∃ m : ℕ, 4 * m = n + 11) (h5 : ∃ l : ℕ, 11 * l = n + 11) : 11 = 11 :=
by
  sorry

end add_least_number_l79_79044


namespace paving_stones_correct_l79_79738

def paving_stone_area : ℕ := 3 * 2
def courtyard_breadth : ℕ := 6
def number_of_paving_stones : ℕ := 15
def courtyard_length : ℕ := 15

theorem paving_stones_correct : 
  number_of_paving_stones * paving_stone_area = courtyard_length * courtyard_breadth :=
by
  sorry

end paving_stones_correct_l79_79738


namespace bhanu_house_rent_l79_79090

theorem bhanu_house_rent (I : ℝ) 
  (h1 : 0.30 * I = 300) 
  (h2 : 210 = 210) : 
  210 / (I - 300) = 0.30 := 
by 
  sorry

end bhanu_house_rent_l79_79090


namespace percentage_reduction_in_production_l79_79070

theorem percentage_reduction_in_production :
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  percentage_reduction = 10 :=
by
  let daily_production_rate := 10
  let days_in_year := 365
  let total_production_first_year := daily_production_rate * days_in_year
  let total_production_second_year := 3285
  let reduction_in_production := total_production_first_year - total_production_second_year
  let percentage_reduction := (reduction_in_production * 100) / total_production_first_year
  sorry

end percentage_reduction_in_production_l79_79070


namespace books_left_on_Fri_l79_79811

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l79_79811


namespace blue_balls_taken_out_l79_79801

theorem blue_balls_taken_out (x : ℕ) :
  ∀ (total_balls : ℕ) (initial_blue_balls : ℕ)
    (remaining_probability : ℚ),
    total_balls = 25 ∧ initial_blue_balls = 9 ∧ remaining_probability = 1/5 →
    (9 - x : ℚ) / (25 - x : ℚ) = 1/5 →
    x = 5 :=
by
  intros total_balls initial_blue_balls remaining_probability
  rintro ⟨h_total_balls, h_initial_blue_balls, h_remaining_probability⟩ h_eq
  -- Proof goes here
  sorry

end blue_balls_taken_out_l79_79801


namespace intersection_complement_l79_79391

open Set

variable (x : ℝ)

def M : Set ℝ := { x | -1 < x ∧ x < 2 }
def N : Set ℝ := { x | 1 ≤ x }

theorem intersection_complement :
  M ∩ (univ \ N) = { x | -1 < x ∧ x < 1 } := by
  sorry

end intersection_complement_l79_79391


namespace train_length_proof_l79_79967

noncomputable def train_length (speed_km_per_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_per_s := (speed_km_per_hr * 1000) / 3600 in
  speed_m_per_s * time_sec

theorem train_length_proof :
  train_length 60 6 = 100.02 :=
by
  sorry

end train_length_proof_l79_79967


namespace interest_rate_is_10_percent_l79_79963

theorem interest_rate_is_10_percent
  (principal : ℝ)
  (interest_rate_c : ℝ) 
  (time : ℝ)
  (gain_b : ℝ)
  (interest_c : ℝ := principal * interest_rate_c / 100 * time)
  (interest_a : ℝ := interest_c - gain_b)
  (expected_rate : ℝ := (interest_a / (principal * time)) * 100)
  (h1: principal = 3500)
  (h2: interest_rate_c = 12)
  (h3: time = 3)
  (h4: gain_b = 210)
  : expected_rate = 10 := 
  by 
  sorry

end interest_rate_is_10_percent_l79_79963


namespace find_first_dimension_l79_79696

variable (w h cost_per_sqft total_cost : ℕ)

def surface_area (l w h : ℕ) : ℕ := 2 * l * w + 2 * l * h + 2 * w * h

def insulation_cost (A cost_per_sqft : ℕ) : ℕ := A * cost_per_sqft

theorem find_first_dimension 
  (w := 7) (h := 2) (cost_per_sqft := 20) (total_cost := 1640) : 
  (∃ l : ℕ, insulation_cost (surface_area l w h) cost_per_sqft = total_cost) → 
  l = 3 := 
sorry

end find_first_dimension_l79_79696


namespace mean_equality_and_find_y_l79_79315

theorem mean_equality_and_find_y : 
  (8 + 9 + 18) / 3 = (15 + (25 / 3)) / 2 :=
by
  sorry

end mean_equality_and_find_y_l79_79315


namespace banana_arrangement_count_l79_79549

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l79_79549


namespace arrangement_of_BANANA_l79_79545

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l79_79545


namespace cows_horses_ratio_l79_79500

theorem cows_horses_ratio (cows horses : ℕ) (h : cows = 21) (ratio : cows / horses = 7 / 2) : horses = 6 :=
sorry

end cows_horses_ratio_l79_79500


namespace factor_poly_l79_79400

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l79_79400


namespace sum_of_integers_ending_in_2_between_100_and_600_l79_79362

theorem sum_of_integers_ending_in_2_between_100_and_600 :
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  ∃ S : ℤ, S = n * (a + l) / 2 ∧ S = 17350 := 
by
  let a := 102
  let d := 10
  let l := 592
  let n := (l - a) / d + 1
  use n * (a + l) / 2
  sorry

end sum_of_integers_ending_in_2_between_100_and_600_l79_79362


namespace wheel_rpm_l79_79209

noncomputable def radius : ℝ := 175
noncomputable def speed_kmh : ℝ := 66
noncomputable def speed_cmm := speed_kmh * 100000 / 60 -- convert from km/h to cm/min
noncomputable def circumference := 2 * Real.pi * radius -- circumference of the wheel
noncomputable def rpm := speed_cmm / circumference -- revolutions per minute

theorem wheel_rpm : rpm = 1000 := by
  sorry

end wheel_rpm_l79_79209


namespace arithmetic_seq_solution_l79_79731

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Definition of arithmetic sequence
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Sum of first n terms of arithmetic sequence
def sum_arithmetic_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = (n + 1) / 2 * (a 0 + a n)

-- Given conditions
def given_conditions (a : ℕ → ℝ) : Prop :=
  a 0 + a 4 + a 8 = 27

-- Main theorem to be proved
theorem arithmetic_seq_solution (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (ha : arithmetic_seq a d)
  (hs : sum_arithmetic_seq S a)
  (h_given : given_conditions a) :
  a 4 = 9 ∧ S 8 = 81 :=
sorry

end arithmetic_seq_solution_l79_79731


namespace find_b_l79_79452

def point := ℝ × ℝ

def dir_vector (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def scale_vector (v : point) (s : ℝ) : point := (s * v.1, s * v.2)

theorem find_b (p1 p2 : point) (b : ℝ) :
  p1 = (-5, 0) → p2 = (-2, 2) →
  dir_vector p1 p2 = (3, 2) →
  scale_vector (3, 2) (2 / 3) = (2, b) →
  b = 4 / 3 :=
by
  intros h1 h2 h3 h4
  sorry

end find_b_l79_79452


namespace min_value_fraction_l79_79253

theorem min_value_fraction (x : ℝ) (h : x > 0) : ∃ y, y = 4 ∧ (∀ z, z = (x + 5) / Real.sqrt (x + 1) → y ≤ z) := sorry

end min_value_fraction_l79_79253


namespace arithmetic_sequence_sum_l79_79903

def f (x : ℝ) : ℝ := (x - 3) ^ 3 + x - 1

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d ≠ 0, ∀ n : ℕ, a (n + 1) = a n + d

-- Problem Statement
theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_f_sum : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) + f (a 7) = 14) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 21 :=
sorry

end arithmetic_sequence_sum_l79_79903


namespace card_toss_sum_one_probability_l79_79470

noncomputable def cardTossProbability : ℂ :=
  let outcomes := [{0, 0}, {0, 1}, {1, 0}, {1, 1}] in
  let favorable := [0, 1] ∪ [1, 0] in
  (favorable.length / outcomes.length : ℂ)

theorem card_toss_sum_one_probability :
  cardTossProbability = 1/2 :=
sorry

end card_toss_sum_one_probability_l79_79470


namespace sum_of_interior_angles_of_polygon_l79_79744

theorem sum_of_interior_angles_of_polygon (exterior_angle : ℝ) (h : exterior_angle = 36) :
  ∃ interior_sum : ℝ, interior_sum = 1440 :=
by
  sorry

end sum_of_interior_angles_of_polygon_l79_79744


namespace problem_l79_79741

theorem problem
  (x y : ℝ)
  (h1 : x + 3 * y = 9)
  (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 :=
sorry

end problem_l79_79741


namespace min_value_inequality_l79_79765

open Real

theorem min_value_inequality (a b c : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ 47 :=
sorry

end min_value_inequality_l79_79765


namespace solve_log_equation_l79_79444

theorem solve_log_equation (x : ℝ) 
  (h1 : 7 * x + 3 > 0)
  (h2 : 4 * x + 5 > 0) :
  (log (sqrt (7 * x + 3)) + log (sqrt (4 * x + 5)) = 1 / 2 + log 3) ↔ x = 1 := 
begin
  sorry
end

end solve_log_equation_l79_79444


namespace min_a2_b2_c2_l79_79431

theorem min_a2_b2_c2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 3 * b + 5 * c = 100) : 
  a^2 + b^2 + c^2 ≥ (5000 / 19) :=
by
  sorry

end min_a2_b2_c2_l79_79431


namespace pradeep_pass_percentage_l79_79779

-- Define the given data as constants
def score : ℕ := 185
def shortfall : ℕ := 25
def maxMarks : ℕ := 840

-- Calculate the passing mark
def passingMark : ℕ := score + shortfall

-- Calculate the percentage needed to pass
def passPercentage (passingMark : ℕ) (maxMarks : ℕ) : ℕ :=
  (passingMark * 100) / maxMarks

-- Statement of the theorem that we aim to prove
theorem pradeep_pass_percentage (score shortfall maxMarks : ℕ)
  (h_score : score = 185) (h_shortfall : shortfall = 25) (h_maxMarks : maxMarks = 840) :
  passPercentage (score + shortfall) maxMarks = 25 :=
by
  -- This is where the proof would go
  sorry

-- Example of calling the function to ensure definitions are correct
#eval passPercentage (score + shortfall) maxMarks -- Should output 25

end pradeep_pass_percentage_l79_79779


namespace solve_system_l79_79646

theorem solve_system (x y z a b c : ℝ)
  (h1 : x * (x + y + z) = a^2)
  (h2 : y * (x + y + z) = b^2)
  (h3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ x = -a^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ y = -b^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (z = c^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) :=
by
  sorry

end solve_system_l79_79646


namespace other_root_of_quadratic_l79_79260

theorem other_root_of_quadratic 
  (a b c: ℝ) 
  (h : a * (b - c - d) * (1:ℝ)^2 + b * (c - a + d) * (1:ℝ) + c * (a - b - d) = 0) : 
  ∃ k: ℝ, k = c * (a - b - d) / (a * (b - c - d)) :=
sorry

end other_root_of_quadratic_l79_79260


namespace arithmetic_progression_x_value_l79_79928

theorem arithmetic_progression_x_value :
  ∃ x : ℝ, (2 * x - 1) + ((5 * x + 6) - (3 * x + 4)) = (3 * x + 4) + ((3 * x + 4) - (2 * x - 1)) ∧ x = 3 :=
by
  sorry

end arithmetic_progression_x_value_l79_79928


namespace max_length_MN_l79_79457

theorem max_length_MN (p : ℝ) (h a b c r : ℝ)
  (h_perimeter : a + b + c = 2 * p)
  (h_tangent : r = (a * h) / (2 * p))
  (h_parallel : ∀ h r : ℝ, ∃ k : ℝ, MN = k * (1 - 2 * r / h)) :
  ∀ k : ℝ, MN = (p / 4) :=
sorry

end max_length_MN_l79_79457


namespace batsman_average_increase_l79_79218

theorem batsman_average_increase
  (prev_avg : ℝ) -- average before the 17th innings
  (total_runs_16 : ℝ := 16 * prev_avg) -- total runs scored in the first 16 innings
  (score_17th : ℝ := 85) -- score in the 17th innings
  (new_avg : ℝ := 37) -- new average after 17 innings
  (total_runs_17 : ℝ := total_runs_16 + score_17th) -- total runs after 17 innings
  (calc_total_runs_17 : ℝ := 17 * new_avg) -- new total runs calculated by the new average
  (h : total_runs_17 = calc_total_runs_17) -- given condition: total_runs_17 = calc_total_runs_17
  : (new_avg - prev_avg) = 3 := 
by
  sorry

end batsman_average_increase_l79_79218


namespace find_a_l79_79999

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {-4, a - 1, a + 1}

theorem find_a (a : ℝ) (h : A a ∩ B a = {-2}) : a = -1 :=
sorry

end find_a_l79_79999


namespace change_factor_w_l79_79130

theorem change_factor_w (w d z F_w : Real)
  (h_q : ∀ w d z, q = 5 * w / (4 * d * z^2))
  (h1 : d' = 2 * d)
  (h2 : z' = 3 * z)
  (h3 : F_q = 0.2222222222222222)
  : F_w = 4 :=
by
  sorry

end change_factor_w_l79_79130


namespace ratio_of_larger_to_smaller_l79_79796

theorem ratio_of_larger_to_smaller (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + y = 7 * (x - y)) : x / y = 4 / 3 := 
by
  sorry

end ratio_of_larger_to_smaller_l79_79796


namespace factor_poly_l79_79399

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l79_79399


namespace molecular_weight_compound_l79_79197

/-- Definition of atomic weights for elements H, Cr, and O in AMU (Atomic Mass Units) --/
def atomic_weight_H : ℝ := 1.008
def atomic_weight_Cr : ℝ := 51.996
def atomic_weight_O : ℝ := 15.999

/-- Proof statement to calculate the molecular weight of a compound with 2 H, 1 Cr, and 4 O --/
theorem molecular_weight_compound :
  2 * atomic_weight_H + 1 * atomic_weight_Cr + 4 * atomic_weight_O = 118.008 :=
by
  sorry

end molecular_weight_compound_l79_79197


namespace f_g_2_equals_169_l79_79012

-- Definitions of f and g
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := 2 * x^2 + x + 3

-- The theorem statement
theorem f_g_2_equals_169 : f (g 2) = 169 :=
by
  sorry

end f_g_2_equals_169_l79_79012


namespace percent_difference_l79_79137

def boys := 100
def girls := 125
def diff := girls - boys
def boys_less_than_girls_percent := (diff : ℚ) / girls  * 100
def girls_more_than_boys_percent := (diff : ℚ) / boys  * 100

theorem percent_difference :
  boys_less_than_girls_percent = 20 ∧ girls_more_than_boys_percent = 25 :=
by
  -- The proof here demonstrates the percentage calculations.
  sorry

end percent_difference_l79_79137


namespace BANANA_arrangements_l79_79515

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l79_79515


namespace part_a_part_b_l79_79948

-- Part (a)

theorem part_a : ∃ (a b : ℕ), 2015^2 + 2017^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

-- Part (b)

theorem part_b (k n : ℕ) : ∃ (a b : ℕ), (2 * k + 1)^2 + (2 * n + 1)^2 = 2 * (a^2 + b^2) :=
by
  -- The proof will go here
  sorry

end part_a_part_b_l79_79948


namespace BANANA_arrangements_l79_79518

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l79_79518


namespace books_left_on_Fri_l79_79809

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l79_79809


namespace find_p_q_l79_79392

variable (R : Set ℝ)

def A (p : ℝ) : Set ℝ := {x | x^2 + p * x + 12 = 0}
def B (q : ℝ) : Set ℝ := {x | x^2 - 5 * x + q = 0}

theorem find_p_q 
  (h : (R \ (A p)) ∩ (B q) = {2}) : p + q = -1 :=
by
  sorry

end find_p_q_l79_79392


namespace kamari_toys_eq_65_l79_79087

-- Define the number of toys Kamari has
def number_of_toys_kamari_has : ℕ := sorry

-- Define the number of toys Anais has in terms of K
def number_of_toys_anais_has (K : ℕ) : ℕ := K + 30

-- Define the total number of toys
def total_number_of_toys (K A : ℕ) := K + A

-- Prove that the number of toys Kamari has is 65
theorem kamari_toys_eq_65 : ∃ K : ℕ, (number_of_toys_anais_has K) = K + 30 ∧ total_number_of_toys K (number_of_toys_anais_has K) = 160 ∧ K = 65 :=
by
  sorry

end kamari_toys_eq_65_l79_79087


namespace height_of_regular_triangular_pyramid_l79_79794

-- Problem Statement: Given conditions and correct answer
theorem height_of_regular_triangular_pyramid
  (a : ℝ)
  (ABC : Triangle)
  (h_reg : is_equilateral ABC)
  (P : Point)
  (M : Point := centroid ABC)
  (P_on_lateral : ∀ (X : Point), X ∈ ABC.vertices → ⦞ (angle (P -ᵥ M) (X -ᵥ M)) = 60) :
  
  height_of_pyramid ABC P = a :=
begin
  sorry
end

end height_of_regular_triangular_pyramid_l79_79794


namespace inequality_proof_l79_79290

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1 / a)^3 + (b + 1 / b)^3 + (c + 1 / c)^3 = 1000 / 9 :=
sorry

end inequality_proof_l79_79290


namespace vector2d_propositions_l79_79887

-- Define the vector structure in ℝ²
structure Vector2D where
  x : ℝ
  y : ℝ

-- Define the relation > on Vector2D
def Vector2D.gt (a1 a2 : Vector2D) : Prop :=
  a1.x > a2.x ∨ (a1.x = a2.x ∧ a1.y > a2.y)

-- Define vectors e1, e2, and 0
def e1 : Vector2D := ⟨ 1, 0 ⟩
def e2 : Vector2D := ⟨ 0, 1 ⟩
def zero : Vector2D := ⟨ 0, 0 ⟩

-- Define propositions
def prop1 : Prop := Vector2D.gt e1 e2 ∧ Vector2D.gt e2 zero
def prop2 (a1 a2 a3 : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt a2 a3 → Vector2D.gt a1 a3
def prop3 (a1 a2 a : Vector2D) : Prop := Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a1.x + a.x) (a1.y + a.y)) (Vector2D.mk (a2.x + a.x) (a2.y + a.y))
def prop4 (a a1 a2 : Vector2D) : Prop := Vector2D.gt a zero → Vector2D.gt a1 a2 → Vector2D.gt (Vector2D.mk (a.x * a1.x + a.y * a1.y) (0)) (Vector2D.mk (a.x * a2.x + a.y * a2.y) 0)

-- Main theorem to prove
theorem vector2d_propositions : prop1 ∧ (∀ a1 a2 a3, prop2 a1 a2 a3) ∧ (∀ a1 a2 a, prop3 a1 a2 a) := 
by
  sorry

end vector2d_propositions_l79_79887


namespace sequence_sum_100_eq_200_l79_79727

theorem sequence_sum_100_eq_200
  (a : ℕ → ℕ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h3 : a 3 = 2)
  (h4 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) ≠ 1)
  (h5 : ∀ n : ℕ, a n * a (n + 1) * a (n + 2) * a (n + 3) = a n + a (n + 1) + a (n + 2) + a (n + 3)) :
  (Finset.range 100).sum (a ∘ Nat.succ) = 200 := by
  sorry

end sequence_sum_100_eq_200_l79_79727


namespace length_OB_max_volume_l79_79655

-- Given conditions as definitions
variables {P A B H C O : ℝ}
def is_isosceles_right_triangle (P A B C : ℝ) : Prop := sorry
def perpendicular (x y : ℝ) : Prop := sorry -- a stub for perpendicular property
def on_circumference (A : ℝ) : Prop := sorry -- a stub for point on circumference

-- The main theorem statement
theorem length_OB_max_volume (P A B H C O : ℝ) (h1 : is_isosceles_right_triangle P A B C) 
  (h2 : on_circumference A) (h3 : perpendicular A B) (h4 : perpendicular O B)
  (h5 : perpendicular O H) (h6 : PA = 4) (h7 : C = (P + A) / 2) :
  ∃ OB : ℝ, OB = (2 * real.sqrt 6) / 3 := 
sorry

end length_OB_max_volume_l79_79655


namespace chris_remaining_money_l79_79709

variable (video_game_cost : ℝ)
variable (discount_rate : ℝ)
variable (candy_cost : ℝ)
variable (tax_rate : ℝ)
variable (shipping_fee : ℝ)
variable (hourly_rate : ℝ)
variable (hours_worked : ℝ)

noncomputable def remaining_money (video_game_cost discount_rate candy_cost tax_rate shipping_fee hourly_rate hours_worked : ℝ) : ℝ :=
  let discount := discount_rate * video_game_cost
  let discounted_price := video_game_cost - discount
  let total_video_game_cost := discounted_price + shipping_fee
  let video_tax := tax_rate * total_video_game_cost
  let candy_tax := tax_rate * candy_cost
  let total_cost := (total_video_game_cost + video_tax) + (candy_cost + candy_tax)
  let earnings := hourly_rate * hours_worked
  earnings - total_cost

theorem chris_remaining_money : remaining_money 60 0.15 5 0.10 3 8 9 = 7.1 :=
by
  sorry

end chris_remaining_money_l79_79709


namespace diana_owes_amount_l79_79679

def principal : ℝ := 75
def rate : ℝ := 0.07
def time : ℝ := 1
def interest : ℝ := principal * rate * time
def total_amount_owed : ℝ := principal + interest

theorem diana_owes_amount :
  total_amount_owed = 80.25 :=
by
  sorry

end diana_owes_amount_l79_79679


namespace annie_hamburgers_l79_79703

theorem annie_hamburgers (H : ℕ) (h₁ : 4 * H + 6 * 5 = 132 - 70) : H = 8 := by
  sorry

end annie_hamburgers_l79_79703


namespace find_x_l79_79674

theorem find_x (x y : ℤ) (h₁ : x + 3 * y = 10) (h₂ : y = 3) : x = 1 := 
by
  sorry

end find_x_l79_79674


namespace probability_point_closer_to_origin_than_4_1_l79_79965

-- Define the rectangular region
def region : set (ℝ × ℝ) :=
  { p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1 }

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the point (4,1)
def point_4_1 : ℝ × ℝ := (4, 1)

-- Function to calculate Euclidean distance
def euclidean_distance (a b : ℝ × ℝ) : ℝ :=
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)

-- Probability statement to be proven
theorem probability_point_closer_to_origin_than_4_1 : 
  measure_theory.measure.region (λ p, euclidean_distance p origin < euclidean_distance p point_4_1) / 
  measure_theory.measure.region region = 2 / 3 :=
sorry

end probability_point_closer_to_origin_than_4_1_l79_79965


namespace isosceles_triangle_base_length_l79_79359

theorem isosceles_triangle_base_length (a b c : ℕ) (h_isosceles : a = b ∨ b = c ∨ c = a)
  (h_perimeter : a + b + c = 16) (h_side_length : a = 6 ∨ b = 6 ∨ c = 6) :
  (a = 4 ∨ b = 4 ∨ c = 4) ∨ (a = 6 ∨ b = 6 ∨ c = 6) :=
sorry

end isosceles_triangle_base_length_l79_79359


namespace number_of_rectangles_in_5x5_grid_l79_79004

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l79_79004


namespace solution_set_of_quadratic_inequality_l79_79662

theorem solution_set_of_quadratic_inequality (a : ℝ) (x : ℝ) :
  (∀ x, 0 < x - 0.5 ∧ x < 2 → ax^2 + 5 * x - 2 > 0) ∧ a = -2 →
  (∀ x, -3 < x ∧ x < 0.5 → a * x^2 - 5 * x + a^2 - 1 > 0) :=
by
  sorry

end solution_set_of_quadratic_inequality_l79_79662


namespace frog_jumps_within_distance_l79_79225

noncomputable section

open ProbTheory

def frog_jumps_probability (n : ℕ) (radius : ℝ) (jump_length : ℝ) : ℝ :=
  -- Here, we'd define the probability based on random walk theory
  sorry -- actual implementation would go here

theorem frog_jumps_within_distance :
  frog_jumps_probability 4 1.5 1 = 1 / 3 :=
sorry -- proof would go here

end frog_jumps_within_distance_l79_79225


namespace no_divisor_30_to_40_of_2_pow_28_minus_1_l79_79983

theorem no_divisor_30_to_40_of_2_pow_28_minus_1 :
  ¬ ∃ n : ℕ, (30 ≤ n ∧ n ≤ 40 ∧ n ∣ (2^28 - 1)) :=
by
  sorry

end no_divisor_30_to_40_of_2_pow_28_minus_1_l79_79983


namespace integer_count_of_sqrt_x_l79_79463

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l79_79463


namespace probability_all_blue_jellybeans_removed_l79_79214

def num_red_jellybeans : ℕ := 10
def num_blue_jellybeans : ℕ := 10
def total_jellybeans : ℕ := num_red_jellybeans + num_blue_jellybeans

def prob_first_blue : ℚ := num_blue_jellybeans / total_jellybeans
def prob_second_blue : ℚ := (num_blue_jellybeans - 1) / (total_jellybeans - 1)
def prob_third_blue : ℚ := (num_blue_jellybeans - 2) / (total_jellybeans - 2)

def prob_all_blue : ℚ := prob_first_blue * prob_second_blue * prob_third_blue

theorem probability_all_blue_jellybeans_removed :
  prob_all_blue = 1 / 9.5 := sorry

end probability_all_blue_jellybeans_removed_l79_79214


namespace margarets_mean_score_l79_79994

noncomputable def mean (scores : List ℝ) : ℝ :=
  scores.sum / scores.length

open List

theorem margarets_mean_score :
  let scores := [86, 88, 91, 93, 95, 97, 99, 100]
  let cyprians_mean := 92
  let num_scores := 8
  let cyprians_scores := 4
  let margarets_scores := num_scores - cyprians_scores
  (scores.sum - cyprians_scores * cyprians_mean) / margarets_scores = 95.25 :=
by
  sorry

end margarets_mean_score_l79_79994


namespace moving_circle_trajectory_l79_79407

theorem moving_circle_trajectory (x y : ℝ) 
  (fixed_circle : x^2 + y^2 = 4): 
  (x^2 + y^2 = 9) ∨ (x^2 + y^2 = 1) :=
sorry

end moving_circle_trajectory_l79_79407


namespace number_of_valid_n_l79_79094

-- The definition for determining the number of positive integers n ≤ 2000 that can be represented as
-- floor(x) + floor(4x) + floor(5x) = n for some real number x.

noncomputable def count_valid_n : ℕ :=
  (200 : ℕ) * 3 + (200 : ℕ) * 2 + 1 + 1

theorem number_of_valid_n : count_valid_n = 802 :=
  sorry

end number_of_valid_n_l79_79094


namespace daria_needs_to_earn_more_money_l79_79842

noncomputable def moneyNeeded (ticket_cost : ℕ) (discount : ℕ) (gift_card : ℕ) 
  (transport_cost : ℕ) (parking_cost : ℕ) (tshirt_cost : ℕ) (current_money : ℕ) (tickets : ℕ) : ℕ :=
  let discounted_ticket_price := ticket_cost - (ticket_cost * discount / 100)
  let total_ticket_cost := discounted_ticket_price * tickets
  let ticket_cost_after_gift_card := total_ticket_cost - gift_card
  let total_cost := ticket_cost_after_gift_card + transport_cost + parking_cost + tshirt_cost
  total_cost - current_money

theorem daria_needs_to_earn_more_money :
  moneyNeeded 90 10 50 20 10 25 189 6 = 302 :=
by
  sorry

end daria_needs_to_earn_more_money_l79_79842


namespace total_students_is_45_l79_79022

def num_students_in_class 
  (excellent_chinese : ℕ) 
  (excellent_math : ℕ) 
  (excellent_both : ℕ) 
  (no_excellent : ℕ) : ℕ :=
  excellent_chinese + excellent_math - excellent_both + no_excellent

theorem total_students_is_45 
  (h1 : excellent_chinese = 15)
  (h2 : excellent_math = 18)
  (h3 : excellent_both = 8)
  (h4 : no_excellent = 20) : 
  num_students_in_class excellent_chinese excellent_math excellent_both no_excellent = 45 := 
  by 
    sorry

end total_students_is_45_l79_79022


namespace jaewoong_ran_the_most_l79_79026

def distance_jaewoong : ℕ := 20000 -- Jaewoong's distance in meters
def distance_seongmin : ℕ := 2600  -- Seongmin's distance in meters
def distance_eunseong : ℕ := 5000  -- Eunseong's distance in meters

theorem jaewoong_ran_the_most : distance_jaewoong > distance_seongmin ∧ distance_jaewoong > distance_eunseong := by
  sorry

end jaewoong_ran_the_most_l79_79026


namespace soccer_team_points_l79_79139

theorem soccer_team_points
  (x y : ℕ)
  (h1 : x + y = 8)
  (h2 : 3 * x - y = 12) : 
  (x + y = 8 ∧ 3 * x - y = 12) :=
by
  exact ⟨h1, h2⟩

end soccer_team_points_l79_79139


namespace find_circle_center_l79_79065

def circle_center_condition (x y : ℝ) : Prop :=
  (3 * x - 4 * y = 24 ∨ 3 * x - 4 * y = -12) ∧ 3 * x + 2 * y = 0

theorem find_circle_center :
  ∃ (x y : ℝ), circle_center_condition x y ∧ (x, y) = (2/3, -1) :=
by
  sorry

end find_circle_center_l79_79065


namespace remainder_when_divided_by_x_minus_2_l79_79198

def f (x : ℝ) : ℝ := x^5 - 4 * x^4 + 6 * x^3 + 25 * x^2 - 20 * x - 24

theorem remainder_when_divided_by_x_minus_2 : f 2 = 52 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l79_79198


namespace Shekar_marks_in_English_l79_79299

theorem Shekar_marks_in_English 
  (math_marks : ℕ) (science_marks : ℕ) (socialstudies_marks : ℕ) (biology_marks : ℕ) (average_marks : ℕ) (num_subjects : ℕ) 
  (mathscore : math_marks = 76)
  (sciencescore : science_marks = 65)
  (socialstudiesscore : socialstudies_marks = 82)
  (biologyscore : biology_marks = 85)
  (averagescore : average_marks = 74)
  (numsubjects : num_subjects = 5) :
  ∃ (english_marks : ℕ), english_marks = 62 :=
by
  sorry

end Shekar_marks_in_English_l79_79299


namespace mean_goals_l79_79658

theorem mean_goals :
  let goals := 2 * 3 + 4 * 2 + 5 * 1 + 6 * 1
  let players := 3 + 2 + 1 + 1
  goals / players = 25 / 7 :=
by
  sorry

end mean_goals_l79_79658


namespace distinct_arrangements_of_BANANA_l79_79556

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l79_79556


namespace candy_remains_unclaimed_l79_79081

theorem candy_remains_unclaimed
  (x : ℚ) (h1 : x > 0) :
  let al_claim := (4 / 9 : ℚ) * x,
      bert_claim := (1 / 3 : ℚ) * x,
      carl_claim := (2 / 9 : ℚ) * x,
      bert_left := x - al_claim,
      bert_take := bert_claim,
      carl_left := bert_left - bert_take,
      carl_take := carl_claim in
  carl_left - carl_take = 0 :=
by sorry

end candy_remains_unclaimed_l79_79081


namespace max_subset_size_l79_79115

open Nat

theorem max_subset_size (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n)
  (h_gcd : a.gcd b = 1) (h_div : (a + b) ∣ n) :
  ∃ S : Finset ℕ, S ⊆ Finset.range (n + 1) ∧ (∀ x y ∈ S, x ≠ y → (x - y).abs ≠ a ∧ (x - y).abs ≠ b) ∧
  S.card = (n / (a + b)) * ((a + b) / 2) :=
by 
  sorry

end max_subset_size_l79_79115


namespace area_comparison_l79_79937

def point := (ℝ × ℝ)

def quadrilateral_I_vertices : List point := [(0, 0), (2, 0), (2, 2), (0, 2)]

def quadrilateral_I_area : ℝ := 4

def quadrilateral_II_vertices : List point := [(1, 0), (4, 0), (4, 4), (1, 3)]

noncomputable def quadrilateral_II_area : ℝ := 10.5

theorem area_comparison :
  quadrilateral_I_area < quadrilateral_II_area :=
  by
    sorry

end area_comparison_l79_79937


namespace domain_of_f_2x_minus_1_l79_79274

theorem domain_of_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -1 ≤ x ∧ x ≤ 1 → (f x ≠ 0)) →
  (∀ y, 0 ≤ y ∧ y ≤ 1 ↔ exists x, (2 * x - 1 = y) ∧ (0 ≤ x ∧ x ≤ 1)) :=
by
  sorry

end domain_of_f_2x_minus_1_l79_79274


namespace train_length_l79_79966

theorem train_length (speed_km_hr : ℝ) (time_sec : ℝ) (h_speed : speed_km_hr = 60) (h_time : time_sec = 6) :
  let speed_ms := (speed_km_hr * 1000) / 3600
  let length_m := speed_ms * time_sec
  length_m ≈ 100.02 :=
by sorry

end train_length_l79_79966


namespace transform_roots_to_quadratic_l79_79319

noncomputable def quadratic_formula (p q : ℝ) (x : ℝ) : ℝ :=
  x^2 + p * x + q

theorem transform_roots_to_quadratic (x₁ x₂ y₁ y₂ p q : ℝ)
  (h₁ : quadratic_formula p q x₁ = 0)
  (h₂ : quadratic_formula p q x₂ = 0)
  (h₃ : x₁ ≠ 1)
  (h₄ : x₂ ≠ 1)
  (hy₁ : y₁ = (x₁ + 1) / (x₁ - 1))
  (hy₂ : y₂ = (x₂ + 1) / (x₂ - 1)) :
  (1 + p + q) * y₁^2 + 2 * (1 - q) * y₁ + (1 - p + q) = 0 ∧
  (1 + p + q) * y₂^2 + 2 * (1 - q) * y₂ + (1 - p + q) = 0 := 
sorry

end transform_roots_to_quadratic_l79_79319


namespace total_skips_correct_l79_79572

def S (n : ℕ) : ℕ := n^2 + n

def TotalSkips5 : ℕ :=
  S 1 + S 2 + S 3 + S 4 + S 5

def Skips6 : ℕ :=
  2 * S 6

theorem total_skips_correct : TotalSkips5 + Skips6 = 154 :=
by
  -- proof goes here
  sorry

end total_skips_correct_l79_79572


namespace find_n_constant_term_l79_79120

-- Given condition as a Lean term
def eq1 (n : ℕ) : ℕ := 2^(2*n) - (2^n + 992)

-- Prove that n = 5 fulfills the condition
theorem find_n : eq1 5 = 0 := by
  sorry

-- Given n = 5, find the constant term in the given expansion
def general_term (n r : ℕ) : ℤ := (-1)^r * (Nat.choose (2*n) r) * (n - 5*r/2)

-- Prove the constant term is 45 when n = 5
theorem constant_term : general_term 5 2 = 45 := by
  sorry

end find_n_constant_term_l79_79120


namespace banana_arrangements_l79_79531

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l79_79531


namespace veranda_width_l79_79043

theorem veranda_width (w : ℝ) (h_room : 18 * 12 = 216) (h_veranda : 136 = 136) : 
  (18 + 2*w) * (12 + 2*w) = 352 → w = 2 :=
by
  sorry

end veranda_width_l79_79043


namespace permutations_of_banana_l79_79564

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l79_79564


namespace ratio_of_numbers_l79_79797

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l79_79797


namespace find_prime_pair_l79_79850
open Int

theorem find_prime_pair :
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ ∃ (p : ℕ), Prime p ∧ p = a * b^2 / (a + b) ∧ (a, b) = (6, 2) := by
  sorry

end find_prime_pair_l79_79850


namespace jason_bought_correct_dozens_l79_79892

-- Given conditions
def cupcakes_per_cousin : Nat := 3
def cousins : Nat := 16
def cupcakes_per_dozen : Nat := 12

-- Calculated value
def total_cupcakes : Nat := cupcakes_per_cousin * cousins
def dozens_of_cupcakes_bought : Nat := total_cupcakes / cupcakes_per_dozen

-- Theorem statement
theorem jason_bought_correct_dozens : dozens_of_cupcakes_bought = 4 := by
  -- Proof omitted
  sorry

end jason_bought_correct_dozens_l79_79892


namespace exist_points_C_and_D_l79_79110

-- Lean 4 statement of the problem
theorem exist_points_C_and_D 
  (S : set (ℝ × ℝ)) -- The circle S
  (A B : ℝ × ℝ) -- Points A and B on the circle S
  (α : ℝ) -- The given angle α
  (hA : A ∈ S) (hB : B ∈ S) -- Conditions: A and B are on the circle S
  (arc_CD : ℝ) -- The arc length CD equal to given α
  (hα : 0 < α ∧ α < 2 * π) -- Angle α is within valid range for a circle
  (hS : ∃ O r, ∀ P, P ∈ S ↔ dist P O = r) : -- Definition of circle S with center O and radius r
  ∃ C D : ℝ × ℝ, C ∈ S ∧ D ∈ S ∧ -- Points C and D are on the circle S
    arc_CD = α ∧ -- The arc length CD is α
    (C.1 = D.1 ∧ A.2 = B.2) := -- CA is parallel to DB (evaluated here simply with Cartesian coordinates)

sorry -- Proof to be filled in

end exist_points_C_and_D_l79_79110


namespace Sam_memorized_more_digits_l79_79783

variable (MinaDigits SamDigits CarlosDigits : ℕ)
variable (h1 : MinaDigits = 6 * CarlosDigits)
variable (h2 : MinaDigits = 24)
variable (h3 : SamDigits = 10)
 
theorem Sam_memorized_more_digits :
  SamDigits - CarlosDigits = 6 :=
by
  -- Let's unfold the statements and perform basic arithmetic.
  sorry

end Sam_memorized_more_digits_l79_79783


namespace prove_sum_eq_9_l79_79258

theorem prove_sum_eq_9 (a b : ℝ) (h : i * (a - i) = b - (2 * i) ^ 3) : a + b = 9 :=
by
  sorry

end prove_sum_eq_9_l79_79258


namespace new_total_lifting_capacity_is_correct_l79_79146

-- Define the initial lifting capacities and improvements
def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50
def clean_and_jerk_multiplier : ℕ := 2
def snatch_increment_percentage : ℕ := 80

-- Calculated values
def new_clean_and_jerk := initial_clean_and_jerk * clean_and_jerk_multiplier
def snatch_increment := initial_snatch * snatch_increment_percentage / 100
def new_snatch := initial_snatch + snatch_increment
def new_total_lifting_capacity := new_clean_and_jerk + new_snatch

-- Theorem statement to be proven
theorem new_total_lifting_capacity_is_correct :
  new_total_lifting_capacity = 250 := 
sorry

end new_total_lifting_capacity_is_correct_l79_79146


namespace hyperbola_eccentricity_l79_79581

theorem hyperbola_eccentricity
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h_hyperbola : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (h_asymptotes_intersect : ∃ A B : ℝ × ℝ, A ≠ B ∧ A.1 = -1 ∧ B.1 = -1 ∧
    ∀ (A B : ℝ × ℝ), ∃ x y : ℝ, (A.2 = y ∧ B.2 = y ∧ x^2 / a^2 - y^2 / b^2 = 1))
  (triangle_area : ∃ A B : ℝ × ℝ, 1 / 2 * abs (A.1 * B.2 - A.2 * B.1) = 2 * Real.sqrt 3) :
  ∃ e : ℝ, e = Real.sqrt 13 :=
by {
  sorry
}

end hyperbola_eccentricity_l79_79581


namespace smallest_number_l79_79694

theorem smallest_number (n : ℕ) :
  (n % 3 = 1) ∧
  (n % 5 = 3) ∧
  (n % 6 = 4) →
  n = 28 :=
sorry

end smallest_number_l79_79694


namespace tangent_line_at_e_range_of_a_l79_79266

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * Real.log x
noncomputable def g (a x : ℝ) : ℝ := f a x - 2 * a * x

theorem tangent_line_at_e (a : ℝ) :
  a = 0 →
  ∃ m b : ℝ, (∀ x, y = m * x + b) ∧ 
             y = (2 / Real.exp 1 - 2 * Real.exp 1) * x + (Real.exp 1)^2 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x ∈ Set.Ioi 1 → g a x < 0) →
  a ∈ Set.Icc (-1) 1 :=
sorry

end tangent_line_at_e_range_of_a_l79_79266


namespace string_length_l79_79069

theorem string_length (cylinder_circumference : ℝ)
  (total_loops : ℕ) (post_height : ℝ)
  (height_per_loop : ℝ := post_height / total_loops)
  (hypotenuse_per_loop : ℝ := Real.sqrt (height_per_loop ^ 2 + cylinder_circumference ^ 2))
  : total_loops = 5 → cylinder_circumference = 4 → post_height = 15 → hypotenuse_per_loop * total_loops = 25 :=
by 
  intros h1 h2 h3
  sorry

end string_length_l79_79069


namespace equation_solutions_l79_79187

theorem equation_solutions (x : ℝ) : x * (2 * x + 1) = 2 * x + 1 ↔ x = -1 / 2 ∨ x = 1 :=
by
  sorry

end equation_solutions_l79_79187


namespace permutations_of_banana_l79_79566

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l79_79566


namespace det_A_eq_l79_79840

open Matrix

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, -3, 3],
    ![x, 5, -1],
    ![4, -2, 1]]

theorem det_A_eq (x : ℝ) : det (A x) = -3 * x - 45 :=
by sorry

end det_A_eq_l79_79840


namespace Gina_tip_is_5_percent_l79_79722

noncomputable def Gina_tip_percentage : ℝ := 5

theorem Gina_tip_is_5_percent (bill_amount : ℝ) (good_tipper_percentage : ℝ)
    (good_tipper_extra_tip_cents : ℝ) (good_tipper_tip : ℝ) 
    (Gina_tip_extra_cents : ℝ):
    bill_amount = 26 ∧
    good_tipper_percentage = 20 ∧
    Gina_tip_extra_cents = 390 ∧
    good_tipper_tip = (20 / 100) * 26 ∧
    Gina_tip_extra_cents = 390 ∧
    (Gina_tip_percentage / 100) * bill_amount + (Gina_tip_extra_cents / 100) = good_tipper_tip
    → Gina_tip_percentage = 5 :=
by
  sorry

end Gina_tip_is_5_percent_l79_79722


namespace BANANA_arrangements_l79_79520

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l79_79520


namespace find_values_l79_79711

theorem find_values (h t u : ℕ) 
  (h0 : u = h - 5) 
  (h1 : (h * 100 + t * 10 + u) - (h * 100 + u * 10 + t) = 96)
  (hu : h < 10 ∧ t < 10 ∧ u < 10) :
  h = 5 ∧ t = 9 ∧ u = 0 :=
by 
  sorry

end find_values_l79_79711


namespace base_conversion_l79_79841

def baseThreeToBaseTen (n : List ℕ) : ℕ :=
  n.reverse.enumFrom 0 |>.map (λ ⟨i, d⟩ => d * 3^i) |>.sum

def baseTenToBaseFive (n : ℕ) : List ℕ :=
  let rec aux (n : ℕ) (acc : List ℕ) : List ℕ :=
    if n = 0 then acc else aux (n / 5) ((n % 5) :: acc)
  aux n []

theorem base_conversion (baseThreeNum : List ℕ) (baseTenNum : ℕ) (baseFiveNum : List ℕ) :
  baseThreeNum = [2, 0, 1, 2, 1] →
  baseTenNum = 178 →
  baseFiveNum = [1, 2, 0, 3] →
  baseThreeToBaseTen baseThreeNum = baseTenNum ∧ baseTenToBaseFive baseTenNum = baseFiveNum :=
by
  intros h1 h2 h3
  unfold baseThreeToBaseTen
  unfold baseTenToBaseFive
  sorry

end base_conversion_l79_79841


namespace hexagon_diagonal_length_is_twice_side_l79_79980

noncomputable def regular_hexagon_side_length : ℝ := 12

def diagonal_length_in_regular_hexagon (s : ℝ) : ℝ :=
2 * s

theorem hexagon_diagonal_length_is_twice_side :
  diagonal_length_in_regular_hexagon regular_hexagon_side_length = 2 * regular_hexagon_side_length :=
by 
  -- Simplify and check the computation according to the understanding of the properties of the hexagon
  sorry

end hexagon_diagonal_length_is_twice_side_l79_79980


namespace find_real_solutions_l79_79100

theorem find_real_solutions (x : ℝ) : 
  x^4 + (3 - x)^4 = 130 ↔ x = 1.5 + Real.sqrt 1.5 ∨ x = 1.5 - Real.sqrt 1.5 :=
sorry

end find_real_solutions_l79_79100


namespace books_at_end_l79_79814

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l79_79814


namespace right_triangle_sides_l79_79569

theorem right_triangle_sides (p m : ℝ)
  (hp : 0 < p)
  (hm : 0 < m) :
  ∃ a b c : ℝ, 
    a + b + c = 2 * p ∧
    a^2 + b^2 = c^2 ∧
    (1 / 2) * a * b = m^2 ∧
    c = (p^2 - m^2) / p ∧
    a = (p^2 + m^2 + Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) ∧
    b = (p^2 + m^2 - Real.sqrt ((p^2 + m^2)^2 - 8 * p^2 * m^2)) / (2 * p) := 
by
  sorry

end right_triangle_sides_l79_79569


namespace sin_mul_cos_eq_quarter_l79_79262

open Real

theorem sin_mul_cos_eq_quarter (α : ℝ) (h : sin α - cos α = sqrt 2 / 2) : sin α * cos α = 1 / 4 :=
by
  sorry

end sin_mul_cos_eq_quarter_l79_79262


namespace number_of_arrangements_of_BANANA_l79_79537

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l79_79537


namespace arithmetic_sequence_third_term_l79_79663

theorem arithmetic_sequence_third_term (a d : ℤ) 
  (h20 : a + 19 * d = 17) (h21 : a + 20 * d = 20) : a + 2 * d = -34 := 
sorry

end arithmetic_sequence_third_term_l79_79663


namespace exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l79_79381

theorem exists_n_such_that_an_is_cube_and_bn_is_fifth_power
  (a b : ℕ) (ha : a > 0) (hb : b > 0) :
  ∃ (n : ℕ), n ≥ 1 ∧ (∃ k : ℤ, a * n = k^3) ∧ (∃ l : ℤ, b * n = l^5) := 
by
  sorry

end exists_n_such_that_an_is_cube_and_bn_is_fifth_power_l79_79381


namespace union_of_A_B_l79_79583

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l79_79583


namespace discount_problem_l79_79316

variable (x : ℝ)

theorem discount_problem :
  (400 * (1 - x)^2 = 225) :=
sorry

end discount_problem_l79_79316


namespace percentage_x_equals_y_l79_79127

theorem percentage_x_equals_y (x y z : ℝ) (p : ℝ)
    (h1 : 0.45 * z = 0.39 * y)
    (h2 : z = 0.65 * x)
    (h3 : y = (p / 100) * x) : 
    p = 75 := 
sorry

end percentage_x_equals_y_l79_79127


namespace permutations_of_banana_l79_79568

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l79_79568


namespace cubed_difference_l79_79603

theorem cubed_difference (x : ℝ) (h : x - 1/x = 3) : (x^3 - 1/x^3 = 36) := 
by
  sorry

end cubed_difference_l79_79603


namespace excircle_identity_l79_79019

variables (a b c r_a r_b r_c : ℝ)

-- Conditions: r_a, r_b, r_c are the radii of the excircles opposite vertices A, B, and C respectively.
-- In the triangle ABC, a, b, c are the sides opposite vertices A, B, and C respectively.

theorem excircle_identity:
  (a^2 / (r_a * (r_b + r_c)) + b^2 / (r_b * (r_c + r_a)) + c^2 / (r_c * (r_a + r_b))) = 2 :=
by
  sorry

end excircle_identity_l79_79019


namespace total_hike_time_l79_79625

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l79_79625


namespace number_of_arrangements_of_BANANA_l79_79536

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l79_79536


namespace simplify_expression_l79_79643

theorem simplify_expression :
  ((45 * 2^10) / (15 * 2^5) * 5) = 480 := by
  sorry

end simplify_expression_l79_79643


namespace total_rainfall_2010_to_2012_l79_79995

noncomputable def average_rainfall (year : ℕ) : ℕ :=
  if year = 2010 then 35
  else if year = 2011 then 38
  else if year = 2012 then 41
  else 0

theorem total_rainfall_2010_to_2012 :
  (12 * average_rainfall 2010) + 
  (12 * average_rainfall 2011) + 
  (12 * average_rainfall 2012) = 1368 :=
by
  sorry

end total_rainfall_2010_to_2012_l79_79995


namespace rectangle_vertex_x_coordinate_l79_79050

theorem rectangle_vertex_x_coordinate
  (x : ℝ)
  (y1 y2 : ℝ)
  (slope : ℝ)
  (h1 : x = 1)
  (h2 : 9 = 9)
  (h3 : slope = 0.2)
  (h4 : y1 = 0)
  (h5 : y2 = 2)
  (h6 : ∀ (x : ℝ), (0.2 * x : ℝ) = 1 → x = 1) :
  x = 1 := 
by sorry

end rectangle_vertex_x_coordinate_l79_79050


namespace amy_work_hours_per_week_l79_79972

/-- Amy works for 40 hours per week for 8 weeks in the summer, making $3200. If she works for 32 weeks during 
the school year at the same rate of pay and needs to make another $4000, we need to prove that she must 
work 12.5 hours per week during the school year. -/
theorem amy_work_hours_per_week
  (summer_hours_per_week : ℕ)
  (summer_weeks : ℕ)
  (summer_money : ℕ)
  (school_year_weeks : ℕ)
  (school_year_money_needed : ℕ) :
  (let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ) in
   let hours_needed := school_year_money_needed / hourly_wage in
   hours_needed / school_year_weeks = 12.5) :=
by
  -- Definitions
  let summer_hours_per_week := 40
  let summer_weeks := 8
  let summer_money := 3200
  let school_year_weeks := 32
  let school_year_money_needed := 4000
  
  -- Calculations
  let hourly_wage := summer_money / (summer_hours_per_week * summer_weeks : ℕ)
  let hours_needed := school_year_money_needed / hourly_wage
  let hours_per_week := hours_needed / school_year_weeks

  -- Goal
  show hours_per_week = 12.5

  -- This proof is omitted.
  sorry

end amy_work_hours_per_week_l79_79972


namespace find_x_l79_79605

theorem find_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 6 * x^2 + 12 * x * y + 6 * y^2 = x^3 + 3 * x^2 * y + 3 * x * y^2) : x = 24 / 7 :=
by
  sorry

end find_x_l79_79605


namespace base_8_to_base_10_2671_to_1465_l79_79345

theorem base_8_to_base_10_2671_to_1465 :
  (2 * 8^3 + 6 * 8^2 + 7 * 8^1 + 1 * 8^0) = 1465 := by
  sorry

end base_8_to_base_10_2671_to_1465_l79_79345


namespace probability_blue_then_yellow_l79_79411

variable (total_chips : ℕ := 15)
variable (blue_chips : ℕ := 10)
variable (yellow_chips : ℕ := 5)
variable (initial_blue_prob : ℚ := blue_chips / total_chips)
variable (remaining_chips : ℕ := total_chips - 1)
variable (next_yellow_prob : ℚ := yellow_chips / remaining_chips)
variable (final_prob : ℚ := initial_blue_prob * next_yellow_prob)

theorem probability_blue_then_yellow :
  final_prob = 5 / 21 :=
sorry

end probability_blue_then_yellow_l79_79411


namespace determine_b_l79_79845

theorem determine_b (b : ℝ) : (∀ x1 x2 : ℝ, x1^2 - x2^2 = 7 → x1 * x2 = 12 → x1 + x2 = b) → (b = 7 ∨ b = -7) := 
by {
  -- Proof needs to be provided
  sorry
}

end determine_b_l79_79845


namespace gcd_m_n_l79_79195

def m := 122^2 + 234^2 + 346^2 + 458^2
def n := 121^2 + 233^2 + 345^2 + 457^2

theorem gcd_m_n : Int.gcd m n = 1 := 
by sorry

end gcd_m_n_l79_79195


namespace part_time_employees_l79_79074

theorem part_time_employees (total_employees : ℕ) (full_time_employees : ℕ) (part_time_employees : ℕ) 
  (h1 : total_employees = 65134) 
  (h2 : full_time_employees = 63093) 
  (h3 : total_employees = full_time_employees + part_time_employees) : 
  part_time_employees = 2041 :=
by 
  sorry

end part_time_employees_l79_79074


namespace new_total_lifting_capacity_is_correct_l79_79145

-- Define the initial lifting capacities and improvements
def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50
def clean_and_jerk_multiplier : ℕ := 2
def snatch_increment_percentage : ℕ := 80

-- Calculated values
def new_clean_and_jerk := initial_clean_and_jerk * clean_and_jerk_multiplier
def snatch_increment := initial_snatch * snatch_increment_percentage / 100
def new_snatch := initial_snatch + snatch_increment
def new_total_lifting_capacity := new_clean_and_jerk + new_snatch

-- Theorem statement to be proven
theorem new_total_lifting_capacity_is_correct :
  new_total_lifting_capacity = 250 := 
sorry

end new_total_lifting_capacity_is_correct_l79_79145


namespace geom_seq_arith_seq_l79_79730

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def isGeomSeq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = q * a n

theorem geom_seq_arith_seq (h1 : ∀ n, 0 < a n) 
  (h2 : isGeomSeq a q)
  (h3 : 2 * (1 / 2 * a 5) = a 3 + a 4)
  : (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := 
sorry

end geom_seq_arith_seq_l79_79730


namespace prod_quality_related_prob_at_least_one_from_A_l79_79488

noncomputable def chi_square_value (a b c d n : ℝ) : ℝ :=
  (n * (a * d - b * c) ^ 2) / ( (a + b) * (c + d) * (a + c) * (b + d) )

theorem prod_quality_related (a b c d n k0 : ℝ) (h_k0 : k0 = 2.706)
  (h_a : a = 40) (h_b : b = 80) (h_c : c = 80) (h_d : d = 100) (h_n : n = 300) :
  chi_square_value a b c d n ≥ k0 :=
by
  sorry

theorem prob_at_least_one_from_A :
  ∃ (A B : ℕ), A = 2 ∧ B = 4 ∧ (choose 6 2) = 15 ∧
  ((A * B + choose A 2 + choose B 2) : ℝ) / (choose 6 2) = (3 : ℝ) / 5 :=
by
  sorry

end prod_quality_related_prob_at_least_one_from_A_l79_79488


namespace truth_values_l79_79513

-- Define the region D as a set
def D (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 ≤ 4

-- Define propositions p and q
def p : Prop := ∀ x y, D x y → 2 * x + y ≤ 8
def q : Prop := ∃ x y, D x y ∧ 2 * x + y ≤ -1

-- State the propositions to be proven
def prop1 : Prop := p ∨ q
def prop2 : Prop := ¬p ∨ q
def prop3 : Prop := p ∧ ¬q
def prop4 : Prop := ¬p ∧ ¬q

-- State the main theorem asserting the truth values of the propositions
theorem truth_values : ¬prop1 ∧ prop2 ∧ ¬prop3 ∧ prop4 :=
by
  sorry

end truth_values_l79_79513


namespace twelve_edge_cubes_painted_faces_l79_79962

theorem twelve_edge_cubes_painted_faces :
  let painted_faces_per_edge_cube := 2
  let num_edge_cubes := 12
  painted_faces_per_edge_cube * num_edge_cubes = 24 :=
by
  sorry

end twelve_edge_cubes_painted_faces_l79_79962


namespace part_a_part_b_part_c_l79_79152

/-
  Define conditions and property \mathcal{S} in Lean 4
-/
def has_property_S (X : Finset ℕ) (p q : ℕ) : Prop :=
  ∀ (B : Finset (Finset ℕ)), 
    B.card = p ∧ (∀ b ∈ B, b.card = q) → 
    ∃ (Y : Finset ℕ),
    Y.card = p ∧ (∀ b ∈ B, (Y ∩ b).card ≤ 1)

/-
  Part (a):  
  Prove that if p = 4 and q = 3, any set X with 9 elements does not satisfy \mathcal{S}
-/
theorem part_a : ¬ (has_property_S (Finset.range 9) 4 3) :=
sorry

/-
  Part (b):  
  Prove that if p, q ≥ 2, any set X with pq - q elements does not satisfy \mathcal{S}
-/
theorem part_b (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) : 
  ¬ (has_property_S (Finset.range (p * q - q)) p q) :=
sorry

/-
  Part (c): 
  Prove that if p, q ≥ 2, any set X with pq - q + 1 elements does satisfy \mathcal{S}
-/
theorem part_c (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) : 
  has_property_S (Finset.range (p * q - q + 1)) p q :=
sorry

end part_a_part_b_part_c_l79_79152


namespace third_side_length_is_six_l79_79045

-- Defining the lengths of the sides of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 6

-- Defining that the third side is an even number between 4 and 8
def is_even (x : ℕ) : Prop := x % 2 = 0
def valid_range (x : ℕ) : Prop := 4 < x ∧ x < 8

-- Stating the theorem
theorem third_side_length_is_six (x : ℕ) (h1 : is_even x) (h2 : valid_range x) : x = 6 :=
by
  sorry

end third_side_length_is_six_l79_79045


namespace cistern_length_is_correct_l79_79960

-- Definitions for the conditions mentioned in the problem
def cistern_width : ℝ := 6
def water_depth : ℝ := 1.25
def wet_surface_area : ℝ := 83

-- The length of the cistern to be proven
def cistern_length : ℝ := 8

-- Theorem statement that length of the cistern must be 8 meters given the conditions
theorem cistern_length_is_correct :
  ∃ (L : ℝ), (wet_surface_area = (L * cistern_width) + (2 * L * water_depth) + (2 * cistern_width * water_depth)) ∧ L = cistern_length :=
  sorry

end cistern_length_is_correct_l79_79960


namespace sphere_surface_area_l79_79889

variable (x y z : ℝ)

theorem sphere_surface_area :
  (x^2 + y^2 + z^2 = 1) → (4 * Real.pi) = 4 * Real.pi :=
by
  intro h
  -- The proof will be inserted here
  sorry

end sphere_surface_area_l79_79889


namespace distance_between_stations_is_correct_l79_79474

noncomputable def distance_between_stations : ℕ := 200

theorem distance_between_stations_is_correct 
  (start_hour_p : ℕ := 7) 
  (speed_p : ℕ := 20) 
  (start_hour_q : ℕ := 8) 
  (speed_q : ℕ := 25) 
  (meeting_hour : ℕ := 12)
  (time_travel_p := meeting_hour - start_hour_p) -- Time traveled by train from P
  (time_travel_q := meeting_hour - start_hour_q) -- Time traveled by train from Q 
  (distance_travel_p := speed_p * time_travel_p) 
  (distance_travel_q := speed_q * time_travel_q) : 
  distance_travel_p + distance_travel_q = distance_between_stations :=
by 
  sorry

end distance_between_stations_is_correct_l79_79474


namespace chromium_atoms_in_compound_l79_79066

-- Definitions of given conditions
def hydrogen_atoms : Nat := 2
def oxygen_atoms : Nat := 4
def compound_molecular_weight : ℝ := 118
def hydrogen_atomic_weight : ℝ := 1
def chromium_atomic_weight : ℝ := 52
def oxygen_atomic_weight : ℝ := 16

-- Problem statement to find the number of Chromium atoms
theorem chromium_atoms_in_compound (hydrogen_atoms : Nat) (oxygen_atoms : Nat) (compound_molecular_weight : ℝ)
    (hydrogen_atomic_weight : ℝ) (chromium_atomic_weight : ℝ) (oxygen_atomic_weight : ℝ) :
  hydrogen_atoms * hydrogen_atomic_weight + 
  oxygen_atoms * oxygen_atomic_weight + 
  chromium_atomic_weight = compound_molecular_weight → 
  chromium_atomic_weight = 52 :=
by
  sorry

end chromium_atoms_in_compound_l79_79066


namespace average_salary_proof_l79_79446

noncomputable def average_salary_of_all_workers (tech_workers : ℕ) (tech_avg_sal : ℕ) (total_workers : ℕ) (non_tech_avg_sal : ℕ) : ℕ :=
  let non_tech_workers := total_workers - tech_workers
  let total_tech_salary := tech_workers * tech_avg_sal
  let total_non_tech_salary := non_tech_workers * non_tech_avg_sal
  let total_salary := total_tech_salary + total_non_tech_salary
  total_salary / total_workers

theorem average_salary_proof : average_salary_of_all_workers 7 14000 28 6000 = 8000 := by
  sorry

end average_salary_proof_l79_79446


namespace tangent_line_at_P_minimized_distance_point_Q_l79_79596

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2

def P : ℝ × ℝ := (1, -2)

def line_l (x y : ℝ) : Prop := x - 2 * y + 3 = 0

theorem tangent_line_at_P :
  let f' := deriv f 1 in
  f' = 1 → (∀ x y : ℝ, y + 2 = x - 1 → x - y - 3 = 0) :=
by
  intros f' hf' x y h_tangent
  sorry

theorem minimized_distance_point_Q :
  ∃ x : ℝ, x = 2 ∧ (∀ y : ℝ, y = Real.log 2 - 2 → let Q := (x, y) in 
  ∀ d : ℝ, d = (abs (Q.1 - 2 * Real.log Q.1 + 7)) / (Real.sqrt 5) → 
  (∀ h : ℝ, h ≠ d → h > d)) :=
by
  intros
  use [2, Real.log 2 - 2]
  sorry

end tangent_line_at_P_minimized_distance_point_Q_l79_79596


namespace initial_fee_l79_79142

theorem initial_fee (initial_fee : ℝ) : 
  (∀ (distance_charge_per_segment travel_total_charge : ℝ), 
    distance_charge_per_segment = 0.35 → 
    3.6 / 0.4 * distance_charge_per_segment + initial_fee = travel_total_charge → 
    travel_total_charge = 5.20)
    → initial_fee = 2.05 :=
by
  intro h
  specialize h 0.35 5.20
  sorry

end initial_fee_l79_79142


namespace common_factor_of_polynomial_l79_79927

variables (x y m n : ℝ)

theorem common_factor_of_polynomial :
  ∃ (k : ℝ), (2 * (m - n)) = k ∧ (4 * x * (m - n) + 2 * y * (m - n)^2) = k * (2 * x * (m - n)) :=
sorry

end common_factor_of_polynomial_l79_79927


namespace pencils_cost_l79_79062

theorem pencils_cost (A B : ℕ) (C D : ℕ) (r : ℚ) : 
    A * 20 = 3200 → B * 20 = 960 → (A / B = 3200 / 960) → (A = 160) → (B = 48) → (C = 3200) → (D = 960) → 160 * 960 / 48 = 3200 :=
by
sorry

end pencils_cost_l79_79062


namespace sum_of_numbers_eq_answer_l79_79318

open Real

noncomputable def sum_of_numbers (x y : ℝ) : ℝ := x + y

theorem sum_of_numbers_eq_answer (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 16) (h4 : (1 / x) = 3 * (1 / y)) :
  sum_of_numbers x y = 16 * Real.sqrt 3 / 3 := 
sorry

end sum_of_numbers_eq_answer_l79_79318


namespace total_snakes_in_park_l79_79468

theorem total_snakes_in_park :
  ∀ (pythons boa_constrictors rattlesnakes total_snakes : ℕ),
    boa_constrictors = 40 →
    pythons = 3 * boa_constrictors →
    rattlesnakes = 40 →
    total_snakes = boa_constrictors + pythons + rattlesnakes →
    total_snakes = 200 :=
by
  intros pythons boa_constrictors rattlesnakes total_snakes h1 h2 h3 h4
  rw [h1, h3] at h4
  rw [h2] at h4
  sorry

end total_snakes_in_park_l79_79468


namespace solve_for_x_l79_79877

theorem solve_for_x (x : ℝ) (h : 144 / 0.144 = x / 0.0144) : x = 14.4 :=
by
  sorry

end solve_for_x_l79_79877


namespace integer_triangle_600_integer_triangle_144_l79_79574

-- Problem Part I
theorem integer_triangle_600 :
  ∃ (a b c : ℕ), a * b * c = 600 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 26 :=
by {
  sorry
}

-- Problem Part II
theorem integer_triangle_144 :
  ∃ (a b c : ℕ), a * b * c = 144 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 16 :=
by {
  sorry
}

end integer_triangle_600_integer_triangle_144_l79_79574


namespace Mary_chewing_gums_count_l79_79314

variable (Mary_gums Sam_gums Sue_gums : ℕ)

-- Define the given conditions
axiom Sam_chewing_gums : Sam_gums = 10
axiom Sue_chewing_gums : Sue_gums = 15
axiom Total_chewing_gums : Mary_gums + Sam_gums + Sue_gums = 30

theorem Mary_chewing_gums_count : Mary_gums = 5 := by
  sorry

end Mary_chewing_gums_count_l79_79314


namespace correct_relative_pronoun_used_l79_79124

theorem correct_relative_pronoun_used (option : String) :
  (option = "where") ↔
  "Giving is a universal opportunity " ++ option ++ " regardless of your age, profession, religion, and background, you have the capacity to create change." =
  "Giving is a universal opportunity where regardless of your age, profession, religion, and background, you have the capacity to create change." :=
by
  sorry

end correct_relative_pronoun_used_l79_79124


namespace ratio_of_solving_linear_equations_to_algebra_problems_l79_79166

theorem ratio_of_solving_linear_equations_to_algebra_problems:
  let total_problems := 140
  let algebra_percentage := 0.40
  let solving_linear_equations := 28
  let total_algebra_problems := algebra_percentage * total_problems
  let ratio := solving_linear_equations / total_algebra_problems
  ratio = 1 / 2 := by
  sorry

end ratio_of_solving_linear_equations_to_algebra_problems_l79_79166


namespace least_whole_number_clock_equiv_l79_79163

theorem least_whole_number_clock_equiv (h : ℕ) (h_gt_10 : h > 10) : 
  ∃ k, k = 12 ∧ (h^2 - h) % 12 = 0 ∧ h = 12 :=
by 
  sorry

end least_whole_number_clock_equiv_l79_79163


namespace ratio_of_numbers_l79_79798

theorem ratio_of_numbers (x y : ℝ) (h₁ : 0 < y) (h₂ : y < x) (h₃ : x + y = 7 * (x - y)) : x / y = 4 / 3 :=
by
  sorry

end ratio_of_numbers_l79_79798


namespace function_f_not_all_less_than_half_l79_79640

theorem function_f_not_all_less_than_half (p q : ℝ) (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f x = x^2 + p*x + q) :
  ¬ (|f 1| < 1 / 2 ∧ |f 2| < 1 / 2 ∧ |f 3| < 1 / 2) :=
sorry

end function_f_not_all_less_than_half_l79_79640


namespace gcd_of_1230_and_920_is_10_l79_79046

theorem gcd_of_1230_and_920_is_10 : Int.gcd 1230 920 = 10 :=
sorry

end gcd_of_1230_and_920_is_10_l79_79046


namespace range_of_m_l79_79880

theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, (x > 3 - m) ∧ (x ≤ 5) ↔ (1 ≤ x ∧ x ≤ 5)) →
  (2 < m ∧ m ≤ 3) := 
by
  sorry

end range_of_m_l79_79880


namespace fire_alarms_and_passengers_discrete_l79_79211

-- Definitions of the random variables
def xi₁ : ℕ := sorry  -- number of fire alarms in a city within one day
def xi₂ : ℝ := sorry  -- temperature in a city within one day
def xi₃ : ℕ := sorry  -- number of passengers at a train station in a city within a month

-- Defining the concept of discrete random variable
def is_discrete (X : Type) : Prop := 
  ∃ f : X → ℕ, ∀ x : X, ∃ n : ℕ, f x = n

-- Statement of the proof problem
theorem fire_alarms_and_passengers_discrete :
  is_discrete ℕ ∧ is_discrete ℕ ∧ ¬ is_discrete ℝ :=
by
  have xi₁_discrete : is_discrete ℕ := sorry
  have xi₃_discrete : is_discrete ℕ := sorry
  have xi₂_not_discrete : ¬ is_discrete ℝ := sorry
  exact ⟨xi₁_discrete, xi₃_discrete, xi₂_not_discrete⟩

end fire_alarms_and_passengers_discrete_l79_79211


namespace find_angle_FYD_l79_79619

variable {Point : Type}
variables (A B C D X Y : Point)
variables (E F G : Point)
variables [Euclidean ⟨_,⟨A, B, C, by sorry⟩⟩]

-- Conditions
variable (h_parallel : parallel (line_through A B) (line_through C D))
variable (h_angle_AXF : angle A X F = 130)
variable (h_isosceles_XFG : is_isosceles_triangle X F G)
variable (h_angle_FXG : angle F X G = 36)

-- Question to prove
theorem find_angle_FYD : angle F Y D = 50 :=
by
  sorry

end find_angle_FYD_l79_79619


namespace initial_volume_of_mixture_l79_79073

theorem initial_volume_of_mixture (V : ℝ) :
  let V_new := V + 8
  let initial_water := 0.20 * V
  let new_water := initial_water + 8
  let new_mixture := V_new
  new_water = 0.25 * new_mixture →
  V = 120 :=
by
  intro h
  sorry

end initial_volume_of_mixture_l79_79073


namespace find_sum_of_extrema_l79_79160

open Real

noncomputable def f (x : ℝ) : ℝ := (x - 2)^2 * sin (x - 2) + 3

def f_max (f : ℝ → ℝ) (a b : ℝ) : ℝ := Sup (set.image f (set.Icc a b))
def f_min (f : ℝ → ℝ) (a b : ℝ) : ℝ := Inf (set.image f (set.Icc a b))

theorem find_sum_of_extrema : 
  let M := f_max f (-1) 5 in
  let m := f_min f (-1) 5 in
  M + m = 6 :=
begin
  sorry
end

end find_sum_of_extrema_l79_79160


namespace largest_fraction_proof_l79_79729

theorem largest_fraction_proof 
  (w x y z : ℕ)
  (hw : 0 < w)
  (hx : w < x)
  (hy : x < y)
  (hz : y < z)
  (w_eq : w = 1)
  (x_eq : x = y - 1)
  (z_eq : z = y + 1)
  (y_eq : y = x!) : 
  (max (max (w + z) (w + x)) (max (x + z) (max (x + y) (y + z))) = 5 / 3) := 
sorry

end largest_fraction_proof_l79_79729


namespace required_points_exist_l79_79768

noncomputable def S := {p : ℝ × ℝ | (0 ≤ p.1 ∧ p.1 ≤ 100) ∧ (0 ≤ p.2 ∧ p.2 ≤ 100)}

structure point_line {n : ℕ} :=
  (points : Fin n → ℝ × ℝ)
  (non_self_intersecting : ∀ i j : Fin n, i ≠ j → points i ≠ points j)
  (inside_square : ∀ i : Fin n, points i ∈ S)

def dist (p q : ℝ × ℝ) : ℝ := Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def path_length {n : ℕ} (L : point_line {n}) (i j : Fin n) : ℝ :=
  if h : i < j then
    (Finset.Ico i j).val.map ((dist_on_curve L.points) ∘ (λ k, (k, Nat.succ k))).sum
  else
    (Finset.Ico j i).val.map ((dist_on_curve L.points) ∘ (λ k, (k, Nat.succ k))).sum

theorem required_points_exist (L : point_line {n})
  (boundary_cond : ∀ P ∈ @boundary_points ℝ 2 S, ∃ Q ∈ L.points, dist P Q ≤ 1/2) :
  ∃ (X Y : Fin n), dist (L.points X) (L.points Y) ≤ 1 ∧ path_length L X Y ≥ 198 :=
begin
  sorry
end

end required_points_exist_l79_79768


namespace value_of_expression_l79_79673

theorem value_of_expression : (180^2 - 150^2) / 30 = 330 := by
  sorry

end value_of_expression_l79_79673


namespace elena_total_pens_l79_79985

theorem elena_total_pens 
  (cost_X : ℝ) (cost_Y : ℝ) (total_spent : ℝ) (num_brand_X : ℕ) (num_brand_Y : ℕ) (total_pens : ℕ)
  (h1 : cost_X = 4.0) 
  (h2 : cost_Y = 2.8) 
  (h3 : total_spent = 40.0) 
  (h4 : num_brand_X = 8) 
  (h5 : total_pens = num_brand_X + num_brand_Y) 
  (h6 : total_spent = num_brand_X * cost_X + num_brand_Y * cost_Y) :
  total_pens = 10 :=
sorry

end elena_total_pens_l79_79985


namespace average_salary_l79_79448

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l79_79448


namespace find_angle_l79_79358

-- Given definitions:
def complement (α : ℝ) : ℝ := 90 - α
def supplement (α : ℝ) : ℝ := 180 - α

-- Condition:
def condition (α : ℝ) : Prop :=
  supplement α = 3 * complement α + 10

-- Statement to prove:
theorem find_angle (α : ℝ) (h : condition α) : α = 50 :=
sorry

end find_angle_l79_79358


namespace number_of_men_in_third_group_l79_79821

theorem number_of_men_in_third_group (m w : ℝ) (x : ℕ) :
  3 * m + 8 * w = 6 * m + 2 * w →
  x * m + 5 * w = 0.9285714285714286 * (6 * m + 2 * w) →
  x = 4 :=
by
  intros h₁ h₂
  sorry

end number_of_men_in_third_group_l79_79821


namespace percentage_of_l_equals_150_percent_k_l79_79742

section

variables (j k l m : ℝ) (x : ℝ)

-- Given conditions
axiom cond1 : 1.25 * j = 0.25 * k
axiom cond2 : 1.50 * k = x / 100 * l
axiom cond3 : 1.75 * l = 0.75 * m
axiom cond4 : 0.20 * m = 7.00 * j

-- Proof statement
theorem percentage_of_l_equals_150_percent_k : x = 50 :=
sorry

end

end percentage_of_l_equals_150_percent_k_l79_79742


namespace union_of_A_and_B_l79_79586

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l79_79586


namespace find_some_number_l79_79805

def simplify_expr (x : ℚ) : Prop :=
  1 / 2 + ((2 / 3 * (3 / 8)) + x) - (8 / 16) = 4.25

theorem find_some_number :
  ∃ x : ℚ, simplify_expr x ∧ x = 4 :=
by
  sorry

end find_some_number_l79_79805


namespace average_height_l79_79654

theorem average_height (avg1 avg2 : ℕ) (n1 n2 : ℕ) (total_students : ℕ)
  (h1 : avg1 = 20) (h2 : avg2 = 20) (h3 : n1 = 20) (h4 : n2 = 11) (h5 : total_students = 31) :
  (n1 * avg1 + n2 * avg2) / total_students = 20 :=
by
  -- Placeholder for the proof
  sorry

end average_height_l79_79654


namespace difference_is_correct_l79_79325

-- Definition of the given numbers
def numbers : List ℕ := [44, 16, 2, 77, 241]

-- Define the sum of the numbers
def sum_numbers := numbers.sum

-- Define the average of the numbers
def average := sum_numbers / numbers.length

-- Define the difference between sum and average
def difference := sum_numbers - average

-- The theorem we need to prove
theorem difference_is_correct : difference = 304 := by
  sorry

end difference_is_correct_l79_79325


namespace bianca_total_drawing_time_l79_79975

def total_drawing_time (a b : ℕ) : ℕ := a + b

theorem bianca_total_drawing_time :
  let a := 22
  let b := 19
  total_drawing_time a b = 41 :=
by
  sorry

end bianca_total_drawing_time_l79_79975


namespace find_M_l79_79321

theorem find_M (a b c M : ℚ) 
  (h1 : a + b + c = 100)
  (h2 : a - 10 = M)
  (h3 : b + 10 = M)
  (h4 : 10 * c = M) : 
  M = 1000 / 21 :=
sorry

end find_M_l79_79321


namespace range_of_x_minus_cos_y_l79_79132

theorem range_of_x_minus_cos_y {x y : ℝ} (h : x^2 + 2 * Real.cos y = 1) :
  ∃ (a b : ℝ), ∀ z, z = x - Real.cos y → a ≤ z ∧ z ≤ b ∧ a = -1 ∧ b = 1 + Real.sqrt 3 :=
by
  sorry

end range_of_x_minus_cos_y_l79_79132


namespace triangle_side_length_difference_l79_79613

theorem triangle_side_length_difference (a b c : ℕ) (hb : b = 8) (hc : c = 3)
  (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  let min_a := 6
  let max_a := 10
  max_a - min_a = 4 :=
by {
  sorry
}

end triangle_side_length_difference_l79_79613


namespace find_c1_minus_c2_l79_79129

theorem find_c1_minus_c2 (c1 c2 : ℝ) (h1 : 2 * 3 + 3 * 5 = c1) (h2 : 5 = c2) : c1 - c2 = 16 := by
  sorry

end find_c1_minus_c2_l79_79129


namespace painters_work_days_l79_79282

theorem painters_work_days 
  (six_painters_days : ℝ) (number_six_painters : ℝ) (total_work_units : ℝ)
  (number_four_painters : ℝ) 
  (h1 : number_six_painters = 6)
  (h2 : six_painters_days = 1.4)
  (h3 : total_work_units = number_six_painters * six_painters_days) 
  (h4 : number_four_painters = 4) :
  2 + 1 / 10 = total_work_units / number_four_painters :=
by
  rw [h3, h1, h2, h4]
  sorry

end painters_work_days_l79_79282


namespace theresa_needs_15_hours_l79_79192

theorem theresa_needs_15_hours 
  (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (h4 : ℕ) (h5 : ℕ) (average : ℕ) (weeks : ℕ) (total_hours_first_5 : ℕ) :
  h1 = 10 → h2 = 13 → h3 = 9 → h4 = 14 → h5 = 11 → average = 12 → weeks = 6 → 
  total_hours_first_5 = h1 + h2 + h3 + h4 + h5 → 
  (total_hours_first_5 + x) / weeks = average → x = 15 :=
by
  intros h1_eq h2_eq h3_eq h4_eq h5_eq avg_eq weeks_eq sum_eq avg_eqn
  sorry

end theresa_needs_15_hours_l79_79192


namespace distance_center_to_line_circle_l79_79279

noncomputable def circle_center : ℝ × ℝ := (2, Real.pi / 2)

noncomputable def distance_from_center_to_line (radius : ℝ) (center : ℝ × ℝ) : ℝ :=
  radius * Real.sin (center.snd - Real.pi / 3)

theorem distance_center_to_line_circle : distance_from_center_to_line 2 circle_center = 1 := by
  sorry

end distance_center_to_line_circle_l79_79279


namespace arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l79_79106

-- Definition of the first proof problem
theorem arrangement_with_one_ball_per_box:
  ∃ n : ℕ, n = 24 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that each box has exactly one ball
    n = Nat.factorial 4 :=
by sorry

-- Definition of the second proof problem
theorem arrangement_with_one_empty_box:
  ∃ n : ℕ, n = 144 ∧ 
    -- Number of ways to arrange 4 different balls in 4 boxes such that exactly one box is empty
    n = Nat.choose 4 2 * Nat.factorial 3 :=
by sorry

end arrangement_with_one_ball_per_box_arrangement_with_one_empty_box_l79_79106


namespace compression_strength_value_l79_79135

def compression_strength (T H : ℕ) : ℚ :=
  (15 * T^5) / (H^3)

theorem compression_strength_value : 
  compression_strength 3 6 = 55 / 13 := by
  sorry

end compression_strength_value_l79_79135


namespace fraction_is_five_over_nine_l79_79606

theorem fraction_is_five_over_nine (f k t : ℝ) (h1 : t = f * (k - 32)) (h2 : t = 50) (h3 : k = 122) : f = 5 / 9 :=
by
  sorry

end fraction_is_five_over_nine_l79_79606


namespace find_x_in_sequence_l79_79756

theorem find_x_in_sequence :
  ∃ x : ℕ, x = 32 ∧
    2 + 3 = 5 ∧
    5 + 6 = 11 ∧
    11 + 9 = 20 ∧
    20 + (9 + 3) = x ∧
    x + (9 + 3 + 3) = 47 :=
by
  sorry

end find_x_in_sequence_l79_79756


namespace cody_tickets_l79_79234

theorem cody_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (won_tickets : ℕ) : 
  initial_tickets = 49 ∧ spent_tickets = 25 ∧ won_tickets = 6 → 
  initial_tickets - spent_tickets + won_tickets = 30 :=
by sorry

end cody_tickets_l79_79234


namespace fraction_decomposition_l79_79311

theorem fraction_decomposition :
  ∀ (A B : ℚ), (∀ x : ℚ, x ≠ -2 → x ≠ 4/3 → 
  (7 * x - 15) / ((3 * x - 4) * (x + 2)) = A / (x + 2) + B / (3 * x - 4)) →
  A = 29 / 10 ∧ B = -17 / 10 :=
by
  sorry

end fraction_decomposition_l79_79311


namespace fraction_a_over_b_l79_79710

theorem fraction_a_over_b (x y a b : ℝ) (hb : b ≠ 0) (h1 : 4 * x - 2 * y = a) (h2 : 9 * y - 18 * x = b) :
  a / b = -2 / 9 :=
by
  sorry

end fraction_a_over_b_l79_79710


namespace mountain_bike_cost_l79_79148

theorem mountain_bike_cost (savings : ℕ) (lawns : ℕ) (lawn_rate : ℕ) (newspapers : ℕ) (paper_rate : ℕ) (dogs : ℕ) (dog_rate : ℕ) (remaining : ℕ) (total_earned : ℕ) (total_before_purchase : ℕ) (cost : ℕ) : 
  savings = 1500 ∧ lawns = 20 ∧ lawn_rate = 20 ∧ newspapers = 600 ∧ paper_rate = 40 ∧ dogs = 24 ∧ dog_rate = 15 ∧ remaining = 155 ∧ 
  total_earned = (lawns * lawn_rate) + (newspapers * paper_rate / 100) + (dogs * dog_rate) ∧
  total_before_purchase = savings + total_earned ∧
  cost = total_before_purchase - remaining →
  cost = 2345 := by
  sorry

end mountain_bike_cost_l79_79148


namespace planted_fraction_l79_79718

theorem planted_fraction (length width radius : ℝ) (h_field : length * width = 24)
  (h_circle : π * radius^2 = π) : (24 - π) / 24 = (24 - π) / 24 :=
by
  -- all proofs are skipped
  sorry

end planted_fraction_l79_79718


namespace glycerin_percentage_proof_l79_79226

-- Conditions given in problem
def original_percentage : ℝ := 0.90
def original_volume : ℝ := 4
def added_volume : ℝ := 0.8

-- Total glycerin in original solution
def glycerin_amount : ℝ := original_percentage * original_volume

-- Total volume after adding water
def new_volume : ℝ := original_volume + added_volume

-- Desired percentage proof statement
theorem glycerin_percentage_proof : 
  (glycerin_amount / new_volume) * 100 = 75 := 
by
  sorry

end glycerin_percentage_proof_l79_79226


namespace desktops_to_sell_l79_79746

theorem desktops_to_sell (laptops desktops : ℕ) (ratio_laptops desktops_sold laptops_expected : ℕ) :
  ratio_laptops = 5 → desktops_sold = 3 → laptops_expected = 40 → 
  desktops = (desktops_sold * laptops_expected) / ratio_laptops :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry -- This is where the proof would go, but it's not needed for this task

end desktops_to_sell_l79_79746


namespace three_exp_eq_l79_79592

theorem three_exp_eq (y : ℕ) (h : 3^y + 3^y + 3^y = 2187) : y = 6 :=
by
  sorry

end three_exp_eq_l79_79592


namespace sheilas_family_contribution_l79_79913

theorem sheilas_family_contribution :
  let initial_amount := 3000
  let monthly_savings := 276
  let duration_years := 4
  let total_after_duration := 23248
  let months_in_year := 12
  let total_months := duration_years * months_in_year
  let savings_over_duration := monthly_savings * total_months
  let sheilas_total_savings := initial_amount + savings_over_duration
  let family_contribution := total_after_duration - sheilas_total_savings
  family_contribution = 7000 :=
by
  sorry

end sheilas_family_contribution_l79_79913


namespace number_of_educated_employees_l79_79749

-- Define the context and input values
variable (T: ℕ) (I: ℕ := 20) (decrease_illiterate: ℕ := 15) (total_decrease_illiterate: ℕ := I * decrease_illiterate) (average_salary_decrease: ℕ := 10)

-- The theorem statement
theorem number_of_educated_employees (h1: total_decrease_illiterate / T = average_salary_decrease) (h2: T = I + 10): L = 10 := by
  sorry

end number_of_educated_employees_l79_79749


namespace num_integers_between_sqrt_range_l79_79460

theorem num_integers_between_sqrt_range :
  {x : ℕ | 5 > Real.sqrt x ∧ Real.sqrt x > 3}.card = 15 :=
by sorry

end num_integers_between_sqrt_range_l79_79460


namespace sequence_sum_l79_79726

theorem sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) 
  (H_n_def : H_n = (a 1 + (2:ℕ) * a 2 + (2:ℕ) ^ (n - 1) * a n) / n)
  (H_n_val : H_n = 2^n) :
  S n = n * (n + 3) / 2 :=
by
  sorry

end sequence_sum_l79_79726


namespace ralph_squares_count_l79_79573

def total_matchsticks := 50
def elvis_square_sticks := 4
def ralph_square_sticks := 8
def elvis_squares := 5
def leftover_sticks := 6

theorem ralph_squares_count : 
  ∃ R : ℕ, 
  (elvis_squares * elvis_square_sticks) + (R * ralph_square_sticks) + leftover_sticks = total_matchsticks ∧ R = 3 :=
by 
  sorry

end ralph_squares_count_l79_79573


namespace total_hike_time_l79_79624

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l79_79624


namespace xy_value_l79_79276

noncomputable def x (y : ℝ) : ℝ := 36 * y

theorem xy_value (y : ℝ) (h1 : y = 0.16666666666666666) : x y * y = 1 :=
by
  rw [h1, x]
  sorry

end xy_value_l79_79276


namespace suzy_final_books_l79_79808

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l79_79808


namespace GregsAgeIs16_l79_79978

def CindyAge := 5
def JanAge := CindyAge + 2
def MarciaAge := 2 * JanAge
def GregAge := MarciaAge + 2

theorem GregsAgeIs16 : GregAge = 16 := by
  sorry

end GregsAgeIs16_l79_79978


namespace intersection_M_N_eq_l79_79158

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N_eq : M ∩ N = {x | 1 ≤ x ∧ x < 2} := by
  sorry

end intersection_M_N_eq_l79_79158


namespace work_days_together_l79_79684

theorem work_days_together (A_rate B_rate : ℚ) (h1 : A_rate = 1 / 12) (h2 : B_rate = 5 / 36) : 
  1 / (A_rate + B_rate) = 4.5 := by
  sorry

end work_days_together_l79_79684


namespace find_angle_B_find_max_k_l79_79277

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

end find_angle_B_find_max_k_l79_79277


namespace speed_of_truck_l79_79222

theorem speed_of_truck
  (v : ℝ)                         -- Let \( v \) be the speed of the truck.
  (car_speed : ℝ := 55)           -- Car speed is 55 mph.
  (start_delay : ℝ := 1)          -- Truck starts 1 hour later.
  (catchup_time : ℝ := 6.5)       -- Truck takes 6.5 hours to pass the car.
  (additional_distance_car : ℝ := car_speed * catchup_time)  -- Additional distance covered by the car in 6.5 hours.
  (total_distance_truck : ℝ := car_speed * start_delay + additional_distance_car)  -- Total distance truck must cover to pass the car.
  (truck_distance_eq : v * catchup_time = total_distance_truck)  -- Distance equation for the truck.
  : v = 63.46 :=                -- Prove the truck's speed is 63.46 mph.
by
  -- Original problem solution confirms truck's speed as 63.46 mph. 
  sorry

end speed_of_truck_l79_79222


namespace BANANA_arrangements_l79_79521

theorem BANANA_arrangements : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by 
  sorry

end BANANA_arrangements_l79_79521


namespace arith_seq_sum_signs_l79_79414

variable {α : Type*} [LinearOrderedField α]
variable {a : ℕ → α} {S : ℕ → α} {d : α}

noncomputable def is_arith_seq (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  n * (a 1 + a n) / 2

-- Given conditions
variable (a_8_neg : a 8 < 0)
variable (a_9_pos : a 9 > 0)
variable (a_9_greater_abs_a_8 : a 9 > abs (a 8))

-- The theorem to prove
theorem arith_seq_sum_signs (h : is_arith_seq a) :
  (∀ n, n ≤ 15 → sum_first_n_terms a n < 0) ∧ (∀ n, n ≥ 16 → sum_first_n_terms a n > 0) :=
sorry

end arith_seq_sum_signs_l79_79414


namespace radius_of_cone_is_8_l79_79678

noncomputable def r_cylinder := 8 -- cm
noncomputable def h_cylinder := 2 -- cm
noncomputable def h_cone := 6 -- cm

theorem radius_of_cone_is_8 :
  exists (r_cone : ℝ), r_cone = 8 ∧ π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone :=
by
  let r_cone := 8
  have eq_volumes : π * r_cylinder^2 * h_cylinder = (1 / 3) * π * r_cone^2 * h_cone := 
    sorry
  exact ⟨r_cone, by simp, eq_volumes⟩

end radius_of_cone_is_8_l79_79678


namespace probability_john_david_chosen_l79_79425

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem probability_john_david_chosen :
  let total_workers := 6
  let choose_two := choose total_workers 2
  let favorable_outcomes := 1
  choose_two = 15 → (favorable_outcomes / choose_two : ℝ) = 1 / 15 :=
by
  intros
  sorry

end probability_john_david_chosen_l79_79425


namespace pears_morning_sales_l79_79829

theorem pears_morning_sales (morning afternoon : ℕ) 
  (h1 : afternoon = 2 * morning)
  (h2 : morning + afternoon = 360) : 
  morning = 120 := 
sorry

end pears_morning_sales_l79_79829


namespace clock_in_probability_l79_79230

-- Definitions
def start_time := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_start := 495 -- 8:15 in minutes from 00:00 (495 minutes)
def arrival_start := 470 -- 7:50 in minutes from 00:00 (470 minutes)
def arrival_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)
def valid_clock_in_end := 510 -- 8:30 in minutes from 00:00 (510 minutes)

-- Conditions
def arrival_window := arrival_end - arrival_start -- 40 minutes window
def valid_clock_in_window := valid_clock_in_end - valid_clock_in_start -- 15 minutes window

-- Required proof statement
theorem clock_in_probability :
  (valid_clock_in_window : ℚ) / (arrival_window : ℚ) = 3 / 8 :=
by
  sorry

end clock_in_probability_l79_79230


namespace prob_three_blue_is_correct_l79_79216

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l79_79216


namespace red_ball_probability_l79_79886

theorem red_ball_probability : 
  let red_A := 2
  let white_A := 3
  let red_B := 4
  let white_B := 1
  let total_A := red_A + white_A
  let total_B := red_B + white_B
  let prob_red_A := red_A / total_A
  let prob_white_A := white_A / total_A
  let prob_red_B_after_red_A := (red_B + 1) / (total_B + 1)
  let prob_red_B_after_white_A := red_B / (total_B + 1)
  (prob_red_A * prob_red_B_after_red_A + prob_white_A * prob_red_B_after_white_A) = 11 / 15 :=
by {
  sorry
}

end red_ball_probability_l79_79886


namespace prime_count_of_first_10_sums_is_2_l79_79303

open Nat

def consecutivePrimes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def consecutivePrimeSums (n : Nat) : List Nat :=
  (List.range n).scanl (λ sum i => sum + consecutivePrimes.getD i 0) 0

theorem prime_count_of_first_10_sums_is_2 :
  let sums := consecutivePrimeSums 10;
  (sums.count isPrime) = 2 :=
by
  sorry

end prime_count_of_first_10_sums_is_2_l79_79303


namespace max_bench_weight_support_l79_79427

/-- Definitions for the given problem conditions -/
def john_weight : ℝ := 250
def bar_weight : ℝ := 550
def total_weight : ℝ := john_weight + bar_weight
def safety_percentage : ℝ := 0.80

/-- Theorem stating the maximum weight the bench can support given the conditions -/
theorem max_bench_weight_support :
  ∀ (W : ℝ), safety_percentage * W = total_weight → W = 1000 :=
by
  sorry

end max_bench_weight_support_l79_79427


namespace correct_calculation_l79_79005

theorem correct_calculation (x : ℕ) (h1 : 21 * x = 63) : x + 40 = 43 :=
by
  -- proof steps would go here, but we skip them with 'sorry'
  sorry

end correct_calculation_l79_79005


namespace count_integers_satisfying_sqrt_condition_l79_79465

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l79_79465


namespace initial_average_mark_l79_79173

theorem initial_average_mark (A : ℝ) (n : ℕ) (excluded_avg remaining_avg : ℝ) :
  n = 25 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (A * n = (n - 5) * remaining_avg + 5 * excluded_avg) →
  A = 80 :=
by
  intros hn_hexcluded_avg hremaining_avg htotal_correct
  sorry

end initial_average_mark_l79_79173


namespace binom_12_11_eq_12_l79_79509

theorem binom_12_11_eq_12 : nat.choose 12 11 = 12 := 
by {
  sorry
}

end binom_12_11_eq_12_l79_79509


namespace recommendation_plans_count_l79_79934

theorem recommendation_plans_count :
  let total_students := 7
  let sports_talents := 2
  let artistic_talents := 2
  let other_talents := 3
  let recommend_count := 4
  let condition_sports := recommend_count >= 1
  let condition_artistic := recommend_count >= 1
  (condition_sports ∧ condition_artistic) → 
  ∃ (n : ℕ), n = 25 := sorry

end recommendation_plans_count_l79_79934


namespace backup_settings_required_l79_79772

-- Definitions for the given conditions
def weight_of_silverware_piece : ℕ := 4
def pieces_of_silverware_per_setting : ℕ := 3
def weight_of_plate : ℕ := 12
def plates_per_setting : ℕ := 2
def tables : ℕ := 15
def settings_per_table : ℕ := 8
def total_weight_ounces : ℕ := 5040

-- Statement to prove
theorem backup_settings_required :
  (total_weight_ounces - 
     (tables * settings_per_table) * 
       (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
        plates_per_setting * weight_of_plate)) /
  (pieces_of_silverware_per_setting * weight_of_silverware_piece + 
   plates_per_setting * weight_of_plate) = 20 := 
by sorry

end backup_settings_required_l79_79772


namespace a_b_c_at_least_one_not_less_than_one_third_l79_79054

theorem a_b_c_at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  ¬ (a < 1/3 ∧ b < 1/3 ∧ c < 1/3) :=
by
  sorry

end a_b_c_at_least_one_not_less_than_one_third_l79_79054


namespace inequality_proof_l79_79287

variable (a b c : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : c > 0)
variable (h4 : a + b + c = 1)

theorem inequality_proof : 
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := 
by 
  sorry

end inequality_proof_l79_79287


namespace suzy_final_books_l79_79807

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l79_79807


namespace profit_per_metre_l79_79078

/-- 
Given:
1. A trader sells 85 meters of cloth for Rs. 8925.
2. The cost price of one metre of cloth is Rs. 95.

Prove:
The profit per metre of cloth is Rs. 10.
-/
theorem profit_per_metre 
  (SP : ℕ) (CP : ℕ)
  (total_SP : SP = 8925)
  (total_meters : ℕ := 85)
  (cost_per_meter : CP = 95) :
  (SP - total_meters * CP) / total_meters = 10 :=
by
  sorry

end profit_per_metre_l79_79078


namespace quadratic_polynomials_perfect_square_l79_79851

variables {x y p q a b c : ℝ}

theorem quadratic_polynomials_perfect_square (h1 : ∃ a, x^2 + p * x + q = (x + a) * (x + a))
  (h2 : ∃ a b, a^2 * x^2 + 2 * b^2 * x * y + c^2 * y^2 = (a * x + b * y) * (a * x + b * y)) :
  q = (p^2 / 4) ∧ b^2 = a * c :=
by
  sorry

end quadratic_polynomials_perfect_square_l79_79851


namespace proof_expression_value_l79_79010

theorem proof_expression_value (x y : ℝ) (h : x + 2 * y = 30) : 
  (x / 5 + 2 * y / 3 + 2 * y / 5 + x / 3) = 16 := 
by 
  sorry

end proof_expression_value_l79_79010


namespace solution_of_two_quadratics_l79_79897

theorem solution_of_two_quadratics (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 := 
by 
  sorry

end solution_of_two_quadratics_l79_79897


namespace complex_imaginary_condition_l79_79016

theorem complex_imaginary_condition (m : ℝ) : (∀ m : ℝ, (m^2 - 3*m - 4 = 0) → (m^2 - 5*m - 6) ≠ 0) ↔ (m ≠ -1 ∧ m ≠ 6) :=
by
  sorry

end complex_imaginary_condition_l79_79016


namespace real_solutions_infinite_l79_79720

theorem real_solutions_infinite : 
  ∃ (S : Set ℝ), (∀ x ∈ S, - (x^2 - 4) ≥ 0) ∧ S.Infinite :=
sorry

end real_solutions_infinite_l79_79720


namespace rides_first_day_l79_79501

variable (total_rides : ℕ) (second_day_rides : ℕ)

theorem rides_first_day (h1 : total_rides = 7) (h2 : second_day_rides = 3) : total_rides - second_day_rides = 4 :=
by
  sorry

end rides_first_day_l79_79501


namespace BANANA_arrangements_l79_79514

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l79_79514


namespace find_number_of_small_gardens_l79_79953

-- Define the conditions
def seeds_total : Nat := 52
def seeds_big_garden : Nat := 28
def seeds_per_small_garden : Nat := 4

-- Define the target value
def num_small_gardens : Nat := 6

-- The statement of the proof problem
theorem find_number_of_small_gardens 
  (H1 : seeds_total = 52) 
  (H2 : seeds_big_garden = 28) 
  (H3 : seeds_per_small_garden = 4) 
  : seeds_total - seeds_big_garden = 24 ∧ (seeds_total - seeds_big_garden) / seeds_per_small_garden = num_small_gardens := 
sorry

end find_number_of_small_gardens_l79_79953


namespace prob_AB_diff_homes_l79_79360

-- Define the volunteers
inductive Volunteer : Type
| A | B | C | D | E

open Volunteer

-- Define the homes
inductive Home : Type
| home1 | home2

open Home

-- Total number of ways to distribute the volunteers
def total_ways : ℕ := 2^5  -- Each volunteer has independently 2 choices

-- Number of ways in which A and B are in different homes
def diff_ways : ℕ := 2 * 4 * 2^3  -- Split the problem down by cases for simplicity

-- Calculate the probability
def probability : ℚ := diff_ways / total_ways

-- The final statement to prove
theorem prob_AB_diff_homes : probability = 8 / 15 := sorry

end prob_AB_diff_homes_l79_79360


namespace cricket_run_rate_l79_79067

theorem cricket_run_rate (run_rate_first_10_overs : ℝ) (total_target : ℝ) (overs_first_period : ℕ) (overs_remaining_period : ℕ)
  (h1 : run_rate_first_10_overs = 3.2)
  (h2 : total_target = 252)
  (h3 : overs_first_period = 10)
  (h4 : overs_remaining_period = 40) :
  (total_target - (run_rate_first_10_overs * overs_first_period)) / overs_remaining_period = 5.5 := 
by
  sorry

end cricket_run_rate_l79_79067


namespace mutually_exclusive_not_opposite_l79_79418

-- Define the given conditions
def boys := 6
def girls := 5
def total_students := boys + girls
def selection := 3

-- Define the mutually exclusive and not opposite events
def event_at_least_2_boys := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (b ≥ 2) ∧ (g ≤ (selection - b))
def event_at_least_2_girls := ∃ (b: ℕ), ∃ (g: ℕ), (b + g = selection) ∧ (g ≥ 2) ∧ (b ≤ (selection - g))

-- Statement that these events are mutually exclusive but not opposite
theorem mutually_exclusive_not_opposite :
  (event_at_least_2_boys ∧ event_at_least_2_girls) → 
  (¬ ((∃ (b: ℕ) (g: ℕ), b + g = selection ∧ b ≥ 2 ∧ g ≥ 2) ∧ ¬(event_at_least_2_boys))) :=
sorry

end mutually_exclusive_not_opposite_l79_79418


namespace perm_banana_l79_79562

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l79_79562


namespace permutations_of_BANANA_l79_79527

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l79_79527


namespace evaluate_expression_l79_79800

def f (x : ℕ) : ℕ :=
  match x with
  | 3 => 10
  | 4 => 17
  | 5 => 26
  | 6 => 37
  | 7 => 50
  | _ => 0  -- for any x not in the table, f(x) is undefined and defaults to 0

def f_inv (y : ℕ) : ℕ :=
  match y with
  | 10 => 3
  | 17 => 4
  | 26 => 5
  | 37 => 6
  | 50 => 7
  | _ => 0  -- for any y not in the table, f_inv(y) is undefined and defaults to 0

theorem evaluate_expression :
  f_inv (f_inv 50 * f_inv 10 + f_inv 26) = 5 :=
by
  sorry

end evaluate_expression_l79_79800


namespace chinaman_change_possible_l79_79648

def pence (x : ℕ) := x -- defining the value of pence as a natural number

def ching_chang_by_value (d : ℕ) := 
  (2 * pence d) + (4 * (2 * pence d) / 15)

def equivalent_value_of_half_crown (d : ℕ) := 30 * pence d

def coin_value_with_holes (holes_value : ℕ) (value_per_eleven : ℕ) := 
  (value_per_eleven * ching_chang_by_value 1) / 11

theorem chinaman_change_possible :
  ∃ (x y z : ℕ), 
  (7 * coin_value_with_holes 15 11) + (1 * coin_value_with_holes 16 11) + (0 * coin_value_with_holes 17 11) = 
  equivalent_value_of_half_crown 1 :=
sorry

end chinaman_change_possible_l79_79648


namespace hannah_remaining_money_l79_79125

-- Define the conditions of the problem
def initial_amount : Nat := 120
def rides_cost : Nat := initial_amount * 40 / 100
def games_cost : Nat := initial_amount * 15 / 100
def remaining_after_rides_games : Nat := initial_amount - rides_cost - games_cost

def dessert_cost : Nat := 8
def cotton_candy_cost : Nat := 5
def hotdog_cost : Nat := 6
def keychain_cost : Nat := 7
def poster_cost : Nat := 10
def additional_attraction_cost : Nat := 15
def total_food_souvenirs_cost : Nat := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + additional_attraction_cost

def final_remaining_amount : Nat := remaining_after_rides_games - total_food_souvenirs_cost

-- Formulate the theorem to prove
theorem hannah_remaining_money : final_remaining_amount = 3 := by
  sorry

end hannah_remaining_money_l79_79125


namespace number_of_k_for_lcm_l79_79853

theorem number_of_k_for_lcm (a b : ℕ) :
  (∀ a b, k = 2^a * 3^b) → 
  (∀ (a : ℕ), 0 ≤ a ∧ a ≤ 24) →
  (∃ b, b = 12) →
  (∀ k, k = 2^a * 3^b) →
  (Nat.lcm (Nat.lcm (6^6) (8^8)) k = 12^12) :=
sorry

end number_of_k_for_lcm_l79_79853


namespace sum_of_A_and_B_l79_79326

theorem sum_of_A_and_B (A B : ℕ) (h1 : 7 - B = 3) (h2 : A - 5 = 4) (h_diff : A ≠ B) : A + B = 13 :=
sorry

end sum_of_A_and_B_l79_79326


namespace percentage_music_l79_79664

variable (students_total : ℕ)
variable (percent_dance percent_art percent_drama percent_sports percent_photography percent_music : ℝ)

-- Define the problem conditions
def school_conditions : Prop :=
  students_total = 3000 ∧
  percent_dance = 0.125 ∧
  percent_art = 0.22 ∧
  percent_drama = 0.135 ∧
  percent_sports = 0.15 ∧
  percent_photography = 0.08 ∧
  percent_music = 1 - (percent_dance + percent_art + percent_drama + percent_sports + percent_photography)

-- Define the proof statement
theorem percentage_music : school_conditions students_total percent_dance percent_art percent_drama percent_sports percent_photography percent_music → percent_music = 0.29 :=
by
  intros h
  rw [school_conditions] at h
  sorry

end percentage_music_l79_79664


namespace first_term_of_arithmetic_progression_l79_79852

theorem first_term_of_arithmetic_progression 
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (nth_term_eq : a + (n - 1) * d = 26)
  (common_diff : d = 2)
  (term_num : n = 10) : 
  a = 8 := 
by 
  sorry

end first_term_of_arithmetic_progression_l79_79852


namespace total_servings_of_vegetables_l79_79758

def carrot_plant_serving : ℕ := 4
def num_green_bean_plants : ℕ := 10
def num_carrot_plants : ℕ := 8
def num_corn_plants : ℕ := 12
def num_tomato_plants : ℕ := 15
def corn_plant_serving : ℕ := 5 * carrot_plant_serving
def green_bean_plant_serving : ℕ := corn_plant_serving / 2
def tomato_plant_serving : ℕ := carrot_plant_serving + 3

theorem total_servings_of_vegetables :
  (num_carrot_plants * carrot_plant_serving) +
  (num_corn_plants * corn_plant_serving) +
  (num_green_bean_plants * green_bean_plant_serving) +
  (num_tomato_plants * tomato_plant_serving) = 477 := by
  sorry

end total_servings_of_vegetables_l79_79758


namespace no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l79_79027

theorem no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1 :
  ∀ (a b n : ℕ), (a > 1) → (b > 1) → (a ∣ 2^n - 1) → (b ∣ 2^n + 1) → ∀ (k : ℕ), ¬ (a ∣ 2^k + 1 ∧ b ∣ 2^k - 1) :=
by
  intros a b n a_gt_1 b_gt_1 a_div_2n_minus_1 b_div_2n_plus_1 k
  sorry

end no_k_such_that_a_divides_2k_plus_1_and_b_divides_2k_minus_1_l79_79027


namespace initial_strawberry_plants_l79_79435

theorem initial_strawberry_plants (P : ℕ) (h1 : 24 * P - 4 = 500) : P = 21 := 
by
  sorry

end initial_strawberry_plants_l79_79435


namespace lucy_apples_per_week_l79_79708

-- Define the conditions
def chandler_apples_per_week := 23
def total_apples_per_month := 168
def weeks_per_month := 4
def chandler_apples_per_month := chandler_apples_per_week * weeks_per_month
def lucy_apples_per_month := total_apples_per_month - chandler_apples_per_month

-- Define the proof problem statement
theorem lucy_apples_per_week :
  lucy_apples_per_month / weeks_per_month = 19 :=
  by sorry

end lucy_apples_per_week_l79_79708


namespace time_after_seconds_l79_79025

def initial_time : Nat := 8 * 60 * 60 -- 8:00:00 a.m. in seconds
def seconds_passed : Nat := 8035
def target_time : Nat := (10 * 60 * 60 + 13 * 60 + 35) -- 10:13:35 in seconds

theorem time_after_seconds : initial_time + seconds_passed = target_time := by
  -- proof skipped
  sorry

end time_after_seconds_l79_79025


namespace scientific_notation_correct_l79_79986

theorem scientific_notation_correct :
  0.00000164 = 1.64 * 10^(-6) :=
sorry

end scientific_notation_correct_l79_79986


namespace sum_of_three_smallest_positive_solutions_l79_79102

theorem sum_of_three_smallest_positive_solutions :
  let sol1 := 2
  let sol2 := 8 / 3
  let sol3 := 7 / 2
  sol1 + sol2 + sol3 = 8 + 1 / 6 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_l79_79102


namespace wall_print_costs_are_15_l79_79242

-- Define the cost of curtains, installation, total cost, and number of wall prints.
variable (cost_curtain : ℕ := 30)
variable (num_curtains : ℕ := 2)
variable (cost_installation : ℕ := 50)
variable (num_wall_prints : ℕ := 9)
variable (total_cost : ℕ := 245)

-- Define the total cost of curtains
def total_cost_curtains : ℕ := num_curtains * cost_curtain

-- Define the total fixed costs
def total_fixed_costs : ℕ := total_cost_curtains + cost_installation

-- Define the total cost of wall prints
def total_cost_wall_prints : ℕ := total_cost - total_fixed_costs

-- Define the cost per wall print
def cost_per_wall_print : ℕ := total_cost_wall_prints / num_wall_prints

-- Prove the cost per wall print is $15.00
theorem wall_print_costs_are_15 : cost_per_wall_print = 15 := by
  -- This is a placeholder for the proof
  sorry

end wall_print_costs_are_15_l79_79242


namespace cost_of_toys_l79_79436

theorem cost_of_toys (x y : ℝ) (h1 : x + y = 40) (h2 : 90 / x = 150 / y) :
  x = 15 ∧ y = 25 :=
sorry

end cost_of_toys_l79_79436


namespace exists_infinite_n_for_multiple_of_prime_l79_79915

theorem exists_infinite_n_for_multiple_of_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ᶠ n in at_top, 2 ^ n - n ≡ 0 [MOD p] :=
by
  sorry

end exists_infinite_n_for_multiple_of_prime_l79_79915


namespace peter_total_miles_l79_79295

-- Definitions based on the conditions
def minutes_per_mile : ℝ := 20
def miles_walked_already : ℝ := 1
def additional_minutes : ℝ := 30

-- The value we want to prove
def total_miles_to_walk : ℝ := 2.5

-- Theorem statement corresponding to the proof problem
theorem peter_total_miles :
  (additional_minutes / minutes_per_mile) + miles_walked_already = total_miles_to_walk :=
sorry

end peter_total_miles_l79_79295


namespace total_books_is_177_l79_79771

-- Define the number of books read (x), books yet to read (y), and the total number of books (T)
def x : Nat := 13
def y : Nat := 8
def T : Nat := x^2 + y

-- Prove that the total number of books in the series is 177
theorem total_books_is_177 : T = 177 :=
  sorry

end total_books_is_177_l79_79771


namespace percentage_of_girl_scouts_with_slips_l79_79056

-- Define the proposition that captures the problem
theorem percentage_of_girl_scouts_with_slips 
    (total_scouts : ℕ)
    (scouts_with_slips : ℕ := total_scouts * 60 / 100)
    (boy_scouts : ℕ := total_scouts * 45 / 100)
    (boy_scouts_with_slips : ℕ := boy_scouts * 50 / 100)
    (girl_scouts : ℕ := total_scouts - boy_scouts)
    (girl_scouts_with_slips : ℕ := scouts_with_slips - boy_scouts_with_slips) :
  (girl_scouts_with_slips * 100 / girl_scouts) = 68 :=
by 
  -- The proof goes here
  sorry

end percentage_of_girl_scouts_with_slips_l79_79056


namespace total_selling_price_l79_79495

theorem total_selling_price (cost_per_meter profit_per_meter : ℕ) (total_meters : ℕ) :
  cost_per_meter = 90 → 
  profit_per_meter = 15 → 
  total_meters = 85 → 
  (cost_per_meter + profit_per_meter) * total_meters = 8925 :=
by
  intros
  sorry

end total_selling_price_l79_79495


namespace apples_per_pie_l79_79922

theorem apples_per_pie (total_apples : ℕ) (apples_given : ℕ) (pies : ℕ) : 
  total_apples = 47 ∧ apples_given = 27 ∧ pies = 5 →
  (total_apples - apples_given) / pies = 4 :=
by
  intros h
  sorry

end apples_per_pie_l79_79922


namespace freddy_call_duration_l79_79107

theorem freddy_call_duration (total_cost : ℕ) (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ) (local_duration : ℕ)
  (total_cost_eq : total_cost = 1000) -- cost in cents
  (local_cost_eq : local_cost_per_minute = 5)
  (international_cost_eq : international_cost_per_minute = 25)
  (local_duration_eq : local_duration = 45) :
  (total_cost - local_duration * local_cost_per_minute) / international_cost_per_minute = 31 :=
by
  sorry

end freddy_call_duration_l79_79107


namespace cost_price_computer_table_l79_79208

-- Define the variables
def cost_price : ℝ := 3840
def selling_price (CP : ℝ) := CP * 1.25

-- State the conditions and the proof problem
theorem cost_price_computer_table 
  (SP : ℝ) 
  (h1 : SP = 4800)
  (h2 : ∀ CP : ℝ, SP = selling_price CP) :
  cost_price = 3840 :=
by 
  sorry

end cost_price_computer_table_l79_79208


namespace rainfall_second_week_l79_79715

theorem rainfall_second_week (r1 r2 : ℝ) (h1 : r1 + r2 = 35) (h2 : r2 = 1.5 * r1) : r2 = 21 := 
  sorry

end rainfall_second_week_l79_79715


namespace coefficient_of_expression_l79_79923

theorem coefficient_of_expression :
  ∀ (a b : ℝ), (∃ (c : ℝ), - (2/3) * (a * b) = c * (a * b)) :=
by
  intros a b
  use (-2/3)
  sorry

end coefficient_of_expression_l79_79923


namespace initial_group_size_l79_79306

theorem initial_group_size (W : ℝ) : 
  (∃ n : ℝ, (W + 15) / n = W / n + 2.5) → n = 6 :=
by
  sorry

end initial_group_size_l79_79306


namespace binom_1300_2_l79_79365

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l79_79365


namespace equations_neither_directly_nor_inversely_proportional_l79_79371

-- Definitions for equations
def equation1 (x y : ℝ) : Prop := 2 * x + 3 * y = 6
def equation2 (x y : ℝ) : Prop := 4 * x * y = 12
def equation3 (x y : ℝ) : Prop := y = 1/2 * x
def equation4 (x y : ℝ) : Prop := 5 * x - 2 * y = 20
def equation5 (x y : ℝ) : Prop := x / y = 5

-- Theorem stating that y is neither directly nor inversely proportional to x for the given equations
theorem equations_neither_directly_nor_inversely_proportional (x y : ℝ) :
  (¬∃ k : ℝ, x = k * y) ∧ (¬∃ k : ℝ, x * y = k) ↔ 
  (equation1 x y ∨ equation4 x y) :=
sorry

end equations_neither_directly_nor_inversely_proportional_l79_79371


namespace sum_of_numbers_l79_79894

theorem sum_of_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_numbers_l79_79894


namespace intersection_x_value_l79_79323

theorem intersection_x_value : ∃ x y : ℝ, y = 3 * x + 7 ∧ 3 * x - 2 * y = -4 ∧ x = -10 / 3 :=
by
  sorry

end intersection_x_value_l79_79323


namespace xyz_zero_if_equation_zero_l79_79428

theorem xyz_zero_if_equation_zero (x y z : ℚ) 
  (h : x^3 + 3 * y^3 + 9 * z^3 - 9 * x * y * z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := 
by 
  sorry

end xyz_zero_if_equation_zero_l79_79428


namespace unique_solution_for_2_3_6_eq_7_l79_79246

theorem unique_solution_for_2_3_6_eq_7 (x : ℝ) : 2^x + 3^x + 6^x = 7^x → x = 2 :=
by
  intro h
  -- Add the relevant proof tactic steps here
  sorry

end unique_solution_for_2_3_6_eq_7_l79_79246


namespace number_of_rectangles_in_5x5_grid_l79_79003

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l79_79003


namespace right_triangle_345_l79_79971

theorem right_triangle_345 :
  ∃ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 ∧ a^2 + b^2 = c^2 :=
by {
  -- Here, we should construct the proof later
  sorry
}

end right_triangle_345_l79_79971


namespace simplify_and_evaluate_l79_79039

noncomputable def simplified_expression (x : ℝ) : ℝ :=
  ((1 / (x - 1)) + (1 / (x + 1))) / (x^2 / (3 * x^2 - 3))

theorem simplify_and_evaluate : simplified_expression (Real.sqrt 2) = 3 * Real.sqrt 2 :=
by 
  sorry

end simplify_and_evaluate_l79_79039


namespace arrangement_of_BANANA_l79_79548

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l79_79548


namespace joan_football_games_l79_79760

theorem joan_football_games (G_total G_last G_this : ℕ) (h1 : G_total = 13) (h2 : G_last = 9) (h3 : G_this = G_total - G_last) : G_this = 4 :=
by
  sorry

end joan_football_games_l79_79760


namespace books_at_end_l79_79813

-- Define the conditions
def initialBooks : ℕ := 98
def checkoutsWednesday : ℕ := 43
def returnsThursday : ℕ := 23
def checkoutsThursday : ℕ := 5
def returnsFriday : ℕ := 7

-- Define the final number of books and the theorem to prove
def finalBooks : ℕ := initialBooks - checkoutsWednesday + returnsThursday - checkoutsThursday + returnsFriday

-- Prove that the final number of books is 80
theorem books_at_end : finalBooks = 80 := by
  sorry

end books_at_end_l79_79813


namespace chorus_group_membership_l79_79175

theorem chorus_group_membership (n : ℕ) : 
  100 < n ∧ n < 200 →
  n % 3 = 1 ∧ 
  n % 4 = 2 ∧ 
  n % 6 = 4 ∧ 
  n % 8 = 6 →
  n = 118 ∨ n = 142 ∨ n = 166 ∨ n = 190 :=
by
  sorry

end chorus_group_membership_l79_79175


namespace ratio_greater_than_one_ratio_greater_than_one_neg_l79_79931

theorem ratio_greater_than_one (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b > 1) : a > b :=
by
  sorry

theorem ratio_greater_than_one_neg (a b : ℝ) (h1 : a < 0) (h2 : b < 0) (h3 : a / b > 1) : a < b :=
by
  sorry

end ratio_greater_than_one_ratio_greater_than_one_neg_l79_79931


namespace men_in_first_scenario_l79_79408

theorem men_in_first_scenario 
  (M : ℕ) 
  (daily_hours_first weekly_earning_first daily_hours_second weekly_earning_second : ℝ) 
  (number_of_men_second : ℕ)
  (days_per_week : ℕ := 7) 
  (h1 : M * daily_hours_first * days_per_week = weekly_earning_first)
  (h2 : number_of_men_second * daily_hours_second * days_per_week = weekly_earning_second) 
  (h1_value : daily_hours_first = 10) 
  (w1_value : weekly_earning_first = 1400) 
  (h2_value : daily_hours_second = 6) 
  (w2_value : weekly_earning_second = 1890)
  (second_scenario_men : number_of_men_second = 9) : 
  M = 4 :=
by
  sorry

end men_in_first_scenario_l79_79408


namespace sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l79_79705

noncomputable def calculation (x y z : ℝ) : ℝ :=
  (Real.sqrt x * Real.sqrt y) / Real.sqrt z

theorem sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3 :
  calculation 12 27 3 = 6 * Real.sqrt 3 :=
by
  sorry

end sqrt_12_times_sqrt_27_div_sqrt_3_eq_6_sqrt_3_l79_79705


namespace largest_consecutive_odd_sum_l79_79932

theorem largest_consecutive_odd_sum (x : ℤ) (h : 20 * (x + 19) = 8000) : x + 38 = 419 := 
by
  sorry

end largest_consecutive_odd_sum_l79_79932


namespace fractional_shaded_area_l79_79228

noncomputable def geometric_series_sum (a r : ℚ) : ℚ := a / (1 - r)

theorem fractional_shaded_area :
  let a := (7 : ℚ) / 16
  let r := (1 : ℚ) / 16
  geometric_series_sum a r = 7 / 15 :=
by
  sorry

end fractional_shaded_area_l79_79228


namespace train_passes_jogger_in_approx_36_seconds_l79_79824

noncomputable def jogger_speed_kmph : ℝ := 8
noncomputable def train_speed_kmph : ℝ := 55
noncomputable def distance_ahead_m : ℝ := 340
noncomputable def train_length_m : ℝ := 130

noncomputable def kmph_to_mps (speed_kmph : ℝ) : ℝ :=
  (speed_kmph * 1000) / 3600

noncomputable def jogger_speed_mps : ℝ :=
  kmph_to_mps jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  kmph_to_mps train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance_m : ℝ :=
  distance_ahead_m + train_length_m

noncomputable def time_to_pass_jogger_s : ℝ :=
  total_distance_m / relative_speed_mps

theorem train_passes_jogger_in_approx_36_seconds : 
  abs (time_to_pass_jogger_s - 36) < 1 := 
sorry

end train_passes_jogger_in_approx_36_seconds_l79_79824


namespace blue_die_prime_yellow_die_power_2_probability_l79_79329

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end blue_die_prime_yellow_die_power_2_probability_l79_79329


namespace ratio_values_l79_79770

theorem ratio_values (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) 
  (h₀ : (x + y) / z = (y + z) / x) (h₀' : (y + z) / x = (z + x) / y) :
  ∃ a : ℝ, a = -1 ∨ a = 8 :=
sorry

end ratio_values_l79_79770


namespace clothing_probability_l79_79006

theorem clothing_probability :
  let total_clothing := 6 + 8 + 9 + 4,
      ways_to_choose_four := Nat.choose total_clothing 4,
      specific_ways := 6 * 8 * 9 * 4 in
  (6 + 8 + 9 + 4 = 27)
  -> (ways_to_choose_four = 17550)
  -> (specific_ways = 1728)
  -> (specific_ways / ways_to_choose_four = 96 / 975) :=
by sorry

end clothing_probability_l79_79006


namespace find_width_l79_79333

-- Definitions and Conditions
def length : ℝ := 6
def depth : ℝ := 2
def total_surface_area : ℝ := 104

-- Statement to prove the width
theorem find_width (width : ℝ) (h : 12 * width + 4 * width + 24 = total_surface_area) : width = 5 := 
by { 
  -- lean 4 statement only, proof omitted
  sorry 
}

end find_width_l79_79333


namespace sqrt_product_simplification_l79_79918

-- Define the main problem
theorem sqrt_product_simplification : Real.sqrt 18 * Real.sqrt 72 = 36 := 
by
  sorry

end sqrt_product_simplification_l79_79918


namespace principal_amount_l79_79317

theorem principal_amount (SI : ℝ) (T : ℝ) (R : ℝ) (P : ℝ) (h1 : SI = 140) (h2 : T = 2) (h3 : R = 17.5) :
  P = 400 :=
by
  -- Formal proof would go here
  sorry

end principal_amount_l79_79317


namespace train_pass_platform_time_l79_79203

-- Define the conditions given in the problem.
def train_length : ℕ := 1200
def platform_length : ℕ := 1100
def time_to_cross_tree : ℕ := 120

-- Define the calculation for speed.
def speed := train_length / time_to_cross_tree

-- Define the combined length of train and platform.
def combined_length := train_length + platform_length

-- Define the expected time to pass the platform.
def expected_time_to_pass_platform := combined_length / speed

-- The theorem to prove.
theorem train_pass_platform_time :
  expected_time_to_pass_platform = 230 :=
by {
  -- Placeholder for the proof.
  sorry
}

end train_pass_platform_time_l79_79203


namespace lines_intersection_l79_79803

theorem lines_intersection (n c : ℝ) : 
    (∀ x y : ℝ, y = n * x + 5 → y = 4 * x + c → (x, y) = (8, 9)) → 
    n + c = -22.5 := 
by
    intro h
    sorry

end lines_intersection_l79_79803


namespace housewife_oil_expense_l79_79827

theorem housewife_oil_expense:
  ∃ M P R: ℝ, (R = 30) ∧ (0.8 * P = R) ∧ ((M / R) - (M / P) = 10) ∧ (M = 1500) :=
by
  sorry

end housewife_oil_expense_l79_79827


namespace percentage_heavier_l79_79426

variables (J M : ℝ)

theorem percentage_heavier (hM : M ≠ 0) : 
  100 * ((J + 3) - M) / M = 100 * ((J + 3) - M) / M := 
sorry

end percentage_heavier_l79_79426


namespace find_P_and_Q_l79_79401

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l79_79401


namespace find_b_neg_l79_79154

noncomputable def h (x : ℝ) : ℝ := if x ≤ 0 then -x else 3 * x - 50

theorem find_b_neg (b : ℝ) (h_neg_b : b < 0) : 
  h (h (h 15)) = h (h (h b)) → b = - (55 / 3) :=
by
  sorry

end find_b_neg_l79_79154


namespace find_x_l79_79356

theorem find_x :
  ∃ X : ℝ, 0.25 * X + 0.20 * 40 = 23 ∧ X = 60 :=
by
  sorry

end find_x_l79_79356


namespace geometric_sequence_properties_l79_79416

noncomputable def geometric_sequence (a2 a5 : ℕ) (n : ℕ) : ℕ :=
  3 ^ (n - 1)

noncomputable def sum_first_n_terms (n : ℕ) : ℕ :=
  (3^n - 1) / 2

def T10_sum_of_sequence : ℚ := 10/11

theorem geometric_sequence_properties :
  (geometric_sequence 3 81 2 = 3) ∧
  (geometric_sequence 3 81 5 = 81) ∧
  (sum_first_n_terms 2 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2)) ∧
  (sum_first_n_terms 5 = (geometric_sequence 3 81 1 + geometric_sequence 3 81 2 + geometric_sequence 3 81 3 + geometric_sequence 3 81 4 + geometric_sequence 3 81 5)) ∧
  T10_sum_of_sequence = 10/11 :=
by
  sorry

end geometric_sequence_properties_l79_79416


namespace sqrt_sixteen_l79_79651

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l79_79651


namespace seventeenth_replacement_month_l79_79632

def months_after_january (n : Nat) : Nat :=
  n % 12

theorem seventeenth_replacement_month :
  months_after_january (7 * 16) = 4 :=
by
  sorry

end seventeenth_replacement_month_l79_79632


namespace modulus_of_complex_number_l79_79121

/-- Definition of the imaginary unit i defined as the square root of -1 --/
def i : ℂ := Complex.I

/-- Statement that the modulus of z = i (1 - i) equals sqrt(2) --/
theorem modulus_of_complex_number : Complex.abs (i * (1 - i)) = Real.sqrt 2 :=
sorry

end modulus_of_complex_number_l79_79121


namespace slower_train_speed_l79_79051

noncomputable def speed_of_slower_train (v_f : ℕ) (l1 l2 : ℚ) (t : ℚ) : ℚ :=
  let total_distance := l1 + l2
  let time_in_hours := t / 3600
  let relative_speed := total_distance / time_in_hours
  relative_speed - v_f

theorem slower_train_speed :
  speed_of_slower_train 210 (11 / 10) (9 / 10) 24 = 90 := by
  sorry

end slower_train_speed_l79_79051


namespace decrease_percent_revenue_l79_79467

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.68 * T
  let new_consumption := 1.12 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 23.84 := by {
    sorry
  }

end decrease_percent_revenue_l79_79467


namespace fred_initial_dimes_l79_79379

theorem fred_initial_dimes (current_dimes borrowed_dimes initial_dimes : ℕ)
  (hc : current_dimes = 4)
  (hb : borrowed_dimes = 3)
  (hi : current_dimes + borrowed_dimes = initial_dimes) :
  initial_dimes = 7 := 
by
  sorry

end fred_initial_dimes_l79_79379


namespace find_salary_of_january_l79_79681

variables (J F M A May : ℝ)

theorem find_salary_of_january
  (h1 : (J + F + M + A) / 4 = 8000)
  (h2 : (F + M + A + May) / 4 = 8600)
  (h3 : May = 6500) :
  J = 4100 := 
sorry

end find_salary_of_january_l79_79681


namespace banana_permutations_l79_79542

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l79_79542


namespace field_trip_count_l79_79822

theorem field_trip_count (vans: ℕ) (buses: ℕ) (people_per_van: ℕ) (people_per_bus: ℕ)
  (hv: vans = 9) (hb: buses = 10) (hpv: people_per_van = 8) (hpb: people_per_bus = 27):
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end field_trip_count_l79_79822


namespace comparison_abc_l79_79635

noncomputable def a : ℝ := 0.98 + Real.sin 0.01
noncomputable def b : ℝ := Real.exp (-0.01)
noncomputable def c : ℝ := 0.5 * (Real.log 2023 / Real.log 2022 + Real.log 2022 / Real.log 2023)

theorem comparison_abc : c > b ∧ b > a := by
  sorry

end comparison_abc_l79_79635


namespace men_in_first_group_l79_79015

variable (M : ℕ) (daily_wage : ℝ)
variable (h1 : M * 10 * daily_wage = 1200)
variable (h2 : 9 * 6 * daily_wage = 1620)
variable (dw_eq : daily_wage = 30)

theorem men_in_first_group : M = 4 :=
by sorry

end men_in_first_group_l79_79015


namespace number_of_ordered_pairs_l79_79250

-- Formal statement of the problem in Lean 4
theorem number_of_ordered_pairs : 
  ∃ (n : ℕ), n = 128 ∧ 
  ∀ (a b : ℝ), (∃ (x y : ℤ), (a * x + b * y = 1) ∧ (x^2 + y^2 = 65)) ↔ n = 128 :=
sorry

end number_of_ordered_pairs_l79_79250


namespace student_count_l79_79450

noncomputable def numberOfStudents (decreaseInAverageWeight totalWeightDecrease : ℕ) : ℕ :=
  totalWeightDecrease / decreaseInAverageWeight

theorem student_count 
  (decreaseInAverageWeight : ℕ)
  (totalWeightDecrease : ℕ)
  (condition_avg_weight_decrease : decreaseInAverageWeight = 4)
  (condition_weight_difference : totalWeightDecrease = 92 - 72) :
  numberOfStudents decreaseInAverageWeight totalWeightDecrease = 5 := by 
  -- We are not providing the proof details as per the instruction
  sorry

end student_count_l79_79450


namespace fraction_is_percent_l79_79059

theorem fraction_is_percent (y : ℝ) (hy : y > 0) : (6 * y / 20 + 3 * y / 10) = (60 / 100) * y :=
by
  sorry

end fraction_is_percent_l79_79059


namespace speed_of_stream_l79_79682

-- Define the speed of the boat in still water
def speed_of_boat_in_still_water : ℝ := 39

-- Define the effective speed upstream and downstream
def effective_speed_upstream (v : ℝ) : ℝ := speed_of_boat_in_still_water - v
def effective_speed_downstream (v : ℝ) : ℝ := speed_of_boat_in_still_water + v

-- Define the condition that time upstream is twice the time downstream
def time_condition (D v : ℝ) : Prop := 
  (D / effective_speed_upstream v = 2 * (D / effective_speed_downstream v))

-- The main theorem stating the speed of the stream
theorem speed_of_stream (D : ℝ) (h : D > 0) : (v : ℝ) → time_condition D v → v = 13 :=
by
  sorry

end speed_of_stream_l79_79682


namespace probability_blue_prime_and_yellow_power_of_two_l79_79328

-- Definitions
def blue_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def yellow_die_outcomes := { n : ℕ | 1 ≤ n ∧ n ≤ 8 }
def prime_numbers := {2, 3, 5, 7}
def powers_of_two := {1, 2, 4, 8}

-- Theorem statement
theorem probability_blue_prime_and_yellow_power_of_two :
  (finset.card (prime_numbers ×ˢ powers_of_two) : ℚ) / (finset.card (blue_die_outcomes ×ˢ yellow_die_outcomes) : ℚ) = 1 / 4 :=
by
  sorry

end probability_blue_prime_and_yellow_power_of_two_l79_79328


namespace lines_do_not_intersect_l79_79713

theorem lines_do_not_intersect (b : ℝ) :
  ∀ s v : ℝ,
    (2 + 3 * s = 5 + 6 * v) →
    (1 + 4 * s = 3 + 3 * v) →
    (b + 5 * s = 1 + 2 * v) →
    b ≠ -4/5 :=
by
  intros s v h1 h2 h3
  sorry

end lines_do_not_intersect_l79_79713


namespace parabola_directrix_distance_l79_79017

theorem parabola_directrix_distance (a : ℝ) : 
  (abs (a / 4 + 1) = 2) → (a = -12 ∨ a = 4) := 
by
  sorry

end parabola_directrix_distance_l79_79017


namespace max_r1_minus_r2_l79_79089

noncomputable def ellipse (x y : ℝ) : Prop :=
  (x^2) / 2 + y^2 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def P (x y : ℝ) : Prop :=
  ellipse x y ∧ x > 0 ∧ y > 0

def r1 (x y : ℝ) (Q2 : ℝ × ℝ) : ℝ := 
  -- Assume a function that calculates the inradius of triangle ΔPF1Q2
  sorry

def r2 (x y : ℝ) (Q1 : ℝ × ℝ) : ℝ :=
  -- Assume a function that calculates the inradius of triangle ΔPF2Q1
  sorry

theorem max_r1_minus_r2 :
  ∃ (x y : ℝ) (Q1 Q2 : ℝ × ℝ), P x y →
    r1 x y Q2 - r2 x y Q1 = 1/3 := 
sorry

end max_r1_minus_r2_l79_79089


namespace number_of_packages_needed_l79_79873

-- Define the problem constants and constraints
def students_per_class := 30
def number_of_classes := 4
def buns_per_student := 2
def buns_per_package := 8

-- Calculate the total number of students
def total_students := number_of_classes * students_per_class

-- Calculate the total number of buns needed
def total_buns := total_students * buns_per_student

-- Calculate the required number of packages
def required_packages := total_buns / buns_per_package

-- Prove that the required number of packages is 30
theorem number_of_packages_needed : required_packages = 30 := by
  -- The proof would be here, but for now we assume it is correct
  sorry

end number_of_packages_needed_l79_79873


namespace calculate_expression_l79_79091

theorem calculate_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) * (3 + 2) = X :=
by
  sorry

end calculate_expression_l79_79091


namespace floor_of_ten_times_expected_value_of_fourth_largest_l79_79938

theorem floor_of_ten_times_expected_value_of_fourth_largest : 
  let n := 90
  let m := 5
  let k := 4
  let E := (k * (n + 1)) / (m + 1)
  ∀ (X : Fin m → Fin n) (h : ∀ i j : Fin m, i ≠ j → X i ≠ X j), 
  Nat.floor (10 * E) = 606 := 
by
  sorry

end floor_of_ten_times_expected_value_of_fourth_largest_l79_79938


namespace root_situation_l79_79186

theorem root_situation (a b : ℝ) : 
  ∃ (m n : ℝ), 
    (x - a) * (x - (a + b)) = 1 → 
    (m < a ∧ a < n) ∨ (n < a ∧ a < m) :=
sorry

end root_situation_l79_79186


namespace solve_inequality_l79_79191

-- Define the domain and inequality conditions
def inequality_condition (x : ℝ) : Prop := (1 / (x - 1)) > 1
def domain_condition (x : ℝ) : Prop := x ≠ 1

-- State the theorem to be proved.
theorem solve_inequality (x : ℝ) : domain_condition x → inequality_condition x → 1 < x ∧ x < 2 :=
by
  intros h_domain h_ineq
  sorry

end solve_inequality_l79_79191


namespace books_left_on_Fri_l79_79810

-- Define the conditions as constants or values
def books_at_beginning : ℕ := 98
def books_checked_out_Wed : ℕ := 43
def books_returned_Thu : ℕ := 23
def books_checked_out_Thu : ℕ := 5
def books_returned_Fri : ℕ := 7

-- The proof statement to verify the final number of books
theorem books_left_on_Fri (b : ℕ) :
  b = (books_at_beginning - books_checked_out_Wed) + books_returned_Thu - books_checked_out_Thu + books_returned_Fri := 
  sorry

end books_left_on_Fri_l79_79810


namespace student_community_arrangements_l79_79254

theorem student_community_arrangements :
  (3 ^ 4) = 81 :=
by
  sorry

end student_community_arrangements_l79_79254


namespace permutations_of_BANANA_l79_79526

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l79_79526


namespace additional_interest_due_to_higher_rate_l79_79064

def principal : ℝ := 2500
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  P * r * t

theorem additional_interest_due_to_higher_rate :
  simple_interest principal rate1 time - simple_interest principal rate2 time = 300 :=
by
  sorry

end additional_interest_due_to_higher_rate_l79_79064


namespace infinite_primes_l79_79179

theorem infinite_primes : ∀ (p : ℕ), Prime p → ¬ (∃ q : ℕ, Prime q ∧ q > p) := sorry

end infinite_primes_l79_79179


namespace ratio_of_rectangle_sides_l79_79757

theorem ratio_of_rectangle_sides (x y : ℝ) (h : x < y) 
  (hs : x + y - Real.sqrt (x^2 + y^2) = (1 / 3) * y) : 
  x / y = 5 / 12 :=
by
  sorry

end ratio_of_rectangle_sides_l79_79757


namespace stephanie_quarters_fraction_l79_79445

/-- Stephanie has a collection containing exactly one of the first 25 U.S. state quarters. 
    The quarters are in the order the states joined the union.
    Suppose 8 states joined the union between 1800 and 1809. -/
theorem stephanie_quarters_fraction :
  (8 / 25 : ℚ) = (8 / 25) :=
by
  sorry

end stephanie_quarters_fraction_l79_79445


namespace solveTheaterProblem_l79_79699

open Nat

def theaterProblem : Prop :=
  ∃ (A C : ℕ), (A + C = 80) ∧ (12 * A + 5 * C = 519) ∧ (C = 63)

theorem solveTheaterProblem : theaterProblem :=
  by
  sorry

end solveTheaterProblem_l79_79699


namespace john_new_total_lifting_capacity_is_correct_l79_79143

def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50

def new_clean_and_jerk : ℕ := 2 * initial_clean_and_jerk
def new_snatch : ℕ := initial_snatch + (initial_snatch * 8 / 10)

def new_combined_total_capacity : ℕ := new_clean_and_jerk + new_snatch

theorem john_new_total_lifting_capacity_is_correct : 
  new_combined_total_capacity = 250 := by
  sorry

end john_new_total_lifting_capacity_is_correct_l79_79143


namespace unique_solution_for_2_3_6_eq_7_l79_79245

theorem unique_solution_for_2_3_6_eq_7 (x : ℝ) : 2^x + 3^x + 6^x = 7^x → x = 2 :=
by
  intro h
  -- Add the relevant proof tactic steps here
  sorry

end unique_solution_for_2_3_6_eq_7_l79_79245


namespace find_m_l79_79745

open Complex

theorem find_m (m : ℝ) : (re ((1 + I) / (1 - I) + m * (1 - I) / (1 + I)) = ((1 + I) / (1 - I) + m * (1 - I) / (1 + I))) → m = 1 :=
by
  sorry

end find_m_l79_79745


namespace mileage_per_gallon_l79_79221

noncomputable def car_mileage (distance: ℝ) (gasoline: ℝ) : ℝ :=
  distance / gasoline

theorem mileage_per_gallon :
  car_mileage 190 4.75 = 40 :=
by
  -- proof omitted
  sorry

end mileage_per_gallon_l79_79221


namespace problem1_problem2_l79_79837

-- Problem 1
theorem problem1 : ((- (1/2) - (1/3) + (3/4)) * -60) = 5 :=
by
  -- The proof steps would go here
  sorry

-- Problem 2
theorem problem2 : ((-1)^4 - (1/6) * (3 - (-3)^2)) = 2 :=
by
  -- The proof steps would go here
  sorry

end problem1_problem2_l79_79837


namespace root_conditions_l79_79390

-- Given conditions and definitions:
def quadratic_eq (m x : ℝ) : ℝ := x^2 + (m - 3) * x + m

-- The proof problem statement
theorem root_conditions (m : ℝ) (h1 : ∃ x y : ℝ, quadratic_eq m x = 0 ∧ quadratic_eq m y = 0 ∧ x > 1 ∧ y < 1) : m < 1 :=
sorry

end root_conditions_l79_79390


namespace total_area_of_field_l79_79072

theorem total_area_of_field (A1 A2 : ℝ) (h1 : A1 = 225)
    (h2 : A2 - A1 = (1 / 5) * ((A1 + A2) / 2)) :
  A1 + A2 = 500 := by
  sorry

end total_area_of_field_l79_79072


namespace projectiles_initial_distance_l79_79193

theorem projectiles_initial_distance (Projectile1_speed Projectile2_speed Time_to_meet : ℕ) 
  (h1 : Projectile1_speed = 444)
  (h2 : Projectile2_speed = 555)
  (h3 : Time_to_meet = 2) : 
  (Projectile1_speed + Projectile2_speed) * Time_to_meet = 1998 := by
  sorry

end projectiles_initial_distance_l79_79193


namespace female_students_count_l79_79653

variable (F : ℕ)

theorem female_students_count
    (avg_all_students : ℕ)
    (avg_male_students : ℕ)
    (avg_female_students : ℕ)
    (num_male_students : ℕ)
    (condition1 : avg_all_students = 90)
    (condition2 : avg_male_students = 82)
    (condition3 : avg_female_students = 92)
    (condition4 : num_male_students = 8)
    (condition5 : 8 * 82 + F * 92 = (8 + F) * 90) : 
    F = 32 := 
by 
  sorry

end female_students_count_l79_79653


namespace remaining_sum_avg_l79_79951

variable (a b : ℕ → ℝ)
variable (h1 : 1 / 6 * (a 1 + a 2 + a 3 + a 4 + a 5 + a 6) = 2.5)
variable (h2 : 1 / 2 * (a 1 + a 2) = 1.1)
variable (h3 : 1 / 2 * (a 3 + a 4) = 1.4)

theorem remaining_sum_avg :
  1 / 2 * (a 5 + a 6) = 5 :=
by
  sorry

end remaining_sum_avg_l79_79951


namespace set_difference_example_l79_79028

-- Define P and Q based on the given conditions
def P : Set ℝ := {x | 0 < x ∧ x < 2}
def Q : Set ℝ := {x | 1 < x ∧ x < 3}

-- State the theorem: P - Q equals to the set {x | 0 < x ≤ 1}
theorem set_difference_example : P \ Q = {x | 0 < x ∧ x ≤ 1} := 
  by
  sorry

end set_difference_example_l79_79028


namespace lagrange_mean_value_problem_l79_79020

noncomputable def g (x : ℝ) : ℝ := (Real.log x) + x

theorem lagrange_mean_value_problem :
  ∃ c : ℝ, c ∈ set.Ioo 1 2 ∧ 
           deriv g c = Real.log 2 + 1 :=
by
  have h1 : continuous_on g (set.Icc 1 2),
  { sorry }
  have h2 : ∀ x ∈ set.Ioo 1 2, differentiable_at ℝ g x,
  { sorry }
  have h3 : ∀ x ∈ set.Ioo 1 2, deriv g x = (1 / x) + 1,
  { sorry }
  use 1 / Real.log 2
  split
  { sorry }
  { sorry }
  sorry


end lagrange_mean_value_problem_l79_79020


namespace optometrist_sales_l79_79833

noncomputable def total_pairs_optometrist_sold (H S : ℕ) (total_sales: ℝ) : Prop :=
  (S = H + 7) ∧ 
  (total_sales = 0.9 * (95 * ↑H + 175 * ↑S)) ∧ 
  (total_sales = 2469)

theorem optometrist_sales :
  ∃ H S : ℕ, total_pairs_optometrist_sold H S 2469 ∧ H + S = 17 :=
by 
  sorry

end optometrist_sales_l79_79833


namespace problem1_problem2_l79_79976

theorem problem1 : 2 * Real.sqrt 3 - Real.sqrt 12 + Real.sqrt 27 = 3 * Real.sqrt 3 := by
  sorry

theorem problem2 : (Real.sqrt 18 - Real.sqrt 3) * Real.sqrt 12 = 6 * Real.sqrt 6 - 6 := by
  sorry

end problem1_problem2_l79_79976


namespace cos_value_l79_79724

theorem cos_value (α : ℝ) (h : Real.sin (α + π / 6) = 1 / 3) : Real.cos (π / 3 - α) = 1 / 3 :=
  sorry

end cos_value_l79_79724


namespace cistern_water_depth_l79_79959

theorem cistern_water_depth:
  ∀ h: ℝ,
  (4 * 4 + 4 * h * 4 + 4 * h * 4 = 36) → h = 1.25 := by
    sorry

end cistern_water_depth_l79_79959


namespace no_solution_exists_l79_79034

theorem no_solution_exists :
  ¬ ∃ (x1 x2 x3 x4 : ℝ), 
    (x1 + x2 = 1) ∧
    (x2 + x3 - x4 = 1) ∧
    (0 ≤ x1) ∧
    (0 ≤ x2) ∧
    (0 ≤ x3) ∧
    (0 ≤ x4) ∧
    ∀ (F : ℝ), F = x1 - x2 + 2 * x3 - x4 → 
    ∀ (b : ℝ), F ≤ b :=
by sorry

end no_solution_exists_l79_79034


namespace minimize_circumscribed_sphere_radius_l79_79639

noncomputable def cylinder_surface_area (r h : ℝ) : ℝ :=
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

noncomputable def circumscribed_sphere_radius (r h : ℝ) : ℝ :=
  (r^2 + (1 / 2 * h)^2).sqrt

theorem minimize_circumscribed_sphere_radius (r : ℝ) (h : ℝ) (hr : cylinder_surface_area r h = 16 * Real.pi) : 
  r^2 = 8 * Real.sqrt 5 / 5 :=
sorry

end minimize_circumscribed_sphere_radius_l79_79639


namespace probability_single_shot_l79_79793

-- Define the event and probability given
def event_A := "shooter hits the target at least once out of three shots"
def probability_event_A : ℝ := 0.875

-- The probability of missing in one shot is q, and missing all three is q^3, 
-- which leads to hitting at least once being 1 - q^3
theorem probability_single_shot (q : ℝ) (h : 1 - q^3 = 0.875) : 1 - q = 0.5 :=
by
  sorry

end probability_single_shot_l79_79793


namespace geometric_progression_arcsin_sin_l79_79511

noncomputable def least_positive_t : ℝ :=
  9 + 4 * Real.sqrt 5

theorem geometric_progression_arcsin_sin 
  (α : ℝ) 
  (hα1: 0 < α) 
  (hα2: α < Real.pi / 2) 
  (t : ℝ) 
  (h : ∀ (a b c d : ℝ), 
    a = Real.arcsin (Real.sin α) ∧ 
    b = Real.arcsin (Real.sin (3 * α)) ∧ 
    c = Real.arcsin (Real.sin (5 * α)) ∧ 
    d = Real.arcsin (Real.sin (t * α)) → 
    b / a = c / b ∧ c / b = d / c) : 
  t = least_positive_t :=
sorry

end geometric_progression_arcsin_sin_l79_79511


namespace arrangement_count_l79_79825

-- Definitions
def volunteers := 4
def elderly := 2
def total_people := volunteers + elderly
def criteria := "The 2 elderly people must be adjacent but not at the ends of the row."

-- Theorem: The number of different valid arrangements is 144
theorem arrangement_count : 
  ∃ (arrangements : Nat), arrangements = (volunteers.factorial * 3 * elderly.factorial) ∧ arrangements = 144 := 
  by 
    sorry

end arrangement_count_l79_79825


namespace total_hike_time_l79_79630

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l79_79630


namespace train_length_l79_79968

theorem train_length (speed_km_hr : ℝ) (time_seconds : ℝ) (speed_conversion_factor : ℝ) (approx_length : ℝ) :
  speed_km_hr = 60 → time_seconds = 6 → speed_conversion_factor = (1000 / 3600) → approx_length = 100.02 →
  speed_km_hr * speed_conversion_factor * time_seconds = approx_length :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3]
  sorry

end train_length_l79_79968


namespace amy_initial_money_l79_79676

-- Define the conditions
variable (left_fair : ℕ) (spent : ℕ)

-- Define the proof problem statement
theorem amy_initial_money (h1 : left_fair = 11) (h2 : spent = 4) : left_fair + spent = 15 := 
by sorry

end amy_initial_money_l79_79676


namespace percentage_difference_l79_79174

theorem percentage_difference (G P R : ℝ) (h1 : P = 0.9 * G) (h2 : R = 1.125 * G) :
  ((1 - P / R) * 100) = 20 :=
by
  sorry

end percentage_difference_l79_79174


namespace linear_function_quadrants_l79_79313

theorem linear_function_quadrants (k : ℝ) :
  (k - 3 > 0) ∧ (-k + 2 < 0) → k > 3 :=
by
  intro h
  sorry

end linear_function_quadrants_l79_79313


namespace quadratic_inequality_l79_79378

noncomputable def quadratic_inequality_solution : Set ℝ :=
  {x | x < 2} ∪ {x | x > 4}

theorem quadratic_inequality (x : ℝ) : (x^2 - 6 * x + 8 > 0) ↔ (x ∈ quadratic_inequality_solution) :=
by
  sorry

end quadratic_inequality_l79_79378


namespace average_of_remaining_two_numbers_l79_79952

theorem average_of_remaining_two_numbers 
  (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 4.60)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.8) :
  ((e + f) / 2) = 6.6 :=
sorry

end average_of_remaining_two_numbers_l79_79952


namespace sqrt_sixteen_l79_79652

theorem sqrt_sixteen : ∃ x : ℝ, x^2 = 16 ∧ x = 4 :=
by
  sorry

end sqrt_sixteen_l79_79652


namespace solution_l79_79251

noncomputable def g (a : ℝ) : Polynomials ℝ := X^3 + a * X^2 + 2 * X + 15
noncomputable def f (a b c : ℝ) : Polynomials ℝ := X^4 + 2 * X^3 + b * X^2 + 150 * X + c

theorem solution (a b c : ℝ) (h1 : (g a).roots.nodup) (h2 : ∀ r ∈ (g a).roots, f a b c).is_root r) :
  (f a b c).eval 1 = -15640 :=
by 
  sorry

end solution_l79_79251


namespace number_of_rectangles_in_5x5_grid_l79_79001

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l79_79001


namespace problem_xyz_l79_79270

theorem problem_xyz (x y : ℝ) (h1 : (x + y)^2 = 16) (h2 : x * y = -8) :
  x^2 + y^2 = 32 :=
by
  sorry

end problem_xyz_l79_79270


namespace find_a_b_l79_79766

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem find_a_b : 
  (∀ x : ℝ, f (g x a b) = 9 * x^2 + 6 * x + 1) ↔ ((a = 3 ∧ b = 1) ∨ (a = -3 ∧ b = -1)) :=
by
  sorry

end find_a_b_l79_79766


namespace cherry_sodas_l79_79688

theorem cherry_sodas (C O : ℕ) (h1 : O = 2 * C) (h2 : C + O = 24) : C = 8 :=
by sorry

end cherry_sodas_l79_79688


namespace profit_increase_l79_79053

theorem profit_increase (x y : ℝ) (a : ℝ)
  (h1 : x = (57 / 20) * y)
  (h2 : (x - y) / y = a / 100)
  (h3 : (x - 0.95 * y) / (0.95 * y) = (a + 15) / 100) :
  a = 185 := sorry

end profit_increase_l79_79053


namespace cistern_fill_time_l79_79677

-- Define the filling rate and emptying rate as given conditions.
def R_fill : ℚ := 1 / 5
def R_empty : ℚ := 1 / 9

-- Define the net rate when both taps are opened simultaneously.
def R_net : ℚ := R_fill - R_empty

-- The total time to fill the cistern when both taps are opened.
def fill_time := 1 / R_net

-- Prove that the total time to fill the cistern is 11.25 hours.
theorem cistern_fill_time : fill_time = 11.25 := 
by 
    -- We include sorry to bypass the actual proof. This will allow the code to compile.
    sorry

end cistern_fill_time_l79_79677


namespace f_satisfies_condition_l79_79397

noncomputable def f (x : ℝ) : ℝ := 2^x

-- Prove that f(x + 1) = 2 * f(x) for the defined function f.
theorem f_satisfies_condition (x : ℝ) : f (x + 1) = 2 * f x := by
  show 2^(x + 1) = 2 * 2^x
  sorry

end f_satisfies_condition_l79_79397


namespace find_q_l79_79429

open Polynomial

-- Define the conditions for the roots of the first polynomial
def roots_of_first_eq (a b m : ℝ) (h : a * b = 3) : Prop := 
  ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)

-- Define the problem statement
theorem find_q (a b m p q : ℝ) 
  (h1 : a * b = 3) 
  (h2 : ∀ x, (x^2 - m*x + 3) = (x - a) * (x - b)) 
  (h3 : ∀ x, (x^2 - p*x + q) = (x - (a + 2/b)) * (x - (b + 2/a))) :
  q = 25 / 3 :=
sorry

end find_q_l79_79429


namespace total_tourists_proof_l79_79789

noncomputable def calculate_total_tourists : ℕ :=
  let start_time := 8  
  let end_time := 17   -- 5 PM in 24-hour format
  let initial_tourists := 120
  let increment := 2
  let number_of_trips := end_time - start_time  -- total number of trips including both start and end
  let first_term := initial_tourists
  let last_term := initial_tourists + increment * (number_of_trips - 1)
  (number_of_trips * (first_term + last_term)) / 2

theorem total_tourists_proof : calculate_total_tourists = 1290 := by
  sorry

end total_tourists_proof_l79_79789


namespace Dawn_sold_glasses_l79_79502

variable (x : ℕ)

def Bea_price_per_glass : ℝ := 0.25
def Dawn_price_per_glass : ℝ := 0.28
def Bea_glasses_sold : ℕ := 10
def Bea_extra_earnings : ℝ := 0.26
def Bea_total_earnings : ℝ := Bea_glasses_sold * Bea_price_per_glass
def Dawn_total_earnings (x : ℕ) : ℝ := x * Dawn_price_per_glass

theorem Dawn_sold_glasses :
  Bea_total_earnings - Bea_extra_earnings = Dawn_total_earnings x → x = 8 :=
by
  sorry

end Dawn_sold_glasses_l79_79502


namespace binom_12_11_l79_79508

theorem binom_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binom_12_11_l79_79508


namespace households_with_at_least_one_appliance_l79_79077

theorem households_with_at_least_one_appliance (total: ℕ) (color_tvs: ℕ) (refrigerators: ℕ) (both: ℕ) :
  total = 100 → color_tvs = 65 → refrigerators = 84 → both = 53 →
  (color_tvs + refrigerators - both) = 96 :=
by
  intros
  sorry

end households_with_at_least_one_appliance_l79_79077


namespace binom_1300_2_eq_844350_l79_79364

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end binom_1300_2_eq_844350_l79_79364


namespace smallest_lcm_value_l79_79273

def is_five_digit (x : ℕ) : Prop :=
  10000 ≤ x ∧ x < 100000

theorem smallest_lcm_value :
  ∃ (m n : ℕ), is_five_digit m ∧ is_five_digit n ∧ Nat.gcd m n = 5 ∧ Nat.lcm m n = 20030010 :=
by
  sorry

end smallest_lcm_value_l79_79273


namespace condition_sufficient_not_necessary_monotonicity_l79_79384

theorem condition_sufficient_not_necessary_monotonicity
  (f : ℝ → ℝ) (a : ℝ) (h_def : ∀ x, f x = 2^(abs (x - a))) :
  (∀ x > 1, x - a ≥ 0) → (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y) ∧
  (∃ a, a ≤ 1 ∧ (∀ x > 1, x - a ≥ 0) ∧ (∀ x y, (x > 1) ∧ (y > 1) ∧ (x ≤ y) → f x ≤ f y)) :=
by
  sorry

end condition_sufficient_not_necessary_monotonicity_l79_79384


namespace geometric_sequence_value_l79_79023

noncomputable def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

theorem geometric_sequence_value
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geo : geometric_sequence a r)
  (h_pos : ∀ n, a n > 0)
  (h_roots : ∀ (a1 a19 : ℝ), a1 = a 1 → a19 = a 19 → a1 * a19 = 16 ∧ a1 + a19 = 10) :
  a 8 * a 10 * a 12 = 64 := 
sorry

end geometric_sequence_value_l79_79023


namespace jordan_meets_emily_after_total_time_l79_79241

noncomputable def meet_time
  (initial_distance : ℝ)
  (speed_ratio : ℝ)
  (decrease_rate : ℝ)
  (time_until_break : ℝ)
  (break_duration : ℝ)
  (total_meet_time : ℝ) : Prop :=
  initial_distance = 30 ∧
  speed_ratio = 2 ∧
  decrease_rate = 2 ∧
  time_until_break = 10 ∧
  break_duration = 5 ∧
  total_meet_time = 17

theorem jordan_meets_emily_after_total_time :
  meet_time 30 2 2 10 5 17 := 
by {
  -- The conditions directly state the requirements needed for the proof.
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ -- This line confirms that all inputs match the given conditions.
}

end jordan_meets_emily_after_total_time_l79_79241


namespace multiple_of_6_and_factor_of_72_l79_79347

open Nat

theorem multiple_of_6_and_factor_of_72 (n : ℕ) :
  (∃ k₁ : ℕ, n = 6 * k₁) ∧ (∃ k₂ : ℕ, 72 = n * k₂) ↔ n ∈ {6, 12, 18, 24, 36, 72} :=
by
  sorry

end multiple_of_6_and_factor_of_72_l79_79347


namespace truck_speed_on_dirt_road_l79_79830

theorem truck_speed_on_dirt_road 
  (total_distance: ℝ) (time_on_dirt: ℝ) (time_on_paved: ℝ) (speed_difference: ℝ)
  (h1: total_distance = 200) (h2: time_on_dirt = 3) (h3: time_on_paved = 2) (h4: speed_difference = 20) : 
  ∃ v: ℝ, (time_on_dirt * v + time_on_paved * (v + speed_difference) = total_distance) ∧ v = 32 := 
sorry

end truck_speed_on_dirt_road_l79_79830


namespace john_new_total_lifting_capacity_is_correct_l79_79144

def initial_clean_and_jerk : ℕ := 80
def initial_snatch : ℕ := 50

def new_clean_and_jerk : ℕ := 2 * initial_clean_and_jerk
def new_snatch : ℕ := initial_snatch + (initial_snatch * 8 / 10)

def new_combined_total_capacity : ℕ := new_clean_and_jerk + new_snatch

theorem john_new_total_lifting_capacity_is_correct : 
  new_combined_total_capacity = 250 := by
  sorry

end john_new_total_lifting_capacity_is_correct_l79_79144


namespace find_m_probability_l79_79040

theorem find_m_probability (m : ℝ) (ξ : ℕ → ℝ) :
  (ξ 1 = m * (2/3)) ∧ (ξ 2 = m * (2/3)^2) ∧ (ξ 3 = m * (2/3)^3) ∧ 
  (ξ 1 + ξ 2 + ξ 3 = 1) → 
  m = 27 / 38 := 
sorry

end find_m_probability_l79_79040


namespace set_operation_result_l79_79432

def M : Set ℕ := {2, 3}

def bin_op (A : Set ℕ) : Set ℕ :=
  {x | ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem set_operation_result : bin_op M = {4, 5, 6} :=
by
  sorry

end set_operation_result_l79_79432


namespace solve_x_l79_79899

theorem solve_x (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 :=
by
  sorry

end solve_x_l79_79899


namespace complex_square_sum_eq_five_l79_79868

theorem complex_square_sum_eq_five (a b : ℝ) (h : (a + b * I) ^ 2 = 3 + 4 * I) : a^2 + b^2 = 5 := 
by sorry

end complex_square_sum_eq_five_l79_79868


namespace race_prob_l79_79950

theorem race_prob :
  let pX := (1 : ℝ) / 8
  let pY := (1 : ℝ) / 12
  let pZ := (1 : ℝ) / 6
  pX + pY + pZ = (3 : ℝ) / 8 :=
by
  sorry

end race_prob_l79_79950


namespace find_4digit_number_l79_79575

theorem find_4digit_number (a b c d n n' : ℕ) :
  n = 1000 * a + 100 * b + 10 * c + d →
  n' = 1000 * d + 100 * c + 10 * b + a →
  n = n' - 7182 →
  n = 1909 :=
by
  intros h1 h2 h3
  sorry

end find_4digit_number_l79_79575


namespace men_l79_79956

-- Given conditions
variable (W M : ℕ)
variable (B : ℕ) [DecidableEq ℕ] -- number of boys
variable (total_earnings : ℕ)

def earnings : ℕ := 5 * M + W * M + 8 * W

-- Total earnings of men, women, and boys is Rs. 150.
def conditions : Prop := 
  5 * M = W * M ∧ 
  W * M = 8 * W ∧ 
  earnings = total_earnings

-- Prove men's wages (total wages for 5 men) is Rs. 50.
theorem men's_wages (hm : total_earnings = 150) (hb : W = 8) : 
  5 * M = 50 :=
by
  sorry

end men_l79_79956


namespace arithmetic_sequence_sum_l79_79614

/-
In an arithmetic sequence, if the sum of terms \( a_2 + a_3 + a_4 + a_5 + a_6 = 90 \), 
prove that \( a_1 + a_7 = 36 \).
-/

theorem arithmetic_sequence_sum (a : ℕ → ℤ) (d : ℤ)
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_sum : a 2 + a 3 + a 4 + a 5 + a 6 = 90) :
  a 1 + a 7 = 36 := by
  sorry

end arithmetic_sequence_sum_l79_79614


namespace hyperbola_eccentricity_l79_79014

theorem hyperbola_eccentricity (a b c : ℝ) (h : (c^2 - a^2 = 5 * a^2)) (hb : a / b = 2) :
  (c / a = Real.sqrt 5) :=
by
  sorry

end hyperbola_eccentricity_l79_79014


namespace problem_l79_79608

theorem problem (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_l79_79608


namespace tickets_difference_l79_79906

-- Define the conditions
def tickets_friday : ℕ := 181
def tickets_sunday : ℕ := 78
def tickets_saturday : ℕ := 2 * tickets_friday

-- The theorem to prove
theorem tickets_difference : tickets_saturday - tickets_sunday = 284 := by
  sorry

end tickets_difference_l79_79906


namespace average_age_of_team_l79_79307

theorem average_age_of_team :
  ∃ A : ℝ,
    (∀ (ages : Fin 11 → ℝ),
      (ages ⟨0, by norm_num⟩ = 26) ∧
      (ages ⟨1, by norm_num⟩ = 29) ∧
      (∀ i (hi : 2 ≤ i ∧ i < 11), ages i = 32)  ∧ 
      (11 * A = sum (range 11) (λ i, ages ⟨i, by norm_num⟩)) ∧
      (9 * (A - 1) = sum (range 2 11) (λ i, ages ⟨i, by norm_num⟩))
    ) →
  A = 32 :=
sorry

end average_age_of_team_l79_79307


namespace total_attendance_l79_79775

theorem total_attendance (first_concert : ℕ) (second_concert : ℕ) (third_concert : ℕ) :
  first_concert = 65899 →
  second_concert = first_concert + 119 →
  third_concert = 2 * second_concert →
  first_concert + second_concert + third_concert = 263953 :=
by
  intros h_first h_second h_third
  rw [h_first, h_second, h_third]
  sorry

end total_attendance_l79_79775


namespace rectangle_area_same_width_l79_79227

theorem rectangle_area_same_width
  (square_area : ℝ) (area_eq : square_area = 36)
  (rect_width_eq_side : ℝ → ℝ → Prop) (width_eq : ∀ s, rect_width_eq_side s s)
  (rect_length_eq_3_times_width : ℝ → ℝ → Prop) (length_eq : ∀ w, rect_length_eq_3_times_width w (3 * w)) :
  (∃ s l w, s = 6 ∧ w = s ∧ l = 3 * w ∧ square_area = s * s ∧ rect_width_eq_side w s ∧ rect_length_eq_3_times_width w l ∧ w * l = 108) :=
by {
  sorry
}

end rectangle_area_same_width_l79_79227


namespace lin_reg_proof_l79_79352

variable (x y : List ℝ)
variable (n : ℝ := 10)
variable (sum_x : ℝ := 80)
variable (sum_y : ℝ := 20)
variable (sum_xy : ℝ := 184)
variable (sum_x2 : ℝ := 720)

noncomputable def mean (lst: List ℝ) (n: ℝ) : ℝ := (List.sum lst) / n

noncomputable def lin_reg_slope (n sum_x sum_y sum_xy sum_x2 : ℝ) : ℝ :=
  (sum_xy - n * (sum_x / n) * (sum_y / n)) / (sum_x2 - n * (sum_x / n) ^ 2)

noncomputable def lin_reg_intercept (sum_x sum_y : ℝ) (slope : ℝ) (n : ℝ) : ℝ :=
  (sum_y / n) - slope * (sum_x / n)

theorem lin_reg_proof :
  lin_reg_slope n sum_x sum_y sum_xy sum_x2 = 0.3 ∧ 
  lin_reg_intercept sum_x sum_y 0.3 n = -0.4 ∧ 
  (0.3 * 7 - 0.4 = 1.7) :=
by
  sorry

end lin_reg_proof_l79_79352


namespace infinite_values_prime_divisor_l79_79297

noncomputable def largestPrimeDivisor (n : ℕ) : ℕ :=
  sorry

theorem infinite_values_prime_divisor :
  ∃ᶠ n in at_top, largestPrimeDivisor (n^2 + n + 1) = largestPrimeDivisor ((n+1)^2 + (n+1) + 1) :=
sorry

end infinite_values_prime_divisor_l79_79297


namespace union_of_A_B_l79_79585

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l79_79585


namespace quadratic_to_vertex_properties_of_quadratic_quadratic_decreasing_interval_quadratic_range_in_interval_l79_79735

-- Define the quadratic function.
def quadratic_function (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Define the vertex form of the quadratic function.
def vertex_form (x : ℝ) : ℝ := (x - 2)^2 - 1

-- Prove that the quadratic function is equal to its vertex form.
theorem quadratic_to_vertex :
  ∀ x : ℝ, quadratic_function(x) = vertex_form(x) :=
by
  sorry

-- Define the axis of symmetry.
def axis_of_symmetry : ℝ := 2

-- Define the vertex coordinates.
def vertex : ℝ × ℝ := (2, -1)

-- Define the minimum value of the quadratic function.
def minimum_value : ℝ := -1

-- Define the interval where the quadratic function decreases.
def decreasing_interval : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

-- Define the range of the quadratic function in the given interval.
def range_in_interval : Set ℝ := {y : ℝ | -1 ≤ y ∧ y ≤ 8}

-- Prove that the axis of symmetry, vertex coordinates, and minimum value are correct.
theorem properties_of_quadratic :
  axios_of_symmetry = 2 ∧ vertex = (2, -1) ∧ minimum_value = -1 :=
by
  sorry

-- Prove the interval where the function decreases.
theorem quadratic_decreasing_interval :
  ∀ x : ℝ, -1 ≤ x ∧ x < 2 → ∃ y : ℝ, quadratic_function(x) = y :=
by
  sorry

-- Prove the range of the function in the given interval.
theorem quadratic_range_in_interval :
  ∀ y : ℝ, -1 ≤ y ∧ y ≤ 8 → ∃ x : ℝ, -1 ≤ x ∧ x < 3 ∧ quadratic_function(x) = y :=
by
  sorry

end quadratic_to_vertex_properties_of_quadratic_quadratic_decreasing_interval_quadratic_range_in_interval_l79_79735


namespace range_of_m_l79_79387

-- Define the conditions:

/-- Proposition p: the equation represents an ellipse with foci on y-axis -/
def proposition_p (m : ℝ) : Prop :=
  0 < m ∧ m < 9 ∧ 9 - m > 2 * m ∧ 2 * m > 0

/-- Proposition q: the eccentricity of the hyperbola is in the interval (\sqrt(3)/2, \sqrt(2)) -/
def proposition_q (m : ℝ) : Prop :=
  0 < m ∧ (5 / 2 < m ∧ m < 5)

def p_or_q (m : ℝ) : Prop := proposition_p m ∨ proposition_q m
def p_and_q (m : ℝ) : Prop := proposition_p m ∧ proposition_q m

-- Mathematically equivalent proof problem in Lean 4:

theorem range_of_m (m : ℝ) : (p_or_q m ∧ ¬p_and_q m) ↔ (m ∈ Set.Ioc 0 (5 / 2) ∪ Set.Icc 3 5) := sorry

end range_of_m_l79_79387


namespace sum_of_remainders_eq_3_l79_79945

theorem sum_of_remainders_eq_3 (a b c : ℕ) (h1 : a % 59 = 28) (h2 : b % 59 = 15) (h3 : c % 59 = 19) (h4 : a = b + d ∨ b = c + d ∨ c = a + d) : 
  (a + b + c) % 59 = 3 :=
by {
  sorry -- Proof to be constructed
}

end sum_of_remainders_eq_3_l79_79945


namespace unique_solution_l79_79244

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^x + 3^x + 6^x = 7^x

theorem unique_solution : ∀ x : ℝ, equation_satisfied x ↔ x = 2 := by
  sorry

end unique_solution_l79_79244


namespace arrangement_of_BANANA_l79_79547

theorem arrangement_of_BANANA : 
  (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 :=
by
  sorry

end arrangement_of_BANANA_l79_79547


namespace gcd_of_44_54_74_l79_79848

theorem gcd_of_44_54_74 : gcd (gcd 44 54) 74 = 2 :=
by
    sorry

end gcd_of_44_54_74_l79_79848


namespace david_initial_money_l79_79843

-- Given conditions as definitions
def spent (S : ℝ) : Prop := S - 800 = 500
def has_left (H : ℝ) : Prop := H = 500

-- The main theorem to prove
theorem david_initial_money (S : ℝ) (X : ℝ) (H : ℝ)
  (h1 : spent S) 
  (h2 : has_left H) 
  : X = S + H → X = 1800 :=
by
  sorry

end david_initial_money_l79_79843


namespace find_digits_of_six_two_digit_sum_equals_528_l79_79947

theorem find_digits_of_six_two_digit_sum_equals_528
  (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_digits : a < 10 ∧ b < 10 ∧ c < 10)
  (h_sum_six_numbers : (10 * a + b) + (10 * a + c) + (10 * b + c) + (10 * b + a) + (10 * c + a) + (10 * c + b) = 528) :
  (a = 7 ∧ b = 8 ∧ c = 9) := 
sorry

end find_digits_of_six_two_digit_sum_equals_528_l79_79947


namespace frac_y_over_x_plus_y_eq_one_third_l79_79396

theorem frac_y_over_x_plus_y_eq_one_third (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end frac_y_over_x_plus_y_eq_one_third_l79_79396


namespace minimum_value_inequality_l79_79901

variable {x y z : ℝ}

theorem minimum_value_inequality (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 4) :
  (1 / x + 4 / y + 9 / z) ≥ 9 :=
sorry

end minimum_value_inequality_l79_79901


namespace initial_men_count_l79_79171

variable (M : ℕ)

theorem initial_men_count
  (work_completion_time : ℕ)
  (men_leaving : ℕ)
  (remaining_work_time : ℕ)
  (completion_days : ℕ) :
  work_completion_time = 40 →
  men_leaving = 20 →
  remaining_work_time = 40 →
  completion_days = 10 →
  M = 80 :=
by
  intros h1 h2 h3 h4
  sorry

end initial_men_count_l79_79171


namespace find_radii_of_circles_l79_79177

theorem find_radii_of_circles (d : ℝ) (ext_tangent : ℝ) (int_tangent : ℝ)
  (hd : d = 65) (hext : ext_tangent = 63) (hint : int_tangent = 25) :
  ∃ (R r : ℝ), R = 38 ∧ r = 22 :=
by 
  sorry

end find_radii_of_circles_l79_79177


namespace count_pairs_A_B_l79_79190

open Finset

theorem count_pairs_A_B (A B : Finset ℕ) : 
  A ∪ B = {0, 1, 2} ∧ A ≠ B → 
  (({0, 1, 2}.powerset.filter (λ A, A ≠ B)).card = 27) := by
  sorry

end count_pairs_A_B_l79_79190


namespace suzy_final_books_l79_79806

def initial_books : ℕ := 98
def wednesday_checkouts : ℕ := 43
def thursday_returns : ℕ := 23
def thursday_checkouts : ℕ := 5
def friday_returns : ℕ := 7

theorem suzy_final_books :
  initial_books - wednesday_checkouts
  + thursday_returns - thursday_checkouts
  + friday_returns = 80 := by
sorry

end suzy_final_books_l79_79806


namespace simplify_expression_l79_79849

open Real

theorem simplify_expression (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  (( (x + 2) ^ 2 * (x ^ 2 - 2 * x + 2) ^ 2 / (x ^ 3 + 8) ^ 2 ) ^ 2 *
   ( (x - 2) ^ 2 * (x ^ 2 + 2 * x + 2) ^ 2 / (x ^ 3 - 8) ^ 2 ) ^ 2 = 1) :=
by
  sorry

end simplify_expression_l79_79849


namespace system_of_equations_property_l79_79733

theorem system_of_equations_property (a x y : ℝ)
  (h1 : x + y = 1 - a)
  (h2 : x - y = 3 * a + 5)
  (h3 : 0 < x)
  (h4 : 0 ≤ y) :
  (a = -5 / 3 → x = y) ∧ 
  (a = -2 → x + y = 5 + a) ∧ 
  (0 < x ∧ x ≤ 1 → 2 ≤ y ∧ y < 4) := 
by
  sorry

end system_of_equations_property_l79_79733


namespace count_integers_satisfying_sqrt_condition_l79_79464

theorem count_integers_satisfying_sqrt_condition :
  ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℕ), (3 < Real.sqrt x ∧ Real.sqrt x < 5) → (9 < x ∧ x < 25) :=
by
  sorry

end count_integers_satisfying_sqrt_condition_l79_79464


namespace g_42_value_l79_79029

noncomputable def g : ℕ → ℕ := sorry

axiom g_increasing (n : ℕ) (hn : n > 0) : g (n + 1) > g n
axiom g_multiplicative (m n : ℕ) (hm : m > 0) (hn : n > 0) : g (m * n) = g m * g n
axiom g_property_iii (m n : ℕ) (hm : m > 0) (hn : n > 0) : (m ≠ n ∧ m^n = n^m) → (g m = n ∨ g n = m)

theorem g_42_value : g 42 = 4410 :=
by
  sorry

end g_42_value_l79_79029


namespace chocolate_bar_pieces_l79_79957

theorem chocolate_bar_pieces (X : ℕ) (h1 : X / 2 + X / 4 + 15 = X) : X = 60 :=
by
  sorry

end chocolate_bar_pieces_l79_79957


namespace train_length_l79_79353

noncomputable def length_of_train (speed_kmph : ℝ) (time_sec : ℝ) (length_platform_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmph * 1000) / 3600
  let distance_covered := speed_ms * time_sec
  distance_covered - length_platform_m

theorem train_length :
  length_of_train 72 25 340.04 = 159.96 := by
  sorry

end train_length_l79_79353


namespace arithmetic_progression_12th_term_l79_79949

theorem arithmetic_progression_12th_term (a d n : ℤ) (h_a : a = 2) (h_d : d = 8) (h_n : n = 12) :
  a + (n - 1) * d = 90 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end arithmetic_progression_12th_term_l79_79949


namespace solution_of_two_quadratics_l79_79898

theorem solution_of_two_quadratics (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 := 
by 
  sorry

end solution_of_two_quadratics_l79_79898


namespace max_a_b_l79_79879

theorem max_a_b (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_eq : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 := sorry

end max_a_b_l79_79879


namespace mabel_petals_remaining_l79_79161

/-- Mabel has 5 daisies, each with 8 petals. If she gives 2 daisies to her teacher,
how many petals does she have on the remaining daisies in her garden? -/
theorem mabel_petals_remaining :
  (5 - 2) * 8 = 24 :=
by
  sorry

end mabel_petals_remaining_l79_79161


namespace votes_cast_l79_79335

-- Define the conditions as given in the problem.
def total_votes (V : ℕ) := 35 * V / 100 + (35 * V / 100 + 2400) = V

-- The goal is to prove that the number of total votes V equals 8000.
theorem votes_cast : ∃ V : ℕ, total_votes V ∧ V = 8000 :=
by
  sorry -- The proof is not required, only the statement.

end votes_cast_l79_79335


namespace mean_of_smallest_and_largest_is_12_l79_79181

-- Definition of the condition: the mean of five consecutive even numbers is 12.
def mean_of_five_consecutive_even_numbers_is_12 (n : ℤ) : Prop :=
  ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 12

-- Theorem stating that the mean of the smallest and largest of these numbers is 12.
theorem mean_of_smallest_and_largest_is_12 (n : ℤ) 
  (h : mean_of_five_consecutive_even_numbers_is_12 n) : 
  (8 + (16 : ℤ)) / (2 : ℤ) = 12 := 
by
  sorry

end mean_of_smallest_and_largest_is_12_l79_79181


namespace jamal_bought_4_half_dozens_l79_79421

/-- Given that each crayon costs $2, the total cost is $48, and a half dozen is 6 crayons,
    prove that Jamal bought 4 half dozens of crayons. -/
theorem jamal_bought_4_half_dozens (cost_per_crayon : ℕ) (total_cost : ℕ) (half_dozen : ℕ) 
  (h1 : cost_per_crayon = 2) (h2 : total_cost = 48) (h3 : half_dozen = 6) : 
  (total_cost / cost_per_crayon) / half_dozen = 4 := 
by 
  sorry

end jamal_bought_4_half_dozens_l79_79421


namespace quadrilateral_property_indeterminate_l79_79141

variable {α : Type*}
variable (Q A : α → Prop)

theorem quadrilateral_property_indeterminate :
  (¬ ∀ x, Q x → A x) → ¬ ((∃ x, Q x ∧ A x) ↔ False) :=
by
  intro h
  sorry

end quadrilateral_property_indeterminate_l79_79141


namespace sum_of_squares_diagonals_cyclic_quadrilateral_l79_79443

theorem sum_of_squares_diagonals_cyclic_quadrilateral 
(a b c d : ℝ) (α : ℝ) 
(hc : c^2 = a^2 + b^2 + 2 * a * b * Real.cos α)
(hd : d^2 = a^2 + b^2 - 2 * a * b * Real.cos α) :
  c^2 + d^2 = 2 * a^2 + 2 * b^2 :=
by
  sorry

end sum_of_squares_diagonals_cyclic_quadrilateral_l79_79443


namespace quadratic_roots_relation_l79_79661

theorem quadratic_roots_relation (a b c d : ℝ) (h : ∀ x : ℝ, (c * x^2 + d * x + a = 0) → 
  (a * (2007 * x)^2 + b * (2007 * x) + c = 0)) : b^2 = d^2 := 
sorry

end quadratic_roots_relation_l79_79661


namespace number_of_extreme_points_l79_79047

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + 3 * x^2 + 4 * x - a

theorem number_of_extreme_points (a : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 6 * x + 4) > 0) →
  0 = 0 :=
by
  intro h
  sorry

end number_of_extreme_points_l79_79047


namespace guilt_proof_l79_79786

theorem guilt_proof (X Y : Prop) (h1 : X ∨ Y) (h2 : ¬X) : Y :=
by
  sorry

end guilt_proof_l79_79786


namespace ben_paints_150_square_feet_l79_79496

-- Define the given conditions
def ratio_allen_ben : ℕ := 3
def ratio_ben_allen : ℕ := 5
def total_work : ℕ := 240

-- Define the total amount of parts
def total_parts : ℕ := ratio_allen_ben + ratio_ben_allen

-- Define the work per part
def work_per_part : ℕ := total_work / total_parts

-- Define the work done by Ben
def ben_parts : ℕ := ratio_ben_allen
def ben_work : ℕ := work_per_part * ben_parts

-- The statement to be proved
theorem ben_paints_150_square_feet : ben_work = 150 :=
by
  sorry

end ben_paints_150_square_feet_l79_79496


namespace investment_C_l79_79080

-- Definitions of the given conditions
def investment_A : ℝ := 6300
def investment_B : ℝ := 4200
def total_profit : ℝ := 12700
def profit_A : ℝ := 3810

-- Defining the total investment, including C's investment
noncomputable def investment_total_including_C (C : ℝ) : ℝ := investment_A + investment_B + C

-- Proving the correct investment for C under the given conditions
theorem investment_C (C : ℝ) :
  (investment_A / investment_total_including_C C) = (profit_A / total_profit) → 
  C = 10500 :=
by
  -- Placeholder for the actual proof
  sorry

end investment_C_l79_79080


namespace max_value_of_quadratic_l79_79944

theorem max_value_of_quadratic :
  ∃ x_max : ℝ, x_max = 1.5 ∧
  ∀ x : ℝ, -3 * x^2 + 9 * x + 24 ≤ -3 * (1.5)^2 + 9 * 1.5 + 24 := by
  sorry

end max_value_of_quadratic_l79_79944


namespace find_f_neg_two_l79_79869

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then x^2 - 1 else sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

variable (f : ℝ → ℝ)

axiom f_odd : is_odd_function f
axiom f_pos : ∀ x, x > 0 → f x = x^2 - 1

theorem find_f_neg_two : f (-2) = -3 :=
by
  sorry

end find_f_neg_two_l79_79869


namespace find_y_l79_79041

theorem find_y (n x y : ℕ) 
    (h1 : (n + 200 + 300 + x) / 4 = 250)
    (h2 : (300 + 150 + n + x + y) / 5 = 200) :
    y = 50 := 
by
  -- Placeholder for the proof
  sorry

end find_y_l79_79041


namespace sin_double_angle_ineq_l79_79876

theorem sin_double_angle_ineq (α : ℝ) (h1 : Real.sin (π - α) = 1 / 3) (h2 : π / 2 ≤ α) (h3 : α ≤ π) :
  Real.sin (2 * α) = - (4 * Real.sqrt 2) / 9 := 
by
  sorry

end sin_double_angle_ineq_l79_79876


namespace banana_permutations_l79_79540

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l79_79540


namespace number_of_members_l79_79336

theorem number_of_members (n : ℕ) (h1 : n * n = 5929) : n = 77 :=
sorry

end number_of_members_l79_79336


namespace camp_cedar_counselors_l79_79235

theorem camp_cedar_counselors (boys : ℕ) (girls : ℕ) 
(counselors_for_boys : ℕ) (counselors_for_girls : ℕ) 
(total_counselors : ℕ) 
(h1 : boys = 80)
(h2 : girls = 6 * boys - 40)
(h3 : counselors_for_boys = boys / 5)
(h4 : counselors_for_girls = (girls + 11) / 12)  -- +11 to account for rounding up
(h5 : total_counselors = counselors_for_boys + counselors_for_girls) : 
total_counselors = 53 :=
by
  sorry

end camp_cedar_counselors_l79_79235


namespace banana_permutations_l79_79541

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l79_79541


namespace binary_operation_correct_l79_79988

-- Define the binary numbers involved
def bin1 := 0b110110 -- 110110_2
def bin2 := 0b101010 -- 101010_2
def bin3 := 0b100    -- 100_2

-- Define the operation in binary
def result := 0b111001101100 -- 111001101100_2

-- Lean statement to verify the operation result
theorem binary_operation_correct : (bin1 * bin2) / bin3 = result :=
by sorry

end binary_operation_correct_l79_79988


namespace inverse_proportion_quadrant_l79_79492

theorem inverse_proportion_quadrant (k : ℝ) (h : k < 0) : 
  ∀ x : ℝ, (0 < x → y = k / x → y < 0) ∧ (x < 0 → y = k / x → 0 < y) :=
by
  sorry

end inverse_proportion_quadrant_l79_79492


namespace initial_books_l79_79872

-- Definitions for the conditions.

def boxes (b : ℕ) : ℕ := 3 * b -- Box count
def booksInRoom : ℕ := 21 -- Books in the room
def booksOnTable : ℕ := 4 -- Books on the coffee table
def cookbooks : ℕ := 18 -- Cookbooks in the kitchen
def booksGrabbed : ℕ := 12 -- Books grabbed from the donation center
def booksNow : ℕ := 23 -- Books Henry has now

-- Define total number of books donated
def totalBooksDonated (inBoxes : ℕ) (additionalBooks : ℕ) : ℕ :=
  inBoxes + additionalBooks - booksGrabbed

-- Define number of books Henry initially had
def initialBooks (netDonated : ℕ) (booksCurrently : ℕ) : ℕ :=
  netDonated + booksCurrently

-- Proof goal
theorem initial_books (b : ℕ) (inBox : ℕ) (additionalBooks : ℕ) : 
  let totalBooks := booksInRoom + booksOnTable + cookbooks
  let inBoxes := boxes b
  let totalDonated := totalBooksDonated inBoxes totalBooks
  initialBooks totalDonated booksNow = 99 :=
by 
  simp [initialBooks, totalBooksDonated, boxes, booksInRoom, booksOnTable, cookbooks, booksGrabbed, booksNow]
  sorry

end initial_books_l79_79872


namespace problem_a4_inv_a4_eq_seven_l79_79011

theorem problem_a4_inv_a4_eq_seven (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^4 + (1/a)^4 = 7 :=
sorry

end problem_a4_inv_a4_eq_seven_l79_79011


namespace find_x_l79_79269

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h : a = (Real.cos (3 * x / 2), Real.sin (3 * x / 2)) ∧ b = (Real.cos (x / 2), -Real.sin (x / 2)) ∧ (a.1 + b.1) ^ 2 + (a.2 + b.2) ^ 2 = 1 ∧ 0 ≤ x ∧ x ≤ Real.pi)  :
  x = Real.pi / 3 ∨ x = 2 * Real.pi / 3 :=
by
  sorry

end find_x_l79_79269


namespace gcd_eq_gcd_of_eq_add_mul_l79_79439

theorem gcd_eq_gcd_of_eq_add_mul (a b q r : Int) (h_q : b > 0) (h_r : 0 ≤ r) (h_ar : a = b * q + r) : Int.gcd a b = Int.gcd b r :=
by
  -- Conditions: constraints and assertion
  exact sorry

end gcd_eq_gcd_of_eq_add_mul_l79_79439


namespace oranges_difference_l79_79736

-- Defining the number of sacks of ripe and unripe oranges
def sacks_ripe_oranges := 44
def sacks_unripe_oranges := 25

-- The statement to be proven
theorem oranges_difference : sacks_ripe_oranges - sacks_unripe_oranges = 19 :=
by
  -- Provide the exact calculation and result expected
  sorry

end oranges_difference_l79_79736


namespace number_of_letters_l79_79747

-- Definitions and Conditions, based on the given problem
variables (n : ℕ) -- n is the number of different letters in the local language

-- Given: The people have lost 129 words due to the prohibition of the seventh letter
def words_lost_due_to_prohibition (n : ℕ) : ℕ := 2 * n

-- The main theorem to prove
theorem number_of_letters (h : 129 = words_lost_due_to_prohibition n) : n = 65 :=
by sorry

end number_of_letters_l79_79747


namespace find_m_l79_79031

-- Definition of the constraints and the values of x and y that satisfy them
def constraint1 (x y : ℝ) : Prop := x + y - 2 ≥ 0
def constraint2 (x y : ℝ) : Prop := x - y + 1 ≥ 0
def constraint3 (x : ℝ) : Prop := x ≤ 3

-- Given conditions
def satisfies_constraints (x y : ℝ) : Prop := 
  constraint1 x y ∧ constraint2 x y ∧ constraint3 x

-- The objective to prove
theorem find_m (x y m : ℝ) (h : satisfies_constraints x y) : 
  (∀ x y, satisfies_constraints x y → (- 3 = m * x + y)) → m = -2 / 3 :=
by
  sorry

end find_m_l79_79031


namespace sum_of_odd_base4_digits_of_152_and_345_l79_79990

def base_4_digit_count (n : ℕ) : ℕ :=
    n.digits 4 |>.filter (λ x => x % 2 = 1) |>.length

theorem sum_of_odd_base4_digits_of_152_and_345 :
    base_4_digit_count 152 + base_4_digit_count 345 = 6 :=
by
    sorry

end sum_of_odd_base4_digits_of_152_and_345_l79_79990


namespace polynomial_strictly_monotonic_l79_79048

variable {P : ℝ → ℝ}

/-- The polynomial P(x) is such that the polynomials P(P(x)) and P(P(P(x))) are strictly monotonic 
on the entire real axis. Prove that P(x) is also strictly monotonic on the entire real axis. -/
theorem polynomial_strictly_monotonic
  (h1 : StrictMono (P ∘ P))
  (h2 : StrictMono (P ∘ P ∘ P)) :
  StrictMono P :=
sorry

end polynomial_strictly_monotonic_l79_79048


namespace problem_proof_l79_79763

def is_multiple_of (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

def y := 80 + 120 + 160 + 200 + 360 + 440 + 4040

theorem problem_proof :
  is_multiple_of 5 y ∧
  is_multiple_of 10 y ∧
  is_multiple_of 20 y ∧
  is_multiple_of 40 y := 
by
  sorry

end problem_proof_l79_79763


namespace minimum_people_l79_79136

def num_photos : ℕ := 10
def num_center_men : ℕ := 10
def num_people_per_photo : ℕ := 3

theorem minimum_people (n : ℕ) (h : n = num_photos) :
  (∃ total_people, total_people = 16) :=
sorry

end minimum_people_l79_79136


namespace profit_percentage_is_25_l79_79498

theorem profit_percentage_is_25 
  (selling_price : ℝ) (cost_price : ℝ) 
  (sp_val : selling_price = 600) 
  (cp_val : cost_price = 480) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_is_25_l79_79498


namespace trip_time_difference_l79_79229

-- Definitions of the given conditions
def speed_AB := 160 -- speed from A to B in km/h
def speed_BA := 120 -- speed from B to A in km/h
def distance_AB := 480 -- distance between A and B in km

-- Calculation of the time for each trip
def time_AB := distance_AB / speed_AB
def time_BA := distance_AB / speed_BA

-- The statement we need to prove
theorem trip_time_difference :
  (time_BA - time_AB) = 1 :=
by
  sorry

end trip_time_difference_l79_79229


namespace minimum_a_l79_79013

def f (x a : ℝ) : ℝ := x^2 - 2*x - abs (x-1-a) - abs (x-2) + 4

theorem minimum_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = -2 :=
sorry

end minimum_a_l79_79013


namespace not_right_triangle_D_l79_79084

theorem not_right_triangle_D : 
  ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 ∧
  (1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2) ∧
  (7^2 + 24^2 = 25^2) ∧
  (5^2 + 12^2 = 13^2) := 
by 
  have hA : 1^2 + (Real.sqrt 2)^2 = (Real.sqrt 3)^2 := by norm_num
  have hB : 7^2 + 24^2 = 25^2 := by norm_num
  have hC : 5^2 + 12^2 = 13^2 := by norm_num
  have hD : ¬ (Real.sqrt 3)^2 + 4^2 = 5^2 := by norm_num
  exact ⟨hD, hA, hB, hC⟩

#print axioms not_right_triangle_D

end not_right_triangle_D_l79_79084


namespace number_of_rectangles_in_5x5_grid_l79_79002

theorem number_of_rectangles_in_5x5_grid : 
  let n := 5 in (n.choose 2) * (n.choose 2) = 100 :=
by
  sorry

end number_of_rectangles_in_5x5_grid_l79_79002


namespace cube_sphere_volume_relation_l79_79354

theorem cube_sphere_volume_relation (n : ℕ) (h : 2 < n)
  (h_volume : n^3 - (n^3 * pi / 6) = (n^3 * pi / 3)) : n = 8 :=
sorry

end cube_sphere_volume_relation_l79_79354


namespace prob_three_blue_is_correct_l79_79217

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end prob_three_blue_is_correct_l79_79217


namespace fixed_point_translation_l79_79725

variable {R : Type*} [LinearOrderedField R]

def passes_through (f : R → R) (p : R × R) : Prop := f p.1 = p.2

theorem fixed_point_translation (f : R → R) (h : f 1 = 1) :
  passes_through (fun x => f (x + 2)) (-1, 1) :=
by
  sorry

end fixed_point_translation_l79_79725


namespace unique_solution_l79_79243

noncomputable def equation_satisfied (x : ℝ) : Prop :=
  2^x + 3^x + 6^x = 7^x

theorem unique_solution : ∀ x : ℝ, equation_satisfied x ↔ x = 2 := by
  sorry

end unique_solution_l79_79243


namespace arcsin_cos_eq_neg_pi_div_six_l79_79505

theorem arcsin_cos_eq_neg_pi_div_six :
  Real.arcsin (Real.cos (2 * Real.pi / 3)) = -Real.pi / 6 :=
by
  sorry

end arcsin_cos_eq_neg_pi_div_six_l79_79505


namespace remainder_when_6n_divided_by_4_l79_79202

theorem remainder_when_6n_divided_by_4 (n : ℤ) (h : n % 4 = 1) : 6 * n % 4 = 2 := by
  sorry

end remainder_when_6n_divided_by_4_l79_79202


namespace ratio_of_edges_l79_79607

theorem ratio_of_edges
  (V₁ V₂ : ℝ)
  (a b : ℝ)
  (hV : V₁ / V₂ = 8 / 1)
  (hV₁ : V₁ = a^3)
  (hV₂ : V₂ = b^3) :
  a / b = 2 / 1 := 
by 
  sorry

end ratio_of_edges_l79_79607


namespace find_number_l79_79881

-- Definitions used in the given problem conditions
def condition (x : ℝ) : Prop := (3.242 * x) / 100 = 0.04863

-- Statement of the problem
theorem find_number (x : ℝ) (h : condition x) : x = 1.5 :=
by
  sorry
 
end find_number_l79_79881


namespace scale_reading_l79_79309

theorem scale_reading (a b c : ℝ) (h₁ : 10.15 < a ∧ a < 10.4) (h₂ : 10.275 = (10.15 + 10.4) / 2) : a = 10.3 := 
by 
  sorry

end scale_reading_l79_79309


namespace trigonometric_identity_l79_79484

theorem trigonometric_identity 
  (α β : ℝ) 
  (h : α + β = π / 3)  -- Note: 60 degrees is π/3 radians
  (tan_add : ∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) 
  (tan_60 : Real.tan (π / 3) = Real.sqrt 3) :
  Real.tan α + Real.tan β + Real.sqrt 3 * Real.tan α * Real.tan β = Real.sqrt 3 :=
sorry

end trigonometric_identity_l79_79484


namespace calculate_dividend_l79_79204

def faceValue : ℕ := 100
def premiumPercent : ℕ := 20
def dividendPercent : ℕ := 5
def investment : ℕ := 14400
def costPerShare : ℕ := faceValue + (premiumPercent * faceValue / 100)
def numberOfShares : ℕ := investment / costPerShare
def dividendPerShare : ℕ := faceValue * dividendPercent / 100
def totalDividend : ℕ := numberOfShares * dividendPerShare

theorem calculate_dividend :
  totalDividend = 600 := 
by
  sorry

end calculate_dividend_l79_79204


namespace shortest_distance_between_circles_l79_79672

theorem shortest_distance_between_circles :
  let circle1 := (x^2 - 12*x + y^2 - 6*y + 9 = 0)
  let circle2 := (x^2 + 10*x + y^2 + 8*y + 34 = 0)
  -- Centers and radii from conditions above:
  let center1 := (6, 3)
  let radius1 := 3
  let center2 := (-5, -4)
  let radius2 := Real.sqrt 7
  let distance_centers := Real.sqrt ((6 - (-5))^2 + (3 - (-4))^2)
  -- Calculate shortest distance
  distance_centers - (radius1 + radius2) = Real.sqrt 170 - 3 - Real.sqrt 7 := sorry

end shortest_distance_between_circles_l79_79672


namespace banana_permutations_l79_79539

theorem banana_permutations : (Nat.factorial 6) / ((Nat.factorial 3) * (Nat.factorial 2)) = 60 := by
  sorry

end banana_permutations_l79_79539


namespace CH4_reaction_with_Cl2_l79_79737

def balanced_chemical_equation (CH4 Cl2 CH3Cl HCl : ℕ) : Prop :=
  CH4 + Cl2 = CH3Cl + HCl

theorem CH4_reaction_with_Cl2
  (CH4 Cl2 CH3Cl HCl : ℕ)
  (balanced_eq : balanced_chemical_equation 1 1 1 1)
  (reaction_cl2 : Cl2 = 2) :
  CH4 = 2 :=
by
  sorry

end CH4_reaction_with_Cl2_l79_79737


namespace find_element_atomic_mass_l79_79361

-- Define the atomic mass of bromine
def atomic_mass_br : ℝ := 79.904

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 267

-- Define the number of bromine atoms in the compound (assuming n = 1)
def n : ℕ := 1

-- Define the atomic mass of the unknown element X
def atomic_mass_x : ℝ := molecular_weight - n * atomic_mass_br

-- State the theorem to prove
theorem find_element_atomic_mass : atomic_mass_x = 187.096 :=
by
  -- placeholder for the proof
  sorry

end find_element_atomic_mass_l79_79361


namespace banana_arrangement_count_l79_79551

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l79_79551


namespace remainder_of_sum_l79_79671

theorem remainder_of_sum : 
  let a := 21160
  let b := 21162
  let c := 21164
  let d := 21166
  let e := 21168
  let f := 21170
  (a + b + c + d + e + f) % 12 = 6 :=
by
  sorry

end remainder_of_sum_l79_79671


namespace reciprocal_of_neg_three_l79_79184

theorem reciprocal_of_neg_three : (1 / (-3 : ℝ)) = (-1 / 3) := by
  sorry

end reciprocal_of_neg_three_l79_79184


namespace probability_all_blue_jellybeans_removed_l79_79215

def num_red_jellybeans : ℕ := 10
def num_blue_jellybeans : ℕ := 10
def total_jellybeans : ℕ := num_red_jellybeans + num_blue_jellybeans

def prob_first_blue : ℚ := num_blue_jellybeans / total_jellybeans
def prob_second_blue : ℚ := (num_blue_jellybeans - 1) / (total_jellybeans - 1)
def prob_third_blue : ℚ := (num_blue_jellybeans - 2) / (total_jellybeans - 2)

def prob_all_blue : ℚ := prob_first_blue * prob_second_blue * prob_third_blue

theorem probability_all_blue_jellybeans_removed :
  prob_all_blue = 1 / 9.5 := sorry

end probability_all_blue_jellybeans_removed_l79_79215


namespace trucks_on_lot_l79_79702

-- We'll state the conditions as hypotheses and then conclude the total number of trucks.
theorem trucks_on_lot (T : ℕ)
  (h₁ : ∀ N : ℕ, 50 ≤ N ∧ N ≤ 20 → N / 2 = 10)
  (h₂ : T ≥ 20 + 10): T = 30 :=
sorry

end trucks_on_lot_l79_79702


namespace seating_arrangement_l79_79061

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem seating_arrangement : 
  let republicans := 6
  let democrats := 4
  (factorial (republicans - 1)) * (binom republicans democrats) * (factorial democrats) = 43200 :=
by
  sorry

end seating_arrangement_l79_79061


namespace average_salary_l79_79449

theorem average_salary
  (num_technicians : ℕ) (avg_salary_technicians : ℝ)
  (num_other_workers : ℕ) (avg_salary_other_workers : ℝ)
  (total_num_workers : ℕ) (avg_salary_all_workers : ℝ) :
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  num_other_workers = total_num_workers - num_technicians →
  avg_salary_other_workers = 6000 →
  total_num_workers = 28 →
  avg_salary_all_workers = (num_technicians * avg_salary_technicians + num_other_workers * avg_salary_other_workers) / total_num_workers →
  avg_salary_all_workers = 8000 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l79_79449


namespace percentage_of_boys_l79_79609

theorem percentage_of_boys (total_students : ℕ) (ratio_boys_to_girls : ℕ) (ratio_girls_to_boys : ℕ) 
  (h_ratio : ratio_boys_to_girls = 3 ∧ ratio_girls_to_boys = 4 ∧ total_students = 42) : 
  (18 / 42) * 100 = 42.857 := 
by 
  sorry

end percentage_of_boys_l79_79609


namespace range_of_fraction_l79_79123

-- Definition of the quadratic equation with roots within specified intervals
variables (a b : ℝ)
variables (x1 x2 : ℝ)
variables (h_distinct_roots : x1 ≠ x2)
variables (h_interval_x1 : 0 < x1 ∧ x1 < 1)
variables (h_interval_x2 : 1 < x2 ∧ x2 < 2)
variables (h_quadratic : ∀ x : ℝ, x^2 + a * x + 2 * b - 2 = 0)

-- Prove range of expression
theorem range_of_fraction (a b : ℝ)
  (x1 x2 h_distinct_roots : ℝ) (h_interval_x1 : 0 < x1 ∧ x1 < 1)
  (h_interval_x2 : 1 < x2 ∧ x2 < 2)
  (h_quadratic : ∀ x, x^2 + a * x + 2 * b - 2 = 0) :
  (1/2 < (b - 4) / (a - 1)) ∧ ((b - 4) / (a - 1) < 3/2) :=
by
  -- proof placeholder
  sorry

end range_of_fraction_l79_79123


namespace problem_solution_l79_79865

theorem problem_solution (a b c : ℝ) (h : (a / (36 - a)) + (b / (45 - b)) + (c / (54 - c)) = 8) :
    (4 / (36 - a)) + (5 / (45 - b)) + (6 / (54 - c)) = 11 / 9 := 
by
  sorry

end problem_solution_l79_79865


namespace ball_hits_ground_approx_time_l79_79451

noncomputable def ball_hits_ground_time (t : ℝ) : ℝ :=
-6 * t^2 - 12 * t + 60

theorem ball_hits_ground_approx_time :
  ∃ t : ℝ, |t - 2.32| < 0.01 ∧ ball_hits_ground_time t = 0 :=
sorry

end ball_hits_ground_approx_time_l79_79451


namespace range_of_f_l79_79101

noncomputable def f (t : ℝ) : ℝ := (t^2 + 2 * t) / (t^2 + 2)

theorem range_of_f : Set.range f = Set.Icc (-1 : ℝ) 2 :=
sorry

end range_of_f_l79_79101


namespace fraction_absent_l79_79885

theorem fraction_absent (p : ℕ) (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : p * 1 = (1 - x) * p * 1.5) : x = 1 / 3 :=
by
  sorry

end fraction_absent_l79_79885


namespace meaningful_iff_gt_3_l79_79201

section meaningful_expression

variable (a : ℝ)

def is_meaningful (a : ℝ) : Prop :=
  (a > 3)

theorem meaningful_iff_gt_3 : (∃ b, b = (a + 3) / Real.sqrt (a - 3)) ↔ is_meaningful a :=
by
  sorry

end meaningful_expression

end meaningful_iff_gt_3_l79_79201


namespace alex_score_l79_79776

theorem alex_score 
    (n : ℕ) -- number of students
    (avg_19 : ℕ) -- average score of first 19 students
    (avg_20 : ℕ) -- average score of all 20 students
    (h_n : n = 20) -- number of students is 20
    (h_avg_19 : avg_19 = 75) -- average score of first 19 students is 75
    (h_avg_20 : avg_20 = 76) -- average score of all 20 students is 76
  : ∃ alex_score : ℕ, alex_score = 95 := 
by
    sorry

end alex_score_l79_79776


namespace kittens_given_away_l79_79231

-- Conditions
def initial_kittens : ℕ := 8
def remaining_kittens : ℕ := 4

-- Statement to prove
theorem kittens_given_away : initial_kittens - remaining_kittens = 4 :=
by
  sorry

end kittens_given_away_l79_79231


namespace find_literate_employees_l79_79750

-- Definitions based on conditions
def illiterate_employees : ℕ := 20
def initial_daily_wage : ℝ := 25
def decreased_daily_wage : ℝ := 10
def decrease_per_employee := initial_daily_wage - decreased_daily_wage
def total_decrease_illiterate := illiterate_employees * decrease_per_employee
def decrease_in_avg_salary : ℝ := 10

-- Question to be proven
def total_employees (literate_employees : ℕ) := literate_employees + illiterate_employees
def total_decrease_all_employees (literate_employees : ℕ) := (total_employees literate_employees) * decrease_in_avg_salary

theorem find_literate_employees : 
  ∃ L : ℕ, total_decrease_all_employees L = total_decrease_illiterate ∧ L = 10 :=
begin
  use 10,
  split,
  {
    change (10 + 20) * 10 = 300,
    norm_num
  },
  refl
end

end find_literate_employees_l79_79750


namespace find_second_number_l79_79342

theorem find_second_number
  (first_number : ℕ)
  (second_number : ℕ)
  (h1 : first_number = 45)
  (h2 : first_number / second_number = 5) : second_number = 9 :=
by
  -- Proof goes here
  sorry

end find_second_number_l79_79342


namespace greatest_integer_value_x_l79_79940

theorem greatest_integer_value_x :
  ∃ x : ℤ, (8 - 3 * (2 * x + 1) > 26) ∧ ∀ y : ℤ, (8 - 3 * (2 * y + 1) > 26) → y ≤ x :=
sorry

end greatest_integer_value_x_l79_79940


namespace find_correct_speed_l79_79293

-- Definitions for given conditions
def distance_traveled (speed : ℝ) (time : ℝ) : ℝ := speed * time

-- Given conditions as definitions
def condition1 (d t : ℝ) : Prop := distance_traveled 35 (t + (5 / 60)) = d
def condition2 (d t : ℝ) : Prop := distance_traveled 55 (t - (5 / 60)) = d

-- Statement to prove
theorem find_correct_speed (d t r : ℝ) (h1 : condition1 d t) (h2 : condition2 d t) :
  r = (d / t) ∧ r = 42.78 :=
by sorry

end find_correct_speed_l79_79293


namespace fixed_point_line_l79_79777

theorem fixed_point_line (m x y : ℝ) (h : (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0) :
  x = 3 ∧ y = 1 :=
sorry

end fixed_point_line_l79_79777


namespace perm_banana_l79_79563

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l79_79563


namespace find_probability_l79_79140

variable {σ : ℝ} (ξ : ℝ)
def is_normal_distribution : Prop := ξ ~ ℕ(0, σ^2)

theorem find_probability (h1 : is_normal_distribution ξ) (h2 : P(-2 ≤ ξ ∧ ξ ≤ 0) = 0.4) : 
  P(ξ > 2) = 0.1 :=
by
  sorry

end find_probability_l79_79140


namespace simple_random_sampling_correct_statements_l79_79085

theorem simple_random_sampling_correct_statements :
  let N : ℕ := 10
  -- Conditions for simple random sampling
  let is_finite (N : ℕ) := N > 0
  let is_non_sequential (N : ℕ) := N > 0 -- represents sampling does not require sequential order
  let without_replacement := true
  let equal_probability := true
  -- Verification
  (is_finite N) ∧ 
  (¬ is_non_sequential N) ∧ 
  without_replacement ∧ 
  equal_probability = true :=
by
  sorry

end simple_random_sampling_correct_statements_l79_79085


namespace jogged_time_l79_79714

theorem jogged_time (J : ℕ) (W : ℕ) (r : ℚ) (h1 : r = 5 / 3) (h2 : W = 9) (h3 : r = J / W) : J = 15 := 
by
  sorry

end jogged_time_l79_79714


namespace jill_sales_goal_l79_79423

def first_customer : ℕ := 5
def second_customer : ℕ := 4 * first_customer
def third_customer : ℕ := second_customer / 2
def fourth_customer : ℕ := 3 * third_customer
def fifth_customer : ℕ := 10
def boxes_sold : ℕ := first_customer + second_customer + third_customer + fourth_customer + fifth_customer
def boxes_left : ℕ := 75
def sales_goal : ℕ := boxes_sold + boxes_left

theorem jill_sales_goal : sales_goal = 150 := by
  sorry

end jill_sales_goal_l79_79423


namespace no_integer_y_such_that_abs_g_y_is_prime_l79_79196

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m > 0 → m ≤ n → m ∣ n → m = 1 ∨ m = n

def g (y : ℤ) : ℤ := 8 * y^2 - 55 * y + 21

theorem no_integer_y_such_that_abs_g_y_is_prime : 
  ∀ y : ℤ, ¬ is_prime (|g y|) :=
by sorry

end no_integer_y_such_that_abs_g_y_is_prime_l79_79196


namespace apples_per_basket_holds_15_l79_79086

-- Conditions as Definitions
def trees := 10
def total_apples := 3000
def baskets_per_tree := 20

-- Definition for apples per tree (from the given total apples and number of trees)
def apples_per_tree : ℕ := total_apples / trees

-- Definition for apples per basket (from apples per tree and baskets per tree)
def apples_per_basket : ℕ := apples_per_tree / baskets_per_tree

-- The statement to prove the equivalent mathematical problem
theorem apples_per_basket_holds_15 
  (H1 : trees = 10)
  (H2 : total_apples = 3000)
  (H3 : baskets_per_tree = 20) :
  apples_per_basket = 15 :=
by 
  sorry

end apples_per_basket_holds_15_l79_79086


namespace italian_clock_hand_coincidence_l79_79021

theorem italian_clock_hand_coincidence :
  let hour_hand_rotation := 1 / 24
  let minute_hand_rotation := 1
  ∃ (t : ℕ), 0 ≤ t ∧ t < 24 ∧ (t * hour_hand_rotation) % 1 = (t * minute_hand_rotation) % 1
:= sorry

end italian_clock_hand_coincidence_l79_79021


namespace oranges_equiv_frac_bananas_l79_79304

theorem oranges_equiv_frac_bananas :
  (3 / 4) * 16 * (1 / 3) * 9 = (3 / 2) * 6 :=
by
  sorry

end oranges_equiv_frac_bananas_l79_79304


namespace quadratic_roots_ratio_l79_79930

theorem quadratic_roots_ratio (m n p : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : p ≠ 0)
    (h₄ : ∀ (s₁ s₂ : ℝ), s₁ + s₂ = -p ∧ s₁ * s₂ = m ∧ 3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) :
    n / p = 27 :=
sorry

end quadratic_roots_ratio_l79_79930


namespace k_inequality_l79_79199

noncomputable def k_value : ℝ :=
  5

theorem k_inequality (x : ℝ) :
  (x * (2 * x + 3) < k_value) ↔ (x > -5 / 2 ∧ x < 1) :=
sorry

end k_inequality_l79_79199


namespace valid_paths_l79_79512

def paths (n m : ℕ) : ℕ := Nat.choose (n + m) m

def total_paths : ℕ := paths 8 4
def blocked_paths_through_C_or_D : ℕ := 4 * 56

theorem valid_paths : total_paths - blocked_paths_through_C_or_D = 271 := by
  sorry

end valid_paths_l79_79512


namespace largest_two_digit_number_l79_79946

-- Define the conditions and the theorem to be proven
theorem largest_two_digit_number (n : ℕ) : 
  (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 4) ∧ (10 ≤ n) ∧ (n < 100) → n = 84 := by
  sorry

end largest_two_digit_number_l79_79946


namespace heartsuit_ratio_l79_79007

-- Define the operation \heartsuit
def heartsuit (n m : ℕ) : ℕ := n^3 * m^2

-- The proposition we want to prove
theorem heartsuit_ratio :
  heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l79_79007


namespace particle_speed_correct_l79_79348

noncomputable def particle_position (t : ℝ) : ℝ × ℝ :=
  (3 * t + 5, 5 * t - 9)

noncomputable def particle_speed : ℝ :=
  Real.sqrt (3 ^ 2 + 5 ^ 2)

theorem particle_speed_correct : particle_speed = Real.sqrt 34 := by
  sorry

end particle_speed_correct_l79_79348


namespace estimate_yellow_balls_l79_79278

theorem estimate_yellow_balls (m : ℕ) (h1: (5 : ℝ) / (5 + m) = 0.2) : m = 20 :=
  sorry

end estimate_yellow_balls_l79_79278


namespace union_of_A_and_B_l79_79589

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l79_79589


namespace boy_present_age_l79_79255

theorem boy_present_age : ∃ x : ℕ, (x + 4 = 2 * (x - 6)) ∧ x = 16 := by
  sorry

end boy_present_age_l79_79255


namespace p_is_necessary_but_not_sufficient_for_q_l79_79858

def p (x : ℝ) : Prop := |2 * x - 3| < 1
def q (x : ℝ) : Prop := x * (x - 3) < 0

theorem p_is_necessary_but_not_sufficient_for_q :
  (∀ x : ℝ, q x → p x) ∧ ¬(∀ x : ℝ, p x → q x) :=
by sorry

end p_is_necessary_but_not_sufficient_for_q_l79_79858


namespace range_of_a_if_in_first_quadrant_l79_79866

noncomputable def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem range_of_a_if_in_first_quadrant (a : ℝ) :
  is_first_quadrant ((1 + a * Complex.I) / (2 - Complex.I)) ↔ (-1/2 : ℝ) < a ∧ a < 2 := 
sorry

end range_of_a_if_in_first_quadrant_l79_79866


namespace largest_integer_divides_expression_l79_79272

theorem largest_integer_divides_expression (x : ℤ) (h : Even x) :
  3 ∣ (10 * x + 1) * (10 * x + 5) * (5 * x + 3) :=
sorry

end largest_integer_divides_expression_l79_79272


namespace right_triangle_condition_l79_79176

theorem right_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  (A + B = 90) → (A + B + C = 180) → (C = 90) := 
by
  sorry

end right_triangle_condition_l79_79176


namespace binom_1300_2_l79_79366

theorem binom_1300_2 : Nat.choose 1300 2 = 844350 := by
  sorry

end binom_1300_2_l79_79366


namespace exists_ij_aij_gt_ij_l79_79116

theorem exists_ij_aij_gt_ij (a : ℕ → ℕ → ℕ) 
  (h_a_positive : ∀ i j, 0 < a i j)
  (h_a_distribution : ∀ k, (∃ S : Finset (ℕ × ℕ), S.card = 8 ∧ ∀ ij : ℕ × ℕ, ij ∈ S ↔ a ij.1 ij.2 = k)) :
  ∃ i j, a i j > i * j :=
by
  sorry

end exists_ij_aij_gt_ij_l79_79116


namespace probability_of_female_selection_probability_of_male_host_selection_l79_79611

/-!
In a competition, there are eight contestants consisting of five females and three males.
If three contestants are chosen randomly to progress to the next round, what is the 
probability that all selected contestants are female? Additionally, from those who 
do not proceed, one is selected as a host. What is the probability that this host is male?
-/

noncomputable def number_of_ways_select_3_from_8 : ℕ := Nat.choose 8 3

noncomputable def number_of_ways_select_3_females_from_5 : ℕ := Nat.choose 5 3

noncomputable def probability_all_3_females : ℚ := number_of_ways_select_3_females_from_5 / number_of_ways_select_3_from_8

noncomputable def number_of_remaining_contestants : ℕ := 8 - 3

noncomputable def number_of_males_remaining : ℕ := 3 - 1

noncomputable def number_of_ways_select_1_male_from_2 : ℕ := Nat.choose 2 1

noncomputable def number_of_ways_select_1_from_5 : ℕ := Nat.choose 5 1

noncomputable def probability_host_is_male : ℚ := number_of_ways_select_1_male_from_2 / number_of_ways_select_1_from_5

theorem probability_of_female_selection : probability_all_3_females = 5 / 28 := by
  sorry

theorem probability_of_male_host_selection : probability_host_is_male = 2 / 5 := by
  sorry

end probability_of_female_selection_probability_of_male_host_selection_l79_79611


namespace solve_system_l79_79647

open Classical

theorem solve_system : ∃ t : ℝ, ∀ (x y z : ℝ), 
  (x^2 - 9 * y^2 = 0 ∧ x + y + z = 0) ↔ 
  (x = 3 * t ∧ y = t ∧ z = -4 * t) 
  ∨ (x = -3 * t ∧ y = t ∧ z = 2 * t) := 
by 
  sorry

end solve_system_l79_79647


namespace determine_phi_l79_79862

variable (ω : ℝ) (varphi : ℝ)

noncomputable def f (ω varphi x: ℝ) : ℝ := Real.sin (ω * x + varphi)

theorem determine_phi
  (hω : ω > 0)
  (hvarphi : 0 < varphi ∧ varphi < π)
  (hx1 : f ω varphi (π/4) = Real.sin (ω * (π / 4) + varphi))
  (hx2 : f ω varphi (5 * π / 4) = Real.sin (ω * (5 * π / 4) + varphi))
  (hsym : ∀ x, f ω varphi x = f ω varphi (π - x))
  : varphi = π / 4 :=
sorry

end determine_phi_l79_79862


namespace points_lie_on_hyperbola_l79_79104

def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * Real.exp t - 2 * Real.exp (-t)
  let y := 4 * (Real.exp t + Real.exp (-t))
  (y^2) / 16 - (x^2) / 4 = 1

theorem points_lie_on_hyperbola : ∀ t : ℝ, point_on_hyperbola t :=
by
  intro t
  sorry

end points_lie_on_hyperbola_l79_79104


namespace convert_to_polar_l79_79368

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (r, θ)

theorem convert_to_polar (x y : ℝ) (hx : x = 8) (hy : y = 3 * Real.sqrt 3) :
  polar_coordinates x y = (Real.sqrt 91, Real.arctan (3 * Real.sqrt 3 / 8)) :=
by
  rw [hx, hy]
  simp [polar_coordinates]
  -- place to handle conversions and simplifications if necessary
  sorry

end convert_to_polar_l79_79368


namespace mb_range_l79_79659

theorem mb_range (m b : ℝ) (hm : m = 3 / 4) (hb : b = -2 / 3) :
  -1 < m * b ∧ m * b < 0 :=
by
  rw [hm, hb]
  sorry

end mb_range_l79_79659


namespace extra_sweets_l79_79294

theorem extra_sweets (S : ℕ) (h1 : ∀ n: ℕ, S = 120 * 38) : 
    (38 - (S / 190) = 14) :=
by
  -- Here we will provide the proof 
  sorry

end extra_sweets_l79_79294


namespace polygon_triangle_division_l79_79780

theorem polygon_triangle_division (n k : ℕ) (h₁ : n ≥ 3) (h₂ : k ≥ 1):
  k ≥ n - 2 :=
sorry

end polygon_triangle_division_l79_79780


namespace BANANA_arrangements_l79_79516

theorem BANANA_arrangements : 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  (Nat.factorial total_letters) / (Nat.factorial A_count * Nat.factorial N_count) = 60 := 
by 
  let total_letters := 6
  let A_count := 3
  let N_count := 2
  sorry

end BANANA_arrangements_l79_79516


namespace syllogism_sequence_correct_l79_79870

-- Definitions based on conditions
def square_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def rectangle_interior_angles_equal : Prop := ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D
def square_is_rectangle : Prop := ∀ (S : Type), S = S

-- Final Goal
theorem syllogism_sequence_correct : (rectangle_interior_angles_equal → square_is_rectangle → square_interior_angles_equal) :=
by
  sorry

end syllogism_sequence_correct_l79_79870


namespace quadratic_equation_root_form_l79_79984

theorem quadratic_equation_root_form
  (a b c : ℤ) (m n p : ℤ)
  (ha : a = 3)
  (hb : b = -4)
  (hc : c = -7)
  (h_discriminant : b^2 - 4 * a * c = n)
  (hgcd_mn : Int.gcd m n = 1)
  (hgcd_mp : Int.gcd m p = 1)
  (hgcd_np : Int.gcd n p = 1) :
  n = 100 :=
by
  sorry

end quadratic_equation_root_form_l79_79984


namespace variance_of_data_set_l79_79385

theorem variance_of_data_set (a : ℝ) (ha : (1 + a + 3 + 6 + 7) / 5 = 4) : 
  (1 / 5) * ((1 - 4)^2 + (a - 4)^2 + (3 - 4)^2 + (6 - 4)^2 + (7 - 4)^2) = 24 / 5 :=
by
  sorry

end variance_of_data_set_l79_79385


namespace probability_perpendicular_lines_l79_79580

open Finset

theorem probability_perpendicular_lines (a b : ℕ) :
  a ∈ (range 6).map (λ x, x + 1) → b ∈ (range 6).map (λ x, x + 1) →
  (filter (λ (p : ℕ × ℕ), let (a, b) := p in a - 2 * b = 0) ((range 6).map (λ x, x + 1)).product ((range 6).map (λ x, x + 1))).card.toRat / 36 = (1 : ℚ) / 12 :=
by
  intros ha hb
  sorry

end probability_perpendicular_lines_l79_79580


namespace value_of_g_at_2_l79_79602

def g (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem value_of_g_at_2 : g 2 = 0 :=
by
  sorry

end value_of_g_at_2_l79_79602


namespace constant_term_in_modified_equation_l79_79620

theorem constant_term_in_modified_equation :
  ∃ (c : ℝ), ∀ (q : ℝ), (3 * (3 * 5 - 3) - 3 + c = 132) → c = 99 := 
by
  sorry

end constant_term_in_modified_equation_l79_79620


namespace percentage_of_girls_after_change_l79_79616

variables (initial_total_children initial_boys initial_girls additional_boys : ℕ)
variables (percentage_boys : ℚ)

-- Initial conditions
def initial_conditions : Prop :=
  initial_total_children = 50 ∧
  percentage_boys = 90 / 100 ∧
  initial_boys = initial_total_children * percentage_boys ∧
  initial_girls = initial_total_children - initial_boys ∧
  additional_boys = 50

-- Statement to prove
theorem percentage_of_girls_after_change :
  initial_conditions initial_total_children initial_boys initial_girls additional_boys percentage_boys →
  (initial_girls / (initial_total_children + additional_boys) * 100 = 5) :=
by
  sorry

end percentage_of_girls_after_change_l79_79616


namespace ed_initial_money_l79_79240

-- Define initial conditions
def cost_per_hour_night : ℝ := 1.50
def hours_at_night : ℕ := 6
def cost_per_hour_morning : ℝ := 2
def hours_in_morning : ℕ := 4
def money_left : ℝ := 63

-- Total cost calculation
def total_cost : ℝ :=
  (cost_per_hour_night * hours_at_night) + (cost_per_hour_morning * hours_in_morning)

-- Problem statement to prove
theorem ed_initial_money : money_left + total_cost = 80 :=
by sorry

end ed_initial_money_l79_79240


namespace banana_arrangement_count_l79_79553

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l79_79553


namespace age_difference_l79_79380

def JobAge := 5
def StephanieAge := 4 * JobAge
def FreddyAge := 18

theorem age_difference : StephanieAge - FreddyAge = 2 := by
  sorry

end age_difference_l79_79380


namespace total_distance_l79_79642

def morning_distance : ℕ := 2
def evening_multiplier : ℕ := 5

theorem total_distance : morning_distance + (evening_multiplier * morning_distance) = 12 :=
by
  sorry

end total_distance_l79_79642


namespace distinct_arrangements_of_BANANA_l79_79555

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l79_79555


namespace LCM_of_8_and_12_l79_79921

-- Definitions based on the provided conditions
def a : ℕ := 8
def x : ℕ := 12

def HCF (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Conditions
def hcf_condition : HCF a x = 4 := by sorry
def x_condition : x = 12 := rfl

-- The proof statement
theorem LCM_of_8_and_12 : LCM a x = 24 :=
by
  have h1 : HCF a x = 4 := hcf_condition
  have h2 : x = 12 := x_condition
  rw [h2] at h1
  sorry

end LCM_of_8_and_12_l79_79921


namespace solve_sqrt_equation_l79_79377

theorem solve_sqrt_equation :
  ∀ (x : ℝ), (3 * Real.sqrt x + 3 * x⁻¹/2 = 7) →
  (x = (49 + 14 * Real.sqrt 13 + 13) / 36 ∨ x = (49 - 14 * Real.sqrt 13 + 13) / 36) :=
by
  intro x hx
  sorry

end solve_sqrt_equation_l79_79377


namespace total_valid_arrangements_l79_79644

-- Define the students and schools
inductive Student
| G1 | G2 | B1 | B2 | B3 | BA
deriving DecidableEq

inductive School
| A | B | C
deriving DecidableEq

-- Define the condition that any two students cannot be in the same school
def is_valid_arrangement (arr : School → Student → Bool) : Bool :=
  (arr School.A Student.G1 ≠ arr School.A Student.G2) ∧ 
  (arr School.B Student.G1 ≠ arr School.B Student.G2) ∧
  (arr School.C Student.G1 ≠ arr School.C Student.G2) ∧
  ¬ arr School.C Student.G1 ∧
  ¬ arr School.C Student.G2 ∧
  ¬ arr School.A Student.BA

-- The theorem to prove the total number of different valid arrangements
theorem total_valid_arrangements : 
  ∃ n : ℕ, n = 18 ∧ ∃ arr : (School → Student → Bool), is_valid_arrangement arr := 
sorry

end total_valid_arrangements_l79_79644


namespace find_possible_values_l79_79896

theorem find_possible_values (a b c k : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (sum_zero : a + b + c = 0) :
  (k * a^2 * b^2 + k * a^2 * c^2 + k * b^2 * c^2) / 
  ((a^2 - b * c) * (b^2 - a * c) + 
   (a^2 - b * c) * (c^2 - a * b) + 
   (b^2 - a * c) * (c^2 - a * b)) 
  = k / 3 :=
by 
  sorry

end find_possible_values_l79_79896


namespace intersection_of_A_and_B_l79_79582

noncomputable def A : Set ℝ := { x | -1 < x - 3 ∧ x - 3 ≤ 2 }
noncomputable def B : Set ℝ := { x | 3 ≤ x ∧ x < 6 }

theorem intersection_of_A_and_B : A ∩ B = { x | 3 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end intersection_of_A_and_B_l79_79582


namespace integer_count_of_sqrt_x_l79_79462

theorem integer_count_of_sqrt_x : ∃ (n : ℕ), n = 15 ∧ ∀ (x : ℤ), (9 < x ∧ x < 25) ↔ (10 ≤ x ∧ x ≤ 24) :=
by
  sorry

end integer_count_of_sqrt_x_l79_79462


namespace volume_of_blue_tetrahedron_l79_79068

theorem volume_of_blue_tetrahedron (side_length : ℝ) (H : side_length = 8) : 
  volume_of_tetrahedron_formed_by_blue_vertices side_length = 512 / 3 :=
by
  sorry

end volume_of_blue_tetrahedron_l79_79068


namespace quadratic_properties_l79_79734

open Real

noncomputable section

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Vertex form of the quadratic
def vertexForm (x : ℝ) : ℝ := (x - 2)^2 - 1

-- Axis of symmetry
def axisOfSymmetry : ℝ := 2

-- Vertex of the quadratic
def vertex : ℝ × ℝ := (2, -1)

-- Minimum value of the quadratic
def minimumValue : ℝ := -1

-- Interval where the function decreases
def decreasingInterval (x : ℝ) : Prop := -1 ≤ x ∧ x < 2

-- Range of y in the interval -1 <= x < 3
def rangeOfY (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 8

-- Main statement
theorem quadratic_properties :
  (∀ x, quadratic x = vertexForm x) ∧
  (∃ x, axisOfSymmetry = x) ∧
  (∃ v, vertex = v) ∧
  (minimumValue = -1) ∧
  (∀ x, -1 ≤ x ∧ x < 2 → quadratic x > quadratic (x + 1)) ∧
  (∀ y, (∃ x, -1 ≤ x ∧ x < 3 ∧ y = quadratic x) → rangeOfY y) :=
sorry

end quadratic_properties_l79_79734


namespace baseball_team_games_l79_79340

theorem baseball_team_games (P Q : ℕ) (hP : P > 3 * Q) (hQ : Q > 3) (hTotal : 2 * P + 6 * Q = 78) :
  2 * P = 54 :=
by
  -- placeholder for the actual proof
  sorry

end baseball_team_games_l79_79340


namespace tank_capacity_l79_79344

theorem tank_capacity
  (w c : ℝ)
  (h1 : w / c = 1 / 3)
  (h2 : (w + 5) / c = 2 / 5) :
  c = 75 :=
by
  sorry

end tank_capacity_l79_79344


namespace cherry_sodas_l79_79687

theorem cherry_sodas (C O : ℕ) (h1 : O = 2 * C) (h2 : C + O = 24) : C = 8 :=
by sorry

end cherry_sodas_l79_79687


namespace common_chord_through_vertex_l79_79597

-- Define the structure for the problem
def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

def passes_through (x y x_f y_f : ℝ) : Prop := (x - x_f) * (x - x_f) + y * y = 0

noncomputable def focus (p : ℝ) : ℝ × ℝ := (p / 2, 0)

-- The main statement to prove
theorem common_chord_through_vertex (p : ℝ)
  (A B C D : ℝ × ℝ)
  (hA : parabola A.snd A.fst p)
  (hB : parabola B.snd B.fst p)
  (hC : parabola C.snd C.fst p)
  (hD : parabola D.snd D.fst p)
  (hAB_f : passes_through A.fst A.snd (focus p).fst (focus p).snd)
  (hCD_f : passes_through C.fst C.snd (focus p).fst (focus p).snd) :
  ∃ k : ℝ, ∀ x y : ℝ, (x + k = 0) → (y + k = 0) :=
by sorry

end common_chord_through_vertex_l79_79597


namespace range_of_m_l79_79109

variable {x m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, m * x^2 + 1 ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem range_of_m (m : ℝ) : (¬ p m ∨ ¬ q m) → m ≥ 2 := 
sorry

end range_of_m_l79_79109


namespace cubic_expression_value_l79_79271

theorem cubic_expression_value (a b c : ℝ) 
  (h1 : a + b + c = 13) 
  (h2 : ab + ac + bc = 32) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 949 := 
by
  sorry

end cubic_expression_value_l79_79271


namespace coeff_x3_in_binom_expansion_l79_79753

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the coefficient function for x^k in the binomial expansion of (x + 1)^n
def binom_coeff (n k : ℕ) : ℕ := binom n k

-- The theorem to prove that the coefficient of x^3 in the expansion of (x + 1)^36 is 7140
theorem coeff_x3_in_binom_expansion : binom_coeff 36 3 = 7140 :=
by
  sorry

end coeff_x3_in_binom_expansion_l79_79753


namespace find_QS_l79_79785

theorem find_QS (RS QR QS : ℕ) (h1 : RS = 13) (h2 : QR = 5) (h3 : QR * 13 = 5 * 13) :
  QS = 12 :=
by
  sorry

end find_QS_l79_79785


namespace max_sin_a_given_sin_a_plus_b_l79_79093

theorem max_sin_a_given_sin_a_plus_b (a b : ℝ) (sin_add : Real.sin (a + b) = Real.sin a + Real.sin b) : 
  Real.sin a ≤ 1 := 
sorry

end max_sin_a_given_sin_a_plus_b_l79_79093


namespace permutations_of_BANANA_l79_79524

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l79_79524


namespace construct_points_PQ_l79_79712

-- Given Conditions
variable (a b c : ℝ)
def triangle_ABC_conditions : Prop := 
  let s := (a + b + c) / 2
  s^2 ≥ 2 * a * b

-- Main Statement
theorem construct_points_PQ (a b c : ℝ) (P Q : ℝ) 
(h1 : triangle_ABC_conditions a b c) :
  let s := (a + b + c) / 2
  let x := (s + Real.sqrt (s^2 - 2 * a * b)) / 2
  let y := (s - Real.sqrt (s^2 - 2 * a * b)) / 2
  x + y = s ∧ x * y = (a * b) / 2 :=
by
  sorry

end construct_points_PQ_l79_79712


namespace find_multiplier_l79_79009

theorem find_multiplier (x y n : ℤ) (h1 : 3 * x + y = 40) (h2 : 2 * x - y = 20) (h3 : y^2 = 16) :
  n * y^2 = 48 :=
by 
  -- proof goes here
  sorry

end find_multiplier_l79_79009


namespace expansion_coefficient_a2_l79_79875

theorem expansion_coefficient_a2 : 
  (∃ (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℝ), 
    (1 - 2*x)^7 = a + a_1*x + a_2*x^2 + a_3*x^3 + a_4*x^4 + a_5*x^5 + a_6*x^6 + a_7*x^7 -> 
    a_2 = 84) :=
sorry

end expansion_coefficient_a2_l79_79875


namespace measure_angle_CAB_in_hexagon_l79_79413

-- Define the problem conditions and goal.
theorem measure_angle_CAB_in_hexagon 
  (ABCDEF_regular: ∀ (AB BC CD DE EF FA: ℝ), AB = BC ∧ BC = CD ∧ CD = DE ∧ DE = EF ∧ EF = FA ∧ FA = AB)
  (interior_angle_120: ∀ (x y z w v u: ℝ), x+y+z+w+v+u = 720)
  (interior_angle_A: ∀ (angle: ℝ), angle = 120 )
  : ∃ (CAB:ℝ), CAB = 30 := 
sorry

end measure_angle_CAB_in_hexagon_l79_79413


namespace lateral_surface_area_cone_l79_79458

theorem lateral_surface_area_cone (r l : ℝ) (h₀ : r = 6) (h₁ : l = 10) : π * r * l = 60 * π := by 
  sorry

end lateral_surface_area_cone_l79_79458


namespace part1_l79_79060

variables {a b c : ℝ}
theorem part1 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a / (b + c) = b / (c + a) - c / (a + b)) : 
    b / (c + a) ≥ (Real.sqrt 17 - 1) / 4 :=
sorry

end part1_l79_79060


namespace tickets_difference_l79_79907

-- Define the conditions
def tickets_friday : ℕ := 181
def tickets_sunday : ℕ := 78
def tickets_saturday : ℕ := 2 * tickets_friday

-- The theorem to prove
theorem tickets_difference : tickets_saturday - tickets_sunday = 284 := by
  sorry

end tickets_difference_l79_79907


namespace eq_has_unique_solution_l79_79370

theorem eq_has_unique_solution : 
  ∃! x : ℝ, (x ≠ 0)
    ∧ ((x < 0 → false) ∧ 
      (x > 0 → (x^18 + 1) * (x^16 + x^14 + x^12 + x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 18 * x^9)) :=
by sorry

end eq_has_unique_solution_l79_79370


namespace cone_volume_l79_79836

theorem cone_volume (R h : ℝ) (hR : 0 ≤ R) (hh : 0 ≤ h) : 
  (∫ x in (0 : ℝ)..h, π * (R / h * x)^2) = (1 / 3) * π * R^2 * h :=
by
  sorry

end cone_volume_l79_79836


namespace solve_for_x_l79_79170

theorem solve_for_x (x : ℝ) : (|2 * x + 8| = 4 - 3 * x) → x = -4 / 5 :=
  sorry

end solve_for_x_l79_79170


namespace math_proof_problem_l79_79637

variable {a_n : ℕ → ℝ} -- sequence a_n
variable {b_n : ℕ → ℝ} -- sequence b_n

-- Given that a_n is an arithmetic sequence with common difference d
def isArithmeticSequence (a_n : ℕ → ℝ) (d : ℝ) : Prop := ∀ n, a_n (n + 1) = a_n n + d

-- Given condition for sequence b_n
def b_n_def (a_n b_n : ℕ → ℝ) : Prop := ∀ n, b_n n = a_n (n + 1) * a_n (n + 2) - a_n n ^ 2

-- Both sequences have common difference d ≠ 0
def common_difference_ne_zero (a_n b_n : ℕ → ℝ) (d : ℝ) : Prop :=
  isArithmeticSequence a_n d ∧ isArithmeticSequence b_n d ∧ d ≠ 0

-- Condition involving positive integers s and t
def integer_condition (a_n b_n : ℕ → ℝ) (s t : ℕ) : Prop :=
  1 ≤ s ∧ 1 ≤ t ∧ ∃ (x : ℤ), a_n s + b_n t = x

-- Theorem to prove that the sequence {b_n} is arithmetic and find minimum value of |a_1|
theorem math_proof_problem
  (a_n b_n : ℕ → ℝ) (d : ℝ) (s t : ℕ)
  (arithmetic_a : isArithmeticSequence a_n d)
  (defined_b : b_n_def a_n b_n)
  (common_diff : common_difference_ne_zero a_n b_n d)
  (int_condition : integer_condition a_n b_n s t) :
  (isArithmeticSequence b_n (3 * d ^ 2)) ∧ (∃ m : ℝ, m = |a_n 1| ∧ m = 1 / 36) :=
  by sorry

end math_proof_problem_l79_79637


namespace factor_polynomial_l79_79367

-- Statement of the proof problem
theorem factor_polynomial (x y z : ℝ) :
    x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
    (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) :=
by
  sorry

end factor_polynomial_l79_79367


namespace problem_solution_l79_79764

noncomputable def ellipse_properties (F1 F2 : ℝ × ℝ) (sum_dists : ℝ) : ℝ × ℝ × ℝ × ℝ := 
  let a := sum_dists / 2 
  let c := (Real.sqrt ((F1.1 - F2.1)^2 + (F1.2 - F2.2)^2)) / 2
  let b := Real.sqrt (a^2 - c^2)
  let h := (F1.1 + F2.1) / 2
  let k := (F1.2 + F2.2) / 2
  (h, k, a, b)

theorem problem_solution :
  let F1 := (0, 1)
  let F2 := (6, 1)
  let sum_dists := 10
  let (h, k, a, b) := ellipse_properties F1 F2 sum_dists
  h + k + a + b = 13 :=
by
  -- assuming the proof here
  sorry

end problem_solution_l79_79764


namespace solve_for_a_l79_79133

theorem solve_for_a (a : ℝ) (y : ℝ) (h1 : 4 * 2 + y = a) (h2 : 2 * 2 + 5 * y = 3 * a) : a = 18 :=
  sorry

end solve_for_a_l79_79133


namespace union_of_A_B_l79_79584

def A : Set ℝ := {x | |x - 3| < 2}
def B : Set ℝ := {x | (x + 1) / (x - 2) ≤ 0}

theorem union_of_A_B : A ∪ B = {x | -1 ≤ x ∧ x < 5} :=
by
  sorry

end union_of_A_B_l79_79584


namespace gcd_of_polynomial_and_multiple_of_12600_l79_79593

theorem gcd_of_polynomial_and_multiple_of_12600 (x : ℕ) (h : 12600 ∣ x) : gcd ((5 * x + 7) * (11 * x + 3) * (17 * x + 8) * (4 * x + 5)) x = 840 := by
  sorry

end gcd_of_polynomial_and_multiple_of_12600_l79_79593


namespace problem1_problem2_l79_79929

open Real

-- Define the line y = 2x + b
def line (x : ℝ) (b : ℝ) : ℝ := 2 * x + b

-- Define the parabola y = (1/2)x^2
def parabola (x : ℝ) : ℝ := (1 / 2) * x^2

-- Define the directrix of the parabola y = (1/2)x^2
def directrix : ℝ := -1 / 2

-- Define the equation of the circle centered at (2, 2) with radius 2.5
def circle (x y : ℝ) : ℝ :=
  (x - 2) ^ 2 + (y - 2) ^ 2

-- Problem 1: Prove that the line y = 2x - 2 is tangent to the parabola y = (1/2)x^2 at (2, 2)
theorem problem1 : 
  ∀ x : ℝ, 
  (∃ b : ℝ, b = -2 → line x b = parabola x) →
  (2 * 2 + -2 = (1 / 2) * 2 ^ 2) :=
by
  sorry

-- Problem 2: Prove that the equation of the circle with center at (2, 2) and tangent to the directrix is (x - 2)^2 + (y - 2)^2 = (5/2)^2
theorem problem2 : 
  ∀ x y : ℝ, 
  (circle x y = (5 / 2) ^ 2) :=
by
  sorry

end problem1_problem2_l79_79929


namespace correct_word_is_any_l79_79622

def words : List String := ["other", "any", "none", "some"]

def is_correct_word (word : String) : Prop :=
  "Jane was asked a lot of questions, but she didn’t answer " ++ word ++ " of them." = 
    "Jane was asked a lot of questions, but she didn’t answer any of them."

theorem correct_word_is_any : is_correct_word "any" :=
by
  sorry

end correct_word_is_any_l79_79622


namespace garden_width_l79_79180

theorem garden_width (w l : ℝ) (h_length : l = 3 * w) (h_area : l * w = 675) : w = 15 :=
by
  sorry

end garden_width_l79_79180


namespace number_of_arrangements_of_BANANA_l79_79534

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l79_79534


namespace find_omitted_angle_l79_79298

-- Definitions and conditions
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

def omitted_angle (calculated_sum actual_sum : ℝ) : ℝ :=
  actual_sum - calculated_sum

-- The theorem to be proven
theorem find_omitted_angle (n : ℕ) (h₁ : 1958 + 22 = sum_of_interior_angles n) :
  omitted_angle 1958 (sum_of_interior_angles n) = 22 :=
by
  sorry

end find_omitted_angle_l79_79298


namespace minimum_g_exists_x_minimum_g_l79_79375

noncomputable def g (x : ℝ) : ℝ := 
  x + x / (x^2 + 1) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem minimum_g (x : ℝ) (hx : x > 0) : g x ≥ 6 :=
by
  sorry

theorem exists_x_minimum_g : ∃ x > 0, g x = 6 :=
by
  sorry

end minimum_g_exists_x_minimum_g_l79_79375


namespace geometric_fraction_l79_79259

noncomputable def a_n : ℕ → ℝ := sorry
axiom a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5
axiom geometric_sequence : ∀ n, a_n (n + 1) = a_n n * a_n (n + 1) / a_n (n - 1) 

theorem geometric_fraction (a_n : ℕ → ℝ) (a1_a5_condition : a_n 1 + a_n 5 = a_n 1 * a_n 5) :
  (a_n 13) / (a_n 9) = 9 :=
sorry

end geometric_fraction_l79_79259


namespace angle_conversion_l79_79332

-- Define the known conditions
def full_circle_vens : ℕ := 800
def full_circle_degrees : ℕ := 360
def given_angle_degrees : ℕ := 135
def expected_vens : ℕ := 300

-- Prove that an angle of 135 degrees corresponds to 300 vens.
theorem angle_conversion :
  (given_angle_degrees * full_circle_vens) / full_circle_degrees = expected_vens := by
  sorry

end angle_conversion_l79_79332


namespace total_lemonade_poured_l79_79096

-- Define the amounts of lemonade served during each intermission.
def first_intermission : ℝ := 0.25
def second_intermission : ℝ := 0.42
def third_intermission : ℝ := 0.25

-- State the theorem that the total amount of lemonade poured is 0.92 pitchers.
theorem total_lemonade_poured : first_intermission + second_intermission + third_intermission = 0.92 :=
by
  -- Placeholders to skip the proof.
  sorry

end total_lemonade_poured_l79_79096


namespace number_of_arrangements_of_BANANA_l79_79535

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l79_79535


namespace find_number_l79_79478

theorem find_number (x : ℤ) (h : x - 7 = 9) : x * 3 = 48 :=
by sorry

end find_number_l79_79478


namespace inequality_proof_l79_79153

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x + y + z = 1) : 
  (x + y)^3 / z + (y + z)^3 / x + (z + x)^3 / y + 9 * x * y * z ≥ 9 * (x * y + y * z + z * x) :=
by 
  sorry

end inequality_proof_l79_79153


namespace complement_of_P_with_respect_to_U_l79_79728

universe u

def U : Set ℤ := {-1, 0, 1, 2}

def P : Set ℤ := {x | x * x < 2}

theorem complement_of_P_with_respect_to_U : U \ P = {2} :=
by
  sorry

end complement_of_P_with_respect_to_U_l79_79728


namespace problem_part_one_problem_part_two_l79_79118

noncomputable def a := 1
noncomputable def b := Real.log 2 - 1
noncomputable def b_min := -1 / 2

theorem problem_part_one (a : ℝ) (b : ℝ) 
(h1 : ∃ x : ℝ, x = 1 ∧ deriv (λ x, a * Real.log x) x = 1 / 2)
(h2 : ∃ x : ℝ, x = 1 ∧ (λ x, (1 / 2) * x + b) x = 0) : 
a = 1 ∧ b = Real.log 2 - 1 := 
sorry

theorem problem_part_two (a : ℝ) (b : ℝ) 
(h : a > 0) 
(h1 : ∃ x₀ : ℝ, 2 * a = x₀ ∧ (λ x₀, (1 / 2) * x₀ + b = a * Real.log x₀)) (h2 : ∃ x₀ : ℝ, g a x₀ = b) :
b = -1 / 2 :=
sorry

-- Define general function g used in the statement
noncomputable def g (a : ℝ) (x : ℝ) := x * Real.log x + x * (Real.log 2 - 1)

end problem_part_one_problem_part_two_l79_79118


namespace tan_420_eq_sqrt3_l79_79476

theorem tan_420_eq_sqrt3 : Real.tan (420 * Real.pi / 180) = Real.sqrt 3 := 
by 
  -- Additional mathematical justification can go here.
  sorry

end tan_420_eq_sqrt3_l79_79476


namespace permutations_of_BANANA_l79_79528

/-- The number of distinct permutations of the word "BANANA" is 60. -/
theorem permutations_of_BANANA : (Nat.factorial 6 / (Nat.factorial 3 * Nat.factorial 2 * Nat.factorial 1)) = 60 := by
  sorry

end permutations_of_BANANA_l79_79528


namespace intersection_necessary_but_not_sufficient_l79_79030

variables {M N P : Set α}

theorem intersection_necessary_but_not_sufficient : 
  (M ∩ P = N ∩ P) → (M ≠ N) :=
sorry

end intersection_necessary_but_not_sufficient_l79_79030


namespace union_of_A_and_B_l79_79587

def setA : Set ℝ := { x : ℝ | abs (x - 3) < 2 }
def setB : Set ℝ := { x : ℝ | (x + 1) / (x - 2) ≤ 0 }

theorem union_of_A_and_B : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_of_A_and_B_l79_79587


namespace permutations_of_banana_l79_79567

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l79_79567


namespace speed_of_first_boy_l79_79330

theorem speed_of_first_boy (x : ℝ) (h1 : 7.5 > 0) (h2 : 16 > 0) (h3 : 32 > 0) (h4 : 32 = 16 * (x - 7.5)) : x = 9.5 :=
by
  sorry

end speed_of_first_boy_l79_79330


namespace city_mpg_l79_79832

-- Definitions
def total_distance := 256.2 -- total distance in miles
def total_gallons := 21.0 -- total gallons of gasoline

-- Theorem statement
theorem city_mpg : total_distance / total_gallons = 12.2 :=
by sorry

end city_mpg_l79_79832


namespace minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l79_79762

noncomputable def minimum_for_specific_values : ℝ :=
  let m := 2 
  let n := 2 
  let p := 2 
  let xyz := 8 
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_specific_values_proof : minimum_for_specific_values = 36 := by
  sorry

noncomputable def minimum_for_arbitrary_values (m n p : ℝ) (h : m * n * p = 8) : ℝ :=
  let x := 2
  let y := 2
  let z := 2
  x^2 + y^2 + z^2 + m * x * y + n * x * z + p * y * z

theorem minimum_for_arbitrary_values_proof (m n p : ℝ) (h : m * n * p = 8) : minimum_for_arbitrary_values m n p h = 12 + 4 * (m + n + p) := by
  sorry

end minimum_for_specific_values_proof_minimum_for_arbitrary_values_proof_l79_79762


namespace maximum_xy_l79_79860

variable {a b c x y : ℝ}

theorem maximum_xy 
  (h1 : a * x + b * y + 2 * c = 0)
  (h2 : c ≠ 0)
  (h3 : a * b - c^2 ≥ 0) :
  ∃ (m : ℝ), m = x * y ∧ m ≤ 1 :=
sorry

end maximum_xy_l79_79860


namespace f_not_surjective_l79_79636

def f : ℝ → ℕ → Prop := sorry

theorem f_not_surjective (f : ℝ → ℕ) 
  (h : ∀ x y : ℝ, f (x + (1 / f y)) = f (y + (1 / f x))) : 
  ¬ (∀ n : ℕ, ∃ x : ℝ, f x = n) :=
sorry

end f_not_surjective_l79_79636


namespace calc_expression_l79_79706

theorem calc_expression : abs (real.sqrt 3 - 2) - (1 / 2)⁻¹ - 2 * real.sin (real.pi / 3) = -2 * real.sqrt 3 := 
by
  sorry

end calc_expression_l79_79706


namespace original_employees_229_l79_79480

noncomputable def original_number_of_employees (reduced_employees : ℕ) (reduction_percentage : ℝ) : ℝ := 
  reduced_employees / (1 - reduction_percentage)

theorem original_employees_229 : original_number_of_employees 195 0.15 = 229 := 
by
  sorry

end original_employees_229_l79_79480


namespace howard_groups_l79_79874

theorem howard_groups :
  (18 : ℕ) / (24 / 4) = 3 := sorry

end howard_groups_l79_79874


namespace vectorBC_computation_l79_79257

open Vector

def vectorAB : ℝ × ℝ := (2, 4)

def vectorAC : ℝ × ℝ := (1, 3)

theorem vectorBC_computation :
  (vectorAC.1 - vectorAB.1, vectorAC.2 - vectorAB.2) = (-1, -1) :=
sorry

end vectorBC_computation_l79_79257


namespace sum_of_even_factors_900_l79_79238

theorem sum_of_even_factors_900 : 
  ∃ (S : ℕ), 
  (∀ a b c : ℕ, 900 = 2^a * 3^b * 5^c → 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → 0 ≤ c ∧ c ≤ 2) → 
  (∀ a : ℕ, 1 ≤ a ∧ a ≤ 2 → ∃ b c : ℕ, 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 ∧ (2^a * 3^b * 5^c = 900 ∧ a ≠ 0)) → 
  S = 2418 := 
sorry

end sum_of_even_factors_900_l79_79238


namespace work_in_one_day_l79_79479

theorem work_in_one_day (A_days B_days : ℕ) (hA : A_days = 18) (hB : B_days = A_days / 2) :
  (1 / A_days + 1 / B_days) = 1 / 6 := 
by
  sorry

end work_in_one_day_l79_79479


namespace find_a_l79_79981

def diamond (a b : ℝ) : ℝ := 3 * a - b^2

theorem find_a (a : ℝ) (h : diamond a 6 = 15) : a = 17 :=
by
  sorry

end find_a_l79_79981


namespace percent_increase_surface_area_l79_79847

theorem percent_increase_surface_area (a b c : ℝ) :
  let S := 2 * (a * b + b * c + a * c)
  let S' := 2 * (1.8 * a * 1.8 * b + 1.8 * b * 1.8 * c + 1.8 * c * 1.8 * a)
  (S' - S) / S * 100 = 224 := by
  sorry

end percent_increase_surface_area_l79_79847


namespace problem_statement_l79_79155

noncomputable def alpha := 3 + Real.sqrt 8
noncomputable def beta := 3 - Real.sqrt 8
noncomputable def x := alpha ^ 1000
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem problem_statement : x * (1 - f) = 1 :=
by sorry

end problem_statement_l79_79155


namespace banana_arrangements_l79_79532

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l79_79532


namespace eqn_of_trajectory_max_area_triangle_l79_79119

open Real

def dist (p q : ℝ × ℝ) : ℝ :=
  sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Given conditions
variables (P : ℝ × ℝ) (O A M : ℝ × ℝ)
variable (r : ℝ)

axiom dist_ratio : dist P O = 2 * dist P A
axiom fixed_points : O = (0,0) ∧ A = (3,0) ∧ M = (4,0)
axiom curve_eq : ∀ x y : ℝ, P = (x, y) → dist P (4, 0) = 2

-- To Prove
theorem eqn_of_trajectory :
  (∀ P : ℝ × ℝ, dist P O = 2 * dist P A → (P.1 - 4)^2 + P.2^2 = 4) :=
by
  sorry

theorem max_area_triangle :
  (∀ A B : ℝ × ℝ, A ≠ B →
    A.1 ≠ M.1 → (∃ k : ℝ, k ≠ 0 ∧ A = (-1, 0) ∧ A.2 = k * (A.1 + 1)) →
    let Δ := (A.1, A.2 - B.2) → Δ = M →
    1/2 * dist A B * dist M (1/2 * (A + B)) ≤ 2) :=
by
  sorry

end eqn_of_trajectory_max_area_triangle_l79_79119


namespace barrels_oil_difference_l79_79469

/--
There are two barrels of oil, A and B.
1. $\frac{1}{3}$ of the oil is poured from barrel A into barrel B.
2. $\frac{1}{5}$ of the oil is poured from barrel B back into barrel A.
3. Each barrel contains 24kg of oil after the transfers.

Prove that originally, barrel A had 6 kg more oil than barrel B.
-/
theorem barrels_oil_difference :
  ∃ (x y : ℝ), (y = 48 - x) ∧
  (24 = (2 / 3) * x + (1 / 5) * (48 - x + (1 / 3) * x)) ∧
  (24 = (48 - x + (1 / 3) * x) * (4 / 5)) ∧
  (x - y = 6) :=
by
  sorry

end barrels_oil_difference_l79_79469


namespace monotonically_decreasing_interval_l79_79265

-- Given conditions
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- The proof problem statement
theorem monotonically_decreasing_interval :
  ∃ a b : ℝ, (0 ≤ a) ∧ (b ≤ 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → (deriv f x ≤ 0)) :=
sorry

end monotonically_decreasing_interval_l79_79265


namespace num_subsets_div_by_three_l79_79239

open Finset

def set : Finset ℕ := {101, 106, 111, 146, 154, 159}

def sums_to_three (s : Finset ℕ) : Prop :=
  s.card = 4 ∧ (s.sum % 3 = 0)

theorem num_subsets_div_by_three :
  (set.powerset.filter sums_to_three).card = 5 := 
sorry

end num_subsets_div_by_three_l79_79239


namespace number_of_arrangements_of_BANANA_l79_79538

theorem number_of_arrangements_of_BANANA :
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 :=
by
  let total_letters := 6
  let count_A := 3
  let count_N := 2
  let count_B := 1
  let factorial := (n : ℕ) => if n = 0 then 1 else n * factorial (n - 1)
  have h : (factorial total_letters) / ((factorial count_A) * (factorial count_N) * (factorial count_B)) = 60 := sorry
  exact h

end number_of_arrangements_of_BANANA_l79_79538


namespace range_of_a_l79_79388

theorem range_of_a (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_mono_inc : ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y)
  (h_ineq : f (a - 3) < f 4) : -1 < a ∧ a < 7 :=
by
  sorry

end range_of_a_l79_79388


namespace smaller_of_x_and_y_is_15_l79_79668

variable {x y : ℕ}

/-- Given two positive numbers x and y are in the ratio 3:5, 
and the sum of x and y plus 10 equals 50,
prove that the smaller of x and y is 15. -/
theorem smaller_of_x_and_y_is_15 (h1 : x * 5 = y * 3) (h2 : x + y + 10 = 50) (h3 : 0 < x) (h4 : 0 < y) : x = 15 :=
by
  sorry

end smaller_of_x_and_y_is_15_l79_79668


namespace time_for_train_to_pass_pole_l79_79817

-- Definitions based on conditions
def train_length_meters : ℕ := 160
def train_speed_kmph : ℕ := 72

-- The calculated speed in m/s
def train_speed_mps : ℕ := train_speed_kmph * 1000 / 3600

-- The calculation of time taken to pass the pole
def time_to_pass_pole : ℕ := train_length_meters / train_speed_mps

-- The theorem statement
theorem time_for_train_to_pass_pole : time_to_pass_pole = 8 := sorry

end time_for_train_to_pass_pole_l79_79817


namespace length_of_first_train_l79_79052

noncomputable def first_train_length 
  (speed_first_train_km_h : ℕ) 
  (speed_second_train_km_h : ℕ) 
  (length_second_train_m : ℕ) 
  (time_seconds : ℝ) 
  (relative_speed_m_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_km_h + speed_second_train_km_h) * (5 / 18)
  let distance_covered := relative_speed_mps * time_seconds
  let length_first_train := distance_covered - length_second_train_m
  length_first_train

theorem length_of_first_train : 
  first_train_length 40 50 165 11.039116870650348 25 = 110.9779217662587 :=
by 
  rw [first_train_length]
  sorry

end length_of_first_train_l79_79052


namespace union_of_A_and_B_l79_79591

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l79_79591


namespace add_to_make_divisible_l79_79475

theorem add_to_make_divisible :
  ∃ n, n = 34 ∧ ∃ k : ℕ, 758492136547 + n = 51 * k := by
  sorry

end add_to_make_divisible_l79_79475


namespace Mitch_needs_to_keep_500_for_license_and_registration_l79_79904

-- Define the constants and variables
def total_savings : ℕ := 20000
def cost_per_foot : ℕ := 1500
def longest_boat_length : ℕ := 12
def docking_fee_factor : ℕ := 3

-- Define the price of the longest boat
def cost_longest_boat : ℕ := longest_boat_length * cost_per_foot

-- Define the amount for license and registration
def license_and_registration (L : ℕ) : Prop :=
  total_savings - cost_longest_boat = L * (docking_fee_factor + 1)

-- The statement to be proved
theorem Mitch_needs_to_keep_500_for_license_and_registration :
  ∃ L : ℕ, license_and_registration L ∧ L = 500 :=
by
  -- Conditions and setup have already been defined, we now state the proof goal.
  sorry

end Mitch_needs_to_keep_500_for_license_and_registration_l79_79904


namespace ratio_of_perimeters_is_one_l79_79695

-- Definitions based on the given conditions
def original_rectangle : ℝ × ℝ := (6, 8)
def folded_rectangle : ℝ × ℝ := (3, 8)
def small_rectangle : ℝ × ℝ := (3, 4)
def large_rectangle : ℝ × ℝ := (3, 4)

-- The perimeter function for a rectangle given its dimensions (length, width)
def perimeter (r : ℝ × ℝ) : ℝ := 2 * (r.1 + r.2)

-- The main theorem to prove
theorem ratio_of_perimeters_is_one : 
  perimeter small_rectangle / perimeter large_rectangle = 1 :=
by
  sorry

end ratio_of_perimeters_is_one_l79_79695


namespace certain_number_l79_79878

theorem certain_number (n w : ℕ) (h1 : w = 132)
  (h2 : ∃ m1 m2 m3, 32 = 2^5 * 3^3 * 11^2 * m1 * m2 * m3)
  (h3 : n * w = 132 * 2^3 * 3^2 * 11)
  (h4 : m1 = 1) (h5 : m2 = 1) (h6 : m3 = 1): 
  n = 792 :=
by sorry

end certain_number_l79_79878


namespace local_minimum_at_1_1_l79_79281

noncomputable def function (x y : ℝ) : ℝ :=
  x^3 + y^3 - 3 * x * y

theorem local_minimum_at_1_1 : 
  ∃ (x y : ℝ), x = 1 ∧ y = 1 ∧ (∀ (z : ℝ), z = function x y → z = -1) :=
sorry

end local_minimum_at_1_1_l79_79281


namespace abs_sum_less_abs_diff_l79_79600

theorem abs_sum_less_abs_diff {a b : ℝ} (hab : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_sum_less_abs_diff_l79_79600


namespace distinct_arrangements_of_BANANA_l79_79554

theorem distinct_arrangements_of_BANANA :
  let total_letters := 6
  let freq_A := 3
  let freq_N := 2
  let freq_B := 1
  (nat.factorial total_letters) / (nat.factorial freq_A * nat.factorial freq_N * nat.factorial freq_B) = 60 :=
by
  sorry

end distinct_arrangements_of_BANANA_l79_79554


namespace gold_bars_per_row_l79_79167

theorem gold_bars_per_row 
  (total_worth : ℝ)
  (total_rows : ℕ)
  (value_per_bar : ℝ)
  (h_total_worth : total_worth = 1600000)
  (h_total_rows : total_rows = 4)
  (h_value_per_bar : value_per_bar = 40000) :
  total_worth / value_per_bar / total_rows = 10 :=
by
  sorry

end gold_bars_per_row_l79_79167


namespace calc_tan_fraction_l79_79504

theorem calc_tan_fraction :
  (1 - Real.tan (15 * Real.pi / 180)) / (1 + Real.tan (15 * Real.pi / 180)) = Real.tan (30 * Real.pi / 180) :=
by
  have h_tan_45 : Real.tan (45 * Real.pi / 180) = 1 := by sorry
  have h_tan_30 : Real.tan (30 * Real.pi / 180) = Real.sqrt 3 / 3 := by sorry
  sorry

end calc_tan_fraction_l79_79504


namespace rhombus_diagonal_l79_79656

/-- Given a rhombus with one diagonal being 11 cm and the area of the rhombus being 88 cm²,
prove that the length of the other diagonal is 16 cm. -/
theorem rhombus_diagonal 
  (d1 : ℝ) (d2 : ℝ) (area : ℝ)
  (h_d1 : d1 = 11)
  (h_area : area = 88)
  (h_area_eq : area = (d1 * d2) / 2) : d2 = 16 :=
sorry

end rhombus_diagonal_l79_79656


namespace total_assignment_schemes_l79_79991

def num_ways_to_assign_teams : ℕ :=
  let ways_A := choose 4 1  -- C^1_4
  let ways_rest := factorial 4  -- 4!
  ways_A * ways_rest

theorem total_assignment_schemes :
    num_ways_to_assign_teams = 96 :=
by
  -- We can define each component individually in Lean 4 if needed, but this compact form achieves the same:
  have h1: choose 4 1 = 4 := by simp [choose]
  have h2: factorial 4 = 24 := by norm_num
  have h3: num_ways_to_assign_teams = 4 * 24 := by simp [num_ways_to_assign_teams, h1, h2]
  rw [h3]
  norm_num -- This computes the final result 4 * 24 = 96

end total_assignment_schemes_l79_79991


namespace tank_capacity_l79_79339

theorem tank_capacity (liters_cost : ℕ) (liters_amount : ℕ) (full_tank_cost : ℕ) (h₁ : liters_cost = 18) (h₂ : liters_amount = 36) (h₃ : full_tank_cost = 32) : 
  (full_tank_cost * liters_amount / liters_cost) = 64 :=
by 
  sorry

end tank_capacity_l79_79339


namespace alloy_price_per_kg_l79_79890

theorem alloy_price_per_kg (cost_A cost_B ratio_A_B total_cost total_weight price_per_kg : ℤ)
  (hA : cost_A = 68) 
  (hB : cost_B = 96) 
  (hRatio : ratio_A_B = 3) 
  (hTotalCost : total_cost = 3 * cost_A + cost_B) 
  (hTotalWeight : total_weight = 3 + 1)
  (hPricePerKg : price_per_kg = total_cost / total_weight) : 
  price_per_kg = 75 := 
by
  sorry

end alloy_price_per_kg_l79_79890


namespace pineapple_cost_l79_79350

variables (P W : ℕ)

theorem pineapple_cost (h1 : 2 * P + 5 * W = 38) : P = 14 :=
sorry

end pineapple_cost_l79_79350


namespace pencil_length_l79_79349

theorem pencil_length (L : ℝ) 
  (h1 : 1 / 8 * L = b) 
  (h2 : 1 / 2 * (L - 1 / 8 * L) = w) 
  (h3 : (L - 1 / 8 * L - 1 / 2 * (L - 1 / 8 * L)) = 7 / 2) :
  L = 8 :=
sorry

end pencil_length_l79_79349


namespace simplify_expression_d_l79_79083

variable (a b c : ℝ)

theorem simplify_expression_d : a - (b - c) = a - b + c :=
  sorry

end simplify_expression_d_l79_79083


namespace bob_initial_cats_l79_79787

theorem bob_initial_cats (B : ℕ) (h : 21 - 4 = B + 14) : B = 3 := 
by
  -- Placeholder for the proof
  sorry

end bob_initial_cats_l79_79787


namespace normal_distribution_prob_l79_79863

open Probability

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := 
λ x, exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * sqrt (2 * π))

theorem normal_distribution_prob {a : ℝ} {σ : ℝ} (hσ : σ > 0)
  (X : ℝ → ℝ) (hX : ∀ x, X x = normal_distribution 2 σ x)
  (h_prob : ∫ x in -∞ .. a, X x = 0.32) :
  ∫ x in a..4-a, X x = 0.36 :=
sorry

end normal_distribution_prob_l79_79863


namespace number_20_l79_79759

def Jo (n : ℕ) : ℕ :=
  1 + 5 * (n - 1)

def Blair (n : ℕ) : ℕ :=
  3 + 5 * (n - 1)

def number_at_turn (k : ℕ) : ℕ :=
  if k % 2 = 1 then Jo ((k + 1) / 2) else Blair (k / 2)

theorem number_20 : number_at_turn 20 = 48 :=
by
  sorry

end number_20_l79_79759


namespace total_hike_time_l79_79623

-- Define the conditions
def distance_to_mount_overlook : ℝ := 12
def pace_to_mount_overlook : ℝ := 4
def pace_return : ℝ := 6

-- Prove the total time for the hike
theorem total_hike_time :
  (distance_to_mount_overlook / pace_to_mount_overlook) +
  (distance_to_mount_overlook / pace_return) = 5 := 
sorry

end total_hike_time_l79_79623


namespace negation_proposition_l79_79454

open Classical

theorem negation_proposition :
  ¬ (∃ x : ℝ, x^2 - 2*x + 1 ≤ 0) ↔ ∀ x : ℝ, x^2 - 2*x + 1 > 0 :=
by
  sorry

end negation_proposition_l79_79454


namespace permutations_of_banana_l79_79565

theorem permutations_of_banana : (Nat.fac 6) / ((Nat.fac 3) * (Nat.fac 2)) = 60 := 
by
  sorry

end permutations_of_banana_l79_79565


namespace sodium_bicarbonate_moles_l79_79249

theorem sodium_bicarbonate_moles (HCl NaHCO3 CO2 : ℕ) (h1 : HCl = 1) (h2 : CO2 = 1) :
  NaHCO3 = 1 :=
by sorry

end sodium_bicarbonate_moles_l79_79249


namespace smallest_w_l79_79434

theorem smallest_w (x y w : ℕ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 ^ x) ∣ (3125 * w)) (h4 : (3 ^ y) ∣ (3125 * w)) 
  (h5 : (5 ^ (x + y)) ∣ (3125 * w)) (h6 : (7 ^ (x - y)) ∣ (3125 * w))
  (h7 : (13 ^ 4) ∣ (3125 * w))
  (h8 : x + y ≤ 10) (h9 : x - y ≥ 2) :
  w = 33592336 :=
by
  sorry

end smallest_w_l79_79434


namespace original_profit_margin_theorem_l79_79958

noncomputable def original_profit_margin (a : ℝ) (x : ℝ) (h : a > 0) : Prop := 
  (a * (1 + x) - a * (1 - 0.064)) / (a * (1 - 0.064)) = x + 0.08

theorem original_profit_margin_theorem (a : ℝ) (x : ℝ) (h : a > 0) :
  original_profit_margin a x h → x = 0.17 :=
sorry

end original_profit_margin_theorem_l79_79958


namespace perm_banana_l79_79559

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l79_79559


namespace solution_set_of_inverse_inequality_l79_79383

open Function

variable {f : ℝ → ℝ}

theorem solution_set_of_inverse_inequality 
  (h_decreasing : ∀ x y, x < y → f y < f x)
  (h_A : f (-2) = 2)
  (h_B : f 2 = -2)
  : { x : ℝ | |(invFun f (x + 1))| ≤ 2 } = { x : ℝ | -3 ≤ x ∧ x ≤ 1 } :=
sorry

end solution_set_of_inverse_inequality_l79_79383


namespace common_factor_l79_79924

-- Definition of the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4 * x * (m - n) + 2 * y * (m - n) ^ 2

-- The theorem statement
theorem common_factor (x y m n : ℝ) : ∃ k : ℝ, k * (m - n) = polynomial x y m n :=
sorry

end common_factor_l79_79924


namespace football_field_area_l79_79691

theorem football_field_area (total_fertilizer : ℝ) (partial_fertilizer : ℝ) (partial_area : ℝ) (fertilizer_rate : ℝ) (total_area : ℝ) 
  (h1 : total_fertilizer = 800)
  (h2: partial_fertilizer = 300)
  (h3: partial_area = 3600)
  (h4: fertilizer_rate = partial_fertilizer / partial_area)
  (h5: total_area = total_fertilizer / fertilizer_rate) 
  : total_area = 9600 := 
sorry

end football_field_area_l79_79691


namespace area_of_rectangle_at_stage_4_l79_79275

def area_at_stage (n : ℕ) : ℕ :=
  let square_area := 16
  let initial_squares := 2
  let common_difference := 2
  let total_squares := initial_squares + common_difference * (n - 1)
  total_squares * square_area

theorem area_of_rectangle_at_stage_4 :
  area_at_stage 4 = 128 :=
by
  -- computation and transformations are omitted
  sorry

end area_of_rectangle_at_stage_4_l79_79275


namespace solve_inequality_l79_79820

theorem solve_inequality (x : ℝ) (h : 0 < x ∧ x < 2) : abs (2 * x - 1) < abs x + 1 :=
by
  sorry

end solve_inequality_l79_79820


namespace ball_hits_ground_at_two_seconds_l79_79178

theorem ball_hits_ground_at_two_seconds :
  (∃ t : ℝ, (-6.1) * t^2 + 2.8 * t + 7 = 0 ∧ t = 2) :=
sorry

end ball_hits_ground_at_two_seconds_l79_79178


namespace percent_deficit_in_width_l79_79751

theorem percent_deficit_in_width (L W : ℝ) (h : 1.08 * (1 - (d : ℝ) / W) = 1.0044) : d = 0.07 * W :=
by sorry

end percent_deficit_in_width_l79_79751


namespace evaluate_F_2_f_3_l79_79740

def f (a : ℤ) : ℤ := a^2 - 2
def F (a b : ℤ) : ℤ := b^3 - a

theorem evaluate_F_2_f_3 : F 2 (f 3) = 341 := by
  sorry

end evaluate_F_2_f_3_l79_79740


namespace percent_of_rs_600_l79_79816

theorem percent_of_rs_600 : (600 * 0.25 = 150) :=
by
  sorry

end percent_of_rs_600_l79_79816


namespace coin_ratio_l79_79685

theorem coin_ratio (n₁ n₅ n₂₅ : ℕ) (total_value : ℕ) 
  (h₁ : n₁ = 40) 
  (h₅ : n₅ = 40) 
  (h₂₅ : n₂₅ = 40) 
  (hv : total_value = 70) 
  (hv_calc : n₁ * 1 + n₅ * (50 / 100) + n₂₅ * (25 / 100) = total_value) : 
  n₁ = n₅ ∧ n₁ = n₂₅ :=
by
  sorry

end coin_ratio_l79_79685


namespace fraction_meaningful_range_l79_79666

-- Define the condition
def meaningful_fraction_condition (x : ℝ) : Prop := (x - 2023) ≠ 0

-- Define the conclusion that we need to prove
def meaningful_fraction_range (x : ℝ) : Prop := x ≠ 2023

theorem fraction_meaningful_range (x : ℝ) : meaningful_fraction_condition x → meaningful_fraction_range x :=
by
  intro h
  -- Proof steps would go here
  sorry

end fraction_meaningful_range_l79_79666


namespace prove_tirzah_handbags_l79_79327
noncomputable def tirzah_has_24_handbags (H : ℕ) : Prop :=
  let P := 26 -- number of purses
  let fakeP := P / 2 -- half of the purses are fake
  let authP := P - fakeP -- number of authentic purses
  let fakeH := H / 4 -- one quarter of the handbags are fake
  let authH := H - fakeH -- number of authentic handbags
  authP + authH = 31 -- total number of authentic items
  → H = 24 -- prove the number of handbags is 24

theorem prove_tirzah_handbags : ∃ H : ℕ, tirzah_has_24_handbags H :=
  by
    use 24
    -- Proof goes here
    sorry

end prove_tirzah_handbags_l79_79327


namespace rows_of_seats_l79_79185

theorem rows_of_seats (r : ℕ) (h : r * 4 = 80) : r = 20 :=
sorry

end rows_of_seats_l79_79185


namespace shifted_parabola_is_correct_l79_79267

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  -((x - 1) ^ 2) + 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ :=
  -((x + 1 - 1) ^ 2) + 4

-- State the theorem
theorem shifted_parabola_is_correct :
  ∀ x : ℝ, shifted_parabola x = -x^2 + 4 :=
by
  -- Proof would go here
  sorry

end shifted_parabola_is_correct_l79_79267


namespace largest_value_l79_79477

noncomputable def a : ℕ := 2 ^ 6
noncomputable def b : ℕ := 3 ^ 5
noncomputable def c : ℕ := 4 ^ 4
noncomputable def d : ℕ := 5 ^ 3
noncomputable def e : ℕ := 6 ^ 2

theorem largest_value : c > a ∧ c > b ∧ c > d ∧ c > e := by
  sorry

end largest_value_l79_79477


namespace largest_of_three_numbers_l79_79471

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end largest_of_three_numbers_l79_79471


namespace correct_answer_l79_79864

def M : Set ℕ := {1}
def N : Set ℕ := {1, 2, 3}

theorem correct_answer : M ⊆ N := by
  sorry

end correct_answer_l79_79864


namespace find_P_and_Q_l79_79402

variables {x P Q b c : ℝ}

theorem find_P_and_Q :
  (∃ b c : ℝ, (x^2 + 3 * x + 7) * (x^2 + b * x + c) = x^4 + P * x^2 + Q) →
  (b + 3 = 0) →
  (3 * b + c + 7 = P) →
  (7 * b + 3 * c = 0) →
  (7 * c = Q) →
  P + Q = 54 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end find_P_and_Q_l79_79402


namespace remainder_83_pow_89_times_5_mod_11_l79_79943

theorem remainder_83_pow_89_times_5_mod_11 : 
  (83^89 * 5) % 11 = 10 := 
by
  have h1 : 83 % 11 = 6 := by sorry
  have h2 : 6^10 % 11 = 1 := by sorry
  have h3 : 89 = 8 * 10 + 9 := by sorry
  sorry

end remainder_83_pow_89_times_5_mod_11_l79_79943


namespace sum_of_coeffs_eq_92_l79_79987

noncomputable def sum_of_integer_coeffs_in_factorization (x y : ℝ) : ℝ :=
  let f := 27 * (x ^ 6) - 512 * (y ^ 6)
  3 - 8 + 9 + 24 + 64  -- Sum of integer coefficients

theorem sum_of_coeffs_eq_92 (x y : ℝ) : sum_of_integer_coeffs_in_factorization x y = 92 :=
by
  -- proof steps go here
  sorry

end sum_of_coeffs_eq_92_l79_79987


namespace concyclic_A_D_E_N_l79_79261

open EuclideanGeometry

variables {A B C D E N : Point}

-- Given conditions
axiom triangleABC : Triangle A B C
axiom Gamma : Circle A B C -- Circumcircle of triangle ABC
axiom D_on_AB : D ∈ Segment A B
axiom E_on_AC : E ∈ Segment A C
axiom BD_CE : Distance B D = Distance C E
axiom N_midpoint_arc : N = midpointArc A B C Gamma

theorem concyclic_A_D_E_N :
  Concyclic {A, D, E, N} :=
by
  sorry

end concyclic_A_D_E_N_l79_79261


namespace find_n_l79_79210

theorem find_n 
  (n : ℕ) 
  (h_lcm : Nat.lcm n 16 = 48) 
  (h_gcf : Nat.gcd n 16 = 18) : 
  n = 54 := 
sorry

end find_n_l79_79210


namespace min_value_of_a_l79_79395

theorem min_value_of_a (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x + y) * (1/x + a/y) ≥ 16) : a ≥ 9 :=
sorry

end min_value_of_a_l79_79395


namespace banana_arrangement_count_l79_79550

theorem banana_arrangement_count :
  let word := "BANANA"
  let length_word := 6
  let a_count := 3
  let n_count := 2
  (length_word)! / (a_count! * n_count!) = 60 := by
  sorry

end banana_arrangement_count_l79_79550


namespace inequality_proof_l79_79769

theorem inequality_proof {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 :=
by
  -- Proof goes here
  sorry

end inequality_proof_l79_79769


namespace red_knights_fraction_magic_l79_79612

theorem red_knights_fraction_magic (total_knights red_knights blue_knights magical_knights : ℕ)
  (h1 : red_knights = (3 / 8 : ℚ) * total_knights)
  (h2 : blue_knights = total_knights - red_knights)
  (h3 : magical_knights = (1 / 4 : ℚ) * total_knights)
  (fraction_red_magic fraction_blue_magic : ℚ) 
  (h4 : fraction_red_magic = 3 * fraction_blue_magic)
  (h5 : magical_knights = red_knights * fraction_red_magic + blue_knights * fraction_blue_magic) :
  fraction_red_magic = 3 / 7 := 
by
  sorry

end red_knights_fraction_magic_l79_79612


namespace total_hike_time_l79_79631

/-!
# Problem Statement
Jeannie hikes the 12 miles to Mount Overlook at a pace of 4 miles per hour, 
and then returns at a pace of 6 miles per hour. Prove that the total time 
Jeannie spent on her hike is 5 hours.
-/

def distance_to_mountain : ℝ := 12
def pace_up : ℝ := 4
def pace_down : ℝ := 6

theorem total_hike_time :
  (distance_to_mountain / pace_up) + (distance_to_mountain / pace_down) = 5 := 
by 
  sorry

end total_hike_time_l79_79631


namespace complaint_online_prob_l79_79964

/-- Define the various probability conditions -/
def prob_online := 4 / 5
def prob_store := 1 / 5
def qual_rate_online := 17 / 20
def qual_rate_store := 9 / 10
def non_qual_rate_online := 1 - qual_rate_online
def non_qual_rate_store := 1 - qual_rate_store
def prob_complaint_online := prob_online * non_qual_rate_online
def prob_complaint_store := prob_store * non_qual_rate_store
def total_prob_complaint := prob_complaint_online + prob_complaint_store

/-- The theorem states that given the conditions, the probability of an online purchase given a complaint is 6/7 -/
theorem complaint_online_prob : 
    (prob_complaint_online / total_prob_complaint) = 6 / 7 := 
by
    sorry

end complaint_online_prob_l79_79964


namespace wire_length_l79_79079

theorem wire_length (S L W : ℝ) (h1 : S = 20) (h2 : S = (2 / 7) * L) (h3 : W = S + L) : W = 90 :=
by sorry

end wire_length_l79_79079


namespace store_revenue_is_1210_l79_79223

noncomputable def shirt_price : ℕ := 10
noncomputable def jeans_price : ℕ := 2 * shirt_price
noncomputable def jacket_price : ℕ := 3 * jeans_price
noncomputable def discounted_jacket_price : ℕ := jacket_price - (jacket_price / 10)

noncomputable def total_revenue : ℕ :=
  20 * shirt_price + 10 * jeans_price + 15 * discounted_jacket_price

theorem store_revenue_is_1210 :
  total_revenue = 1210 :=
by
  sorry

end store_revenue_is_1210_l79_79223


namespace find_amount_with_R_l79_79483

variable (P_amount Q_amount R_amount : ℝ)
variable (total_amount : ℝ) (r_has_twothirds : Prop)

noncomputable def amount_with_R (total_amount : ℝ) : ℝ :=
  let R_amount := 2 / 3 * (total_amount - R_amount)
  R_amount

theorem find_amount_with_R (P_amount Q_amount R_amount : ℝ) (total_amount : ℝ)
  (h_total : total_amount = 5000)
  (h_two_thirds : R_amount = 2 / 3 * (P_amount + Q_amount)) :
  R_amount = 2000 := by sorry

end find_amount_with_R_l79_79483


namespace card_arrangement_probability_l79_79088

/-- 
This problem considers the probability of arranging four distinct cards,
each labeled with a unique character, in such a way that they form one of two specific
sequences. Specifically, the sequences are "我爱数学" (I love mathematics) and "数学爱我" (mathematics loves me).
-/
theorem card_arrangement_probability :
  let cards := ["我", "爱", "数", "学"]
  let total_permutations := 24
  let favorable_outcomes := 2
  let probability := favorable_outcomes / total_permutations
  probability = 1 / 12 :=
by
  sorry

end card_arrangement_probability_l79_79088


namespace coefficient_x5y3_in_expansion_l79_79466

theorem coefficient_x5y3_in_expansion :
  let f (x y : ℤ) (n : ℕ) := (x^2 - x + 2 * y)^n
  f 1 1 6 = 64 →
  (binom 6 3) * (2^3) * (6 - 3) * (-3) = -480 :=
by
  intros f h
  have h1 : (1^2 - 1 + 2 * 1)^6 = 64 := h
  have h2 := binom 6 3 * 2^3 * (6 - 3) * (-3)
  sorry

end coefficient_x5y3_in_expansion_l79_79466


namespace shaded_region_equality_l79_79618

-- Define the necessary context and variables
variable {r : ℝ} -- radius of the circle
variable {θ : ℝ} -- angle measured in degrees

-- Define the relevant trigonometric functions
noncomputable def tan_degrees (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)
noncomputable def tan_half_degrees (x : ℝ) : ℝ := Real.tan ((x / 2) * Real.pi / 180)

-- State the theorem we need to prove given the conditions
theorem shaded_region_equality (hθ1 : θ / 2 = 90 - θ) :
  tan_degrees θ + (tan_degrees θ)^2 * tan_half_degrees θ = (θ * Real.pi) / 180 - (θ^2 * Real.pi) / 360 :=
  sorry

end shaded_region_equality_l79_79618


namespace solve_x_l79_79900

theorem solve_x (x : ℝ) (h1 : 8 * x^2 + 7 * x - 1 = 0) (h2 : 24 * x^2 + 53 * x - 7 = 0) : x = 1 / 8 :=
by
  sorry

end solve_x_l79_79900


namespace largest_pack_size_of_markers_l79_79424

theorem largest_pack_size_of_markers (markers_John markers_Alex : ℕ) (h_John : markers_John = 36) (h_Alex : markers_Alex = 60) : 
  ∃ (n : ℕ), (∀ (x : ℕ), (∀ (y : ℕ), (x * n = markers_John ∧ y * n = markers_Alex) → n ≤ 12) ∧ (12 * x = markers_John ∨ 12 * y = markers_Alex)) :=
by 
  sorry

end largest_pack_size_of_markers_l79_79424


namespace sum_arithmetic_sequence_l79_79657

def first_term (k : ℕ) : ℕ := k^2 - k + 1

def sum_of_first_k_plus_3_terms (k : ℕ) : ℕ := (k + 3) * (k^2 + (k / 2) + 2)

theorem sum_arithmetic_sequence (k : ℕ) (k_pos : 0 < k) : 
    sum_of_first_k_plus_3_terms k = k^3 + (7 * k^2) / 2 + (15 * k) / 2 + 6 := 
by
  sorry

end sum_arithmetic_sequence_l79_79657


namespace geometric_sequence_value_l79_79799

theorem geometric_sequence_value (a : ℝ) (h₁ : 280 ≠ 0) (h₂ : 35 ≠ 0) : 
  (∃ r : ℝ, 280 * r = a ∧ a * r = 35 / 8 ∧ a > 0) → a = 35 :=
by {
  sorry
}

end geometric_sequence_value_l79_79799


namespace simple_interest_rate_l79_79683

theorem simple_interest_rate (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 130) (h2 : P = 780) (h3 : T = 4) :
  R = 4.17 :=
sorry

end simple_interest_rate_l79_79683


namespace candy_bar_cost_l79_79935

/-- Problem statement:
Todd had 85 cents and spent 53 cents in total on a candy bar and a box of cookies.
The box of cookies cost 39 cents. How much did the candy bar cost? --/
theorem candy_bar_cost (t c s b : ℕ) (ht : t = 85) (hc : c = 39) (hs : s = 53) (h_total : s = b + c) : b = 14 :=
by
  sorry

end candy_bar_cost_l79_79935


namespace solve_for_x_l79_79645

theorem solve_for_x (x : ℝ) (h : x + 3 * x = 500 - (4 * x + 5 * x)) : x = 500 / 13 := 
by 
  sorry

end solve_for_x_l79_79645


namespace inequality_proof_l79_79165

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  sqrt (a^2 + b^2 - sqrt 2 * a * b) + sqrt (b^2 + c^2 - sqrt 2 * b * c)  ≥ sqrt (a^2 + c^2) :=
by sorry

end inequality_proof_l79_79165


namespace value_of_r_minus_p_l79_79680

-- Define the arithmetic mean conditions
def arithmetic_mean1 (p q : ℝ) : Prop :=
  (p + q) / 2 = 10

def arithmetic_mean2 (q r : ℝ) : Prop :=
  (q + r) / 2 = 27

-- Prove that r - p = 34 based on the conditions
theorem value_of_r_minus_p (p q r : ℝ)
  (h1 : arithmetic_mean1 p q)
  (h2 : arithmetic_mean2 q r) :
  r - p = 34 :=
by
  sorry

end value_of_r_minus_p_l79_79680


namespace exponent_identity_l79_79404

variable (x : ℝ) (m n : ℝ)
axiom h1 : x^m = 6
axiom h2 : x^n = 9

theorem exponent_identity : x^(2 * m - n) = 4 :=
by
  sorry

end exponent_identity_l79_79404


namespace perpendicular_lines_unique_a_l79_79114

open Real

theorem perpendicular_lines_unique_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, (a - 1) * x + y - 1 = 0) 
  (l2 : ∀ x y : ℝ, 3 * x + a * y + 2 = 0) 
  (perpendicular : True) : 
  a = 3 / 4 := 
sorry

end perpendicular_lines_unique_a_l79_79114


namespace event_A_probability_l79_79942

theorem event_A_probability (n : ℕ) (m₀ : ℕ) (H_n : n = 120) (H_m₀ : m₀ = 32) (p : ℝ) :
  (n * p - (1 - p) ≤ m₀) ∧ (n * p + p ≥ m₀) → 
  (32 / 121 : ℝ) ≤ p ∧ p ≤ (33 / 121 : ℝ) :=
sorry

end event_A_probability_l79_79942


namespace banana_arrangements_l79_79529

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l79_79529


namespace repeating_decimal_as_fraction_l79_79374

theorem repeating_decimal_as_fraction :
  ∃ x : ℚ, x = 6 / 10 + 7 / 90 ∧ x = 61 / 90 :=
by
  sorry

end repeating_decimal_as_fraction_l79_79374


namespace find_ks_l79_79721

def is_valid_function (f : ℕ → ℤ) (k : ℤ) : Prop :=
  ∀ x y : ℕ, f (x * y) = f x + f y + k * f (Nat.gcd x y)

theorem find_ks (f : ℕ → ℤ) :
  (f 2006 = 2007) →
  is_valid_function f k →
  k = 0 ∨ k = -1 :=
sorry

end find_ks_l79_79721


namespace total_surface_area_of_cuboid_l79_79594

variables (l w h : ℝ)
variables (lw_area wh_area lh_area : ℝ)

def box_conditions :=
  lw_area = l * w ∧
  wh_area = w * h ∧
  lh_area = l * h

theorem total_surface_area_of_cuboid (hc : box_conditions l w h 120 72 60) :
  2 * (120 + 72 + 60) = 504 :=
sorry

end total_surface_area_of_cuboid_l79_79594


namespace Mike_got_18_cards_l79_79773

theorem Mike_got_18_cards (original_cards : ℕ) (total_cards : ℕ) : 
  original_cards = 64 → total_cards = 82 → total_cards - original_cards = 18 :=
by
  intros h1 h2
  sorry

end Mike_got_18_cards_l79_79773


namespace minimum_broken_lines_cover_all_vertices_l79_79237

-- Define the grid dimensions
def gridWidth : ℕ := 100
def gridHeight : ℕ := 100

-- Define the properties of a shortest path from one corner to the other in the grid
def shortestPath(c1 c2 : ℕ × ℕ) : Prop :=
  c1.2 = 0 ∧ c2.1 = gridWidth ∧ ∀ i, (i < gridWidth → (i+1, i) = (i, i+1)) ∧

-- Define a function that checks if a vertex is covered by a path
def isVertexCovered (p : array (ℕ × ℕ)) (v : ℕ × ℕ) : Prop :=
  ∃ i, p.get? i = some v

-- Define the statement to be proven
theorem minimum_broken_lines_cover_all_vertices :
  ∃ n : ℕ, n = 101 ∧ ∀ v : ℕ × ℕ, v.1 ≤ gridWidth ∧ v.2 ≤ gridHeight → 
    (∃ paths : array (array (ℕ × ℕ)), paths.size ≤ n ∧ 
    ∀ path, path ∈ paths → shortestPath (0,0) (gridWidth, gridHeight) ∧
            ∀ pathVertex v, isVertexCovered path pathVertex →
            isVertexCovered pathVertex v) := by
  sorry

end minimum_broken_lines_cover_all_vertices_l79_79237


namespace pairs_sum_less_than_100_l79_79996

open Nat

def count_pairs_under_100 : ℕ :=
  card {p : ℕ × ℕ | p.1 < p.2 ∧ p.fst + p.snd < 100 ∧ p.1 ≤ 100 ∧ p.2 ≤ 100}

theorem pairs_sum_less_than_100 :
  count_pairs_under_100 = 2401 :=
sorry

end pairs_sum_less_than_100_l79_79996


namespace root_of_quadratic_l79_79441

theorem root_of_quadratic (a b c : ℝ) :
  (4 * a + 2 * b + c = 0) ↔ (a * 2^2 + b * 2 + c = 0) :=
by
  sorry

end root_of_quadratic_l79_79441


namespace find_a2_l79_79998

-- Definitions from conditions
def is_arithmetic_sequence (u : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, u (n + 1) = u n + d

def is_geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a2
  (u : ℕ → ℤ) (a1 a3 a4 : ℤ)
  (h1 : is_arithmetic_sequence u 3)
  (h2 : is_geometric_sequence a1 a3 a4)
  (h3 : a1 = u 1)
  (h4 : a3 = u 3)
  (h5 : a4 = u 4) :
  u 2 = -9 :=
by  
  sorry

end find_a2_l79_79998


namespace salary_after_cuts_l79_79912

noncomputable def finalSalary (init_salary : ℝ) (cuts : List ℝ) : ℝ :=
  cuts.foldl (λ salary cut => salary * (1 - cut)) init_salary

theorem salary_after_cuts :
  finalSalary 5000 [0.0525, 0.0975, 0.146, 0.128] = 3183.63 :=
by
  sorry

end salary_after_cuts_l79_79912


namespace c_minus_a_equals_90_l79_79207

variable (a b c : ℝ)

def average_a_b (a b : ℝ) : Prop := (a + b) / 2 = 45
def average_b_c (b c : ℝ) : Prop := (b + c) / 2 = 90

theorem c_minus_a_equals_90
  (h1 : average_a_b a b)
  (h2 : average_b_c b c) :
  c - a = 90 :=
  sorry

end c_minus_a_equals_90_l79_79207


namespace find_w_l79_79406

theorem find_w (a w : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * w) : w = 49 :=
by
  sorry

end find_w_l79_79406


namespace sum_first_third_numbers_l79_79189

theorem sum_first_third_numbers (A B C : ℕ)
    (h1 : A + B + C = 98)
    (h2 : A * 3 = B * 2)
    (h3 : B * 8 = C * 5)
    (h4 : B = 30) :
    A + C = 68 :=
by
-- Data is sufficient to conclude that A + C = 68
sorry

end sum_first_third_numbers_l79_79189


namespace compare_negative_fractions_l79_79507

theorem compare_negative_fractions :
  (-5 : ℝ) / 6 < (-4 : ℝ) / 5 :=
sorry

end compare_negative_fractions_l79_79507


namespace calculate_expression_l79_79506

theorem calculate_expression (m : ℝ) : (-m)^2 * m^5 = m^7 := 
sorry

end calculate_expression_l79_79506


namespace total_bulbs_needed_l79_79905

-- Definitions according to the conditions.
variables (T S M L XL : ℕ)

-- Conditions
variables (cond1 : L = 2 * M)
variables (cond2 : S = 5 * M / 4)  -- since 1.25M = 5/4M
variables (cond3 : XL = S - T)
variables (cond4 : 4 * T = 3 * M) -- equivalent to T / M = 3 / 4
variables (cond5 : 2 * S + 3 * M = 4 * L + 5 * XL)
variables (cond6 : XL = 14)

-- Prove total bulbs needed
theorem total_bulbs_needed :
  T + 2 * S + 3 * M + 4 * L + 5 * XL = 469 :=
sorry

end total_bulbs_needed_l79_79905


namespace onewaynia_road_closure_l79_79417

variable {V : Type} -- Denoting the type of cities
variable (G : V → V → Prop) -- G represents the directed graph

-- Conditions
variables (outdegree : V → Nat) (indegree : V → Nat)
variables (two_ways : ∀ (u v : V), u ≠ v → ¬(G u v ∧ G v u))
variables (two_out : ∀ v : V, outdegree v = 2)
variables (two_in : ∀ v : V, indegree v = 2)

theorem onewaynia_road_closure:
  ∃ n : Nat, n ≥ 1 ∧ (number_of_closures : Nat) = 2 ^ n :=
by
  sorry

end onewaynia_road_closure_l79_79417


namespace inner_rectangle_length_is_4_l79_79828

-- Define the conditions
def inner_rectangle_width : ℝ := 2
def shaded_region_width : ℝ := 2

-- Define the lengths and areas of the respective regions
def inner_rectangle_length (x : ℝ) : ℝ := x
def second_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 4, 6)
def largest_rectangle_dimensions (x : ℝ) : (ℝ × ℝ) := (x + 8, 10)

def inner_rectangle_area (x : ℝ) : ℝ := inner_rectangle_length x * inner_rectangle_width
def second_rectangle_area (x : ℝ) : ℝ := (second_rectangle_dimensions x).1 * (second_rectangle_dimensions x).2
def largest_rectangle_area (x : ℝ) : ℝ := (largest_rectangle_dimensions x).1 * (largest_rectangle_dimensions x).2

def first_shaded_region_area (x : ℝ) : ℝ := second_rectangle_area x - inner_rectangle_area x
def second_shaded_region_area (x : ℝ) : ℝ := largest_rectangle_area x - second_rectangle_area x

-- Define the arithmetic progression condition
def arithmetic_progression (x : ℝ) : Prop :=
  (first_shaded_region_area x - inner_rectangle_area x) = (second_shaded_region_area x - first_shaded_region_area x)

-- State the theorem
theorem inner_rectangle_length_is_4 :
  ∃ x : ℝ, arithmetic_progression x ∧ inner_rectangle_length x = 4 := 
by
  use 4
  -- Proof goes here
  sorry

end inner_rectangle_length_is_4_l79_79828


namespace eval_dollar_expr_l79_79854

noncomputable def dollar (k : ℝ) (a b : ℝ) := k * (a - b) ^ 2

theorem eval_dollar_expr (x y : ℝ) : dollar 3 ((2 * x - 3 * y) ^ 2) ((3 * y - 2 * x) ^ 2) = 0 :=
by sorry

end eval_dollar_expr_l79_79854


namespace perfect_square_expression_5_l79_79846

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def expression_1 : ℕ := 3^3 * 4^4 * 7^7
def expression_2 : ℕ := 3^4 * 4^3 * 7^6
def expression_3 : ℕ := 3^5 * 4^6 * 7^5
def expression_4 : ℕ := 3^6 * 4^5 * 7^4
def expression_5 : ℕ := 3^4 * 4^6 * 7^4

theorem perfect_square_expression_5 : is_perfect_square expression_5 :=
sorry

end perfect_square_expression_5_l79_79846


namespace blue_length_is_2_l79_79405

-- Define the lengths of the parts
def total_length : ℝ := 4
def purple_length : ℝ := 1.5
def black_length : ℝ := 0.5

-- Define the length of the blue part with the given conditions
def blue_length : ℝ := total_length - (purple_length + black_length)

-- State the theorem we need to prove
theorem blue_length_is_2 : blue_length = 2 :=
by 
  sorry

end blue_length_is_2_l79_79405


namespace trihedral_angle_sum_gt_180_l79_79781

theorem trihedral_angle_sum_gt_180
    (a' b' c' α β γ : ℝ)
    (Sabc : Prop)
    (h1 : b' = π - α)
    (h2 : c' = π - β)
    (h3 : a' = π - γ)
    (triangle_inequality : a' + b' + c' < 2 * π) :
    α + β + γ > π :=
by
  sorry

end trihedral_angle_sum_gt_180_l79_79781


namespace total_sequences_correct_l79_79324

/-- 
Given 6 blocks arranged such that:
1. Block 1 must be removed first.
2. Blocks 2 and 3 become accessible after Block 1 is removed.
3. Blocks 4, 5, and 6 become accessible after Blocks 2 and 3 are removed.
4. A block can only be removed if no other block is stacked on top of it. 

Prove that the total number of possible sequences to remove all the blocks is 10.
-/
def total_sequences_to_remove_blocks : ℕ := 10

theorem total_sequences_correct : 
  total_sequences_to_remove_blocks = 10 :=
sorry

end total_sequences_correct_l79_79324


namespace toby_sharing_proof_l79_79473

theorem toby_sharing_proof (initial_amt amount_left num_brothers : ℕ) 
(h_init : initial_amt = 343)
(h_left : amount_left = 245)
(h_bros : num_brothers = 2) : 
(initial_amt - amount_left) / (initial_amt * num_brothers) = 1 / 7 := 
sorry

end toby_sharing_proof_l79_79473


namespace total_salary_correct_l79_79882

-- Define the daily salaries
def owner_salary : ℕ := 20
def manager_salary : ℕ := 15
def cashier_salary : ℕ := 10
def clerk_salary : ℕ := 5
def bagger_salary : ℕ := 3

-- Define the number of employees
def num_owners : ℕ := 1
def num_managers : ℕ := 3
def num_cashiers : ℕ := 5
def num_clerks : ℕ := 7
def num_baggers : ℕ := 9

-- Define the total salary calculation
def total_daily_salary : ℕ :=
  (num_owners * owner_salary) +
  (num_managers * manager_salary) +
  (num_cashiers * cashier_salary) +
  (num_clerks * clerk_salary) +
  (num_baggers * bagger_salary)

-- The theorem we need to prove
theorem total_salary_correct :
  total_daily_salary = 177 :=
by
  -- Proof can be filled in later
  sorry

end total_salary_correct_l79_79882


namespace problem_l79_79634

variable (x : ℝ)

theorem problem (A B : ℝ) 
  (h : (A / (x - 3) + B * (x + 2) = (-5 * x^2 + 18 * x + 26) / (x - 3))): 
  A + B = 15 := by
  sorry

end problem_l79_79634


namespace geometric_sequence_b_l79_79049

theorem geometric_sequence_b (b : ℝ) (h1 : b > 0) (h2 : 30 * (b / 30) = b) (h3 : b * (b / 30) = 9 / 4) :
  b = 3 * Real.sqrt 30 / 2 :=
by
  sorry

end geometric_sequence_b_l79_79049


namespace correct_evaluation_at_3_l79_79835

noncomputable def polynomial (x : ℝ) : ℝ := 
  (4 * x^3 - 6 * x + 5) * (9 - 3 * x)

def expanded_poly (x : ℝ) : ℝ := 
  -12 * x^4 + 36 * x^3 + 18 * x^2 - 69 * x + 45

theorem correct_evaluation_at_3 :
  polynomial = expanded_poly →
  (12 * (-12) + 6 * 36 + 3 * 18 - 69) = 57 := 
by
  intro h
  sorry

end correct_evaluation_at_3_l79_79835


namespace ratio_right_to_left_l79_79351

theorem ratio_right_to_left (L C R : ℕ) (hL : L = 12) (hC : C = L + 2) (hTotal : L + C + R = 50) :
  R / L = 2 :=
by
  sorry

end ratio_right_to_left_l79_79351


namespace number_of_skew_line_pairs_in_cube_l79_79692

theorem number_of_skew_line_pairs_in_cube : 
  let vertices := 8
  let total_lines := 28
  let sets_of_4_points := Nat.choose 8 4 - 12
  let skew_pairs_per_set := 3
  let number_of_skew_pairs := sets_of_4_points * skew_pairs_per_set
  number_of_skew_pairs = 174 := sorry

end number_of_skew_line_pairs_in_cube_l79_79692


namespace speed_of_second_train_l79_79700

def speed_of_first_train := 40 -- speed of the first train in kmph
def distance_from_mumbai := 120 -- distance from Mumbai where the trains meet in km
def head_start_time := 1 -- head start time in hours for the first train
def total_remaining_distance := distance_from_mumbai - speed_of_first_train * head_start_time -- remaining distance for the first train to travel in km after head start
def time_to_meet_first_train := total_remaining_distance / speed_of_first_train -- time in hours for the first train to reach the meeting point after head start
def second_train_meeting_time := time_to_meet_first_train -- the second train takes the same time to meet the first train
def distance_covered_by_second_train := distance_from_mumbai -- same meeting point distance for second train from Mumbai

theorem speed_of_second_train : 
  ∃ v : ℝ, v = distance_covered_by_second_train / second_train_meeting_time ∧ v = 60 :=
by
  sorry

end speed_of_second_train_l79_79700


namespace necklace_stand_capacity_l79_79491

def necklace_stand_initial := 5
def ring_display_capacity := 30
def ring_display_current := 18
def bracelet_display_capacity := 15
def bracelet_display_current := 8
def cost_per_necklace := 4
def cost_per_ring := 10
def cost_per_bracelet := 5
def total_cost := 183

theorem necklace_stand_capacity : necklace_stand_current + (total_cost - (ring_display_capacity - ring_display_current) * cost_per_ring - (bracelet_display_capacity - bracelet_display_current) * cost_per_bracelet) / cost_per_necklace = 12 :=
by
  sorry

end necklace_stand_capacity_l79_79491


namespace banana_arrangements_l79_79533

theorem banana_arrangements : 
  let letters := "BANANA".toList
  let n := letters.length
  let countA := letters.count (fun c => c = 'A')
  let countN := letters.count (fun c => c = 'N')
  let countB := letters.count (fun c => c = 'B')
  n = 6 ∧ countA = 3 ∧ countN = 2 ∧ countB = 1 → (Nat.factorial n) / ((Nat.factorial countA) * (Nat.factorial countN) * (Nat.factorial countB)) = 60 := 
by
  intros letters n countA countN countB h
  sorry

end banana_arrangements_l79_79533


namespace substitutions_made_in_first_half_l79_79698

-- Definitions based on given problem conditions
def total_players : ℕ := 24
def starters : ℕ := 11
def non_players : ℕ := 7
def first_half_substitutions (S : ℕ) : ℕ := S
def second_half_substitutions (S : ℕ) : ℕ := 2 * S
def total_players_played (S : ℕ) := starters + first_half_substitutions S + second_half_substitutions S
def remaining_players : ℕ := total_players - non_players

-- Proof problem statement
theorem substitutions_made_in_first_half (S : ℕ) (h : total_players_played S = remaining_players) : S = 2 :=
by
  sorry

end substitutions_made_in_first_half_l79_79698


namespace find_a_l79_79280

variable (m n a : ℝ)
variable (h1 : m = 2 * n + 5)
variable (h2 : m + a = 2 * (n + 1.5) + 5)

theorem find_a : a = 3 := by
  sorry

end find_a_l79_79280


namespace min_ab_bound_l79_79440

theorem min_ab_bound (a b n : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hpos_n : 0 < n) 
                      (h : ∀ i j, i ≤ n → j ≤ n → Nat.gcd (a + i) (b + j) > 1) :
  ∃ c > 0, min a b > c^n * n^(n/2) :=
sorry

end min_ab_bound_l79_79440


namespace S13_is_52_l79_79320

variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
variable {n : ℕ}

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (a1 d : ℕ), ∀ n, a n = a1 + (n - 1) * d

-- Define the sum of the first n terms
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem S13_is_52 (h1 : is_arithmetic_sequence a)
                  (h2 : a 3 + a 7 + a 11 = 12)
                  (h3 : sum_of_first_n_terms S a) :
  S 13 = 52 :=
by sorry

end S13_is_52_l79_79320


namespace total_annual_gain_l79_79911

theorem total_annual_gain (x : ℝ) 
    (Lakshmi_share : ℝ) 
    (Lakshmi_share_eq: Lakshmi_share = 12000) : 
    (3 * Lakshmi_share = 36000) :=
by
  sorry

end total_annual_gain_l79_79911


namespace red_button_probability_l79_79284

theorem red_button_probability :
  let jarA_red := 6
  let jarA_blue := 9
  let jarA_total := jarA_red + jarA_blue
  let jarA_half := jarA_total / 2
  let removed_total := jarA_total - jarA_half
  let removed_red := removed_total / 2
  let removed_blue := removed_total / 2
  let jarA_red_remaining := jarA_red - removed_red
  let jarA_blue_remaining := jarA_blue - removed_blue
  let jarB_red := removed_red
  let jarB_blue := removed_blue
  let jarA_total_remaining := jarA_red_remaining + jarA_blue_remaining
  let jarB_total := jarB_red + jarB_blue
  (jarA_total = 15) →
  (jarA_red_remaining = 6 - removed_red) →
  (jarA_blue_remaining = 9 - removed_blue) →
  (jarB_red = removed_red) →
  (jarB_blue = removed_blue) →
  (jarA_red_remaining + jarA_blue_remaining = 9) →
  (jarB_red + jarB_blue = 6) →
  let prob_red_JarA := jarA_red_remaining / jarA_total_remaining
  let prob_red_JarB := jarB_red / jarB_total
  prob_red_JarA * prob_red_JarB = 1 / 6 := by sorry

end red_button_probability_l79_79284


namespace vector_calculation_l79_79598

def a :ℝ × ℝ := (1, 2)
def b :ℝ × ℝ := (1, -1)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem vector_calculation : scalar_mult (1/3) a - scalar_mult (4/3) b = (-1, 2) :=
by sorry

end vector_calculation_l79_79598


namespace blue_ball_higher_probability_l79_79219

noncomputable def probability_blue_ball_higher : ℝ :=
  let p (k : ℕ) : ℝ := 1 / (2^k : ℝ)
  let same_bin_prob := ∑' k : ℕ, (p (k + 1))^2
  let higher_prob := (1 - same_bin_prob) / 2
  higher_prob

theorem blue_ball_higher_probability :
  probability_blue_ball_higher = 1 / 3 :=
by
  sorry

end blue_ball_higher_probability_l79_79219


namespace pyramid_volume_l79_79782

theorem pyramid_volume 
(EF FG QE : ℝ) 
(base_area : ℝ) 
(volume : ℝ)
(h1 : EF = 10)
(h2 : FG = 5)
(h3 : base_area = EF * FG)
(h4 : QE = 9)
(h5 : volume = (1 / 3) * base_area * QE) : 
volume = 150 :=
by
  simp [h1, h2, h3, h4, h5]
  sorry

end pyramid_volume_l79_79782


namespace correct_assignment_l79_79831

-- Definition of conditions
def is_variable_free (e : String) : Prop := -- a simplistic placeholder
  e ∈ ["A", "B", "C", "D", "x"]

def valid_assignment (lhs : String) (rhs : String) : Prop :=
  is_variable_free lhs ∧ ¬(is_variable_free rhs)

-- The statement of the proof problem
theorem correct_assignment : valid_assignment "A" "A * A + A - 2" :=
by
  sorry

end correct_assignment_l79_79831


namespace find_other_root_l79_79117

-- Definitions based on conditions
def quadratic_equation (k : ℝ) (x : ℝ) : Prop := x^2 + 2 * k * x + k - 1 = 0

def is_root (k : ℝ) (x : ℝ) : Prop := quadratic_equation k x = true

-- The theorem to prove
theorem find_other_root (k x t: ℝ) (h₁ : is_root k 0) : t = -2 :=
sorry

end find_other_root_l79_79117


namespace additional_bureaus_needed_correct_l79_79686

-- The number of bureaus the company has
def total_bureaus : ℕ := 192

-- The number of offices
def total_offices : ℕ := 36

-- The additional bureaus needed to ensure each office gets an equal number
def additional_bureaus_needed (bureaus : ℕ) (offices : ℕ) : ℕ :=
  let bureaus_per_office := bureaus / offices
  let rounded_bureaus_per_office := bureaus_per_office + if bureaus % offices = 0 then 0 else 1
  let total_bureaus_needed := offices * rounded_bureaus_per_office
  total_bureaus_needed - bureaus

-- Problem Statement: Prove that at least 24 more bureaus are needed
theorem additional_bureaus_needed_correct : 
  additional_bureaus_needed total_bureaus total_offices = 24 := 
by
  sorry

end additional_bureaus_needed_correct_l79_79686


namespace face_opposite_A_l79_79818
noncomputable def cube_faces : List String := ["A", "B", "C", "D", "E", "F"]

theorem face_opposite_A (cube_faces : List String) 
  (h1 : cube_faces.length = 6)
  (h2 : "A" ∈ cube_faces) 
  (h3 : "B" ∈ cube_faces)
  (h4 : "C" ∈ cube_faces) 
  (h5 : "D" ∈ cube_faces)
  (h6 : "E" ∈ cube_faces) 
  (h7 : "F" ∈ cube_faces)
  : ("D" ≠ "A") := 
by
  sorry

end face_opposite_A_l79_79818


namespace not_divisible_by_n_plus_4_l79_79296

theorem not_divisible_by_n_plus_4 (n : ℕ) (h : n > 0) : ¬ ∃ k : ℕ, n^2 + 8 * n + 15 = k * (n + 4) := by
  sorry

end not_divisible_by_n_plus_4_l79_79296


namespace sequence_bound_l79_79151

noncomputable def sequenceProperties (a : ℕ → ℝ) (c : ℝ) : Prop :=
  (∀ i, 0 ≤ a i ∧ a i ≤ c) ∧ (∀ i j, i ≠ j → abs (a i - a j) ≥ 1 / (i + j))

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) (h : sequenceProperties a c) : 
  c ≥ 1 :=
by
  sorry

end sequence_bound_l79_79151


namespace total_soda_consumption_l79_79838

variables (c_soda b_soda c_consumed b_consumed b_remaining carol_final bob_final total_consumed : ℕ)

-- Define the conditions
def carol_soda_size : ℕ := 20
def bob_soda_25_percent_more : ℕ := carol_soda_size + carol_soda_size * 25 / 100
def carol_consumed : ℕ := carol_soda_size * 80 / 100
def bob_consumed : ℕ := bob_soda_25_percent_more * 80 / 100
def carol_remaining : ℕ := carol_soda_size - carol_consumed
def bob_remaining : ℕ := bob_soda_25_percent_more - bob_consumed
def bob_gives_carol : ℕ := bob_remaining / 2 + 3
def carol_final_consumption : ℕ := carol_consumed + bob_gives_carol
def bob_final_consumption : ℕ := bob_consumed - bob_gives_carol
def total_soda_consumed : ℕ := carol_final_consumption + bob_final_consumption

-- The theorem to prove the total amount of soda consumed by Carol and Bob together is 36 ounces
theorem total_soda_consumption : total_soda_consumed = 36 := by {
  sorry
}

end total_soda_consumption_l79_79838


namespace number_of_flowers_alissa_picked_l79_79082

-- Define the conditions
variable (A : ℕ) -- Number of flowers Alissa picked
variable (M : ℕ) -- Number of flowers Melissa picked
variable (flowers_gifted : ℕ := 18) -- Flowers given to mother
variable (flowers_left : ℕ := 14) -- Flowers left after gifting

-- Define that Melissa picked the same number of flowers as Alissa
axiom pick_equal : M = A

-- Define the total number of flowers they had initially
axiom total_flowers : 2 * A = flowers_gifted + flowers_left

-- Prove that Alissa picked 16 flowers
theorem number_of_flowers_alissa_picked : A = 16 := by
  -- Use placeholders for proof steps
  sorry

end number_of_flowers_alissa_picked_l79_79082


namespace birds_flew_up_l79_79819

-- Definitions based on conditions in the problem
def initial_birds : ℕ := 29
def new_total_birds : ℕ := 42

-- The statement to be proven
theorem birds_flew_up (x y z : ℕ) (h1 : x = initial_birds) (h2 : y = new_total_birds) (h3 : z = y - x) : z = 13 :=
by
  -- Proof will go here
  sorry

end birds_flew_up_l79_79819


namespace field_trip_count_l79_79823

theorem field_trip_count (vans: ℕ) (buses: ℕ) (people_per_van: ℕ) (people_per_bus: ℕ)
  (hv: vans = 9) (hb: buses = 10) (hpv: people_per_van = 8) (hpb: people_per_bus = 27):
  vans * people_per_van + buses * people_per_bus = 342 := by
  sorry

end field_trip_count_l79_79823


namespace line_through_center_eq_line_bisects_chord_eq_l79_79382

section Geometry

-- Define the circle C
def circleC (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 16

-- Define the point P
def P := (2, 2)

-- Define when line l passes through the center of the circle
def line_through_center (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Define when line l bisects chord AB by point P
def line_bisects_chord (x y : ℝ) : Prop := x + 2 * y - 6 = 0

-- Prove the equation of line l passing through the center
theorem line_through_center_eq : 
  (∀ (x y : ℝ), line_through_center x y → circleC x y → (x, y) = (1, 0)) →
  2 * (2:ℝ) - 2 - 2 = 0 := sorry

-- Prove the equation of line l bisects chord AB by point P
theorem line_bisects_chord_eq:
  (∀ (x y : ℝ), line_bisects_chord x y → circleC x y → (2, 2) = P) →
  (2 + 2 * 2 - 6 = 0) := sorry

end Geometry

end line_through_center_eq_line_bisects_chord_eq_l79_79382


namespace find_principal_amount_l79_79076

-- Definitions based on conditions
def A : ℝ := 3969
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The statement to be proved
theorem find_principal_amount : ∃ P : ℝ, A = P * (1 + r/n)^(n * t) ∧ P = 3600 :=
by
  use 3600
  sorry

end find_principal_amount_l79_79076


namespace find_missing_number_l79_79410

theorem find_missing_number :
  (12 + x + 42 + 78 + 104) / 5 = 62 →
  (a + 255 + 511 + 1023 + x) / 5 = 398.2 →
  a = 128 :=
by
  intros h1 h2
  sorry

end find_missing_number_l79_79410


namespace distance_between_houses_l79_79188

theorem distance_between_houses (d d_JS d_QS : ℝ) (h1 : d_JS = 3) (h2 : d_QS = 1) :
  (2 ≤ d ∧ d ≤ 4) → d = 3 :=
sorry

end distance_between_houses_l79_79188


namespace perm_banana_l79_79560

theorem perm_banana : 
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  (fact total_letters) / ((fact A_letters) * (fact N_letters) * fact (total_letters - A_letters - N_letters)) = 60 :=
by
  let total_letters := 6
  let A_letters := 3
  let N_letters := 2
  have h1 : fact total_letters = 720 := by decide
  have h2 : fact A_letters = 6 := by decide
  have h3 : fact N_letters = 2 := by decide
  have h4 : fact (total_letters - A_letters - N_letters) = fact (6 - 3 - 2) := by decide
  have h5 : fact (total_letters - A_letters - N_letters) = 1 := by decide
  calc
    (720 / (6 * 2 * 1) : ℝ)
    _ = 60 := by norm_num

end perm_banana_l79_79560


namespace minimum_value_inequality_l79_79156

theorem minimum_value_inequality (x y z : ℝ) (hx : 2 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 5) :
    (x - 2)^2 + (y / x - 2)^2 + (z / y - 2)^2 + (5 / z - 2)^2 ≥ 4 * (Real.sqrt (Real.sqrt 5) - 2)^2 := 
    sorry

end minimum_value_inequality_l79_79156


namespace complement_set_U_A_l79_79393

-- Definitions of U and A
def U : Set ℝ := { x : ℝ | x^2 ≤ 4 }
def A : Set ℝ := { x : ℝ | |x - 1| ≤ 1 }

-- Theorem statement
theorem complement_set_U_A : (U \ A) = { x : ℝ | -2 ≤ x ∧ x < 0 } := 
by
  sorry

end complement_set_U_A_l79_79393


namespace proof_problem1_proof_problem2_l79_79707

noncomputable def problem1_lhs : ℝ := 
  1 / (Real.sqrt 3 + 1) - Real.sin (Real.pi / 3) + Real.sqrt 32 * Real.sqrt (1 / 8)

noncomputable def problem1_rhs : ℝ := 3 / 2

theorem proof_problem1 : problem1_lhs = problem1_rhs :=
by 
  sorry

noncomputable def problem2_lhs : ℝ := 
  2^(-2 : ℤ) - Real.sqrt ((-2)^2) + 6 * Real.sin (Real.pi / 4) - Real.sqrt 18

noncomputable def problem2_rhs : ℝ := -7 / 4

theorem proof_problem2 : problem2_lhs = problem2_rhs :=
by 
  sorry

end proof_problem1_proof_problem2_l79_79707


namespace min_sum_of_perpendicular_sides_l79_79111

noncomputable def min_sum_perpendicular_sides (a b : ℝ) (h : a * b = 100) : ℝ :=
a + b

theorem min_sum_of_perpendicular_sides {a b : ℝ} (h : a * b = 100) : min_sum_perpendicular_sides a b h = 20 :=
sorry

end min_sum_of_perpendicular_sides_l79_79111


namespace sin_cos_theta_l79_79856

-- Define the problem conditions and the question as a Lean statement
theorem sin_cos_theta (θ : ℝ) (h : Real.tan (θ + Real.pi / 2) = 2) : Real.sin θ * Real.cos θ = -2 / 5 := by
  sorry

end sin_cos_theta_l79_79856


namespace minimum_value_of_sum_of_squares_l79_79263

noncomputable def minimum_of_sum_of_squares (a b : ℝ) : ℝ :=
  a^2 + b^2

theorem minimum_value_of_sum_of_squares (a b : ℝ) (h : |a * b| = 6) :
  a^2 + b^2 ≥ 12 :=
by {
  sorry
}

end minimum_value_of_sum_of_squares_l79_79263


namespace hiking_time_l79_79628

-- Define the conditions
def Distance : ℕ := 12
def Pace_up : ℕ := 4
def Pace_down : ℕ := 6

-- Statement to be proved
theorem hiking_time (d : ℕ) (pu : ℕ) (pd : ℕ) (h₁ : d = Distance) (h₂ : pu = Pace_up) (h₃ : pd = Pace_down) :
  d / pu + d / pd = 5 :=
by sorry

end hiking_time_l79_79628


namespace fraction_simplification_l79_79126

theorem fraction_simplification (a b : ℚ) (h : b / a = 3 / 5) : (a - b) / a = 2 / 5 :=
by
  sorry

end fraction_simplification_l79_79126


namespace simplify_and_evaluate_problem_l79_79038

noncomputable def problem_expression (a : ℤ) : ℚ :=
  (1 - (3 : ℚ) / (a + 1)) / ((a^2 - 4 * a + 4 : ℚ) / (a + 1))

theorem simplify_and_evaluate_problem :
  ∀ (a : ℤ), -2 ≤ a ∧ a ≤ 2 → a ≠ -1 → a ≠ 2 →
  (problem_expression a = 1 / (a - 2 : ℚ)) ∧
  (a = 0 → problem_expression a = -1 / 2) ∧
  (a = 1 → problem_expression a = -1) :=
sorry

end simplify_and_evaluate_problem_l79_79038


namespace remainder_of_trailing_zeros_l79_79895

def count_trailing_zeros (n : ℕ) : ℕ :=
  let rec helper (n : ℕ) (count : ℕ) : ℕ :=
    if n = 0 then count
    else helper (n / 10) (count + n % 10 = 0)
  helper n 0

def product_factorials (n : ℕ) : ℕ :=
  (List.range (n+1)).map (λ x, Nat.factorial x) |>.foldl (· * ·) 1

theorem remainder_of_trailing_zeros :
  let n := product_factorials 50 
  let m := count_trailing_zeros n
  m % 1000 = 702 := sorry

end remainder_of_trailing_zeros_l79_79895


namespace days_considered_l79_79969

theorem days_considered (visitors_current : ℕ) (visitors_previous : ℕ) (total_visitors : ℕ)
  (h1 : visitors_current = 132) (h2 : visitors_previous = 274) (h3 : total_visitors = 406)
  (h_total : visitors_current + visitors_previous = total_visitors) :
  2 = 2 :=
by
  sorry

end days_considered_l79_79969


namespace cubes_with_painted_faces_l79_79961

theorem cubes_with_painted_faces :
  ∀ (n : ℕ), (∃ (cubes : ℕ), cubes = 27 ∧ ∃ (face_painted_cubes : ℕ → ℕ), 
  face_painted_cubes 2 = 12) → 12 * 2 = 24 :=
by
  intro n
  assume h
  cases h with cubes hc
  cases hc with hc_painted hc_cubes
  simp at hc_painted
  simp at hc_cubes
  rw [hc_painted, hc_cubes]
  sorry

end cubes_with_painted_faces_l79_79961


namespace cost_to_fix_car_l79_79633

variable {S A : ℝ}

theorem cost_to_fix_car (h1 : A = 3 * S + 50) (h2 : S + A = 450) : A = 350 := 
by
  sorry

end cost_to_fix_car_l79_79633


namespace apples_in_basket_l79_79018

-- Define the conditions in Lean
def four_times_as_many_apples (O A : ℕ) : Prop :=
  A = 4 * O

def emiliano_consumes (O A : ℕ) : Prop :=
  (2/3 : ℚ) * O + (2/3 : ℚ) * A = 50

-- Formulate the main proposition to prove there are 60 apples
theorem apples_in_basket (O A : ℕ) (h1 : four_times_as_many_apples O A) (h2 : emiliano_consumes O A) : A = 60 := 
by
  sorry

end apples_in_basket_l79_79018


namespace johns_percentage_increase_l79_79482

theorem johns_percentage_increase (original_amount new_amount : ℕ) (h₀ : original_amount = 30) (h₁ : new_amount = 40) :
  (new_amount - original_amount) * 100 / original_amount = 33 :=
by
  sorry

end johns_percentage_increase_l79_79482


namespace marshmallow_per_smore_l79_79283

theorem marshmallow_per_smore (graham_crackers : ℕ) (initial_marshmallows : ℕ) (additional_marshmallows : ℕ) 
                               (graham_crackers_per_smore : ℕ) :
  graham_crackers = 48 ∧ initial_marshmallows = 6 ∧ additional_marshmallows = 18 ∧ graham_crackers_per_smore = 2 →
  (initial_marshmallows + additional_marshmallows) / (graham_crackers / graham_crackers_per_smore) = 1 :=
by
  intro h
  sorry

end marshmallow_per_smore_l79_79283


namespace total_pears_picked_l79_79761

theorem total_pears_picked (keith_pears jason_pears : ℕ) (h1 : keith_pears = 3) (h2 : jason_pears = 2) : keith_pears + jason_pears = 5 :=
by
  sorry

end total_pears_picked_l79_79761


namespace factor_poly_l79_79398

theorem factor_poly (P Q : ℝ) (h1 : ∃ b c : ℝ, 
  (x^2 + 3*x + 7) * (x^2 + b*x + c) = x^4 + P*x^2 + Q)
  : P + Q = 50 :=
sorry

end factor_poly_l79_79398


namespace common_factor_l79_79925

-- Definition of the polynomial
def polynomial (x y m n : ℝ) : ℝ := 4 * x * (m - n) + 2 * y * (m - n) ^ 2

-- The theorem statement
theorem common_factor (x y m n : ℝ) : ∃ k : ℝ, k * (m - n) = polynomial x y m n :=
sorry

end common_factor_l79_79925


namespace min_xy_when_a_16_min_expr_when_a_0_l79_79732

-- Problem 1: Minimum value of xy when a = 16
theorem min_xy_when_a_16 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y + 16) : 16 ≤ x * y :=
    sorry

-- Problem 2: Minimum value of x + y + 2 / x + 1 / (2 * y) when a = 0
theorem min_expr_when_a_0 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y) : (11 : ℝ) / 2 ≤ x + y + 2 / x + 1 / (2 * y) :=
    sorry

end min_xy_when_a_16_min_expr_when_a_0_l79_79732


namespace area_of_30_60_90_triangle_l79_79172

theorem area_of_30_60_90_triangle (altitude : ℝ) (h : altitude = 3) : 
  ∃ (area : ℝ), area = 6 * Real.sqrt 3 := 
sorry

end area_of_30_60_90_triangle_l79_79172


namespace diff_between_percent_and_fraction_l79_79804

theorem diff_between_percent_and_fraction :
  (0.75 * 800) - ((7 / 8) * 1200) = -450 :=
by
  sorry

end diff_between_percent_and_fraction_l79_79804


namespace prob_at_least_one_head_is_7_over_8_l79_79224

-- Define the event and probability calculation
def probability_of_tails_all_three_tosses : ℚ :=
  (1 / 2) ^ 3

def probability_of_at_least_one_head : ℚ :=
  1 - probability_of_tails_all_three_tosses

-- Prove the probability of at least one head is 7/8
theorem prob_at_least_one_head_is_7_over_8 : probability_of_at_least_one_head = 7 / 8 :=
by
  sorry

end prob_at_least_one_head_is_7_over_8_l79_79224


namespace correct_choice_D_l79_79386

variables {Ω : Type*} {P : ProbabilityMassFunction Ω}

def A : Set Ω := sorry -- define event A
def B : Set Ω := sorry -- define event B

theorem correct_choice_D (hA : P(A) = 0.2) (hB : P(B) = 0.8) (hInd : P(A ∩ B) = P(A) * P(B)) :
    P(A ∪ B) = 0.84 ∧ P(A ∩ B) = 0.16 :=
by
  rw hInd
  have hAB : P(A ∩ B) = 0.2 * 0.8, by rw [hA, hB]; ring
  have hUnion : P(A ∪ B) = P(A) + P(B) - P(A ∩ B), by sorry -- Add calculation steps as needed
  rw [hAB, hA, hB] at hUnion
  norm_num at hUnion
  split
  · exact hUnion
  · exact hAB

end correct_choice_D_l79_79386


namespace stocks_higher_price_l79_79438

theorem stocks_higher_price (total_stocks lower_price higher_price: ℝ)
  (h_total: total_stocks = 8000)
  (h_ratio: higher_price = 1.5 * lower_price)
  (h_sum: lower_price + higher_price = total_stocks) :
  higher_price = 4800 :=
by
  sorry

end stocks_higher_price_l79_79438


namespace people_speak_neither_l79_79748

-- Define the total number of people
def total_people : ℕ := 25

-- Define the number of people who can speak Latin
def speak_latin : ℕ := 13

-- Define the number of people who can speak French
def speak_french : ℕ := 15

-- Define the number of people who can speak both Latin and French
def speak_both : ℕ := 9

-- Prove that the number of people who don't speak either Latin or French is 6
theorem people_speak_neither : (total_people - (speak_latin + speak_french - speak_both)) = 6 := by
  sorry

end people_speak_neither_l79_79748


namespace probability_log_product_lt_zero_l79_79844

open Real

noncomputable def chosen_elements : Finset ℝ := {0.3, 0.5, 3.0, 4.0, 5.0, 6.0}

theorem probability_log_product_lt_zero : 
  let p := 3 / 5 in 
  ∃ a b c ∈ chosen_elements.to_list, 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  (log a) * (log b) * (log c) < 0 :=
by sorry

end probability_log_product_lt_zero_l79_79844


namespace car_distance_l79_79063

theorem car_distance (t : ℚ) (s : ℚ) (d : ℚ) 
(h1 : t = 2 + 2 / 5) 
(h2 : s = 260) 
(h3 : d = s * t) : 
d = 624 := by
  sorry

end car_distance_l79_79063


namespace simplify_and_evaluate_expression_l79_79302

theorem simplify_and_evaluate_expression (a b : ℝ) (h1 : a = -1) (h2 : a * b = 2) :
  3 * (2 * a^2 * b + a * b^2) - (3 * a * b^2 - a^2 * b) = -14 := by
  sorry

end simplify_and_evaluate_expression_l79_79302


namespace negate_even_condition_l79_79035

theorem negate_even_condition (a b c : ℤ) :
  (¬(∀ a b c : ℤ, ∃ x : ℚ, a * x^2 + b * x + c = 0 → Even a ∧ Even b ∧ Even c)) →
  (¬Even a ∨ ¬Even b ∨ ¬Even c) :=
by
  sorry

end negate_even_condition_l79_79035


namespace find_a_minus_inv_a_l79_79857

variable (a : ℝ)
variable (h : a + 1 / a = Real.sqrt 13)

theorem find_a_minus_inv_a : a - 1 / a = 3 ∨ a - 1 / a = -3 := by
  sorry

end find_a_minus_inv_a_l79_79857
