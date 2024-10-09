import Mathlib

namespace difference_between_q_and_r_l2380_238050

-- Define the variables for shares with respect to the common multiple x
def p_share (x : Nat) : Nat := 3 * x
def q_share (x : Nat) : Nat := 7 * x
def r_share (x : Nat) : Nat := 12 * x

-- Given condition: The difference between q's share and p's share is Rs. 4000
def condition_1 (x : Nat) : Prop := (q_share x - p_share x = 4000)

-- Define the theorem to prove the difference between r and q's share is Rs. 5000
theorem difference_between_q_and_r (x : Nat) (h : condition_1 x) : r_share x - q_share x = 5000 :=
by
  sorry

end difference_between_q_and_r_l2380_238050


namespace welders_correct_l2380_238074

-- Define the initial number of welders
def initial_welders := 12

-- Define the conditions:
-- 1. Total work is 1 job that welders can finish in 3 days.
-- 2. 9 welders leave after the first day.
-- 3. The remaining work is completed by (initial_welders - 9) in 8 days.

theorem welders_correct (W : ℕ) (h1 : W * 1/3 = 1) (h2 : (W - 9) * 8 = 2 * W) : 
  W = initial_welders :=
by
  sorry

end welders_correct_l2380_238074


namespace arithmetic_sequence_problem_l2380_238068

theorem arithmetic_sequence_problem 
  (a : ℕ → ℕ)
  (h_sequence : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_sum : a 3 + a 8 = 10) :
  3 * a 5 + a 7 = 20 :=
sorry

end arithmetic_sequence_problem_l2380_238068


namespace sum_S9_l2380_238045

variable (a : ℕ → ℤ)

-- Define the arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

-- Given condition for the sum of specific terms
def condition_given (a : ℕ → ℤ) : Prop :=
  a 2 + a 5 + a 8 = 12

-- Sum of the first 9 terms
def sum_of_first_nine_terms (a : ℕ → ℤ) : ℤ :=
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8

-- Problem statement: Prove that given the arithmetic sequence and the condition,
-- the sum of the first 9 terms is 36
theorem sum_S9 :
  arithmetic_sequence a → condition_given a → sum_of_first_nine_terms a = 36 :=
by
  intros
  sorry

end sum_S9_l2380_238045


namespace average_price_initial_l2380_238027

noncomputable def total_cost_initial (P : ℕ) := 5 * P
noncomputable def total_cost_remaining := 3 * 12
noncomputable def total_cost_returned := 2 * 32

theorem average_price_initial (P : ℕ) : total_cost_initial P = total_cost_remaining + total_cost_returned → P = 20 := 
by
  sorry

end average_price_initial_l2380_238027


namespace find_number_l2380_238083

theorem find_number :
  ∃ x : ℝ, (10 + x + 60) / 3 = (10 + 40 + 25) / 3 + 5 ∧ x = 20 :=
by
  sorry

end find_number_l2380_238083


namespace mow_lawn_time_l2380_238090

noncomputable def time_to_mow (length width swath_width overlap speed : ℝ) : ℝ :=
  let effective_swath := (swath_width - overlap) / 12 -- Convert inches to feet
  let strips_needed := width / effective_swath
  let total_distance := strips_needed * length
  total_distance / speed

theorem mow_lawn_time : time_to_mow 100 140 30 6 4500 = 1.6 :=
by
  sorry

end mow_lawn_time_l2380_238090


namespace find_salary_for_january_l2380_238014

-- Definitions based on problem conditions
variables (J F M A May : ℝ)
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (hMay : May = 6500)

-- Lean statement
theorem find_salary_for_january : J = 5700 :=
by {
  sorry
}

end find_salary_for_january_l2380_238014


namespace salt_solution_problem_l2380_238053

theorem salt_solution_problem
  (x y : ℝ)
  (h1 : 70 + x + y = 200)
  (h2 : 0.20 * 70 + 0.60 * x + 0.35 * y = 0.45 * 200) :
  x = 122 ∧ y = 8 :=
by
  sorry

end salt_solution_problem_l2380_238053


namespace Shiela_stars_per_bottle_l2380_238087

theorem Shiela_stars_per_bottle (total_stars : ℕ) (total_classmates : ℕ) (h1 : total_stars = 45) (h2 : total_classmates = 9) :
  total_stars / total_classmates = 5 := 
by 
  sorry

end Shiela_stars_per_bottle_l2380_238087


namespace inequality_relationship_l2380_238006

noncomputable def a := 1 / 2023
noncomputable def b := Real.exp (-2022 / 2023)
noncomputable def c := (Real.cos (1 / 2023)) / 2023

theorem inequality_relationship : b > a ∧ a > c :=
by
  -- Initializing and defining the variables
  let a := a
  let b := b
  let c := c
  -- Providing the required proof
  sorry

end inequality_relationship_l2380_238006


namespace inv_proportion_through_point_l2380_238024

theorem inv_proportion_through_point (m : ℝ) (x y : ℝ) (h1 : y = m / x) (h2 : x = 2) (h3 : y = -3) : m = -6 := by
  sorry

end inv_proportion_through_point_l2380_238024


namespace sqrt_four_eq_two_l2380_238059

theorem sqrt_four_eq_two : Real.sqrt 4 = 2 :=
by
  sorry

end sqrt_four_eq_two_l2380_238059


namespace conditional_probability_l2380_238064

def prob_event_A : ℚ := 7 / 8 -- Probability of event A (at least one occurrence of tails)
def prob_event_AB : ℚ := 3 / 8 -- Probability of both events A and B happening (at least one occurrence of tails and exactly one occurrence of heads)

theorem conditional_probability (prob_A : ℚ) (prob_AB : ℚ) 
  (h1: prob_A = 7 / 8) (h2: prob_AB = 3 / 8) : 
  (prob_AB / prob_A) = 3 / 7 := 
by
  rw [h1, h2]
  norm_num

end conditional_probability_l2380_238064


namespace quadratic_real_roots_quadratic_product_of_roots_l2380_238035

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 - 2 * m * x + m^2 + m - 3 = 0) ↔ m ≤ 3 := by
{
  sorry
}

theorem quadratic_product_of_roots (m : ℝ) (α β : ℝ) :
  α * β = 17 ∧ α^2 - 2 * m * α + m^2 + m - 3 = 0 ∧ β^2 - 2 * m * β + m^2 + m - 3 = 0 →
  m = -5 := by
{
  sorry
}

end quadratic_real_roots_quadratic_product_of_roots_l2380_238035


namespace largest_angle_triangle_l2380_238063

-- Definition of constants and conditions
def right_angle : ℝ := 90
def angle_sum : ℝ := 120
def angle_difference : ℝ := 20

-- Given two angles of a triangle sum to 120 degrees and one is 20 degrees greater than the other,
-- Prove the largest angle in the triangle is 70 degrees
theorem largest_angle_triangle (A B C : ℝ) (hA : A + B = angle_sum) (hB : B = A + angle_difference) (hC : A + B + C = 180) : 
  max A (max B C) = 70 := 
by 
  sorry

end largest_angle_triangle_l2380_238063


namespace solve_eq_roots_l2380_238067

noncomputable def solve_equation (x : ℝ) : Prop :=
  (7 * x + 2) / (3 * x^2 + 7 * x - 6) = (3 * x) / (3 * x - 2)

theorem solve_eq_roots (x : ℝ) (h₁ : x ≠ 2 / 3) :
  solve_equation x ↔ (x = (-1 + Real.sqrt 7) / 3 ∨ x = (-1 - Real.sqrt 7) / 3) :=
by
  sorry

end solve_eq_roots_l2380_238067


namespace exists_n_for_all_k_l2380_238080

theorem exists_n_for_all_k (k : ℕ) : ∃ n : ℕ, 5^k ∣ (n^2 + 1) :=
sorry

end exists_n_for_all_k_l2380_238080


namespace power_sum_is_99_l2380_238000

theorem power_sum_is_99 : 3^4 + (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 99 :=
by sorry

end power_sum_is_99_l2380_238000


namespace cost_price_of_article_l2380_238022

theorem cost_price_of_article (x : ℝ) (h : 66 - x = x - 22) : x = 44 :=
sorry

end cost_price_of_article_l2380_238022


namespace complement_of_angle_l2380_238005

theorem complement_of_angle (A : ℝ) (hA : A = 35) : 180 - A = 145 := by
  sorry

end complement_of_angle_l2380_238005


namespace area_of_rectangular_field_l2380_238043

theorem area_of_rectangular_field (L W A : ℕ) (h1 : L = 10) (h2 : 2 * W + L = 130) :
  A = 600 :=
by
  -- Proof will go here
  sorry

end area_of_rectangular_field_l2380_238043


namespace original_height_l2380_238028

theorem original_height (total_travel : ℝ) (h : ℝ) (half: h/2 = (1/2 * h)): 
  (total_travel = h + 2 * (h / 2) + 2 * (h / 4)) → total_travel = 260 → h = 104 :=
by
  intro travel_eq
  intro travel_value
  sorry

end original_height_l2380_238028


namespace sum_of_primes_final_sum_l2380_238095

theorem sum_of_primes (p : ℕ) (hp : Nat.Prime p) :
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) →
  p = 2 ∨ p = 5 :=
sorry

theorem final_sum :
  (∀ p : ℕ, Nat.Prime p → (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) → p = 2 ∨ p = 5) →
  (2 + 5 = 7) :=
sorry

end sum_of_primes_final_sum_l2380_238095


namespace one_thirds_in_nine_thirds_l2380_238058

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l2380_238058


namespace determine_a_l2380_238007

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end determine_a_l2380_238007


namespace find_y_l2380_238015

theorem find_y (x y : ℝ) (h1 : 2 * (x - y) = 12) (h2 : x + y = 14) : y = 4 := 
by
  sorry

end find_y_l2380_238015


namespace max_correct_answers_l2380_238048

/--
In a 50-question multiple-choice math contest, students receive 5 points for a correct answer, 
0 points for an answer left blank, and -2 points for an incorrect answer. Jesse’s total score 
on the contest was 115. Prove that the maximum number of questions that Jesse could have answered 
correctly is 30.
-/
theorem max_correct_answers (a b c : ℕ) (h1 : a + b + c = 50) (h2 : 5 * a - 2 * c = 115) : a ≤ 30 :=
by
  sorry

end max_correct_answers_l2380_238048


namespace cat_and_mouse_positions_after_317_moves_l2380_238042

-- Define the conditions of the problem
def cat_positions : List String := ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
def mouse_positions : List String := ["Top Left", "Top Middle", "Top Right", "Right Middle", "Bottom Right", "Bottom Middle", "Bottom Left", "Left Middle"]

-- Calculate the position of the cat after n moves
def cat_position_after_moves (n : Nat) : String :=
  cat_positions.get! (n % 4)

-- Calculate the position of the mouse after n moves
def mouse_position_after_moves (n : Nat) : String :=
  mouse_positions.get! (n % 8)

-- Prove the final positions of the cat and mouse after 317 moves
theorem cat_and_mouse_positions_after_317_moves :
  cat_position_after_moves 317 = "Top Left" ∧ mouse_position_after_moves 317 = "Bottom Middle" :=
by
  sorry

end cat_and_mouse_positions_after_317_moves_l2380_238042


namespace five_minus_x_eight_l2380_238069

theorem five_minus_x_eight (x y : ℤ) (h1 : 5 + x = 3 - y) (h2 : 2 + y = 6 + x) : 5 - x = 8 :=
by
  sorry

end five_minus_x_eight_l2380_238069


namespace geometric_sequence_sum_5_is_75_l2380_238049

noncomputable def geometric_sequence_sum_5 (a r : ℝ) : ℝ :=
  a * (1 + r + r^2 + r^3 + r^4)

theorem geometric_sequence_sum_5_is_75 (a r : ℝ)
  (h1 : a * (1 + r + r^2) = 13)
  (h2 : a * (1 - r^7) / (1 - r) = 183) :
  geometric_sequence_sum_5 a r = 75 :=
sorry

end geometric_sequence_sum_5_is_75_l2380_238049


namespace investment_growth_theorem_l2380_238099

variable (x : ℝ)

-- Defining the initial and final investments
def initial_investment : ℝ := 800
def final_investment : ℝ := 960

-- Defining the growth equation
def growth_equation (x : ℝ) : Prop := initial_investment * (1 + x) ^ 2 = final_investment

-- The theorem statement that needs to be proven
theorem investment_growth_theorem : growth_equation x := sorry

end investment_growth_theorem_l2380_238099


namespace xyz_sum_48_l2380_238071

theorem xyz_sum_48 (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x * y + z = 47) (h2 : y * z + x = 47) (h3 : z * x + y = 47) : 
  x + y + z = 48 :=
sorry

end xyz_sum_48_l2380_238071


namespace smallest_k_l2380_238047

theorem smallest_k (p : ℕ) (hp : p = 997) : 
  ∃ k : ℕ, (p^2 - k) % 10 = 0 ∧ k = 9 :=
by
  sorry

end smallest_k_l2380_238047


namespace find_k_l2380_238001

theorem find_k 
  (c : ℝ) (a₁ : ℝ) (S : ℕ → ℝ) (k : ℝ)
  (h1 : ∀ n, S (n+1) = c * S n) 
  (h2 : S 1 = 3 + k)
  (h3 : ∀ n, S n = 3^n + k) :
  k = -1 :=
sorry

end find_k_l2380_238001


namespace option_c_is_always_odd_l2380_238016

theorem option_c_is_always_odd (n : ℤ) : ∃ (q : ℤ), n^2 + n + 5 = 2*q + 1 := by
  sorry

end option_c_is_always_odd_l2380_238016


namespace smallest_positive_integer_23n_mod_5678_mod_11_l2380_238057

theorem smallest_positive_integer_23n_mod_5678_mod_11 :
  ∃ n : ℕ, 0 < n ∧ 23 * n % 11 = 5678 % 11 ∧ ∀ m : ℕ, 0 < m ∧ 23 * m % 11 = 5678 % 11 → n ≤ m :=
by
  sorry

end smallest_positive_integer_23n_mod_5678_mod_11_l2380_238057


namespace scientific_notation_five_hundred_billion_l2380_238093

theorem scientific_notation_five_hundred_billion :
  500000000000 = 5 * 10^11 := by
  sorry

end scientific_notation_five_hundred_billion_l2380_238093


namespace find_p7_value_l2380_238034

def quadratic (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem find_p7_value (d e f : ℝ)
  (h1 : quadratic d e f 1 = 4)
  (h2 : quadratic d e f 2 = 4) :
  quadratic d e f 7 = 5 := by
  sorry

end find_p7_value_l2380_238034


namespace exact_value_range_l2380_238040

theorem exact_value_range (a : ℝ) (h : |170 - a| < 0.5) : 169.5 ≤ a ∧ a < 170.5 :=
by
  sorry

end exact_value_range_l2380_238040


namespace width_of_rectangle_l2380_238054

-- Define the given values
def length : ℝ := 2
def area : ℝ := 8

-- State the theorem
theorem width_of_rectangle : ∃ width : ℝ, area = length * width ∧ width = 4 :=
by
  -- The proof is omitted
  sorry

end width_of_rectangle_l2380_238054


namespace oil_needed_to_half_fill_tanker_l2380_238073

theorem oil_needed_to_half_fill_tanker :
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  let current_tanker_oil := initial_tanker_oil + poured_oil
  let half_tanker_capacity := initial_tanker_capacity / 2
  let needed_oil := half_tanker_capacity - current_tanker_oil
  needed_oil = 4000 :=
by
  let initial_tank_capacity := 4000
  let poured_fraction := 3 / 4
  let initial_tanker_capacity := 20000
  let initial_tanker_oil := 3000
  let poured_oil := poured_fraction * initial_tank_capacity
  have h1 : poured_oil = 3000 := by sorry
  let current_tanker_oil := initial_tanker_oil + poured_oil
  have h2 : current_tanker_oil = 6000 := by sorry
  let half_tanker_capacity := initial_tanker_capacity / 2
  have h3 : half_tanker_capacity = 10000 := by sorry
  let needed_oil := half_tanker_capacity - current_tanker_oil
  have h4 : needed_oil = 4000 := by sorry
  exact h4

end oil_needed_to_half_fill_tanker_l2380_238073


namespace power_sum_l2380_238055

theorem power_sum : (-2) ^ 2007 + (-2) ^ 2008 = 2 ^ 2007 := by
  sorry

end power_sum_l2380_238055


namespace tom_roses_per_day_l2380_238072

-- Define variables and conditions
def total_roses := 168
def days_in_week := 7
def dozen := 12

-- Theorem to prove
theorem tom_roses_per_day : (total_roses / dozen) / days_in_week = 2 :=
by
  -- The actual proof would go here, using the sorry placeholder
  sorry

end tom_roses_per_day_l2380_238072


namespace ratio_R_U_l2380_238056

theorem ratio_R_U : 
  let spacing := 1 / 4
  let R := 3 * spacing
  let U := 6 * spacing
  R / U = 0.5 := 
by
  sorry

end ratio_R_U_l2380_238056


namespace distance_between_towns_l2380_238018

theorem distance_between_towns 
  (x : ℝ) 
  (h1 : x / 100 - x / 110 = 0.15) : 
  x = 165 := 
by 
  sorry

end distance_between_towns_l2380_238018


namespace area_DEF_l2380_238098

structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -3, y := 4}
def E : Point := {x := 1, y := 7}
def F : Point := {x := 3, y := -1}

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)|

theorem area_DEF : area_of_triangle D E F = 16 := by
  sorry

end area_DEF_l2380_238098


namespace det_A_eq_l2380_238026

open Matrix

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, -3, 3],
    ![x, 5, -1],
    ![4, -2, 1]]

theorem det_A_eq (x : ℝ) : det (A x) = -3 * x - 45 :=
by sorry

end det_A_eq_l2380_238026


namespace original_set_cardinality_l2380_238025

-- Definitions based on conditions
def is_reversed_error (n : ℕ) : Prop :=
  ∃ (A B C : ℕ), 100 * A + 10 * B + C = n ∧ 100 * C + 10 * B + A = n + 198 ∧ C - A = 2

-- The theorem to prove
theorem original_set_cardinality : ∃ n : ℕ, is_reversed_error n ∧ n = 10 := by
  sorry

end original_set_cardinality_l2380_238025


namespace calculate_8b_l2380_238052

-- Define the conditions \(6a + 3b = 0\), \(b - 3 = a\), and \(b + c = 5\)
variables (a b c : ℝ)

theorem calculate_8b :
  (6 * a + 3 * b = 0) → (b - 3 = a) → (b + c = 5) → (8 * b = 16) :=
by
  intros h1 h2 h3
  -- Proof goes here, but we will use sorry to skip the proof.
  sorry

end calculate_8b_l2380_238052


namespace cylinder_surface_area_l2380_238061

theorem cylinder_surface_area (h : ℝ) (c : ℝ) (r : ℝ) 
  (h_eq : h = 2) (c_eq : c = 2 * Real.pi) (circumference_formula : c = 2 * Real.pi * r) : 
  2 * (Real.pi * r^2) + (2 * Real.pi * r * h) = 6 * Real.pi := 
by
  sorry

end cylinder_surface_area_l2380_238061


namespace sea_star_collection_l2380_238036

theorem sea_star_collection (S : ℕ) (initial_seashells : ℕ) (initial_snails : ℕ) (lost_sea_creatures : ℕ) (remaining_items : ℕ) :
  initial_seashells = 21 →
  initial_snails = 29 →
  lost_sea_creatures = 25 →
  remaining_items = 59 →
  S + initial_seashells + initial_snails = remaining_items + lost_sea_creatures →
  S = 34 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  sorry

end sea_star_collection_l2380_238036


namespace scientific_notation_example_l2380_238065

theorem scientific_notation_example : 3790000 = 3.79 * 10^6 := 
sorry

end scientific_notation_example_l2380_238065


namespace problem_statement_l2380_238031

theorem problem_statement (m n : ℝ) (h1 : 1 + 27 = m) (h2 : 3 + 9 = n) : |m - n| = 16 := by
  sorry

end problem_statement_l2380_238031


namespace water_outflow_time_l2380_238084

theorem water_outflow_time (H R : ℝ) (flow_rate : ℝ → ℝ)
  (h_initial : ℝ) (t_initial : ℝ) (empty_height : ℝ) :
  H = 12 →
  R = 3 →
  (∀ h, flow_rate h = -h) →
  h_initial = 12 →
  t_initial = 0 →
  empty_height = 0 →
  ∃ t, t = (72 : ℝ) * π / 16 :=
by
  intros hL R_eq flow_rate_eq h_initial_eq t_initial_eq empty_height_eq
  sorry

end water_outflow_time_l2380_238084


namespace total_flags_l2380_238013

theorem total_flags (x : ℕ) (hx1 : 4 * x + 20 > 8 * (x - 1)) (hx2 : 4 * x + 20 < 8 * x) : 4 * 6 + 20 = 44 :=
by sorry

end total_flags_l2380_238013


namespace slope_and_intercept_of_line_l2380_238097

theorem slope_and_intercept_of_line :
  ∀ (x y : ℝ), 3 * x + 2 * y + 6 = 0 → y = - (3 / 2) * x - 3 :=
by
  intros x y h
  sorry

end slope_and_intercept_of_line_l2380_238097


namespace contrapositive_of_real_roots_l2380_238010

variable {a : ℝ}

theorem contrapositive_of_real_roots :
  (1 + 4 * a < 0) → (a < 0) := by
  sorry

end contrapositive_of_real_roots_l2380_238010


namespace probability_sum_six_two_dice_l2380_238086

theorem probability_sum_six_two_dice :
  let total_outcomes := 36
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes = 5 / 36 := by
  let total_outcomes := 6 * 6
  let favorable_outcomes := 5
  sorry

end probability_sum_six_two_dice_l2380_238086


namespace SeedMixtureWeights_l2380_238081

theorem SeedMixtureWeights (x y z : ℝ) (h1 : x + y + z = 8) (h2 : x / 3 = y / 2) (h3 : x / 3 = z / 3) :
  x = 3 ∧ y = 2 ∧ z = 3 :=
by
  sorry

end SeedMixtureWeights_l2380_238081


namespace solve_basketball_points_l2380_238029

noncomputable def y_points_other_members (x : ℕ) : ℕ :=
  let d_points := (1 / 3) * x
  let e_points := (3 / 8) * x
  let f_points := 18
  let total := x
  total - d_points - e_points - f_points

theorem solve_basketball_points (x : ℕ) (h1: x > 0) (h2: ∃ y ≤ 24, y = y_points_other_members x) :
  ∃ y, y = 21 :=
by
  sorry

end solve_basketball_points_l2380_238029


namespace jerry_added_action_figures_l2380_238094

theorem jerry_added_action_figures (x : ℕ) (h1 : 7 + x - 10 = 8) : x = 11 :=
by
  sorry

end jerry_added_action_figures_l2380_238094


namespace range_of_m_l2380_238060

noncomputable def f (a x : ℝ) : ℝ := a * x - (2 * a + 1) / x

theorem range_of_m (a m : ℝ) (h₀ : a > 0) (h₁ : f a (m^2 + 1) > f a (m^2 - m + 3)) 
  : m > 2 :=
sorry

end range_of_m_l2380_238060


namespace middle_digit_base7_l2380_238037

theorem middle_digit_base7 (a b c : ℕ) 
  (h1 : N = 49 * a + 7 * b + c) 
  (h2 : N = 81 * c + 9 * b + a)
  (h3 : a < 7 ∧ b < 7 ∧ c < 7) : 
  b = 0 :=
by sorry

end middle_digit_base7_l2380_238037


namespace find_a_2_find_a_n_l2380_238078

-- Define the problem conditions and questions as types
def S_3 (a_1 a_2 a_3 : ℝ) : Prop := a_1 + a_2 + a_3 = 7
def arithmetic_mean_condition (a_1 a_2 a_3 : ℝ) : Prop :=
  (a_1 + 3 + a_3 + 4) / 2 = 3 * a_2

-- Prove that a_2 = 2 given the conditions
theorem find_a_2 (a_1 a_2 a_3 : ℝ) (h1 : S_3 a_1 a_2 a_3) (h2: arithmetic_mean_condition a_1 a_2 a_3) :
  a_2 = 2 := 
sorry

-- Define the general term for a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- Prove the formula for the general term of the geometric sequence given the conditions and a_2 found
theorem find_a_n (a : ℕ → ℝ) (q : ℝ) (h1 : S_3 (a 1) (a 2) (a 3)) (h2 : arithmetic_mean_condition (a 1) (a 2) (a 3)) (h3 : geometric_sequence a q) : 
  (q = (1/2) → ∀ n, a n = (1 / 2)^(n - 3))
  ∧ (q = 2 → ∀ n, a n = 2^(n - 1)) := 
sorry

end find_a_2_find_a_n_l2380_238078


namespace value_of_expr_l2380_238062

theorem value_of_expr (a b c d : ℝ) (h1 : a = 3 * b) (h2 : b = 3 * c) (h3 : c = 5 * d) : 
  a * c / (b * d) = 15 := 
by
  sorry

end value_of_expr_l2380_238062


namespace intersection_M_N_l2380_238021

def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l2380_238021


namespace find_n_on_angle_bisector_l2380_238011

theorem find_n_on_angle_bisector (M : ℝ × ℝ) (hM : M = (3 * n - 2, 2 * n + 7) ∧ M.1 + M.2 = 0) : 
    n = -1 :=
by
  sorry

end find_n_on_angle_bisector_l2380_238011


namespace solution_eq_l2380_238020

theorem solution_eq (a x : ℚ) :
  (2 * (x - 2 * (x - a / 4)) = 3 * x) ∧ ((x + a) / 9 - (1 - 3 * x) / 12 = 1) → 
  a = 65 / 11 ∧ x = 13 / 11 :=
by
  sorry

end solution_eq_l2380_238020


namespace angles_sum_540_l2380_238075

theorem angles_sum_540 (p q r s : ℝ) (h1 : ∀ a, a + (180 - a) = 180)
  (h2 : ∀ a b, (180 - a) + (180 - b) = 360 - a - b)
  (h3 : ∀ p q r, (360 - p - q) + (180 - r) = 540 - p - q - r) :
  p + q + r + s = 540 :=
sorry

end angles_sum_540_l2380_238075


namespace intersection_of_M_and_N_l2380_238009

def M : Set ℝ := { x | |x + 1| ≤ 1}

def N : Set ℝ := {-1, 0, 1}

theorem intersection_of_M_and_N : M ∩ N = {-1, 0} :=
by
  sorry

end intersection_of_M_and_N_l2380_238009


namespace correct_assignment_statement_l2380_238076

-- Definitions according to the problem conditions
def input_statement (x : Nat) : Prop := x = 3
def assignment_statement1 (A B : Nat) : Prop := A = B ∧ B = 2
def assignment_statement2 (T : Nat) : Prop := T = T * T
def output_statement (A : Nat) : Prop := A = 4

-- Lean statement for the problem. We need to prove that the assignment_statement2 is correct.
theorem correct_assignment_statement (T : Nat) : assignment_statement2 T :=
by sorry

end correct_assignment_statement_l2380_238076


namespace missed_field_goals_l2380_238046

theorem missed_field_goals (TotalAttempts MissedFraction WideRightPercentage : ℕ) 
  (TotalAttempts_eq : TotalAttempts = 60)
  (MissedFraction_eq : MissedFraction = 15)
  (WideRightPercentage_eq : WideRightPercentage = 3) : 
  (TotalAttempts * (1 / 4) * (20 / 100) = 3) :=
  by
    sorry

end missed_field_goals_l2380_238046


namespace number_of_true_propositions_l2380_238032

noncomputable def f : ℝ → ℝ := sorry -- since it's not specified, we use sorry here

-- Definitions for the conditions
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Original proposition
def original_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, is_odd f → f 0 = 0

-- Converse proposition
def converse_proposition (f : ℝ → ℝ) :=
  f 0 = 0 → ∀ x : ℝ, is_odd f

-- Inverse proposition (logically equivalent to the converse)
def inverse_proposition (f : ℝ → ℝ) :=
  ∀ x : ℝ, ¬(is_odd f) → f 0 ≠ 0

-- Contrapositive proposition (logically equivalent to the original)
def contrapositive_proposition (f : ℝ → ℝ) :=
  f 0 ≠ 0 → ∀ x : ℝ, ¬(is_odd f)

-- Theorem statement
theorem number_of_true_propositions (f : ℝ → ℝ) :
  (original_proposition f → true) ∧
  (converse_proposition f → false) ∧
  (inverse_proposition f → false) ∧
  (contrapositive_proposition f → true) →
  2 = 2 := 
by 
  sorry -- proof to be inserted

end number_of_true_propositions_l2380_238032


namespace proof_of_ratio_l2380_238070

def f (x : ℤ) : ℤ := 3 * x + 4

def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_of_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 151 / 121 :=
by
  sorry

end proof_of_ratio_l2380_238070


namespace mechanical_moles_l2380_238019

-- Define the conditions
def condition_one (x y : ℝ) : Prop :=
  x + y = 1 / 5

def condition_two (x y : ℝ) : Prop :=
  (1 / (3 * x)) + (2 / (3 * y)) = 10

-- Define the main theorem using the defined conditions
theorem mechanical_moles (x y : ℝ) (h1 : condition_one x y) (h2 : condition_two x y) :
  x = 1 / 30 ∧ y = 1 / 6 :=
  sorry

end mechanical_moles_l2380_238019


namespace prob_draw_l2380_238033

theorem prob_draw (p_not_losing p_winning p_drawing : ℝ) (h1 : p_not_losing = 0.6) (h2 : p_winning = 0.5) :
  p_drawing = 0.1 :=
by
  sorry

end prob_draw_l2380_238033


namespace range_of_a_l2380_238008

variable (a x y : ℝ)

def proposition_p : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q : Prop :=
  (1 - a) * (a - 3) < 0

theorem range_of_a (h1 : proposition_p a) (h2 : proposition_q a) : 
  (0 ≤ a ∧ a < 1) ∨ (3 < a ∧ a < 4) :=
by
  sorry

end range_of_a_l2380_238008


namespace option_C_correct_l2380_238030

theorem option_C_correct (a : ℤ) : (a = 3 → a = a + 1 → a = 4) :=
by {
  sorry
}

end option_C_correct_l2380_238030


namespace condition1_a_geq_1_l2380_238077

theorem condition1_a_geq_1 (a : ℝ) :
  (∀ x ∈ ({1, 2, 3} : Set ℝ), a * x - 1 ≥ 0) → a ≥ 1 :=
by
sorry

end condition1_a_geq_1_l2380_238077


namespace shaded_areas_are_different_l2380_238096

theorem shaded_areas_are_different :
  let shaded_area_I := 3 / 8
  let shaded_area_II := 1 / 3
  let shaded_area_III := 1 / 2
  (shaded_area_I ≠ shaded_area_II) ∧ (shaded_area_I ≠ shaded_area_III) ∧ (shaded_area_II ≠ shaded_area_III) :=
by
  sorry

end shaded_areas_are_different_l2380_238096


namespace find_divided_number_l2380_238089

theorem find_divided_number:
  ∃ x : ℕ, (x % 127 = 6) ∧ (2037 % 127 = 5) ∧ x = 2038 :=
by
  sorry

end find_divided_number_l2380_238089


namespace championship_positions_l2380_238017

def positions_valid : Prop :=
  ∃ (pos_A pos_B pos_D pos_E pos_V pos_G : ℕ),
  (pos_A = pos_B + 3) ∧
  (pos_D < pos_E ∧ pos_E < pos_B) ∧
  (pos_V < pos_G) ∧
  (pos_D = 1) ∧
  (pos_E = 2) ∧
  (pos_B = 3) ∧
  (pos_V = 4) ∧
  (pos_G = 5) ∧
  (pos_A = 6)

theorem championship_positions : positions_valid :=
by
  sorry

end championship_positions_l2380_238017


namespace eighth_square_more_tiles_than_seventh_l2380_238091

-- Define the total number of tiles in the nth square
def total_tiles (n : ℕ) : ℕ := n^2 + 2 * n

-- Formulate the theorem statement
theorem eighth_square_more_tiles_than_seventh :
  total_tiles 8 - total_tiles 7 = 17 := by
  sorry

end eighth_square_more_tiles_than_seventh_l2380_238091


namespace find_p_q_r_l2380_238082

def is_rel_prime (m n : ℕ) : Prop := Nat.gcd m n = 1

theorem find_p_q_r (x : ℝ) (p q r : ℕ)
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9 / 4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p / q - Real.sqrt r)
  (hpq_rel_prime : is_rel_prime p q)
  (hp : 0 < p)
  (hq : 0 < q)
  (hr : 0 < r) :
  p + q + r = 26 :=
sorry

end find_p_q_r_l2380_238082


namespace probability_of_drawing_red_ball_l2380_238085

noncomputable def probability_of_red_ball (total_balls red_balls : ℕ) : ℚ :=
  red_balls / total_balls

theorem probability_of_drawing_red_ball:
  probability_of_red_ball 5 3 = 3 / 5 :=
by
  unfold probability_of_red_ball
  norm_num

end probability_of_drawing_red_ball_l2380_238085


namespace g_g_g_3_l2380_238066

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 2*n + 1 else 4*n - 3

theorem g_g_g_3 : g (g (g 3)) = 241 := by
  sorry

end g_g_g_3_l2380_238066


namespace decimal_representation_of_7_over_12_eq_0_point_5833_l2380_238088

theorem decimal_representation_of_7_over_12_eq_0_point_5833 : (7 : ℝ) / 12 = 0.5833 :=
by
  sorry

end decimal_representation_of_7_over_12_eq_0_point_5833_l2380_238088


namespace long_fur_brown_dogs_l2380_238023

-- Defining the basic parameters given in the problem
def total_dogs : ℕ := 45
def long_fur : ℕ := 26
def brown_dogs : ℕ := 30
def neither_long_fur_nor_brown : ℕ := 8

-- Statement of the theorem
theorem long_fur_brown_dogs : ∃ LB : ℕ, LB = 27 ∧ total_dogs = long_fur + brown_dogs - LB + neither_long_fur_nor_brown :=
by {
  -- skipping the proof
  sorry
}

end long_fur_brown_dogs_l2380_238023


namespace distance_light_300_years_eq_l2380_238003

-- Define the constant distance light travels in one year
def distance_light_year : ℕ := 9460800000000

-- Define the time period in years
def time_period : ℕ := 300

-- Define the expected distance light travels in 300 years in scientific notation
def expected_distance : ℝ := 28382 * 10^13

-- The theorem to prove
theorem distance_light_300_years_eq :
  (distance_light_year * time_period) = 2838200000000000 :=
by
  sorry

end distance_light_300_years_eq_l2380_238003


namespace original_price_of_painting_l2380_238079

theorem original_price_of_painting (purchase_price : ℝ) (fraction : ℝ) (original_price : ℝ) :
  purchase_price = 200 → fraction = 1/4 → purchase_price = original_price * fraction → original_price = 800 :=
by
  intros h1 h2 h3
  -- proof steps here
  sorry

end original_price_of_painting_l2380_238079


namespace square_garden_perimeter_l2380_238038

theorem square_garden_perimeter (A : ℝ) (hA : A = 450) : 
    ∃ P : ℝ, P = 60 * Real.sqrt 2 :=
  by
    sorry

end square_garden_perimeter_l2380_238038


namespace minimum_value_l2380_238041

theorem minimum_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ x : ℝ, x = 4 * a^3 + 16 * b^3 + 25 * c^3 + 1 / (5 * a * b * c) ∧ x ≥ 4 * Real.sqrt 3 :=
by sorry

end minimum_value_l2380_238041


namespace negative_fraction_comparison_l2380_238012

theorem negative_fraction_comparison : (-3/4 : ℚ) > (-4/5 : ℚ) :=
sorry

end negative_fraction_comparison_l2380_238012


namespace factorization_correct_l2380_238092

theorem factorization_correct (C D : ℤ) (h : 15 = C * D ∧ 48 = 8 * 6 ∧ -56 = -8 * D - 6 * C):
  C * D + C = 18 :=
  sorry

end factorization_correct_l2380_238092


namespace average_of_remaining_ten_numbers_l2380_238002

theorem average_of_remaining_ten_numbers
  (avg_50 : ℝ)
  (n_50 : ℝ)
  (avg_40 : ℝ)
  (n_40 : ℝ)
  (sum_50 : n_50 * avg_50 = 3800)
  (sum_40 : n_40 * avg_40 = 3200)
  (n_10 : n_50 - n_40 = 10)
  : (3800 - 3200) / 10 = 60 :=
by
  sorry

end average_of_remaining_ten_numbers_l2380_238002


namespace ball_returns_to_bella_after_13_throws_l2380_238039

def girl_after_throws (start : ℕ) (throws : ℕ) : ℕ :=
  (start + throws * 5) % 13

theorem ball_returns_to_bella_after_13_throws :
  girl_after_throws 1 13 = 1 :=
sorry

end ball_returns_to_bella_after_13_throws_l2380_238039


namespace hans_deposit_l2380_238051

noncomputable def calculate_deposit : ℝ :=
  let flat_fee := 30
  let kid_deposit := 2 * 3
  let adult_deposit := 8 * 6
  let senior_deposit := 5 * 4
  let student_deposit := 3 * 4.5
  let employee_deposit := 2 * 2.5
  let total_deposit_before_service := flat_fee + kid_deposit + adult_deposit + senior_deposit + student_deposit + employee_deposit
  let service_charge := total_deposit_before_service * 0.05
  total_deposit_before_service + service_charge

theorem hans_deposit : calculate_deposit = 128.63 :=
by
  sorry

end hans_deposit_l2380_238051


namespace initial_men_invited_l2380_238044

theorem initial_men_invited (M W C : ℕ) (h1 : W = M / 2) (h2 : C + 10 = 30) (h3 : M + W + C = 80) (h4 : C = 20) : M = 40 :=
sorry

end initial_men_invited_l2380_238044


namespace average_of_remaining_two_numbers_l2380_238004

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 2.5)
  (h2 : (a + b) / 2 = 1.1)
  (h3 : (c + d) / 2 = 1.4) : 
  (e + f) / 2 = 5 :=
by
  sorry

end average_of_remaining_two_numbers_l2380_238004
