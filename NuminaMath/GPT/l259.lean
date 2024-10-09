import Mathlib

namespace transformed_parabola_is_correct_l259_25950

-- Definitions based on conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 3
def shifted_left (x : ℝ) : ℝ := original_parabola (x - 2)
def shifted_up (y : ℝ) : ℝ := y + 2

-- Theorem statement
theorem transformed_parabola_is_correct :
  ∀ x : ℝ, shifted_up (shifted_left x) = 3 * x^2 + 6 * x - 1 :=
by 
  -- Proof will be filled in here
  sorry

end transformed_parabola_is_correct_l259_25950


namespace smallest_class_number_selected_l259_25972

theorem smallest_class_number_selected
  {n k : ℕ} (hn : n = 30) (hk : k = 5) (h_sum : ∃ x : ℕ, x + (x + 6) + (x + 12) + (x + 18) + (x + 24) = 75) :
  ∃ x : ℕ, x = 3 := 
sorry

end smallest_class_number_selected_l259_25972


namespace circle_diameter_l259_25993

theorem circle_diameter (r : ℝ) (h : π * r^2 = 16 * π) : 2 * r = 8 :=
by
  sorry

end circle_diameter_l259_25993


namespace equilateral_triangle_BJ_l259_25978

-- Define points G, F, H, J and their respective lengths on sides AB and BC
def equilateral_triangle_AG_GF_HJ_FC (AG GF HJ FC BJ : ℕ) : Prop :=
  AG = 3 ∧ GF = 11 ∧ HJ = 5 ∧ FC = 4 ∧ 
    (∀ (side_length : ℕ), side_length = AG + GF + HJ + FC → 
    (∀ (length_J : ℕ), length_J = side_length - (AG + HJ) → BJ = length_J))

-- Example usage statement
theorem equilateral_triangle_BJ : 
  ∃ BJ, equilateral_triangle_AG_GF_HJ_FC 3 11 5 4 BJ ∧ BJ = 15 :=
by
  use 15
  sorry

end equilateral_triangle_BJ_l259_25978


namespace maria_sister_drank_l259_25975

-- Define the conditions
def initial_bottles : ℝ := 45.0
def maria_drank : ℝ := 14.0
def remaining_bottles : ℝ := 23.0

-- Define the problem statement to prove the number of bottles Maria's sister drank
theorem maria_sister_drank (initial_bottles maria_drank remaining_bottles : ℝ) : 
    (initial_bottles - maria_drank) - remaining_bottles = 8.0 :=
by
  sorry

end maria_sister_drank_l259_25975


namespace problem1_solution_set_problem2_min_value_l259_25908

-- For Problem (1)
def f (x a b : ℝ) : ℝ := |x + a| + |x - b|

theorem problem1_solution_set (x : ℝ) (h : f x 1 1 ≤ 4) : 
  -2 ≤ x ∧ x ≤ 2 :=
sorry

-- For Problem (2)
theorem problem2_min_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
  (h3 : ∀ x : ℝ, f x a b ≥ 2) : 
  (1 / a) + (2 / b) = 3 :=
sorry

end problem1_solution_set_problem2_min_value_l259_25908


namespace evaluate_expression_l259_25911

theorem evaluate_expression : 
  (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3) = 3 + 2 * Real.sqrt 3 / 3) :=
by
  sorry

end evaluate_expression_l259_25911


namespace satisfy_eq_pairs_l259_25928

theorem satisfy_eq_pairs (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ (x = 4 ∧ (y = 1 ∨ y = -3) ∨ x = -4 ∧ (y = 1 ∨ y = -3)) :=
by
  sorry

end satisfy_eq_pairs_l259_25928


namespace sphere_volume_proof_l259_25918

noncomputable def sphereVolume (d : ℝ) (S : ℝ) : ℝ :=
  let r := Real.sqrt (S / Real.pi)
  let R := Real.sqrt (r^2 + d^2)
  (4 / 3) * Real.pi * R^3

theorem sphere_volume_proof : sphereVolume 1 (2 * Real.pi) = 4 * Real.sqrt 3 * Real.pi :=
by
  sorry

end sphere_volume_proof_l259_25918


namespace combinations_x_eq_2_or_8_l259_25914

theorem combinations_x_eq_2_or_8 (x : ℕ) (h_pos : 0 < x) (h_comb : Nat.choose 10 x = Nat.choose 10 2) : x = 2 ∨ x = 8 :=
sorry

end combinations_x_eq_2_or_8_l259_25914


namespace f_triple_application_l259_25963

-- Define the function f : ℕ → ℕ such that f(x) = 3x + 2
def f (x : ℕ) : ℕ := 3 * x + 2

-- Theorem statement to prove f(f(f(1))) = 53
theorem f_triple_application : f (f (f 1)) = 53 := 
by 
  sorry

end f_triple_application_l259_25963


namespace wall_building_time_l259_25953

variables (f b c y : ℕ) 

theorem wall_building_time :
  (y = 2 * f * c / b) 
  ↔ 
  (f > 0 ∧ b > 0 ∧ c > 0 ∧ (f * b * y = 2 * b * c)) := 
sorry

end wall_building_time_l259_25953


namespace number_of_routes_l259_25926

open Nat

theorem number_of_routes (south_cities north_cities : ℕ) 
  (connections : south_cities = 4 ∧ north_cities = 5) : 
  ∃ routes, routes = (factorial 3) * (5 ^ 4) := 
by
  sorry

end number_of_routes_l259_25926


namespace equal_playing_time_for_each_player_l259_25916

-- Defining the conditions
def num_players : ℕ := 10
def players_on_field : ℕ := 8
def match_duration : ℕ := 45
def total_field_time : ℕ := players_on_field * match_duration
def equal_playing_time : ℕ := total_field_time / num_players

-- Stating the question and the proof problem
theorem equal_playing_time_for_each_player : equal_playing_time = 36 := 
  by sorry

end equal_playing_time_for_each_player_l259_25916


namespace optimal_discount_order_l259_25954

variables (p : ℝ) (d1 : ℝ) (d2 : ℝ)

-- Original price of "Stars Beyond" is 30 dollars
def original_price : ℝ := 30

-- Fixed discount is 5 dollars
def discount_5 : ℝ := 5

-- 25% discount represented as a multiplier
def discount_25 : ℝ := 0.75

-- Applying $5 discount first and then 25% discount
def price_after_5_then_25_discount := discount_25 * (original_price - discount_5)

-- Applying 25% discount first and then $5 discount
def price_after_25_then_5_discount := (discount_25 * original_price) - discount_5

-- The additional savings when applying 25% discount first
def additional_savings := price_after_5_then_25_discount - price_after_25_then_5_discount

theorem optimal_discount_order : 
  additional_savings = 1.25 :=
sorry

end optimal_discount_order_l259_25954


namespace tangent_sum_half_angles_l259_25987

-- Lean statement for the proof problem
theorem tangent_sum_half_angles (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.tan (A / 2) * Real.tan (B / 2) + 
  Real.tan (B / 2) * Real.tan (C / 2) + 
  Real.tan (C / 2) * Real.tan (A / 2) = 1 := 
by
  sorry

end tangent_sum_half_angles_l259_25987


namespace find_PA_values_l259_25938

theorem find_PA_values :
  ∃ P A : ℕ, 10 ≤ P * 10 + A ∧ P * 10 + A < 100 ∧
            (P * 10 + A) ^ 2 / 1000 = P ∧ (P * 10 + A) ^ 2 % 10 = A ∧
            ((P = 9 ∧ A = 5) ∨ (P = 9 ∧ A = 6)) := by
  sorry

end find_PA_values_l259_25938


namespace sum_of_first_ten_nice_numbers_is_182_l259_25902

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, is_proper_divisor n m → ∃ p q, n = p * q ∧ p ≠ q

def first_ten_nice_numbers : List ℕ := [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sum_first_ten_nice_numbers : ℕ := first_ten_nice_numbers.sum

theorem sum_of_first_ten_nice_numbers_is_182 :
  sum_first_ten_nice_numbers = 182 :=
by
  sorry

end sum_of_first_ten_nice_numbers_is_182_l259_25902


namespace only_natural_number_dividing_power_diff_l259_25996

theorem only_natural_number_dividing_power_diff (n : ℕ) (h : n ∣ (2^n - 1)) : n = 1 :=
by
  sorry

end only_natural_number_dividing_power_diff_l259_25996


namespace boys_at_beginning_is_15_l259_25989

noncomputable def number_of_boys_at_beginning (B : ℝ) : Prop :=
  let girls_start := 1.20 * B
  let girls_end := 2 * girls_start
  let total_students := B + girls_end
  total_students = 51 

theorem boys_at_beginning_is_15 : number_of_boys_at_beginning 15 := 
  by
  -- Sorry is added to skip the proof
  sorry

end boys_at_beginning_is_15_l259_25989


namespace factor_x4_minus_81_l259_25944

theorem factor_x4_minus_81 :
  ∀ x : ℝ, x^4 - 81 = (x - 3) * (x + 3) * (x^2 + 9) :=
by
  intros x
  sorry

end factor_x4_minus_81_l259_25944


namespace probability_of_three_white_balls_equals_8_over_65_l259_25986

noncomputable def probability_three_white_balls (n_white n_black : ℕ) (draws : ℕ) : ℚ :=
  (Nat.choose n_white draws : ℚ) / Nat.choose (n_white + n_black) draws

theorem probability_of_three_white_balls_equals_8_over_65 :
  probability_three_white_balls 8 7 3 = 8 / 65 :=
by
  sorry

end probability_of_three_white_balls_equals_8_over_65_l259_25986


namespace eval_complex_fraction_expr_l259_25922

def complex_fraction_expr : ℚ :=
  2 + (3 / (4 + (5 / (6 + (7 / 8)))))

theorem eval_complex_fraction_expr : complex_fraction_expr = 137 / 52 :=
by
  -- we skip the actual proof but ensure it can build successfully.
  sorry

end eval_complex_fraction_expr_l259_25922


namespace sum_all_possible_values_l259_25957

theorem sum_all_possible_values (x : ℝ) (h : x^2 = 16) :
  (x = 4 ∨ x = -4) → (4 + (-4) = 0) :=
by
  intro h1
  have : 4 + (-4) = 0 := by norm_num
  exact this

end sum_all_possible_values_l259_25957


namespace is_factorization_l259_25952

-- given an equation A,
-- Prove A is factorization: 
-- i.e., x^3 - x = x * (x + 1) * (x - 1)

theorem is_factorization (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end is_factorization_l259_25952


namespace second_option_feasible_l259_25910

def Individual : Type := String
def M : Individual := "M"
def I : Individual := "I"
def P : Individual := "P"
def A : Individual := "A"

variable (is_sitting : Individual → Prop)

-- Given conditions
axiom fact1 : ¬ is_sitting M
axiom fact2 : ¬ is_sitting A
axiom fact3 : ¬ is_sitting M → is_sitting I
axiom fact4 : is_sitting I → is_sitting P

theorem second_option_feasible :
  is_sitting I ∧ is_sitting P ∧ ¬ is_sitting M ∧ ¬ is_sitting A :=
by
  sorry

end second_option_feasible_l259_25910


namespace graph_intersects_x_axis_once_l259_25981

noncomputable def f (m x : ℝ) : ℝ := (m - 1) * x^2 - 6 * x + (3 / 2) * m

theorem graph_intersects_x_axis_once (m : ℝ) :
  (∃ x : ℝ, f m x = 0 ∧ ∀ y : ℝ, f m y = 0 → y = x) ↔ (m = 1 ∨ m = 3 ∨ m = -2) :=
by
  sorry

end graph_intersects_x_axis_once_l259_25981


namespace largest_n_unique_k_l259_25968

theorem largest_n_unique_k : ∃ n : ℕ, n = 24 ∧ (∃! k : ℕ, 
  3 / 7 < n / (n + k: ℤ) ∧ n / (n + k: ℤ) < 8 / 19) :=
by
  sorry

end largest_n_unique_k_l259_25968


namespace passenger_catches_bus_l259_25932

-- Definitions based on conditions from part a)
def P_route3 := 0.20
def P_route6 := 0.60

-- Statement to prove based on part c)
theorem passenger_catches_bus : 
  P_route3 + P_route6 = 0.80 := 
by
  sorry

end passenger_catches_bus_l259_25932


namespace psychology_majors_percentage_in_liberal_arts_l259_25979

theorem psychology_majors_percentage_in_liberal_arts 
  (total_students : ℕ) 
  (percent_freshmen : ℝ) 
  (percent_freshmen_liberal_arts : ℝ) 
  (percent_freshmen_psych_majors_liberal_arts : ℝ) 
  (h1: percent_freshmen = 0.40) 
  (h2: percent_freshmen_liberal_arts = 0.50)
  (h3: percent_freshmen_psych_majors_liberal_arts = 0.10) :
  ((percent_freshmen_psych_majors_liberal_arts / (percent_freshmen * percent_freshmen_liberal_arts)) * 100 = 50) :=
by
  sorry

end psychology_majors_percentage_in_liberal_arts_l259_25979


namespace sum_of_first_50_digits_of_one_over_1234_l259_25933

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l259_25933


namespace negation_proof_l259_25999

theorem negation_proof :
  (¬ ∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∃ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_proof_l259_25999


namespace subset_proper_l259_25931

def M : Set ℝ := {x | x^2 - x ≤ 0}

def N : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem subset_proper : N ⊂ M := by
  sorry

end subset_proper_l259_25931


namespace henry_socks_l259_25994

theorem henry_socks : 
  ∃ a b c : ℕ, 
    a + b + c = 15 ∧ 
    2 * a + 3 * b + 5 * c = 36 ∧ 
    a ≥ 1 ∧ b ≥ 1 ∧ c ≥ 1 ∧ 
    a = 11 :=
by
  sorry

end henry_socks_l259_25994


namespace find_fraction_l259_25923

def f (x : ℤ) : ℤ := 3 * x + 4
def g (x : ℤ) : ℤ := 4 * x - 3

theorem find_fraction :
  (f (g (f 2)):ℚ) / (g (f (g 2)):ℚ) = 115 / 73 := by
  sorry

end find_fraction_l259_25923


namespace taxi_ride_distance_l259_25943

theorem taxi_ride_distance
  (initial_charge : ℝ) (additional_charge : ℝ) 
  (total_charge : ℝ) (initial_increment : ℝ) (distance_increment : ℝ)
  (initial_charge_eq : initial_charge = 2.10) 
  (additional_charge_eq : additional_charge = 0.40) 
  (total_charge_eq : total_charge = 17.70) 
  (initial_increment_eq : initial_increment = 1/5) 
  (distance_increment_eq : distance_increment = 1/5) : 
  (distance : ℝ) = 8 :=
by sorry

end taxi_ride_distance_l259_25943


namespace theresa_needs_15_hours_l259_25940

theorem theresa_needs_15_hours 
  (h1 : ℕ) (h2 : ℕ) (h3 : ℕ) (h4 : ℕ) (h5 : ℕ) (average : ℕ) (weeks : ℕ) (total_hours_first_5 : ℕ) :
  h1 = 10 → h2 = 13 → h3 = 9 → h4 = 14 → h5 = 11 → average = 12 → weeks = 6 → 
  total_hours_first_5 = h1 + h2 + h3 + h4 + h5 → 
  (total_hours_first_5 + x) / weeks = average → x = 15 :=
by
  intros h1_eq h2_eq h3_eq h4_eq h5_eq avg_eq weeks_eq sum_eq avg_eqn
  sorry

end theresa_needs_15_hours_l259_25940


namespace find_power_l259_25971

theorem find_power (x y : ℕ) (h1 : 2^x - 2^y = 3 * 2^11) (h2 : x = 13) : y = 11 :=
sorry

end find_power_l259_25971


namespace part_one_part_two_l259_25915

-- First part: Prove that \( (1)(-1)^{2017}+(\frac{1}{2})^{-2}+(3.14-\pi)^{0} = 4\)
theorem part_one : (1 * (-1:ℤ)^2017 + (1/2)^(-2:ℤ) + (3.14 - Real.pi)^0 : ℝ) = 4 := 
  sorry

-- Second part: Prove that \( ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 \)
theorem part_two (x : ℝ) : ((-2*x^2)^3 + 4*x^3*x^3) = -4*x^6 := 
  sorry

end part_one_part_two_l259_25915


namespace fraction_division_l259_25983

theorem fraction_division (a b c d e : ℚ)
  (h1 : a = 3 / 7)
  (h2 : b = 1 / 3)
  (h3 : d = 2 / 5)
  (h4 : c = a + b)
  (h5 : e = c / d):
  e = 40 / 21 := by
  sorry

end fraction_division_l259_25983


namespace curve_is_line_l259_25977

theorem curve_is_line (r θ x y : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  x + y = 1 := by
  sorry

end curve_is_line_l259_25977


namespace exponents_divisible_by_8_l259_25941

theorem exponents_divisible_by_8 (n : ℕ) : 8 ∣ (3^(4 * n + 1) + 5^(2 * n + 1)) :=
by
-- Base case and inductive step will be defined here.
sorry

end exponents_divisible_by_8_l259_25941


namespace B_pow_101_l259_25984

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]
  ]

theorem B_pow_101 :
  B ^ 101 = ![
    ![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]
  ] :=
  sorry

end B_pow_101_l259_25984


namespace sum_of_quotient_and_remainder_is_184_l259_25962

theorem sum_of_quotient_and_remainder_is_184 
  (q r : ℕ)
  (h1 : 23 * 17 + 19 = q)
  (h2 : q * 10 = r)
  (h3 : r / 23 = 178)
  (h4 : r % 23 = 6) :
  178 + 6 = 184 :=
by
  -- Inform Lean that we are skipping the proof
  sorry

end sum_of_quotient_and_remainder_is_184_l259_25962


namespace lena_more_than_nicole_l259_25969

theorem lena_more_than_nicole (L K N : ℕ) 
  (h1 : L = 23)
  (h2 : 4 * K = L + 7)
  (h3 : K = N - 6) : L - N = 10 := sorry

end lena_more_than_nicole_l259_25969


namespace john_initial_pairs_9_l259_25988

-- Definitions based on the conditions in the problem

def john_initial_pairs (x : ℕ) := 2 * x   -- Each pair consists of 2 socks

def john_remaining_socks (x : ℕ) := john_initial_pairs x - 5   -- John loses 5 individual socks

def john_max_pairs_left := 7
def john_minimum_socks_required := john_max_pairs_left * 2  -- 7 pairs mean he needs 14 socks

-- Theorem statement proving John initially had 9 pairs of socks
theorem john_initial_pairs_9 : 
  ∀ (x : ℕ), john_remaining_socks x ≥ john_minimum_socks_required → x = 9 := by
  sorry

end john_initial_pairs_9_l259_25988


namespace height_of_square_pyramid_is_13_l259_25955

noncomputable def square_pyramid_height (base_edge : ℝ) (adjacent_face_angle : ℝ) : ℝ :=
  let half_diagonal := base_edge * (Real.sqrt 2) / 2
  let sin_angle := Real.sin (adjacent_face_angle / 2 : ℝ)
  let opp_side := half_diagonal * sin_angle
  let height := half_diagonal * sin_angle / (Real.sqrt 3)
  height

theorem height_of_square_pyramid_is_13 :
  ∀ (base_edge : ℝ) (adjacent_face_angle : ℝ), 
  base_edge = 26 → 
  adjacent_face_angle = 120 → 
  square_pyramid_height base_edge adjacent_face_angle = 13 :=
by
  intros base_edge adjacent_face_angle h_base_edge h_adj_face_angle
  rw [h_base_edge, h_adj_face_angle]
  have half_diagonal := 26 * (Real.sqrt 2) / 2
  have sin_angle := Real.sin (120 / 2 : ℝ) -- sin 60 degrees
  have sqrt_three := Real.sqrt 3
  have height := (half_diagonal * sin_angle) / sqrt_three
  sorry

end height_of_square_pyramid_is_13_l259_25955


namespace arithmetic_sequence_geometric_sequence_l259_25982

-- Arithmetic sequence proof
theorem arithmetic_sequence (d n : ℕ) (a_n a_1 : ℤ) (s_n : ℤ) :
  d = 2 → n = 15 → a_n = -10 →
  a_1 = -38 ∧ s_n = -360 :=
sorry

-- Geometric sequence proof
theorem geometric_sequence (a_1 a_4 q s_3 : ℤ) :
  a_1 = -1 → a_4 = 64 →
  q = -4 ∧ s_3 = -13 :=
sorry

end arithmetic_sequence_geometric_sequence_l259_25982


namespace Q_neither_necessary_nor_sufficient_l259_25913

-- Define the propositions P and Q
def PropositionP (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  ∀ x : ℝ, (a1*x^2 + b1*x + c1 > 0) ↔ (a2*x^2 + b2*x + c2 > 0)

def PropositionQ (a1 b1 c1 a2 b2 c2 : ℝ) : Prop :=
  (a1 / a2 = b1 / b2) ∧ (b1 / b2 = c1 / c2)

-- The final statement to prove that Q is neither necessary nor sufficient for P
theorem Q_neither_necessary_nor_sufficient (a1 b1 c1 a2 b2 c2 : ℝ) :
  ¬ ((PropositionQ a1 b1 c1 a2 b2 c2) ↔ (PropositionP a1 b1 c1 a2 b2 c2)) := sorry

end Q_neither_necessary_nor_sufficient_l259_25913


namespace total_cost_of_rolls_l259_25980

-- Defining the conditions
def price_per_dozen : ℕ := 5
def total_rolls_bought : ℕ := 36
def rolls_per_dozen : ℕ := 12

-- Prove the total cost calculation
theorem total_cost_of_rolls : (total_rolls_bought / rolls_per_dozen) * price_per_dozen = 15 :=
by
  sorry

end total_cost_of_rolls_l259_25980


namespace ratio_proof_l259_25964

variables {F : Type*} [Field F] 
variables (w x y z : F)

theorem ratio_proof 
  (h1 : w / x = 4 / 3) 
  (h2 : y / z = 3 / 2) 
  (h3 : z / x = 1 / 6) : 
  w / y = 16 / 3 :=
by sorry

end ratio_proof_l259_25964


namespace vectors_parallel_perpendicular_l259_25985

theorem vectors_parallel_perpendicular (t t1 t2 : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) 
    (h_a : a = (2, t)) (h_b : b = (1, 2)) :
    ((2 * 2 = t * 1) → t1 = 4) ∧ ((2 * 1 + 2 * t = 0) → t2 = -1) :=
by 
  sorry

end vectors_parallel_perpendicular_l259_25985


namespace a_minus_2_values_l259_25958

theorem a_minus_2_values (a : ℝ) (h : |a| = 3) : a - 2 = 1 ∨ a - 2 = -5 :=
by {
  -- the theorem states that given the absolute value condition, a - 2 can be 1 or -5
  sorry
}

end a_minus_2_values_l259_25958


namespace necess_suff_cond_odd_function_l259_25901

noncomputable def f (ω : ℝ) (ϕ : ℝ) (x : ℝ) := Real.sin (ω * x + ϕ)

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def P (ω ϕ : ℝ) : Prop := f ω ϕ 0 = 0
def Q (ω ϕ : ℝ) : Prop := is_odd (f ω ϕ)

theorem necess_suff_cond_odd_function (ω ϕ : ℝ) : P ω ϕ ↔ Q ω ϕ := by
  sorry

end necess_suff_cond_odd_function_l259_25901


namespace tan_alpha_plus_pi_div4_sin2alpha_over_expr_l259_25925

variables (α : ℝ) (h : Real.tan α = 3)

-- Problem 1
theorem tan_alpha_plus_pi_div4 : Real.tan (α + π / 4) = -2 :=
by
  sorry

-- Problem 2
theorem sin2alpha_over_expr : (Real.sin (2 * α)) / (Real.sin α ^ 2 + Real.sin α * Real.cos α - Real.cos (2 * α) - 1) = 3 / 5 :=
by
  sorry

end tan_alpha_plus_pi_div4_sin2alpha_over_expr_l259_25925


namespace diameter_of_inscribed_circle_l259_25903

theorem diameter_of_inscribed_circle (a b c r : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_radius : r = (a + b - c) / 2) : 
  2 * r = a + b - c :=
by
  sorry

end diameter_of_inscribed_circle_l259_25903


namespace cone_shape_in_spherical_coordinates_l259_25945

-- Define the conditions as given in the problem
def spherical_coordinates (rho theta phi c : ℝ) : Prop := 
  rho = c * Real.sin phi

-- Define the main statement to prove
theorem cone_shape_in_spherical_coordinates (rho theta phi c : ℝ) (hpos : 0 < c) :
  spherical_coordinates rho theta phi c → 
  ∃ cone : Prop, cone :=
sorry

end cone_shape_in_spherical_coordinates_l259_25945


namespace percentage_increase_in_yield_after_every_harvest_is_20_l259_25935

theorem percentage_increase_in_yield_after_every_harvest_is_20
  (P : ℝ)
  (h1 : ∀ n : ℕ, n = 1 → 20 * n = 20)
  (h2 : 20 + 20 * (1 + P / 100) = 44) :
  P = 20 := 
sorry

end percentage_increase_in_yield_after_every_harvest_is_20_l259_25935


namespace total_time_spent_l259_25919

-- Definition of the problem conditions
def warm_up_time : ℕ := 10
def additional_puzzles : ℕ := 2
def multiplier : ℕ := 3

-- Statement to prove the total time spent solving puzzles
theorem total_time_spent : warm_up_time + (additional_puzzles * (multiplier * warm_up_time)) = 70 :=
by
  sorry

end total_time_spent_l259_25919


namespace candle_problem_l259_25947

theorem candle_problem :
  ∃ x : ℚ,
    (1 - x / 6 = 3 * (1 - x / 5)) ∧
    x = 60 / 13 :=
by
  -- let initial_height_first_candle be 1
  -- let rate_first_burns be 1 / 6
  -- let initial_height_second_candle be 1
  -- let rate_second_burns be 1 / 5
  -- We want to prove:
  -- 1 - x / 6 = 3 * (1 - x / 5) ∧ x = 60 / 13
  sorry

end candle_problem_l259_25947


namespace inequality_am_gm_l259_25927

variable {a b c : ℝ}

theorem inequality_am_gm (habc_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_abc_eq_1 : a * b * c = 1) : 
  a^3 + b^3 + c^3 + (a * b / (a^2 + b^2) + b * c / (b^2 + c^2) + c * a / (c^2 + a^2)) ≥ 9 / 2 := 
by
  sorry

end inequality_am_gm_l259_25927


namespace speed_ratio_l259_25991

theorem speed_ratio (v1 v2 : ℝ) 
  (h1 : v1 > 0) 
  (h2 : v2 > 0) 
  (h : v2 / v1 - v1 / v2 = 35 / 60) : v1 / v2 = 3 / 4 := 
sorry

end speed_ratio_l259_25991


namespace product_of_repeating_decimal_l259_25907

-- Define the repeating decimal 0.3
def repeating_decimal : ℚ := 1 / 3
-- Define the question
def product (a b : ℚ) := a * b

-- State the theorem to be proved
theorem product_of_repeating_decimal :
  product repeating_decimal 8 = 8 / 3 :=
sorry

end product_of_repeating_decimal_l259_25907


namespace tip_percentage_l259_25951

theorem tip_percentage (cost_of_crown : ℕ) (total_paid : ℕ) (h1 : cost_of_crown = 20000) (h2 : total_paid = 22000) :
  (total_paid - cost_of_crown) * 100 / cost_of_crown = 10 :=
by
  sorry

end tip_percentage_l259_25951


namespace find_N_l259_25995

def consecutive_product_sum_condition (a : ℕ) : Prop :=
  a*(a + 1)*(a + 2) = 8*(a + (a + 1) + (a + 2))

theorem find_N : ∃ (N : ℕ), N = 120 ∧ ∃ (a : ℕ), a > 0 ∧ consecutive_product_sum_condition a := by
  sorry

end find_N_l259_25995


namespace algebraic_expression_l259_25973

theorem algebraic_expression (m : ℝ) (hm : m^2 + m - 1 = 0) : 
  m^3 + 2 * m^2 + 2014 = 2015 := 
by
  sorry

end algebraic_expression_l259_25973


namespace chessboard_overlap_area_l259_25921

theorem chessboard_overlap_area :
  let n := 8
  let cell_area := 1
  let side_length := 8
  let overlap_area := 32 * (Real.sqrt 2 - 1)
  (∃ black_overlap_area : ℝ, black_overlap_area = overlap_area) :=
by
  sorry

end chessboard_overlap_area_l259_25921


namespace factor_polynomial_l259_25990

theorem factor_polynomial (x : ℝ) : 
    54 * x^4 - 135 * x^8 = -27 * x^4 * (5 * x^4 - 2) := 
by 
  sorry

end factor_polynomial_l259_25990


namespace find_q_zero_l259_25997

theorem find_q_zero
  (p q r : ℝ → ℝ)  -- Define p, q, r as functions from ℝ to ℝ (since they are polynomials)
  (h1 : ∀ x, r x = p x * q x + 2)  -- Condition 1: r(x) = p(x) * q(x) + 2
  (h2 : p 0 = 6)                   -- Condition 2: constant term of p(x) is 6
  (h3 : r 0 = 5)                   -- Condition 3: constant term of r(x) is 5
  : q 0 = 1 / 2 :=                 -- Conclusion: q(0) = 1/2
sorry

end find_q_zero_l259_25997


namespace question_1_question_2_l259_25909

open Real

theorem question_1 (a b m : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) : ab < m / 2 → m > 2 := sorry

theorem question_2 (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) (h4 : 9 / a + 1 / b ≥ |x - 1| + |x + 2|) :
  -9/2 ≤ x ∧ x ≤ 7/2 := sorry

end question_1_question_2_l259_25909


namespace interest_first_year_l259_25976
-- Import the necessary math library

-- Define the conditions and proof the interest accrued in the first year
theorem interest_first_year :
  ∀ (P B₁ : ℝ) (r₂ increase_ratio: ℝ),
    P = 1000 →
    B₁ = 1100 →
    r₂ = 0.20 →
    increase_ratio = 0.32 →
    (B₁ - P) = 100 :=
by
  intros P B₁ r₂ increase_ratio P_def B₁_def r₂_def increase_ratio_def
  sorry

end interest_first_year_l259_25976


namespace function_identity_l259_25936

-- Definitions of the problem
def f (n : ℕ) : ℕ := sorry

-- Main theorem to prove
theorem function_identity (f : ℕ → ℕ) 
  (h : ∀ m n : ℕ, m > 0 → n > 0 → f (m + n) * f (m - n) = f (m * m)) : 
  ∀ n : ℕ, n > 0 → f n = 1 := 
sorry

end function_identity_l259_25936


namespace smallest_sum_symmetrical_dice_l259_25942

theorem smallest_sum_symmetrical_dice (p : ℝ) (N : ℕ) (h₁ : p > 0) (h₂ : 6 * N = 2022) : N = 337 := 
by
  -- Proof can be filled in here
  sorry

end smallest_sum_symmetrical_dice_l259_25942


namespace sum_of_possible_M_l259_25905

theorem sum_of_possible_M (M : ℝ) (h : M * (M - 8) = -8) : M = 4 ∨ M = 4 := 
by sorry

end sum_of_possible_M_l259_25905


namespace highest_probability_ksi_expected_value_ksi_equals_l259_25998

noncomputable def probability_ksi_equals (k : ℕ) : ℚ :=
  match k with
  | 2 => 9 / 64
  | 3 => 18 / 64
  | 4 => 21 / 64
  | 5 => 12 / 64
  | 6 => 4 / 64
  | _ => 0

noncomputable def expected_value_ksi : ℚ :=
  2 * (9 / 64) + 3 * (18 / 64) + 4 * (21 / 64) + 5 * (12 / 64) + 6 * (4 / 64)

theorem highest_probability_ksi :
  ∃ k : ℕ, (∀ m : ℕ, probability_ksi_equals k ≥ probability_ksi_equals m) ∧ k = 4 :=
by
  sorry

theorem expected_value_ksi_equals :
  expected_value_ksi = 15 / 4 :=
by
  sorry

end highest_probability_ksi_expected_value_ksi_equals_l259_25998


namespace lcm_of_three_numbers_l259_25930

theorem lcm_of_three_numbers (x : ℕ) :
  (Nat.gcd (3 * x) (Nat.gcd (4 * x) (5 * x)) = 40) →
  Nat.lcm (3 * x) (Nat.lcm (4 * x) (5 * x)) = 2400 :=
by
  sorry

end lcm_of_three_numbers_l259_25930


namespace total_heartbeats_during_race_l259_25906

-- Definitions for the conditions
def heart_beats_per_minute : ℕ := 160
def pace_in_minutes_per_mile : ℕ := 6
def total_distance_in_miles : ℕ := 30

-- Main theorem statement
theorem total_heartbeats_during_race : 
  heart_beats_per_minute * pace_in_minutes_per_mile * total_distance_in_miles = 28800 :=
by
  -- Place the proof here
  sorry

end total_heartbeats_during_race_l259_25906


namespace polynomial_roots_sum_l259_25974

theorem polynomial_roots_sum (a b c d e : ℝ) (h₀ : a ≠ 0)
  (h1 : a * 5^4 + b * 5^3 + c * 5^2 + d * 5 + e = 0)
  (h2 : a * (-3)^4 + b * (-3)^3 + c * (-3)^2 + d * (-3) + e = 0)
  (h3 : a * 2^4 + b * 2^3 + c * 2^2 + d * 2 + e = 0) :
  (b + d) / a = -2677 := 
sorry

end polynomial_roots_sum_l259_25974


namespace martha_clothes_total_l259_25959

-- Given conditions
def jackets_bought : Nat := 4
def t_shirts_bought : Nat := 9
def free_jacket_ratio : Nat := 2
def free_t_shirt_ratio : Nat := 3

-- Problem statement to prove
theorem martha_clothes_total :
  (jackets_bought + jackets_bought / free_jacket_ratio) + 
  (t_shirts_bought + t_shirts_bought / free_t_shirt_ratio) = 18 := 
by 
  sorry

end martha_clothes_total_l259_25959


namespace angle_same_terminal_side_l259_25934

theorem angle_same_terminal_side (θ : ℝ) (α : ℝ) 
  (hθ : θ = -950) 
  (hα_range : 0 ≤ α ∧ α ≤ 180) 
  (h_terminal_side : ∃ k : ℤ, θ = α + k * 360) : 
  α = 130 := by
  sorry

end angle_same_terminal_side_l259_25934


namespace part1_part2_l259_25961

noncomputable def f (x : ℝ) := 2 * Real.sin x * (Real.sin x + Real.cos x)

theorem part1 : f (Real.pi / 4) = 2 := sorry

theorem part2 : ∀ k : ℤ, ∀ x : ℝ, k * Real.pi - Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 3 * Real.pi / 8 → 
  (2 * Real.sqrt 2 * Real.cos (2 * x - Real.pi / 4) > 0) := sorry

end part1_part2_l259_25961


namespace conference_total_duration_is_715_l259_25924

structure ConferenceSession where
  hours : ℕ
  minutes : ℕ

def totalDuration (s1 s2 : ConferenceSession): ℕ :=
  (s1.hours * 60 + s1.minutes) + (s2.hours * 60 + s2.minutes)

def session1 : ConferenceSession := { hours := 8, minutes := 15 }
def session2 : ConferenceSession := { hours := 3, minutes := 40 }

theorem conference_total_duration_is_715 :
  totalDuration session1 session2 = 715 := 
sorry

end conference_total_duration_is_715_l259_25924


namespace uncle_bradley_money_l259_25904

-- Definitions of the variables and conditions
variables (F H M : ℝ)
variables (h1 : F + H = 13)
variables (h2 : 50 * F = (3 / 10) * M)
variables (h3 : 100 * H = (7 / 10) * M)

-- The theorem statement
theorem uncle_bradley_money : M = 1300 :=
by
  sorry

end uncle_bradley_money_l259_25904


namespace hour_hand_degrees_noon_to_2_30_l259_25900

def degrees_moved (hours: ℕ) : ℝ := (hours * 30)

theorem hour_hand_degrees_noon_to_2_30 :
  degrees_moved 2 + degrees_moved 1 / 2 = 75 :=
sorry

end hour_hand_degrees_noon_to_2_30_l259_25900


namespace combined_rate_is_29_l259_25920

def combined_rate_of_mpg (miles_ray : ℕ) (mpg_ray : ℕ) (miles_tom : ℕ) (mpg_tom : ℕ) (miles_jerry : ℕ) (mpg_jerry : ℕ) : ℕ :=
  let gallons_ray := miles_ray / mpg_ray
  let gallons_tom := miles_tom / mpg_tom
  let gallons_jerry := miles_jerry / mpg_jerry
  let total_gallons := gallons_ray + gallons_tom + gallons_jerry
  let total_miles := miles_ray + miles_tom + miles_jerry
  total_miles / total_gallons

theorem combined_rate_is_29 :
  combined_rate_of_mpg 60 50 60 20 60 30 = 29 :=
by
  sorry

end combined_rate_is_29_l259_25920


namespace inequality_abc_geq_36_l259_25965

theorem inequality_abc_geq_36 (a b c : ℝ) (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_prod : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 :=
by
  sorry

end inequality_abc_geq_36_l259_25965


namespace water_used_for_plates_and_clothes_is_48_l259_25948

noncomputable def waterUsedToWashPlatesAndClothes : ℕ := 
  let barrel1 := 65 
  let barrel2 := (75 * 80) / 100 
  let barrel3 := (45 * 60) / 100 
  let totalCollected := barrel1 + barrel2 + barrel3
  let usedForCars := 7 * 2
  let usedForPlants := 15
  let usedForDog := 10
  let usedForCooking := 5
  let usedForBathing := 12
  let totalUsed := usedForCars + usedForPlants + usedForDog + usedForCooking + usedForBathing
  let remainingWater := totalCollected - totalUsed
  remainingWater / 2

theorem water_used_for_plates_and_clothes_is_48 : 
  waterUsedToWashPlatesAndClothes = 48 :=
by
  sorry

end water_used_for_plates_and_clothes_is_48_l259_25948


namespace newOp_of_M_and_N_l259_25949

def newOp (A B : Set ℕ) : Set ℕ :=
  {x | x ∈ A ∨ x ∈ B ∧ x ∉ (A ∩ B)}

theorem newOp_of_M_and_N (M N : Set ℕ) :
  M = {0, 2, 4, 6, 8, 10} →
  N = {0, 3, 6, 9, 12, 15} →
  newOp (newOp M N) M = N :=
by
  intros hM hN
  sorry

end newOp_of_M_and_N_l259_25949


namespace cube_inequality_contradiction_l259_25956

theorem cube_inequality_contradiction (a b : Real) (h : a > b) : ¬(a^3 <= b^3) := by
  sorry

end cube_inequality_contradiction_l259_25956


namespace swap_original_x_y_l259_25937

variables (x y z : ℕ)

theorem swap_original_x_y (x_original y_original : ℕ) 
  (step1 : z = x_original)
  (step2 : x = y_original)
  (step3 : y = z) :
  x = y_original ∧ y = x_original :=
sorry

end swap_original_x_y_l259_25937


namespace max_volume_prism_l259_25917

theorem max_volume_prism (V : ℝ) (h l w : ℝ) 
  (h_eq_2h : l = 2 * h ∧ w = 2 * h) 
  (surface_area_eq : l * h + w * h + l * w = 36) : 
  V = 27 * Real.sqrt 2 := 
  sorry

end max_volume_prism_l259_25917


namespace heavy_operators_earn_129_dollars_per_day_l259_25966

noncomputable def heavy_operator_daily_wage (H : ℕ) : Prop :=
  let laborer_wage := 82
  let total_people := 31
  let total_payroll := 3952
  let laborers_count := 1
  let heavy_operators_count := total_people - laborers_count
  let heavy_operators_payroll := total_payroll - (laborer_wage * laborers_count)
  H = heavy_operators_payroll / heavy_operators_count

theorem heavy_operators_earn_129_dollars_per_day : heavy_operator_daily_wage 129 :=
by
  unfold heavy_operator_daily_wage
  sorry

end heavy_operators_earn_129_dollars_per_day_l259_25966


namespace determine_constants_l259_25967

theorem determine_constants :
  ∃ (a b c p : ℝ), (a = -1) ∧ (b = -1) ∧ (c = -1) ∧ (p = 3) ∧
  (∀ x : ℝ, x^3 + p*x^2 + 3*x - 10 = 0 ↔ (x = a ∨ x = b ∨ x = c)) ∧ 
  c - b = b - a ∧ c - b > 0 :=
by
  sorry

end determine_constants_l259_25967


namespace Jane_exercises_days_per_week_l259_25946

theorem Jane_exercises_days_per_week 
  (goal_hours_per_day : ℕ)
  (weeks : ℕ)
  (total_hours : ℕ)
  (exercise_days_per_week : ℕ) 
  (h_goal : goal_hours_per_day = 1)
  (h_weeks : weeks = 8)
  (h_total_hours : total_hours = 40)
  (h_exercise_hours_weekly : total_hours / weeks = exercise_days_per_week) :
  exercise_days_per_week = 5 :=
by
  sorry

end Jane_exercises_days_per_week_l259_25946


namespace triangle_count_l259_25992

-- Define the function to compute the binomial coefficient
def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define the number of points on each side
def pointsAB : ℕ := 6
def pointsBC : ℕ := 7

-- Compute the number of triangles that can be formed
theorem triangle_count (h₁ : pointsAB = 6) (h₂ : pointsBC = 7) : 
  (binom pointsAB 2) * (binom pointsBC 1) + (binom pointsBC 2) * (binom pointsAB 1) = 231 := by
  sorry

end triangle_count_l259_25992


namespace isosceles_triangle_sides_l259_25960

theorem isosceles_triangle_sides (a b c : ℝ) (h_iso : a = b ∨ b = c ∨ c = a) (h_perimeter : a + b + c = 14) (h_side : a = 4 ∨ b = 4 ∨ c = 4) : 
  (a = 4 ∧ b = 5 ∧ c = 5) ∨ (a = 5 ∧ b = 5 ∧ c = 4) ∨ (a = 4 ∧ b = 4 ∧ c = 6) ∨ (a = 4 ∧ b = 6 ∧ c = 4) :=
  sorry

end isosceles_triangle_sides_l259_25960


namespace divisibility_by_six_l259_25970

theorem divisibility_by_six (a x: ℤ) : ∃ t: ℤ, x = 3 * t ∨ x = 3 * t - a^2 → 6 ∣ a * (x^3 + a^2 * x^2 + a^2 - 1) :=
by
  sorry

end divisibility_by_six_l259_25970


namespace wildcats_points_l259_25939

theorem wildcats_points (panthers_points wildcats_additional_points wildcats_points : ℕ)
  (h_panthers : panthers_points = 17)
  (h_wildcats : wildcats_additional_points = 19)
  (h_wildcats_points : wildcats_points = panthers_points + wildcats_additional_points) :
  wildcats_points = 36 :=
by
  have h1 : panthers_points = 17 := h_panthers
  have h2 : wildcats_additional_points = 19 := h_wildcats
  have h3 : wildcats_points = panthers_points + wildcats_additional_points := h_wildcats_points
  sorry

end wildcats_points_l259_25939


namespace charity_donation_ratio_l259_25929

theorem charity_donation_ratio :
  let total_winnings := 114
  let hot_dog_cost := 2
  let remaining_amount := 55
  let donation_amount := 114 - (remaining_amount + hot_dog_cost)
  donation_amount = 55 :=
by
  sorry

end charity_donation_ratio_l259_25929


namespace cookies_per_batch_l259_25912

theorem cookies_per_batch
  (bag_chips : ℕ)
  (batches : ℕ)
  (chips_per_cookie : ℕ)
  (total_chips : ℕ)
  (h1 : bag_chips = total_chips)
  (h2 : batches = 3)
  (h3 : chips_per_cookie = 9)
  (h4 : total_chips = 81) :
  (bag_chips / batches) / chips_per_cookie = 3 := 
by
  sorry

end cookies_per_batch_l259_25912
