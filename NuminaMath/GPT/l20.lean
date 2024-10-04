import Mathlib

namespace johns_equation_l20_20633

theorem johns_equation (a b c d e : ℤ) (ha : a = 2) (hb : b = 3) 
  (hc : c = 4) (hd : d = 5) : 
  a - (b - (c * (d - e))) = a - b - c * d + e ↔ e = 8 := 
by
  sorry

end johns_equation_l20_20633


namespace non_empty_prime_subsets_count_l20_20890

-- Definition of the set S
def S : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- Definition of primes in S
def prime_subset_S : Set ℕ := {x ∈ S | Nat.Prime x}

-- The statement to prove
theorem non_empty_prime_subsets_count : 
  ∃ n, n = 15 ∧ ∀ T ⊆ prime_subset_S, T ≠ ∅ → ∃ m, n = 2^m - 1 := 
by
  sorry

end non_empty_prime_subsets_count_l20_20890


namespace washer_and_dryer_proof_l20_20412

noncomputable def washer_and_dryer_problem : Prop :=
  ∃ (price_of_washer price_of_dryer : ℕ),
    price_of_washer + price_of_dryer = 600 ∧
    (∃ (k : ℕ), price_of_washer = k * price_of_dryer) ∧
    price_of_dryer = 150 ∧
    price_of_washer / price_of_dryer = 3

theorem washer_and_dryer_proof : washer_and_dryer_problem :=
sorry

end washer_and_dryer_proof_l20_20412


namespace isosceles_triangle_vertex_angle_l20_20897

theorem isosceles_triangle_vertex_angle (exterior_angle : ℝ) (h1 : exterior_angle = 40) : 
  ∃ vertex_angle : ℝ, vertex_angle = 140 :=
by
  sorry

end isosceles_triangle_vertex_angle_l20_20897


namespace ben_bonus_amount_l20_20548

variables (B : ℝ)

-- Conditions
def condition1 := B - (1/22) * B - (1/4) * B - (1/8) * B = 867

-- Theorem statement
theorem ben_bonus_amount (h : condition1 B) : B = 1496.50 := 
sorry

end ben_bonus_amount_l20_20548


namespace product_eval_l20_20435

theorem product_eval (a : ℝ) (h : a = 1) : (a - 3) * (a - 2) * (a - 1) * a * (a + 1) = 0 :=
by
  sorry

end product_eval_l20_20435


namespace sequence_26th_term_l20_20734

theorem sequence_26th_term (a d : ℕ) (n : ℕ) (h_a : a = 4) (h_d : d = 3) (h_n : n = 26) :
  a + (n - 1) * d = 79 :=
by
  sorry

end sequence_26th_term_l20_20734


namespace range_of_a_l20_20089

theorem range_of_a (a x : ℝ) (h1 : 1 ≤ x ∧ x ≤ 3) (h2 : ∀ x, 1 ≤ x ∧ x ≤ 3 → |x - a| < 2) : 1 < a ∧ a < 3 := by
  sorry

end range_of_a_l20_20089


namespace sum_eq_3_or_7_l20_20761

theorem sum_eq_3_or_7 {x y z : ℝ} 
  (h1 : x + y / z = 2)
  (h2 : y + z / x = 2)
  (h3 : z + x / y = 2) : 
  x + y + z = 3 ∨ x + y + z = 7 :=
by
  sorry

end sum_eq_3_or_7_l20_20761


namespace a3_value_l20_20303

theorem a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) (x : ℝ) :
  ( (1 + x) * (a - x) ^ 6 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4 + a₅ * x^5 + a₆ * x^6 + a₇ * x^7 ) →
  ( a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0 ) →
  a = 1 →
  a₃ = -5 :=
by
  sorry

end a3_value_l20_20303


namespace triangle_side_length_l20_20906

theorem triangle_side_length (a b c : ℝ) (B : ℝ) (ha : a = 2) (hB : B = 60) (hc : c = 3) :
  b = Real.sqrt 7 :=
by
  sorry

end triangle_side_length_l20_20906


namespace least_four_digit_integer_has_3_7_11_as_factors_l20_20960

theorem least_four_digit_integer_has_3_7_11_as_factors :
  ∃ x : ℕ, (1000 ≤ x ∧ x < 10000) ∧ (3 ∣ x) ∧ (7 ∣ x) ∧ (11 ∣ x) ∧ x = 1155 := by
  sorry

end least_four_digit_integer_has_3_7_11_as_factors_l20_20960


namespace evaluate_expression_l20_20342

theorem evaluate_expression (x : Int) (h : x = -2023) : abs (abs (abs x - x) + abs x) + x = 4046 :=
by
  rw [h]
  sorry

end evaluate_expression_l20_20342


namespace product_of_two_numbers_l20_20803

theorem product_of_two_numbers (a b : ℕ) (h_gcd : Nat.gcd a b = 8) (h_lcm : Nat.lcm a b = 72) : a * b = 576 := 
by
  sorry

end product_of_two_numbers_l20_20803


namespace jack_mopping_rate_l20_20771

variable (bathroom_floor_area : ℕ) (kitchen_floor_area : ℕ) (time_mopped : ℕ)

theorem jack_mopping_rate
  (h_bathroom : bathroom_floor_area = 24)
  (h_kitchen : kitchen_floor_area = 80)
  (h_time : time_mopped = 13) :
  (bathroom_floor_area + kitchen_floor_area) / time_mopped = 8 :=
by
  sorry

end jack_mopping_rate_l20_20771


namespace complex_subtraction_l20_20260

theorem complex_subtraction (a b : ℂ) (ha : a = 5 - 3 * complex.I) (hb : b = 2 + 3 * complex.I) : 
  a - 3 * b = -1 - 12 * complex.I :=
by
  rw [ha, hb]
  sorry

end complex_subtraction_l20_20260


namespace lift_cars_and_trucks_l20_20213

theorem lift_cars_and_trucks :
  (let car := 5 in let truck := car * 2 in
   let P_cars := 6 * car in
   let P_trucks := 3 * truck in
   P_cars + P_trucks = 60) := 
by
  sorry

end lift_cars_and_trucks_l20_20213


namespace graph_is_hyperbola_l20_20520

theorem graph_is_hyperbola : 
  ∀ x y : ℝ, (x + y)^2 = x^2 + y^2 + 4 ↔ x * y = 2 := 
by
  sorry

end graph_is_hyperbola_l20_20520


namespace trig_identity_l20_20894

variable (α : ℝ)
variable (h : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
variable (h₁ : Real.sin α = 4 / 5)

theorem trig_identity : Real.sin (α + Real.pi / 4) + Real.cos (α + Real.pi / 4) = -3 * Real.sqrt 2 / 5 := 
by 
  sorry

end trig_identity_l20_20894


namespace bob_ears_left_l20_20420

namespace CornProblem

-- Definitions of the given conditions
def initial_bob_bushels : ℕ := 120
def ears_per_bushel : ℕ := 15

def given_away_bushels_terry : ℕ := 15
def given_away_bushels_jerry : ℕ := 8
def given_away_bushels_linda : ℕ := 25
def given_away_ears_stacy : ℕ := 42
def given_away_bushels_susan : ℕ := 9
def given_away_bushels_tim : ℕ := 4
def given_away_ears_tim : ℕ := 18

-- Calculate initial ears of corn
noncomputable def initial_ears_of_corn : ℕ := initial_bob_bushels * ears_per_bushel

-- Calculate total ears given away in bushels
def total_ears_given_away_bushels : ℕ :=
  (given_away_bushels_terry + given_away_bushels_jerry + given_away_bushels_linda +
   given_away_bushels_susan + given_away_bushels_tim) * ears_per_bushel

-- Calculate total ears directly given away
def total_ears_given_away_direct : ℕ :=
  given_away_ears_stacy + given_away_ears_tim

-- Calculate total ears given away
def total_ears_given_away : ℕ :=
  total_ears_given_away_bushels + total_ears_given_away_direct

-- Calculate ears of corn Bob has left
noncomputable def ears_left : ℕ :=
  initial_ears_of_corn - total_ears_given_away

-- The proof statement
theorem bob_ears_left : ears_left = 825 := by
  sorry

end CornProblem

end bob_ears_left_l20_20420


namespace primitive_root_exists_mod_pow_of_two_l20_20101

theorem primitive_root_exists_mod_pow_of_two (n : ℕ) : 
  (∃ x : ℤ, ∀ k : ℕ, 1 ≤ k → x^k % (2^n) ≠ 1 % (2^n)) ↔ (n ≤ 2) := sorry

end primitive_root_exists_mod_pow_of_two_l20_20101


namespace pentagon_inequality_l20_20107

-- Definitions
variables {S R1 R2 R3 R4 R5 : ℝ}
noncomputable def sine108 := Real.sin (108 * Real.pi / 180)

-- Theorem statement
theorem pentagon_inequality (h_area : S > 0) (h_radii : R1 > 0 ∧ R2 > 0 ∧ R3 > 0 ∧ R4 > 0 ∧ R5 > 0) :
  R1^4 + R2^4 + R3^4 + R4^4 + R5^4 ≥ (4 / (5 * sine108^2)) * S^2 :=
by
  sorry

end pentagon_inequality_l20_20107


namespace rectangle_area_error_l20_20821

/-
  Problem: 
  Given:
  1. One side of the rectangle is taken 20% in excess.
  2. The other side of the rectangle is taken 10% in deficit.
  Prove:
  The error percentage in the calculated area is 8%.
-/

noncomputable def error_percentage (L W : ℝ) := 
  let actual_area : ℝ := L * W
  let measured_length : ℝ := 1.20 * L
  let measured_width : ℝ := 0.90 * W
  let measured_area : ℝ := measured_length * measured_width
  ((measured_area - actual_area) / actual_area) * 100

theorem rectangle_area_error
  (L W : ℝ) : error_percentage L W = 8 := 
  sorry

end rectangle_area_error_l20_20821


namespace larger_number_l20_20948

theorem larger_number (t a b : ℝ) (h1 : a + b = t) (h2 : a ^ 2 - b ^ 2 = 208) (ht : t = 104) :
  a = 53 :=
by
  sorry

end larger_number_l20_20948


namespace max_value_2x_minus_y_l20_20456

theorem max_value_2x_minus_y (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) : 2 * x - y ≤ 5 :=
sorry

end max_value_2x_minus_y_l20_20456


namespace remainder_calculation_l20_20178

theorem remainder_calculation :
  (7 * 10^23 + 3^25) % 11 = 5 :=
by
  sorry

end remainder_calculation_l20_20178


namespace find_two_numbers_l20_20388

noncomputable def x := 5 + 2 * Real.sqrt 5
noncomputable def y := 5 - 2 * Real.sqrt 5

theorem find_two_numbers :
  (x * y = 5) ∧ (x + y = 10) :=
by {
  sorry
}

end find_two_numbers_l20_20388


namespace rank_friends_l20_20717

variable (Amy Bill Celine : Prop)

-- Statement definitions
def statement_I := Bill
def statement_II := ¬Amy
def statement_III := ¬Celine

-- Exactly one of the statements is true
def exactly_one_true (s1 s2 s3 : Prop) :=
  (s1 ∧ ¬s2 ∧ ¬s3) ∨ (¬s1 ∧ s2 ∧ ¬s3) ∨ (¬s1 ∧ ¬s2 ∧ s3)

theorem rank_friends (h : exactly_one_true (statement_I Bill) (statement_II Amy) (statement_III Celine)) :
  (Amy ∧ ¬Bill ∧ Celine) :=
sorry

end rank_friends_l20_20717


namespace max_diagonals_in_chessboard_l20_20816

/-- The maximum number of non-intersecting diagonals that can be drawn in an 8x8 chessboard is 36. -/
theorem max_diagonals_in_chessboard : 
  ∃ (diagonals : Finset (ℕ × ℕ)), 
  diagonals.card = 36 ∧ 
  ∀ (d1 d2 : ℕ × ℕ), d1 ∈ diagonals → d2 ∈ diagonals → d1 ≠ d2 → d1.fst ≠ d2.fst ∧ d1.snd ≠ d2.snd := 
  sorry

end max_diagonals_in_chessboard_l20_20816


namespace tickets_savings_percentage_l20_20158

theorem tickets_savings_percentage (P S : ℚ) (h : 8 * S = 5 * P) :
  (12 * P - 12 * S) / (12 * P) * 100 = 37.5 :=
by 
  sorry

end tickets_savings_percentage_l20_20158


namespace probability_three_white_two_black_l20_20691

-- Define the total number of balls
def total_balls : ℕ := 17

-- Define the number of white balls
def white_balls : ℕ := 8

-- Define the number of black balls
def black_balls : ℕ := 9

-- Define the number of balls drawn
def balls_drawn : ℕ := 5

-- Define three white balls drawn
def three_white_drawn : ℕ := 3

-- Define two black balls drawn
def two_black_drawn : ℕ := 2

-- Define the combination formula
noncomputable def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Calculate the probability
noncomputable def probability : ℚ :=
  (combination white_balls three_white_drawn * combination black_balls two_black_drawn : ℚ) 
  / combination total_balls balls_drawn

-- Statement to prove
theorem probability_three_white_two_black :
  probability = 672 / 2063 := by
  sorry

end probability_three_white_two_black_l20_20691


namespace algebraic_expression_for_A_l20_20891

variable {x y A : ℝ}

theorem algebraic_expression_for_A
  (h : (3 * x + 2 * y) ^ 2 = (3 * x - 2 * y) ^ 2 + A) :
  A = 24 * x * y :=
sorry

end algebraic_expression_for_A_l20_20891


namespace remove_6_maximizes_probability_l20_20390

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define what it means to maximize the probability of pairs summing to 12
def maximize_probability (l : List Int) : Prop :=
  ∀ x y, x ≠ y → x ∈ l → y ∈ l → x + y = 12

-- Prove that removing 6 maximizes the probability that the sum of the two chosen numbers is 12
theorem remove_6_maximizes_probability :
  maximize_probability (original_list.erase 6) :=
sorry

end remove_6_maximizes_probability_l20_20390


namespace triangle_is_isosceles_or_right_l20_20908

theorem triangle_is_isosceles_or_right (A B C a b : ℝ) (h : a * Real.cos (π - A) + b * Real.sin (π / 2 + B) = 0)
  (h1 : a = 2 * R * Real.sin A) 
  (h2 : b = 2 * R * Real.sin B) : 
  (A = B ∨ A + B = π / 2) := 
sorry

end triangle_is_isosceles_or_right_l20_20908


namespace relationship_of_y_values_l20_20575

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l20_20575


namespace arithmetic_sequence_first_term_and_difference_l20_20369

theorem arithmetic_sequence_first_term_and_difference
  (a1 d : ℤ)
  (h1 : (a1 + 2 * d) * (a1 + 5 * d) = 406)
  (h2 : a1 + 8 * d = 2 * (a1 + 3 * d) + 6) : 
  a1 = 4 ∧ d = 5 :=
by 
  sorry

end arithmetic_sequence_first_term_and_difference_l20_20369


namespace problem_statement_l20_20917

noncomputable def f : ℝ → ℝ := sorry  -- Define f as a noncomputable function to accommodate the problem constraints

variables (a : ℝ)

theorem problem_statement (periodic_f : ∀ x, f (x + 3) = f x)
    (odd_f : ∀ x, f (-x) = -f x)
    (ineq_f1 : f 1 < 1)
    (eq_f2 : f 2 = (2*a-1)/(a+1)) :
    a < -1 ∨ 0 < a :=
by
  sorry

end problem_statement_l20_20917


namespace hcf_of_three_numbers_l20_20621

theorem hcf_of_three_numbers (a b c : ℕ) (h1 : Nat.lcm a (Nat.lcm b c) = 45600) (h2 : a * b * c = 109183500000) :
  Nat.gcd a (Nat.gcd b c) = 2393750 := by
  sorry

end hcf_of_three_numbers_l20_20621


namespace solve_recurrence_relation_l20_20104

def recurrence_relation (a : ℕ → ℤ) : Prop :=
  ∀ n ≥ 3, a n = 3 * a (n - 1) - 3 * a (n - 2) + a (n - 3) + 24 * n - 6

def initial_conditions (a : ℕ → ℤ) : Prop :=
  a 0 = -4 ∧ a 1 = -2 ∧ a 2 = 2

def explicit_solution (n : ℕ) : ℤ :=
  -4 + 17 * n - 21 * n^2 + 5 * n^3 + n^4

theorem solve_recurrence_relation :
  ∀ (a : ℕ → ℤ),
    recurrence_relation a →
    initial_conditions a →
    ∀ n, a n = explicit_solution n := by
  intros a h_recur h_init n
  sorry

end solve_recurrence_relation_l20_20104


namespace min_jumps_required_to_visit_all_points_and_return_l20_20432

theorem min_jumps_required_to_visit_all_points_and_return :
  ∀ (n : ℕ), n = 2016 →
  ∀ jumps : ℕ → ℕ, (∀ i, jumps i = 2 ∨ jumps i = 3) →
  (∀ i, (jumps (i + 1) + jumps (i + 2)) % n = 0) →
  ∃ (min_jumps : ℕ), min_jumps = 2017 :=
by
  sorry

end min_jumps_required_to_visit_all_points_and_return_l20_20432


namespace final_rider_is_C_l20_20015

def initial_order : List Char := ['A', 'B', 'C']

def leader_changes : Nat := 19
def third_place_changes : Nat := 17

def B_finishes_third (final_order: List Char) : Prop :=
  final_order.get! 2 = 'B'

def total_transpositions (a b : Nat) : Nat :=
  a + b

theorem final_rider_is_C (final_order: List Char) :
  B_finishes_third final_order →
  total_transpositions leader_changes third_place_changes % 2 = 0 →
  final_order = ['C', 'A', 'B'] → 
  final_order.get! 0 = 'C' :=
by
  sorry

end final_rider_is_C_l20_20015


namespace find_product_l20_20684

theorem find_product (a b c d : ℚ) 
  (h₁ : 2 * a + 4 * b + 6 * c + 8 * d = 48)
  (h₂ : 4 * (d + c) = b)
  (h₃ : 4 * b + 2 * c = a)
  (h₄ : c + 1 = d) :
  a * b * c * d = -319603200 / 10503489 := sorry

end find_product_l20_20684


namespace relationship_y1_y2_y3_l20_20581

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l20_20581


namespace burmese_python_eats_alligators_l20_20975

theorem burmese_python_eats_alligators (snake_length : ℝ) (alligator_length : ℝ) (alligator_per_week : ℝ) (total_alligators : ℝ) :
  snake_length = 1.4 → alligator_length = 0.5 → alligator_per_week = 1 → total_alligators = 88 →
  (total_alligators / alligator_per_week) * 7 = 616 := by
  intros
  sorry

end burmese_python_eats_alligators_l20_20975


namespace rectangular_park_area_l20_20988

/-- Define the conditions for the rectangular park -/
def rectangular_park (w l : ℕ) : Prop :=
  l = 3 * w ∧ 2 * (w + l) = 72

/-- Prove that the area of the rectangular park is 243 square meters -/
theorem rectangular_park_area (w l : ℕ) (h : rectangular_park w l) : w * l = 243 := by
  sorry

end rectangular_park_area_l20_20988


namespace chord_division_ratio_l20_20936

theorem chord_division_ratio (R AB PO DP PC x AP PB : ℝ)
  (hR : R = 11)
  (hAB : AB = 18)
  (hPO : PO = 7)
  (hDP : DP = R - PO)
  (hPC : PC = R + PO)
  (hPower : AP * PB = DP * PC)
  (hChord : AP + PB = AB) :
  AP = 12 ∧ PB = 6 ∨ AP = 6 ∧ PB = 12 :=
by
  -- Structure of the theorem is provided.
  -- Proof steps are skipped and marked with sorry.
  sorry

end chord_division_ratio_l20_20936


namespace average_problem_l20_20932

noncomputable def avg2 (a b : ℚ) := (a + b) / 2
noncomputable def avg3 (a b c : ℚ) := (a + b + c) / 3

theorem average_problem :
  avg3 (avg3 2 2 1) (avg2 1 2) 1 = 25 / 18 :=
by
  sorry

end average_problem_l20_20932


namespace geometric_series_sum_first_four_terms_l20_20424

theorem geometric_series_sum_first_four_terms :
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  (a * (1 - r^n) / (1 - r)) = 40 / 27 := by
  let a : ℝ := 1
  let r : ℝ := 1 / 3
  let n : ℕ := 4
  sorry

end geometric_series_sum_first_four_terms_l20_20424


namespace meadow_grazing_days_l20_20143

theorem meadow_grazing_days 
    (a b x : ℝ) 
    (h1 : a + 6 * b = 27 * 6 * x)
    (h2 : a + 9 * b = 23 * 9 * x)
    : ∃ y : ℝ, (a + y * b = 21 * y * x) ∧ y = 12 := 
by
    sorry

end meadow_grazing_days_l20_20143


namespace multiply_identity_l20_20094

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity_l20_20094


namespace genevieve_drinks_pints_l20_20005

theorem genevieve_drinks_pints (total_gallons : ℝ) (thermoses : ℕ) 
  (gallons_to_pints : ℝ) (genevieve_thermoses : ℕ) 
  (h1 : total_gallons = 4.5) (h2 : thermoses = 18) 
  (h3 : gallons_to_pints = 8) (h4 : genevieve_thermoses = 3) : 
  (total_gallons * gallons_to_pints / thermoses) * genevieve_thermoses = 6 := 
by
  admit

end genevieve_drinks_pints_l20_20005


namespace prob_both_primes_l20_20956

-- Define the set of integers from 1 through 30
def int_set : Set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers between 1 and 30
def primes_between_1_and_30 : Set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Calculate the number of ways to choose two distinct elements from a set
noncomputable def combination (n k : ℕ) : ℕ := if k > n then 0 else n.choose k

-- Define the probabilities
noncomputable def prob_primes : ℚ :=
  (combination 10 2) / (combination 30 2)

-- State the theorem to prove
theorem prob_both_primes : prob_primes = 10 / 87 := by
  sorry

end prob_both_primes_l20_20956


namespace square_perimeter_l20_20843

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l20_20843


namespace problem_l20_20507

theorem problem (a b : ℕ) (h1 : ∃ k : ℕ, a * b = k * k) (h2 : ∃ m : ℕ, (2 * a + 1) * (2 * b + 1) = m * m) :
  ∃ n : ℕ, n % 2 = 0 ∧ n > 2 ∧ ∃ p : ℕ, (a + n) * (b + n) = p * p :=
by
  sorry

end problem_l20_20507


namespace percent_increase_sales_l20_20968

-- Define constants for sales
def sales_last_year : ℕ := 320
def sales_this_year : ℕ := 480

-- Define the percent increase formula
def percent_increase (old_value new_value : ℕ) : ℚ :=
  ((new_value - old_value) / old_value) * 100

-- Prove the percent increase from last year to this year is 50%
theorem percent_increase_sales : percent_increase sales_last_year sales_this_year = 50 := by
  sorry

end percent_increase_sales_l20_20968


namespace geometric_series_sum_y_equals_nine_l20_20288

theorem geometric_series_sum_y_equals_nine : 
  (∑' n : ℕ, (1 / 3) ^ n) * (∑' n : ℕ, (-1 / 3) ^ n) = ∑' n : ℕ, (1 / (9 ^ n)) :=
by
  sorry

end geometric_series_sum_y_equals_nine_l20_20288


namespace parabola_unique_solution_l20_20710

theorem parabola_unique_solution (b c : ℝ) :
  (∀ x y : ℝ, (x, y) = (-2, -8) ∨ (x, y) = (4, 28) ∨ (x, y) = (1, 4) →
    (y = x^2 + b * x + c)) →
  b = 4 ∧ c = -1 :=
by
  intro h
  have h₁ := h (-2) (-8) (Or.inl rfl)
  have h₂ := h 4 28 (Or.inr (Or.inl rfl))
  have h₃ := h 1 4 (Or.inr (Or.inr rfl))
  sorry

end parabola_unique_solution_l20_20710


namespace wendy_furniture_time_l20_20391

theorem wendy_furniture_time (chairs tables minutes_per_piece : ℕ) 
    (h_chairs : chairs = 4) 
    (h_tables : tables = 4) 
    (h_minutes_per_piece : minutes_per_piece = 6) : 
    chairs + tables * minutes_per_piece = 48 := 
by 
    sorry

end wendy_furniture_time_l20_20391


namespace postage_problem_l20_20567

theorem postage_problem (n : ℕ) (h_positive : n > 0) (h_postage : ∀ k, k ∈ List.range 121 → ∃ a b c : ℕ, 6 * a + n * b + (n + 2) * c = k) :
  6 * n * (n + 2) - (6 + n + (n + 2)) = 120 → n = 8 := 
by
  sorry

end postage_problem_l20_20567


namespace absolute_value_equation_solution_l20_20299

theorem absolute_value_equation_solution (x : ℝ) : |x - 30| + |x - 24| = |3 * x - 72| ↔ x = 26 :=
by sorry

end absolute_value_equation_solution_l20_20299


namespace hurricanes_valid_lineups_l20_20366

/-- Define the set of Hurricanes' players indexed from 1 to 15. -/
def players := Finset.range 15

/-- Anne is player 0, Rick is player 1, Sam is player 2, John is player 3. -/
def Anne := 0
def Rick := 1
def Sam := 2
def John := 3

/-- The number of valid starting lineups (of 5 players) for which Anne and Rick cannot both play together and Sam and John cannot both play together. -/
def valid_starting_lineups : ℕ :=
  let total_lineups := players.card.choose 5 in
  let invalid_AR := ((players.erase Anne).erase Rick).card.choose 4 in
  let invalid_SJ := ((players.erase Sam).erase John).card.choose 4 in
  let intersection_AR_SJ := ((players.erase Anne).erase Rick).erase Sam).erase John).card.choose 3 in
  total_lineups - invalid_AR - invalid_SJ + intersection_AR_SJ

theorem hurricanes_valid_lineups : valid_starting_lineups = 
    -- Final desired count, combining cases 
    sorry

end hurricanes_valid_lineups_l20_20366


namespace trig_proof_l20_20462

noncomputable def trig_problem (α : ℝ) (h : Real.tan α = 3) : Prop :=
  Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5

theorem trig_proof (α : ℝ) (h : Real.tan α = 3) : Real.cos (α + Real.pi / 4) ^ 2 - Real.cos (α - Real.pi / 4) ^ 2 = -3 / 5 :=
by
  sorry

end trig_proof_l20_20462


namespace event_properties_l20_20469

open MeasureTheory ProbabilityTheory

noncomputable def Ball := { red := 0.5, white := 0.5 }

def event_A := λ (ball1 ball2 : Ball), ball1 = ball2
def event_B := λ (ball1 : Ball), ball1 = Ball.red
def event_C := λ (ball2 : Ball), ball2 = Ball.red
def event_D := λ (ball1 ball2 : Ball), ball1 ≠ ball2

theorem event_properties :
  (¬(MutuallyExclusive event_B event_C)) ∧
  (Complementary event_A event_D) ∧
  (Independent event_A event_B) ∧
  (Independent event_C event_D) :=
by
  simp; sorry

end event_properties_l20_20469


namespace simplify_and_evaluate_expression_l20_20651

theorem simplify_and_evaluate_expression (x : ℤ) (hx : x = 3) : 
  (1 - (x / (x + 1))) / ((x^2 - 2 * x + 1) / (x^2 - 1)) = 1 / 2 := by
  rw [hx]
  -- Here we perform the necessary rewrites and simplifications as shown in the steps
  sorry

end simplify_and_evaluate_expression_l20_20651


namespace stratified_sampling_l20_20279

theorem stratified_sampling
  (students_class1 : ℕ)
  (students_class2 : ℕ)
  (formation_slots : ℕ)
  (total_students : ℕ)
  (prob_selected: ℚ)
  (selected_class1 : ℕ)
  (selected_class2 : ℕ)
  (h1 : students_class1 = 54)
  (h2 : students_class2 = 42)
  (h3 : formation_slots = 16)
  (h4 : total_students = students_class1 + students_class2)
  (h5 : prob_selected = formation_slots / total_students)
  (h6 : selected_class1 = students_class1 * prob_selected)
  (h7 : selected_class2 = students_class2 * prob_selected)
  : selected_class1 = 9 ∧ selected_class2 = 7 := by
  sorry

end stratified_sampling_l20_20279


namespace least_multiple_of_7_not_lucky_l20_20708

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_multiple_of_7_not_lucky : ∃ n, is_multiple_of_7 n ∧ ¬ is_lucky n ∧ n = 14 :=
by
  sorry

end least_multiple_of_7_not_lucky_l20_20708


namespace value_of_expression_l20_20461

theorem value_of_expression (x Q : ℝ) (π : Real) (h : 5 * (3 * x - 4 * π) = Q) : 10 * (6 * x - 8 * π) = 4 * Q :=
by 
  sorry

end value_of_expression_l20_20461


namespace sqrt_7_minus_a_l20_20264

theorem sqrt_7_minus_a (a : ℝ) (h : a = -1) : Real.sqrt (7 - a) = 2 * Real.sqrt 2 := by
  sorry

end sqrt_7_minus_a_l20_20264


namespace gcd_5800_14025_l20_20565

theorem gcd_5800_14025 : Int.gcd 5800 14025 = 25 := by
  sorry

end gcd_5800_14025_l20_20565


namespace proof_statement_B_proof_statement_D_proof_statement_E_l20_20725

def statement_B (x : ℝ) : Prop := x^2 = 0 → x = 0

def statement_D (x : ℝ) : Prop := x^2 < 2 * x → x > 0

def statement_E (x : ℝ) : Prop := x > 2 → x^2 > x

theorem proof_statement_B (x : ℝ) : statement_B x := sorry

theorem proof_statement_D (x : ℝ) : statement_D x := sorry

theorem proof_statement_E (x : ℝ) : statement_E x := sorry

end proof_statement_B_proof_statement_D_proof_statement_E_l20_20725


namespace arithmetic_geometric_inequality_l20_20501

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ := a1 + n * d

noncomputable def geometric_sequence (b1 r : ℝ) (n : ℕ) : ℝ := b1 * r^n

theorem arithmetic_geometric_inequality
  (a1 b1 : ℝ) (d r : ℝ) (n : ℕ)
  (h_pos : 0 < a1) 
  (ha1_eq_b1 : a1 = b1) 
  (h_eq_2np1 : arithmetic_sequence a1 d (2*n+1) = geometric_sequence b1 r (2*n+1)) :
  arithmetic_sequence a1 d (n+1) ≥ geometric_sequence b1 r (n+1) :=
sorry

end arithmetic_geometric_inequality_l20_20501


namespace simon_can_make_blueberry_pies_l20_20360

theorem simon_can_make_blueberry_pies (bush1 bush2 blueberries_per_pie : ℕ) (h1 : bush1 = 100) (h2 : bush2 = 200) (h3 : blueberries_per_pie = 100) : 
  (bush1 + bush2) / blueberries_per_pie = 3 :=
by
  -- Proof goes here
  sorry

end simon_can_make_blueberry_pies_l20_20360


namespace number_of_dodge_trucks_l20_20513

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l20_20513


namespace percentage_increase_x_y_l20_20204

theorem percentage_increase_x_y (Z Y X : ℝ) (h1 : Z = 300) (h2 : Y = 1.20 * Z) (h3 : X = 1110 - Y - Z) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_x_y_l20_20204


namespace volleyball_champion_probability_l20_20076

theorem volleyball_champion_probability
    (p : ℚ) 
    (A_needs_one_more_win : ℕ) 
    (B_needs_two_more_wins : ℕ) 
    (equal_win_probability : p = 1/2)
    (team_A_wins : ℚ)
    (prob_team_A_wins_first_game : team_A_wins = p)
    (prob_team_A_loses_first_wins_second : team_A_wins = p * p)
  : team_A_wins = 3 / 4 := by
  sorry

end volleyball_champion_probability_l20_20076


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20829

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20829


namespace projectile_hits_ground_at_5_over_2_l20_20800

theorem projectile_hits_ground_at_5_over_2 :
  ∃ t : ℚ, (-20) * t ^ 2 + 26 * t + 60 = 0 ∧ t = 5 / 2 :=
sorry

end projectile_hits_ground_at_5_over_2_l20_20800


namespace cars_without_features_l20_20645

theorem cars_without_features (total_cars cars_with_air_bags cars_with_power_windows cars_with_sunroofs 
                               cars_with_air_bags_and_power_windows cars_with_air_bags_and_sunroofs 
                               cars_with_power_windows_and_sunroofs cars_with_all_features: ℕ)
                               (h1 : total_cars = 80)
                               (h2 : cars_with_air_bags = 45)
                               (h3 : cars_with_power_windows = 40)
                               (h4 : cars_with_sunroofs = 25)
                               (h5 : cars_with_air_bags_and_power_windows = 20)
                               (h6 : cars_with_air_bags_and_sunroofs = 15)
                               (h7 : cars_with_power_windows_and_sunroofs = 10)
                               (h8 : cars_with_all_features = 8) : 
    total_cars - (cars_with_air_bags + cars_with_power_windows + cars_with_sunroofs 
                 - cars_with_air_bags_and_power_windows - cars_with_air_bags_and_sunroofs 
                 - cars_with_power_windows_and_sunroofs + cars_with_all_features) = 7 :=
by sorry

end cars_without_features_l20_20645


namespace inscribed_triangle_area_l20_20152

theorem inscribed_triangle_area 
  (r : ℝ) (theta : ℝ) 
  (A B C : ℝ) (arc1 arc2 arc3 : ℝ)
  (h_arc1 : arc1 = 5)
  (h_arc2 : arc2 = 7)
  (h_arc3 : arc3 = 8)
  (h_sum_arcs : arc1 + arc2 + arc3 = 2 * π * r)
  (h_theta : theta = 20)
  -- in radians: h_theta_rad : θ = (20 * π / 180)
  (h_A : A = 100)
  (h_B : B = 140)
  (h_C : C = 120) :
  let sin_A := sin (A * π / 180)
    sin_B := sin (B * π / 180)
    sin_C := sin (C * π / 180) in
  1 / 2 * (10 / π) ^ 2 * (sin_A + sin_B + sin_C) = 249.36 / π^2 := 
sorry

end inscribed_triangle_area_l20_20152


namespace triangle_base_angles_eq_l20_20349

theorem triangle_base_angles_eq
  (A B C C1 C2 : ℝ)
  (h1 : A > B)
  (h2 : C1 = 2 * C2)
  (h3 : A + B + C = 180)
  (h4 : B + C2 = 90)
  (h5 : C = C1 + C2) :
  A = B := by
  sorry

end triangle_base_angles_eq_l20_20349


namespace sqrt_product_eq_six_l20_20128

theorem sqrt_product_eq_six (sqrt24 sqrtThreeOverTwo: ℝ)
    (h1 : sqrt24 = Real.sqrt 24)
    (h2 : sqrtThreeOverTwo = Real.sqrt (3 / 2))
    : sqrt24 * sqrtThreeOverTwo = 6 := by
  sorry

end sqrt_product_eq_six_l20_20128


namespace abs_a_gt_neg_b_l20_20573

variable {a b : ℝ}

theorem abs_a_gt_neg_b (h : a < b ∧ b < 0) : |a| > -b :=
by
  sorry

end abs_a_gt_neg_b_l20_20573


namespace calc_x_equals_condition_l20_20551

theorem calc_x_equals_condition (m n p q x : ℝ) :
  x^2 + (2 * m * p + 2 * n * q) ^ 2 + (2 * m * q - 2 * n * p) ^ 2 = (m ^ 2 + n ^ 2 + p ^ 2 + q ^ 2) ^ 2 →
  x = m ^ 2 + n ^ 2 - p ^ 2 - q ^ 2 ∨ x = - m ^ 2 - n ^ 2 + p ^ 2 + q ^ 2 :=
by
  sorry

end calc_x_equals_condition_l20_20551


namespace probability_of_two_in_decimal_rep_of_eight_over_eleven_l20_20229

theorem probability_of_two_in_decimal_rep_of_eight_over_eleven : 
  (∃ B : List ℕ, (B = [7, 2]) ∧ (1 = (B.count 2) / (B.length)) ∧ 
  (0 + B.sum + 1) / 11 = 8 / 11) := sorry

end probability_of_two_in_decimal_rep_of_eight_over_eleven_l20_20229


namespace minimum_number_of_peanuts_l20_20742

/--
Five monkeys share a pile of peanuts.
Each monkey divides the peanuts into five piles, leaves one peanut which it eats, and takes away one pile.
This process continues in the same manner until the fifth monkey, who also evenly divides the remaining peanuts into five piles and has one peanut left over.
Prove that the minimum number of peanuts in the pile originally is 3121.
-/
theorem minimum_number_of_peanuts : ∃ N : ℕ, N = 3121 ∧
  (N - 1) % 5 = 0 ∧
  ((4 * ((N - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 4) % 5 = 0 :=
by
  sorry

end minimum_number_of_peanuts_l20_20742


namespace max_full_pikes_l20_20271

theorem max_full_pikes (initial_pikes : ℕ) (pike_full_condition : ℕ → Prop) (remaining_pikes : ℕ) 
  (h_initial : initial_pikes = 30)
  (h_condition : ∀ n, pike_full_condition n → n ≥ 3)
  (h_remaining : remaining_pikes ≥ 1) :
    ∃ max_full : ℕ, max_full ≤ 9 := 
sorry

end max_full_pikes_l20_20271


namespace range_of_x_l20_20073

theorem range_of_x (x : ℝ) : (x ≠ -3) ∧ (x ≤ 4) ↔ (x ≤ 4) ∧ (x ≠ -3) :=
by { sorry }

end range_of_x_l20_20073


namespace harkamal_purchase_mangoes_l20_20756

variable (m : ℕ)

def cost_of_grapes (cost_per_kg grapes_weight : ℕ) : ℕ := cost_per_kg * grapes_weight
def cost_of_mangoes (cost_per_kg mangoes_weight : ℕ) : ℕ := cost_per_kg * mangoes_weight

theorem harkamal_purchase_mangoes :
  (cost_of_grapes 70 10 + cost_of_mangoes 55 m = 1195) → m = 9 :=
by
  sorry

end harkamal_purchase_mangoes_l20_20756


namespace math_proof_problem_l20_20421

theorem math_proof_problem (a b : ℝ) (h1 : 64 = 8^2) (h2 : 16 = 8^2) :
  8^15 / (64^7) * 16 = 512 :=
by
  sorry

end math_proof_problem_l20_20421


namespace divides_necklaces_l20_20483

/-- Define the number of ways to make an even number of necklaces each of length at least 3. -/
def D_0 (n : ℕ) : ℕ := sorry

/-- Define the number of ways to make an odd number of necklaces each of length at least 3. -/
def D_1 (n : ℕ) : ℕ := sorry

/-- Main theorem: Prove that (n - 1) divides (D_1(n) - D_0(n)) for n ≥ 2 -/
theorem divides_necklaces (n : ℕ) (h : n ≥ 2) : (n - 1) ∣ (D_1 n - D_0 n) := sorry

end divides_necklaces_l20_20483


namespace exponential_equivalence_l20_20524

theorem exponential_equivalence (a : ℝ) : a^4 * a^4 = (a^2)^4 := by
  sorry

end exponential_equivalence_l20_20524


namespace original_group_men_l20_20840

-- Let's define the parameters of the problem
def original_days := 55
def absent_men := 15
def completed_days := 60

-- We need to show that the number of original men (x) is 180
theorem original_group_men (x : ℕ) (h : x * original_days = (x - absent_men) * completed_days) : x = 180 :=
by
  sorry

end original_group_men_l20_20840


namespace total_paintable_area_is_2006_l20_20293

-- Define the dimensions of the bedrooms and the hallway
def bedroom_length := 14
def bedroom_width := 11
def bedroom_height := 9

def hallway_length := 20
def hallway_width := 7
def hallway_height := 9

def num_bedrooms := 4
def doorway_window_area := 70

-- Compute the areas of the bedroom walls and the hallway walls
def bedroom_wall_area : ℕ :=
  2 * (bedroom_length * bedroom_height) +
  2 * (bedroom_width * bedroom_height)

def paintable_bedroom_wall_area : ℕ :=
  bedroom_wall_area - doorway_window_area

def total_paintable_bedroom_area : ℕ :=
  num_bedrooms * paintable_bedroom_wall_area

def hallway_wall_area : ℕ :=
  2 * (hallway_length * hallway_height) +
  2 * (hallway_width * hallway_height)

-- Compute the total paintable area
def total_paintable_area : ℕ :=
  total_paintable_bedroom_area + hallway_wall_area

-- Theorem stating the total paintable area is 2006 sq ft
theorem total_paintable_area_is_2006 : total_paintable_area = 2006 := 
  by
    unfold total_paintable_area
    rw [total_paintable_bedroom_area, paintable_bedroom_wall_area, bedroom_wall_area]
    rw [hallway_wall_area]
    norm_num
    sorry -- Proof omitted

end total_paintable_area_is_2006_l20_20293


namespace circular_garden_area_l20_20699

theorem circular_garden_area (r : ℝ) (A C : ℝ) (h_radius : r = 6) (h_relationship : C = (1 / 3) * A) 
  (h_circumference : C = 2 * Real.pi * r) (h_area : A = Real.pi * r ^ 2) : 
  A = 36 * Real.pi :=
by
  sorry

end circular_garden_area_l20_20699


namespace find_x_intercept_of_line_through_points_l20_20506

-- Definitions based on the conditions
def point1 : ℝ × ℝ := (-1, 1)
def point2 : ℝ × ℝ := (0, 3)

-- Statement: The x-intercept of the line passing through the given points is -3/2
theorem find_x_intercept_of_line_through_points :
  let x1 := point1.1
  let y1 := point1.2
  let x2 := point2.1
  let y2 := point2.2
  ∃ x_intercept : ℝ, x_intercept = -3 / 2 ∧ 
    (∀ x, ∀ y, (x2 - x1) * (y - y1) = (y2 - y1) * (x - x1) → y = 0 → x = x_intercept) :=
by
  sorry

end find_x_intercept_of_line_through_points_l20_20506


namespace problem_statement_l20_20886

theorem problem_statement (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (D : ℕ) (M : ℕ) (h_gcd : D = Nat.gcd (Nat.gcd a b) c) (h_lcm : M = Nat.lcm (Nat.lcm a b) c) :
  ((D * M = a * b * c) ∧ ((Nat.gcd a b = 1) ∧ (Nat.gcd b c = 1) ∧ (Nat.gcd a c = 1) → (D * M = a * b * c))) :=
by sorry

end problem_statement_l20_20886


namespace cos_sin_identity_l20_20446

theorem cos_sin_identity (x : ℝ) (h : Real.cos (x - Real.pi / 3) = 1 / 3) :
  Real.cos (2 * x - 5 * Real.pi / 3) + Real.sin (Real.pi / 3 - x) ^ 2 = 5 / 3 :=
sorry

end cos_sin_identity_l20_20446


namespace length_of_water_fountain_l20_20410

theorem length_of_water_fountain :
  (∀ (L1 : ℕ), 20 * 14 = L1) ∧
  (35 * 3 = 21) →
  (20 * 14 = 56) := by
sorry

end length_of_water_fountain_l20_20410


namespace skips_per_meter_l20_20934

variable (a b c d e f g h : ℕ)

theorem skips_per_meter 
  (hops_skips : a * skips = b * hops)
  (jumps_hops : c * jumps = d * hops)
  (leaps_jumps : e * leaps = f * jumps)
  (leaps_meters : g * leaps = h * meters) :
  1 * skips = (g * b * f * d) / (a * e * h * c) * skips := 
sorry

end skips_per_meter_l20_20934


namespace eval_expr_l20_20163

theorem eval_expr : 8^3 + 8^3 + 8^3 + 8^3 - 2^6 * 2^3 = 1536 := by
  -- Proof will go here
  sorry

end eval_expr_l20_20163


namespace least_number_to_subtract_l20_20819

theorem least_number_to_subtract (x : ℕ) (h : x = 7538 % 14) : (7538 - x) % 14 = 0 :=
by
  -- Proof goes here
  sorry

end least_number_to_subtract_l20_20819


namespace minimum_gloves_needed_l20_20937

-- Definitions based on conditions:
def participants : Nat := 43
def gloves_per_participant : Nat := 2

-- Problem statement proving the minimum number of gloves needed
theorem minimum_gloves_needed : participants * gloves_per_participant = 86 := by
  -- sorry allows us to omit the proof, focusing only on the formal statement
  sorry

end minimum_gloves_needed_l20_20937


namespace kite_area_is_192_l20_20042

-- Define the points with doubled dimensions
def A : (ℝ × ℝ) := (0, 16)
def B : (ℝ × ℝ) := (8, 24)
def C : (ℝ × ℝ) := (16, 16)
def D : (ℝ × ℝ) := (8, 0)

-- Calculate the area of the kite
noncomputable def kiteArea (A B C D : ℝ × ℝ) : ℝ :=
  let baseUpper := abs (C.1 - A.1)
  let heightUpper := abs (B.2 - A.2)
  let areaUpper := 1 / 2 * baseUpper * heightUpper
  let baseLower := baseUpper
  let heightLower := abs (B.2 - D.2)
  let areaLower := 1 / 2 * baseLower * heightLower
  areaUpper + areaLower

-- State the theorem to prove the kite area is 192 square inches
theorem kite_area_is_192 : kiteArea A B C D = 192 := 
  sorry

end kite_area_is_192_l20_20042


namespace sum_of_ratios_is_3_or_neg3_l20_20081

theorem sum_of_ratios_is_3_or_neg3 
  (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a / b + b / c + c / a : ℚ).den = 1 ) 
  (h5 : (b / a + c / b + a / c : ℚ).den = 1) :
  (a / b + b / c + c / a = 3 ∨ a / b + b / c + c / a = -3) ∧ 
  (b / a + c / b + a / c = 3 ∨ b / a + c / b + a / c = -3) := 
sorry

end sum_of_ratios_is_3_or_neg3_l20_20081


namespace renata_final_money_l20_20923

-- Defining the initial condition and the sequence of financial transactions.
def initial_money := 10
def donation := 4
def prize := 90
def slot_loss1 := 50
def slot_loss2 := 10
def slot_loss3 := 5
def water_cost := 1
def lottery_ticket_cost := 1
def lottery_prize := 65

-- Prove that given all these transactions, the final amount of money is $94.
theorem renata_final_money :
  initial_money 
  - donation 
  + prize 
  - slot_loss1 
  - slot_loss2 
  - slot_loss3 
  - water_cost 
  - lottery_ticket_cost 
  + lottery_prize 
  = 94 := 
by
  sorry

end renata_final_money_l20_20923


namespace combined_weight_l20_20385

theorem combined_weight (y z : ℝ) 
  (h_avg : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y + z) / 6) :
  y + z = 62 :=
by
  sorry

end combined_weight_l20_20385


namespace geom_sequence_2010th_term_l20_20367

theorem geom_sequence_2010th_term (p q r a_n : ℕ) (h1 : ∀ p q r, 3p / q = 3 * p * q)
  (h2: p * r = 9) (h3: 9 * r = 3 * p / q) (h4: r = 9 / p):
  a_n = 9 :=
by
  sorry

end geom_sequence_2010th_term_l20_20367


namespace factor_polynomial_l20_20571

theorem factor_polynomial : ∀ y : ℝ, 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) :=
by
  intros y
  sorry

end factor_polynomial_l20_20571


namespace compare_a_b_c_l20_20348

noncomputable def a := Real.sin (Real.pi / 5)
noncomputable def b := Real.logb (Real.sqrt 2) (Real.sqrt 3)
noncomputable def c := (1 / 4)^(2 / 3)

theorem compare_a_b_c : c < a ∧ a < b := by
  sorry

end compare_a_b_c_l20_20348


namespace john_steps_l20_20777

/-- John climbs up 9 flights of stairs. Each flight is 10 feet. -/
def flights := 9
def flight_height_feet := 10

/-- Conversion factor between feet and inches. -/
def feet_to_inches := 12

/-- Each step is 18 inches. -/
def step_height_inches := 18

/-- The total number of steps John climbs. -/
theorem john_steps :
  (flights * flight_height_feet * feet_to_inches) / step_height_inches = 60 :=
by
  sorry

end john_steps_l20_20777


namespace line_equation_l20_20448

open Real

theorem line_equation (x y : Real) : 
  (3 * x + 2 * y - 1 = 0) ↔ (y = (-(3 / 2)) * x + 2.5) :=
by
  sorry

end line_equation_l20_20448


namespace deposit_on_Jan_1_2008_l20_20354

-- Let a be the initial deposit amount in yuan.
-- Let x be the annual interest rate.

def compound_interest (a : ℝ) (x : ℝ) (n : ℕ) : ℝ :=
  a * (1 + x) ^ n

theorem deposit_on_Jan_1_2008 (a : ℝ) (x : ℝ) : 
  compound_interest a x 5 = a * (1 + x) ^ 5 := 
by
  sorry

end deposit_on_Jan_1_2008_l20_20354


namespace reading_rate_l20_20378

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l20_20378


namespace sequence_a_2011_l20_20480

noncomputable def sequence_a : ℕ → ℕ
| 0       => 2
| 1       => 3
| (n+2)   => (sequence_a (n+1) * sequence_a n) % 10

theorem sequence_a_2011 : sequence_a 2010 = 2 :=
by
  sorry

end sequence_a_2011_l20_20480


namespace mandy_cinnamon_nutmeg_difference_l20_20638

theorem mandy_cinnamon_nutmeg_difference :
  0.67 - 0.5 = 0.17 :=
by
  sorry

end mandy_cinnamon_nutmeg_difference_l20_20638


namespace second_discount_percentage_l20_20697

theorem second_discount_percentage (x : ℝ) :
  9356.725146198829 * 0.8 * (1 - x / 100) * 0.95 = 6400 → x = 10 :=
by
  sorry

end second_discount_percentage_l20_20697


namespace proof_of_problem_l20_20750

variable (f : ℝ → ℝ)
variable (h_nonzero : ∀ x, f x ≠ 0)
variable (h_equation : ∀ x y, f (x * y) = y * f x + x * f y)

theorem proof_of_problem :
  f 1 = 0 ∧ f (-1) = 0 ∧ (∀ x, f (-x) = -f x) :=
by
  sorry

end proof_of_problem_l20_20750


namespace max_value_of_expression_l20_20343

theorem max_value_of_expression (x y : ℝ) (h : x + y = 4) :
  x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 ≤ 7225 / 28 :=
sorry

end max_value_of_expression_l20_20343


namespace total_population_estimate_l20_20767

def average_population_min : ℕ := 3200
def average_population_max : ℕ := 3600
def towns : ℕ := 25

theorem total_population_estimate : 
    ∃ x : ℕ, average_population_min ≤ x ∧ x ≤ average_population_max ∧ towns * x = 85000 :=
by 
  sorry

end total_population_estimate_l20_20767


namespace quadratic_function_negative_values_l20_20618

theorem quadratic_function_negative_values (a : ℝ) : 
  (∃ x : ℝ, (x^2 - a*x + 1) < 0) ↔ (a > 2 ∨ a < -2) :=
by
  sorry

end quadratic_function_negative_values_l20_20618


namespace stratified_sampling_expected_elderly_chosen_l20_20118

theorem stratified_sampling_expected_elderly_chosen :
  let total := 165
  let to_choose := 15
  let elderly := 22
  (22 : ℚ) / 165 * 15 = 2 := sorry

end stratified_sampling_expected_elderly_chosen_l20_20118


namespace union_of_A_B_l20_20222

open Set

variable {α : Type*} [LinearOrder α]

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | 1 < x ∧ x < 3 }

theorem union_of_A_B : A ∪ B = { x | -1 < x ∧ x < 3 } :=
sorry

end union_of_A_B_l20_20222


namespace min_value_quadratic_expr_l20_20678

-- Define the quadratic function
def quadratic_expr (x : ℝ) : ℝ := 8 * x^2 - 24 * x + 1729

-- State the theorem to prove the minimum value
theorem min_value_quadratic_expr : (∃ x : ℝ, ∀ y : ℝ, quadratic_expr y ≥ quadratic_expr x) ∧ ∃ x : ℝ, quadratic_expr x = 1711 :=
by
  -- The proof will go here
  sorry

end min_value_quadratic_expr_l20_20678


namespace expected_number_of_defective_products_l20_20417

theorem expected_number_of_defective_products 
  (N : ℕ) (D : ℕ) (n : ℕ) (hN : N = 15000) (hD : D = 1000) (hn : n = 150) :
  n * (D / N : ℚ) = 10 := 
by {
  sorry
}

end expected_number_of_defective_products_l20_20417


namespace sufficient_but_not_necessary_l20_20406

theorem sufficient_but_not_necessary (m : ℕ) :
  m = 9 → m > 8 ∧ ∃ k : ℕ, k > 8 ∧ k ≠ 9 :=
by
  sorry

end sufficient_but_not_necessary_l20_20406


namespace domain_of_f_l20_20799

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt (x - 3)) / (abs (x + 1) - 5)

theorem domain_of_f :
  {x : ℝ | x - 3 ≥ 0 ∧ abs (x + 1) - 5 ≠ 0} = {x : ℝ | (3 ≤ x ∧ x < 4) ∨ (4 < x)} :=
by
  sorry

end domain_of_f_l20_20799


namespace rectangle_ratio_l20_20040

theorem rectangle_ratio (s : ℝ) (x y : ℝ) 
  (h_outer_area : x * y * 4 + s^2 = 9 * s^2)
  (h_inner_outer_relation : s + 2 * y = 3 * s) :
  x / y = 2 :=
by {
  sorry
}

end rectangle_ratio_l20_20040


namespace cheryl_same_color_probability_l20_20008

/-- Defines the probability of Cheryl picking 3 marbles of the same color from the given box setup. -/
def probability_cheryl_picks_same_color : ℚ :=
  let total_ways := (Nat.choose 9 3) * (Nat.choose 6 3) * (Nat.choose 3 3)
  let favorable_ways := 3 * (Nat.choose 6 3)
  (favorable_ways : ℚ) / (total_ways : ℚ)

/-- Theorem stating the probability that Cheryl picks 3 marbles of the same color is 1/28. -/
theorem cheryl_same_color_probability :
  probability_cheryl_picks_same_color = 1 / 28 :=
by
  sorry

end cheryl_same_color_probability_l20_20008


namespace no_equal_differences_between_products_l20_20352

theorem no_equal_differences_between_products (a b c d : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
    (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_order : a < b ∧ b < c ∧ c < d) :
    ¬ (∃ k : ℕ, ac - ab = k ∧ ad - ac = k ∧ bc - ad = k ∧ bd - bc = k ∧ cd - bd = k) :=
by
  sorry

end no_equal_differences_between_products_l20_20352


namespace Laura_pays_more_l20_20631

theorem Laura_pays_more 
  (slices : ℕ) 
  (cost_plain : ℝ) 
  (cost_mushrooms : ℝ) 
  (laura_mushroom_slices : ℕ) 
  (laura_plain_slices : ℕ) 
  (jessica_plain_slices: ℕ) :
  slices = 12 →
  cost_plain = 12 →
  cost_mushrooms = 3 →
  laura_mushroom_slices = 4 →
  laura_plain_slices = 2 →
  jessica_plain_slices = 6 →
  15 / 12 * (laura_mushroom_slices + laura_plain_slices) - 
  (cost_plain / 12 * jessica_plain_slices) = 1.5 :=
by
  intro slices_eq
  intro cost_plain_eq
  intro cost_mushrooms_eq
  intro laura_mushroom_slices_eq
  intro laura_plain_slices_eq
  intro jessica_plain_slices_eq
  sorry

end Laura_pays_more_l20_20631


namespace neither_happy_nor_sad_boys_is_5_l20_20227

-- Define the total number of children
def total_children := 60

-- Define the number of happy children
def happy_children := 30

-- Define the number of sad children
def sad_children := 10

-- Define the number of neither happy nor sad children
def neither_happy_nor_sad_children := 20

-- Define the number of boys
def boys := 17

-- Define the number of girls
def girls := 43

-- Define the number of happy boys
def happy_boys := 6

-- Define the number of sad girls
def sad_girls := 4

-- Define the number of neither happy nor sad boys
def neither_happy_nor_sad_boys := boys - (happy_boys + (sad_children - sad_girls))

theorem neither_happy_nor_sad_boys_is_5 :
  neither_happy_nor_sad_boys = 5 :=
by
  -- This skips the proof
  sorry

end neither_happy_nor_sad_boys_is_5_l20_20227


namespace disease_given_positive_test_l20_20500

variable (D T : Event)
variable [ProbabilityMeasure Ω] [MeasureSpace Ω]

-- Conditions
variable (h_PD : Pr(D) = 1 / 1000)
variable (h_PT_D : Pr(T | D) = 1)
variable (D_compl : Event) (h_D_compl : D_compl = Dᶜ)
variable (h_PT_Dcomplement : Pr(T | D_compl) = 0.05)

-- Goal
theorem disease_given_positive_test :
  Pr(D | T) = 100 / 5095 :=
by
  sorry

end disease_given_positive_test_l20_20500


namespace solve_trig_equation_l20_20362

theorem solve_trig_equation (x : ℝ) : 
  (∃ (k : ℤ), x = (Real.pi / 16) * (4 * k + 1)) ↔ 2 * (Real.cos (4 * x) - Real.sin x * Real.cos (3 * x)) = Real.sin (4 * x) + Real.sin (2 * x) :=
by
  -- The full proof detail goes here.
  sorry

end solve_trig_equation_l20_20362


namespace time_difference_l20_20910

-- Define the conditions
def time_to_nile_delta : Nat := 4
def number_of_alligators : Nat := 7
def combined_walking_time : Nat := 46

-- Define the mathematical statement we want to prove
theorem time_difference (x : Nat) :
  4 + 7 * (time_to_nile_delta + x) = combined_walking_time → x = 2 :=
by
  sorry

end time_difference_l20_20910


namespace function_satisfies_conditions_l20_20050

-- Define the conditions
def f (n : ℕ) : ℕ := n + 1

-- Prove that the function f satisfies the given conditions
theorem function_satisfies_conditions : 
  (f 0 = 1) ∧ (f 2012 = 2013) :=
by
  sorry

end function_satisfies_conditions_l20_20050


namespace smaller_circle_radius_l20_20332

theorem smaller_circle_radius
  (radius_largest : ℝ)
  (h1 : radius_largest = 10)
  (aligned_circles : ℝ)
  (h2 : 4 * aligned_circles = 2 * radius_largest) :
  aligned_circles / 2 = 2.5 :=
by
  sorry

end smaller_circle_radius_l20_20332


namespace range_of_x_l20_20074

theorem range_of_x : ∀ x : ℝ, (¬ (x + 3 = 0)) ∧ (4 - x ≥ 0) ↔ x ≤ 4 ∧ x ≠ -3 := by
  sorry

end range_of_x_l20_20074


namespace dog_weight_ratio_l20_20286

theorem dog_weight_ratio
  (w7 : ℕ) (r : ℕ) (w13 : ℕ) (w21 : ℕ) (w52 : ℕ):
  (w7 = 6) →
  (w13 = 12 * r) →
  (w21 = 2 * w13) →
  (w52 = w21 + 30) →
  (w52 = 78) →
  r = 2 :=
by 
  sorry

end dog_weight_ratio_l20_20286


namespace classify_event_l20_20813

-- Define the conditions of the problem
def involves_variables_and_uncertainties (event: String) : Prop := 
  event = "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'"

-- Define the type of event as a string
def event_type : String := "random"

-- The theorem to prove the classification of the event
theorem classify_event : involves_variables_and_uncertainties "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'" →
  event_type = "random" :=
by
  intro h
  -- Proof is skipped
  sorry

end classify_event_l20_20813


namespace expressions_inequivalence_l20_20729

theorem expressions_inequivalence (x : ℝ) (h : x > 0) :
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (x + 1) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (x + 1) ^ (2 * x + 2)) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ 2 * (0.5 * x + x) ^ x) ∧ 
  (∀ x > 0, 2 * (x + 1) ^ (x + 1) ≠ (2 * x + 2) ^ (2 * x + 2)) := by
  sorry

end expressions_inequivalence_l20_20729


namespace jill_first_show_length_l20_20772

theorem jill_first_show_length : 
  ∃ (x : ℕ), (x + 4 * x = 150) ∧ (x = 30) :=
sorry

end jill_first_show_length_l20_20772


namespace solve_for_y_l20_20467

variable (k y : ℝ)

-- Define the first equation for x
def eq1 (x : ℝ) : Prop := (1 / 2023) * x - 2 = 3 * x + k

-- Define the condition that x = -5 satisfies eq1
def condition1 : Prop := eq1 k (-5)

-- Define the second equation for y
def eq2 : Prop := (1 / 2023) * (2 * y + 1) - 5 = 6 * y + k

-- Prove that given condition1, y = -3 satisfies eq2
theorem solve_for_y : condition1 k → eq2 k (-3) :=
sorry

end solve_for_y_l20_20467


namespace which_polygon_covers_ground_l20_20965

def is_tessellatable (n : ℕ) : Prop :=
  let interior_angle := (n - 2) * 180 / n
  360 % interior_angle = 0

theorem which_polygon_covers_ground :
  is_tessellatable 6 ∧ ¬is_tessellatable 5 ∧ ¬is_tessellatable 8 ∧ ¬is_tessellatable 12 :=
by
  sorry

end which_polygon_covers_ground_l20_20965


namespace smallest_gcd_value_l20_20198

theorem smallest_gcd_value (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : Nat.gcd m n = 8) : Nat.gcd (8 * m) (12 * n) = 32 :=
by
  sorry

end smallest_gcd_value_l20_20198


namespace distance_to_city_center_l20_20243

theorem distance_to_city_center 
  (D : ℕ) 
  (H1 : D = 200 + 200 + D) 
  (H_total : 900 = 200 + 200 + D) : 
  D = 500 :=
by { sorry }

end distance_to_city_center_l20_20243


namespace metal_rods_per_sheet_l20_20275

theorem metal_rods_per_sheet :
  (∀ (metal_rod_for_sheets metal_rod_for_beams total_metal_rod num_sheet_per_panel num_panel num_rod_per_beam),
    (num_rod_per_beam = 4) →
    (total_metal_rod = 380) →
    (metal_rod_for_beams = num_panel * (2 * num_rod_per_beam)) →
    (metal_rod_for_sheets = total_metal_rod - metal_rod_for_beams) →
    (num_sheet_per_panel = 3) →
    (num_panel = 10) →
    (metal_rod_per_sheet = metal_rod_for_sheets / (num_panel * num_sheet_per_panel)) →
    metal_rod_per_sheet = 10
  ) := sorry

end metal_rods_per_sheet_l20_20275


namespace commute_time_x_l20_20030

theorem commute_time_x (d : ℝ) (walk_speed : ℝ) (train_speed : ℝ) (extra_time : ℝ) (diff_time : ℝ) :
  d = 1.5 →
  walk_speed = 3 →
  train_speed = 20 →
  diff_time = 10 →
  (diff_time : ℝ) * 60 = (d / walk_speed - (d / train_speed + extra_time / 60)) * 60 →
  extra_time = 15.5 :=
by
  sorry

end commute_time_x_l20_20030


namespace evaluate_expression_l20_20437

theorem evaluate_expression : (723 * 723) - (722 * 724) = 1 :=
by
  sorry

end evaluate_expression_l20_20437


namespace part_a_l20_20470

theorem part_a (cities : Finset (ℝ × ℝ)) (h_cities : cities.card = 100) 
  (distances : Finset (ℝ × ℝ → ℝ)) (h_distances : distances.card = 4950) :
  ∃ (erased_distance : ℝ × ℝ → ℝ), ¬ ∃ (restored_distance : ℝ × ℝ → ℝ), 
    restored_distance = erased_distance :=
sorry

end part_a_l20_20470


namespace find_other_number_l20_20656

theorem find_other_number (a b : ℕ) (h1 : (a + b) / 2 = 7) (h2 : a = 5) : b = 9 :=
by
  sorry

end find_other_number_l20_20656


namespace find_number_l20_20398

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l20_20398


namespace wage_increase_l20_20247

-- Definition: Regression line equation
def regression_line (x : ℝ) : ℝ := 80 * x + 50

-- Theorem: On average, when the labor productivity increases by 1000 yuan, the wage increases by 80 yuan
theorem wage_increase (x : ℝ) : regression_line (x + 1) - regression_line x = 80 :=
by
  sorry

end wage_increase_l20_20247


namespace fractions_are_integers_l20_20921

theorem fractions_are_integers (x y : ℕ) 
    (h : ∃ k : ℤ, (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) = k) :
    ∃ u v : ℤ, (x^2 - 1) = u * (y + 1) ∧ (y^2 - 1) = v * (x + 1) := 
by
  sorry

end fractions_are_integers_l20_20921


namespace sum_value_l20_20384

variable (T R S PV : ℝ)
variable (TD SI : ℝ) (h_td : TD = 80) (h_si : SI = 88)
variable (h1 : SI = TD + (TD * R * T) / 100)
variable (h2 : (PV * R * T) / 100 = TD)
variable (h3 : PV = S - TD)
variable (h4 : R * T = 10)

theorem sum_value : S = 880 := by
  sorry

end sum_value_l20_20384


namespace cats_after_purchasing_l20_20488

/-- Mrs. Sheridan's total number of cats after purchasing more -/
theorem cats_after_purchasing (a b : ℕ) (h₀ : a = 11) (h₁ : b = 43) : a + b = 54 := by
  sorry

end cats_after_purchasing_l20_20488


namespace evaluate_expression_l20_20452

variable (x y : ℝ)
variable (h₀ : x ≠ 0)
variable (h₁ : y ≠ 0)
variable (h₂ : 5 * x ≠ 3 * y)

theorem evaluate_expression : 
  (5 * x - 3 * y)⁻¹ * ((5 * x)⁻¹ - (3 * y)⁻¹) = -1 / (15 * x * y) :=
sorry

end evaluate_expression_l20_20452


namespace pumpkin_weight_difference_l20_20110

variable (Brad_weight Jessica_weight Betty_weight : ℕ)

theorem pumpkin_weight_difference :
  Brad_weight = 54 →
  Jessica_weight = Brad_weight / 2 →
  Betty_weight = 4 * Jessica_weight →
  Betty_weight - Jessica_weight = 81 := by
  sorry

end pumpkin_weight_difference_l20_20110


namespace floor_equation_solution_l20_20171

theorem floor_equation_solution (x : ℝ) :
  (⌊⌊3 * x⌋ + 1/3⌋ = ⌊x + 5⌋) ↔ (7/3 ≤ x ∧ x < 3) := 
sorry

end floor_equation_solution_l20_20171


namespace sum_a1_to_a12_l20_20330

variable {a : ℕ → ℕ}

axiom geom_seq (n : ℕ) : a n * a (n + 1) * a (n + 2) = 8
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2

theorem sum_a1_to_a12 : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_a1_to_a12_l20_20330


namespace eq_infinite_solutions_pos_int_l20_20034

noncomputable def eq_has_inf_solutions_in_positive_integers (m : ℕ) : Prop :=
    ∀ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 → 
    ∃ (a' b' c' : ℕ), 
    a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    (1 / a + 1 / b + 1 / c + 1 / (a * b * c) = m / (a + b + c))

theorem eq_infinite_solutions_pos_int (m : ℕ) (hm : m > 0) : eq_has_inf_solutions_in_positive_integers m := 
by 
  sorry

end eq_infinite_solutions_pos_int_l20_20034


namespace ad_equals_two_l20_20586

noncomputable def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem ad_equals_two (a b c d : ℝ) 
  (h1 : geometric_sequence a b c d) 
  (h2 : ∃ (b c : ℝ), (1, 2) = (b, c) ∧ b = 1 ∧ c = 2) :
  a * d = 2 :=
by
  sorry

end ad_equals_two_l20_20586


namespace cheapest_pie_cost_is_18_l20_20556

noncomputable def crust_cost : ℝ := 2 + 1 + 1.5
noncomputable def blueberry_container_cost : ℝ := 2.25
noncomputable def blueberry_containers_needed : ℕ := 3 * (16 / 8)
noncomputable def blueberry_filling_cost : ℝ := blueberry_containers_needed * blueberry_container_cost
noncomputable def cherry_filling_cost : ℝ := 14
noncomputable def cheapest_filling_cost : ℝ := min blueberry_filling_cost cherry_filling_cost
noncomputable def total_cheapest_pie_cost : ℝ := crust_cost + cheapest_filling_cost

theorem cheapest_pie_cost_is_18 : total_cheapest_pie_cost = 18 := by
  sorry

end cheapest_pie_cost_is_18_l20_20556


namespace egg_problem_l20_20841

theorem egg_problem :
  ∃ (N F E : ℕ), N + F + E = 100 ∧ 5 * N + F + E / 2 = 100 ∧ (N = F ∨ N = E ∨ F = E) ∧ N = 10 ∧ F = 10 ∧ E = 80 :=
by
  sorry

end egg_problem_l20_20841


namespace solve_a_l20_20757

variable (a : ℝ)

theorem solve_a (h : ∃ b : ℝ, (9 * x^2 + 12 * x + a) = (3 * x + b) ^ 2) : a = 4 :=
by
   sorry

end solve_a_l20_20757


namespace M_subset_P_l20_20338

def M := {x : ℕ | ∃ a : ℕ, 0 < a ∧ x = a^2 + 1}
def P := {y : ℕ | ∃ b : ℕ, 0 < b ∧ y = b^2 - 4*b + 5}

theorem M_subset_P : M ⊂ P :=
by
  sorry

end M_subset_P_l20_20338


namespace quadratic_inequality_l20_20370

theorem quadratic_inequality (x : ℝ) : x^2 - x + 1 ≥ 0 :=
sorry

end quadratic_inequality_l20_20370


namespace number_of_outliers_l20_20241

def data_set : List ℕ := [4, 23, 27, 27, 35, 37, 37, 39, 47, 53]

def Q1 : ℕ := 27
def Q3 : ℕ := 39

def IQR : ℕ := Q3 - Q1
def lower_threshold : ℕ := Q1 - (3 * IQR / 2)
def upper_threshold : ℕ := Q3 + (3 * IQR / 2)

def outliers (s : List ℕ) (low high : ℕ) : List ℕ :=
  s.filter (λ x => x < low ∨ x > high)

theorem number_of_outliers :
  outliers data_set lower_threshold upper_threshold = [4] :=
by
  sorry

end number_of_outliers_l20_20241


namespace polyhedron_inequality_proof_l20_20473

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l20_20473


namespace value_of_a_l20_20182

def f (a x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem value_of_a (a : ℝ) (h : f_prime a (-1) = 4) : a = 10 / 3 :=
by
  -- Proof goes here
  sorry

end value_of_a_l20_20182


namespace smaller_circle_area_l20_20517

theorem smaller_circle_area (r R : ℝ) (hR : R = 3 * r)
  (hTangentLines : ∀ (P A B A' B' : ℝ), P = 5 ∧ A = 5 ∧ PA = 5 ∧ A' = 5 ∧ PA' = 5 ∧ AB = 5 ∧ A'B' = 5 ) :
  π * r^2 = 25 / 3 * π := by
  sorry

end smaller_circle_area_l20_20517


namespace fraction_division_l20_20676

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l20_20676


namespace convex_pentagons_l20_20438

theorem convex_pentagons (P : Finset ℝ) (h : P.card = 15) : 
  (P.card.choose 5) = 3003 := 
by
  sorry

end convex_pentagons_l20_20438


namespace problem_solution_l20_20759

theorem problem_solution (m n p : ℝ) 
  (h1 : 1 * m + 4 * p - 2 = 0) 
  (h2 : 2 * 1 - 5 * p + n = 0) 
  (h3 : (m / (-4)) * (2 / 5) = -1) :
  n = -12 :=
sorry

end problem_solution_l20_20759


namespace eval_complex_fraction_expr_l20_20295

def complex_fraction_expr : ℚ :=
  2 + (3 / (4 + (5 / (6 + (7 / 8)))))

theorem eval_complex_fraction_expr : complex_fraction_expr = 137 / 52 :=
by
  -- we skip the actual proof but ensure it can build successfully.
  sorry

end eval_complex_fraction_expr_l20_20295


namespace workshop_output_comparison_l20_20481

theorem workshop_output_comparison (a x : ℝ)
  (h1 : ∀n:ℕ, n ≥ 0 → (1 + n * a) = (1 + x)^n) :
  (1 + 3 * a) > (1 + x)^3 := sorry

end workshop_output_comparison_l20_20481


namespace inches_of_rain_received_so_far_l20_20329

def total_days_in_year : ℕ := 365
def days_left_in_year : ℕ := 100
def rain_per_day_initial_avg : ℝ := 2
def rain_per_day_required_avg : ℝ := 3

def total_annually_expected_rain : ℝ := rain_per_day_initial_avg * total_days_in_year
def days_passed_in_year : ℕ := total_days_in_year - days_left_in_year
def total_rain_needed_remaining : ℝ := rain_per_day_required_avg * days_left_in_year

variable (S : ℝ) -- inches of rain received so far

theorem inches_of_rain_received_so_far (S : ℝ) :
  S + total_rain_needed_remaining = total_annually_expected_rain → S = 430 :=
  by
  sorry

end inches_of_rain_received_so_far_l20_20329


namespace max_knights_is_seven_l20_20536

-- Definitions of conditions
def students : ℕ := 11
def total_statements : ℕ := students * (students - 1)
def liar_statements : ℕ := 56

-- Definition translating the problem statement
theorem max_knights_is_seven : ∃ (k li : ℕ), 
  (k + li = students) ∧ 
  (k * li = liar_statements) ∧ 
  (k = 7) := 
by
  sorry

end max_knights_is_seven_l20_20536


namespace pythagorean_triple_fits_l20_20278

theorem pythagorean_triple_fits 
  (k : ℤ) (n : ℤ) : 
  (∃ k, (n = 5 * k ∨ n = 12 * k ∨ n = 13 * k) ∧ 
      (n = 62 ∨ n = 96 ∨ n = 120 ∨ n = 91 ∨ n = 390)) ↔ 
      (n = 120 ∨ n = 91) := by 
  sorry

end pythagorean_triple_fits_l20_20278


namespace multiples_of_7_between_15_and_200_l20_20889

theorem multiples_of_7_between_15_and_200 : ∃ n : ℕ, n = 26 ∧ ∃ (a₁ a_n d : ℕ), 
  a₁ = 21 ∧ a_n = 196 ∧ d = 7 ∧ (a₁ + (n - 1) * d = a_n) := 
by
  sorry

end multiples_of_7_between_15_and_200_l20_20889


namespace calc_a_minus_3b_l20_20261

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end calc_a_minus_3b_l20_20261


namespace triangle_ineq_l20_20062

noncomputable def TriangleSidesProof (AB AC BC : ℝ) :=
  AB = AC ∧ BC = 10 ∧ 2 * AB + BC ≤ 44 → 5 < AB ∧ AB ≤ 17

-- Statement for the proof problem
theorem triangle_ineq (AB AC BC : ℝ) (h1 : AB = AC) (h2 : BC = 10) (h3 : 2 * AB + BC ≤ 44) :
  5 < AB ∧ AB ≤ 17 :=
sorry

end triangle_ineq_l20_20062


namespace work_completion_time_l20_20273

theorem work_completion_time (days_B days_C days_all : ℝ) (h_B : days_B = 5) (h_C : days_C = 12) (h_all : days_all = 2.2222222222222223) : 
    (1 / ((days_all / 9) * 10) - 1 / days_B - 1 / days_C)⁻¹ = 60 / 37 := by 
  sorry

end work_completion_time_l20_20273


namespace intersection_point_l20_20634

noncomputable def f (x : ℝ) : ℝ := (x^2 - 4 * x + 3) / (2 * x - 6)
noncomputable def g (a b c x : ℝ) : ℝ := (a * x^2 + b * x + c) / (x - 3)

theorem intersection_point (a b c : ℝ) (h_asymp : ¬(2 = 0) ∧ (a ≠ 0 ∨ b ≠ 0)) (h_perpendicular : True) (h_y_intersect : g a b c 0 = 1) (h_intersects : f (-1) = g a b c (-1)):
  f 1 = 0 :=
by
  dsimp [f, g] at *
  sorry

end intersection_point_l20_20634


namespace range_of_x_l20_20075

theorem range_of_x : ∀ x : ℝ, (¬ (x + 3 = 0)) ∧ (4 - x ≥ 0) ↔ x ≤ 4 ∧ x ≠ -3 := by
  sorry

end range_of_x_l20_20075


namespace pirate_treasure_probability_l20_20711

theorem pirate_treasure_probability :
  let p_treasure := 1 / 5
  let p_trap_no_treasure := 1 / 10
  let p_notreasure_notrap := 7 / 10
  let combinatorial_factor := Nat.choose 8 4
  let probability := (combinatorial_factor * (p_treasure ^ 4) * (p_notreasure_notrap ^ 4))
  probability = 33614 / 1250000 :=
by
  sorry

end pirate_treasure_probability_l20_20711


namespace largest_tile_size_l20_20145

def length_cm : ℕ := 378
def width_cm : ℕ := 525

theorem largest_tile_size :
  Nat.gcd length_cm width_cm = 21 := by
  sorry

end largest_tile_size_l20_20145


namespace find_probability_of_B_l20_20626

-- Define the conditions and the problem
def system_A_malfunction_prob := 1 / 10
def at_least_one_not_malfunction_prob := 49 / 50

/-- The probability that System B malfunctions given that 
  the probability of at least one system not malfunctioning 
  is 49/50 and the probability of System A malfunctioning is 1/10 -/
theorem find_probability_of_B (p : ℝ) 
  (h1 : system_A_malfunction_prob = 1 / 10) 
  (h2 : at_least_one_not_malfunction_prob = 49 / 50) 
  (h3 : 1 - (system_A_malfunction_prob * p) = at_least_one_not_malfunction_prob) : 
  p = 1 / 5 :=
sorry

end find_probability_of_B_l20_20626


namespace three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l20_20116

theorem three_hundred_thousand_times_three_hundred_thousand_minus_one_million :
  (300000 * 300000) - 1000000 = 89990000000 := by
  sorry 

end three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l20_20116


namespace percentage_problem_l20_20463

theorem percentage_problem (x : ℝ) (h : 0.3 * 0.4 * x = 45) : 0.4 * 0.3 * x = 45 :=
by
  sorry

end percentage_problem_l20_20463


namespace crumbs_triangle_area_l20_20788

theorem crumbs_triangle_area :
  ∀ (table_length table_width : ℝ) (crumbs : ℕ),
    table_length = 2 ∧ table_width = 1 ∧ crumbs = 500 →
    ∃ (triangle_area : ℝ), (triangle_area < 0.005 ∧ ∃ (a b c : Type), a ≠ b ∧ b ≠ c ∧ a ≠ c) :=
by
  sorry

end crumbs_triangle_area_l20_20788


namespace volume_of_one_piece_l20_20146

open Real

noncomputable def pizza_thickness : ℝ := 1 / 2
noncomputable def pizza_diameter : ℝ := 18
noncomputable def pizza_radius : ℝ := pizza_diameter / 2
noncomputable def number_of_pieces : ℕ := 16
noncomputable def total_volume : ℝ := π * (pizza_radius ^ 2) * pizza_thickness

theorem volume_of_one_piece : 
  (total_volume / number_of_pieces) = (2.53125 * π) := by
  -- proof omitted
  sorry

end volume_of_one_piece_l20_20146


namespace compute_fraction_pow_mul_l20_20169

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l20_20169


namespace trajectory_of_P_below_x_axis_l20_20985

theorem trajectory_of_P_below_x_axis (x y : ℝ) (P_below_x_axis : y < 0)
    (tangent_to_parabola : ∃ A B: ℝ × ℝ, A.1^2 = 2 * A.2 ∧ B.1^2 = 2 * B.2 ∧ (x^2 + y^2 = 1))
    (AB_tangent_to_circle : ∀ (x0 y0 : ℝ), x0^2 + y0^2 = 1 → x0 * x + y0 * y = 1) :
    y^2 - x^2 = 1 :=
sorry

end trajectory_of_P_below_x_axis_l20_20985


namespace meet_second_time_4_5_minutes_l20_20572

-- Define the initial conditions
def opposite_ends := true      -- George and Henry start from opposite ends
def pass_in_center := 1.5      -- They pass each other in the center after 1.5 minutes
def no_time_lost := true       -- No time lost in turning
def constant_speeds := true    -- They maintain their respective speeds

-- Prove that they pass each other the second time after 4.5 minutes
theorem meet_second_time_4_5_minutes :
  opposite_ends ∧ pass_in_center = 1.5 ∧ no_time_lost ∧ constant_speeds → 
  ∃ t : ℝ, t = 4.5 := by
  sorry

end meet_second_time_4_5_minutes_l20_20572


namespace budget_left_equals_16_l20_20854

def initial_budget : ℤ := 200
def expense_shirt : ℤ := 30
def expense_pants : ℤ := 46
def expense_coat : ℤ := 38
def expense_socks : ℤ := 11
def expense_belt : ℤ := 18
def expense_shoes : ℤ := 41

def total_expenses : ℤ := 
  expense_shirt + expense_pants + expense_coat + expense_socks + expense_belt + expense_shoes

def budget_left : ℤ := initial_budget - total_expenses

theorem budget_left_equals_16 : 
  budget_left = 16 := by
  sorry

end budget_left_equals_16_l20_20854


namespace trajectory_midpoint_l20_20557

theorem trajectory_midpoint (P M D : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 16) (hD : D = (P.1, 0)) (hM : M = ((P.1 + D.1)/2, (P.2 + D.2)/2)) :
  (M.1 ^ 2) / 4 + (M.2 ^ 2) / 16 = 1 :=
by
  sorry

end trajectory_midpoint_l20_20557


namespace bug_move_probability_l20_20692

noncomputable def probability_bug_visits_all_vertices_in_seven_moves : ℚ :=
  8 / 729

theorem bug_move_probability :
  (∀ v : ℕ × ℕ × ℕ, v ∈ {(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)} →
   ∀ n : ℕ, n = 7 →
   ∀ start_v end_v : ℕ × ℕ × ℕ, 
     start_v ≠ end_v ∧ ((start_v.1 + start_v.2 + start_v.3) % 2 ≠ (end_v.1 + end_v.2 + end_v.3) % 2)
   ∧ 
   (∀ i : ℕ, i < n → 
     (start_v.1 + start_v.2 + start_v.3) % 2 ≠ (start_v.1 + start_v.2 + start_v.3) % 2 →
     -- condition to ensure alternating parity and visiting all vertices
    /* actual path definitions and alternations omitted */) →
   probability_bug_visits_all_vertices_in_seven_moves = 8 / 729 :=
begin
  sorry
end

end bug_move_probability_l20_20692


namespace number_of_white_tiles_l20_20476

-- Definition of conditions in the problem
def side_length_large_square := 81
def area_large_square := side_length_large_square * side_length_large_square
def area_black_tiles := 81
def num_red_tiles := 154
def area_red_tiles := num_red_tiles * 4
def area_covered_by_black_and_red := area_black_tiles + area_red_tiles
def remaining_area_for_white_tiles := area_large_square - area_covered_by_black_and_red
def area_of_one_white_tile := 2
def expected_num_white_tiles := 2932

-- The theorem to prove
theorem number_of_white_tiles :
  remaining_area_for_white_tiles / area_of_one_white_tile = expected_num_white_tiles :=
by
  sorry

end number_of_white_tiles_l20_20476


namespace complement_A_union_B_l20_20605

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l20_20605


namespace alex_average_speed_l20_20154

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end alex_average_speed_l20_20154


namespace range_of_a_l20_20464

noncomputable theory

def f (a : ℝ) (x : ℝ) : ℝ := log x + a * x^2 - 2

theorem range_of_a (a : ℝ) (h : ∀ x ∈ Ioo (1 / 4) 1, (1 / x) + 2 * a * x > 0) : a > -8 :=
by {
  sorry  -- proof steps would go here
}

end range_of_a_l20_20464


namespace convex_polyhedron_inequality_l20_20474

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l20_20474


namespace Rachel_plant_arrangement_l20_20357

-- We define Rachel's plants and lamps
inductive Plant : Type
| basil1
| basil2
| aloe
| cactus

inductive Lamp : Type
| white1
| white2
| red1
| red2

def arrangements (plants : List Plant) (lamps : List Lamp) : Nat :=
  -- This would be the function counting all valid arrangements
  -- I'm skipping the implementation
  sorry

def Rachel_arrangement_count : Nat :=
  arrangements [Plant.basil1, Plant.basil2, Plant.aloe, Plant.cactus]
                [Lamp.white1, Lamp.white2, Lamp.red1, Lamp.red2]

theorem Rachel_plant_arrangement : Rachel_arrangement_count = 22 := by
  sorry

end Rachel_plant_arrangement_l20_20357


namespace Liza_reads_more_pages_than_Suzie_l20_20350

def Liza_reading_speed : ℕ := 20
def Suzie_reading_speed : ℕ := 15
def hours : ℕ := 3

theorem Liza_reads_more_pages_than_Suzie :
  Liza_reading_speed * hours - Suzie_reading_speed * hours = 15 := by
  sorry

end Liza_reads_more_pages_than_Suzie_l20_20350


namespace complement_union_A_B_in_U_l20_20617

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l20_20617


namespace inverse_of_f_inverse_of_f_inv_l20_20939

noncomputable def f (x : ℝ) : ℝ := 3^(x - 1) + 1

noncomputable def f_inv (x : ℝ) : ℝ := 1 + Real.log x / Real.log 3

theorem inverse_of_f (x : ℝ) (hx : x > 1) : f_inv (f x) = x :=
by
  sorry

theorem inverse_of_f_inv (x : ℝ) (hx : x > 1) : f (f_inv x) = x :=
by
  sorry

end inverse_of_f_inverse_of_f_inv_l20_20939


namespace unsold_tomatoes_l20_20641

theorem unsold_tomatoes (total_harvest sold_maxwell sold_wilson : ℝ) 
(h_total_harvest : total_harvest = 245.5)
(h_sold_maxwell : sold_maxwell = 125.5)
(h_sold_wilson : sold_wilson = 78) :
(total_harvest - (sold_maxwell + sold_wilson) = 42) :=
by {
  sorry
}

end unsold_tomatoes_l20_20641


namespace distance_on_dirt_section_distance_on_muddy_section_l20_20712

section RaceProblem

variables {v_h v_d v_m : ℕ} (initial_gap : ℕ)

-- Problem conditions
def highway_speed := 150 -- km/h
def dirt_road_speed := 60 -- km/h
def muddy_section_speed := 18 -- km/h
def initial_gap_start := 300 -- meters

-- Convert km/h to m/s
def to_m_per_s (speed : ℕ) : ℕ := (speed * 1000) / 3600

-- Speeds in m/s
def highway_speed_mps := to_m_per_s highway_speed
def dirt_road_speed_mps := to_m_per_s dirt_road_speed
def muddy_section_speed_mps := to_m_per_s muddy_section_speed

-- Questions
theorem distance_on_dirt_section :
  ∃ (d : ℕ), (d = 120) :=
sorry

theorem distance_on_muddy_section :
  ∃ (d : ℕ), (d = 36) :=
sorry

end RaceProblem

end distance_on_dirt_section_distance_on_muddy_section_l20_20712


namespace smallest_multiple_3_4_5_l20_20147

theorem smallest_multiple_3_4_5 : ∃ (n : ℕ), (∀ (m : ℕ), (m % 3 = 0 ∧ m % 4 = 0 ∧ m % 5 = 0) → n ≤ m) ∧ n = 60 := 
sorry

end smallest_multiple_3_4_5_l20_20147


namespace units_digit_7_pow_3_pow_5_l20_20443

theorem units_digit_7_pow_3_pow_5 : ∀ (n : ℕ), n % 4 = 3 → ∀ k, 7 ^ k ≡ 3 [MOD 10] :=
by 
    sorry

end units_digit_7_pow_3_pow_5_l20_20443


namespace mass_of_man_proof_l20_20272

def volume_displaced (L B h : ℝ) : ℝ :=
  L * B * h

def mass_of_man (V ρ : ℝ) : ℝ :=
  ρ * V

theorem mass_of_man_proof :
  ∀ (L B h ρ : ℝ), L = 9 → B = 3 → h = 0.01 → ρ = 1000 →
  mass_of_man (volume_displaced L B h) ρ = 270 :=
by
  intros L B h ρ L_eq B_eq h_eq ρ_eq
  rw [L_eq, B_eq, h_eq, ρ_eq]
  unfold volume_displaced
  unfold mass_of_man
  simp
  sorry

end mass_of_man_proof_l20_20272


namespace inflation_over_two_years_real_interest_rate_l20_20827

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l20_20827


namespace red_other_side_probability_is_one_l20_20134

/-- Definitions from the problem conditions --/
def total_cards : ℕ := 10
def green_both_sides : ℕ := 5
def green_red_sides : ℕ := 2
def red_both_sides : ℕ := 3
def red_faces : ℕ := 6 -- 3 cards × 2 sides each

/-- The theorem proves the probability is 1 that the other side is red given that one side seen is red --/
theorem red_other_side_probability_is_one
  (h_total_cards : total_cards = 10)
  (h_green_both : green_both_sides = 5)
  (h_green_red : green_red_sides = 2)
  (h_red_both : red_both_sides = 3)
  (h_red_faces : red_faces = 6) :
  1 = (red_faces / red_faces) :=
by
  -- Write the proof steps here
  sorry

end red_other_side_probability_is_one_l20_20134


namespace average_riding_speed_l20_20159

theorem average_riding_speed
  (initial_reading : ℕ) (final_reading : ℕ) (time_day1 : ℕ) (time_day2 : ℕ)
  (h_initial : initial_reading = 2332)
  (h_final : final_reading = 2552)
  (h_time_day1 : time_day1 = 5)
  (h_time_day2 : time_day2 = 4) :
  (final_reading - initial_reading) / (time_day1 + time_day2) = 220 / 9 :=
by
  sorry

end average_riding_speed_l20_20159


namespace find_roots_l20_20869

theorem find_roots (x : ℝ) :
  5 * x^4 - 28 * x^3 + 46 * x^2 - 28 * x + 5 = 0 → x = 3.2 ∨ x = 0.8 ∨ x = 1 :=
by
  intro h
  sorry

end find_roots_l20_20869


namespace alyssa_cut_11_roses_l20_20670

theorem alyssa_cut_11_roses (initial_roses cut_roses final_roses : ℕ) 
  (h1 : initial_roses = 3) 
  (h2 : final_roses = 14) 
  (h3 : initial_roses + cut_roses = final_roses) : 
  cut_roses = 11 :=
by
  rw [h1, h2] at h3
  sorry

end alyssa_cut_11_roses_l20_20670


namespace roots_of_polynomial_l20_20741

theorem roots_of_polynomial :
  {r : ℝ | (10 * r^4 - 55 * r^3 + 96 * r^2 - 55 * r + 10 = 0)} = {2, 1, 1 / 2} :=
sorry

end roots_of_polynomial_l20_20741


namespace smallest_lcm_for_80k_quadruples_l20_20115

-- Declare the gcd and lcm functions for quadruples
def gcd_quad (a b c d : ℕ) : ℕ := Nat.gcd (Nat.gcd a b) (Nat.gcd c d)
def lcm_quad (a b c d : ℕ) : ℕ := Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

-- Main statement we need to prove
theorem smallest_lcm_for_80k_quadruples :
  ∃ m : ℕ, (∃ (a b c d : ℕ), gcd_quad a b c d = 100 ∧ lcm_quad a b c d = m) ∧
    (∀ m', m' < m → ¬ (∃ (a' b' c' d' : ℕ), gcd_quad a' b' c' d' = 100 ∧ lcm_quad a' b' c' d' = m')) ∧
    m = 2250000 :=
sorry

end smallest_lcm_for_80k_quadruples_l20_20115


namespace problem_lean_statement_l20_20812

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

theorem problem_lean_statement : 
  (∀ x, g x = 2 * cos (2 * x)) ∧ (∀ x, g (x) = g (-x)) ∧ (∀ x, g (x + π) = g (x)) :=
  sorry

end problem_lean_statement_l20_20812


namespace line_through_points_a_plus_b_l20_20465

theorem line_through_points_a_plus_b :
  ∃ a b : ℝ, (∀ x y : ℝ, (y = a * x + b) → ((x, y) = (6, 7)) ∨ ((x, y) = (10, 23))) ∧ (a + b = -13) :=
sorry

end line_through_points_a_plus_b_l20_20465


namespace total_amount_shared_l20_20216

theorem total_amount_shared (jane mike nora total : ℝ) 
  (h1 : jane = 30) 
  (h2 : jane / 2 = mike / 3) 
  (h3 : mike / 3 = nora / 8) 
  (h4 : total = jane + mike + nora) : 
  total = 195 :=
by
  sorry

end total_amount_shared_l20_20216


namespace find_pair_l20_20013

theorem find_pair :
  ∃ x y : ℕ, (1984 * x - 1983 * y = 1985) ∧ (x = 27764) ∧ (y = 27777) :=
by
  sorry

end find_pair_l20_20013


namespace minimum_value_of_function_on_interval_l20_20186

open set

theorem minimum_value_of_function_on_interval :
  let f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 3 in
  ∃ x ∈ (Icc (-2 : ℝ) 2), (∀ y ∈ (Icc (-2 : ℝ) 2), f x ≤ f y) ∧ f x = -37 :=
by
  let f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 3
  have h_max : (∀ x ∈ (Icc (-2 : ℝ) 2), f x ≤ 3) ∧ ∃ x ∈ (Icc (-2 : ℝ) 2), f x = 3 := sorry
  sorry

end minimum_value_of_function_on_interval_l20_20186


namespace Eva_is_6_l20_20294

def ages : Set ℕ := {2, 4, 6, 8, 10}

def conditions : Prop :=
  ∃ a b, a ∈ ages ∧ b ∈ ages ∧ a + b = 12 ∧
  b ≠ 2 ∧ b ≠ 10 ∧ a ≠ 2 ∧ a ≠ 10 ∧
  (∃ c d, c ∈ ages ∧ d ∈ ages ∧ c = 2 ∧ d = 10 ∧
           (∃ e, e ∈ ages ∧ e = 4 ∧
           ∃ eva, eva ∈ ages ∧ eva ≠ 2 ∧ eva ≠ 4 ∧ eva ≠ 8 ∧ eva ≠ 10 ∧ eva = 6))

theorem Eva_is_6 (h : conditions) : ∃ eva, eva ∈ ages ∧ eva = 6 := sorry

end Eva_is_6_l20_20294


namespace sin_double_angle_sub_pi_over_4_l20_20047

open Real

theorem sin_double_angle_sub_pi_over_4 (x : ℝ) (h : sin x = (sqrt 5 - 1) / 2) : 
  sin (2 * (x - π / 4)) = 2 - sqrt 5 :=
by
  sorry

end sin_double_angle_sub_pi_over_4_l20_20047


namespace solve_inequality_l20_20731

theorem solve_inequality (x : ℝ) : x^2 - 3 * x - 10 < 0 ↔ -2 < x ∧ x < 5 := 
by
  sorry

end solve_inequality_l20_20731


namespace complement_union_eq_l20_20612

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l20_20612


namespace total_jellybeans_l20_20426

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l20_20426


namespace area_enclosed_curve_l20_20677

-- The proof statement
theorem area_enclosed_curve (x y : ℝ) : (x^2 + y^2 = 2 * (|x| + |y|)) → 
  (area_of_enclosed_region = 2 * π + 8) :=
sorry

end area_enclosed_curve_l20_20677


namespace billy_apples_ratio_l20_20160

theorem billy_apples_ratio :
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  thursday / friday = 4 := 
by
  let monday := 2
  let tuesday := 2 * monday
  let wednesday := 9
  let friday := monday / 2
  let total_apples := 20
  let thursday := total_apples - (monday + tuesday + wednesday + friday)
  sorry

end billy_apples_ratio_l20_20160


namespace area_of_fourth_rectangle_l20_20704

theorem area_of_fourth_rectangle (A B C D E F G H I J K L : Type) 
  (x y z w : ℕ) (a1 : x * y = 20) (a2 : x * w = 12) (a3 : z * w = 16) : 
  y * w = 16 :=
by sorry

end area_of_fourth_rectangle_l20_20704


namespace symmetric_points_sum_l20_20451

theorem symmetric_points_sum {c e : ℤ} 
  (P : ℤ × ℤ × ℤ) 
  (sym_xoy : ℤ × ℤ × ℤ) 
  (sym_y : ℤ × ℤ × ℤ) 
  (hP : P = (-4, -2, 3)) 
  (h_sym_xoy : sym_xoy = (-4, -2, -3)) 
  (h_sym_y : sym_y = (4, -2, 3)) 
  (hc : c = -3) 
  (he : e = 4) : 
  c + e = 1 :=
by
  -- Proof goes here
  sorry

end symmetric_points_sum_l20_20451


namespace rational_expression_simplification_l20_20748

theorem rational_expression_simplification
  (a b c : ℚ) 
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a * b^2 = c / a - b) :
  ( ((a^2 * b^2) / c^2 - (2 / c) + (1 / (a^2 * b^2)) + (2 * a * b) / c^2 - (2 / (a * b * c))) 
      / ((2 / (a * b)) - (2 * a * b) / c) ) 
      / (101 / c) = - (1 / 202) :=
by sorry

end rational_expression_simplification_l20_20748


namespace calculation_correct_l20_20552

def expression : ℝ := 200 * 375 * 0.0375 * 5

theorem calculation_correct : expression = 14062.5 := 
by
  sorry

end calculation_correct_l20_20552


namespace dividend_rate_correct_l20_20993

def stock_price : ℝ := 150
def yield_percentage : ℝ := 0.08
def dividend_rate : ℝ := stock_price * yield_percentage

theorem dividend_rate_correct : dividend_rate = 12 := by
  sorry

end dividend_rate_correct_l20_20993


namespace exponent_property_l20_20892

variable {a : ℝ} {m n : ℕ}

theorem exponent_property (h1 : a^m = 2) (h2 : a^n = 3) : a^(2*m + n) = 12 :=
sorry

end exponent_property_l20_20892


namespace convex_polyhedron_inequality_l20_20475

noncomputable def convex_polyhedron (B P T : ℕ) : Prop :=
  ∀ (B P T : ℕ), B > 0 ∧ P > 0 ∧ T >= 0 → B * (Nat.sqrt (P + T)) ≥ 2 * P

theorem convex_polyhedron_inequality (B P T : ℕ) (h : convex_polyhedron B P T) : 
  B * (Nat.sqrt (P + T)) ≥ 2 * P :=
by
  sorry

end convex_polyhedron_inequality_l20_20475


namespace find_f_prime_one_l20_20749

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x)
def f_condition (x : ℝ) : Prop := f (1 / x) = x / (1 + x)

theorem find_f_prime_one : f_condition 1 → deriv f 1 = -1 / 4 := by
  intro h
  sorry

end find_f_prime_one_l20_20749


namespace divisible_check_l20_20938

theorem divisible_check (n : ℕ) (h : n = 287) : 
  ¬ (n % 3 = 0) ∧  ¬ (n % 4 = 0) ∧  ¬ (n % 5 = 0) ∧ ¬ (n % 6 = 0) ∧ (n % 7 = 0) := 
by {
  sorry
}

end divisible_check_l20_20938


namespace total_students_like_sports_l20_20902

def Total_students := 30

def B : ℕ := 12
def C : ℕ := 10
def S : ℕ := 8
def BC : ℕ := 4
def BS : ℕ := 3
def CS : ℕ := 2
def BCS : ℕ := 1

theorem total_students_like_sports : 
  (B + C + S - (BC + BS + CS) + BCS = 22) := by
  sorry

end total_students_like_sports_l20_20902


namespace find_missing_number_l20_20060

theorem find_missing_number (x y : ℝ) 
  (h1 : (x + 50 + 78 + 104 + y) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 76.4) : 
  y = 28 :=
by
  sorry

end find_missing_number_l20_20060


namespace find_n_eq_l20_20301

theorem find_n_eq : 
  let a := 2^4
  let b := 3^3
  ∃ (n : ℤ), a - 7 = b + n :=
by
  let a := 2^4
  let b := 3^3
  use -18
  sorry

end find_n_eq_l20_20301


namespace relationship_between_y_coordinates_l20_20580

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l20_20580


namespace base7_calculation_result_l20_20281

-- Define the base 7 addition and multiplication
def base7_add (a b : ℕ) := (a + b)
def base7_mul (a b : ℕ) := (a * b)

-- Represent the given numbers in base 10 for calculations:
def num1 : ℕ := 2 * 7 + 5 -- 25 in base 7
def num2 : ℕ := 3 * 7^2 + 3 * 7 + 4 -- 334 in base 7
def mul_factor : ℕ := 2 -- 2 in base 7

-- Addition result
def sum : ℕ := base7_add num1 num2

-- Multiplication result
def result : ℕ := base7_mul sum mul_factor

-- Proving the result is equal to the final answer in base 7
theorem base7_calculation_result : result = 6 * 7^2 + 6 * 7 + 4 := 
by sorry

end base7_calculation_result_l20_20281


namespace simplify_expression_sum_of_coefficients_l20_20933

theorem simplify_expression (k : ℝ) (h : k ≠ 0) : (8 * k + 3 + 6 * k^2) + (5 * k^2 + 4 * k + 7) = 11 * k^2 + 12 * k + 10 :=
by sorry

theorem sum_of_coefficients : 11 + 12 + 10 = 33 :=
by rfl

example (k : ℝ) (h : k ≠ 0) : (8 * k + 3 + 6 * k^2 + 5 * k^2 + 4 * k + 7) = 11 * k^2 + 12 * k + 10 ∧ (11 + 12 + 10 = 33) :=
by exact ⟨simplify_expression k h, sum_of_coefficients⟩

end simplify_expression_sum_of_coefficients_l20_20933


namespace distance_between_pathway_lines_is_5_l20_20537

-- Define the conditions
def parallel_lines_distance (distance : ℤ) : Prop :=
  distance = 30

def pathway_length_between_lines (length : ℤ) : Prop :=
  length = 10

def pathway_line_length (length : ℤ) : Prop :=
  length = 60

-- Main proof problem
theorem distance_between_pathway_lines_is_5:
  ∀ (d : ℤ), parallel_lines_distance 30 → 
  pathway_length_between_lines 10 → 
  pathway_line_length 60 → 
  d = 5 := 
by
  sorry

end distance_between_pathway_lines_is_5_l20_20537


namespace symmetric_point_x_axis_l20_20109

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetricWithRespectToXAxis (p : Point3D) : Point3D :=
  {x := p.x, y := -p.y, z := -p.z}

theorem symmetric_point_x_axis :
  symmetricWithRespectToXAxis ⟨-1, -2, 3⟩ = ⟨-1, 2, -3⟩ :=
  by
    sorry

end symmetric_point_x_axis_l20_20109


namespace number_of_shortest_paths_l20_20515

-- Define the concept of shortest paths
def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

-- State the theorem that needs to be proved
theorem number_of_shortest_paths (m n : ℕ) : shortest_paths m n = Nat.choose (m + n) m :=
by 
  sorry

end number_of_shortest_paths_l20_20515


namespace continuity_at_2_l20_20347

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
if x ≤ 2 then 4 * x^2 + 5 else b * x + 3

theorem continuity_at_2 (b : ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 2) < δ → abs (f x b - f 2 b) < ε) → b = 9 :=
by
  sorry  

end continuity_at_2_l20_20347


namespace probability_rolling_odd_l20_20701

-- Define the faces of the die
def die_faces : List ℕ := [1, 1, 1, 2, 3, 3]

-- Define a predicate to check if a number is odd
def is_odd (n : ℕ) : Prop := n % 2 = 1

-- Define the probability function
def probability_of_odd : ℚ := 
  let total_outcomes := die_faces.length
  let favorable_outcomes := die_faces.countp is_odd
  favorable_outcomes / total_outcomes

theorem probability_rolling_odd :
  probability_of_odd = 5 / 6 := by
  sorry

end probability_rolling_odd_l20_20701


namespace tate_total_years_l20_20935

-- Define the conditions
def high_school_years : Nat := 3
def gap_years : Nat := 2
def bachelor_years : Nat := 2 * high_school_years
def certification_years : Nat := 1
def work_experience_years : Nat := 1
def master_years : Nat := bachelor_years / 2
def phd_years : Nat := 3 * (high_school_years + bachelor_years + master_years)

-- Define the total years Tate spent
def total_years : Nat :=
  high_school_years + gap_years +
  bachelor_years + certification_years +
  work_experience_years + master_years + phd_years

-- State the theorem
theorem tate_total_years : total_years = 52 := by
  sorry

end tate_total_years_l20_20935


namespace remainder_6n_mod_4_l20_20963

theorem remainder_6n_mod_4 (n : ℕ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := by
  sorry

end remainder_6n_mod_4_l20_20963


namespace amount_first_set_correct_l20_20103

-- Define the amounts as constants
def total_amount : ℝ := 900.00
def amount_second_set : ℝ := 260.00
def amount_third_set : ℝ := 315.00

-- Define the amount given to the first set
def amount_first_set : ℝ :=
  total_amount - amount_second_set - amount_third_set

-- Statement: prove that the amount given to the first set of families equals $325.00
theorem amount_first_set_correct :
  amount_first_set = 325.00 :=
sorry

end amount_first_set_correct_l20_20103


namespace percentage_deficit_for_second_side_l20_20331

-- Defining the given conditions and the problem statement
def side1_excess : ℚ := 0.14
def area_error : ℚ := 0.083
def original_length (L : ℚ) := L
def original_width (W : ℚ) := W
def measured_length_side1 (L : ℚ) := (1 + side1_excess) * L
def measured_width_side2 (W : ℚ) (x : ℚ) := W * (1 - 0.01 * x)
def original_area (L W : ℚ) := L * W
def calculated_area (L W x : ℚ) := 
  measured_length_side1 L * measured_width_side2 W x

theorem percentage_deficit_for_second_side (L W : ℚ) :
  (calculated_area L W 5) / (original_area L W) = 1 + area_error :=
by
  sorry

end percentage_deficit_for_second_side_l20_20331


namespace garden_width_is_correct_l20_20114

noncomputable def width_of_garden : ℝ :=
  let w := 12 -- We will define the width to be 12 as the final correct answer.
  w

theorem garden_width_is_correct (h_length : ∀ {w : ℝ}, 3 * w = 432 / w) : width_of_garden = 12 := by
  sorry

end garden_width_is_correct_l20_20114


namespace inequality_solution_l20_20562

theorem inequality_solution (x : ℝ) :
  (7 : ℝ) / 30 + abs (x - 7 / 60) < 11 / 20 ↔ -1 / 5 < x ∧ x < 13 / 30 :=
by
  sorry

end inequality_solution_l20_20562


namespace part1_part2_l20_20597

def f (x a : ℝ) := |x - a| + 2 * |x + 1|

-- Part 1: Solve the inequality f(x) > 4 when a = 2
theorem part1 (x : ℝ) : f x 2 > 4 ↔ (x < -4/3 ∨ x > 0) := by
  sorry

-- Part 2: If the solution set of the inequality f(x) < 3x + 4 is {x | x > 2}, find the value of a.
theorem part2 (a : ℝ) : (∀ x : ℝ, (f x a < 3 * x + 4 ↔ x > 2)) → a = 6 := by
  sorry

end part1_part2_l20_20597


namespace rate_of_A_is_8_l20_20814

noncomputable def rate_of_A (a b : ℕ) : ℕ :=
  if b = a + 4 ∧ 48 * b = 72 * a then a else 0

theorem rate_of_A_is_8 {a b : ℕ} 
  (h1 : b = a + 4)
  (h2 : 48 * b = 72 * a) : 
  rate_of_A a b = 8 :=
by
  -- proof steps can be added here
  sorry

end rate_of_A_is_8_l20_20814


namespace costco_container_holds_one_gallon_l20_20217

theorem costco_container_holds_one_gallon
  (costco_cost : ℕ := 8)
  (store_cost_per_bottle : ℕ := 3)
  (savings : ℕ := 16)
  (ounces_per_bottle : ℕ := 16)
  (ounces_per_gallon : ℕ := 128) :
  ∃ (gallons : ℕ), gallons = 1 :=
by
  sorry

end costco_container_holds_one_gallon_l20_20217


namespace max_distance_between_P_and_Q_l20_20590

-- Definitions of the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop := P.1^2 + (P.2 - 6)^2 = 2
def is_on_ellipse (Q : ℝ × ℝ) : Prop := (Q.1^2) / 10 + Q.2^2 = 1

-- The maximum distance between any point on the circle and any point on the ellipse
theorem max_distance_between_P_and_Q :
  ∃ P Q : ℝ × ℝ, is_on_circle P ∧ is_on_ellipse Q ∧ dist P Q = 6 * Real.sqrt 2 :=
sorry

end max_distance_between_P_and_Q_l20_20590


namespace problem1_problem2_l20_20785

noncomputable def f (x a : ℝ) := |x - 2 * a|
noncomputable def g (x a : ℝ) := |x + a|

theorem problem1 (x m : ℝ): (∃ x, f x 1 - g x 1 ≥ m) → m ≤ 3 :=
by
  sorry

theorem problem2 (a : ℝ): (∀ x, f x a + g x a ≥ 3) → (a ≥ 1 ∨ a ≤ -1) :=
by
  sorry

end problem1_problem2_l20_20785


namespace kilometers_driven_equal_l20_20168

theorem kilometers_driven_equal (x : ℝ) :
  (20 + 0.25 * x = 24 + 0.16 * x) → x = 44 := by
  sorry

end kilometers_driven_equal_l20_20168


namespace molecular_weight_CaH2_correct_l20_20679

-- Define the atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_H : ℝ := 1.008

-- Define the formula to compute the molecular weight
def molecular_weight_CaH2 (atomic_weight_Ca : ℝ) (atomic_weight_H : ℝ) : ℝ :=
  (1 * atomic_weight_Ca) + (2 * atomic_weight_H)

-- Theorem stating that the molecular weight of CaH2 is 42.096 g/mol
theorem molecular_weight_CaH2_correct : molecular_weight_CaH2 atomic_weight_Ca atomic_weight_H = 42.096 := 
by 
  sorry

end molecular_weight_CaH2_correct_l20_20679


namespace find_a_with_constraints_l20_20105

theorem find_a_with_constraints (x y a : ℝ) 
  (h1 : 2 * x - y + 2 ≥ 0) 
  (h2 : x - 3 * y + 1 ≤ 0)
  (h3 : x + y - 2 ≤ 0)
  (h4 : a > 0)
  (h5 : ∃ (x1 x2 x3 y1 y2 y3 : ℝ), 
    ((x1, y1) = (1, 1) ∨ (x1, y1) = (5 / 3, 1 / 3) ∨ (x1, y1) = (2, 0)) ∧ 
    ((x2, y2) = (1, 1) ∨ (x2, y2) = (5 / 3, 1 / 3) ∨ (x2, y2) = (2, 0)) ∧ 
    ((x3, y3) = (1, 1) ∨ (x3, y3) = (5 / 3, 1 / 3) ∨ (x3, y3) = (2, 0)) ∧ 
    (ax1 - y1 = ax2 - y2) ∧ (ax2 - y2 = ax3 - y3)) :
  a = 1 / 3 :=
sorry

end find_a_with_constraints_l20_20105


namespace f_2015_equals_2_l20_20306

noncomputable def f : ℝ → ℝ :=
sorry

theorem f_2015_equals_2 (f_even : ∀ x : ℝ, f (-x) = f x)
    (f_shift : ∀ x : ℝ, f (-x) = f (2 + x))
    (f_log : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f x = Real.log (3 * x + 1) / Real.log 2) :
    f 2015 = 2 :=
sorry

end f_2015_equals_2_l20_20306


namespace power_sum_eq_l20_20023

theorem power_sum_eq (n : ℕ) : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end power_sum_eq_l20_20023


namespace load_transportable_l20_20982

theorem load_transportable :
  ∃ (n : ℕ), n ≤ 11 ∧ (∀ (box_weight : ℕ) (total_weight : ℕ),
    total_weight = 13500 ∧ 
    box_weight ≤ 350 ∧ 
    (n * 1500) ≥ total_weight) :=
by
  sorry

end load_transportable_l20_20982


namespace inequality_solution_l20_20201

theorem inequality_solution (a : ℝ)
  (h : ∀ x, x ∈ Set.Icc (1/2 : ℝ) 2 → a * x^2 - 2 * x + 2 > 0) :
  a > 1/2 := 
sorry

end inequality_solution_l20_20201


namespace chandler_total_rolls_l20_20179

-- Definitions based on given conditions
def rolls_sold_grandmother : ℕ := 3
def rolls_sold_uncle : ℕ := 4
def rolls_sold_neighbor : ℕ := 3
def rolls_needed_more : ℕ := 2

-- Total rolls sold so far and needed
def total_rolls_to_sell : ℕ :=
  rolls_sold_grandmother + rolls_sold_uncle + rolls_sold_neighbor + rolls_needed_more

theorem chandler_total_rolls : total_rolls_to_sell = 12 :=
by
  sorry

end chandler_total_rolls_l20_20179


namespace find_a3_l20_20574

-- Define the sequence sum S_n
def S (n : ℕ) : ℚ := (n + 1) / (n + 2)

-- Define the sequence term a_n using S_n
def a (n : ℕ) : ℚ :=
  if h : n = 1 then S 1 else S n - S (n - 1)

-- State the theorem to find the value of a_3
theorem find_a3 : a 3 = 1 / 20 :=
by
  -- The proof is omitted, use sorry to skip it
  sorry

end find_a3_l20_20574


namespace cube_mono_increasing_l20_20317

theorem cube_mono_increasing (a b : ℝ) (h : a > b) : a^3 > b^3 := sorry

end cube_mono_increasing_l20_20317


namespace sum_of_angles_of_solutions_l20_20371

theorem sum_of_angles_of_solutions : 
  ∀ (z : ℂ), z^5 = 32 * Complex.I → ∃ θs : Fin 5 → ℝ, 
  (∀ k, 0 ≤ θs k ∧ θs k < 360) ∧ (θs 0 + θs 1 + θs 2 + θs 3 + θs 4 = 810) :=
by
  sorry

end sum_of_angles_of_solutions_l20_20371


namespace find_ck_l20_20943

theorem find_ck 
  (d r : ℕ)                -- d : common difference, r : common ratio
  (k : ℕ)                  -- k : integer such that certain conditions hold
  (hn2 : (k-2) > 0)        -- ensure (k-2) > 0
  (hk1 : (k+1) > 0)        -- ensure (k+1) > 0
  (h1 : 1 + (k-3) * d + r^(k-3) = 120) -- c_{k-1} = 120
  (h2 : 1 + k * d + r^k = 1200) -- c_{k+1} = 1200
  : (1 + (k-1) * d + r^(k-1)) = 263 := -- c_k = 263
sorry

end find_ck_l20_20943


namespace remove_blue_to_get_80_percent_red_l20_20857

-- Definitions from the conditions
def total_balls : ℕ := 150
def red_balls : ℕ := 60
def initial_blue_balls : ℕ := total_balls - red_balls
def desired_percentage_red : ℤ := 80

-- Lean statement of the proof problem
theorem remove_blue_to_get_80_percent_red :
  ∃ (x : ℕ), (x ≤ initial_blue_balls) ∧ (red_balls * 100 = desired_percentage_red * (total_balls - x)) → x = 75 := sorry

end remove_blue_to_get_80_percent_red_l20_20857


namespace jill_trips_to_fill_tank_l20_20215

-- Definitions as per the conditions specified
def tank_capacity : ℕ := 600
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def leak_per_trip : ℕ := 2

-- Prove that the number of trips Jill will make = 20 given the above conditions
theorem jill_trips_to_fill_tank : 
  (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) * (tank_capacity / ((jack_trips_ratio + jill_trips_ratio) * (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) / (jack_trips_ratio + jill_trips_ratio)))  = 20 := 
sorry

end jill_trips_to_fill_tank_l20_20215


namespace cost_of_math_books_l20_20519

theorem cost_of_math_books (M : ℕ) : 
  (∃ (total_books math_books history_books total_cost : ℕ),
    total_books = 90 ∧
    math_books = 60 ∧
    history_books = total_books - math_books ∧
    history_books * 5 + math_books * M = total_cost ∧
    total_cost = 390) → 
  M = 4 :=
by
  -- We provide the assumed conditions
  intro h
  -- We will skip the proof with sorry
  sorry

end cost_of_math_books_l20_20519


namespace regular_pay_calculation_l20_20709

theorem regular_pay_calculation
  (R : ℝ)  -- defining the regular pay per hour
  (H1 : 40 * R + 20 * R = 180):  -- condition given based on the total actual pay calculation.
  R = 3 := 
by
  -- Skipping the proof
  sorry

end regular_pay_calculation_l20_20709


namespace total_apples_l20_20132

theorem total_apples (baskets apples_per_basket : ℕ) (h1 : baskets = 37) (h2 : apples_per_basket = 17) : baskets * apples_per_basket = 629 := by
  sorry

end total_apples_l20_20132


namespace quarters_to_dimes_difference_l20_20365

variable (p : ℝ)

theorem quarters_to_dimes_difference :
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  difference_dimes = 12.5 * p - 15 :=
by
  let Susan_quarters := 7 * p + 3
  let George_quarters := 2 * p + 9
  let difference_quarters := Susan_quarters - George_quarters
  let difference_dimes := 2.5 * difference_quarters
  sorry

end quarters_to_dimes_difference_l20_20365


namespace area_of_diamond_l20_20324

theorem area_of_diamond (x y : ℝ) : (|x / 2| + |y / 2| = 1) → 
∃ (area : ℝ), area = 8 :=
by sorry

end area_of_diamond_l20_20324


namespace proof_problem_l20_20061

variable (x y : ℕ) -- define x and y as natural numbers

-- Define the problem-specific variables m and n
variable (m n : ℕ)

-- Assume the conditions given in the problem
axiom H1 : 2 = m
axiom H2 : n = 3

-- The goal is to prove that -m^n equals -8 given the conditions H1 and H2
theorem proof_problem : - (m^n : ℤ) = -8 :=
by
  sorry

end proof_problem_l20_20061


namespace exists_pos_integers_l20_20495

theorem exists_pos_integers (r : ℚ) (hr : r > 0) : 
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r = (a^3 + b^3) / (c^3 + d^3) :=
by sorry

end exists_pos_integers_l20_20495


namespace xiao_ming_english_score_l20_20647

theorem xiao_ming_english_score :
  let a := 92
  let b := 90
  let c := 95
  let w_a := 3
  let w_b := 3
  let w_c := 4
  let total_weight := (w_a + w_b + w_c)
  let score := (a * w_a + b * w_b + c * w_c) / total_weight
  score = 92.6 :=
by
  sorry

end xiao_ming_english_score_l20_20647


namespace find_largest_number_l20_20444

theorem find_largest_number (a b c d e : ℕ)
    (h1 : a + b + c + d = 240)
    (h2 : a + b + c + e = 260)
    (h3 : a + b + d + e = 280)
    (h4 : a + c + d + e = 300)
    (h5 : b + c + d + e = 320)
    (h6 : a + b = 40) :
    max a (max b (max c (max d e))) = 160 := by
  sorry

end find_largest_number_l20_20444


namespace find_a_l20_20298

theorem find_a (a x y : ℝ) 
  (h1 : (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0) 
  (h2 : (x + 2)^2 + (y + 4)^2 = a) 
  (h3 : ∃! x y, (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0 ∧ (x + 2)^2 + (y + 4)^2 = a) :
  a = 9 ∨ a = 23 + 4 * Real.sqrt 15 :=
sorry

end find_a_l20_20298


namespace sliderB_moves_distance_l20_20071

theorem sliderB_moves_distance :
  ∀ (A B : ℝ) (rod_length : ℝ),
    (A = 20) →
    (B = 15) →
    (rod_length = Real.sqrt (20^2 + 15^2)) →
    (rod_length = 25) →
    (B_new = 25 - 15) →
    B_new = 10 := by
  sorry

end sliderB_moves_distance_l20_20071


namespace multiply_polynomials_l20_20095

variable {x y : ℝ}

theorem multiply_polynomials (x y : ℝ) :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 :=
by
  sorry

end multiply_polynomials_l20_20095


namespace find_fraction_l20_20310

theorem find_fraction {a b : ℕ} 
  (h1 : 32016 + (a / b) = 2016 * 3 + (a / b)) 
  (ha : a = 2016) 
  (hb : b = 2016^3 - 1) : 
  (b + 1) / a^2 = 2016 := 
by 
  sorry

end find_fraction_l20_20310


namespace find_b_l20_20235

theorem find_b : ∃ b : ℤ, 0 ≤ b ∧ b ≤ 19 ∧ (527816429 - b) % 17 = 0 ∧ b = 8 := 
by 
  sorry

end find_b_l20_20235


namespace Amanda_lost_notebooks_l20_20282

theorem Amanda_lost_notebooks (initial_notebooks ordered additional_notebooks remaining_notebooks : ℕ)
  (h1 : initial_notebooks = 10)
  (h2 : ordered = 6)
  (h3 : remaining_notebooks = 14) :
  initial_notebooks + ordered - remaining_notebooks = 2 := by
sorry

end Amanda_lost_notebooks_l20_20282


namespace birds_landing_l20_20669

theorem birds_landing (initial_birds total_birds birds_landed : ℤ) 
  (h_initial : initial_birds = 12) 
  (h_total : total_birds = 20) :
  birds_landed = total_birds - initial_birds :=
by
  sorry

end birds_landing_l20_20669


namespace problem_solution_l20_20440

noncomputable def y (x : ℝ) : ℝ := Real.sqrt (x - 1)

theorem problem_solution (x : ℝ) : x ≥ 1 →
  (Real.sqrt (x + 5 - 6 * Real.sqrt (x - 1)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 1)) = 2) ↔ (x = 13.25) :=
sorry

end problem_solution_l20_20440


namespace focus_of_parabola_l20_20797

theorem focus_of_parabola (x y : ℝ) : 
  (∃ x y : ℝ, x^2 = -2 * y) → (0, -1/2) = (0, -1/2) :=
sorry

end focus_of_parabola_l20_20797


namespace min_pounds_of_beans_l20_20419

theorem min_pounds_of_beans : 
  ∃ (b : ℕ), (∀ (r : ℝ), (r ≥ 8 + b / 3 ∧ r ≤ 3 * b) → b ≥ 3) :=
sorry

end min_pounds_of_beans_l20_20419


namespace correct_expression_l20_20401

theorem correct_expression (a : ℝ) :
  (a^3 * a^2 = a^5) ∧ ¬((a^2)^3 = a^5) ∧ ¬(2 * a^2 + 3 * a^3 = 5 * a^5) ∧ ¬((a - 1)^2 = a^2 - 1) :=
by
  sorry

end correct_expression_l20_20401


namespace total_product_l20_20855

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12 
  else if n % 2 = 0 then 4 
  else 0 

def allie_rolls : List ℕ := [2, 6, 3, 1, 6]
def betty_rolls : List ℕ := [4, 6, 3, 5]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem total_product : total_points allie_rolls * total_points betty_rolls = 1120 := sorry

end total_product_l20_20855


namespace coins_dimes_count_l20_20967

theorem coins_dimes_count :
  ∃ (p n d q : ℕ), 
    p + n + d + q = 10 ∧ 
    p + 5 * n + 10 * d + 25 * q = 110 ∧ 
    p ≥ 1 ∧ n ≥ 1 ∧ d ≥ 1 ∧ q ≥ 2 ∧ d = 5 :=
by {
    sorry
}

end coins_dimes_count_l20_20967


namespace compute_fg_difference_l20_20779

def f (x : ℕ) : ℕ := x^2 + 3
def g (x : ℕ) : ℕ := 2 * x + 5

theorem compute_fg_difference : f (g 5) - g (f 5) = 167 := by
  sorry

end compute_fg_difference_l20_20779


namespace problem_solution_l20_20635

noncomputable def is_valid_permutation (s : List Char) : Prop :=
  s.length = 16 ∧
  (∀ i, i < 4 → s.nthLe i sorry ≠ 'A') ∧
  (∀ i, 4 ≤ i ∧ i < 9 → s.nthLe i sorry ≠ 'B') ∧
  (∀ i, 9 ≤ i → s.nthLe i sorry ≠ 'C' ∧ s.nthLe i sorry ≠ 'D')

def count_valid_permutations : Nat :=
  List.permutations "AAAABBBBBCCCCDDD".toList.count is_valid_permutation

def answer : Nat :=
  count_valid_permutations % 1000

theorem problem_solution : answer = 62 :=
  sorry

end problem_solution_l20_20635


namespace angle_perpendicular_sides_l20_20751

theorem angle_perpendicular_sides (α β : ℝ) (hα : α = 80) 
  (h_perp : ∀ {x y}, ((x = α → y = 180 - x) ∨ (y = 180 - α → x = y))) : 
  β = 80 ∨ β = 100 :=
by
  sorry

end angle_perpendicular_sides_l20_20751


namespace total_cost_is_72_l20_20530

-- Definitions based on conditions
def adults (total_people : ℕ) (kids : ℕ) : ℕ := total_people - kids
def cost_per_adult_meal (cost_per_meal : ℕ) (adults : ℕ) : ℕ := cost_per_meal * adults
def total_cost (total_people : ℕ) (kids : ℕ) (cost_per_meal : ℕ) : ℕ := 
  cost_per_adult_meal cost_per_meal (adults total_people kids)

-- Given values
def total_people := 11
def kids := 2
def cost_per_meal := 8

-- Theorem statement
theorem total_cost_is_72 : total_cost total_people kids cost_per_meal = 72 := by
  sorry

end total_cost_is_72_l20_20530


namespace minimize_notch_volume_l20_20411

noncomputable def total_volume (theta phi : ℝ) : ℝ :=
  let part1 := (2 / 3) * Real.tan phi
  let part2 := (2 / 3) * Real.tan (theta - phi)
  part1 + part2

theorem minimize_notch_volume :
  ∀ (theta : ℝ), (0 < theta ∧ theta < π) →
  ∃ (phi : ℝ), (0 < phi ∧ phi < θ) ∧
  (∀ ψ : ℝ, (0 < ψ ∧ ψ < θ) → total_volume theta ψ ≥ total_volume theta (theta / 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_notch_volume_l20_20411


namespace combined_rate_last_year_l20_20718

noncomputable def combine_effective_rate_last_year (r_increased: ℝ) (r_this_year: ℝ) : ℝ :=
  r_this_year / r_increased

theorem combined_rate_last_year
  (compounding_frequencies : List String)
  (r_increased : ℝ)
  (r_this_year : ℝ)
  (combined_interest_rate_this_year : r_this_year = 0.11)
  (interest_rate_increase : r_increased = 1.10) :
  combine_effective_rate_last_year r_increased r_this_year = 0.10 :=
by
  sorry

end combined_rate_last_year_l20_20718


namespace solve_for_a_l20_20880

theorem solve_for_a (x a : ℤ) (h : 2 * x - a - 5 = 0) (hx : x = 3) : a = 1 :=
by sorry

end solve_for_a_l20_20880


namespace relationship_of_y_values_l20_20577

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l20_20577


namespace probability_of_same_color_correct_l20_20532

/-- Define events and their probabilities based on the given conditions --/
def probability_of_two_black_stones : ℚ := 1 / 7
def probability_of_two_white_stones : ℚ := 12 / 35

/-- Define the probability of drawing two stones of the same color --/
def probability_of_two_same_color_stones : ℚ :=
  probability_of_two_black_stones + probability_of_two_white_stones

theorem probability_of_same_color_correct :
  probability_of_two_same_color_stones = 17 / 35 :=
by
  -- We only set up the theorem, the proof is not considered here
  sorry

end probability_of_same_color_correct_l20_20532


namespace calc_expression_l20_20270

theorem calc_expression : 2^1 + 1^0 - 3^2 = -6 := by
  sorry

end calc_expression_l20_20270


namespace find_y_plus_inv_y_l20_20312

theorem find_y_plus_inv_y (y : ℝ) (h : y^3 + 1 / y^3 = 110) : y + 1 / y = 5 :=
sorry

end find_y_plus_inv_y_l20_20312


namespace problem_1_problem_2_l20_20090

noncomputable def f (x a : ℝ) : ℝ := |x - a|

theorem problem_1 (x : ℝ) : (f x 2) ≥ (7 - |x - 1|) ↔ (x ≤ -2 ∨ x ≥ 5) := 
by
  sorry

theorem problem_2 (m n : ℝ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) 
  (h : (f (1/m) 1) + (f (1/(2*n)) 1) = 1) : m + 4 * n ≥ 2 * Real.sqrt 2 + 3 := 
by
  sorry

end problem_1_problem_2_l20_20090


namespace compute_product_sum_l20_20162

theorem compute_product_sum (a b c : ℕ) (ha : a = 3) (hb : b = 4) (hc : c = 5) :
  (a * b * c) * ((1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c) = 47 :=
by
  sorry

end compute_product_sum_l20_20162


namespace total_savings_correct_l20_20994

theorem total_savings_correct :
  let price_chlorine := 10
  let discount1_chlorine := 0.20
  let discount2_chlorine := 0.10
  let price_soap := 16
  let discount1_soap := 0.25
  let discount2_soap := 0.05
  let price_wipes := 8
  let bogo_discount_wipes := 0.50
  let quantity_chlorine := 4
  let quantity_soap := 6
  let quantity_wipes := 8
  let final_chlorine_price := (price_chlorine * (1 - discount1_chlorine)) * (1 - discount2_chlorine)
  let final_soap_price := (price_soap * (1 - discount1_soap)) * (1 - discount2_soap)
  let final_wipes_price_per_two := price_wipes + price_wipes * bogo_discount_wipes
  let final_wipes_price := final_wipes_price_per_two / 2
  let total_original_price := quantity_chlorine * price_chlorine + quantity_soap * price_soap + quantity_wipes * price_wipes
  let total_final_price := quantity_chlorine * final_chlorine_price + quantity_soap * final_soap_price + quantity_wipes * final_wipes_price
  let total_savings := total_original_price - total_final_price
  total_savings = 55.80 :=
by sorry

end total_savings_correct_l20_20994


namespace find_number_l20_20399

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end find_number_l20_20399


namespace boat_speed_still_water_l20_20720

theorem boat_speed_still_water (v c : ℝ) (h1 : v + c = 13) (h2 : v - c = 4) : v = 8.5 :=
by sorry

end boat_speed_still_water_l20_20720


namespace nth_term_sequence_l20_20226

def sequence (n : ℕ) : ℕ :=
  if n = 0 then 1
  else (2 ^ n) - 1

theorem nth_term_sequence (n : ℕ) : 
  sequence n = 2 ^ n - 1 :=
by
  sorry

end nth_term_sequence_l20_20226


namespace servings_of_peanut_butter_l20_20012

theorem servings_of_peanut_butter :
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  (peanutButterInJar / oneServing) = servings :=
by
  let peanutButterInJar := 37 + 4 / 5
  let oneServing := 1 + 1 / 2
  let servings := 25 + 1 / 5
  sorry

end servings_of_peanut_butter_l20_20012


namespace sum_of_a_and_b_l20_20700

-- Define conditions
def population_size : ℕ := 55
def sample_size : ℕ := 5
def interval : ℕ := population_size / sample_size
def sample_indices : List ℕ := [6, 28, 50]

-- Assume a and b are such that the systematic sampling is maintained
variable (a b : ℕ)
axiom a_idx : a = sample_indices.head! + interval
axiom b_idx : b = sample_indices.getLast! - interval

-- Define Lean 4 statement to prove
theorem sum_of_a_and_b :
  (a + b) = 56 :=
by
  -- This will be the place where the proof is inserted
  sorry

end sum_of_a_and_b_l20_20700


namespace AM_GM_Inequality_l20_20346

theorem AM_GM_Inequality 
  (a b c : ℝ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end AM_GM_Inequality_l20_20346


namespace max_chain_length_in_subdivided_triangle_l20_20175

-- Define an equilateral triangle subdivision
structure EquilateralTriangleSubdivided (n : ℕ) :=
(n_squares : ℕ)
(n_squares_eq : n_squares = n^2)

-- Define the problem's chain concept
def maximum_chain_length (n : ℕ) : ℕ :=
n^2 - n + 1

-- Main statement
theorem max_chain_length_in_subdivided_triangle
  (n : ℕ) (triangle : EquilateralTriangleSubdivided n) :
  maximum_chain_length n = n^2 - n + 1 :=
by sorry

end max_chain_length_in_subdivided_triangle_l20_20175


namespace quadrilateral_is_trapezium_l20_20106

-- Define the angles of the quadrilateral and the sum of the angles condition
variables {x : ℝ}
def sum_of_angles (x : ℝ) : Prop := x + 5 * x + 2 * x + 4 * x = 360

-- State the theorem
theorem quadrilateral_is_trapezium (x : ℝ) (h : sum_of_angles x) : 
  30 + 150 = 180 ∧ 60 + 120 = 180 → is_trapezium :=
sorry

end quadrilateral_is_trapezium_l20_20106


namespace same_candy_probability_l20_20976

noncomputable def candy_probability : ℚ :=
let total_permutations := (20.choose 3 : ℚ),
    alice_picks_three_red := (12.choose 3 : ℚ) / total_permutations,
    alice_picks_two_red_one_green := ((12.choose 2 : ℕ) * (8.choose 1 : ℕ) : ℚ) / total_permutations,
    bob_picks_three_red := (9.choose 3 : ℚ) / (17.choose 3 : ℚ),
    bob_picks_two_red_one_green := ((10.choose 2 : ℕ) * (7.choose 1 : ℕ) : ℚ) / (17.choose 3 : ℚ) in
alice_picks_three_red * bob_picks_three_red + alice_picks_two_red_one_green * bob_picks_two_red_one_green

theorem same_candy_probability :
    candy_probability = 231 / 1060 :=
sorry

end same_candy_probability_l20_20976


namespace probability_of_two_primes_l20_20958

-- Define the set of integers from 1 to 30
def finite_set : set ℕ := {n | 1 ≤ n ∧ n ≤ 30}

-- Define the set of prime numbers from 1 to 30
def primes_set : set ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

-- Define the probability of choosing two different primes
def probability_two_primes : ℚ :=
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ primes_set ∧ p.2 ∈ primes_set}) /
  (fintype.card {p : ℕ × ℕ // p.1 < p.2 ∧ p.1 ∈ finite_set ∧ p.2 ∈ finite_set})

-- Prove that the probability is 1/29
theorem probability_of_two_primes :
  probability_two_primes = 1 / 29 :=
sorry

end probability_of_two_primes_l20_20958


namespace Genevieve_drinks_pints_l20_20004

theorem Genevieve_drinks_pints :
  ∀ (total_gallons : ℝ) (num_thermoses : ℕ) (pints_per_gallon : ℝ) (genevieve_thermoses : ℕ),
  total_gallons = 4.5 → num_thermoses = 18 → pints_per_gallon = 8 → genevieve_thermoses = 3 →
  (genevieve_thermoses * ((total_gallons / num_thermoses) * pints_per_gallon) = 6) :=
by
  intros total_gallons num_thermoses pints_per_gallon genevieve_thermoses
  intros h1 h2 h3 h4
  sorry

end Genevieve_drinks_pints_l20_20004


namespace averageFishIs75_l20_20490

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l20_20490


namespace solution_set_inequality_l20_20181

theorem solution_set_inequality (t : ℝ) (ht : 0 < t ∧ t < 1) :
  {x : ℝ | x^2 - (t + t⁻¹) * x + 1 < 0} = {x : ℝ | t < x ∧ x < t⁻¹} :=
sorry

end solution_set_inequality_l20_20181


namespace reading_rate_l20_20377

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l20_20377


namespace pages_per_hour_l20_20381

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l20_20381


namespace a_speed_calculation_l20_20402

def meeting_time : ℝ := 179.98560115190787
def track_length : ℝ := 900
def b_speed_kmph : ℝ := 54
def b_speed_mps : ℝ := b_speed_kmph * (1000 / 3600)
def b_distance_covered : ℝ := b_speed_mps * meeting_time
def b_laps : ℝ := b_distance_covered / track_length

theorem a_speed_calculation
  (meeting_time: ℝ)
  (track_length: ℝ)
  (b_speed_kmph: ℝ)
  (a_speed: ℝ)
  (h_meet_time: meeting_time = 179.98560115190787)
  (h_track_length: track_length = 900)
  (h_b_speed: b_speed_kmph = 54)
  : a_speed = 54 := by
  sorry

end a_speed_calculation_l20_20402


namespace total_chairs_l20_20276

-- Define the conditions as constants
def living_room_chairs : ℕ := 3
def kitchen_chairs : ℕ := 6
def dining_room_chairs : ℕ := 8
def outdoor_patio_chairs : ℕ := 12

-- State the goal to prove
theorem total_chairs : 
  living_room_chairs + kitchen_chairs + dining_room_chairs + outdoor_patio_chairs = 29 := 
by
  -- The proof is not required as per instructions
  sorry

end total_chairs_l20_20276


namespace find_lambda_l20_20194

variable {R : Type*} [CommRing R]

def vector (n : ℕ) := fin n → R

def coplanar (a b c : vector 3) : Prop :=
∃ (p q : R), c = p • a + q • b

theorem find_lambda
  (a : vector 3 := ![2, -1, 3])
  (b : vector 3 := ![-1, 4, -2])
  (c : vector 3 := ![7, 5, (65 : R) / 7])
  (h : coplanar a b c) :
  c.last = (65 : R) / 7 :=
by { sorry }

end find_lambda_l20_20194


namespace part1_part2_l20_20596

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem part1 (x : ℝ) : f x ≥ 1 ↔ (x ≤ -5/2 ∨ x ≥ 3/2) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end part1_part2_l20_20596


namespace problem_statement_l20_20187

theorem problem_statement (p q : ℕ) (prime_p : Nat.Prime p) (prime_q : Nat.Prime q) (r s : ℕ)
  (consecutive_primes : Nat.Prime r ∧ Nat.Prime s ∧ (r + 1 = s ∨ s + 1 = r))
  (roots_condition : r + s = p ∧ r * s = 2 * q) :
  (r * s = 2 * q) ∧ (Nat.Prime (p^2 - 2 * q)) ∧ (Nat.Prime (p + 2 * q)) :=
by
  sorry

end problem_statement_l20_20187


namespace Felicity_family_store_visits_l20_20296

theorem Felicity_family_store_visits
  (lollipop_stick : ℕ := 1)
  (fort_total_sticks : ℕ := 400)
  (fort_completion_percent : ℕ := 60)
  (weeks_collected : ℕ := 80)
  (sticks_collected : ℕ := (fort_total_sticks * fort_completion_percent) / 100)
  (store_visits_per_week : ℕ := sticks_collected / weeks_collected) :
  store_visits_per_week = 3 := by
  sorry

end Felicity_family_store_visits_l20_20296


namespace unique_integer_solution_quad_eqns_l20_20722

def is_single_digit_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

theorem unique_integer_solution_quad_eqns : 
  ∃ (S : Finset (ℕ × ℕ × ℕ)), 
    (∀ (a b c : ℕ), (a, b, c) ∈ S ↔ is_single_digit_prime a ∧ is_single_digit_prime b ∧ is_single_digit_prime c ∧ 
                     ∃ x : ℤ, a * x^2 + b * x + c = 0) ∧ S.card = 7 :=
by
  sorry

end unique_integer_solution_quad_eqns_l20_20722


namespace boys_without_calculators_l20_20063

/-- In Mrs. Robinson's math class, there are 20 boys, and 30 of her students bring their calculators to class. 
    If 18 of the students who brought calculators are girls, then the number of boys who didn't bring their calculators is 8. -/
theorem boys_without_calculators (num_boys : ℕ) (num_students_with_calculators : ℕ) (num_girls_with_calculators : ℕ)
  (h1 : num_boys = 20)
  (h2 : num_students_with_calculators = 30)
  (h3 : num_girls_with_calculators = 18) :
  num_boys - (num_students_with_calculators - num_girls_with_calculators) = 8 :=
by 
  -- proof goes here
  sorry

end boys_without_calculators_l20_20063


namespace not_right_angled_triangle_l20_20033

theorem not_right_angled_triangle 
  (m n : ℝ) 
  (h1 : m > n) 
  (h2 : n > 0)
  : ¬ (m^2 + n^2)^2 = (mn)^2 + (m^2 - n^2)^2 :=
sorry

end not_right_angled_triangle_l20_20033


namespace truck_sand_amount_l20_20851

theorem truck_sand_amount (initial_sand loss_sand final_sand : ℝ) (h1 : initial_sand = 4.1) (h2 : loss_sand = 2.4) :
  initial_sand - loss_sand = final_sand ↔ final_sand = 1.7 := 
by
  sorry

end truck_sand_amount_l20_20851


namespace simple_interest_time_l20_20300

-- Definitions based on given conditions
def SI : ℝ := 640           -- Simple interest
def P : ℝ := 4000           -- Principal
def R : ℝ := 8              -- Rate
def T : ℝ := 2              -- Time in years (correct answer to be proved)

-- Theorem statement
theorem simple_interest_time :
  SI = (P * R * T) / 100 := 
by 
  sorry

end simple_interest_time_l20_20300


namespace value_of_3W5_l20_20290

-- Define the operation W
def W (a b : ℝ) : ℝ := b + 15 * a - a^3

-- State the theorem to prove
theorem value_of_3W5 : W 3 5 = 23 := by
    sorry

end value_of_3W5_l20_20290


namespace solve_fraction_l20_20498

theorem solve_fraction (x : ℚ) : (x^2 + 3*x + 5) / (x + 6) = x + 7 ↔ x = -37 / 10 :=
by
  sorry

end solve_fraction_l20_20498


namespace least_remaining_marbles_l20_20953

/-- 
There are 60 identical marbles forming a tetrahedral pile.
The formula for the number of marbles in a tetrahedral pile up to the k-th level is given by:
∑_(i=1)^k (i * (i + 1)) / 6 = k * (k + 1) * (k + 2) / 6.

We must show that the least number of remaining marbles when 60 marbles are used to form the pile is 4.
-/
theorem least_remaining_marbles : ∃ k : ℕ, (60 - k * (k + 1) * (k + 2) / 6) = 4 :=
by
  sorry

end least_remaining_marbles_l20_20953


namespace fg_eq_gf_condition_l20_20340

/-- Definitions of the functions f and g --/
def f (m n c x : ℝ) : ℝ := m * x + n + c
def g (p q c x : ℝ) : ℝ := p * x + q + c

/-- The main theorem stating the equivalence of the condition for f(g(x)) = g(f(x)) --/
theorem fg_eq_gf_condition (m n p q c x : ℝ) :
  f m n c (g p q c x) = g p q c (f m n c x) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end fg_eq_gf_condition_l20_20340


namespace prime_number_identity_l20_20561

theorem prime_number_identity (p m : ℕ) (h1 : Nat.Prime p) (h2 : m > 0) (h3 : 2 * p^2 + p + 9 = m^2) :
  p = 5 ∧ m = 8 :=
sorry

end prime_number_identity_l20_20561


namespace complement_union_eq_l20_20611

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l20_20611


namespace find_s_l20_20088

def f (x s : ℝ) := 3 * x^5 + 2 * x^4 - x^3 + 4 * x^2 - 5 * x + s

theorem find_s (s : ℝ) (h : f 3 s = 0) : s = -885 :=
  by sorry

end find_s_l20_20088


namespace garden_area_l20_20251

def radius : ℝ := 0.6
def pi_approx : ℝ := 3
def circle_area (r : ℝ) (π : ℝ) := π * r^2

theorem garden_area : circle_area radius pi_approx = 1.08 :=
by
  sorry

end garden_area_l20_20251


namespace taco_castle_parking_lot_l20_20511

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l20_20511


namespace no_overlapping_sale_days_l20_20834

def bookstore_sale_days (d : ℕ) : Prop :=
  d % 4 = 0 ∧ 1 ≤ d ∧ d ≤ 31

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ k : ℕ, d = 2 + 8 * k ∧ 1 ≤ d ∧ d ≤ 31

theorem no_overlapping_sale_days : 
  ∀ d : ℕ, bookstore_sale_days d → ¬ shoe_store_sale_days d :=
by
  intros d h1 h2
  sorry

end no_overlapping_sale_days_l20_20834


namespace total_cost_l20_20952

variable (a b : ℝ)

theorem total_cost (ha : a ≥ 0) (hb : b ≥ 0) : 3 * a + 4 * b = 3 * a + 4 * b :=
by sorry

end total_cost_l20_20952


namespace emma_still_missing_fraction_l20_20434

variable (x : ℕ)  -- Total number of coins Emma received 

-- Conditions
def emma_lost_half (x : ℕ) : ℕ := x / 2
def emma_found_four_fifths (lost : ℕ) : ℕ := 4 * lost / 5

-- Question to prove
theorem emma_still_missing_fraction :
  (x - (x / 2 + emma_found_four_fifths (emma_lost_half x))) / x = 1 / 10 := 
by
  sorry

end emma_still_missing_fraction_l20_20434


namespace parallelogram_area_l20_20987

theorem parallelogram_area (base height : ℝ) (h_base : base = 10) (h_height : height = 7) :
  base * height = 70 := by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l20_20987


namespace gcd_135_81_l20_20442

-- Define the numbers
def a : ℕ := 135
def b : ℕ := 81

-- State the goal: greatest common divisor of a and b is 27
theorem gcd_135_81 : Nat.gcd a b = 27 := by
  sorry

end gcd_135_81_l20_20442


namespace complement_union_A_B_in_U_l20_20616

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l20_20616


namespace find_equation_of_perpendicular_line_l20_20564

noncomputable def line_through_point_perpendicular
    (A : ℝ × ℝ) (a b c : ℝ) (hA : A = (2, 3)) (hLine : a = 2 ∧ b = 1 ∧ c = -5) :
    Prop :=
  ∃ (m : ℝ) (b1 : ℝ), (m = (1 / 2)) ∧
    (b1 = 3 - m * 2) ∧
    (∀ (x y : ℝ), y = m * (x - 2) + 3 → a * x + b * y + c = 0 → x - 2 * y + 4 = 0)

theorem find_equation_of_perpendicular_line :
  line_through_point_perpendicular (2, 3) 2 1 (-5) rfl ⟨rfl, rfl, rfl⟩ :=
sorry

end find_equation_of_perpendicular_line_l20_20564


namespace remainder_when_7n_divided_by_5_l20_20127

theorem remainder_when_7n_divided_by_5 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
  sorry

end remainder_when_7n_divided_by_5_l20_20127


namespace probability_sum_10_three_dice_l20_20900

theorem probability_sum_10_three_dice : 
  let die_faces := {1, 2, 3, 4, 5}
  let total_outcomes := 5^3
  let favorable_outcomes := 
    { (a, b, c) ∈ (die_faces × die_faces × die_faces) | a + b + c = 10 }.card
  let probability := favorable_outcomes / total_outcomes
  probability = 21 / 125 :=
by
  sorry

end probability_sum_10_three_dice_l20_20900


namespace largest_three_digit_number_satisfying_conditions_l20_20078

def valid_digits (a b c : ℕ) : Prop :=
  1 ≤ a ∧ a ≤ 9 ∧ 
  1 ≤ b ∧ b ≤ 9 ∧ 
  1 ≤ c ∧ c ≤ 9 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def sum_of_two_digit_permutations_eq (a b c : ℕ) : Prop :=
  22 * (a + b + c) = 100 * a + 10 * b + c

theorem largest_three_digit_number_satisfying_conditions (a b c : ℕ) :
  valid_digits a b c →
  sum_of_two_digit_permutations_eq a b c →
  100 * a + 10 * b + c ≤ 396 :=
sorry

end largest_three_digit_number_satisfying_conditions_l20_20078


namespace nitrogen_mass_percentage_in_ammonium_phosphate_l20_20291

def nitrogen_mass_percentage
  (molar_mass_N : ℚ)
  (molar_mass_H : ℚ)
  (molar_mass_P : ℚ)
  (molar_mass_O : ℚ)
  : ℚ :=
  let molar_mass_NH4 := molar_mass_N + 4 * molar_mass_H
  let molar_mass_PO4 := molar_mass_P + 4 * molar_mass_O
  let molar_mass_NH4_3_PO4 := 3 * molar_mass_NH4 + molar_mass_PO4
  let mass_N_in_NH4_3_PO4 := 3 * molar_mass_N
  (mass_N_in_NH4_3_PO4 / molar_mass_NH4_3_PO4) * 100

theorem nitrogen_mass_percentage_in_ammonium_phosphate
  (molar_mass_N : ℚ := 14.01)
  (molar_mass_H : ℚ := 1.01)
  (molar_mass_P : ℚ := 30.97)
  (molar_mass_O : ℚ := 16.00)
  : nitrogen_mass_percentage molar_mass_N molar_mass_H molar_mass_P molar_mass_O = 28.19 :=
by
  sorry

end nitrogen_mass_percentage_in_ammonium_phosphate_l20_20291


namespace arlene_average_pace_l20_20018

theorem arlene_average_pace :
  ∃ pace : ℝ, pace = 24 / (6 - 0.75) ∧ pace = 4.57 := 
by
  sorry

end arlene_average_pace_l20_20018


namespace max_integer_k_l20_20192

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x - 1)) / (x - 2)

theorem max_integer_k (x : ℝ) (k : ℕ) (hx : x > 2) :
  (∀ x, x > 2 → f x > (k : ℝ) / (x - 1)) ↔ k ≤ 3 :=
sorry

end max_integer_k_l20_20192


namespace tobee_points_l20_20763

theorem tobee_points (T J S : ℕ) (h1 : J = T + 6) (h2 : S = 2 * (T + 3) - 2) (h3 : T + J + S = 26) : T = 4 := 
by
  sorry

end tobee_points_l20_20763


namespace total_stones_l20_20794

theorem total_stones (x : ℕ) 
  (h1 : x + 6 * x = x * 7 ∧ 7 * x + 6 * x = 2 * x) 
  (h2 : 2 * x = 7 * x - 10) 
  (h3 : 14 * x / 2 = 7 * x) :
  2 * 2 + 14 * 2 + 2 + 7 * 2 + 6 * 2 = 60 := 
by {
  sorry
}

end total_stones_l20_20794


namespace john_steps_l20_20776

/-- John climbs up 9 flights of stairs. Each flight is 10 feet. -/
def flights := 9
def flight_height_feet := 10

/-- Conversion factor between feet and inches. -/
def feet_to_inches := 12

/-- Each step is 18 inches. -/
def step_height_inches := 18

/-- The total number of steps John climbs. -/
theorem john_steps :
  (flights * flight_height_feet * feet_to_inches) / step_height_inches = 60 :=
by
  sorry

end john_steps_l20_20776


namespace cut_scene_length_l20_20978

theorem cut_scene_length (original_length final_length : ℕ) (h1 : original_length = 60) (h2 : final_length = 52) : original_length - final_length = 8 := 
by 
  sorry

end cut_scene_length_l20_20978


namespace ayen_total_jog_time_l20_20418

def jog_time_weekday : ℕ := 30
def jog_time_tuesday : ℕ := jog_time_weekday + 5
def jog_time_friday : ℕ := jog_time_weekday + 25

def total_weekday_jog_time : ℕ := jog_time_weekday * 3
def total_jog_time : ℕ := total_weekday_jog_time + jog_time_tuesday + jog_time_friday

theorem ayen_total_jog_time : total_jog_time / 60 = 3 := by
  sorry

end ayen_total_jog_time_l20_20418


namespace original_total_movies_is_293_l20_20526

noncomputable def original_movies (dvd_to_bluray_ratio : ℕ × ℕ) (additional_blurays : ℕ) (new_ratio : ℕ × ℕ) : ℕ :=
  let original_dvds := dvd_to_bluray_ratio.1
  let original_blurays := dvd_to_bluray_ratio.2
  let added_blurays := additional_blurays
  let new_dvds := new_ratio.1
  let new_blurays := new_ratio.2
  let x := (new_dvds * original_blurays - new_blurays * original_dvds) / (new_blurays * original_dvds - added_blurays * original_blurays)
  let total_movies := (original_dvds * x + original_blurays * x)
  let blurays_after_purchase := original_blurays * x + added_blurays

  if (new_dvds * (original_blurays * x + added_blurays) = new_blurays * (original_dvds * x))
  then 
    (original_dvds * x + original_blurays * x)
  else
    0 -- This case should theoretically never happen if the input ratios are consistent.

theorem original_total_movies_is_293 : original_movies (7, 2) 5 (13, 4) = 293 :=
by sorry

end original_total_movies_is_293_l20_20526


namespace negation_of_p_is_neg_p_l20_20898

-- Define the proposition p
def p : Prop := ∀ x : ℝ, x > 3 → x^3 - 27 > 0

-- Define the negation of proposition p
def neg_p : Prop := ∃ x : ℝ, x > 3 ∧ x^3 - 27 ≤ 0

-- The Lean statement that proves the problem
theorem negation_of_p_is_neg_p : ¬ p ↔ neg_p := by
  sorry

end negation_of_p_is_neg_p_l20_20898


namespace pages_read_per_hour_l20_20376

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l20_20376


namespace actual_time_l20_20735

variables (m_pos : ℕ) (h_pos : ℕ)

-- The mirrored positions
def minute_hand_in_mirror : ℕ := 10
def hour_hand_in_mirror : ℕ := 5

theorem actual_time (m_pos h_pos : ℕ) 
  (hm : m_pos = 2) 
  (hh : h_pos < 7 ∧ h_pos ≥ 6) : 
  m_pos = 10 ∧ h_pos < 7 ∧ h_pos ≥ 6 :=
sorry

end actual_time_l20_20735


namespace sin_cos_identity_angle_between_vectors_l20_20584

theorem sin_cos_identity (θ : ℝ)
  (h1 : (Real.cos θ - 2, Real.sin θ) • (Real.cos θ, Real.sin θ - 2) = -1 / 3) :
  Real.sin θ * Real.cos θ = -5 / 18 :=
begin
  sorry,
end

theorem angle_between_vectors (θ : ℝ)
  (h2 : (Real.cos θ + 2)^2 + (Real.sin θ)^2 = 7)
  (h3 : 0 < θ ∧ θ < Real.pi / 2) :
  Real.cos (Real.angle (0, 2) (Real.cos θ, Real.sin θ)) = Real.sqrt 3 / 2 :=
begin
  sorry,
end

end sin_cos_identity_angle_between_vectors_l20_20584


namespace average_score_for_entire_class_l20_20268

def total_students : ℕ := 100
def assigned_day_percentage : ℝ := 0.70
def make_up_day_percentage : ℝ := 0.30
def assigned_day_avg_score : ℝ := 65
def make_up_day_avg_score : ℝ := 95

theorem average_score_for_entire_class :
  (assigned_day_percentage * total_students * assigned_day_avg_score + make_up_day_percentage * total_students * make_up_day_avg_score) / total_students = 74 := by
  sorry

end average_score_for_entire_class_l20_20268


namespace pages_read_per_hour_l20_20374

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l20_20374


namespace unique_ab_not_determined_l20_20916

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + b * x - Real.sqrt 2

theorem unique_ab_not_determined :
  ∀ (a b : ℝ), a > 0 → b > 0 → 
  f a b (f a b (Real.sqrt 2)) = 1 → False := 
by
  sorry

end unique_ab_not_determined_l20_20916


namespace factor_expression_l20_20873

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l20_20873


namespace equation_descr_circle_l20_20865

theorem equation_descr_circle : ∀ (x y : ℝ), (x - 0) ^ 2 + (y - 0) ^ 2 = 25 → ∃ (c : ℝ × ℝ) (r : ℝ), c = (0, 0) ∧ r = 5 ∧ ∀ (p : ℝ × ℝ), (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2 :=
by
  sorry

end equation_descr_circle_l20_20865


namespace thirtieth_number_in_base_5_l20_20479

theorem thirtieth_number_in_base_5 : Nat.toDigits 5 30 = [1, 1, 0] :=
by
  sorry

end thirtieth_number_in_base_5_l20_20479


namespace range_of_derivative_common_tangent_eq_l20_20000

-- Problem 1
theorem range_of_derivative {θ : ℝ} (hθ : θ ∈ set.Icc 0 (5/12 * real.pi)) :
  let f : ℝ → ℝ := λ x, (sin θ / 3) * x^3 + (sqrt 3 * cos θ / 2) * x^2 + tan θ in
  let f' := λ x, sin θ * x^2 + sqrt 3 * cos θ * x in
  set.Icc (sqrt 2) 2 = { y | ∃ x, f' x = y ? f' 1 } :=
sorry

-- Problem 2
theorem common_tangent_eq {a s t : ℝ} (h : 0 < a) (P : s > 0)
  (commons : let curve1 := λ x : ℝ, a * x^2 ;
             let curve2 := λ x, log x ;
             (let dy1 dx := 2 * a * P s;
              let dy2 dx := 1 / P ;
              let yval := a * P s ^ 2 ?= log P :
              dy1 ?= dy2) ∧ (t = a * P s ^ 2) ∧ (t = log P s)) :
  ∃ t : ℝ, 2 * s - 2 * sqrt e * t - sqrt e = (y | y * (dx / dy)) :=
sorry

end range_of_derivative_common_tangent_eq_l20_20000


namespace multiply_identity_l20_20093

variable (x y : ℝ)

theorem multiply_identity :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 := by
  sorry

end multiply_identity_l20_20093


namespace number_of_dodge_trucks_l20_20512

theorem number_of_dodge_trucks (V T F D : ℕ) (h1 : V = 5)
  (h2 : T = 2 * V) 
  (h3 : F = 2 * T)
  (h4 : F = D / 3) :
  D = 60 := 
by
  sorry

end number_of_dodge_trucks_l20_20512


namespace triangle_area_l20_20907

theorem triangle_area 
  (DE EL EF : ℝ)
  (hDE : DE = 14)
  (hEL : EL = 9)
  (hEF : EF = 17)
  (DL : ℝ)
  (hDL : DE^2 = DL^2 + EL^2)
  (hDL_val : DL = Real.sqrt 115):
  (1/2) * EF * DL = 17 * Real.sqrt 115 / 2 :=
by
  -- Sorry, as the proof is not required.
  sorry

end triangle_area_l20_20907


namespace exsphere_identity_l20_20927

-- Given definitions for heights and radii
variables {h1 h2 h3 h4 r1 r2 r3 r4 : ℝ}

-- Definition of the relationship that needs to be proven
theorem exsphere_identity 
  (h1 h2 h3 h4 r1 r2 r3 r4 : ℝ) :
  2 * (1 / h1 + 1 / h2 + 1 / h3 + 1 / h4) = 1 / r1 + 1 / r2 + 1 / r3 + 1 / r4 := 
sorry

end exsphere_identity_l20_20927


namespace maximum_time_for_3_digit_combination_lock_l20_20977

def max_time_to_open_briefcase : ℕ :=
  let num_combinations := 9 * 9 * 9
  let time_per_trial := 3
  num_combinations * time_per_trial

theorem maximum_time_for_3_digit_combination_lock :
  max_time_to_open_briefcase = 2187 :=
by
  sorry

end maximum_time_for_3_digit_combination_lock_l20_20977


namespace total_jellybeans_l20_20429

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end total_jellybeans_l20_20429


namespace seating_arrangements_l20_20787

theorem seating_arrangements (n : ℕ) (p : ℕ) (p_fixed : ℕ) : 
  n = 9 ∧ p = 8 ∧ p_fixed = 1 →
  let ways_to_seat := (nat.factorial (p - 1)) * (n - p + p_fixed) + 
                      (nat.factorial p / p)
  in ways_to_seat = 10800 :=
by
  intros
  sorry

end seating_arrangements_l20_20787


namespace largest_inscribed_rightangled_parallelogram_l20_20733

theorem largest_inscribed_rightangled_parallelogram (r : ℝ) (x y : ℝ) 
  (parallelogram_inscribed : x = 2 * r * Real.sin (45 * π / 180) ∧ y = 2 * r * Real.cos (45 * π / 180)) :
  x = r * Real.sqrt 2 ∧ y = r * Real.sqrt 2 := 
by 
  sorry

end largest_inscribed_rightangled_parallelogram_l20_20733


namespace green_bows_count_l20_20765

noncomputable def total_bows : ℕ := 36 * 4

def fraction_green : ℚ := 1/6

theorem green_bows_count (red blue green total yellow : ℕ) (h_red : red = total / 4)
  (h_blue : blue = total / 3) (h_green : green = total / 6)
  (h_yellow : yellow = total - red - blue - green)
  (h_yellow_count : yellow = 36) : green = 24 := by
  sorry

end green_bows_count_l20_20765


namespace problem1_problem2_l20_20972

-- Problem 1
theorem problem1 :
  (Real.sqrt 48 / Real.sqrt 3 - Real.sqrt (1 / 2) * Real.sqrt 48 + Real.sqrt 54 = 4 + Real.sqrt 6) :=
by
  sorry

-- Problem 2
theorem problem2 :
  (Real.sqrt 8 + Real.sqrt 32 - Real.sqrt 2 = 5 * Real.sqrt 2) :=
by
  sorry

end problem1_problem2_l20_20972


namespace probability_even_distinct_digits_l20_20284

theorem probability_even_distinct_digits :
  let count_even_distinct := 1960
  let total_numbers := 8000
  count_even_distinct / total_numbers = 49 / 200 :=
by
  sorry

end probability_even_distinct_digits_l20_20284


namespace second_caterer_cheaper_l20_20489

theorem second_caterer_cheaper (x : ℕ) :
  (∀ n : ℕ, n < x → 150 + 18 * n ≤ 250 + 15 * n) ∧ (150 + 18 * x > 250 + 15 * x) ↔ x = 34 :=
by sorry

end second_caterer_cheaper_l20_20489


namespace algebraic_expression_value_l20_20652

variable (a : ℝ)

theorem algebraic_expression_value (h : a = Real.sqrt 2) :
  (a / (a - 1)^2) / (1 + 1 / (a - 1)) = Real.sqrt 2 + 1 :=
by
  sorry

end algebraic_expression_value_l20_20652


namespace triangle_construction_possible_l20_20029

-- Define the entities involved
variables {α β : ℝ} {a c : ℝ}

-- State the theorem
theorem triangle_construction_possible (a c : ℝ) (h : α = 2 * β) : a > (2 / 3) * c :=
sorry

end triangle_construction_possible_l20_20029


namespace cat_finishes_food_on_tuesday_second_week_l20_20016

def initial_cans : ℚ := 8
def extra_treat : ℚ := 1 / 6
def morning_diet : ℚ := 1 / 4
def evening_diet : ℚ := 1 / 5

def daily_consumption (morning_diet evening_diet : ℚ) : ℚ :=
  morning_diet + evening_diet

def first_day_consumption (daily_consumption extra_treat : ℚ) : ℚ :=
  daily_consumption + extra_treat

theorem cat_finishes_food_on_tuesday_second_week 
  (initial_cans extra_treat morning_diet evening_diet : ℚ)
  (h1 : initial_cans = 8)
  (h2 : extra_treat = 1 / 6)
  (h3 : morning_diet = 1 / 4)
  (h4 : evening_diet = 1 / 5) :
  -- The computation must be performed here or defined previously
  -- The proof of this theorem is the task, the result is postulated as a theorem
  final_day = "Tuesday (second week)" :=
sorry

end cat_finishes_food_on_tuesday_second_week_l20_20016


namespace sculpture_and_base_height_l20_20287

def height_sculpture : ℕ := 2 * 12 + 10
def height_base : ℕ := 8
def total_height : ℕ := 42

theorem sculpture_and_base_height :
  height_sculpture + height_base = total_height :=
by
  -- provide the necessary proof steps here
  sorry

end sculpture_and_base_height_l20_20287


namespace min_overlap_percentage_l20_20098

theorem min_overlap_percentage (A B : ℝ) (hA : A = 0.9) (hB : B = 0.8) : ∃ x, x = 0.7 := 
by sorry

end min_overlap_percentage_l20_20098


namespace tan_alpha_plus_pi_div_four_l20_20887

theorem tan_alpha_plus_pi_div_four
  (α : ℝ)
  (a : ℝ × ℝ := (3, 4))
  (b : ℝ × ℝ := (Real.sin α, Real.cos α))
  (h_parallel : ∃ k : ℝ, b = (k * a.1, k * a.2)) :
  Real.tan (α + Real.pi / 4) = 7 := by
  sorry

end tan_alpha_plus_pi_div_four_l20_20887


namespace ratio_of_shaded_to_white_l20_20817

theorem ratio_of_shaded_to_white (A : ℝ) : 
  let shaded_area := 5 * A
  let unshaded_area := 3 * A
  shaded_area / unshaded_area = 5 / 3 := by
  sorry

end ratio_of_shaded_to_white_l20_20817


namespace people_per_bus_l20_20535

def num_vans : ℝ := 6.0
def num_buses : ℝ := 8.0
def people_per_van : ℝ := 6.0
def extra_people : ℝ := 108.0

theorem people_per_bus :
  let people_vans := num_vans * people_per_van
  let people_buses := people_vans + extra_people
  let people_per_bus := people_buses / num_buses
  people_per_bus = 18.0 :=
by 
  sorry

end people_per_bus_l20_20535


namespace owen_profit_l20_20231

/-- Given the initial purchases and sales, calculate Owen's overall profit. -/
theorem owen_profit :
  let boxes_9_dollars := 8
  let boxes_12_dollars := 4
  let cost_9_dollars := 9
  let cost_12_dollars := 12
  let masks_per_box := 50
  let packets_25_pieces := 100
  let price_25_pieces := 5
  let packets_100_pieces := 28
  let price_100_pieces := 12
  let remaining_masks1 := 150
  let price_remaining1 := 3
  let remaining_masks2 := 150
  let price_remaining2 := 4
  let total_cost := (boxes_9_dollars * cost_9_dollars) + (boxes_12_dollars * cost_12_dollars)
  let total_repacked_masks := (packets_25_pieces * price_25_pieces) + (packets_100_pieces * price_100_pieces)
  let total_remaining_masks := (remaining_masks1 * price_remaining1) + (remaining_masks2 * price_remaining2)
  let total_revenue := total_repacked_masks + total_remaining_masks
  let overall_profit := total_revenue - total_cost
  overall_profit = 1766 := by
  sorry

end owen_profit_l20_20231


namespace find_common_ratio_l20_20372

noncomputable def geometric_series (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * (1 - q^n) / (1 - q)

theorem find_common_ratio (a_1 : ℝ) (q : ℝ) (n : ℕ) (S_n : ℕ → ℝ)
  (h1 : ∀ n, S_n n = geometric_series a_1 q n)
  (h2 : S_n 3 = (2 * a_1 + a_1 * q) / 2)
  : q = -1/2 :=
  sorry

end find_common_ratio_l20_20372


namespace max_value_f_l20_20304

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 1)

theorem max_value_f (h : ∀ ε > (0 : ℝ), ∃ x : ℝ, x < 1 ∧ ε < f x) : ∀ x : ℝ, x < 1 → f x ≤ -1 :=
by
  intros x hx
  dsimp [f]
  -- Proof steps are omitted.
  sorry

example (h: ∀ ε > 0, ∃ x : ℝ, x < 1 ∧ ε < f x) : ∃ x : ℝ, x < 1 ∧ f x = -1 :=
by
  use 0
  -- Proof steps are omitted.
  sorry

end max_value_f_l20_20304


namespace number_of_oranges_l20_20796

theorem number_of_oranges (B T O : ℕ) (h₁ : B + T = 178) (h₂ : B + T + O = 273) : O = 95 :=
by
  -- Begin proof here
  sorry

end number_of_oranges_l20_20796


namespace total_people_needed_to_lift_l20_20212

theorem total_people_needed_to_lift (lift_car : ℕ) (lift_truck : ℕ) (num_cars : ℕ) (num_trucks : ℕ) : 
  lift_car = 5 → 
  lift_truck = 2 * lift_car → 
  num_cars = 6 → 
  num_trucks = 3 → 
  6 * 5 + 3 * (2 * 5) = 60 := 
by
  intros hc ht hcars htrucks
  rw[hc, hcars, htrucks]
  rw[ht]
  sorry

end total_people_needed_to_lift_l20_20212


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20831

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20831


namespace right_triangle_angles_l20_20746

theorem right_triangle_angles (c : ℝ) (t : ℝ) (h : t = c^2 / 8) :
  ∃(A B: ℝ), A = 90 ∧ (B = 75 ∨ B = 15) :=
by
  sorry

end right_triangle_angles_l20_20746


namespace taco_castle_parking_lot_l20_20510

variable (D F T V : ℕ)

theorem taco_castle_parking_lot (h1 : F = D / 3) (h2 : F = 2 * T) (h3 : V = T / 2) (h4 : V = 5) : D = 60 :=
by
  sorry

end taco_castle_parking_lot_l20_20510


namespace eight_pow_2012_mod_10_l20_20881

theorem eight_pow_2012_mod_10 : (8 ^ 2012) % 10 = 2 :=
by {
  sorry
}

end eight_pow_2012_mod_10_l20_20881


namespace arithmetic_sequence_a5_value_l20_20477

variable (a : ℕ → ℝ)
variable (a_2 a_5 a_8 : ℝ)
variable (h1 : a 2 + a 8 = 15 - a 5)

/-- In an arithmetic sequence {a_n}, given that a_2 + a_8 = 15 - a_5, prove that a_5 equals 5. -/ 
theorem arithmetic_sequence_a5_value (h1 : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
sorry

end arithmetic_sequence_a5_value_l20_20477


namespace pages_read_per_hour_l20_20375

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end pages_read_per_hour_l20_20375


namespace complement_union_eq_l20_20610

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l20_20610


namespace solve_x_minus_y_l20_20387

theorem solve_x_minus_y :
  (2 = 0.25 * x) → (2 = 0.1 * y) → (x - y = -12) :=
by
  sorry

end solve_x_minus_y_l20_20387


namespace real_yield_correct_l20_20823

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l20_20823


namespace successive_product_l20_20942

theorem successive_product (n : ℤ) (h : n * (n + 1) = 4160) : n = 64 :=
sorry

end successive_product_l20_20942


namespace daryl_max_crate_weight_l20_20031

variable (crates : ℕ) (weight_nails : ℕ) (bags_nails : ℕ)
variable (weight_hammers : ℕ) (bags_hammers : ℕ) (weight_planks : ℕ)
variable (bags_planks : ℕ) (weight_left_out : ℕ)

def max_weight_per_crate (total_weight: ℕ) (total_crates: ℕ) : ℕ :=
  total_weight / total_crates

-- State the problem in Lean
theorem daryl_max_crate_weight
  (h1 : crates = 15) 
  (h2 : bags_nails = 4) 
  (h3 : weight_nails = 5)
  (h4 : bags_hammers = 12) 
  (h5 : weight_hammers = 5) 
  (h6 : bags_planks = 10) 
  (h7 : weight_planks = 30) 
  (h8 : weight_left_out = 80):
  max_weight_per_crate ((bags_nails * weight_nails + bags_hammers * weight_hammers + bags_planks * weight_planks) - weight_left_out) crates = 20 :=
  by sorry

end daryl_max_crate_weight_l20_20031


namespace complement_union_A_B_eq_neg2_0_l20_20608

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l20_20608


namespace solution_set_inequality_l20_20945

theorem solution_set_inequality (x : ℝ) : 4 * x < 3 * x + 2 → x < 2 :=
by
  intro h
  -- Add actual proof here, but for now; we use sorry
  sorry

end solution_set_inequality_l20_20945


namespace total_jellybeans_l20_20427

-- Definitions of the conditions
def caleb_jellybeans : ℕ := 3 * 12
def sophie_jellybeans (caleb_jellybeans : ℕ) : ℕ := caleb_jellybeans / 2

-- Statement of the proof problem
theorem total_jellybeans (C : caleb_jellybeans = 36) (S : sophie_jellybeans 36 = 18) :
  caleb_jellybeans + sophie_jellybeans 36 = 54 :=
by
  sorry

end total_jellybeans_l20_20427


namespace total_cost_of_soup_l20_20516

theorem total_cost_of_soup :
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  in cost_of_vegetables + cost_of_beef = 36 :=
by
  let beef_pounds := 4
  let vegetable_pounds := 6
  let vegetable_cost_per_pound := 2
  let beef_cost_per_pound := 3 * vegetable_cost_per_pound
  let cost_of_vegetables := vegetable_cost_per_pound * vegetable_pounds
  let cost_of_beef := beef_cost_per_pound * beef_pounds
  show cost_of_vegetables + cost_of_beef = 36
  sorry

end total_cost_of_soup_l20_20516


namespace find_value_l20_20409

theorem find_value (x : ℤ) (h : 3 * x - 45 = 159) : (x + 32) * 12 = 1200 :=
by
  sorry

end find_value_l20_20409


namespace pradeep_passing_percentage_l20_20100

theorem pradeep_passing_percentage (score failed_by max_marks : ℕ) :
  score = 185 → failed_by = 25 → max_marks = 600 →
  ((score + failed_by) / max_marks : ℚ) * 100 = 35 :=
by
  intros h_score h_failed_by h_max_marks
  sorry

end pradeep_passing_percentage_l20_20100


namespace Samanta_points_diff_l20_20064

variables (Samanta Mark Eric : ℕ)

/-- In a game, Samanta has some more points than Mark, Mark has 50% more points than Eric,
Eric has 6 points, and Samanta, Mark, and Eric have a total of 32 points. Prove that Samanta
has 8 more points than Mark. -/
theorem Samanta_points_diff 
    (h1 : Mark = Eric + Eric / 2) 
    (h2 : Eric = 6) 
    (h3 : Samanta + Mark + Eric = 32)
    : Samanta - Mark = 8 :=
sorry

end Samanta_points_diff_l20_20064


namespace greta_received_more_letters_l20_20195

noncomputable def number_of_letters_difference : ℕ :=
  let B := 40
  let M (G : ℕ) := 2 * (G + B)
  let total (G : ℕ) := G + B + M G
  let G := 50 -- Solved from the total equation
  G - B

theorem greta_received_more_letters : number_of_letters_difference = 10 :=
by
  sorry

end greta_received_more_letters_l20_20195


namespace hyperbola_focus_proof_l20_20563

noncomputable def hyperbola_focus : ℝ × ℝ :=
  (-3, 2.5 + 2 * Real.sqrt 3)

theorem hyperbola_focus_proof :
  ∃ x y : ℝ, 
  -2 * x^2 + 4 * y^2 - 12 * x - 20 * y + 5 = 0 
  → (x = -3) ∧ (y = 2.5 + 2 * Real.sqrt 3) := 
by 
  sorry

end hyperbola_focus_proof_l20_20563


namespace isosceles_triangle_perimeter_l20_20327

theorem isosceles_triangle_perimeter 
    (a b : ℕ) (h_iso : a = 3 ∨ a = 5) (h_other : b = 3 ∨ b = 5) 
    (h_distinct : a ≠ b) : 
    ∃ p : ℕ, p = (3 + 3 + 5) ∨ p = (5 + 5 + 3) :=
by
  sorry

end isosceles_triangle_perimeter_l20_20327


namespace no_obtuse_equilateral_triangle_exists_l20_20559

theorem no_obtuse_equilateral_triangle_exists :
  ¬(∃ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = π ∧ a > π/2 ∧ b > π/2 ∧ c > π/2) :=
sorry

end no_obtuse_equilateral_triangle_exists_l20_20559


namespace remainder_7n_mod_5_l20_20125

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
by 
  sorry

end remainder_7n_mod_5_l20_20125


namespace drawings_with_colored_pencils_l20_20811

-- Definitions based on conditions
def total_drawings : Nat := 25
def blending_markers_drawings : Nat := 7
def charcoal_drawings : Nat := 4
def colored_pencils_drawings : Nat := total_drawings - (blending_markers_drawings + charcoal_drawings)

-- Theorem to be proven
theorem drawings_with_colored_pencils : colored_pencils_drawings = 14 :=
by
  sorry

end drawings_with_colored_pencils_l20_20811


namespace find_a_range_l20_20053

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 1 then (a + 3) * x - 5 else 2 * a / x

theorem find_a_range (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≤ f a x2) → -2 ≤ a ∧ a < 0 :=
by
  sorry

end find_a_range_l20_20053


namespace area_of_moving_point_l20_20131

theorem area_of_moving_point (a b : ℝ) :
  (∀ (x y : ℝ), abs x ≤ 1 ∧ abs y ≤ 1 → a * x - 2 * b * y ≤ 2) →
  ∃ (A : ℝ), A = 8 := sorry

end area_of_moving_point_l20_20131


namespace max_value_of_expression_l20_20782

open Real

theorem max_value_of_expression (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  a + sqrt (a * b) + (a * b * c) ^ (1 / 4) ≤ 10 / 3 := sorry

end max_value_of_expression_l20_20782


namespace trip_total_time_l20_20533

theorem trip_total_time 
  (x : ℕ) 
  (h1 : 30 * 5 = 150) 
  (h2 : 42 * x + 150 = 38 * (x + 5)) 
  (h3 : 38 = (150 + 42 * x) / (5 + x)) : 
  5 + x = 15 := by
  sorry

end trip_total_time_l20_20533


namespace daniel_total_earnings_l20_20558

-- Definitions of conditions
def fabric_delivered_monday : ℕ := 20
def fabric_delivered_tuesday : ℕ := 2 * fabric_delivered_monday
def fabric_delivered_wednesday : ℕ := fabric_delivered_tuesday / 4
def total_fabric_delivered : ℕ := fabric_delivered_monday + fabric_delivered_tuesday + fabric_delivered_wednesday

def cost_per_yard : ℕ := 2
def total_earnings : ℕ := total_fabric_delivered * cost_per_yard

-- Proposition to be proved
theorem daniel_total_earnings : total_earnings = 140 := by
  sorry

end daniel_total_earnings_l20_20558


namespace number_of_non_Speedsters_l20_20696

theorem number_of_non_Speedsters (V : ℝ) (h0 : (4 / 15) * V = 12) : (2 / 3) * V = 30 :=
by
  -- The conditions are such that:
  -- V is the total number of vehicles.
  -- (4 / 15) * V = 12 means 4/5 of 1/3 of the total vehicles are convertibles.
  -- We need to prove that 2/3 of the vehicles are not Speedsters.
  sorry

end number_of_non_Speedsters_l20_20696


namespace p_necessary_not_sufficient_q_l20_20341

theorem p_necessary_not_sufficient_q (x : ℝ) : (|x| = 2) → (x = 2) → (|x| = 2 ∧ (x ≠ 2 ∨ x = -2)) := by
  intros h_p h_q
  sorry

end p_necessary_not_sufficient_q_l20_20341


namespace original_price_is_100_l20_20632

variable (P : ℝ) -- Declare the original price P as a real number
variable (h : 0.10 * P = 10) -- The condition given in the problem

theorem original_price_is_100 (P : ℝ) (h : 0.10 * P = 10) : P = 100 := by
  sorry

end original_price_is_100_l20_20632


namespace find_PS_l20_20077

theorem find_PS 
    (P Q R S : Type)
    (PQ PR : ℝ)
    (h : ℝ) 
    (ratio_QS_SR : ℝ)
    (hyp1 : PQ = 13)
    (hyp2 : PR = 20)
    (hyp3 : ratio_QS_SR = 3/7) :
    h = Real.sqrt (117.025) :=
by
  -- Proof steps would go here, but we are just stating the theorem
  sorry

end find_PS_l20_20077


namespace range_of_a_l20_20505

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3 * a else a^x - 2

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by sorry

end range_of_a_l20_20505


namespace circle_center_radius_l20_20657

theorem circle_center_radius :
  ∃ (h : ℝ × ℝ) (r : ℝ),
    (h = (1, -3)) ∧ (r = 2) ∧ ∀ x y : ℝ, 
    (x - h.1)^2 + (y - h.2)^2 = 4 → x^2 + y^2 - 2*x + 6*y + 6 = 0 :=
sorry

end circle_center_radius_l20_20657


namespace compute_expression_l20_20028

theorem compute_expression : (-1) ^ 2014 + (π - 3.14) ^ 0 - (1 / 2) ^ (-2) = -2 := by
  sorry

end compute_expression_l20_20028


namespace problem_geometric_description_of_set_T_l20_20319

open Complex

def set_T (a b : ℝ) : ℂ := a + b * I

theorem problem_geometric_description_of_set_T :
  {w : ℂ | ∃ a b : ℝ, w = set_T a b ∧
    (im ((5 - 3 * I) * w) = 2 * re ((5 - 3 * I) * w))} =
  {w : ℂ | ∃ a : ℝ, w = set_T a (-(13/5) * a)} :=
sorry

end problem_geometric_description_of_set_T_l20_20319


namespace train_passing_time_l20_20850

/-- The problem defines a train of length 110 meters traveling at 40 km/hr, 
    passing a man who is running at 5 km/hr in the opposite direction.
    We want to prove that the time it takes for the train to pass the man is 8.8 seconds. -/
theorem train_passing_time :
  ∀ (train_length : ℕ) (train_speed man_speed : ℕ), 
  train_length = 110 → train_speed = 40 → man_speed = 5 →
  (∃ time : ℚ, time = 8.8) :=
by
  intros train_length train_speed man_speed h_train_length h_train_speed h_man_speed
  sorry

end train_passing_time_l20_20850


namespace line_equation_l20_20190

theorem line_equation {x y : ℝ} (m b : ℝ) (h1 : m = 2) (h2 : b = -3) :
    (∃ (f : ℝ → ℝ), (∀ x, f x = m * x + b) ∧ (∀ x, 2 * x - f x - 3 = 0)) :=
by
  sorry

end line_equation_l20_20190


namespace most_likely_outcome_l20_20858

/-- Definition of being a boy or girl -/
inductive Gender
| boy : Gender
| girl : Gender

/-- Probability of each gender -/
def prob_gender (g : Gender) : ℚ :=
  1 / 2

/-- Calculate the probability of all children being a specific gender -/
def prob_all_same_gender (n : ℕ) (g : Gender) : ℚ :=
  (prob_gender g) ^ n

/-- Number of ways to choose k girls out of n children -/
def num_ways_k_girls_out_of_n (n k : ℕ) : ℕ :=
  Nat.choose n k

/-- Calculate the probability of having exactly k girls out of n children -/
def prob_exactly_k_girls (n k : ℕ) : ℚ :=
  (num_ways_k_girls_out_of_n n k) * (prob_gender Gender.girl) ^ k * (prob_gender Gender.boy) ^ (n - k)

theorem most_likely_outcome :
  let A := prob_all_same_gender 8 Gender.boy,
      B := prob_all_same_gender 8 Gender.girl,
      C := prob_exactly_k_girls 8 4,
      D := prob_exactly_k_girls 8 6 + prob_exactly_k_girls 8 2 in
  C > A ∧ C > B ∧ C > D :=
by
  sorry

end most_likely_outcome_l20_20858


namespace find_multiple_l20_20986

theorem find_multiple (x m : ℕ) (h₁ : x = 69) (h₂ : x - 18 = m * (86 - x)) : m = 3 :=
by
  sorry

end find_multiple_l20_20986


namespace triangle_area_ellipse_l20_20313

open Real

noncomputable def ellipse_foci (a b : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let c := sqrt (a^2 + b^2)
  ((-c, 0), (c, 0))

theorem triangle_area_ellipse 
  (a : ℝ) (b : ℝ) 
  (h1 : a = sqrt 2) (h2 : b = 1) 
  (F1 F2 : ℝ × ℝ) 
  (hfoci : ellipse_foci a b = (F1, F2))
  (hF2 : F2 = (sqrt 3, 0))
  (A B : ℝ × ℝ)
  (hA : A = (0, -1))
  (hB : B = (0, -1))
  (h_inclination : ∃ θ, θ = pi / 4 ∧ (B.1 - A.1) / (B.2 - A.2) = tan θ) :
  F1 = (-sqrt 3, 0) → 
  1/2 * (B.1 - A.1) * (B.2 - A.2) = 4/3 :=
sorry

end triangle_area_ellipse_l20_20313


namespace probability_of_meeting_l20_20959

noncomputable def meeting_probability : ℝ := 
  let f := λ (x y : ℝ), |x - y| ≤ (1 / 4)
  let P := MeasureTheory.Measure.univ_measure
  MeasureTheory.Probability.ProbabilityMeasure.measure (MeasureTheory.Interval.unitSquare) {p : ℝ × ℝ | f p.1 p.2}

theorem probability_of_meeting : meeting_probability = 7 / 16 :=
by
  sorry

end probability_of_meeting_l20_20959


namespace total_birds_caught_l20_20137

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l20_20137


namespace find_divisor_l20_20522

def positive_integer := {e : ℕ // e > 0}

theorem find_divisor (d : ℕ) :
  (∃ e : positive_integer, (e.val % 13 = 2)) →
  (∃ n : ℕ, n < 180 ∧ n % d = 5 ∧ ∀ m < 180, m % d = 5 → m = n) →
  d = 175 :=
by
  sorry

end find_divisor_l20_20522


namespace inflation_over_two_years_real_interest_rate_l20_20828

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l20_20828


namespace relatively_prime_sequence_l20_20221

theorem relatively_prime_sequence (k : ℤ) (hk : k > 1) :
  ∃ (a b : ℤ) (x : ℕ → ℤ),
    a > 0 ∧ b > 0 ∧
    (∀ n, x (n + 2) = x (n + 1) + x n) ∧
    x 0 = a ∧ x 1 = b ∧ ∀ n, gcd (x n) (4 * k^2 - 5) = 1 :=
by
  sorry

end relatively_prime_sequence_l20_20221


namespace find_bases_of_isosceles_trapezoid_l20_20981

noncomputable def isosceles_trapezoid_bases (c d : ℝ) (h : c < d) : Prop :=
  ∃ b1 b2 : ℝ, b1 = (sqrt (d + c) + sqrt (d - c))^2 / 2 ∧ b2 = (sqrt (d + c) - sqrt (d - c))^2 / 2

theorem find_bases_of_isosceles_trapezoid (c d : ℝ) (h : c < d) :
  ∃ (b1 b2 : ℝ), b1 = (sqrt (d + c) + sqrt (d - c))^2 / 2 ∧ b2 = (sqrt (d + c) - sqrt (d - c))^2 / 2 := 
sorry

end find_bases_of_isosceles_trapezoid_l20_20981


namespace quadratic_specific_a_l20_20325

noncomputable def quadratic_root_condition (a : ℝ) : Prop :=
  ∃ x : ℝ, (a + 2) * x^2 + 2 * a * x + 1 = 0

theorem quadratic_specific_a (a : ℝ) (h : quadratic_root_condition a) :
  a = 2 ∨ a = -1 :=
sorry

end quadratic_specific_a_l20_20325


namespace M_supseteq_P_l20_20755

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 - 4}
def P : Set ℝ := {y | |y - 3| ≤ 1}

theorem M_supseteq_P : M ⊇ P := 
sorry

end M_supseteq_P_l20_20755


namespace blue_chairs_fewer_than_yellow_l20_20468

theorem blue_chairs_fewer_than_yellow :
  ∀ (red_chairs yellow_chairs chairs_left total_chairs blue_chairs : ℕ),
    red_chairs = 4 →
    yellow_chairs = 2 * red_chairs →
    chairs_left = 15 →
    total_chairs = chairs_left + 3 →
    blue_chairs = total_chairs - (red_chairs + yellow_chairs) →
    yellow_chairs - blue_chairs = 2 :=
by sorry

end blue_chairs_fewer_than_yellow_l20_20468


namespace modular_inverse_7_10000_l20_20521

theorem modular_inverse_7_10000 :
  (7 * 8571) % 10000 = 1 := 
sorry

end modular_inverse_7_10000_l20_20521


namespace apple_price_difference_l20_20250

variable (S R F : ℝ)

theorem apple_price_difference (h1 : S + R > R + F) (h2 : F = S - 250) :
  (S + R) - (R + F) = 250 :=
by
  sorry

end apple_price_difference_l20_20250


namespace product_quantities_l20_20211

theorem product_quantities (a b x y : ℝ) 
  (h1 : a * x + b * y = 1500)
  (h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529)
  (h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5)
  (h4 : 205 < 2 * x + y ∧ 2 * x + y < 210) :
  (x + 2 * y = 186) ∧ (73 ≤ x ∧ x ≤ 75) :=
by
  sorry

end product_quantities_l20_20211


namespace g_sum_zero_l20_20111

def g (x : ℝ) : ℝ := x^2 - 2013 * x

theorem g_sum_zero (a b : ℝ) (h₁ : g a = g b) (h₂ : a ≠ b) : g (a + b) = 0 :=
sorry

end g_sum_zero_l20_20111


namespace quadratic_roots_difference_square_l20_20219

theorem quadratic_roots_difference_square (a b : ℝ) (h : 2 * a^2 - 8 * a + 6 = 0 ∧ 2 * b^2 - 8 * b + 6 = 0) :
  (a - b) ^ 2 = 4 :=
sorry

end quadratic_roots_difference_square_l20_20219


namespace parabola_tangent_circle_l20_20055

noncomputable def parabola_focus (p : ℝ) (hp : p > 0) : ℝ × ℝ := (p / 2, 0)

theorem parabola_tangent_circle (p : ℝ) (hp : p > 0)
  (x0 : ℝ) (hx0 : x0 = p)
  (M : ℝ × ℝ) (hM : M = (x0, 2 * (Real.sqrt 2)))
  (MA AF : ℝ) (h_ratio : MA / AF = 2) :
  p = 2 :=
by
  sorry

end parabola_tangent_circle_l20_20055


namespace speed_of_second_train_l20_20541

noncomputable def speed_of_first_train_kmph := 60 -- km/h
noncomputable def speed_of_first_train_mps := (speed_of_first_train_kmph * 1000) / 3600 -- m/s
noncomputable def length_of_first_train := 145 -- m
noncomputable def length_of_second_train := 165 -- m
noncomputable def time_to_cross := 8 -- seconds
noncomputable def total_distance := length_of_first_train + length_of_second_train -- m
noncomputable def relative_speed := total_distance / time_to_cross -- m/s

theorem speed_of_second_train (V : ℝ) :
  V * 1000 / 3600 + 60 * 1000 / 3600 = 38.75 →
  V = 79.5 := by {
  sorry
}

end speed_of_second_train_l20_20541


namespace least_multiple_of_seven_not_lucky_is_14_l20_20705

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven_not_lucky (n : ℕ) : Prop :=
  n % 7 = 0 ∧ ¬ is_lucky_integer n

theorem least_multiple_of_seven_not_lucky_is_14 : 
  ∃ n : ℕ, is_multiple_of_seven_not_lucky n ∧ ∀ m, (is_multiple_of_seven_not_lucky m → n ≤ m) :=
⟨ 14, 
  by {
    -- Proof is provided here, but for now, we use "sorry"
    sorry
  }⟩

end least_multiple_of_seven_not_lucky_is_14_l20_20705


namespace winner_votes_more_than_loser_l20_20904

-- Define the initial conditions.
def winner_percentage : ℝ := 0.55
def total_students : ℕ := 2000
def voting_percentage : ℝ := 0.25

-- Define the number of students who voted
def num_voted := total_students * (voting_percentage : ℕ)

-- Define the votes received by the winner and the loser,
-- based on the percentage of the total votes.
def winner_votes := num_voted * winner_percentage
def loser_votes := num_voted * (1 - winner_percentage)

-- Define the difference in votes between the winner and the loser.
def vote_difference := winner_votes - loser_votes

-- The target theorem to prove.
theorem winner_votes_more_than_loser : vote_difference = 50 := by
  sorry

end winner_votes_more_than_loser_l20_20904


namespace smallest_integer_C_l20_20172

-- Define the function f(n) = 6^n / n!
def f (n : ℕ) : ℚ := (6 ^ n) / (Nat.factorial n)

theorem smallest_integer_C (C : ℕ) (h : ∀ n : ℕ, n > 0 → f n ≤ C) : C = 65 :=
by
  sorry

end smallest_integer_C_l20_20172


namespace log_sum_l20_20550

-- Define the common logarithm function using Lean's natural logarithm with a change of base
noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_sum : log_base_10 5 + log_base_10 0.2 = 0 :=
by
  -- Placeholder for the proof to be completed
  sorry

end log_sum_l20_20550


namespace intercept_condition_slope_condition_l20_20637

theorem intercept_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 - 2 * m - 3) * -3 + (2 * m^2 + m - 1) * 0 + (-2 * m + 6) = 0 → 
  m = -5 / 3 := 
  sorry

theorem slope_condition (m : ℚ) (h : m ≠ -1) : 
  (m^2 + 2 * m - 4) = 0 → 
  m = 4 / 3 := 
  sorry

end intercept_condition_slope_condition_l20_20637


namespace avery_donation_clothes_l20_20019

theorem avery_donation_clothes :
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  shirts + pants + shorts = 16 :=
by
  let shirts := 4
  let pants := 2 * shirts
  let shorts := pants / 2
  show shirts + pants + shorts = 16
  sorry

end avery_donation_clothes_l20_20019


namespace part1_part2_l20_20883

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 2 * Real.log x - a * x ^ 2 + 1

theorem part1 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ (0 < a ∧ a < 1) :=
sorry

theorem part2 (a : ℝ) :
  (∃ α β m : ℝ, 1 ≤ α ∧ α ≤ 4 ∧ 1 ≤ β ∧ β ≤ 4 ∧ β - α = 1 ∧ f α a = m ∧ f β a = m) ↔ 
  (Real.log 4 / 3 * (2 / 7) ≤ a ∧ a ≤ Real.log 2 * (2 / 3)) :=
sorry

end part1_part2_l20_20883


namespace pages_per_hour_l20_20382

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l20_20382


namespace triangle_segments_equivalence_l20_20762

variable {a b c p : ℝ}

theorem triangle_segments_equivalence (h_acute : a^2 + b^2 > c^2) 
  (h_perpendicular : ∃ h: ℝ, h^2 = c^2 - (a - p)^2 ∧ h^2 = b^2 - p^2) :
  a / (c + b) = (c - b) / (a - 2 * p) := by
sorry

end triangle_segments_equivalence_l20_20762


namespace find_s_is_neg4_l20_20781

def g (x s : ℝ) : ℝ := 3 * x^4 + 2 * x^3 - x^2 - 4 * x + s

theorem find_s_is_neg4 : (∃ s : ℝ, g (-1) s = 0) ↔ (s = -4) :=
sorry

end find_s_is_neg4_l20_20781


namespace power_addition_l20_20021

theorem power_addition :
  (-2 : ℤ) ^ 2009 + (-2 : ℤ) ^ 2010 = 2 ^ 2009 :=
by
  sorry

end power_addition_l20_20021


namespace total_weight_of_fruits_l20_20620

/-- Define the given conditions in Lean -/
def weight_of_orange_bags (n : ℕ) : ℝ :=
  if n = 12 then 24 else 0

def weight_of_apple_bags (n : ℕ) : ℝ :=
  if n = 8 then 30 else 0

/-- Prove that the total weight of 5 bags of oranges and 4 bags of apples is 25 pounds given the conditions -/
theorem total_weight_of_fruits :
  weight_of_orange_bags 12 / 12 * 5 + weight_of_apple_bags 8 / 8 * 4 = 25 :=
by sorry

end total_weight_of_fruits_l20_20620


namespace remainder_7n_mod_5_l20_20124

theorem remainder_7n_mod_5 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
by 
  sorry

end remainder_7n_mod_5_l20_20124


namespace five_times_remaining_is_400_l20_20790

-- Define the conditions
def original_marbles := 800
def marbles_per_friend := 120
def num_friends := 6

-- Calculate total marbles given away
def marbles_given_away := num_friends * marbles_per_friend

-- Calculate marbles remaining after giving away
def marbles_remaining := original_marbles - marbles_given_away

-- Question: what is five times the marbles remaining?
def five_times_remaining_marbles := 5 * marbles_remaining

-- The proof problem: prove that this equals 400
theorem five_times_remaining_is_400 : five_times_remaining_marbles = 400 :=
by
  -- The proof would go here
  sorry

end five_times_remaining_is_400_l20_20790


namespace complement_union_A_B_eq_neg2_0_l20_20609

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l20_20609


namespace charlie_certain_instrument_l20_20431

theorem charlie_certain_instrument :
  ∃ (x : ℕ), (1 + 2 + x) + (2 + 1 + 0) = 7 → x = 1 :=
by
  sorry

end charlie_certain_instrument_l20_20431


namespace martin_distance_l20_20351

def speed : ℝ := 12.0  -- Speed in miles per hour
def time : ℝ := 6.0    -- Time in hours

theorem martin_distance : (speed * time) = 72.0 :=
by
  sorry

end martin_distance_l20_20351


namespace pirate_prob_l20_20538

def probability_treasure_no_traps := 1 / 3
def probability_traps_no_treasure := 1 / 6
def probability_neither := 1 / 2

theorem pirate_prob : (70 : ℝ) * ((1 / 3)^4 * (1 / 2)^4) = 35 / 648 := by
  sorry

end pirate_prob_l20_20538


namespace pepperoni_ratio_l20_20911

-- Definition of the problem's conditions
def total_pepperoni_slices : ℕ := 40
def slice_given_to_jelly_original : ℕ := 10
def slice_fallen_off : ℕ := 1

-- Our goal is to prove that the ratio is 3:10
theorem pepperoni_ratio (total_pepperoni_slices : ℕ) (slice_given_to_jelly_original : ℕ) (slice_fallen_off : ℕ) :
  (slice_given_to_jelly_original - slice_fallen_off) / (total_pepperoni_slices - slice_given_to_jelly_original) = 3 / 10 :=
by
  sorry

end pepperoni_ratio_l20_20911


namespace product_of_two_numbers_l20_20805

-- Define HCF function
def HCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM function
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

-- Define the conditions for the problem
def problem_conditions (x y : ℕ) : Prop :=
  HCF x y = 55 ∧ LCM x y = 1500

-- State the theorem that should be proven
theorem product_of_two_numbers (x y : ℕ) (h_conditions : problem_conditions x y) :
  x * y = 82500 :=
by
  sorry

end product_of_two_numbers_l20_20805


namespace probability_f4_positive_l20_20589

theorem probability_f4_positive {f : ℝ → ℝ} (h_odd : ∀ x, f (-x) = -f x)
  (h_fn : ∀ x < 0, f x = a + x + Real.logb 2 (-x)) (h_a : a > -4 ∧ a < 5) :
  (1/3 : ℝ) < (2/3 : ℝ) :=
sorry

end probability_f4_positive_l20_20589


namespace div_fraction_fraction_division_eq_l20_20673

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l20_20673


namespace max_value_of_f_l20_20188

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem max_value_of_f 
  (ω a : ℝ) 
  (h1 : 0 < ω) 
  (h2 : (2 * Real.pi / ω) = Real.pi) 
  (h3 : ∃ k : ℤ, ω * (Real.pi / 12) + (k : ℝ) * Real.pi + Real.pi / 3 = Real.pi / 2 + (k : ℝ) * Real.pi) :
  ∃ x : ℝ, f ω a x = 2 := by
  sorry

end max_value_of_f_l20_20188


namespace polyhedron_inequality_proof_l20_20472

noncomputable def polyhedron_inequality (B : ℕ) (P : ℕ) (T : ℕ) : Prop :=
  B * Real.sqrt (P + T) ≥ 2 * P

theorem polyhedron_inequality_proof (B P T : ℕ) 
  (h1 : 0 < B) (h2 : 0 < P) (h3 : 0 < T) 
  (condition_is_convex_polyhedron : true) : 
  polyhedron_inequality B P T :=
sorry

end polyhedron_inequality_proof_l20_20472


namespace darnell_phone_minutes_l20_20726

theorem darnell_phone_minutes
  (unlimited_cost : ℕ)
  (text_cost : ℕ)
  (call_cost : ℕ)
  (texts_per_dollar : ℕ)
  (minutes_per_dollar : ℕ)
  (total_texts : ℕ)
  (cost_difference : ℕ)
  (alternative_total_cost : ℕ)
  (M : ℕ)
  (text_cost_condition : unlimited_cost - cost_difference = alternative_total_cost)
  (text_formula : M / minutes_per_dollar * call_cost + total_texts / texts_per_dollar * text_cost = alternative_total_cost)
  : M = 60 :=
sorry

end darnell_phone_minutes_l20_20726


namespace arithmetic_seq_sum_mul_3_l20_20998

-- Definition of the arithmetic sequence
def arithmetic_sequence := [101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121]

-- Prove that 3 times the sum of the arithmetic sequence is 3663
theorem arithmetic_seq_sum_mul_3 : 
  3 * (arithmetic_sequence.sum) = 3663 :=
by
  sorry

end arithmetic_seq_sum_mul_3_l20_20998


namespace find_q_l20_20322

theorem find_q (q : ℤ) (h1 : lcm (lcm 12 16) (lcm 18 q) = 144) : q = 1 := sorry

end find_q_l20_20322


namespace calculate_expression_l20_20422

theorem calculate_expression :
  (-0.125)^2022 * 8^2023 = 8 :=
sorry

end calculate_expression_l20_20422


namespace new_fig_sides_l20_20789

def hexagon_side := 1
def triangle_side := 1
def hexagon_sides := 6
def triangle_sides := 3
def joined_sides := 2
def total_initial_sides := hexagon_sides + triangle_sides
def lost_sides := joined_sides * 2
def new_shape_sides := total_initial_sides - lost_sides

theorem new_fig_sides : new_shape_sides = 5 := by
  sorry

end new_fig_sides_l20_20789


namespace money_distribution_l20_20032

theorem money_distribution (Maggie_share : ℝ) (fraction_Maggie : ℝ) (total_sum : ℝ) :
  Maggie_share = 7500 →
  fraction_Maggie = (1/8) →
  total_sum = Maggie_share / fraction_Maggie →
  total_sum = 60000 :=
by 
  intros h1 h2 h3
  rw [h1, h2] at h3
  linarith

end money_distribution_l20_20032


namespace parametric_function_f_l20_20368

theorem parametric_function_f (f : ℚ → ℚ)
  (x y : ℝ) (t : ℚ) :
  y = 20 * t - 10 →
  y = (3 / 4 : ℝ) * x - 15 →
  x = f t →
  f t = (80 / 3) * t + 20 / 3 :=
by
  sorry

end parametric_function_f_l20_20368


namespace valid_5_digit_numbers_l20_20509

noncomputable def num_valid_numbers (d : ℕ) (h : d ≠ 7) (h_valid : d < 10) (h_pos : d ≠ 0) : ℕ :=
  let choices_first_place := 7   -- choices for the first digit (1-9, excluding d and 7)
  let choices_other_places := 8  -- choices for other digits (0-9, excluding d and 7)
  choices_first_place * choices_other_places ^ 4

theorem valid_5_digit_numbers (d : ℕ) (h_d_ne_7 : d ≠ 7) (h_d_valid : d < 10) (h_d_pos : d ≠ 0) :
  num_valid_numbers d h_d_ne_7 h_d_valid h_d_pos = 28672 := sorry

end valid_5_digit_numbers_l20_20509


namespace area_of_inscribed_triangle_l20_20151

noncomputable def area_inscribed_triangle (r : ℝ) (a b c : ℝ) : ℝ :=
  1 / 2 * r^2 * (Real.sin a + Real.sin b + Real.sin c)

theorem area_of_inscribed_triangle : 
  ∃ (r a b c : ℝ),
    a + b + c = 2 * π ∧
    r = 10 / π ∧
    a = 5 * (18 * π / 180) ∧
    b = 7 * (18 * π / 180) ∧
    c = 8 * (18 * π / 180) ∧
    area_inscribed_triangle r a b c = 119.84 / π^2 :=
begin
  sorry
end

end area_of_inscribed_triangle_l20_20151


namespace complement_A_union_B_l20_20602

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l20_20602


namespace reduce_consumption_percentage_l20_20687

theorem reduce_consumption_percentage :
  ∀ (current_rate old_rate : ℝ), 
  current_rate = 20 → 
  old_rate = 16 → 
  ((current_rate - old_rate) / old_rate * 100) = 25 :=
by
  intros current_rate old_rate h_current h_old
  sorry

end reduce_consumption_percentage_l20_20687


namespace total_oil_leakage_l20_20856

def oil_leaked_before : ℕ := 6522
def oil_leaked_during : ℕ := 5165
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leakage : total_oil_leaked = 11687 := by
  sorry

end total_oil_leakage_l20_20856


namespace expected_num_red_light_l20_20848

noncomputable def jiaRedLightExpectedValue : ℝ :=
  let n := 3
  let p := 2 / 5
  n * p

theorem expected_num_red_light
  (n : ℕ := 3)
  (p : ℝ := 2 / 5)
  (ξ : ℕ → ℝ := λ k, if k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 then real.to_ennreal 1 else 0)
  (hx : ξ 0 + ξ 1 + ξ 2 + ξ 3 = 1)
  (hx_eq : ξ 0 = real.to_ennreal 1 - 3 * p ∧
            ξ 1 = 3 * p * (1 - p)^2 ∧
            ξ 2 = 3 * p^2 * (1 - p) ∧
            ξ 3 = p^3) :
  (∑ k in Finset.range (3 + 1), k * ξ k) = 3 * (2/5) := sorry

end expected_num_red_light_l20_20848


namespace bat_wings_area_l20_20791

structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨0, 0⟩
def Q : Point := ⟨5, 0⟩
def R : Point := ⟨5, 2⟩
def S : Point := ⟨0, 2⟩
def A : Point := ⟨5, 1⟩
def T : Point := ⟨3, 2⟩

def area_triangle (P Q R : Point) : ℝ :=
  0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x))

theorem bat_wings_area :
  area_triangle P A T = 5.5 :=
sorry

end bat_wings_area_l20_20791


namespace votes_cast_l20_20685

theorem votes_cast (V : ℝ) (h1 : ∃ V, (0.65 * V) = (0.35 * V + 2340)) : V = 7800 :=
by
  sorry

end votes_cast_l20_20685


namespace smallest_base_converted_l20_20017

def convert_to_decimal_base_3 (n : ℕ) : ℕ :=
  1 * 3^3 + 0 * 3^2 + 0 * 3^1 + 2 * 3^0

def convert_to_decimal_base_6 (n : ℕ) : ℕ :=
  2 * 6^2 + 1 * 6^1 + 0 * 6^0

def convert_to_decimal_base_4 (n : ℕ) : ℕ :=
  1 * 4^3 + 0 * 4^2 + 0 * 4^1 + 0 * 4^0

def convert_to_decimal_base_2 (n : ℕ) : ℕ :=
  1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem smallest_base_converted :
  min (convert_to_decimal_base_3 1002) 
      (min (convert_to_decimal_base_6 210) 
           (min (convert_to_decimal_base_4 1000) 
                (convert_to_decimal_base_2 111111))) = convert_to_decimal_base_3 1002 :=
by sorry

end smallest_base_converted_l20_20017


namespace simplify_expression_l20_20650

variable (a b : Real)

theorem simplify_expression (a b : Real) : 
    3 * b * (3 * b ^ 2 + 2 * b) - b ^ 2 + 2 * a * (2 * a ^ 2 - 3 * a) - 4 * a * b = 
    9 * b ^ 3 + 5 * b ^ 2 + 4 * a ^ 3 - 6 * a ^ 2 - 4 * a * b := by
  sorry

end simplify_expression_l20_20650


namespace alexander_first_gallery_pictures_l20_20853

def pictures_for_new_galleries := 5 * 2
def pencils_for_new_galleries := pictures_for_new_galleries * 4
def total_exhibitions := 1 + 5
def pencils_for_signing := total_exhibitions * 2
def total_pencils := 88
def pencils_for_first_gallery := total_pencils - pencils_for_new_galleries - pencils_for_signing
def pictures_for_first_gallery := pencils_for_first_gallery / 4

theorem alexander_first_gallery_pictures : pictures_for_first_gallery = 9 :=
by
  sorry

end alexander_first_gallery_pictures_l20_20853


namespace inflation_over_two_years_real_interest_rate_l20_20826

-- Definitions for conditions
def annual_inflation_rate : ℝ := 0.025
def nominal_interest_rate : ℝ := 0.06

-- Lean statement for the first problem: Inflation over two years
theorem inflation_over_two_years :
  (1 + annual_inflation_rate) ^ 2 - 1 = 0.050625 := 
by sorry

-- Lean statement for the second problem: Real interest rate after inflation
theorem real_interest_rate (inflation_rate_two_years : ℝ)
  (h_inflation_rate : inflation_rate_two_years = 0.050625) :
  (nominal_interest_rate + 1) ^ 2 / (1 + inflation_rate_two_years) - 1 = 0.069459 :=
by sorry

end inflation_over_two_years_real_interest_rate_l20_20826


namespace quadratic_inequality_solution_l20_20189

variables {x : ℝ} {f : ℝ → ℝ}

def is_quadratic_and_opens_downwards (f : ℝ → ℝ) : Prop :=
  ∃ a b c, a < 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def is_symmetric_at_two (f : ℝ → ℝ) : Prop :=
  ∀ x, f (2 - x) = f (2 + x)

theorem quadratic_inequality_solution
  (h_quadratic : is_quadratic_and_opens_downwards f)
  (h_symmetric : is_symmetric_at_two f) :
  (1 - (Real.sqrt 14) / 4) < x ∧ x < (1 + (Real.sqrt 14) / 4) ↔
  f (Real.log ((1 / (1 / 4)) * (x^2 + x + 1 / 2))) <
  f (Real.log ((1 / (1 / 2)) * (2 * x^2 - x + 5 / 8))) :=
sorry

end quadratic_inequality_solution_l20_20189


namespace find_angle_C_find_side_c_l20_20770

variable {A B C a b c : ℝ}
variable {AD CD area_ABD : ℝ}

-- Conditions for question 1
variable (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))

-- Conditions for question 2
variable (h2 : AD = 4)
variable (h3 : CD = 4)
variable (h4 : area_ABD = 8 * Real.sqrt 3)
variable (h5 : C = Real.pi / 3)

-- Lean 4 statement for both parts of the problem
theorem find_angle_C (h1 : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A)) : 
  C = Real.pi / 3 :=
sorry

theorem find_side_c (h2 : AD = 4) (h3 : CD = 4) (h4 : area_ABD = 8 * Real.sqrt 3) (h5 : C = Real.pi / 3) : 
  c = 4 * Real.sqrt 7 :=
sorry

end find_angle_C_find_side_c_l20_20770


namespace cos_pi_div_3_l20_20736

theorem cos_pi_div_3 : Real.cos (π / 3) = 1 / 2 := 
by
  sorry

end cos_pi_div_3_l20_20736


namespace max_value_trig_expression_l20_20084

variable (a b φ θ : ℝ)

theorem max_value_trig_expression :
  ∃ θ : ℝ, a * Real.cos θ + b * Real.sin (θ + φ) ≤ Real.sqrt (a^2 + 2 * a * b * Real.sin φ + b^2) := sorry

end max_value_trig_expression_l20_20084


namespace find_function_l20_20659

/-- A function f satisfies the equation f(x) + (x + 1/2) * f(1 - x) = 1. -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f x + (x + 1 / 2) * f (1 - x) = 1

/-- We want to prove two things:
 1) f(0) = 2 and f(1) = -2
 2) f(x) =  2 / (1 - 2x) for x ≠ 1/2
 -/
theorem find_function (f : ℝ → ℝ) (h : satisfies_equation f) :
  (f 0 = 2 ∧ f 1 = -2) ∧ (∀ x : ℝ, x ≠ 1 / 2 → f x = 2 / (1 - 2 * x)) ∧ (f (1 / 2) = 1 / 2) :=
by
  sorry

end find_function_l20_20659


namespace sequence_value_a1_l20_20450

theorem sequence_value_a1 (a : ℕ → ℝ) 
  (h₁ : ∀ n, a (n + 1) = (1 / 2) * a n) 
  (h₂ : a 4 = 8) : a 1 = 64 :=
sorry

end sequence_value_a1_l20_20450


namespace lucas_should_give_fraction_l20_20224

-- Conditions as Lean definitions
variables (n : ℕ) -- Number of shells Noah has
def Noah_shells := n
def Emma_shells := 2 * n -- Emma has twice as many shells as Noah
def Lucas_shells := 8 * n -- Lucas has four times as many shells as Emma

-- Desired distribution
def Total_shells := Noah_shells n + Emma_shells n + Lucas_shells n
def Each_person_shells := Total_shells n / 3

-- Fraction calculation
def Shells_needed_by_Emma := Each_person_shells n - Emma_shells n
def Fraction_of_Lucas_shells_given_to_Emma := Shells_needed_by_Emma n / Lucas_shells n 

theorem lucas_should_give_fraction :
  Fraction_of_Lucas_shells_given_to_Emma n = 5 / 24 := 
by
  sorry

end lucas_should_give_fraction_l20_20224


namespace small_rectangular_prisms_intersect_diagonal_l20_20309

def lcm (a b c : Nat) : Nat :=
  Nat.lcm a (Nat.lcm b c)

def inclusion_exclusion (n : Nat) : Nat :=
  n / 2 + n / 3 + n / 5 - n / (2 * 3) - n / (3 * 5) - n / (5 * 2) + n / (2 * 3 * 5)

theorem small_rectangular_prisms_intersect_diagonal :
  ∀ (a b c : Nat) (L : Nat), a = 2 → b = 3 → c = 5 → L = 90 →
  lcm a b c = 30 → 3 * inclusion_exclusion (lcm a b c) = 66 :=
by
  intros
  sorry

end small_rectangular_prisms_intersect_diagonal_l20_20309


namespace domain_of_ln_function_l20_20866

theorem domain_of_ln_function (x : ℝ) : 3 - 4 * x > 0 ↔ x < 3 / 4 := 
by
  sorry

end domain_of_ln_function_l20_20866


namespace relationship_y1_y2_y3_l20_20583

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l20_20583


namespace integral_abs_sin_from_0_to_2pi_l20_20863

theorem integral_abs_sin_from_0_to_2pi : ∫ x in (0 : ℝ)..(2 * Real.pi), |Real.sin x| = 4 := 
by
  sorry

end integral_abs_sin_from_0_to_2pi_l20_20863


namespace distinct_c_values_l20_20087

theorem distinct_c_values (c r s t : ℂ) 
  (h_distinct : r ≠ s ∧ s ≠ t ∧ r ≠ t)
  (h_unity : ∃ ω : ℂ, ω^3 = 1 ∧ r = 1 ∧ s = ω ∧ t = ω^2)
  (h_eq : ∀ z : ℂ, (z - r) * (z - s) * (z - t) = (z - c * r) * (z - c * s) * (z - c * t)) :
  ∃ (c_vals : Finset ℂ), c_vals.card = 3 ∧ ∀ (c' : ℂ), c' ∈ c_vals → c'^3 = 1 :=
by
  sorry

end distinct_c_values_l20_20087


namespace positive_integer_solutions_l20_20737

theorem positive_integer_solutions (n x y z t : ℕ) (h_n : n > 0) (h_n_neq_1 : n ≠ 1) (h_x : x > 0) (h_y : y > 0) (h_z : z > 0) (h_t : t > 0) :
  (n ^ x ∣ n ^ y + n ^ z ↔ n ^ x = n ^ t) →
  ((n = 2 ∧ y = x ∧ z = x + 1 ∧ t = x + 2) ∨ (n = 3 ∧ y = x ∧ z = x ∧ t = x + 1)) :=
by
  sorry

end positive_integer_solutions_l20_20737


namespace part1_part2_l20_20009

namespace ClothingFactory

variables {x y m : ℝ} -- defining variables

-- The conditions
def condition1 : Prop := x + 2 * y = 5
def condition2 : Prop := 3 * x + y = 7
def condition3 : Prop := 1.8 * (100 - m) + 1.6 * m ≤ 168

-- Theorems to Prove
theorem part1 (h1 : x + 2 * y = 5) (h2 : 3 * x + y = 7) : 
  x = 1.8 ∧ y = 1.6 := 
sorry

theorem part2 (h1 : x = 1.8) (h2 : y = 1.6) (h3 : 1.8 * (100 - m) + 1.6 * m ≤ 168) : 
  m ≥ 60 := 
sorry

end ClothingFactory

end part1_part2_l20_20009


namespace frac_square_between_half_and_one_l20_20665

theorem frac_square_between_half_and_one :
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  (1 / 2) < expr ∧ expr < 1 :=
by
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  have h1 : (1 / 2) < expr := sorry
  have h2 : expr < 1 := sorry
  exact ⟨h1, h2⟩

end frac_square_between_half_and_one_l20_20665


namespace find_b_when_a_equals_neg10_l20_20804

theorem find_b_when_a_equals_neg10 
  (ab_k : ∀ a b : ℝ, (a * b) = 675) 
  (sum_60 : ∀ a b : ℝ, (a + b = 60 → a = 3 * b)) 
  (a_eq_neg10 : ∀ a : ℝ, a = -10) : 
  ∃ b : ℝ, b = -67.5 := 
by 
  sorry

end find_b_when_a_equals_neg10_l20_20804


namespace trig_identity_l20_20875

open Real

theorem trig_identity (α : ℝ) (h : tan α = -1/2) : 1 - sin (2 * α) = 9/5 := 
  sorry

end trig_identity_l20_20875


namespace certain_number_is_213_l20_20057

theorem certain_number_is_213 (n : ℕ) (h : n * 16 = 3408) : n = 213 :=
sorry

end certain_number_is_213_l20_20057


namespace pokemon_cards_per_friend_l20_20234

theorem pokemon_cards_per_friend (total_cards : ℕ) (num_friends : ℕ) 
  (hc : total_cards = 56) (hf : num_friends = 4) : (total_cards / num_friends) = 14 := 
by
  sorry

end pokemon_cards_per_friend_l20_20234


namespace fixed_monthly_fee_l20_20430

-- Define the problem parameters and assumptions
variables (x y : ℝ)
axiom february_bill : x + y = 20.72
axiom march_bill : x + 3 * y = 35.28

-- State the Lean theorem that we want to prove
theorem fixed_monthly_fee : x = 13.44 :=
by
  sorry

end fixed_monthly_fee_l20_20430


namespace largest_possible_b_l20_20085

theorem largest_possible_b (b : ℝ) (h : (3 * b + 6) * (b - 2) = 9 * b) : b ≤ 4 := 
by {
  -- leaving the proof as an exercise, using 'sorry' to complete the statement
  sorry
}

end largest_possible_b_l20_20085


namespace smallest_integer_five_consecutive_sum_2025_l20_20947

theorem smallest_integer_five_consecutive_sum_2025 :
  ∃ n : ℤ, 5 * n + 10 = 2025 ∧ n = 403 :=
by
  sorry

end smallest_integer_five_consecutive_sum_2025_l20_20947


namespace parabola_chord_length_l20_20449

theorem parabola_chord_length (x₁ x₂ : ℝ) (y₁ y₂ : ℝ) 
(h1 : y₁^2 = 4 * x₁) 
(h2 : y₂^2 = 4 * x₂) 
(h3 : x₁ + x₂ = 6) : 
|y₁ - y₂| = 8 :=
sorry

end parabola_chord_length_l20_20449


namespace chemical_transformations_correct_l20_20525

def ethylbenzene : String := "C6H5CH2CH3"
def brominate (A : String) : String := "C6H5CH(Br)CH3"
def hydrolyze (B : String) : String := "C6H5CH(OH)CH3"
def dehydrate (C : String) : String := "C6H5CH=CH2"
def oxidize (D : String) : String := "C6H5COOH"
def brominate_with_catalyst (E : String) : String := "m-C6H4(Br)COOH"

def sequence_of_transformations : Prop :=
  ethylbenzene = "C6H5CH2CH3" ∧
  brominate ethylbenzene = "C6H5CH(Br)CH3" ∧
  hydrolyze (brominate ethylbenzene) = "C6H5CH(OH)CH3" ∧
  dehydrate (hydrolyze (brominate ethylbenzene)) = "C6H5CH=CH2" ∧
  oxidize (dehydrate (hydrolyze (brominate ethylbenzene))) = "C6H5COOH" ∧
  brominate_with_catalyst (oxidize (dehydrate (hydrolyze (brominate ethylbenzene)))) = "m-C6H4(Br)COOH"

theorem chemical_transformations_correct : sequence_of_transformations :=
by
  -- proof would go here
  sorry

end chemical_transformations_correct_l20_20525


namespace john_spent_30_l20_20527

/-- At a supermarket, John spent 1/5 of his money on fresh fruits and vegetables, 1/3 on meat products, and 1/10 on bakery products. If he spent the remaining $11 on candy, how much did John spend at the supermarket? -/
theorem john_spent_30 (X : ℝ) (h1 : X * (1/5) + X * (1/3) + X * (1/10) + 11 = X) : X = 30 := 
by 
  sorry

end john_spent_30_l20_20527


namespace expand_expression_l20_20036

variable (x y : ℝ)

theorem expand_expression (x y : ℝ) : 12 * (3 * x + 4 - 2 * y) = 36 * x + 48 - 24 * y :=
by
  sorry

end expand_expression_l20_20036


namespace initial_carrots_count_l20_20644

theorem initial_carrots_count (x : ℕ) (h1 : x - 2 + 21 = 31) : x = 12 := by
  sorry

end initial_carrots_count_l20_20644


namespace casper_initial_candies_l20_20353

theorem casper_initial_candies 
  (x : ℚ)
  (h1 : ∃ y : ℚ, y = x - (1/4) * x - 3) 
  (h2 : ∃ z : ℚ, z = y - (1/5) * y - 5) 
  (h3 : z - 10 = 10) : x = 224 / 3 :=
by
  sorry

end casper_initial_candies_l20_20353


namespace range_of_m_l20_20485

variable (x m : ℝ)

def alpha (x : ℝ) : Prop := x ≤ -5
def beta (x m : ℝ) : Prop := 2 * m - 3 ≤ x ∧ x ≤ 2 * m + 1

theorem range_of_m (x : ℝ) : (∀ x, beta x m → alpha x) → m ≤ -3 := by
  sorry

end range_of_m_l20_20485


namespace sales_value_minimum_l20_20119

theorem sales_value_minimum (V : ℝ) (base_salary new_salary : ℝ) (commission_rate sales_needed old_salary : ℝ)
    (h_base_salary : base_salary = 45000 )
    (h_new_salary : new_salary = base_salary + 0.15 * V * sales_needed)
    (h_sales_needed : sales_needed = 266.67)
    (h_old_salary : old_salary = 75000) :
    new_salary ≥ old_salary ↔ V ≥ 750 := 
by
  sorry

end sales_value_minimum_l20_20119


namespace perimeter_is_11_or_13_l20_20326

-- Define the basic length properties of the isosceles triangle
structure IsoscelesTriangle where
  a b c : ℝ
  a_eq_b_or_a_eq_c : a = b ∨ a = c

-- Conditions of the problem
def sides : IsoscelesTriangle where
  a := 3
  b := 5
  c := 5
  a_eq_b_or_a_eq_c := Or.inr rfl

def sides' : IsoscelesTriangle where
  a := 3
  b := 3
  c := 5
  a_eq_b_or_a_eq_c := Or.inl rfl

-- Prove the perimeters
theorem perimeter_is_11_or_13 : (sides.a + sides.b + sides.c = 13) ∨ (sides'.a + sides'.b + sides'.c = 11) :=
by
  sorry

end perimeter_is_11_or_13_l20_20326


namespace mountain_height_is_1700m_l20_20373

noncomputable def height_of_mountain (temp_base : ℝ) (temp_summit : ℝ) (rate_decrease : ℝ) : ℝ :=
  ((temp_base - temp_summit) / rate_decrease) * 100

theorem mountain_height_is_1700m :
  height_of_mountain 26 14.1 0.7 = 1700 :=
by
  sorry

end mountain_height_is_1700m_l20_20373


namespace number_of_blue_candles_l20_20165

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l20_20165


namespace shoes_no_bad_pairings_l20_20292

theorem shoes_no_bad_pairings :
  ∃ m n : ℕ, Nat.Coprime m n ∧ m + n = 399 ∧ 
  (∃! (pairs : Finset (Fin 8 × Fin 8)), 
    ∀ k < 4, ∀ s : Finset (Fin 8 × Fin 8), s.card = k → 
        ¬ ∀ i ∈ s, (↑(i.1) = ↑(i.2))) →
  ∑ p in pairs, 1 = 7728 := sorry

end shoes_no_bad_pairings_l20_20292


namespace red_pigment_weight_in_brown_paint_l20_20534

theorem red_pigment_weight_in_brown_paint :
  ∀ (M G : ℝ), 
    (M + G = 10) → 
    (0.5 * M + 0.3 * G = 4) →
    0.5 * M = 2.5 :=
by sorry

end red_pigment_weight_in_brown_paint_l20_20534


namespace range_of_a_l20_20065

open Real

noncomputable def C1 (t a : ℝ) : ℝ × ℝ := (2 * t + 2 * a, -t)
noncomputable def C2 (θ : ℝ) : ℝ × ℝ := (2 * cos θ, 2 + 2 * sin θ)

theorem range_of_a {a : ℝ} :
  (∃ (t θ : ℝ), C1 t a = C2 θ) ↔ 2 - sqrt 5 ≤ a ∧ a ≤ 2 + sqrt 5 :=
by 
  sorry

end range_of_a_l20_20065


namespace domain_of_sqrt_fraction_l20_20738

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (x - 2 ≥ 0 ∧ 5 - x > 0) ↔ (2 ≤ x ∧ x < 5) :=
by
  sorry

end domain_of_sqrt_fraction_l20_20738


namespace discount_percentage_correct_l20_20773

-- Definitions corresponding to the conditions
def number_of_toys : ℕ := 5
def cost_per_toy : ℕ := 3
def total_price_paid : ℕ := 12
def original_price : ℕ := number_of_toys * cost_per_toy
def discount_amount : ℕ := original_price - total_price_paid
def discount_percentage : ℕ := (discount_amount * 100) / original_price

-- Statement of the problem
theorem discount_percentage_correct :
  discount_percentage = 20 := 
  sorry

end discount_percentage_correct_l20_20773


namespace necessary_not_sufficient_condition_l20_20592

theorem necessary_not_sufficient_condition (x : ℝ) : (x < 2) → (x^2 - x - 2 < 0) :=
by {
  sorry
}

end necessary_not_sufficient_condition_l20_20592


namespace complement_A_union_B_l20_20604

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l20_20604


namespace triangle_area_l20_20864

theorem triangle_area {a b m : ℝ} (h1 : a = 27) (h2 : b = 29) (h3 : m = 26) : 
  ∃ (area : ℝ), area = 270 :=
by
  sorry

end triangle_area_l20_20864


namespace distribute_marbles_correct_l20_20459

def distribute_marbles (total_marbles : Nat) (num_boys : Nat) : Nat :=
  total_marbles / num_boys

theorem distribute_marbles_correct :
  distribute_marbles 20 2 = 10 := 
by 
  sorry

end distribute_marbles_correct_l20_20459


namespace max_profit_achieved_l20_20643

theorem max_profit_achieved :
  ∃ x : ℤ, 
    (x = 21) ∧ 
    (21 + 14 = 35) ∧ 
    (30 - 21 = 9) ∧ 
    (21 - 5 = 16) ∧
    (-x + 1965 = 1944) :=
by
  sorry

end max_profit_achieved_l20_20643


namespace find_b_value_l20_20183

theorem find_b_value
    (k1 k2 b : ℝ)
    (y1 y2 : ℝ → ℝ)
    (a n : ℝ)
    (h1 : ∀ x, y1 x = k1 / x)
    (h2 : ∀ x, y2 x = k2 * x + b)
    (intersection_A : y1 1 = 4)
    (intersection_B : y2 a = 1 ∧ y1 a = 1)
    (translated_C_y1 : y1 (-1) = n + 6)
    (translated_C_y2 : y2 1 = n)
    (k1k2_nonzero : k1 ≠ 0 ∧ k2 ≠ 0)
    (sum_k1_k2 : k1 + k2 = 0) :
  b = -6 :=
sorry

end find_b_value_l20_20183


namespace white_square_area_l20_20487

theorem white_square_area
  (edge_length : ℝ)
  (total_green_area : ℝ)
  (faces : ℕ)
  (green_per_face : ℝ)
  (total_surface_area : ℝ)
  (white_area_per_face : ℝ) :
  edge_length = 12 ∧ total_green_area = 432 ∧ faces = 6 ∧ total_surface_area = 864 ∧ green_per_face = total_green_area / faces ∧ white_area_per_face = total_surface_area / faces - green_per_face → white_area_per_face = 72 :=
by
  sorry

end white_square_area_l20_20487


namespace total_number_of_workers_l20_20108

theorem total_number_of_workers 
    (W N : ℕ) 
    (h1 : 8000 * W = 12000 * 8 + 6000 * N) 
    (h2 : W = 8 + N) : 
    W = 24 :=
by
  sorry

end total_number_of_workers_l20_20108


namespace product_multiple_of_3_probability_l20_20218

theorem product_multiple_of_3_probability :
  let P_Juan_rolls_3_or_6 := 1 / 4
  let P_Juan_does_not_roll_3_or_6 := 3 / 4
  let P_Amal_rolls_3_or_6 := 1 / 3
  let P_Scenario_1 := P_Juan_rolls_3_or_6
  let P_Scenario_2 := P_Juan_does_not_roll_3_or_6 * P_Amal_rolls_3_or_6
  (P_Scenario_1 + P_Scenario_2 = 1 / 2) := sorry

end product_multiple_of_3_probability_l20_20218


namespace fractional_eq_no_solution_l20_20454

theorem fractional_eq_no_solution (m : ℝ) :
  ¬ ∃ x, (x - 2) / (x + 2) - (m * x) / (x^2 - 4) = 1 ↔ m = -4 :=
by
  sorry

end fractional_eq_no_solution_l20_20454


namespace polynomial_real_root_condition_l20_20861

theorem polynomial_real_root_condition (b : ℝ) :
    (∃ x : ℝ, x^4 + b * x^3 + x^2 + b * x - 1 = 0) ↔ (b ≥ 1 / 2) :=
by sorry

end polynomial_real_root_condition_l20_20861


namespace partition_value_l20_20837

variable {a m n p x k l : ℝ}

theorem partition_value :
  (m * (a - n * x) = k * (a - n * x)) ∧
  (n * x = l * x) ∧
  (a - x = p * (a - m * (a - n * x)))
  → x = (a * (m * p - p + 1)) / (n * m * p + 1) :=
by
  sorry

end partition_value_l20_20837


namespace absolute_difference_55_l20_20180

def tau (n: ℕ) : ℕ := DivisorFinset.card (Finset.filter (λ k, k ∣ n) (Finset.range (n + 1)))

noncomputable def S (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), tau k

def oddIntegersCount (n : ℕ) : ℕ := (Finset.range n).filter (λ k, S k % 2 = 1).card

def evenIntegersCount (n : ℕ) : ℕ := (Finset.range n).filter (λ k, S k % 2 = 0).card

theorem absolute_difference_55 : |oddIntegersCount 3000 - evenIntegersCount 3000| = 55 :=
sorry

end absolute_difference_55_l20_20180


namespace pencil_notebook_cost_l20_20847

theorem pencil_notebook_cost (p n : ℝ)
  (h1 : 9 * p + 10 * n = 5.35)
  (h2 : 6 * p + 4 * n = 2.50) :
  24 * 0.9 * p + 15 * n = 9.24 :=
by 
  sorry

end pencil_notebook_cost_l20_20847


namespace find_sachins_age_l20_20404

variable (S R : ℕ)

theorem find_sachins_age (h1 : R = S + 8) (h2 : S * 9 = R * 7) : S = 28 := by
  sorry

end find_sachins_age_l20_20404


namespace bald_eagle_dive_time_l20_20240

-- Definitions as per the conditions in the problem
def speed_bald_eagle : ℝ := 100
def speed_peregrine_falcon : ℝ := 2 * speed_bald_eagle
def time_peregrine_falcon : ℝ := 15

-- The theorem to prove
theorem bald_eagle_dive_time : (speed_bald_eagle * 30) = (speed_peregrine_falcon * time_peregrine_falcon) := by
  sorry

end bald_eagle_dive_time_l20_20240


namespace extreme_values_f_range_of_a_l20_20599

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x - a
noncomputable def df (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem extreme_values_f (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), df x₁ = 0 ∧ df x₂ = 0 ∧ f x₁ a = (5 / 27) - a ∧ f x₂ a = -1 - a :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ (a : ℝ), f (-1/3) a < 0 ∧ f 1 a > 0) ↔ (a < -1 ∨ a > 5 / 27) :=
sorry

end extreme_values_f_range_of_a_l20_20599


namespace shaded_to_white_ratio_l20_20818

theorem shaded_to_white_ratio (shaded_area : ℕ) (white_area : ℕ) (h_shaded : shaded_area = 5) (h_white : white_area = 3) : shaded_area / white_area = 5 / 3 := 
by
  rw [h_shaded, h_white]
  norm_num

end shaded_to_white_ratio_l20_20818


namespace average_score_of_male_students_standard_deviation_of_all_students_l20_20849

def students : ℕ := 5
def total_average_score : ℝ := 80
def male_student_variance : ℝ := 150
def female_student1_score : ℝ := 85
def female_student2_score : ℝ := 75
def male_student_average_score : ℝ := 80 -- From solution step (1)
def total_standard_deviation : ℝ := 10 -- From solution step (2)

theorem average_score_of_male_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  male_student_average_score = 80 :=
by sorry

theorem standard_deviation_of_all_students :
  (students = 5) →
  (total_average_score = 80) →
  (male_student_variance = 150) →
  (female_student1_score = 85) →
  (female_student2_score = 75) →
  total_standard_deviation = 10 :=
by sorry

end average_score_of_male_students_standard_deviation_of_all_students_l20_20849


namespace total_apples_l20_20205

variable (A : ℕ)
variables (too_small not_ripe perfect : ℕ)

-- Conditions
axiom small_fraction : too_small = A / 6
axiom ripe_fraction  : not_ripe = A / 3
axiom remaining_fraction : perfect = A / 2
axiom perfect_count : perfect = 15

theorem total_apples : A = 30 :=
sorry

end total_apples_l20_20205


namespace find_area_of_triangle_ABQ_l20_20333

noncomputable def area_triangle_ABQ {A B C P Q R : Type*}
  (AP PB : ℝ) (area_ABC area_ABQ : ℝ) (h_areas_equal : area_ABQ = 15 / 2)
  (h_triangle_area : area_ABC = 15) (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) : Prop := area_ABQ = 15

theorem find_area_of_triangle_ABQ
  (A B C P Q R : Type*) (AP PB : ℝ)
  (h_triangle_area : area_ABC = 15)
  (h_AP : AP = 3) (h_PB : PB = 2)
  (h_AB : AB = AP + PB) (h_areas_equal : area_ABQ = 15 / 2) :
  area_ABQ = 15 := sorry

end find_area_of_triangle_ABQ_l20_20333


namespace calculate_blue_candles_l20_20167

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l20_20167


namespace average_is_correct_l20_20392

def nums : List ℝ := [13, 14, 510, 520, 530, 1115, 1120, 1, 1252140, 2345]

theorem average_is_correct :
  (nums.sum / nums.length) = 125830.8 :=
by sorry

end average_is_correct_l20_20392


namespace general_solution_of_differential_equation_l20_20177

theorem general_solution_of_differential_equation (a₀ : ℝ) (x : ℝ) :
  ∃ y : ℝ → ℝ, (∀ x, deriv y x = (y x)^2) ∧ y x = a₀ / (1 - a₀ * x) :=
sorry

end general_solution_of_differential_equation_l20_20177


namespace steps_climbed_l20_20774

-- Definitions
def flights : ℕ := 9
def feet_per_flight : ℕ := 10
def inches_per_step : ℕ := 18

-- Proving the number of steps John climbs up
theorem steps_climbed : 
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  steps = 60 := 
by
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  sorry

end steps_climbed_l20_20774


namespace count_squares_below_graph_l20_20244

theorem count_squares_below_graph (x y: ℕ) (h : 5 * x + 195 * y = 975) :
  ∃ n : ℕ, n = 388 ∧ 
  ∀ a b : ℕ, 0 ≤ a ∧ a ≤ 195 ∧ 0 ≤ b ∧ b ≤ 5 →
    1 * a + 1 * b < 195 * 5 →
    n = 388 := 
sorry

end count_squares_below_graph_l20_20244


namespace remainder_of_x_mod_10_l20_20833

def x : ℕ := 2007 ^ 2008

theorem remainder_of_x_mod_10 : x % 10 = 1 := by
  sorry

end remainder_of_x_mod_10_l20_20833


namespace pies_sold_l20_20862

-- Define the conditions in Lean
def num_cakes : ℕ := 453
def price_per_cake : ℕ := 12
def total_earnings : ℕ := 6318
def price_per_pie : ℕ := 7

-- Define the problem
theorem pies_sold (P : ℕ) (h1 : num_cakes * price_per_cake + P * price_per_pie = total_earnings) : P = 126 := 
by 
  sorry

end pies_sold_l20_20862


namespace true_proposition_is_D_l20_20157

open Real

theorem true_proposition_is_D :
  (∃ x_0 : ℝ, exp x_0 ≤ 0) = False ∧
  (∀ x : ℝ, 2 ^ x > x ^ 2) = False ∧
  (∀ a b : ℝ, a + b = 0 ↔ a / b = -1) = False ∧
  (∀ a b : ℝ, a > 1 ∧ b > 1 → a * b > 1) = True :=
by
    sorry

end true_proposition_is_D_l20_20157


namespace determine_a_l20_20619

theorem determine_a (a x : ℝ) (h : x = 1) (h_eq : a * x + 2 * x = 3) : a = 1 :=
by
  subst h
  simp at h_eq
  linarith

end determine_a_l20_20619


namespace max_median_of_pos_integers_l20_20237

theorem max_median_of_pos_integers
  (k m p r s t u : ℕ)
  (h_avg : (k + m + p + r + s + t + u) / 7 = 24)
  (h_order : k < m ∧ m < p ∧ p < r ∧ r < s ∧ s < t ∧ t < u)
  (h_t : t = 54)
  (h_km_sum : k + m ≤ 20)
  : r ≤ 53 :=
sorry

end max_median_of_pos_integers_l20_20237


namespace multiply_polynomials_l20_20096

variable {x y : ℝ}

theorem multiply_polynomials (x y : ℝ) :
  (3 * x ^ 4 - 2 * y ^ 3) * (9 * x ^ 8 + 6 * x ^ 4 * y ^ 3 + 4 * y ^ 6) = 27 * x ^ 12 - 8 * y ^ 9 :=
by
  sorry

end multiply_polynomials_l20_20096


namespace find_coeff_and_root_range_l20_20051

def f (x : ℝ) (a b : ℝ) : ℝ := a * x^3 - b * x + 4

theorem find_coeff_and_root_range (a b : ℝ)
  (h1 : f 2 a b = - (4/3))
  (h2 : deriv (λ x => f x a b) 2 = 0) :
  a = 1 / 3 ∧ b = 4 ∧ 
  (∀ k : ℝ, (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 (1/3) 4 = k ∧ f x2 (1/3) 4 = k ∧ f x3 (1/3) 4 = k) ↔ - (4/3) < k ∧ k < 28/3) :=
sorry

end find_coeff_and_root_range_l20_20051


namespace alex_friends_invite_l20_20542

theorem alex_friends_invite (burger_buns_per_pack : ℕ)
                            (packs_of_buns : ℕ)
                            (buns_needed_by_each_guest : ℕ)
                            (total_buns : ℕ)
                            (friends_who_dont_eat_buns : ℕ)
                            (friends_who_dont_eat_meat : ℕ)
                            (total_friends_invited : ℕ) 
                            (h1 : burger_buns_per_pack = 8)
                            (h2 : packs_of_buns = 3)
                            (h3 : buns_needed_by_each_guest = 3)
                            (h4 : total_buns = packs_of_buns * burger_buns_per_pack)
                            (h5 : friends_who_dont_eat_buns = 1)
                            (h6 : friends_who_dont_eat_meat = 1)
                            (h7 : total_friends_invited = (total_buns / buns_needed_by_each_guest) + friends_who_dont_eat_buns) :
  total_friends_invited = 9 :=
by sorry

end alex_friends_invite_l20_20542


namespace solve_linear_system_l20_20457

theorem solve_linear_system (m x y : ℝ) 
  (h1 : x + y = 3 * m) 
  (h2 : x - y = 5 * m)
  (h3 : 2 * x + 3 * y = 10) : 
  m = 2 := 
by 
  sorry

end solve_linear_system_l20_20457


namespace possible_values_of_cubic_sum_l20_20912

theorem possible_values_of_cubic_sum (x y z : ℂ) (h1 : (Matrix.of ![
    ![x, y, z],
    ![y, z, x],
    ![z, x, y]
  ] ^ 2 = 3 • (1 : Matrix (Fin 3) (Fin 3) ℂ))) (h2 : x * y * z = -1) :
  x^3 + y^3 + z^3 = -3 + 3 * Real.sqrt 3 ∨ x^3 + y^3 + z^3 = -3 - 3 * Real.sqrt 3 := by
  sorry

end possible_values_of_cubic_sum_l20_20912


namespace solution_correctness_l20_20043

theorem solution_correctness:
  ∀ (x1 : ℝ) (θ : ℝ), (θ = (5 * Real.pi / 13)) →
  (0 ≤ x1 ∧ x1 ≤ Real.pi / 2) →
  ∃ (x2 : ℝ), (0 ≤ x2 ∧ x2 ≤ Real.pi / 2) ∧ 
  (Real.sin x1 - 2 * Real.sin (x2 + θ) = -1) :=
by 
  intros x1 θ hθ hx1;
  sorry

end solution_correctness_l20_20043


namespace averageFishIs75_l20_20491

-- Introduce the number of fish in Boast Pool
def BoastPool : ℕ := 75

-- Introduce the number of fish in Onum Lake
def OnumLake : ℕ := BoastPool + 25

-- Introduce the number of fish in Riddle Pond
def RiddlePond : ℕ := OnumLake / 2

-- Define the total number of fish in all three bodies of water
def totalFish : ℕ := BoastPool + OnumLake + RiddlePond

-- Define the average number of fish in all three bodies of water
def averageFish : ℕ := totalFish / 3

-- Prove that the average number of fish is 75
theorem averageFishIs75 : averageFish = 75 :=
by
  -- We need to provide the proof steps here but using sorry to skip
  sorry

end averageFishIs75_l20_20491


namespace task_completion_time_l20_20995

theorem task_completion_time :
  let rate_A := 1 / 10
  let rate_B := 1 / 15
  let rate_C := 1 / 15
  let combined_rate := rate_A + rate_B + rate_C
  let working_days_A := 2
  let working_days_B := 1
  let rest_day_A := 1
  let rest_days_B := 2
  let work_done_A := rate_A * working_days_A
  let work_done_B := rate_B * working_days_B
  let work_done_C := rate_C * (working_days_A + rest_day_A)
  let work_done := work_done_A + work_done_B + work_done_C
  let remaining_work := 1 - work_done
  let total_days := (work_done / combined_rate) + rest_day_A + rest_days_B
  total_days = 4 + 1 / 7 := by sorry

end task_completion_time_l20_20995


namespace Z_is_divisible_by_10001_l20_20339

theorem Z_is_divisible_by_10001
    (Z : ℕ) (a b c d : ℕ) (ha : a ≠ 0)
    (hZ : Z = 1000 * 10001 * a + 100 * 10001 * b + 10 * 10001 * c + 10001 * d)
    : 10001 ∣ Z :=
by {
    -- Proof omitted
    sorry
}

end Z_is_divisible_by_10001_l20_20339


namespace matrix_pow_expression_l20_20914

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_pow_expression :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by
  sorry

end matrix_pow_expression_l20_20914


namespace zero_in_A_l20_20193

def A : Set ℝ := {x | x * (x + 1) = 0}

theorem zero_in_A : 0 ∈ A := by
  sorry

end zero_in_A_l20_20193


namespace major_axis_double_minor_axis_l20_20842

-- Define the radius of the right circular cylinder.
def cylinder_radius := 2

-- Define the minor axis length based on the cylinder's radius.
def minor_axis_length := 2 * cylinder_radius

-- Define the major axis length as double the minor axis length.
def major_axis_length := 2 * minor_axis_length

-- State the theorem to prove the major axis length.
theorem major_axis_double_minor_axis : major_axis_length = 8 := by
  sorry

end major_axis_double_minor_axis_l20_20842


namespace square_perimeter_l20_20844

theorem square_perimeter (s : ℝ) (h1 : ∀ r : ℝ, r = 2 * (s + s / 4)) (r_perimeter_eq_40 : r = 40) :
  4 * s = 64 := by
  sorry

end square_perimeter_l20_20844


namespace stock_status_after_limit_moves_l20_20713

theorem stock_status_after_limit_moves (initial_value : ℝ) (h₁ : initial_value = 1)
  (limit_up_factor : ℝ) (h₂ : limit_up_factor = 1 + 0.10)
  (limit_down_factor : ℝ) (h₃ : limit_down_factor = 1 - 0.10) :
  (limit_up_factor^5 * limit_down_factor^5) < initial_value :=
by
  sorry

end stock_status_after_limit_moves_l20_20713


namespace quadratic_inequality_solution_l20_20893

theorem quadratic_inequality_solution
  (x : ℝ)
  (h : x^2 - 5 * x + 6 < 0) :
  2 < x ∧ x < 3 :=
by sorry

end quadratic_inequality_solution_l20_20893


namespace negation_of_p_is_neg_p_l20_20585

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- Define the negation of p
def neg_p : Prop := ∀ m : ℝ, m > 0 → ¬ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- The theorem statement
theorem negation_of_p_is_neg_p : (¬ p) = neg_p := 
by
  sorry

end negation_of_p_is_neg_p_l20_20585


namespace div_fraction_fraction_division_eq_l20_20674

theorem div_fraction (a b : ℕ) (h : b ≠ 0) : (a : ℚ) / b = (a : ℚ) * (1 / (b : ℚ)) := 
by sorry

theorem fraction_division_eq : (3 : ℚ) / 7 / 4 = 3 / 28 := 
by 
  calc
    (3 : ℚ) / 7 / 4 = (3 / 7) * (1 / 4) : by rw [div_fraction] 
                ... = 3 / 28            : by normalization_tactic -- Use appropriate tactic for simplification
                ... = 3 / 28            : by rfl

end div_fraction_fraction_division_eq_l20_20674


namespace solve_quadratic_eqn_l20_20946

theorem solve_quadratic_eqn (x : ℝ) : 3 * x ^ 2 = 27 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end solve_quadratic_eqn_l20_20946


namespace order_of_magnitude_l20_20663

noncomputable def a : Real := 70.3
noncomputable def b : Real := 70.2
noncomputable def c : Real := Real.log 0.3

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l20_20663


namespace karl_savings_proof_l20_20337

-- Definitions based on the conditions
def original_price_per_notebook : ℝ := 3.00
def sale_discount : ℝ := 0.25
def extra_discount_threshold : ℝ := 10
def extra_discount_rate : ℝ := 0.05

-- The number of notebooks Karl could have purchased instead
def notebooks_purchased : ℝ := 12

-- The total savings calculation
noncomputable def total_savings : ℝ := 
  let original_total := notebooks_purchased * original_price_per_notebook
  let discounted_price_per_notebook := original_price_per_notebook * (1 - sale_discount)
  let extra_discount := if notebooks_purchased > extra_discount_threshold then discounted_price_per_notebook * extra_discount_rate else 0
  let total_price_after_discounts := notebooks_purchased * discounted_price_per_notebook - notebooks_purchased * extra_discount
  original_total - total_price_after_discounts

-- Formal statement to prove
theorem karl_savings_proof : total_savings = 10.35 := 
  sorry

end karl_savings_proof_l20_20337


namespace rational_pair_exists_l20_20225

theorem rational_pair_exists (a b : ℚ) (h1 : a = 3/2) (h2 : b = 3) : a ≠ b ∧ a + b = a * b :=
by {
  sorry
}

end rational_pair_exists_l20_20225


namespace RiverJoe_popcorn_shrimp_price_l20_20496

theorem RiverJoe_popcorn_shrimp_price
  (price_catfish : ℝ)
  (total_orders : ℕ)
  (total_revenue : ℝ)
  (orders_popcorn_shrimp : ℕ)
  (catfish_revenue : ℝ)
  (popcorn_shrimp_price : ℝ) :
  price_catfish = 6.00 →
  total_orders = 26 →
  total_revenue = 133.50 →
  orders_popcorn_shrimp = 9 →
  catfish_revenue = (total_orders - orders_popcorn_shrimp) * price_catfish →
  catfish_revenue + orders_popcorn_shrimp * popcorn_shrimp_price = total_revenue →
  popcorn_shrimp_price = 3.50 :=
by
  intros price_catfish_eq total_orders_eq total_revenue_eq orders_popcorn_shrimp_eq catfish_revenue_eq revenue_eq
  sorry

end RiverJoe_popcorn_shrimp_price_l20_20496


namespace midpoint_chord_hyperbola_l20_20066

-- Definitions to use in our statement
variables (a b x y : ℝ)
def ellipse : Prop := (x^2)/(a^2) + (y^2)/(b^2) = 1
def line_ellipse : Prop := x / (a^2) + y / (b^2) = 0
def hyperbola : Prop := (x^2)/(a^2) - (y^2)/(b^2) = 1
def line_hyperbola : Prop := x / (a^2) - y / (b^2) = 0

-- The theorem to prove
theorem midpoint_chord_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (x y : ℝ) 
    (h_ellipse : ellipse a b x y)
    (h_line_ellipse : line_ellipse a b x y)
    (h_hyperbola : hyperbola a b x y) :
    line_hyperbola a b x y :=
sorry

end midpoint_chord_hyperbola_l20_20066


namespace polynomial_factorization_l20_20601

variable (a b c : ℝ)

theorem polynomial_factorization :
  2 * a * (b - c)^3 + 3 * b * (c - a)^3 + 2 * c * (a - b)^3 =
  (a - b) * (b - c) * (c - a) * (5 * b - c) :=
by sorry

end polynomial_factorization_l20_20601


namespace brick_width_l20_20274

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

end brick_width_l20_20274


namespace algebra_1_algebra_2_l20_20587

variable (x1 x2 : ℝ)
variable (h_root1 : x1^2 - 2*x1 - 1 = 0)
variable (h_root2 : x2^2 - 2*x2 - 1 = 0)
variable (h_sum : x1 + x2 = 2)
variable (h_prod : x1 * x2 = -1)

theorem algebra_1 : (x1 + x2) * (x1 * x2) = -2 := by
  -- Proof here
  sorry

theorem algebra_2 : (x1 - x2)^2 = 8 := by
  -- Proof here
  sorry

end algebra_1_algebra_2_l20_20587


namespace speed_of_train_A_is_90_kmph_l20_20389

-- Definitions based on the conditions
def train_length_A := 225 -- in meters
def train_length_B := 150 -- in meters
def crossing_time := 15 -- in seconds

-- The total distance covered by train A to cross train B
def total_distance := train_length_A + train_length_B

-- The speed of train A in m/s
def speed_in_mps := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def mps_to_kmph (mps: ℕ) := mps * 36 / 10

-- The speed of train A in km/hr
def speed_in_kmph := mps_to_kmph speed_in_mps

-- The theorem to be proved
theorem speed_of_train_A_is_90_kmph : speed_in_kmph = 90 := by
  -- Proof steps go here
  sorry

end speed_of_train_A_is_90_kmph_l20_20389


namespace sufficient_but_not_necessary_l20_20784

theorem sufficient_but_not_necessary (a b : ℝ) (h₁ : a > 1) (h₂ : b > 1) : 
  (a > 1 ∧ b > 1 → a * b > 1) ∧ ¬(a * b > 1 → a > 1 ∧ b > 1) :=
by
  sorry

end sufficient_but_not_necessary_l20_20784


namespace system_of_equations_value_l20_20724

theorem system_of_equations_value (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 4 * y - 10 * z = 0)
  (hz : z ≠ 0) :
  (x^2 + 4 * x * y) / (y^2 + z^2) = 96 / 13 := 
sorry

end system_of_equations_value_l20_20724


namespace sarah_driving_distance_l20_20793

def sarah_car_mileage (miles_per_gallon : ℕ) (tank_capacity : ℕ) (initial_drive : ℕ) (refuel : ℕ) (remaining_fraction : ℚ) : Prop :=
  ∃ (total_drive : ℚ),
    (initial_drive / miles_per_gallon + refuel - (tank_capacity * remaining_fraction / 1)) * miles_per_gallon = total_drive ∧
    total_drive = 467

theorem sarah_driving_distance :
  sarah_car_mileage 28 16 280 6 (1 / 3) :=
by
  sorry

end sarah_driving_distance_l20_20793


namespace particular_solution_correct_l20_20753

-- Define the fundamental solutions y₁ and y₂
def y₁ (x : ℝ) : ℝ := Real.log x
def y₂ (x : ℝ) : ℝ := x

-- Define the differential equation and the particular solution
def differential_eq (y y' y'' : ℝ → ℝ) (x : ℝ) : Prop :=
  x^2 * (1 - y₁ x) * y'' x + x * y' x - y x = (1 - y₁ x)^2 / x

-- Define the candidate particular solution
noncomputable def particular_solution (x : ℝ) : ℝ :=
  (1 - 2 * y₁ x) / (4 * x)

-- Define the limit condition
def limit_condition (y : ℝ → ℝ) : Prop :=
  Filter.Tendsto y Filter.atTop (nhds 0)

-- Main theorem statement
theorem particular_solution_correct :
  (∀ x, differential_eq (λ x, particular_solution x) (deriv particular_solution x) (deriv (deriv particular_solution) x) x) ∧
  limit_condition particular_solution :=
  sorry

end particular_solution_correct_l20_20753


namespace inequality_logarithms_l20_20690

noncomputable def a : ℝ := Real.log 3.6 / Real.log 2
noncomputable def b : ℝ := Real.log 3.2 / Real.log 4
noncomputable def c : ℝ := Real.log 3.6 / Real.log 4

theorem inequality_logarithms : a > c ∧ c > b :=
by
  -- the proof will be written here
  sorry

end inequality_logarithms_l20_20690


namespace jackson_email_problem_l20_20630

variables (E_0 E_1 E_2 E_3 X : ℕ)

/-- Jackson's email deletion and receipt problem -/
theorem jackson_email_problem
  (h1 : E_1 = E_0 - 50 + 15)
  (h2 : E_2 = E_1 - X + 5)
  (h3 : E_3 = E_2 + 10)
  (h4 : E_3 = 30) :
  X = 50 :=
sorry

end jackson_email_problem_l20_20630


namespace number_of_blue_candles_l20_20164

-- Conditions
def grandfather_age : ℕ := 79
def yellow_candles : ℕ := 27
def red_candles : ℕ := 14
def total_candles : ℕ := grandfather_age
def yellow_red_candles : ℕ := yellow_candles + red_candles
def blue_candles : ℕ := total_candles - yellow_red_candles

-- Proof statement
theorem number_of_blue_candles : blue_candles = 38 :=
by
  -- sorry indicates the proof is omitted
  sorry

end number_of_blue_candles_l20_20164


namespace total_blocks_in_pyramid_l20_20702

-- Define the number of blocks in each layer
def blocks_in_layer (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n+1) => 3 * blocks_in_layer n

-- Prove the total number of blocks in the four-layer pyramid
theorem total_blocks_in_pyramid : 
  (blocks_in_layer 0) + (blocks_in_layer 1) + (blocks_in_layer 2) + (blocks_in_layer 3) = 40 :=
by
  sorry

end total_blocks_in_pyramid_l20_20702


namespace jerry_initial_candy_l20_20336

theorem jerry_initial_candy
  (total_bags : ℕ)
  (chocolate_hearts_bags : ℕ)
  (chocolate_kisses_bags : ℕ)
  (nonchocolate_pieces : ℕ)
  (remaining_bags : ℕ)
  (pieces_per_bag : ℕ)
  (initial_candy : ℕ)
  (h_total_bags : total_bags = 9)
  (h_chocolate_hearts_bags : chocolate_hearts_bags = 2)
  (h_chocolate_kisses_bags : chocolate_kisses_bags = 3)
  (h_nonchocolate_pieces : nonchocolate_pieces = 28)
  (h_remaining_bags : remaining_bags = total_bags - chocolate_hearts_bags - chocolate_kisses_bags)
  (h_pieces_per_bag : pieces_per_bag = nonchocolate_pieces / remaining_bags)
  (h_initial_candy : initial_candy = total_bags * pieces_per_bag) :
  initial_candy = 63 := by
  sorry

end jerry_initial_candy_l20_20336


namespace line_transformation_equiv_l20_20044

theorem line_transformation_equiv :
  (∀ x y: ℝ, (2 * x - y - 3 = 0) ↔
    (7 * (x + 2 * y) - 5 * (-x + 4 * y) - 18 = 0)) :=
sorry

end line_transformation_equiv_l20_20044


namespace cheapest_pie_cost_l20_20555

def crust_cost : ℝ := 2 + 1 + 1.5

def blueberry_pie_cost : ℝ :=
  let blueberries_needed := 3 * 16
  let containers_required := blueberries_needed / 8
  let blueberries_cost := containers_required * 2.25
  crust_cost + blueberries_cost

def cherry_pie_cost : ℝ := crust_cost + 14

theorem cheapest_pie_cost : blueberry_pie_cost = 18 :=
by sorry

end cheapest_pie_cost_l20_20555


namespace power_sum_eq_l20_20024

theorem power_sum_eq (n : ℕ) : (-2)^2009 + (-2)^2010 = 2^2009 := by
  sorry

end power_sum_eq_l20_20024


namespace total_jellybeans_l20_20428

-- Define the conditions
def a := 3 * 12       -- Caleb's jellybeans
def b := a / 2        -- Sophie's jellybeans

-- Define the goal
def total := a + b    -- Total jellybeans

-- The theorem statement
theorem total_jellybeans : total = 54 :=
by
  -- Proof placeholder
  sorry

end total_jellybeans_l20_20428


namespace price_of_shirt_l20_20383

theorem price_of_shirt (T S : ℝ) 
  (h1 : T + S = 80.34) 
  (h2 : T = S - 7.43) : 
  T = 36.455 :=
by
  sorry

end price_of_shirt_l20_20383


namespace book_cost_l20_20320

-- Definitions from conditions
def priceA : ℝ := 340
def priceB : ℝ := 350
def gain_percent_more : ℝ := 0.05

-- proof problem
theorem book_cost (C : ℝ) (G : ℝ) :
  (priceA - C = G) →
  (priceB - C = (1 + gain_percent_more) * G) →
  C = 140 :=
by
  intros
  sorry

end book_cost_l20_20320


namespace moles_of_CO2_formed_l20_20723

variables (CH4 O2 C2H2 CO2 H2O : Type)
variables (nCH4 nO2 nC2H2 nCO2 : ℕ)
variables (reactsCompletely : Prop)

-- Balanced combustion equations
axiom combustion_methane : ∀ (mCH4 mO2 mCO2 mH2O : ℕ), mCH4 = 1 → mO2 = 2 → mCO2 = 1 → mH2O = 2 → Prop
axiom combustion_acetylene : ∀ (aC2H2 aO2 aCO2 aH2O : ℕ), aC2H2 = 2 → aO2 = 5 → aCO2 = 4 → aH2O = 2 → Prop

-- Given conditions
axiom conditions :
  nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely

-- Prove the number of moles of CO2 formed
theorem moles_of_CO2_formed : 
  (nCH4 = 3 ∧ nO2 = 6 ∧ nC2H2 = 2 ∧ reactsCompletely) →
  nCO2 = 3
:= by
  intros h
  sorry

end moles_of_CO2_formed_l20_20723


namespace tank_capacity_l20_20199

theorem tank_capacity (T : ℝ) (h1 : T * (4 / 5) - T * (5 / 8) = 15) : T = 86 :=
by
  sorry

end tank_capacity_l20_20199


namespace part1_minimum_value_part2_zeros_inequality_l20_20052

noncomputable def f (x a : ℝ) := x * Real.exp x - a * (Real.log x + x)

theorem part1_minimum_value (a : ℝ) :
  (∀ x > 0, f x a > 0) ∨ (∃ x > 0, f x a = a - a * Real.log a) :=
sorry

theorem part2_zeros_inequality (a x₁ x₂ : ℝ) (hx₁ : f x₁ a = 0) (hx₂ : f x₂ a = 0) :
  Real.exp (x₁ + x₂ - 2) > 1 / (x₁ * x₂) :=
sorry

end part1_minimum_value_part2_zeros_inequality_l20_20052


namespace algebraic_expression_value_l20_20056

theorem algebraic_expression_value (x y : ℕ) (h : 3 * x - y = 1) : (8^x : ℝ) / (2^y) / 2 = 1 := 
by 
  sorry

end algebraic_expression_value_l20_20056


namespace intersection_point_of_lines_l20_20661

theorem intersection_point_of_lines : ∃ (x y : ℝ), x + y = 5 ∧ x - y = 1 ∧ x = 3 ∧ y = 2 :=
by
  sorry

end intersection_point_of_lines_l20_20661


namespace find_c_l20_20680

theorem find_c (c : ℝ) :
  (∀ x : ℝ, -2 * x^2 + c * x - 8 < 0 ↔ x ∈ Set.Iio 2 ∪ Set.Ioi 6) → c = 16 :=
by
  intros h
  sorry

end find_c_l20_20680


namespace angle_B_of_isosceles_triangle_l20_20311

theorem angle_B_of_isosceles_triangle (A B C : ℝ) (h_iso : (A = B ∨ A = C) ∨ (B = C ∨ B = A) ∨ (C = A ∨ C = B)) (h_angle_A : A = 70) :
  B = 70 ∨ B = 55 :=
by
  sorry

end angle_B_of_isosceles_triangle_l20_20311


namespace gcd_2024_1728_l20_20393

theorem gcd_2024_1728 : Int.gcd 2024 1728 = 8 := 
by
  sorry

end gcd_2024_1728_l20_20393


namespace pyramid_blocks_count_l20_20703

theorem pyramid_blocks_count :
  ∀ (n : ℕ), n = 4 →
    ∀ (a₀ : ℕ), a₀ = 1 →
      let a₁ := 3 * a₀,
      let a₂ := 3 * a₁,
      let a₃ := 3 * a₂ in
      a₀ + a₁ + a₂ + a₃ = 40 :=
by
  intros n hn a₀ h₀
  rw [hn, h₀]
  let a₁ := 3 * 1
  let a₂ := 3 * a₁
  let a₃ := 3 * a₂
  calc
    1 + a₁ + a₂ + a₃
        = 1 + 3 + a₂ + a₃   : by rw [show a₁ = 3, from rfl]
    ... = 1 + 3 + 9 + a₃    : by rw [show a₂ = 3 * 3, from rfl]
    ... = 1 + 3 + 9 + 27    : by rw [show a₃ = 3 * (3 * 3), from rfl]
    ... = 40                : rfl

end pyramid_blocks_count_l20_20703


namespace sufficient_condition_l20_20884

theorem sufficient_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 2, (1/2 : ℝ) * x^2 - a ≥ 0) → a ≤ 0 :=
by
  sorry

end sufficient_condition_l20_20884


namespace connected_components_after_optimal_play_l20_20543

open Finset Nat

def is_black_connected_component (board : ℕ × ℕ → Prop) (comp : Finset (ℕ × ℕ)) : Prop :=
  ∀ p ∈ comp, ∀ q ∈ comp, p ≠ q → (∃ path : list (ℕ × ℕ), list.chain' (λ a b, (abs (a.1 - b.1) + abs (a.2 - b.2) = 1) ∧ board a ∧ board b) (p :: path ++ [q]))

def board : ℕ × ℕ → Prop := sorry -- condition to determine if a square is black

theorem connected_components_after_optimal_play : 
  ∃ comps : Finset (Finset (ℕ × ℕ)), 
    (∀ c ∈ comps, is_black_connected_component board c) → 
    (comps.card = 16) :=
by
  sorry

end connected_components_after_optimal_play_l20_20543


namespace ratio_third_to_others_l20_20010

-- Definitions of the heights
def H1 := 600
def H2 := 2 * H1
def H3 := 7200 - (H1 + H2)

-- Definition of the ratio to be proved
def ratio := H3 / (H1 + H2)

-- The theorem statement in Lean 4
theorem ratio_third_to_others : ratio = 3 := by
  have hH1 : H1 = 600 := rfl
  have hH2 : H2 = 2 * 600 := rfl
  have hH3 : H3 = 7200 - (600 + 1200) := rfl
  have h_total : 600 + 1200 + H3 = 7200 := sorry
  have h_ratio : (7200 - (600 + 1200)) / (600 + 1200) = 3 := by sorry
  sorry

end ratio_third_to_others_l20_20010


namespace calc_miscellaneous_collective_expenses_l20_20838

def individual_needed_amount : ℕ := 450
def additional_needed_amount : ℕ := 475
def total_students : ℕ := 6
def first_day_amount : ℕ := 600
def second_day_amount : ℕ := 900
def third_day_amount : ℕ := 400
def days : ℕ := 4

def total_individual_goal : ℕ := individual_needed_amount + additional_needed_amount
def total_students_goal : ℕ := total_individual_goal * total_students
def total_first_3_days : ℕ := first_day_amount + second_day_amount + third_day_amount
def total_next_4_days : ℕ := (total_first_3_days / 2) * days
def total_raised : ℕ := total_first_3_days + total_next_4_days

def miscellaneous_collective_expenses : ℕ := total_raised - total_students_goal

theorem calc_miscellaneous_collective_expenses : miscellaneous_collective_expenses = 150 := by
  sorry

end calc_miscellaneous_collective_expenses_l20_20838


namespace option_D_correct_l20_20045

-- Definitions representing conditions
variables (a b : Line) (α : Plane)

-- Conditions
def line_parallel_plane (a : Line) (α : Plane) : Prop := sorry
def line_parallel_line (a b : Line) : Prop := sorry
def line_in_plane (b : Line) (α : Plane) : Prop := sorry

-- Theorem stating the correctness of option D
theorem option_D_correct (h1 : line_parallel_plane a α)
                         (h2 : line_parallel_line a b) :
                         (line_in_plane b α) ∨ (line_parallel_plane b α) :=
by
  sorry

end option_D_correct_l20_20045


namespace fraction_simplified_form_l20_20732

variables (a b c : ℝ)

noncomputable def fraction : ℝ := (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b)

theorem fraction_simplified_form (h : a^2 - c^2 + b^2 + 2 * a * b ≠ 0) :
  fraction a b c = (a^2 - b^2 + c^2 + 2 * b * c) / (a^2 - c^2 + b^2 + 2 * a * b) :=
by sorry

end fraction_simplified_form_l20_20732


namespace john_marks_wrongly_entered_as_l20_20079

-- Definitions based on the conditions
def john_correct_marks : ℤ := 62
def num_students : ℤ := 80
def avg_increase : ℤ := 1/2
def total_increase : ℤ := num_students * avg_increase

-- Statement to prove
theorem john_marks_wrongly_entered_as (x : ℤ) :
  (total_increase = (x - john_correct_marks)) → x = 102 :=
by {
  -- Placeholder for proof
  sorry
}

end john_marks_wrongly_entered_as_l20_20079


namespace inequality_of_sums_l20_20307

theorem inequality_of_sums (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_ineq : a > b ∧ b > c ∧ c > d) :
  (a + b + c + d)^2 > a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 :=
by
  sorry

end inequality_of_sums_l20_20307


namespace rectangle_area_l20_20688

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * (L + B) = 266) : L * B = 4290 :=
by
  sorry

end rectangle_area_l20_20688


namespace find_difference_l20_20514

theorem find_difference (x y : ℕ) (hx : ∃ k : ℕ, x = k^2) (h_sum_prod : x + y = x * y - 2006) : y - x = 666 :=
sorry

end find_difference_l20_20514


namespace exist_abc_l20_20097

theorem exist_abc (n k : ℕ) (h1 : 20 < n) (h2 : 1 < k) (h3 : n % k^2 = 0) :
  ∃ a b c : ℕ, n = a * b + b * c + c * a :=
sorry

end exist_abc_l20_20097


namespace interval_length_l20_20629

theorem interval_length (a b m h : ℝ) (h_eq : h = m / |a - b|) : |a - b| = m / h := 
by 
  sorry

end interval_length_l20_20629


namespace combined_reach_l20_20778

theorem combined_reach (barry_reach : ℝ) (larry_height : ℝ) (shoulder_ratio : ℝ) :
  barry_reach = 5 → larry_height = 5 → shoulder_ratio = 0.80 → 
  (larry_height * shoulder_ratio + barry_reach) = 9 :=
by
  intros h1 h2 h3
  sorry

end combined_reach_l20_20778


namespace second_number_pascal_triangle_with_n_plus_two_equals_43_l20_20394

theorem second_number_pascal_triangle_with_n_plus_two_equals_43 :
  (Nat.choose 42 1) = 42 :=
by
  -- Pascal's triangle's (n)th row has (n+1) numbers
  -- We solved k + 1 = 43 to get k = 42
  -- The second number in 43-numbered row is C(42, 1) = 42
  sorry 

end second_number_pascal_triangle_with_n_plus_two_equals_43_l20_20394


namespace watermelons_left_to_be_sold_tomorrow_l20_20155

def initial_watermelons : ℕ := 10 * 12
def sold_yesterday : ℕ := initial_watermelons * 40 / 100
def remaining_after_yesterday : ℕ := initial_watermelons - sold_yesterday
def sold_today : ℕ := remaining_after_yesterday / 4
def remaining_after_today : ℕ := remaining_after_yesterday - sold_today

theorem watermelons_left_to_be_sold_tomorrow : remaining_after_today = 54 := 
by
  sorry

end watermelons_left_to_be_sold_tomorrow_l20_20155


namespace olivia_race_time_l20_20560

theorem olivia_race_time (total_time : ℕ) (time_difference : ℕ) (olivia_time : ℕ)
  (h1 : total_time = 112) (h2 : time_difference = 4) (h3 : olivia_time + (olivia_time - time_difference) = total_time) :
  olivia_time = 58 :=
by
  sorry

end olivia_race_time_l20_20560


namespace delphine_chocolates_l20_20196

theorem delphine_chocolates (x : ℕ) 
  (h1 : ∃ n, n = (2 * x - 3)) 
  (h2 : ∃ m, m = (x - 2))
  (h3 : ∃ p, p = (x - 3))
  (total_eq : x + (2 * x - 3) + (x - 2) + (x - 3) + 12 = 24) : 
  x = 4 := 
sorry

end delphine_chocolates_l20_20196


namespace raw_materials_amount_true_l20_20080

def machinery_cost : ℝ := 2000
def total_amount : ℝ := 5555.56
def cash (T : ℝ) : ℝ := 0.10 * T
def raw_materials_cost (T : ℝ) : ℝ := T - machinery_cost - cash T

theorem raw_materials_amount_true :
  raw_materials_cost total_amount = 3000 := 
  by
  sorry

end raw_materials_amount_true_l20_20080


namespace train_crosses_bridge_in_12_2_seconds_l20_20969

def length_of_train : ℕ := 110
def speed_of_train_kmh : ℕ := 72
def length_of_bridge : ℕ := 134

def speed_of_train_ms : ℚ := speed_of_train_kmh * (1000 : ℚ) / (3600 : ℚ)
def total_distance : ℕ := length_of_train + length_of_bridge

noncomputable def time_to_cross_bridge : ℚ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_12_2_seconds : time_to_cross_bridge = 12.2 := by
  sorry

end train_crosses_bridge_in_12_2_seconds_l20_20969


namespace soup_adult_feeding_l20_20693

theorem soup_adult_feeding (cans_of_soup : ℕ) (cans_for_children : ℕ) (feeding_ratio : ℕ) 
  (children : ℕ) (adults : ℕ) :
  feeding_ratio = 4 → cans_of_soup = 10 → children = 20 →
  cans_for_children = (children / feeding_ratio) → 
  adults = feeding_ratio * (cans_of_soup - cans_for_children) →
  adults = 20 :=
by
  intros h1 h2 h3 h4 h5
  -- proof goes here
  sorry

end soup_adult_feeding_l20_20693


namespace measure_angle_F_l20_20210

theorem measure_angle_F :
  ∃ (F : ℝ), F = 18 ∧
  ∃ (D E : ℝ),
  D = 75 ∧
  E = 15 + 4 * F ∧
  D + E + F = 180 :=
by
  sorry

end measure_angle_F_l20_20210


namespace avg_difference_is_5_l20_20236

def avg (s : List ℕ) : ℕ :=
  s.sum / s.length

def set1 := [20, 40, 60]
def set2 := [20, 60, 25]

theorem avg_difference_is_5 :
  avg set1 - avg set2 = 5 :=
by
  sorry

end avg_difference_is_5_l20_20236


namespace average_is_correct_l20_20020

theorem average_is_correct (x : ℝ) : 
  (2 * x + 12 + 3 * x + 3 + 5 * x - 8) / 3 = 3 * x + 2 → x = -1 :=
by
  sorry

end average_is_correct_l20_20020


namespace f_2017_equals_neg_one_fourth_l20_20112

noncomputable def f : ℝ → ℝ := sorry -- Original definition will be derived from the conditions

axiom symmetry_about_y_axis : ∀ (x : ℝ), f (-x) = f x
axiom periodicity : ∀ (x : ℝ), f (x + 3) = -f x
axiom specific_interval : ∀ (x : ℝ), (3/2 < x ∧ x < 5/2) → f x = (1/2)^x

theorem f_2017_equals_neg_one_fourth : f 2017 = -1/4 :=
by sorry

end f_2017_equals_neg_one_fourth_l20_20112


namespace andy_late_minutes_l20_20546

theorem andy_late_minutes (school_starts_at : Nat) (normal_travel_time : Nat) 
  (stop_per_light : Nat) (red_lights : Nat) (construction_wait : Nat) 
  (left_house_at : Nat) : 
  let total_delay := (stop_per_light * red_lights) + construction_wait
  let total_travel_time := normal_travel_time + total_delay
  let arrive_time := left_house_at + total_travel_time
  let late_time := arrive_time - school_starts_at
  late_time = 7 :=
by
  sorry

end andy_late_minutes_l20_20546


namespace circle_radius_tangent_l20_20140

theorem circle_radius_tangent (a : ℝ) (R : ℝ) (h1 : a = 25)
  (h2 : ∀ BP DE CP CE, BP = 2 ∧ DE = 2 ∧ CP = 23 ∧ CE = 23 ∧ BP + CP = a ∧ DE + CE = a)
  : R = 17 :=
sorry

end circle_radius_tangent_l20_20140


namespace sin_2BPC_l20_20494

open Real

theorem sin_2BPC (A B C D E P : Type) (h : Equally_Spaced A B C D E)
  (h1 : cos (angle P A C) = 3 / 5)
  (h2 : cos (angle P B D) = 1 / 2) : sin (2 * angle P B C) = 3 * sqrt 3 / 5 :=
sorry

end sin_2BPC_l20_20494


namespace find_number_l20_20397

theorem find_number (N : ℕ) : 
  (N % 13 = 11) ∧ (N % 17 = 9) ↔ N = 141 :=
by 
  sorry

end find_number_l20_20397


namespace initial_chocolate_bars_l20_20139

theorem initial_chocolate_bars (B : ℕ) 
  (H1 : Thomas_and_friends_take = B / 4)
  (H2 : One_friend_returns_5 = Thomas_and_friends_take - 5)
  (H3 : Piper_takes = Thomas_and_friends_take - 5 - 5)
  (H4 : Remaining_bars = B - Thomas_and_friends_take - Piper_takes)
  (H5 : Remaining_bars = 110) :
  B = 190 := 
sorry

end initial_chocolate_bars_l20_20139


namespace simplify_and_evaluate_expression_l20_20361

theorem simplify_and_evaluate_expression
    (a b : ℤ)
    (h1 : a = -1/3)
    (h2 : b = -2) :
  ((3 * a + b)^2 - (3 * a + b) * (3 * a - b)) / (2 * b) = -3 :=
by
  sorry

end simplify_and_evaluate_expression_l20_20361


namespace vertex_on_x_axis_iff_t_eq_neg_4_l20_20808

theorem vertex_on_x_axis_iff_t_eq_neg_4 (t : ℝ) :
  (∃ x : ℝ, (4 + t) = 0) ↔ t = -4 :=
by
  sorry

end vertex_on_x_axis_iff_t_eq_neg_4_l20_20808


namespace assignment_increment_l20_20245

theorem assignment_increment (M : ℤ) : (M = M + 3) → false :=
by
  sorry

end assignment_increment_l20_20245


namespace find_d_l20_20086

-- Definitions based on conditions
def f (x : ℝ) (c : ℝ) := 5 * x + c
def g (x : ℝ) (c : ℝ) := c * x + 3

-- The theorem statement
theorem find_d (c d : ℝ) (h₁ : f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry -- Proof is omitted as per the instructions

end find_d_l20_20086


namespace bacteria_after_7_hours_l20_20979

noncomputable def bacteria_growth (initial : ℝ) (t : ℝ) (k : ℝ) : ℝ := initial * (10 * (Real.exp (k * t)))

noncomputable def solve_bacteria_problem : ℝ :=
let doubling_time := 1 / 60 -- In hours, since 60 minutes is 1 hour
-- Given that it doubles in 1 hour, we expect the growth to be such that y = initial * (2) in 1 hour.
let k := Real.log 2 -- Since when t = 1, we have 10 * e^(k * 1) = 2 * 10
bacteria_growth 10 7 k

theorem bacteria_after_7_hours :
  solve_bacteria_problem = 1280 :=
by
  sorry

end bacteria_after_7_hours_l20_20979


namespace find_p_q_d_l20_20658

noncomputable def cubic_polynomial_real_root (p q d : ℕ) (x : ℝ) : Prop :=
  27 * x^3 - 12 * x^2 - 4 * x - 1 = 0 ∧ x = (p^(1/3) + q^(1/3) + 1) / d ∧
  p > 0 ∧ q > 0 ∧ d > 0

theorem find_p_q_d :
  ∃ (p q d : ℕ), cubic_polynomial_real_root p q d 1 ∧ p + q + d = 3 :=
by
  sorry

end find_p_q_d_l20_20658


namespace probability_king_even_coords_2008_l20_20924

noncomputable def king_probability_even_coords (turns : ℕ) : ℝ :=
  let p_stay := 0.4
  let p_edge := 0.1
  let p_diag := 0.05
  if turns = 2008 then
    (5 ^ 2008 + 1) / (2 * 5 ^ 2008)
  else
    0 -- default value for other cases

theorem probability_king_even_coords_2008 :
  king_probability_even_coords 2008 = (5 ^ 2008 + 1) / (2 * 5 ^ 2008) :=
by
  sorry

end probability_king_even_coords_2008_l20_20924


namespace Aiyanna_has_more_cookies_l20_20832

def Alyssa_cookies : ℕ := 129
def Aiyanna_cookies : ℕ := 140

theorem Aiyanna_has_more_cookies :
  Aiyanna_cookies - Alyssa_cookies = 11 := by
  sorry

end Aiyanna_has_more_cookies_l20_20832


namespace weight_of_replaced_person_l20_20238

-- Define the conditions in Lean 4
variables {w_replaced : ℝ}   -- Weight of the person who was replaced
variables {w_new : ℝ}        -- Weight of the new person
variables {n : ℕ}            -- Number of persons
variables {avg_increase : ℝ} -- Increase in average weight

-- Set up the given conditions
axiom h1 : n = 8
axiom h2 : avg_increase = 2.5
axiom h3 : w_new = 40

-- Theorem that states the weight of the replaced person
theorem weight_of_replaced_person : w_replaced = 20 :=
by
  sorry

end weight_of_replaced_person_l20_20238


namespace log_function_fixed_point_l20_20445

noncomputable def f (a x : ℝ) := log a x + 2

theorem log_function_fixed_point (a : ℝ) (h : (0 < a ∧ a < 1) ∨ 1 < a) : (1, 2) ∈ set_of (λ p : ℝ × ℝ, ∃ x, p = (x, f a x)) :=
by
  sorry

end log_function_fixed_point_l20_20445


namespace cylinder_height_relationship_l20_20254

theorem cylinder_height_relationship
  (r1 h1 r2 h2 : ℝ)
  (vol_eq : π * r1^2 * h1 = π * r2^2 * h2)
  (radius_rel : r2 = (6 / 5) * r1) : h1 = (36 / 25) * h2 := 
sorry

end cylinder_height_relationship_l20_20254


namespace two_digit_number_with_tens_5_l20_20852

-- Definitions and conditions
variable (A : Nat)

-- Problem statement as a Lean theorem
theorem two_digit_number_with_tens_5 (hA : A < 10) : (10 * 5 + A) = 50 + A := by
  sorry

end two_digit_number_with_tens_5_l20_20852


namespace tile_size_l20_20482

theorem tile_size (length width : ℕ) (total_tiles : ℕ) 
  (h_length : length = 48) 
  (h_width : width = 72) 
  (h_total_tiles : total_tiles = 96) : 
  ((length * width) / total_tiles) = 36 := 
by
  sorry

end tile_size_l20_20482


namespace intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l20_20594

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x + Real.log (2 - x) + a * x

theorem intervals_of_monotonicity_a_eq_1 : 
  ∀ x : ℝ, (0 < x ∧ x < Real.sqrt 2) → 
  f x 1 < f (Real.sqrt 2) 1 ∧ 
  ∀ x : ℝ, (Real.sqrt 2 < x ∧ x < 2) → 
  f x 1 > f (Real.sqrt 2) 1 := 
sorry

theorem max_value_implies_a_half : 
  ∀ x : ℝ, (0 < x ∧ x ≤ 1) ∧ f 1 a = 1/2 → a = 1/2 := 
sorry

end intervals_of_monotonicity_a_eq_1_max_value_implies_a_half_l20_20594


namespace carol_weight_l20_20664

variable (a c : ℝ)

theorem carol_weight (h1 : a + c = 240) (h2 : c - a = (2 / 3) * c) : c = 180 :=
by
  sorry

end carol_weight_l20_20664


namespace exists_infinitely_many_n_with_increasing_ω_l20_20915

open Nat

/--
  Let ω(n) represent the number of distinct prime factors of a natural number n (where n > 1).
  Prove that there exist infinitely many n such that ω(n) < ω(n + 1) < ω(n + 2).
-/
theorem exists_infinitely_many_n_with_increasing_ω (ω : ℕ → ℕ) (hω : ∀ (n : ℕ), n > 1 → ∃ k, ω k < ω (k + 1) ∧ ω (k + 1) < ω (k + 2)) :
  ∃ (infinitely_many : ℕ → Prop), ∀ N : ℕ, ∃ n : ℕ, N < n ∧ infinitely_many n :=
by
  sorry

end exists_infinitely_many_n_with_increasing_ω_l20_20915


namespace compute_fraction_pow_mul_l20_20170

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l20_20170


namespace leo_weight_l20_20895

-- Definitions from the conditions
variable (L K J M : ℝ)

-- Conditions 
def condition1 : Prop := L + 15 = 1.60 * K
def condition2 : Prop := L + 15 = 0.40 * J
def condition3 : Prop := J = K + 25
def condition4 : Prop := M = K - 20
def condition5 : Prop := L + K + J + M = 350

-- Final statement to prove based on the conditions
theorem leo_weight (h1 : condition1 L K) (h2 : condition2 L J) (h3 : condition3 J K) 
                   (h4 : condition4 M K) (h5 : condition5 L K J M) : L = 110.22 :=
by 
  sorry

end leo_weight_l20_20895


namespace circle_radius_l20_20455

-- Define the general equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Prove the radius of the circle given by the equation is √5
theorem circle_radius :
  (∀ x y : ℝ, circle_eq x y) →
  (∃ r : ℝ, r = Real.sqrt 5) :=
by
  sorry

end circle_radius_l20_20455


namespace monotonic_increasing_interval_l20_20191

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem monotonic_increasing_interval :
  ∃ a b : ℝ, a < b ∧
    ∀ x y : ℝ, (a < x ∧ x < b) → (a < y ∧ y < b) → x < y → f x < f y ∧ a = -Real.pi / 6 ∧ b = Real.pi / 3 :=
by
  sorry

end monotonic_increasing_interval_l20_20191


namespace total_chairs_l20_20239

theorem total_chairs (indoor_tables outdoor_tables chairs_per_table : ℕ) (h_indoor : indoor_tables = 8) (h_outdoor : outdoor_tables = 12) (h_chairs : chairs_per_table = 3) :
  indoor_tables * chairs_per_table + outdoor_tables * chairs_per_table = 60 :=
by
  rw [h_indoor, h_outdoor, h_chairs]
  norm_num


end total_chairs_l20_20239


namespace shape_with_circular_views_is_sphere_l20_20414

-- Definitions of the views of different geometric shapes
inductive Shape
| Cuboid : Shape
| Cylinder : Shape
| Cone : Shape
| Sphere : Shape

def front_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def left_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Rectangle
| Shape.Cone := False  -- Isosceles Triangle
| Shape.Sphere := True  -- Circle

def top_view : Shape → Prop
| Shape.Cuboid := False  -- Rectangle
| Shape.Cylinder := False  -- Circle, but not all views
| Shape.Cone := False  -- Circle, but not all views
| Shape.Sphere := True  -- Circle

-- The theorem to be proved
theorem shape_with_circular_views_is_sphere (s : Shape) :
  (front_view s ↔ True) ∧ (left_view s ↔ True) ∧ (top_view s ↔ True) ↔ s = Shape.Sphere :=
by sorry

end shape_with_circular_views_is_sphere_l20_20414


namespace two_students_solved_at_least_five_common_problems_l20_20049

open Finset Nat

theorem two_students_solved_at_least_five_common_problems
  (students : Fin 31) (problems : Fin 10)
  (solves_problems : students → Finset problems)
  (H1 : ∀ s : students, (solves_problems s).card ≥ 6) :
  ∃ (s1 s2 : students), s1 ≠ s2 ∧ ((solves_problems s1) ∩ (solves_problems s2)).card ≥ 5 := by
  sorry

end two_students_solved_at_least_five_common_problems_l20_20049


namespace R_transformed_is_R_l20_20102

-- Define the initial coordinates of the rectangle PQRS
def P : ℝ × ℝ := (3, 4)
def Q : ℝ × ℝ := (6, 4)
def R : ℝ × ℝ := (6, 1)
def S : ℝ × ℝ := (3, 1)

-- Define the reflection across the x-axis
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

-- Define the translation down by 2 units
def translate_down_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 - 2)

-- Define the reflection across the line y = -x
def reflect_y_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

-- Define the translation up by 2 units
def translate_up_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, p.2 + 2)

-- Define the transformation to find R''
def transform (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_up_2 (reflect_y_neg_x (translate_down_2 (reflect_x p)))

-- Prove that the result of transforming R is (-3, -4)
theorem R_transformed_is_R'' : transform R = (-3, -4) :=
  by sorry

end R_transformed_is_R_l20_20102


namespace find_a_plus_b_eq_102_l20_20364

theorem find_a_plus_b_eq_102 :
  ∃ (a b : ℕ), (1600^(1 / 2) - 24 = (a^(1 / 2) - b)^2) ∧ (a + b = 102) :=
by {
  sorry
}

end find_a_plus_b_eq_102_l20_20364


namespace complement_union_eq_l20_20613

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l20_20613


namespace complement_union_A_B_eq_neg2_0_l20_20607

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l20_20607


namespace max_xy_l20_20308

theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  xy ≤ 1 / 4 := 
sorry

end max_xy_l20_20308


namespace philip_car_mileage_typical_week_l20_20951

-- Definitions for distances and frequencies
def distance_to_school := 2.5
def distance_to_market := 2
def school_round_trips_per_day := 2
def school_days_per_week := 4
def market_trips_per_week := 1

-- Calculate total weekly mileage
def total_weekly_mileage := 
  let school_trip := distance_to_school * 2
  let market_trip := distance_to_market * 2
  let weekly_school_miles := school_trip * school_round_trips_per_day * school_days_per_week
  let weekly_market_miles := market_trip * market_trips_per_week
  weekly_school_miles + weekly_market_miles

-- Theorem statement for the proof problem
theorem philip_car_mileage_typical_week : total_weekly_mileage = 44 := by
  sorry

end philip_car_mileage_typical_week_l20_20951


namespace sum_of_x_coordinates_of_other_vertices_l20_20318

theorem sum_of_x_coordinates_of_other_vertices {x1 y1 x2 y2 x3 y3 x4 y4: ℝ} 
    (h1 : (x1, y1) = (2, 12))
    (h2 : (x2, y2) = (8, 3))
    (midpoint_eq : (x1 + x2) / 2 = (x3 + x4) / 2) 
    : x3 + x4 = 10 := 
by
  have h4 : (2 + 8) / 2 = 5 := by norm_num
  have h5 : 2 * 5 = 10 := by norm_num
  sorry

end sum_of_x_coordinates_of_other_vertices_l20_20318


namespace smallest_natural_with_properties_l20_20870

theorem smallest_natural_with_properties :
  ∃ n : ℕ, (∃ N : ℕ, n = 10 * N + 6) ∧ 4 * (10 * N + 6) = 6 * 10^(5 : ℕ) + N ∧ n = 153846 := sorry

end smallest_natural_with_properties_l20_20870


namespace projection_of_m_on_n_l20_20458

noncomputable def a : ℝ × ℝ := (√3, 0)
noncomputable def b : ℝ × ℝ := (1, 0)
noncomputable def m : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)
noncomputable def n : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)
noncomputable def dot_product (x y : ℝ × ℝ) := x.1 * y.1 + x.2 * y.2
noncomputable def magnitude (x : ℝ × ℝ) := Real.sqrt (dot_product x x)

theorem projection_of_m_on_n : 
  (dot_product m n) / (magnitude n) = 2 :=
by
  sorry

end projection_of_m_on_n_l20_20458


namespace tiger_speed_l20_20014

variable (v_t : ℝ) (hours_head_start : ℝ := 5) (hours_zebra_to_catch : ℝ := 6) (speed_zebra : ℝ := 55)

-- Define the distance covered by the tiger and the zebra
def distance_tiger (v_t : ℝ) (hours : ℝ) : ℝ := v_t * hours
def distance_zebra (hours : ℝ) (speed_zebra : ℝ) : ℝ := speed_zebra * hours

theorem tiger_speed :
  v_t * hours_head_start + v_t * hours_zebra_to_catch = distance_zebra hours_zebra_to_catch speed_zebra →
  v_t = 30 :=
by
  sorry

end tiger_speed_l20_20014


namespace f_2009_is_one_l20_20822

   -- Define the properties of the function f
   variables (f : ℤ → ℤ)
   variable (h_even : ∀ x : ℤ, f x = f (-x))
   variable (h1 : f 1 = 1)
   variable (h2008 : f 2008 ≠ 1)
   variable (h_max : ∀ a b : ℤ, f (a + b) ≤ max (f a) (f b))

   -- Prove that f(2009) = 1
   theorem f_2009_is_one : f 2009 = 1 :=
   sorry
   
end f_2009_is_one_l20_20822


namespace min_value_expression_l20_20566

theorem min_value_expression :
  ∃ θ : ℝ, 0 < θ ∧ θ < π / 2 ∧ (∀ θ' : ℝ, 0 < θ' ∧ θ' < π / 2 → 
    (3 * Real.sin θ' + 4 / Real.cos θ' + 2 * Real.sqrt 3 * Real.tan θ') ≥ 9 * Real.sqrt 3) ∧ 
    (3 * Real.sin θ + 4 / Real.cos θ + 2 * Real.sqrt 3 * Real.tan θ = 9 * Real.sqrt 3) :=
by
  sorry

end min_value_expression_l20_20566


namespace eighth_term_in_arithmetic_sequence_l20_20123

theorem eighth_term_in_arithmetic_sequence : 
  ∀ (a1 d : ℚ), a1 = 2 / 3 → d = 1 / 3 → (a1 + 7 * d) = 3 :=
by
  intros a1 d h1 h2
  rw [h1, h2]
  simp
  norm_num
  sorry

end eighth_term_in_arithmetic_sequence_l20_20123


namespace stock_worth_l20_20267

theorem stock_worth (X : ℝ)
  (H1 : 0.2 * X * 0.1 = 0.02 * X)  -- 20% of stock at 10% profit given in condition.
  (H2 : 0.8 * X * 0.05 = 0.04 * X) -- Remaining 80% of stock at 5% loss given in condition.
  (H3 : 0.04 * X - 0.02 * X = 400) -- Overall loss incurred is Rs. 400.
  : X = 20000 := 
sorry

end stock_worth_l20_20267


namespace largest_pillar_radius_l20_20835

-- Define the dimensions of the crate
def crate_length := 12
def crate_width := 8
def crate_height := 3

-- Define the condition that the pillar is a right circular cylinder
def is_right_circular_cylinder (r : ℝ) (h : ℝ) : Prop :=
  r > 0 ∧ h > 0

-- The theorem stating the radius of the largest volume pillar that can fit in the crate
theorem largest_pillar_radius (r h : ℝ) (cylinder_fits : is_right_circular_cylinder r h) :
  r = 1.5 := 
sorry

end largest_pillar_radius_l20_20835


namespace concurrency_of_lines_l20_20624

noncomputable def circle (C : Type) := 
  { center : C, radius : ℝ }

structure TangencyCondition (C : Type) :=
  (common_tangent : C → C → set (line C))

-- Given conditions
variables {C : Type} [euclidean_space C]

def circle_c : circle C := ⟨O, r⟩
def circle_c1 : circle C := ⟨O₁, r₁⟩
def circle_c2 : circle C := ⟨O₂, r₂⟩

-- Tangency conditions
variable (A₁ A₂ A : C)

axiom is_internally_tangent_circle_c_c1 : ∀ ⦃x : C⦄, x = A₁ → ∀ {y : C}, y = O₁ → (x ≠ y) ∧ (O + O₁).intersection (O₁ + A₁) = A₁
axiom is_internally_tangent_circle_c_c2 : ∀ ⦃x : C⦄, x = A₂ → ∀ {y : C}, y = O₂ → (x ≠ y) ∧ (O + O₂).intersection (O₂ + A₂) = A₂
axiom is_externally_tangent_circle_c1_c2 : ∀ ⦃x : C⦄, x = A → ∀ {y : C}, y = A → (y ≠ x) ∧ (O₁ + O₂).intersection (O₁ + A) = A

-- Goal to prove: concurrency of lines
theorem concurrency_of_lines :
  ( (line_through O A) ∩ (line_through O₁ A₂) ) ∩ ( (line_through O₂ A₁) ) ≠ ∅ :=
sorry

end concurrency_of_lines_l20_20624


namespace average_stamps_per_day_l20_20639

theorem average_stamps_per_day :
  let a1 := 8
  let d := 8
  let n := 6
  let stamps_collected : Fin n → ℕ := λ i => a1 + i * d
  -- sum the stamps collected over six days
  let S := List.sum (List.ofFn stamps_collected)
  -- calculate average
  let average := S / n
  average = 28 :=
by sorry

end average_stamps_per_day_l20_20639


namespace sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l20_20263

theorem sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine : 
  Real.sqrt (3^3 + 3^3 + 3^3) = 9 :=
by 
  sorry

end sqrt_three_pow_three_plus_three_pow_three_plus_three_pow_three_eq_nine_l20_20263


namespace sum_first_n_terms_arithmetic_sequence_l20_20783

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n m : ℕ, a (m + 1) - a m = d

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (a12 : a 12 = -8) (S9 : S 9 = -9) (h_arith : is_arithmetic_sequence a) :
  S 16 = -72 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l20_20783


namespace right_triangle_equation_l20_20233

-- Let a, b, and c be the sides of a right triangle with a^2 + b^2 = c^2
variables (a b c : ℕ)
-- Define the semiperimeter
def semiperimeter (a b c : ℕ) : ℕ := (a + b + c) / 2
-- Define the radius of the inscribed circle
def inscribed_radius (a b c : ℕ) : ℚ := (a * b) / (2 * semiperimeter a b c)
-- State the theorem to prove
theorem right_triangle_equation : 
    ∀ a b c : ℕ, a^2 + b^2 = c^2 → semiperimeter a b c + inscribed_radius a b c = a + b := by
  sorry

end right_triangle_equation_l20_20233


namespace mark_trees_total_l20_20919

def mark_trees (current_trees new_trees : Nat) : Nat :=
  current_trees + new_trees

theorem mark_trees_total (x y : Nat) (h1 : x = 13) (h2 : y = 12) :
  mark_trees x y = 25 :=
by
  rw [h1, h2]
  sorry

end mark_trees_total_l20_20919


namespace division_problem_l20_20266

variables (a b c : ℤ)

theorem division_problem 
  (h1 : a ∣ b * c - 1)
  (h2 : b ∣ c * a - 1)
  (h3 : c ∣ a * b - 1) : 
  abc ∣ ab + bc + ca - 1 := 
sorry

end division_problem_l20_20266


namespace domain_of_f_l20_20798

def f (x : ℝ) : ℝ := (sqrt (x - 3)) / (|x + 1| - 5)

theorem domain_of_f :
  {x : ℝ | (3 ≤ x ∧ x < 4) ∨ (4 < x)} = [3, 4) ∪ (4, +∞) :=
by
  sorry

end domain_of_f_l20_20798


namespace find_matrix_M_l20_20740

-- Define the given matrix with real entries
def matrix_M : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 2], ![-1, 0]]

-- Define the function for matrix operations
def M_calc (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (M * M * M) - (M * M) + (2 • M)

-- Define the target matrix
def target_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 3], ![-2, 0]]

-- Problem statement: The matrix M should satisfy the given matrix equation
theorem find_matrix_M (M : Matrix (Fin 2) (Fin 2) ℝ) :
  M_calc M = target_matrix ↔ M = matrix_M :=
sorry

end find_matrix_M_l20_20740


namespace distance_from_A_to_O_is_3_l20_20209

-- Define polar coordinates with the given conditions
def point_A : ℝ × ℝ := (3, -4)

-- Define the distance function in terms of polar coordinates
def distance_to_pole_O (coords : ℝ × ℝ) : ℝ := coords.1

-- The main theorem to be proved
theorem distance_from_A_to_O_is_3 : distance_to_pole_O point_A = 3 := by
  sorry

end distance_from_A_to_O_is_3_l20_20209


namespace area_of_inscribed_triangle_l20_20150

noncomputable def calculate_triangle_area_inscribed_in_circle 
  (arc1 : ℝ) (arc2 : ℝ) (arc3 : ℝ) (total_circumference := arc1 + arc2 + arc3)
  (radius := total_circumference / (2 * Real.pi))
  (theta := (2 * Real.pi) / total_circumference)
  (angle1 := 5 * theta) (angle2 := 7 * theta) (angle3 := 8 * theta) : ℝ :=
  0.5 * (radius ^ 2) * (Real.sin angle1 + Real.sin angle2 + Real.sin angle3)

theorem area_of_inscribed_triangle : 
  calculate_triangle_area_inscribed_in_circle 5 7 8 = 119.85 / (Real.pi ^ 2) :=
by
  sorry

end area_of_inscribed_triangle_l20_20150


namespace sum_after_third_rotation_max_sum_of_six_faces_l20_20258

variable (a b c : ℕ) (a' b': ℕ)

-- Initial Conditions
axiom sum_initial : a + b + c = 42

-- Conditions after first rotation
axiom a_prime : a' = a - 8
axiom sum_first_rotation : b + c + a' = 34

-- Conditions after second rotation
axiom b_prime : b' = b + 19
axiom sum_second_rotation : c + a' + b' = 53

-- The cube always rests on the face with number 6
axiom bottom_face : c = 6

-- Prove question 1:
theorem sum_after_third_rotation : (b + 19) + a + c = 61 :=
by sorry

-- Prove question 2:
theorem max_sum_of_six_faces : 
∃ d e f: ℕ, d = a ∧ e = b ∧ f = c ∧ d + e + f + (a - 8) + (b + 19) + 6 = 100 :=
by sorry

end sum_after_third_rotation_max_sum_of_six_faces_l20_20258


namespace sequence_exists_and_unique_l20_20667

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l20_20667


namespace alpha_half_in_II_IV_l20_20046

theorem alpha_half_in_II_IV (k : ℤ) (α : ℝ) (h : 2 * k * π - π / 2 < α ∧ α < 2 * k * π) : 
  (k * π - π / 4 < (α / 2) ∧ (α / 2) < k * π) :=
by
  sorry

end alpha_half_in_II_IV_l20_20046


namespace new_average_weight_l20_20386

def num_people := 6
def avg_weight1 := 154
def weight_seventh := 133

theorem new_average_weight :
  (num_people * avg_weight1 + weight_seventh) / (num_people + 1) = 151 := by
  sorry

end new_average_weight_l20_20386


namespace calculation_l20_20719

theorem calculation : 
  ((18 ^ 13 * 18 ^ 11) ^ 2 / 6 ^ 8) * 3 ^ 4 = 2 ^ 40 * 3 ^ 92 :=
by sorry

end calculation_l20_20719


namespace ratio_second_week_to_first_week_l20_20922

def cases_week_1 : ℤ := 5000
def cases_week_2 : ℤ := 1250
def cases_week_3 : ℤ := cases_week_2 + 2000
def total_cases := cases_week_1 + cases_week_2 + cases_week_3

theorem ratio_second_week_to_first_week : 
  total_cases = 9500 → 
  (cases_week_2 : ℚ) / cases_week_1 = 1 / 4 :=
by
  intros h_total_cases
  have h_cases_week_2 : cases_week_2 = 1250,
  sorry
  have h_cases_week_1 : cases_week_1 = 5000,
  sorry
  rw [h_cases_week_2, h_cases_week_1],
  norm_num

end ratio_second_week_to_first_week_l20_20922


namespace product_of_values_of_x_l20_20408

theorem product_of_values_of_x : 
  (∃ x : ℝ, |x^2 - 7| - 3 = -1) → 
  (∀ x1 x2 x3 x4 : ℝ, 
    (|x1^2 - 7| - 3 = -1) ∧
    (|x2^2 - 7| - 3 = -1) ∧
    (|x3^2 - 7| - 3 = -1) ∧
    (|x4^2 - 7| - 3 = -1) 
    → x1 * x2 * x3 * x4 = 45) :=
sorry

end product_of_values_of_x_l20_20408


namespace constant_term_in_binomial_expansion_is_40_l20_20069

-- Define the binomial coefficient C(n, k)
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the expression for the binomial expansion of (x^2 + 2/x^3)^5
def term (r : ℕ) : ℕ := binom 5 r * 2^r

theorem constant_term_in_binomial_expansion_is_40 
  (x : ℝ) (h : x ≠ 0) : 
  (∃ r : ℕ, 10 - 5 * r = 0) ∧ term 2 = 40 :=
by 
  sorry

end constant_term_in_binomial_expansion_is_40_l20_20069


namespace negative_expressions_l20_20940

-- Define the approximated values for P, Q, R, S, and T
def P : ℝ := 3.5
def Q : ℝ := 1.1
def R : ℝ := -0.1
def S : ℝ := 0.9
def T : ℝ := 1.5

-- State the theorem to be proved
theorem negative_expressions : 
  (R / (P * Q) < 0) ∧ ((S + T) / R < 0) :=
by
  sorry

end negative_expressions_l20_20940


namespace calculate_subtraction_l20_20671

theorem calculate_subtraction :
  ∀ (x : ℕ), (49 = 50 - 1) → (49^2 = 50^2 - 99)
  := by
  intros x h
  sorry

end calculate_subtraction_l20_20671


namespace quadratic_real_roots_l20_20743

theorem quadratic_real_roots (k : ℝ) (h : ∀ x : ℝ, k * x^2 - 4 * x + 1 = 0) : k ≤ 4 ∧ k ≠ 0 :=
by
  sorry

end quadratic_real_roots_l20_20743


namespace least_multiple_of_7_not_lucky_l20_20707

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_multiple_of_7_not_lucky : ∃ n, is_multiple_of_7 n ∧ ¬ is_lucky n ∧ n = 14 :=
by
  sorry

end least_multiple_of_7_not_lucky_l20_20707


namespace passing_percentage_is_correct_l20_20148

theorem passing_percentage_is_correct :
  ∀ (marks_obtained : ℕ) (marks_failed_by : ℕ) (max_marks : ℕ),
    marks_obtained = 59 →
    marks_failed_by = 40 →
    max_marks = 300 →
    (marks_obtained + marks_failed_by) / max_marks * 100 = 33 :=
by
  intros marks_obtained marks_failed_by max_marks h1 h2 h3
  sorry

end passing_percentage_is_correct_l20_20148


namespace halfway_between_one_third_and_one_eighth_l20_20867

theorem halfway_between_one_third_and_one_eighth : (1/3 + 1/8) / 2 = 11 / 48 :=
by
  -- The proof goes here
  sorry

end halfway_between_one_third_and_one_eighth_l20_20867


namespace Yi_visited_city_A_l20_20807

variable (visited : String -> String -> Prop) -- denote visited "Student" "City"
variables (Jia Yi Bing : String) (A B C : String)

theorem Yi_visited_city_A
  (h1 : visited Jia A ∧ visited Jia C ∧ ¬ visited Jia B)
  (h2 : ¬ visited Yi C)
  (h3 : visited Jia A ∧ visited Yi A ∧ visited Bing A) :
  visited Yi A :=
by
  sorry

end Yi_visited_city_A_l20_20807


namespace opposite_of_neg_2022_l20_20941

theorem opposite_of_neg_2022 : -(-2022) = 2022 :=
by
  sorry

end opposite_of_neg_2022_l20_20941


namespace calculate_possible_change_l20_20918

structure ChangeProblem where
  (change : ℕ)
  (h1 : change < 100)
  (h2 : ∃ (q : ℕ), change = 25 * q + 10 ∧ q ≤ 3)
  (h3 : ∃ (d : ℕ), change = 10 * d + 20 ∧ d ≤ 9)

theorem calculate_possible_change (p1 p2 p3 p4 : ChangeProblem) :
  p1.change + p2.change + p3.change = 180 :=
by
  sorry

end calculate_possible_change_l20_20918


namespace aaronTotalOwed_l20_20716

def monthlyPayment : ℝ := 100
def numberOfMonths : ℕ := 12
def interestRate : ℝ := 0.1

def totalCostWithoutInterest : ℝ := monthlyPayment * (numberOfMonths : ℝ)
def interestAmount : ℝ := totalCostWithoutInterest * interestRate
def totalAmountOwed : ℝ := totalCostWithoutInterest + interestAmount

theorem aaronTotalOwed : totalAmountOwed = 1320 := by
  sorry

end aaronTotalOwed_l20_20716


namespace domain_of_f_l20_20739

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x) + Real.sqrt (x * (x + 1))

theorem domain_of_f :
  {x : ℝ | -x ≥ 0 ∧ x * (x + 1) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x = 0} :=
by
  sorry

end domain_of_f_l20_20739


namespace equilateral_triangle_area_with_inscribed_circle_l20_20120

theorem equilateral_triangle_area_with_inscribed_circle
  (r : ℝ) (area_circle : ℝ) (area_triangle : ℝ) 
  (h_inscribed_circle_area : area_circle = 9 * Real.pi)
  (h_radius : r = 3) :
  area_triangle = 27 * Real.sqrt 3 :=
by
  sorry

end equilateral_triangle_area_with_inscribed_circle_l20_20120


namespace cat_birds_total_l20_20136

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l20_20136


namespace quadratic_roots_range_l20_20593

theorem quadratic_roots_range (m : ℝ) : 
  (2 * x^2 - (m + 1) * x + m = 0) → 
  (m^2 - 6 * m + 1 > 0) → 
  (0 < m) → 
  (0 < m ∧ m < 3 - 2 * Real.sqrt 2 ∨ m > 3 + 2 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_range_l20_20593


namespace polynomial_evaluation_l20_20395

theorem polynomial_evaluation :
  102^5 - 5 * 102^4 + 10 * 102^3 - 10 * 102^2 + 5 * 102 - 1 = 101^5 :=
by
  sorry

end polynomial_evaluation_l20_20395


namespace perimeter_of_square_is_64_l20_20846

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l20_20846


namespace maxEccentricity_l20_20453

noncomputable def majorAxisLength := 4
noncomputable def majorSemiAxis := 2
noncomputable def leftVertexParabolaEq (y : ℝ) := y^2 = -3
noncomputable def distanceCondition (c : ℝ) := 2^2 / c - 2 ≥ 1

theorem maxEccentricity : ∃ c : ℝ, distanceCondition c ∧ (c ≤ 4 / 3) ∧ (c / majorSemiAxis = 2 / 3) :=
by
  sorry

end maxEccentricity_l20_20453


namespace cheapest_pie_l20_20554

def cost_flour : ℝ := 2
def cost_sugar : ℝ := 1
def cost_eggs_butter : ℝ := 1.5
def cost_crust : ℝ := cost_flour + cost_sugar + cost_eggs_butter

def weight_blueberries : ℝ := 3
def container_weight : ℝ := 0.5 -- 8 oz in pounds
def price_per_blueberry_container : ℝ := 2.25
def cost_blueberries (weight: ℝ) (container_weight: ℝ) (price_per_container: ℝ) : ℝ :=
  (weight / container_weight) * price_per_container

def weight_cherries : ℝ := 4
def price_cherry_bag : ℝ := 14

def cost_blueberry_pie : ℝ := cost_crust + cost_blueberries weight_blueberries container_weight price_per_blueberry_container
def cost_cherry_pie : ℝ := cost_crust + price_cherry_bag

theorem cheapest_pie : min cost_blueberry_pie cost_cherry_pie = 18 := by
  sorry

end cheapest_pie_l20_20554


namespace Genevieve_drinks_pints_l20_20003

theorem Genevieve_drinks_pints :
  ∀ (total_gallons : ℝ) (num_thermoses : ℕ) (pints_per_gallon : ℝ) (genevieve_thermoses : ℕ),
  total_gallons = 4.5 → num_thermoses = 18 → pints_per_gallon = 8 → genevieve_thermoses = 3 →
  (genevieve_thermoses * ((total_gallons / num_thermoses) * pints_per_gallon) = 6) :=
by
  intros total_gallons num_thermoses pints_per_gallon genevieve_thermoses
  intros h1 h2 h3 h4
  sorry

end Genevieve_drinks_pints_l20_20003


namespace average_price_correct_l20_20929

-- Define the conditions
def books_shop1 : ℕ := 65
def price_shop1 : ℕ := 1480
def books_shop2 : ℕ := 55
def price_shop2 : ℕ := 920

-- Define the total books and total price based on conditions
def total_books : ℕ := books_shop1 + books_shop2
def total_price : ℕ := price_shop1 + price_shop2

-- Define the average price based on total books and total price
def average_price : ℕ := total_price / total_books

-- Theorem stating the average price per book Sandy paid
theorem average_price_correct : average_price = 20 :=
  by
  sorry

end average_price_correct_l20_20929


namespace factorial_inequality_l20_20232

theorem factorial_inequality (n : ℕ) (h : n ≥ 1) : n! ≤ ((n+1)/2)^n := 
by {
  sorry
}

end factorial_inequality_l20_20232


namespace speed_of_current_l20_20715

theorem speed_of_current (v_w v_c : ℝ) (h_downstream : 125 = (v_w + v_c) * 10)
                         (h_upstream : 60 = (v_w - v_c) * 10) :
  v_c = 3.25 :=
by {
  sorry
}

end speed_of_current_l20_20715


namespace integers_in_range_eq_l20_20113

theorem integers_in_range_eq :
  {i : ℤ | i > -2 ∧ i ≤ 3} = {-1, 0, 1, 2, 3} :=
by
  sorry

end integers_in_range_eq_l20_20113


namespace find_pairs_l20_20176

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ n : ℕ, (n > 0) ∧ (a = n ∧ b = n) ∨ (a = n ∧ b = 1)) ↔ 
  (a^3 ∣ b^2) ∧ ((b - 1) ∣ (a - 1)) :=
by {
  sorry
}

end find_pairs_l20_20176


namespace reading_rate_l20_20379

-- Definitions based on conditions
def one_way_trip_time : ℕ := 4
def round_trip_time : ℕ := 2 * one_way_trip_time
def read_book_time : ℕ := 2 * round_trip_time
def book_pages : ℕ := 4000

-- The theorem to prove Juan's reading rate is 250 pages per hour.
theorem reading_rate : book_pages / read_book_time = 250 := by
  sorry

end reading_rate_l20_20379


namespace sum_angles_of_two_triangles_l20_20625

theorem sum_angles_of_two_triangles (a1 a3 a5 a2 a4 a6 : ℝ) 
  (hABC : a1 + a3 + a5 = 180) (hDEF : a2 + a4 + a6 = 180) : 
  a1 + a2 + a3 + a4 + a5 + a6 = 360 :=
by
  sorry

end sum_angles_of_two_triangles_l20_20625


namespace find_multiplier_value_l20_20144

def number : ℤ := 18
def increase : ℤ := 198

theorem find_multiplier_value (x : ℤ) (h : number * x = number + increase) : x = 12 :=
by
  sorry

end find_multiplier_value_l20_20144


namespace fraction_division_l20_20675

theorem fraction_division (a b c d : ℚ) (h : b ≠ 0 ∧ d ≠ 0) : (a / b) / c = a / (b * c) := sorry

example : (3 / 7) / 4 = 3 / 28 := by
  apply fraction_division
  exact ⟨by norm_num, by norm_num⟩

end fraction_division_l20_20675


namespace percentage_both_correct_l20_20321

variable (A B : Type) 

noncomputable def percentage_of_test_takers_correct_first : ℝ := 0.85
noncomputable def percentage_of_test_takers_correct_second : ℝ := 0.70
noncomputable def percentage_of_test_takers_neither_correct : ℝ := 0.05

theorem percentage_both_correct :
  percentage_of_test_takers_correct_first + 
  percentage_of_test_takers_correct_second - 
  (1 - percentage_of_test_takers_neither_correct) = 0.60 := by
  sorry

end percentage_both_correct_l20_20321


namespace smallest_yummy_integer_l20_20497

theorem smallest_yummy_integer :
  ∃ (n A : ℤ), 4046 = n * (2 * A + n - 1) ∧ A ≥ 0 ∧ (∀ m, 4046 = m * (2 * A + m - 1) ∧ m ≥ 0 → A ≤ 1011) :=
sorry

end smallest_yummy_integer_l20_20497


namespace train_passes_bridge_in_52_seconds_l20_20130

def length_of_train : ℕ := 510
def speed_of_train_kmh : ℕ := 45
def length_of_bridge : ℕ := 140
def total_distance := length_of_train + length_of_bridge
def speed_of_train_ms := speed_of_train_kmh * 1000 / 3600
def time_to_pass_bridge := total_distance / speed_of_train_ms

theorem train_passes_bridge_in_52_seconds :
  time_to_pass_bridge = 52 := sorry

end train_passes_bridge_in_52_seconds_l20_20130


namespace fred_green_balloons_l20_20744

theorem fred_green_balloons (initial : ℕ) (given : ℕ) (final : ℕ) (h1 : initial = 709) (h2 : given = 221) (h3 : final = initial - given) : final = 488 :=
by
  sorry

end fred_green_balloons_l20_20744


namespace amount_after_a_year_l20_20654

def initial_amount : ℝ := 90
def interest_rate : ℝ := 0.10

theorem amount_after_a_year : initial_amount * (1 + interest_rate) = 99 := 
by
  -- Here 'sorry' indicates that the proof is not provided.
  sorry

end amount_after_a_year_l20_20654


namespace squirrel_acorns_l20_20992

theorem squirrel_acorns :
  ∀ (total_acorns : ℕ)
    (first_month_percent second_month_percent third_month_percent : ℝ)
    (first_month_consumed second_month_consumed third_month_consumed : ℝ),
    total_acorns = 500 →
    first_month_percent = 0.40 →
    second_month_percent = 0.30 →
    third_month_percent = 0.30 →
    first_month_consumed = 0.20 →
    second_month_consumed = 0.25 →
    third_month_consumed = 0.15 →
    let first_month_acorns := total_acorns * first_month_percent
    let second_month_acorns := total_acorns * second_month_percent
    let third_month_acorns := total_acorns * third_month_percent
    let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
    let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
    let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
    remaining_first_month + remaining_second_month + remaining_third_month = 400 := 
by
  intros 
    total_acorns
    first_month_percent second_month_percent third_month_percent
    first_month_consumed second_month_consumed third_month_consumed
    h_total
    h_first_percent
    h_second_percent
    h_third_percent
    h_first_consumed
    h_second_consumed
    h_third_consumed
  let first_month_acorns := total_acorns * first_month_percent
  let second_month_acorns := total_acorns * second_month_percent
  let third_month_acorns := total_acorns * third_month_percent
  let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
  let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
  let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
  sorry

end squirrel_acorns_l20_20992


namespace total_birds_caught_l20_20138

theorem total_birds_caught 
  (day_birds : ℕ) 
  (night_birds : ℕ)
  (h1 : day_birds = 8) 
  (h2 : night_birds = 2 * day_birds) 
  : day_birds + night_birds = 24 := 
by 
  sorry

end total_birds_caught_l20_20138


namespace problem_statement_l20_20220

def f (x : ℤ) : ℤ := 2 * x ^ 2 + 3 * x - 1

theorem problem_statement : f (f 3) = 1429 := by
  sorry

end problem_statement_l20_20220


namespace find_avg_speed_l20_20141

variables (v t : ℝ)

noncomputable def avg_speed_cond := 
  (v + Real.sqrt 15) * (t - Real.pi / 4) = v * t

theorem find_avg_speed (h : avg_speed_cond v t) : v = Real.sqrt 15 :=
by
  sorry

end find_avg_speed_l20_20141


namespace sum_of_fractions_and_decimal_l20_20025

theorem sum_of_fractions_and_decimal :
  (3 / 10) + (5 / 100) + (7 / 1000) + 0.001 = 0.358 :=
by
  sorry

end sum_of_fractions_and_decimal_l20_20025


namespace real_yield_correct_l20_20825

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l20_20825


namespace maximum_value_f_zeros_l20_20315

noncomputable def f (x : ℝ) (k : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if 1 < x then k * x + 1
  else 0

theorem maximum_value_f_zeros (k : ℝ) (x1 x2 : ℝ) :
  0 < k ∧ ∀ x, f x k = 0 ↔ x = x1 ∨ x = x2 → x1 ≠ x2 →
  x1 > 0 → x2 > 0 → -1 < k ∧ k < 0 →
  (x1 = -1 / k) ∧ (x2 = 1 / (1 + Real.sqrt (1 + k))) →
  ∃ y, (1 / x1) + (1 / x2) = y ∧ y = 9 / 4 := sorry

end maximum_value_f_zeros_l20_20315


namespace find_positive_integers_with_divisors_and_sum_l20_20439

theorem find_positive_integers_with_divisors_and_sum (n : ℕ) :
  (∃ d1 d2 d3 d4 d5 d6 : ℕ,
    (n ≠ 0) ∧ (n ≠ 1) ∧ 
    n = d1 * d2 * d3 * d4 * d5 * d6 ∧
    d1 ≠ 1 ∧ d2 ≠ 1 ∧ d3 ≠ 1 ∧ d4 ≠ 1 ∧ d5 ≠ 1 ∧ d6 ≠ 1 ∧
    (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ (d1 ≠ d6) ∧
    (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ (d2 ≠ d6) ∧
    (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ (d3 ≠ d6) ∧
    (d4 ≠ d5) ∧ (d4 ≠ d6) ∧
    (d5 ≠ d6) ∧
    d1 + d2 + d3 + d4 + d5 + d6 = 14133
  ) -> 
  (n = 16136 ∨ n = 26666) :=
sorry

end find_positive_integers_with_divisors_and_sum_l20_20439


namespace boiling_point_of_water_l20_20256

theorem boiling_point_of_water :
  (boiling_point_F : ℝ) = 212 →
  (boiling_point_C : ℝ) = (5 / 9) * (boiling_point_F - 32) →
  boiling_point_C = 100 :=
by
  intro h1 h2
  sorry

end boiling_point_of_water_l20_20256


namespace compare_expressions_l20_20588

theorem compare_expressions (x y : ℝ) (h1: x * y > 0) (h2: x ≠ y) : 
  x^4 + 6 * x^2 * y^2 + y^4 > 4 * x * y * (x^2 + y^2) :=
by
  sorry

end compare_expressions_l20_20588


namespace solve_for_x_l20_20460

theorem solve_for_x (x y : ℚ) (h1 : 2 * x - 3 * y = 15) (h2 : x + 2 * y = 8) : x = 54 / 7 :=
sorry

end solve_for_x_l20_20460


namespace painters_complete_three_rooms_in_three_hours_l20_20323

theorem painters_complete_three_rooms_in_three_hours :
  ∃ P, (∀ (P : ℕ), (P * 3) = 3) ∧ (9 * 9 = 27) → P = 3 := by
  sorry

end painters_complete_three_rooms_in_three_hours_l20_20323


namespace probability_of_A_given_B_l20_20997

def medical_teams : Type := {A, B, C, D}
def countries : Type := {Country1, Country2, Country3, Country4}

def EventA (assignment : medical_teams → countries) : Prop :=
  Function.Injective assignment

def EventB (assignment : medical_teams → countries) : Prop :=
  ∃ c : countries, ∀ t ≠ medical_teams.A, assignment t ≠ c

theorem probability_of_A_given_B : 
  P(EventA | EventB) = 2 / 9 :=
by
  sorry

end probability_of_A_given_B_l20_20997


namespace x_varies_as_sin_squared_l20_20344

variable {k j z : ℝ}
variable (x y : ℝ)

-- condition: x is proportional to y^2
def proportional_xy_square (x y : ℝ) (k : ℝ) : Prop :=
  x = k * y ^ 2

-- condition: y is proportional to sin(z)
def proportional_y_sin (y : ℝ) (j z : ℝ) : Prop :=
  y = j * Real.sin z

-- statement to prove: x is proportional to (sin(z))^2
theorem x_varies_as_sin_squared (k j z : ℝ) (x y : ℝ)
  (h1 : proportional_xy_square x y k)
  (h2 : proportional_y_sin y j z) :
  ∃ m, x = m * (Real.sin z) ^ 2 :=
by
  sorry

end x_varies_as_sin_squared_l20_20344


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20830

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l20_20830


namespace megan_initial_strawberry_jelly_beans_l20_20092

variables (s g : ℕ)

theorem megan_initial_strawberry_jelly_beans :
  (s = 3 * g) ∧ (s - 15 = 4 * (g - 15)) → s = 135 :=
by
  sorry

end megan_initial_strawberry_jelly_beans_l20_20092


namespace determine_f_peak_tourism_season_l20_20471

noncomputable def f (n : ℕ) : ℝ := 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300

theorem determine_f :
  (∀ n : ℕ, f n = 200 * Real.cos ((Real.pi / 6) * n + 2 * Real.pi / 3) + 300) ∧
  (f 8 - f 2 = 400) ∧
  (f 2 = 100) :=
sorry

theorem peak_tourism_season (n : ℤ) :
  (6 ≤ n ∧ n ≤ 10) ↔ (200 * Real.cos (((Real.pi / 6) * n) + 2 * Real.pi / 3) + 300 >= 400) :=
sorry

end determine_f_peak_tourism_season_l20_20471


namespace average_speed_l20_20721

section
def flat_sand_speed : ℕ := 60
def downhill_slope_speed : ℕ := flat_sand_speed + 12
def uphill_slope_speed : ℕ := flat_sand_speed - 18

/-- Conner's average speed on flat, downhill, and uphill slopes, each of which he spends one-third of his time traveling on, is 58 miles per hour -/
theorem average_speed : (flat_sand_speed + downhill_slope_speed + uphill_slope_speed) / 3 = 58 := by
  sorry

end

end average_speed_l20_20721


namespace extreme_point_inequality_l20_20598

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x - a / x - 2 * Real.log x

theorem extreme_point_inequality (x₁ x₂ a : ℝ) (h1 : x₁ < x₂) (h2 : f x₁ a = 0) (h3 : f x₂ a = 0) 
(h_a_range : 0 < a) (h_a_lt_1 : a < 1) :
  f x₂ a < x₂ - 1 :=
sorry

end extreme_point_inequality_l20_20598


namespace probability_two_primes_1_to_30_l20_20957

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23 ∨ n = 29

def count_combinations (n k : ℕ) : ℕ := nat.choose n k

theorem probability_two_primes_1_to_30 :
  let total_combinations := count_combinations 30 2 in
  let prime_combinations := count_combinations 10 2 in
  total_combinations = 435 ∧ 
  prime_combinations = 45 ∧ 
  (prime_combinations : ℚ) / total_combinations = 15 / 145 :=
by { sorry }

end probability_two_primes_1_to_30_l20_20957


namespace range_of_a_l20_20868

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
by
  sorry

end range_of_a_l20_20868


namespace board_rook_placement_l20_20996

-- Define the color function for the board
def color (n i j : ℕ) : ℕ :=
  min (i + j - 1) (2 * n - i - j + 1)

-- Conditions: It is possible to place n rooks such that no two attack each other and 
-- no two rooks stand on cells of the same color
def non_attacking_rooks (n : ℕ) (rooks : Fin n → Fin n) : Prop :=
  ∀ i j : Fin n, i ≠ j → rooks i ≠ rooks j ∧ color n i.val (rooks i).val ≠ color n j.val (rooks j).val

-- Main theorem to be proven
theorem board_rook_placement (n : ℕ) :
  (∃ rooks : Fin n → Fin n, non_attacking_rooks n rooks) →
  n % 4 = 0 ∨ n % 4 = 1 :=
by
  intros h
  sorry

end board_rook_placement_l20_20996


namespace length_of_uncovered_side_l20_20540

variables (L W : ℝ)

-- Conditions
def area_eq_680 := (L * W = 680)
def fence_eq_178 := (2 * W + L = 178)

-- Theorem statement to prove the length of the uncovered side
theorem length_of_uncovered_side (h1 : area_eq_680 L W) (h2 : fence_eq_178 L W) : L = 170 := 
sorry

end length_of_uncovered_side_l20_20540


namespace election_problem_l20_20208

theorem election_problem :
  ∃ (n : ℕ), n = (10 * 9) * Nat.choose 8 3 :=
  by
  use 5040
  sorry

end election_problem_l20_20208


namespace prime_sum_and_difference_l20_20876

theorem prime_sum_and_difference (m n p : ℕ) (hmprime : Nat.Prime m) (hnprime : Nat.Prime n) (hpprime: Nat.Prime p)
  (h1: m > n)
  (h2: n > p)
  (h3 : m + n + p = 74) 
  (h4 : m - n - p = 44) : 
  m = 59 ∧ n = 13 ∧ p = 2 :=
by
  sorry

end prime_sum_and_difference_l20_20876


namespace hillary_descending_rate_l20_20316

def baseCampDistance : ℕ := 4700
def hillaryClimbingRate : ℕ := 800
def eddyClimbingRate : ℕ := 500
def hillaryStopShort : ℕ := 700
def departTime : ℕ := 6 -- time is represented in hours from midnight
def passTime : ℕ := 12 -- time is represented in hours from midnight

theorem hillary_descending_rate :
  ∃ r : ℕ, r = 1000 := by
  sorry

end hillary_descending_rate_l20_20316


namespace robotics_club_non_participants_l20_20627

theorem robotics_club_non_participants (club_students electronics_students programming_students both_students : ℕ) 
  (h1 : club_students = 80) 
  (h2 : electronics_students = 45) 
  (h3 : programming_students = 50) 
  (h4 : both_students = 30) : 
  club_students - (electronics_students - both_students + programming_students - both_students + both_students) = 15 :=
by
  -- The proof would be here
  sorry

end robotics_club_non_participants_l20_20627


namespace canoes_rented_more_than_kayaks_l20_20259

-- Defining the constants
def canoe_cost : ℕ := 11
def kayak_cost : ℕ := 16
def total_revenue : ℕ := 460
def canoe_ratio : ℕ := 4
def kayak_ratio : ℕ := 3

-- Main statement to prove
theorem canoes_rented_more_than_kayaks :
  ∃ (C K : ℕ), canoe_cost * C + kayak_cost * K = total_revenue ∧ (canoe_ratio * K = kayak_ratio * C) ∧ (C - K = 5) :=
by
  have h1 : canoe_cost = 11 := rfl
  have h2 : kayak_cost = 16 := rfl
  have h3 : total_revenue = 460 := rfl
  have h4 : canoe_ratio = 4 := rfl
  have h5 : kayak_ratio = 3 := rfl
  sorry

end canoes_rented_more_than_kayaks_l20_20259


namespace area_of_plot_l20_20920

def central_square_area : ℕ := 64

def common_perimeter : ℕ := 32

-- This statement formalizes the proof problem: "The area of Mrs. Lígia's plot is 256 m² given the provided conditions."
theorem area_of_plot (a b : ℕ) 
  (h1 : a * a = central_square_area)
  (h2 : b = a) 
  (h3 : 4 * a = common_perimeter)  
  (h4 : ∀ (x y : ℕ), x + y = 16)
  (h5 : ∀ (x : ℕ), x + a = 16) 
  : a * 16 = 256 :=
sorry

end area_of_plot_l20_20920


namespace range_of_a_l20_20568

variable (a : ℝ) (x : ℝ)

theorem range_of_a
  (h1 : 2 * x < 3 * (x - 3) + 1)
  (h2 : (3 * x + 2) / 4 > x + a) :
  -11 / 4 ≤ a ∧ a < -5 / 2 :=
sorry

end range_of_a_l20_20568


namespace total_bills_is_126_l20_20695

noncomputable def F : ℕ := 84  -- number of 5-dollar bills
noncomputable def T : ℕ := (840 - 5 * F) / 10  -- derive T based on the total value and F
noncomputable def total_bills : ℕ := F + T

theorem total_bills_is_126 : total_bills = 126 :=
by
  -- Placeholder for the proof
  sorry

end total_bills_is_126_l20_20695


namespace sum_first_10_terms_l20_20059

def a_n (n : ℕ) : ℤ := (-1)^n * (3 * n - 2)

theorem sum_first_10_terms :
  (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) +
  (a_n 6) + (a_n 7) + (a_n 8) + (a_n 9) + (a_n 10) = 15 :=
by
  sorry

end sum_first_10_terms_l20_20059


namespace candidate_failed_by_25_marks_l20_20694

-- Define the given conditions
def maximum_marks : ℝ := 127.27
def passing_percentage : ℝ := 0.55
def marks_secured : ℝ := 45

-- Define the minimum passing marks
def minimum_passing_marks : ℝ := passing_percentage * maximum_marks

-- Define the number of failing marks the candidate missed
def failing_marks : ℝ := minimum_passing_marks - marks_secured

-- Define the main theorem to prove the candidate failed by 25 marks
theorem candidate_failed_by_25_marks :
  failing_marks = 25 := 
by
  sorry

end candidate_failed_by_25_marks_l20_20694


namespace car_total_travel_time_l20_20228

-- Define the given conditions
def travel_time_ngapara_zipra : ℝ := 60
def travel_time_ningi_zipra : ℝ := 0.8 * travel_time_ngapara_zipra
def speed_limit_zone_fraction : ℝ := 0.25
def speed_reduction_factor : ℝ := 0.5
def travel_time_zipra_varnasi : ℝ := 0.75 * travel_time_ningi_zipra

-- Total adjusted travel time from Ningi to Zipra including speed limit delay
def adjusted_travel_time_ningi_zipra : ℝ :=
  let delayed_time := speed_limit_zone_fraction * travel_time_ningi_zipra * (2 - speed_reduction_factor)
  travel_time_ningi_zipra + delayed_time

-- Total travel time in the day
def total_travel_time : ℝ :=
  travel_time_ngapara_zipra + adjusted_travel_time_ningi_zipra + travel_time_zipra_varnasi

-- Proposition to prove
theorem car_total_travel_time : total_travel_time = 156 :=
by
  -- We skip the proof for now
  sorry

end car_total_travel_time_l20_20228


namespace M_eq_N_l20_20944

-- Define the sets M and N
def M : Set ℤ := {u | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r}

-- Prove that M equals N
theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l20_20944


namespace area_reachable_points_is_40pi_l20_20070

def point : Type := (ℝ × ℝ)

variable (A B C: point)
variable (side_length : ℝ)
variable (reachable_radius : ℝ)
variable (extra_radius : ℝ)

-- Define an equilateral triangle
def is_equilateral_triangle (A B C : point) (side_length : ℝ) : Prop :=
  dist A B = side_length ∧ dist B C = side_length ∧ dist C A = side_length

-- Define distance between two points
def dist (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define reachability condition
def is_reachable (X : point) (A B C : point) (reachable_radius : ℝ) : Prop :=
  dist A X ≤ reachable_radius ∧ ¬(∃ t ∈ Icc 0 1, (X = (t * B.1 + (1 - t) * C.1, t * B.2 + (1 - t) * C.2)))

-- Define the set of reachable points
def reachable_set (A B C : point) (reachable_radius : ℝ) : set point :=
  {X | is_reachable X A B C reachable_radius}

-- Theorem: Area of the set of reachable points
noncomputable def area_of_set (s : set point) : ℝ := sorry

noncomputable def area_of_reachable_set (A B C : point) (reachable_radius : ℝ) (extra_radius : ℝ) : ℝ := 
  let main_circle_area := π * reachable_radius^2 / 2 in
  let side_circle_area := 2 * (π * extra_radius^2) in
  main_circle_area + side_circle_area

theorem area_reachable_points_is_40pi (A B C : point) (side_length : ℝ) (reachable_radius : ℝ) (extra_radius : ℝ)
  (h_triangle : is_equilateral_triangle A B C side_length)
  (h_side_length : side_length = 6)
  (h_reachable_radius : reachable_radius = 8)
  (h_extra_radius : extra_radius = 2) :
  area_of_reachable_set A B C reachable_radius extra_radius = 40 * π := sorry

end area_reachable_points_is_40pi_l20_20070


namespace possible_degrees_of_remainder_l20_20964

theorem possible_degrees_of_remainder (p : Polynomial ℝ) (h : p = 3 * X^3 - 5 * X^2 + 2 * X - 8) :
  ∃ d : Finset ℕ, d = {0, 1, 2} :=
by
  sorry

end possible_degrees_of_remainder_l20_20964


namespace small_pizza_slices_l20_20153

-- Definitions based on conditions
def large_pizza_slices : ℕ := 16
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Statement to prove
theorem small_pizza_slices (S : ℕ) (H : num_large_pizzas * large_pizza_slices + num_small_pizzas * S = total_slices_eaten) : S = 8 :=
by
  sorry

end small_pizza_slices_l20_20153


namespace cuboid_first_dimension_l20_20011

theorem cuboid_first_dimension (x : ℕ)
  (h₁ : ∃ n : ℕ, n = 24) 
  (h₂ : ∃ a b c d e f g : ℕ, x = a ∧ 9 = b ∧ 12 = c ∧ a * b * c = d * e * f ∧ g = Nat.gcd b c ∧ f = (g^3) ∧ e = (n * f) ∧ d = 648) : 
  x = 6 :=
by
  sorry

end cuboid_first_dimension_l20_20011


namespace motorboat_speeds_l20_20815

theorem motorboat_speeds (v a x : ℝ) (d : ℝ)
  (h1 : ∀ t1 t2 t1' t2', 
        t1 = d / (v - a) ∧ t1' = d / (v + x - a) ∧ 
        t2 = d / (v + a) ∧ t2' = d / (v + a - x) ∧ 
        (t1 - t1' = t2' - t2)) 
        : x = 2 * a := 
sorry

end motorboat_speeds_l20_20815


namespace blue_chip_value_l20_20768

noncomputable def yellow_chip_value := 2
noncomputable def green_chip_value := 5
noncomputable def total_product_value := 16000
noncomputable def num_yellow_chips := 4

def blue_chip_points (b n : ℕ) :=
  yellow_chip_value ^ num_yellow_chips * b ^ n * green_chip_value ^ n = total_product_value

theorem blue_chip_value (b : ℕ) (n : ℕ) (h : blue_chip_points b n) (hn : b^n = 8) : b = 8 :=
by
  have h1 : ∀ k : ℕ, k ^ n = 8 → k = 8 ∧ n = 3 := sorry
  exact (h1 b hn).1

end blue_chip_value_l20_20768


namespace pyramid_max_volume_height_l20_20591

-- Define the conditions and the theorem
theorem pyramid_max_volume_height
  (a h V : ℝ)
  (SA : ℝ := 2 * Real.sqrt 3)
  (h_eq : h = Real.sqrt (SA^2 - (Real.sqrt 2 * a / 2)^2))
  (V_eq : V = (1 / 3) * a^2 * h)
  (derivative_at_max : ∀ a, (48 * a^3 - 3 * a^5 = 0) → (a = 0 ∨ a = 4))
  (max_a_value : a = 4):
  h = 2 :=
by
  sorry

end pyramid_max_volume_height_l20_20591


namespace hadassah_additional_paintings_l20_20888

noncomputable def hadassah_initial_paintings : ℕ := 12
noncomputable def hadassah_initial_hours : ℕ := 6
noncomputable def hadassah_total_hours : ℕ := 16

theorem hadassah_additional_paintings 
  (initial_paintings : ℕ)
  (initial_hours : ℕ)
  (total_hours : ℕ) :
  initial_paintings = hadassah_initial_paintings →
  initial_hours = hadassah_initial_hours →
  total_hours = hadassah_total_hours →
  let additional_hours := total_hours - initial_hours
  let painting_rate := initial_paintings / initial_hours
  let additional_paintings := painting_rate * additional_hours
  additional_paintings = 20 :=
by
  sorry

end hadassah_additional_paintings_l20_20888


namespace son_age_is_10_l20_20983

-- Define the conditions
variables (S F : ℕ)
axiom condition1 : F = S + 30
axiom condition2 : F + 5 = 3 * (S + 5)

-- State the theorem to prove the son's age
theorem son_age_is_10 : S = 10 :=
by
  sorry

end son_age_is_10_l20_20983


namespace even_function_has_specific_a_l20_20466

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ :=
  a * x ^ 2 + (2 * a ^ 2 - a) * x + 1

-- State the proof problem
theorem even_function_has_specific_a (a : ℝ) :
  (∀ x : ℝ, f a (-x) = f a x) → a = 1 / 2 :=
by
  intros h
  sorry

end even_function_has_specific_a_l20_20466


namespace next_elements_l20_20289

-- Define the conditions and the question
def next_elements_in_sequence (n : ℕ) : String :=
  match n with
  | 1 => "О"  -- "Один"
  | 2 => "Д"  -- "Два"
  | 3 => "Т"  -- "Три"
  | 4 => "Ч"  -- "Четыре"
  | 5 => "П"  -- "Пять"
  | 6 => "Ш"  -- "Шесть"
  | 7 => "С"  -- "Семь"
  | 8 => "В"  -- "Восемь"
  | _ => "?"

theorem next_elements (n : ℕ) :
  next_elements_in_sequence 7 = "С" ∧ next_elements_in_sequence 8 = "В" := by
  sorry

end next_elements_l20_20289


namespace janet_income_difference_l20_20335

def janet_current_job_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def janet_freelance_income (hours_per_week : ℕ) (weeks_per_month : ℕ) (hourly_rate : ℝ) : ℝ :=
  hours_per_week * weeks_per_month * hourly_rate

def extra_fica_taxes (weekly_tax : ℝ) (weeks_per_month : ℕ) : ℝ :=
  weekly_tax * weeks_per_month

def healthcare_premiums (monthly_premium : ℝ) : ℝ :=
  monthly_premium

def janet_net_freelance_income (freelance_income : ℝ) (additional_costs : ℝ) : ℝ :=
  freelance_income - additional_costs

theorem janet_income_difference
  (hours_per_week : ℕ)
  (weeks_per_month : ℕ)
  (current_hourly_rate : ℝ)
  (freelance_hourly_rate : ℝ)
  (weekly_tax : ℝ)
  (monthly_premium : ℝ)
  (H_hours : hours_per_week = 40)
  (H_weeks : weeks_per_month = 4)
  (H_current_rate : current_hourly_rate = 30)
  (H_freelance_rate : freelance_hourly_rate = 40)
  (H_weekly_tax : weekly_tax = 25)
  (H_monthly_premium : monthly_premium = 400) :
  janet_net_freelance_income (janet_freelance_income 40 4 40) (extra_fica_taxes 25 4 + healthcare_premiums 400) 
  - janet_current_job_income 40 4 30 = 1100 := 
  by 
    sorry

end janet_income_difference_l20_20335


namespace fill_tank_time_l20_20356

theorem fill_tank_time (t_A t_B : ℕ) (hA : t_A = 20) (hB : t_B = t_A / 4) :
  t_B = 4 := by
  sorry

end fill_tank_time_l20_20356


namespace calculate_sum_l20_20026

theorem calculate_sum : (2 / 20) + (3 / 50 * 5 / 100) + (4 / 1000) + (6 / 10000) = 0.1076 := 
by
  sorry

end calculate_sum_l20_20026


namespace fluorescent_tubes_count_l20_20949

theorem fluorescent_tubes_count 
  (x y : ℕ)
  (h1 : x + y = 13)
  (h2 : x / 3 + y / 2 = 5) : x = 9 :=
by
  sorry

end fluorescent_tubes_count_l20_20949


namespace operation_5_7_eq_35_l20_20727

noncomputable def operation (x y : ℝ) : ℝ := sorry

axiom condition1 :
  ∀ (x y : ℝ), (x * y > 0) → (operation (x * y) y = x * (operation y y))

axiom condition2 :
  ∀ (x : ℝ), (x > 0) → (operation (operation x 1) x = operation x 1)

axiom condition3 :
  (operation 1 1 = 2)

theorem operation_5_7_eq_35 : operation 5 7 = 35 :=
by
  sorry

end operation_5_7_eq_35_l20_20727


namespace bridget_apples_l20_20549

theorem bridget_apples :
  ∃ x : ℕ, (x - x / 3 - 4) = 6 :=
by
  sorry

end bridget_apples_l20_20549


namespace simplify_fraction_l20_20653

theorem simplify_fraction :
  (3 * (Real.sqrt 3 + Real.sqrt 8)) / (2 * Real.sqrt (3 + Real.sqrt 5)) = 
  (297 - 99 * Real.sqrt 5 + 108 * Real.sqrt 6 - 36 * Real.sqrt 30) / 16 := by
  sorry

end simplify_fraction_l20_20653


namespace min_value_x_plus_3y_min_value_xy_l20_20877

variable {x y : ℝ}

theorem min_value_x_plus_3y (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x + 3 * y ≥ 16 :=
sorry

theorem min_value_xy (hx : x > 0) (hy : y > 0) (h : 1/x + 3/y = 1) : x * y ≥ 12 :=
sorry

end min_value_x_plus_3y_min_value_xy_l20_20877


namespace rowan_distance_downstream_l20_20358

-- Conditions
def speed_still : ℝ := 9.75
def downstream_time : ℝ := 2
def upstream_time : ℝ := 4

-- Statement to prove
theorem rowan_distance_downstream : ∃ (d : ℝ) (c : ℝ), 
  d / (speed_still + c) = downstream_time ∧
  d / (speed_still - c) = upstream_time ∧
  d = 26 := by
    sorry

end rowan_distance_downstream_l20_20358


namespace find_asymptote_slope_l20_20660

theorem find_asymptote_slope (x y : ℝ) (h : (y^2) / 9 - (x^2) / 4 = 1) : y = 3 / 2 * x :=
sorry

end find_asymptote_slope_l20_20660


namespace ratio_of_spending_is_one_to_two_l20_20810

-- Definitions
def initial_amount : ℕ := 24
def doris_spent : ℕ := 6
def final_amount : ℕ := 15

-- Amount remaining after Doris spent
def remaining_after_doris : ℕ := initial_amount - doris_spent

-- Amount Martha spent
def martha_spent : ℕ := remaining_after_doris - final_amount

-- Ratio of the amounts spent
def ratio_martha_doris : ℕ × ℕ := (martha_spent, doris_spent)

-- Theorem to prove
theorem ratio_of_spending_is_one_to_two : ratio_martha_doris = (1, 2) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_spending_is_one_to_two_l20_20810


namespace complement_union_A_B_in_U_l20_20614

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l20_20614


namespace digits_sum_l20_20068

theorem digits_sum (P Q R : ℕ) (h1 : P < 10) (h2 : Q < 10) (h3 : R < 10)
  (h_eq : 100 * P + 10 * Q + R + 10 * Q + R = 1012) :
  P + Q + R = 20 :=
by {
  -- Implementation of the proof will go here
  sorry
}

end digits_sum_l20_20068


namespace least_multiple_of_seven_not_lucky_is_14_l20_20706

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n > 0 ∧ n % sum_of_digits n = 0

def is_multiple_of_seven_not_lucky (n : ℕ) : Prop :=
  n % 7 = 0 ∧ ¬ is_lucky_integer n

theorem least_multiple_of_seven_not_lucky_is_14 : 
  ∃ n : ℕ, is_multiple_of_seven_not_lucky n ∧ ∀ m, (is_multiple_of_seven_not_lucky m → n ≤ m) :=
⟨ 14, 
  by {
    -- Proof is provided here, but for now, we use "sorry"
    sorry
  }⟩

end least_multiple_of_seven_not_lucky_is_14_l20_20706


namespace cat_birds_total_l20_20135

def day_birds : ℕ := 8
def night_birds : ℕ := 2 * day_birds
def total_birds : ℕ := day_birds + night_birds

theorem cat_birds_total : total_birds = 24 :=
by
  -- proof goes here
  sorry

end cat_birds_total_l20_20135


namespace relationship_between_y_coordinates_l20_20578

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l20_20578


namespace probability_both_numbers_prime_l20_20954

open Finset

def primes_between_1_and_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_both_numbers_prime :
  (∃ (a b : ℕ), a ≠ b ∧ a ∈ primes_between_1_and_30 ∧ b ∈ primes_between_1_and_30 ∧
  (45 / 435) = (1 / 9)) :=
by
  have h_prime_count : ∃ (s : Finset ℕ), s.card = 10 ∧ s = primes_between_1_and_30 := sorry,
  have h_total_pairs : ∃ (n : ℕ), n = 435 := sorry,
  have h_prime_pairs : ∃ (m : ℕ), m = 45 := sorry,
  have h_fraction : (45 : ℚ) / 435 = (1 : ℚ) / 9 := sorry,
  exact ⟨45, 435, by simp [h_prime_pairs, h_total_pairs, h_fraction]⟩

end probability_both_numbers_prime_l20_20954


namespace perimeter_of_square_is_64_l20_20845

noncomputable def side_length_of_square (s : ℝ) :=
  let rect_height := s
  let rect_width := s / 4
  let perimeter_of_rectangle := 2 * (rect_height + rect_width)
  perimeter_of_rectangle = 40

theorem perimeter_of_square_is_64 (s : ℝ) (h1 : side_length_of_square s) : 4 * s = 64 :=
by
  sorry

end perimeter_of_square_is_64_l20_20845


namespace range_of_function_l20_20436

theorem range_of_function : 
  ∀ x ∈ set.Ioo (-1 : ℝ) 1,
    ∃ y : ℝ, 
      y = real.arcsin x + real.arccos x + real.arctan x - real.arctanh x :=
by
  sorry

end range_of_function_l20_20436


namespace arithmetic_sequence_50th_term_l20_20901

-- Define the arithmetic sequence parameters
def first_term : Int := 2
def common_difference : Int := 5

-- Define the formula to calculate the n-th term of the sequence
def nth_term (n : Nat) : Int :=
  first_term + (n - 1) * common_difference

-- Prove that the 50th term of the sequence is 247
theorem arithmetic_sequence_50th_term : nth_term 50 = 247 :=
  by
  -- Proof goes here
  sorry

end arithmetic_sequence_50th_term_l20_20901


namespace Ferris_break_length_l20_20285

noncomputable def Audrey_rate_per_hour := (1:ℝ) / 4
noncomputable def Ferris_rate_per_hour := (1:ℝ) / 3
noncomputable def total_completion_time := (2:ℝ)
noncomputable def number_of_breaks := (6:ℝ)
noncomputable def job_completion_audrey := total_completion_time * Audrey_rate_per_hour
noncomputable def job_completion_ferris := 1 - job_completion_audrey
noncomputable def working_time_ferris := job_completion_ferris / Ferris_rate_per_hour
noncomputable def total_break_time := total_completion_time - working_time_ferris
noncomputable def break_length := total_break_time / number_of_breaks

theorem Ferris_break_length :
  break_length = (5:ℝ) / 60 := 
sorry

end Ferris_break_length_l20_20285


namespace represent_1947_as_squares_any_integer_as_squares_l20_20792

theorem represent_1947_as_squares :
  ∃ (a b c : ℤ), 1947 = a * a - b * b - c * c :=
by
  use 488, 486, 1
  sorry

theorem any_integer_as_squares (n : ℤ) :
  ∃ (a b c d : ℤ), n = a * a + b * b + c * c + d * d :=
by
  sorry

end represent_1947_as_squares_any_integer_as_squares_l20_20792


namespace enclosed_polygons_l20_20989

theorem enclosed_polygons (n : ℕ) :
  (∃ α β : ℝ, (15 * β) = 360 ∧ β = 180 - α ∧ (15 * α) = 180 * (n - 2) / n) ↔ n = 15 :=
by sorry

end enclosed_polygons_l20_20989


namespace ratio_q_p_l20_20499

variable (p q : ℝ)
variable (hpq_pos : 0 < p ∧ 0 < q)
variable (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18)

theorem ratio_q_p (p q : ℝ) (hpq_pos : 0 < p ∧ 0 < q) 
    (hlog : Real.log p / Real.log 8 = Real.log q / Real.log 12 ∧ Real.log q / Real.log 12 = Real.log (p - q) / Real.log 18) :
    q / p = (Real.sqrt 5 - 1) / 2 :=
  sorry

end ratio_q_p_l20_20499


namespace B_necessary_not_sufficient_for_A_l20_20931

def A (x : ℝ) : Prop := 0 < x ∧ x < 5
def B (x : ℝ) : Prop := |x - 2| < 3

theorem B_necessary_not_sufficient_for_A (x : ℝ) :
  (A x → B x) ∧ (∃ x, B x ∧ ¬ A x) :=
by
  sorry

end B_necessary_not_sufficient_for_A_l20_20931


namespace trains_meeting_time_l20_20122

noncomputable def kmph_to_mps (speed : ℕ) : ℕ := speed * 1000 / 3600

noncomputable def time_to_meet (L1 L2 D S1 S2 : ℕ) : ℕ := 
  let S1_mps := kmph_to_mps S1
  let S2_mps := kmph_to_mps S2
  let relative_speed := S1_mps + S2_mps
  let total_distance := L1 + L2 + D
  total_distance / relative_speed

theorem trains_meeting_time : time_to_meet 210 120 160 74 92 = 10620 / 1000 :=
by
  sorry

end trains_meeting_time_l20_20122


namespace percentage_employees_at_picnic_l20_20206

theorem percentage_employees_at_picnic (total_employees men_attend men_percentage women_attend women_percentage : ℝ)
  (h1 : men_attend = 0.20 * (men_percentage * total_employees))
  (h2 : women_attend = 0.40 * ((1 - men_percentage) * total_employees))
  (h3 : men_percentage = 0.30)
  : ((men_attend + women_attend) / total_employees) * 100 = 34 := by
sorry

end percentage_employees_at_picnic_l20_20206


namespace answered_both_correctly_l20_20403

variable (A B : Prop)
variable (P_A P_B P_not_A_and_not_B P_A_and_B : ℝ)

axiom P_A_eq : P_A = 0.75
axiom P_B_eq : P_B = 0.35
axiom P_not_A_and_not_B_eq : P_not_A_and_not_B = 0.20

theorem answered_both_correctly (h1 : P_A = 0.75) (h2 : P_B = 0.35) (h3 : P_not_A_and_not_B = 0.20) : 
  P_A_and_B = 0.30 :=
by
  sorry

end answered_both_correctly_l20_20403


namespace real_root_if_and_only_if_l20_20433

theorem real_root_if_and_only_if (b : ℝ) : 
  (∃ x : ℝ, x^2 + b * x + 25 = 0) ↔ b ∈ Set.Iic (-10) ∪ Set.Ici 10 :=
by
  sorry

end real_root_if_and_only_if_l20_20433


namespace x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l20_20203

theorem x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero
  (x y : ℝ)
  (h1 : x + y = 2)
  (h2 : x * y = 4) : x^2 * y^3 + y^2 * x^3 = 0 := 
sorry

end x_squared_y_cubed_plus_y_squared_x_cubed_eq_zero_l20_20203


namespace relationship_y1_y2_y3_l20_20582

variable (y1 y2 y3 b : ℝ)
variable (h1 : y1 = 3 * (-3) - b)
variable (h2 : y2 = 3 * 1 - b)
variable (h3 : y3 = 3 * (-1) - b)

theorem relationship_y1_y2_y3 : y1 < y3 ∧ y3 < y2 := by
  sorry

end relationship_y1_y2_y3_l20_20582


namespace find_Y_l20_20305

theorem find_Y (Y : ℕ) 
  (h_top : 2 + 1 + Y + 3 = 6 + Y)
  (h_bottom : 4 + 3 + 1 + 5 = 13)
  (h_equal : 6 + Y = 13) : 
  Y = 7 := 
by
  sorry

end find_Y_l20_20305


namespace power_addition_l20_20022

theorem power_addition :
  (-2 : ℤ) ^ 2009 + (-2 : ℤ) ^ 2010 = 2 ^ 2009 :=
by
  sorry

end power_addition_l20_20022


namespace m_range_l20_20249

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    (x + m) / (x - 2) - 3 = (x - 1) / (2 - x) ∧ 
    x ≥ 0

theorem m_range (m : ℝ) : 
  range_of_m m ↔ m ≥ -5 ∧ m ≠ -3 := 
sorry

end m_range_l20_20249


namespace solve_for_C_and_D_l20_20872

theorem solve_for_C_and_D (C D : ℚ) (h1 : 2 * C + 3 * D + 4 = 31) (h2 : D = C + 2) :
  C = 21 / 5 ∧ D = 31 / 5 :=
by
  sorry

end solve_for_C_and_D_l20_20872


namespace S_ploughing_time_l20_20269

theorem S_ploughing_time (R S : ℝ) (hR_rate : R = 1 / 15) (h_combined_rate : R + S = 1 / 10) : S = 1 / 30 := sorry

end S_ploughing_time_l20_20269


namespace total_wet_surface_area_eq_l20_20129

-- Definitions based on given conditions
def length_cistern : ℝ := 10
def width_cistern : ℝ := 6
def height_water : ℝ := 1.35

-- Problem statement: Prove the total wet surface area is as calculated
theorem total_wet_surface_area_eq :
  let area_bottom : ℝ := length_cistern * width_cistern
  let area_longer_sides : ℝ := 2 * (length_cistern * height_water)
  let area_shorter_sides : ℝ := 2 * (width_cistern * height_water)
  let total_wet_surface_area : ℝ := area_bottom + area_longer_sides + area_shorter_sides
  total_wet_surface_area = 103.2 :=
by
  -- Since we do not need the proof, we use sorry here
  sorry

end total_wet_surface_area_eq_l20_20129


namespace fraction_of_second_year_students_l20_20764

-- Define the fractions of first-year and second-year students
variables (F S f s: ℝ)

-- Conditions
axiom h1 : F + S = 1
axiom h2 : f = (1 / 5) * F
axiom h3 : s = 4 * f
axiom h4 : S - s = 0.2

-- The theorem statement to prove the fraction of second-year students is 2 / 3
theorem fraction_of_second_year_students (F S f s: ℝ) 
    (h1: F + S = 1) 
    (h2: f = (1 / 5) * F) 
    (h3: s = 4 * f) 
    (h4: S - s = 0.2) : 
    S = 2 / 3 :=
by 
    sorry

end fraction_of_second_year_students_l20_20764


namespace sphere_has_circular_views_l20_20416

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end sphere_has_circular_views_l20_20416


namespace solve_ineqs_l20_20363

theorem solve_ineqs (a x : ℝ) (h1 : |x - 2 * a| ≤ 3) (h2 : 0 < x + a ∧ x + a ≤ 4) 
  (ha : a = 3) (hx : x = 1) : 
  (|x - 2 * a| ≤ 3) ∧ (0 < x + a ∧ x + a ≤ 4) :=
by
  sorry

end solve_ineqs_l20_20363


namespace div_by_self_condition_l20_20966

theorem div_by_self_condition (n : ℤ) (h : n^2 + 1 ∣ n) : n = 0 :=
by sorry

end div_by_self_condition_l20_20966


namespace xyz_value_l20_20747

-- Define the real numbers x, y, and z
variables (x y z : ℝ)

-- Condition 1
def condition1 := (x + y + z) * (x * y + x * z + y * z) = 49

-- Condition 2
def condition2 := x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 19

-- Main theorem statement
theorem xyz_value (h1 : condition1 x y z) (h2 : condition2 x y z) : x * y * z = 10 :=
sorry

end xyz_value_l20_20747


namespace rolling_green_probability_l20_20257

/-- A cube with 5 green faces and 1 yellow face. -/
structure ColoredCube :=
  (green_faces : ℕ)
  (yellow_face : ℕ)
  (total_faces : ℕ)

def example_cube : ColoredCube :=
  { green_faces := 5, yellow_face := 1, total_faces := 6 }

/-- The probability of rolling a green face on a given cube. -/
def probability_of_rolling_green (cube : ColoredCube) : ℚ :=
  cube.green_faces / cube.total_faces

theorem rolling_green_probability :
  probability_of_rolling_green example_cube = 5 / 6 :=
by simp [probability_of_rolling_green, example_cube]

end rolling_green_probability_l20_20257


namespace calculate_blue_candles_l20_20166

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l20_20166


namespace shape_with_circular_views_is_sphere_l20_20415

/-- Define the views of the cuboid, cylinder, cone, and sphere. -/
structure Views (shape : Type) :=
(front_view : Type)
(left_view : Type)
(top_view : Type)

def is_cuboid (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Rectangle

def is_cylinder (s : Views) : Prop :=
s.front_view = Rectangle ∧ s.left_view = Rectangle ∧ s.top_view = Circle

def is_cone (s : Views) : Prop :=
s.front_view = IsoscelesTriangle ∧ s.left_view = IsoscelesTriangle ∧ s.top_view = Circle

def is_sphere (s : Views) : Prop :=
s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle

/-- Proof problem: Prove that the only shape with circular views in all three perspectives (front, left, top) is the sphere. -/
theorem shape_with_circular_views_is_sphere :
  ∀ (s : Views), 
    (s.front_view = Circle ∧ s.left_view = Circle ∧ s.top_view = Circle) → 
    is_sphere s ∧ ¬ is_cuboid s ∧ ¬ is_cylinder s ∧ ¬ is_cone s :=
by
  intro s h
  sorry

end shape_with_circular_views_is_sphere_l20_20415


namespace integer_solutions_l20_20728

theorem integer_solutions (x : ℤ) : 
  (⌊(x : ℚ) / 2⌋ * ⌊(x : ℚ) / 3⌋ * ⌊(x : ℚ) / 4⌋ = x^2) ↔ (x = 0 ∨ x = 24) := 
sorry

end integer_solutions_l20_20728


namespace weekly_car_mileage_l20_20950

-- Definitions of the conditions
def dist_school := 2.5 
def dist_market := 2 
def school_days := 4
def school_trips_per_day := 2
def market_trips_per_week := 1

-- Proof statement
theorem weekly_car_mileage : 
  4 * 2 * (2.5 * 2) + (1 * (2 * 2)) = 44 :=
by
  -- The goal is to prove that 4 days of 2 round trips to school plus 1 round trip to market equals 44 miles
  sorry

end weekly_car_mileage_l20_20950


namespace prime_probability_l20_20955

theorem prime_probability (P : Finset ℕ) : 
  (P = {p | p ≤ 30 ∧ Nat.Prime p}).card = 10 → 
  (Finset.Icc 1 30).card = 30 →
  ((Finset.Icc 1 30).card).choose 2 = 435 →
  (P.card).choose 2 = 45 →
  45 / 435 = 1 / 29 :=
by
  intros hP hIcc30 hChoose30 hChoosePrime
  sorry

end prime_probability_l20_20955


namespace plane_stops_at_20_seconds_l20_20502

/-- The analytical expression of the function of the distance s the plane travels during taxiing 
after landing with respect to the time t is given by s = -1.5t^2 + 60t. 

Prove that the plane stops after taxiing for 20 seconds. -/

noncomputable def plane_distance (t : ℝ) : ℝ :=
  -1.5 * t^2 + 60 * t

theorem plane_stops_at_20_seconds :
  ∃ t : ℝ, t = 20 ∧ plane_distance t = plane_distance (20 : ℝ) :=
by
  sorry

end plane_stops_at_20_seconds_l20_20502


namespace unique_root_ln_eqn_l20_20570

/-- For what values of the parameter \(a\) does the equation
   \(\ln(x - 2a) - 3(x - 2a)^2 + 2a = 0\) have a unique root? -/
theorem unique_root_ln_eqn (a : ℝ) :
  ∃! x : ℝ, (Real.log (x - 2 * a) - 3 * (x - 2 * a) ^ 2 + 2 * a = 0) ↔
  a = (1 + Real.log 6) / 4 :=
sorry

end unique_root_ln_eqn_l20_20570


namespace locus_of_M_l20_20355

theorem locus_of_M (k : ℝ) (A B M : ℝ × ℝ) (hA : A.1 ≥ 0 ∧ A.2 = 0) (hB : B.2 ≥ 0 ∧ B.1 = 0) (h_sum : A.1 + B.2 = k) :
    ∃ (M : ℝ × ℝ), (M.1 - k / 2)^2 + (M.2 - k / 2)^2 = k^2 / 2 :=
by
  sorry

end locus_of_M_l20_20355


namespace selling_price_correct_l20_20405

-- Define the conditions
def purchase_price : ℝ := 12000
def repair_costs : ℝ := 5000
def transportation_charges : ℝ := 1000
def profit_percentage : ℝ := 0.50

-- Calculate total cost
def total_cost : ℝ := purchase_price + repair_costs + transportation_charges

-- Define the selling price and the proof goal
def selling_price : ℝ := total_cost + (profit_percentage * total_cost)

-- Prove that the selling price equals Rs 27000
theorem selling_price_correct : selling_price = 27000 := 
by 
  -- Proof is not required, so we use sorry
  sorry

end selling_price_correct_l20_20405


namespace total_days_correct_l20_20425

-- Defining the years and the conditions given.
def year_1999 := 1999
def year_2000 := 2000
def year_2001 := 2001
def year_2002 := 2002

-- Defining the leap year and regular year days
def days_in_regular_year := 365
def days_in_leap_year := 366

-- Noncomputable version to skip the proof
noncomputable def total_days_from_1999_to_2002 : ℕ :=
  3 * days_in_regular_year + days_in_leap_year

-- The theorem stating the problem, which we need to prove
theorem total_days_correct : total_days_from_1999_to_2002 = 1461 := by
  sorry

end total_days_correct_l20_20425


namespace xy_value_l20_20328

theorem xy_value (x y : ℝ) (h1 : (x + y) / 3 = 1.222222222222222) : x + y = 3.666666666666666 :=
by
  sorry

end xy_value_l20_20328


namespace sum_of_consecutive_odds_mod_16_l20_20262

theorem sum_of_consecutive_odds_mod_16 :
  (12001 + 12003 + 12005 + 12007 + 12009 + 12011 + 12013) % 16 = 1 :=
by
  sorry

end sum_of_consecutive_odds_mod_16_l20_20262


namespace students_trip_twice_l20_20230

-- Definitions of conditions
constant Students : Type
constant is_nonempty : Nonempty Students
constant total_students : Finset Students
constant trips : Finset (Finset Students)
constant num_students : total_students.card = 30
constant num_trips : trips.card = 16
constant trip_size : ∀ t ∈ trips, t.card = 8

-- The proof problem
theorem students_trip_twice :
  ∃ (s1 s2 : Students), s1 ≠ s2 ∧ ∃ t1 t2 ∈ trips, t1 ≠ t2 ∧ {s1, s2} ⊆ t1 ∧ {s1, s2} ⊆ t2 :=
sorry

end students_trip_twice_l20_20230


namespace sequence_ratio_proof_l20_20885

variable {a : ℕ → ℤ}

-- Sequence definition
axiom a₁ : a 1 = 3
axiom a_recurrence : ∀ n : ℕ, a (n + 1) = 4 * a n + 3

-- The theorem to be proved
theorem sequence_ratio_proof (n : ℕ) : (a (n + 1) + 1) / (a n + 1) = 4 := by
  sorry

end sequence_ratio_proof_l20_20885


namespace tiffany_reading_homework_pages_l20_20117

theorem tiffany_reading_homework_pages 
  (math_pages : ℕ)
  (problems_per_page : ℕ)
  (total_problems : ℕ)
  (reading_pages : ℕ)
  (H1 : math_pages = 6)
  (H2 : problems_per_page = 3)
  (H3 : total_problems = 30)
  (H4 : reading_pages = (total_problems - math_pages * problems_per_page) / problems_per_page) 
  : reading_pages = 4 := 
sorry

end tiffany_reading_homework_pages_l20_20117


namespace calculate_area_of_pentagon_l20_20161

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℝ :=
  let triangle_area := (1/2 : ℝ) * b * a
  let trapezoid_area := (1/2 : ℝ) * (c + e) * d
  triangle_area + trapezoid_area

theorem calculate_area_of_pentagon : area_of_pentagon 18 25 28 30 25 = 1020 :=
sorry

end calculate_area_of_pentagon_l20_20161


namespace solution_exists_l20_20041

theorem solution_exists (a b : ℝ) (h1 : 4 * a + b = 60) (h2 : 6 * a - b = 30) :
  a = 9 ∧ b = 24 :=
by
  sorry

end solution_exists_l20_20041


namespace sum_of_first_2018_terms_is_3_over_2_l20_20802

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (π / 6))

def a (n : ℕ) : ℝ := f (n * π / 6)

def sum_first_2018_terms : ℝ :=
  ∑ i in Finset.range 2018, a i

theorem sum_of_first_2018_terms_is_3_over_2 :
  sum_first_2018_terms = 3 / 2 :=
by
  sorry

end sum_of_first_2018_terms_is_3_over_2_l20_20802


namespace racing_championship_guarantee_l20_20766

/-- 
In a racing championship consisting of five races, the points awarded are as follows: 
6 points for first place, 4 points for second place, and 2 points for third place, with no ties possible. 
What is the smallest number of points a racer must accumulate in these five races to be guaranteed of having more points than any other racer? 
-/
theorem racing_championship_guarantee :
  ∀ (points_1st : ℕ) (points_2nd : ℕ) (points_3rd : ℕ) (races : ℕ),
  points_1st = 6 → points_2nd = 4 → points_3rd = 2 → 
  races = 5 →
  (∃ min_points : ℕ, min_points = 26 ∧ 
    ∀ (possible_points : ℕ), possible_points ≠ min_points → 
    (possible_points < min_points)) :=
by
  sorry

end racing_championship_guarantee_l20_20766


namespace complement_union_A_B_eq_neg2_0_l20_20606

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_eq_neg2_0 :
  U \ (A ∪ B) = {-2, 0} := by
  sorry

end complement_union_A_B_eq_neg2_0_l20_20606


namespace max_value_of_sum_l20_20529

theorem max_value_of_sum (x y z : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) 
  (eq : x^2 + y^2 + z^2 + x + 2*y + 3*z = (13 : ℝ) / 4) : x + y + z ≤ 3 / 2 :=
sorry

end max_value_of_sum_l20_20529


namespace ratio_of_counters_l20_20928

theorem ratio_of_counters (C_K M_K C_total M_ratio : ℕ)
  (h1 : C_K = 40)
  (h2 : M_K = 50)
  (h3 : M_ratio = 4 * M_K)
  (h4 : C_total = C_K + M_ratio)
  (h5 : C_total = 320) :
  C_K ≠ 0 → (320 - M_ratio) / C_K = 3 :=
by
  sorry

end ratio_of_counters_l20_20928


namespace find_legs_of_triangle_l20_20909

-- Definition of the problem conditions
def right_triangle (x y : ℝ) := x * y = 200 ∧ 4 * (y - 4) = 8 * (x - 8)

-- Theorem we want to prove
theorem find_legs_of_triangle : 
  ∃ (x y : ℝ), right_triangle x y ∧ ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) :=
by
  sorry

end find_legs_of_triangle_l20_20909


namespace divide_45_to_get_900_l20_20174

theorem divide_45_to_get_900 (x : ℝ) (h : 45 / x = 900) : x = 0.05 :=
by
  sorry

end divide_45_to_get_900_l20_20174


namespace common_ratio_of_geometric_series_l20_20545

theorem common_ratio_of_geometric_series (a S r : ℝ) (h₁ : a = 400) (h₂ : S = 2500) :
  S = a / (1 - r) → r = 21 / 25 :=
by
  intros h₃
  rw [h₁, h₂] at h₃
  sorry

end common_ratio_of_geometric_series_l20_20545


namespace rectangle_length_fraction_l20_20662

theorem rectangle_length_fraction 
  (s r : ℝ) 
  (A b ℓ : ℝ)
  (area_square : s * s = 1600)
  (radius_eq_side : r = s)
  (area_rectangle : A = ℓ * b)
  (breadth_rect : b = 10)
  (area_rect_val : A = 160) :
  ℓ / r = 2 / 5 := 
by
  sorry

end rectangle_length_fraction_l20_20662


namespace eval_custom_op_l20_20899

def custom_op (a b : ℤ) : ℤ := 2 * b + 5 * a - a^2 - b

theorem eval_custom_op : custom_op 3 4 = 10 :=
by
  sorry

end eval_custom_op_l20_20899


namespace square_area_l20_20991

theorem square_area :
  ∃ (s : ℝ), (8 * s - 2 = 30) ∧ (s ^ 2 = 16) :=
by
  sorry

end square_area_l20_20991


namespace mark_total_eggs_in_a_week_l20_20640

-- Define the given conditions
def first_store_eggs_per_day := 5 * 12 -- 5 dozen eggs per day
def second_store_eggs_per_day := 30
def third_store_eggs_per_odd_day := 25 * 12 -- 25 dozen eggs per odd day
def third_store_eggs_per_even_day := 15 * 12 -- 15 dozen eggs per even day
def days_per_week := 7
def odd_days_per_week := 4
def even_days_per_week := 3

-- Lean theorem statement to prove the total eggs supplied in a week
theorem mark_total_eggs_in_a_week : 
    first_store_eggs_per_day * days_per_week + 
    second_store_eggs_per_day * days_per_week + 
    third_store_eggs_per_odd_day * odd_days_per_week + 
    third_store_eggs_per_even_day * even_days_per_week =
    2370 := 
    sorry  -- Placeholder for the actual proof

end mark_total_eggs_in_a_week_l20_20640


namespace complement_A_union_B_l20_20603

-- Define the universal set U
def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

-- Define the set A
def A : Set ℤ := {-1, 2}

-- Define the set B using the quadratic equation condition
def B : Set ℤ := {x | x^2 - 4*x + 3 = 0}

-- State the theorem we want to prove
theorem complement_A_union_B : (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_A_union_B_l20_20603


namespace difference_of_fractions_l20_20970

theorem difference_of_fractions (a : ℝ) (b : ℝ) (h₁ : a = 7000) (h₂ : b = 1/10) :
  (a * b - a * (0.1 / 100)) = 693 :=
by 
  sorry

end difference_of_fractions_l20_20970


namespace number_of_true_propositions_is_two_l20_20314

def proposition1 (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (1 + x) = f (1 - x)

def proposition2 : Prop :=
∀ x : ℝ, 2 * Real.sin x * Real.cos (abs x) -- minimum period not 1
  -- We need to define proper periodicity which is complex; so here's a simplified representation
  ≠ 2 * Real.sin (x + 1) * Real.cos (abs (x + 1))

def increasing_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

def proposition3 (k : ℝ) : Prop :=
∀ n : ℕ, n > 0 → increasing_sequence (fun n => n^2 + k * n + 2)

def condition (f : ℝ → ℝ) (k : ℝ) : Prop :=
proposition1 f ∧ proposition2 ∧ proposition3 k

theorem number_of_true_propositions_is_two (f : ℝ → ℝ) (k : ℝ) :
  condition f k → 2 = 2 :=
by
  sorry

end number_of_true_propositions_is_two_l20_20314


namespace odd_function_periodic_example_l20_20780

theorem odd_function_periodic_example (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_segment : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (10 * Real.sqrt 3) = 36 - 20 * Real.sqrt 3 := 
sorry

end odd_function_periodic_example_l20_20780


namespace max_lines_between_points_l20_20666

noncomputable def maxLines (points : Nat) := 
  let deg := [1, 2, 3, 4, 5]
  (1 * (points - 1) + 2 * (points - 2) + 3 * (points - 3) + 4 * (points - 4) + 5 * (points - 5)) / 2

theorem max_lines_between_points :
  ∀ (n : Nat), n = 15 → maxLines n = 85 :=
by
  intros n hn
  sorry

end max_lines_between_points_l20_20666


namespace part1_part2_l20_20035

def f (x a : ℝ) := abs (x - a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, (f x a) ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 :=
by
  intros h
  sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f (2 * x) 3 + f (x + 2) 3 ≥ m) → m ≤ 1 / 2 :=
by
  intros h
  sorry

end part1_part2_l20_20035


namespace tangent_line_at_e_l20_20503

noncomputable def f (x : ℝ) := x * Real.log x

theorem tangent_line_at_e : ∀ x y : ℝ, (x = Real.exp 1) → (y = f x) → (y = 2 * x - Real.exp 1) :=
by
  intros x y hx hy
  sorry

end tangent_line_at_e_l20_20503


namespace solution_set_of_inequality_range_of_a_for_gx_zero_l20_20752

-- Define f(x) and g(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 1) + abs (x + a)

def g (x : ℝ) (a : ℝ) : ℝ := f x a - abs (3 + a)

-- The first Lean statement
theorem solution_set_of_inequality (a : ℝ) (h : a = 3) :
  ∀ x : ℝ, f x a > 6 ↔ x < -4 ∨ (-3 < x ∧ x < 1) ∨ 2 < x := by
  sorry

-- The second Lean statement
theorem range_of_a_for_gx_zero (a : ℝ) :
  (∃ x : ℝ, g x a = 0) ↔ a ≥ -2 := by
  sorry

end solution_set_of_inequality_range_of_a_for_gx_zero_l20_20752


namespace Clea_escalator_time_l20_20999

variable {s : ℝ} -- speed of the escalator at its normal operating speed
variable {c : ℝ} -- speed of Clea walking down the escalator
variable {d : ℝ} -- distance of the escalator

theorem Clea_escalator_time :
  (30 * (c + s) = 72 * c) →
  (s = (7 * c) / 5) →
  (t = (72 * c) / ((3 / 2) * s)) →
  t = 240 / 7 :=
by
  sorry

end Clea_escalator_time_l20_20999


namespace range_log_plus_three_is_l20_20508

noncomputable def range_log_plus_three (x : ℝ) (h : x ≥ 1) : ℝ := 
(log 2 x) + 3

theorem range_log_plus_three_is (s : Set ℝ) (h : ∀ x, x ≥ 1 → range_log_plus_three x h ∈ s) : 
s = {y : ℝ | y ≥ 3} :=
sorry

end range_log_plus_three_is_l20_20508


namespace find_g_l20_20037

variable (x : ℝ)

theorem find_g :
  ∃ g : ℝ → ℝ, 2 * x ^ 5 + 4 * x ^ 3 - 3 * x + 5 + g x = 3 * x ^ 4 + 7 * x ^ 2 - 2 * x - 4 ∧
                g x = -2 * x ^ 5 + 3 * x ^ 4 - 4 * x ^ 3 + 7 * x ^ 2 - x - 9 :=
sorry

end find_g_l20_20037


namespace tomatoes_not_sold_l20_20642

theorem tomatoes_not_sold (total_harvested sold_mrs_maxwell sold_mr_wilson : ℝ)
  (h1 : total_harvested = 245.5)
  (h2 : sold_mrs_maxwell = 125.5)
  (h3 : sold_mr_wilson = 78) :
total_harvested - (sold_mrs_maxwell + sold_mr_wilson) = 42 :=
by {
  rw [h1, h2, h3],
  norm_num,
  sorry
}

end tomatoes_not_sold_l20_20642


namespace yellow_ball_range_l20_20623

-- Definitions
def probability_condition (x : ℕ) : Prop :=
  (20 / 100 : ℝ) ≤ (4 * x / ((x + 2) * (x + 1))) ∧ (4 * x / ((x + 2) * (x + 1))) ≤ (33 / 100)

theorem yellow_ball_range (x : ℕ) : probability_condition x ↔ 9 ≤ x ∧ x ≤ 16 := 
by
  sorry

end yellow_ball_range_l20_20623


namespace larger_investment_value_l20_20007

-- Definitions of the conditions given in the problem
def investment_value_1 : ℝ := 500
def yearly_return_rate_1 : ℝ := 0.07
def yearly_return_rate_2 : ℝ := 0.27
def combined_return_rate : ℝ := 0.22

-- Stating the proof problem
theorem larger_investment_value :
  ∃ X : ℝ, X = 1500 ∧ 
    yearly_return_rate_1 * investment_value_1 + yearly_return_rate_2 * X = combined_return_rate * (investment_value_1 + X) :=
by {
  sorry -- Proof is omitted as per instructions
}

end larger_investment_value_l20_20007


namespace product_divisible_by_3_l20_20255

noncomputable def dice_prob_divisible_by_3 (n : ℕ) (faces : List ℕ) : ℚ := 
  let probability_div_3 := (1 / 3 : ℚ)
  let probability_not_div_3 := (2 / 3 : ℚ)
  1 - probability_not_div_3 ^ n

theorem product_divisible_by_3 (faces : List ℕ) (h_faces : faces = [1, 2, 3, 4, 5, 6]) :
  dice_prob_divisible_by_3 6 faces = 665 / 729 := 
  by 
    sorry

end product_divisible_by_3_l20_20255


namespace largest_value_among_l20_20185

theorem largest_value_among (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hneq : a ≠ b) :
  max (a + b) (max (2 * Real.sqrt (a * b)) ((a^2 + b^2) / (2 * a * b))) = a + b :=
sorry

end largest_value_among_l20_20185


namespace find_a_l20_20202

theorem find_a {a : ℝ} (h : {x : ℝ | (1/2 : ℝ) < x ∧ x < 2} = {x : ℝ | 0 < ax^2 + 5 * x - 2}) : a = -2 :=
sorry

end find_a_l20_20202


namespace inequality_C_incorrect_l20_20156

theorem inequality_C_incorrect (x : ℝ) (h : x ≠ 0) : ¬(e^x < 1 + x) → (e^1 ≥ 1 + 1) :=
by {
  sorry
}

end inequality_C_incorrect_l20_20156


namespace derivative_at_pi_over_3_l20_20882

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_at_pi_over_3 : 
  (deriv f) (Real.pi / 3) = 0 := 
by 
  sorry

end derivative_at_pi_over_3_l20_20882


namespace problem1_problem2_l20_20745

noncomputable def f (a x : ℝ) := Real.log x / Real.log a
noncomputable def g (a x t : ℝ) := 2 * Real.log (2 * x + t - 2) / Real.log a
noncomputable def F (a x t : ℝ) := g a x t - f a x

-- Problem 1: Proving a = 4
theorem problem1 (a : ℝ) (h_a_pos : 0 < a) (h_a_ne_one : a ≠ 1) :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → F a x 4 = 2) → a = 4 :=
sorry

-- Problem 2: Proving t >= 17/8
theorem problem2 (a t : ℝ) (h_a_pos : 0 < a) (h_a_lt_one : a < 1) :
  (∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → f a x ≥ g a x t) → t ≥ 17 / 8 :=
sorry


end problem1_problem2_l20_20745


namespace fx_fixed_point_l20_20531

theorem fx_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y, (x = -1) ∧ (y = 3) ∧ (a * (x + 1) + 2 = y) :=
by
  sorry

end fx_fixed_point_l20_20531


namespace average_number_of_fish_is_75_l20_20492

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l20_20492


namespace roger_allowance_spend_l20_20649

variable (A m s : ℝ)

-- Conditions from the problem
def condition1 : Prop := m = 0.25 * (A - 2 * s)
def condition2 : Prop := s = 0.10 * (A - 0.5 * m)
def goal : Prop := m + s = 0.59 * A

theorem roger_allowance_spend (h1 : condition1 A m s) (h2 : condition2 A m s) : goal A m s :=
  sorry

end roger_allowance_spend_l20_20649


namespace remainder_when_7n_divided_by_5_l20_20126

theorem remainder_when_7n_divided_by_5 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
  sorry

end remainder_when_7n_divided_by_5_l20_20126


namespace range_of_m_l20_20600

noncomputable def f (x : ℝ) : ℝ := |x - 3| - 2
noncomputable def g (x : ℝ) : ℝ := -|x + 1| + 4

theorem range_of_m :
  (∀ x : ℝ, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by sorry

end range_of_m_l20_20600


namespace students_play_alto_saxophone_l20_20859

def roosevelt_high_school :=
  let total_students := 600
  let marching_band_students := total_students / 5
  let brass_instrument_students := marching_band_students / 2
  let saxophone_students := brass_instrument_students / 5
  let alto_saxophone_students := saxophone_students / 3
  alto_saxophone_students

theorem students_play_alto_saxophone :
  roosevelt_high_school = 4 :=
  by
    sorry

end students_play_alto_saxophone_l20_20859


namespace clowns_to_guppies_ratio_l20_20646

theorem clowns_to_guppies_ratio
  (C : ℕ)
  (tetra : ℕ)
  (guppies : ℕ)
  (total_animals : ℕ)
  (h1 : tetra = 4 * C)
  (h2 : guppies = 30)
  (h3 : total_animals = 330)
  (h4 : total_animals = tetra + C + guppies) :
  C / guppies = 2 :=
by
  sorry

end clowns_to_guppies_ratio_l20_20646


namespace parallelogram_area_l20_20820

variable (d : ℕ) (h : ℕ)

theorem parallelogram_area (h_d : d = 30) (h_h : h = 20) : 
  ∃ a : ℕ, a = 600 := 
by
  sorry

end parallelogram_area_l20_20820


namespace simplify_and_evaluate_expression_l20_20930

theorem simplify_and_evaluate_expression (x y : ℝ) (hx : x = -1) (hy : y = -1) :
  (5 * x ^ 2 - 2 * (3 * y ^ 2 + 6 * x) + (2 * y ^ 2 - 5 * x ^ 2)) = 8 :=
by
  sorry

end simplify_and_evaluate_expression_l20_20930


namespace ratio_of_areas_l20_20806

theorem ratio_of_areas (aC aD : ℕ) (hC : aC = 48) (hD : aD = 60) : 
  (aC^2 : ℚ) / (aD^2 : ℚ) = (16 : ℚ) / (25 : ℚ) := 
by
  sorry

end ratio_of_areas_l20_20806


namespace correct_exponentiation_l20_20523

theorem correct_exponentiation (a : ℝ) : a^5 / a = a^4 := 
  sorry

end correct_exponentiation_l20_20523


namespace arithmetic_sequence_problem_l20_20628

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 = 15)
  (h2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 :=
sorry

end arithmetic_sequence_problem_l20_20628


namespace carol_extra_invitations_l20_20553

theorem carol_extra_invitations : 
  let invitations_per_pack := 3
  let packs_bought := 2
  let friends_to_invite := 9
  packs_bought * invitations_per_pack < friends_to_invite → 
  friends_to_invite - (packs_bought * invitations_per_pack) = 3 :=
by 
  intros _  -- Introduce the condition
  exact sorry  -- Placeholder for the proof

end carol_extra_invitations_l20_20553


namespace difference_between_numbers_l20_20242

theorem difference_between_numbers 
  (A B : ℝ)
  (h1 : 0.075 * A = 0.125 * B)
  (h2 : A = 2430 ∨ B = 2430) :
  A - B = 972 :=
by
  sorry

end difference_between_numbers_l20_20242


namespace complement_union_A_B_in_U_l20_20615

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}

def A : Set ℤ := {-1, 2}

def B : Set ℤ := {x | x^2 - 4 * x + 3 = 0}

theorem complement_union_A_B_in_U :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_union_A_B_in_U_l20_20615


namespace a5_value_l20_20769

def seq (a : ℕ → ℤ) (a1 : a 1 = 2) (rec : ∀ n, a (n + 1) = 2 * a n - 1) : Prop := True

theorem a5_value : 
  ∀ (a : ℕ → ℤ)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, a (n + 1) = 2 * a n - 1),
  seq a h1 recurrence → a 5 = 17 :=
by
  intros a h1 recurrence seq_a
  sorry

end a5_value_l20_20769


namespace smallest_n_for_107n_same_last_two_digits_l20_20871

theorem smallest_n_for_107n_same_last_two_digits :
  ∃ n : ℕ, n > 0 ∧ (107 * n) % 100 = n % 100 ∧ n = 50 :=
by {
  sorry
}

end smallest_n_for_107n_same_last_two_digits_l20_20871


namespace intersection_of_A_and_B_l20_20184

def A : Set ℚ := { x | x^2 - 4*x + 3 < 0 }
def B : Set ℚ := { x | 2 < x ∧ x < 4 }

theorem intersection_of_A_and_B : A ∩ B = { x | 2 < x ∧ x < 3 } := by
  sorry

end intersection_of_A_and_B_l20_20184


namespace sequence_properties_l20_20878

def f (x : ℝ) : ℝ := x^3 + 3 * x

variables {a_5 a_8 : ℝ}
variables {S_12 : ℝ}

axiom a5_condition : (a_5 - 1)^3 + 3 * a_5 = 4
axiom a8_condition : (a_8 - 1)^3 + 3 * a_8 = 2

theorem sequence_properties : (a_5 > a_8) ∧ (S_12 = 12) :=
by {
  sorry
}

end sequence_properties_l20_20878


namespace jace_gave_to_neighbor_l20_20214

theorem jace_gave_to_neighbor
  (earnings : ℕ) (debt : ℕ) (remaining : ℕ) (cents_per_dollar : ℕ) :
  earnings = 1000 →
  debt = 358 →
  remaining = 642 →
  cents_per_dollar = 100 →
  earnings - debt - remaining = 0
:= by
  intros h1 h2 h3 h4
  sorry

end jace_gave_to_neighbor_l20_20214


namespace volume_tetrahedron_OABC_correct_l20_20253

noncomputable def volume_tetrahedron_OABC : ℝ :=
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  (1 / 6) * a * b * c

theorem volume_tetrahedron_OABC_correct :
  let a := Real.sqrt 33
  let b := 4
  let c := 4 * Real.sqrt 3
  let volume := (1 / 6) * a * b * c
  volume = 8 * Real.sqrt 99 / 3 :=
by
  sorry

end volume_tetrahedron_OABC_correct_l20_20253


namespace genevieve_drinks_pints_l20_20006

theorem genevieve_drinks_pints (total_gallons : ℝ) (thermoses : ℕ) 
  (gallons_to_pints : ℝ) (genevieve_thermoses : ℕ) 
  (h1 : total_gallons = 4.5) (h2 : thermoses = 18) 
  (h3 : gallons_to_pints = 8) (h4 : genevieve_thermoses = 3) : 
  (total_gallons * gallons_to_pints / thermoses) * genevieve_thermoses = 6 := 
by
  admit

end genevieve_drinks_pints_l20_20006


namespace average_of_other_half_l20_20528

theorem average_of_other_half (avg : ℝ) (sum_half : ℝ) (n : ℕ) (n_half : ℕ)
    (h_avg : avg = 43.1)
    (h_sum_half : sum_half = 158.4)
    (h_n : n = 8)
    (h_n_half : n_half = n / 2) :
    ((n * avg - sum_half) / n_half) = 46.6 :=
by
  -- The proof steps would be given here. We're omitting them as the prompt instructs.
  sorry

end average_of_other_half_l20_20528


namespace complex_sum_to_zero_l20_20636

noncomputable def z : ℂ := sorry

theorem complex_sum_to_zero 
  (h₁ : z ^ 3 = 1) 
  (h₂ : z ≠ 1) : 
  z ^ 103 + z ^ 104 + z ^ 105 + z ^ 106 + z ^ 107 + z ^ 108 = 0 :=
sorry

end complex_sum_to_zero_l20_20636


namespace find_value_of_x2_div_y2_l20_20002

theorem find_value_of_x2_div_y2 (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z)
    (h7 : (y^2 / (x^2 - z^2) = (x^2 + y^2) / z^2))
    (h8 : (x^2 + y^2) / z^2 = x^2 / y^2) : x^2 / y^2 = 2 := by
  sorry

end find_value_of_x2_div_y2_l20_20002


namespace seven_lines_divide_into_29_regions_l20_20359

open Function

theorem seven_lines_divide_into_29_regions : 
  ∀ n : ℕ, (∀ l m : ℕ, l ≠ m → l < n ∧ m < n) → 1 + n + (n.choose 2) = 29 :=
by
  sorry

end seven_lines_divide_into_29_regions_l20_20359


namespace find_some_number_l20_20973

-- Definitions based on the given condition
def some_number : ℝ := sorry
def equation := some_number * 3.6 / (0.04 * 0.1 * 0.007) = 990.0000000000001

-- An assertion/proof that given the equation, some_number equals 7.7
theorem find_some_number (h : equation) : some_number = 7.7 :=
sorry

end find_some_number_l20_20973


namespace min_correct_answers_l20_20478

theorem min_correct_answers (total_questions correct_points incorrect_points target_score : ℕ)
                            (h_total : total_questions = 22)
                            (h_correct_points : correct_points = 4)
                            (h_incorrect_points : incorrect_points = 2)
                            (h_target : target_score = 81) :
  ∃ x : ℕ, 4 * x - 2 * (22 - x) > 81 ∧ x ≥ 21 :=
by {
  sorry
}

end min_correct_answers_l20_20478


namespace complex_b_value_l20_20760

open Complex

theorem complex_b_value (b : ℝ) (h : (2 - b * I) / (1 + 2 * I) = (2 - 2 * b) / 5 + ((-4 - b) / 5) * I) :
  b = -2 / 3 :=
sorry

end complex_b_value_l20_20760


namespace primes_not_exist_d_10_primes_not_exist_d_11_l20_20971

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def exists_three_distinct_primes_d_10 : Prop :=
  ¬∃ (p q r : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    (q * r ∣ (p ^ 2 + 10)) ∧ 
    (r * p ∣ (q ^ 2 + 10)) ∧ 
    (p * q ∣ (r ^ 2 + 10))

def exists_three_distinct_primes_d_11 : Prop :=
  ¬∃ (p q r : ℕ), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    is_prime p ∧ is_prime q ∧ is_prime r ∧
    (q * r ∣ (p ^ 2 + 11)) ∧ 
    (r * p ∣ (q ^ 2 + 11)) ∧ 
    (p * q ∣ (r ^ 2 + 11))

theorem primes_not_exist_d_10 : exists_three_distinct_primes_d_10 := by 
  sorry

theorem primes_not_exist_d_11 : exists_three_distinct_primes_d_11 := by 
  sorry

end primes_not_exist_d_10_primes_not_exist_d_11_l20_20971


namespace pencil_ratio_l20_20223

theorem pencil_ratio (B G : ℕ) (h1 : ∀ (n : ℕ), n = 20) 
  (h2 : ∀ (n : ℕ), n = 40) 
  (h3 : ∀ (n : ℕ), n = 160) 
  (h4 : G = 20 + B)
  (h5 : B + 20 + G + 40 = 160) : 
  (B / 20) = 4 := 
  by sorry

end pencil_ratio_l20_20223


namespace volume_of_tetrahedron_l20_20547

theorem volume_of_tetrahedron 
  (A B C D E : ℝ)
  (AB AD AE: ℝ)
  (h_AB : AB = 3)
  (h_AD : AD = 4)
  (h_AE : AE = 1)
  (V : ℝ) :
  (V = (4 * Real.sqrt 3) / 3) :=
sorry

end volume_of_tetrahedron_l20_20547


namespace find_slope_l20_20142

noncomputable def slope_of_first_line
    (m : ℝ)
    (intersect_point : ℝ × ℝ)
    (slope_second_line : ℝ)
    (x_intercept_distance : ℝ) 
    : Prop :=
  let (x₀, y₀) := intersect_point
  let x_intercept_first := (40 * m - 30) / m
  let x_intercept_second := 35
  abs (x_intercept_first - x_intercept_second) = x_intercept_distance

theorem find_slope : ∃ m : ℝ, slope_of_first_line m (40, 30) 6 10 :=
by
  use 2
  sorry

end find_slope_l20_20142


namespace range_of_x_l20_20072

theorem range_of_x (x : ℝ) : (x ≠ -3) ∧ (x ≤ 4) ↔ (x ≤ 4) ∧ (x ≠ -3) :=
by { sorry }

end range_of_x_l20_20072


namespace avg_wx_half_l20_20200

noncomputable def avg_wx {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) : ℝ :=
(w + x) / 2

theorem avg_wx_half {w x y : ℝ} (h1 : 5 / w + 5 / x = 5 / y) (h2 : w * x = y) :
  avg_wx h1 h2 = 1 / 2 :=
sorry

end avg_wx_half_l20_20200


namespace midpoint_sum_coordinates_l20_20423

theorem midpoint_sum_coordinates :
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  (midpoint.1 + midpoint.2) = 9 :=
by
  let p1 : ℝ × ℝ := (8, -4)
  let p2 : ℝ × ℝ := (-2, 16)
  let midpoint : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  show midpoint.1 + midpoint.2 = 9
  sorry

end midpoint_sum_coordinates_l20_20423


namespace weight_of_7_weights_l20_20681

theorem weight_of_7_weights :
  ∀ (w : ℝ), (16 * w + 0.6 = 17.88) → 7 * w = 7.56 :=
by
  intros w h
  sorry

end weight_of_7_weights_l20_20681


namespace charity_fundraising_l20_20683

theorem charity_fundraising (num_people : ℕ) (amount_event1 amount_event2 : ℕ) (total_amount_per_person : ℕ) :
  num_people = 8 →
  amount_event1 = 2000 →
  amount_event2 = 1000 →
  total_amount_per_person = (amount_event1 + amount_event2) / num_people →
  total_amount_per_person = 375 :=
by
  intros h1 h2 h3 h4
  sorry

end charity_fundraising_l20_20683


namespace steps_climbed_l20_20775

-- Definitions
def flights : ℕ := 9
def feet_per_flight : ℕ := 10
def inches_per_step : ℕ := 18

-- Proving the number of steps John climbs up
theorem steps_climbed : 
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  steps = 60 := 
by
  let feet_total := flights * feet_per_flight
  let inches_total := feet_total * 12
  let steps := inches_total / inches_per_step
  sorry

end steps_climbed_l20_20775


namespace number_of_sets_C_l20_20754

open Set

def A := {x : ℝ | x^2 - 3 * x + 2 = 0}

def B := {x : ℕ | 0 < x ∧ x < 5}

theorem number_of_sets_C :
  ∃ (n : ℕ), n = 3 ∧ 
  ∀ C : Set ℕ, A ⊆ C ∧ C ⊆ B ∧ A != C → n = 3 := 
sorry

end number_of_sets_C_l20_20754


namespace average_number_of_fish_is_75_l20_20493

-- Define the number of fish in Boast Pool and conditions for other bodies of water
def Boast_Pool_fish : ℕ := 75
def Onum_Lake_fish : ℕ := Boast_Pool_fish + 25
def Riddle_Pond_fish : ℕ := Onum_Lake_fish / 2

-- Define the average number of fish in all three bodies of water
def average_fish : ℕ := (Onum_Lake_fish + Boast_Pool_fish + Riddle_Pond_fish) / 3

-- Prove that the average number of fish in all three bodies of water is 75
theorem average_number_of_fish_is_75 : average_fish = 75 := by
  sorry

end average_number_of_fish_is_75_l20_20493


namespace solve_quadratic_eqn_l20_20173

theorem solve_quadratic_eqn:
  (∃ x: ℝ, (x + 10)^2 = (4 * x + 6) * (x + 8)) ↔ 
  (∀ x: ℝ, x = 2.131 ∨ x = -8.131) := 
by
  sorry

end solve_quadratic_eqn_l20_20173


namespace program_exists_l20_20486
open Function

-- Define the chessboard and labyrinth
namespace ChessMaze

structure Position :=
  (row : Nat)
  (col : Nat)
  (h_row : row < 8)
  (h_col : col < 8)

inductive Command
| RIGHT | LEFT | UP | DOWN

structure Labyrinth :=
  (barriers : Position → Position → Bool) -- True if there's a barrier between the two positions

def accessible (L : Labyrinth) (start : Position) (cmd : List Command) : Set Position :=
  -- The set of positions accessible after applying the commands from start in labyrinth L
  sorry

-- The main theorem we want to prove
theorem program_exists : 
  ∃ (cmd : List Command), ∀ (L : Labyrinth) (start : Position), ∀ pos ∈ accessible L start cmd, ∃ p : Position, p = pos :=
  sorry

end ChessMaze

end program_exists_l20_20486


namespace smallest_n_divisible_11_remainder_1_l20_20961

theorem smallest_n_divisible_11_remainder_1 :
  ∃ (n : ℕ), (n % 2 = 1) ∧ (n % 3 = 1) ∧ (n % 4 = 1) ∧ (n % 5 = 1) ∧ (n % 7 = 1) ∧ (n % 11 = 0) ∧ 
    (∀ m : ℕ, (m % 2 = 1) ∧ (m % 3 = 1) ∧ (m % 4 = 1) ∧ (m % 5 = 1) ∧ (m % 7 = 1) ∧ (m % 11 = 0) → 2521 ≤ m) :=
by
  sorry

end smallest_n_divisible_11_remainder_1_l20_20961


namespace two_digit_numbers_condition_l20_20297

theorem two_digit_numbers_condition : ∃ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧
    10 * a + b ≥ 10 ∧ 10 * a + b ≤ 99 ∧
    (10 * a + b) / (a + b) = (a + b) / 3 ∧ 
    (10 * a + b = 27 ∨ 10 * a + b = 48) := 
by
    sorry

end two_digit_numbers_condition_l20_20297


namespace sequence_exists_and_unique_l20_20668

theorem sequence_exists_and_unique (a : ℕ → ℕ) :
  a 0 = 11 ∧ a 7 = 12 ∧
  (∀ n : ℕ, n < 6 → a n + a (n + 1) + a (n + 2) = 50) →
  (a 0 = 11 ∧ a 1 = 12 ∧ a 2 = 27 ∧ a 3 = 11 ∧ a 4 = 12 ∧
   a 5 = 27 ∧ a 6 = 11 ∧ a 7 = 12) :=
by
  sorry

end sequence_exists_and_unique_l20_20668


namespace find_alpha_l20_20048

theorem find_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) 
  (l1 : ∀ x y : ℝ, x * Real.cos α - y - 1 = 0) 
  (l2 : ∀ x y : ℝ, x + y * Real.sin α + 1 = 0) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
sorry

end find_alpha_l20_20048


namespace real_solutions_eq_pos_neg_2_l20_20441

theorem real_solutions_eq_pos_neg_2 (x : ℝ) :
  ( (x - 1) ^ 2 * (x - 5) * (x - 5) / (x - 5) = 4) ↔ (x = 3 ∨ x = -1) :=
by
  sorry

end real_solutions_eq_pos_neg_2_l20_20441


namespace max_m_l20_20730

theorem max_m : ∃ m A B : ℤ, (AB = 90 ∧ m = 5 * B + A) ∧ (∀ m' A' B', (A' * B' = 90 ∧ m' = 5 * B' + A') → m' ≤ 451) ∧ m = 451 :=
by
  sorry

end max_m_l20_20730


namespace winner_more_votes_l20_20905

variable (totalStudents : ℕ) (votingPercentage : ℤ) (winnerPercentage : ℤ) (loserPercentage : ℤ)

theorem winner_more_votes
    (h1 : totalStudents = 2000)
    (h2 : votingPercentage = 25)
    (h3 : winnerPercentage = 55)
    (h4 : loserPercentage = 100 - winnerPercentage)
    (h5 : votingStudents = votingPercentage * totalStudents / 100)
    (h6 : winnerVotes = winnerPercentage * votingStudents / 100)
    (h7 : loserVotes = loserPercentage * votingStudents / 100)
    : winnerVotes - loserVotes = 50 := by
  sorry

end winner_more_votes_l20_20905


namespace relationship_of_y_values_l20_20576

theorem relationship_of_y_values (b y1 y2 y3 : ℝ) (h1 : y1 = 3 * (-3) - b)
                                (h2 : y2 = 3 * 1 - b)
                                (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 :=
by
  sorry

end relationship_of_y_values_l20_20576


namespace original_number_of_people_l20_20334

/-- Initially, one-third of the people in a room left.
Then, one-fourth of those remaining started to dance.
There were then 18 people who were not dancing.
What was the original number of people in the room? -/
theorem original_number_of_people (x : ℕ) 
  (h_one_third_left : ∀ y : ℕ, 2 * y / 3 = x) 
  (h_one_fourth_dancing : ∀ y : ℕ, y / 4 = x) 
  (h_non_dancers : x / 2 = 18) : 
  x = 36 :=
sorry

end original_number_of_people_l20_20334


namespace perfect_square_tens_place_l20_20714

/-- A whole number ending in 5 can only be a perfect square if the tens place is 2. -/
theorem perfect_square_tens_place (n : ℕ) (h₁ : n % 10 = 5) : ∃ k : ℕ, n = k * k → (n / 10) % 10 = 2 :=
sorry

end perfect_square_tens_place_l20_20714


namespace rectangle_area_l20_20539

theorem rectangle_area (x : ℕ) (hx : x > 0)
  (h₁ : (x + 5) * 2 * (x + 10) = 3 * x * (x + 10))
  (h₂ : (x - 10) = x + 10 - 10) :
  x * (x + 10) = 200 :=
by {
  sorry
}

end rectangle_area_l20_20539


namespace tangent_line_exists_l20_20913

noncomputable def tangent_line_problem := ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  Int.gcd (Int.gcd a b) c = 1 ∧ 
  (∀ x y : ℝ, a * x + b * (x^2 + 52 / 25) = c ∧ a * (y^2 + 81 / 16) + b * y = c) ∧ 
  a + b + c = 168

theorem tangent_line_exists : tangent_line_problem := by
  sorry

end tangent_line_exists_l20_20913


namespace fred_sheets_l20_20302

theorem fred_sheets (initial_sheets : ℕ) (received_sheets : ℕ) (given_sheets : ℕ) :
  initial_sheets = 212 → received_sheets = 307 → given_sheets = 156 →
  (initial_sheets + received_sheets - given_sheets) = 363 :=
by
  intros h_initial h_received h_given
  rw [h_initial, h_received, h_given]
  sorry

end fred_sheets_l20_20302


namespace solve_for_A_l20_20396

theorem solve_for_A (A : ℕ) (h1 : 3 + 68 * A = 691) (h2 : 68 * A < 1000) (h3 : 68 * A ≥ 100) : A = 8 :=
by
  sorry

end solve_for_A_l20_20396


namespace correct_props_l20_20039

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 2 * (Real.sin x + Real.cos x)

theorem correct_props : 
  (∃ φ : ℝ, ∀ x : ℝ, f(x + φ) = -f(x)) ∧
  (∃ k : ℤ, ∀ x : ℝ, f(x) = f(-x + -3 * Real.pi / 4)) :=
by 
  sorry

end correct_props_l20_20039


namespace circle_line_bisect_l20_20622

theorem circle_line_bisect (a : ℝ) :
    (∀ x y : ℝ, (x - (-1))^2 + (y - 2)^2 = 5 → 3 * x + y + a = 0) → a = 1 :=
sorry

end circle_line_bisect_l20_20622


namespace triangle_area_inscribed_in_circle_l20_20149

noncomputable def circle_inscribed_triangle_area : ℝ :=
  let r := 10 / Real.pi
  let angle_A := Real.pi / 2
  let angle_B := 7 * Real.pi / 10
  let angle_C := 4 * Real.pi / 5
  let sin_sum := Real.sin(angle_A) + Real.sin(angle_B) + Real.sin(angle_C)
  1 / 2 * r^2 * sin_sum

theorem triangle_area_inscribed_in_circle (h_circumference : 5 + 7 + 8 = 20)
  (h_radius : 10 / Real.pi * 2 * Real.pi = 20) :
  circle_inscribed_triangle_area = 138.005 / Real.pi^2 :=
by
  sorry

end triangle_area_inscribed_in_circle_l20_20149


namespace regular_polygon_sides_l20_20896

theorem regular_polygon_sides (D : ℕ) (h : D = 30) :
  ∃ n : ℕ, D = n * (n - 3) / 2 ∧ n = 9 :=
by
  use 9
  rw [h]
  norm_num
  sorry

end regular_polygon_sides_l20_20896


namespace additional_telephone_lines_l20_20246

theorem additional_telephone_lines :
  let lines_six_digits := 9 * 10^5
  let lines_seven_digits := 9 * 10^6
  let additional_lines := lines_seven_digits - lines_six_digits
  additional_lines = 81 * 10^5 :=
by
  sorry

end additional_telephone_lines_l20_20246


namespace gcd_lcm_sum_l20_20345

-- Define the necessary components: \( A \) as the greatest common factor and \( B \) as the least common multiple of 16, 32, and 48
def A := Int.gcd (Int.gcd 16 32) 48
def B := Int.lcm (Int.lcm 16 32) 48

-- Statement that needs to be proved
theorem gcd_lcm_sum : A + B = 112 := by
  sorry

end gcd_lcm_sum_l20_20345


namespace expression_subtracted_from_3_pow_k_l20_20058

theorem expression_subtracted_from_3_pow_k (k : ℕ) (h : 15^k ∣ 759325) : 3^k - 0 = 1 :=
sorry

end expression_subtracted_from_3_pow_k_l20_20058


namespace relationship_between_y_coordinates_l20_20579

theorem relationship_between_y_coordinates (b y1 y2 y3 : ℝ)
  (h1 : y1 = 3 * (-3) - b)
  (h2 : y2 = 3 * 1 - b)
  (h3 : y3 = 3 * (-1) - b) :
  y1 < y3 ∧ y3 < y2 := 
sorry

end relationship_between_y_coordinates_l20_20579


namespace total_copies_l20_20686

-- Conditions: Defining the rates of two copy machines and the time duration
def rate1 : ℕ := 35 -- rate in copies per minute for the first machine
def rate2 : ℕ := 65 -- rate in copies per minute for the second machine
def time : ℕ := 30 -- time in minutes

-- The theorem stating that the total number of copies made by both machines in 30 minutes is 3000
theorem total_copies : rate1 * time + rate2 * time = 3000 := by
  sorry

end total_copies_l20_20686


namespace remainder_97_pow_50_mod_100_l20_20962

theorem remainder_97_pow_50_mod_100 :
  (97 ^ 50) % 100 = 49 := 
by
  sorry

end remainder_97_pow_50_mod_100_l20_20962


namespace specific_gravity_proof_l20_20283

noncomputable def specific_gravity_condition (M R h : ℝ) : ℝ :=
  let vol_cone := (1/3) * π * R^2 * M
  let vol_subcone := (1/3) * π * R^2 * (1/√3 * M)
  let water_density := (1/3) * π * R^2 * (M * √3 - h)
  vol_subcone / water_density

theorem specific_gravity_proof : 
  (∀ M R m, specific_gravity_condition M R m = 1 - (√3 / 9)) :=
by
  intros
  unfold specific_gravity_condition
  sorry

end specific_gravity_proof_l20_20283


namespace average_value_is_2020_l20_20980

namespace CardsAverage

theorem average_value_is_2020 (n : ℕ) (h : (2020 * 3 * ((n * (n + 1)) + 2) = n * (n + 1) * (2 * n + 1) + 6 * (n + 1))) : n = 3015 := 
by
  sorry

end CardsAverage

end average_value_is_2020_l20_20980


namespace anthony_balloon_count_l20_20252

variable (Tom Luke Anthony : ℕ)

theorem anthony_balloon_count
  (h1 : Tom = 3 * Luke)
  (h2 : Luke = Anthony / 4)
  (hTom : Tom = 33) :
  Anthony = 44 := by
    sorry

end anthony_balloon_count_l20_20252


namespace f_zero_is_118_l20_20082

theorem f_zero_is_118
  (f : ℕ → ℕ)
  (eq1 : ∀ m n : ℕ, f (m^2 + n^2) = (f m - f n)^2 + f (2 * m * n))
  (eq2 : 8 * f 0 + 9 * f 1 = 2006) :
  f 0 = 118 :=
sorry

end f_zero_is_118_l20_20082


namespace meal_preppers_activity_setters_count_l20_20277

-- Definitions for the problem conditions
def num_friends : ℕ := 6
def num_meal_preppers : ℕ := 3

-- Statement of the theorem
theorem meal_preppers_activity_setters_count :
  (num_friends.choose num_meal_preppers) = 20 :=
by
  -- Proof would go here
  sorry

end meal_preppers_activity_setters_count_l20_20277


namespace scientific_notation_of_600_million_l20_20099

theorem scientific_notation_of_600_million : 600000000 = 6 * 10^7 := 
sorry

end scientific_notation_of_600_million_l20_20099


namespace determine_m_n_l20_20879

theorem determine_m_n 
  {a b c d m n : ℕ} 
  (h₁ : a + b + c + d = m^2)
  (h₂ : a^2 + b^2 + c^2 + d^2 = 1989)
  (h₃ : max (max a b) (max c d) = n^2) 
  : m = 9 ∧ n = 6 := by 
  sorry

end determine_m_n_l20_20879


namespace part1_l20_20595

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, x > 0 → f x < 0) :
  a > 1 :=
sorry

end part1_l20_20595


namespace find_number_l20_20400

noncomputable def N : ℕ :=
  76

theorem find_number :
  (N % 13 = 11) ∧ (N % 17 = 9) :=
by
  -- These are the conditions translated to Lean 4, as stated:
  have h1 : N % 13 = 11 := by sorry
  have h2 : N % 17 = 9 := by sorry
  exact ⟨h1, h2⟩

end find_number_l20_20400


namespace derivative_at_one_third_l20_20054

open Real

def f (x : ℝ) : ℝ := log (2 - 3 * x)

theorem derivative_at_one_third : deriv f (1 / 3) = -3 :=
by
  sorry

end derivative_at_one_third_l20_20054


namespace total_lives_correct_l20_20839

namespace VideoGame

def num_friends : ℕ := 8
def lives_each : ℕ := 8

def total_lives (n : ℕ) (l : ℕ) : ℕ := n * l 

theorem total_lives_correct : total_lives num_friends lives_each = 64 := by
  sorry

end total_lives_correct_l20_20839


namespace natasha_destination_distance_l20_20786

theorem natasha_destination_distance
  (over_speed : ℕ)
  (time : ℕ)
  (speed_limit : ℕ)
  (actual_speed : ℕ)
  (distance : ℕ) :
  (over_speed = 10) →
  (time = 1) →
  (speed_limit = 50) →
  (actual_speed = speed_limit + over_speed) →
  (distance = actual_speed * time) →
  (distance = 60) :=
by
  sorry

end natasha_destination_distance_l20_20786


namespace pages_per_hour_l20_20380

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l20_20380


namespace chess_pieces_present_l20_20091

theorem chess_pieces_present (total_pieces : ℕ) (missing_pieces : ℕ) (h1 : total_pieces = 32) (h2 : missing_pieces = 4) : (total_pieces - missing_pieces) = 28 := 
by sorry

end chess_pieces_present_l20_20091


namespace distance_traveled_by_bus_l20_20984

noncomputable def total_distance : ℕ := 900
noncomputable def distance_by_plane : ℕ := total_distance / 3
noncomputable def distance_by_bus : ℕ := 360
noncomputable def distance_by_train : ℕ := (2 * distance_by_bus) / 3

theorem distance_traveled_by_bus :
  distance_by_plane + distance_by_train + distance_by_bus = total_distance :=
by
  sorry

end distance_traveled_by_bus_l20_20984


namespace area_of_triangle_l20_20207

noncomputable def circumradius (a b c : ℝ) (α : ℝ) : ℝ := a / (2 * Real.sin α)

theorem area_of_triangle (A B C a b c R : ℝ) (h₁ : b * Real.cos C + c * Real.cos B = Real.sqrt 3 * R)
  (h₂ : a = 2) (h₃ : b + c = 4) : 
  1 / 2 * b * (c * Real.sin A) = Real.sqrt 3 :=
by
  sorry

end area_of_triangle_l20_20207


namespace factor_expression_l20_20874

theorem factor_expression (x : ℝ) : 6 * x ^ 3 - 54 * x = 6 * x * (x + 3) * (x - 3) :=
by {
  sorry
}

end factor_expression_l20_20874


namespace Tara_savings_after_one_year_l20_20655

theorem Tara_savings_after_one_year :
  ∀ (initial_amount : ℝ) (interest_rate : ℝ),
    initial_amount = 90 → interest_rate = 0.1 →
    let interest_earned := initial_amount * interest_rate in
    let total_amount := initial_amount + interest_earned in
    total_amount = 99 :=
by
  intros initial_amount interest_rate h_initial_amount h_interest_rate
  rw [h_initial_amount, h_interest_rate]
  let interest_earned := initial_amount * interest_rate
  let total_amount := initial_amount + interest_earned
  rw [h_initial_amount, h_interest_rate]
  sorry

end Tara_savings_after_one_year_l20_20655


namespace probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l20_20801

noncomputable def germination_rate : ℝ := 0.9
noncomputable def non_germination_rate : ℝ := 1 - germination_rate
noncomputable def strong_seedling_rate : ℝ := 0.6
noncomputable def non_strong_seedling_rate : ℝ := 1 - strong_seedling_rate

theorem probability_two_seeds_missing_seedlings :
  (non_germination_rate ^ 2) = 0.01 := sorry

theorem probability_two_seeds_no_strong_seedlings :
  (non_strong_seedling_rate ^ 2) = 0.16 := sorry

theorem probability_three_seeds_having_seedlings :
  (1 - non_germination_rate ^ 3) = 0.999 := sorry

theorem probability_three_seeds_having_strong_seedlings :
  (1 - non_strong_seedling_rate ^ 3) = 0.936 := sorry

end probability_two_seeds_missing_seedlings_probability_two_seeds_no_strong_seedlings_probability_three_seeds_having_seedlings_probability_three_seeds_having_strong_seedlings_l20_20801


namespace find_number_l20_20001

theorem find_number (x : ℤ) (n : ℤ) (h1 : x = 88320) (h2 : x + 1315 + n - 1569 = 11901) : n = -75165 :=
by 
  sorry

end find_number_l20_20001


namespace geometric_sequence_S6_l20_20447

-- Assume we have a geometric sequence {a_n} and the sum of the first n terms is denoted as S_n
variable (S : ℕ → ℝ)

-- Conditions given in the problem
axiom S2_eq : S 2 = 2
axiom S4_eq : S 4 = 8

-- The goal is to find the value of S 6
theorem geometric_sequence_S6 : S 6 = 26 := 
by 
  sorry

end geometric_sequence_S6_l20_20447


namespace mascot_toy_profit_l20_20698

theorem mascot_toy_profit (x : ℝ) :
  (∀ (c : ℝ) (sales : ℝ), c = 40 → sales = 1000 - 10 * x → (x - c) * sales = 8000) →
  (x = 60 ∨ x = 80) :=
by
  intro h
  sorry

end mascot_toy_profit_l20_20698


namespace absent_children_on_teachers_day_l20_20925

theorem absent_children_on_teachers_day (A : ℕ) (h1 : ∀ n : ℕ, n = 190)
(h2 : ∀ s : ℕ, s = 38) (h3 : ∀ extra : ℕ, extra = 14) :
  (190 - A) * 38 = 190 * 24 → A = 70 :=
by
  sorry

end absent_children_on_teachers_day_l20_20925


namespace real_yield_correct_l20_20824

-- Define the annual inflation rate and nominal interest rate
def annualInflationRate : ℝ := 0.025
def nominalInterestRate : ℝ := 0.06

-- Define the number of years
def numberOfYears : ℕ := 2

-- Calculate total inflation over two years
def totalInflation (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n - 1

-- Calculate the compounded nominal rate
def compoundedNominalRate (r : ℝ) (n : ℕ) : ℝ := (1 + r) ^ n

-- Calculate the real yield considering inflation
def realYield (nominalRate : ℝ) (inflationRate : ℝ) : ℝ :=
  (nominalRate / (1 + inflationRate)) - 1

-- Proof problem statement
theorem real_yield_correct :
  let inflation_two_years := totalInflation annualInflationRate numberOfYears
  let nominal_two_years := compoundedNominalRate nominalInterestRate numberOfYears
  inflation_two_years * 100 = 5.0625 ∧
  (realYield nominal_two_years inflation_two_years) * 100 = 6.95 :=
by
  sorry

end real_yield_correct_l20_20824


namespace sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l20_20860

theorem sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6 : 
  (Nat.sqrt 8 - Nat.sqrt 2 - (Nat.sqrt (1 / 3) * Nat.sqrt 6) = 0) :=
by
  sorry

theorem sqrt15_div_sqrt3_add_sqrt5_sub1_sq : 
  (Nat.sqrt 15 / Nat.sqrt 3 + (Nat.sqrt 5 - 1) ^ 2 = 6 - Nat.sqrt 5) :=
by
  sorry

end sqrt8_sub_sqrt2_sub_sqrt1div3_mul_sqrt6_sqrt15_div_sqrt3_add_sqrt5_sub1_sq_l20_20860


namespace sum_is_2000_l20_20518

theorem sum_is_2000 (x y : ℝ) (h : x ≠ y) (h_eq : x^2 - 2000 * x = y^2 - 2000 * y) : x + y = 2000 := by
  sorry

end sum_is_2000_l20_20518


namespace yoque_payment_months_l20_20265

-- Define the conditions
def monthly_payment : ℝ := 15
def amount_borrowed : ℝ := 150
def total_payment : ℝ := amount_borrowed * 1.1

-- Define the proof problem
theorem yoque_payment_months :
  ∃ (n : ℕ), n * monthly_payment = total_payment :=
by 
  have monthly_payment : ℝ := 15
  have amount_borrowed : ℝ := 150
  have total_payment : ℝ := amount_borrowed * 1.1
  use 11
  sorry

end yoque_payment_months_l20_20265


namespace unreasonable_inference_l20_20413

theorem unreasonable_inference:
  (∀ (S T : Type) (P : S → Prop) (Q : T → Prop), (∀ x y, P x → ¬ Q y) → ¬ (∀ x, P x) → (∃ y, ¬ Q y))
  ∧ ¬ (∀ s : ℝ, (s = 100) → ∀ t : ℝ, t = 100) :=
sorry

end unreasonable_inference_l20_20413


namespace q_investment_l20_20689

theorem q_investment (p_investment : ℝ) (profit_ratio_p : ℝ) (profit_ratio_q : ℝ) (q_investment : ℝ) 
  (h1 : p_investment = 40000) 
  (h2 : profit_ratio_p / profit_ratio_q = 2 / 3) 
  : q_investment = 60000 := 
sorry

end q_investment_l20_20689


namespace target_hit_probability_l20_20133

theorem target_hit_probability (prob_A_hits : ℝ) (prob_B_hits : ℝ) (hA : prob_A_hits = 0.5) (hB : prob_B_hits = 0.6) :
  (1 - (1 - prob_A_hits) * (1 - prob_B_hits)) = 0.8 := 
by 
  sorry

end target_hit_probability_l20_20133


namespace find_k_intersection_on_line_l20_20038

theorem find_k_intersection_on_line (k : ℝ) :
  (∃ (x y : ℝ), x - 2 * y - 2 * k = 0 ∧ 2 * x - 3 * y - k = 0 ∧ 3 * x - y = 0) → k = 0 :=
by
  sorry

end find_k_intersection_on_line_l20_20038


namespace find_a_mul_b_l20_20795

theorem find_a_mul_b (x y z a b : ℝ)
  (h1 : a = x)
  (h2 : b = y)
  (h3 : x + x = y * x)
  (h4 : b = z)
  (h5 : x + x = z * z)
  (h6 : y = 3)
  : a * b = 4 := by
  sorry

end find_a_mul_b_l20_20795


namespace not_a_fraction_l20_20682

axiom x : ℝ
axiom a : ℝ
axiom b : ℝ

noncomputable def A := 1 / (x^2)
noncomputable def B := (b + 3) / a
noncomputable def C := (x^2 - 1) / (x + 1)
noncomputable def D := (2 / 7) * a

theorem not_a_fraction : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) :=
by 
  sorry

end not_a_fraction_l20_20682


namespace neg_three_lt_neg_sqrt_eight_l20_20027

theorem neg_three_lt_neg_sqrt_eight : -3 < -Real.sqrt 8 := 
sorry

end neg_three_lt_neg_sqrt_eight_l20_20027


namespace ram_balance_speed_l20_20648

theorem ram_balance_speed
  (part_speed : ℝ)
  (balance_distance : ℝ)
  (total_distance : ℝ)
  (total_time : ℝ)
  (part_time : ℝ)
  (balance_speed : ℝ)
  (h1 : part_speed = 20)
  (h2 : total_distance = 400)
  (h3 : total_time = 8)
  (h4 : part_time = 3.2)
  (h5 : balance_distance = total_distance - part_speed * part_time)
  (h6 : balance_speed = balance_distance / (total_time - part_time)) :
  balance_speed = 70 :=
by
  simp [h1, h2, h3, h4, h5, h6]
  sorry

end ram_balance_speed_l20_20648


namespace incorrect_expression_D_l20_20083

noncomputable def E : ℝ := sorry
def R : ℕ := sorry
def S : ℕ := sorry
def m : ℕ := sorry
def t : ℕ := sorry

-- E is a repeating decimal
-- R is the non-repeating part of E with m digits
-- S is the repeating part of E with t digits

theorem incorrect_expression_D : ¬ (10^m * (10^t - 1) * E = S * (R - 1)) :=
sorry

end incorrect_expression_D_l20_20083


namespace num_members_in_league_l20_20067

theorem num_members_in_league :
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  num_members = 150 :=
by
  let sock_cost := 5
  let tshirt_cost := 11
  let total_exp := 3100
  let cost_per_member_before_discount := 2 * (sock_cost + tshirt_cost)
  let discount := 3
  let effective_cost_per_member := cost_per_member_before_discount - discount
  let num_members := total_exp / effective_cost_per_member
  sorry

end num_members_in_league_l20_20067


namespace total_sign_up_methods_l20_20809

theorem total_sign_up_methods (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) :
  k ^ n = 16 :=
by
  rw [h1, h2]
  norm_num

end total_sign_up_methods_l20_20809


namespace ordered_triples_lcm_l20_20197

def lcm_equal (a b n : ℕ) : Prop :=
  a * b / (Nat.gcd a b) = n

theorem ordered_triples_lcm :
  ∀ (x y z : ℕ), 0 < x → 0 < y → 0 < z → 
  lcm_equal x y 48 → lcm_equal x z 900 → lcm_equal y z 180 →
  false :=
by sorry

end ordered_triples_lcm_l20_20197


namespace certain_number_l20_20758

theorem certain_number (p q : ℝ) (h1 : 3 / p = 6) (h2 : p - q = 0.3) : 3 / q = 15 :=
by
  sorry

end certain_number_l20_20758


namespace contemporaries_probability_l20_20672

theorem contemporaries_probability:
  (∀ (x y : ℝ),
    0 ≤ x ∧ x ≤ 400 ∧
    0 ≤ y ∧ y ≤ 400 ∧
    (x < y + 80) ∧ (y < x + 80)) →
    (∃ p : ℝ, p = 9 / 25) :=
by sorry

end contemporaries_probability_l20_20672


namespace solve_n_is_2_l20_20407

noncomputable def problem_statement (n : ℕ) : Prop :=
  ∃ m : ℕ, 9 * n^2 + 5 * n - 26 = m * (m + 1)

theorem solve_n_is_2 : problem_statement 2 :=
  sorry

end solve_n_is_2_l20_20407


namespace volume_tetrahedral_region_is_correct_l20_20248

noncomputable def volume_of_tetrahedral_region (a : ℝ) : ℝ :=
  (81 - 8 * Real.pi) * a^3 / 486

theorem volume_tetrahedral_region_is_correct (a : ℝ) :
  volume_of_tetrahedral_region a = (81 - 8 * Real.pi) * a^3 / 486 :=
by
  sorry

end volume_tetrahedral_region_is_correct_l20_20248


namespace common_ratio_of_geometric_series_l20_20544

theorem common_ratio_of_geometric_series (a S r : ℚ) (ha : a = 400) (hS : S = 2500) (hS_eq : S = a / (1 - r)) : r = 21 / 25 :=
by {
  rw [ha, hS] at hS_eq,
  sorry
}

end common_ratio_of_geometric_series_l20_20544


namespace product_diff_squares_l20_20484

theorem product_diff_squares (a b c d x1 y1 x2 y2 x3 y3 x4 y4 : ℕ) 
  (ha : a = x1^2 - y1^2) 
  (hb : b = x2^2 - y2^2) 
  (hc : c = x3^2 - y3^2) 
  (hd : d = x4^2 - y4^2)
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) :
  ∃ X Y : ℕ, a * b * c * d = X^2 - Y^2 :=
by
  sorry

end product_diff_squares_l20_20484


namespace unoccupied_garden_area_is_correct_l20_20280

noncomputable def area_unoccupied_by_pond_trees_bench (π : ℝ) : ℝ :=
  let garden_area := 144
  let pond_area_rectangle := 6
  let pond_area_semi_circle := 2 * π
  let trees_area := 3
  let bench_area := 3
  garden_area - (pond_area_rectangle + pond_area_semi_circle + trees_area + bench_area)

theorem unoccupied_garden_area_is_correct : 
  area_unoccupied_by_pond_trees_bench Real.pi = 132 - 2 * Real.pi :=
by
  sorry

end unoccupied_garden_area_is_correct_l20_20280


namespace quadratic_coefficients_l20_20569

theorem quadratic_coefficients (a b c : ℝ) (h₀: 0 < a) 
  (h₁: |a + b + c| = 3) 
  (h₂: |4 * a + 2 * b + c| = 3) 
  (h₃: |9 * a + 3 * b + c| = 3) : 
  (a = 6 ∧ b = -24 ∧ c = 21) ∨ (a = 3 ∧ b = -15 ∧ c = 15) ∨ (a = 3 ∧ b = -9 ∧ c = 3) :=
sorry

end quadratic_coefficients_l20_20569


namespace distinct_sequences_six_sided_die_rolled_six_times_l20_20836

theorem distinct_sequences_six_sided_die_rolled_six_times :
  let count := 6
  (count ^ 6 = 46656) :=
by
  let count := 6
  sorry

end distinct_sequences_six_sided_die_rolled_six_times_l20_20836


namespace target_average_income_l20_20990

variable (past_incomes : List ℕ) (next_average : ℕ)

def total_past_income := past_incomes.sum
def total_next_income := next_average * 5
def total_ten_week_income := total_past_income past_incomes + total_next_income next_average

theorem target_average_income (h1 : past_incomes = [406, 413, 420, 436, 395])
                              (h2 : next_average = 586) :
  total_ten_week_income past_incomes next_average / 10 = 500 := by
  sorry

end target_average_income_l20_20990


namespace distance_between_trees_l20_20974

def yard_length : ℕ := 350
def num_trees : ℕ := 26
def num_intervals : ℕ := num_trees - 1

theorem distance_between_trees :
  yard_length / num_intervals = 14 := 
sorry

end distance_between_trees_l20_20974


namespace exists_linear_function_intersecting_negative_axes_l20_20926

theorem exists_linear_function_intersecting_negative_axes :
  ∃ (k b : ℝ), k < 0 ∧ b < 0 ∧ (∃ x, k * x + b = 0 ∧ x < 0) ∧ (k * 0 + b < 0) :=
by
  sorry

end exists_linear_function_intersecting_negative_axes_l20_20926


namespace first_diamond_second_spade_prob_l20_20121

/--
Given a standard deck of 52 cards, there are 13 cards of each suit.
What is the probability that the first card dealt is a diamond (♦) 
and the second card dealt is a spade (♠)?
-/
theorem first_diamond_second_spade_prob : 
  let total_cards := 52
  let diamonds := 13
  let spades := 13
  let first_diamond_prob := diamonds / total_cards
  let second_spade_prob_after_diamond := spades / (total_cards - 1)
  let combined_prob := first_diamond_prob * second_spade_prob_after_diamond
  combined_prob = 13 / 204 := 
by
  sorry

end first_diamond_second_spade_prob_l20_20121


namespace find_f_of_9_l20_20504

variable (f : ℝ → ℝ)

-- Conditions
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_of_3 : f 3 = 4

-- Theorem statement to prove
theorem find_f_of_9 : f 9 = 64 := by
  sorry

end find_f_of_9_l20_20504


namespace rectangle_percentage_excess_l20_20903

variable (L W : ℝ) -- The lengths of the sides of the rectangle
variable (x : ℝ) -- The percentage excess for the first side (what we want to prove)

theorem rectangle_percentage_excess 
    (h1 : W' = W * 0.95)                    -- Condition: second side is taken with 5% deficit
    (h2 : L' = L * (1 + x/100))             -- Condition: first side is taken with x% excess
    (h3 : A = L * W)                        -- Actual area of the rectangle
    (h4 : 1.064 = (L' * W') / A) :           -- Condition: error percentage in the area is 6.4%
    x = 12 :=                                -- Proof that x equals 12
sorry

end rectangle_percentage_excess_l20_20903
