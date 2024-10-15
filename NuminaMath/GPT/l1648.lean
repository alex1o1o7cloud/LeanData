import Mathlib

namespace NUMINAMATH_GPT_number_of_days_l1648_164861

theorem number_of_days (d : ℝ) (h : 2 * d = 1.5 * d + 3) : d = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_days_l1648_164861


namespace NUMINAMATH_GPT_total_birds_and_storks_l1648_164895

theorem total_birds_and_storks (initial_birds initial_storks additional_storks : ℕ) 
  (h1 : initial_birds = 3) 
  (h2 : initial_storks = 4) 
  (h3 : additional_storks = 6) 
  : initial_birds + initial_storks + additional_storks = 13 := 
  by sorry

end NUMINAMATH_GPT_total_birds_and_storks_l1648_164895


namespace NUMINAMATH_GPT_price_reduction_correct_eqn_l1648_164801

theorem price_reduction_correct_eqn (x : ℝ) :
  120 * (1 - x)^2 = 85 :=
sorry

end NUMINAMATH_GPT_price_reduction_correct_eqn_l1648_164801


namespace NUMINAMATH_GPT_new_mean_l1648_164881

-- Define the conditions
def mean_of_numbers (n : ℕ) (mean : ℝ) : ℝ := n * mean
def added_to_each (n : ℕ) (addend : ℝ) : ℝ := n * addend

-- The proof problem
theorem new_mean (n : ℕ) (mean addend : ℝ) (h1 : mean_of_numbers n mean = 600) (h2 : added_to_each n addend = 150) (h3 : n = 15) (h4 : mean = 40) (h5 : addend = 10) :
  (mean_of_numbers n mean + added_to_each n addend) / n = 50 :=
by
  sorry

end NUMINAMATH_GPT_new_mean_l1648_164881


namespace NUMINAMATH_GPT_sum_x_y_eq_l1648_164800

noncomputable def equation (x y : ℝ) : Prop :=
  2 * x^2 - 4 * x * y + 4 * y^2 + 6 * x + 9 = 0

theorem sum_x_y_eq (x y : ℝ) (h : equation x y) : x + y = -9 / 2 :=
by sorry

end NUMINAMATH_GPT_sum_x_y_eq_l1648_164800


namespace NUMINAMATH_GPT_Donny_spends_28_on_Thursday_l1648_164847

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end NUMINAMATH_GPT_Donny_spends_28_on_Thursday_l1648_164847


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1648_164867

-- Definitions for lines a and b, and planes alpha and beta
variables {a b : Type} {α β : Type}

-- predicate for line a being in plane α
def line_in_plane (a : Type) (α : Type) : Prop := sorry

-- predicate for line b being perpendicular to plane β
def line_perpendicular_plane (b : Type) (β : Type) : Prop := sorry

-- predicate for plane α being parallel to plane β
def plane_parallel_plane (α : Type) (β : Type) : Prop := sorry

-- predicate for line a being perpendicular to line b
def line_perpendicular_line (a : Type) (b : Type) : Prop := sorry

-- Proof of the statement: The condition of line a being in plane α, line b being perpendicular to plane β,
-- and plane α being parallel to plane β is sufficient but not necessary for line a being perpendicular to line b.
theorem sufficient_but_not_necessary
  (a b : Type) (α β : Type)
  (h1 : line_in_plane a α)
  (h2 : line_perpendicular_plane b β)
  (h3 : plane_parallel_plane α β) :
  line_perpendicular_line a b :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1648_164867


namespace NUMINAMATH_GPT_determine_m_l1648_164857

theorem determine_m (a b : ℝ) (m : ℝ) :
  (a^2 + 2 * a * b - b^2) - (a^2 + m * a * b + 2 * b^2) = (2 - m) * a * b - 3 * b^2 →
  (∀ a b : ℝ, (2 - m) * a * b = 0) →
  m = 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l1648_164857


namespace NUMINAMATH_GPT_remainder_div_2DD_l1648_164884

theorem remainder_div_2DD' (P D D' Q R Q' R' : ℕ) 
  (h1 : P = Q * D + R) 
  (h2 : Q = 2 * D' * Q' + R') :
  P % (2 * D * D') = D * R' + R :=
sorry

end NUMINAMATH_GPT_remainder_div_2DD_l1648_164884


namespace NUMINAMATH_GPT_sum_of_primes_between_30_and_50_l1648_164876

-- Define a prime number
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- List of prime numbers between 30 and 50
def prime_numbers_between_30_and_50 : List ℕ := [31, 37, 41, 43, 47]

-- Sum of prime numbers between 30 and 50
def sum_prime_numbers_between_30_and_50 : ℕ :=
  prime_numbers_between_30_and_50.sum

-- Theorem: The sum of prime numbers between 30 and 50 is 199
theorem sum_of_primes_between_30_and_50 :
  sum_prime_numbers_between_30_and_50 = 199 := by
    sorry

end NUMINAMATH_GPT_sum_of_primes_between_30_and_50_l1648_164876


namespace NUMINAMATH_GPT_jakes_present_weight_l1648_164888

theorem jakes_present_weight (J S : ℕ) (h1 : J - 32 = 2 * S) (h2 : J + S = 212) : J = 152 :=
by
  sorry

end NUMINAMATH_GPT_jakes_present_weight_l1648_164888


namespace NUMINAMATH_GPT_balance_balls_l1648_164802

noncomputable def green_weight := (9 : ℝ) / 4
noncomputable def yellow_weight := (7 : ℝ) / 3
noncomputable def white_weight := (3 : ℝ) / 2

theorem balance_balls (B : ℝ) : 
  5 * green_weight * B + 4 * yellow_weight * B + 3 * white_weight * B = (301 / 12) * B :=
by
  sorry

end NUMINAMATH_GPT_balance_balls_l1648_164802


namespace NUMINAMATH_GPT_tangent_line_x_squared_l1648_164892

theorem tangent_line_x_squared (P : ℝ × ℝ) (hP : P = (1, -1)) :
  ∃ (a : ℝ), a = 1 + Real.sqrt 2 ∨ a = 1 - Real.sqrt 2 ∧
    ((∀ x : ℝ, (2 * (1 + Real.sqrt 2) * x - (3 + 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 + Real.sqrt 2) * P.1 - (3 + 2 * Real.sqrt 2))) ∨
    (∀ x : ℝ, (2 * (1 - Real.sqrt 2) * x - (3 - 2 * Real.sqrt 2)) = P.2 → 
      P.2 = (2 * (1 - Real.sqrt 2) * P.1 - (3 - 2 * Real.sqrt 2)))) := by
  sorry

end NUMINAMATH_GPT_tangent_line_x_squared_l1648_164892


namespace NUMINAMATH_GPT_apples_difference_l1648_164871

theorem apples_difference 
  (father_apples : ℕ := 8)
  (mother_apples : ℕ := 13)
  (jungkook_apples : ℕ := 7)
  (brother_apples : ℕ := 5) :
  max father_apples (max mother_apples (max jungkook_apples brother_apples)) - 
  min father_apples (min mother_apples (min jungkook_apples brother_apples)) = 8 :=
by
  sorry

end NUMINAMATH_GPT_apples_difference_l1648_164871


namespace NUMINAMATH_GPT_num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l1648_164866

def is_prime (n : ℕ) : Prop := Nat.Prime n
def ends_with_7 (n : ℕ) : Prop := n % 10 = 7

theorem num_prime_numbers_with_units_digit_7 (n : ℕ) (h1 : n < 100) (h2 : ends_with_7 n) : is_prime n :=
by sorry

theorem num_prime_numbers_less_than_100_with_units_digit_7 : 
  ∃ (l : List ℕ), (∀ x ∈ l, x < 100 ∧ ends_with_7 x ∧ is_prime x) ∧ l.length = 6 :=
by sorry

end NUMINAMATH_GPT_num_prime_numbers_with_units_digit_7_num_prime_numbers_less_than_100_with_units_digit_7_l1648_164866


namespace NUMINAMATH_GPT_xy_value_l1648_164887

theorem xy_value (x y : ℝ) (h : x * (x + y) = x^2 + 12) : x * y = 12 := by
  sorry

end NUMINAMATH_GPT_xy_value_l1648_164887


namespace NUMINAMATH_GPT_man_l1648_164816

theorem man's_speed_against_current :
  ∀ (V_down V_c V_m V_up : ℝ),
    (V_down = 15) →
    (V_c = 2.8) →
    (V_m = V_down - V_c) →
    (V_up = V_m - V_c) →
    V_up = 9.4 :=
by
  intros V_down V_c V_m V_up
  intros hV_down hV_c hV_m hV_up
  sorry

end NUMINAMATH_GPT_man_l1648_164816


namespace NUMINAMATH_GPT_rotation_90_ccw_l1648_164891

-- Define the complex number before the rotation
def initial_complex : ℂ := -4 - 2 * Complex.I

-- Define the resulting complex number after a 90-degree counter-clockwise    rotation
def result_complex : ℂ := 2 - 4 * Complex.I

-- State the theorem to be proved
theorem rotation_90_ccw (z : ℂ) (h : z = initial_complex) :
  Complex.I * z = result_complex :=
by sorry

end NUMINAMATH_GPT_rotation_90_ccw_l1648_164891


namespace NUMINAMATH_GPT_complex_third_quadrant_l1648_164890

-- Define the imaginary unit i.
def i : ℂ := Complex.I 

-- Define the complex number z = i * (1 + i).
def z : ℂ := i * (1 + i)

-- Prove that z lies in the third quadrant.
theorem complex_third_quadrant : z.re < 0 ∧ z.im < 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_third_quadrant_l1648_164890


namespace NUMINAMATH_GPT_intersection_result_l1648_164859

def A : Set ℝ := {x | |x - 2| ≤ 2}

def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

def intersection : Set ℝ := {x | 0 ≤ x ∧ x ≤ 2}

theorem intersection_result : (A ∩ B) = intersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_result_l1648_164859


namespace NUMINAMATH_GPT_original_candle_length_l1648_164808

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (h_current : current_length = 48) (h_factor : factor = 1.33) :
  (current_length * factor = 63.84) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_original_candle_length_l1648_164808


namespace NUMINAMATH_GPT_children_got_on_bus_l1648_164828

theorem children_got_on_bus (initial_children total_children children_added : ℕ) 
  (h_initial : initial_children = 64) 
  (h_total : total_children = 78) : 
  children_added = total_children - initial_children :=
by
  sorry

end NUMINAMATH_GPT_children_got_on_bus_l1648_164828


namespace NUMINAMATH_GPT_age_of_third_boy_l1648_164851

theorem age_of_third_boy (a b c : ℕ) (h1 : a = 9) (h2 : b = 9) (h_sum : a + b + c = 29) : c = 11 :=
by
  sorry

end NUMINAMATH_GPT_age_of_third_boy_l1648_164851


namespace NUMINAMATH_GPT_replacement_fraction_l1648_164822

variable (Q : ℝ) (x : ℝ)

def initial_concentration : ℝ := 0.70
def new_concentration : ℝ := 0.35
def replacement_concentration : ℝ := 0.25

theorem replacement_fraction (h1 : 0.70 * Q - 0.70 * x * Q + 0.25 * x * Q = 0.35 * Q) :
  x = 7 / 9 :=
by
  sorry

end NUMINAMATH_GPT_replacement_fraction_l1648_164822


namespace NUMINAMATH_GPT_tan_alpha_sub_pi_over_8_l1648_164812

theorem tan_alpha_sub_pi_over_8 (α : ℝ) (h : 2 * Real.tan α = 3 * Real.tan (Real.pi / 8)) :
  Real.tan (α - Real.pi / 8) = (5 * Real.sqrt 2 + 1) / 49 :=
by sorry

end NUMINAMATH_GPT_tan_alpha_sub_pi_over_8_l1648_164812


namespace NUMINAMATH_GPT_flag_arrangement_modulo_1000_l1648_164869

theorem flag_arrangement_modulo_1000 :
  let red_flags := 8
  let white_flags := 8
  let black_flags := 1
  let total_flags := red_flags + white_flags + black_flags
  let number_of_gaps := total_flags + 1
  let valid_arrangements := (Nat.choose number_of_gaps white_flags) * (number_of_gaps - 2)
  valid_arrangements % 1000 = 315 :=
by
  sorry

end NUMINAMATH_GPT_flag_arrangement_modulo_1000_l1648_164869


namespace NUMINAMATH_GPT_simplify_expression_l1648_164834

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : (3 * b - 3 - 5 * b) / 3 = - (2 / 3) * b - 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1648_164834


namespace NUMINAMATH_GPT_vector_parallel_l1648_164837

variables {t : ℝ}

def a : ℝ × ℝ := (1, 3)
def b : ℝ × ℝ := (3, t)

theorem vector_parallel (h : (1 : ℝ) / (3 : ℝ) = (3 : ℝ) / t) : t = 9 :=
by 
  sorry

end NUMINAMATH_GPT_vector_parallel_l1648_164837


namespace NUMINAMATH_GPT_repeated_pair_exists_l1648_164878

theorem repeated_pair_exists (a : Fin 99 → Fin 10)
  (h1 : ∀ n : Fin 98, a n = 1 → a (n + 1) ≠ 2)
  (h2 : ∀ n : Fin 98, a n = 3 → a (n + 1) ≠ 4) :
  ∃ k l : Fin 98, k ≠ l ∧ a k = a l ∧ a (k + 1) = a (l + 1) :=
sorry

end NUMINAMATH_GPT_repeated_pair_exists_l1648_164878


namespace NUMINAMATH_GPT_center_of_circle_l1648_164824

theorem center_of_circle (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 10) (h4 : y2 = 7) :
  (x1 + x2) / 2 = 6 ∧ (y1 + y2) / 2 = 2 :=
by
  rw [h1, h2, h3, h4]
  constructor
  · norm_num
  · norm_num

end NUMINAMATH_GPT_center_of_circle_l1648_164824


namespace NUMINAMATH_GPT_marble_problem_l1648_164804

variable (A V M : ℕ)

theorem marble_problem
  (h1 : A + 5 = V - 5)
  (h2 : V + 2 * (A + 5) = A - 2 * (A + 5) + M) :
  M = 10 :=
sorry

end NUMINAMATH_GPT_marble_problem_l1648_164804


namespace NUMINAMATH_GPT_cubical_block_weight_l1648_164854

-- Given conditions
variables (s : ℝ) (volume_ratio : ℝ) (weight2 : ℝ)
variable (h : volume_ratio = 8)
variable (h_weight : weight2 = 40)

-- The problem statement
theorem cubical_block_weight (weight1 : ℝ) :
  volume_ratio * weight1 = weight2 → weight1 = 5 :=
by
  -- Assume volume ratio as 8, weight of the second cube as 40 pounds
  have h1 : volume_ratio = 8 := h
  have h2 : weight2 = 40 := h_weight
  -- sorry is here to indicate we are skipping the proof
  sorry

end NUMINAMATH_GPT_cubical_block_weight_l1648_164854


namespace NUMINAMATH_GPT_proof_l1648_164885

variable {S : Type} 
variable (op : S → S → S)

-- Condition given in the problem
def condition (a b : S) : Prop :=
  op (op a b) a = b

-- Statement to be proven
theorem proof (h : ∀ a b : S, condition op a b) :
  ∀ a b : S, op a (op b a) = b :=
by
  intros a b
  sorry

end NUMINAMATH_GPT_proof_l1648_164885


namespace NUMINAMATH_GPT_feifei_reaches_school_at_828_l1648_164803

-- Definitions for all conditions
def start_time : Nat := 8 * 60 + 10  -- Feifei starts walking at 8:10 AM in minutes since midnight
def dog_delay : Nat := 3             -- Dog starts chasing after 3 minutes
def catch_up_200m_time : ℕ := 1      -- Time for dog to catch Feifei at 200 meters
def catch_up_400m_time : ℕ := 4      -- Time for dog to catch Feifei at 400 meters
def school_distance : ℕ := 800       -- Distance from home to school
def feifei_speed : ℕ := 2            -- assumed speed of Feifei where distance covered uniformly
def dog_speed : ℕ := 6               -- dog speed is three times Feifei's speed
def catch_times := [200, 400, 800]   -- Distances (in meters) where dog catches Feifei

-- Derived condition:
def total_travel_time : ℕ := 
  let time_for_200m := catch_up_200m_time + catch_up_200m_time;
  let time_for_400m_and_back := 2* catch_up_400m_time ;
  (time_for_200m + time_for_400m_and_back + (school_distance - 400))

-- The statement we wish to prove:
theorem feifei_reaches_school_at_828 : 
  (start_time + total_travel_time - dog_delay/2) % 60 = 28 :=
sorry

end NUMINAMATH_GPT_feifei_reaches_school_at_828_l1648_164803


namespace NUMINAMATH_GPT_cos_2pi_minus_alpha_tan_alpha_minus_7pi_l1648_164880

open Real

variables (α : ℝ)
variables (h1 : sin (π + α) = -1 / 3) (h2 : π / 2 < α ∧ α < π)

-- Statement for the problem (Ⅰ)
theorem cos_2pi_minus_alpha :
  cos (2 * π - α) = -2 * sqrt 2 / 3 :=
sorry

-- Statement for the problem (Ⅱ)
theorem tan_alpha_minus_7pi :
  tan (α - 7 * π) = -sqrt 2 / 4 :=
sorry

end NUMINAMATH_GPT_cos_2pi_minus_alpha_tan_alpha_minus_7pi_l1648_164880


namespace NUMINAMATH_GPT_heaviest_lightest_difference_l1648_164825

-- Define 4 boxes' weights
variables {a b c d : ℕ}

-- Define given pairwise weights
axiom w1 : a + b = 22
axiom w2 : a + c = 23
axiom w3 : c + d = 30
axiom w4 : b + d = 29

-- Define the inequality among the weights
axiom h1 : a < b
axiom h2 : b < c
axiom h3 : c < d

-- Prove the heaviest box is 7 kg heavier than the lightest
theorem heaviest_lightest_difference : d - a = 7 :=
by sorry

end NUMINAMATH_GPT_heaviest_lightest_difference_l1648_164825


namespace NUMINAMATH_GPT_employee_pay_l1648_164832

variable (X Y : ℝ)

theorem employee_pay (h1: X + Y = 572) (h2: X = 1.2 * Y) : Y = 260 :=
by
  sorry

end NUMINAMATH_GPT_employee_pay_l1648_164832


namespace NUMINAMATH_GPT_percentage_problem_l1648_164899

theorem percentage_problem (P : ℝ) (h : (P / 100) * 180 - (1 / 3) * (P / 100) * 180 = 42) : P = 35 := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_problem_l1648_164899


namespace NUMINAMATH_GPT_bacon_suggestion_count_l1648_164858

theorem bacon_suggestion_count (B : ℕ) (h1 : 408 = B + 366) : B = 42 :=
by
  sorry

end NUMINAMATH_GPT_bacon_suggestion_count_l1648_164858


namespace NUMINAMATH_GPT_workers_combined_time_l1648_164862

theorem workers_combined_time (g_rate a_rate c_rate : ℝ)
  (hg : g_rate = 1 / 70)
  (ha : a_rate = 1 / 30)
  (hc : c_rate = 1 / 42) :
  1 / (g_rate + a_rate + c_rate) = 14 :=
by
  sorry

end NUMINAMATH_GPT_workers_combined_time_l1648_164862


namespace NUMINAMATH_GPT_fill_time_l1648_164811

-- Definition of the conditions
def faster_pipe_rate (t : ℕ) := 1 / t
def slower_pipe_rate (t : ℕ) := 1 / (4 * t)
def combined_rate (t : ℕ) := faster_pipe_rate t + slower_pipe_rate t
def time_to_fill_tank (t : ℕ) := 1 / combined_rate t

-- Given t = 50, prove the combined fill time is 40 minutes which is equal to the target time to fill the tank.
theorem fill_time (t : ℕ) (h : 4 * t = 200) : t = 50 → time_to_fill_tank t = 40 :=
by
  intros ht
  rw [ht]
  sorry

end NUMINAMATH_GPT_fill_time_l1648_164811


namespace NUMINAMATH_GPT_susan_bought_36_items_l1648_164848

noncomputable def cost_per_pencil : ℝ := 0.25
noncomputable def cost_per_pen : ℝ := 0.80
noncomputable def pencils_bought : ℕ := 16
noncomputable def total_spent : ℝ := 20.0

theorem susan_bought_36_items :
  ∃ (pens_bought : ℕ), pens_bought * cost_per_pen + pencils_bought * cost_per_pencil = total_spent ∧ pencils_bought + pens_bought = 36 := 
sorry

end NUMINAMATH_GPT_susan_bought_36_items_l1648_164848


namespace NUMINAMATH_GPT_M_intersection_N_l1648_164889

-- Define the sets M and N
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}
def N : Set ℝ := {x | -1 < x ∧ x < 1}

-- Define the proof problem
theorem M_intersection_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_M_intersection_N_l1648_164889


namespace NUMINAMATH_GPT_palm_trees_total_l1648_164863

theorem palm_trees_total
  (forest_palm_trees : ℕ := 5000)
  (desert_palm_trees : ℕ := forest_palm_trees - (3 * forest_palm_trees / 5)) :
  desert_palm_trees + forest_palm_trees = 7000 :=
by
  sorry

end NUMINAMATH_GPT_palm_trees_total_l1648_164863


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l1648_164833

theorem arithmetic_expression_evaluation :
  (3 + 9) ^ 2 + (3 ^ 2) * (9 ^ 2) = 873 :=
by
  -- Proof is skipped, using sorry for now.
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l1648_164833


namespace NUMINAMATH_GPT_largest_integer_is_222_l1648_164819

theorem largest_integer_is_222
  (a b c d : ℤ)
  (h_distinct : a < b ∧ b < c ∧ c < d)
  (h_mean : (a + b + c + d) / 4 = 72)
  (h_min_a : a ≥ 21) 
  : d = 222 :=
sorry

end NUMINAMATH_GPT_largest_integer_is_222_l1648_164819


namespace NUMINAMATH_GPT_hexagons_formed_square_z_l1648_164835

theorem hexagons_formed_square_z (a b s z : ℕ) (hexagons_congruent : a = 9 ∧ b = 16 ∧ s = 12 ∧ z = 4): 
(z = 4) := by
  sorry

end NUMINAMATH_GPT_hexagons_formed_square_z_l1648_164835


namespace NUMINAMATH_GPT_algorithm_can_contain_all_structures_l1648_164864

def sequential_structure : Prop := sorry
def conditional_structure : Prop := sorry
def loop_structure : Prop := sorry

def algorithm_contains_structure (str : Prop) : Prop := sorry

theorem algorithm_can_contain_all_structures :
  algorithm_contains_structure sequential_structure ∧
  algorithm_contains_structure conditional_structure ∧
  algorithm_contains_structure loop_structure := sorry

end NUMINAMATH_GPT_algorithm_can_contain_all_structures_l1648_164864


namespace NUMINAMATH_GPT_original_quantity_of_ghee_l1648_164849

theorem original_quantity_of_ghee (Q : ℝ) (h1 : 0.6 * Q = 9) (h2 : 0.4 * Q = 6) (h3 : 0.4 * Q = 0.2 * (Q + 10)) : Q = 10 :=
by sorry

end NUMINAMATH_GPT_original_quantity_of_ghee_l1648_164849


namespace NUMINAMATH_GPT_find_a_l1648_164870

noncomputable def A (a : ℝ) : Set ℝ := {a + 2, (a + 1) ^ 2, a ^ 2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1648_164870


namespace NUMINAMATH_GPT_specific_heat_capacity_l1648_164855

variable {k x p S V α ν R μ : Real}
variable (p x V α : Real) (hp : p = α * V)
variable (hk : k * x = p * S)
variable (hα : α = k / (S^2))

theorem specific_heat_capacity 
  (hk : k * x = p * S) 
  (hp : p = α * V)
  (hα : α = k / (S^2)) 
  (hR : R > 0) 
  (hν : ν > 0) 
  (hμ : μ > 0)
  : (2 * R / μ) = 4155 := 
sorry

end NUMINAMATH_GPT_specific_heat_capacity_l1648_164855


namespace NUMINAMATH_GPT_sheelas_total_net_monthly_income_l1648_164875

noncomputable def totalNetMonthlyIncome
    (PrimaryJobIncome : ℝ)
    (FreelanceIncome : ℝ)
    (FreelanceIncomeTaxRate : ℝ)
    (AnnualInterestIncome : ℝ)
    (InterestIncomeTaxRate : ℝ) : ℝ :=
    let PrimaryJobMonthlyIncome := 5000 / 0.20
    let FreelanceIncomeTax := FreelanceIncome * FreelanceIncomeTaxRate
    let NetFreelanceIncome := FreelanceIncome - FreelanceIncomeTax
    let InterestIncomeTax := AnnualInterestIncome * InterestIncomeTaxRate
    let NetAnnualInterestIncome := AnnualInterestIncome - InterestIncomeTax
    let NetMonthlyInterestIncome := NetAnnualInterestIncome / 12
    PrimaryJobMonthlyIncome + NetFreelanceIncome + NetMonthlyInterestIncome

theorem sheelas_total_net_monthly_income :
    totalNetMonthlyIncome 25000 3000 0.10 2400 0.05 = 27890 := 
by
    sorry

end NUMINAMATH_GPT_sheelas_total_net_monthly_income_l1648_164875


namespace NUMINAMATH_GPT_chord_length_proof_tangent_lines_through_M_l1648_164874

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 4

noncomputable def line_l (x y : ℝ) : Prop := 2*x - y + 4 = 0

noncomputable def point_M : (ℝ × ℝ) := (3, 1)

noncomputable def chord_length : ℝ := 4 * Real.sqrt (5) / 5

noncomputable def tangent_line_1 (x y : ℝ) : Prop := 3*x - 4*y - 5 = 0
noncomputable def tangent_line_2 (x : ℝ) : Prop := x = 3

theorem chord_length_proof :
  ∀ x y : ℝ, circle_C x y → line_l x y → chord_length = 4 * Real.sqrt (5) / 5 :=
by sorry

theorem tangent_lines_through_M :
  ∀ x y : ℝ, circle_C x y → (tangent_line_1 x y ∨ tangent_line_2 x) :=
by sorry

end NUMINAMATH_GPT_chord_length_proof_tangent_lines_through_M_l1648_164874


namespace NUMINAMATH_GPT_man_l1648_164846

-- Define the conditions
def speed_downstream : ℕ := 8
def speed_upstream : ℕ := 4

-- Define the man's rate in still water
def rate_in_still_water : ℕ := (speed_downstream + speed_upstream) / 2

-- The target theorem
theorem man's_rate_in_still_water : rate_in_still_water = 6 := by
  -- The statement is set up. Proof to be added later.
  sorry

end NUMINAMATH_GPT_man_l1648_164846


namespace NUMINAMATH_GPT_f_at_neg_one_l1648_164868

def f (x : ℝ) : ℝ := sorry

theorem f_at_neg_one (h : ∀ x : ℝ, f (x - 1) = x^2 + 1) : f (-1) = 1 :=
by sorry

end NUMINAMATH_GPT_f_at_neg_one_l1648_164868


namespace NUMINAMATH_GPT_smallest_digit_divisible_by_9_l1648_164829

theorem smallest_digit_divisible_by_9 :
  ∃ d : ℕ, (0 ≤ d ∧ d < 10) ∧ (∃ k : ℕ, 26 + d = 9 * k) ∧ d = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_divisible_by_9_l1648_164829


namespace NUMINAMATH_GPT_max_value_E_X_E_Y_l1648_164850

open MeasureTheory

-- Defining the random variables and their ranges
variables {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω)
variable (X : Ω → ℝ) (Y : Ω → ℝ)

-- Condition: 2 ≤ X ≤ 3
def condition1 : Prop := ∀ ω, 2 ≤ X ω ∧ X ω ≤ 3

-- Condition: XY = 1
def condition2 : Prop := ∀ ω, X ω * Y ω = 1

-- The theorem statement
theorem max_value_E_X_E_Y (h1 : condition1 X) (h2 : condition2 X Y) : 
  ∃ E_X E_Y, (E_X = ∫ ω, X ω ∂μ) ∧ (E_Y = ∫ ω, Y ω ∂μ) ∧ (E_X * E_Y = 25 / 24) := 
sorry

end NUMINAMATH_GPT_max_value_E_X_E_Y_l1648_164850


namespace NUMINAMATH_GPT_hypotenuse_length_l1648_164896

theorem hypotenuse_length (a b c : ℝ) (h1 : a + b + c = 32) (h2 : a * b = 40) (h3 : a^2 + b^2 = c^2) : 
  c = 59 / 4 :=
by
  sorry

end NUMINAMATH_GPT_hypotenuse_length_l1648_164896


namespace NUMINAMATH_GPT_length_of_DE_l1648_164809

-- Given conditions
variables (AB DE : ℝ) (area_projected area_ABC : ℝ)

-- Hypotheses
def base_length (AB : ℝ) : Prop := AB = 15
def projected_area_ratio (area_projected area_ABC : ℝ) : Prop := area_projected = 0.25 * area_ABC
def parallel_lines (DE AB : ℝ) : Prop := ∀ x : ℝ, DE = 0.5 * AB

-- The theorem to prove
theorem length_of_DE (h1 : base_length AB) (h2 : projected_area_ratio area_projected area_ABC) (h3 : parallel_lines DE AB) : DE = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_length_of_DE_l1648_164809


namespace NUMINAMATH_GPT_D_72_eq_93_l1648_164898

def D (n : ℕ) : ℕ :=
-- The function definition of D would go here, but we leave it abstract for now.
sorry

theorem D_72_eq_93 : D 72 = 93 :=
sorry

end NUMINAMATH_GPT_D_72_eq_93_l1648_164898


namespace NUMINAMATH_GPT_back_wheel_revolutions_l1648_164843

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (rev_front : ℝ) (r_front_eq : r_front = 3)
  (r_back_eq : r_back = 0.5) (rev_front_eq : rev_front = 50) :
  let C_front := 2 * Real.pi * r_front
  let D_front := C_front * rev_front
  let C_back := 2 * Real.pi * r_back
  let rev_back := D_front / C_back
  rev_back = 300 := by
  sorry

end NUMINAMATH_GPT_back_wheel_revolutions_l1648_164843


namespace NUMINAMATH_GPT_tickets_sold_l1648_164882

theorem tickets_sold (student_tickets non_student_tickets student_ticket_price non_student_ticket_price total_revenue : ℕ)
  (h1 : student_ticket_price = 5)
  (h2 : non_student_ticket_price = 8)
  (h3 : total_revenue = 930)
  (h4 : student_tickets = 90)
  (h5 : non_student_tickets = 60) :
  student_tickets + non_student_tickets = 150 := 
by 
  sorry

end NUMINAMATH_GPT_tickets_sold_l1648_164882


namespace NUMINAMATH_GPT_smallest_y_value_l1648_164841

theorem smallest_y_value (y : ℚ) (h : y / 7 + 2 / (7 * y) = 1 / 3) : y = 2 / 3 :=
sorry

end NUMINAMATH_GPT_smallest_y_value_l1648_164841


namespace NUMINAMATH_GPT_range_of_a_l1648_164877

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3 * a) ↔ -1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1648_164877


namespace NUMINAMATH_GPT_part1_part2_l1648_164836

noncomputable def f (a x : ℝ) : ℝ := (a * Real.exp x - a - x) * Real.exp x

theorem part1 (a : ℝ) (h0 : a ≥ 0) (h1 : ∀ x : ℝ, f a x ≥ 0) : a = 1 := 
sorry

theorem part2 (h1 : ∀ x : ℝ, f 1 x ≥ 0) :
  ∃! x0 : ℝ, (∀ x : ℝ, x0 = x → 
  (f 1 x0) = (f 1 x)) ∧ (0 < f 1 x0 ∧ f 1 x0 < 1/4) :=
sorry

end NUMINAMATH_GPT_part1_part2_l1648_164836


namespace NUMINAMATH_GPT_star_15_star_eq_neg_15_l1648_164879

-- Define the operations as given
def y_star (y : ℤ) := 9 - y
def star_y (y : ℤ) := y - 9

-- The theorem stating the required proof
theorem star_15_star_eq_neg_15 : star_y (y_star 15) = -15 :=
by
  sorry

end NUMINAMATH_GPT_star_15_star_eq_neg_15_l1648_164879


namespace NUMINAMATH_GPT_different_distributions_l1648_164827

def arrangement_methods (students teachers: Finset ℕ) : ℕ :=
  students.card.factorial * (students.card - 1).factorial * ((students.card - 1) - 1).factorial

theorem different_distributions :
  ∀ (students teachers : Finset ℕ), 
  students.card = 3 ∧ teachers.card = 3 →
  arrangement_methods students teachers = 72 :=
by sorry

end NUMINAMATH_GPT_different_distributions_l1648_164827


namespace NUMINAMATH_GPT_quadratic_root_l1648_164821

theorem quadratic_root (k : ℝ) (h : (1:ℝ)^2 - 3 * (1 : ℝ) - k = 0) : k = -2 :=
sorry

end NUMINAMATH_GPT_quadratic_root_l1648_164821


namespace NUMINAMATH_GPT_nine_digit_not_perfect_square_l1648_164826

theorem nine_digit_not_perfect_square (D : ℕ) (h1 : 100000000 ≤ D) (h2 : D < 1000000000)
  (h3 : ∀ c : ℕ, (c ∈ D.digits 10) → (c ≠ 0)) (h4 : D % 10 = 5) :
  ¬ ∃ A : ℕ, D = A ^ 2 := 
sorry

end NUMINAMATH_GPT_nine_digit_not_perfect_square_l1648_164826


namespace NUMINAMATH_GPT_distinct_primes_sum_reciprocal_l1648_164831

open Classical

theorem distinct_primes_sum_reciprocal (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r)
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) 
  (hineq: (1 / p : ℚ) + (1 / q) + (1 / r) ≥ 1) 
  : (p = 2 ∧ q = 3 ∧ r = 5) ∨ (p = 2 ∧ q = 5 ∧ r = 3) ∨ (p = 3 ∧ q = 2 ∧ r = 5) ∨
    (p = 3 ∧ q = 5 ∧ r = 2) ∨ (p = 5 ∧ q = 2 ∧ r = 3) ∨ (p = 5 ∧ q = 3 ∧ r = 2) := 
sorry

end NUMINAMATH_GPT_distinct_primes_sum_reciprocal_l1648_164831


namespace NUMINAMATH_GPT_calculate_expression_l1648_164815

variable (x : ℝ)

theorem calculate_expression : ((3 * x)^2) * (x^2) = 9 * (x^4) := 
sorry

end NUMINAMATH_GPT_calculate_expression_l1648_164815


namespace NUMINAMATH_GPT_range_of_slope_exists_k_for_collinearity_l1648_164820

def line_equation (k x : ℝ) : ℝ := k * x + 1

def circle_eq (x y : ℝ) : ℝ := x^2 + y^2 - 4 * x + 3

noncomputable def intersect_points (k : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  sorry  -- Assume a function that computes the intersection points (x₁, y₁) and (x₂, y₂)

def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v2 = (c * v1.1, c * v1.2)

theorem range_of_slope (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0) :
  -4/3 < k ∧ k < 0 := 
sorry

theorem exists_k_for_collinearity (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ)
  (h1 : line_equation k x₁ = y₁) 
  (h2 : line_equation k x₂ = y₂)
  (h3 : circle_eq x₁ y₁ = 0)
  (h4 : circle_eq x₂ y₂ = 0)
  (h5 : -4/3 < k ∧ k < 0) :
  collinear (2 - x₁ - x₂, -(y₁ + y₂)) (-2, 1) ↔ k = -1/2 :=
sorry


end NUMINAMATH_GPT_range_of_slope_exists_k_for_collinearity_l1648_164820


namespace NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l1648_164883

theorem cos_alpha_in_second_quadrant (α : ℝ) (hα : π / 2 < α ∧ α < π) (h_tan : Real.tan α = -1 / 2) :
  Real.cos α = -2 * Real.sqrt 5 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_alpha_in_second_quadrant_l1648_164883


namespace NUMINAMATH_GPT_correct_expression_l1648_164830

theorem correct_expression (a b : ℚ) (h1 : 3 * a = 4 * b) (h2 : a ≠ 0) (h3 : b ≠ 0) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_correct_expression_l1648_164830


namespace NUMINAMATH_GPT_donut_selection_l1648_164814

theorem donut_selection :
  ∃ (ways : ℕ), ways = Nat.choose 8 3 ∧ ways = 56 :=
by
  sorry

end NUMINAMATH_GPT_donut_selection_l1648_164814


namespace NUMINAMATH_GPT_inequality_proof_l1648_164865

theorem inequality_proof 
(x1 x2 y1 y2 z1 z2 : ℝ) 
(hx1 : x1 > 0) 
(hx2 : x2 > 0) 
(hineq1 : x1 * y1 - z1^2 > 0) 
(hineq2 : x2 * y2 - z2^2 > 0)
: 
  8 / ((x1 + x2)*(y1 + y2) - (z1 + z2)^2) <= 
  1 / (x1 * y1 - z1^2) + 
  1 / (x2 * y2 - z2^2) := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1648_164865


namespace NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_is_20_l1648_164838

theorem number_of_terms_in_arithmetic_sequence_is_20
  (a : ℕ → ℤ)
  (common_difference : ℤ)
  (h1 : common_difference = 2)
  (even_num_terms : ℕ)
  (h2 : ∃ k, even_num_terms = 2 * k)
  (sum_odd_terms sum_even_terms : ℤ)
  (h3 : sum_odd_terms = 15)
  (h4 : sum_even_terms = 35)
  (h5 : ∀ n, a n = a 0 + n * common_difference) :
  even_num_terms = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_arithmetic_sequence_is_20_l1648_164838


namespace NUMINAMATH_GPT_fourth_grade_students_l1648_164806

theorem fourth_grade_students (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ) 
  (h_initial : initial_students = 35) (h_left : students_left = 10) (h_new : new_students = 10) :
  initial_students - students_left + new_students = 35 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_fourth_grade_students_l1648_164806


namespace NUMINAMATH_GPT_curve_is_line_l1648_164817

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * Real.sin θ - Real.cos θ)) :
  ∃ (a b c : ℝ), a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 ∧
  (a, b, c) = (-1, 2, -1) := sorry

end NUMINAMATH_GPT_curve_is_line_l1648_164817


namespace NUMINAMATH_GPT_number_of_factors_27648_l1648_164813

-- Define the number in question
def n : ℕ := 27648

-- State the prime factorization
def n_prime_factors : Nat := 2^10 * 3^3

-- State the theorem to be proven
theorem number_of_factors_27648 : 
  ∃ (f : ℕ), 
  (f = (10+1) * (3+1)) ∧ (f = 44) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_number_of_factors_27648_l1648_164813


namespace NUMINAMATH_GPT_ABCD_eq_neg1_l1648_164844

noncomputable def A := (Real.sqrt 2013 + Real.sqrt 2012)
noncomputable def B := (- Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def C := (Real.sqrt 2013 - Real.sqrt 2012)
noncomputable def D := (Real.sqrt 2012 - Real.sqrt 2013)

theorem ABCD_eq_neg1 : A * B * C * D = -1 :=
by sorry

end NUMINAMATH_GPT_ABCD_eq_neg1_l1648_164844


namespace NUMINAMATH_GPT_percent_preferred_apples_l1648_164853

def frequencies : List ℕ := [75, 80, 45, 100, 50]
def frequency_apples : ℕ := 75
def total_frequency : ℕ := frequency_apples + frequencies[1] + frequencies[2] + frequencies[3] + frequencies[4]

theorem percent_preferred_apples :
  (frequency_apples * 100) / total_frequency = 21 := by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_percent_preferred_apples_l1648_164853


namespace NUMINAMATH_GPT_value_of_M_in_equation_l1648_164893

theorem value_of_M_in_equation :
  ∀ {M : ℕ}, (32 = 2^5) ∧ (8 = 2^3) → (32^3 * 8^4 = 2^M) → M = 27 :=
by
  intros M h1 h2
  sorry

end NUMINAMATH_GPT_value_of_M_in_equation_l1648_164893


namespace NUMINAMATH_GPT_idiom_describes_random_event_l1648_164860

-- Define the idioms as propositions.
def FishingForMoonInWater : Prop := ∀ (x : Type), x -> False
def CastlesInTheAir : Prop := ∀ (y : Type), y -> False
def WaitingByStumpForHare : Prop := ∃ (z : Type), True
def CatchingTurtleInJar : Prop := ∀ (w : Type), w -> False

-- Define the main theorem to state that WaitingByStumpForHare describes a random event.
theorem idiom_describes_random_event : WaitingByStumpForHare :=
  sorry

end NUMINAMATH_GPT_idiom_describes_random_event_l1648_164860


namespace NUMINAMATH_GPT_state_a_selection_percentage_l1648_164856

-- Definitions based on the conditions
variables {P : ℕ} -- percentage of candidates selected in State A

theorem state_a_selection_percentage 
  (candidates : ℕ) 
  (state_b_percentage : ℕ) 
  (extra_selected_in_b : ℕ) 
  (total_selected_in_b : ℕ) 
  (total_selected_in_a : ℕ)
  (appeared_in_each_state : ℕ) 
  (H1 : appeared_in_each_state = 8200)
  (H2 : state_b_percentage = 7)
  (H3 : extra_selected_in_b = 82)
  (H4 : total_selected_in_b = (state_b_percentage * appeared_in_each_state) / 100)
  (H5 : total_selected_in_a = total_selected_in_b - extra_selected_in_b)
  (H6 : total_selected_in_a = (P * appeared_in_each_state) / 100)
  : P = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_state_a_selection_percentage_l1648_164856


namespace NUMINAMATH_GPT_choose_two_out_of_three_l1648_164872

-- Define the number of vegetables as n and the number to choose as k
def n : ℕ := 3
def k : ℕ := 2

-- The combination formula C(n, k) == n! / (k! * (n - k)!)
def combination (n k : ℕ) : ℕ := n.choose k

-- Problem statement: Prove that the number of ways to choose 2 out of 3 vegetables is 3
theorem choose_two_out_of_three : combination n k = 3 :=
by
  sorry

end NUMINAMATH_GPT_choose_two_out_of_three_l1648_164872


namespace NUMINAMATH_GPT_range_of_t_l1648_164823

theorem range_of_t (a b t : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 2 * a + b = 1) 
    (h_ineq : 2 * Real.sqrt (a * b) - 4 * a^2 - b^2 ≤ t - 1 / 2):
    t = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_t_l1648_164823


namespace NUMINAMATH_GPT_eric_age_l1648_164852

theorem eric_age (B E : ℕ) (h1 : B = E + 4) (h2 : B + E = 28) : E = 12 :=
by
  sorry

end NUMINAMATH_GPT_eric_age_l1648_164852


namespace NUMINAMATH_GPT_average_weight_of_arun_l1648_164873

variable (weight : ℝ)

def arun_constraint := 61 < weight ∧ weight < 72
def brother_constraint := 60 < weight ∧ weight < 70
def mother_constraint := weight ≤ 64
def father_constraint := 62 < weight ∧ weight < 73
def sister_constraint := 59 < weight ∧ weight < 68

theorem average_weight_of_arun : 
  (∃ w : ℝ, arun_constraint w ∧ brother_constraint w ∧ mother_constraint w ∧ father_constraint w ∧ sister_constraint w) →
  (63.5 = (63 + 64) / 2) := 
by
  sorry

end NUMINAMATH_GPT_average_weight_of_arun_l1648_164873


namespace NUMINAMATH_GPT_minimum_value_of_sum_l1648_164886

variable (x y : ℝ)

theorem minimum_value_of_sum (hx : x > 0) (hy : y > 0) : ∃ x y, x > 0 ∧ y > 0 ∧ (x + 2 * y) = 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_sum_l1648_164886


namespace NUMINAMATH_GPT_sandwiches_cost_l1648_164840

theorem sandwiches_cost (sandwiches sodas : ℝ) 
  (cost_sandwich : ℝ := 2.44)
  (cost_soda : ℝ := 0.87)
  (num_sodas : ℕ := 4)
  (total_cost : ℝ := 8.36)
  (total_soda_cost : ℝ := cost_soda * num_sodas)
  (total_sandwich_cost : ℝ := total_cost - total_soda_cost):
  sandwiches = (total_sandwich_cost / cost_sandwich) → sandwiches = 2 := by 
  sorry

end NUMINAMATH_GPT_sandwiches_cost_l1648_164840


namespace NUMINAMATH_GPT_sales_worth_l1648_164845

variable (S : ℝ)
def old_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_remuneration S = old_remuneration S + 600 → S = 24000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sales_worth_l1648_164845


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l1648_164807

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), y = k * x - k ∧ x = 1 ∧ y = 0 :=
by
  use 1
  use 0
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l1648_164807


namespace NUMINAMATH_GPT_constant_term_expansion_l1648_164805

theorem constant_term_expansion :
  (∃ c : ℤ, ∀ x : ℝ, (2 * x - 1 / x) ^ 4 = c * x^0) ∧ c = 24 :=
by
  sorry

end NUMINAMATH_GPT_constant_term_expansion_l1648_164805


namespace NUMINAMATH_GPT_base_of_parallelogram_l1648_164818

theorem base_of_parallelogram (A h b : ℝ) (hA : A = 960) (hh : h = 16) :
  A = h * b → b = 60 :=
by
  sorry

end NUMINAMATH_GPT_base_of_parallelogram_l1648_164818


namespace NUMINAMATH_GPT_number_of_cookies_l1648_164842

def total_cake := 22
def total_chocolate := 16
def total_groceries := 42

theorem number_of_cookies :
  ∃ C : ℕ, total_groceries = total_cake + total_chocolate + C ∧ C = 4 := 
by
  sorry

end NUMINAMATH_GPT_number_of_cookies_l1648_164842


namespace NUMINAMATH_GPT_rational_includes_integers_and_fractions_l1648_164810

def is_integer (x : ℤ) : Prop := true
def is_fraction (x : ℚ) : Prop := true
def is_rational (x : ℚ) : Prop := true

theorem rational_includes_integers_and_fractions : 
  (∀ x : ℤ, is_integer x → is_rational (x : ℚ)) ∧ 
  (∀ x : ℚ, is_fraction x → is_rational x) :=
by {
  sorry -- Proof to be filled in
}

end NUMINAMATH_GPT_rational_includes_integers_and_fractions_l1648_164810


namespace NUMINAMATH_GPT_ellen_bakes_6_balls_of_dough_l1648_164897

theorem ellen_bakes_6_balls_of_dough (rising_time baking_time total_time : ℕ) (h_rise : rising_time = 3) (h_bake : baking_time = 2) (h_total : total_time = 20) :
  ∃ n : ℕ, (rising_time + baking_time) + rising_time * (n - 1) = total_time ∧ n = 6 :=
by sorry

end NUMINAMATH_GPT_ellen_bakes_6_balls_of_dough_l1648_164897


namespace NUMINAMATH_GPT_compute_fraction_value_l1648_164839

theorem compute_fraction_value : 2 + 3 / (4 + 5 / 6) = 76 / 29 := by
  sorry

end NUMINAMATH_GPT_compute_fraction_value_l1648_164839


namespace NUMINAMATH_GPT_u2008_is_5898_l1648_164894

-- Define the sequence as given in the problem.
def u (n : ℕ) : ℕ := sorry  -- The nth term of the sequence defined in the problem.

-- The main theorem stating u_{2008} = 5898.
theorem u2008_is_5898 : u 2008 = 5898 := sorry

end NUMINAMATH_GPT_u2008_is_5898_l1648_164894
