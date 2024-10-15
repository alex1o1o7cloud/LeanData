import Mathlib

namespace NUMINAMATH_GPT_number_of_cows_l75_7512

-- Define conditions
def total_bags_consumed_by_some_cows := 45
def bags_consumed_by_one_cow := 1

-- State the theorem to prove the number of cows
theorem number_of_cows (h1 : total_bags_consumed_by_some_cows = 45) (h2 : bags_consumed_by_one_cow = 1) : 
  total_bags_consumed_by_some_cows / bags_consumed_by_one_cow = 45 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_number_of_cows_l75_7512


namespace NUMINAMATH_GPT_remainder_twice_original_l75_7514

def findRemainder (N : ℕ) (D : ℕ) (r : ℕ) : ℕ :=
  2 * N % D

theorem remainder_twice_original
  (N : ℕ) (D : ℕ)
  (hD : D = 367)
  (hR : N % D = 241) :
  findRemainder N D 2 = 115 := by
  sorry

end NUMINAMATH_GPT_remainder_twice_original_l75_7514


namespace NUMINAMATH_GPT_water_level_drop_recording_l75_7562

theorem water_level_drop_recording (rise6_recorded: Int): 
    (rise6_recorded = 6) → (6 = -rise6_recorded) :=
by
  sorry

end NUMINAMATH_GPT_water_level_drop_recording_l75_7562


namespace NUMINAMATH_GPT_donation_to_first_home_l75_7591

theorem donation_to_first_home :
  let total_donation := 700
  let donation_to_second := 225
  let donation_to_third := 230
  total_donation - donation_to_second - donation_to_third = 245 :=
by
  sorry

end NUMINAMATH_GPT_donation_to_first_home_l75_7591


namespace NUMINAMATH_GPT_permutations_of_three_digit_numbers_from_set_l75_7543

theorem permutations_of_three_digit_numbers_from_set {digits : Finset ℕ} (h : digits = {1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = (Finset.card digits) * (Finset.card digits - 1) * (Finset.card digits - 2) ∧ n = 60 :=
by
  sorry

end NUMINAMATH_GPT_permutations_of_three_digit_numbers_from_set_l75_7543


namespace NUMINAMATH_GPT_evaluate_expression_l75_7595

variable (a b c d : ℝ)

theorem evaluate_expression :
  (a - (b - (c + d))) - ((a + b) - (c - d)) = -2 * b + 2 * c :=
sorry

end NUMINAMATH_GPT_evaluate_expression_l75_7595


namespace NUMINAMATH_GPT_norris_savings_l75_7526

theorem norris_savings:
  ∀ (N : ℕ), 
  (29 + 25 + N = 85) → N = 31 :=
by
  intros N h
  sorry

end NUMINAMATH_GPT_norris_savings_l75_7526


namespace NUMINAMATH_GPT_constant_term_exists_l75_7579

theorem constant_term_exists:
  ∃ (n : ℕ), 2 ≤ n ∧ n ≤ 10 ∧ 
  (∃ r : ℕ, n = 3 * r) ∧ (∃ k : ℕ, n = 2 * k) ∧ 
  n = 6 :=
sorry

end NUMINAMATH_GPT_constant_term_exists_l75_7579


namespace NUMINAMATH_GPT_total_votes_cast_l75_7588

/-- Define the conditions for Elvis's votes and percentage representation -/
def elvis_votes : ℕ := 45
def percentage_representation : ℚ := 1 / 4

/-- The main theorem that proves the total number of votes cast -/
theorem total_votes_cast : (elvis_votes: ℚ) / percentage_representation = 180 := by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l75_7588


namespace NUMINAMATH_GPT_train_speed_in_km_per_hr_l75_7583

-- Conditions
def time_in_seconds : ℕ := 9
def length_in_meters : ℕ := 175

-- Conversion factor from m/s to km/hr
def meters_per_sec_to_km_per_hr (speed_m_per_s : ℚ) : ℚ :=
  speed_m_per_s * 3.6

-- Question as statement
theorem train_speed_in_km_per_hr :
  meters_per_sec_to_km_per_hr ((length_in_meters : ℚ) / (time_in_seconds : ℚ)) = 70 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_km_per_hr_l75_7583


namespace NUMINAMATH_GPT_cucumbers_for_20_apples_l75_7576

theorem cucumbers_for_20_apples (A B C : ℝ) (h1 : 10 * A = 5 * B) (h2 : 3 * B = 4 * C) :
  20 * A = 40 / 3 * C :=
by
  sorry

end NUMINAMATH_GPT_cucumbers_for_20_apples_l75_7576


namespace NUMINAMATH_GPT_choose_most_suitable_l75_7572

def Survey := ℕ → Bool
structure Surveys :=
  (A B C D : Survey)
  (census_suitable : Survey)

theorem choose_most_suitable (s : Surveys) :
  s.census_suitable = s.C :=
sorry

end NUMINAMATH_GPT_choose_most_suitable_l75_7572


namespace NUMINAMATH_GPT_least_number_to_add_l75_7544

theorem least_number_to_add (x : ℕ) (h1 : (1789 + x) % 6 = 0) (h2 : (1789 + x) % 4 = 0) (h3 : (1789 + x) % 3 = 0) : x = 7 := 
sorry

end NUMINAMATH_GPT_least_number_to_add_l75_7544


namespace NUMINAMATH_GPT_rectangle_area_ratio_l75_7510

noncomputable def area_ratio (s : ℝ) : ℝ :=
  let area_square := s^2
  let longer_side := 1.15 * s
  let shorter_side := 0.95 * s
  let area_rectangle := longer_side * shorter_side
  area_rectangle / area_square

theorem rectangle_area_ratio (s : ℝ) : area_ratio s = 109.25 / 100 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_ratio_l75_7510


namespace NUMINAMATH_GPT_plastic_bag_estimation_l75_7549

theorem plastic_bag_estimation (a b c d e f : ℕ) (class_size : ℕ) (h1 : a = 33) 
  (h2 : b = 25) (h3 : c = 28) (h4 : d = 26) (h5 : e = 25) (h6 : f = 31) (h_class_size : class_size = 45) :
  let count := a + b + c + d + e + f
  let average := count / 6
  average * class_size = 1260 := by
{ 
  sorry 
}

end NUMINAMATH_GPT_plastic_bag_estimation_l75_7549


namespace NUMINAMATH_GPT_intersect_point_one_l75_7548

theorem intersect_point_one (k : ℝ) : 
  (∀ y : ℝ, (x = -3 * y^2 - 2 * y + 4 ↔ x = k)) ↔ k = 13 / 3 := 
by
  sorry

end NUMINAMATH_GPT_intersect_point_one_l75_7548


namespace NUMINAMATH_GPT_min_value_a_l75_7558

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, (3 * x - 5 * y) ≥ 0 → x > 0 → y > 0 → (1 - a) * x ^ 2 + 2 * x * y - a * y ^ 2 ≤ 0) ↔ a ≥ 55 / 34 := 
by 
  sorry

end NUMINAMATH_GPT_min_value_a_l75_7558


namespace NUMINAMATH_GPT_find_a8_l75_7537

-- Define the arithmetic sequence aₙ
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) := a₁ + (n - 1) * d

-- The given condition
def condition (a₁ d : ℕ) :=
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 15 = 12

-- The value we want to prove
def a₈ (a₁ d : ℕ ) : ℕ :=
  arithmetic_sequence a₁ d 8

theorem find_a8 (a₁ d : ℕ) (h : condition a₁ d) : a₈ a₁ d = 4 :=
  sorry

end NUMINAMATH_GPT_find_a8_l75_7537


namespace NUMINAMATH_GPT_handshake_remainder_l75_7500

noncomputable def handshakes (n : ℕ) (k : ℕ) : ℕ := sorry

theorem handshake_remainder :
  handshakes 12 3 % 1000 = 850 :=
sorry

end NUMINAMATH_GPT_handshake_remainder_l75_7500


namespace NUMINAMATH_GPT_y_at_x_equals_2sqrt3_l75_7592

theorem y_at_x_equals_2sqrt3 (k : ℝ) (y : ℝ → ℝ)
  (h1 : ∀ x : ℝ, y x = k * x^(1/3))
  (h2 : y 64 = 4 * Real.sqrt 3) :
  y 8 = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_y_at_x_equals_2sqrt3_l75_7592


namespace NUMINAMATH_GPT_max_number_of_cubes_l75_7596

theorem max_number_of_cubes (l w h v_cube : ℕ) (h_l : l = 8) (h_w : w = 9) (h_h : h = 12) (h_v_cube : v_cube = 27) :
  (l * w * h) / v_cube = 32 :=
by
  sorry

end NUMINAMATH_GPT_max_number_of_cubes_l75_7596


namespace NUMINAMATH_GPT_largest_divisor_of_m_p1_l75_7566

theorem largest_divisor_of_m_p1 (m : ℕ) (h1 : m > 0) (h2 : 72 ∣ m^3) : 6 ∣ m :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_m_p1_l75_7566


namespace NUMINAMATH_GPT_find_shortest_side_of_triangle_l75_7513

def Triangle (A B C : Type) := true -- Dummy definition for a triangle

structure Segments :=
(BD DE EC : ℝ)

def angle_ratios (AD AE : ℝ) (r1 r2 : ℕ) := true -- Dummy definition for angle ratios

def triangle_conditions (ABC : Type) (s : Segments) (r1 r2 : ℕ)
  (h1 : angle_ratios AD AE r1 r2)
  (h2 : s.BD = 4)
  (h3 : s.DE = 2)
  (h4 : s.EC = 5) : Prop := True

noncomputable def shortestSide (ABC : Type) (s : Segments) (r1 r2 : ℕ) : ℝ := 
  if true then sorry else 0 -- Placeholder for the shortest side length function

theorem find_shortest_side_of_triangle (ABC : Type) (s : Segments)
  (h1 : angle_ratios AD AE 2 3) (h2 : angle_ratios AE AD 1 1)
  (h3 : s.BD = 4) (h4 : s.DE = 2) (h5 : s.EC = 5) :
  shortestSide ABC s 2 3 = 30 / 11 :=
sorry

end NUMINAMATH_GPT_find_shortest_side_of_triangle_l75_7513


namespace NUMINAMATH_GPT_possible_values_a_l75_7536

-- Define the problem statement
theorem possible_values_a :
  (∃ a b c : ℤ, ∀ x : ℝ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) → (a = 1 ∨ a = 9) :=
by 
  -- Variable declaration and theorem body will be placed here
  sorry

end NUMINAMATH_GPT_possible_values_a_l75_7536


namespace NUMINAMATH_GPT_greatest_int_satisfying_inequality_l75_7582

theorem greatest_int_satisfying_inequality : ∃ n : ℤ, (∀ m : ℤ, m^2 - 13 * m + 40 ≤ 0 → m ≤ n) ∧ n = 8 := 
sorry

end NUMINAMATH_GPT_greatest_int_satisfying_inequality_l75_7582


namespace NUMINAMATH_GPT_M_plus_N_eq_2_l75_7554

noncomputable def M : ℝ := 1^5 + 2^4 * 3^3 - (4^2 / 5^1)
noncomputable def N : ℝ := 1^5 - 2^4 * 3^3 + (4^2 / 5^1)

theorem M_plus_N_eq_2 : M + N = 2 := by
  sorry

end NUMINAMATH_GPT_M_plus_N_eq_2_l75_7554


namespace NUMINAMATH_GPT_yankees_mets_ratio_l75_7540

-- Given conditions
def num_mets_fans : ℕ := 104
def total_fans : ℕ := 390
def ratio_mets_to_redsox : ℚ := 4 / 5

-- Definitions
def num_redsox_fans (M : ℕ) := (5 / 4) * M
def num_yankees_fans (Y M B : ℕ) := (total_fans - M - B)

-- Theorem statement
theorem yankees_mets_ratio (Y M B : ℕ)
  (h1 : M = num_mets_fans)
  (h2 : Y + M + B = total_fans)
  (h3 : (M : ℚ) / (B : ℚ) = ratio_mets_to_redsox) :
  (Y : ℚ) / (M : ℚ) = 3 / 2 :=
sorry

end NUMINAMATH_GPT_yankees_mets_ratio_l75_7540


namespace NUMINAMATH_GPT_min_value_y_l75_7553

noncomputable def y (x : ℝ) := x^4 - 4*x + 3

theorem min_value_y : ∃ x ∈ Set.Icc (-2 : ℝ) 3, y x = 0 ∧ ∀ x' ∈ Set.Icc (-2 : ℝ) 3, y x' ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_min_value_y_l75_7553


namespace NUMINAMATH_GPT_solve_speed_of_second_train_l75_7503

open Real

noncomputable def speed_of_second_train
  (L1 : ℝ) (L2 : ℝ) (S1 : ℝ) (T : ℝ) : ℝ :=
  let D := (L1 + L2) / 1000   -- Total distance in kilometers
  let H := T / 3600           -- Time in hours
  let relative_speed := D / H -- Relative speed in km/h
  relative_speed - S1         -- Speed of the second train

theorem solve_speed_of_second_train :
  speed_of_second_train 100 220 42 15.99872010239181 = 30 := by
  sorry

end NUMINAMATH_GPT_solve_speed_of_second_train_l75_7503


namespace NUMINAMATH_GPT_sum_solution_equation_l75_7532

theorem sum_solution_equation (n : ℚ) : (∃ x : ℚ, (n / x = 3 - n) ∧ (x = 1 / (n + (3 - n)))) → n = 3 / 4 := by
  intros h
  sorry

end NUMINAMATH_GPT_sum_solution_equation_l75_7532


namespace NUMINAMATH_GPT_rational_terms_count_l75_7556

noncomputable def number_of_rational_terms (n : ℕ) (x : ℝ) : ℕ :=
  -- The count of rational terms in the expansion
  17

theorem rational_terms_count (n : ℕ) (x : ℝ) :
  (number_of_rational_terms 100 x) = 17 := by
  sorry

end NUMINAMATH_GPT_rational_terms_count_l75_7556


namespace NUMINAMATH_GPT_odds_burning_out_during_second_period_l75_7538

def odds_burning_out_during_first_period := 1 / 3
def odds_not_burning_out_first_period := 1 - odds_burning_out_during_first_period
def odds_not_burning_out_next_period := odds_not_burning_out_first_period / 2

theorem odds_burning_out_during_second_period :
  (1 - odds_not_burning_out_next_period) = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_odds_burning_out_during_second_period_l75_7538


namespace NUMINAMATH_GPT_david_presents_l75_7535

variables (C B E : ℕ)

def total_presents (C B E : ℕ) : ℕ := C + B + E

theorem david_presents :
  C = 60 →
  B = 3 * E →
  E = (C / 2) - 10 →
  total_presents C B E = 140 :=
by
  intros hC hB hE
  sorry

end NUMINAMATH_GPT_david_presents_l75_7535


namespace NUMINAMATH_GPT_increasing_g_on_neg_l75_7530

variable {R : Type*} [LinearOrderedField R]

-- Assumptions: 
-- 1. f is an increasing function on R
-- 2. (h_neg : ∀ x : R, f x < 0)

theorem increasing_g_on_neg (f : R → R) (h_inc : ∀ x y : R, x < y → f x < f y) (h_neg : ∀ x : R, f x < 0) :
  ∀ x y : R, x < y → x < 0 → y < 0 → (x^2 * f x < y^2 * f y) :=
by
  sorry

end NUMINAMATH_GPT_increasing_g_on_neg_l75_7530


namespace NUMINAMATH_GPT_value_of_a_even_function_monotonicity_on_interval_l75_7593

noncomputable def f (x : ℝ) := (1 / x^2) + 0 * x

theorem value_of_a_even_function 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (-x) = f x) : 
  (∃ a : ℝ, ∀ x, f x = (1 / x^2) + a * x) → a = 0 := by
  -- Placeholder for the proof
  sorry

theorem monotonicity_on_interval 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = (1 / x^2) + 0 * x) 
  (h2 : ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2) : 
  ∀ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 → f x1 > f x2 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_value_of_a_even_function_monotonicity_on_interval_l75_7593


namespace NUMINAMATH_GPT_division_problem_l75_7517

theorem division_problem :
  250 / (5 + 12 * 3^2) = 250 / 113 :=
by sorry

end NUMINAMATH_GPT_division_problem_l75_7517


namespace NUMINAMATH_GPT_polynomial_roots_unique_b_c_l75_7542

theorem polynomial_roots_unique_b_c :
    ∀ (r : ℝ), (r ^ 2 - 2 * r - 1 = 0) → (r ^ 5 - 29 * r - 12 = 0) :=
by
    sorry

end NUMINAMATH_GPT_polynomial_roots_unique_b_c_l75_7542


namespace NUMINAMATH_GPT_steps_in_five_days_l75_7581

def steps_to_school : ℕ := 150
def daily_steps : ℕ := steps_to_school * 2
def days : ℕ := 5

theorem steps_in_five_days : daily_steps * days = 1500 := by
  sorry

end NUMINAMATH_GPT_steps_in_five_days_l75_7581


namespace NUMINAMATH_GPT_percentage_defective_units_shipped_l75_7589

noncomputable def defective_percent : ℝ := 0.07
noncomputable def shipped_percent : ℝ := 0.05

theorem percentage_defective_units_shipped :
  defective_percent * shipped_percent * 100 = 0.35 :=
by
  -- Proof body here
  sorry

end NUMINAMATH_GPT_percentage_defective_units_shipped_l75_7589


namespace NUMINAMATH_GPT_total_cantaloupes_l75_7539

def cantaloupes (fred : ℕ) (tim : ℕ) := fred + tim

theorem total_cantaloupes : cantaloupes 38 44 = 82 := by
  sorry

end NUMINAMATH_GPT_total_cantaloupes_l75_7539


namespace NUMINAMATH_GPT_sequence_formula_l75_7528

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 2) :
  ∀ n : ℕ, a n = 3^(n - 1) + 1 :=
by sorry

end NUMINAMATH_GPT_sequence_formula_l75_7528


namespace NUMINAMATH_GPT_contrapositive_of_x_squared_eq_one_l75_7527

theorem contrapositive_of_x_squared_eq_one (x : ℝ) 
  (h : x^2 = 1 → x = 1 ∨ x = -1) : (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_x_squared_eq_one_l75_7527


namespace NUMINAMATH_GPT_sam_age_l75_7598

variable (Sam Drew : ℕ)

theorem sam_age :
  (Sam + Drew = 54) →
  (Sam = Drew / 2) →
  Sam = 18 :=
by intros h1 h2; sorry

end NUMINAMATH_GPT_sam_age_l75_7598


namespace NUMINAMATH_GPT_ellipse_parameters_sum_l75_7507

def ellipse_sum (h k a b : ℝ) : ℝ :=
  h + k + a + b

theorem ellipse_parameters_sum :
  let h := 5
  let k := -3
  let a := 7
  let b := 4
  ellipse_sum h k a b = 13 := by
  sorry

end NUMINAMATH_GPT_ellipse_parameters_sum_l75_7507


namespace NUMINAMATH_GPT_engineer_last_name_is_smith_l75_7518

/-- Given these conditions:
 1. Businessman Robinson and a conductor live in Sheffield.
 2. Businessman Jones and a stoker live in Leeds.
 3. Businessman Smith and the railroad engineer live halfway between Leeds and Sheffield.
 4. The conductor’s namesake earns $10,000 a year.
 5. The engineer earns exactly 1/3 of what the businessman who lives closest to him earns.
 6. Railroad worker Smith beats the stoker at billiards.
 
We need to prove that the last name of the engineer is Smith. -/
theorem engineer_last_name_is_smith
  (lives_in_Sheffield_Robinson : Prop)
  (lives_in_Sheffield_conductor : Prop)
  (lives_in_Leeds_Jones : Prop)
  (lives_in_Leeds_stoker : Prop)
  (lives_in_halfway_Smith : Prop)
  (lives_in_halfway_engineer : Prop)
  (conductor_namesake_earns_10000 : Prop)
  (engineer_earns_one_third_closest_bizman : Prop)
  (railway_worker_Smith_beats_stoker_at_billiards : Prop) :
  (engineer_last_name = "Smith") :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_engineer_last_name_is_smith_l75_7518


namespace NUMINAMATH_GPT_greatest_possible_value_l75_7584

theorem greatest_possible_value (x : ℝ) : 
  (∃ (k : ℝ), k = (5 * x - 25) / (4 * x - 5) ∧ k^2 + k = 20) → x ≤ 2 := 
sorry

end NUMINAMATH_GPT_greatest_possible_value_l75_7584


namespace NUMINAMATH_GPT_number_of_lines_l75_7555

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the distances from the points
def d_A : ℝ := 1
def d_B : ℝ := 2

-- A theorem stating the number of lines under the given conditions
theorem number_of_lines (A B : ℝ × ℝ) (d_A d_B : ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) (hdA : d_A = 1) (hdB : d_B = 2) :
  ∃ n : ℕ, n = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_lines_l75_7555


namespace NUMINAMATH_GPT_max_integer_k_l75_7502

-- First, define the sequence a_n
def a (n : ℕ) : ℕ := n + 5

-- Define the sequence b_n given the recurrence relation and initial condition
def b (n : ℕ) : ℕ := 3 * n + 2

-- Define the sequence c_n
def c (n : ℕ) : ℚ := 3 / ((2 * a n - 11) * (2 * b n - 1))

-- Define the sum T_n of the first n terms of the sequence c_n
def T (n : ℕ) : ℚ := (1 / 2) * (1 - (1 / (2 * n + 1)))

-- The theorem to prove
theorem max_integer_k :
  ∃ k : ℕ, ∀ n : ℕ, n > 0 → T n > (k : ℚ) / 57 ∧ k = 18 :=
by
  sorry

end NUMINAMATH_GPT_max_integer_k_l75_7502


namespace NUMINAMATH_GPT_keychain_arrangement_count_l75_7531

-- Definitions of the keys
inductive Key
| house
| car
| office
| other1
| other2

-- Function to count the number of distinct arrangements on a keychain
noncomputable def distinct_keychain_arrangements : ℕ :=
  sorry -- This will be the placeholder for the proof

-- The ultimate theorem stating the solution
theorem keychain_arrangement_count : distinct_keychain_arrangements = 2 :=
  sorry -- This will be the placeholder for the proof

end NUMINAMATH_GPT_keychain_arrangement_count_l75_7531


namespace NUMINAMATH_GPT_complement_intersection_l75_7587

-- Define the universal set U.
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the set M.
def M : Set ℕ := {2, 3}

-- Define the set N.
def N : Set ℕ := {1, 3}

-- Define the complement of set M in U.
def complement_U_M : Set ℕ := {x ∈ U | x ∉ M}

-- Define the complement of set N in U.
def complement_U_N : Set ℕ := {x ∈ U | x ∉ N}

-- The statement to be proven.
theorem complement_intersection :
  (complement_U_M ∩ complement_U_N) = {4, 5, 6} :=
sorry

end NUMINAMATH_GPT_complement_intersection_l75_7587


namespace NUMINAMATH_GPT_inverse_prop_l75_7551

theorem inverse_prop (x : ℝ) : x < 0 → x^2 > 0 :=
by
  sorry

end NUMINAMATH_GPT_inverse_prop_l75_7551


namespace NUMINAMATH_GPT_not_periodic_l75_7567

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin x + Real.sin (a * x)

theorem not_periodic {a : ℝ} (ha : Irrational a) : ¬ ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f a (x + T) = f a x :=
  sorry

end NUMINAMATH_GPT_not_periodic_l75_7567


namespace NUMINAMATH_GPT_abs_eq_linear_eq_l75_7504

theorem abs_eq_linear_eq (x : ℝ) : (|x - 5| = 3 * x + 1) ↔ x = 1 := by
  sorry

end NUMINAMATH_GPT_abs_eq_linear_eq_l75_7504


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l75_7594

noncomputable def f (x : ℝ) : ℝ := x^4 - 3 * x^3 + 2 * x^2 + 11 * x - 6

theorem remainder_when_divided_by_x_minus_2 :
  (f 2) = 16 := by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l75_7594


namespace NUMINAMATH_GPT_find_larger_number_l75_7585

theorem find_larger_number (x y : ℝ) (h1 : 4 * y = 6 * x) (h2 : x + y = 36) : y = 21.6 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l75_7585


namespace NUMINAMATH_GPT_fixed_point_of_inverse_l75_7506

-- Define an odd function f on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f (x)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) := f (x + 1) - 2

-- Define the condition for a point to be on the inverse of a function
def inv_contains (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = f p.1

-- The theorem statement
theorem fixed_point_of_inverse (f : ℝ → ℝ) 
  (Hf_odd : odd_function f) :
  inv_contains (λ y => g f (y)) (-2, -1) :=
sorry

end NUMINAMATH_GPT_fixed_point_of_inverse_l75_7506


namespace NUMINAMATH_GPT_test_question_count_l75_7590

theorem test_question_count :
  ∃ (x : ℕ), 
    (20 / x: ℚ) > 0.60 ∧ 
    (20 / x: ℚ) < 0.70 ∧ 
    (4 ∣ x) ∧ 
    x = 32 := 
by
  sorry

end NUMINAMATH_GPT_test_question_count_l75_7590


namespace NUMINAMATH_GPT_average_brown_mms_l75_7520

def brown_mms_bag_1 := 9
def brown_mms_bag_2 := 12
def brown_mms_bag_3 := 8
def brown_mms_bag_4 := 8
def brown_mms_bag_5 := 3

def total_brown_mms : ℕ := brown_mms_bag_1 + brown_mms_bag_2 + brown_mms_bag_3 + brown_mms_bag_4 + brown_mms_bag_5

theorem average_brown_mms :
  (total_brown_mms / 5) = 8 := by
  rw [total_brown_mms]
  norm_num
  sorry

end NUMINAMATH_GPT_average_brown_mms_l75_7520


namespace NUMINAMATH_GPT_average_speed_round_trip_l75_7508

theorem average_speed_round_trip (m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (2 * m * n) / (m + n) = (2 * (m * n)) / (m + n) :=
  sorry

end NUMINAMATH_GPT_average_speed_round_trip_l75_7508


namespace NUMINAMATH_GPT_translate_line_upwards_l75_7573

theorem translate_line_upwards (x y y' : ℝ) (h : y = -2 * x) (t : y' = y + 4) : y' = -2 * x + 4 :=
by
  sorry

end NUMINAMATH_GPT_translate_line_upwards_l75_7573


namespace NUMINAMATH_GPT_right_isosceles_areas_l75_7578

theorem right_isosceles_areas (A B C : ℝ) (hA : A = 1 / 2 * 5 * 5) (hB : B = 1 / 2 * 12 * 12) (hC : C = 1 / 2 * 13 * 13) :
  A + B = C :=
by
  sorry

end NUMINAMATH_GPT_right_isosceles_areas_l75_7578


namespace NUMINAMATH_GPT_rectangle_square_problem_l75_7560

theorem rectangle_square_problem
  (m n x : ℕ)
  (h : 2 * (m + n) + 2 * x = m * n)
  (h2 : m * n - x^2 = 2 * (m + n)) :
  x = 2 ∧ ((m = 3 ∧ n = 10) ∨ (m = 6 ∧ n = 4)) :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_rectangle_square_problem_l75_7560


namespace NUMINAMATH_GPT_kim_total_water_drank_l75_7534

noncomputable def total_water_kim_drank : Float :=
  let water_from_bottle := 1.5 * 32
  let water_from_can := 12
  let shared_bottle := (3 / 5) * 32
  water_from_bottle + water_from_can + shared_bottle

theorem kim_total_water_drank :
  total_water_kim_drank = 79.2 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_kim_total_water_drank_l75_7534


namespace NUMINAMATH_GPT_fraction_of_red_knights_magical_l75_7563

def total_knights : ℕ := 28
def red_fraction : ℚ := 3 / 7
def magical_fraction : ℚ := 1 / 4
def red_magical_to_blue_magical_ratio : ℚ := 3

theorem fraction_of_red_knights_magical :
  let red_knights := red_fraction * total_knights
  let blue_knights := total_knights - red_knights
  let total_magical := magical_fraction * total_knights
  let red_magical_fraction := 21 / 52
  let blue_magical_fraction := red_magical_fraction / red_magical_to_blue_magical_ratio
  red_knights * red_magical_fraction + blue_knights * blue_magical_fraction = total_magical :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_red_knights_magical_l75_7563


namespace NUMINAMATH_GPT_new_class_mean_l75_7557

theorem new_class_mean :
  let students1 := 45
  let mean1 := 80
  let students2 := 4
  let mean2 := 85
  let students3 := 1
  let score3 := 90
  let total_students := students1 + students2 + students3
  let total_score := (students1 * mean1) + (students2 * mean2) + (students3 * score3)
  let class_mean := total_score / total_students
  class_mean = 80.6 := 
by
  sorry

end NUMINAMATH_GPT_new_class_mean_l75_7557


namespace NUMINAMATH_GPT_reassemble_square_with_hole_l75_7525

theorem reassemble_square_with_hole 
  (a b c d k1 k2 : ℝ)
  (h1 : a = b)
  (h2 : c = d)
  (h3 : k1 = k2) :
  ∃ (f gh ef gh' : ℝ), 
    f = a - c ∧
    gh = b - d ∧
    ef = f ∧
    gh' = gh := 
by sorry

end NUMINAMATH_GPT_reassemble_square_with_hole_l75_7525


namespace NUMINAMATH_GPT_division_in_base_5_l75_7571

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end NUMINAMATH_GPT_division_in_base_5_l75_7571


namespace NUMINAMATH_GPT_product_plus_one_square_l75_7586

theorem product_plus_one_square (n : ℕ):
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3 * n + 1)^2 := 
  sorry

end NUMINAMATH_GPT_product_plus_one_square_l75_7586


namespace NUMINAMATH_GPT_total_unique_working_games_l75_7552

-- Define the given conditions
def initial_games_from_friend := 25
def non_working_games_from_friend := 12

def games_from_garage_sale := 15
def non_working_games_from_garage_sale := 8
def duplicate_games := 3

-- Calculate the number of working games from each source
def working_games_from_friend := initial_games_from_friend - non_working_games_from_friend
def total_garage_sale_games := games_from_garage_sale - non_working_games_from_garage_sale
def unique_working_games_from_garage_sale := total_garage_sale_games - duplicate_games

-- Theorem statement
theorem total_unique_working_games : 
  working_games_from_friend + unique_working_games_from_garage_sale = 17 := by
  sorry

end NUMINAMATH_GPT_total_unique_working_games_l75_7552


namespace NUMINAMATH_GPT_find_interest_rate_l75_7523

-- conditions
def P : ℝ := 6200
def t : ℕ := 10

def interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * t
def I : ℝ := P - 3100

-- problem statement
theorem find_interest_rate (r : ℝ) :
  interest P r t = I → r = 0.05 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l75_7523


namespace NUMINAMATH_GPT_find_value_of_a_l75_7568

theorem find_value_of_a (a : ℝ) (h : 2 - a = 0) : a = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_value_of_a_l75_7568


namespace NUMINAMATH_GPT_money_distribution_l75_7505

theorem money_distribution (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 310) (h3 : C = 10) : A + B + C = 500 :=
by
  sorry

end NUMINAMATH_GPT_money_distribution_l75_7505


namespace NUMINAMATH_GPT_one_angle_not_greater_than_60_l75_7564

theorem one_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := 
sorry

end NUMINAMATH_GPT_one_angle_not_greater_than_60_l75_7564


namespace NUMINAMATH_GPT_principal_amount_l75_7521

theorem principal_amount (SI R T : ℕ) (P : ℕ) : SI = 160 ∧ R = 5 ∧ T = 4 → P = 800 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l75_7521


namespace NUMINAMATH_GPT_largest_divisor_l75_7515

theorem largest_divisor (A B : ℕ) (h : 24 = A * B + 4) : A ≤ 20 :=
sorry

end NUMINAMATH_GPT_largest_divisor_l75_7515


namespace NUMINAMATH_GPT_students_taking_one_language_l75_7575

-- Definitions based on the conditions
def french_class_students : ℕ := 21
def spanish_class_students : ℕ := 21
def both_languages_students : ℕ := 6
def total_students : ℕ := french_class_students + spanish_class_students - both_languages_students

-- The theorem we want to prove
theorem students_taking_one_language :
    total_students = 36 :=
by
  -- Add the proof here
  sorry

end NUMINAMATH_GPT_students_taking_one_language_l75_7575


namespace NUMINAMATH_GPT_tangent_parallel_to_line_l75_7599

def f (x : ℝ) : ℝ := x ^ 3 + x - 2

theorem tangent_parallel_to_line (x y : ℝ) : 
  (y = 4 * x - 1) ↔ (x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = -4) :=
by
  sorry

end NUMINAMATH_GPT_tangent_parallel_to_line_l75_7599


namespace NUMINAMATH_GPT_calculate_weight_of_first_batch_jelly_beans_l75_7546

theorem calculate_weight_of_first_batch_jelly_beans (J : ℝ)
    (h1 : 16 = 8 * (J * 4)) : J = 2 := 
  sorry

end NUMINAMATH_GPT_calculate_weight_of_first_batch_jelly_beans_l75_7546


namespace NUMINAMATH_GPT_range_of_set_l75_7529

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end NUMINAMATH_GPT_range_of_set_l75_7529


namespace NUMINAMATH_GPT_P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l75_7561

-- Conditions
def center_C : (ℝ × ℝ) := (6, 8)
def radius : ℝ := 10
def circle_eq (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 100
def origin_O : (ℝ × ℝ) := (0, 0)

-- (a) Point of intersection of the circle with the x-axis
def point_P : (ℝ × ℝ) := (12, 0)
theorem P_on_x_axis : circle_eq (point_P.1) (point_P.2) ∧ point_P.2 = 0 := sorry

-- (b) Point on the circle with maximum y-coordinate
def point_Q : (ℝ × ℝ) := (6, 18)
theorem Q_max_y : circle_eq (point_Q.1) (point_Q.2) ∧ ∀ y : ℝ, (circle_eq 6 y → y ≤ 18) := sorry

-- (c) Point on the circle such that ∠PQR = 90°
def point_R : (ℝ × ℝ) := (0, 16)
theorem PQR_90_deg : circle_eq (point_R.1) (point_R.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧ (point_P.1 - point_R.1) * (Q.1 - point_Q.1) + (point_P.2 - point_R.2) * (Q.2 - point_Q.2) = 0 := sorry

-- (d) Two points on the circle such that ∠PQS = ∠PQT = 45°
def point_S : (ℝ × ℝ) := (14, 14)
def point_T : (ℝ × ℝ) := (-2, 2)
theorem PQS_PQT_45_deg : circle_eq (point_S.1) (point_S.2) ∧ circle_eq (point_T.1) (point_T.2) ∧
  ∃ Q : (ℝ × ℝ), circle_eq (Q.1) (Q.2) ∧ (Q = point_Q) ∧
  ((point_P.1 - Q.1) * (point_S.1 - Q.1) + (point_P.2 - Q.2) * (point_S.2 - Q.2) =
  (point_P.1 - Q.1) * (point_T.1 - Q.1) + (point_P.2 - Q.2) * (point_T.2 - Q.2)) := sorry

end NUMINAMATH_GPT_P_on_x_axis_Q_max_y_PQR_90_deg_PQS_PQT_45_deg_l75_7561


namespace NUMINAMATH_GPT_hyperbola_equation_l75_7509

theorem hyperbola_equation (c : ℝ) (b a : ℝ) 
  (h₁ : c = 2 * Real.sqrt 5) 
  (h₂ : a^2 + b^2 = c^2) 
  (h₃ : b / a = 1 / 2) : 
  (x y : ℝ) → (x^2 / 16) - (y^2 / 4) = 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_equation_l75_7509


namespace NUMINAMATH_GPT_natasha_average_speed_l75_7574

theorem natasha_average_speed :
  ∀ (time_up time_down : ℝ) (speed_up : ℝ),
  time_up = 4 →
  time_down = 2 →
  speed_up = 2.25 →
  (2 * (time_up * speed_up) / (time_up + time_down) = 3) :=
by
  intros time_up time_down speed_up h_time_up h_time_down h_speed_up
  rw [h_time_up, h_time_down, h_speed_up]
  sorry

end NUMINAMATH_GPT_natasha_average_speed_l75_7574


namespace NUMINAMATH_GPT_sum_smallest_largest_eq_2y_l75_7533

theorem sum_smallest_largest_eq_2y (n : ℕ) (y a : ℕ) 
  (h1 : 2 * a + 2 * (n - 1) / n = y) : 
  2 * y = (2 * a + 2 * (n - 1)) := 
sorry

end NUMINAMATH_GPT_sum_smallest_largest_eq_2y_l75_7533


namespace NUMINAMATH_GPT_probability_not_win_l75_7550

theorem probability_not_win (n : ℕ) (h : 1 - 1 / (n : ℝ) = 0.9375) : n = 16 :=
sorry

end NUMINAMATH_GPT_probability_not_win_l75_7550


namespace NUMINAMATH_GPT_circle_equation_l75_7597

theorem circle_equation (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 4) :
    x^2 + y^2 - 2 * x - 3 = 0 :=
sorry

end NUMINAMATH_GPT_circle_equation_l75_7597


namespace NUMINAMATH_GPT_chick_hit_count_l75_7522

theorem chick_hit_count :
  ∃ x y z : ℕ,
    9 * x + 5 * y + 2 * z = 61 ∧
    x + y + z = 10 ∧
    x ≥ 1 ∧
    y ≥ 1 ∧
    z ≥ 1 ∧
    x = 5 :=
by
  sorry

end NUMINAMATH_GPT_chick_hit_count_l75_7522


namespace NUMINAMATH_GPT_integer_for_all_n_l75_7511

theorem integer_for_all_n
  (x y : ℝ)
  (f : ℕ → ℤ)
  (h : ∀ n : ℕ, 1 ≤ n ∧ n ≤ 4 → f n = ((x^n - y^n) / (x - y))) :
  ∀ n : ℕ, 0 < n → f n = ((x^n - y^n) / (x - y)) :=
by sorry

end NUMINAMATH_GPT_integer_for_all_n_l75_7511


namespace NUMINAMATH_GPT_inequality_solution_l75_7516

/-- Define conditions and state the corresponding theorem -/
theorem inequality_solution (a x : ℝ) (h : a < 0) : ax - 1 > 0 ↔ x < 1 / a :=
by sorry

end NUMINAMATH_GPT_inequality_solution_l75_7516


namespace NUMINAMATH_GPT_correct_operation_l75_7547

variable {R : Type*} [CommRing R] (x y : R)

theorem correct_operation : x * (1 + y) = x + x * y :=
by sorry

end NUMINAMATH_GPT_correct_operation_l75_7547


namespace NUMINAMATH_GPT_function_properties_l75_7519

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a*x + b

theorem function_properties (a b : ℝ) (h : (a - 1) ^ 2 - 4 * b < 0) : 
  (∀ x : ℝ, f x a b > x) ∧ (∀ x : ℝ, f (f x a b) a b > x) ∧ (a + b > 0) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l75_7519


namespace NUMINAMATH_GPT_simplify_expression_l75_7524

theorem simplify_expression : 3000 * 3000^3000 = 3000^(3001) := 
by 
  sorry

end NUMINAMATH_GPT_simplify_expression_l75_7524


namespace NUMINAMATH_GPT_smallest_n_such_that_no_n_digit_is_11_power_l75_7565

theorem smallest_n_such_that_no_n_digit_is_11_power (log_11 : Real) (h : log_11 = 1.0413) : 
  ∃ n > 1, ∀ k : ℕ, ¬ (10 ^ (n - 1) ≤ 11 ^ k ∧ 11 ^ k < 10 ^ n) :=
sorry

end NUMINAMATH_GPT_smallest_n_such_that_no_n_digit_is_11_power_l75_7565


namespace NUMINAMATH_GPT_extreme_value_f_range_of_a_l75_7501

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3
noncomputable def h (x : ℝ) : ℝ := 2 * Real.log x + x + 3 / x

theorem extreme_value_f : ∃ x, f x = -1 / Real.exp 1 :=
by sorry

theorem range_of_a (a : ℝ) : (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
by sorry

end NUMINAMATH_GPT_extreme_value_f_range_of_a_l75_7501


namespace NUMINAMATH_GPT_fraction_of_blue_cars_l75_7577

-- Definitions of the conditions
def total_cars : ℕ := 516
def red_cars : ℕ := total_cars / 2
def black_cars : ℕ := 86
def blue_cars : ℕ := total_cars - (red_cars + black_cars)

-- Statement to prove that the fraction of blue cars is 1/3
theorem fraction_of_blue_cars :
  (blue_cars : ℚ) / total_cars = 1 / 3 :=
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_fraction_of_blue_cars_l75_7577


namespace NUMINAMATH_GPT_problem_part1_problem_part2_l75_7541

open Real

variables {α : ℝ}

theorem problem_part1 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) :
  sin α * cos α = 3 / 10 := sorry

theorem problem_part2 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) (h3 : sin α * cos α = 3 / 10) :
  sin α + cos α = - (2 * sqrt 10 / 5) := sorry

end NUMINAMATH_GPT_problem_part1_problem_part2_l75_7541


namespace NUMINAMATH_GPT_best_fitting_model_l75_7570

theorem best_fitting_model :
  ∀ R1 R2 R3 R4 : ℝ, 
  R1 = 0.21 → R2 = 0.80 → R3 = 0.50 → R4 = 0.98 → 
  abs (R4 - 1) < abs (R1 - 1) ∧ abs (R4 - 1) < abs (R2 - 1) 
    ∧ abs (R4 - 1) < abs (R3 - 1) :=
by
  intros R1 R2 R3 R4 hR1 hR2 hR3 hR4
  rw [hR1, hR2, hR3, hR4]
  exact sorry

end NUMINAMATH_GPT_best_fitting_model_l75_7570


namespace NUMINAMATH_GPT_tan_double_angle_l75_7580

theorem tan_double_angle (theta : ℝ) (h : 2 * Real.sin (Real.pi / 2 + theta) + Real.sin (Real.pi + theta) = 0) :
  Real.tan (2 * theta) = -4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l75_7580


namespace NUMINAMATH_GPT_race_participants_minimum_l75_7559

theorem race_participants_minimum : ∃ (n : ℕ), 
  (∃ (x : ℕ), n = 3 * x + 1) ∧ 
  (∃ (y : ℕ), n = 4 * y + 1) ∧ 
  (∃ (z : ℕ), n = 5 * z + 1) ∧ 
  n = 61 :=
by
  sorry

end NUMINAMATH_GPT_race_participants_minimum_l75_7559


namespace NUMINAMATH_GPT_percentage_increase_l75_7545

theorem percentage_increase (total_capacity : ℝ) (additional_water : ℝ) (percentage_capacity : ℝ) (current_water : ℝ) : 
    additional_water + current_water = percentage_capacity * total_capacity →
    percentage_capacity = 0.70 →
    total_capacity = 1857.1428571428573 →
    additional_water = 300 →
    current_water = ((percentage_capacity * total_capacity) - additional_water) →
    (additional_water / current_water) * 100 = 30 :=
by
    sorry

end NUMINAMATH_GPT_percentage_increase_l75_7545


namespace NUMINAMATH_GPT_hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l75_7569

-- Definitions for conditions of the problem
def ellipse_C1 (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def is_sister_conic_section (e1 e2 : ℝ) : Prop :=
  e1 * e2 = Real.sqrt 15 / 4

def hyperbola_C2 (x y : ℝ) : Prop := (x^2) / 4 - y^2 = 1

variable {b : ℝ} (hb : 0 < b ∧ b < 2)
variable {e1 e2 : ℝ} (heccentricities : is_sister_conic_section e1 e2)

theorem hyperbola_C2_equation :
  ∃ (x y : ℝ), ellipse_C1 x y b → hyperbola_C2 x y := sorry

theorem constant_ratio_kAM_kBN (G : ℝ × ℝ) :
  G = (4,0) → 
  ∀ (M N : ℝ × ℝ) (kAM kBN : ℝ), 
  (kAM / kBN = -1/3) := sorry

theorem range_of_w_kAM_kBN (kAM kBN : ℝ) :
  ∃ (w : ℝ),
  w = kAM^2 + (2 / 3) * kBN →
  (w ∈ Set.Icc (-3 / 4) (-11 / 36) ∪ Set.Icc (13 / 36) (5 / 4)) := sorry

end NUMINAMATH_GPT_hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l75_7569
