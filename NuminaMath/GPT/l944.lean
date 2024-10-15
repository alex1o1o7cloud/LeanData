import Mathlib

namespace NUMINAMATH_GPT_fountain_pen_price_l944_94487

theorem fountain_pen_price
  (n_fpens : ℕ) (n_mpens : ℕ) (total_cost : ℕ) (avg_cost_mpens : ℝ)
  (hpens : n_fpens = 450) (mpens : n_mpens = 3750) 
  (htotal : total_cost = 11250) (havg_mpens : avg_cost_mpens = 2.25) : 
  (total_cost - n_mpens * avg_cost_mpens) / n_fpens = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_fountain_pen_price_l944_94487


namespace NUMINAMATH_GPT_exists_num_with_digit_sum_div_by_11_l944_94496

-- Helper function to sum the digits of a natural number
def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Main theorem statement
theorem exists_num_with_digit_sum_div_by_11 (N : ℕ) :
  ∃ k : ℕ, k < 39 ∧ (digit_sum (N + k)) % 11 = 0 :=
sorry

end NUMINAMATH_GPT_exists_num_with_digit_sum_div_by_11_l944_94496


namespace NUMINAMATH_GPT_dima_story_telling_l944_94407

theorem dima_story_telling (initial_spoons final_spoons : ℕ) 
  (h1 : initial_spoons = 26) (h2 : final_spoons = 33696)
  (h3 : ∃ (n : ℕ), final_spoons = initial_spoons * (2^5 * 3^4) * 13) : 
  ∃ n : ℕ, n = 9 := 
sorry

end NUMINAMATH_GPT_dima_story_telling_l944_94407


namespace NUMINAMATH_GPT_bureaucrats_total_l944_94449

-- Define the parameters and conditions as stated in the problem
variables (a b c : ℕ)

-- Conditions stated in the problem
def condition_1 : Prop :=
  ∀ (i j : ℕ) (h1 : i ≠ j), 
    (10 * a * b = 10 * a * c ∧ 10 * b * c = 10 * a * b)

-- The main goal: proving the total number of bureaucrats
theorem bureaucrats_total (h1 : a = b) (h2 : b = c) (h3 : condition_1 a b c) : 
  3 * a = 120 :=
by sorry

end NUMINAMATH_GPT_bureaucrats_total_l944_94449


namespace NUMINAMATH_GPT_calculate_expression_l944_94478

theorem calculate_expression :
  |(-Real.sqrt 3)| - (1/3)^(-1/2 : ℝ) + 2 / (Real.sqrt 3 - 1) - 12^(1/2 : ℝ) = 1 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l944_94478


namespace NUMINAMATH_GPT_ratio_of_time_charged_l944_94442

theorem ratio_of_time_charged (P K M : ℕ) (r : ℚ) 
  (h1 : P + K + M = 144) 
  (h2 : P = r * K)
  (h3 : P = 1/3 * M)
  (h4 : M = K + 80) : 
  r = 2 := 
  sorry

end NUMINAMATH_GPT_ratio_of_time_charged_l944_94442


namespace NUMINAMATH_GPT_sum_of_numbers_l944_94432

theorem sum_of_numbers : 217 + 2.017 + 0.217 + 2.0017 = 221.2357 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l944_94432


namespace NUMINAMATH_GPT_can_measure_all_weights_l944_94465

def weights : List ℕ := [1, 3, 9, 27]

theorem can_measure_all_weights :
  (∀ n, 1 ≤ n ∧ n ≤ 40 → ∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = n) ∧ 
  (∃ (a b c d : ℕ), a * 1 + b * 3 + c * 9 + d * 27 = 40) :=
by
  sorry

end NUMINAMATH_GPT_can_measure_all_weights_l944_94465


namespace NUMINAMATH_GPT_find_number_l944_94474

noncomputable def S (x : ℝ) : ℝ :=
  -- Assuming S(x) is a non-trivial function that sums the digits
  sorry

theorem find_number (x : ℝ) (hx_nonzero : x ≠ 0) (h_cond : x = (S x) / 5) : x = 1.8 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l944_94474


namespace NUMINAMATH_GPT_average_age_of_team_l944_94458

theorem average_age_of_team
    (A : ℝ)
    (captain_age : ℝ)
    (wicket_keeper_age : ℝ)
    (bowlers_count : ℝ)
    (batsmen_count : ℝ)
    (team_members_count : ℝ)
    (avg_bowlers_age : ℝ)
    (avg_batsmen_age : ℝ)
    (total_age_team : ℝ) :
    captain_age = 28 →
    wicket_keeper_age = 31 →
    bowlers_count = 5 →
    batsmen_count = 4 →
    avg_bowlers_age = A - 2 →
    avg_batsmen_age = A + 3 →
    total_age_team = 28 + 31 + 5 * (A - 2) + 4 * (A + 3) →
    team_members_count * A = total_age_team →
    team_members_count = 11 →
    A = 30.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_age_of_team_l944_94458


namespace NUMINAMATH_GPT_scatter_plot_variable_placement_l944_94492

theorem scatter_plot_variable_placement
  (forecast explanatory : Type)
  (scatter_plot : explanatory → forecast → Prop) : 
  ∀ (x : explanatory) (y : forecast), scatter_plot x y → (True -> True) := 
by
  intros x y h
  sorry

end NUMINAMATH_GPT_scatter_plot_variable_placement_l944_94492


namespace NUMINAMATH_GPT_greatest_prime_factor_of_341_l944_94488

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem greatest_prime_factor_of_341 : ∃ p, is_prime p ∧ p ∣ 341 ∧ ∀ q, is_prime q → q ∣ 341 → q ≤ p :=
by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_of_341_l944_94488


namespace NUMINAMATH_GPT_total_amount_proof_l944_94470

-- Definitions of the base 8 numbers
def silks_base8 := 5267
def stones_base8 := 6712
def spices_base8 := 327

-- Conversion function from base 8 to base 10
def base8_to_base10 (n : ℕ) : ℕ := sorry -- Assume this function converts a base 8 number to base 10

-- Converted values
def silks_base10 := base8_to_base10 silks_base8
def stones_base10 := base8_to_base10 stones_base8
def spices_base10 := base8_to_base10 spices_base8

-- Total amount calculation in base 10
def total_amount_base10 := silks_base10 + stones_base10 + spices_base10

-- The theorem that we want to prove
theorem total_amount_proof : total_amount_base10 = 6488 :=
by
  -- The proof is omitted here.
  sorry

end NUMINAMATH_GPT_total_amount_proof_l944_94470


namespace NUMINAMATH_GPT_hayden_ironing_weeks_l944_94491

variable (total_daily_minutes : Nat := 5 + 3)
variable (days_per_week : Nat := 5)
variable (total_minutes : Nat := 160)

def calculate_weeks (total_daily_minutes : Nat) (days_per_week : Nat) (total_minutes : Nat) : Nat :=
  total_minutes / (total_daily_minutes * days_per_week)

theorem hayden_ironing_weeks :
  calculate_weeks (5 + 3) 5 160 = 4 := 
by
  sorry

end NUMINAMATH_GPT_hayden_ironing_weeks_l944_94491


namespace NUMINAMATH_GPT_common_difference_value_l944_94499

-- Define the arithmetic sequence and the sum of the first n terms
def sum_of_arithmetic_sequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  (n * (2 * a1 + (n - 1) * d)) / 2

-- Define the given condition in terms of the arithmetic sequence
def given_condition (a1 d : ℚ) : Prop :=
  (sum_of_arithmetic_sequence a1 d 2017) / 2017 - (sum_of_arithmetic_sequence a1 d 17) / 17 = 100

-- Prove the common difference d is 1/10 given the condition
theorem common_difference_value (a1 d : ℚ) :
  given_condition a1 d → d = 1/10 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_value_l944_94499


namespace NUMINAMATH_GPT_intersection_points_l944_94483

open Real

def parabola1 (x : ℝ) : ℝ := x^2 - 3 * x + 2
def parabola2 (x : ℝ) : ℝ := 3 * x^2 + 2 * x - 1

theorem intersection_points : 
  ∃ x y : ℝ, 
  (parabola1 x = y ∧ parabola2 x = y) ∧ 
  ((x = 1/2 ∧ y = 3/4) ∨ (x = -3 ∧ y = 20)) :=
by sorry

end NUMINAMATH_GPT_intersection_points_l944_94483


namespace NUMINAMATH_GPT_arithmetic_seq_n_possible_values_l944_94403

theorem arithmetic_seq_n_possible_values
  (a1 : ℕ) (a_n : ℕ → ℕ) (d : ℕ) (n : ℕ):
  a1 = 1 → 
  (∀ n, n ≥ 3 → a_n n = 100) → 
  (∃ d : ℕ, ∀ n, n ≥ 3 → a_n n = a1 + (n - 1) * d) → 
  (n = 4 ∨ n = 10 ∨ n = 12 ∨ n = 34 ∨ n = 100) := by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_n_possible_values_l944_94403


namespace NUMINAMATH_GPT_systematic_sampling_eighth_group_l944_94434

theorem systematic_sampling_eighth_group
  (total_employees : ℕ)
  (target_sample : ℕ)
  (third_group_value : ℕ)
  (group_count : ℕ)
  (common_difference : ℕ)
  (eighth_group_value : ℕ) :
  total_employees = 840 →
  target_sample = 42 →
  third_group_value = 44 →
  group_count = total_employees / target_sample →
  common_difference = group_count →
  eighth_group_value = third_group_value + (8 - 3) * common_difference →
  eighth_group_value = 144 :=
sorry

end NUMINAMATH_GPT_systematic_sampling_eighth_group_l944_94434


namespace NUMINAMATH_GPT_find_AC_find_angle_A_l944_94476

noncomputable def triangle_AC (AB BC : ℝ) (sinC_over_sinB : ℝ) : ℝ :=
  if h : sinC_over_sinB = 3 / 5 ∧ AB = 3 ∧ BC = 7 then 5 else 0

noncomputable def triangle_angle_A (AB AC BC : ℝ) : ℝ :=
  if h : AB = 3 ∧ AC = 5 ∧ BC = 7 then 120 else 0

theorem find_AC (BC AB : ℝ) (sinC_over_sinB : ℝ) (h : BC = 7 ∧ AB = 3 ∧ sinC_over_sinB = 3 / 5) : 
  triangle_AC AB BC sinC_over_sinB = 5 := by
  sorry

theorem find_angle_A (BC AB AC : ℝ) (h : BC = 7 ∧ AB = 3 ∧ AC = 5) : 
  triangle_angle_A AB AC BC = 120 := by
  sorry

end NUMINAMATH_GPT_find_AC_find_angle_A_l944_94476


namespace NUMINAMATH_GPT_average_balance_correct_l944_94481

-- Define the monthly balances
def january_balance : ℕ := 120
def february_balance : ℕ := 240
def march_balance : ℕ := 180
def april_balance : ℕ := 180
def may_balance : ℕ := 210
def june_balance : ℕ := 300

-- List of all balances
def balances : List ℕ := [january_balance, february_balance, march_balance, april_balance, may_balance, june_balance]

-- Define the function to calculate the average balance
def average_balance (balances : List ℕ) : ℕ :=
  (balances.sum / balances.length)

-- Define the target average balance
def target_average_balance : ℕ := 205

-- The theorem we need to prove
theorem average_balance_correct :
  average_balance balances = target_average_balance :=
by
  sorry

end NUMINAMATH_GPT_average_balance_correct_l944_94481


namespace NUMINAMATH_GPT_sum_from_one_to_twelve_l944_94454

-- Define the sum of an arithmetic series
def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Theorem stating the sum of numbers from 1 to 12
theorem sum_from_one_to_twelve : sum_arithmetic_series 12 1 12 = 78 := by
  sorry

end NUMINAMATH_GPT_sum_from_one_to_twelve_l944_94454


namespace NUMINAMATH_GPT_dots_not_visible_l944_94473

-- Define the sum of numbers on a single die
def sum_die_faces : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Define the sum of numbers on four dice
def total_dots_on_four_dice : ℕ := 4 * sum_die_faces

-- List the visible numbers
def visible_numbers : List ℕ := [1, 2, 2, 3, 3, 4, 5, 5, 6]

-- Calculate the sum of visible numbers
def sum_visible_numbers : ℕ := (visible_numbers.sum)

-- Define the math proof problem
theorem dots_not_visible : total_dots_on_four_dice - sum_visible_numbers = 53 := by
  sorry

end NUMINAMATH_GPT_dots_not_visible_l944_94473


namespace NUMINAMATH_GPT_power_sum_l944_94461

theorem power_sum : 1^234 + 4^6 / 4^4 = 17 :=
by
  sorry

end NUMINAMATH_GPT_power_sum_l944_94461


namespace NUMINAMATH_GPT_algebraic_expression_value_l944_94464

theorem algebraic_expression_value (x : ℝ) :
  let a := 2003 * x + 2001
  let b := 2003 * x + 2002
  let c := 2003 * x + 2003
  a^2 + b^2 + c^2 - a * b - a * c - b * c = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l944_94464


namespace NUMINAMATH_GPT_mod_sum_l944_94484

theorem mod_sum : 
  (5432 + 5433 + 5434 + 5435) % 7 = 2 := 
by
  sorry

end NUMINAMATH_GPT_mod_sum_l944_94484


namespace NUMINAMATH_GPT_product_decrease_increase_fifteenfold_l944_94463

theorem product_decrease_increase_fifteenfold (a1 a2 a3 a4 a5 : ℕ) :
  ((a1 - 3) * (a2 - 3) * (a3 - 3) * (a4 - 3) * (a5 - 3) = 15 * a1 * a2 * a3 * a4 * a5) → true :=
by
  sorry

end NUMINAMATH_GPT_product_decrease_increase_fifteenfold_l944_94463


namespace NUMINAMATH_GPT_truck_transportation_l944_94494

theorem truck_transportation
  (x y t : ℕ) 
  (h1 : xt - yt = 60)
  (h2 : (x - 4) * (t + 10) = xt)
  (h3 : (y - 3) * (t + 10) = yt)
  (h4 : xt = x * t)
  (h5 : yt = y * t) : 
  x - 4 = 8 ∧ y - 3 = 6 ∧ t + 10 = 30 := 
by
  sorry

end NUMINAMATH_GPT_truck_transportation_l944_94494


namespace NUMINAMATH_GPT_agency_comparison_l944_94410

variable (days m : ℝ)

theorem agency_comparison (h : 20.25 * days + 0.14 * m < 18.25 * days + 0.22 * m) : m > 25 * days :=
by
  sorry

end NUMINAMATH_GPT_agency_comparison_l944_94410


namespace NUMINAMATH_GPT_degree_measure_OC1D_l944_94419

/-- Define points on the sphere -/
structure Point (latitude longitude : ℝ) :=
(lat : ℝ := latitude)
(long : ℝ := longitude)

noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

noncomputable def angle_OC1D : ℝ :=
  Real.arccos ((cos_deg 44) * (cos_deg (-123)))

/-- The main theorem: the degree measure of ∠OC₁D is 113 -/
theorem degree_measure_OC1D :
  angle_OC1D = 113 := sorry

end NUMINAMATH_GPT_degree_measure_OC1D_l944_94419


namespace NUMINAMATH_GPT_avg_math_chem_l944_94406

variables (M P C : ℕ)

def total_marks (M P : ℕ) := M + P = 50
def chemistry_marks (P C : ℕ) := C = P + 20

theorem avg_math_chem (M P C : ℕ) (h1 : total_marks M P) (h2 : chemistry_marks P C) :
  (M + C) / 2 = 35 :=
by
  sorry

end NUMINAMATH_GPT_avg_math_chem_l944_94406


namespace NUMINAMATH_GPT_compare_neg_fractions_l944_94414

theorem compare_neg_fractions : - (3 / 5 : ℚ) < - (1 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_compare_neg_fractions_l944_94414


namespace NUMINAMATH_GPT_arithmetic_geometric_means_l944_94457

theorem arithmetic_geometric_means (a b : ℝ) 
  (h1 : a + b = 40) 
  (h2 : a * b = 110) : 
  a^2 + b^2 = 1380 :=
sorry

end NUMINAMATH_GPT_arithmetic_geometric_means_l944_94457


namespace NUMINAMATH_GPT_peter_takes_last_stone_l944_94447

theorem peter_takes_last_stone (n : ℕ) (h : ∀ p, Nat.Prime p → p < n) :
  ∃ P, ∀ stones: ℕ, stones > n^2 → (∃ k : ℕ, 
  ((k = 1 ∨ (∃ p : ℕ, Nat.Prime p ∧ p < n ∧ k = p) ∨ (∃ m : ℕ, k = m * n)) ∧
  stones ≥ k ∧ stones - k > n^2) →
  P = stones - k) := 
sorry

end NUMINAMATH_GPT_peter_takes_last_stone_l944_94447


namespace NUMINAMATH_GPT_math_problem_l944_94453

noncomputable def f (x : ℝ) := (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8)

theorem math_problem : f 6 = 43264 := by
  sorry

end NUMINAMATH_GPT_math_problem_l944_94453


namespace NUMINAMATH_GPT_sum_of_xyz_l944_94471

theorem sum_of_xyz (x y z : ℝ) 
  (h1 : 1/x + y + z = 3) 
  (h2 : x + 1/y + z = 3) 
  (h3 : x + y + 1/z = 3) : 
  ∃ m n : ℕ, m = 9 ∧ n = 2 ∧ Nat.gcd m n = 1 ∧ 100 * m + n = 902 := 
sorry

end NUMINAMATH_GPT_sum_of_xyz_l944_94471


namespace NUMINAMATH_GPT_systematic_sampling_probability_l944_94477

/-- Given a population of 1002 individuals, if we remove 2 randomly and then pick 50 out of the remaining 1000, then the probability of picking each individual is 50/1002. 
This is because the process involves two independent steps: not being removed initially and then being chosen in the sample of size 50. --/
theorem systematic_sampling_probability :
  let population_size := 1002
  let removal_count := 2
  let sample_size := 50
  ∀ p : ℕ, p = 50 / (1002 : ℚ) := sorry

end NUMINAMATH_GPT_systematic_sampling_probability_l944_94477


namespace NUMINAMATH_GPT_ratio_male_whales_l944_94446

def num_whales_first_trip_males : ℕ := 28
def num_whales_first_trip_females : ℕ := 56
def num_whales_second_trip_babies : ℕ := 8
def num_whales_second_trip_parents_males : ℕ := 8
def num_whales_second_trip_parents_females : ℕ := 8
def num_whales_third_trip_females : ℕ := 56
def total_whales : ℕ := 178

theorem ratio_male_whales (M : ℕ) (ratio : ℕ × ℕ) 
  (h_total_whales : num_whales_first_trip_males + num_whales_first_trip_females 
    + num_whales_second_trip_babies + num_whales_second_trip_parents_males 
    + num_whales_second_trip_parents_females + M + num_whales_third_trip_females = total_whales) 
  (h_ratio : ratio = ((M : ℕ) / Nat.gcd M num_whales_first_trip_males, 
                       num_whales_first_trip_males / Nat.gcd M num_whales_first_trip_males)) 
  : ratio = (1, 2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_male_whales_l944_94446


namespace NUMINAMATH_GPT_karen_group_size_l944_94426

theorem karen_group_size (total_students : ℕ) (zack_group_size number_of_groups : ℕ) (karen_group_size : ℕ) (h1 : total_students = 70) (h2 : zack_group_size = 14) (h3 : number_of_groups = total_students / zack_group_size) (h4 : number_of_groups = total_students / karen_group_size) : karen_group_size = 14 :=
by
  sorry

end NUMINAMATH_GPT_karen_group_size_l944_94426


namespace NUMINAMATH_GPT_probability_point_in_circle_l944_94423

theorem probability_point_in_circle (r : ℝ) (h: r = 2) :
  let side_length := 2 * r
  let area_square := side_length ^ 2
  let area_circle := Real.pi * r ^ 2
  (area_circle / area_square) = Real.pi / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_point_in_circle_l944_94423


namespace NUMINAMATH_GPT_additional_days_needed_is_15_l944_94408

-- Definitions and conditions from the problem statement
def good_days_2013 : ℕ := 365 * 479 / 100  -- Number of good air quality days in 2013
def target_increase : ℕ := 20              -- Target increase in percentage for 2014
def additional_days_first_half_2014 : ℕ := 20 -- Additional good air quality days in first half of 2014 compared to 2013
def half_good_days_2013 : ℕ := good_days_2013 / 2 -- Good air quality days in first half of 2013

-- Target number of good air quality days for 2014
def target_days_2014 : ℕ := good_days_2013 * (100 + target_increase) / 100

-- Good air quality days in the first half of 2014
def good_days_first_half_2014 : ℕ := half_good_days_2013 + additional_days_first_half_2014

-- Additional good air quality days needed in the second half of 2014
def additional_days_2014_second_half (target_days good_days_first_half_2014 : ℕ) : ℕ := 
  target_days - good_days_first_half_2014 - half_good_days_2013

-- Final theorem verifying the number of additional days needed in the second half of 2014 is 15
theorem additional_days_needed_is_15 : 
  additional_days_2014_second_half target_days_2014 good_days_first_half_2014 = 15 :=
sorry

end NUMINAMATH_GPT_additional_days_needed_is_15_l944_94408


namespace NUMINAMATH_GPT_sequence_sum_square_l944_94428

-- Definition of the sum of the symmetric sequence.
def sequence_sum (n : ℕ) : ℕ :=
  (List.range' 1 (n+1)).sum + (List.range' 1 n).sum

-- The conjecture that the sum of the sequence equals n^2.
theorem sequence_sum_square (n : ℕ) : sequence_sum n = n^2 := by
  sorry

end NUMINAMATH_GPT_sequence_sum_square_l944_94428


namespace NUMINAMATH_GPT_ab_divisibility_l944_94438

theorem ab_divisibility (a b : ℕ) (h_a : a ≥ 2) (h_b : b ≥ 2) : 
  (ab - 1) % ((a - 1) * (b - 1)) = 0 ↔ (a = 2 ∧ b = 2) ∨ (a = 3 ∧ b = 3) :=
sorry

end NUMINAMATH_GPT_ab_divisibility_l944_94438


namespace NUMINAMATH_GPT_ratio_of_a_to_b_l944_94456

variable (a b x m : ℝ)
variable (h_a_pos : a > 0) (h_b_pos : b > 0)
variable (h_x : x = 1.25 * a) (h_m : m = 0.6 * b)
variable (h_ratio : m / x = 0.6)

theorem ratio_of_a_to_b (h_x : x = 1.25 * a) (h_m : m = 0.6 * b) (h_ratio : m / x = 0.6) : a / b = 0.8 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_b_l944_94456


namespace NUMINAMATH_GPT_opposite_of_neg_one_third_l944_94402

theorem opposite_of_neg_one_third : (-(-1/3)) = (1/3) := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_one_third_l944_94402


namespace NUMINAMATH_GPT_triangle_height_l944_94445

theorem triangle_height (x y : ℝ) :
  let area := (x^3 * y)^2
  let base := (2 * x * y)^2
  base ≠ 0 →
  (2 * area) / base = x^4 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l944_94445


namespace NUMINAMATH_GPT_cab_driver_income_l944_94413

theorem cab_driver_income (x : ℕ)
  (h1 : 50 + 60 + 65 + 70 + x = 5 * 58) :
  x = 45 :=
by
  sorry

end NUMINAMATH_GPT_cab_driver_income_l944_94413


namespace NUMINAMATH_GPT_remainder_mod_8_l944_94466

theorem remainder_mod_8 (x : ℤ) (h : x % 63 = 25) : x % 8 = 1 := 
sorry

end NUMINAMATH_GPT_remainder_mod_8_l944_94466


namespace NUMINAMATH_GPT_proj_a_b_l944_94440

open Real

def vector (α : Type*) := (α × α)

noncomputable def dot_product (a b: vector ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

noncomputable def magnitude (v: vector ℝ) : ℝ := sqrt (v.1^2 + v.2^2)

noncomputable def projection (a b: vector ℝ) : ℝ := (dot_product a b) / (magnitude b)

-- Define the vectors a and b
def a : vector ℝ := (-1, 3)
def b : vector ℝ := (3, 4)

-- The projection of a in the direction of b
theorem proj_a_b : projection a b = 9 / 5 := 
  by sorry

end NUMINAMATH_GPT_proj_a_b_l944_94440


namespace NUMINAMATH_GPT_odometer_problem_l944_94424

theorem odometer_problem
    (x a b c : ℕ)
    (h_dist : 60 * x = (100 * b + 10 * c + a) - (100 * a + 10 * b + c))
    (h_b_ge_1 : b ≥ 1)
    (h_sum_le_9 : a + b + c ≤ 9) :
    a^2 + b^2 + c^2 = 29 :=
sorry

end NUMINAMATH_GPT_odometer_problem_l944_94424


namespace NUMINAMATH_GPT_find_y_l944_94404

theorem find_y (x y : ℝ) (h₁ : x^2 - 2 * x + 5 = y + 3) (h₂ : x = 5) : y = 17 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l944_94404


namespace NUMINAMATH_GPT_calc_f_18_48_l944_94451

def f (x y : ℕ) : ℕ := sorry

axiom f_self (x : ℕ) : f x x = x
axiom f_symm (x y : ℕ) : f x y = f y x
axiom f_third_cond (x y : ℕ) : (x + y) * f x y = x * f x (x + y)

theorem calc_f_18_48 : f 18 48 = 48 := sorry

end NUMINAMATH_GPT_calc_f_18_48_l944_94451


namespace NUMINAMATH_GPT_simplify_expression_l944_94455

theorem simplify_expression : (2^3002 * 3^3004) / 6^3003 = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l944_94455


namespace NUMINAMATH_GPT_find_x_l944_94421

-- Let \( x \) be a real number.
variable (x : ℝ)

-- Condition given in the problem.
def condition : Prop := x = (3 / 7) * x + 200

-- The main statement to be proved.
theorem find_x (h : condition x) : x = 350 :=
  sorry

end NUMINAMATH_GPT_find_x_l944_94421


namespace NUMINAMATH_GPT_remove_green_balls_l944_94427

theorem remove_green_balls (total_balls green_balls yellow_balls x : ℕ) 
  (h1 : total_balls = 600)
  (h2 : green_balls = 420)
  (h3 : yellow_balls = 180)
  (h4 : green_balls = 70 * total_balls / 100)
  (h5 : yellow_balls = total_balls - green_balls)
  (h6 : (green_balls - x) = 60 * (total_balls - x) / 100) :
  x = 150 := 
by {
  -- sorry placeholder for proof.
  sorry
}

end NUMINAMATH_GPT_remove_green_balls_l944_94427


namespace NUMINAMATH_GPT_sqrt_infinite_series_eq_two_l944_94412

theorem sqrt_infinite_series_eq_two (m : ℝ) (hm : 0 < m) :
  (m ^ 2 = 2 + m) → m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_sqrt_infinite_series_eq_two_l944_94412


namespace NUMINAMATH_GPT_total_flour_l944_94486

theorem total_flour (original_flour extra_flour : Real) (h_orig : original_flour = 7.0) (h_extra : extra_flour = 2.0) : original_flour + extra_flour = 9.0 :=
sorry

end NUMINAMATH_GPT_total_flour_l944_94486


namespace NUMINAMATH_GPT_viewers_watching_program_A_l944_94479

theorem viewers_watching_program_A (T : ℕ) (hT : T = 560) (x : ℕ)
  (h_ratio : 1 * x + (2 * x - x) + (3 * x - x) = T) : 2 * x = 280 :=
by
  -- by solving the given equation, we find x = 140
  -- substituting x = 140 in 2 * x gives 2 * x = 280
  sorry

end NUMINAMATH_GPT_viewers_watching_program_A_l944_94479


namespace NUMINAMATH_GPT_algebraic_expression_value_l944_94472

namespace MathProof

variables {α β : ℝ} 

-- Given conditions
def is_root (a : ℝ) : Prop := a^2 - a - 1 = 0
def roots_of_quadratic (α β : ℝ) : Prop := is_root α ∧ is_root β

-- The proof problem statement
theorem algebraic_expression_value (h : roots_of_quadratic α β) : α^2 + α * (β^2 - 2) = 0 := 
by sorry

end MathProof

end NUMINAMATH_GPT_algebraic_expression_value_l944_94472


namespace NUMINAMATH_GPT_jill_and_bob_payment_l944_94431

-- Definitions of the conditions
def price_of_first_house (X : ℝ) := X
def price_of_second_house (Y X : ℝ) := 2 * X

theorem jill_and_bob_payment :
  ∃ X, ∃ Y, Y = 2 * X ∧ X + Y = 600000 ∧ X = 200000 :=
by
  sorry

end NUMINAMATH_GPT_jill_and_bob_payment_l944_94431


namespace NUMINAMATH_GPT_total_tourists_proof_l944_94480

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

end NUMINAMATH_GPT_total_tourists_proof_l944_94480


namespace NUMINAMATH_GPT_polynomial_simplification_l944_94493

variable (x : ℝ)

theorem polynomial_simplification : 
  ((3 * x - 2) * (5 * x ^ 12 + 3 * x ^ 11 - 4 * x ^ 9 + x ^ 8)) = 
  (15 * x ^ 13 - x ^ 12 - 6 * x ^ 11 - 12 * x ^ 10 + 11 * x ^ 9 - 2 * x ^ 8) := by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l944_94493


namespace NUMINAMATH_GPT_tiger_speed_l944_94420

variable (v_t : ℝ) (hours_head_start : ℝ := 5) (hours_zebra_to_catch : ℝ := 6) (speed_zebra : ℝ := 55)

-- Define the distance covered by the tiger and the zebra
def distance_tiger (v_t : ℝ) (hours : ℝ) : ℝ := v_t * hours
def distance_zebra (hours : ℝ) (speed_zebra : ℝ) : ℝ := speed_zebra * hours

theorem tiger_speed :
  v_t * hours_head_start + v_t * hours_zebra_to_catch = distance_zebra hours_zebra_to_catch speed_zebra →
  v_t = 30 :=
by
  sorry

end NUMINAMATH_GPT_tiger_speed_l944_94420


namespace NUMINAMATH_GPT_slices_left_l944_94400

variable (total_pieces: ℕ) (joe_fraction: ℚ) (darcy_fraction: ℚ)
variable (carl_fraction: ℚ) (emily_fraction: ℚ)

theorem slices_left 
  (h1 : total_pieces = 24)
  (h2 : joe_fraction = 1/3)
  (h3 : darcy_fraction = 1/4)
  (h4 : carl_fraction = 1/6)
  (h5 : emily_fraction = 1/8) :
  total_pieces - (total_pieces * joe_fraction + total_pieces * darcy_fraction + total_pieces * carl_fraction + total_pieces * emily_fraction) = 3 := 
  by 
  sorry

end NUMINAMATH_GPT_slices_left_l944_94400


namespace NUMINAMATH_GPT_fraction_value_l944_94444

theorem fraction_value : (5 * 7 : ℝ) / 10 = 3.5 := by
  sorry

end NUMINAMATH_GPT_fraction_value_l944_94444


namespace NUMINAMATH_GPT_find_divisor_l944_94422

variable (r q d v : ℕ)
variable (h1 : r = 8)
variable (h2 : q = 43)
variable (h3 : d = 997)

theorem find_divisor : d = v * q + r → v = 23 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l944_94422


namespace NUMINAMATH_GPT_max_b_value_l944_94467

theorem max_b_value (a b c : ℕ) (h_volume : a * b * c = 360) (h_conditions : 1 < c ∧ c < b ∧ b < a) : b = 12 :=
  sorry

end NUMINAMATH_GPT_max_b_value_l944_94467


namespace NUMINAMATH_GPT_total_asphalt_used_1520_tons_l944_94435

noncomputable def asphalt_used (L W : ℕ) (asphalt_per_100m2 : ℕ) : ℕ :=
  (L * W / 100) * asphalt_per_100m2

theorem total_asphalt_used_1520_tons :
  asphalt_used 800 50 3800 = 1520000 := by
  sorry

end NUMINAMATH_GPT_total_asphalt_used_1520_tons_l944_94435


namespace NUMINAMATH_GPT_sum_of_three_numbers_l944_94462

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h : a + (b * c) = (a + b) * (a + c)) : a + b + c = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l944_94462


namespace NUMINAMATH_GPT_sum_of_n_values_l944_94460

theorem sum_of_n_values (sum_n : ℕ) : (∀ n : ℕ, 0 < n ∧ 24 % (2 * n - 1) = 0) → sum_n = 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_n_values_l944_94460


namespace NUMINAMATH_GPT_pieces_cut_from_rod_l944_94468

theorem pieces_cut_from_rod (rod_length_m : ℝ) (piece_length_cm : ℝ) (rod_length_cm_eq : rod_length_m * 100 = 4250) (piece_length_eq : piece_length_cm = 85) :
  (4250 / 85) = 50 :=
by sorry

end NUMINAMATH_GPT_pieces_cut_from_rod_l944_94468


namespace NUMINAMATH_GPT_find_a7_l944_94436

variable {a : ℕ → ℝ}

-- Conditions
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

axiom a3_eq_4 : a 3 = 4
axiom harmonic_condition : (1 / a 1 + 1 / a 5 = 5 / 8)
axiom increasing_geometric : is_increasing_geometric_sequence a

-- The problem is to prove that a 7 = 16 given the above conditions.
theorem find_a7 : a 7 = 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_find_a7_l944_94436


namespace NUMINAMATH_GPT_num_triangles_l944_94489

def vertices := 10
def chosen_vertices := 3

theorem num_triangles : (Nat.choose vertices chosen_vertices) = 120 := by
  sorry

end NUMINAMATH_GPT_num_triangles_l944_94489


namespace NUMINAMATH_GPT_range_of_a_l944_94417

def p (a m : ℝ) : Prop := m^2 + 12 * a^2 < 7 * a * m ∧ a > 0
def q (m : ℝ) : Prop := 1 < m ∧ m < 3 / 2

theorem range_of_a (a : ℝ) :
  (∀ m : ℝ, p a m → q m) → 
  (∃ (a_lower a_upper : ℝ), a_lower ≤ a ∧ a ≤ a_upper ∧ a_lower = 1 / 3 ∧ a_upper = 3 / 8) :=
sorry

end NUMINAMATH_GPT_range_of_a_l944_94417


namespace NUMINAMATH_GPT_kevin_total_miles_l944_94411

theorem kevin_total_miles : 
  ∃ (d1 d2 d3 d4 d5 : ℕ), 
  d1 = 60 / 6 ∧ 
  d2 = 60 / (6 + 6 * 1) ∧ 
  d3 = 60 / (6 + 6 * 2) ∧ 
  d4 = 60 / (6 + 6 * 3) ∧ 
  d5 = 60 / (6 + 6 * 4) ∧ 
  (d1 + d2 + d3 + d4 + d5) = 13 := 
by
  sorry

end NUMINAMATH_GPT_kevin_total_miles_l944_94411


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l944_94475

theorem sum_of_other_endpoint_coordinates (x y : ℝ) (hx : (x + 5) / 2 = 3) (hy : (y - 2) / 2 = 4) :
  x + y = 11 :=
sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l944_94475


namespace NUMINAMATH_GPT_part1_solution_set_part2_range_a_l944_94416

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3} = {x : ℝ | x ≤ -3 / 2} ∪ {x : ℝ | x ≥ 3 / 2} := 
sorry

theorem part2_range_a (a : ℝ) : 
  (∀ x : ℝ, f x ≥ a^2 - a) ↔ (-1 ≤ a ∧ a ≤ 2) := 
sorry

end NUMINAMATH_GPT_part1_solution_set_part2_range_a_l944_94416


namespace NUMINAMATH_GPT_optimal_hospital_location_l944_94485

-- Define the coordinates for points A, B, and C
def A : ℝ × ℝ := (0, 12)
def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

-- Define the distance function
def dist_sq (p q : ℝ × ℝ) : ℝ := (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the statement to be proved: minimizing sum of squares of distances
theorem optimal_hospital_location : ∃ y : ℝ, 
  (∀ (P : ℝ × ℝ), P = (0, y) → (dist_sq P A + dist_sq P B + dist_sq P C) = 146) ∧ y = 4 :=
by sorry

end NUMINAMATH_GPT_optimal_hospital_location_l944_94485


namespace NUMINAMATH_GPT_frac_equiv_l944_94429

-- Define the given values of x and y.
def x : ℚ := 2 / 7
def y : ℚ := 8 / 11

-- Define the statement to prove.
theorem frac_equiv : (7 * x + 11 * y) / (77 * x * y) = 5 / 8 :=
by
  -- The proof will go here (use 'sorry' for now)
  sorry

end NUMINAMATH_GPT_frac_equiv_l944_94429


namespace NUMINAMATH_GPT_problem_statement_l944_94452

theorem problem_statement (m n : ℝ) 
  (h₁ : m^2 - 1840 * m + 2009 = 0)
  (h₂ : n^2 - 1840 * n + 2009 = 0) : 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 :=
sorry

end NUMINAMATH_GPT_problem_statement_l944_94452


namespace NUMINAMATH_GPT_area_of_rectangle_is_108_l944_94448

-- Define the conditions and parameters
variables (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
variable (isTangentToSides : Prop)
variable (centersFormLineParallelToLongerSide : Prop)

-- Assume the given conditions
axiom h1 : diameter = 6
axiom h2 : isTangentToSides
axiom h3 : centersFormLineParallelToLongerSide

-- Define the goal to prove
theorem area_of_rectangle_is_108 (P Q R S : Type) (diameter : ℝ) (height : ℝ) (width : ℝ) (area : ℝ)
    (isTangentToSides : Prop) (centersFormLineParallelToLongerSide : Prop)
    (h1 : diameter = 6)
    (h2 : isTangentToSides)
    (h3 : centersFormLineParallelToLongerSide) :
    area = 108 :=
by
  -- Lean code requires an actual proof here, but for now, we'll use sorry.
  sorry

end NUMINAMATH_GPT_area_of_rectangle_is_108_l944_94448


namespace NUMINAMATH_GPT_triangle_side_range_l944_94490

theorem triangle_side_range (x : ℝ) (hx1 : 8 + 10 > x) (hx2 : 10 + x > 8) (hx3 : x + 8 > 10) : 2 < x ∧ x < 18 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_range_l944_94490


namespace NUMINAMATH_GPT_angle_in_fourth_quadrant_l944_94439

theorem angle_in_fourth_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 270 < 360 - α ∧ 360 - α < 360 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_fourth_quadrant_l944_94439


namespace NUMINAMATH_GPT_percentage_increase_of_x_compared_to_y_l944_94450

-- We are given that y = 0.5 * z and x = 0.6 * z
-- We need to prove that the percentage increase of x compared to y is 20%

theorem percentage_increase_of_x_compared_to_y (x y z : ℝ) 
  (h1 : y = 0.5 * z) 
  (h2 : x = 0.6 * z) : 
  (x / y - 1) * 100 = 20 :=
by 
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_percentage_increase_of_x_compared_to_y_l944_94450


namespace NUMINAMATH_GPT_find_f_neg2007_l944_94433

variable (f : ℝ → ℝ)

-- Conditions
axiom cond1 (x y w : ℝ) (hx : x > y) (hw : f x + x ≥ w ∧ w ≥ f y + y) : 
  ∃ z ∈ Set.Icc y x, f z = w - z

axiom cond2 : ∃ u, f u = 0 ∧ ∀ v, f v = 0 → u ≤ v

axiom cond3 : f 0 = 1

axiom cond4 : f (-2007) ≤ 2008

axiom cond5 (x y : ℝ) : f x * f y = f (x * f y + y * f x + x * y)

theorem find_f_neg2007 : f (-2007) = 2008 := 
sorry

end NUMINAMATH_GPT_find_f_neg2007_l944_94433


namespace NUMINAMATH_GPT_combined_weight_loss_l944_94495

theorem combined_weight_loss (a_weekly_loss : ℝ) (a_weeks : ℕ) (x_weekly_loss : ℝ) (x_weeks : ℕ)
  (h1 : a_weekly_loss = 1.5) (h2 : a_weeks = 10) (h3 : x_weekly_loss = 2.5) (h4 : x_weeks = 8) :
  a_weekly_loss * a_weeks + x_weekly_loss * x_weeks = 35 := 
by
  -- We will not provide the proof body; the goal is to ensure the statement compiles.
  sorry

end NUMINAMATH_GPT_combined_weight_loss_l944_94495


namespace NUMINAMATH_GPT_number_is_minus_72_l944_94482

noncomputable def find_number (x : ℝ) : Prop :=
  0.833 * x = -60

theorem number_is_minus_72 : ∃ x : ℝ, find_number x ∧ x = -72 :=
by
  sorry

end NUMINAMATH_GPT_number_is_minus_72_l944_94482


namespace NUMINAMATH_GPT_interval_length_l944_94443

theorem interval_length (c : ℝ) (h : ∀ x : ℝ, 3 ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ c → 
                             (3 * (x) + 4 ≤ c ∧ 3 ≤ 3 * x + 4)) :
  (∃ c : ℝ, ((c - 4) / 3) - ((-1) / 3) = 15) → (c - 3 = 45) :=
sorry

end NUMINAMATH_GPT_interval_length_l944_94443


namespace NUMINAMATH_GPT_pieces_per_box_l944_94497

theorem pieces_per_box 
  (a : ℕ) -- Adam bought 13 boxes of chocolate candy 
  (g : ℕ) -- Adam gave 7 boxes to his little brother 
  (p : ℕ) -- Adam still has 36 pieces 
  (n : ℕ) (b : ℕ) 
  (h₁ : a = 13) 
  (h₂ : g = 7) 
  (h₃ : p = 36) 
  (h₄ : n = a - g) 
  (h₅ : p = n * b) 
  : b = 6 :=
by 
  sorry

end NUMINAMATH_GPT_pieces_per_box_l944_94497


namespace NUMINAMATH_GPT_ratio_unit_price_l944_94418

theorem ratio_unit_price
  (v : ℝ) (p : ℝ) (h_v : v > 0) (h_p : p > 0)
  (vol_A : ℝ := 1.25 * v)
  (price_A : ℝ := 0.85 * p) :
  (price_A / vol_A) / (p / v) = 17 / 25 :=
by
  sorry

end NUMINAMATH_GPT_ratio_unit_price_l944_94418


namespace NUMINAMATH_GPT_solve_inequality_l944_94459

theorem solve_inequality (a x : ℝ) :
  (a - x) * (x - 1) < 0 ↔
  (a > 1 ∧ (x < 1 ∨ x > a)) ∨
  (a < 1 ∧ (x < a ∨ x > 1)) ∨
  (a = 1 ∧ x ≠ 1) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l944_94459


namespace NUMINAMATH_GPT_other_juice_cost_l944_94401

theorem other_juice_cost (total_spent : ℕ := 94)
    (mango_cost_per_glass : ℕ := 5)
    (other_total_spent : ℕ := 54)
    (total_people : ℕ := 17) : 
  other_total_spent / (total_people - (total_spent - other_total_spent) / mango_cost_per_glass) = 6 := 
sorry

end NUMINAMATH_GPT_other_juice_cost_l944_94401


namespace NUMINAMATH_GPT_difference_of_squares_l944_94425

theorem difference_of_squares (x y : ℕ) (h₁ : x + y = 22) (h₂ : x * y = 120) (h₃ : x > y) : 
  x^2 - y^2 = 44 :=
sorry

end NUMINAMATH_GPT_difference_of_squares_l944_94425


namespace NUMINAMATH_GPT_greatest_integer_inequality_l944_94415

theorem greatest_integer_inequality : 
  ⌊ (3 ^ 100 + 2 ^ 100 : ℝ) / (3 ^ 96 + 2 ^ 96) ⌋ = 80 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_inequality_l944_94415


namespace NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l944_94469

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : (ℝ × ℝ × ℝ) :=
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z)

theorem spherical_to_rectangular_coordinates :
  sphericalToRectangular 10 (5 * Real.pi / 4) (Real.pi / 4) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_spherical_to_rectangular_coordinates_l944_94469


namespace NUMINAMATH_GPT_ineq_x4_y4_l944_94409

theorem ineq_x4_y4 (x y : ℝ) (h1 : x > Real.sqrt 2) (h2 : y > Real.sqrt 2) :
    x^4 - x^3 * y + x^2 * y^2 - x * y^3 + y^4 > x^2 + y^2 :=
by
  sorry

end NUMINAMATH_GPT_ineq_x4_y4_l944_94409


namespace NUMINAMATH_GPT_minimum_value_f_maximum_value_f_l944_94437

-- Problem 1: Minimum value of f(x) = 12/x + 3x for x > 0
theorem minimum_value_f (x : ℝ) (h : x > 0) : 
  (12 / x + 3 * x) ≥ 12 :=
sorry

-- Problem 2: Maximum value of f(x) = x(1 - 3x) for 0 < x < 1/3
theorem maximum_value_f (x : ℝ) (h1 : 0 < x) (h2 : x < 1 / 3) :
  x * (1 - 3 * x) ≤ 1 / 12 :=
sorry

end NUMINAMATH_GPT_minimum_value_f_maximum_value_f_l944_94437


namespace NUMINAMATH_GPT_bananas_count_l944_94430

theorem bananas_count 
  (total_oranges : ℕ)
  (total_percentage_good : ℝ)
  (percentage_rotten_oranges : ℝ)
  (percentage_rotten_bananas : ℝ)
  (total_good_fruits_percentage : ℝ)
  (B : ℝ) :
  total_oranges = 600 →
  total_percentage_good = 0.85 →
  percentage_rotten_oranges = 0.15 →
  percentage_rotten_bananas = 0.03 →
  total_good_fruits_percentage = 0.898 →
  B = 400  :=
by
  intros h_oranges h_good_percentage h_rotten_oranges h_rotten_bananas h_good_fruits_percentage
  sorry

end NUMINAMATH_GPT_bananas_count_l944_94430


namespace NUMINAMATH_GPT_algebraic_expression_value_l944_94498

variable (a b : ℝ)
axiom h1 : a = 3
axiom h2 : a - b = 1

theorem algebraic_expression_value :
  a^2 - a * b = 3 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l944_94498


namespace NUMINAMATH_GPT_convert_base8_to_base10_l944_94405

def base8_to_base10 (n : Nat) : Nat := 
  -- Assuming a specific function that converts from base 8 to base 10
  sorry 

theorem convert_base8_to_base10 :
  base8_to_base10 5624 = 2964 :=
by
  sorry

end NUMINAMATH_GPT_convert_base8_to_base10_l944_94405


namespace NUMINAMATH_GPT_average_remaining_two_numbers_l944_94441

theorem average_remaining_two_numbers 
  (a1 a2 a3 a4 a5 a6 : ℝ)
  (h_avg_6 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 2.80)
  (h_avg_2_1 : (a1 + a2) / 2 = 2.4)
  (h_avg_2_2 : (a3 + a4) / 2 = 2.3) :
  (a5 + a6) / 2 = 3.7 :=
by
  sorry

end NUMINAMATH_GPT_average_remaining_two_numbers_l944_94441
