import Mathlib

namespace right_triangle_sides_l690_69098

theorem right_triangle_sides 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_area : (1 / 2) * a * b = 150) 
  (h_perimeter : a + b + c = 60) 
  : (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by
  sorry

end right_triangle_sides_l690_69098


namespace domain_of_x_l690_69028

-- Conditions
def is_defined_num (x : ℝ) : Prop := x + 1 >= 0
def not_zero_den (x : ℝ) : Prop := x ≠ 2

-- Proof problem statement
theorem domain_of_x (x : ℝ) : (is_defined_num x ∧ not_zero_den x) ↔ (x >= -1 ∧ x ≠ 2) := by
  sorry

end domain_of_x_l690_69028


namespace fourth_term_of_geometric_sequence_l690_69016

theorem fourth_term_of_geometric_sequence 
  (a r : ℕ) 
  (h₁ : a = 3)
  (h₂ : a * r^2 = 75) :
  a * r^3 = 375 := 
by
  sorry

end fourth_term_of_geometric_sequence_l690_69016


namespace fixed_point_exists_trajectory_M_trajectory_equation_l690_69053

variable (m : ℝ)
def line_l (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0
def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0

theorem fixed_point_exists :
  ∃ x y : ℝ, (line_l m x y ∧ x = 1 ∧ y = -2) :=
by
  sorry

theorem trajectory_M :
  ∃ (M: ℝ × ℝ), (line_l m M.1 M.2 ∧ M = (0, -1)) :=
by
  sorry

theorem trajectory_equation (x y : ℝ) :
  ∃ (x y : ℝ), (x + 1) ^ 2  + y ^ 2 = 2 :=
by
  sorry

end fixed_point_exists_trajectory_M_trajectory_equation_l690_69053


namespace sasha_stickers_l690_69083

variables (m n : ℕ) (t : ℝ)

-- Conditions
def conditions : Prop :=
  m < n ∧ -- Fewer coins than stickers
  m ≥ 1 ∧ -- At least one coin
  n ≥ 1 ∧ -- At least one sticker
  t > 1 ∧ -- t is greater than 1
  m * t + n = 100 ∧ -- Coin increase condition
  m + n * t = 101 -- Sticker increase condition

-- Theorem stating that the number of stickers must be 34 or 66
theorem sasha_stickers : conditions m n t → n = 34 ∨ n = 66 :=
sorry

end sasha_stickers_l690_69083


namespace inradius_of_right_triangle_l690_69055

-- Define the side lengths of the triangle
def a : ℕ := 9
def b : ℕ := 40
def c : ℕ := 41

-- Define the semiperimeter of the triangle
def s : ℕ := (a + b + c) / 2

-- Define the area of a right triangle
def A : ℕ := (a * b) / 2

-- Define the inradius of the triangle
def inradius : ℕ := A / s

theorem inradius_of_right_triangle : inradius = 4 :=
by
  -- The proof is omitted since only the statement is requested
  sorry

end inradius_of_right_triangle_l690_69055


namespace expression_undefined_count_l690_69045

theorem expression_undefined_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x : ℝ,
  ((x = x1 ∨ x = x2) ↔ (x^2 - 2*x - 3 = 0 ∨ x - 3 = 0)) ∧ 
  ((x^2 - 2*x - 3) * (x - 3) = 0 → (x = x1 ∨ x = x2)) :=
by
  sorry

end expression_undefined_count_l690_69045


namespace field_trip_vans_l690_69071

-- Define the number of students and adults
def students := 12
def adults := 3

-- Define the capacity of each van
def van_capacity := 5

-- Total number of people
def total_people := students + adults

-- Calculate the number of vans needed
def vans_needed := (total_people + van_capacity - 1) / van_capacity  -- For rounding up division

theorem field_trip_vans : vans_needed = 3 :=
by
  -- Calculation and proof would go here
  sorry

end field_trip_vans_l690_69071


namespace area_of_triangle_ACD_l690_69091

theorem area_of_triangle_ACD :
  ∀ (AD AC height_AD height_AC : ℝ),
  AD = 6 → height_AD = 3 → AC = 3 → height_AC = 3 →
  (1 / 2 * AD * height_AD - 1 / 2 * AC * height_AC) = 4.5 :=
by
  intros AD AC height_AD height_AC hAD hheight_AD hAC hheight_AC
  sorry

end area_of_triangle_ACD_l690_69091


namespace max_value_x3y2z_l690_69023

theorem max_value_x3y2z
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h_total : x + 2 * y + 3 * z = 1)
  : x^3 * y^2 * z ≤ 2048 / 11^6 := 
by
  sorry

end max_value_x3y2z_l690_69023


namespace quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l690_69042

theorem quadratic_eq_real_roots_m_ge_neg1 (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0) →
  m ≥ -1 :=
sorry

theorem quadratic_eq_real_roots_cond (m : ℝ) (x1 x2 : ℝ) :
  x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0 ∧
  (x1 - x2)^2 = 16 - x1 * x2 →
  m = 1 :=
sorry

end quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l690_69042


namespace polynomial_coefficients_l690_69088

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (x + 2)^5 = (x + 1)^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_0 = 31 ∧ a_1 = 75 :=
by
  sorry

end polynomial_coefficients_l690_69088


namespace fish_added_l690_69020

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end fish_added_l690_69020


namespace infection_probability_l690_69048

theorem infection_probability
  (malaria_percent : ℝ)
  (zika_percent : ℝ)
  (vaccine_reduction : ℝ)
  (prob_random_infection : ℝ)
  (P : ℝ) :
  malaria_percent = 0.40 →
  zika_percent = 0.20 →
  vaccine_reduction = 0.50 →
  prob_random_infection = 0.15 →
  0.15 = (0.40 * 0.50 * P) + (0.20 * P) →
  P = 0.375 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end infection_probability_l690_69048


namespace total_pounds_of_peppers_l690_69069

def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335
def total_peppers : ℝ := 5.666666666666667

theorem total_pounds_of_peppers :
  green_peppers + red_peppers = total_peppers :=
by
  -- sorry: Proof is omitted
  sorry

end total_pounds_of_peppers_l690_69069


namespace average_speed_is_9_mph_l690_69017

-- Define the conditions
def distance_north_ft := 5280
def north_speed_min_per_mile := 3
def rest_time_min := 10
def south_speed_miles_per_min := 3

-- Define a function to convert feet to miles
def feet_to_miles (ft : ℕ) : ℕ := ft / 5280

-- Define the time calculation for north and south trips
def time_north_min (speed : ℕ) (distance_ft : ℕ) : ℕ :=
  speed * feet_to_miles distance_ft

def time_south_min (speed_miles_per_min : ℕ) (distance_ft : ℕ) : ℕ :=
  (feet_to_miles distance_ft) / speed_miles_per_min

def total_time_min (time_north rest_time time_south : ℕ) : Rat :=
  time_north + rest_time + time_south

-- Convert total time into hours
def total_time_hr (total_time_min : Rat) : Rat :=
  total_time_min / 60

-- Define the total distance in miles
def total_distance_miles (distance_ft : ℕ) : ℕ :=
  2 * feet_to_miles distance_ft

-- Calculate the average speed
def average_speed (total_distance : ℕ) (total_time_hr : Rat) : Rat :=
  total_distance / total_time_hr

-- Prove the average speed is 9 miles per hour
theorem average_speed_is_9_mph : 
  average_speed (total_distance_miles distance_north_ft)
                (total_time_hr (total_time_min (time_north_min north_speed_min_per_mile distance_north_ft)
                                              rest_time_min
                                              (time_south_min south_speed_miles_per_min distance_north_ft)))
    = 9 := by
  sorry

end average_speed_is_9_mph_l690_69017


namespace find_largest_number_l690_69049

noncomputable def largest_number (a b c : ℚ) : ℚ :=
  if a + b + c = 77 ∧ c - b = 9 ∧ b - a = 5 then c else 0

theorem find_largest_number (a b c : ℚ) 
  (h1 : a + b + c = 77) 
  (h2 : c - b = 9) 
  (h3 : b - a = 5) : 
  c = 100 / 3 := 
sorry

end find_largest_number_l690_69049


namespace convert_speed_72_kmph_to_mps_l690_69062

theorem convert_speed_72_kmph_to_mps :
  let kmph := 72
  let factor_km_to_m := 1000
  let factor_hr_to_s := 3600
  (kmph * factor_km_to_m) / factor_hr_to_s = 20 := by
  -- (72 kmph * (1000 meters / 1 kilometer)) / (3600 seconds / 1 hour) = 20 meters per second
  sorry

end convert_speed_72_kmph_to_mps_l690_69062


namespace BP_PA_ratio_l690_69086

section

variable (A B C P : Type)
variable {AC BC PA PB BP : ℕ}

-- Conditions:
-- 1. In triangle ABC, the ratio AC:CB = 2:5.
axiom AC_CB_ratio : 2 * BC = 5 * AC

-- 2. The bisector of the exterior angle at C intersects the extension of BA at P,
--    such that B is between P and A.
axiom Angle_Bisector_Theorem : PA * BC = PB * AC

theorem BP_PA_ratio (h1 : 2 * BC = 5 * AC) (h2 : PA * BC = PB * AC) :
  BP * PA = 5 * PA := sorry

end

end BP_PA_ratio_l690_69086


namespace probability_one_black_one_red_l690_69021

theorem probability_one_black_one_red (R B : Finset ℕ) (hR : R.card = 2) (hB : B.card = 3) :
  (2 : ℚ) / 5 = (6 + 6) / (5 * 4) := by
  sorry

end probability_one_black_one_red_l690_69021


namespace player_catches_ball_in_5_seconds_l690_69068

theorem player_catches_ball_in_5_seconds
    (s_ball : ℕ → ℝ) (s_player : ℕ → ℝ)
    (t_ball : ℕ)
    (t_player : ℕ)
    (d_player_initial : ℝ)
    (d_sideline : ℝ) :
  (∀ t, s_ball t = (4.375 * t - 0.375 * t^2)) →
  (∀ t, s_player t = (3.25 * t + 0.25 * t^2)) →
  (d_player_initial = 10) →
  (d_sideline = 23) →
  t_player = 5 →
  s_player t_player + d_player_initial = s_ball t_player ∧ s_ball t_player < d_sideline := 
by sorry

end player_catches_ball_in_5_seconds_l690_69068


namespace S_equals_l690_69050
noncomputable def S : Real :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_equals : S = 2 * Real.sqrt 23 - 2 :=
by
  sorry

end S_equals_l690_69050


namespace exists_fraction_bound_infinite_no_fraction_bound_l690_69085

-- Problem 1: Statement 1
theorem exists_fraction_bound (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

-- Problem 2: Statement 2
theorem infinite_no_fraction_bound :
  ∃ᶠ n : ℕ in Filter.atTop, ¬ ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

end exists_fraction_bound_infinite_no_fraction_bound_l690_69085


namespace exp_fixed_point_l690_69052

theorem exp_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : a^0 = 1 :=
by
  exact one_pow 0

end exp_fixed_point_l690_69052


namespace iggy_running_hours_l690_69040

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end iggy_running_hours_l690_69040


namespace apple_consumption_l690_69011

-- Definitions for the portions of the apple above and below water
def portion_above_water := 1 / 5
def portion_below_water := 4 / 5

-- Rates of consumption by fish and bird
def fish_rate := 120  -- grams per minute
def bird_rate := 60  -- grams per minute

-- The question statements with the correct answers
theorem apple_consumption :
  (portion_below_water * (fish_rate / (fish_rate + bird_rate)) = 2 / 3) ∧ 
  (portion_above_water * (bird_rate / (fish_rate + bird_rate)) = 1 / 3) := 
sorry

end apple_consumption_l690_69011


namespace last_two_nonzero_digits_of_70_factorial_are_04_l690_69036

-- Given conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial_are_04 :
  let n := 70;
  ∀ t : ℕ, 
    t = factorial n → t % 100 ≠ 0 → (t % 100) / 10 != 0 → 
    (t % 100) = 04 :=
sorry

end last_two_nonzero_digits_of_70_factorial_are_04_l690_69036


namespace opposite_of_half_l690_69026

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end opposite_of_half_l690_69026


namespace highest_prob_red_ball_l690_69018

-- Definitions
def total_red_balls : ℕ := 5
def total_white_balls : ℕ := 12
def total_balls : ℕ := total_red_balls + total_white_balls

-- Condition that neither bag is empty
def neither_bag_empty (r1 w1 r2 w2 : ℕ) : Prop :=
  (r1 + w1 > 0) ∧ (r2 + w2 > 0)

-- Define the probability of drawing a red ball from a bag
def prob_red (r w : ℕ) : ℚ :=
  if (r + w) = 0 then 0 else r / (r + w)

-- Define the overall probability if choosing either bag with equal probability
def overall_prob_red (r1 w1 r2 w2 : ℕ) : ℚ :=
  (prob_red r1 w1 + prob_red r2 w2) / 2

-- Problem statement to be proved
theorem highest_prob_red_ball :
  ∃ (r1 w1 r2 w2 : ℕ),
    neither_bag_empty r1 w1 r2 w2 ∧
    r1 + r2 = total_red_balls ∧
    w1 + w2 = total_white_balls ∧
    (overall_prob_red r1 w1 r2 w2 = 0.625) :=
sorry

end highest_prob_red_ball_l690_69018


namespace proof_problem_l690_69073

variable (p q : Prop)

theorem proof_problem
  (h₁ : p ∨ q)
  (h₂ : ¬p) :
  ¬p ∧ q :=
by
  sorry

end proof_problem_l690_69073


namespace relation_between_p_and_q_l690_69039

theorem relation_between_p_and_q (p q : ℝ) (α : ℝ) 
  (h1 : α + 2 * α = -p) 
  (h2 : α * (2 * α) = q) : 
  2 * p^2 = 9 * q := 
by 
  -- simplifying the provided conditions
  sorry

end relation_between_p_and_q_l690_69039


namespace number_50_is_sample_size_l690_69008

def number_of_pairs : ℕ := 50
def is_sample_size (n : ℕ) : Prop := n = number_of_pairs

-- We are to show that 50 represents the sample size
theorem number_50_is_sample_size : is_sample_size 50 :=
sorry

end number_50_is_sample_size_l690_69008


namespace benny_january_savings_l690_69081

theorem benny_january_savings :
  ∃ x : ℕ, x + x + 8 = 46 ∧ x = 19 :=
by
  sorry

end benny_january_savings_l690_69081


namespace certain_number_value_l690_69057

theorem certain_number_value (x : ℝ) (certain_number : ℝ) 
  (h1 : x = 0.25) 
  (h2 : 625^(-x) + 25^(-2 * x) + certain_number^(-4 * x) = 11) : 
  certain_number = 5 / 53 := 
sorry

end certain_number_value_l690_69057


namespace max_ad_minus_bc_l690_69015

theorem max_ad_minus_bc (a b c d : ℤ) (ha : a ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hb : b ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hc : c ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hd : d ∈ Set.image (fun x => x) {(-1), 1, 2}) :
  ad - bc ≤ 6 :=
sorry

end max_ad_minus_bc_l690_69015


namespace Mater_costs_10_percent_of_Lightning_l690_69096

-- Conditions
def price_Lightning : ℕ := 140000
def price_Sally : ℕ := 42000
def price_Mater : ℕ := price_Sally / 3

-- The theorem we want to prove
theorem Mater_costs_10_percent_of_Lightning :
  (price_Mater * 100 / price_Lightning) = 10 := 
by 
  sorry

end Mater_costs_10_percent_of_Lightning_l690_69096


namespace line_tangent_to_ellipse_l690_69041

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1 → m^2 = 35/9) := 
sorry

end line_tangent_to_ellipse_l690_69041


namespace min_quotient_l690_69090

theorem min_quotient {a b : ℕ} (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 400 ≤ b) (h₄ : b ≤ 800) (h₅ : a + b ≤ 950) : a / b = 1 / 8 := 
by
  sorry

end min_quotient_l690_69090


namespace two_a_minus_b_values_l690_69046

theorem two_a_minus_b_values (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 5) (h3 : |a + b| = -(a + b)) :
  (2 * a - b = 13) ∨ (2 * a - b = -3) :=
sorry

end two_a_minus_b_values_l690_69046


namespace negation_of_proposition_l690_69080

theorem negation_of_proposition : (¬ (∀ x : ℝ, x > 2 → x > 3)) = ∃ x > 2, x ≤ 3 := by
  sorry

end negation_of_proposition_l690_69080


namespace remaining_food_can_cater_children_l690_69009

theorem remaining_food_can_cater_children (A C : ℝ) 
  (h_food_adults : 70 * A = 90 * C) 
  (h_35_adults_ate : ∀ n: ℝ, (n = 35) → 35 * A = 35 * (9/7) * C) : 
  70 * A - 35 * A = 45 * C :=
by
  sorry

end remaining_food_can_cater_children_l690_69009


namespace find_2023rd_letter_in_sequence_l690_69082

def repeating_sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F', 'E', 'D', 'C', 'B', 'A']

def nth_in_repeating_sequence (n : ℕ) : Char :=
  repeating_sequence.get! (n % 13)

theorem find_2023rd_letter_in_sequence :
  nth_in_repeating_sequence 2023 = 'H' :=
by
  sorry

end find_2023rd_letter_in_sequence_l690_69082


namespace log_identity_l690_69077

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_identity :
    2 * log_base_10 2 + log_base_10 (5 / 8) - log_base_10 25 = -1 :=
by 
  sorry

end log_identity_l690_69077


namespace smallest_number_of_slices_l690_69025

-- Definition of the number of slices in each type of cheese package
def slices_of_cheddar : ℕ := 12
def slices_of_swiss : ℕ := 28

-- Predicate stating that the smallest number of slices of each type Randy could have bought is 84
theorem smallest_number_of_slices : Nat.lcm slices_of_cheddar slices_of_swiss = 84 := by
  sorry

end smallest_number_of_slices_l690_69025


namespace geometric_seq_common_ratio_l690_69067

theorem geometric_seq_common_ratio 
  (a : ℝ) (q : ℝ)
  (h1 : a * q^2 = 4)
  (h2 : a * q^5 = 1 / 2) : 
  q = 1 / 2 := 
by
  sorry

end geometric_seq_common_ratio_l690_69067


namespace soccer_match_outcome_l690_69033

theorem soccer_match_outcome :
  ∃ n : ℕ, n = 4 ∧
  (∃ (num_wins num_draws num_losses : ℕ),
     num_wins * 3 + num_draws * 1 + num_losses * 0 = 19 ∧
     num_wins + num_draws + num_losses = 14) :=
sorry

end soccer_match_outcome_l690_69033


namespace calculation_proof_l690_69005

theorem calculation_proof :
  5^(Real.log 9 / Real.log 5) + (1 / 2) * (Real.log 32 / Real.log 2) - Real.log (Real.log 8 / Real.log 2) / Real.log 3 = 21 / 2 := 
  sorry

end calculation_proof_l690_69005


namespace relatively_prime_sums_l690_69070

theorem relatively_prime_sums (x y : ℤ) (h : Int.gcd x y = 1) 
  : Int.gcd (x^2 + x * y + y^2) (x^2 + 3 * x * y + y^2) = 1 :=
by
  sorry

end relatively_prime_sums_l690_69070


namespace quadratic_two_distinct_real_roots_l690_69002

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end quadratic_two_distinct_real_roots_l690_69002


namespace compute_value_l690_69095

theorem compute_value : (7^2 - 6^2)^3 = 2197 := by
  sorry

end compute_value_l690_69095


namespace percentage_increase_l690_69029

theorem percentage_increase (L : ℝ) (h : L + 60 = 240) : ((60 / L) * 100 = 33 + (1 / 3) * 100) :=
by
  sorry

end percentage_increase_l690_69029


namespace elderly_people_sampled_l690_69075

theorem elderly_people_sampled (total_population : ℕ) (children : ℕ) (elderly : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_population = 1500)
  (h2 : ∃ d, children + d = elderly ∧ elderly + d = middle_aged)
  (h3 : total_population = children + elderly + middle_aged)
  (h4 : sample_size = 60) :
  elderly * (sample_size / total_population) = 20 :=
by
  -- Proof will be written here
  sorry

end elderly_people_sampled_l690_69075


namespace one_in_M_l690_69066

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := sorry

end one_in_M_l690_69066


namespace max_min_value_of_fg_l690_69032

noncomputable def f (x : ℝ) : ℝ := 4 - x^2
noncomputable def g (x : ℝ) : ℝ := 3 * x
noncomputable def min' (a b : ℝ) : ℝ := if a < b then a else b

theorem max_min_value_of_fg : ∃ x : ℝ, min' (f x) (g x) = 3 :=
by
  sorry

end max_min_value_of_fg_l690_69032


namespace total_amount_paid_is_correct_l690_69093

def rate_per_kg_grapes := 98
def quantity_grapes := 15
def rate_per_kg_mangoes := 120
def quantity_mangoes := 8
def rate_per_kg_pineapples := 75
def quantity_pineapples := 5
def rate_per_kg_oranges := 60
def quantity_oranges := 10

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes
def cost_pineapples := rate_per_kg_pineapples * quantity_pineapples
def cost_oranges := rate_per_kg_oranges * quantity_oranges

def total_amount_paid := cost_grapes + cost_mangoes + cost_pineapples + cost_oranges

theorem total_amount_paid_is_correct : total_amount_paid = 3405 := by
  sorry

end total_amount_paid_is_correct_l690_69093


namespace find_a_b_c_sum_l690_69072

theorem find_a_b_c_sum (a b c : ℤ)
  (h_gcd : gcd (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x + 1)
  (h_lcm : lcm (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x ^ 3 - 4 * x ^ 2 + x + 6) :
  a + b + c = -6 := 
sorry

end find_a_b_c_sum_l690_69072


namespace constant_k_independent_of_b_l690_69059

noncomputable def algebraic_expression (a b k : ℝ) : ℝ :=
  a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2)

theorem constant_k_independent_of_b (a : ℝ) : (algebraic_expression a b 2) = (algebraic_expression a 1 2) :=
by
  sorry

end constant_k_independent_of_b_l690_69059


namespace intersection_of_M_and_N_l690_69056

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 5}
noncomputable def N : Set ℝ := {x | x * (x - 4) > 0}

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5) } := by
  sorry

end intersection_of_M_and_N_l690_69056


namespace probability_of_draw_l690_69003

-- Let P be the probability of the game ending in a draw.
-- Let PA be the probability of Player A winning.

def PA_not_losing := 0.8
def PB_not_losing := 0.7

theorem probability_of_draw : ¬ (1 - PA_not_losing + PB_not_losing ≠ 1.5) → PA_not_losing + (1 - PB_not_losing) = 1.5 → PB_not_losing + 0.5 = 1 := by
  intros
  sorry

end probability_of_draw_l690_69003


namespace calculation_result_l690_69054

theorem calculation_result :
  -Real.sqrt 4 + abs (-Real.sqrt 2 - 1) + (Real.pi - 2013) ^ 0 - (1/5) ^ 0 = Real.sqrt 2 - 1 :=
by
  sorry

end calculation_result_l690_69054


namespace cricket_run_target_l690_69024

theorem cricket_run_target
  (run_rate_1st_period : ℝ)
  (overs_1st_period : ℕ)
  (run_rate_2nd_period : ℝ)
  (overs_2nd_period : ℕ)
  (target_runs : ℝ)
  (h1 : run_rate_1st_period = 3.2)
  (h2 : overs_1st_period = 10)
  (h3 : run_rate_2nd_period = 5)
  (h4 : overs_2nd_period = 50) :
  target_runs = (run_rate_1st_period * overs_1st_period) + (run_rate_2nd_period * overs_2nd_period) :=
by
  sorry

end cricket_run_target_l690_69024


namespace initial_bones_count_l690_69031

theorem initial_bones_count (B : ℕ) (h1 : B + 8 = 23) : B = 15 :=
sorry

end initial_bones_count_l690_69031


namespace three_digit_even_with_sum_twelve_l690_69006

theorem three_digit_even_with_sum_twelve :
  ∃ n: ℕ, n = 36 ∧ 
    (∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 2 = 0 ∧ 
          ((x / 10) % 10 + x % 10 = 12) → x = n) :=
sorry

end three_digit_even_with_sum_twelve_l690_69006


namespace arithmetic_sequence_sum_l690_69060

variable {a : ℕ → ℝ}

noncomputable def sum_of_first_ten_terms (a : ℕ → ℝ) : ℝ :=
  (10 / 2) * (a 1 + a 10)

theorem arithmetic_sequence_sum (h : a 5 + a 6 = 28) :
  sum_of_first_ten_terms a = 140 :=
by
  sorry

end arithmetic_sequence_sum_l690_69060


namespace area_inequality_equality_condition_l690_69043

variable (a b c d S : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
variable (s : ℝ) (h5 : s = (a + b + c + d) / 2)
variable (h6 : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)))

theorem area_inequality (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  S ≤ Real.sqrt (a * b * c * d) :=
sorry

theorem equality_condition (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  (S = Real.sqrt (a * b * c * d)) ↔ (a = c ∧ b = d ∨ a = d ∧ b = c) :=
sorry

end area_inequality_equality_condition_l690_69043


namespace complex_purely_imaginary_m_value_l690_69099

theorem complex_purely_imaginary_m_value (m : ℝ) :
  (m^2 - 1 = 0) ∧ (m + 1 ≠ 0) → m = 1 :=
by
  sorry

end complex_purely_imaginary_m_value_l690_69099


namespace yolkino_palkino_l690_69001

open Nat

/-- On every kilometer of the highway between the villages Yolkino and Palkino, there is a post with a sign.
    On one side of the sign, the distance to Yolkino is written, and on the other side, the distance to Palkino is written.
    The sum of all the digits on each post equals 13.
    Prove that the distance from Yolkino to Palkino is 49 kilometers. -/
theorem yolkino_palkino (n : ℕ) (h : ∀ k : ℕ, k ≤ n → (digits 10 k).sum + (digits 10 (n - k)).sum = 13) : n = 49 :=
by
  sorry

end yolkino_palkino_l690_69001


namespace other_root_of_quadratic_l690_69014

theorem other_root_of_quadratic (m : ℝ) (h : (m + 2) * 0^2 - 0 + m^2 - 4 = 0) : 
  ∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x ≠ 0 ∧ x = 1/4 := 
sorry

end other_root_of_quadratic_l690_69014


namespace triangle_division_point_distances_l690_69012

theorem triangle_division_point_distances 
  {a b c : ℝ} 
  (h1 : a = 13) 
  (h2 : b = 17) 
  (h3 : c = 24)
  (h4 : ∃ p q : ℝ, p = 9 ∧ q = 11) : 
  ∃ p q : ℝ, p = 9 ∧ q = 11 :=
  sorry

end triangle_division_point_distances_l690_69012


namespace ratio_of_areas_l690_69030

theorem ratio_of_areas (s L : ℝ) (h1 : (π * L^2) / (π * s^2) = 9 / 4) : L - s = (1/2) * s :=
by
  sorry

end ratio_of_areas_l690_69030


namespace sum_of_squares_of_roots_l690_69007

theorem sum_of_squares_of_roots 
  (x1 x2 : ℝ) 
  (h₁ : 5 * x1^2 - 6 * x1 - 4 = 0)
  (h₂ : 5 * x2^2 - 6 * x2 - 4 = 0)
  (h₃ : x1 ≠ x2) :
  x1^2 + x2^2 = 76 / 25 := sorry

end sum_of_squares_of_roots_l690_69007


namespace units_digit_product_is_2_l690_69047

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end units_digit_product_is_2_l690_69047


namespace no_solution_2023_l690_69019

theorem no_solution_2023 (a b c : ℕ) (h₁ : a + b + c = 2023) (h₂ : (b + c) ∣ a) (h₃ : (b - c + 1) ∣ (b + c)) : false :=
by
  sorry

end no_solution_2023_l690_69019


namespace compound_propositions_l690_69092

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

-- Define the propositions p and q
def p : Prop := divides 6 12
def q : Prop := divides 6 24

-- Prove the compound propositions
theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ q) ∧ ¬¬p :=
by
  -- We are proving three statements:
  -- 1. "p or q" is true.
  -- 2. "p and q" is true.
  -- 3. "not p" is false (which is equivalent to "¬¬p" being true).
  -- The actual proof will be constructed here.
  sorry

end compound_propositions_l690_69092


namespace exists_positive_integer_solution_l690_69058

theorem exists_positive_integer_solution (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ n / m = ⌊(n^2 : ℝ)^(1/3)⌋ + ⌊(n : ℝ)^(1/2)⌋ + 1 := 
by
  sorry

end exists_positive_integer_solution_l690_69058


namespace line_through_origin_and_conditions_l690_69074

-- Definitions:
def system_defines_line (m n p x y z : ℝ) : Prop :=
  (x / m = y / n) ∧ (y / n = z / p)

def lies_in_coordinate_plane (m n p : ℝ) : Prop :=
  (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)

def coincides_with_coordinate_axis (m n p : ℝ) : Prop :=
  (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)

-- Theorem statement:
theorem line_through_origin_and_conditions (m n p x y z : ℝ) :
  system_defines_line m n p x y z →
  (∀ m n p, lies_in_coordinate_plane m n p ↔ (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)) ∧
  (∀ m n p, coincides_with_coordinate_axis m n p ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)) :=
by
  sorry

end line_through_origin_and_conditions_l690_69074


namespace find_functions_l690_69037

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem find_functions (f : ℝ × ℝ → ℝ) :
  (is_non_decreasing (λ x => f (0, x))) →
  (∀ x y, f (x, y) = f (y, x)) →
  (∀ x y z, (f (x, y) - f (y, z)) * (f (y, z) - f (z, x)) * (f (z, x) - f (x, y)) = 0) →
  (∀ x y a, f (x + a, y + a) = f (x, y) + a) →
  (∃ a : ℝ, (∀ x y, f (x, y) = a + min x y) ∨ (∀ x y, f (x, y) = a + max x y)) :=
  by sorry

end find_functions_l690_69037


namespace problem_l690_69035

variables (x : ℝ)

-- Define the condition
def condition (x : ℝ) : Prop :=
  0.3 * (0.2 * x) = 24

-- Define the target statement
def target (x : ℝ) : Prop :=
  0.2 * (0.3 * x) = 24

-- The theorem we want to prove
theorem problem (x : ℝ) (h : condition x) : target x :=
sorry

end problem_l690_69035


namespace maximum_value_x_squared_plus_2y_l690_69022

theorem maximum_value_x_squared_plus_2y (x y b : ℝ) (h_curve : x^2 / 4 + y^2 / b^2 = 1) (h_b_positive : b > 0) : 
  x^2 + 2 * y ≤ max (b^2 / 4 + 4) (2 * b) :=
sorry

end maximum_value_x_squared_plus_2y_l690_69022


namespace bus_routes_arrangement_l690_69089

-- Define the lines and intersection points (stops).
def routes := Fin 10
def stops (r1 r2 : routes) : Prop := r1 ≠ r2 -- Representing intersection

-- First condition: Any subset of 9 routes will cover all stops.
def covers_all_stops (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 9 → ∀ r1 r2 : routes, r1 ≠ r2 → stops r1 r2

-- Second condition: Any subset of 8 routes will miss at least one stop.
def misses_at_least_one_stop (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 8 → ∃ r1 r2 : routes, r1 ≠ r2 ∧ ¬stops r1 r2

-- The theorem to prove that this arrangement is possible.
theorem bus_routes_arrangement : 
  (∃ stops_scheme : routes → routes → Prop, 
    (∀ subset_9 : Finset routes, covers_all_stops subset_9) ∧ 
    (∀ subset_8 : Finset routes, misses_at_least_one_stop subset_8)) :=
by
  sorry

end bus_routes_arrangement_l690_69089


namespace range_of_a_plus_c_l690_69094

-- Let a, b, c be the sides of the triangle opposite to angles A, B, and C respectively.
variable (a b c A B C : ℝ)

-- Given conditions
variable (h1 : b = Real.sqrt 3)
variable (h2 : (2 * c - a) / b * Real.cos B = Real.cos A)
variable (h3 : 0 < A ∧ A < Real.pi / 2)
variable (h4 : 0 < B ∧ B < Real.pi / 2)
variable (h5 : 0 < C ∧ C < Real.pi / 2)
variable (h6 : A + B + C = Real.pi)

-- The range of a + c
theorem range_of_a_plus_c (a b c A B C : ℝ) (h1 : b = Real.sqrt 3)
  (h2 : (2 * c - a) / b * Real.cos B = Real.cos A) (h3 : 0 < A ∧ A < Real.pi / 2)
  (h4 : 0 < B ∧ B < Real.pi / 2) (h5 : 0 < C ∧ C < Real.pi / 2) (h6 : A + B + C = Real.pi) :
  a + c ∈ Set.Ioc (Real.sqrt 3) (2 * Real.sqrt 3) :=
  sorry

end range_of_a_plus_c_l690_69094


namespace trip_time_difference_l690_69076

theorem trip_time_difference 
  (speed : ℕ) (dist1 dist2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 60) 
  (h_dist1 : dist1 = 360) 
  (h_dist2 : dist2 = 420) 
  (h_time_per_hour : time_per_hour = 60) : 
  ((dist2 / speed - dist1 / speed) * time_per_hour) = 60 := 
by
  sorry

end trip_time_difference_l690_69076


namespace sin_330_eq_negative_half_l690_69097

theorem sin_330_eq_negative_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_eq_negative_half_l690_69097


namespace roger_initial_money_l690_69013

theorem roger_initial_money (spent_on_game : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_spent : ℕ) :
  spent_on_game = 48 →
  cost_per_toy = 3 →
  num_toys = 5 →
  total_money_spent = spent_on_game + num_toys * cost_per_toy →
  total_money_spent = 63 :=
by
  intros h_game h_toy_cost h_num_toys h_total_spent
  rw [h_game, h_toy_cost, h_num_toys] at h_total_spent
  exact h_total_spent

end roger_initial_money_l690_69013


namespace probability_calculation_l690_69078

def p_X := 1 / 5
def p_Y := 1 / 2
def p_Z := 5 / 8
def p_not_Z := 1 - p_Z

theorem probability_calculation : 
    (p_X * p_Y * p_not_Z) = (3 / 80) := by
    sorry

end probability_calculation_l690_69078


namespace necessary_but_not_sufficient_condition_l690_69027

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 5) → (x > 4) :=
by 
  intro h
  linarith

end necessary_but_not_sufficient_condition_l690_69027


namespace sin_theta_of_triangle_l690_69051

theorem sin_theta_of_triangle (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ)
  (h_area : area = 30)
  (h_side : side = 10)
  (h_median : median = 9) :
  Real.sin θ = 2 / 3 := by
  sorry

end sin_theta_of_triangle_l690_69051


namespace number_of_multiples_840_in_range_l690_69084

theorem number_of_multiples_840_in_range :
  ∃ n, n = 1 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 2500 ∧ (840 ∣ x) → x = 1680 :=
by
  sorry

end number_of_multiples_840_in_range_l690_69084


namespace knights_max_seated_between_knights_l690_69000

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end knights_max_seated_between_knights_l690_69000


namespace find_value_l690_69087

theorem find_value (number remainder certain_value : ℕ) (h1 : number = 26)
  (h2 : certain_value / 2 = remainder) 
  (h3 : remainder = ((number + 20) * 2 / 2) - 2) :
  certain_value = 88 :=
by
  sorry

end find_value_l690_69087


namespace integer_values_b_for_three_integer_solutions_l690_69079

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end integer_values_b_for_three_integer_solutions_l690_69079


namespace cardboard_box_height_l690_69004

theorem cardboard_box_height :
  ∃ (x : ℕ), x ≥ 0 ∧ 10 * x^2 + 4 * x ≥ 130 ∧ (2 * x + 1) = 9 :=
sorry

end cardboard_box_height_l690_69004


namespace four_points_no_obtuse_triangle_l690_69010

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end four_points_no_obtuse_triangle_l690_69010


namespace substitution_correct_l690_69061

theorem substitution_correct (x y : ℝ) (h1 : y = x - 1) (h2 : x - 2 * y = 7) :
  x - 2 * x + 2 = 7 :=
by
  sorry

end substitution_correct_l690_69061


namespace number_of_ordered_pairs_l690_69064

theorem number_of_ordered_pairs {x y: ℕ} (h1 : x < y) (h2 : 2 * x * y / (x + y) = 4^30) : 
  ∃ n, n = 61 :=
sorry

end number_of_ordered_pairs_l690_69064


namespace max_positive_n_l690_69038

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

noncomputable def sequence_condition (a : ℕ → ℤ) : Prop :=
a 1010 / a 1009 < -1

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * (a 1 + a n) / 2

theorem max_positive_n (a : ℕ → ℤ) (h1 : is_arithmetic_sequence a) 
    (h2 : sequence_condition a) : n = 2018 ∧ sum_of_first_n_terms a 2018 > 0 := sorry

end max_positive_n_l690_69038


namespace tower_height_count_l690_69044

theorem tower_height_count (bricks : ℕ) (height1 height2 height3 : ℕ) :
  height1 = 3 → height2 = 11 → height3 = 18 → bricks = 100 →
  (∃ (h : ℕ),  h = 1404) :=
by
  sorry

end tower_height_count_l690_69044


namespace sam_initial_investment_is_6000_l690_69063

variables (P : ℝ)
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.10 / 2) ^ (2 * 1)

theorem sam_initial_investment_is_6000 :
  final_amount 6000 = 6615 :=
by
  unfold final_amount
  sorry

end sam_initial_investment_is_6000_l690_69063


namespace burmese_pythons_required_l690_69034

theorem burmese_pythons_required (single_python_rate : ℕ) (total_alligators : ℕ) (total_weeks : ℕ) (required_pythons : ℕ) :
  single_python_rate = 1 →
  total_alligators = 15 →
  total_weeks = 3 →
  required_pythons = total_alligators / total_weeks →
  required_pythons = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at *
  simp at h4
  sorry

end burmese_pythons_required_l690_69034


namespace flat_fee_for_solar_panel_equipment_l690_69065

theorem flat_fee_for_solar_panel_equipment
  (land_acreage : ℕ)
  (land_cost_per_acre : ℕ)
  (house_cost : ℕ)
  (num_cows : ℕ)
  (cow_cost_per_cow : ℕ)
  (num_chickens : ℕ)
  (chicken_cost_per_chicken : ℕ)
  (installation_hours : ℕ)
  (installation_cost_per_hour : ℕ)
  (total_cost : ℕ)
  (total_spent : ℕ) :
  land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour = total_spent →
  total_cost = total_spent →
  total_cost - (land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour) = 26000 := by 
  sorry

end flat_fee_for_solar_panel_equipment_l690_69065
