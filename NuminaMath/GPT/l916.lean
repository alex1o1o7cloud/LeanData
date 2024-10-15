import Mathlib

namespace NUMINAMATH_GPT_second_term_deposit_interest_rate_l916_91691

theorem second_term_deposit_interest_rate
  (initial_deposit : ℝ)
  (first_term_annual_rate : ℝ)
  (first_term_months : ℝ)
  (second_term_initial_value : ℝ)
  (second_term_final_value : ℝ)
  (s : ℝ)
  (first_term_value : initial_deposit * (1 + first_term_annual_rate / 100 / 12 * first_term_months) = second_term_initial_value)
  (second_term_value : second_term_initial_value * (1 + s / 100 / 12 * first_term_months) = second_term_final_value) :
  s = 11.36 :=
by
  sorry

end NUMINAMATH_GPT_second_term_deposit_interest_rate_l916_91691


namespace NUMINAMATH_GPT_largest_not_sum_of_two_composites_l916_91654

-- Define a natural number to be composite if it is divisible by some natural number other than itself and one
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

-- Define the predicate that states a number cannot be expressed as the sum of two composite numbers
def not_sum_of_two_composites (n : ℕ) : Prop :=
  ¬∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ n = a + b

-- Formal statement of the problem
theorem largest_not_sum_of_two_composites : not_sum_of_two_composites 11 :=
  sorry

end NUMINAMATH_GPT_largest_not_sum_of_two_composites_l916_91654


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_odd_function_l916_91667

def is_odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f (x)

theorem necessary_but_not_sufficient_for_odd_function (f : ℝ → ℝ) :
  (f 0 = 0) ↔ is_odd_function f :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_odd_function_l916_91667


namespace NUMINAMATH_GPT_metal_detector_time_on_less_crowded_days_l916_91639

variable (find_parking_time walk_time crowded_metal_detector_time total_time_per_week : ℕ)
variable (week_days crowded_days less_crowded_days : ℕ)

theorem metal_detector_time_on_less_crowded_days
  (h1 : find_parking_time = 5)
  (h2 : walk_time = 3)
  (h3 : crowded_metal_detector_time = 30)
  (h4 : total_time_per_week = 130)
  (h5 : week_days = 5)
  (h6 : crowded_days = 2)
  (h7 : less_crowded_days = 3) :
  (total_time_per_week = (find_parking_time * week_days) + (walk_time * week_days) + (crowded_metal_detector_time * crowded_days) + (10 * less_crowded_days)) :=
sorry

end NUMINAMATH_GPT_metal_detector_time_on_less_crowded_days_l916_91639


namespace NUMINAMATH_GPT_orange_jellybeans_count_l916_91681

theorem orange_jellybeans_count (total blue purple red : Nat)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_red : red = 120) :
  ∃ orange : Nat, orange = total - (blue + purple + red) ∧ orange = 40 :=
by
  sorry

end NUMINAMATH_GPT_orange_jellybeans_count_l916_91681


namespace NUMINAMATH_GPT_sum_of_three_digit_numbers_l916_91631

theorem sum_of_three_digit_numbers :
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  let Sum := n / 2 * (first_term + last_term)
  Sum = 494550 :=
by {
  let first_term := 100
  let last_term := 999
  let n := (last_term - first_term) + 1
  have n_def : n = 900 := by norm_num [n]
  let Sum := n / 2 * (first_term + last_term)
  have sum_def : Sum = 450 * (100 + 999) := by norm_num [Sum, first_term, last_term, n_def]
  have final_sum : Sum = 494550 := by norm_num [sum_def]
  exact final_sum
}

end NUMINAMATH_GPT_sum_of_three_digit_numbers_l916_91631


namespace NUMINAMATH_GPT_aarti_three_times_work_l916_91633

theorem aarti_three_times_work (d : ℕ) (h : d = 5) : 3 * d = 15 :=
by
  sorry

end NUMINAMATH_GPT_aarti_three_times_work_l916_91633


namespace NUMINAMATH_GPT_bob_hair_length_l916_91649

-- Define the current length of Bob's hair
def current_length : ℝ := 36

-- Define the growth rate in inches per month
def growth_rate : ℝ := 0.5

-- Define the duration in years
def duration_years : ℕ := 5

-- Define the total growth over the duration in years
def total_growth : ℝ := growth_rate * 12 * duration_years

-- Define the length of Bob's hair when he last cut it
def initial_length : ℝ := current_length - total_growth

-- Theorem stating that the length of Bob's hair when he last cut it was 6 inches
theorem bob_hair_length :
  initial_length = 6 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_bob_hair_length_l916_91649


namespace NUMINAMATH_GPT_angle_E_measure_l916_91666

theorem angle_E_measure {D E F : Type} (angle_D angle_E angle_F : ℝ) 
  (h1 : angle_E = angle_F)
  (h2 : angle_F = 3 * angle_D)
  (h3 : angle_D = (1/2) * angle_E) 
  (h_sum : angle_D + angle_E + angle_F = 180) :
  angle_E = 540 / 7 := 
by
  sorry

end NUMINAMATH_GPT_angle_E_measure_l916_91666


namespace NUMINAMATH_GPT_chocolate_ice_cream_ordered_l916_91636

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end NUMINAMATH_GPT_chocolate_ice_cream_ordered_l916_91636


namespace NUMINAMATH_GPT_maximum_value_expression_l916_91609

theorem maximum_value_expression (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_sum : a + b + c + d ≤ 4) :
  (Real.sqrt (Real.sqrt (a^2 + 3 * a * b)) + Real.sqrt (Real.sqrt (b^2 + 3 * b * c)) +
   Real.sqrt (Real.sqrt (c^2 + 3 * c * d)) + Real.sqrt (Real.sqrt (d^2 + 3 * d * a))) ≤ 4 * Real.sqrt 2 :=
by 
  sorry

end NUMINAMATH_GPT_maximum_value_expression_l916_91609


namespace NUMINAMATH_GPT_total_charging_time_l916_91693

def charge_smartphone_full : ℕ := 26
def charge_tablet_full : ℕ := 53
def charge_phone_half : ℕ := charge_smartphone_full / 2
def charge_tablet : ℕ := charge_tablet_full

theorem total_charging_time : 
  charge_phone_half + charge_tablet = 66 := by
  sorry

end NUMINAMATH_GPT_total_charging_time_l916_91693


namespace NUMINAMATH_GPT_one_fourth_more_than_x_equals_twenty_percent_less_than_80_l916_91698

theorem one_fourth_more_than_x_equals_twenty_percent_less_than_80 :
  ∃ n : ℝ, (80 - 0.30 * 80 = 56) ∧ (5 / 4 * n = 56) ∧ (n = 45) :=
by
  sorry

end NUMINAMATH_GPT_one_fourth_more_than_x_equals_twenty_percent_less_than_80_l916_91698


namespace NUMINAMATH_GPT_exists_positive_integer_divisible_by_15_and_sqrt_in_range_l916_91651

theorem exists_positive_integer_divisible_by_15_and_sqrt_in_range :
  ∃ (n : ℕ), (n % 15 = 0) ∧ (28 < Real.sqrt n) ∧ (Real.sqrt n < 28.5) ∧ (n = 795) :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_divisible_by_15_and_sqrt_in_range_l916_91651


namespace NUMINAMATH_GPT_simplify_fraction_l916_91629

namespace FractionSimplify

-- Define the fraction 48/72
def original_fraction : ℚ := 48 / 72

-- The goal is to prove that this fraction simplifies to 2/3
theorem simplify_fraction : original_fraction = 2 / 3 := by
  sorry

end FractionSimplify

end NUMINAMATH_GPT_simplify_fraction_l916_91629


namespace NUMINAMATH_GPT_cross_country_meet_winning_scores_l916_91601

theorem cross_country_meet_winning_scores :
  ∃ (scores : Finset ℕ), scores.card = 13 ∧
    ∀ s ∈ scores, s ≥ 15 ∧ s ≤ 27 :=
by
  sorry

end NUMINAMATH_GPT_cross_country_meet_winning_scores_l916_91601


namespace NUMINAMATH_GPT_ratio_volume_surface_area_l916_91603

noncomputable def volume : ℕ := 10
noncomputable def surface_area : ℕ := 45

theorem ratio_volume_surface_area : volume / surface_area = 2 / 9 := by
  sorry

end NUMINAMATH_GPT_ratio_volume_surface_area_l916_91603


namespace NUMINAMATH_GPT_audrey_sleep_time_l916_91650

theorem audrey_sleep_time (T : ℝ) (h1 : (3 / 5) * T = 6) : T = 10 :=
by
  sorry

end NUMINAMATH_GPT_audrey_sleep_time_l916_91650


namespace NUMINAMATH_GPT_reflected_ray_eqn_l916_91623

theorem reflected_ray_eqn : 
  ∃ a b c : ℝ, (∀ x y : ℝ, 2 * x - y + 5 = 0 → (a * x + b * y + c = 0)) → -- Condition for the line
  (∀ x y : ℝ, x = 1 ∧ y = 3 → (a * x + b * y + c = 0)) → -- Condition for point (1, 3)
  (a = 1 ∧ b = -5 ∧ c = 14) := -- Assertion about the line equation
by
  sorry

end NUMINAMATH_GPT_reflected_ray_eqn_l916_91623


namespace NUMINAMATH_GPT_deceased_member_income_l916_91628

theorem deceased_member_income
  (initial_income_4_members : ℕ)
  (initial_members : ℕ := 4)
  (initial_average_income : ℕ := 840)
  (final_income_3_members : ℕ)
  (remaining_members : ℕ := 3)
  (final_average_income : ℕ := 650)
  (total_income_initial : initial_income_4_members = initial_average_income * initial_members)
  (total_income_final : final_income_3_members = final_average_income * remaining_members)
  (income_deceased : ℕ) :
  income_deceased = initial_income_4_members - final_income_3_members :=
by
  -- sorry indicates this part of the proof is left as an exercise
  sorry

end NUMINAMATH_GPT_deceased_member_income_l916_91628


namespace NUMINAMATH_GPT_equilateral_triangle_of_ap_angles_gp_sides_l916_91675

theorem equilateral_triangle_of_ap_angles_gp_sides
  (A B C : ℝ)
  (α β γ : ℝ)
  (hαβγ_sum : α + β + γ = 180)
  (h_ap_angles : 2 * β = α + γ)
  (a b c : ℝ)
  (h_gp_sides : b^2 = a * c) :
  α = β ∧ β = γ ∧ a = b ∧ b = c :=
sorry

end NUMINAMATH_GPT_equilateral_triangle_of_ap_angles_gp_sides_l916_91675


namespace NUMINAMATH_GPT_average_speed_l916_91614

-- Define the average speed v
variable {v : ℝ}

-- Conditions
def day1_distance : ℝ := 160  -- 160 miles on the first day
def day2_distance : ℝ := 280  -- 280 miles on the second day
def time_difference : ℝ := 3  -- 3 hours difference

-- Theorem to prove the average speed
theorem average_speed (h1 : day1_distance / v + time_difference = day2_distance / v) : v = 40 := 
by 
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_average_speed_l916_91614


namespace NUMINAMATH_GPT_cylinder_ellipse_major_axis_l916_91617

theorem cylinder_ellipse_major_axis :
  ∀ (r : ℝ), r = 2 →
  ∀ (minor_axis : ℝ), minor_axis = 2 * r →
  ∀ (major_axis : ℝ), major_axis = 1.4 * minor_axis →
  major_axis = 5.6 :=
by
  intros r hr minor_axis hminor major_axis hmajor
  sorry

end NUMINAMATH_GPT_cylinder_ellipse_major_axis_l916_91617


namespace NUMINAMATH_GPT_candle_burning_time_l916_91605

theorem candle_burning_time :
  ∃ T : ℝ, 
    (∀ T, 0 ≤ T ∧ T ≤ 4 → thin_candle_length = 24 - 6 * T) ∧
    (∀ T, 0 ≤ T ∧ T ≤ 6 → thick_candle_length = 24 - 4 * T) ∧
    (2 * (24 - 6 * T) = 24 - 4 * T) →
    T = 3 :=
by
  sorry

end NUMINAMATH_GPT_candle_burning_time_l916_91605


namespace NUMINAMATH_GPT_models_kirsty_can_buy_l916_91671

def savings := 30 * 0.45
def new_price := 0.50

theorem models_kirsty_can_buy : savings / new_price = 27 := by
  sorry

end NUMINAMATH_GPT_models_kirsty_can_buy_l916_91671


namespace NUMINAMATH_GPT_base9_addition_correct_l916_91659

-- Definition of base 9 addition problem.
def add_base9 (a b c : ℕ) : ℕ :=
  let sum := a + b + c -- Sum in base 10
  let d0 := sum % 9 -- Least significant digit in base 9
  let carry1 := sum / 9
  (carry1 + carry1 / 9 * 9 + carry1 % 9) + d0 -- Sum in base 9 considering carry

-- The specific values converted to base 9 integers
def n1 := 3 * 9^2 + 4 * 9 + 6
def n2 := 8 * 9^2 + 0 * 9 + 2
def n3 := 1 * 9^2 + 5 * 9 + 7

-- The expected result converted to base 9 integer
def expected_sum := 1 * 9^3 + 4 * 9^2 + 1 * 9 + 6

theorem base9_addition_correct : add_base9 n1 n2 n3 = expected_sum := by
  -- Proof will be provided here
  sorry

end NUMINAMATH_GPT_base9_addition_correct_l916_91659


namespace NUMINAMATH_GPT_alice_forest_walks_l916_91695

theorem alice_forest_walks
  (morning_distance : ℕ)
  (total_distance : ℕ)
  (days_per_week : ℕ)
  (forest_distance : ℕ) :
  morning_distance = 10 →
  total_distance = 110 →
  days_per_week = 5 →
  (total_distance - morning_distance * days_per_week) / days_per_week = forest_distance →
  forest_distance = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_alice_forest_walks_l916_91695


namespace NUMINAMATH_GPT_bus_stop_time_l916_91668

theorem bus_stop_time (v_exclude_stop v_include_stop : ℕ) (h1 : v_exclude_stop = 54) (h2 : v_include_stop = 36) : 
  ∃ t: ℕ, t = 20 :=
by
  sorry

end NUMINAMATH_GPT_bus_stop_time_l916_91668


namespace NUMINAMATH_GPT_inequality_proof_l916_91670

variable (a b : ℝ)
variable (ha : a > 0)
variable (hb : b > 0)

theorem inequality_proof :
  (a / (a + b)) * ((a + 2 * b) / (a + 3 * b)) < Real.sqrt (a / (a + 4 * b)) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l916_91670


namespace NUMINAMATH_GPT_smallest_angle_l916_91627

theorem smallest_angle (k : ℝ) (h1 : 4 * k + 5 * k + 7 * k = 180) : 4 * k = 45 :=
by sorry

end NUMINAMATH_GPT_smallest_angle_l916_91627


namespace NUMINAMATH_GPT_average_children_in_families_with_children_l916_91644

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end NUMINAMATH_GPT_average_children_in_families_with_children_l916_91644


namespace NUMINAMATH_GPT_length_of_segment_correct_l916_91645

noncomputable def length_of_segment (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem length_of_segment_correct :
  length_of_segment 5 (-1) 13 11 = 4 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_GPT_length_of_segment_correct_l916_91645


namespace NUMINAMATH_GPT_range_of_x_l916_91672

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = (2 / (Real.sqrt (x - 1)))) → (x > 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l916_91672


namespace NUMINAMATH_GPT_arccos_sin_three_l916_91638

theorem arccos_sin_three : Real.arccos (Real.sin 3) = 3 - Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_arccos_sin_three_l916_91638


namespace NUMINAMATH_GPT_insects_per_group_correct_l916_91652

-- Define the numbers of insects collected by boys and girls
def boys_insects : ℕ := 200
def girls_insects : ℕ := 300
def total_insects : ℕ := boys_insects + girls_insects

-- Define the number of groups
def groups : ℕ := 4

-- Define the expected number of insects per group using total insects and groups
def insects_per_group : ℕ := total_insects / groups

-- Prove that each group gets 125 insects
theorem insects_per_group_correct : insects_per_group = 125 :=
by
  -- The proof is omitted (just setting up the theorem statement)
  sorry

end NUMINAMATH_GPT_insects_per_group_correct_l916_91652


namespace NUMINAMATH_GPT_initial_marbles_l916_91630

variable (C_initial : ℕ)
variable (marbles_given : ℕ := 42)
variable (marbles_left : ℕ := 5)

theorem initial_marbles :
  C_initial = marbles_given + marbles_left :=
sorry

end NUMINAMATH_GPT_initial_marbles_l916_91630


namespace NUMINAMATH_GPT_fraction_nonnegative_iff_l916_91677

theorem fraction_nonnegative_iff (x : ℝ) :
  (x - 12 * x^2 + 36 * x^3) / (9 - x^3) ≥ 0 ↔ 0 ≤ x ∧ x < 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_fraction_nonnegative_iff_l916_91677


namespace NUMINAMATH_GPT_eval_expression_l916_91660

-- Define the redefined operation
def red_op (a b : ℝ) : ℝ := (a + b)^2

-- Define the target expression to be evaluated
def expr (x y : ℝ) : ℝ := red_op ((x + y)^2) ((x - y)^2)

-- State the theorem
theorem eval_expression (x y : ℝ) : expr x y = 4 * (x^2 + y^2)^2 := by
  sorry

end NUMINAMATH_GPT_eval_expression_l916_91660


namespace NUMINAMATH_GPT_negation_of_statement_6_l916_91682

variable (Teenager Adult : Type)
variable (CanCookWell : Teenager → Prop)
variable (CanCookWell' : Adult → Prop)

-- Conditions from the problem
def all_teenagers_can_cook_well : Prop :=
  ∀ t : Teenager, CanCookWell t

def some_teenagers_can_cook_well : Prop :=
  ∃ t : Teenager, CanCookWell t

def no_adults_can_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def all_adults_cannot_cook_well : Prop :=
  ∀ a : Adult, ¬CanCookWell' a

def at_least_one_adult_cannot_cook_well : Prop :=
  ∃ a : Adult, ¬CanCookWell' a

def all_adults_can_cook_well : Prop :=
  ∀ a : Adult, CanCookWell' a

-- Theorem to prove
theorem negation_of_statement_6 :
  at_least_one_adult_cannot_cook_well Adult CanCookWell' = ¬ all_adults_can_cook_well Adult CanCookWell' :=
sorry

end NUMINAMATH_GPT_negation_of_statement_6_l916_91682


namespace NUMINAMATH_GPT_michael_large_balls_l916_91656

theorem michael_large_balls (total_rubber_bands : ℕ) (small_ball_rubber_bands : ℕ) (large_ball_rubber_bands : ℕ) (small_balls_made : ℕ)
  (h_total_rubber_bands : total_rubber_bands = 5000)
  (h_small_ball_rubber_bands : small_ball_rubber_bands = 50)
  (h_large_ball_rubber_bands : large_ball_rubber_bands = 300)
  (h_small_balls_made : small_balls_made = 22) :
  (total_rubber_bands - small_balls_made * small_ball_rubber_bands) / large_ball_rubber_bands = 13 :=
by {
  sorry
}

end NUMINAMATH_GPT_michael_large_balls_l916_91656


namespace NUMINAMATH_GPT_minimum_guests_l916_91610

-- Define the conditions as variables
def total_food : ℕ := 4875
def max_food_per_guest : ℕ := 3

-- Define the theorem we need to prove
theorem minimum_guests : ∃ g : ℕ, g * max_food_per_guest = total_food ∧ g >= 1625 := by
  sorry

end NUMINAMATH_GPT_minimum_guests_l916_91610


namespace NUMINAMATH_GPT_find_x_plus_y_l916_91662

theorem find_x_plus_y (x y : ℝ) (hx : |x| = 3) (hy : |y| = 2) (hxy : |x - y| = y - x) :
  (x + y = -1) ∨ (x + y = -5) :=
sorry

end NUMINAMATH_GPT_find_x_plus_y_l916_91662


namespace NUMINAMATH_GPT_total_percentage_increase_l916_91624

def initial_time : ℝ := 45
def additive_A_increase : ℝ := 0.35
def additive_B_increase : ℝ := 0.20

theorem total_percentage_increase :
  let time_after_A := initial_time * (1 + additive_A_increase)
  let time_after_B := time_after_A * (1 + additive_B_increase)
  (time_after_B - initial_time) / initial_time * 100 = 62 :=
  sorry

end NUMINAMATH_GPT_total_percentage_increase_l916_91624


namespace NUMINAMATH_GPT_factor_expression_l916_91692

variables (b : ℝ)

theorem factor_expression :
  (8 * b ^ 3 + 45 * b ^ 2 - 10) - (-12 * b ^ 3 + 5 * b ^ 2 - 10) = 20 * b ^ 2 * (b + 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l916_91692


namespace NUMINAMATH_GPT_find_third_side_length_l916_91680

noncomputable def triangle_third_side_length (a b c : ℝ) (B C : ℝ) 
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) : Prop :=
a = 16

theorem find_third_side_length (a b c : ℝ) (B C : ℝ)
  (h1 : B = 3 * C) (h2 : b = 12) (h3 : c = 20) :
  triangle_third_side_length a b c B C h1 h2 h3 :=
sorry

end NUMINAMATH_GPT_find_third_side_length_l916_91680


namespace NUMINAMATH_GPT_simplify_polynomial_l916_91686

theorem simplify_polynomial (x : ℝ) : 
  (2 * x^5 - 3 * x^3 + 5 * x^2 - 8 * x + 15) + (3 * x^4 + 2 * x^3 - 4 * x^2 + 3 * x - 7) = 
  2 * x^5 + 3 * x^4 - x^3 + x^2 - 5 * x + 8 :=
by sorry

end NUMINAMATH_GPT_simplify_polynomial_l916_91686


namespace NUMINAMATH_GPT_find_larger_number_l916_91689

theorem find_larger_number :
  ∃ (x y : ℝ), (y = x + 10) ∧ (x = y / 2) ∧ (x + y = 34) → y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l916_91689


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l916_91699

-- Definition of given quantities and conditions
variables (a b x : ℝ) (α β : ℝ)

-- Given Conditions
@[simp] def cond1 := true
@[simp] def cond2 := true
@[simp] def cond3 := true
@[simp] def cond4 := true

-- First Question
theorem problem1 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    a * Real.sin α = b * Real.sin β := sorry

-- Second Question
theorem problem2 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    Real.sin β ≤ a / b := sorry

-- Third Question
theorem problem3 (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) : 
    x = a * (1 - Real.cos α) + b * (1 - Real.cos β) := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l916_91699


namespace NUMINAMATH_GPT_no_real_roots_of_quadratic_l916_91665

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem no_real_roots_of_quadratic :
  let a := 2
  let b := -5
  let c := 6
  discriminant a b c < 0 → ¬∃ x : ℝ, 2 * x ^ 2 - 5 * x + 6 = 0 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_no_real_roots_of_quadratic_l916_91665


namespace NUMINAMATH_GPT_savings_percentage_correct_l916_91616

def coat_price : ℝ := 120
def hat_price : ℝ := 30
def gloves_price : ℝ := 50

def coat_discount : ℝ := 0.20
def hat_discount : ℝ := 0.40
def gloves_discount : ℝ := 0.30

def original_total : ℝ := coat_price + hat_price + gloves_price
def coat_savings : ℝ := coat_price * coat_discount
def hat_savings : ℝ := hat_price * hat_discount
def gloves_savings : ℝ := gloves_price * gloves_discount
def total_savings : ℝ := coat_savings + hat_savings + gloves_savings

theorem savings_percentage_correct :
  (total_savings / original_total) * 100 = 25.5 := by
  sorry

end NUMINAMATH_GPT_savings_percentage_correct_l916_91616


namespace NUMINAMATH_GPT_mean_temperature_l916_91685

def temperatures : List ℚ := [80, 79, 81, 85, 87, 89, 87, 90, 89, 88]

theorem mean_temperature :
  let n := temperatures.length
  let sum := List.sum temperatures
  (sum / n : ℚ) = 85.5 :=
by
  sorry

end NUMINAMATH_GPT_mean_temperature_l916_91685


namespace NUMINAMATH_GPT_sum_of_ages_is_220_l916_91640

-- Definitions based on the conditions
def father_age (S : ℕ) := (7 * S) / 4
def sum_ages (F S : ℕ) := F + S

-- The proof statement
theorem sum_of_ages_is_220 (F S : ℕ) (h1 : 4 * F = 7 * S)
  (h2 : 3 * (F + 10) = 5 * (S + 10)) : sum_ages F S = 220 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_ages_is_220_l916_91640


namespace NUMINAMATH_GPT_sum_of_roots_of_f_l916_91669

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x

noncomputable def f_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

theorem sum_of_roots_of_f (f : ℝ → ℝ) (m : ℝ) (x1 x2 x3 x4 : ℝ)
  (h1 : odd_function f)
  (h2 : ∀ x, f (x - 4) = - f x)
  (h3 : f_increasing_on f 0 2)
  (h4 : m > 0)
  (h5 : f x1 = m)
  (h6 : f x2 = m)
  (h7 : f x3 = m)
  (h8 : f x4 = m)
  (h9 : x1 ≠ x2)
  (h10 : x1 ≠ x3)
  (h11 : x1 ≠ x4)
  (h12 : x2 ≠ x3)
  (h13 : x2 ≠ x4)
  (h14 : x3 ≠ x4)
  (h15 : ∀ x, -8 ≤ x ∧ x ≤ 8 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_of_f_l916_91669


namespace NUMINAMATH_GPT_lucy_last_10_shots_l916_91658

variable (shots_30 : ℕ) (percentage_30 : ℚ) (total_shots : ℕ) (percentage_40 : ℚ)
variable (shots_made_30 : ℕ) (shots_made_40 : ℕ) (shots_made_last_10 : ℕ)

theorem lucy_last_10_shots 
    (h1 : shots_30 = 30) 
    (h2 : percentage_30 = 0.60) 
    (h3 : total_shots = 40) 
    (h4 : percentage_40 = 0.62 )
    (h5 : shots_made_30 = Nat.floor (percentage_30 * shots_30)) 
    (h6 : shots_made_40 = Nat.floor (percentage_40 * total_shots))
    (h7 : shots_made_last_10 = shots_made_40 - shots_made_30) 
    : shots_made_last_10 = 7 := sorry

end NUMINAMATH_GPT_lucy_last_10_shots_l916_91658


namespace NUMINAMATH_GPT_curve_crossing_self_l916_91647

theorem curve_crossing_self (t t' : ℝ) :
  (t^3 - t - 2 = t'^3 - t' - 2) ∧ (t ≠ t') ∧ 
  (t^3 - t^2 - 9 * t + 5 = t'^3 - t'^2 - 9 * t' + 5) → 
  (t = 3 ∧ t' = -3) ∨ (t = -3 ∧ t' = 3) →
  (t^3 - t - 2 = 22) ∧ (t^3 - t^2 - 9 * t + 5 = -4) :=
by
  sorry

end NUMINAMATH_GPT_curve_crossing_self_l916_91647


namespace NUMINAMATH_GPT_differential_solution_l916_91694

theorem differential_solution (C : ℝ) : 
  ∃ y : ℝ → ℝ, (∀ x : ℝ, y x = C * (1 + x^2)) := 
by
  sorry

end NUMINAMATH_GPT_differential_solution_l916_91694


namespace NUMINAMATH_GPT_Tony_temp_above_fever_threshold_l916_91641

def normal_temp : ℕ := 95
def illness_A : ℕ := 10
def illness_B : ℕ := 4
def illness_C : Int := -2
def fever_threshold : ℕ := 100

theorem Tony_temp_above_fever_threshold :
  let T := normal_temp + illness_A + illness_B + illness_C
  T = 107 ∧ (T - fever_threshold) = 7 := by
  -- conditions
  let t_0 := normal_temp
  let T_A := illness_A
  let T_B := illness_B
  let T_C := illness_C
  let F := fever_threshold
  -- calculations
  let T := t_0 + T_A + T_B + T_C
  show T = 107 ∧ (T - F) = 7
  sorry

end NUMINAMATH_GPT_Tony_temp_above_fever_threshold_l916_91641


namespace NUMINAMATH_GPT_total_money_spent_l916_91634

/-- 
John buys a gaming PC for $1200.
He decides to replace the video card in it.
He sells the old card for $300 and buys a new one for $500.
Prove total money spent on the computer after counting the savings from selling the old card is $1400.
-/
theorem total_money_spent (initial_cost : ℕ) (sale_price_old_card : ℕ) (price_new_card : ℕ) : 
  (initial_cost = 1200) → (sale_price_old_card = 300) → (price_new_card = 500) → 
  (initial_cost + (price_new_card - sale_price_old_card) = 1400) :=
by 
  intros
  sorry

end NUMINAMATH_GPT_total_money_spent_l916_91634


namespace NUMINAMATH_GPT_speed_difference_is_zero_l916_91643

theorem speed_difference_is_zero :
  let distance_bike := 72
  let time_bike := 9
  let distance_truck := 72
  let time_truck := 9
  let speed_bike := distance_bike / time_bike
  let speed_truck := distance_truck / time_truck
  (speed_truck - speed_bike) = 0 := by
  sorry

end NUMINAMATH_GPT_speed_difference_is_zero_l916_91643


namespace NUMINAMATH_GPT_savings_after_increase_l916_91618

/-- A man saves 20% of his monthly salary. If on account of dearness of things
    he is to increase his monthly expenses by 20%, he is only able to save a
    certain amount per month. His monthly salary is Rs. 6250. -/
theorem savings_after_increase (monthly_salary : ℝ) (initial_savings_percentage : ℝ)
  (increase_expenses_percentage : ℝ) (final_savings : ℝ) :
  monthly_salary = 6250 ∧
  initial_savings_percentage = 0.20 ∧
  increase_expenses_percentage = 0.20 →
  final_savings = 250 :=
by
  sorry

end NUMINAMATH_GPT_savings_after_increase_l916_91618


namespace NUMINAMATH_GPT_machine_purchase_price_l916_91620

theorem machine_purchase_price (P : ℝ) (h : 0.80 * P = 6400) : P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_machine_purchase_price_l916_91620


namespace NUMINAMATH_GPT_inequality_proof_l916_91684

theorem inequality_proof 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 ≥ 2 * (a^3 + b^3 + c^3) / (a * b * c) + 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l916_91684


namespace NUMINAMATH_GPT_ratio_of_sopranos_to_altos_l916_91626

theorem ratio_of_sopranos_to_altos (S A : ℕ) :
  (10 = 5 * S) ∧ (15 = 5 * A) → (S : ℚ) / (A : ℚ) = 2 / 3 :=
by sorry

end NUMINAMATH_GPT_ratio_of_sopranos_to_altos_l916_91626


namespace NUMINAMATH_GPT_fibonacci_coprime_l916_91676

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_coprime (n : ℕ) (hn : n ≥ 1) :
  Nat.gcd (fibonacci n) (fibonacci (n - 1)) = 1 := by
  sorry

end NUMINAMATH_GPT_fibonacci_coprime_l916_91676


namespace NUMINAMATH_GPT_max_area_rectangle_l916_91632

theorem max_area_rectangle (perimeter : ℕ) (a b : ℕ) (h1 : perimeter = 30) 
  (h2 : b = a + 3) : a * b = 54 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l916_91632


namespace NUMINAMATH_GPT_divide_19_degree_angle_into_19_equal_parts_l916_91661

/-- Divide a 19° angle into 19 equal parts, resulting in each part being 1° -/
theorem divide_19_degree_angle_into_19_equal_parts
  (α : ℝ) (hα : α = 19) :
  α / 19 = 1 :=
by
  sorry

end NUMINAMATH_GPT_divide_19_degree_angle_into_19_equal_parts_l916_91661


namespace NUMINAMATH_GPT_students_not_enrolled_in_either_l916_91602

variable (total_students french_students german_students both_students : ℕ)

theorem students_not_enrolled_in_either (h1 : total_students = 60)
                                        (h2 : french_students = 41)
                                        (h3 : german_students = 22)
                                        (h4 : both_students = 9) :
    total_students - (french_students + german_students - both_students) = 6 := by
  sorry

end NUMINAMATH_GPT_students_not_enrolled_in_either_l916_91602


namespace NUMINAMATH_GPT_total_walnut_trees_in_park_l916_91674

-- Define initial number of walnut trees in the park
def initial_walnut_trees : ℕ := 22

-- Define number of walnut trees planted by workers
def planted_walnut_trees : ℕ := 33

-- Prove the total number of walnut trees in the park
theorem total_walnut_trees_in_park : initial_walnut_trees + planted_walnut_trees = 55 := by
  sorry

end NUMINAMATH_GPT_total_walnut_trees_in_park_l916_91674


namespace NUMINAMATH_GPT_max_real_solutions_l916_91673

noncomputable def max_number_of_real_solutions (n : ℕ) (y : ℝ) : ℕ :=
if (n + 1) % 2 = 1 then 1 else 0

theorem max_real_solutions (n : ℕ) (hn : 0 < n) (y : ℝ) :
  max_number_of_real_solutions n y = 1 :=
by
  sorry

end NUMINAMATH_GPT_max_real_solutions_l916_91673


namespace NUMINAMATH_GPT_division_proof_l916_91611

-- Define the given condition
def given_condition : Prop :=
  2084.576 / 135.248 = 15.41

-- Define the problem statement we want to prove
def problem_statement : Prop :=
  23.8472 / 13.5786 = 1.756

-- Main theorem stating that under the given condition, the problem statement holds
theorem division_proof (h : given_condition) : problem_statement :=
by sorry

end NUMINAMATH_GPT_division_proof_l916_91611


namespace NUMINAMATH_GPT_total_hours_correct_l916_91646

def hours_watching_tv_per_day : ℕ := 4
def days_per_week : ℕ := 7
def days_playing_video_games_per_week : ℕ := 3

def tv_hours_per_week : ℕ := hours_watching_tv_per_day * days_per_week
def video_game_hours_per_day : ℕ := hours_watching_tv_per_day / 2
def video_game_hours_per_week : ℕ := video_game_hours_per_day * days_playing_video_games_per_week

def total_hours_per_week : ℕ := tv_hours_per_week + video_game_hours_per_week

theorem total_hours_correct :
  total_hours_per_week = 34 := by
  sorry

end NUMINAMATH_GPT_total_hours_correct_l916_91646


namespace NUMINAMATH_GPT_triangle_sum_is_16_l916_91613

-- Definition of the triangle operation
def triangle (a b c : ℕ) : ℕ := a * b - c

-- Lean theorem statement
theorem triangle_sum_is_16 : 
  triangle 2 4 3 + triangle 3 6 7 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_sum_is_16_l916_91613


namespace NUMINAMATH_GPT_perpendicular_tangent_l916_91683

noncomputable def f (x a : ℝ) := (x + a) * Real.exp x -- Defines the function

theorem perpendicular_tangent (a : ℝ) : 
  ∀ (tangent_slope perpendicular_slope : ℝ), 
  (tangent_slope = 1) → 
  (perpendicular_slope = -1) →
  tangent_slope = Real.exp 0 * (a + 1) →
  tangent_slope + perpendicular_slope = 0 → 
  a = 0 := by 
  intros tangent_slope perpendicular_slope htangent hperpendicular hderiv hperpendicular_slope
  sorry

end NUMINAMATH_GPT_perpendicular_tangent_l916_91683


namespace NUMINAMATH_GPT_find_n_l916_91648

variable {a : ℕ → ℝ} (h1 : a 4 = 7) (h2 : a 3 + a 6 = 16)

theorem find_n (n : ℕ) (h3 : a n = 31) : n = 16 := by
  sorry

end NUMINAMATH_GPT_find_n_l916_91648


namespace NUMINAMATH_GPT_find_sum_a100_b100_l916_91678

-- Definitions of arithmetic sequences and their properties
structure arithmetic_sequence (an : ℕ → ℝ) :=
  (a1 : ℝ)
  (d : ℝ)
  (def_seq : ∀ n, an n = a1 + (n - 1) * d)

-- Given conditions
variables (a_n b_n : ℕ → ℝ)
variables (ha : arithmetic_sequence a_n)
variables (hb : arithmetic_sequence b_n)

-- Specified conditions
axiom cond1 : a_n 5 + b_n 5 = 3
axiom cond2 : a_n 9 + b_n 9 = 19

-- The goal to be proved
theorem find_sum_a100_b100 : a_n 100 + b_n 100 = 383 :=
sorry

end NUMINAMATH_GPT_find_sum_a100_b100_l916_91678


namespace NUMINAMATH_GPT_molecular_weight_compound_l916_91600

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00

def molecular_weight (n_H n_Br n_O : ℕ) : ℝ :=
  n_H * atomic_weight_H + n_Br * atomic_weight_Br + n_O * atomic_weight_O

theorem molecular_weight_compound : 
  molecular_weight 1 1 3 = 128.91 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_molecular_weight_compound_l916_91600


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l916_91625

-- Define the conditions
inductive Color
| blue
| red
| green
| yellow

-- Each square can be painted in one of the colors: blue, red, or green.
def square_colors : List Color := [Color.blue, Color.red, Color.green]

-- Each triangle can be painted in one of the colors: blue, red, or yellow.
def triangle_colors : List Color := [Color.blue, Color.red, Color.yellow]

-- Condition that polygons with a common side cannot share the same color
def different_color (c1 c2 : Color) : Prop := c1 ≠ c2

-- Part (a)
theorem part_a : ∃ n : Nat, n = 7 := sorry

-- Part (b)
theorem part_b : ∃ n : Nat, n = 43 := sorry

-- Part (c)
theorem part_c : ∃ n : Nat, n = 667 := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l916_91625


namespace NUMINAMATH_GPT_sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l916_91615

theorem sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h : 100^2 + 1^2 = p * q ∧ 65^2 + 76^2 = p * q) : p + q = 210 := 
sorry

end NUMINAMATH_GPT_sum_of_primes_100_sq_plus_1_sq_eq_65_sq_plus_76_sq_l916_91615


namespace NUMINAMATH_GPT_geometric_progression_condition_l916_91604

theorem geometric_progression_condition
  (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (a_seq : ℕ → ℝ) 
  (h_def : ∀ n, a_seq (n+2) = k * a_seq n * a_seq (n+1)) :
  (a_seq 1 = a ∧ a_seq 2 = b) ↔ a_seq 1 = a_seq 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_condition_l916_91604


namespace NUMINAMATH_GPT_opposite_of_neg_one_div_2023_l916_91664

theorem opposite_of_neg_one_div_2023 : 
  (∃ x : ℚ, - (1 : ℚ) / 2023 + x = 0) ∧ 
  (∀ x : ℚ, - (1 : ℚ) / 2023 + x = 0 → x = 1 / 2023) := 
sorry

end NUMINAMATH_GPT_opposite_of_neg_one_div_2023_l916_91664


namespace NUMINAMATH_GPT_pizza_consumption_order_l916_91622

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

end NUMINAMATH_GPT_pizza_consumption_order_l916_91622


namespace NUMINAMATH_GPT_complement_A_in_U_l916_91607

noncomputable def U : Set ℝ := {x | x > -Real.sqrt 3}
noncomputable def A : Set ℝ := {x | 1 < 4 - x^2 ∧ 4 - x^2 ≤ 2}

theorem complement_A_in_U :
  (U \ A) = {x | -Real.sqrt 3 < x ∧ x ≤ -Real.sqrt 2} ∪ {x | Real.sqrt 2 ≤ x ∧ x < (Real.sqrt 3) ∨ Real.sqrt 3 ≤ x} :=
by
  sorry

end NUMINAMATH_GPT_complement_A_in_U_l916_91607


namespace NUMINAMATH_GPT_nature_of_roots_Q_l916_91697

noncomputable def Q (x : ℝ) : ℝ := x^6 - 4 * x^5 + 3 * x^4 - 7 * x^3 - x^2 + x + 10

theorem nature_of_roots_Q : 
  ∃ (negative_roots positive_roots : Finset ℝ),
    (∀ r ∈ negative_roots, r < 0) ∧
    (∀ r ∈ positive_roots, r > 0) ∧
    negative_roots.card = 1 ∧
    positive_roots.card > 1 ∧
    ∀ r, r ∈ negative_roots ∨ r ∈ positive_roots → Q r = 0 :=
sorry

end NUMINAMATH_GPT_nature_of_roots_Q_l916_91697


namespace NUMINAMATH_GPT_find_original_number_l916_91696

theorem find_original_number (x : ℤ) (h : 3 * (2 * x + 9) = 57) : x = 5 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l916_91696


namespace NUMINAMATH_GPT_total_cars_at_end_of_play_l916_91619

def carsInFront : ℕ := 100
def carsInBack : ℕ := 2 * carsInFront
def additionalCars : ℕ := 300

theorem total_cars_at_end_of_play : carsInFront + carsInBack + additionalCars = 600 := by
  sorry

end NUMINAMATH_GPT_total_cars_at_end_of_play_l916_91619


namespace NUMINAMATH_GPT_percentage_increase_l916_91679

theorem percentage_increase
  (black_and_white_cost color_cost : ℕ)
  (h_bw : black_and_white_cost = 160)
  (h_color : color_cost = 240) :
  ((color_cost - black_and_white_cost) * 100) / black_and_white_cost = 50 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l916_91679


namespace NUMINAMATH_GPT_find_integers_for_perfect_square_l916_91688

theorem find_integers_for_perfect_square (x : ℤ) :
  (∃ k : ℤ, x * (x + 1) * (x + 7) * (x + 8) = k^2) ↔ 
  x = -9 ∨ x = -8 ∨ x = -7 ∨ x = -4 ∨ x = -1 ∨ x = 0 ∨ x = 1 :=
sorry

end NUMINAMATH_GPT_find_integers_for_perfect_square_l916_91688


namespace NUMINAMATH_GPT_trains_cross_time_l916_91653

def length_train1 := 140 -- in meters
def length_train2 := 160 -- in meters

def speed_train1_kmph := 60 -- in km/h
def speed_train2_kmph := 48 -- in km/h

def kmph_to_mps (speed : ℕ) := speed * 1000 / 3600

def speed_train1_mps := kmph_to_mps speed_train1_kmph
def speed_train2_mps := kmph_to_mps speed_train2_kmph

def relative_speed_mps := speed_train1_mps + speed_train2_mps

def total_length := length_train1 + length_train2

def time_to_cross := total_length / relative_speed_mps

theorem trains_cross_time : time_to_cross = 10 :=
  by sorry

end NUMINAMATH_GPT_trains_cross_time_l916_91653


namespace NUMINAMATH_GPT_total_packs_l916_91621

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end NUMINAMATH_GPT_total_packs_l916_91621


namespace NUMINAMATH_GPT_train_average_speed_with_stoppages_l916_91657

theorem train_average_speed_with_stoppages :
  (∀ d t_without_stops t_with_stops : ℝ, t_without_stops = d / 400 → 
  t_with_stops = d / (t_without_stops * (10/9)) → 
  t_with_stops = d / 360) :=
sorry

end NUMINAMATH_GPT_train_average_speed_with_stoppages_l916_91657


namespace NUMINAMATH_GPT_find_n_modulo_l916_91642

theorem find_n_modulo :
  ∀ n : ℤ, (0 ≤ n ∧ n < 25 ∧ -175 % 25 = n % 25) → n = 0 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_find_n_modulo_l916_91642


namespace NUMINAMATH_GPT_exam_total_questions_l916_91637

/-- 
In an examination, a student scores 4 marks for every correct answer 
and loses 1 mark for every wrong answer. The student secures 140 marks 
in total. Given that the student got 40 questions correct, 
prove that the student attempted a total of 60 questions. 
-/
theorem exam_total_questions (C W T : ℕ) 
  (score_correct : C = 40)
  (total_score : 4 * C - W = 140)
  (total_questions : T = C + W) : 
  T = 60 := 
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_exam_total_questions_l916_91637


namespace NUMINAMATH_GPT_problem_l916_91606

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

axiom universal_set : U = {1, 2, 3, 4, 5, 6, 7}
axiom set_M : M = {3, 4, 5}
axiom set_N : N = {1, 3, 6}

def complement (U M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

theorem problem :
  {1, 6} = (complement U M) ∩ N :=
by
  sorry

end NUMINAMATH_GPT_problem_l916_91606


namespace NUMINAMATH_GPT_conference_center_people_count_l916_91612

-- Definition of the conditions
def rooms : ℕ := 6
def capacity_per_room : ℕ := 80
def fraction_full : ℚ := 2/3

-- Total capacity of the conference center
def total_capacity := rooms * capacity_per_room

-- Number of people in the conference center when 2/3 full
def num_people := fraction_full * total_capacity

-- The theorem stating the problem
theorem conference_center_people_count :
  num_people = 320 := 
by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_conference_center_people_count_l916_91612


namespace NUMINAMATH_GPT_full_house_plus_two_probability_l916_91690

def total_ways_to_choose_7_cards_from_52 : ℕ :=
  Nat.choose 52 7

def ways_for_full_house_plus_two : ℕ :=
  13 * 4 * 12 * 6 * 55 * 16

def probability_full_house_plus_two : ℚ :=
  (ways_for_full_house_plus_two : ℚ) / (total_ways_to_choose_7_cards_from_52 : ℚ)

theorem full_house_plus_two_probability :
  probability_full_house_plus_two = 13732 / 3344614 :=
by
  sorry

end NUMINAMATH_GPT_full_house_plus_two_probability_l916_91690


namespace NUMINAMATH_GPT_g_neg_9_equiv_78_l916_91608

noncomputable def f (x : ℝ) : ℝ := 2 * x + 3
noncomputable def g (y : ℝ) : ℝ := 3 * (y / 2 - 3 / 2)^2 + 4 * (y / 2 - 3 / 2) - 6

theorem g_neg_9_equiv_78 : g (-9) = 78 := by
  sorry

end NUMINAMATH_GPT_g_neg_9_equiv_78_l916_91608


namespace NUMINAMATH_GPT_molecular_weight_correct_l916_91687

noncomputable def molecular_weight : ℝ := 
  let N_count := 2
  let H_count := 6
  let Br_count := 1
  let O_count := 1
  let C_count := 3
  let N_weight := 14.01
  let H_weight := 1.01
  let Br_weight := 79.90
  let O_weight := 16.00
  let C_weight := 12.01
  N_count * N_weight + 
  H_count * H_weight + 
  Br_count * Br_weight + 
  O_count * O_weight +
  C_count * C_weight

theorem molecular_weight_correct :
  molecular_weight = 166.01 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l916_91687


namespace NUMINAMATH_GPT_must_divide_a_l916_91635

-- Definitions of positive integers and their gcd conditions
variables {a b c d : ℕ}

-- The conditions given in the problem
axiom h1 : gcd a b = 24
axiom h2 : gcd b c = 36
axiom h3 : gcd c d = 54
axiom h4 : 70 < gcd d a ∧ gcd d a < 100

-- We need to prove that 13 divides a
theorem must_divide_a : 13 ∣ a :=
by sorry

end NUMINAMATH_GPT_must_divide_a_l916_91635


namespace NUMINAMATH_GPT_integer_roots_iff_floor_square_l916_91655

variable (α β : ℝ)
variable (m n : ℕ)
variable (real_roots : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0)

noncomputable def are_integers (α β : ℝ) : Prop := (∃ (a b : ℤ), α = a ∧ β = b)

theorem integer_roots_iff_floor_square (m n : ℕ) (α β : ℝ)
  (hmn : 0 ≤ m ∧ 0 ≤ n)
  (roots_real : α^2 - m*α + n = 0 ∧ β^2 - m*β + n = 0) :
  (are_integers α β) ↔ (∃ k : ℤ, (⌊m * α⌋ + ⌊m * β⌋) = k^2) :=
sorry

end NUMINAMATH_GPT_integer_roots_iff_floor_square_l916_91655


namespace NUMINAMATH_GPT_odd_nat_existence_l916_91663

theorem odd_nat_existence (a b : ℕ) (h1 : a % 2 = 1) (h2 : b % 2 = 1) (n : ℕ) :
  ∃ m : ℕ, (a^m * b^2 - 1) % 2^n = 0 ∨ (b^m * a^2 - 1) % 2^n = 0 := 
by
  sorry

end NUMINAMATH_GPT_odd_nat_existence_l916_91663
