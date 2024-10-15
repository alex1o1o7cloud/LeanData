import Mathlib

namespace NUMINAMATH_GPT_fraction_defined_l1096_109638

theorem fraction_defined (x : ℝ) : (1 - 2 * x ≠ 0) ↔ (x ≠ 1 / 2) :=
by sorry

end NUMINAMATH_GPT_fraction_defined_l1096_109638


namespace NUMINAMATH_GPT_total_shingles_for_all_roofs_l1096_109610

def roof_A_length : ℕ := 20
def roof_A_width : ℕ := 40
def roof_A_shingles_per_sqft : ℕ := 8

def roof_B_length : ℕ := 25
def roof_B_width : ℕ := 35
def roof_B_shingles_per_sqft : ℕ := 10

def roof_C_length : ℕ := 30
def roof_C_width : ℕ := 30
def roof_C_shingles_per_sqft : ℕ := 12

def area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

def total_area (length : ℕ) (width : ℕ) : ℕ :=
  2 * area length width

def total_shingles_needed (length : ℕ) (width : ℕ) (shingles_per_sqft : ℕ) : ℕ :=
  total_area length width * shingles_per_sqft

theorem total_shingles_for_all_roofs :
  total_shingles_needed roof_A_length roof_A_width roof_A_shingles_per_sqft +
  total_shingles_needed roof_B_length roof_B_width roof_B_shingles_per_sqft +
  total_shingles_needed roof_C_length roof_C_width roof_C_shingles_per_sqft = 51900 :=
by
  sorry

end NUMINAMATH_GPT_total_shingles_for_all_roofs_l1096_109610


namespace NUMINAMATH_GPT_rotten_tomatoes_l1096_109644

-- Conditions
def weight_per_crate := 20
def num_crates := 3
def total_cost := 330
def selling_price_per_kg := 6
def profit := 12

-- Derived data
def total_weight := num_crates * weight_per_crate
def total_revenue := profit + total_cost
def sold_weight := total_revenue / selling_price_per_kg

-- Proof statement
theorem rotten_tomatoes : total_weight - sold_weight = 3 := by
  sorry

end NUMINAMATH_GPT_rotten_tomatoes_l1096_109644


namespace NUMINAMATH_GPT_calculate_expression_l1096_109626

theorem calculate_expression : 7 * (12 + 2 / 5) - 3 = 83.8 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1096_109626


namespace NUMINAMATH_GPT_area_of_region_below_and_left_l1096_109643

theorem area_of_region_below_and_left (x y : ℝ) :
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 4^2) ∧ y ≤ 0 ∧ y ≤ x - 4 →
  π * 4^2 / 4 = 4 * π :=
by sorry

end NUMINAMATH_GPT_area_of_region_below_and_left_l1096_109643


namespace NUMINAMATH_GPT_smallest_rectangle_area_contains_L_shape_l1096_109694

-- Condition: Side length of each square
def side_length : ℕ := 8

-- Condition: Number of squares
def num_squares : ℕ := 6

-- The correct answer (to be proven equivalent)
def expected_area : ℕ := 768

-- The main theorem stating the expected proof problem
theorem smallest_rectangle_area_contains_L_shape 
  (side_length : ℕ) (num_squares : ℕ) (h_shape : side_length = 8 ∧ num_squares = 6) : 
  ∃area, area = expected_area :=
by
  sorry

end NUMINAMATH_GPT_smallest_rectangle_area_contains_L_shape_l1096_109694


namespace NUMINAMATH_GPT_negation_exists_eq_forall_l1096_109603

theorem negation_exists_eq_forall (h : ¬ ∃ x : ℝ, x^2 + 2*x + 5 = 0) : ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_negation_exists_eq_forall_l1096_109603


namespace NUMINAMATH_GPT_rachel_homework_difference_l1096_109631

def pages_of_math_homework : Nat := 5
def pages_of_reading_homework : Nat := 2

theorem rachel_homework_difference : 
  pages_of_math_homework - pages_of_reading_homework = 3 :=
sorry

end NUMINAMATH_GPT_rachel_homework_difference_l1096_109631


namespace NUMINAMATH_GPT_map_distance_l1096_109669

theorem map_distance (scale : ℝ) (d_actual_km : ℝ) (d_actual_m : ℝ) (d_actual_cm : ℝ) (d_map : ℝ) :
  scale = 1 / 250000 →
  d_actual_km = 5 →
  d_actual_m = d_actual_km * 1000 →
  d_actual_cm = d_actual_m * 100 →
  d_map = (1 * d_actual_cm) / (1 / scale) →
  d_map = 2 :=
by sorry

end NUMINAMATH_GPT_map_distance_l1096_109669


namespace NUMINAMATH_GPT_sandy_initial_cost_l1096_109677

theorem sandy_initial_cost 
  (repairs_cost : ℝ)
  (selling_price : ℝ)
  (gain_percent : ℝ)
  (h1 : repairs_cost = 200)
  (h2 : selling_price = 1400)
  (h3 : gain_percent = 40) :
  ∃ P : ℝ, P = 800 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_sandy_initial_cost_l1096_109677


namespace NUMINAMATH_GPT_sale_day_intersection_in_july_l1096_109654

def is_multiple_of_five (d : ℕ) : Prop :=
  d % 5 = 0

def shoe_store_sale_days (d : ℕ) : Prop :=
  ∃ (k : ℕ), d = 3 + k * 6

theorem sale_day_intersection_in_july : 
  (∃ d, is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31) = (1 = Nat.card {d | is_multiple_of_five d ∧ shoe_store_sale_days d ∧ 1 ≤ d ∧ d ≤ 31}) :=
by
  sorry

end NUMINAMATH_GPT_sale_day_intersection_in_july_l1096_109654


namespace NUMINAMATH_GPT_inequality_proof_l1096_109635

theorem inequality_proof (x : ℝ) (h₁ : 3/2 ≤ x) (h₂ : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1096_109635


namespace NUMINAMATH_GPT_student_passing_percentage_l1096_109632

variable (marks_obtained failed_by max_marks : ℕ)

def passing_marks (marks_obtained failed_by : ℕ) : ℕ :=
  marks_obtained + failed_by

def percentage_needed (passing_marks max_marks : ℕ) : ℚ :=
  (passing_marks : ℚ) / (max_marks : ℚ) * 100

theorem student_passing_percentage
  (h1 : marks_obtained = 125)
  (h2 : failed_by = 40)
  (h3 : max_marks = 500) :
  percentage_needed (passing_marks marks_obtained failed_by) max_marks = 33 := by
  sorry

end NUMINAMATH_GPT_student_passing_percentage_l1096_109632


namespace NUMINAMATH_GPT_ed_initial_money_l1096_109698

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

end NUMINAMATH_GPT_ed_initial_money_l1096_109698


namespace NUMINAMATH_GPT_number_is_93_75_l1096_109646

theorem number_is_93_75 (x : ℝ) (h : 0.16 * (0.40 * x) = 6) : x = 93.75 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_number_is_93_75_l1096_109646


namespace NUMINAMATH_GPT_jerry_total_mean_l1096_109604

def receivedFromAunt : ℕ := 9
def receivedFromUncle : ℕ := 9
def receivedFromBestFriends : List ℕ := [22, 23, 22, 22]
def receivedFromSister : ℕ := 7

def totalAmountReceived : ℕ :=
  receivedFromAunt + receivedFromUncle +
  receivedFromBestFriends.sum + receivedFromSister

def totalNumberOfGifts : ℕ :=
  1 + 1 + receivedFromBestFriends.length + 1

def meanAmountReceived : ℚ :=
  totalAmountReceived / totalNumberOfGifts

theorem jerry_total_mean :
  meanAmountReceived = 16.29 := by
sorry

end NUMINAMATH_GPT_jerry_total_mean_l1096_109604


namespace NUMINAMATH_GPT_contrapositive_statement_l1096_109645

theorem contrapositive_statement (m : ℝ) (h : ¬ ∃ x : ℝ, x^2 = m) : m < 0 :=
sorry

end NUMINAMATH_GPT_contrapositive_statement_l1096_109645


namespace NUMINAMATH_GPT_f_at_five_l1096_109674

def f (n : ℕ) : ℕ := n^3 + 2 * n^2 + 3 * n + 17

theorem f_at_five : f 5 = 207 := 
by 
sorry

end NUMINAMATH_GPT_f_at_five_l1096_109674


namespace NUMINAMATH_GPT_driving_speed_l1096_109617

variable (total_distance : ℝ) (break_time : ℝ) (total_trip_time : ℝ)

theorem driving_speed (h1 : total_distance = 480)
                      (h2 : break_time = 1)
                      (h3 : total_trip_time = 9) : 
  (total_distance / (total_trip_time - break_time)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_driving_speed_l1096_109617


namespace NUMINAMATH_GPT_remainder_17_pow_2023_mod_28_l1096_109640

theorem remainder_17_pow_2023_mod_28 :
  17^2023 % 28 = 17 := 
by sorry

end NUMINAMATH_GPT_remainder_17_pow_2023_mod_28_l1096_109640


namespace NUMINAMATH_GPT_soda_cost_is_20_l1096_109602

noncomputable def cost_of_soda (b s : ℕ) : Prop :=
  4 * b + 3 * s = 500 ∧ 3 * b + 2 * s = 370

theorem soda_cost_is_20 {b s : ℕ} (h : cost_of_soda b s) : s = 20 :=
  by sorry

end NUMINAMATH_GPT_soda_cost_is_20_l1096_109602


namespace NUMINAMATH_GPT_at_least_one_not_less_than_one_third_l1096_109679

theorem at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  a ≥ 1/3 ∨ b ≥ 1/3 ∨ c ≥ 1/3 :=
sorry

end NUMINAMATH_GPT_at_least_one_not_less_than_one_third_l1096_109679


namespace NUMINAMATH_GPT_runners_meet_time_l1096_109611

theorem runners_meet_time (t_P t_Q : ℕ) (hP: t_P = 252) (hQ: t_Q = 198) : Nat.lcm t_P t_Q = 2772 :=
by
  rw [hP, hQ]
  -- The proof can be continued by proving the LCM calculation step, which we omit here
  sorry

end NUMINAMATH_GPT_runners_meet_time_l1096_109611


namespace NUMINAMATH_GPT_polynomial_is_monic_l1096_109658

noncomputable def f : ℝ → ℝ := sorry

variables (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + 6*x - 4)

theorem polynomial_is_monic (f : ℝ → ℝ) (h1 : f 1 = 3) (h2 : f 2 = 12) (h3 : ∀ x : ℝ, f x = x^2 + x + b) : 
  ∀ x : ℝ, f x = x^2 + 6*x - 4 :=
by sorry

end NUMINAMATH_GPT_polynomial_is_monic_l1096_109658


namespace NUMINAMATH_GPT_units_digit_m_sq_plus_2_m_l1096_109613

def m := 2017^2 + 2^2017

theorem units_digit_m_sq_plus_2_m (m := 2017^2 + 2^2017) : (m^2 + 2^m) % 10 = 3 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_m_sq_plus_2_m_l1096_109613


namespace NUMINAMATH_GPT_good_students_count_l1096_109685

noncomputable def student_count := 25

def is_good_student (s : Nat) := s ≤ student_count

def is_troublemaker (s : Nat) := s ≤ student_count

def always_tell_truth (s : Nat) := is_good_student s

def always_lie (s : Nat) := is_troublemaker s

def condition1 (E B : Nat) := E + B = student_count

def condition2 := ∀ (x : Nat), x ≤ 5 → is_good_student x → 
  ∃ (B : Nat), B > 24 / 2

def condition3 := ∀ (x : Nat), x ≤ 20 → is_troublemaker x → 
  ∃ (B E : Nat), B = 3 * (E - 1) ∧ E + B = student_count

theorem good_students_count :
  ∃ (E : Nat), condition1 E (student_count - E) ∧ condition2 ∧ condition3 :=
sorry  -- the actual proof is not required

end NUMINAMATH_GPT_good_students_count_l1096_109685


namespace NUMINAMATH_GPT_football_team_selection_l1096_109693

theorem football_team_selection :
  let team_members : ℕ := 12
  let offensive_lineman_choices : ℕ := 4
  let tight_end_choices : ℕ := 2
  let players_left_after_offensive : ℕ := team_members - 1
  let players_left_after_tightend : ℕ := players_left_after_offensive - 1
  let quarterback_choices : ℕ := players_left_after_tightend
  let players_left_after_quarterback : ℕ := quarterback_choices - 1
  let running_back_choices : ℕ := players_left_after_quarterback
  let players_left_after_runningback : ℕ := running_back_choices - 1
  let wide_receiver_choices : ℕ := players_left_after_runningback
  offensive_lineman_choices * tight_end_choices * 
  quarterback_choices * running_back_choices * 
  wide_receiver_choices = 5760 := 
by 
  sorry

end NUMINAMATH_GPT_football_team_selection_l1096_109693


namespace NUMINAMATH_GPT_divisors_remainders_l1096_109665

theorem divisors_remainders (n : ℕ) (h : ∀ k : ℕ, 1001 ≤ k ∧ k ≤ 2012 → ∃ d : ℕ, d ∣ n ∧ d % 2013 = k) :
  ∀ m : ℕ, 1 ≤ m ∧ m ≤ 2012 → ∃ d : ℕ, d ∣ n^2 ∧ d % 2013 = m :=
by sorry

end NUMINAMATH_GPT_divisors_remainders_l1096_109665


namespace NUMINAMATH_GPT_find_m_l1096_109691

theorem find_m (m : ℝ) : (∀ x : ℝ, x^2 - 4 * x + m = 0) → m = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_m_l1096_109691


namespace NUMINAMATH_GPT_call_charge_ratio_l1096_109608

def elvin_jan_total_bill : ℕ := 46
def elvin_feb_total_bill : ℕ := 76
def elvin_internet_charge : ℕ := 16
def elvin_call_charge_ratio : ℕ := 2

theorem call_charge_ratio : 
  (elvin_feb_total_bill - elvin_internet_charge) / (elvin_jan_total_bill - elvin_internet_charge) = elvin_call_charge_ratio := 
by
  sorry

end NUMINAMATH_GPT_call_charge_ratio_l1096_109608


namespace NUMINAMATH_GPT_area_of_region_eq_24π_l1096_109625

theorem area_of_region_eq_24π :
  (∃ R, R > 0 ∧ ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 8 * x + 18 * y + 73 = R ^ 2) →
  ∃ π : ℝ, π > 0 ∧ area = 24 * π :=
by
  sorry

end NUMINAMATH_GPT_area_of_region_eq_24π_l1096_109625


namespace NUMINAMATH_GPT_xiao_ming_min_correct_answers_l1096_109678

theorem xiao_ming_min_correct_answers (x : ℕ) : (10 * x - 5 * (20 - x) > 100) → (x ≥ 14) := by
  sorry

end NUMINAMATH_GPT_xiao_ming_min_correct_answers_l1096_109678


namespace NUMINAMATH_GPT_player_one_wins_l1096_109683

theorem player_one_wins (initial_coins : ℕ) (h_initial : initial_coins = 2015) : 
  ∃ first_move : ℕ, (1 ≤ first_move ∧ first_move ≤ 99 ∧ first_move % 2 = 1) ∧ 
  (∀ move : ℕ, (2 ≤ move ∧ move ≤ 100 ∧ move % 2 = 0) → 
   ∃ next_move : ℕ, (1 ≤ next_move ∧ next_move ≤ 99 ∧ next_move % 2 = 1) → 
   initial_coins - first_move - move - next_move < 101) → first_move = 95 :=
by 
  sorry

end NUMINAMATH_GPT_player_one_wins_l1096_109683


namespace NUMINAMATH_GPT_product_of_functions_l1096_109612

noncomputable def f (x : ℝ) : ℝ := (x - 3) / (x + 3)
noncomputable def g (x : ℝ) : ℝ := x + 3

theorem product_of_functions (x : ℝ) (hx : x ≠ -3) : f x * g x = x - 3 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_product_of_functions_l1096_109612


namespace NUMINAMATH_GPT_complement_intersection_l1096_109622

open Set

theorem complement_intersection (U A B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 5})
  (hB : B = {2, 4}) :
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1096_109622


namespace NUMINAMATH_GPT_find_fraction_sum_l1096_109630

theorem find_fraction_sum (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = -2) : (1 / x) + (1 / y) = -3 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_sum_l1096_109630


namespace NUMINAMATH_GPT_car_mpg_in_city_l1096_109675

theorem car_mpg_in_city
  (H C T : ℕ)
  (h1 : H * T = 462)
  (h2 : C * T = 336)
  (h3 : C = H - 9) : C = 24 := by
  sorry

end NUMINAMATH_GPT_car_mpg_in_city_l1096_109675


namespace NUMINAMATH_GPT_abc_sum_l1096_109637

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_abc_sum_l1096_109637


namespace NUMINAMATH_GPT_inequality_not_necessarily_hold_l1096_109655

theorem inequality_not_necessarily_hold (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) : ¬ (a + d > b + c) :=
sorry

end NUMINAMATH_GPT_inequality_not_necessarily_hold_l1096_109655


namespace NUMINAMATH_GPT_three_card_deal_probability_l1096_109628

theorem three_card_deal_probability :
  (4 / 52) * (4 / 51) * (4 / 50) = 16 / 33150 := 
by 
  sorry

end NUMINAMATH_GPT_three_card_deal_probability_l1096_109628


namespace NUMINAMATH_GPT_placing_2_flowers_in_2_vases_l1096_109623

noncomputable def num_ways_to_place_flowers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : ℕ :=
  Nat.choose n k * 2

theorem placing_2_flowers_in_2_vases :
  num_ways_to_place_flowers 5 2 rfl rfl = 20 := 
by
  sorry

end NUMINAMATH_GPT_placing_2_flowers_in_2_vases_l1096_109623


namespace NUMINAMATH_GPT_distance_to_x_axis_l1096_109684

theorem distance_to_x_axis (P : ℝ × ℝ) (h : P = (-3, -2)) : |P.2| = 2 := 
by sorry

end NUMINAMATH_GPT_distance_to_x_axis_l1096_109684


namespace NUMINAMATH_GPT_part1_part2_l1096_109652

noncomputable def f (x a : ℝ) : ℝ := x^2 - (a+1)*x + a

theorem part1 (a x : ℝ) :
  (a < 1 ∧ f x a < 0 ↔ a < x ∧ x < 1) ∧
  (a = 1 ∧ ¬(f x a < 0)) ∧
  (a > 1 ∧ f x a < 0 ↔ 1 < x ∧ x < a) :=
sorry

theorem part2 (a : ℝ) :
  (∀ x, 1 < x → f x a ≥ -1) → a ≤ 3 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1096_109652


namespace NUMINAMATH_GPT_third_quadrant_angle_to_fourth_l1096_109682

theorem third_quadrant_angle_to_fourth {α : ℝ} (k : ℤ) 
  (h : 180 + k * 360 < α ∧ α < 270 + k * 360) : 
  -90 - k * 360 < 180 - α ∧ 180 - α < -k * 360 :=
by
  sorry

end NUMINAMATH_GPT_third_quadrant_angle_to_fourth_l1096_109682


namespace NUMINAMATH_GPT_sin_theta_value_l1096_109642

theorem sin_theta_value 
  (θ : ℝ)
  (h1 : Real.sin θ + Real.cos θ = 7/5)
  (h2 : Real.tan θ < 1) :
  Real.sin θ = 3/5 :=
sorry

end NUMINAMATH_GPT_sin_theta_value_l1096_109642


namespace NUMINAMATH_GPT_solve_system_of_equations_l1096_109601

theorem solve_system_of_equations :
  ∃ x y : ℝ, 4 * x - 6 * y = -3 ∧ 9 * x + 3 * y = 6.3 ∧ x = 0.436 ∧ y = 0.792 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1096_109601


namespace NUMINAMATH_GPT_problem1_proof_problem2_proof_l1096_109606

-- Problem 1 proof statement
theorem problem1_proof : (-1)^10 * 2 + (-2)^3 / 4 = 0 := 
by
  sorry

-- Problem 2 proof statement
theorem problem2_proof : -24 * (5 / 6 - 4 / 3 + 3 / 8) = 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_proof_problem2_proof_l1096_109606


namespace NUMINAMATH_GPT_circle_equation_l1096_109653

theorem circle_equation :
  ∃ (h k r : ℝ), 
    (∀ (x y : ℝ), (x, y) = (-6, 2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ (∀ (x y : ℝ), (x, y) = (2, -2) → (x - h)^2 + (y - k)^2 = r^2)
    ∧ r = 5
    ∧ h - k = -1
    ∧ (x + 3)^2 + (y + 2)^2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_circle_equation_l1096_109653


namespace NUMINAMATH_GPT_unique_integer_n_l1096_109680

theorem unique_integer_n (n : ℤ) (h : ⌊(n^2 : ℚ) / 5⌋ - ⌊(n / 2 : ℚ)⌋^2 = 3) : n = 5 :=
  sorry

end NUMINAMATH_GPT_unique_integer_n_l1096_109680


namespace NUMINAMATH_GPT_domain_of_f_x_squared_l1096_109659

theorem domain_of_f_x_squared {f : ℝ → ℝ} (h : ∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) : 
  ∀ x, -2 ≤ x ∧ x ≤ 2 → ∃ y, f (x ^ 2) = y := 
by 
  sorry

end NUMINAMATH_GPT_domain_of_f_x_squared_l1096_109659


namespace NUMINAMATH_GPT_largest_divisor_of_5_consecutive_integers_l1096_109672

theorem largest_divisor_of_5_consecutive_integers :
  ∀ (a b c d e : ℤ), 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e →
    (∃ k : ℤ, k ∣ (a * b * c * d * e) ∧ k = 60) :=
by 
  intro a b c d e h
  sorry

end NUMINAMATH_GPT_largest_divisor_of_5_consecutive_integers_l1096_109672


namespace NUMINAMATH_GPT_q_sufficient_not_necessary_p_l1096_109615

theorem q_sufficient_not_necessary_p (x : ℝ) (p : Prop) (q : Prop) :
  (p ↔ |x| < 2) →
  (q ↔ x^2 - x - 2 < 0) →
  (q → p) ∧ (p ∧ ¬q) :=
by
  sorry

end NUMINAMATH_GPT_q_sufficient_not_necessary_p_l1096_109615


namespace NUMINAMATH_GPT_vip_seat_cost_is_65_l1096_109668

noncomputable def cost_of_VIP_seat (G V_T V : ℕ) (cost : ℕ) : Prop :=
  G + V_T = 320 ∧
  (15 * G + V * V_T = cost) ∧
  V_T = G - 212 → V = 65

theorem vip_seat_cost_is_65 :
  ∃ (G V_T V : ℕ), cost_of_VIP_seat G V_T V 7500 :=
  sorry

end NUMINAMATH_GPT_vip_seat_cost_is_65_l1096_109668


namespace NUMINAMATH_GPT_find_N_l1096_109618

/--
If 15% of N is 45% of 2003, then N is 6009.
-/
theorem find_N (N : ℕ) (h : 15 / 100 * N = 45 / 100 * 2003) : 
  N = 6009 :=
sorry

end NUMINAMATH_GPT_find_N_l1096_109618


namespace NUMINAMATH_GPT_solve_otimes_n_1_solve_otimes_2005_2_l1096_109614

-- Define the operation ⊗
noncomputable def otimes (x y : ℕ) : ℕ :=
sorry -- the definition is abstracted away as per conditions

-- Conditions from the problem
axiom otimes_cond_1 : ∀ x : ℕ, otimes x 0 = x + 1
axiom otimes_cond_2 : ∀ x : ℕ, otimes 0 (x + 1) = otimes 1 x
axiom otimes_cond_3 : ∀ x y : ℕ, otimes (x + 1) (y + 1) = otimes (otimes x (y + 1)) y

-- Prove the required equalities
theorem solve_otimes_n_1 (n : ℕ) : otimes n 1 = n + 2 :=
sorry

theorem solve_otimes_2005_2 : otimes 2005 2 = 4013 :=
sorry

end NUMINAMATH_GPT_solve_otimes_n_1_solve_otimes_2005_2_l1096_109614


namespace NUMINAMATH_GPT_extra_bananas_each_child_gets_l1096_109619

-- Define the total number of students and the number of absent students
def total_students : ℕ := 260
def absent_students : ℕ := 130

-- Define the total number of bananas
variable (B : ℕ)

-- The proof statement
theorem extra_bananas_each_child_gets :
  ∀ B : ℕ, (B / (total_students - absent_students)) = (B / total_students) + (B / total_students) :=
by
  intro B
  sorry

end NUMINAMATH_GPT_extra_bananas_each_child_gets_l1096_109619


namespace NUMINAMATH_GPT_height_of_water_in_cylindrical_tank_l1096_109692

theorem height_of_water_in_cylindrical_tank :
  let r_cone := 15  -- radius of base of conical tank in cm
  let h_cone := 24  -- height of conical tank in cm
  let r_cylinder := 18  -- radius of base of cylindrical tank in cm
  let V_cone := (1 / 3 : ℝ) * Real.pi * r_cone^2 * h_cone  -- volume of conical tank
  let h_cyl := V_cone / (Real.pi * r_cylinder^2)  -- height of water in cylindrical tank
  h_cyl = 5.56 :=
by
  sorry

end NUMINAMATH_GPT_height_of_water_in_cylindrical_tank_l1096_109692


namespace NUMINAMATH_GPT_line_passing_through_points_l1096_109656

-- Definition of points
def point1 : ℝ × ℝ := (1, 0)
def point2 : ℝ × ℝ := (0, -2)

-- Definition of the line equation
def line_eq (x y : ℝ) : Prop := 2 * x - y - 2 = 0

-- Theorem statement
theorem line_passing_through_points : 
  line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2 :=
by
  sorry

end NUMINAMATH_GPT_line_passing_through_points_l1096_109656


namespace NUMINAMATH_GPT_probability_two_consecutive_pairs_of_four_dice_correct_l1096_109667

open Classical

noncomputable def probability_two_consecutive_pairs_of_four_dice : ℚ :=
  let total_outcomes := 6^4
  let favorable_outcomes := 48
  favorable_outcomes / total_outcomes

theorem probability_two_consecutive_pairs_of_four_dice_correct :
  probability_two_consecutive_pairs_of_four_dice = 1 / 27 := 
by
  sorry

end NUMINAMATH_GPT_probability_two_consecutive_pairs_of_four_dice_correct_l1096_109667


namespace NUMINAMATH_GPT_remainder_of_division_l1096_109664

noncomputable def P (x : ℝ) := x ^ 888
noncomputable def Q (x : ℝ) := (x ^ 2 - x + 1) * (x + 1)

theorem remainder_of_division :
  ∀ x : ℝ, (P x) % (Q x) = 1 :=
sorry

end NUMINAMATH_GPT_remainder_of_division_l1096_109664


namespace NUMINAMATH_GPT_part1_part2_l1096_109639

def f (x a : ℝ) : ℝ := |x - a| + |x + 5 - a|

theorem part1 (a : ℝ) :
  (Set.Icc (a - 7) (a - 3)) = (Set.Icc (-5 : ℝ) (-1 : ℝ)) -> a = 2 :=
by
  intro h
  sorry

theorem part2 (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 2 < 4 * m + m^2) -> (m < -5 ∨ m > 1) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_part1_part2_l1096_109639


namespace NUMINAMATH_GPT_three_w_seven_l1096_109688

def operation_w (a b : ℤ) : ℤ := b + 5 * a - 3 * a^2

theorem three_w_seven : operation_w 3 7 = -5 :=
by
  sorry

end NUMINAMATH_GPT_three_w_seven_l1096_109688


namespace NUMINAMATH_GPT_algebraic_expression_value_l1096_109699

theorem algebraic_expression_value (a b : ℝ) (h1 : a * b = 2) (h2 : a - b = 3) :
  2 * a^3 * b - 4 * a^2 * b^2 + 2 * a * b^3 = 36 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1096_109699


namespace NUMINAMATH_GPT_f_increasing_maximum_b_condition_approximate_ln2_l1096_109649

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) - 2 * x

theorem f_increasing : ∀ x y : ℝ, x < y → f x ≤ f y := 
sorry

noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (2 * x) - 4 * b * f x

theorem maximum_b_condition (x : ℝ) (H : 0 < x): ∃ b, g x b > 0 ∧ b ≤ 2 := 
sorry

theorem approximate_ln2 :
  0.692 ≤ Real.log 2 ∧ Real.log 2 ≤ 0.694 :=
sorry

end NUMINAMATH_GPT_f_increasing_maximum_b_condition_approximate_ln2_l1096_109649


namespace NUMINAMATH_GPT_sumata_miles_per_day_l1096_109641

theorem sumata_miles_per_day (total_miles : ℝ) (total_days : ℝ) (h1 : total_miles = 250.0) (h2 : total_days = 5.0) :
  total_miles / total_days = 50.0 :=
by
  sorry

end NUMINAMATH_GPT_sumata_miles_per_day_l1096_109641


namespace NUMINAMATH_GPT_faster_train_length_l1096_109636

noncomputable def length_of_faster_train 
    (speed_train_1_kmph : ℤ) 
    (speed_train_2_kmph : ℤ) 
    (time_seconds : ℤ) : ℤ := 
    (speed_train_1_kmph + speed_train_2_kmph) * 1000 / 3600 * time_seconds

theorem faster_train_length 
    (speed_train_1_kmph : ℤ)
    (speed_train_2_kmph : ℤ)
    (time_seconds : ℤ)
    (h1 : speed_train_1_kmph = 36)
    (h2 : speed_train_2_kmph = 45)
    (h3 : time_seconds = 12) :
    length_of_faster_train speed_train_1_kmph speed_train_2_kmph time_seconds = 270 :=
by
    sorry

end NUMINAMATH_GPT_faster_train_length_l1096_109636


namespace NUMINAMATH_GPT_sum_of_powers_of_two_l1096_109651

theorem sum_of_powers_of_two : 2^4 + 2^4 + 2^4 = 2^5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_powers_of_two_l1096_109651


namespace NUMINAMATH_GPT_other_endpoint_l1096_109670

theorem other_endpoint (M : ℝ × ℝ) (A : ℝ × ℝ) (x y : ℝ) :
  M = (2, 3) ∧ A = (5, -1) ∧ (M = ((A.1 + x) / 2, (A.2 + y) / 2)) → (x, y) = (-1, 7) := by
  sorry

end NUMINAMATH_GPT_other_endpoint_l1096_109670


namespace NUMINAMATH_GPT_carol_used_tissue_paper_l1096_109650

theorem carol_used_tissue_paper (initial_pieces : ℕ) (remaining_pieces : ℕ) (usage: ℕ)
  (h1 : initial_pieces = 97)
  (h2 : remaining_pieces = 93)
  (h3: usage = initial_pieces - remaining_pieces) : 
  usage = 4 :=
by
  -- We only need to set up the problem; proof can be provided later.
  sorry

end NUMINAMATH_GPT_carol_used_tissue_paper_l1096_109650


namespace NUMINAMATH_GPT_vaishali_total_stripes_l1096_109695

theorem vaishali_total_stripes
  (hats1 : ℕ) (stripes1 : ℕ)
  (hats2 : ℕ) (stripes2 : ℕ)
  (hats3 : ℕ) (stripes3 : ℕ)
  (hats4 : ℕ) (stripes4 : ℕ)
  (total_stripes : ℕ) :
  hats1 = 4 → stripes1 = 3 →
  hats2 = 3 → stripes2 = 4 →
  hats3 = 6 → stripes3 = 0 →
  hats4 = 2 → stripes4 = 5 →
  total_stripes = (hats1 * stripes1) + (hats2 * stripes2) + (hats3 * stripes3) + (hats4 * stripes4) →
  total_stripes = 34 := by
  sorry

end NUMINAMATH_GPT_vaishali_total_stripes_l1096_109695


namespace NUMINAMATH_GPT_set_operation_result_l1096_109671

def M : Set ℕ := {2, 3}

def bin_op (A : Set ℕ) : Set ℕ :=
  {x | ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ x = a + b}

theorem set_operation_result : bin_op M = {4, 5, 6} :=
by
  sorry

end NUMINAMATH_GPT_set_operation_result_l1096_109671


namespace NUMINAMATH_GPT_Q_is_perfect_square_trinomial_l1096_109605

def is_perfect_square_trinomial (p : ℤ → ℤ) :=
∃ (b : ℤ), ∀ a : ℤ, p a = (a + b) * (a + b)

def P (a b : ℤ) : ℤ := a^2 + 2 * a * b - b^2
def Q (a : ℤ) : ℤ := a^2 + 2 * a + 1
def R (a b : ℤ) : ℤ := a^2 + a * b + b^2
def S (a : ℤ) : ℤ := a^2 + 2 * a - 1

theorem Q_is_perfect_square_trinomial : is_perfect_square_trinomial Q :=
sorry -- Proof goes here

end NUMINAMATH_GPT_Q_is_perfect_square_trinomial_l1096_109605


namespace NUMINAMATH_GPT_sebastian_older_than_jeremy_by_4_l1096_109634

def J : ℕ := 40
def So : ℕ := 60 - 3
def sum_ages_in_3_years (S : ℕ) : Prop := (J + 3) + (S + 3) + (So + 3) = 150

theorem sebastian_older_than_jeremy_by_4 (S : ℕ) (h : sum_ages_in_3_years S) : S - J = 4 := by
  -- proof will be filled in
  sorry

end NUMINAMATH_GPT_sebastian_older_than_jeremy_by_4_l1096_109634


namespace NUMINAMATH_GPT_cakes_difference_l1096_109673

-- Definitions of the given conditions
def cakes_sold : ℕ := 78
def cakes_bought : ℕ := 31

-- The theorem to prove
theorem cakes_difference : cakes_sold - cakes_bought = 47 :=
by sorry

end NUMINAMATH_GPT_cakes_difference_l1096_109673


namespace NUMINAMATH_GPT_percent_increase_l1096_109662

theorem percent_increase (x : ℝ) (h : (1 / 2) * x = 1) : ((x - (1 / 2)) / (1 / 2)) * 100 = 300 := by
  sorry

end NUMINAMATH_GPT_percent_increase_l1096_109662


namespace NUMINAMATH_GPT_part_a_l1096_109609

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem part_a :
  ∀ (N : ℕ), (N = (sum_of_digits N) ^ 2) → (N = 1 ∨ N = 81) :=
by
  intros N h
  sorry

end NUMINAMATH_GPT_part_a_l1096_109609


namespace NUMINAMATH_GPT_rubble_money_left_l1096_109657

/-- Rubble has $15 in his pocket. -/
def rubble_initial_amount : ℝ := 15

/-- Each notebook costs $4.00. -/
def notebook_price : ℝ := 4

/-- Each pen costs $1.50. -/
def pen_price : ℝ := 1.5

/-- Rubble needs to buy 2 notebooks. -/
def num_notebooks : ℝ := 2

/-- Rubble needs to buy 2 pens. -/
def num_pens : ℝ := 2

/-- The total cost of the notebooks. -/
def total_notebook_cost : ℝ := num_notebooks * notebook_price

/-- The total cost of the pens. -/
def total_pen_cost : ℝ := num_pens * pen_price

/-- The total amount Rubble spends. -/
def total_spent : ℝ := total_notebook_cost + total_pen_cost

/-- The remaining amount Rubble has after the purchase. -/
def rubble_remaining_amount : ℝ := rubble_initial_amount - total_spent

theorem rubble_money_left :
  rubble_remaining_amount = 4 := 
by
  -- Some necessary steps to complete the proof
  sorry

end NUMINAMATH_GPT_rubble_money_left_l1096_109657


namespace NUMINAMATH_GPT_sequence_recurrence_l1096_109633

theorem sequence_recurrence (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : a 2 = 2) (h₃ : ∀ n, n ≥ 1 → a (n + 2) / a n = (a (n + 1) ^ 2 + 1) / (a n ^ 2 + 1)):
  (∀ n, a (n + 1) = a n + 1 / a n) ∧ 63 < a 2008 ∧ a 2008 < 78 :=
by
  sorry

end NUMINAMATH_GPT_sequence_recurrence_l1096_109633


namespace NUMINAMATH_GPT_gcd_problem_l1096_109661

-- Define the variables according to the conditions
def m : ℤ := 123^2 + 235^2 + 347^2
def n : ℤ := 122^2 + 234^2 + 348^2

-- Lean statement for the proof problem
theorem gcd_problem : Int.gcd m n = 1 := sorry

end NUMINAMATH_GPT_gcd_problem_l1096_109661


namespace NUMINAMATH_GPT_total_loads_washed_l1096_109686

theorem total_loads_washed (a b : ℕ) (h1 : a = 8) (h2 : b = 6) : a + b = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_loads_washed_l1096_109686


namespace NUMINAMATH_GPT_oil_tank_depth_l1096_109696

theorem oil_tank_depth (L r A : ℝ) (h : ℝ) (L_pos : L = 8) (r_pos : r = 2) (A_pos : A = 16) :
  h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_oil_tank_depth_l1096_109696


namespace NUMINAMATH_GPT_ratio_milk_water_larger_vessel_l1096_109616

-- Definitions for the conditions given in the problem
def ratio_volume (V1 V2 : ℝ) : Prop := V1 / V2 = 3 / 5
def ratio_milk_water_vessel1 (M1 W1 : ℝ) : Prop := M1 / W1 = 1 / 2
def ratio_milk_water_vessel2 (M2 W2 : ℝ) : Prop := M2 / W2 = 3 / 2

-- The final goal to prove
theorem ratio_milk_water_larger_vessel (V1 V2 M1 W1 M2 W2 : ℝ)
  (h1 : ratio_volume V1 V2) 
  (h2 : V1 = M1 + W1) 
  (h3 : V2 = M2 + W2) 
  (h4 : ratio_milk_water_vessel1 M1 W1) 
  (h5 : ratio_milk_water_vessel2 M2 W2) :
  (M1 + M2) / (W1 + W2) = 1 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_ratio_milk_water_larger_vessel_l1096_109616


namespace NUMINAMATH_GPT_correct_masks_l1096_109627

def elephant_mask := 6
def mouse_mask := 4
def pig_mask := 8
def panda_mask := 1

theorem correct_masks :
  (elephant_mask = 6) ∧
  (mouse_mask = 4) ∧
  (pig_mask = 8) ∧
  (panda_mask = 1) := 
by
  sorry

end NUMINAMATH_GPT_correct_masks_l1096_109627


namespace NUMINAMATH_GPT_problem_solution_l1096_109660

theorem problem_solution (r : ℝ) (h : (r + 1/r)^2 = 5) : r^4 + 1/r^4 = 7 := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1096_109660


namespace NUMINAMATH_GPT_necessary_and_sufficient_problem_l1096_109620

theorem necessary_and_sufficient_problem : 
  (¬ (∀ x : ℝ, (-2 < x ∧ x < 1) → (|x| > 1)) ∧ ¬ (∀ x : ℝ, (|x| > 1) → (-2 < x ∧ x < 1))) :=
by {
  sorry
}

end NUMINAMATH_GPT_necessary_and_sufficient_problem_l1096_109620


namespace NUMINAMATH_GPT_inequality_pow_l1096_109666

variable {n : ℕ}

theorem inequality_pow (hn : n > 0) : 
  (3:ℝ) / 2 ≤ (1 + (1:ℝ) / (2 * n)) ^ n ∧ (1 + (1:ℝ) / (2 * n)) ^ n < 2 := 
sorry

end NUMINAMATH_GPT_inequality_pow_l1096_109666


namespace NUMINAMATH_GPT_retail_price_eq_120_l1096_109681

noncomputable def retail_price : ℝ :=
  let W := 90
  let P := 0.20 * W
  let SP := W + P
  SP / 0.90

theorem retail_price_eq_120 : retail_price = 120 := by
  sorry

end NUMINAMATH_GPT_retail_price_eq_120_l1096_109681


namespace NUMINAMATH_GPT_A_in_second_quadrant_l1096_109697

-- Define the coordinates of point A
def A_x : ℝ := -2
def A_y : ℝ := 3

-- Define the condition that point A lies in the second quadrant
def is_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- State the theorem
theorem A_in_second_quadrant : is_second_quadrant A_x A_y :=
by
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_A_in_second_quadrant_l1096_109697


namespace NUMINAMATH_GPT_surface_area_reduction_of_spliced_cuboid_l1096_109687

theorem surface_area_reduction_of_spliced_cuboid 
  (initial_faces : ℕ := 12)
  (faces_lost : ℕ := 2)
  (percentage_reduction : ℝ := (2 / 12) * 100) :
  percentage_reduction = 16.7 :=
by
  sorry

end NUMINAMATH_GPT_surface_area_reduction_of_spliced_cuboid_l1096_109687


namespace NUMINAMATH_GPT_age_proof_l1096_109648

theorem age_proof (y d : ℕ)
  (h1 : y = 4 * d)
  (h2 : y - 7 = 11 * (d - 7)) :
  y = 48 ∧ d = 12 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_age_proof_l1096_109648


namespace NUMINAMATH_GPT_total_crayons_in_drawer_l1096_109624

-- Definitions of conditions from a)
def initial_crayons : Nat := 9
def additional_crayons : Nat := 3

-- Statement to prove that total crayons in the drawer is 12
theorem total_crayons_in_drawer : initial_crayons + additional_crayons = 12 := sorry

end NUMINAMATH_GPT_total_crayons_in_drawer_l1096_109624


namespace NUMINAMATH_GPT_joey_pills_sum_one_week_l1096_109600

def joey_pills (n : ℕ) : ℕ :=
  1 + 2 * n

theorem joey_pills_sum_one_week : 
  (joey_pills 0) + (joey_pills 1) + (joey_pills 2) + (joey_pills 3) + (joey_pills 4) + (joey_pills 5) + (joey_pills 6) = 49 :=
by
  sorry

end NUMINAMATH_GPT_joey_pills_sum_one_week_l1096_109600


namespace NUMINAMATH_GPT_interval_length_difference_l1096_109676

noncomputable def log2_abs (x : ℝ) : ℝ := |Real.log x / Real.log 2|

theorem interval_length_difference :
  ∀ (a b : ℝ), (∀ x, a ≤ x ∧ x ≤ b → 0 ≤ log2_abs x ∧ log2_abs x ≤ 2) → 
               (b - a = 15 / 4 - 3 / 4) :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_interval_length_difference_l1096_109676


namespace NUMINAMATH_GPT_subset_complU_N_l1096_109621

variable {U : Type} {M N : Set U}

-- Given conditions
axiom non_empty_M : ∃ x, x ∈ M
axiom non_empty_N : ∃ y, y ∈ N
axiom subset_complU_M : N ⊆ Mᶜ

-- Prove the statement that M is a subset of the complement of N
theorem subset_complU_N : M ⊆ Nᶜ := by
  sorry

end NUMINAMATH_GPT_subset_complU_N_l1096_109621


namespace NUMINAMATH_GPT_quadratic_radical_simplified_l1096_109607

theorem quadratic_radical_simplified (a : ℕ) : 
  (∃ (b : ℕ), a = 3 * b^2) -> a = 3 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_radical_simplified_l1096_109607


namespace NUMINAMATH_GPT_expression_value_l1096_109690

theorem expression_value (x y z : ℕ) (hx : x = 2) (hy : y = 5) (hz : z = 3) :
  (3 * x^5 + 4 * y^3 + z^2) / 7 = 605 / 7 := by
  rw [hx, hy, hz]
  sorry

end NUMINAMATH_GPT_expression_value_l1096_109690


namespace NUMINAMATH_GPT_min_value_A_mul_abs_x1_minus_x2_l1096_109663

noncomputable def f (x : ℝ) : ℝ :=
  Real.sin (2017 * x + Real.pi / 6) + Real.cos (2017 * x - Real.pi / 3)

theorem min_value_A_mul_abs_x1_minus_x2 :
  ∃ x1 x2 : ℝ, (∀ x : ℝ, f x1 ≤ f x ∧ f x ≤ f x2) →
  2 * |x1 - x2| = (2 * Real.pi) / 2017 :=
sorry

end NUMINAMATH_GPT_min_value_A_mul_abs_x1_minus_x2_l1096_109663


namespace NUMINAMATH_GPT_cos_neg_60_equals_half_l1096_109629

  theorem cos_neg_60_equals_half : Real.cos (-60 * Real.pi / 180) = 1 / 2 :=
  by
    sorry
  
end NUMINAMATH_GPT_cos_neg_60_equals_half_l1096_109629


namespace NUMINAMATH_GPT_max_f_geq_fraction_3_sqrt3_over_2_l1096_109689

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.sin (2 * x) + Real.sin (3 * x)

theorem max_f_geq_fraction_3_sqrt3_over_2 : ∃ x : ℝ, f x ≥ (3 + Real.sqrt 3) / 2 := 
sorry

end NUMINAMATH_GPT_max_f_geq_fraction_3_sqrt3_over_2_l1096_109689


namespace NUMINAMATH_GPT_min_value_zero_l1096_109647

noncomputable def f (k x y : ℝ) : ℝ :=
  3 * x^2 - 8 * k * x * y + (4 * k^2 + 3) * y^2 - 6 * x - 6 * y + 9

theorem min_value_zero (k : ℝ) :
  (∀ x y : ℝ, f k x y ≥ 0) ↔ (k = 3 / 2 ∨ k = -3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_zero_l1096_109647
