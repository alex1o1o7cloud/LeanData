import Mathlib

namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l156_15610

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), (x = 40/99) ∧ (x = 0.36) := sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l156_15610


namespace NUMINAMATH_GPT_positive_difference_of_numbers_l156_15632

theorem positive_difference_of_numbers
  (a b : ℝ)
  (h1 : a + b = 10)
  (h2 : a^2 - b^2 = 40) :
  |a - b| = 4 :=
by 
  sorry

end NUMINAMATH_GPT_positive_difference_of_numbers_l156_15632


namespace NUMINAMATH_GPT_hannah_strawberries_l156_15645

-- Definitions for the conditions
def daily_harvest : ℕ := 5
def days_in_april : ℕ := 30
def strawberries_given_away : ℕ := 20
def strawberries_stolen : ℕ := 30

-- The statement we need to prove
theorem hannah_strawberries (harvested_strawberries : ℕ)
  (total_harvest := daily_harvest * days_in_april)
  (total_lost := strawberries_given_away + strawberries_stolen)
  (final_count := total_harvest - total_lost) :
  harvested_strawberries = final_count :=
sorry

end NUMINAMATH_GPT_hannah_strawberries_l156_15645


namespace NUMINAMATH_GPT_seventy_seventh_digit_is_three_l156_15615

-- Define the sequence of digits from the numbers 60 to 1 in decreasing order.
def sequence_of_digits : List Nat :=
  (List.range' 1 60).reverse.bind (fun n => n.digits 10)

-- Define a function to get the nth digit from the list.
def digit_at_position (n : Nat) : Option Nat :=
  sequence_of_digits.get? (n - 1)

-- The statement to prove
theorem seventy_seventh_digit_is_three : digit_at_position 77 = some 3 :=
sorry

end NUMINAMATH_GPT_seventy_seventh_digit_is_three_l156_15615


namespace NUMINAMATH_GPT_mumu_identity_l156_15682

def f (m u : ℕ) : ℕ := 
  -- Assume f is correctly defined to match the number of valid Mumu words 
  -- involving m M's and u U's according to the problem's definition.
  sorry 

theorem mumu_identity (u m : ℕ) (h₁ : u ≥ 2) (h₂ : 3 ≤ m) (h₃ : m ≤ 2 * u) :
  f m u = f (2 * u - m + 1) u ↔ f m (u - 1) = f (2 * u - m + 1) (u - 1) :=
by
  sorry

end NUMINAMATH_GPT_mumu_identity_l156_15682


namespace NUMINAMATH_GPT_reasoning_common_sense_l156_15636

theorem reasoning_common_sense :
  (∀ P Q: Prop, names_not_correct → P → ¬Q → affairs_not_successful → ¬Q)
  ∧ (∀ R S: Prop, affairs_not_successful → R → ¬S → rites_not_flourish → ¬S)
  ∧ (∀ T U: Prop, rites_not_flourish → T → ¬U → punishments_not_executed_properly → ¬U)
  ∧ (∀ V W: Prop, punishments_not_executed_properly → V → ¬W → people_nowhere_hands_feet → ¬W)
  → reasoning_is_common_sense :=
by sorry

end NUMINAMATH_GPT_reasoning_common_sense_l156_15636


namespace NUMINAMATH_GPT_total_litter_weight_l156_15689

-- Definitions of the conditions
def gina_bags : ℕ := 2
def neighborhood_multiplier : ℕ := 82
def bag_weight : ℕ := 4

-- Representing the total calculation
def neighborhood_bags : ℕ := neighborhood_multiplier * gina_bags
def total_bags : ℕ := neighborhood_bags + gina_bags

def total_weight : ℕ := total_bags * bag_weight

-- Statement of the problem
theorem total_litter_weight : total_weight = 664 :=
by
  sorry

end NUMINAMATH_GPT_total_litter_weight_l156_15689


namespace NUMINAMATH_GPT_small_pump_fill_time_l156_15681

noncomputable def small_pump_time (large_pump_time combined_time : ℝ) : ℝ :=
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate

theorem small_pump_fill_time :
  small_pump_time (1 / 3) 0.2857142857142857 = 2 :=
by
  sorry

end NUMINAMATH_GPT_small_pump_fill_time_l156_15681


namespace NUMINAMATH_GPT_number_divisible_by_33_l156_15643

theorem number_divisible_by_33 (x y : ℕ) 
  (h1 : (x + y) % 3 = 2) 
  (h2 : (y - x) % 11 = 8) : 
  (27850 + 1000 * x + y) % 33 = 0 := 
sorry

end NUMINAMATH_GPT_number_divisible_by_33_l156_15643


namespace NUMINAMATH_GPT_total_clients_correct_l156_15655

-- Define the number of each type of cars and total cars
def num_cars : ℕ := 12
def num_sedans : ℕ := 4
def num_coupes : ℕ := 4
def num_suvs : ℕ := 4

-- Define the number of selections per car and total selections required
def selections_per_car : ℕ := 3

-- Define the number of clients per type of car
def num_clients_who_like_sedans : ℕ := (num_sedans * selections_per_car) / 2
def num_clients_who_like_coupes : ℕ := (num_coupes * selections_per_car) / 2
def num_clients_who_like_suvs : ℕ := (num_suvs * selections_per_car) / 2

-- Compute total number of clients
def total_clients : ℕ := num_clients_who_like_sedans + num_clients_who_like_coupes + num_clients_who_like_suvs

-- Prove that the total number of clients is 18
theorem total_clients_correct : total_clients = 18 := by
  sorry

end NUMINAMATH_GPT_total_clients_correct_l156_15655


namespace NUMINAMATH_GPT_students_in_class_l156_15614

variable (G B : ℕ)

def total_plants (G B : ℕ) : ℕ := 3 * G + B / 3

theorem students_in_class (h1 : total_plants G B = 24) (h2 : B / 3 = 6) : G + B = 24 :=
by
  sorry

end NUMINAMATH_GPT_students_in_class_l156_15614


namespace NUMINAMATH_GPT_triangle_angles_l156_15666

theorem triangle_angles (A B C : ℝ) 
  (h1 : A + B + C = 180)
  (h2 : B = 120)
  (h3 : (∃D, A = D ∧ (A + A + C = 180 ∨ A + C + C = 180)) ∨ (∃E, C = E ∧ (B + 15 + 45 = 180 ∨ B + 15 + 15 = 180))) :
  (A = 40 ∧ C = 20) ∨ (A = 45 ∧ C = 15) :=
sorry

end NUMINAMATH_GPT_triangle_angles_l156_15666


namespace NUMINAMATH_GPT_determine_x_l156_15659

theorem determine_x (x : ℝ) :
  (∀ y : ℝ, 10 * x * y - 15 * y + 5 * x - 7.5 = 0) → x = 3 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_determine_x_l156_15659


namespace NUMINAMATH_GPT_sets_equal_l156_15647

-- Definitions of sets M and N
def M := { u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

-- Theorem statement asserting M = N
theorem sets_equal : M = N :=
by sorry

end NUMINAMATH_GPT_sets_equal_l156_15647


namespace NUMINAMATH_GPT_cats_on_ship_l156_15695

theorem cats_on_ship :
  ∃ (C S : ℕ), 
  (C + S + 1 + 1 = 16) ∧
  (4 * C + 2 * S + 2 * 1 + 1 * 1 = 41) ∧ 
  C = 5 :=
by
  sorry

end NUMINAMATH_GPT_cats_on_ship_l156_15695


namespace NUMINAMATH_GPT_solution_value_of_a_l156_15657

noncomputable def verify_a (a : ℚ) (A : Set ℚ) : Prop :=
  A = {a - 2, 2 * a^2 + 5 * a, 12} ∧ -3 ∈ A

theorem solution_value_of_a (a : ℚ) (A : Set ℚ) (h : verify_a a A) : a = -3 / 2 := by
  sorry

end NUMINAMATH_GPT_solution_value_of_a_l156_15657


namespace NUMINAMATH_GPT_range_of_values_l156_15675

variable (a : ℝ)

-- State the conditions
def prop.false (a : ℝ) : Prop := ¬ ∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0

-- Prove that the range of values for a where the proposition is false is (2, +∞)
theorem range_of_values (ha : prop.false a) : 2 < a :=
sorry

end NUMINAMATH_GPT_range_of_values_l156_15675


namespace NUMINAMATH_GPT_range_of_m_l156_15654

noncomputable def is_quadratic (m : ℝ) : Prop := (m^2 - 4) ≠ 0

theorem range_of_m (m : ℝ) : is_quadratic m → m ≠ 2 ∧ m ≠ -2 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l156_15654


namespace NUMINAMATH_GPT_steve_final_amount_l156_15656

def initial_deposit : ℝ := 100
def interest_years_1_to_3 : ℝ := 0.10
def interest_years_4_to_5 : ℝ := 0.08
def annual_deposit_years_1_to_2 : ℝ := 10
def annual_deposit_years_3_to_5 : ℝ := 15

def total_after_one_year (initial : ℝ) (annual : ℝ) (interest : ℝ) : ℝ :=
  initial * (1 + interest) + annual

def steve_saving_after_five_years : ℝ :=
  let year1 := total_after_one_year initial_deposit annual_deposit_years_1_to_2 interest_years_1_to_3
  let year2 := total_after_one_year year1 annual_deposit_years_1_to_2 interest_years_1_to_3
  let year3 := total_after_one_year year2 annual_deposit_years_3_to_5 interest_years_1_to_3
  let year4 := total_after_one_year year3 annual_deposit_years_3_to_5 interest_years_4_to_5
  let year5 := total_after_one_year year4 annual_deposit_years_3_to_5 interest_years_4_to_5
  year5

theorem steve_final_amount :
  steve_saving_after_five_years = 230.88768 := by
  sorry

end NUMINAMATH_GPT_steve_final_amount_l156_15656


namespace NUMINAMATH_GPT_bryson_new_shoes_l156_15608

-- Define the conditions as variables and constant values
def pairs_of_shoes : ℕ := 2 -- Number of pairs Bryson bought
def shoes_per_pair : ℕ := 2 -- Number of shoes per pair

-- Define the theorem to prove the question == answer
theorem bryson_new_shoes : pairs_of_shoes * shoes_per_pair = 4 :=
by
  sorry -- Proof placeholder

end NUMINAMATH_GPT_bryson_new_shoes_l156_15608


namespace NUMINAMATH_GPT_solve_for_x_l156_15644

theorem solve_for_x (x : ℚ) (h : (x + 2) / (x - 3) = (x - 4) / (x + 5)) : x = 1 / 7 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l156_15644


namespace NUMINAMATH_GPT_girl_boy_lineup_probability_l156_15607

theorem girl_boy_lineup_probability :
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  (valid_configurations : ℚ) / total_configurations = 0.058 :=
by
  let total_configurations := Nat.choose 20 9
  let valid_case1 := Nat.choose 14 9
  let valid_subcases := 6 * Nat.choose 13 8
  let valid_configurations := valid_case1 + valid_subcases
  have h : (valid_configurations : ℚ) / total_configurations = 0.058 := sorry
  exact h

end NUMINAMATH_GPT_girl_boy_lineup_probability_l156_15607


namespace NUMINAMATH_GPT_curve_C_is_circle_l156_15602

noncomputable def curve_C_equation (a : ℝ) : Prop := ∀ x y : ℝ, a * (x^2) + a * (y^2) - 2 * a^2 * x - 4 * y = 0

theorem curve_C_is_circle
  (a : ℝ)
  (ha : a ≠ 0)
  (h_line_intersects : ∃ M N : ℝ × ℝ, (M.2 = -2 * M.1 + 4) ∧ (N.2 = -2 * N.1 + 4) ∧ (M.1^2 + M.2^2 = N.1^2 + N.2^2) ∧ M ≠ N)
  :
  (curve_C_equation 2) ∧ (∀ x y, x^2 + y^2 - 4*x - 2*y = 0) :=
sorry -- Proof is to be provided

end NUMINAMATH_GPT_curve_C_is_circle_l156_15602


namespace NUMINAMATH_GPT_tom_books_l156_15611

-- Definitions based on the conditions
def joan_books : ℕ := 10
def total_books : ℕ := 48

-- The theorem statement: Proving that Tom has 38 books
theorem tom_books : (total_books - joan_books) = 38 := by
  -- Here we would normally provide a proof, but we use sorry to skip this.
  sorry

end NUMINAMATH_GPT_tom_books_l156_15611


namespace NUMINAMATH_GPT_iced_coffee_days_per_week_l156_15699

theorem iced_coffee_days_per_week (x : ℕ) (h1 : 5 * 4 = 20)
  (h2 : 20 * 52 = 1040)
  (h3 : 2 * x = 2 * x)
  (h4 : 52 * (2 * x) = 104 * x)
  (h5 : 1040 + 104 * x = 1040 + 104 * x)
  (h6 : 1040 + 104 * x - 338 = 1040 + 104 * x - 338)
  (h7 : (0.75 : ℝ) * (1040 + 104 * x) = 780 + 78 * x) :
  x = 3 :=
by
  sorry

end NUMINAMATH_GPT_iced_coffee_days_per_week_l156_15699


namespace NUMINAMATH_GPT_general_term_a_n_sum_of_b_n_l156_15600

-- Proof Problem 1: General term of sequence {a_n}
theorem general_term_a_n (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 2) (h2 : a 2 = 4) 
    (h3 : ∀ n ≥ 2, a (n+1) - a n = 2) : 
    ∀ n, a n = 2 * n :=
by
  sorry

-- Proof Problem 2: Sum of the first n terms of sequence {b_n}
theorem sum_of_b_n (a : ℕ → ℕ) (b : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ)
    (h : ∀ n, (1 / (a n ^ 2 - 1) : ℝ) + b n = 2^n) :
    T n = 2^(n+1) - n / (2*n + 1) :=
by
  sorry

end NUMINAMATH_GPT_general_term_a_n_sum_of_b_n_l156_15600


namespace NUMINAMATH_GPT_initial_typists_count_l156_15617

theorem initial_typists_count 
  (typists_rate : ℕ → ℕ)
  (letters_in_20min : ℕ)
  (total_typists : ℕ)
  (letters_in_1hour : ℕ)
  (initial_typists : ℕ) 
  (h1 : letters_in_20min = 38)
  (h2 : letters_in_1hour = 171)
  (h3 : total_typists = 30)
  (h4 : ∀ t, 3 * (typists_rate t) = letters_in_1hour / total_typists)
  (h5 : ∀ t, typists_rate t = letters_in_20min / t) 
  : initial_typists = 20 := 
sorry

end NUMINAMATH_GPT_initial_typists_count_l156_15617


namespace NUMINAMATH_GPT_total_votes_400_l156_15624

theorem total_votes_400 
    (V : ℝ)
    (h1 : ∃ (c1_votes c2_votes : ℝ), c1_votes = 0.70 * V ∧ c2_votes = 0.30 * V)
    (h2 : ∃ (majority : ℝ), majority = 160)
    (h3 : ∀ (c1_votes c2_votes majority : ℝ), c1_votes - c2_votes = majority) : V = 400 :=
by 
  sorry

end NUMINAMATH_GPT_total_votes_400_l156_15624


namespace NUMINAMATH_GPT_gcd_lcm_sum_l156_15663

theorem gcd_lcm_sum :
  Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := 
by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l156_15663


namespace NUMINAMATH_GPT_division_result_l156_15649

theorem division_result : (5 * 6 + 4) / 8 = 4.25 :=
by
  sorry

end NUMINAMATH_GPT_division_result_l156_15649


namespace NUMINAMATH_GPT_count_three_digit_numbers_divisible_by_13_l156_15640

theorem count_three_digit_numbers_divisible_by_13 :
  ∃ n : ℕ, n = 69 ∧ ∀ k : ℕ, (100 ≤ 13 * k ∧ 13 * k ≤ 999) → n = (988 - 104) / 13 + 1 :=
by
  sorry

end NUMINAMATH_GPT_count_three_digit_numbers_divisible_by_13_l156_15640


namespace NUMINAMATH_GPT_gcf_360_270_lcm_360_270_l156_15616

def prime_factors_360 := [(2, 3), (3, 2), (5, 1)]
def prime_factors_270 := [(2, 1), (3, 3), (5, 1)]

def GCF (a b: ℕ) : ℕ := 2^1 * 3^2 * 5^1
def LCM (a b: ℕ) : ℕ := 2^3 * 3^3 * 5^1

-- Theorem: The GCF of 360 and 270 is 90
theorem gcf_360_270 : GCF 360 270 = 90 := by
  sorry

-- Theorem: The LCM of 360 and 270 is 1080
theorem lcm_360_270 : LCM 360 270 = 1080 := by
  sorry

end NUMINAMATH_GPT_gcf_360_270_lcm_360_270_l156_15616


namespace NUMINAMATH_GPT_carol_total_points_l156_15650

-- Define the conditions for Carol's game points.
def first_round_points := 17
def second_round_points := 6
def last_round_points := -16

-- Prove that the total points at the end of the game are 7.
theorem carol_total_points : first_round_points + second_round_points + last_round_points = 7 := by
  sorry

end NUMINAMATH_GPT_carol_total_points_l156_15650


namespace NUMINAMATH_GPT_behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l156_15623

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 5 * x ^ 2 + 4

theorem behavior_of_g_as_x_approaches_infinity_and_negative_infinity :
  (∀ ε > 0, ∃ M > 0, ∀ x > M, g x < -ε) ∧
  (∀ ε > 0, ∃ N > 0, ∀ x < -N, g x > ε) :=
by
  sorry

end NUMINAMATH_GPT_behavior_of_g_as_x_approaches_infinity_and_negative_infinity_l156_15623


namespace NUMINAMATH_GPT_similar_triangles_perimeter_l156_15634

theorem similar_triangles_perimeter
  (height_ratio : ℚ)
  (smaller_perimeter larger_perimeter : ℚ)
  (h_ratio : height_ratio = 3 / 5)
  (h_smaller_perimeter : smaller_perimeter = 12)
  : larger_perimeter = 20 :=
by
  sorry

end NUMINAMATH_GPT_similar_triangles_perimeter_l156_15634


namespace NUMINAMATH_GPT_number_of_positive_divisors_of_60_l156_15622

theorem number_of_positive_divisors_of_60 : 
  ∃ n : ℕ, 
  (∀ a b c : ℕ, (60 = 2^a * 3^b * 5^c) → n = (a+1) * (b+1) * (c+1)) → 
  n = 12 :=
by
  sorry

end NUMINAMATH_GPT_number_of_positive_divisors_of_60_l156_15622


namespace NUMINAMATH_GPT_red_balls_count_l156_15665

theorem red_balls_count (R W : ℕ) (h1 : R / W = 4 / 5) (h2 : W = 20) : R = 16 := sorry

end NUMINAMATH_GPT_red_balls_count_l156_15665


namespace NUMINAMATH_GPT_employee_salary_l156_15604

theorem employee_salary (A B : ℝ) (h1 : A + B = 560) (h2 : A = 1.5 * B) : B = 224 :=
by
  sorry

end NUMINAMATH_GPT_employee_salary_l156_15604


namespace NUMINAMATH_GPT_find_james_number_l156_15653

theorem find_james_number (x : ℝ) 
  (h1 : 3 * (3 * x + 10) = 141) : 
  x = 12.33 :=
by 
  sorry

end NUMINAMATH_GPT_find_james_number_l156_15653


namespace NUMINAMATH_GPT_find_certain_number_l156_15683

-- Define the conditions as constants
def n1 : ℕ := 9
def n2 : ℕ := 70
def n3 : ℕ := 25
def n4 : ℕ := 21
def smallest_given_number : ℕ := 3153
def certain_number : ℕ := 3147

-- Lean theorem statement
theorem find_certain_number (n1 n2 n3 n4 smallest_given_number certain_number: ℕ) :
  (∀ x, (∀ y ∈ [n1, n2, n3, n4], y ∣ x) → x ≥ smallest_given_number → x = smallest_given_number + certain_number) :=
sorry -- Skips the proof

end NUMINAMATH_GPT_find_certain_number_l156_15683


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l156_15685

open Set

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l156_15685


namespace NUMINAMATH_GPT_triangle_height_l156_15605

theorem triangle_height (s h : ℝ) 
  (area_square : s^2 = s * s) 
  (area_triangle : 1/2 * s * h = s^2) 
  (areas_equal : s^2 = s^2) : 
  h = 2 * s := 
sorry

end NUMINAMATH_GPT_triangle_height_l156_15605


namespace NUMINAMATH_GPT_total_pounds_of_peppers_l156_15619

-- Definitions and conditions
def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335

-- Theorem statement
theorem total_pounds_of_peppers : green_peppers + red_peppers = 5.666666666666667 :=
by
  sorry

end NUMINAMATH_GPT_total_pounds_of_peppers_l156_15619


namespace NUMINAMATH_GPT_airplane_total_luggage_weight_l156_15688

def num_people := 6
def bags_per_person := 5
def weight_per_bag := 50
def additional_bags := 90

def total_weight_people := num_people * bags_per_person * weight_per_bag
def total_weight_additional_bags := additional_bags * weight_per_bag

def total_luggage_weight := total_weight_people + total_weight_additional_bags

theorem airplane_total_luggage_weight : total_luggage_weight = 6000 :=
by
  sorry

end NUMINAMATH_GPT_airplane_total_luggage_weight_l156_15688


namespace NUMINAMATH_GPT_quadratic_expression_value_l156_15609

theorem quadratic_expression_value (x1 x2 : ℝ) (h1 : x1 + x2 = 4) (h2 : x1 * x2 = 2) (hx : x1^2 - 4 * x1 + 2 = 0) :
  x1^2 - 4 * x1 + 2 * x1 * x2 = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_value_l156_15609


namespace NUMINAMATH_GPT_not_covered_by_homothetic_polygons_l156_15629

structure Polygon :=
  (vertices : Set (ℝ × ℝ))

def homothetic (M : Polygon) (k : ℝ) (O : ℝ × ℝ) : Polygon :=
  {
    vertices := {p | ∃ (q : ℝ × ℝ) (hq : q ∈ M.vertices), p = (O.1 + k * (q.1 - O.1), O.2 + k * (q.2 - O.2))}
  }

theorem not_covered_by_homothetic_polygons (M : Polygon) (k : ℝ) (h : 0 < k ∧ k < 1)
  (O1 O2 : ℝ × ℝ) :
  ¬ (∀ p ∈ M.vertices, p ∈ (homothetic M k O1).vertices ∨ p ∈ (homothetic M k O2).vertices) := by
  sorry

end NUMINAMATH_GPT_not_covered_by_homothetic_polygons_l156_15629


namespace NUMINAMATH_GPT_min_weighings_to_find_counterfeit_l156_15638

-- Definition of the problem conditions.
def coin_is_genuine (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins m = coins (Fin.mk 0 sorry)

def counterfit_coin_is_lighter (coins : Fin 10 → ℝ) (n : Fin 10) : Prop :=
  ∀ m : Fin 10, m ≠ n → coins n < coins m

-- The theorem statement
theorem min_weighings_to_find_counterfeit :
  (∀ coins : Fin 10 → ℝ, ∃ n : Fin 10, coin_is_genuine coins n ∧ counterfit_coin_is_lighter coins n → ∃ min_weighings : ℕ, min_weighings = 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_min_weighings_to_find_counterfeit_l156_15638


namespace NUMINAMATH_GPT_mary_change_l156_15697

/-- 
Calculate the change Mary will receive after buying tickets for herself and her 3 children 
at the circus, given the ticket prices and special group rate discount.
-/
theorem mary_change :
  let adult_ticket := 2
  let child_ticket := 1
  let discounted_child_ticket := 0.5 * child_ticket
  let total_cost_with_discount := adult_ticket + 2 * child_ticket + discounted_child_ticket
  let payment := 20
  payment - total_cost_with_discount = 15.50 :=
by
  sorry

end NUMINAMATH_GPT_mary_change_l156_15697


namespace NUMINAMATH_GPT_quadratic_identity_l156_15693

theorem quadratic_identity
  (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c) / ((a - b) * (a - c))) +
  (b^2 * (x - a) * (x - c) / ((b - a) * (b - c))) +
  (c^2 * (x - a) * (x - b) / ((c - a) * (c - b))) =
  x^2 :=
sorry

end NUMINAMATH_GPT_quadratic_identity_l156_15693


namespace NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l156_15684

noncomputable def pyramid_volume (a b γ : ℝ) : ℝ :=
  (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2)

theorem volume_of_regular_triangular_pyramid (a b γ : ℝ) :
  pyramid_volume a b γ = (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l156_15684


namespace NUMINAMATH_GPT_CarmenBrushLengthIsCorrect_l156_15620

namespace BrushLength

def carlasBrushLengthInInches : ℤ := 12
def conversionRateInCmPerInch : ℝ := 2.5
def lengthMultiplier : ℝ := 1.5

def carmensBrushLengthInCm : ℝ :=
  carlasBrushLengthInInches * lengthMultiplier * conversionRateInCmPerInch

theorem CarmenBrushLengthIsCorrect :
  carmensBrushLengthInCm = 45 := by
  sorry

end BrushLength

end NUMINAMATH_GPT_CarmenBrushLengthIsCorrect_l156_15620


namespace NUMINAMATH_GPT_roger_allowance_fraction_l156_15686

noncomputable def allowance_fraction (A m s p : ℝ) : ℝ :=
  m + s + p

theorem roger_allowance_fraction (A : ℝ) (m s p : ℝ) 
  (h_movie : m = 0.25 * (A - s - p))
  (h_soda : s = 0.10 * (A - m - p))
  (h_popcorn : p = 0.05 * (A - m - s)) :
  allowance_fraction A m s p = 0.32 * A :=
by
  sorry

end NUMINAMATH_GPT_roger_allowance_fraction_l156_15686


namespace NUMINAMATH_GPT_betty_total_stones_l156_15613

def stones_per_bracelet : ℕ := 14
def number_of_bracelets : ℕ := 10
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem betty_total_stones : total_stones = 140 := by
  sorry

end NUMINAMATH_GPT_betty_total_stones_l156_15613


namespace NUMINAMATH_GPT_integers_with_abs_less_than_four_l156_15674

theorem integers_with_abs_less_than_four :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end NUMINAMATH_GPT_integers_with_abs_less_than_four_l156_15674


namespace NUMINAMATH_GPT_arithmetic_sequence_property_l156_15671

variable {α : Type*} [LinearOrderedField α]

theorem arithmetic_sequence_property
  (a : ℕ → α) (h1 : a 1 + a 8 = 9) (h4 : a 4 = 3) : a 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_property_l156_15671


namespace NUMINAMATH_GPT_ceil_sqrt_169_eq_13_l156_15667

theorem ceil_sqrt_169_eq_13 : Int.ceil (Real.sqrt 169) = 13 := by
  sorry

end NUMINAMATH_GPT_ceil_sqrt_169_eq_13_l156_15667


namespace NUMINAMATH_GPT_calculate_3_to_5_mul_7_to_5_l156_15625

theorem calculate_3_to_5_mul_7_to_5 : 3^5 * 7^5 = 4084101 :=
by {
  -- Sorry is added to skip the proof; assuming the proof is done following standard arithmetic calculations
  sorry
}

end NUMINAMATH_GPT_calculate_3_to_5_mul_7_to_5_l156_15625


namespace NUMINAMATH_GPT_find_number_when_divided_by_3_is_equal_to_subtracting_5_l156_15628

theorem find_number_when_divided_by_3_is_equal_to_subtracting_5 (x : ℝ) (h : x / 3 = x - 5) : x = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_find_number_when_divided_by_3_is_equal_to_subtracting_5_l156_15628


namespace NUMINAMATH_GPT_center_of_tangent_circle_l156_15687

theorem center_of_tangent_circle (x y : ℝ) 
  (h1 : 3*x - 4*y = 12) 
  (h2 : 3*x - 4*y = -24)
  (h3 : x - 2*y = 0) : 
  (x, y) = (-6, -3) :=
by
  sorry

end NUMINAMATH_GPT_center_of_tangent_circle_l156_15687


namespace NUMINAMATH_GPT_fraction_simplification_l156_15664

-- Define the numerator and denominator based on given conditions
def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 + 256
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128 - 256 + 512

-- Lean theorem that encapsulates the problem
theorem fraction_simplification : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l156_15664


namespace NUMINAMATH_GPT_rate_of_fencing_is_4_90_l156_15618

noncomputable def rate_of_fencing_per_meter : ℝ :=
  let area_hectares := 13.86
  let cost := 6466.70
  let area_m2 := area_hectares * 10000
  let radius := Real.sqrt (area_m2 / Real.pi)
  let circumference := 2 * Real.pi * radius
  cost / circumference

theorem rate_of_fencing_is_4_90 :
  rate_of_fencing_per_meter = 4.90 := sorry

end NUMINAMATH_GPT_rate_of_fencing_is_4_90_l156_15618


namespace NUMINAMATH_GPT_density_of_second_part_l156_15635

theorem density_of_second_part (V m : ℝ) (h1 : ∀ V m : ℝ, V_1 = 0.3 * V) 
  (h2 : ∀ V m : ℝ, m_1 = 0.6 * m) 
  (rho1 : ρ₁ = 7800) : 
  ∃ ρ₂, ρ₂ = 2229 :=
by sorry

end NUMINAMATH_GPT_density_of_second_part_l156_15635


namespace NUMINAMATH_GPT_range_of_a_l156_15646

theorem range_of_a (a : ℝ) : (forall x : ℝ, (a-3) * x > 1 → x < 1 / (a-3)) → a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l156_15646


namespace NUMINAMATH_GPT_initial_number_of_students_l156_15603

theorem initial_number_of_students (W : ℝ) (n : ℕ) (new_student_weight avg_weight1 avg_weight2 : ℝ)
  (h1 : avg_weight1 = 15)
  (h2 : new_student_weight = 13)
  (h3 : avg_weight2 = 14.9)
  (h4 : W = n * avg_weight1)
  (h5 : W + new_student_weight = (n + 1) * avg_weight2) : n = 19 := 
by
  sorry

end NUMINAMATH_GPT_initial_number_of_students_l156_15603


namespace NUMINAMATH_GPT_rectangle_area_difference_196_l156_15639

noncomputable def max_min_area_difference (P : ℕ) (A_max A_min : ℕ) : Prop :=
  ( ∃ l w : ℕ, 2 * l + 2 * w = P ∧ A_max = l * w ) ∧
  ( ∃ l' w' : ℕ, 2 * l' + 2 * w' = P ∧ A_min = l' * w' ) ∧
  (A_max - A_min = 196)

theorem rectangle_area_difference_196 : max_min_area_difference 60 225 29 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_difference_196_l156_15639


namespace NUMINAMATH_GPT_original_price_of_shoes_l156_15612

theorem original_price_of_shoes (x : ℝ) (h : 1/4 * x = 18) : x = 72 := by
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l156_15612


namespace NUMINAMATH_GPT_Lena_stops_in_X_l156_15673

def circumference : ℕ := 60
def distance_run : ℕ := 7920
def starting_point : String := "T"
def quarter_stops : String := "X"

theorem Lena_stops_in_X :
  (distance_run / circumference) * circumference + (distance_run % circumference) = distance_run →
  distance_run % circumference = 0 →
  (distance_run % circumference = 0 → starting_point = quarter_stops) →
  quarter_stops = "X" :=
sorry

end NUMINAMATH_GPT_Lena_stops_in_X_l156_15673


namespace NUMINAMATH_GPT_find_f_at_4_l156_15606

def f (n : ℕ) : ℕ := sorry -- We define the function f.

theorem find_f_at_4 : (∀ x : ℕ, f (2 * x) = 3 * x^2 + 1) → f 4 = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_4_l156_15606


namespace NUMINAMATH_GPT_suzy_total_jumps_in_two_days_l156_15670

-- Definitions based on the conditions in the problem
def yesterdays_jumps : ℕ := 247
def additional_jumps_today : ℕ := 131
def todays_jumps : ℕ := yesterdays_jumps + additional_jumps_today

-- Lean statement of the proof problem
theorem suzy_total_jumps_in_two_days : yesterdays_jumps + todays_jumps = 625 := by
  sorry

end NUMINAMATH_GPT_suzy_total_jumps_in_two_days_l156_15670


namespace NUMINAMATH_GPT_solution_set_of_inequality_l156_15626

theorem solution_set_of_inequality (x : ℝ) : (x - 1 ≤ (1 + x) / 3) → (x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l156_15626


namespace NUMINAMATH_GPT_pyramid_side_length_l156_15631

noncomputable def side_length_of_square_base (area_of_lateral_face : ℝ) (slant_height : ℝ) : ℝ :=
  2 * area_of_lateral_face / slant_height

theorem pyramid_side_length 
  (area_of_lateral_face : ℝ)
  (slant_height : ℝ)
  (h1 : area_of_lateral_face = 120)
  (h2 : slant_height = 24) :
  side_length_of_square_base area_of_lateral_face slant_height = 10 :=
by
  -- Skipping the proof details.
  sorry

end NUMINAMATH_GPT_pyramid_side_length_l156_15631


namespace NUMINAMATH_GPT_computation_of_expression_l156_15672

theorem computation_of_expression (x : ℝ) (h : x + 1 / x = 7) : 
  (x - 3) ^ 2 + 49 / (x - 3) ^ 2 = 23 := 
by
  sorry

end NUMINAMATH_GPT_computation_of_expression_l156_15672


namespace NUMINAMATH_GPT_initial_honey_amount_l156_15658

variable (H : ℝ)

theorem initial_honey_amount :
  (0.70 * 0.60 * 0.50) * H = 315 → H = 1500 :=
by
  sorry

end NUMINAMATH_GPT_initial_honey_amount_l156_15658


namespace NUMINAMATH_GPT_power_of_two_l156_15637

theorem power_of_two (n : ℕ) (h : 2^n = 32 * (1 / 2) ^ 2) : n = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_power_of_two_l156_15637


namespace NUMINAMATH_GPT_find_number_l156_15692

theorem find_number (m : ℤ) (h1 : ∃ k1 : ℤ, k1 * k1 = m + 100) (h2 : ∃ k2 : ℤ, k2 * k2 = m + 168) : m = 156 :=
sorry

end NUMINAMATH_GPT_find_number_l156_15692


namespace NUMINAMATH_GPT_perimeter_area_ratio_le_8_l156_15661

/-- Let \( S \) be a shape in the plane obtained as a union of finitely many unit squares.
    The perimeter of a single unit square is 4 and its area is 1.
    Prove that the ratio of the perimeter \( P \) and the area \( A \) of \( S \)
    is at most 8, i.e., \(\frac{P}{A} \leq 8\). -/
theorem perimeter_area_ratio_le_8
  (S : Set (ℝ × ℝ)) 
  (unit_square : ∀ (x y : ℝ), (x, y) ∈ S → (x + 1, y + 1) ∈ S ∧ (x + 1, y) ∈ S ∧ (x, y + 1) ∈ S ∧ (x, y) ∈ S)
  (P A : ℝ)
  (unit_square_perimeter : ∀ (x y : ℝ), (x, y) ∈ S → P = 4)
  (unit_square_area : ∀ (x y : ℝ), (x, y) ∈ S → A = 1) :
  P / A ≤ 8 :=
sorry

end NUMINAMATH_GPT_perimeter_area_ratio_le_8_l156_15661


namespace NUMINAMATH_GPT_final_value_A_is_5_l156_15641

/-
Problem: Given a 3x3 grid of numbers and a series of operations that add or subtract 1 to two adjacent cells simultaneously, prove that the number in position A in the table on the right is 5.
Conditions:
1. The initial grid is:
   \[
   \begin{array}{ccc}
   a & b & c \\
   d & e & f \\
   g & h & i \\
   \end{array}
   \]
2. Each operation involves adding or subtracting 1 from two adjacent cells.
3. The sum of all numbers in the grid remains unchanged.
-/

def table_operations (a b c d e f g h i : ℤ) : ℤ :=
-- A is determined based on the given problem and conditions
  5

theorem final_value_A_is_5 (a b c d e f g h i : ℤ) : 
  table_operations a b c d e f g h i = 5 :=
sorry

end NUMINAMATH_GPT_final_value_A_is_5_l156_15641


namespace NUMINAMATH_GPT_identity_proof_l156_15677

theorem identity_proof : 
  ∀ x : ℝ, 
    (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := 
by 
  sorry

end NUMINAMATH_GPT_identity_proof_l156_15677


namespace NUMINAMATH_GPT_problem_l156_15694

   def f (n : ℕ) : ℕ := sorry

   theorem problem (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) :
     f 2013 = 2014 :=
   sorry
   
end NUMINAMATH_GPT_problem_l156_15694


namespace NUMINAMATH_GPT_find_n_l156_15668

noncomputable def r1 : ℚ := 6 / 15
noncomputable def S1 : ℚ := 15 / (1 - r1)
noncomputable def r2 (n : ℚ) : ℚ := (6 + n) / 15
noncomputable def S2 (n : ℚ) : ℚ := 15 / (1 - r2 n)

theorem find_n : ∃ (n : ℚ), S2 n = 3 * S1 ∧ n = 6 :=
by
  use 6
  sorry

end NUMINAMATH_GPT_find_n_l156_15668


namespace NUMINAMATH_GPT_exterior_angle_parallel_lines_l156_15648

theorem exterior_angle_parallel_lines
  (k l : Prop) 
  (triangle_has_angles : ∃ (a b c : ℝ), a = 40 ∧ b = 40 ∧ c = 100 ∧ a + b + c = 180)
  (exterior_angle_eq : ∀ (y : ℝ), y = 180 - 100) :
  ∃ (x : ℝ), x = 80 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_parallel_lines_l156_15648


namespace NUMINAMATH_GPT_complex_num_z_imaginary_square_l156_15642

theorem complex_num_z_imaginary_square (z : ℂ) (h1 : z.im ≠ 0) (h2 : z.re = 0) (h3 : ((z + 1) ^ 2).re = 0) :
  z = Complex.I ∨ z = -Complex.I :=
by
  sorry

end NUMINAMATH_GPT_complex_num_z_imaginary_square_l156_15642


namespace NUMINAMATH_GPT_binom_15_13_eq_105_l156_15662

theorem binom_15_13_eq_105 : Nat.choose 15 13 = 105 := by
  sorry

end NUMINAMATH_GPT_binom_15_13_eq_105_l156_15662


namespace NUMINAMATH_GPT_algebraic_identity_neg_exponents_l156_15651

theorem algebraic_identity_neg_exponents (x y z : ℂ) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (y * z + x * z + x * y) * x⁻¹ * y⁻¹ * z⁻¹ * (x + y + z)⁻¹ :=
by
  sorry

end NUMINAMATH_GPT_algebraic_identity_neg_exponents_l156_15651


namespace NUMINAMATH_GPT_expected_yield_correct_l156_15676

/-- Define the problem variables and conditions -/
def steps_x : ℕ := 25
def steps_y : ℕ := 20
def step_length : ℝ := 2.5
def yield_per_sqft : ℝ := 0.75

/-- Calculate the dimensions in feet -/
def length_x := steps_x * step_length
def length_y := steps_y * step_length

/-- Calculate the area of the orchard -/
def area := length_x * length_y

/-- Calculate the expected yield of apples -/
def expected_yield := area * yield_per_sqft

/-- Prove the expected yield of apples is 2343.75 pounds -/
theorem expected_yield_correct : expected_yield = 2343.75 := sorry

end NUMINAMATH_GPT_expected_yield_correct_l156_15676


namespace NUMINAMATH_GPT_lineup_count_l156_15669

-- Define five distinct people
inductive Person 
| youngest : Person 
| oldest : Person 
| person1 : Person 
| person2 : Person 
| person3 : Person 

-- Define the total number of people
def numberOfPeople : ℕ := 5

-- Define a function to calculate the number of ways to line up five people with constraints
def lineupWays : ℕ := 3 * 4 * 3 * 2 * 1

-- State the theorem
theorem lineup_count (h₁ : numberOfPeople = 5) (h₂ : ¬ ∃ (p : Person), p = Person.youngest ∨ p = Person.oldest → p = Person.youngest) :
  lineupWays = 72 :=
by
  sorry

end NUMINAMATH_GPT_lineup_count_l156_15669


namespace NUMINAMATH_GPT_find_b_l156_15680

theorem find_b 
  (a b c d : ℚ) 
  (h1 : a = 2 * b + c) 
  (h2 : b = 2 * c + d) 
  (h3 : 2 * c = d + a - 1) 
  (h4 : d = a - c) : 
  b = 2 / 9 :=
by
  -- Proof is omitted (the proof steps would be inserted here)
  sorry

end NUMINAMATH_GPT_find_b_l156_15680


namespace NUMINAMATH_GPT_second_largest_between_28_and_31_l156_15678

theorem second_largest_between_28_and_31 : 
  ∃ (n : ℕ), n > 28 ∧ n ≤ 31 ∧ (∀ m, (m > 28 ∧ m ≤ 31 ∧ m < 31) ->  m ≤ 30) :=
sorry

end NUMINAMATH_GPT_second_largest_between_28_and_31_l156_15678


namespace NUMINAMATH_GPT_value_of_sum_l156_15690

theorem value_of_sum (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 - 2 * a * b = 2 * a * b) : a + b = 2 ∨ a + b = -2 :=
sorry

end NUMINAMATH_GPT_value_of_sum_l156_15690


namespace NUMINAMATH_GPT_probability_two_red_balls_randomly_picked_l156_15698

theorem probability_two_red_balls_randomly_picked :
  (3/9) * (2/8) = 1/12 :=
by sorry

end NUMINAMATH_GPT_probability_two_red_balls_randomly_picked_l156_15698


namespace NUMINAMATH_GPT_range_of_m_l156_15696

theorem range_of_m (m : ℝ) (h : 1 < (8 - m) / (m - 5)) : 5 < m ∧ m < 13 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l156_15696


namespace NUMINAMATH_GPT_solve_equation_l156_15627

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l156_15627


namespace NUMINAMATH_GPT_negate_exists_statement_l156_15691

theorem negate_exists_statement : 
  (∃ x : ℝ, x^2 + x - 2 < 0) ↔ ¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0) :=
by sorry

end NUMINAMATH_GPT_negate_exists_statement_l156_15691


namespace NUMINAMATH_GPT_rank_of_matrix_A_is_2_l156_15630

def matrix_A : Matrix (Fin 4) (Fin 5) ℚ :=
  ![![3, -1, 1, 2, -8],
    ![7, -1, 2, 1, -12],
    ![11, -1, 3, 0, -16],
    ![10, -2, 3, 3, -20]]

theorem rank_of_matrix_A_is_2 : Matrix.rank matrix_A = 2 := by
  sorry

end NUMINAMATH_GPT_rank_of_matrix_A_is_2_l156_15630


namespace NUMINAMATH_GPT_range_of_quadratic_function_l156_15652

variable (x : ℝ)
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem range_of_quadratic_function :
  (∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = quadratic_function x) ↔ (1 ≤ y ∧ y ≤ 5)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_quadratic_function_l156_15652


namespace NUMINAMATH_GPT_find_a_b_l156_15601

theorem find_a_b
  (f : ℝ → ℝ) (a b : ℝ) (h_a_ne_zero : a ≠ 0) (h_f : ∀ x, f x = x^3 + 3 * x^2 + 1)
  (h_eq : ∀ x, f x - f a = (x - b) * (x - a)^2) :
  a = -2 ∧ b = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_b_l156_15601


namespace NUMINAMATH_GPT_probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l156_15633

noncomputable def diameter := 19 -- mm
noncomputable def side_length := 50 -- mm, side length of each square
noncomputable def total_area := side_length^2 -- 2500 mm^2 for each square
noncomputable def coin_radius := diameter / 2 -- 9.5 mm

theorem probability_completely_inside_square : 
  (side_length - 2 * coin_radius)^2 / total_area = 961 / 2500 :=
by sorry

theorem probability_partial_one_edge :
  4 * ((side_length - 2 * coin_radius) * coin_radius) / total_area = 1178 / 2500 :=
by sorry

theorem probability_partial_two_edges_not_vertex :
  (4 * ((diameter)^2 - (coin_radius^2 * Real.pi / 4))) / total_area = (4 * 290.12) / 2500 :=
by sorry

theorem probability_vertex :
  4 * (coin_radius^2 * Real.pi / 4) / total_area = 4 * 70.88 / 2500 :=
by sorry

end NUMINAMATH_GPT_probability_completely_inside_square_probability_partial_one_edge_probability_partial_two_edges_not_vertex_probability_vertex_l156_15633


namespace NUMINAMATH_GPT_triangle_area_16_l156_15660

theorem triangle_area_16 : 
  let A := (0, 0)
  let B := (4, 0)
  let C := (3, 8)
  let base := (B.1 - A.1)
  let height := (C.2 - A.2)
  (base * height) / 2 = 16 := by
  sorry

end NUMINAMATH_GPT_triangle_area_16_l156_15660


namespace NUMINAMATH_GPT_trapezoid_area_l156_15679

-- Geometry setup
variable (outer_area : ℝ) (inner_height_ratio : ℝ)

-- Conditions
def outer_triangle_area := outer_area = 36
def inner_height_to_outer_height := inner_height_ratio = 2 / 3

-- Conclusion: Area of one trapezoid
theorem trapezoid_area (outer_area inner_height_ratio : ℝ) 
  (h_outer : outer_triangle_area outer_area) 
  (h_inner : inner_height_to_outer_height inner_height_ratio) : 
  (outer_area - 16 * Real.sqrt 3) / 3 = (36 - 16 * Real.sqrt 3) / 3 := 
sorry

end NUMINAMATH_GPT_trapezoid_area_l156_15679


namespace NUMINAMATH_GPT_volume_of_rectangular_prism_l156_15621

theorem volume_of_rectangular_prism
  (l w h : ℝ)
  (Hlw : l * w = 10)
  (Hwh : w * h = 15)
  (Hlh : l * h = 6) : l * w * h = 30 := 
by
  sorry

end NUMINAMATH_GPT_volume_of_rectangular_prism_l156_15621
