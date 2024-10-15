import Mathlib

namespace NUMINAMATH_GPT_woman_age_multiple_l1040_104040

theorem woman_age_multiple (S : ℕ) (W : ℕ) (k : ℕ) 
  (h1 : S = 27)
  (h2 : W + S = 84)
  (h3 : W = k * S + 3) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_woman_age_multiple_l1040_104040


namespace NUMINAMATH_GPT_farmer_field_m_value_l1040_104073

theorem farmer_field_m_value (m : ℝ) 
    (h_length : ∀ m, m > -4 → 2 * m + 9 > 0) 
    (h_breadth : ∀ m, m > -4 → m - 4 > 0)
    (h_area : (2 * m + 9) * (m - 4) = 88) : 
    m = 7.5 :=
by
  sorry

end NUMINAMATH_GPT_farmer_field_m_value_l1040_104073


namespace NUMINAMATH_GPT_divisor_inequality_l1040_104036

-- Definition of our main inequality theorem
theorem divisor_inequality (n : ℕ) (h1 : n > 0) (h2 : n % 8 = 4)
    (divisors : List ℕ) (h3 : divisors = (List.range (n + 1)).filter (λ x => n % x = 0)) 
    (i : ℕ) (h4 : i < divisors.length - 1) (h5 : i % 3 ≠ 0) : 
    divisors[i + 1] ≤ 2 * divisors[i] := sorry

end NUMINAMATH_GPT_divisor_inequality_l1040_104036


namespace NUMINAMATH_GPT_pilot_fish_final_speed_relative_to_ocean_l1040_104034

-- Define conditions
def keanu_speed : ℝ := 20 -- Keanu's speed in mph
def wind_speed : ℝ := 5 -- Wind speed in mph
def shark_speed (initial_speed: ℝ) : ℝ := 2 * initial_speed -- Shark doubles its speed

-- The pilot fish increases its speed by half the shark's increase
def pilot_fish_speed (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  initial_pilot_fish_speed + 0.5 * shark_initial_speed

-- Define the speed of the pilot fish relative to the ocean
def pilot_fish_speed_relative_to_ocean (initial_pilot_fish_speed shark_initial_speed : ℝ) : ℝ := 
  pilot_fish_speed initial_pilot_fish_speed shark_initial_speed - wind_speed

-- Initial assumptions
def initial_pilot_fish_speed : ℝ := keanu_speed -- Pilot fish initially swims at the same speed as Keanu
def initial_shark_speed : ℝ := keanu_speed -- Let us assume the shark initially swims at the same speed as Keanu for simplicity

-- Prove the final speed of the pilot fish relative to the ocean
theorem pilot_fish_final_speed_relative_to_ocean : 
  pilot_fish_speed_relative_to_ocean initial_pilot_fish_speed initial_shark_speed = 25 := 
by sorry

end NUMINAMATH_GPT_pilot_fish_final_speed_relative_to_ocean_l1040_104034


namespace NUMINAMATH_GPT_andy_last_problem_l1040_104017

theorem andy_last_problem (s t : ℕ) (start : s = 75) (total : t = 51) : (s + t - 1) = 125 :=
by
  sorry

end NUMINAMATH_GPT_andy_last_problem_l1040_104017


namespace NUMINAMATH_GPT_red_tint_percentage_new_mixture_l1040_104002

-- Definitions of the initial conditions
def original_volume : ℝ := 50
def red_tint_percentage : ℝ := 0.20
def added_red_tint : ℝ := 6

-- Definition for the proof
theorem red_tint_percentage_new_mixture : 
  let original_red_tint := red_tint_percentage * original_volume
  let new_red_tint := original_red_tint + added_red_tint
  let new_total_volume := original_volume + added_red_tint
  (new_red_tint / new_total_volume) * 100 = 28.57 :=
by
  sorry

end NUMINAMATH_GPT_red_tint_percentage_new_mixture_l1040_104002


namespace NUMINAMATH_GPT_length_second_train_is_125_l1040_104072

noncomputable def length_second_train (speed_faster speed_slower distance1 : ℕ) (time_minutes : ℝ) : ℝ :=
  let relative_speed_m_per_minute := (speed_faster - speed_slower) * 1000 / 60
  let total_distance_covered := relative_speed_m_per_minute * time_minutes
  total_distance_covered - distance1

theorem length_second_train_is_125 :
  length_second_train 50 40 125 1.5 = 125 :=
  by sorry

end NUMINAMATH_GPT_length_second_train_is_125_l1040_104072


namespace NUMINAMATH_GPT_no_simultaneous_inequalities_l1040_104067

theorem no_simultaneous_inequalities (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  ¬ ((a + b < c + d) ∧ ((a + b) * (c + d) < a * b + c * d) ∧ ((a + b) * c * d < a * b * (c + d))) :=
by
  sorry

end NUMINAMATH_GPT_no_simultaneous_inequalities_l1040_104067


namespace NUMINAMATH_GPT_area_of_triangle_l1040_104025

theorem area_of_triangle 
  (h : ∀ x y : ℝ, (x / 5 + y / 2 = 1) → ((x = 5 ∧ y = 0) ∨ (x = 0 ∧ y = 2))) : 
  ∃ t : ℝ, t = 1 / 2 * 2 * 5 := 
sorry

end NUMINAMATH_GPT_area_of_triangle_l1040_104025


namespace NUMINAMATH_GPT_average_of_scores_l1040_104088

theorem average_of_scores :
  let scores := [50, 60, 70, 80, 80]
  let total := 340
  let num_subjects := 5
  let average := total / num_subjects
  average = 68 :=
by
  sorry

end NUMINAMATH_GPT_average_of_scores_l1040_104088


namespace NUMINAMATH_GPT_gcd_of_78_and_36_l1040_104065

theorem gcd_of_78_and_36 :
  Nat.gcd 78 36 = 6 :=
by
  sorry

end NUMINAMATH_GPT_gcd_of_78_and_36_l1040_104065


namespace NUMINAMATH_GPT_average_of_six_numbers_l1040_104064

theorem average_of_six_numbers :
  (∀ a b : ℝ, (a + b) / 2 = 6.2) →
  (∀ c d : ℝ, (c + d) / 2 = 6.1) →
  (∀ e f : ℝ, (e + f) / 2 = 6.9) →
  ((a + b + c + d + e + f) / 6 = 6.4) :=
by
  intros h1 h2 h3
  -- Proof goes here, but will be skipped with sorry.
  sorry

end NUMINAMATH_GPT_average_of_six_numbers_l1040_104064


namespace NUMINAMATH_GPT_simone_finishes_task_at_1115_l1040_104056

noncomputable def simone_finish_time
  (start_time: Nat) -- Start time in minutes past midnight
  (task_1_duration: Nat) -- Duration of the first task in minutes
  (task_2_duration: Nat) -- Duration of the second task in minutes
  (break_duration: Nat) -- Duration of the break in minutes
  (task_3_duration: Nat) -- Duration of the third task in minutes
  (end_time: Nat) := -- End time to be proven
  start_time + task_1_duration + task_2_duration + break_duration + task_3_duration = end_time

theorem simone_finishes_task_at_1115 :
  simone_finish_time 480 45 45 15 90 675 := -- 480 minutes is 8:00 AM; 675 minutes is 11:15 AM
  by sorry

end NUMINAMATH_GPT_simone_finishes_task_at_1115_l1040_104056


namespace NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l1040_104042

theorem tenth_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 2 * d = 14)
  (h2 : a + 5 * d = 32) : 
  (a + 9 * d = 56) ∧ (d = 6) := 
by
  sorry

end NUMINAMATH_GPT_tenth_term_of_arithmetic_sequence_l1040_104042


namespace NUMINAMATH_GPT_triangle_equilateral_if_condition_l1040_104052

-- Define the given conditions
variables {A B C : ℝ} -- Angles
variables {a b c : ℝ} -- Opposite sides

-- Assume the condition that a/ cos(A) = b/ cos(B) = c/ cos(C)
def triangle_condition (A B C a b c : ℝ) : Prop :=
  a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C

-- The theorem to prove under these conditions
theorem triangle_equilateral_if_condition (A B C a b c : ℝ) 
  (h : triangle_condition A B C a b c) : 
  A = B ∧ B = C :=
sorry

end NUMINAMATH_GPT_triangle_equilateral_if_condition_l1040_104052


namespace NUMINAMATH_GPT_f_2023_l1040_104070

noncomputable def f : ℕ → ℤ := sorry

axiom f_defined_for_all : ∀ x : ℕ, f x ≠ 0 → (x ≥ 0)
axiom f_one : f 1 = 1
axiom f_functional_eq : ∀ a b : ℕ, f (a + b) = f a + f b - 3 * f (a * b)

theorem f_2023 : f 2023 = -(2^2022 - 1) := sorry

end NUMINAMATH_GPT_f_2023_l1040_104070


namespace NUMINAMATH_GPT_value_of_b_l1040_104055

-- Define the variables and conditions
variables (a b c : ℚ)
axiom h1 : a + b + c = 150
axiom h2 : a + 10 = b - 3
axiom h3 : b - 3 = 4 * c 

-- The statement we want to prove
theorem value_of_b : b = 655 / 9 := 
by 
  -- We start with assumptions h1, h2, and h3
  sorry

end NUMINAMATH_GPT_value_of_b_l1040_104055


namespace NUMINAMATH_GPT_train_passing_tree_time_l1040_104081

theorem train_passing_tree_time
  (train_length : ℝ) (train_speed_kmhr : ℝ) (conversion_factor : ℝ)
  (train_speed_ms : train_speed_ms = train_speed_kmhr * conversion_factor) :
  train_length = 500 → train_speed_kmhr = 72 → conversion_factor = 5 / 18 →
  500 / (72 * (5 / 18)) = 25 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_passing_tree_time_l1040_104081


namespace NUMINAMATH_GPT_zoo_individuals_remaining_l1040_104059

noncomputable def initial_students_class1 := 10
noncomputable def initial_students_class2 := 10
noncomputable def chaperones := 5
noncomputable def teachers := 2
noncomputable def students_left := 10
noncomputable def chaperones_left := 2

theorem zoo_individuals_remaining :
  let total_initial_individuals := initial_students_class1 + initial_students_class2 + chaperones + teachers
  let total_left := students_left + chaperones_left
  total_initial_individuals - total_left = 15 := by
  sorry

end NUMINAMATH_GPT_zoo_individuals_remaining_l1040_104059


namespace NUMINAMATH_GPT_x_y_z_sum_l1040_104045

theorem x_y_z_sum :
  ∃ (x y z : ℕ), (16 / 3)^x * (27 / 25)^y * (5 / 4)^z = 256 ∧ x + y + z = 6 :=
by
  -- Proof can be completed here
  sorry

end NUMINAMATH_GPT_x_y_z_sum_l1040_104045


namespace NUMINAMATH_GPT_dog_food_cans_l1040_104018

theorem dog_food_cans 
  (packages_cat_food : ℕ)
  (cans_per_package_cat_food : ℕ)
  (packages_dog_food : ℕ)
  (additional_cans_cat_food : ℕ)
  (total_cans_cat_food : ℕ)
  (total_cans_dog_food : ℕ)
  (num_cans_dog_food_package : ℕ) :
  packages_cat_food = 9 →
  cans_per_package_cat_food = 10 →
  packages_dog_food = 7 →
  additional_cans_cat_food = 55 →
  total_cans_cat_food = packages_cat_food * cans_per_package_cat_food →
  total_cans_dog_food = packages_dog_food * num_cans_dog_food_package →
  total_cans_cat_food = total_cans_dog_food + additional_cans_cat_food →
  num_cans_dog_food_package = 5 :=
by
  sorry

end NUMINAMATH_GPT_dog_food_cans_l1040_104018


namespace NUMINAMATH_GPT_percentage_increase_l1040_104071

theorem percentage_increase (N P : ℕ) (h1 : N = 40)
       (h2 : (N + (P / 100) * N) - (N - (30 / 100) * N) = 22) : P = 25 :=
by 
  have p1 := h1
  have p2 := h2
  sorry

end NUMINAMATH_GPT_percentage_increase_l1040_104071


namespace NUMINAMATH_GPT_probability_X_l1040_104057

theorem probability_X (P : ℕ → ℚ) (h1 : P 1 = 1/10) (h2 : P 2 = 2/10) (h3 : P 3 = 3/10) (h4 : P 4 = 4/10) :
  P 2 + P 3 = 1/2 :=
by
  sorry

end NUMINAMATH_GPT_probability_X_l1040_104057


namespace NUMINAMATH_GPT_inequality_one_inequality_two_l1040_104022

theorem inequality_one (a b : ℝ) : 
    a^2 + b^2 ≥ (a + b)^2 / 2 := 
by
    sorry

theorem inequality_two (a b : ℝ) : 
    a^2 + b^2 ≥ 2 * (a - b - 1) := 
by
    sorry

end NUMINAMATH_GPT_inequality_one_inequality_two_l1040_104022


namespace NUMINAMATH_GPT_total_toes_on_bus_l1040_104004

/-- Definition for the number of toes a Hoopit has -/
def toes_per_hoopit : ℕ := 4 * 3

/-- Definition for the number of toes a Neglart has -/
def toes_per_neglart : ℕ := 5 * 2

/-- Definition for the total number of Hoopits on the bus -/
def hoopit_students_on_bus : ℕ := 7

/-- Definition for the total number of Neglarts on the bus -/
def neglart_students_on_bus : ℕ := 8

/-- Proving that the total number of toes on the Popton school bus is 164 -/
theorem total_toes_on_bus : hoopit_students_on_bus * toes_per_hoopit + neglart_students_on_bus * toes_per_neglart = 164 := by
  sorry

end NUMINAMATH_GPT_total_toes_on_bus_l1040_104004


namespace NUMINAMATH_GPT_inclination_angle_of_line_l1040_104046

theorem inclination_angle_of_line
  (α : ℝ) (h1 : α > 0) (h2 : α < 180)
  (hslope : Real.tan α = - (Real.sqrt 3) / 3) :
  α = 150 :=
sorry

end NUMINAMATH_GPT_inclination_angle_of_line_l1040_104046


namespace NUMINAMATH_GPT_ratio_male_to_female_l1040_104093

theorem ratio_male_to_female (total_members female_members : ℕ) (h_total : total_members = 18) (h_female : female_members = 6) :
  (total_members - female_members) / Nat.gcd (total_members - female_members) female_members = 2 ∧
  female_members / Nat.gcd (total_members - female_members) female_members = 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_male_to_female_l1040_104093


namespace NUMINAMATH_GPT_min_value_2013_Quanzhou_simulation_l1040_104066

theorem min_value_2013_Quanzhou_simulation:
  ∃ (x y : ℝ), (x - y - 1 = 0) ∧ (x - 2) ^ 2 + (y - 2) ^ 2 = 1 / 2 :=
by
  use 2
  use 3
  sorry

end NUMINAMATH_GPT_min_value_2013_Quanzhou_simulation_l1040_104066


namespace NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1040_104077

def digits_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_23 : ∃ p > 0, Prime p ∧ digits_sum p = 23 ∧ ∀ q, Prime q ∧ digits_sum q = 23 → q ≥ p :=
by
  exists 887
  sorry -- The detailed proof steps will be provided here.

end NUMINAMATH_GPT_smallest_prime_with_digit_sum_23_l1040_104077


namespace NUMINAMATH_GPT_initial_amount_l1040_104095

theorem initial_amount (P R : ℝ) (h1 : 956 = P * (1 + (3 * R) / 100)) (h2 : 1052 = P * (1 + (3 * (R + 4)) / 100)) : P = 800 := 
by
  -- We would provide the proof steps here normally
  sorry

end NUMINAMATH_GPT_initial_amount_l1040_104095


namespace NUMINAMATH_GPT_investor_pieces_impossible_to_be_2002_l1040_104050

theorem investor_pieces_impossible_to_be_2002 : 
  ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := 
by
  sorry

end NUMINAMATH_GPT_investor_pieces_impossible_to_be_2002_l1040_104050


namespace NUMINAMATH_GPT_suraj_average_increase_l1040_104048

theorem suraj_average_increase
  (A : ℝ)
  (h1 : 9 * A + 200 = 10 * 128) :
  128 - A = 8 :=
by
  sorry

end NUMINAMATH_GPT_suraj_average_increase_l1040_104048


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_enlargement_l1040_104099

theorem right_triangle_hypotenuse_enlargement
  (a b c : ℝ)
  (h : c^2 = a^2 + b^2) :
  ((5 * a)^2 + (5 * b)^2 = (5 * c)^2) :=
by sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_enlargement_l1040_104099


namespace NUMINAMATH_GPT_parakeets_per_cage_l1040_104078

-- Define total number of cages
def num_cages: Nat := 6

-- Define number of parrots per cage
def parrots_per_cage: Nat := 2

-- Define total number of birds in the store
def total_birds: Nat := 54

-- Theorem statement: prove the number of parakeets per cage
theorem parakeets_per_cage : (total_birds - num_cages * parrots_per_cage) / num_cages = 7 :=
by
  sorry

end NUMINAMATH_GPT_parakeets_per_cage_l1040_104078


namespace NUMINAMATH_GPT_pythagorean_triple_l1040_104098

theorem pythagorean_triple {c a b : ℕ} (h1 : a = 24) (h2 : b = 7) (h3 : c = 25) : a^2 + b^2 = c^2 :=
by
  rw [h1, h2, h3]
  norm_num

end NUMINAMATH_GPT_pythagorean_triple_l1040_104098


namespace NUMINAMATH_GPT_simplify_expression_l1040_104028

noncomputable def m : ℝ := Real.tan (Real.pi / 3) - 1

theorem simplify_expression (h : m = Real.tan (Real.pi / 3) - 1) :
  (1 - 2 / (m + 1)) / ((m^2 - 2 * m + 1) / (m^2 - m)) = (3 - Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1040_104028


namespace NUMINAMATH_GPT_div_by_13_l1040_104020

theorem div_by_13 (a b c : ℤ) (h : (a + b + c) % 13 = 0) : 
  (a^2007 + b^2007 + c^2007 + 2 * 2007 * a * b * c) % 13 = 0 :=
by
  sorry

end NUMINAMATH_GPT_div_by_13_l1040_104020


namespace NUMINAMATH_GPT_minimum_value_l1040_104024

theorem minimum_value (a b : ℝ) (h1 : 2 * a + 3 * b = 5) (h2 : a > 0) (h3 : b > 0) : 
  (1 / a) + (1 / b) = 5 + 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l1040_104024


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1040_104079

theorem arithmetic_sequence_sum (a : ℕ → Int) (a1 a2017 : Int)
  (h1 : a 1 = a1) 
  (h2017 : a 2017 = a2017)
  (roots_eq : ∀ x, x^2 - 10 * x + 16 = 0 → (x = a1 ∨ x = a2017))
  (arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) :
  a 2 + a 1009 + a 2016 = 15 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1040_104079


namespace NUMINAMATH_GPT_Q_difference_l1040_104082

def Q (x n : ℕ) : ℕ :=
  (Finset.range (10^n)).sum (λ k => x / (k + 1))

theorem Q_difference (n : ℕ) : 
  Q (10^n) n - Q (10^n - 1) n = (n + 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_Q_difference_l1040_104082


namespace NUMINAMATH_GPT_estimate_students_spending_more_than_60_l1040_104075

-- Definition of the problem
def students_surveyed : ℕ := 50
def students_inclined_to_subscribe : ℕ := 8
def total_students : ℕ := 1000
def estimated_students : ℕ := 600

-- Define the proof task
theorem estimate_students_spending_more_than_60 :
  (students_inclined_to_subscribe : ℝ) / (students_surveyed : ℝ) * (total_students : ℝ) = estimated_students :=
by
  sorry

end NUMINAMATH_GPT_estimate_students_spending_more_than_60_l1040_104075


namespace NUMINAMATH_GPT_perpendicular_lines_a_eq_1_l1040_104003

-- Definitions for the given conditions
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + y + 3 = 0
def line2 (a : ℝ) (x y : ℝ) : Prop := x + (2 * a - 3) * y = 4

-- Condition that the lines are perpendicular
def perpendicular_lines (a : ℝ) : Prop := a + (2 * a - 3) = 0

-- Proof problem to be solved
theorem perpendicular_lines_a_eq_1 (a : ℝ) (h : perpendicular_lines a) : a = 1 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_eq_1_l1040_104003


namespace NUMINAMATH_GPT_train_speed_l1040_104043

/--
Given:
  Length of the train = 500 m
  Length of the bridge = 350 m
  The train takes 60 seconds to completely cross the bridge.

Prove:
  The speed of the train is exactly 14.1667 m/s
-/
theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 500) (h_bridge : length_bridge = 350) (h_time : time = 60) :
  (length_train + length_bridge) / time = 14.1667 :=
by
  rw [h_train, h_bridge, h_time]
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_l1040_104043


namespace NUMINAMATH_GPT_eggs_remainder_l1040_104058

def daniel_eggs := 53
def eliza_eggs := 68
def fiona_eggs := 26
def george_eggs := 47
def total_eggs := daniel_eggs + eliza_eggs + fiona_eggs + george_eggs

theorem eggs_remainder :
  total_eggs % 15 = 14 :=
by
  sorry

end NUMINAMATH_GPT_eggs_remainder_l1040_104058


namespace NUMINAMATH_GPT_cost_price_of_cloth_l1040_104087

theorem cost_price_of_cloth:
  ∀ (meters_sold profit_per_meter : ℕ) (selling_price : ℕ),
  meters_sold = 45 →
  profit_per_meter = 12 →
  selling_price = 4500 →
  (selling_price - (profit_per_meter * meters_sold)) / meters_sold = 88 :=
by
  intros meters_sold profit_per_meter selling_price h1 h2 h3
  sorry

end NUMINAMATH_GPT_cost_price_of_cloth_l1040_104087


namespace NUMINAMATH_GPT_students_playing_both_l1040_104094

theorem students_playing_both
    (total_students baseball_team hockey_team : ℕ)
    (h1 : total_students = 36)
    (h2 : baseball_team = 25)
    (h3 : hockey_team = 19)
    (h4 : total_students = baseball_team + hockey_team - students_both) :
    students_both = 8 := by
  sorry

end NUMINAMATH_GPT_students_playing_both_l1040_104094


namespace NUMINAMATH_GPT_pet_store_cages_l1040_104029

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h_initial : initial_puppies = 13) (h_sold : sold_puppies = 7) (h_per_cage : puppies_per_cage = 2) : (initial_puppies - sold_puppies) / puppies_per_cage = 3 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_cages_l1040_104029


namespace NUMINAMATH_GPT_geometric_series_sum_l1040_104096

theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  (a + a * r + a * r^2 + a * r^3 + a * r^4) = 341 / 1024 :=
by
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1040_104096


namespace NUMINAMATH_GPT_range_of_a_l1040_104011

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then (1/2 : ℝ) * x - 1 else 1 / x

theorem range_of_a (a : ℝ) : f a > a ↔ a < -1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1040_104011


namespace NUMINAMATH_GPT_square_presses_exceed_1000_l1040_104049

theorem square_presses_exceed_1000:
  ∃ n : ℕ, (n = 3) ∧ (3 ^ (2^n) > 1000) :=
by
  sorry

end NUMINAMATH_GPT_square_presses_exceed_1000_l1040_104049


namespace NUMINAMATH_GPT_coordinates_of_C_prime_l1040_104063

-- Define the given vertices of the triangle
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (3, 2)

-- Define the similarity ratio
def similarity_ratio : ℝ := 2

-- Define the function for the similarity transformation
def similarity_transform (center : ℝ × ℝ) (ratio : ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := point
  (ratio * x, ratio * y)

-- Prove the coordinates of C'
theorem coordinates_of_C_prime :
  similarity_transform (0, 0) similarity_ratio C = (6, 4) ∨ 
  similarity_transform (0, 0) similarity_ratio C = (-6, -4) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_C_prime_l1040_104063


namespace NUMINAMATH_GPT_cos_2beta_proof_l1040_104032

theorem cos_2beta_proof (α β : ℝ)
  (h1 : Real.sin (α - β) = 3 / 5)
  (h2 : Real.sin (α + β) = -3 / 5)
  (h3 : α - β ∈ Set.Ioo (π / 2) π)
  (h4 : α + β ∈ Set.Ioo (3 * π / 2) (2 * π)) :
  Real.cos (2 * β) = -7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_cos_2beta_proof_l1040_104032


namespace NUMINAMATH_GPT_janet_extra_flowers_l1040_104084

-- Define the number of flowers Janet picked for each type
def tulips : ℕ := 5
def roses : ℕ := 10
def daisies : ℕ := 8
def lilies : ℕ := 4

-- Define the number of flowers Janet used
def used : ℕ := 19

-- Calculate the total number of flowers Janet picked
def total_picked : ℕ := tulips + roses + daisies + lilies

-- Calculate the number of extra flowers
def extra_flowers : ℕ := total_picked - used

-- The theorem to be proven
theorem janet_extra_flowers : extra_flowers = 8 :=
by
  -- You would provide the proof here, but it's not required as per instructions
  sorry

end NUMINAMATH_GPT_janet_extra_flowers_l1040_104084


namespace NUMINAMATH_GPT_sqrt_calc1_sqrt_calc2_l1040_104038

-- Problem 1 proof statement
theorem sqrt_calc1 : ( (Real.sqrt 2 + Real.sqrt 3) ^ 2 - (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) = 4 + 2 * Real.sqrt 6 ) :=
  sorry

-- Problem 2 proof statement
theorem sqrt_calc2 : ( (2 - Real.sqrt 3) ^ 2023 * (2 + Real.sqrt 3) ^ 2023 - 2 * abs (-Real.sqrt 3 / 2) - (-Real.sqrt 2) ^ 0 = -Real.sqrt 3 ) :=
  sorry

end NUMINAMATH_GPT_sqrt_calc1_sqrt_calc2_l1040_104038


namespace NUMINAMATH_GPT_female_democrats_l1040_104001

/-
There are 810 male and female participants in a meeting.
Half of the female participants and one-quarter of the male participants are Democrats.
One-third of all the participants are Democrats.
Prove that the number of female Democrats is 135.
-/

theorem female_democrats (F M : ℕ) (h : F + M = 810)
  (female_democrats : F / 2 = F / 2)
  (male_democrats : M / 4 = M / 4)
  (total_democrats : (F / 2 + M / 4) = 810 / 3) : 
  F / 2 = 135 := by
  sorry

end NUMINAMATH_GPT_female_democrats_l1040_104001


namespace NUMINAMATH_GPT_no_injective_function_exists_l1040_104026

theorem no_injective_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∀ x, f (x^2) - (f x)^2 ≥ 1/4) ∧ (∀ x y, f x = f y → x = y) := 
sorry

end NUMINAMATH_GPT_no_injective_function_exists_l1040_104026


namespace NUMINAMATH_GPT_charlie_contribution_l1040_104097

theorem charlie_contribution (a b c : ℝ) (h₁ : a + b + c = 72) (h₂ : a = 1/4 * (b + c)) (h₃ : b = 1/5 * (a + c)) :
  c = 49 :=
by sorry

end NUMINAMATH_GPT_charlie_contribution_l1040_104097


namespace NUMINAMATH_GPT_line_through_two_points_l1040_104044

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) = (1, 3) ∨ (x, y) = (3, 7) → y = m * x + b) ∧ (m + b = 3) := by
{ sorry }

end NUMINAMATH_GPT_line_through_two_points_l1040_104044


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1040_104054

def A : Set ℝ := {x | x ≤ 1}
def B : Set ℝ := {x | -Real.sqrt 3 < x ∧ x < Real.sqrt 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x | -Real.sqrt 3 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1040_104054


namespace NUMINAMATH_GPT_find_fourth_intersection_point_l1040_104030

theorem find_fourth_intersection_point 
  (a b r: ℝ) 
  (h4 : ∃ a b r, ∀ x y, (x - a)^2 + (y - b)^2 = r^2 → (x, y) = (4, 1) ∨ (x, y) = (-2, -2) ∨ (x, y) = (8, 1/2) ∨ (x, y) = (-1/4, -16)):
  ∃ (x y : ℝ), (x - a)^2 + (y - b)^2 = r^2 → x * y = 4 → (x, y) = (-1/4, -16) := 
sorry

end NUMINAMATH_GPT_find_fourth_intersection_point_l1040_104030


namespace NUMINAMATH_GPT_ax_plus_by_equals_d_set_of_solutions_l1040_104010

theorem ax_plus_by_equals_d (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  ∀ (x y : ℤ), (a * x + b * y = d) ↔ ∃ k : ℤ, x = u + k * b ∧ y = v - k * a :=
by
  sorry

theorem set_of_solutions (a b d : ℤ) (u v : ℤ) (h_d : d = a.gcd b) (h_uv : a * u + b * v = d) :
  {p : ℤ × ℤ | a * p.1 + b * p.2 = d} = {p : ℤ × ℤ | ∃ k : ℤ, p = (u + k * b, v - k * a)} :=
by
  sorry

end NUMINAMATH_GPT_ax_plus_by_equals_d_set_of_solutions_l1040_104010


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_pi_l1040_104005

theorem sufficient_but_not_necessary_pi (x : ℝ) : 
  (x = Real.pi → Real.sin x = 0) ∧ (Real.sin x = 0 → ∃ k : ℤ, x = k * Real.pi) → ¬(Real.sin x = 0 → x = Real.pi) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_pi_l1040_104005


namespace NUMINAMATH_GPT_binomial_12_6_eq_1848_l1040_104035

theorem binomial_12_6_eq_1848 : (Nat.choose 12 6) = 1848 :=
  sorry

end NUMINAMATH_GPT_binomial_12_6_eq_1848_l1040_104035


namespace NUMINAMATH_GPT_largest_lcm_value_l1040_104060

open Nat

theorem largest_lcm_value :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end NUMINAMATH_GPT_largest_lcm_value_l1040_104060


namespace NUMINAMATH_GPT_find_x_from_triangle_area_l1040_104051

theorem find_x_from_triangle_area :
  ∀ (x : ℝ), x > 0 ∧ (1 / 2) * x * 3 * x = 96 → x = 8 :=
by
  intros x hx
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_find_x_from_triangle_area_l1040_104051


namespace NUMINAMATH_GPT_inserted_number_sq_property_l1040_104080

noncomputable def inserted_number (n : ℕ) : ℕ :=
  (5 * 10^n - 1) * 10^(n+1) + 1

theorem inserted_number_sq_property (n : ℕ) : (inserted_number n)^2 = (10^(n+1) - 1)^2 :=
by sorry

end NUMINAMATH_GPT_inserted_number_sq_property_l1040_104080


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1040_104014

theorem sufficient_but_not_necessary {a b : ℝ} (h1 : a > 2) (h2 : b > 2) : 
  a + b > 4 ∧ a * b > 4 := 
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1040_104014


namespace NUMINAMATH_GPT_smallest_positive_period_max_min_values_l1040_104090

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem smallest_positive_period (x : ℝ) :
  ∃ T, T > 0 ∧ ∀ x, f (x + T) = f x ∧ ∀ T', T' > 0 ∧ ∀ x, f (x + T') = f x → T ≤ T' :=
  sorry

theorem max_min_values : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  ∀ x, 0 ≤ x ∧ x ≤ Real.pi / 2 →
  min ≤ f x ∧ f x ≤ max :=
  sorry

end NUMINAMATH_GPT_smallest_positive_period_max_min_values_l1040_104090


namespace NUMINAMATH_GPT_find_number_l1040_104006

-- Definitions of the fractions involved
def frac_2_15 : ℚ := 2 / 15
def frac_1_5 : ℚ := 1 / 5
def frac_1_2 : ℚ := 1 / 2

-- Condition that the number is greater than the sum of frac_2_15 and frac_1_5 by frac_1_2 
def number : ℚ := frac_2_15 + frac_1_5 + frac_1_2

-- Theorem statement matching the math proof problem
theorem find_number : number = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1040_104006


namespace NUMINAMATH_GPT_sin_18_cos_36_eq_quarter_l1040_104021

theorem sin_18_cos_36_eq_quarter : Real.sin (Real.pi / 10) * Real.cos (Real.pi / 5) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sin_18_cos_36_eq_quarter_l1040_104021


namespace NUMINAMATH_GPT_second_man_speed_l1040_104069

/-- A formal statement of the problem -/
theorem second_man_speed (v : ℝ) 
  (start_same_place : ∀ t : ℝ, t ≥ 0 → 2 * t = (10 - v) * 1) : 
  v = 8 :=
by
  sorry

end NUMINAMATH_GPT_second_man_speed_l1040_104069


namespace NUMINAMATH_GPT_odd_function_neg_value_l1040_104086

theorem odd_function_neg_value (f : ℝ → ℝ) (h_odd : ∀ x : ℝ, f (-x) = -f x) (h_value : f 1 = 1) : f (-1) = -1 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_neg_value_l1040_104086


namespace NUMINAMATH_GPT_tanks_fill_l1040_104076

theorem tanks_fill
  (c : ℕ) -- capacity of each tank
  (h1 : 300 < c) -- first tank is filled with 300 liters, thus c > 300
  (h2 : 450 < c) -- second tank is filled with 450 liters, thus c > 450
  (h3 : (45 : ℝ) / 100 = (450 : ℝ) / c) -- second tank is 45% filled, thus 0.45 * c = 450
  (h4 : 300 + 450 < 2 * c) -- the two tanks have the same capacity, thus they must have enough capacity to be filled more than 750 liters
  : c - 300 + (c - 450) = 1250 :=
sorry

end NUMINAMATH_GPT_tanks_fill_l1040_104076


namespace NUMINAMATH_GPT_simplify_fractions_l1040_104023

theorem simplify_fractions : 
  (150 / 225) + (90 / 135) = 4 / 3 := by 
  sorry

end NUMINAMATH_GPT_simplify_fractions_l1040_104023


namespace NUMINAMATH_GPT_no_real_roots_x2_plus_4_l1040_104074

theorem no_real_roots_x2_plus_4 : ¬ ∃ x : ℝ, x^2 + 4 = 0 := by
  sorry

end NUMINAMATH_GPT_no_real_roots_x2_plus_4_l1040_104074


namespace NUMINAMATH_GPT_evaluate_expression_l1040_104015

theorem evaluate_expression : 6 - 8 * (9 - 4^2) / 2 - 3 = 31 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1040_104015


namespace NUMINAMATH_GPT_max_a_no_lattice_points_l1040_104062

theorem max_a_no_lattice_points :
  ∀ (m : ℝ), (1 / 3) < m → m < (17 / 51) →
  ¬ ∃ (x : ℕ) (y : ℕ), 0 < x ∧ x ≤ 50 ∧ y = m * x + 3 := 
by
  sorry

end NUMINAMATH_GPT_max_a_no_lattice_points_l1040_104062


namespace NUMINAMATH_GPT_FatherCandyCount_l1040_104083

variables (a b c d e : ℕ)

-- Conditions
def BillyInitial := 6
def CalebInitial := 11
def AndyInitial := 9
def BillyReceived := 8
def CalebReceived := 11
def AndyHasMore := 4

-- Define number of candies Andy has now based on Caleb's candies
def AndyTotal (b c : ℕ) : ℕ := c + AndyHasMore

-- Define number of candies received by Andy
def AndyReceived (a b c d e : ℕ) : ℕ := (AndyTotal b c) - AndyInitial

-- Define total candies bought by father
def FatherBoughtCandies (d e f : ℕ) : ℕ := d + e + f

theorem FatherCandyCount : FatherBoughtCandies BillyReceived CalebReceived (AndyReceived BillyInitial CalebInitial AndyInitial BillyReceived CalebReceived)  = 36 :=
by
  sorry

end NUMINAMATH_GPT_FatherCandyCount_l1040_104083


namespace NUMINAMATH_GPT_increased_colored_area_l1040_104012

theorem increased_colored_area
  (P : ℝ) -- Perimeter of the original convex pentagon
  (s : ℝ) -- Distance from the points colored originally
  : 
  s * P + π * s^2 = 23.14 :=
by
  sorry

end NUMINAMATH_GPT_increased_colored_area_l1040_104012


namespace NUMINAMATH_GPT_weight_ratio_mars_moon_l1040_104027

theorem weight_ratio_mars_moon :
  (∀ iron carbon other_elements_moon other_elements_mars wt_moon wt_mars : ℕ, 
    wt_moon = 250 ∧ 
    iron = 50 ∧ 
    carbon = 20 ∧ 
    other_elements_moon + 50 + 20 = 100 ∧ 
    other_elements_moon * wt_moon / 100 = 75 ∧ 
    other_elements_mars = 150 ∧ 
    wt_mars = (other_elements_mars * wt_moon) / other_elements_moon
  → wt_mars / wt_moon = 2) := 
sorry

end NUMINAMATH_GPT_weight_ratio_mars_moon_l1040_104027


namespace NUMINAMATH_GPT_non_congruent_triangles_perimeter_18_l1040_104047

theorem non_congruent_triangles_perimeter_18 : ∃ N : ℕ, N = 11 ∧
  ∀ (a b c : ℕ), a + b + c = 18 → a ≤ b → b ≤ c →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  ∃! t : Nat × Nat × Nat, t = ⟨a, b, c⟩  :=
by
  sorry

end NUMINAMATH_GPT_non_congruent_triangles_perimeter_18_l1040_104047


namespace NUMINAMATH_GPT_minimum_value_l1040_104091

def minimum_value_problem (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : Prop :=
  ∃ c : ℝ, c = (1 / (a + 1) + 4 / (b + 1)) ∧ c = 9 / 4

theorem minimum_value (a b : ℝ) (h_pos : 0 < a ∧ 0 < b) (h_sum : a + b = 2) : 
  (1 / (a + 1) + 4 / (b + 1)) = 9 / 4 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_minimum_value_l1040_104091


namespace NUMINAMATH_GPT_Kaleb_candies_l1040_104089

theorem Kaleb_candies 
  (tickets_whack_a_mole : ℕ) 
  (tickets_skee_ball : ℕ) 
  (candy_cost : ℕ)
  (h1 : tickets_whack_a_mole = 8)
  (h2 : tickets_skee_ball = 7)
  (h3 : candy_cost = 5) : 
  (tickets_whack_a_mole + tickets_skee_ball) / candy_cost = 3 := 
by
  sorry

end NUMINAMATH_GPT_Kaleb_candies_l1040_104089


namespace NUMINAMATH_GPT_trig_identity_l1040_104085

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) : 
  (1 / (Real.cos α ^ 2 + Real.sin (2 * α))) = 10 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l1040_104085


namespace NUMINAMATH_GPT_number_value_proof_l1040_104068

theorem number_value_proof (x y : ℝ) (h1 : 0.5 * x = y + 20) (h2 : x - 2 * y = 40) : x = 40 := 
by
  sorry

end NUMINAMATH_GPT_number_value_proof_l1040_104068


namespace NUMINAMATH_GPT_correct_transformation_l1040_104000

theorem correct_transformation (x : ℝ) :
  3 + x = 7 ∧ ¬ (x = 7 + 3) ∧
  5 * x = -4 ∧ ¬ (x = -5 / 4) ∧
  (7 / 4) * x = 3 ∧ ¬ (x = 3 * (7 / 4)) ∧
  -((x - 2) / 4) = 1 ∧ (-(x - 2)) = 4 :=
by
  sorry

end NUMINAMATH_GPT_correct_transformation_l1040_104000


namespace NUMINAMATH_GPT_min_value_of_fraction_l1040_104016

theorem min_value_of_fraction 
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : (Real.sqrt 3) = Real.sqrt (3 ^ a * 3 ^ (2 * b))) : 
  ∃ (min : ℝ), min = (2 / a + 1 / b) ∧ min = 8 :=
by
  -- proof will be skipped using sorry
  sorry

end NUMINAMATH_GPT_min_value_of_fraction_l1040_104016


namespace NUMINAMATH_GPT_total_students_in_school_l1040_104007

-- Definitions and conditions
def number_of_blind_students (B : ℕ) : Prop := ∃ B, 3 * B = 180
def number_of_other_disabilities (O : ℕ) (B : ℕ) : Prop := O = 2 * B
def total_students (T : ℕ) (D : ℕ) (B : ℕ) (O : ℕ) : Prop := T = D + B + O

theorem total_students_in_school : 
  ∃ (T B O : ℕ), number_of_blind_students B ∧ 
                 number_of_other_disabilities O B ∧ 
                 total_students T 180 B O ∧ 
                 T = 360 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_school_l1040_104007


namespace NUMINAMATH_GPT_jake_sold_tuesday_correct_l1040_104092

def jake_initial_pieces : ℕ := 80
def jake_sold_monday : ℕ := 15
def jake_remaining_wednesday : ℕ := 7

def pieces_sold_tuesday (initial : ℕ) (sold_monday : ℕ) (remaining_wednesday : ℕ) : ℕ :=
  initial - sold_monday - remaining_wednesday

theorem jake_sold_tuesday_correct :
  pieces_sold_tuesday jake_initial_pieces jake_sold_monday jake_remaining_wednesday = 58 :=
by
  unfold pieces_sold_tuesday
  norm_num
  sorry

end NUMINAMATH_GPT_jake_sold_tuesday_correct_l1040_104092


namespace NUMINAMATH_GPT_total_students_in_class_l1040_104009

-- Definitions based on the conditions
def num_girls : ℕ := 140
def num_boys_absent : ℕ := 40
def num_boys_present := num_girls / 2
def num_boys := num_boys_present + num_boys_absent
def total_students := num_girls + num_boys

-- Theorem to be proved
theorem total_students_in_class : total_students = 250 :=
by
  sorry

end NUMINAMATH_GPT_total_students_in_class_l1040_104009


namespace NUMINAMATH_GPT_bananas_indeterminate_l1040_104033

namespace RubyBananaProblem

variables (number_of_candies : ℕ) (number_of_friends : ℕ) (candies_per_friend : ℕ)
           (number_of_bananas : Option ℕ)

-- Given conditions
def Ruby_has_36_candies := number_of_candies = 36
def Ruby_has_9_friends := number_of_friends = 9
def Each_friend_gets_4_candies := candies_per_friend = 4
def Can_distribute_candies := number_of_candies = number_of_friends * candies_per_friend

-- Mathematical statement
theorem bananas_indeterminate (h1 : Ruby_has_36_candies number_of_candies)
                              (h2 : Ruby_has_9_friends number_of_friends)
                              (h3 : Each_friend_gets_4_candies candies_per_friend)
                              (h4 : Can_distribute_candies number_of_candies number_of_friends candies_per_friend) :
  number_of_bananas = none :=
by
  sorry

end RubyBananaProblem

end NUMINAMATH_GPT_bananas_indeterminate_l1040_104033


namespace NUMINAMATH_GPT_find_a2016_l1040_104013

variable {a : ℕ → ℤ} {S : ℕ → ℤ}

-- Given conditions
def cond1 : S 1 = 6 := by sorry
def cond2 : S 2 = 4 := by sorry
def cond3 (n : ℕ) : S n > 0 := by sorry
def cond4 (n : ℕ) : S (2 * n - 1) ^ 2 = S (2 * n) * S (2 * n + 2) := by sorry
def cond5 (n : ℕ) : 2 * S (2 * n + 2) = S (2 * n - 1) + S (2 * n + 1) := by sorry

theorem find_a2016 : a 2016 = -1009 := by
  -- Use the provided conditions to prove the statement
  sorry

end NUMINAMATH_GPT_find_a2016_l1040_104013


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1040_104031

-- Define the predicate that defines the condition for the ordered pairs (m, n)
def satisfies_condition (m n : ℕ) : Prop :=
  6 % m = 0 ∧ 3 % n = 0 ∧ 6 / m + 3 / n = 1

-- Define the main theorem for the problem statement
theorem number_of_ordered_pairs : 
  (∃ count : ℕ, count = 6 ∧ 
  (∀ m n : ℕ, satisfies_condition m n → m > 0 ∧ n > 0)) :=
by {
 sorry -- Placeholder for the actual proof
}

end NUMINAMATH_GPT_number_of_ordered_pairs_l1040_104031


namespace NUMINAMATH_GPT_find_a_plus_b_l1040_104008

def f (x a b : ℝ) := x^3 + a * x^2 + b * x + a^2

def extremum_at_one (a b : ℝ) : Prop :=
  f 1 a b = 10 ∧ (3 * 1^2 + 2 * a * 1 + b = 0)

theorem find_a_plus_b (a b : ℝ) (h : extremum_at_one a b) : a + b = -7 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1040_104008


namespace NUMINAMATH_GPT_range_of_k_l1040_104039

theorem range_of_k (f : ℝ → ℝ) (h_odd : ∀ x, f (-x) = -f x) (h_decreasing : ∀ ⦃x y⦄, 0 ≤ x → x < y → f y < f x) 
  (h_inequality : ∀ x, f (k * x ^ 2 + 2) + f (k * x + k) ≤ 0) : 0 ≤ k :=
sorry

end NUMINAMATH_GPT_range_of_k_l1040_104039


namespace NUMINAMATH_GPT_license_plate_problem_l1040_104037

noncomputable def license_plate_ways : ℕ :=
  let letters := 26
  let digits := 10
  let both_same := letters * digits * 1 * 1
  let digits_adj_same := letters * digits * 1 * letters
  let letters_adj_same := letters * digits * digits * 1
  digits_adj_same + letters_adj_same - both_same

theorem license_plate_problem :
  9100 = license_plate_ways :=
by
  -- Skipping the detailed proof for now
  sorry

end NUMINAMATH_GPT_license_plate_problem_l1040_104037


namespace NUMINAMATH_GPT_tangent_line_equation_l1040_104053

noncomputable def y (x : ℝ) := (2 * x - 1) / (x + 2)
def point : ℝ × ℝ := (-1, -3)
def tangent_eq (x y : ℝ) : Prop := 5 * x - y + 2 = 0

theorem tangent_line_equation :
  tangent_eq (-1) (-3) := 
sorry

end NUMINAMATH_GPT_tangent_line_equation_l1040_104053


namespace NUMINAMATH_GPT_remaining_stickers_l1040_104041

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end NUMINAMATH_GPT_remaining_stickers_l1040_104041


namespace NUMINAMATH_GPT_largest_no_solution_l1040_104019

theorem largest_no_solution (a : ℕ) (h_odd : a % 2 = 1) (h_pos : a > 0) :
  ∃ n : ℕ, ∀ x y z : ℕ, x > 0 → y > 0 → z > 0 → a * x + (a + 1) * y + (a + 2) * z ≠ n :=
sorry

end NUMINAMATH_GPT_largest_no_solution_l1040_104019


namespace NUMINAMATH_GPT_a_c_sum_l1040_104061

theorem a_c_sum (a b c d : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : d = a * b * c) (h5 : 233 % d = 79) : a + c = 13 :=
sorry

end NUMINAMATH_GPT_a_c_sum_l1040_104061
