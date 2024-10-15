import Mathlib

namespace NUMINAMATH_GPT_total_students_correct_l1930_193083

-- Given conditions
def number_of_buses : ℕ := 95
def number_of_seats_per_bus : ℕ := 118

-- Definition for the total number of students
def total_number_of_students : ℕ := number_of_buses * number_of_seats_per_bus

-- Problem statement
theorem total_students_correct :
  total_number_of_students = 11210 :=
by
  -- Proof is omitted, hence we use sorry.
  sorry

end NUMINAMATH_GPT_total_students_correct_l1930_193083


namespace NUMINAMATH_GPT_probability_of_two_eights_l1930_193049

-- Define a function that calculates the factorial of a number
noncomputable def fact (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * fact (n - 1)

-- Definition of binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ :=
  fact n / (fact k * fact (n - k))

-- Probability of exactly two dice showing 8 out of eight 8-sided dice
noncomputable def prob_exactly_two_eights : ℚ :=
  binom 8 2 * ((1 / 8 : ℚ) ^ 2) * ((7 / 8 : ℚ) ^ 6)

-- Main theorem statement
theorem probability_of_two_eights :
  prob_exactly_two_eights = 0.196 := by
  sorry

end NUMINAMATH_GPT_probability_of_two_eights_l1930_193049


namespace NUMINAMATH_GPT_dice_probability_abs_diff_2_l1930_193080

theorem dice_probability_abs_diff_2 :
  let total_outcomes := 36
  let favorable_outcomes := 8
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_dice_probability_abs_diff_2_l1930_193080


namespace NUMINAMATH_GPT_exists_prime_and_cube_root_l1930_193044

theorem exists_prime_and_cube_root (n : ℕ) (hn : 0 < n) :
  ∃ (p m : ℕ), p.Prime ∧ p % 6 = 5 ∧ ¬p ∣ n ∧ n ≡ m^3 [MOD p] :=
sorry

end NUMINAMATH_GPT_exists_prime_and_cube_root_l1930_193044


namespace NUMINAMATH_GPT_average_people_per_boat_correct_l1930_193086

-- Define number of boats and number of people
def num_boats := 3.0
def num_people := 5.0

-- Definition for average people per boat
def avg_people_per_boat := num_people / num_boats

-- Theorem to prove the average number of people per boat is 1.67
theorem average_people_per_boat_correct : avg_people_per_boat = 1.67 := by
  sorry

end NUMINAMATH_GPT_average_people_per_boat_correct_l1930_193086


namespace NUMINAMATH_GPT_value_of_a_l1930_193030

theorem value_of_a (x a : ℝ) (h1 : 0 < x) (h2 : x < 1 / a) (h3 : ∀ x, x * (1 - a * x) ≤ 1 / 12) : a = 3 :=
sorry

end NUMINAMATH_GPT_value_of_a_l1930_193030


namespace NUMINAMATH_GPT_shaded_region_area_is_15_l1930_193094

noncomputable def area_of_shaded_region : ℝ :=
  let radius := 1
  let area_of_one_circle := Real.pi * (radius ^ 2)
  4 * area_of_one_circle + 3 * (4 - area_of_one_circle)

theorem shaded_region_area_is_15 : 
  abs (area_of_shaded_region - 15) < 1 :=
by
  exact sorry

end NUMINAMATH_GPT_shaded_region_area_is_15_l1930_193094


namespace NUMINAMATH_GPT_original_price_of_petrol_l1930_193099

theorem original_price_of_petrol (P : ℝ) :
  (∃ P, 
    ∀ (GA GB GC : ℝ),
    0.8 * P = 0.8 * P ∧
    GA = 200 / P ∧
    GB = 300 / P ∧
    GC = 400 / P ∧
    200 = (GA + 8) * 0.8 * P ∧
    300 = (GB + 15) * 0.8 * P ∧
    400 = (GC + 22) * 0.8 * P) → 
  P = 6.25 :=
by
  sorry

end NUMINAMATH_GPT_original_price_of_petrol_l1930_193099


namespace NUMINAMATH_GPT_find_largest_N_l1930_193037

noncomputable def largest_N : ℕ :=
  by
    -- This proof needs to demonstrate the solution based on constraints.
    -- Proof will be filled here.
    sorry

theorem find_largest_N :
  largest_N = 44 := 
  by
    -- Proof to establish the largest N will be completed here.
    sorry

end NUMINAMATH_GPT_find_largest_N_l1930_193037


namespace NUMINAMATH_GPT_volume_is_85_l1930_193093

/-!
# Proof Problem
Prove that the total volume of Carl's and Kate's cubes is 85, given the conditions,
Carl has 3 cubes each with a side length of 3, and Kate has 4 cubes each with a side length of 1.
-/

-- Definitions for the problem conditions:
def volume_of_cube (s : ℕ) : ℕ := s^3

def total_volume (n : ℕ) (s : ℕ) : ℕ := n * volume_of_cube s

-- Given conditions
def carls_cubes_volume : ℕ := total_volume 3 3
def kates_cubes_volume : ℕ := total_volume 4 1

-- The total volume of Carl's and Kate's cubes:
def total_combined_volume : ℕ := carls_cubes_volume + kates_cubes_volume

-- Prove the total volume is 85
theorem volume_is_85 : total_combined_volume = 85 :=
by sorry

end NUMINAMATH_GPT_volume_is_85_l1930_193093


namespace NUMINAMATH_GPT_unit_cost_of_cranberry_juice_l1930_193067

theorem unit_cost_of_cranberry_juice (total_cost : ℕ) (ounces : ℕ) (h1 : total_cost = 84) (h2 : ounces = 12) :
  total_cost / ounces = 7 :=
by
  sorry

end NUMINAMATH_GPT_unit_cost_of_cranberry_juice_l1930_193067


namespace NUMINAMATH_GPT_number_of_students_l1930_193079

theorem number_of_students (n : ℕ) (h1 : n < 50) (h2 : n % 8 = 5) (h3 : n % 6 = 4) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l1930_193079


namespace NUMINAMATH_GPT_find_pairs_l1930_193053

-- Definitions for the conditions in the problem
def is_positive (x : ℝ) : Prop := x > 0

def equations (x y : ℝ) : Prop :=
  (Real.log (x^2 + y^2) / Real.log 10 = 2) ∧ 
  (Real.log x / Real.log 2 - 4 = Real.log 3 / Real.log 2 - Real.log y / Real.log 2)

-- Lean 4 Statement
theorem find_pairs (x y : ℝ) : 
  is_positive x ∧ is_positive y ∧ equations x y → (x, y) = (8, 6) ∨ (x, y) = (6, 8) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1930_193053


namespace NUMINAMATH_GPT_angle_with_same_terminal_side_315_l1930_193011

def same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 360 + β

theorem angle_with_same_terminal_side_315:
  same_terminal_side (-45) 315 :=
by
  sorry

end NUMINAMATH_GPT_angle_with_same_terminal_side_315_l1930_193011


namespace NUMINAMATH_GPT_union_A_B_eq_A_union_B_l1930_193091

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | x > 3 / 2 }

theorem union_A_B_eq_A_union_B :
  (A ∪ B) = { x | -1 ≤ x } :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_eq_A_union_B_l1930_193091


namespace NUMINAMATH_GPT_max_perimeter_of_triangle_l1930_193098

theorem max_perimeter_of_triangle (A B C a b c p : ℝ) 
  (h_angle_A : A = 2 * Real.pi / 3)
  (h_a : a = 3)
  (h_perimeter : p = a + b + c) 
  (h_sine_law : b = 2 * Real.sqrt 3 * Real.sin B ∧ c = 2 * Real.sqrt 3 * Real.sin C) :
  p ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_perimeter_of_triangle_l1930_193098


namespace NUMINAMATH_GPT_ratio_a_b_l1930_193042

variables {x y a b : ℝ}

theorem ratio_a_b (h1 : 8 * x - 6 * y = a)
                  (h2 : 12 * y - 18 * x = b)
                  (hx : x ≠ 0)
                  (hy : y ≠ 0)
                  (hb : b ≠ 0) :
  a / b = -4 / 9 :=
sorry

end NUMINAMATH_GPT_ratio_a_b_l1930_193042


namespace NUMINAMATH_GPT_find_odd_natural_numbers_l1930_193088

-- Definition of a friendly number
def is_friendly (n : ℕ) : Prop :=
  ∀ i, (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 + 1 ∨ (n / 10^i) % 10 = (n / 10^(i + 1)) % 10 - 1

-- Given condition: n is divisible by 64m
def is_divisible_by_64m (n m : ℕ) : Prop :=
  64 * m ∣ n

-- Proof problem statement
theorem find_odd_natural_numbers (m : ℕ) (hm1 : m % 2 = 1) :
  (5 ∣ m → ¬ ∃ n, is_friendly n ∧ is_divisible_by_64m n m) ∧ 
  (¬ 5 ∣ m → ∃ n, is_friendly n ∧ is_divisible_by_64m n m) :=
by
  sorry

end NUMINAMATH_GPT_find_odd_natural_numbers_l1930_193088


namespace NUMINAMATH_GPT_walkway_area_correct_l1930_193054

-- Define the dimensions and conditions
def bed_width : ℝ := 4
def bed_height : ℝ := 3
def walkway_width : ℝ := 2
def num_rows : ℕ := 4
def num_columns : ℕ := 3
def num_beds : ℕ := num_rows * num_columns

-- Total dimensions of garden including walkways
def total_width : ℝ := (num_columns * bed_width) + ((num_columns + 1) * walkway_width)
def total_height : ℝ := (num_rows * bed_height) + ((num_rows + 1) * walkway_width)

-- Areas
def total_garden_area : ℝ := total_width * total_height
def total_bed_area : ℝ := (bed_width * bed_height) * num_beds

-- Correct answer we want to prove
def walkway_area : ℝ := total_garden_area - total_bed_area

theorem walkway_area_correct : walkway_area = 296 := by
  sorry

end NUMINAMATH_GPT_walkway_area_correct_l1930_193054


namespace NUMINAMATH_GPT_proof_problem_l1930_193022

def setA : Set ℝ := {x | -1 < x ∧ x ≤ 5}
def setB : Set ℝ := {x | -1 < x ∧ x < 3}

def complementB : Set ℝ := {x | x ≥ 3 ∨ x ≤ -1}
def intersection : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

theorem proof_problem :
  (setA ∩ complementB) = intersection := 
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1930_193022


namespace NUMINAMATH_GPT_factor_expression_l1930_193034

-- Problem Statement
theorem factor_expression (x y : ℝ) : 60 * x ^ 2 + 40 * y = 20 * (3 * x ^ 2 + 2 * y) :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_factor_expression_l1930_193034


namespace NUMINAMATH_GPT_solve_system_l1930_193043

theorem solve_system :
  (∃ x y : ℝ, 4 * x + y = 5 ∧ 2 * x - 3 * y = 13) ↔ (x = 2 ∧ y = -3) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1930_193043


namespace NUMINAMATH_GPT_coloring_two_corners_removed_l1930_193081

noncomputable def coloring_count (total_ways : Nat) (ways_without_corner_a : Nat) : Nat :=
  total_ways - 2 * (total_ways - ways_without_corner_a) / 2 + 
  (ways_without_corner_a - (total_ways - ways_without_corner_a) / 2)

theorem coloring_two_corners_removed : coloring_count 120 96 = 78 := by
  sorry

end NUMINAMATH_GPT_coloring_two_corners_removed_l1930_193081


namespace NUMINAMATH_GPT_intersection_A_B_l1930_193082

-- Definitions of sets A and B
def A : Set ℕ := {2, 3, 5, 7}
def B : Set ℕ := {1, 2, 3, 5, 8}

-- Prove that the intersection of sets A and B is {2, 3, 5}
theorem intersection_A_B :
  A ∩ B = {2, 3, 5} :=
sorry

end NUMINAMATH_GPT_intersection_A_B_l1930_193082


namespace NUMINAMATH_GPT_total_peaches_l1930_193040

-- Definitions of conditions
def initial_peaches : ℕ := 13
def picked_peaches : ℕ := 55

-- Proof problem statement
theorem total_peaches : initial_peaches + picked_peaches = 68 :=
by
  -- Including sorry to skip the actual proof
  sorry

end NUMINAMATH_GPT_total_peaches_l1930_193040


namespace NUMINAMATH_GPT_roots_of_quadratic_l1930_193068

theorem roots_of_quadratic (α β : ℝ) (h1 : α^2 - 4*α - 5 = 0) (h2 : β^2 - 4*β - 5 = 0) :
  3*α^4 + 10*β^3 = 2593 := 
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_l1930_193068


namespace NUMINAMATH_GPT_diagonal_of_rectangular_prism_l1930_193055

noncomputable def diagonal_length (l w h : ℝ) : ℝ :=
  Real.sqrt (l^2 + w^2 + h^2)

theorem diagonal_of_rectangular_prism :
  diagonal_length 15 25 20 = 25 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_of_rectangular_prism_l1930_193055


namespace NUMINAMATH_GPT_find_f_2024_l1930_193056

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_f_2024 (a b c : ℝ)
  (h1 : f 2021 a b c = 2021)
  (h2 : f 2022 a b c = 2022)
  (h3 : f 2023 a b c = 2023) :
  f 2024 a b c = 2030 := sorry

end NUMINAMATH_GPT_find_f_2024_l1930_193056


namespace NUMINAMATH_GPT_total_spent_on_date_l1930_193024

-- Constants representing costs
def ticket_cost : ℝ := 10.00
def combo_meal_cost : ℝ := 11.00
def candy_cost : ℝ := 2.50

-- Numbers of items to buy
def num_tickets : ℝ := 2
def num_candies : ℝ := 2

-- Total cost calculation
def total_cost : ℝ := (ticket_cost * num_tickets) + (candy_cost * num_candies) + combo_meal_cost

-- Prove that the total cost is $36.00
theorem total_spent_on_date : total_cost = 36.00 := by
  sorry

end NUMINAMATH_GPT_total_spent_on_date_l1930_193024


namespace NUMINAMATH_GPT_bella_age_l1930_193050

theorem bella_age (B : ℕ) 
  (h1 : (B + 9) + B + B / 2 = 27) 
  : B = 6 :=
by sorry

end NUMINAMATH_GPT_bella_age_l1930_193050


namespace NUMINAMATH_GPT_log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l1930_193084

variable (a : ℝ) (b : ℝ)

-- Conditions
axiom base_pos (h : a > 0) : a ≠ 1
axiom integer_exponents_only (h : ∃ n : ℤ, b = a^n) : True
axiom positive_indices_only (h : ∃ n : ℕ, b = a^n) : 0 < b ∧ b < 1 → False

-- Theorem: If we only knew integer exponents, the logarithm of any number b in base a is defined for powers of a.
theorem log_defined_for_powers_of_a_if_integer_exponents (h : ∃ n : ℤ, b = a^n) : True :=
by sorry

-- Theorem: If we only knew positive exponents, the logarithm of any number b in base a is undefined for all 0 < b < 1
theorem log_undefined_if_only_positive_indices : (∃ n : ℕ, b = a^n) → (0 < b ∧ b < 1 → False) :=
by sorry

end NUMINAMATH_GPT_log_defined_for_powers_of_a_if_integer_exponents_log_undefined_if_only_positive_indices_l1930_193084


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l1930_193013

theorem swimming_speed_in_still_water (v : ℝ) (current_speed : ℝ) (time : ℝ) (distance : ℝ) (effective_speed : current_speed = 10) (time_to_return : time = 6) (distance_to_return : distance = 12) (speed_eq : v - current_speed = distance / time) : v = 12 :=
by
  sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l1930_193013


namespace NUMINAMATH_GPT_problem_statement_l1930_193021

theorem problem_statement : 
  10 - 1.05 / (5.2 * 14.6 - (9.2 * 5.2 + 5.4 * 3.7 - 4.6 * 1.5)) = 9.93 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1930_193021


namespace NUMINAMATH_GPT_gumball_water_wednesday_l1930_193035

theorem gumball_water_wednesday :
  ∀ (total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water : ℕ),
  total_weekly_water = 60 →
  monday_thursday_saturday_water = 9 →
  tuesday_friday_sunday_water = 8 →
  total_weekly_water - (monday_thursday_saturday_water * 3 + tuesday_friday_sunday_water * 3) = 9 :=
by
  intros total_weekly_water monday_thursday_saturday_water tuesday_friday_sunday_water
  sorry

end NUMINAMATH_GPT_gumball_water_wednesday_l1930_193035


namespace NUMINAMATH_GPT_parabola_coefficient_c_l1930_193003

def parabola (b c x : ℝ) : ℝ := x^2 + b * x + c

theorem parabola_coefficient_c (b c : ℝ) (h1 : parabola b c 1 = -1) (h2 : parabola b c 3 = 9) : 
  c = -3 := 
by
  sorry

end NUMINAMATH_GPT_parabola_coefficient_c_l1930_193003


namespace NUMINAMATH_GPT_parabola_constant_term_l1930_193051

theorem parabola_constant_term 
  (b c : ℝ)
  (h1 : 2 = 2 * (1 : ℝ)^2 + b * (1 : ℝ) + c)
  (h2 : 2 = 2 * (3 : ℝ)^2 + b * (3 : ℝ) + c) : 
  c = 8 :=
by
  sorry

end NUMINAMATH_GPT_parabola_constant_term_l1930_193051


namespace NUMINAMATH_GPT_triangle_side_length_l1930_193010

-- Defining basic properties and known lengths of the similar triangles
def GH : ℝ := 8
def HI : ℝ := 16
def YZ : ℝ := 24
def XY : ℝ := 12

-- Defining the similarity condition for triangles GHI and XYZ
def triangles_similar : Prop := 
  -- The similarity of the triangles implies proportionality of the sides
  (XY / GH = YZ / HI)

-- The theorem statement to prove
theorem triangle_side_length (h_sim : triangles_similar) : XY = 12 :=
by
  -- assuming the similarity condition and known lengths
  sorry -- This will be the detailed proof

end NUMINAMATH_GPT_triangle_side_length_l1930_193010


namespace NUMINAMATH_GPT_polygon_sides_l1930_193062

theorem polygon_sides {R : ℝ} (hR : R > 0) : 
  (∃ n : ℕ, n > 2 ∧ (1/2) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) → 
  ∃ n : ℕ, n = 15 :=
by
  sorry

end NUMINAMATH_GPT_polygon_sides_l1930_193062


namespace NUMINAMATH_GPT_total_apples_eq_l1930_193038

-- Define the conditions for the problem
def baskets : ℕ := 37
def apples_per_basket : ℕ := 17

-- Define the theorem to prove the total number of apples
theorem total_apples_eq : baskets * apples_per_basket = 629 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_eq_l1930_193038


namespace NUMINAMATH_GPT_smallest_D_l1930_193071

theorem smallest_D {A B C D : ℕ} (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h2 : (A * 100 + B * 10 + C) * B = D * 1000 + C * 100 + B * 10 + D) : 
  D = 1 :=
sorry

end NUMINAMATH_GPT_smallest_D_l1930_193071


namespace NUMINAMATH_GPT_athul_downstream_distance_l1930_193025

-- Define the conditions
def upstream_distance : ℝ := 16
def upstream_time : ℝ := 4
def speed_of_stream : ℝ := 1
def downstream_time : ℝ := 4

-- Translate the conditions into properties and prove the downstream distance
theorem athul_downstream_distance (V : ℝ) 
  (h1 : upstream_distance = (V - speed_of_stream) * upstream_time) :
  (V + speed_of_stream) * downstream_time = 24 := 
by
  -- Given the conditions, the proof would be filled here
  sorry

end NUMINAMATH_GPT_athul_downstream_distance_l1930_193025


namespace NUMINAMATH_GPT_red_tulips_l1930_193045

theorem red_tulips (white_tulips : ℕ) (bouquets : ℕ)
  (hw : white_tulips = 21)
  (hb : bouquets = 7)
  (div_prop : ∀ n, white_tulips % n = 0 ↔ bouquets % n = 0) : 
  ∃ red_tulips : ℕ, red_tulips = 7 :=
by
  sorry

end NUMINAMATH_GPT_red_tulips_l1930_193045


namespace NUMINAMATH_GPT_number_of_different_duty_schedules_l1930_193085

-- Define a structure for students
inductive Student
| A | B | C

-- Define days of the week excluding Sunday as all duties are from Monday to Saturday
inductive Day
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

-- Define the conditions in Lean
def condition_A_does_not_take_Monday (schedules : Day → Student) : Prop :=
  schedules Day.Monday ≠ Student.A

def condition_B_does_not_take_Saturday (schedules : Day → Student) : Prop :=
  schedules Day.Saturday ≠ Student.B

-- Define the function to count valid schedules
noncomputable def count_valid_schedules : ℕ :=
  sorry  -- This would be the computation considering combinatorics

-- Theorem statement to prove the correct answer
theorem number_of_different_duty_schedules 
    (schedules : Day → Student)
    (h1 : condition_A_does_not_take_Monday schedules)
    (h2 : condition_B_does_not_take_Saturday schedules)
    : count_valid_schedules = 42 :=
sorry

end NUMINAMATH_GPT_number_of_different_duty_schedules_l1930_193085


namespace NUMINAMATH_GPT_ylona_initial_bands_l1930_193097

variable (B J Y : ℕ)  -- Represents the initial number of rubber bands for Bailey, Justine, and Ylona respectively

-- Define the conditions
axiom h1 : J = B + 10
axiom h2 : J = Y - 2
axiom h3 : B - 4 = 8

-- Formulate the statement
theorem ylona_initial_bands : Y = 24 :=
by
  sorry

end NUMINAMATH_GPT_ylona_initial_bands_l1930_193097


namespace NUMINAMATH_GPT_car_A_faster_than_car_B_l1930_193029

noncomputable def car_A_speed := 
  let t_A1 := 50 / 60 -- time for the first 50 miles at 60 mph
  let t_A2 := 50 / 40 -- time for the next 50 miles at 40 mph
  let t_A := t_A1 + t_A2 -- total time for Car A
  100 / t_A -- average speed of Car A

noncomputable def car_B_speed := 
  let t_B := 1 + (1 / 4) + 1 -- total time for Car B, including a 15-minute stop
  100 / t_B -- average speed of Car B

theorem car_A_faster_than_car_B : car_A_speed > car_B_speed := 
by sorry

end NUMINAMATH_GPT_car_A_faster_than_car_B_l1930_193029


namespace NUMINAMATH_GPT_compare_answers_l1930_193087

def num : ℕ := 384
def correct_answer : ℕ := (5 * num) / 16
def students_answer : ℕ := (5 * num) / 6
def difference : ℕ := students_answer - correct_answer

theorem compare_answers : difference = 200 := 
by
  sorry

end NUMINAMATH_GPT_compare_answers_l1930_193087


namespace NUMINAMATH_GPT_bertha_gave_away_balls_l1930_193077

def balls_initial := 2
def balls_worn_out := 20 / 10
def balls_lost := 20 / 5
def balls_purchased := (20 / 4) * 3
def balls_after_20_games_without_giveaway := balls_initial - balls_worn_out - balls_lost + balls_purchased
def balls_after_20_games := 10

theorem bertha_gave_away_balls : balls_after_20_games_without_giveaway - balls_after_20_games = 1 := by
  sorry

end NUMINAMATH_GPT_bertha_gave_away_balls_l1930_193077


namespace NUMINAMATH_GPT_total_distance_walked_l1930_193036

theorem total_distance_walked (t1 t2 : ℝ) (r : ℝ) (total_distance : ℝ)
  (h1 : t1 = 15 / 60)  -- Convert 15 minutes to hours
  (h2 : t2 = 25 / 60)  -- Convert 25 minutes to hours
  (h3 : r = 3)         -- Average speed in miles per hour
  (h4 : total_distance = r * (t1 + t2))
  : total_distance = 2 :=
by
  -- here is where the proof would go
  sorry

end NUMINAMATH_GPT_total_distance_walked_l1930_193036


namespace NUMINAMATH_GPT_find_a_b_l1930_193019

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem find_a_b (a b c : ℝ) (h1 : (12 * a + b = 0)) (h2 : (4 * a + b = -3)) :
  a = 3 / 8 ∧ b = -9 / 2 := by
  sorry

end NUMINAMATH_GPT_find_a_b_l1930_193019


namespace NUMINAMATH_GPT_find_sum_fusion_number_l1930_193014

def sum_fusion_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * (2 * k + 1)

theorem find_sum_fusion_number (n : ℕ) :
  n = 2020 ↔ sum_fusion_number n :=
sorry

end NUMINAMATH_GPT_find_sum_fusion_number_l1930_193014


namespace NUMINAMATH_GPT_nested_function_evaluation_l1930_193074

def f (x : ℕ) : ℕ := x + 3
def g (x : ℕ) : ℕ := x / 2
def f_inv (x : ℕ) : ℕ := x - 3
def g_inv (x : ℕ) : ℕ := 2 * x

theorem nested_function_evaluation : 
  f (g_inv (f_inv (g_inv (f_inv (g (f 15)))))) = 21 := 
by 
  sorry

end NUMINAMATH_GPT_nested_function_evaluation_l1930_193074


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1930_193023

open Set

noncomputable def A : Set ℝ := { x | (x - 2) / (x + 5) < 0 }
noncomputable def B : Set ℝ := { x | x^2 - 2 * x - 3 ≥ 0 }

theorem intersection_of_A_and_B : A ∩ B = { x : ℝ | -5 < x ∧ x ≤ -1 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1930_193023


namespace NUMINAMATH_GPT_percentage_of_students_in_60_to_69_range_is_20_l1930_193039

theorem percentage_of_students_in_60_to_69_range_is_20 :
  let scores := [4, 8, 6, 5, 2]
  let total_students := scores.sum
  let students_in_60_to_69 := 5
  (students_in_60_to_69 * 100 / total_students) = 20 := by
  sorry

end NUMINAMATH_GPT_percentage_of_students_in_60_to_69_range_is_20_l1930_193039


namespace NUMINAMATH_GPT_angle_skew_lines_range_l1930_193005

theorem angle_skew_lines_range (θ : ℝ) (h1 : 0 < θ) (h2 : θ ≤ 90) : 0 < θ ∧ θ ≤ 90 :=
by sorry

end NUMINAMATH_GPT_angle_skew_lines_range_l1930_193005


namespace NUMINAMATH_GPT_edge_length_of_inscribed_cube_in_sphere_l1930_193059

noncomputable def edge_length_of_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) : ℝ :=
  let x := 2 * Real.sqrt 3
  x

theorem edge_length_of_inscribed_cube_in_sphere (surface_area_sphere : ℝ) (π_cond : surface_area_sphere = 36 * Real.pi) :
  edge_length_of_cube_in_sphere surface_area_sphere π_cond = 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_edge_length_of_inscribed_cube_in_sphere_l1930_193059


namespace NUMINAMATH_GPT_find_total_price_l1930_193048

noncomputable def total_price (p : ℝ) : Prop := 0.20 * p = 240

theorem find_total_price (p : ℝ) (h : total_price p) : p = 1200 :=
by sorry

end NUMINAMATH_GPT_find_total_price_l1930_193048


namespace NUMINAMATH_GPT_slope_of_parallel_line_l1930_193066

/-- A line is described by the equation 3x - 6y = 12. The slope of a line 
    parallel to this line is 1/2. -/
theorem slope_of_parallel_line (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1/2 := by
  sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l1930_193066


namespace NUMINAMATH_GPT_sugar_content_of_mixture_l1930_193069

theorem sugar_content_of_mixture 
  (volume_juice1 : ℝ) (conc_juice1 : ℝ)
  (volume_juice2 : ℝ) (conc_juice2 : ℝ) 
  (total_volume : ℝ) (total_sugar : ℝ) 
  (resulting_sugar_content : ℝ) :
  volume_juice1 = 2 →
  conc_juice1 = 0.1 →
  volume_juice2 = 3 →
  conc_juice2 = 0.15 →
  total_volume = volume_juice1 + volume_juice2 →
  total_sugar = (conc_juice1 * volume_juice1) + (conc_juice2 * volume_juice2) →
  resulting_sugar_content = (total_sugar / total_volume) * 100 →
  resulting_sugar_content = 13 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sugar_content_of_mixture_l1930_193069


namespace NUMINAMATH_GPT_milton_books_l1930_193070

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end NUMINAMATH_GPT_milton_books_l1930_193070


namespace NUMINAMATH_GPT_parallel_vectors_implies_x_value_l1930_193041

variable (x : ℝ)

def vec_a : ℝ × ℝ := (1, 2)
def vec_b (x : ℝ) : ℝ × ℝ := (x, 1)
def scalar_mul (c : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

theorem parallel_vectors_implies_x_value :
  (∃ k : ℝ, vec_add vec_a (scalar_mul 2 (vec_b x)) = scalar_mul k (vec_sub (scalar_mul 2 vec_a) (scalar_mul 2 (vec_b x)))) →
  x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_vectors_implies_x_value_l1930_193041


namespace NUMINAMATH_GPT_slices_all_three_toppings_l1930_193064

def slices_with_all_toppings (total_slices pepperoni_slices mushroom_slices olive_slices : ℕ) : ℕ := 
  (12 : ℕ)

theorem slices_all_three_toppings
  (total_slices : ℕ)
  (pepperoni_slices : ℕ)
  (mushroom_slices : ℕ)
  (olive_slices : ℕ)
  (h : total_slices = 24)
  (h1 : pepperoni_slices = 12)
  (h2 : mushroom_slices = 14)
  (h3 : olive_slices = 16)
  (hc : total_slices ≥ 0)
  (hc1 : pepperoni_slices ≥ 0)
  (hc2 : mushroom_slices ≥ 0)
  (hc3 : olive_slices ≥ 0) :
  slices_with_all_toppings total_slices pepperoni_slices mushroom_slices olive_slices = 2 :=
  sorry

end NUMINAMATH_GPT_slices_all_three_toppings_l1930_193064


namespace NUMINAMATH_GPT_max_value_pq_qr_rs_sp_l1930_193002

def max_pq_qr_rs_sp (p q r s : ℕ) : ℕ :=
  p * q + q * r + r * s + s * p

theorem max_value_pq_qr_rs_sp :
  ∀ (p q r s : ℕ), (p = 1 ∨ p = 5 ∨ p = 3 ∨ p = 6) → 
                    (q = 1 ∨ q = 5 ∨ q = 3 ∨ q = 6) →
                    (r = 1 ∨ r = 5 ∨ r = 3 ∨ r = 6) → 
                    (s = 1 ∨ s = 5 ∨ s = 3 ∨ s = 6) →
                    p ≠ q → p ≠ r → p ≠ s → q ≠ r → q ≠ s → r ≠ s → 
                    max_pq_qr_rs_sp p q r s ≤ 56 := by
  sorry

end NUMINAMATH_GPT_max_value_pq_qr_rs_sp_l1930_193002


namespace NUMINAMATH_GPT_bus_cost_proof_l1930_193032

-- Define conditions
def train_cost (bus_cost : ℚ) : ℚ := bus_cost + 6.85
def discount_rate : ℚ := 0.15
def service_fee : ℚ := 1.25
def combined_cost : ℚ := 10.50

-- Formula for the total cost after discount
def discounted_train_cost (bus_cost : ℚ) : ℚ := (train_cost bus_cost) * (1 - discount_rate)
def total_cost (bus_cost : ℚ) : ℚ := discounted_train_cost bus_cost + bus_cost + service_fee

-- Lean 4 statement asserting the cost of the bus ride before service fee
theorem bus_cost_proof : ∃ (B : ℚ), total_cost B = combined_cost ∧ B = 1.85 :=
sorry

end NUMINAMATH_GPT_bus_cost_proof_l1930_193032


namespace NUMINAMATH_GPT_aftershave_lotion_volume_l1930_193058

theorem aftershave_lotion_volume (V : ℝ) (h1 : 0.30 * V = 0.1875 * (V + 30)) : V = 50 := 
by 
-- sorry is added to indicate proof is omitted.
sorry

end NUMINAMATH_GPT_aftershave_lotion_volume_l1930_193058


namespace NUMINAMATH_GPT_cars_left_in_parking_lot_l1930_193031

-- Define constants representing the initial number of cars and cars that went out.
def initial_cars : ℕ := 24
def first_out : ℕ := 8
def second_out : ℕ := 6

-- State the theorem to prove the remaining cars in the parking lot.
theorem cars_left_in_parking_lot : 
  initial_cars - first_out - second_out = 10 := 
by {
  -- Here, 'sorry' is used to indicate the proof is omitted.
  sorry
}

end NUMINAMATH_GPT_cars_left_in_parking_lot_l1930_193031


namespace NUMINAMATH_GPT_average_speed_second_bus_l1930_193052

theorem average_speed_second_bus (x : ℝ) (h1 : x > 0) :
  (12 / x) - (12 / (1.2 * x)) = 3 / 60 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_second_bus_l1930_193052


namespace NUMINAMATH_GPT_one_more_square_possible_l1930_193008

def grid_size : ℕ := 29
def total_cells : ℕ := grid_size * grid_size
def number_of_squares_removed : ℕ := 99
def cells_per_square : ℕ := 4
def total_removed_cells : ℕ := number_of_squares_removed * cells_per_square
def remaining_cells : ℕ := total_cells - total_removed_cells

theorem one_more_square_possible :
  remaining_cells ≥ cells_per_square :=
sorry

end NUMINAMATH_GPT_one_more_square_possible_l1930_193008


namespace NUMINAMATH_GPT_integers_even_condition_l1930_193009

-- Definitions based on conditions
def is_even (n : ℤ) : Prop := n % 2 = 0

def exactly_one_even (a b c : ℤ) : Prop :=
(is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ is_even b ∧ ¬ is_even c) ∨
(¬ is_even a ∧ ¬ is_even b ∧ is_even c)

-- Proof statement
theorem integers_even_condition (a b c : ℤ) (h : ¬ exactly_one_even a b c) :
  (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨
  (is_even a ∧ is_even b) ∨
  (is_even a ∧ is_even c) ∨
  (is_even b ∧ is_even c) :=
sorry

end NUMINAMATH_GPT_integers_even_condition_l1930_193009


namespace NUMINAMATH_GPT_james_pre_injury_miles_600_l1930_193072

-- Define the conditions
def james_pre_injury_miles (x : ℝ) : Prop :=
  ∃ goal_increase : ℝ, ∃ days : ℝ, ∃ weekly_increase : ℝ,
  goal_increase = 1.2 * x ∧
  days = 280 ∧
  weekly_increase = 3 ∧
  (days / 7) * weekly_increase = (goal_increase - x)

-- Define the main theorem to be proved
theorem james_pre_injury_miles_600 : james_pre_injury_miles 600 :=
sorry

end NUMINAMATH_GPT_james_pre_injury_miles_600_l1930_193072


namespace NUMINAMATH_GPT_correct_option_l1930_193065

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem correct_option : M ∪ (U \ N) = U :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l1930_193065


namespace NUMINAMATH_GPT_comb_product_l1930_193075

theorem comb_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 :=
by
  sorry

end NUMINAMATH_GPT_comb_product_l1930_193075


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1930_193006

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem monotonic_increasing_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → ∀ (x1 x2 : ℝ), 0 ≤ x1 ∧ x1 ≤ 1 → 0 ≤ x2 ∧ x2 ≤ 1 → x1 ≤ x2 → sqrt (- x1 ^ 2 + 2 * x1) ≤ sqrt (- x2 ^ 2 + 2 * x2) :=
sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1930_193006


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1930_193018

theorem solution_set_of_inequality :
  { x : ℝ | |x + 1| + |x - 4| ≥ 7 } = { x : ℝ | x ≤ -2 ∨ x ≥ 5 } := sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1930_193018


namespace NUMINAMATH_GPT_gcd_60_75_l1930_193028

theorem gcd_60_75 : Nat.gcd 60 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_60_75_l1930_193028


namespace NUMINAMATH_GPT_ratio_of_area_of_shaded_square_l1930_193004

theorem ratio_of_area_of_shaded_square 
  (large_square : Type) 
  (smaller_squares : Finset large_square) 
  (area_large_square : ℝ) 
  (area_smaller_square : ℝ) 
  (h_division : smaller_squares.card = 25)
  (h_equal_area : ∀ s ∈ smaller_squares, area_smaller_square = (area_large_square / 25))
  (shaded_square : Finset large_square)
  (h_shaded_sub : shaded_square ⊆ smaller_squares)
  (h_shaded_card : shaded_square.card = 5) :
  (5 * area_smaller_square) / area_large_square = 1 / 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_area_of_shaded_square_l1930_193004


namespace NUMINAMATH_GPT_candy_totals_l1930_193012

-- Definitions of the conditions
def sandra_bags := 2
def sandra_pieces_per_bag := 6

def roger_bags1 := 11
def roger_bags2 := 3

def emily_bags1 := 4
def emily_bags2 := 7
def emily_bags3 := 5

-- Definitions of total pieces of candy
def sandra_total_candy := sandra_bags * sandra_pieces_per_bag
def roger_total_candy := roger_bags1 + roger_bags2
def emily_total_candy := emily_bags1 + emily_bags2 + emily_bags3

-- The proof statement
theorem candy_totals :
  sandra_total_candy = 12 ∧ roger_total_candy = 14 ∧ emily_total_candy = 16 :=
by
  -- Here we would provide the proof but we'll use sorry to skip it
  sorry

end NUMINAMATH_GPT_candy_totals_l1930_193012


namespace NUMINAMATH_GPT_chocolate_bars_per_box_l1930_193027

-- Definitions for the given conditions
def total_chocolate_bars : ℕ := 849
def total_boxes : ℕ := 170

-- The statement to prove
theorem chocolate_bars_per_box : total_chocolate_bars / total_boxes = 5 :=
by 
  -- Proof is omitted here
  sorry

end NUMINAMATH_GPT_chocolate_bars_per_box_l1930_193027


namespace NUMINAMATH_GPT_line_does_not_pass_through_fourth_quadrant_l1930_193089

theorem line_does_not_pass_through_fourth_quadrant
  (A B C : ℝ) (hAB : A * B < 0) (hBC : B * C < 0) :
  ¬ ∃ x y : ℝ, x > 0 ∧ y < 0 ∧ A * x + B * y + C = 0 :=
by
  sorry

end NUMINAMATH_GPT_line_does_not_pass_through_fourth_quadrant_l1930_193089


namespace NUMINAMATH_GPT_increased_contact_area_effect_l1930_193076

-- Define the conditions as assumptions
theorem increased_contact_area_effect (k : ℝ) (A₁ A₂ : ℝ) (dTdx : ℝ) (Q₁ Q₂ : ℝ) :
  (A₂ > A₁) →
  (Q₁ = -k * A₁ * dTdx) →
  (Q₂ = -k * A₂ * dTdx) →
  (Q₂ > Q₁) →
  ∃ increased_sensation : Prop, increased_sensation :=
by 
  exfalso
  sorry

end NUMINAMATH_GPT_increased_contact_area_effect_l1930_193076


namespace NUMINAMATH_GPT_sum_of_coordinates_inv_graph_l1930_193090

variable {f : ℝ → ℝ}
variable (hf : f 2 = 12)

theorem sum_of_coordinates_inv_graph :
  ∃ (x y : ℝ), y = f⁻¹ x / 3 ∧ x = 12 ∧ y = 2 / 3 ∧ x + y = 38 / 3 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_inv_graph_l1930_193090


namespace NUMINAMATH_GPT_total_fencing_costs_l1930_193033

theorem total_fencing_costs (c1 c2 c3 c4 l1 l2 l3 : ℕ) 
    (h_c1 : c1 = 79) (h_c2 : c2 = 92) (h_c3 : c3 = 85) (h_c4 : c4 = 96)
    (h_l1 : l1 = 5) (h_l2 : l2 = 7) (h_l3 : l3 = 9) :
    (c1 + c2 + c3 + c4) * l1 = 1760 ∧ 
    (c1 + c2 + c3 + c4) * l2 = 2464 ∧ 
    (c1 + c2 + c3 + c4) * l3 = 3168 := 
by {
    sorry -- Proof to be constructed
}

end NUMINAMATH_GPT_total_fencing_costs_l1930_193033


namespace NUMINAMATH_GPT_car_and_bus_speeds_l1930_193073

-- Definitions of given conditions
def car_speed : ℕ := 44
def bus_speed : ℕ := 52

-- Definition of total distance after 4 hours
def total_distance (car_speed bus_speed : ℕ) := 4 * car_speed + 4 * bus_speed

-- Definition of fact that cars started from the same point and traveled in opposite directions
def cars_from_same_point (car_speed bus_speed : ℕ) := car_speed + bus_speed

theorem car_and_bus_speeds :
  total_distance car_speed (car_speed + 8) = 384 :=
by
  -- Proof constructed based on the conditions given
  sorry

end NUMINAMATH_GPT_car_and_bus_speeds_l1930_193073


namespace NUMINAMATH_GPT_polyhedron_value_l1930_193001

theorem polyhedron_value (T H V E : ℕ) (h t : ℕ) 
  (F : ℕ) (h_eq : h = 10) (t_eq : t = 10)
  (F_eq : F = 20)
  (edges_eq : E = (3 * t + 6 * h) / 2)
  (vertices_eq : V = E - F + 2)
  (T_value : T = 2) (H_value : H = 2) :
  100 * H + 10 * T + V = 227 := by
  sorry

end NUMINAMATH_GPT_polyhedron_value_l1930_193001


namespace NUMINAMATH_GPT_smallest_difference_l1930_193061

theorem smallest_difference (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 362880) (h_order : a < b ∧ b < c) : c - a = 92 := 
sorry

end NUMINAMATH_GPT_smallest_difference_l1930_193061


namespace NUMINAMATH_GPT_parallel_lines_condition_l1930_193026

theorem parallel_lines_condition (a : ℝ) : 
    (∀ x y : ℝ, 2 * x + a * y + 2 ≠ (a - 1) * x + y - 2) ↔ a = 2 := 
sorry

end NUMINAMATH_GPT_parallel_lines_condition_l1930_193026


namespace NUMINAMATH_GPT_simplify_expression_l1930_193096

theorem simplify_expression :
  10 / (2 / 0.3) / (0.3 / 0.04) / (0.04 / 0.05) = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1930_193096


namespace NUMINAMATH_GPT_num_two_wheelers_wheels_eq_two_l1930_193092

variable (num_two_wheelers num_four_wheelers total_wheels : ℕ)

def total_wheels_eq : Prop :=
  2 * num_two_wheelers + 4 * num_four_wheelers = total_wheels

theorem num_two_wheelers_wheels_eq_two (h1 : num_four_wheelers = 13)
                                        (h2 : total_wheels = 54)
                                        (h_total_eq : total_wheels_eq num_two_wheelers num_four_wheelers total_wheels) :
  2 * num_two_wheelers = 2 :=
by
  unfold total_wheels_eq at h_total_eq
  sorry

end NUMINAMATH_GPT_num_two_wheelers_wheels_eq_two_l1930_193092


namespace NUMINAMATH_GPT_compare_abc_l1930_193000

noncomputable def a : ℝ := 4 ^ (Real.log 2 / Real.log 3)
noncomputable def b : ℝ := 4 ^ (Real.log 6 / (2 * Real.log 3))
noncomputable def c : ℝ := 2 ^ (Real.sqrt 5)

theorem compare_abc : c > b ∧ b > a := 
by
  sorry

end NUMINAMATH_GPT_compare_abc_l1930_193000


namespace NUMINAMATH_GPT_inequality_ln_l1930_193015

theorem inequality_ln (x : ℝ) (h₁ : x > -1) (h₂ : x ≠ 0) :
    (2 * abs x) / (2 + x) < abs (Real.log (1 + x)) ∧ abs (Real.log (1 + x)) < (abs x) / Real.sqrt (1 + x) :=
by
  sorry

end NUMINAMATH_GPT_inequality_ln_l1930_193015


namespace NUMINAMATH_GPT_smaller_prime_factor_l1930_193046

theorem smaller_prime_factor (a b : ℕ) (prime_a : Nat.Prime a) (prime_b : Nat.Prime b) (distinct : a ≠ b)
  (product : a * b = 316990099009901) :
  min a b = 4002001 :=
  sorry

end NUMINAMATH_GPT_smaller_prime_factor_l1930_193046


namespace NUMINAMATH_GPT_prob_A_prob_B_l1930_193078

variable (a b : ℝ) -- Declare variables a and b as real numbers
variable (h_ab : a + b = 1) -- Declare the condition a + b = 1
variable (h_pos_a : 0 < a) -- Declare a is a positive real number
variable (h_pos_b : 0 < b) -- Declare b is a positive real number

-- Prove that 1/a + 1/b ≥ 4 under the given conditions
theorem prob_A (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Prove that a^2 + b^2 ≥ 1/2 under the given conditions
theorem prob_B (h_ab : a + b = 1) (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  a^2 + b^2 ≥ 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_prob_A_prob_B_l1930_193078


namespace NUMINAMATH_GPT_increasing_interval_of_g_l1930_193095

noncomputable def f (x : ℝ) : ℝ :=
  (Real.cos (Real.pi / 3 - 2 * x)) -
  2 * (Real.sin (Real.pi / 4 + x) * Real.sin (Real.pi / 4 - x))

noncomputable def g (x : ℝ) : ℝ :=
  f (x + Real.pi / 12)

theorem increasing_interval_of_g :
  ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2),
  ∃ a b, a = -Real.pi / 12 ∧ b = Real.pi / 4 ∧
      (∀ x y, (a ≤ x ∧ x ≤ y ∧ y ≤ b) → g x ≤ g y) :=
sorry

end NUMINAMATH_GPT_increasing_interval_of_g_l1930_193095


namespace NUMINAMATH_GPT_drummer_difference_l1930_193057

def flute_players : Nat := 5
def trumpet_players : Nat := 3 * flute_players
def trombone_players : Nat := trumpet_players - 8
def clarinet_players : Nat := 2 * flute_players
def french_horn_players : Nat := trombone_players + 3
def total_seats_needed : Nat := 65
def total_seats_taken : Nat := flute_players + trumpet_players + trombone_players + clarinet_players + french_horn_players
def drummers : Nat := total_seats_needed - total_seats_taken

theorem drummer_difference : drummers - trombone_players = 11 := by
  sorry

end NUMINAMATH_GPT_drummer_difference_l1930_193057


namespace NUMINAMATH_GPT_problem_solution_l1930_193060

def diamond (x y k : ℝ) : ℝ := x^2 - k * y

theorem problem_solution (h : ℝ) (k : ℝ) (hk : k = 3) : 
  diamond h (diamond h h k) k = -2 * h^2 + 9 * h :=
by
  rw [hk, diamond, diamond]
  sorry

end NUMINAMATH_GPT_problem_solution_l1930_193060


namespace NUMINAMATH_GPT_death_rate_is_three_l1930_193007

-- Let birth_rate be the average birth rate in people per two seconds
def birth_rate : ℕ := 6
-- Let net_population_increase be the net population increase per day
def net_population_increase : ℕ := 129600
-- Let seconds_per_day be the total number of seconds in a day
def seconds_per_day : ℕ := 86400

noncomputable def death_rate_per_two_seconds : ℕ :=
  let net_increase_per_second := net_population_increase / seconds_per_day
  let birth_rate_per_second := birth_rate / 2
  2 * (birth_rate_per_second - net_increase_per_second)

theorem death_rate_is_three :
  death_rate_per_two_seconds = 3 := by
  sorry

end NUMINAMATH_GPT_death_rate_is_three_l1930_193007


namespace NUMINAMATH_GPT_overlapping_squares_proof_l1930_193016

noncomputable def overlapping_squares_area (s : ℝ) : ℝ :=
  let AB := s
  let MN := s
  let areaMN := s^2
  let intersection_area := areaMN / 4
  intersection_area

theorem overlapping_squares_proof (s : ℝ) :
  overlapping_squares_area s = s^2 / 4 := by
    -- proof would go here
    sorry

end NUMINAMATH_GPT_overlapping_squares_proof_l1930_193016


namespace NUMINAMATH_GPT_expected_games_is_correct_l1930_193047

def prob_A_wins : ℚ := 2 / 3
def prob_B_wins : ℚ := 1 / 3
def max_games : ℕ := 6

noncomputable def expected_games : ℚ :=
  2 * (prob_A_wins^2 + prob_B_wins^2) +
  4 * (prob_A_wins * prob_B_wins * (prob_A_wins^2 + prob_B_wins^2)) +
  6 * (prob_A_wins * prob_B_wins)^2

theorem expected_games_is_correct : expected_games = 266 / 81 := by
  sorry

end NUMINAMATH_GPT_expected_games_is_correct_l1930_193047


namespace NUMINAMATH_GPT_village_distance_l1930_193020

theorem village_distance
  (d : ℝ)
  (uphill_speed : ℝ) (downhill_speed : ℝ)
  (total_time : ℝ)
  (h1 : uphill_speed = 15)
  (h2 : downhill_speed = 30)
  (h3 : total_time = 4) :
  d = 40 :=
by
  sorry

end NUMINAMATH_GPT_village_distance_l1930_193020


namespace NUMINAMATH_GPT_function_no_real_zeros_l1930_193017

variable (a b c : ℝ)

-- Conditions: a, b, c form a geometric sequence and ac > 0
def geometric_sequence (a b c : ℝ) : Prop := b^2 = a * c
def positive_product (a c : ℝ) : Prop := a * c > 0

theorem function_no_real_zeros (h_geom : geometric_sequence a b c) (h_pos : positive_product a c) :
  ∀ x : ℝ, a * x^2 + b * x + c ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_function_no_real_zeros_l1930_193017


namespace NUMINAMATH_GPT_city_schools_count_l1930_193063

theorem city_schools_count (a b c : ℕ) (schools : ℕ) : 
  b = 40 → c = 51 → b < a → a < c → 
  (a > b ∧ a < c ∧ (a - 1) * 3 < (c - b + 1) * 3 + 1) → 
  schools = (c - 1) / 3 :=
by
  sorry

end NUMINAMATH_GPT_city_schools_count_l1930_193063
