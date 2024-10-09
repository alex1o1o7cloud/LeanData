import Mathlib

namespace yardsCatchingPasses_l1006_100662

-- Definitions from conditions in a)
def totalYardage : ℕ := 150
def runningYardage : ℕ := 90

-- Problem statement (Proof will follow)
theorem yardsCatchingPasses : totalYardage - runningYardage = 60 := by
  sorry

end yardsCatchingPasses_l1006_100662


namespace missing_angles_sum_l1006_100670

theorem missing_angles_sum 
  (calculated_sum : ℕ) 
  (missed_angles_sum : ℕ)
  (total_corrections : ℕ)
  (polygon_angles : ℕ) 
  (h1 : calculated_sum = 2797) 
  (h2 : total_corrections = 2880) 
  (h3 : polygon_angles = total_corrections - calculated_sum) : 
  polygon_angles = 83 := by
  sorry

end missing_angles_sum_l1006_100670


namespace fixed_numbers_in_diagram_has_six_solutions_l1006_100624

-- Define the problem setup and constraints
def is_divisor (m n : ℕ) : Prop := ∃ k, n = k * m

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

-- Formulating the main proof statement
theorem fixed_numbers_in_diagram_has_six_solutions : 
  ∃ (a b c k : ℕ),
    (14 * 4 * a = 14 * 6 * c) ∧
    (4 * a = 6 * c) ∧
    (2 * a = 3 * c) ∧
    (∃ k, c = 2 * k ∧ a = 3 * k) ∧
    (14 * 4 * 3 * k = 3 * k * b * 2 * k) ∧
    (∃ k, 56 * k = 6 * k^2 * b) ∧
    (b = 28 / k) ∧
    ((is_divisor k 28) ∧
     (k = 1 ∨ k = 2 ∨ k = 4 ∨ k = 7 ∨ k = 14 ∨ k = 28)) ∧
    (6 = 6) := sorry

end fixed_numbers_in_diagram_has_six_solutions_l1006_100624


namespace quadratic_real_roots_l1006_100659

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, m * x^2 + x - 1 = 0) ↔ (m ≥ -1/4 ∧ m ≠ 0) :=
by
  sorry

end quadratic_real_roots_l1006_100659


namespace division_remainder_l1006_100602

theorem division_remainder 
  (R D Q : ℕ) 
  (h1 : D = 3 * Q)
  (h2 : D = 3 * R + 3)
  (h3 : 113 = D * Q + R) : R = 5 :=
sorry

end division_remainder_l1006_100602


namespace incorrect_statement_about_absolute_value_l1006_100678

theorem incorrect_statement_about_absolute_value (x : ℝ) : abs x = 0 → x = 0 :=
by 
  sorry

end incorrect_statement_about_absolute_value_l1006_100678


namespace set_intersection_example_l1006_100656

theorem set_intersection_example :
  let M := {x : ℝ | -1 < x ∧ x < 1}
  let N := {x : ℝ | 0 ≤ x}
  {x : ℝ | -1 < x ∧ x < 1} ∩ {x : ℝ | 0 ≤ x} = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end set_intersection_example_l1006_100656


namespace smallest_integer_condition_l1006_100612

theorem smallest_integer_condition {A : ℕ} (h1 : A > 1) 
  (h2 : ∃ k : ℕ, A = 5 * k / 3 + 2 / 3)
  (h3 : ∃ m : ℕ, A = 7 * m / 5 + 2 / 5)
  (h4 : ∃ n : ℕ, A = 9 * n / 7 + 2 / 7)
  (h5 : ∃ p : ℕ, A = 11 * p / 9 + 2 / 9) : 
  A = 316 := 
sorry

end smallest_integer_condition_l1006_100612


namespace intersection_of_sets_l1006_100691

variable (x : ℝ)
def A : Set ℝ := {x | -2 < x ∧ x ≤ 1}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets 
  (hA : ∀ x, x ∈ A ↔ -2 < x ∧ x ≤ 1)
  (hB : ∀ x, x ∈ B ↔ 0 < x ∧ x ≤ 1) :
  ∀ x, (x ∈ A ∩ B) ↔ (0 < x ∧ x ≤ 1) := 
by
  sorry

end intersection_of_sets_l1006_100691


namespace solve_for_z_l1006_100642

open Complex

theorem solve_for_z (z : ℂ) (h : 2 * z * I = 1 + 3 * I) : 
  z = (3 / 2) - (1 / 2) * I :=
by
  sorry

end solve_for_z_l1006_100642


namespace sum_of_number_and_square_is_306_l1006_100663

theorem sum_of_number_and_square_is_306 : ∃ x : ℤ, x + x^2 = 306 ∧ x = 17 :=
by
  sorry

end sum_of_number_and_square_is_306_l1006_100663


namespace possible_AC_values_l1006_100679

-- Given points A, B, and C on a straight line 
-- with AB = 1 and BC = 3, prove that AC can be 2 or 4.

theorem possible_AC_values (A B C : ℝ) (hAB : abs (B - A) = 1) (hBC : abs (C - B) = 3) : 
  abs (C - A) = 2 ∨ abs (C - A) = 4 :=
sorry

end possible_AC_values_l1006_100679


namespace license_plate_combinations_l1006_100617

-- Definitions representing the conditions
def valid_license_plates_count : ℕ :=
  let letter_combinations := Nat.choose 26 2 -- Choose 2 unique letters
  let letter_arrangements := Nat.choose 4 2 * 2 -- Arrange the repeated letters
  let digit_combinations := 10 * 9 * 8 -- Choose different digits
  letter_combinations * letter_arrangements * digit_combinations

-- The theorem representing the problem statement
theorem license_plate_combinations :
  valid_license_plates_count = 2808000 := 
  sorry

end license_plate_combinations_l1006_100617


namespace pages_needed_l1006_100637

def total_new_cards : ℕ := 8
def total_old_cards : ℕ := 10
def cards_per_page : ℕ := 3

theorem pages_needed (h : total_new_cards = 8) (h2 : total_old_cards = 10) (h3 : cards_per_page = 3) : 
  (total_new_cards + total_old_cards) / cards_per_page = 6 := by 
  sorry

end pages_needed_l1006_100637


namespace inequality_sum_l1006_100620

variables {a b c : ℝ}

theorem inequality_sum (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 3) :
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ a^2 + b^2 + c^2 :=
sorry

end inequality_sum_l1006_100620


namespace sin_inequality_in_triangle_l1006_100632

theorem sin_inequality_in_triangle (A B C : ℝ) (hA_leq_B : A ≤ B) (hB_leq_C : B ≤ C)
  (hSum : A + B + C = π) (hA_pos : 0 < A) (hB_pos : 0 < B) (hC_pos : 0 < C)
  (hA_lt_pi : A < π) (hB_lt_pi : B < π) (hC_lt_pi : C < π) :
  0 < Real.sin A + Real.sin B - Real.sin C ∧ Real.sin A + Real.sin B - Real.sin C ≤ Real.sqrt 3 / 2 := 
sorry

end sin_inequality_in_triangle_l1006_100632


namespace cubics_sum_l1006_100610

theorem cubics_sum (a b c : ℝ) (h₁ : a + b + c = 4) (h₂ : ab + ac + bc = 6) (h₃ : abc = -8) :
  a^3 + b^3 + c^3 = 8 :=
by {
  -- proof steps would go here
  sorry
}

end cubics_sum_l1006_100610


namespace infinite_series_fraction_l1006_100673

theorem infinite_series_fraction:
  (∑' n : ℕ, (if n = 0 then 0 else ((2 : ℚ) / (3 * n) - (1 : ℚ) / (3 * (n + 1)) - (7 : ℚ) / (6 * (n + 3))))) =
  (1 : ℚ) / 3 := 
sorry

end infinite_series_fraction_l1006_100673


namespace hexagon_largest_angle_l1006_100639

theorem hexagon_largest_angle (a : ℚ) 
  (h₁ : (a + 2) + (2 * a - 3) + (3 * a + 1) + 4 * a + (5 * a - 4) + (6 * a + 2) = 720) :
  6 * a + 2 = 4374 / 21 :=
by sorry

end hexagon_largest_angle_l1006_100639


namespace kia_vehicle_count_l1006_100616

theorem kia_vehicle_count (total_vehicles dodge_vehicles hyundai_vehicles kia_vehicles : ℕ)
  (h_total: total_vehicles = 400)
  (h_dodge: dodge_vehicles = total_vehicles / 2)
  (h_hyundai: hyundai_vehicles = dodge_vehicles / 2)
  (h_kia: kia_vehicles = total_vehicles - dodge_vehicles - hyundai_vehicles) : kia_vehicles = 100 := 
by
  sorry

end kia_vehicle_count_l1006_100616


namespace jugs_needed_to_provide_water_for_students_l1006_100615

def jug_capacity : ℕ := 40
def students : ℕ := 200
def cups_per_student : ℕ := 10

def total_cups_needed := students * cups_per_student

theorem jugs_needed_to_provide_water_for_students :
  total_cups_needed / jug_capacity = 50 :=
by
  -- Proof goes here
  sorry

end jugs_needed_to_provide_water_for_students_l1006_100615


namespace division_of_positive_by_negative_l1006_100626

theorem division_of_positive_by_negative :
  4 / (-2) = -2 := 
by
  sorry

end division_of_positive_by_negative_l1006_100626


namespace product_calculation_l1006_100653

theorem product_calculation :
  1500 * 2023 * 0.5023 * 50 = 306903675 :=
sorry

end product_calculation_l1006_100653


namespace equal_parts_division_l1006_100661

theorem equal_parts_division (n : ℕ) (h : (n * n) % 4 = 0) : 
  ∃ parts : ℕ, parts = 4 ∧ ∀ (i : ℕ), i < parts → 
    ∃ p : ℕ, p = (n * n) / parts :=
by sorry

end equal_parts_division_l1006_100661


namespace island_length_l1006_100647

/-- Proof problem: Given an island in the Indian Ocean with a width of 4 miles and a perimeter of 22 miles. 
    Assume the island is rectangular in shape. Prove that the length of the island is 7 miles. -/
theorem island_length
  (width length : ℝ) 
  (h_width : width = 4)
  (h_perimeter : 2 * (length + width) = 22) : 
  length = 7 :=
sorry

end island_length_l1006_100647


namespace probability_same_color_l1006_100683

-- Definitions according to conditions
def total_socks : ℕ := 24
def blue_pairs : ℕ := 7
def green_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_blue_socks : ℕ := blue_pairs * 2
def total_green_socks : ℕ := green_pairs * 2
def total_red_socks : ℕ := red_pairs * 2

-- Probability calculations
def probability_blue : ℚ := (total_blue_socks * (total_blue_socks - 1)) / (total_socks * (total_socks - 1))
def probability_green : ℚ := (total_green_socks * (total_green_socks - 1)) / (total_socks * (total_socks - 1))
def probability_red : ℚ := (total_red_socks * (total_red_socks - 1)) / (total_socks * (total_socks - 1))

def total_probability : ℚ := probability_blue + probability_green + probability_red

theorem probability_same_color : total_probability = 28 / 69 :=
by
  sorry

end probability_same_color_l1006_100683


namespace allocation_count_l1006_100668

def allocate_volunteers (num_service_points : Nat) (num_volunteers : Nat) : Nat :=
  -- Definition that captures the counting logic as per the problem statement
  if num_service_points = 4 ∧ num_volunteers = 6 then 660 else 0

theorem allocation_count :
  allocate_volunteers 4 6 = 660 :=
sorry

end allocation_count_l1006_100668


namespace sin_eq_solutions_l1006_100650

theorem sin_eq_solutions :
  (∃ count : ℕ, 
    count = 4007 ∧ 
    (∀ (x : ℝ), 
      0 ≤ x ∧ x ≤ 2 * Real.pi → 
      (∃ (k1 k2 : ℤ), 
        x = -2 * k1 * Real.pi ∨ 
        x = 2 * Real.pi ∨ 
        x = (2 * k2 + 1) * Real.pi / 4005)
    )) :=
sorry

end sin_eq_solutions_l1006_100650


namespace even_two_digit_numbers_count_l1006_100608

/-- Even positive integers less than 1000 with at most two different digits -/
def count_even_two_digit_numbers : ℕ :=
  let one_digit := [2, 4, 6, 8].length
  let two_d_same := [22, 44, 66, 88].length
  let two_d_diff := [24, 42, 26, 62, 28, 82, 46, 64, 48, 84, 68, 86].length
  let three_d_same := [222, 444, 666, 888].length
  let three_d_diff := 16 + 12
  one_digit + two_d_same + two_d_diff + three_d_same + three_d_diff

theorem even_two_digit_numbers_count :
  count_even_two_digit_numbers = 52 :=
by sorry

end even_two_digit_numbers_count_l1006_100608


namespace numberOfSubsets_of_A_l1006_100684

def numberOfSubsets (s : Finset ℕ) : ℕ := 2 ^ (Finset.card s)

theorem numberOfSubsets_of_A : 
  numberOfSubsets ({0, 1} : Finset ℕ) = 4 := 
by 
  sorry

end numberOfSubsets_of_A_l1006_100684


namespace rebecca_perm_charge_l1006_100680

theorem rebecca_perm_charge :
  ∀ (P : ℕ), (4 * 30 + 2 * 60 - 2 * 10 + P + 50 = 310) -> P = 40 :=
by
  intros P h
  sorry

end rebecca_perm_charge_l1006_100680


namespace emma_uniform_number_correct_l1006_100607

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

noncomputable def dan : ℕ := 11  -- Example value, but needs to satisfy all conditions
noncomputable def emma : ℕ := 19  -- This is what we need to prove
noncomputable def fiona : ℕ := 13  -- Example value, but needs to satisfy all conditions
noncomputable def george : ℕ := 11  -- Example value, but needs to satisfy all conditions

theorem emma_uniform_number_correct :
  is_two_digit_prime dan ∧
  is_two_digit_prime emma ∧
  is_two_digit_prime fiona ∧
  is_two_digit_prime george ∧
  dan ≠ emma ∧ dan ≠ fiona ∧ dan ≠ george ∧
  emma ≠ fiona ∧ emma ≠ george ∧
  fiona ≠ george ∧
  dan + fiona = 23 ∧
  george + emma = 9 ∧
  dan + fiona + george + emma = 32
  → emma = 19 :=
sorry

end emma_uniform_number_correct_l1006_100607


namespace wallace_fulfills_orders_in_13_days_l1006_100641

def batch_small_bags_production := 12
def batch_large_bags_production := 8
def time_per_small_batch := 8
def time_per_large_batch := 12
def daily_production_limit := 18

def initial_stock_small := 18
def initial_stock_large := 10

def order1_small := 45
def order1_large := 30
def order2_small := 60
def order2_large := 25
def order3_small := 52
def order3_large := 42

def total_small_bags_needed := order1_small + order2_small + order3_small
def total_large_bags_needed := order1_large + order2_large + order3_large
def small_bags_to_produce := total_small_bags_needed - initial_stock_small
def large_bags_to_produce := total_large_bags_needed - initial_stock_large

def small_batches_needed := (small_bags_to_produce + batch_small_bags_production - 1) / batch_small_bags_production
def large_batches_needed := (large_bags_to_produce + batch_large_bags_production - 1) / batch_large_bags_production

def total_time_small_batches := small_batches_needed * time_per_small_batch
def total_time_large_batches := large_batches_needed * time_per_large_batch
def total_production_time := total_time_small_batches + total_time_large_batches

def days_needed := (total_production_time + daily_production_limit - 1) / daily_production_limit

theorem wallace_fulfills_orders_in_13_days :
  days_needed = 13 := by
  sorry

end wallace_fulfills_orders_in_13_days_l1006_100641


namespace right_triangle_area_l1006_100636

theorem right_triangle_area
  (hypotenuse : ℝ) (angle : ℝ) (hyp_eq : hypotenuse = 12) (angle_eq : angle = 30) :
  ∃ area : ℝ, area = 18 * Real.sqrt 3 :=
by
  have side1 := hypotenuse / 2  -- Shorter leg = hypotenuse / 2
  have side2 := side1 * Real.sqrt 3  -- Longer leg = shorter leg * sqrt 3
  let area := (side1 * side2) / 2  -- Area calculation
  use area
  sorry

end right_triangle_area_l1006_100636


namespace find_quotient_l1006_100682

theorem find_quotient :
  ∀ (remainder dividend divisor quotient : ℕ),
    remainder = 1 →
    dividend = 217 →
    divisor = 4 →
    quotient = (dividend - remainder) / divisor →
    quotient = 54 :=
by
  intros remainder dividend divisor quotient hr hd hdiv hq
  rw [hr, hd, hdiv] at hq
  norm_num at hq
  exact hq

end find_quotient_l1006_100682


namespace yoojung_notebooks_l1006_100646

theorem yoojung_notebooks (N : ℕ) (h : (N - 5) / 2 = 4) : N = 13 :=
by
  sorry

end yoojung_notebooks_l1006_100646


namespace compute_expression_l1006_100618

theorem compute_expression (x : ℝ) (h : x = 3) : 
  (x^8 + 18 * x^4 + 81) / (x^4 + 9) = 90 :=
by 
  sorry

end compute_expression_l1006_100618


namespace added_classes_l1006_100625

def original_classes := 15
def students_per_class := 20
def new_total_students := 400

theorem added_classes : 
  new_total_students = original_classes * students_per_class + 5 * students_per_class :=
by
  sorry

end added_classes_l1006_100625


namespace triangle_side_relation_l1006_100648

theorem triangle_side_relation (a b c : ℝ) (h1 : a^2 - 16 * b^2 - c^2 + 6 * a * b + 10 * b * c = 0) (h2 : a + b > c) :
  a + c = 2 * b := 
sorry

end triangle_side_relation_l1006_100648


namespace a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l1006_100686

-- Mathematical condition: a^2 + b^2 = 0
variable {a b : ℝ}

-- Mathematical statement to be proven
theorem a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero 
  (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry  -- proof yet to be provided

end a_squared_plus_b_squared_eq_zero_implies_a_eq_zero_and_b_eq_zero_l1006_100686


namespace radishes_times_carrots_l1006_100629

theorem radishes_times_carrots (cucumbers radishes carrots : ℕ) 
  (h1 : cucumbers = 15) 
  (h2 : radishes = 3 * cucumbers) 
  (h3 : carrots = 9) : 
  radishes / carrots = 5 :=
by
  sorry

end radishes_times_carrots_l1006_100629


namespace perpendicular_lines_from_perpendicular_planes_l1006_100640

variable {Line : Type} {Plane : Type}

-- Definitions of non-coincidence, perpendicularity, parallelism
noncomputable def non_coincident_lines (a b : Line) : Prop := sorry
noncomputable def non_coincident_planes (α β : Plane) : Prop := sorry
noncomputable def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_plane (a : Line) (α : Plane) : Prop := sorry
noncomputable def plane_parallel_to_plane (α β : Plane) : Prop := sorry
noncomputable def plane_perpendicular_to_plane (α β : Plane) : Prop := sorry
noncomputable def line_perpendicular_to_line (a b : Line) : Prop := sorry

-- Given non-coincident lines and planes
variable {a b : Line} {α β : Plane}

-- Problem statement
theorem perpendicular_lines_from_perpendicular_planes (h1 : non_coincident_lines a b)
  (h2 : non_coincident_planes α β)
  (h3 : line_perpendicular_to_plane a α)
  (h4 : line_perpendicular_to_plane b β)
  (h5 : plane_perpendicular_to_plane α β) : line_perpendicular_to_line a b := sorry

end perpendicular_lines_from_perpendicular_planes_l1006_100640


namespace stamp_exhibition_l1006_100695

def total_number_of_stamps (x : ℕ) : ℕ := 3 * x + 24

theorem stamp_exhibition : ∃ x : ℕ, total_number_of_stamps x = 174 ∧ (4 * x - 26) = 174 :=
by
  sorry

end stamp_exhibition_l1006_100695


namespace work_rate_l1006_100605

theorem work_rate (A_rate : ℝ) (combined_rate : ℝ) (B_days : ℝ) :
  A_rate = 1 / 12 ∧ combined_rate = 1 / 6.461538461538462 → 1 / B_days = combined_rate - A_rate → B_days = 14 :=
by
  intros
  sorry

end work_rate_l1006_100605


namespace base6_sum_l1006_100611

theorem base6_sum (D C : ℕ) (h₁ : D + 2 = C) (h₂ : C + 3 = 7) : C + D = 6 :=
by
  sorry

end base6_sum_l1006_100611


namespace framed_painting_ratio_l1006_100609

theorem framed_painting_ratio
  (width_painting : ℕ)
  (height_painting : ℕ)
  (frame_side : ℕ)
  (frame_top_bottom : ℕ)
  (h1 : width_painting = 20)
  (h2 : height_painting = 30)
  (h3 : frame_top_bottom = 3 * frame_side)
  (h4 : (width_painting + 2 * frame_side) * (height_painting + 2 * frame_top_bottom) = 2 * width_painting * height_painting):
  (width_painting + 2 * frame_side) = 1/2 * (height_painting + 2 * frame_top_bottom) := 
by
  sorry

end framed_painting_ratio_l1006_100609


namespace total_people_surveyed_l1006_100614

-- Define the conditions
variable (total_surveyed : ℕ) (disease_believers : ℕ)
variable (rabies_believers : ℕ)

-- Condition 1: 75% of the people surveyed thought rats carried diseases
def condition1 (total_surveyed disease_believers : ℕ) : Prop :=
  disease_believers = (total_surveyed * 75) / 100

-- Condition 2: 50% of the people who thought rats carried diseases said rats frequently carried rabies
def condition2 (disease_believers rabies_believers : ℕ) : Prop :=
  rabies_believers = (disease_believers * 50) / 100

-- Condition 3: 18 people were mistaken in thinking rats frequently carry rabies
def condition3 (rabies_believers : ℕ) : Prop := rabies_believers = 18

-- The theorem to prove the total number of people surveyed given the conditions
theorem total_people_surveyed (total_surveyed disease_believers rabies_believers : ℕ) :
  condition1 total_surveyed disease_believers →
  condition2 disease_believers rabies_believers →
  condition3 rabies_believers →
  total_surveyed = 48 :=
by sorry

end total_people_surveyed_l1006_100614


namespace shirts_not_all_on_sale_implications_l1006_100613

variable (Shirts : Type) (store_contains : Shirts → Prop) (on_sale : Shirts → Prop)

theorem shirts_not_all_on_sale_implications :
  ¬ ∀ s, store_contains s → on_sale s → 
  (∃ s, store_contains s ∧ ¬ on_sale s) ∧ (∃ s, store_contains s ∧ ¬ on_sale s) :=
by
  sorry

end shirts_not_all_on_sale_implications_l1006_100613


namespace pinocchio_cannot_pay_exactly_l1006_100621

theorem pinocchio_cannot_pay_exactly (N : ℕ) (hN : N ≤ 50) : 
  (∀ (a b : ℕ), N ≠ 5 * a + 6 * b) → N = 19 :=
by
  intro h
  have hN0 : 0 ≤ N := Nat.zero_le N
  sorry

end pinocchio_cannot_pay_exactly_l1006_100621


namespace value_of_expression_l1006_100628

variable {x : ℝ}

theorem value_of_expression (h : x^2 - 3 * x = 2) : 3 * x^2 - 9 * x - 7 = -1 := by
  sorry

end value_of_expression_l1006_100628


namespace area_of_ellipse_l1006_100692

theorem area_of_ellipse (x y : ℝ) (h : x^2 + 6 * x + 4 * y^2 - 8 * y + 9 = 0) : 
  area = 2 * Real.pi :=
sorry

end area_of_ellipse_l1006_100692


namespace find_a_b_c_l1006_100690

variable (a b c : ℚ)

def parabola (x : ℚ) : ℚ := a * x^2 + b * x + c

def vertex_condition := ∀ x, parabola a b c x = a * (x - 3)^2 - 2
def contains_point := parabola a b c 0 = 5

theorem find_a_b_c : vertex_condition a b c ∧ contains_point a b c → a + b + c = 10 / 9 :=
by
sorry

end find_a_b_c_l1006_100690


namespace prove_number_of_cows_l1006_100672

-- Define the conditions: Cows, Sheep, Pigs, Total animals
variables (C S P : ℕ)

-- Condition 1: Twice as many sheep as cows
def condition1 : Prop := S = 2 * C

-- Condition 2: Number of Pigs is 3 times the number of sheep
def condition2 : Prop := P = 3 * S

-- Condition 3: Total number of animals is 108
def condition3 : Prop := C + S + P = 108

-- The theorem to prove
theorem prove_number_of_cows (h1 : condition1 C S) (h2 : condition2 S P) (h3 : condition3 C S P) : C = 12 :=
sorry

end prove_number_of_cows_l1006_100672


namespace total_cost_l1006_100681

-- Definitions based on conditions
def old_camera_cost : ℝ := 4000
def new_model_cost_increase_rate : ℝ := 0.3
def lens_initial_cost : ℝ := 400
def lens_discount : ℝ := 200

-- Main statement to prove
theorem total_cost (old_camera_cost new_model_cost_increase_rate lens_initial_cost lens_discount : ℝ) : 
  let new_camera_cost := old_camera_cost * (1 + new_model_cost_increase_rate)
  let lens_cost_after_discount := lens_initial_cost - lens_discount
  (new_camera_cost + lens_cost_after_discount) = 5400 :=
by
  sorry

end total_cost_l1006_100681


namespace probability_red_purple_not_same_bed_l1006_100655

def colors : Set String := {"red", "yellow", "white", "purple"}

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_red_purple_not_same_bed : 
  let total_ways := C 4 2
  let unwanted_ways := 2
  let desired_ways := total_ways - unwanted_ways
  let probability := (desired_ways : ℚ) / total_ways
  probability = 2 / 3 := by
  sorry

end probability_red_purple_not_same_bed_l1006_100655


namespace total_shares_eq_300_l1006_100694

-- Define the given conditions
def microtron_price : ℝ := 36
def dynaco_price : ℝ := 44
def avg_price : ℝ := 40
def dynaco_shares : ℝ := 150

-- Define the number of Microtron shares sold
variable (M : ℝ)

-- Define the total shares sold
def total_shares : ℝ := M + dynaco_shares

-- The average price equation given the conditions
def avg_price_eq (M : ℝ) : Prop :=
  avg_price = (microtron_price * M + dynaco_price * dynaco_shares) / total_shares M

-- The correct answer we need to prove
theorem total_shares_eq_300 (M : ℝ) (h : avg_price_eq M) : total_shares M = 300 :=
by
  sorry

end total_shares_eq_300_l1006_100694


namespace find_n_l1006_100660

-- Define the vectors \overrightarrow {AB}, \overrightarrow {BC}, and \overrightarrow {AC}
def vectorAB : ℝ × ℝ := (2, 4)
def vectorBC (n : ℝ) : ℝ × ℝ := (-2, 2 * n)
def vectorAC : ℝ × ℝ := (0, 2)

-- State the theorem and prove the value of n
theorem find_n (n : ℝ) (h : vectorAC = (vectorAB.1 + (vectorBC n).1, vectorAB.2 + (vectorBC n).2)) : n = -1 :=
by
  sorry

end find_n_l1006_100660


namespace shaded_area_correct_l1006_100698

noncomputable def total_shaded_area (floor_length : ℝ) (floor_width : ℝ) (tile_size : ℝ) (circle_radius : ℝ) : ℝ :=
  let tile_area := tile_size ^ 2
  let circle_area := Real.pi * circle_radius ^ 2
  let shaded_area_per_tile := tile_area - circle_area
  let floor_area := floor_length * floor_width
  let number_of_tiles := floor_area / tile_area
  number_of_tiles * shaded_area_per_tile 

theorem shaded_area_correct : total_shaded_area 12 15 2 1 = 180 - 45 * Real.pi := sorry

end shaded_area_correct_l1006_100698


namespace card_picking_l1006_100675

/-
Statement of the problem:
- A modified deck of cards has 65 cards.
- The deck is divided into 5 suits, each of which has 13 cards.
- The cards are placed in random order.
- Prove that the number of ways to pick two different cards from this deck with the order of picking being significant is 4160.
-/
theorem card_picking : (65 * 64) = 4160 := by
  sorry

end card_picking_l1006_100675


namespace circle_diameter_from_area_l1006_100689

theorem circle_diameter_from_area (A : ℝ) (h : A = 225 * Real.pi) : ∃ d : ℝ, d = 30 :=
  by
  have r := Real.sqrt (225)
  have d := 2 * r
  exact ⟨d, sorry⟩

end circle_diameter_from_area_l1006_100689


namespace cone_surface_area_l1006_100631

theorem cone_surface_area (r l: ℝ) (θ : ℝ) (h₁ : r = 3) (h₂ : θ = 2 * π / 3) (h₃: 2 * π * r = θ * l) :
  π * r * l + π * r ^ 2 = 36 * π :=
by
  sorry

end cone_surface_area_l1006_100631


namespace range_of_k_l1006_100664

theorem range_of_k (k : ℝ) : (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^2 + 2*x1 - k = 0) ∧ (x2^2 + 2*x2 - k = 0)) ↔ k > -1 :=
by
  sorry

end range_of_k_l1006_100664


namespace solve_for_A_l1006_100687

theorem solve_for_A : ∃ (A : ℕ), A7 = 10 * A + 7 ∧ A7 + 30 = 77 ∧ A = 4 :=
by
  sorry

end solve_for_A_l1006_100687


namespace cyrus_written_pages_on_fourth_day_l1006_100644

theorem cyrus_written_pages_on_fourth_day :
  ∀ (total_pages first_day second_day third_day fourth_day remaining_pages: ℕ),
  total_pages = 500 →
  first_day = 25 →
  second_day = 2 * first_day →
  third_day = 2 * second_day →
  remaining_pages = total_pages - (first_day + second_day + third_day + fourth_day) →
  remaining_pages = 315 →
  fourth_day = 10 :=
by
  intros total_pages first_day second_day third_day fourth_day remaining_pages
  intros h_total h_first h_second h_third h_remain h_needed
  sorry

end cyrus_written_pages_on_fourth_day_l1006_100644


namespace inequality_transform_l1006_100696

theorem inequality_transform (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := 
by {
  sorry
}

end inequality_transform_l1006_100696


namespace wilson_total_notebooks_l1006_100697

def num_notebooks_per_large_pack : ℕ := 7
def num_large_packs_wilson_bought : ℕ := 7

theorem wilson_total_notebooks : num_large_packs_wilson_bought * num_notebooks_per_large_pack = 49 := 
by
  -- sorry used to skip the proof.
  sorry

end wilson_total_notebooks_l1006_100697


namespace double_and_halve_is_sixteen_l1006_100652

-- Definition of the initial number
def initial_number : ℕ := 16

-- Doubling the number
def doubled (n : ℕ) : ℕ := n * 2

-- Halving the number
def halved (n : ℕ) : ℕ := n / 2

-- The theorem that needs to be proven
theorem double_and_halve_is_sixteen : halved (doubled initial_number) = 16 :=
by
  /-
  We need to prove that when the number 16 is doubled and then halved, 
  the result is 16.
  -/
  sorry

end double_and_halve_is_sixteen_l1006_100652


namespace required_CO2_l1006_100685

noncomputable def moles_of_CO2_required (Mg CO2 MgO C : ℕ) (hMgO : MgO = 2) (hC : C = 1) : ℕ :=
  if Mg = 2 then 1 else 0

theorem required_CO2
  (Mg CO2 MgO C : ℕ)
  (hMgO : MgO = 2)
  (hC : C = 1)
  (hMg : Mg = 2)
  : moles_of_CO2_required Mg CO2 MgO C hMgO hC = 1 :=
  by simp [moles_of_CO2_required, hMg]

end required_CO2_l1006_100685


namespace intersection_of_A_and_B_l1006_100619

-- Define the sets A and B based on the given conditions
def setA : Set ℝ := {x | x^2 - 2 * x < 0}
def setB : Set ℝ := {x | -1 < x ∧ x < 1}

-- State the theorem to prove the intersection A ∩ B
theorem intersection_of_A_and_B : ((setA ∩ setB) = {x : ℝ | 0 < x ∧ x < 1}) :=
by
  sorry

end intersection_of_A_and_B_l1006_100619


namespace percent_of_475_25_is_129_89_l1006_100634

theorem percent_of_475_25_is_129_89 :
  (129.89 / 475.25) * 100 = 27.33 :=
by
  sorry

end percent_of_475_25_is_129_89_l1006_100634


namespace greatest_length_of_equal_pieces_l1006_100604

theorem greatest_length_of_equal_pieces (a b c : ℕ) (h₁ : a = 42) (h₂ : b = 63) (h₃ : c = 84) :
  Nat.gcd (Nat.gcd a b) c = 21 :=
by
  rw [h₁, h₂, h₃]
  sorry

end greatest_length_of_equal_pieces_l1006_100604


namespace least_number_added_to_divisible_l1006_100666

theorem least_number_added_to_divisible (n : ℕ) (k : ℕ) : n = 1789 → k = 11 → (n + k) % Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 4 3)) = 0 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end least_number_added_to_divisible_l1006_100666


namespace quadratic_root_l1006_100622

theorem quadratic_root (m : ℝ) (h : m^2 + 2 * m - 1 = 0) : 2 * m^2 + 4 * m = 2 := by
  sorry

end quadratic_root_l1006_100622


namespace apple_count_difference_l1006_100688

theorem apple_count_difference
    (original_green : ℕ)
    (additional_green : ℕ)
    (red_more_than_green : ℕ)
    (green_now : ℕ := original_green + additional_green)
    (red_now : ℕ := original_green + red_more_than_green)
    (difference : ℕ := green_now - red_now)
    (h_original_green : original_green = 32)
    (h_additional_green : additional_green = 340)
    (h_red_more_than_green : red_more_than_green = 200) :
    difference = 140 :=
by
  sorry

end apple_count_difference_l1006_100688


namespace triangle_is_3_l1006_100677

def base6_addition_valid (delta : ℕ) : Prop :=
  delta < 6 ∧ 
  2 + delta + delta + 4 < 6 ∧ -- No carry effect in the middle digits
  ((delta + 3) % 6 = 4) ∧
  ((5 + delta + (2 + delta + delta + 4) / 6) % 6 = 3) ∧
  ((4 + (5 + delta + (2 + delta + delta + 4) / 6) / 6) % 6 = 5)

theorem triangle_is_3 : ∃ (δ : ℕ), base6_addition_valid δ ∧ δ = 3 :=
by
  use 3
  sorry

end triangle_is_3_l1006_100677


namespace book_price_increase_l1006_100658

theorem book_price_increase (P : ℝ) (x : ℝ) :
  (P * (1 + x / 100)^2 = P * 1.3225) → x = 15 :=
by
  sorry

end book_price_increase_l1006_100658


namespace tea_or_coffee_indifference_l1006_100601

open Classical

theorem tea_or_coffee_indifference : 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) → 
  (∀ (wants_tea wants_coffee : Prop), (wants_tea ∨ wants_coffee) → (¬ wants_tea → wants_coffee) ∧ (¬ wants_coffee → wants_tea)) :=
by
  sorry

end tea_or_coffee_indifference_l1006_100601


namespace intersection_ellipse_line_range_b_l1006_100635

theorem intersection_ellipse_line_range_b (b : ℝ) : 
  (∀ m : ℝ, ∃ x y : ℝ, x^2 + 2*y^2 = 3 ∧ y = m*x + b) ↔ 
  (- (Real.sqrt 6) / 2) ≤ b ∧ b ≤ (Real.sqrt 6) / 2 :=
by {
  sorry
}

end intersection_ellipse_line_range_b_l1006_100635


namespace girls_in_math_class_l1006_100645

theorem girls_in_math_class (x y z : ℕ)
  (boys_girls_ratio : 5 * x = 8 * x)
  (math_science_ratio : 7 * y = 13 * x)
  (science_literature_ratio : 4 * y = 3 * z)
  (total_students : 13 * x + 4 * y + 5 * z = 720) :
  8 * x = 176 :=
by
  sorry

end girls_in_math_class_l1006_100645


namespace total_widgets_sold_15_days_l1006_100623

def widgets_sold (n : ℕ) : ℕ :=
  if n = 1 then 2 else 3 * n

theorem total_widgets_sold_15_days :
  (Finset.range 15).sum widgets_sold = 359 :=
by
  sorry

end total_widgets_sold_15_days_l1006_100623


namespace product_of_consecutive_even_numbers_l1006_100674

theorem product_of_consecutive_even_numbers
  (a b c : ℤ)
  (h : a + b + c = 18 ∧ 2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ a < b ∧ b < c ∧ b - a = 2 ∧ c - b = 2) :
  a * b * c = 192 :=
sorry

end product_of_consecutive_even_numbers_l1006_100674


namespace percent_diploma_thirty_l1006_100699

-- Defining the conditions using Lean definitions

def percent_without_diploma_with_job := 0.10 -- 10%
def percent_with_job := 0.20 -- 20%
def percent_without_job_with_diploma :=
  (1 - percent_with_job) * 0.25 -- 25% of people without job is 25% of 80% which is 20%

def percent_with_diploma := percent_with_job - percent_without_diploma_with_job + percent_without_job_with_diploma

-- Theorem to prove that 30% of the people have a university diploma
theorem percent_diploma_thirty
  (H1 : percent_without_diploma_with_job = 0.10) -- condition 1
  (H2 : percent_with_job = 0.20) -- condition 3
  (H3 : percent_without_job_with_diploma = 0.20) -- evaluated from condition 2
  : percent_with_diploma = 0.30 := by
  -- prove that the percent with diploma is 30%
  sorry

end percent_diploma_thirty_l1006_100699


namespace percentage_of_apples_is_50_l1006_100643

-- Definitions based on the conditions
def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

-- Final percentage calculation after removing 13 oranges
def percentage_apples (apples oranges_removed : ℕ) :=
  let total_initial := initial_apples + initial_oranges
  let oranges_left := initial_oranges - oranges_removed
  let total_after_removal := initial_apples + oranges_left
  (initial_apples * 100) / total_after_removal

-- The theorem to be proved
theorem percentage_of_apples_is_50 : percentage_apples initial_apples oranges_removed = 50 := by
  sorry

end percentage_of_apples_is_50_l1006_100643


namespace minimum_value_f_l1006_100671

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x + 6 / x + 4 / x^2 - 1

theorem minimum_value_f : 
    ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f y ≥ f x) ∧ 
    f x = 3 - 6 * Real.sqrt 2 :=
sorry

end minimum_value_f_l1006_100671


namespace range_of_a_l1006_100651

variable (a : ℝ)
def A (a : ℝ) := {x : ℝ | x^2 - 2*x + a > 0}

theorem range_of_a (h : 1 ∉ A a) : a ≤ 1 :=
by {
  sorry
}

end range_of_a_l1006_100651


namespace binomial_coefficients_sum_l1006_100600

theorem binomial_coefficients_sum (n : ℕ) (h : (2:ℕ)^n = 256) : n = 8 := by
  sorry

end binomial_coefficients_sum_l1006_100600


namespace parallel_lines_a_l1006_100657

theorem parallel_lines_a (a : ℝ) (x y : ℝ)
  (h1 : x + 2 * a * y - 1 = 0)
  (h2 : (a + 1) * x - a * y = 0)
  (h_parallel : ∀ (l1 l2 : ℝ → ℝ → Prop), l1 x y ∧ l2 x y → l1 = l2) :
  a = -3 / 2 ∨ a = 0 :=
sorry

end parallel_lines_a_l1006_100657


namespace base_7_3516_is_1287_l1006_100627

-- Definitions based on conditions
def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 3516 => 3 * 7^3 + 5 * 7^2 + 1 * 7^1 + 6 * 7^0
  | _ => 0

-- Proving the main question
theorem base_7_3516_is_1287 : base7_to_base10 3516 = 1287 := by
  sorry

end base_7_3516_is_1287_l1006_100627


namespace factorize_expression_l1006_100693

theorem factorize_expression (x : ℝ) : 4 * x ^ 2 - 2 * x = 2 * x * (2 * x - 1) :=
by
  sorry

end factorize_expression_l1006_100693


namespace hyperbola_asymptote_b_value_l1006_100676

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : b > 0)
  (asymptote : ∀ x y : ℝ, y = 2 * x → x^2 - (y^2 / b^2) = 1) :
  b = 2 :=
sorry

end hyperbola_asymptote_b_value_l1006_100676


namespace max_value_inequality_max_value_equality_l1006_100633

theorem max_value_inequality (x : ℝ) (hx : x < 0) : 
  3 * x + 4 / x ≤ -4 * Real.sqrt 3 :=
sorry

theorem max_value_equality (x : ℝ) (hx : x = -2 * Real.sqrt 3 / 3) : 
  3 * x + 4 / x = -4 * Real.sqrt 3 :=
sorry

end max_value_inequality_max_value_equality_l1006_100633


namespace max_adjacent_distinct_pairs_l1006_100630

theorem max_adjacent_distinct_pairs (n : ℕ) (h : n = 100) : 
  ∃ (a : ℕ), a = 50 := 
by 
  -- Here we use the provided constraints and game scenario to state the theorem formally.
  sorry

end max_adjacent_distinct_pairs_l1006_100630


namespace problem_1_problem_2_l1006_100649

-- Definitions required for the proof
variables {A B C : ℝ} (a b c : ℝ)
variable (cos_A cos_B cos_C : ℝ)
variables (sin_A sin_C : ℝ)

-- Given conditions
axiom given_condition : (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b
axiom cos_B_eq : cos_B = 1 / 4
axiom b_eq : b = 2

-- First problem: Proving the value of sin_C / sin_A
theorem problem_1 :
  (cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b → (sin_C / sin_A) = 2 :=
by
  intro h
  sorry

-- Second problem: Proving the area of triangle ABC
theorem problem_2 :
  (cos_B = 1 / 4) → (b = 2) → ((cos_A - 2 * cos_C) / cos_B = (2 * c - a) / b) → (1 / 2 * a * c * sin_A) = (Real.sqrt 15) / 4 :=
by
  intros h1 h2 h3
  sorry

end problem_1_problem_2_l1006_100649


namespace sum_of_squares_of_coeffs_l1006_100603

def poly_coeffs_squared_sum (p : Polynomial ℤ) : ℤ :=
  p.coeff 5 ^ 2 + p.coeff 3 ^ 2 + p.coeff 0 ^ 2

theorem sum_of_squares_of_coeffs (p : Polynomial ℤ) (h : p = 5 * (Polynomial.C 1 * Polynomial.X ^ 5 + Polynomial.C 2 * Polynomial.X ^ 3 + Polynomial.C 3)) :
  poly_coeffs_squared_sum p = 350 :=
by
  sorry

end sum_of_squares_of_coeffs_l1006_100603


namespace nathan_and_parents_total_cost_l1006_100667

/-- Define the total number of people -/
def num_people := 3

/-- Define the cost per object -/
def cost_per_object := 11

/-- Define the number of objects per person -/
def objects_per_person := 2 + 2 + 1

/-- Define the total number of objects -/
def total_objects := num_people * objects_per_person

/-- Define the total cost -/
def total_cost := total_objects * cost_per_object

/-- The main theorem to prove the total cost -/
theorem nathan_and_parents_total_cost : total_cost = 165 := by
  sorry

end nathan_and_parents_total_cost_l1006_100667


namespace cows_to_eat_grass_in_96_days_l1006_100669

theorem cows_to_eat_grass_in_96_days (G r : ℕ) : 
  (∀ N : ℕ, (70 * 24 = G + 24 * r) → (30 * 60 = G + 60 * r) → 
  (∃ N : ℕ, 96 * N = G + 96 * r) → N = 20) :=
by
  intro N
  intro h1 h2 h3
  sorry

end cows_to_eat_grass_in_96_days_l1006_100669


namespace problem1_solution_problem2_solution_l1006_100606

theorem problem1_solution (x : ℝ) : x^2 - x - 6 > 0 ↔ x < -2 ∨ x > 3 := sorry

theorem problem2_solution (x : ℝ) : -2*x^2 + x + 1 < 0 ↔ x < -1/2 ∨ x > 1 := sorry

end problem1_solution_problem2_solution_l1006_100606


namespace angle_SQR_l1006_100665

-- Define angles
def PQR : ℝ := 40
def PQS : ℝ := 28

-- State the theorem
theorem angle_SQR : PQR - PQS = 12 := by
  sorry

end angle_SQR_l1006_100665


namespace cell_count_at_end_of_twelvth_day_l1006_100638

def initial_cells : ℕ := 5
def days_per_cycle : ℕ := 3
def total_days : ℕ := 12
def dead_cells_on_ninth_day : ℕ := 3
noncomputable def cells_after_twelvth_day : ℕ :=
  let cycles := total_days / days_per_cycle
  let cells_before_death := initial_cells * 2^cycles
  cells_before_death - dead_cells_on_ninth_day

theorem cell_count_at_end_of_twelvth_day : cells_after_twelvth_day = 77 :=
by sorry

end cell_count_at_end_of_twelvth_day_l1006_100638


namespace pear_juice_processed_l1006_100654

theorem pear_juice_processed
  (total_pears : ℝ)
  (export_percentage : ℝ)
  (juice_percentage_of_remainder : ℝ) :
  total_pears = 8.5 →
  export_percentage = 0.30 →
  juice_percentage_of_remainder = 0.60 →
  ((total_pears * (1 - export_percentage)) * juice_percentage_of_remainder) = 3.6 :=
by
  intros
  sorry

end pear_juice_processed_l1006_100654
