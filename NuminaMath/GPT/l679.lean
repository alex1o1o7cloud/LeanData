import Mathlib

namespace NUMINAMATH_GPT_proof_problem_l679_67954

variable {R : Type} [OrderedRing R]

-- Definitions and conditions
variable (g : R → R) (f : R → R) (k a m : R)
variable (h_odd : ∀ x : R, g (-x) = -g x)
variable (h_f_def : ∀ x : R, f x = g x + k)
variable (h_f_neg_a : f (-a) = m)

-- Theorem statement
theorem proof_problem : f a = 2 * k - m :=
by
  -- Here is where the proof would go.
  sorry

end NUMINAMATH_GPT_proof_problem_l679_67954


namespace NUMINAMATH_GPT_longest_side_in_ratio_5_6_7_l679_67984

theorem longest_side_in_ratio_5_6_7 (x : ℕ) (h : 5 * x + 6 * x + 7 * x = 720) : 7 * x = 280 := 
by
  sorry

end NUMINAMATH_GPT_longest_side_in_ratio_5_6_7_l679_67984


namespace NUMINAMATH_GPT_intersection_A_B_l679_67975

def A := { x : ℝ | x / (x - 1) ≥ 0 }
def B := { y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B : A ∩ B = { y : ℝ | y > 1 } :=
by sorry

end NUMINAMATH_GPT_intersection_A_B_l679_67975


namespace NUMINAMATH_GPT_sqrt_two_squared_l679_67999

noncomputable def sqrt_two : Real := Real.sqrt 2

theorem sqrt_two_squared : (sqrt_two) ^ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_two_squared_l679_67999


namespace NUMINAMATH_GPT_area_of_triangle_l679_67970

namespace TriangleArea

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

noncomputable def area (A B C : Point3D) : ℚ :=
  let x1 := A.x
  let y1 := A.y
  let z1 := A.z
  let x2 := B.x
  let y2 := B.y
  let z2 := B.z
  let x3 := C.x
  let y3 := C.y
  let z3 := C.z
  1 / 2 * ( (x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)) )

def A : Point3D := ⟨0, 3, 6⟩
def B : Point3D := ⟨-2, 2, 2⟩
def C : Point3D := ⟨-5, 5, 2⟩

theorem area_of_triangle : area A B C = 4.5 :=
by
  sorry

end TriangleArea

end NUMINAMATH_GPT_area_of_triangle_l679_67970


namespace NUMINAMATH_GPT_abs_difference_of_numbers_l679_67991

theorem abs_difference_of_numbers (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 391) :
  |x - y| = 6 :=
sorry

end NUMINAMATH_GPT_abs_difference_of_numbers_l679_67991


namespace NUMINAMATH_GPT_evaluates_to_m_times_10_pow_1012_l679_67973

theorem evaluates_to_m_times_10_pow_1012 :
  let a := (3:ℤ) ^ 1010
  let b := (4:ℤ) ^ 1012
  (a + b) ^ 2 - (a - b) ^ 2 = 10 ^ 3642 := by
  sorry

end NUMINAMATH_GPT_evaluates_to_m_times_10_pow_1012_l679_67973


namespace NUMINAMATH_GPT_area_and_cost_of_path_l679_67948

variables (length_field width_field path_width : ℝ) (cost_per_sq_m : ℝ)

noncomputable def area_of_path (length_field width_field path_width : ℝ) : ℝ :=
  let total_length := length_field + 2 * path_width
  let total_width := width_field + 2 * path_width
  let area_with_path := total_length * total_width
  let area_grass_field := length_field * width_field
  area_with_path - area_grass_field

noncomputable def cost_of_path (area_of_path cost_per_sq_m : ℝ) : ℝ :=
  area_of_path * cost_per_sq_m

theorem area_and_cost_of_path
  (length_field width_field path_width : ℝ)
  (cost_per_sq_m : ℝ)
  (h_length_field : length_field = 75)
  (h_width_field : width_field = 55)
  (h_path_width : path_width = 2.5)
  (h_cost_per_sq_m : cost_per_sq_m = 10) :
  area_of_path length_field width_field path_width = 675 ∧
  cost_of_path (area_of_path length_field width_field path_width) cost_per_sq_m = 6750 :=
by
  rw [h_length_field, h_width_field, h_path_width, h_cost_per_sq_m]
  simp [area_of_path, cost_of_path]
  sorry

end NUMINAMATH_GPT_area_and_cost_of_path_l679_67948


namespace NUMINAMATH_GPT_min_value_c_l679_67907

-- Define the problem using Lean
theorem min_value_c 
    (a b c d e : ℕ)
    (h1 : a + 1 = b) 
    (h2 : b + 1 = c)
    (h3 : c + 1 = d)
    (h4 : d + 1 = e)
    (h5 : ∃ n : ℕ, 5 * c = n ^ 3)
    (h6 : ∃ m : ℕ, 3 * c = m ^ 2) : 
    c = 675 := 
sorry

end NUMINAMATH_GPT_min_value_c_l679_67907


namespace NUMINAMATH_GPT_min_m_squared_plus_n_squared_l679_67952

theorem min_m_squared_plus_n_squared {m n : ℝ} (h : 4 * m - 3 * n - 5 * Real.sqrt 2 = 0) :
  m^2 + n^2 = 2 :=
sorry

end NUMINAMATH_GPT_min_m_squared_plus_n_squared_l679_67952


namespace NUMINAMATH_GPT_lemons_needed_l679_67928

theorem lemons_needed (lemons32 : ℕ) (lemons4 : ℕ) (h1 : lemons32 = 24) (h2 : (24 : ℕ) / 32 = (lemons4 : ℕ) / 4) : lemons4 = 3 := 
sorry

end NUMINAMATH_GPT_lemons_needed_l679_67928


namespace NUMINAMATH_GPT_find_biology_marks_l679_67919

theorem find_biology_marks (english math physics chemistry : ℕ) (avg_marks : ℕ) (biology : ℕ)
  (h_english : english = 86) (h_math : math = 89) (h_physics : physics = 82)
  (h_chemistry : chemistry = 87) (h_avg_marks : avg_marks = 85) :
  (english + math + physics + chemistry + biology) = avg_marks * 5 →
  biology = 81 :=
by
  sorry

end NUMINAMATH_GPT_find_biology_marks_l679_67919


namespace NUMINAMATH_GPT_find_length_of_MN_l679_67926

theorem find_length_of_MN (A B C M N : ℝ × ℝ)
  (AB AC : ℝ) (M_midpoint : M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))
  (N_midpoint : N = ((A.1 + C.1) / 2, (A.2 + C.2) / 2))
  (length_AB : abs (B.1 - A.1) + abs (B.2 - A.2) = 15)
  (length_AC : abs (C.1 - A.1) + abs (C.2 - A.2) = 20) :
  abs (N.1 - M.1) + abs (N.2 - M.2) = 40 / 3 := sorry

end NUMINAMATH_GPT_find_length_of_MN_l679_67926


namespace NUMINAMATH_GPT_volume_region_between_spheres_l679_67963

theorem volume_region_between_spheres 
    (r1 r2 : ℝ) 
    (h1 : r1 = 4) 
    (h2 : r2 = 7) 
    : 
    ( (4/3) * π * r2^3 - (4/3) * π * r1^3 ) = 372 * π := 
    sorry

end NUMINAMATH_GPT_volume_region_between_spheres_l679_67963


namespace NUMINAMATH_GPT_distance_between_points_l679_67951

theorem distance_between_points :
  let x1 := 1
  let y1 := 3
  let z1 := 2
  let x2 := 4
  let y2 := 1
  let z2 := 6
  let distance : ℝ := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)
  distance = Real.sqrt 29 := by
  sorry

end NUMINAMATH_GPT_distance_between_points_l679_67951


namespace NUMINAMATH_GPT_Mark_owes_total_l679_67912

noncomputable def base_fine : ℕ := 50

def additional_fine (speed_over_limit : ℕ) : ℕ :=
  let first_10 := min speed_over_limit 10 * 2
  let next_5 := min (speed_over_limit - 10) 5 * 3
  let next_10 := min (speed_over_limit - 15) 10 * 5
  let remaining := max (speed_over_limit - 25) 0 * 6
  first_10 + next_5 + next_10 + remaining

noncomputable def total_fine (base : ℕ) (additional : ℕ) (school_zone : Bool) : ℕ :=
  let fine := base + additional
  if school_zone then fine * 2 else fine

def court_costs : ℕ := 350

noncomputable def processing_fee (fine : ℕ) : ℕ := fine / 10

def lawyer_fees (hourly_rate : ℕ) (hours : ℕ) : ℕ := hourly_rate * hours

theorem Mark_owes_total :
  let speed_over_limit := 45
  let base := base_fine
  let additional := additional_fine speed_over_limit
  let school_zone := true
  let fine := total_fine base additional school_zone
  let total_fine_with_costs := fine + court_costs
  let processing := processing_fee total_fine_with_costs
  let lawyer := lawyer_fees 100 4
  let total := total_fine_with_costs + processing + lawyer
  total = 1346 := sorry

end NUMINAMATH_GPT_Mark_owes_total_l679_67912


namespace NUMINAMATH_GPT_triangle_angle_A_l679_67934

theorem triangle_angle_A (A B a b : ℝ) (h1 : b = 2 * a) (h2 : B = A + 60) : A = 30 :=
by sorry

end NUMINAMATH_GPT_triangle_angle_A_l679_67934


namespace NUMINAMATH_GPT_square_completing_l679_67962

theorem square_completing (b c : ℤ) (h : (x^2 - 10 * x + 15 = 0) → ((x + b)^2 = c)) : 
  b + c = 5 :=
sorry

end NUMINAMATH_GPT_square_completing_l679_67962


namespace NUMINAMATH_GPT_series_sum_eq_negative_one_third_l679_67964

noncomputable def series_sum : ℝ :=
  ∑' n, (2 * n + 1) / (n * (n + 1) * (n + 2) * (n + 3))

theorem series_sum_eq_negative_one_third : series_sum = -1 / 3 := sorry

end NUMINAMATH_GPT_series_sum_eq_negative_one_third_l679_67964


namespace NUMINAMATH_GPT_average_speed_correct_l679_67992

noncomputable def total_distance := 120 + 70
noncomputable def total_time := 2
noncomputable def average_speed := total_distance / total_time

theorem average_speed_correct :
  average_speed = 95 := by
  sorry

end NUMINAMATH_GPT_average_speed_correct_l679_67992


namespace NUMINAMATH_GPT_all_possible_triples_l679_67909

theorem all_possible_triples (x y : ℕ) (z : ℤ) (hz : z % 2 = 1)
                            (h : x.factorial + y.factorial = 8 * z + 2017) :
                            (x = 1 ∧ y = 4 ∧ z = -249) ∨
                            (x = 4 ∧ y = 1 ∧ z = -249) ∨
                            (x = 1 ∧ y = 5 ∧ z = -237) ∨
                            (x = 5 ∧ y = 1 ∧ z = -237) := 
  sorry

end NUMINAMATH_GPT_all_possible_triples_l679_67909


namespace NUMINAMATH_GPT_find_digits_l679_67994

theorem find_digits (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h_sum : 100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a)) :
  a = 1 ∧ b = 9 ∧ c = 8 := by
  sorry

end NUMINAMATH_GPT_find_digits_l679_67994


namespace NUMINAMATH_GPT_root_ratios_equal_l679_67906

theorem root_ratios_equal (a : ℝ) (ha : 0 < a)
  (hroots : ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁^3 + 1 = a * x₁ ∧ x₂^3 + 1 = a * x₂ ∧ x₂ / x₁ = 2018) :
  ∃ y₁ y₂ : ℝ, 0 < y₁ ∧ 0 < y₂ ∧ y₁^3 + 1 = a * y₁^2 ∧ y₂^3 + 1 = a * y₂^2 ∧ y₂ / y₁ = 2018 :=
sorry

end NUMINAMATH_GPT_root_ratios_equal_l679_67906


namespace NUMINAMATH_GPT_sequence_value_G_50_l679_67935

theorem sequence_value_G_50 :
  ∀ G : ℕ → ℚ, (∀ n : ℕ, G (n + 1) = (3 * G n + 1) / 3) ∧ G 1 = 3 → G 50 = 152 / 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_value_G_50_l679_67935


namespace NUMINAMATH_GPT_apple_bags_l679_67979

theorem apple_bags (n : ℕ) (h₁ : n ≥ 70) (h₂ : n ≤ 80) (h₃ : n % 6 = 0) : n = 72 ∨ n = 78 := 
sorry

end NUMINAMATH_GPT_apple_bags_l679_67979


namespace NUMINAMATH_GPT_subtraction_of_7_305_from_neg_3_219_l679_67978

theorem subtraction_of_7_305_from_neg_3_219 :
  -3.219 - 7.305 = -10.524 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_subtraction_of_7_305_from_neg_3_219_l679_67978


namespace NUMINAMATH_GPT_probability_red_given_spade_or_king_l679_67968

def num_cards := 52
def num_spades := 13
def num_kings := 4
def num_red_kings := 2

def num_non_spade_kings := num_kings - 1
def num_spades_or_kings := num_spades + num_non_spade_kings

theorem probability_red_given_spade_or_king :
  (num_red_kings : ℚ) / num_spades_or_kings = 1 / 8 :=
sorry

end NUMINAMATH_GPT_probability_red_given_spade_or_king_l679_67968


namespace NUMINAMATH_GPT_decorate_eggs_time_calculation_l679_67905

/-- Definition of Mia's and Billy's egg decorating rates, total number of eggs to be decorated, and the calculated time when working together --/
def MiaRate : ℕ := 24
def BillyRate : ℕ := 10
def totalEggs : ℕ := 170
def combinedRate : ℕ := MiaRate + BillyRate

theorem decorate_eggs_time_calculation :
  (totalEggs / combinedRate) = 5 := by
  sorry

end NUMINAMATH_GPT_decorate_eggs_time_calculation_l679_67905


namespace NUMINAMATH_GPT_probability_of_specific_sequence_l679_67941

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end NUMINAMATH_GPT_probability_of_specific_sequence_l679_67941


namespace NUMINAMATH_GPT_pepperoni_ratio_l679_67956

-- Definition of the problem's conditions
def total_pepperoni_slices : ℕ := 40
def slice_given_to_jelly_original : ℕ := 10
def slice_fallen_off : ℕ := 1

-- Our goal is to prove that the ratio is 3:10
theorem pepperoni_ratio (total_pepperoni_slices : ℕ) (slice_given_to_jelly_original : ℕ) (slice_fallen_off : ℕ) :
  (slice_given_to_jelly_original - slice_fallen_off) / (total_pepperoni_slices - slice_given_to_jelly_original) = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_pepperoni_ratio_l679_67956


namespace NUMINAMATH_GPT_Jasmine_gets_off_work_at_4pm_l679_67958

-- Conditions
def commute_time : ℕ := 30
def grocery_time : ℕ := 30
def dry_clean_time : ℕ := 10
def groomer_time : ℕ := 20
def cook_time : ℕ := 90
def dinner_time : ℕ := 19 * 60  -- 7:00 pm in minutes

-- Question to prove
theorem Jasmine_gets_off_work_at_4pm : 
  (dinner_time - cook_time - groomer_time - dry_clean_time - grocery_time - commute_time = 16 * 60) := sorry

end NUMINAMATH_GPT_Jasmine_gets_off_work_at_4pm_l679_67958


namespace NUMINAMATH_GPT_square_of_rational_l679_67903

theorem square_of_rational (b : ℚ) : b^2 = b * b :=
sorry

end NUMINAMATH_GPT_square_of_rational_l679_67903


namespace NUMINAMATH_GPT_spend_on_rent_and_utilities_l679_67946

variable (P : ℝ) -- The percentage of her income she used to spend on rent and utilities
variable (I : ℝ) -- Her previous monthly income
variable (increase : ℝ) -- Her salary increase
variable (new_percentage : ℝ) -- The new percentage her rent and utilities amount to

noncomputable def initial_conditions : Prop :=
I = 1000 ∧ increase = 600 ∧ new_percentage = 0.25

theorem spend_on_rent_and_utilities (h : initial_conditions I increase new_percentage) :
    (P / 100) * I = 0.25 * (I + increase) → 
    P = 40 :=
by
  sorry

end NUMINAMATH_GPT_spend_on_rent_and_utilities_l679_67946


namespace NUMINAMATH_GPT_octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l679_67942

-- Part (a) - Octahedron
/- 
A connected graph representing an octahedron. 
Each vertex has a degree of 4, making the graph Eulerian.
-/
theorem octahedron_has_eulerian_circuit : 
  ∃ circuit : List (ℕ × ℕ), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

-- Part (b) - Cube
/- 
A connected graph representing a cube.
Each vertex has a degree of 3, making it impossible for the graph to be Eulerian.
-/
theorem cube_has_no_eulerian_circuit : 
  ¬ ∃ (circuit : List (ℕ × ℕ)), 
    (∀ (u v : ℕ), List.elem (u, v) circuit ↔ List.elem (v, u) circuit) ∧
    (∃ start, ∀ v ∈ circuit, v = start) :=
sorry

end NUMINAMATH_GPT_octahedron_has_eulerian_circuit_cube_has_no_eulerian_circuit_l679_67942


namespace NUMINAMATH_GPT_total_pencils_is_54_l679_67959

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_is_54_l679_67959


namespace NUMINAMATH_GPT_least_positive_multiple_24_gt_450_l679_67917

theorem least_positive_multiple_24_gt_450 : ∃ n : ℕ, n > 450 ∧ n % 24 = 0 ∧ n = 456 :=
by
  use 456
  sorry

end NUMINAMATH_GPT_least_positive_multiple_24_gt_450_l679_67917


namespace NUMINAMATH_GPT_ratio_of_men_to_women_l679_67932

def num_cannoneers : ℕ := 63
def num_people : ℕ := 378
def num_women (C : ℕ) : ℕ := 2 * C
def num_men (total : ℕ) (women : ℕ) : ℕ := total - women

theorem ratio_of_men_to_women : 
  let C := num_cannoneers
  let total := num_people
  let W := num_women C
  let M := num_men total W
  M / W = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_men_to_women_l679_67932


namespace NUMINAMATH_GPT_correct_operation_l679_67945

theorem correct_operation (a : ℝ) :
  (a^5)^2 = a^10 :=
by sorry

end NUMINAMATH_GPT_correct_operation_l679_67945


namespace NUMINAMATH_GPT_rainfall_sunday_l679_67949

theorem rainfall_sunday 
  (rain_sun rain_mon rain_tue : ℝ)
  (h1 : rain_mon = rain_sun + 3)
  (h2 : rain_tue = 2 * rain_mon)
  (h3 : rain_sun + rain_mon + rain_tue = 25) :
  rain_sun = 4 :=
by
  sorry

end NUMINAMATH_GPT_rainfall_sunday_l679_67949


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_necessary_condition_l679_67961

variable {a b m : ℝ}

theorem sufficient_but_not_necessary_condition (h : a * m^2 < b * m^2) : a < b := by
  sorry

-- Additional statements to express the sufficiency and not necessity nature:
theorem not_necessary_condition (h : a < b) (hm : m = 0) : ¬ (a * m^2 < b * m^2) := by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_not_necessary_condition_l679_67961


namespace NUMINAMATH_GPT_henry_income_percent_increase_l679_67930

theorem henry_income_percent_increase :
  let original_income : ℝ := 120
  let new_income : ℝ := 180
  let increase := new_income - original_income
  let percent_increase := (increase / original_income) * 100
  percent_increase = 50 :=
by
  sorry

end NUMINAMATH_GPT_henry_income_percent_increase_l679_67930


namespace NUMINAMATH_GPT_angle_degrees_l679_67927

-- Define the conditions
def sides_parallel (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = θ₂ ∨ (θ₁ + θ₂ = 180)

def angle_relation (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20

-- Statement of the problem
theorem angle_degrees (θ₁ θ₂ : ℝ) (h_parallel : sides_parallel θ₁ θ₂) (h_relation : angle_relation θ₁ θ₂) :
  (θ₁ = 10 ∧ θ₂ = 10) ∨ (θ₁ = 50 ∧ θ₂ = 130) ∨ (θ₁ = 130 ∧ θ₂ = 50) ∨ θ₁ + θ₂ = 180 ∧ (θ₁ = 3 * θ₂ - 20 ∨ θ₂ = 3 * θ₁ - 20) :=
by sorry

end NUMINAMATH_GPT_angle_degrees_l679_67927


namespace NUMINAMATH_GPT_product_of_repeating_decimals_l679_67950

theorem product_of_repeating_decimals :
  let x := (4 / 9 : ℚ)
  let y := (7 / 9 : ℚ)
  x * y = 28 / 81 :=
by
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimals_l679_67950


namespace NUMINAMATH_GPT_jam_cost_l679_67911

theorem jam_cost (N B J H : ℕ) (h1 : N > 1) (h2 : N * (3 * B + 6 * J + 2 * H) = 342) :
  6 * N * J = 270 := 
sorry

end NUMINAMATH_GPT_jam_cost_l679_67911


namespace NUMINAMATH_GPT_intersecting_circles_range_of_m_l679_67993

theorem intersecting_circles_range_of_m
  (x y m : ℝ)
  (C₁_eq : x^2 + y^2 - 2 * m * x + m^2 - 4 = 0)
  (C₂_eq : x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0)
  (intersect : ∃ x y : ℝ, (x^2 + y^2 - 2 * m * x + m^2 - 4 = 0) ∧ (x^2 + y^2 + 2 * x - 4 * m * y + 4 * m^2 - 8 = 0))
  : m ∈ Set.Ioo (-12/5) (-2/5) ∪ Set.Ioo (3/5) 2 := 
sorry

end NUMINAMATH_GPT_intersecting_circles_range_of_m_l679_67993


namespace NUMINAMATH_GPT_closest_points_distance_l679_67976

theorem closest_points_distance :
  let center1 := (2, 2)
  let center2 := (17, 10)
  let radius1 := 2
  let radius2 := 10
  let distance_centers := Nat.sqrt ((center2.1 - center1.1) ^ 2 + (center2.2 - center1.2) ^ 2)
  distance_centers = 17 → (distance_centers - radius1 - radius2) = 5 := by
  sorry

end NUMINAMATH_GPT_closest_points_distance_l679_67976


namespace NUMINAMATH_GPT_decimal_multiplication_l679_67924

theorem decimal_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by sorry

end NUMINAMATH_GPT_decimal_multiplication_l679_67924


namespace NUMINAMATH_GPT_solve_system_eqns_l679_67957

theorem solve_system_eqns 
  {a b c : ℝ} (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
  {x y z : ℝ} 
  (h4 : a^3 + a^2 * x + a * y + z = 0)
  (h5 : b^3 + b^2 * x + b * y + z = 0)
  (h6 : c^3 + c^2 * x + c * y + z = 0) :
  x = -(a + b + c) ∧ y = ab + bc + ca ∧ z = -abc :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_system_eqns_l679_67957


namespace NUMINAMATH_GPT_isosceles_triangle_height_eq_four_times_base_l679_67982

theorem isosceles_triangle_height_eq_four_times_base (b h : ℝ) 
    (same_area : (b * 2 * b) = (1/2 * b * h)) : 
    h = 4 * b :=
by 
  -- sorry allows us to skip the proof steps
  sorry

end NUMINAMATH_GPT_isosceles_triangle_height_eq_four_times_base_l679_67982


namespace NUMINAMATH_GPT_find_m_l679_67960

variable {a : ℕ → ℝ}
variable {q : ℝ}
variable {m : ℕ}

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def initial_condition (a : ℕ → ℝ) : Prop :=
  a 1 = 1

def q_condition (q : ℝ) : Prop :=
  abs q ≠ 1

def a_m_condition (a : ℕ → ℝ) (m : ℕ) : Prop :=
  a m = a 1 * a 2 * a 3 * a 4 * a 5

-- Theorem to prove
theorem find_m (h1 : geometric_sequence a q) (h2 : initial_condition a) (h3 : q_condition q) (h4 : a_m_condition a m) : m = 11 :=
  sorry

end NUMINAMATH_GPT_find_m_l679_67960


namespace NUMINAMATH_GPT_sum_of_series_l679_67915

open BigOperators

-- Define the sequence a(n) = 2 / (n * (n + 3))
def a (n : ℕ) : ℚ := 2 / (n * (n + 3))

-- Prove the sum of the first 20 terms of sequence a equals 10 / 9.
theorem sum_of_series : (∑ n in Finset.range 20, a (n + 1)) = 10 / 9 := by
  sorry

end NUMINAMATH_GPT_sum_of_series_l679_67915


namespace NUMINAMATH_GPT_units_digit_l679_67940

noncomputable def C := 20 + Real.sqrt 153
noncomputable def D := 20 - Real.sqrt 153

theorem units_digit (h : ∀ n ≥ 1, 20 ^ n % 10 = 0) :
  (C ^ 12 + D ^ 12) % 10 = 0 :=
by
  -- Proof will be provided based on the outlined solution
  sorry

end NUMINAMATH_GPT_units_digit_l679_67940


namespace NUMINAMATH_GPT_find_certain_number_l679_67938

theorem find_certain_number (x y : ℝ)
  (h1 : (28 + x + 42 + y + 104) / 5 = 62)
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) :
  y = 78 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l679_67938


namespace NUMINAMATH_GPT_decrease_percent_revenue_l679_67998

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let original_revenue := T * C
  let new_tax := 0.68 * T
  let new_consumption := 1.12 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 23.84 := by {
    sorry
  }

end NUMINAMATH_GPT_decrease_percent_revenue_l679_67998


namespace NUMINAMATH_GPT_probability_red_or_white_l679_67931

-- Define the total number of marbles and the counts of blue and red marbles.
def total_marbles : Nat := 60
def blue_marbles : Nat := 5
def red_marbles : Nat := 9

-- Define the remainder to calculate white marbles.
def white_marbles : Nat := total_marbles - (blue_marbles + red_marbles)

-- Lean proof statement to show the probability of selecting a red or white marble.
theorem probability_red_or_white :
  (red_marbles + white_marbles) / total_marbles = 11 / 12 :=
by
  sorry

end NUMINAMATH_GPT_probability_red_or_white_l679_67931


namespace NUMINAMATH_GPT_first_floor_cost_l679_67986

-- Definitions and assumptions
variables (F : ℝ)
variables (earnings_first_floor earnings_second_floor earnings_third_floor : ℝ)
variables (total_monthly_earnings : ℝ)

-- Conditions from the problem
def costs := F
def second_floor_costs := F + 20
def third_floor_costs := 2 * F
def first_floor_rooms := 3 * costs
def second_floor_rooms := 3 * second_floor_costs
def third_floor_rooms := 3 * third_floor_costs

-- Total monthly earnings
def total_earnings := first_floor_rooms + second_floor_rooms + third_floor_rooms

-- Equality condition
axiom total_earnings_is_correct : total_earnings = 165

-- Theorem to be proved
theorem first_floor_cost :
  (F = 8.75) :=
by
  have earnings_first_floor_eq := first_floor_rooms
  have earnings_second_floor_eq := second_floor_rooms
  have earnings_third_floor_eq := third_floor_rooms
  have total_earning_eq := total_earnings_is_correct
  sorry

end NUMINAMATH_GPT_first_floor_cost_l679_67986


namespace NUMINAMATH_GPT_fraction_subtraction_simplified_l679_67922

theorem fraction_subtraction_simplified : (8 / 19 - 5 / 57) = (1 / 3) := by
  sorry

end NUMINAMATH_GPT_fraction_subtraction_simplified_l679_67922


namespace NUMINAMATH_GPT_right_triangle_geo_seq_ratio_l679_67910

theorem right_triangle_geo_seq_ratio (l r : ℝ) (ht : 0 < l)
  (hr : 1 < r) (hgeo : l^2 + (l * r)^2 = (l * r^2)^2) :
  (l * r^2) / l = (1 + Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_geo_seq_ratio_l679_67910


namespace NUMINAMATH_GPT_third_year_increment_l679_67921

-- Define the conditions
def total_payments : ℕ := 96
def first_year_cost : ℕ := 20
def second_year_cost : ℕ := first_year_cost + 2
def third_year_cost (x : ℕ) : ℕ := second_year_cost + x
def fourth_year_cost (x : ℕ) : ℕ := third_year_cost x + 4

-- The main proof statement
theorem third_year_increment (x : ℕ) 
  (H : first_year_cost + second_year_cost + third_year_cost x + fourth_year_cost x = total_payments) :
  x = 2 :=
sorry

end NUMINAMATH_GPT_third_year_increment_l679_67921


namespace NUMINAMATH_GPT_victor_cannot_escape_k4_l679_67965

theorem victor_cannot_escape_k4
  (r : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ) 
  (k : ℝ)
  (hr : r = 1)
  (hk : k = 4)
  (hA_speed : speed_A = 4 * speed_B)
  (B_starts_at_center : ∃ (B : ℝ), B = 0):
  ¬(∃ (escape_strategy : ℝ → ℝ), escape_strategy 0 = 0 → escape_strategy r = 1) :=
sorry

end NUMINAMATH_GPT_victor_cannot_escape_k4_l679_67965


namespace NUMINAMATH_GPT_min_sum_of_inverses_l679_67936

theorem min_sum_of_inverses 
  (x y z p q r : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) 
  (h_sum : x + y + z + p + q + r = 10) :
  (1 / x + 9 / y + 4 / z + 25 / p + 16 / q + 36 / r) = 44.1 :=
sorry

end NUMINAMATH_GPT_min_sum_of_inverses_l679_67936


namespace NUMINAMATH_GPT_pie_contest_l679_67943

def first_student_pie := 7 / 6
def second_student_pie := 4 / 3
def third_student_eats_from_first := 1 / 2
def third_student_eats_from_second := 1 / 3

theorem pie_contest :
  (first_student_pie - third_student_eats_from_first = 2 / 3) ∧
  (second_student_pie - third_student_eats_from_second = 1) ∧
  (third_student_eats_from_first + third_student_eats_from_second = 5 / 6) :=
by
  sorry

end NUMINAMATH_GPT_pie_contest_l679_67943


namespace NUMINAMATH_GPT_sum_units_tens_not_divisible_by_4_l679_67971

theorem sum_units_tens_not_divisible_by_4 :
  ∃ (n : ℕ), (n = 3674 ∨ n = 3684 ∨ n = 3694 ∨ n = 3704 ∨ n = 3714 ∨ n = 3722) ∧
  (¬ (∃ k, (n % 100) = 4 * k)) ∧
  ((n % 10) + (n / 10 % 10) = 11) :=
sorry

end NUMINAMATH_GPT_sum_units_tens_not_divisible_by_4_l679_67971


namespace NUMINAMATH_GPT_maximum_value_of_f_on_interval_l679_67988

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x + Real.sin x

theorem maximum_value_of_f_on_interval :
  ∃ M, M = Real.pi ∧ ∀ x ∈ Set.Icc 0 (2 * Real.pi), f x ≤ M :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_of_f_on_interval_l679_67988


namespace NUMINAMATH_GPT_right_triangle_leg_length_l679_67904

theorem right_triangle_leg_length
  (A : ℝ)
  (b h : ℝ)
  (hA : A = 800)
  (hb : b = 40)
  (h_area : A = (1 / 2) * b * h) :
  h = 40 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_leg_length_l679_67904


namespace NUMINAMATH_GPT_gcd_pow_minus_one_l679_67980

theorem gcd_pow_minus_one {m n a : ℕ} (hm : 0 < m) (hn : 0 < n) (ha : 2 ≤ a) : 
  Nat.gcd (a^n - 1) (a^m - 1) = a^(Nat.gcd m n) - 1 := 
sorry

end NUMINAMATH_GPT_gcd_pow_minus_one_l679_67980


namespace NUMINAMATH_GPT_no_perpendicular_hatching_other_than_cube_l679_67901

def is_convex_polyhedron (P : Polyhedron) : Prop :=
  -- Definition of a convex polyhedron
  sorry

def number_of_faces (P : Polyhedron) : ℕ :=
  -- Function returning the number of faces of polyhedron P
  sorry

def hatching_perpendicular (P : Polyhedron) : Prop :=
  -- Definition that checks if the hatching on adjacent faces of P is perpendicular
  sorry

theorem no_perpendicular_hatching_other_than_cube :
  ∀ (P : Polyhedron), is_convex_polyhedron P ∧ number_of_faces P ≠ 6 → ¬hatching_perpendicular P :=
by
  sorry

end NUMINAMATH_GPT_no_perpendicular_hatching_other_than_cube_l679_67901


namespace NUMINAMATH_GPT_equation_of_tangent_line_l679_67987

noncomputable def f (m x : ℝ) := m * Real.exp x - x - 1

def passes_through_P (m : ℝ) : Prop :=
  f m 0 = 1

theorem equation_of_tangent_line (m : ℝ) (h : passes_through_P m) :
  (f m) 0 = 1 → (2 - 1 = 1) ∧ ((y - 1 = x) → (x - y + 1 = 0)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_equation_of_tangent_line_l679_67987


namespace NUMINAMATH_GPT_three_pumps_drain_time_l679_67990

-- Definitions of the rates of each pump
def rate1 := 1 / 9
def rate2 := 1 / 6
def rate3 := 1 / 12

-- Combined rate of all three pumps working together
def combined_rate := rate1 + rate2 + rate3

-- Time to drain the lake with all three pumps working together
def time_to_drain := 1 / combined_rate

-- Theorem: The time it takes for three pumps working together to drain the lake is 36/13 hours
theorem three_pumps_drain_time : time_to_drain = 36 / 13 := by
  sorry

end NUMINAMATH_GPT_three_pumps_drain_time_l679_67990


namespace NUMINAMATH_GPT_remi_spilled_second_time_l679_67995

-- Defining the conditions from the problem
def bottle_capacity : ℕ := 20
def daily_refills : ℕ := 3
def total_days : ℕ := 7
def total_water_consumed : ℕ := 407
def first_spill : ℕ := 5

-- Using the conditions to define the total amount of water that Remi would have drunk without spilling.
def no_spill_total : ℕ := bottle_capacity * daily_refills * total_days

-- Defining the second spill
def second_spill : ℕ := no_spill_total - first_spill - total_water_consumed

-- Stating the theorem that we need to prove
theorem remi_spilled_second_time : second_spill = 8 :=
by
  sorry

end NUMINAMATH_GPT_remi_spilled_second_time_l679_67995


namespace NUMINAMATH_GPT_intercept_sum_l679_67923

theorem intercept_sum (x0 y0 : ℕ) (h1 : x0 < 17) (h2 : y0 < 17)
  (hx : 7 * x0 ≡ 2 [MOD 17]) (hy : 3 * y0 ≡ 15 [MOD 17]) : x0 + y0 = 17 :=
sorry

end NUMINAMATH_GPT_intercept_sum_l679_67923


namespace NUMINAMATH_GPT_value_of_abc_l679_67914

-- Conditions
def cond1 (a b : ℤ) : Prop := ∀ x : ℤ, x^2 + 19 * x + 88 = (x + a) * (x + b)
def cond2 (b c : ℤ) : Prop := ∀ x : ℤ, x^2 - 23 * x + 132 = (x - b) * (x - c)

-- Theorem statement
theorem value_of_abc (a b c : ℤ) (h₁ : cond1 a b) (h₂ : cond2 b c) : a + b + c = 31 :=
sorry

end NUMINAMATH_GPT_value_of_abc_l679_67914


namespace NUMINAMATH_GPT_smallest_positive_integer_l679_67918

theorem smallest_positive_integer :
  ∃ x : ℕ, x > 0 ∧ x % 5 = 2 ∧ x % 3 = 1 ∧ x % 7 = 3 ∧ ∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 3 = 1 ∧ y % 7 = 3 → x ≤ y :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_integer_l679_67918


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l679_67925

variable {a : ℕ → ℝ}

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n+1) = a n + d

-- Definition of the fourth term condition
def a4_condition (a : ℕ → ℝ) : Prop :=
  a 4 = 2 - a 3

-- Definition of the sum of the first 6 terms
def sum_first_six_terms (a : ℕ → ℝ) : ℝ :=
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

-- Proof statement
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a →
  a4_condition a →
  sum_first_six_terms a = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l679_67925


namespace NUMINAMATH_GPT_samson_mother_age_l679_67977

variable (S M : ℕ)
variable (x : ℕ)

def problem_statement : Prop :=
  S = 6 ∧
  S - x = 2 ∧
  M - x = 4 * 2 →
  M = 16

theorem samson_mother_age (S M x : ℕ) (h : problem_statement S M x) : M = 16 :=
by
  sorry

end NUMINAMATH_GPT_samson_mother_age_l679_67977


namespace NUMINAMATH_GPT_bike_cost_l679_67996

-- Defining the problem conditions
def jars : ℕ := 5
def quarters_per_jar : ℕ := 160
def leftover : ℚ := 20  -- 20 dollars left over
def quarter_value : ℚ := 0.25

-- Define the total quarters Jenn has
def total_quarters := jars * quarters_per_jar

-- Define the total amount of money from quarters
def total_money_quarters := total_quarters * quarter_value

-- Prove that the cost of the bike is $200
theorem bike_cost : total_money_quarters + leftover - 20 = 200 :=
sorry

end NUMINAMATH_GPT_bike_cost_l679_67996


namespace NUMINAMATH_GPT_ellipse_equation_is_standard_form_l679_67939

theorem ellipse_equation_is_standard_form (m n : ℝ) (h_m_pos : m > 0) (h_n_pos : n > 0) (h_mn_neq : m ≠ n) :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ (∀ x y : ℝ, mx^2 + ny^2 = 1 ↔ x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_is_standard_form_l679_67939


namespace NUMINAMATH_GPT_proportion_of_white_pieces_l679_67989

theorem proportion_of_white_pieces (x : ℕ) (h1 : 0 < x) :
  let total_pieces := 3 * x
  let white_pieces := x + (1 - (5 / 9)) * x
  (white_pieces / total_pieces) = (13 / 27) :=
by
  sorry

end NUMINAMATH_GPT_proportion_of_white_pieces_l679_67989


namespace NUMINAMATH_GPT_negation_equiv_exists_l679_67944

theorem negation_equiv_exists : 
  ¬ (∀ x : ℝ, x^2 + 1 > 0) ↔ ∃ x_0 : ℝ, x_0^2 + 1 ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_negation_equiv_exists_l679_67944


namespace NUMINAMATH_GPT_area_of_triangle_ABC_l679_67966

theorem area_of_triangle_ABC :
  let A'B' := 4
  let B'C' := 3
  let angle_A'B'C' := 60
  let area_A'B'C' := (1 / 2) * A'B' * B'C' * Real.sin (angle_A'B'C' * Real.pi / 180)
  let ratio := 2 * Real.sqrt 2
  let area_ABC := ratio * area_A'B'C'
  area_ABC = 6 * Real.sqrt 6 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_ABC_l679_67966


namespace NUMINAMATH_GPT_oldest_sister_clothing_l679_67900

-- Define the initial conditions
def Nicole_initial := 10
def First_sister := Nicole_initial / 2
def Next_sister := Nicole_initial + 2
def Nicole_end := 36

-- Define the proof statement
theorem oldest_sister_clothing : 
    (First_sister + Next_sister + Nicole_initial + x = Nicole_end) → x = 9 :=
by
  sorry

end NUMINAMATH_GPT_oldest_sister_clothing_l679_67900


namespace NUMINAMATH_GPT_solve_for_2023_minus_a_minus_2b_l679_67974

theorem solve_for_2023_minus_a_minus_2b (a b : ℝ) (h : 1^2 + a*1 + 2*b = 0) : 2023 - a - 2*b = 2024 := 
by sorry

end NUMINAMATH_GPT_solve_for_2023_minus_a_minus_2b_l679_67974


namespace NUMINAMATH_GPT_union_of_sets_l679_67947

def A : Set ℝ := {x | x^2 + x - 2 < 0}
def B : Set ℝ := {x | x > 0}
def C : Set ℝ := {x | x > -2}

theorem union_of_sets (A B : Set ℝ) : (A ∪ B) = C :=
  sorry

end NUMINAMATH_GPT_union_of_sets_l679_67947


namespace NUMINAMATH_GPT_rationalize_denominator_l679_67969

theorem rationalize_denominator :
  (7 / (Real.sqrt 175 - Real.sqrt 75)) = (7 * (Real.sqrt 7 + Real.sqrt 3) / 20) :=
by
  have h1 : Real.sqrt 175 = 5 * Real.sqrt 7 := sorry
  have h2 : Real.sqrt 75 = 5 * Real.sqrt 3 := sorry
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l679_67969


namespace NUMINAMATH_GPT_cinco_de_mayo_day_days_between_feb_14_and_may_5_l679_67920

theorem cinco_de_mayo_day {
  feb_14_is_tuesday : ∃ n : ℕ, n % 7 = 2
}: 
∃ n : ℕ, n % 7 = 5 := sorry

theorem days_between_feb_14_and_may_5: 
  ∃ d : ℕ, 
  d = 81 := sorry

end NUMINAMATH_GPT_cinco_de_mayo_day_days_between_feb_14_and_may_5_l679_67920


namespace NUMINAMATH_GPT_pens_bought_is_17_l679_67916

def number_of_pens_bought (C S : ℝ) (bought_pens : ℝ) : Prop :=
  (bought_pens * C = 12 * S) ∧ (0.4 = (S - C) / C)

theorem pens_bought_is_17 (C S : ℝ) (bought_pens : ℝ) 
  (h1 : bought_pens * C = 12 * S)
  (h2 : 0.4 = (S - C) / C) :
  bought_pens = 17 :=
sorry

end NUMINAMATH_GPT_pens_bought_is_17_l679_67916


namespace NUMINAMATH_GPT_find_2a_plus_b_l679_67981

noncomputable def f (a b x : ℝ) : ℝ := a * x - b
noncomputable def g (x : ℝ) : ℝ := -4 * x + 6
noncomputable def h (a b x : ℝ) : ℝ := f a b (g x)
noncomputable def h_inv (x : ℝ) : ℝ := x + 9

theorem find_2a_plus_b (a b : ℝ) (h_inv_eq: ∀ x : ℝ, h a b (h_inv x) = x) : 2 * a + b = 7 :=
sorry

end NUMINAMATH_GPT_find_2a_plus_b_l679_67981


namespace NUMINAMATH_GPT_find_y_value_l679_67937

theorem find_y_value (a y : ℕ) (h1 : (15^2) * y^3 / 256 = a) (h2 : a = 450) : y = 8 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_value_l679_67937


namespace NUMINAMATH_GPT_other_root_is_minus_two_l679_67997

theorem other_root_is_minus_two (b : ℝ) (h : 1^2 + b * 1 - 2 = 0) : 
  ∃ (x : ℝ), x = -2 ∧ x^2 + b * x - 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_other_root_is_minus_two_l679_67997


namespace NUMINAMATH_GPT_ab_is_zero_l679_67933

theorem ab_is_zero (a b : ℝ) (h₁ : a + b = 5) (h₂ : a^3 + b^3 = 125) : a * b = 0 :=
by
  -- Begin proof here
  sorry

end NUMINAMATH_GPT_ab_is_zero_l679_67933


namespace NUMINAMATH_GPT_investment_amount_l679_67983

-- Conditions and given problem rewrite in Lean 4
theorem investment_amount (P y : ℝ) (h1 : P * y * 2 / 100 = 500) (h2 : P * (1 + y / 100) ^ 2 - P = 512.50) : P = 5000 :=
sorry

end NUMINAMATH_GPT_investment_amount_l679_67983


namespace NUMINAMATH_GPT_jackson_collection_goal_l679_67967

theorem jackson_collection_goal 
  (days_in_week : ℕ)
  (goal : ℕ)
  (earned_mon : ℕ)
  (earned_tue : ℕ)
  (avg_collect_per_4house : ℕ)
  (remaining_days : ℕ)
  (remaining_goal : ℕ)
  (daily_target : ℕ)
  (collect_per_house : ℚ)
  :
  days_in_week = 5 →
  goal = 1000 →
  earned_mon = 300 →
  earned_tue = 40 →
  avg_collect_per_4house = 10 →
  remaining_goal = goal - earned_mon - earned_tue →
  remaining_days = days_in_week - 2 →
  daily_target = remaining_goal / remaining_days →
  collect_per_house = avg_collect_per_4house / 4 →
  (daily_target : ℚ) / collect_per_house = 88 := 
by sorry

end NUMINAMATH_GPT_jackson_collection_goal_l679_67967


namespace NUMINAMATH_GPT_cab_driver_income_l679_67953

theorem cab_driver_income (x2 : ℕ) :
  (600 + x2 + 450 + 400 + 800) / 5 = 500 → x2 = 250 :=
by
  sorry

end NUMINAMATH_GPT_cab_driver_income_l679_67953


namespace NUMINAMATH_GPT_barry_wand_trick_l679_67913

theorem barry_wand_trick (n : ℕ) (h : (n + 3 : ℝ) / 3 = 50) : n = 147 := by
  sorry

end NUMINAMATH_GPT_barry_wand_trick_l679_67913


namespace NUMINAMATH_GPT_machines_working_together_l679_67955

theorem machines_working_together (x : ℝ) :
  (∀ P Q R : ℝ, P = x + 4 ∧ Q = x + 2 ∧ R = 2 * x + 2 ∧ (1 / P + 1 / Q + 1 / R = 1 / x)) ↔ (x = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_machines_working_together_l679_67955


namespace NUMINAMATH_GPT_original_number_of_men_l679_67972

theorem original_number_of_men (x : ℕ) 
  (h1 : 17 * x = 21 * (x - 8)) : x = 42 := 
by {
   -- proof steps can be filled in here
   sorry
}

end NUMINAMATH_GPT_original_number_of_men_l679_67972


namespace NUMINAMATH_GPT_solve_for_x_l679_67929

theorem solve_for_x (x : ℝ) : (5 : ℝ)^(x + 6) = (625 : ℝ)^x → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l679_67929


namespace NUMINAMATH_GPT_quadratic_equation_solution_l679_67908

-- We want to prove that for the conditions given, the only possible value of m is 3
theorem quadratic_equation_solution (m : ℤ) (h1 : m^2 - 7 = 2) (h2 : m + 3 ≠ 0) : m = 3 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l679_67908


namespace NUMINAMATH_GPT_purely_periodic_period_le_T_l679_67902

theorem purely_periodic_period_le_T {a b : ℚ} (T : ℕ) 
  (ha : ∃ m, a = m / (10^T - 1)) 
  (hb : ∃ n, b = n / (10^T - 1)) :
  (∃ T₁, T₁ ≤ T ∧ ∃ p, a = p / (10^T₁ - 1)) ∧ 
  (∃ T₂, T₂ ≤ T ∧ ∃ q, b = q / (10^T₂ - 1)) := 
sorry

end NUMINAMATH_GPT_purely_periodic_period_le_T_l679_67902


namespace NUMINAMATH_GPT_brick_width_l679_67985

variable (w : ℝ)

theorem brick_width :
  ∃ (w : ℝ), 2 * (10 * w + 10 * 3 + 3 * w) = 164 → w = 4 :=
by
  sorry

end NUMINAMATH_GPT_brick_width_l679_67985
