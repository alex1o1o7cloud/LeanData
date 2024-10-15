import Mathlib

namespace NUMINAMATH_GPT_older_brother_catches_younger_brother_l2378_237811

theorem older_brother_catches_younger_brother
  (y_time_reach_school o_time_reach_school : ℕ) 
  (delay : ℕ) 
  (catchup_time : ℕ) 
  (h1 : y_time_reach_school = 25) 
  (h2 : o_time_reach_school = 15) 
  (h3 : delay = 8) 
  (h4 : catchup_time = 17):
  catchup_time = delay + ((8 * y_time_reach_school) / (o_time_reach_school - y_time_reach_school) * (y_time_reach_school / 25)) :=
by
  sorry

end NUMINAMATH_GPT_older_brother_catches_younger_brother_l2378_237811


namespace NUMINAMATH_GPT_measure_ADC_l2378_237867

-- Definitions
def angle_measures (x y ADC : ℝ) : Prop :=
  2 * x + 60 + 2 * y = 180 ∧ x + y = 60 ∧ x + y + ADC = 180

-- Goal
theorem measure_ADC (x y ADC : ℝ) (h : angle_measures x y ADC) : ADC = 120 :=
by {
  -- Solution could go here, skipped for brevity
  sorry
}

end NUMINAMATH_GPT_measure_ADC_l2378_237867


namespace NUMINAMATH_GPT_expected_number_of_shots_l2378_237814

def probability_hit : ℝ := 0.8
def probability_miss := 1 - probability_hit
def max_shots : ℕ := 3

theorem expected_number_of_shots : ∃ ξ : ℝ, ξ = 1.24 := by
  sorry

end NUMINAMATH_GPT_expected_number_of_shots_l2378_237814


namespace NUMINAMATH_GPT_max_distinct_integer_solutions_le_2_l2378_237806

def f (a b c x : ℝ) : ℝ := a*x^2 + b*x + c

theorem max_distinct_integer_solutions_le_2 
  (a b c : ℝ) (h₀ : a > 100) :
  ∀ (x : ℤ), |f a b c (x : ℝ)| ≤ 50 → 
  ∃ (x₁ x₂ : ℤ), x = x₁ ∨ x = x₂ :=
by
  sorry

end NUMINAMATH_GPT_max_distinct_integer_solutions_le_2_l2378_237806


namespace NUMINAMATH_GPT_tan_alpha_eq_neg_five_twelfths_l2378_237864

-- Define the angle α and the given conditions
variables (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π)

-- The goal is to prove that tan α = -5 / 12
theorem tan_alpha_eq_neg_five_twelfths (α : ℝ) (h1 : Real.sin α = 5 / 13) (h2 : π / 2 < α ∧ α < π) :
  Real.tan α = -5 / 12 :=
sorry

end NUMINAMATH_GPT_tan_alpha_eq_neg_five_twelfths_l2378_237864


namespace NUMINAMATH_GPT_find_m_l2378_237817

variables {a1 a2 b1 b2 c1 c2 : ℝ} {m : ℝ}
def vectorA := (3, -2 * m)
def vectorB := (m - 1, 2)
def vectorC := (-2, 1)
def vectorAC := (5, -2 * m - 1)

theorem find_m (h : (5 * (m - 1) + (-2 * m - 1) * 2) = 0) : 
  m = 7 := 
  sorry

end NUMINAMATH_GPT_find_m_l2378_237817


namespace NUMINAMATH_GPT_certain_fraction_is_half_l2378_237801

theorem certain_fraction_is_half (n : ℕ) (fraction : ℚ) (h : (37 + 1/2) / fraction = 75) : fraction = 1/2 :=
by
    sorry

end NUMINAMATH_GPT_certain_fraction_is_half_l2378_237801


namespace NUMINAMATH_GPT_original_price_of_sarees_l2378_237889
open Real

theorem original_price_of_sarees (P : ℝ) (h : 0.70 * 0.80 * P = 224) : P = 400 :=
sorry

end NUMINAMATH_GPT_original_price_of_sarees_l2378_237889


namespace NUMINAMATH_GPT_hyperbola_center_coordinates_l2378_237865

theorem hyperbola_center_coordinates :
  ∃ (h k : ℝ), 
  (∀ x y : ℝ, 
    ((4 * y - 6) ^ 2 / 36 - (5 * x - 3) ^ 2 / 49 = -1) ↔
    ((x - h) ^ 2 / ((7 / 5) ^ 2) - (y - k) ^ 2 / ((3 / 2) ^ 2) = 1)) ∧
  h = 3 / 5 ∧ k = 3 / 2 :=
by sorry

end NUMINAMATH_GPT_hyperbola_center_coordinates_l2378_237865


namespace NUMINAMATH_GPT_vertex_parabola_is_parabola_l2378_237846

variables {a c : ℝ} (h_a : 0 < a) (h_c : 0 < c)

theorem vertex_parabola_is_parabola :
  ∀ (x y : ℝ), (∃ b : ℝ, x = -b / (2 * a) ∧ y = a * (-b / (2 * a)) ^ 2 + b * (-b / (2 * a)) + c) ↔ y = -a * x ^ 2 + c :=
by sorry

end NUMINAMATH_GPT_vertex_parabola_is_parabola_l2378_237846


namespace NUMINAMATH_GPT_shorter_trisector_length_eq_l2378_237895

theorem shorter_trisector_length_eq :
  ∀ (DE EF DF FG : ℝ), DE = 6 → EF = 8 → DF = Real.sqrt (DE^2 + EF^2) → 
  FG = 2 * (24 / (3 + 4 * Real.sqrt 3)) → 
  FG = (192 * Real.sqrt 3 - 144) / 39 :=
by
  intros
  sorry

end NUMINAMATH_GPT_shorter_trisector_length_eq_l2378_237895


namespace NUMINAMATH_GPT_tourism_revenue_scientific_notation_l2378_237874

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end NUMINAMATH_GPT_tourism_revenue_scientific_notation_l2378_237874


namespace NUMINAMATH_GPT_yellow_surface_area_fraction_minimal_l2378_237891

theorem yellow_surface_area_fraction_minimal 
  (total_cubes : ℕ)
  (edge_length : ℕ)
  (yellow_cubes : ℕ)
  (blue_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (yellow_fraction : ℚ) :
  total_cubes = 64 ∧
  edge_length = 4 ∧
  yellow_cubes = 16 ∧
  blue_cubes = 48 ∧
  total_surface_area = 6 * edge_length * edge_length ∧
  yellow_surface_area = 15 →
  yellow_fraction = (yellow_surface_area : ℚ) / total_surface_area :=
sorry

end NUMINAMATH_GPT_yellow_surface_area_fraction_minimal_l2378_237891


namespace NUMINAMATH_GPT_bill_difference_is_zero_l2378_237887

theorem bill_difference_is_zero
    (a b : ℝ)
    (h1 : 0.25 * a = 5)
    (h2 : 0.15 * b = 3) :
    a - b = 0 := 
by 
  sorry

end NUMINAMATH_GPT_bill_difference_is_zero_l2378_237887


namespace NUMINAMATH_GPT_second_runner_stop_time_l2378_237831

-- Definitions provided by the conditions
def pace_first := 8 -- pace of the first runner in minutes per mile
def pace_second := 7 -- pace of the second runner in minutes per mile
def time_elapsed := 56 -- time elapsed in minutes before the second runner stops
def distance_first := time_elapsed / pace_first -- distance covered by the first runner in miles
def distance_second := time_elapsed / pace_second -- distance covered by the second runner in miles
def distance_gap := distance_second - distance_first -- gap between the runners in miles

-- Statement of the proof problem
theorem second_runner_stop_time :
  8 = distance_gap * pace_first :=
by
sorry

end NUMINAMATH_GPT_second_runner_stop_time_l2378_237831


namespace NUMINAMATH_GPT_quadratic_solution_l2378_237800

-- Definitions come from the conditions of the problem
def satisfies_equation (y : ℝ) : Prop := 6 * y^2 + 2 = 4 * y + 12

-- Statement of the proof
theorem quadratic_solution (y : ℝ) (hy : satisfies_equation y) : (12 * y - 2)^2 = 324 ∨ (12 * y - 2)^2 = 196 := 
sorry

end NUMINAMATH_GPT_quadratic_solution_l2378_237800


namespace NUMINAMATH_GPT_probability_of_getting_specific_clothing_combination_l2378_237820

def total_articles := 21

def ways_to_choose_4_articles : ℕ := Nat.choose total_articles 4

def ways_to_choose_2_shirts_from_6 : ℕ := Nat.choose 6 2

def ways_to_choose_1_pair_of_shorts_from_7 : ℕ := Nat.choose 7 1

def ways_to_choose_1_pair_of_socks_from_8 : ℕ := Nat.choose 8 1

def favorable_outcomes := 
  ways_to_choose_2_shirts_from_6 * 
  ways_to_choose_1_pair_of_shorts_from_7 * 
  ways_to_choose_1_pair_of_socks_from_8

def probability := (favorable_outcomes : ℚ) / (ways_to_choose_4_articles : ℚ)

theorem probability_of_getting_specific_clothing_combination : 
  probability = 56 / 399 := by
  sorry

end NUMINAMATH_GPT_probability_of_getting_specific_clothing_combination_l2378_237820


namespace NUMINAMATH_GPT_evaluate_expression_l2378_237823

def x : ℝ := 2
def y : ℝ := 4

theorem evaluate_expression : y * (y - 2 * x) = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2378_237823


namespace NUMINAMATH_GPT_investment_value_l2378_237809

noncomputable def compound_interest (P : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  P * (1 + r)^n

theorem investment_value :
  ∀ (P : ℕ) (r : ℚ) (n : ℕ),
  P = 8000 →
  r = 0.05 →
  n = 3 →
  compound_interest P r n = 9250 := by
    intros P r n hP hr hn
    unfold compound_interest
    -- calculation steps would be here
    sorry

end NUMINAMATH_GPT_investment_value_l2378_237809


namespace NUMINAMATH_GPT_remainder_of_a55_l2378_237810

def concatenate_integers (n : ℕ) : ℕ :=
  -- Function to concatenate integers from 1 to n into a single number.
  -- This is a placeholder, actual implementation may vary.
  sorry

theorem remainder_of_a55 (n : ℕ) (hn : n = 55) :
  concatenate_integers n % 55 = 0 := by
  -- Proof is omitted, provided as a guideline.
  sorry

end NUMINAMATH_GPT_remainder_of_a55_l2378_237810


namespace NUMINAMATH_GPT_Mairead_triathlon_l2378_237885

noncomputable def convert_km_to_miles (km: Float) : Float :=
  0.621371 * km

noncomputable def convert_yards_to_miles (yd: Float) : Float :=
  0.000568182 * yd

noncomputable def convert_feet_to_miles (ft: Float) : Float :=
  0.000189394 * ft

noncomputable def total_distance_in_miles := 
  let run_distance_km := 40.0
  let run_distance_miles := convert_km_to_miles run_distance_km
  let walk_distance_miles := 3.0/5.0 * run_distance_miles
  let jog_distance_yd := 5.0 * (walk_distance_miles * 1760.0)
  let jog_distance_miles := convert_yards_to_miles jog_distance_yd
  let bike_distance_ft := 3.0 * (jog_distance_miles * 5280.0)
  let bike_distance_miles := convert_feet_to_miles bike_distance_ft
  let swim_distance_miles := 2.5
  run_distance_miles + walk_distance_miles + jog_distance_miles + bike_distance_miles + swim_distance_miles

theorem Mairead_triathlon:
  total_distance_in_miles = 340.449562 ∧
  (convert_km_to_miles 40.0) / 10.0 = 2.485484 ∧
  (3.0/5.0 * (convert_km_to_miles 40.0)) / 10.0 = 1.4912904 ∧
  (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0))) / 10.0 = 7.45454544 ∧
  (convert_feet_to_miles (3.0 * (convert_yards_to_miles (5.0 * (3.0/5.0 * (convert_km_to_miles 40.0) * 1760.0)) * 5280.0))) / 10.0 = 22.36363636 ∧
  2.5 / 10.0 = 0.25 := sorry

end NUMINAMATH_GPT_Mairead_triathlon_l2378_237885


namespace NUMINAMATH_GPT_system_of_equations_solution_l2378_237837

theorem system_of_equations_solution (x y z : ℝ) (hx : x = Real.exp (Real.log y))
(hy : y = Real.exp (Real.log z)) (hz : z = Real.exp (Real.log x)) : x = y ∧ y = z ∧ z = x ∧ x = Real.exp 1 :=
by
  sorry

end NUMINAMATH_GPT_system_of_equations_solution_l2378_237837


namespace NUMINAMATH_GPT_complete_square_variant_l2378_237844

theorem complete_square_variant (x : ℝ) :
    3 * x^2 + 4 * x + 1 = 0 → (x + 2 / 3) ^ 2 = 1 / 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_variant_l2378_237844


namespace NUMINAMATH_GPT_train_speed_in_kmh_l2378_237862

theorem train_speed_in_kmh 
  (train_length : ℕ) 
  (crossing_time : ℕ) 
  (conversion_factor : ℕ) 
  (hl : train_length = 120) 
  (ht : crossing_time = 6) 
  (hc : conversion_factor = 36) :
  train_length / crossing_time * conversion_factor / 10 = 72 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmh_l2378_237862


namespace NUMINAMATH_GPT_find_m_for_even_function_l2378_237816

def f (x : ℝ) (m : ℝ) := x^2 + (m - 1) * x + 3

theorem find_m_for_even_function : ∃ m : ℝ, (∀ x : ℝ, f (-x) m = f x m) ∧ m = 1 :=
sorry

end NUMINAMATH_GPT_find_m_for_even_function_l2378_237816


namespace NUMINAMATH_GPT_shirt_price_l2378_237834

theorem shirt_price (S : ℝ) (h : (5 * S + 5 * 3) / 2 = 10) : S = 1 :=
by
  sorry

end NUMINAMATH_GPT_shirt_price_l2378_237834


namespace NUMINAMATH_GPT_tens_digit_less_than_5_probability_l2378_237841

theorem tens_digit_less_than_5_probability 
  (n : ℕ) 
  (hn : 10000 ≤ n ∧ n ≤ 99999)
  (h_even : ∃ k, n % 10 = 2 * k ∧ k < 5) :
  (∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_tens_digit_less_than_5_probability_l2378_237841


namespace NUMINAMATH_GPT_find_x_l2378_237830

theorem find_x (x y : ℝ) (h : y ≠ -5 * x) : (x - 5) / (5 * x + y) = 0 → x = 5 := by
  sorry

end NUMINAMATH_GPT_find_x_l2378_237830


namespace NUMINAMATH_GPT_largest_sample_number_l2378_237835

theorem largest_sample_number (n : ℕ) (start interval total : ℕ) (h1 : start = 7) (h2 : interval = 25) (h3 : total = 500) (h4 : n = total / interval) : 
(start + interval * (n - 1) = 482) :=
sorry

end NUMINAMATH_GPT_largest_sample_number_l2378_237835


namespace NUMINAMATH_GPT_area_of_rhombus_l2378_237847

-- Defining conditions for the problem
def d1 : ℝ := 40   -- Length of the first diagonal in meters
def d2 : ℝ := 30   -- Length of the second diagonal in meters

-- Calculating the area of the rhombus
noncomputable def area : ℝ := (d1 * d2) / 2

-- Statement of the theorem
theorem area_of_rhombus : area = 600 := by
  sorry

end NUMINAMATH_GPT_area_of_rhombus_l2378_237847


namespace NUMINAMATH_GPT_total_people_correct_l2378_237863

-- Define the daily changes as given conditions
def daily_changes : List ℝ := [1.6, 0.8, 0.4, -0.4, -0.8, 0.2, -1.2]

-- Define the total number of people given 'a' and daily changes
def total_people (a : ℝ) : ℝ :=
  7 * a + daily_changes.sum

-- Lean statement for proving the total number of people
theorem total_people_correct (a : ℝ) : 
  total_people a = 7 * a + 13.2 :=
by
  -- This statement needs a proof, so we leave a placeholder 'sorry'
  sorry

end NUMINAMATH_GPT_total_people_correct_l2378_237863


namespace NUMINAMATH_GPT_li_to_zhang_l2378_237852

theorem li_to_zhang :
  (∀ (meter chi : ℕ), 3 * meter = chi) →
  (∀ (zhang chi : ℕ), 10 * zhang = chi) →
  (∀ (kilometer li : ℕ), 2 * li = kilometer) →
  (1 * lin = 150 * zhang) :=
by
  intro h_meter h_zhang h_kilometer
  sorry

end NUMINAMATH_GPT_li_to_zhang_l2378_237852


namespace NUMINAMATH_GPT_frisbee_sales_l2378_237812

/-- A sporting goods store sold some frisbees, with $3 and $4 price points.
The total receipts from frisbee sales were $204. The fewest number of $4 frisbees that could have been sold is 24.
Prove the total number of frisbees sold is 60. -/
theorem frisbee_sales (x y : ℕ) (h1 : 3 * x + 4 * y = 204) (h2 : 24 ≤ y) : x + y = 60 :=
by {
  -- Proof skipped
  sorry
}

end NUMINAMATH_GPT_frisbee_sales_l2378_237812


namespace NUMINAMATH_GPT_midpoint_plane_distance_l2378_237838

noncomputable def midpoint_distance (A B : ℝ) (dA dB : ℝ) : ℝ :=
  (dA + dB) / 2

theorem midpoint_plane_distance (A B : ℝ) (dA dB : ℝ) (hA : dA = 1) (hB : dB = 3) :
  midpoint_distance A B dA dB = 1 ∨ midpoint_distance A B dA dB = 2 :=
by
  sorry

end NUMINAMATH_GPT_midpoint_plane_distance_l2378_237838


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l2378_237808

theorem ratio_of_boys_to_girls (boys : ℕ) (students : ℕ) (h1 : boys = 42) (h2 : students = 48) : (boys : ℚ) / (students - boys : ℚ) = 7 / 1 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l2378_237808


namespace NUMINAMATH_GPT_part1_part2_l2378_237898

def f (x : ℝ) : ℝ := |x - 1|
def g (x : ℝ) (m : ℝ) : ℝ := -|x + 3| + m
def h (x : ℝ) : ℝ := |x - 1| + |x + 3|

theorem part1 (x : ℝ) : f x + x^2 - 1 > 0 ↔ x > 1 ∨ x < 0 :=
by
  sorry

theorem part2 (m : ℝ) : (∃ x : ℝ, f x < g x m) ↔ m > 4 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l2378_237898


namespace NUMINAMATH_GPT_heather_blocks_remaining_l2378_237873

-- Definitions of the initial amount of blocks and the amount shared
def initial_blocks : ℕ := 86
def shared_blocks : ℕ := 41

-- The statement to be proven
theorem heather_blocks_remaining : (initial_blocks - shared_blocks = 45) :=
by sorry

end NUMINAMATH_GPT_heather_blocks_remaining_l2378_237873


namespace NUMINAMATH_GPT_cubes_product_fraction_l2378_237896

theorem cubes_product_fraction :
  (4^3 * 6^3 * 8^3 * 9^3 : ℚ) / (10^3 * 12^3 * 14^3 * 15^3) = 576 / 546875 := 
sorry

end NUMINAMATH_GPT_cubes_product_fraction_l2378_237896


namespace NUMINAMATH_GPT_inequality_pos_distinct_l2378_237840

theorem inequality_pos_distinct (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
    (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
    (a + b + c) * (1/a + 1/b + 1/c) > 9 := by
  sorry

end NUMINAMATH_GPT_inequality_pos_distinct_l2378_237840


namespace NUMINAMATH_GPT_ratio_lateral_surface_area_to_surface_area_l2378_237856

theorem ratio_lateral_surface_area_to_surface_area (r : ℝ) (h : ℝ) (V_sphere V_cone A_cone A_sphere : ℝ)
    (h_eq : h = r)
    (V_sphere_eq : V_sphere = (4 / 3) * Real.pi * r^3)
    (V_cone_eq : V_cone = (1 / 3) * Real.pi * (2 * r)^2 * h)
    (V_eq : V_sphere = V_cone)
    (A_cone_eq : A_cone = 2 * Real.sqrt 5 * Real.pi * r^2)
    (A_sphere_eq : A_sphere = 4 * Real.pi * r^2) :
    A_cone / A_sphere = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_GPT_ratio_lateral_surface_area_to_surface_area_l2378_237856


namespace NUMINAMATH_GPT_tank_cost_correct_l2378_237826

noncomputable def tankPlasteringCost (l w d cost_per_m2 : ℝ) : ℝ :=
  let long_walls_area := 2 * (l * d)
  let short_walls_area := 2 * (w * d)
  let bottom_area := l * w
  let total_area := long_walls_area + short_walls_area + bottom_area
  total_area * cost_per_m2

theorem tank_cost_correct :
  tankPlasteringCost 25 12 6 0.75 = 558 := by
  sorry

end NUMINAMATH_GPT_tank_cost_correct_l2378_237826


namespace NUMINAMATH_GPT_plane_equation_l2378_237833

noncomputable def equation_of_plane (x y z : ℝ) :=
  3 * x + 2 * z - 1

theorem plane_equation :
  ∀ (x y z : ℝ), 
    (∃ (p : ℝ × ℝ × ℝ), p = (1, 2, -1) ∧ 
                         (∃ (n : ℝ × ℝ × ℝ), n = (3, 0, 2) ∧ 
                                              equation_of_plane x y z = 0)) :=
by
  -- The statement setup is done. The proof is not included as per instructions.
  sorry

end NUMINAMATH_GPT_plane_equation_l2378_237833


namespace NUMINAMATH_GPT_parallelepiped_analogy_l2378_237853

-- Define the possible plane figures
inductive PlaneFigure
| Triangle
| Trapezoid
| Parallelogram
| Rectangle

-- Define the concept of a parallelepiped
structure Parallelepiped : Type

-- The theorem asserting the parallelogram is the correct analogy
theorem parallelepiped_analogy : 
  ∀ (fig : PlaneFigure), 
    (fig = PlaneFigure.Parallelogram) ↔ 
    (fig = PlaneFigure.Parallelogram) :=
by sorry

end NUMINAMATH_GPT_parallelepiped_analogy_l2378_237853


namespace NUMINAMATH_GPT_wall_length_l2378_237879

theorem wall_length (mirror_side length width : ℝ) (h1 : mirror_side = 21) (h2 : width = 28) 
  (h3 : 2 * mirror_side^2 = width * length) : length = 31.5 := by
  sorry

end NUMINAMATH_GPT_wall_length_l2378_237879


namespace NUMINAMATH_GPT_distance_from_circumcenter_to_orthocenter_l2378_237839

variables {A B C A1 H O : Type}

-- Condition Definitions
variable (acute_triangle : Prop)
variable (is_altitude : Prop)
variable (is_orthocenter : Prop)
variable (AH_dist : ℝ := 3)
variable (A1H_dist : ℝ := 2)
variable (circum_radius : ℝ := 4)

-- Prove the distance from O to H
theorem distance_from_circumcenter_to_orthocenter
  (h1 : acute_triangle)
  (h2 : is_altitude)
  (h3 : is_orthocenter)
  (h4 : AH_dist = 3)
  (h5 : A1H_dist = 2)
  (h6 : circum_radius = 4) : 
  ∃ (d : ℝ), d = 2 := 
sorry

end NUMINAMATH_GPT_distance_from_circumcenter_to_orthocenter_l2378_237839


namespace NUMINAMATH_GPT_cost_of_each_shirt_is_8_l2378_237868

-- Define the conditions
variables (S : ℝ)
def shirts_cost := 4 * S
def pants_cost := 2 * 18
def jackets_cost := 2 * 60
def total_cost := shirts_cost S + pants_cost + jackets_cost
def carrie_pays := 94

-- The goal is to prove that S equals 8 given the conditions above
theorem cost_of_each_shirt_is_8
  (h1 : carrie_pays = total_cost S / 2) : S = 8 :=
sorry

end NUMINAMATH_GPT_cost_of_each_shirt_is_8_l2378_237868


namespace NUMINAMATH_GPT_find_y_squared_l2378_237871

theorem find_y_squared (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : 2 * x - y = 20) : y ^ 2 = 4 := 
sorry

end NUMINAMATH_GPT_find_y_squared_l2378_237871


namespace NUMINAMATH_GPT_condition_not_right_triangle_l2378_237857

theorem condition_not_right_triangle 
  (AB BC AC : ℕ) (angleA angleB angleC : ℕ)
  (h_A : AB = 3 ∧ BC = 4 ∧ AC = 5)
  (h_B : AB / BC = 3 / 4 ∧ BC / AC = 4 / 5 ∧ AB + BC > AC ∧ AB + AC > BC ∧ BC + AC > AB)
  (h_C : angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 ∧ angleA + angleB + angleC = 180)
  (h_D : angleA = 40 ∧ angleB = 50 ∧ angleA + angleB + angleC = 180) :
  angleA = 45 ∧ angleB = 60 ∧ angleC = 75 ∧ (¬ (angleA = 90 ∨ angleB = 90 ∨ angleC = 90)) :=
sorry

end NUMINAMATH_GPT_condition_not_right_triangle_l2378_237857


namespace NUMINAMATH_GPT_elliot_book_pages_l2378_237866

theorem elliot_book_pages : 
  ∀ (initial_pages read_per_day days_in_week remaining_pages total_pages: ℕ), 
    initial_pages = 149 → 
    read_per_day = 20 → 
    days_in_week = 7 → 
    remaining_pages = 92 → 
    total_pages = initial_pages + (read_per_day * days_in_week) + remaining_pages → 
    total_pages = 381 :=
by
  intros initial_pages read_per_day days_in_week remaining_pages total_pages
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  simp at h5
  assumption

end NUMINAMATH_GPT_elliot_book_pages_l2378_237866


namespace NUMINAMATH_GPT_inequality_proof_l2378_237832

theorem inequality_proof (x y : ℝ) (h1 : x^2 ≥ y) (h2 : y^2 ≥ x) : 
  (x / (y^2 + 1) + y / (x^2 + 1) ≤ 1) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2378_237832


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2378_237858

theorem minimum_value_of_expression {x y : ℝ} (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : x + 2 * y = 1) : 
  ∃ m : ℝ, m = 0.75 ∧ ∀ z : ℝ, (∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ x + 2 * y = 1 ∧ z = 2 * x + 3 * y ^ 2) → z ≥ m :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2378_237858


namespace NUMINAMATH_GPT_mass_percentage_O_is_26_2_l2378_237884

noncomputable def mass_percentage_O_in_Benzoic_acid : ℝ :=
  let molar_mass_C := 12.01
  let molar_mass_H := 1.01
  let molar_mass_O := 16.00
  let molar_mass_Benzoic_acid := (7 * molar_mass_C) + (6 * molar_mass_H) + (2 * molar_mass_O)
  let mass_O_in_Benzoic_acid := 2 * molar_mass_O
  (mass_O_in_Benzoic_acid / molar_mass_Benzoic_acid) * 100

theorem mass_percentage_O_is_26_2 :
  mass_percentage_O_in_Benzoic_acid = 26.2 := by
  sorry

end NUMINAMATH_GPT_mass_percentage_O_is_26_2_l2378_237884


namespace NUMINAMATH_GPT_probability_of_same_color_pairs_left_right_l2378_237819

-- Define the counts of different pairs
def total_pairs := 15
def black_pairs := 8
def red_pairs := 4
def white_pairs := 3

-- Define the total number of shoes
def total_shoes := 30

-- Define the total ways to choose any 2 shoes out of total_shoes
def total_ways := Nat.choose total_shoes 2

-- Define the ways to choose one left and one right for each color
def black_ways := black_pairs * black_pairs
def red_ways := red_pairs * red_pairs
def white_ways := white_pairs * white_pairs

-- Define the total favorable outcomes for same color pairs
def total_favorable := black_ways + red_ways + white_ways

-- Define the probability
def probability := (total_favorable, total_ways)

-- Statement to prove
theorem probability_of_same_color_pairs_left_right :
  probability = (89, 435) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_same_color_pairs_left_right_l2378_237819


namespace NUMINAMATH_GPT_at_least_30_cents_probability_l2378_237872

theorem at_least_30_cents_probability :
  let penny := 1
  let nickel := 5
  let dime := 10
  let quarter := 25
  let half_dollar := 50
  let all_possible_outcomes := 2^5
  let successful_outcomes := 
    -- Half-dollar and quarter heads: 2^3 = 8 combinations
    2^3 + 
    -- Quarter heads and half-dollar tails (nickel and dime heads): 2 combinations
    2^1 + 
    -- Quarter tails and half-dollar heads: 2^3 = 8 combinations
    2^3
  let probability := successful_outcomes / all_possible_outcomes
  probability = 9 / 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_at_least_30_cents_probability_l2378_237872


namespace NUMINAMATH_GPT_red_paint_cans_l2378_237807

theorem red_paint_cans (total_cans : ℕ) (ratio_red_blue : ℕ) (ratio_blue : ℕ) (h_ratio : ratio_red_blue = 4) (h_blue : ratio_blue = 1) (h_total_cans : total_cans = 50) : 
  (total_cans * ratio_red_blue) / (ratio_red_blue + ratio_blue) = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end NUMINAMATH_GPT_red_paint_cans_l2378_237807


namespace NUMINAMATH_GPT_number_of_children_l2378_237854

theorem number_of_children 
  (A C : ℕ) 
  (h1 : A + C = 201) 
  (h2 : 8 * A + 4 * C = 964) : 
  C = 161 := 
sorry

end NUMINAMATH_GPT_number_of_children_l2378_237854


namespace NUMINAMATH_GPT_solution_correct_l2378_237818

variable (a b c d : ℝ)

theorem solution_correct (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end NUMINAMATH_GPT_solution_correct_l2378_237818


namespace NUMINAMATH_GPT_interval_monotonic_increase_max_min_values_range_of_m_l2378_237877

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin x, -Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.cos x, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1/2

-- The interval of monotonic increase for f(x)
theorem interval_monotonic_increase :
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} = 
  {x : ℝ | ∃ k : ℤ, - (π / 6) + k * π ≤ x ∧ x ≤ π / 3 + k * π} := 
by sorry

-- Maximum and minimum values of f(x) when x ∈ [π/4, π/2]
theorem max_min_values (x : ℝ) (h : x ∈ Set.Icc (π / 4) (π / 2)) :
  (f x ≤ 0 ∧ (f x = 0 ↔ x = π / 3)) ∧ (f x ≥ -1/2 ∧ (f x = -1/2 ↔ x = π / 2)) :=
by sorry

-- Range of m for the inequality |f(x) - m| < 1 when x ∈ [π/4, π/2]
theorem range_of_m (m : ℝ) (h : ∀ x ∈ Set.Icc (π / 4) (π / 2), |f x - m| < 1) :
  m ∈ Set.Ioo (-1) (1/2) :=
by sorry

end NUMINAMATH_GPT_interval_monotonic_increase_max_min_values_range_of_m_l2378_237877


namespace NUMINAMATH_GPT_cube_surface_area_l2378_237821

theorem cube_surface_area (edge_length : ℝ) (h : edge_length = 20) : 6 * (edge_length * edge_length) = 2400 := by
  -- We state our theorem and assumptions here
  sorry

end NUMINAMATH_GPT_cube_surface_area_l2378_237821


namespace NUMINAMATH_GPT_pairs_satisfied_condition_l2378_237880

def set_A : Set ℕ := {1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 20, 22, 30, 33, 44, 55, 60, 66, 110, 132, 165, 220, 330, 660}
def set_B : Set ℕ := {1, 2, 3, 4, 6, 8, 9, 12, 18, 24, 36, 72}

def is_valid_pair (a b : ℕ) := a ∈ set_A ∧ b ∈ set_B ∧ (a - b = 4)

def valid_pairs : Set (ℕ × ℕ) := 
  {(6, 2), (10, 6), (12, 8), (22, 18)}

theorem pairs_satisfied_condition :
  { (a, b) | is_valid_pair a b } = valid_pairs := 
sorry

end NUMINAMATH_GPT_pairs_satisfied_condition_l2378_237880


namespace NUMINAMATH_GPT_product_of_solutions_abs_eq_l2378_237888

theorem product_of_solutions_abs_eq (x : ℝ) :
  (∃ x1 x2 : ℝ, |6 * x1 + 2| + 5 = 47 ∧ |6 * x2 + 2| + 5 = 47 ∧ x ≠ x1 ∧ x ≠ x2 ∧ x1 * x2 = -440 / 9) :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_abs_eq_l2378_237888


namespace NUMINAMATH_GPT_correct_calculation_l2378_237845

theorem correct_calculation (a : ℝ) :
  (¬ (a^2 + a^2 = a^4)) ∧ (¬ (a^2 * a^3 = a^6)) ∧ (¬ ((a + 1)^2 = a^2 + 1)) ∧ ((-a^2)^2 = a^4) :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2378_237845


namespace NUMINAMATH_GPT_quadratic_roots_l2378_237894

theorem quadratic_roots (a b k : ℝ) (h₁ : a + b = -2) (h₂ : a * b = k / 3)
    (h₃ : |a - b| = 1/2 * (a^2 + b^2)) : k = 0 ∨ k = 6 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_l2378_237894


namespace NUMINAMATH_GPT_red_balls_number_l2378_237882

namespace BallDrawing

variable (x : ℕ) -- define x as the number of red balls

noncomputable def total_balls : ℕ := x + 4
noncomputable def yellow_ball_probability : ℚ := 4 / total_balls x

theorem red_balls_number : yellow_ball_probability x = 0.2 → x = 16 :=
by
  unfold yellow_ball_probability
  sorry

end BallDrawing

end NUMINAMATH_GPT_red_balls_number_l2378_237882


namespace NUMINAMATH_GPT_student_count_l2378_237827

theorem student_count 
( M S N : ℕ ) 
(h1 : N - M = 10) 
(h2 : N - S = 15) 
(h3 : N - (M + S - 7) = 2) : 
N = 34 :=
by
  sorry

end NUMINAMATH_GPT_student_count_l2378_237827


namespace NUMINAMATH_GPT_prob_CD_l2378_237886

variable (P : String → ℚ)
variable (x : ℚ)

axiom probA : P "A" = 1 / 3
axiom probB : P "B" = 1 / 4
axiom probC : P "C" = 2 * x
axiom probD : P "D" = x
axiom sumProb : P "A" + P "B" + P "C" + P "D" = 1

theorem prob_CD :
  P "D" = 5 / 36 ∧ P "C" = 5 / 18 := by
  sorry

end NUMINAMATH_GPT_prob_CD_l2378_237886


namespace NUMINAMATH_GPT_t_shaped_grid_sum_l2378_237828

open Finset

theorem t_shaped_grid_sum :
  ∃ (a b c d e : ℕ), 
    a ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    b ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    c ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    d ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    e ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Finset ℕ) ∧
    (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (a ≠ e) ∧
    (b ≠ c) ∧ (b ≠ d) ∧ (b ≠ e) ∧
    (c ≠ d) ∧ (c ≠ e) ∧
    (d ≠ e) ∧
    a + b + c = 20 ∧
    d + e = 7 ∧
    (a + b + c + d + e + b) = 33 :=
sorry

end NUMINAMATH_GPT_t_shaped_grid_sum_l2378_237828


namespace NUMINAMATH_GPT_rhombus_perimeter_l2378_237824

theorem rhombus_perimeter (d1 d2 : ℕ) (h1 : d1 = 16) (h2 : d2 = 30) :
  4 * (Nat.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)) = 68 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l2378_237824


namespace NUMINAMATH_GPT_magnitude_of_vector_l2378_237861

open Complex

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - I) : 
  ‖(2 / z + z^2)‖ = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_vector_l2378_237861


namespace NUMINAMATH_GPT_rooms_per_floor_l2378_237869

-- Definitions for each of the conditions
def numberOfFloors : ℕ := 4
def hoursPerRoom : ℕ := 6
def hourlyRate : ℕ := 15
def totalEarnings : ℕ := 3600

-- Statement of the problem
theorem rooms_per_floor : 
  (totalEarnings / hourlyRate) / hoursPerRoom / numberOfFloors = 10 := 
  sorry

end NUMINAMATH_GPT_rooms_per_floor_l2378_237869


namespace NUMINAMATH_GPT_ice_cream_cone_cost_l2378_237850

theorem ice_cream_cone_cost (total_sales : ℝ) (free_cones_given : ℕ) (cost_per_cone : ℝ) 
  (customers_per_group : ℕ) (cones_sold_per_group : ℕ) 
  (h1 : total_sales = 100)
  (h2: free_cones_given = 10)
  (h3: customers_per_group = 6)
  (h4: cones_sold_per_group = 5) :
  cost_per_cone = 2 := sorry

end NUMINAMATH_GPT_ice_cream_cone_cost_l2378_237850


namespace NUMINAMATH_GPT_infinite_solutions_or_no_solutions_l2378_237890

theorem infinite_solutions_or_no_solutions (a b : ℚ) :
  (∃ (x y : ℚ), a * x^2 + b * y^2 = 1) →
  (∀ (k : ℚ), a * k^2 + b ≠ 0 → ∃ (x_k y_k : ℚ), a * x_k^2 + b * y_k^2 = 1) :=
by
  intro h_sol h_k
  sorry

end NUMINAMATH_GPT_infinite_solutions_or_no_solutions_l2378_237890


namespace NUMINAMATH_GPT_four_digit_integer_l2378_237829

theorem four_digit_integer (a b c d : ℕ) (h1 : a + b + c + d = 18)
  (h2 : b + c = 11) (h3 : a - d = 1) (h4 : 11 ∣ (1000 * a + 100 * b + 10 * c + d)) :
  1000 * a + 100 * b + 10 * c + d = 4653 :=
by sorry

end NUMINAMATH_GPT_four_digit_integer_l2378_237829


namespace NUMINAMATH_GPT_percentage_discount_l2378_237804

theorem percentage_discount (discounted_price original_price : ℝ) (h1 : discounted_price = 560) (h2 : original_price = 700) :
  (original_price - discounted_price) / original_price * 100 = 20 :=
by
  simp [h1, h2]
  sorry

end NUMINAMATH_GPT_percentage_discount_l2378_237804


namespace NUMINAMATH_GPT_compute_abs_a_plus_b_plus_c_l2378_237849

variable (a b c : ℝ)

theorem compute_abs_a_plus_b_plus_c (h1 : a^2 - b * c = 14)
                                   (h2 : b^2 - c * a = 14)
                                   (h3 : c^2 - a * b = -3) :
                                   |a + b + c| = 5 :=
sorry

end NUMINAMATH_GPT_compute_abs_a_plus_b_plus_c_l2378_237849


namespace NUMINAMATH_GPT_even_iff_b_eq_zero_l2378_237883

variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

def f' (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

-- Given that f' is an even function, prove that b = 0.
theorem even_iff_b_eq_zero (h : ∀ x : ℝ, f' x = f' (-x)) : b = 0 :=
  sorry

end NUMINAMATH_GPT_even_iff_b_eq_zero_l2378_237883


namespace NUMINAMATH_GPT_smallest_z_is_14_l2378_237859

-- Define the consecutive even integers and the equation.
def w (k : ℕ) := 2 * k
def x (k : ℕ) := 2 * k + 2
def y (k : ℕ) := 2 * k + 4
def z (k : ℕ) := 2 * k + 6

theorem smallest_z_is_14 : ∃ k : ℕ, z k = 14 ∧ w k ^ 3 + x k ^ 3 + y k ^ 3 = z k ^ 3 :=
by sorry

end NUMINAMATH_GPT_smallest_z_is_14_l2378_237859


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2378_237875

/-- Given a geometric sequence with common ratio r = 2, and the sum of the first four terms
    equals 1, the sum of the first eight terms is 17. -/
theorem geometric_sequence_sum (a r : ℝ) (h : r = 2) (h_sum_four : a * (1 + r + r^2 + r^3) = 1) :
  a * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) = 17 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2378_237875


namespace NUMINAMATH_GPT_calories_per_slice_l2378_237881

theorem calories_per_slice (n k t c : ℕ) (h1 : n = 8) (h2 : k = n / 2) (h3 : k * c = t) (h4 : t = 1200) : c = 300 :=
by sorry

end NUMINAMATH_GPT_calories_per_slice_l2378_237881


namespace NUMINAMATH_GPT_pyramid_surface_area_l2378_237803

noncomputable def total_surface_area : Real :=
  let ab := 14
  let bc := 8
  let pf := 15
  let base_area := ab * bc
  let fm := ab / 2
  let pm_ab := Real.sqrt (pf^2 + fm^2)
  let pm_bc := Real.sqrt (pf^2 + (bc / 2)^2)
  base_area + 2 * (ab / 2 * pm_ab) + 2 * (bc / 2 * pm_bc)

theorem pyramid_surface_area :
  total_surface_area = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 := by
  sorry

end NUMINAMATH_GPT_pyramid_surface_area_l2378_237803


namespace NUMINAMATH_GPT_puppies_per_cage_l2378_237870

theorem puppies_per_cage (initial_puppies : ℕ) (sold_puppies : ℕ) (remaining_puppies : ℕ) (cages : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 9)
  (h5 : puppies_per_cage = remaining_puppies / cages) : 
  puppies_per_cage = 9 := 
by
  -- The proof should go here
  sorry

end NUMINAMATH_GPT_puppies_per_cage_l2378_237870


namespace NUMINAMATH_GPT_sequence_a31_value_l2378_237860

theorem sequence_a31_value 
  (a : ℕ → ℝ) 
  (b : ℕ → ℝ) 
  (h₀ : a 1 = 0) 
  (h₁ : ∀ n, a (n + 1) = a n + b n) 
  (h₂ : b 15 + b 16 = 15)
  (h₃ : ∀ m n : ℕ, (b n - b m) = (n - m) * (b 2 - b 1)) :
  a 31 = 225 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a31_value_l2378_237860


namespace NUMINAMATH_GPT_vasya_numbers_l2378_237825

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end NUMINAMATH_GPT_vasya_numbers_l2378_237825


namespace NUMINAMATH_GPT_condition_on_a_and_b_l2378_237842

variable (x a b : ℝ)

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem condition_on_a_and_b
  (h1 : a > 0)
  (h2 : b > 0) :
  (∀ x : ℝ, |f x + 3| < a ↔ |x - 1| < b) ↔ (b^2 + 2*b + 3 ≤ a) :=
sorry

end NUMINAMATH_GPT_condition_on_a_and_b_l2378_237842


namespace NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l2378_237822

section Problem1

def f1 (x : ℝ) : ℝ := |x - 4| + |x - 2|

theorem problem1_solution_set (a : ℝ) (h : a = 2) :
  { x : ℝ | f1 x > 10 } = { x : ℝ | x > 8 ∨ x < -2 } := sorry

end Problem1


section Problem2

def f2 (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem problem2_range_of_a (f_geq : ∀ x : ℝ, f2 x a ≥ 1) :
  a ≥ 5 ∨ a ≤ 3 := sorry

end Problem2

end NUMINAMATH_GPT_problem1_solution_set_problem2_range_of_a_l2378_237822


namespace NUMINAMATH_GPT_range_of_b_l2378_237893

noncomputable def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x + 2
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := f (f x b) b

theorem range_of_b (b : ℝ) : 
  (∀ y : ℝ, ∃ x : ℝ, f x b = y) → (∀ z : ℝ, ∃ x : ℝ, g x b = z) → b ≥ 4 ∨ b ≤ -2 :=
sorry

end NUMINAMATH_GPT_range_of_b_l2378_237893


namespace NUMINAMATH_GPT_cos_difference_simplify_l2378_237848

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cos_difference_simplify_l2378_237848


namespace NUMINAMATH_GPT_probability_of_A_not_losing_l2378_237897

/-- The probability of player A winning is 0.3,
    and the probability of a draw between player A and player B is 0.4.
    Hence, the probability of player A not losing is 0.7. -/
theorem probability_of_A_not_losing (pA_win p_draw : ℝ) (hA_win : pA_win = 0.3) (h_draw : p_draw = 0.4) : 
  (pA_win + p_draw = 0.7) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_A_not_losing_l2378_237897


namespace NUMINAMATH_GPT_xyz_sum_fraction_l2378_237805

theorem xyz_sum_fraction (a1 a2 a3 b1 b2 b3 c1 c2 c3 a b c : ℤ) 
  (h1 : a1 * (b2 * c3 - b3 * c2) - a2 * (b1 * c3 - b3 * c1) + a3 * (b1 * c2 - b2 * c1) = 9)
  (h2 : a * (b2 * c3 - b3 * c2) - a2 * (b * c3 - b3 * c) + a3 * (b * c2 - b2 * c) = 17)
  (h3 : a1 * (b * c3 - b3 * c) - a * (b1 * c3 - b3 * c1) + a3 * (b1 * c - b * c1) = -8)
  (h4 : a1 * (b2 * c - b * c2) - a2 * (b1 * c - b * c1) + a * (b1 * c2 - b2 * c1) = 7)
  (eq1 : a1 * x + a2 * y + a3 * z = a)
  (eq2 : b1 * x + b2 * y + b3 * z = b)
  (eq3 : c1 * x + c2 * y + c3 * z = c)
  : x + y + z = 16 / 9 := 
sorry

end NUMINAMATH_GPT_xyz_sum_fraction_l2378_237805


namespace NUMINAMATH_GPT_valid_triangle_count_l2378_237892

def point := (ℤ × ℤ)

def isValidPoint (p : point) : Prop := 
  1 ≤ p.1 ∧ p.1 ≤ 4 ∧ 1 ≤ p.2 ∧ p.2 ≤ 4

def isCollinear (p1 p2 p3 : point) : Prop :=
  (p2.2 - p1.2) * (p3.1 - p1.1) = (p3.2 - p1.2) * (p2.1 - p1.1)

def isValidTriangle (p1 p2 p3 : point) : Prop :=
  isValidPoint p1 ∧ isValidPoint p2 ∧ isValidPoint p3 ∧ ¬isCollinear p1 p2 p3

def numberOfValidTriangles : ℕ :=
  sorry -- This will contain the combinatorial calculations from the solution.

theorem valid_triangle_count : numberOfValidTriangles = 520 :=
  sorry -- Proof will show combinatorial result from counting non-collinear combinations.

end NUMINAMATH_GPT_valid_triangle_count_l2378_237892


namespace NUMINAMATH_GPT_antifreeze_solution_l2378_237899

theorem antifreeze_solution (x : ℝ) 
  (h1 : 26 * x + 13 * 0.54 = 39 * 0.58) : 
  x = 0.6 := 
by 
  sorry

end NUMINAMATH_GPT_antifreeze_solution_l2378_237899


namespace NUMINAMATH_GPT_probability_spinner_lands_in_shaded_region_l2378_237843

theorem probability_spinner_lands_in_shaded_region :
  let total_regions := 4
  let shaded_regions := 3
  (shaded_regions: ℝ) / total_regions = 3 / 4 :=
by
  let total_regions := 4
  let shaded_regions := 3
  sorry

end NUMINAMATH_GPT_probability_spinner_lands_in_shaded_region_l2378_237843


namespace NUMINAMATH_GPT_jessica_has_100_dollars_l2378_237855

-- Define the variables for Rodney, Ian, and Jessica
variables (R I J : ℝ)

-- Given conditions
axiom rodney_more_than_ian : R = I + 35
axiom ian_half_of_jessica : I = J / 2
axiom jessica_more_than_rodney : J = R + 15

-- The statement to prove
theorem jessica_has_100_dollars : J = 100 :=
by
  -- Proof will be completed here
  sorry

end NUMINAMATH_GPT_jessica_has_100_dollars_l2378_237855


namespace NUMINAMATH_GPT_cost_of_eraser_l2378_237878

theorem cost_of_eraser 
  (s n c : ℕ)
  (h1 : s > 18)
  (h2 : n > 2)
  (h3 : c > n)
  (h4 : s * c * n = 3978) : 
  c = 17 :=
sorry

end NUMINAMATH_GPT_cost_of_eraser_l2378_237878


namespace NUMINAMATH_GPT_quadratic_complete_square_l2378_237836

theorem quadratic_complete_square (x m n : ℝ) 
  (h : 9 * x^2 - 36 * x - 81 = 0) :
  (x + m)^2 = n ∧ m + n = 11 :=
sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2378_237836


namespace NUMINAMATH_GPT_max_value_of_f_l2378_237815

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_value_of_f : ∃ x_max : ℝ, (∀ x : ℝ, f x ≤ f x_max) ∧ f (Real.exp 1) = 1 / Real.exp 1 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l2378_237815


namespace NUMINAMATH_GPT_line_intersects_y_axis_at_point_l2378_237851

def line_intersects_y_axis (x1 y1 x2 y2 : ℚ) : Prop :=
  ∃ c : ℚ, ∀ x : ℚ, y1 + (y2 - y1) / (x2 - x1) * (x - x1) = (y2 - y1) / (x2 - x1) * x + c

theorem line_intersects_y_axis_at_point :
  line_intersects_y_axis 3 21 (-9) (-6) :=
  sorry

end NUMINAMATH_GPT_line_intersects_y_axis_at_point_l2378_237851


namespace NUMINAMATH_GPT_Maria_students_l2378_237802

variable (M J : ℕ)

def conditions : Prop :=
  (M = 4 * J) ∧ (M + J = 2500)

theorem Maria_students : conditions M J → M = 2000 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Maria_students_l2378_237802


namespace NUMINAMATH_GPT_angle_value_l2378_237876

theorem angle_value (x y : ℝ) (h_parallel : True)
  (h_alt_int_ang : x = y)
  (h_triangle_sum : 2 * x + x + 60 = 180) : 
  y = 40 := 
by
  sorry

end NUMINAMATH_GPT_angle_value_l2378_237876


namespace NUMINAMATH_GPT_train_speed_l2378_237813

theorem train_speed (L V : ℝ) (h1 : L = V * 10) (h2 : L + 500 = V * 35) : V = 20 :=
by {
  -- Proof is skipped with sorry
  sorry
}

end NUMINAMATH_GPT_train_speed_l2378_237813
