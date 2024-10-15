import Mathlib

namespace NUMINAMATH_GPT_locus_of_tangent_circle_is_hyperbola_l998_99879

theorem locus_of_tangent_circle_is_hyperbola :
  ∀ (P : ℝ × ℝ) (r : ℝ),
    (P.1 ^ 2 + P.2 ^ 2).sqrt = 1 + r ∧ ((P.1 - 4) ^ 2 + P.2 ^ 2).sqrt = 2 + r →
    ∃ (a b : ℝ), (P.1 - a) ^ 2 / b ^ 2 - (P.2 / a) ^ 2 / b ^ 2 = 1 :=
sorry

end NUMINAMATH_GPT_locus_of_tangent_circle_is_hyperbola_l998_99879


namespace NUMINAMATH_GPT_circle_radius_l998_99873

theorem circle_radius :
  ∃ c : ℝ × ℝ, 
    c.2 = 0 ∧
    (dist c (2, 3)) = (dist c (3, 7)) ∧
    (dist c (2, 3)) = (Real.sqrt 1717) / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l998_99873


namespace NUMINAMATH_GPT_wholesale_cost_l998_99819

theorem wholesale_cost (W R : ℝ) (h1 : R = 1.20 * W) (h2 : 0.70 * R = 168) : W = 200 :=
by
  sorry

end NUMINAMATH_GPT_wholesale_cost_l998_99819


namespace NUMINAMATH_GPT_count_distinct_m_in_right_triangle_l998_99885

theorem count_distinct_m_in_right_triangle (k : ℝ) (hk : k > 0) :
  ∃! m : ℝ, (m = -3/8 ∨ m = -3/4) :=
by
  sorry

end NUMINAMATH_GPT_count_distinct_m_in_right_triangle_l998_99885


namespace NUMINAMATH_GPT_circle_eq_of_given_center_and_radius_l998_99854

theorem circle_eq_of_given_center_and_radius :
  (∀ (x y : ℝ),
    let C := (-1, 2)
    let r := 4
    (x + 1) ^ 2 + (y - 2) ^ 2 = 16) :=
by
  sorry

end NUMINAMATH_GPT_circle_eq_of_given_center_and_radius_l998_99854


namespace NUMINAMATH_GPT_gcd_1407_903_l998_99842

theorem gcd_1407_903 : Nat.gcd 1407 903 = 21 := 
  sorry

end NUMINAMATH_GPT_gcd_1407_903_l998_99842


namespace NUMINAMATH_GPT_reflection_point_sum_l998_99808

theorem reflection_point_sum (m b : ℝ) (H : ∀ x y : ℝ, (1, 2) = (x, y) ∨ (7, 6) = (x, y) → 
    y = m * x + b) : m + b = 8.5 := by
  sorry

end NUMINAMATH_GPT_reflection_point_sum_l998_99808


namespace NUMINAMATH_GPT_intersection_is_open_interval_l998_99802

open Set
open Real

noncomputable def M : Set ℝ := {x | x < 1}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

theorem intersection_is_open_interval :
  M ∩ N = { x | 0 < x ∧ x < 1 } := by
  sorry

end NUMINAMATH_GPT_intersection_is_open_interval_l998_99802


namespace NUMINAMATH_GPT_Faye_can_still_make_8_bouquets_l998_99888

theorem Faye_can_still_make_8_bouquets (total_flowers : ℕ) (wilted_flowers : ℕ) (flowers_per_bouquet : ℕ) 
(h1 : total_flowers = 88) 
(h2 : wilted_flowers = 48) 
(h3 : flowers_per_bouquet = 5) : 
(total_flowers - wilted_flowers) / flowers_per_bouquet = 8 := 
by
  sorry

end NUMINAMATH_GPT_Faye_can_still_make_8_bouquets_l998_99888


namespace NUMINAMATH_GPT_rectangular_prism_volume_l998_99826

theorem rectangular_prism_volume
  (L W h : ℝ)
  (h1 : L - W = 23)
  (h2 : 2 * L + 2 * W = 166) :
  L * W * h = 1590 * h :=
by
  sorry

end NUMINAMATH_GPT_rectangular_prism_volume_l998_99826


namespace NUMINAMATH_GPT_larger_segment_length_l998_99810

theorem larger_segment_length 
  (x y : ℝ)
  (h1 : 40^2 = x^2 + y^2)
  (h2 : 90^2 = (110 - x)^2 + y^2) :
  110 - x = 84.55 :=
by
  sorry

end NUMINAMATH_GPT_larger_segment_length_l998_99810


namespace NUMINAMATH_GPT_bone_meal_percentage_growth_l998_99855

-- Definitions for the problem conditions
def control_height : ℝ := 36
def cow_manure_height : ℝ := 90
def bone_meal_to_cow_manure_ratio : ℝ := 0.5 -- since cow manure plant is 200% the height of bone meal plant

noncomputable def bone_meal_height : ℝ := cow_manure_height * bone_meal_to_cow_manure_ratio

-- The main theorem to prove
theorem bone_meal_percentage_growth : 
  ( (bone_meal_height - control_height) / control_height ) * 100 = 25 := 
by
  sorry

end NUMINAMATH_GPT_bone_meal_percentage_growth_l998_99855


namespace NUMINAMATH_GPT_cube_decomposition_smallest_number_91_l998_99809

theorem cube_decomposition_smallest_number_91 (m : ℕ) (h1 : 0 < m) (h2 : (91 - 1) / 2 + 2 = m * m - m + 1) : m = 10 := by {
  sorry
}

end NUMINAMATH_GPT_cube_decomposition_smallest_number_91_l998_99809


namespace NUMINAMATH_GPT_smallest_whole_number_greater_than_sum_is_12_l998_99899

-- Definitions of the mixed numbers as improper fractions
def a : ℚ := 5 / 3
def b : ℚ := 9 / 4
def c : ℚ := 27 / 8
def d : ℚ := 25 / 6

-- The target sum and the required proof statement
theorem smallest_whole_number_greater_than_sum_is_12 : 
  let sum := a + b + c + d
  let smallest_whole_number_greater_than_sum := Nat.ceil sum
  smallest_whole_number_greater_than_sum = 12 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_whole_number_greater_than_sum_is_12_l998_99899


namespace NUMINAMATH_GPT_ethan_presents_l998_99894

theorem ethan_presents (ethan alissa : ℕ) 
  (h1 : alissa = ethan + 22) 
  (h2 : alissa = 53) : 
  ethan = 31 := 
by
  sorry

end NUMINAMATH_GPT_ethan_presents_l998_99894


namespace NUMINAMATH_GPT_z_plus_inv_y_eq_10_div_53_l998_99843

-- Define the conditions for x, y, z being positive real numbers such that
-- xyz = 1, x + 1/z = 8, and y + 1/x = 20
variables (x y z : ℝ)
variables (hx : x > 0)
variables (hy : y > 0)
variables (hz : z > 0)
variables (h1 : x * y * z = 1)
variables (h2 : x + 1 / z = 8)
variables (h3 : y + 1 / x = 20)

-- The goal is to prove that z + 1/y = 10 / 53
theorem z_plus_inv_y_eq_10_div_53 : z + 1 / y = 10 / 53 :=
by {
  sorry
}

end NUMINAMATH_GPT_z_plus_inv_y_eq_10_div_53_l998_99843


namespace NUMINAMATH_GPT_roots_of_quadratic_eval_l998_99827

theorem roots_of_quadratic_eval :
  ∀ x₁ x₂ : ℝ, (x₁^2 + 4 * x₁ + 2 = 0) ∧ (x₂^2 + 4 * x₂ + 2 = 0) ∧ (x₁ + x₂ = -4) ∧ (x₁ * x₂ = 2) →
    x₁^3 + 14 * x₂ + 55 = 7 :=
by
  sorry

end NUMINAMATH_GPT_roots_of_quadratic_eval_l998_99827


namespace NUMINAMATH_GPT_bunnies_burrow_exit_counts_l998_99890

theorem bunnies_burrow_exit_counts :
  let groupA_bunnies := 40
  let groupA_rate := 3  -- times per minute per bunny
  let groupB_bunnies := 30
  let groupB_rate := 5 / 2 -- times per minute per bunny
  let groupC_bunnies := 30
  let groupC_rate := 8 / 5 -- times per minute per bunny
  let total_bunnies := 100
  let minutes_per_day := 1440
  let days_per_week := 7
  let pre_change_rate_per_min := groupA_bunnies * groupA_rate + groupB_bunnies * groupB_rate + groupC_bunnies * groupC_rate
  let post_change_rate_per_min := pre_change_rate_per_min * 0.5
  let total_pre_change_counts := pre_change_rate_per_min * minutes_per_day * days_per_week
  let total_post_change_counts := post_change_rate_per_min * minutes_per_day * (days_per_week * 2)
  total_pre_change_counts + total_post_change_counts = 4897920 := by
    sorry

end NUMINAMATH_GPT_bunnies_burrow_exit_counts_l998_99890


namespace NUMINAMATH_GPT_polynomial_divisibility_l998_99805

theorem polynomial_divisibility (m : ℕ) (h_pos : 0 < m) : 
  ∀ x : ℝ, x * (x + 1) * (2 * x + 1) ∣ (x + 1)^(2 * m) - x^(2 * m) - 2 * x - 1 :=
sorry

end NUMINAMATH_GPT_polynomial_divisibility_l998_99805


namespace NUMINAMATH_GPT_monthly_average_growth_rate_price_reduction_for_profit_l998_99887

-- Part 1: Monthly average growth rate of sales volume
theorem monthly_average_growth_rate (x : ℝ) : 
  256 * (1 + x) ^ 2 = 400 ↔ x = 0.25 :=
by
  sorry

-- Part 2: Price reduction to achieve profit of $4250
theorem price_reduction_for_profit (m : ℝ) : 
  (40 - m - 25) * (400 + 5 * m) = 4250 ↔ m = 5 :=
by
  sorry

end NUMINAMATH_GPT_monthly_average_growth_rate_price_reduction_for_profit_l998_99887


namespace NUMINAMATH_GPT_find_solutions_l998_99831

noncomputable def cuberoot (x : ℝ) : ℝ := x^(1/3)

theorem find_solutions :
    {x : ℝ | cuberoot x = 15 / (8 - cuberoot x)} = {125, 27} :=
by
  sorry

end NUMINAMATH_GPT_find_solutions_l998_99831


namespace NUMINAMATH_GPT_find_certain_age_l998_99838

theorem find_certain_age 
(Kody_age : ℕ) 
(Mohamed_age : ℕ) 
(certain_age : ℕ) 
(h1 : Kody_age = 32) 
(h2 : Mohamed_age = 2 * certain_age) 
(h3 : ∀ four_years_ago, four_years_ago = Kody_age - 4 → four_years_ago * 2 = Mohamed_age - 4) :
  certain_age = 30 := sorry

end NUMINAMATH_GPT_find_certain_age_l998_99838


namespace NUMINAMATH_GPT_number_of_envelopes_l998_99822

theorem number_of_envelopes (total_weight_grams : ℕ) (weight_per_envelope_grams : ℕ) (n : ℕ) :
  total_weight_grams = 7480 ∧ weight_per_envelope_grams = 8500 ∧ n = 880 → total_weight_grams = n * weight_per_envelope_grams := 
sorry

end NUMINAMATH_GPT_number_of_envelopes_l998_99822


namespace NUMINAMATH_GPT_go_total_pieces_l998_99803

theorem go_total_pieces (T : ℕ) (h : T > 0) (prob_black : T = (3 : ℕ) * 4) : T = 12 := by
  sorry

end NUMINAMATH_GPT_go_total_pieces_l998_99803


namespace NUMINAMATH_GPT_condition_A_sufficient_not_necessary_condition_B_l998_99811

theorem condition_A_sufficient_not_necessary_condition_B {a b : ℝ} (hA : a > 1 ∧ b > 1) : 
  (a + b > 2 ∧ ab > 1) ∧ ¬∀ a b, (a + b > 2 ∧ ab > 1) → (a > 1 ∧ b > 1) :=
by
  sorry

end NUMINAMATH_GPT_condition_A_sufficient_not_necessary_condition_B_l998_99811


namespace NUMINAMATH_GPT_light_coloured_blocks_in_tower_l998_99828

theorem light_coloured_blocks_in_tower :
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  total_light_coloured_blocks = 20 :=
by
  let central_blocks := 4
  let outer_columns := 8
  let height_per_outer_column := 2
  let total_light_coloured_blocks := central_blocks + outer_columns * height_per_outer_column
  show total_light_coloured_blocks = 20
  sorry

end NUMINAMATH_GPT_light_coloured_blocks_in_tower_l998_99828


namespace NUMINAMATH_GPT_sufficient_not_necessary_l998_99893

theorem sufficient_not_necessary (x : ℝ) (h1 : -1 < x) (h2 : x < 3) :
    x^2 - 2*x < 8 :=
by
    -- Proof to be filled in.
    sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l998_99893


namespace NUMINAMATH_GPT_maximum_M_for_right_triangle_l998_99867

theorem maximum_M_for_right_triangle (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  (1 / a + 1 / b + 1 / c) ≥ (5 + 3 * Real.sqrt 2) / (a + b + c) :=
sorry

end NUMINAMATH_GPT_maximum_M_for_right_triangle_l998_99867


namespace NUMINAMATH_GPT_Joey_swimming_days_l998_99813

-- Define the conditions and required proof statement
theorem Joey_swimming_days (E : ℕ) (h1 : 3 * E / 4 = 9) : E / 2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_Joey_swimming_days_l998_99813


namespace NUMINAMATH_GPT_max_license_plates_is_correct_l998_99882

theorem max_license_plates_is_correct :
  let letters := 26
  let digits := 10
  (letters * (letters - 1) * digits^3 = 26 * 25 * 10^3) :=
by 
  sorry

end NUMINAMATH_GPT_max_license_plates_is_correct_l998_99882


namespace NUMINAMATH_GPT_sphere_surface_area_of_solid_l998_99839

theorem sphere_surface_area_of_solid (l w h : ℝ) (hl : l = 2) (hw : w = 1) (hh : h = 2) 
: 4 * Real.pi * ((Real.sqrt (l^2 + w^2 + h^2) / 2)^2) = 9 * Real.pi := 
by 
  sorry

end NUMINAMATH_GPT_sphere_surface_area_of_solid_l998_99839


namespace NUMINAMATH_GPT_smallest_slice_area_l998_99880

theorem smallest_slice_area
  (a₁ : ℕ) (d : ℕ) (total_angle : ℕ) (r : ℕ) 
  (h₁ : a₁ = 30) (h₂ : d = 2) (h₃ : total_angle = 360) (h₄ : r = 10) :
  ∃ (n : ℕ) (smallest_angle : ℕ),
  n = 9 ∧ smallest_angle = 18 ∧ 
  ∃ (area : ℝ), area = 5 * Real.pi :=
by
  sorry


end NUMINAMATH_GPT_smallest_slice_area_l998_99880


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l998_99877

def proposition_p (m : ℝ) : Prop := ∀ x : ℝ, |x + 1| + |x - 1| ≥ m
def proposition_q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 - 2 * m * x₀ + m^2 + m - 3 = 0

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (proposition_p m → proposition_q m) ∧ ¬ (proposition_q m → proposition_p m) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l998_99877


namespace NUMINAMATH_GPT_tangent_line_at_origin_l998_99851

/-- 
The curve is given by y = exp x.
The tangent line to this curve that passes through the origin (0, 0) 
has the equation y = exp 1 * x.
-/
theorem tangent_line_at_origin :
  ∀ (x y : ℝ), y = Real.exp x → (∃ k : ℝ, ∀ x, y = k * x ∧ k = Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_origin_l998_99851


namespace NUMINAMATH_GPT_stella_glasses_count_l998_99853

-- Definitions for the conditions
def dolls : ℕ := 3
def clocks : ℕ := 2
def price_per_doll : ℕ := 5
def price_per_clock : ℕ := 15
def price_per_glass : ℕ := 4
def total_cost : ℕ := 40
def profit : ℕ := 25

-- The proof statement
theorem stella_glasses_count (dolls clocks price_per_doll price_per_clock price_per_glass total_cost profit : ℕ) :
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost = total_cost + profit → 
  (dolls * price_per_doll + clocks * price_per_clock) + profit + total_cost - (dolls * price_per_doll + clocks * price_per_clock) = price_per_glass * 5 :=
sorry

end NUMINAMATH_GPT_stella_glasses_count_l998_99853


namespace NUMINAMATH_GPT_smaller_number_l998_99841

theorem smaller_number (x y : ℝ) (h1 : x + y = 16) (h2 : x - y = 4) (h3 : x * y = 60) : y = 6 :=
sorry

end NUMINAMATH_GPT_smaller_number_l998_99841


namespace NUMINAMATH_GPT_matrix_power_B_l998_99815

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![0, 1, 0],
  ![0, 0, 1],
  ![1, 0, 0]
]

theorem matrix_power_B :
  B ^ 150 = 1 :=
by sorry

end NUMINAMATH_GPT_matrix_power_B_l998_99815


namespace NUMINAMATH_GPT_expected_value_of_win_is_162_l998_99835

noncomputable def expected_value_of_win : ℝ :=
  (1/8) * (1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 + 7^3 + 8^3)

theorem expected_value_of_win_is_162 : expected_value_of_win = 162 := 
by 
  sorry

end NUMINAMATH_GPT_expected_value_of_win_is_162_l998_99835


namespace NUMINAMATH_GPT_graph_passes_through_quadrants_l998_99862

def linear_function (x : ℝ) : ℝ := -5 * x + 5

theorem graph_passes_through_quadrants :
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y > 0) ∧  -- Quadrant I
  (∃ x y : ℝ, linear_function x = y ∧ x < 0 ∧ y > 0) ∧  -- Quadrant II
  (∃ x y : ℝ, linear_function x = y ∧ x > 0 ∧ y < 0)    -- Quadrant IV
  :=
by
  sorry

end NUMINAMATH_GPT_graph_passes_through_quadrants_l998_99862


namespace NUMINAMATH_GPT_final_portfolio_value_l998_99869

-- Define the initial conditions and growth rates
def initial_investment : ℝ := 80
def first_year_growth_rate : ℝ := 0.15
def additional_investment : ℝ := 28
def second_year_growth_rate : ℝ := 0.10

-- Calculate the values of the portfolio at each step
def after_first_year_investment : ℝ := initial_investment * (1 + first_year_growth_rate)
def after_addition : ℝ := after_first_year_investment + additional_investment
def after_second_year_investment : ℝ := after_addition * (1 + second_year_growth_rate)

theorem final_portfolio_value : after_second_year_investment = 132 := by
  -- This is where the proof would go, but we are omitting it
  sorry

end NUMINAMATH_GPT_final_portfolio_value_l998_99869


namespace NUMINAMATH_GPT_find_rate_of_grapes_l998_99814

def rate_per_kg_of_grapes (G : ℝ) : Prop :=
  let cost_of_grapes := 8 * G
  let cost_of_mangoes := 10 * 55
  let total_paid := 1110
  cost_of_grapes + cost_of_mangoes = total_paid

theorem find_rate_of_grapes : rate_per_kg_of_grapes 70 :=
by
  unfold rate_per_kg_of_grapes
  sorry

end NUMINAMATH_GPT_find_rate_of_grapes_l998_99814


namespace NUMINAMATH_GPT_complete_square_expression_l998_99895

theorem complete_square_expression :
  ∃ (a h k : ℝ), (∀ x : ℝ, 2 * x^2 + 8 * x + 6 = a * (x - h)^2 + k) ∧ (a + h + k = -2) :=
by
  sorry

end NUMINAMATH_GPT_complete_square_expression_l998_99895


namespace NUMINAMATH_GPT_least_x_y_z_value_l998_99825

theorem least_x_y_z_value :
  ∃ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (3 * x = 4 * y) ∧ (4 * y = 7 * z) ∧ (3 * x = 7 * z) ∧ (x - y + z = 19) :=
by
  sorry

end NUMINAMATH_GPT_least_x_y_z_value_l998_99825


namespace NUMINAMATH_GPT_leif_has_more_oranges_than_apples_l998_99812

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end NUMINAMATH_GPT_leif_has_more_oranges_than_apples_l998_99812


namespace NUMINAMATH_GPT_shortest_side_l998_99865

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end NUMINAMATH_GPT_shortest_side_l998_99865


namespace NUMINAMATH_GPT_goldfish_added_per_day_is_7_l998_99878

def initial_koi_fish : ℕ := 227 - 2
def initial_goldfish : ℕ := 280 - initial_koi_fish
def added_goldfish : ℕ := 200 - initial_goldfish
def days_in_three_weeks : ℕ := 3 * 7
def goldfish_added_per_day : ℕ := (added_goldfish + days_in_three_weeks - 1) / days_in_three_weeks -- rounding to nearest integer 

theorem goldfish_added_per_day_is_7 : goldfish_added_per_day = 7 :=
by 
-- sorry to skip the proof
sorry

end NUMINAMATH_GPT_goldfish_added_per_day_is_7_l998_99878


namespace NUMINAMATH_GPT_remaining_fuel_relation_l998_99848

-- Define the car's travel time and remaining fuel relation
def initial_fuel : ℝ := 100

def fuel_consumption_rate : ℝ := 6

def remaining_fuel (t : ℝ) : ℝ := initial_fuel - fuel_consumption_rate * t

-- Prove that the remaining fuel after t hours is given by the linear relationship Q = 100 - 6t
theorem remaining_fuel_relation (t : ℝ) : remaining_fuel t = 100 - 6 * t := by
  -- Proof is omitted, as per instructions
  sorry

end NUMINAMATH_GPT_remaining_fuel_relation_l998_99848


namespace NUMINAMATH_GPT_equation_of_line_l998_99816

noncomputable def vector := (Real × Real)
noncomputable def point := (Real × Real)

def line_equation (x y : Real) : Prop := 
  let v1 : vector := (-1, 2)
  let p : point := (3, -4)
  let lhs := (v1.1 * (x - p.1) + v1.2 * (y - p.2)) = 0
  lhs

theorem equation_of_line (x y : Real) :
  line_equation x y ↔ y = (1/2) * x - (11/2) := 
  sorry

end NUMINAMATH_GPT_equation_of_line_l998_99816


namespace NUMINAMATH_GPT_james_huskies_count_l998_99801

theorem james_huskies_count 
  (H : ℕ) 
  (pitbulls : ℕ := 2) 
  (golden_retrievers : ℕ := 4) 
  (husky_pups_per_husky : ℕ := 3) 
  (pitbull_pups_per_pitbull : ℕ := 3) 
  (extra_pups_per_golden_retriever : ℕ := 2) 
  (pup_difference : ℕ := 30) :
  H + pitbulls + golden_retrievers + pup_difference = 3 * H + pitbulls * pitbull_pups_per_pitbull + golden_retrievers * (husky_pups_per_husky + extra_pups_per_golden_retriever) :=
sorry

end NUMINAMATH_GPT_james_huskies_count_l998_99801


namespace NUMINAMATH_GPT_part_a_part_b_l998_99840

-- Definitions for maximum factor increases
def f (n : ℕ) (a : ℕ) : ℚ := sorry
def t (n : ℕ) (a : ℕ) : ℚ := sorry

-- Part (a): Prove the factor increase for exactly 1 blue cube in 100 boxes
theorem part_a : f 100 1 = 2^100 / 100 := sorry

-- Part (b): Prove the factor increase for some integer \( k \) blue cubes in 100 boxes, \( 1 < k \leq 100 \)
theorem part_b (k : ℕ) (hk : 1 < k ∧ k ≤ 100) : t 100 k = 2^100 / (2^100 - k - 1) := sorry

end NUMINAMATH_GPT_part_a_part_b_l998_99840


namespace NUMINAMATH_GPT_find_f_ln_inv_6_l998_99874

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x + 2 / x^3 - 3

theorem find_f_ln_inv_6 (k : ℝ) (h : f k (Real.log 6) = 1) : f k (Real.log (1 / 6)) = -7 :=
by
  sorry

end NUMINAMATH_GPT_find_f_ln_inv_6_l998_99874


namespace NUMINAMATH_GPT_total_distance_traveled_l998_99892

theorem total_distance_traveled :
  let time1 := 3  -- hours
  let speed1 := 70  -- km/h
  let time2 := 4  -- hours
  let speed2 := 80  -- km/h
  let time3 := 3  -- hours
  let speed3 := 65  -- km/h
  let time4 := 2  -- hours
  let speed4 := 90  -- km/h
  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4
  distance1 + distance2 + distance3 + distance4 = 905 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l998_99892


namespace NUMINAMATH_GPT_pure_ghee_percentage_l998_99829

theorem pure_ghee_percentage (Q : ℝ) (vanaspati_percentage : ℝ:= 0.40) (additional_pure_ghee : ℝ := 10) (new_vanaspati_percentage : ℝ := 0.20) (original_quantity : ℝ := 10) :
  (Q = original_quantity) ∧ (vanaspati_percentage = 0.40) ∧ (additional_pure_ghee = 10) ∧ (new_vanaspati_percentage = 0.20) →
  (100 - (vanaspati_percentage * 100)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_pure_ghee_percentage_l998_99829


namespace NUMINAMATH_GPT_germs_killed_in_common_l998_99870

theorem germs_killed_in_common :
  ∃ x : ℝ, x = 5 ∧
    ∀ A B C : ℝ, A = 50 → 
    B = 25 → 
    C = 30 → 
    x = A + B - (100 - C) := sorry

end NUMINAMATH_GPT_germs_killed_in_common_l998_99870


namespace NUMINAMATH_GPT_contrapositive_of_x_squared_lt_one_is_true_l998_99820

variable {x : ℝ}

theorem contrapositive_of_x_squared_lt_one_is_true
  (h : ∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1) :
  ∀ x : ℝ, x ≤ -1 ∨ x ≥ 1 → x^2 ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_of_x_squared_lt_one_is_true_l998_99820


namespace NUMINAMATH_GPT_cost_per_taco_is_1_50_l998_99850

namespace TacoTruck

def total_beef : ℝ := 100
def beef_per_taco : ℝ := 0.25
def taco_price : ℝ := 2
def profit : ℝ := 200

theorem cost_per_taco_is_1_50 :
  let total_tacos := total_beef / beef_per_taco
  let total_revenue := total_tacos * taco_price
  let total_cost := total_revenue - profit
  total_cost / total_tacos = 1.50 := 
by
  sorry

end TacoTruck

end NUMINAMATH_GPT_cost_per_taco_is_1_50_l998_99850


namespace NUMINAMATH_GPT_number_of_ways_to_assign_roles_l998_99891

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 7
  let male_roles := 3
  let female_roles := 3
  let neutral_roles := 2
  let ways_male_roles := men * (men - 1) * (men - 2)
  let ways_female_roles := women * (women - 1) * (women - 2)
  let ways_neutral_roles := (men + women - male_roles - female_roles) * (men + women - male_roles - female_roles - 1)
  ways_male_roles * ways_female_roles * ways_neutral_roles = 1058400 := 
by
  sorry

end NUMINAMATH_GPT_number_of_ways_to_assign_roles_l998_99891


namespace NUMINAMATH_GPT_math_problem_l998_99868

variable (a b c d : ℝ)

-- The initial condition provided in the problem
def given_condition : Prop := (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7

-- The statement that needs to be proven
theorem math_problem 
  (h : given_condition a b c d) : 
  (a - c) * (b - d) / ((a - b) * (c - d)) = -1 := 
by 
  sorry

end NUMINAMATH_GPT_math_problem_l998_99868


namespace NUMINAMATH_GPT_center_of_circle_l998_99837

theorem center_of_circle (x y : ℝ) : 
  (x - 1) ^ 2 + (y + 1) ^ 2 = 4 ↔ (x^2 + y^2 - 2*x + 2*y - 2 = 0) :=
sorry

end NUMINAMATH_GPT_center_of_circle_l998_99837


namespace NUMINAMATH_GPT_wednesday_more_than_half_millet_l998_99884

namespace BirdFeeder

-- Define the initial conditions
def initial_amount_millet (total_seeds : ℚ) : ℚ := 0.4 * total_seeds
def initial_amount_other (total_seeds : ℚ) : ℚ := 0.6 * total_seeds

-- Define the daily consumption
def eaten_millet (millet : ℚ) : ℚ := 0.2 * millet
def eaten_other (other : ℚ) : ℚ := other

-- Define the seed addition every other day
def add_seeds (day : ℕ) (seeds : ℚ) : Prop :=
  day % 2 = 1 → seeds = 1

-- Define the daily update of the millet and other seeds in the feeder
def daily_update (day : ℕ) (millet : ℚ) (other : ℚ) : ℚ × ℚ :=
  let remaining_millet := (1 - 0.2) * millet
  let remaining_other := 0
  if day % 2 = 1 then
    (remaining_millet + initial_amount_millet 1, initial_amount_other 1)
  else
    (remaining_millet, remaining_other)

-- Define the main property to prove
def more_than_half_millet (day : ℕ) (millet : ℚ) (other : ℚ) : Prop :=
  millet > 0.5 * (millet + other)

-- Define the theorem statement
theorem wednesday_more_than_half_millet
  (millet : ℚ := initial_amount_millet 1)
  (other : ℚ := initial_amount_other 1) :
  ∃ day, day = 3 ∧ more_than_half_millet day millet other :=
  by
  sorry

end BirdFeeder

end NUMINAMATH_GPT_wednesday_more_than_half_millet_l998_99884


namespace NUMINAMATH_GPT_water_filter_capacity_l998_99832

theorem water_filter_capacity (x : ℝ) (h : 0.30 * x = 36) : x = 120 :=
sorry

end NUMINAMATH_GPT_water_filter_capacity_l998_99832


namespace NUMINAMATH_GPT_shortest_distance_dasha_vasya_l998_99824

variables (dasha galia asya borya vasya : Type)
variables (dist : ∀ (a b : Type), ℕ)
variables (dist_dasha_galia : dist dasha galia = 15)
variables (dist_vasya_galia : dist vasya galia = 17)
variables (dist_asya_galia : dist asya galia = 12)
variables (dist_galia_borya : dist galia borya = 10)
variables (dist_asya_borya : dist asya borya = 8)

theorem shortest_distance_dasha_vasya : dist dasha vasya = 18 :=
by sorry

end NUMINAMATH_GPT_shortest_distance_dasha_vasya_l998_99824


namespace NUMINAMATH_GPT_sum_moments_equal_l998_99800

theorem sum_moments_equal
  (x1 x2 x3 y1 y2 : ℝ)
  (m1 m2 m3 n1 n2 : ℝ) :
  n1 * y1 + n2 * y2 = m1 * x1 + m2 * x2 + m3 * x3 :=
sorry

end NUMINAMATH_GPT_sum_moments_equal_l998_99800


namespace NUMINAMATH_GPT_plane_overtake_time_is_80_minutes_l998_99817

noncomputable def plane_overtake_time 
  (speed_a speed_b : ℝ)
  (head_start : ℝ) 
  (t : ℝ) : Prop :=
  speed_a * (t + head_start) = speed_b * t

theorem plane_overtake_time_is_80_minutes :
  plane_overtake_time 200 300 (2/3) (80 / 60)
:=
  sorry

end NUMINAMATH_GPT_plane_overtake_time_is_80_minutes_l998_99817


namespace NUMINAMATH_GPT_Priyanka_chocolates_l998_99864

variable (N S So P Sa T : ℕ)

theorem Priyanka_chocolates :
  (N + S = 10) →
  (So + P = 15) →
  (Sa + T = 10) →
  (N = 4) →
  ((S = 2 * y) ∨ (P = 2 * So)) →
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_Priyanka_chocolates_l998_99864


namespace NUMINAMATH_GPT_time_per_student_l998_99847

-- Given Conditions
def total_students : ℕ := 18
def groups : ℕ := 3
def minutes_per_group : ℕ := 24

-- Mathematical proof problem
theorem time_per_student :
  (minutes_per_group / (total_students / groups)) = 4 := by
  -- Proof not required, adding placeholder
  sorry

end NUMINAMATH_GPT_time_per_student_l998_99847


namespace NUMINAMATH_GPT_zoo_problem_l998_99896

variables
  (parrots : ℕ)
  (snakes : ℕ)
  (monkeys : ℕ)
  (elephants : ℕ)
  (zebras : ℕ)
  (f : ℚ)

-- Conditions from the problem
theorem zoo_problem
  (h1 : parrots = 8)
  (h2 : snakes = 3 * parrots)
  (h3 : monkeys = 2 * snakes)
  (h4 : elephants = f * (parrots + snakes))
  (h5 : zebras = elephants - 3)
  (h6 : monkeys - zebras = 35) :
  f = 1 / 2 :=
sorry

end NUMINAMATH_GPT_zoo_problem_l998_99896


namespace NUMINAMATH_GPT_probability_A8_l998_99856

/-- Define the probability of event A_n where the sum of die rolls equals n -/
def P (n : ℕ) : ℚ :=
  1/7 * (if n = 8 then 5/36 + 21/216 + 35/1296 + 35/7776 + 21/46656 +
    7/279936 + 1/1679616 else 0)

theorem probability_A8 : P 8 = (1/7) * (5/36 + 21/216 + 35/1296 + 35/7776 + 
  21/46656 + 7/279936 + 1/1679616) :=
by
  sorry

end NUMINAMATH_GPT_probability_A8_l998_99856


namespace NUMINAMATH_GPT_simplify_fraction_l998_99836

theorem simplify_fraction (x y : ℝ) : (x - y) / (y - x) = -1 :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l998_99836


namespace NUMINAMATH_GPT_starting_number_l998_99846

theorem starting_number (n : ℕ) (h1 : 200 ≥ n) (h2 : 33 = ((200 / 3) - (n / 3))) : n = 102 :=
by
  sorry

end NUMINAMATH_GPT_starting_number_l998_99846


namespace NUMINAMATH_GPT_range_a_l998_99852

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x + 2 / x - a
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 2 / x

theorem range_a (a : ℝ) : (∃ x > 0, f x a = 0) → a ≥ 3 :=
by
sorry

end NUMINAMATH_GPT_range_a_l998_99852


namespace NUMINAMATH_GPT_trapezoid_midsegment_l998_99857

theorem trapezoid_midsegment (h : ℝ) :
  ∃ k : ℝ, (∃ θ : ℝ, θ = 120 ∧ k = 2 * h * Real.cos (θ / 2)) ∧
  (∃ m : ℝ, m = k / 2) ∧
  (∃ midsegment : ℝ, midsegment = m / Real.sqrt 3 ∧ midsegment = h / Real.sqrt 3) :=
by
  -- This is where the proof would go.
  sorry

end NUMINAMATH_GPT_trapezoid_midsegment_l998_99857


namespace NUMINAMATH_GPT_ratio_correct_l998_99807

def cost_of_flasks := 150
def remaining_budget := 25
def total_budget := 325
def spent_budget := total_budget - remaining_budget
def cost_of_test_tubes := 100
def cost_of_safety_gear := cost_of_test_tubes / 2
def ratio_test_tubes_flasks := cost_of_test_tubes / cost_of_flasks

theorem ratio_correct :
  spent_budget = cost_of_flasks + cost_of_test_tubes + cost_of_safety_gear → 
  ratio_test_tubes_flasks = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_correct_l998_99807


namespace NUMINAMATH_GPT_children_boys_count_l998_99876

theorem children_boys_count (girls : ℕ) (total_children : ℕ) (boys : ℕ) 
  (h₁ : girls = 35) (h₂ : total_children = 62) : boys = 27 :=
by
  sorry

end NUMINAMATH_GPT_children_boys_count_l998_99876


namespace NUMINAMATH_GPT_algebra_expression_value_l998_99806

theorem algebra_expression_value (a : ℤ) (h : (2023 - a) ^ 2 + (a - 2022) ^ 2 = 7) :
  (2023 - a) * (a - 2022) = -3 := 
sorry

end NUMINAMATH_GPT_algebra_expression_value_l998_99806


namespace NUMINAMATH_GPT_percent_less_50000_l998_99844

variable (A B C : ℝ) -- Define the given percentages
variable (h1 : A = 0.45) -- 45% of villages have populations from 20,000 to 49,999
variable (h2 : B = 0.30) -- 30% of villages have fewer than 20,000 residents
variable (h3 : C = 0.25) -- 25% of villages have 50,000 or more residents

theorem percent_less_50000 : A + B = 0.75 := by
  sorry

end NUMINAMATH_GPT_percent_less_50000_l998_99844


namespace NUMINAMATH_GPT_totalStudents_correct_l998_99860

-- Defining the initial number of classes, students per class, and new classes
def initialClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def newClasses : ℕ := 5

-- Prove that the total number of students is 400
theorem totalStudents_correct : 
  initialClasses * studentsPerClass + newClasses * studentsPerClass = 400 := by
  sorry

end NUMINAMATH_GPT_totalStudents_correct_l998_99860


namespace NUMINAMATH_GPT_square_side_length_l998_99818

theorem square_side_length (x : ℝ) (h : x^2 = (1/2) * x * 2) : x = 1 := by
  sorry

end NUMINAMATH_GPT_square_side_length_l998_99818


namespace NUMINAMATH_GPT_shortest_hypotenuse_max_inscribed_circle_radius_l998_99830

variable {a b c r : ℝ}

-- Condition 1: The perimeter of the right-angled triangle is 1 meter.
def perimeter_condition (a b : ℝ) : Prop :=
  a + b + Real.sqrt (a^2 + b^2) = 1

-- Problem 1: Prove the shortest length of the hypotenuse is √2 - 1.
theorem shortest_hypotenuse (a b : ℝ) (h : perimeter_condition a b) :
  Real.sqrt (a^2 + b^2) = Real.sqrt 2 - 1 :=
sorry

-- Problem 2: Prove the maximum value of the inscribed circle radius is 3/2 - √2.
theorem max_inscribed_circle_radius (a b r : ℝ) (h : perimeter_condition a b) :
  (a * b = r) → r = 3/2 - Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_shortest_hypotenuse_max_inscribed_circle_radius_l998_99830


namespace NUMINAMATH_GPT_more_girls_than_boys_l998_99886

theorem more_girls_than_boys (num_students : ℕ) (boys_ratio : ℕ) (girls_ratio : ℕ) (total_students : ℕ) (total_students_eq : num_students = 42) (ratio_eq : boys_ratio = 3 ∧ girls_ratio = 4) : (4 * 6) - (3 * 6) = 6 := by
  sorry

end NUMINAMATH_GPT_more_girls_than_boys_l998_99886


namespace NUMINAMATH_GPT_xyz_inequality_l998_99845

theorem xyz_inequality (x y z : ℝ) : x^2 + y^2 + z^2 ≥ x * y + y * z + z * x := 
  sorry

end NUMINAMATH_GPT_xyz_inequality_l998_99845


namespace NUMINAMATH_GPT_average_weight_l998_99875

theorem average_weight (w_girls w_boys : ℕ) (avg_girls avg_boys : ℕ) (n : ℕ) : 
  n = 5 → avg_girls = 45 → avg_boys = 55 → 
  w_girls = n * avg_girls → w_boys = n * avg_boys →
  ∀ total_weight, total_weight = w_girls + w_boys →
  ∀ avg_weight, avg_weight = total_weight / (2 * n) →
  avg_weight = 50 :=
by
  intros h_n h_avg_girls h_avg_boys h_w_girls h_w_boys h_total_weight h_avg_weight
  -- here you would start the proof, but it is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_average_weight_l998_99875


namespace NUMINAMATH_GPT_orthogonality_implies_x_value_l998_99898

theorem orthogonality_implies_x_value :
  ∀ (x : ℝ),
  let a : ℝ × ℝ := (x, 2)
  let b : ℝ × ℝ := (2, -1)
  a.1 * b.1 + a.2 * b.2 = 0 → x = 1 :=
sorry

end NUMINAMATH_GPT_orthogonality_implies_x_value_l998_99898


namespace NUMINAMATH_GPT_absolute_sum_of_roots_l998_99863

theorem absolute_sum_of_roots (d e f n : ℤ) (h1 : d + e + f = 0) (h2 : d * e + e * f + f * d = -2023) : |d| + |e| + |f| = 98 := 
sorry

end NUMINAMATH_GPT_absolute_sum_of_roots_l998_99863


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l998_99833

-- Problem 1
theorem problem1_solution (x y : ℝ) (h1 : 2 * x + 3 * y = 8) (h2 : x = y - 1) : x = 1 ∧ y = 2 := by
  sorry

-- Problem 2
theorem problem2_solution (x y : ℝ) (h1 : 2 * x - y = -1) (h2 : x + 3 * y = 17) : x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l998_99833


namespace NUMINAMATH_GPT_repetend_of_5_over_17_is_294117_l998_99881

theorem repetend_of_5_over_17_is_294117 :
  (∀ n : ℕ, (5 / 17 : ℚ) - (294117 : ℚ) / (10^6 : ℚ) ^ n = 0) :=
by
  sorry

end NUMINAMATH_GPT_repetend_of_5_over_17_is_294117_l998_99881


namespace NUMINAMATH_GPT_red_peaches_count_l998_99861

-- Definitions for the conditions
def yellow_peaches : ℕ := 11
def extra_red_peaches : ℕ := 8

-- The proof statement that the number of red peaches is 19
theorem red_peaches_count : (yellow_peaches + extra_red_peaches = 19) :=
by
  sorry

end NUMINAMATH_GPT_red_peaches_count_l998_99861


namespace NUMINAMATH_GPT_solve_system_of_equations_l998_99889

-- Given conditions
variables {a b c k x y z : ℝ}
variables (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
variables (eq1 : a * x + b * y + c * z = k)
variables (eq2 : a^2 * x + b^2 * y + c^2 * z = k^2)
variables (eq3 : a^3 * x + b^3 * y + c^3 * z = k^3)

-- Statement to be proved
theorem solve_system_of_equations :
  x = k * (k - c) * (k - b) / (a * (a - c) * (a - b)) ∧
  y = k * (k - c) * (k - a) / (b * (b - c) * (b - a)) ∧
  z = k * (k - a) * (k - b) / (c * (c - a) * (c - b)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l998_99889


namespace NUMINAMATH_GPT_black_squares_covered_by_trominoes_l998_99871

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

noncomputable def min_trominoes (n : ℕ) : ℕ :=
  ((n + 1) ^ 2) / 4

theorem black_squares_covered_by_trominoes (n : ℕ) (h1 : n ≥ 7) (h2 : is_odd n):
  ∀ n : ℕ, ∃ k : ℕ, k = min_trominoes n :=
by
  sorry

end NUMINAMATH_GPT_black_squares_covered_by_trominoes_l998_99871


namespace NUMINAMATH_GPT_simplify_expression_l998_99858

open Real

-- Assume that x, y, z are non-zero real numbers
variables (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)

theorem simplify_expression : (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ := 
by
  -- Proof would go here.
  sorry

end NUMINAMATH_GPT_simplify_expression_l998_99858


namespace NUMINAMATH_GPT_bridge_construction_l998_99866

-- Definitions used in the Lean statement based on conditions.
def rate (workers : ℕ) (days : ℕ) : ℚ := 1 / (workers * days)

-- The problem statement: prove that if 60 workers working together can build the bridge in 3 days, 
-- then 120 workers will take 1.5 days to build the bridge.
theorem bridge_construction (t : ℚ) : 
  (rate 60 3) * 120 * t = 1 → t = 1.5 := by
  sorry

end NUMINAMATH_GPT_bridge_construction_l998_99866


namespace NUMINAMATH_GPT_integer_solutions_count_eq_11_l998_99834

theorem integer_solutions_count_eq_11 :
  ∃ (count : ℕ), (∀ n : ℤ, (n + 2) * (n - 5) + n ≤ 10 ↔ (n ≥ -4 ∧ n ≤ 6)) ∧ count = 11 :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_eq_11_l998_99834


namespace NUMINAMATH_GPT_find_slower_speed_l998_99872

-- Variables and conditions definitions
variable (v : ℝ)

def slower_speed (v : ℝ) : Prop :=
  (20 / v = 2) ∧ (v = 10)

-- The statement to be proven
theorem find_slower_speed : slower_speed 10 :=
by
  sorry

end NUMINAMATH_GPT_find_slower_speed_l998_99872


namespace NUMINAMATH_GPT_twenty_percent_greater_l998_99821

theorem twenty_percent_greater (x : ℝ) (h : x = 52 + 0.2 * 52) : x = 62.4 :=
by {
  sorry
}

end NUMINAMATH_GPT_twenty_percent_greater_l998_99821


namespace NUMINAMATH_GPT_inequality_proof_l998_99804

theorem inequality_proof (x y : ℝ) (hx : |x| < 1) (hy : |y| < 1) :
  (1 / (1 - x^2)) + (1 / (1 - y^2)) ≥ (2 / (1 - x * y)) :=
by sorry

end NUMINAMATH_GPT_inequality_proof_l998_99804


namespace NUMINAMATH_GPT_flowers_per_vase_l998_99883

-- Definitions of conditions in Lean 4
def number_of_carnations : ℕ := 7
def number_of_roses : ℕ := 47
def total_number_of_flowers : ℕ := number_of_carnations + number_of_roses
def number_of_vases : ℕ := 9

-- Statement in Lean 4
theorem flowers_per_vase : total_number_of_flowers / number_of_vases = 6 := by
  unfold total_number_of_flowers
  show (7 + 47) / 9 = 6
  sorry

end NUMINAMATH_GPT_flowers_per_vase_l998_99883


namespace NUMINAMATH_GPT_solve_3x_plus_7y_eq_23_l998_99823

theorem solve_3x_plus_7y_eq_23 :
  ∃ (x y : ℕ), 3 * x + 7 * y = 23 ∧ x = 3 ∧ y = 2 := by
sorry

end NUMINAMATH_GPT_solve_3x_plus_7y_eq_23_l998_99823


namespace NUMINAMATH_GPT_brick_width_is_10_cm_l998_99897

-- Define the conditions
def courtyard_length_meters := 25
def courtyard_width_meters := 16
def brick_length_cm := 20
def number_of_bricks := 20000

-- Convert courtyard dimensions to area in square centimeters
def area_of_courtyard_cm2 := courtyard_length_meters * 100 * courtyard_width_meters * 100

-- Total area covered by bricks
def total_brick_area_cm2 := area_of_courtyard_cm2

-- Area covered by one brick
def area_per_brick := total_brick_area_cm2 / number_of_bricks

-- Find the brick width
def brick_width_cm := area_per_brick / brick_length_cm

-- Prove the width of each brick is 10 cm
theorem brick_width_is_10_cm : brick_width_cm = 10 := 
by 
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_brick_width_is_10_cm_l998_99897


namespace NUMINAMATH_GPT_max_statements_true_l998_99859

noncomputable def max_true_statements (a b : ℝ) : ℕ :=
  (if (a^2 > b^2) then 1 else 0) +
  (if (a < b) then 1 else 0) +
  (if (a < 0) then 1 else 0) +
  (if (b < 0) then 1 else 0) +
  (if (1 / a < 1 / b) then 1 else 0)

theorem max_statements_true : ∀ (a b : ℝ), max_true_statements a b ≤ 4 :=
by
  intro a b
  sorry

end NUMINAMATH_GPT_max_statements_true_l998_99859


namespace NUMINAMATH_GPT_xy_squared_value_l998_99849

variable {x y : ℝ}

theorem xy_squared_value :
  (y + 6 = (x - 3)^2) ∧ (x + 6 = (y - 3)^2) ∧ (x ≠ y) → (x^2 + y^2 = 25) := 
by
  sorry

end NUMINAMATH_GPT_xy_squared_value_l998_99849
