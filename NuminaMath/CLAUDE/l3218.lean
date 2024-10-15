import Mathlib

namespace NUMINAMATH_CALUDE_fraction_sum_theorem_l3218_321835

theorem fraction_sum_theorem (x y : ℚ) (h : y ≠ 0) : 
  x / y = 2 / 3 → (x + y) / y = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_theorem_l3218_321835


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_17_is_correct_l3218_321899

/-- The least five-digit positive integer congruent to 7 (mod 17) -/
def least_five_digit_congruent_to_7_mod_17 : ℕ := 10003

/-- A number is five-digit if it's between 10000 and 99999 inclusive -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

theorem least_five_digit_congruent_to_7_mod_17_is_correct :
  is_five_digit least_five_digit_congruent_to_7_mod_17 ∧
  least_five_digit_congruent_to_7_mod_17 % 17 = 7 ∧
  ∀ n : ℕ, is_five_digit n ∧ n % 17 = 7 → n ≥ least_five_digit_congruent_to_7_mod_17 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_17_is_correct_l3218_321899


namespace NUMINAMATH_CALUDE_max_residents_top_floor_l3218_321813

/-- Represents the number of people living on a floor --/
def residents (floor : ℕ) : ℕ := floor

/-- The number of floors in the building --/
def num_floors : ℕ := 10

/-- Theorem: The floor with the most residents is the top floor --/
theorem max_residents_top_floor :
  ∀ k : ℕ, k ≤ num_floors → residents k ≤ residents num_floors :=
by
  sorry

#check max_residents_top_floor

end NUMINAMATH_CALUDE_max_residents_top_floor_l3218_321813


namespace NUMINAMATH_CALUDE_leila_weekly_earnings_l3218_321810

/-- Represents the earnings of a vlogger over a week -/
def weekly_earnings (daily_viewers : ℕ) (earnings_per_view : ℚ) : ℚ :=
  daily_viewers * earnings_per_view * 7

/-- Proves that Leila earns $350 per week given the conditions -/
theorem leila_weekly_earnings : 
  let voltaire_viewers : ℕ := 50
  let leila_viewers : ℕ := 2 * voltaire_viewers
  let earnings_per_view : ℚ := 1/2
  weekly_earnings leila_viewers earnings_per_view = 350 := by
sorry

end NUMINAMATH_CALUDE_leila_weekly_earnings_l3218_321810


namespace NUMINAMATH_CALUDE_second_subdivision_house_count_l3218_321871

/-- The number of houses in the second subdivision where Billy goes trick-or-treating -/
def second_subdivision_houses : ℕ := 75

/-- Anna's candy per house -/
def anna_candy_per_house : ℕ := 14

/-- Number of houses Anna visits -/
def anna_houses : ℕ := 60

/-- Billy's candy per house -/
def billy_candy_per_house : ℕ := 11

/-- Difference in total candy between Anna and Billy -/
def candy_difference : ℕ := 15

theorem second_subdivision_house_count :
  anna_candy_per_house * anna_houses = 
  billy_candy_per_house * second_subdivision_houses + candy_difference := by
  sorry

end NUMINAMATH_CALUDE_second_subdivision_house_count_l3218_321871


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3218_321840

theorem absolute_value_inequality (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3218_321840


namespace NUMINAMATH_CALUDE_price_reduction_percentage_l3218_321814

theorem price_reduction_percentage (original_price reduction : ℝ) : 
  original_price = 500 → reduction = 200 → (reduction / original_price) * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_price_reduction_percentage_l3218_321814


namespace NUMINAMATH_CALUDE_project_hours_difference_l3218_321886

theorem project_hours_difference (total_hours : ℝ) 
  (h_total : total_hours = 350) 
  (h_pat_kate : ∃ k : ℝ, pat = 2 * k ∧ kate = k)
  (h_pat_mark : ∃ m : ℝ, pat = (1/3) * m ∧ mark = m)
  (h_alex_kate : ∃ k : ℝ, alex = 1.5 * k ∧ kate = k)
  (h_sum : pat + kate + mark + alex = total_hours) :
  mark - (kate + alex) = 350/3 :=
sorry

end NUMINAMATH_CALUDE_project_hours_difference_l3218_321886


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3218_321894

theorem arithmetic_sequence_problem (a : ℕ → ℕ) (n : ℕ) :
  (∀ k, a (k + 1) = a k + 5) →  -- arithmetic sequence with common difference 5
  a 1 = 1 →                    -- first term is 1
  a n = 2016 →                 -- n-th term is 2016
  n = 404 :=                   -- prove n is 404
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3218_321894


namespace NUMINAMATH_CALUDE_smaller_number_in_ratio_l3218_321829

theorem smaller_number_in_ratio (x y a b c : ℝ) 
  (h_pos_x : x > 0) 
  (h_pos_y : y > 0) 
  (h_ratio : x / y = a / b) 
  (h_a_lt_b : 0 < a ∧ a < b) 
  (h_sum : x + y = c) : 
  min x y = a * c / (a + b) := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_in_ratio_l3218_321829


namespace NUMINAMATH_CALUDE_intersection_condition_area_condition_l3218_321833

/-- The hyperbola C: x² - y² = 1 -/
def C (x y : ℝ) : Prop := x^2 - y^2 = 1

/-- The line L: y = kx - 1 -/
def L (k x y : ℝ) : Prop := y = k * x - 1

/-- L intersects C at two distinct points -/
def intersects_at_two_points (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂

/-- The area of triangle AOB is √2 -/
def triangle_area_sqrt_2 (k : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ, C x₁ y₁ ∧ C x₂ y₂ ∧ L k x₁ y₁ ∧ L k x₂ y₂ ∧
    (1/2 : ℝ) * |x₁ - x₂| = Real.sqrt 2

theorem intersection_condition (k : ℝ) :
  intersects_at_two_points k ↔ -Real.sqrt 2 < k ∧ k < -1 :=
sorry

theorem area_condition (k : ℝ) :
  triangle_area_sqrt_2 k ↔ k = 0 ∨ k = Real.sqrt 6 / 2 ∨ k = -Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_intersection_condition_area_condition_l3218_321833


namespace NUMINAMATH_CALUDE_sphere_radius_tangent_to_truncated_cone_l3218_321823

/-- The radius of a sphere tangent to a truncated cone -/
theorem sphere_radius_tangent_to_truncated_cone 
  (r_bottom r_top h : ℝ) 
  (h_positive : 0 < h) 
  (r_bottom_positive : 0 < r_bottom) 
  (r_top_positive : 0 < r_top) 
  (r_bottom_gt_r_top : r_top < r_bottom) 
  (h_truncated_cone : r_bottom = 24 ∧ r_top = 6 ∧ h = 20) :
  let r := (17 * Real.sqrt 2) / 2
  r = (Real.sqrt ((h^2 + (r_bottom - r_top)^2)) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_tangent_to_truncated_cone_l3218_321823


namespace NUMINAMATH_CALUDE_womens_doubles_handshakes_l3218_321857

/-- The number of handshakes in a women's doubles tennis tournament --/
theorem womens_doubles_handshakes :
  let num_teams : ℕ := 4
  let team_size : ℕ := 2
  let total_players : ℕ := num_teams * team_size
  let handshakes_per_player : ℕ := total_players - team_size
  total_players * handshakes_per_player / 2 = 24 := by
  sorry

end NUMINAMATH_CALUDE_womens_doubles_handshakes_l3218_321857


namespace NUMINAMATH_CALUDE_union_complement_A_with_B_l3218_321838

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {2, 5}
def B : Set Nat := {1, 3, 5}

theorem union_complement_A_with_B :
  (U \ A) ∪ B = {1, 3, 4, 5} := by sorry

end NUMINAMATH_CALUDE_union_complement_A_with_B_l3218_321838


namespace NUMINAMATH_CALUDE_total_weight_calculation_l3218_321815

/-- The molecular weight of a compound in grams per mole -/
def molecular_weight : ℝ := 72

/-- The number of moles of the compound -/
def number_of_moles : ℝ := 4

/-- The total weight of the compound in grams -/
def total_weight : ℝ := molecular_weight * number_of_moles

theorem total_weight_calculation : total_weight = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_calculation_l3218_321815


namespace NUMINAMATH_CALUDE_milk_water_ratio_problem_l3218_321896

/-- Given two vessels with volumes in ratio 3:5, where the first vessel has a milk to water ratio
    of 1:2, and when mixed the overall milk to water ratio is 1:1, prove that the milk to water
    ratio in the second vessel must be 3:2. -/
theorem milk_water_ratio_problem (v : ℝ) (x y : ℝ) (h_x_pos : x > 0) (h_y_pos : y > 0) : 
  (1 : ℝ) + (5 * x) / (x + y) = (2 : ℝ) + (5 * y) / (x + y) → x / y = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_milk_water_ratio_problem_l3218_321896


namespace NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l3218_321836

theorem tan_value_from_sin_plus_cos (α : Real) 
  (h1 : α ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_sin_plus_cos_l3218_321836


namespace NUMINAMATH_CALUDE_range_of_a_l3218_321872

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - a| < 1}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 2}

-- Define the complement of B
def complementB : Set ℝ := {x : ℝ | x ∉ B}

-- Theorem statement
theorem range_of_a (a : ℝ) :
  A a ⊆ complementB → 3 ≤ a ∧ a ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3218_321872


namespace NUMINAMATH_CALUDE_star_sqrt3_minus_one_minus_sqrt7_l3218_321878

/-- Custom operation ※ -/
def star (a b : ℝ) : ℝ := (a + 1)^2 - b^2

/-- Theorem stating that (√3-1)※(-√7) = -4 -/
theorem star_sqrt3_minus_one_minus_sqrt7 :
  star (Real.sqrt 3 - 1) (-Real.sqrt 7) = -4 := by
  sorry

end NUMINAMATH_CALUDE_star_sqrt3_minus_one_minus_sqrt7_l3218_321878


namespace NUMINAMATH_CALUDE_simplify_fraction_simplify_and_evaluate_evaluate_at_two_l3218_321824

-- Part 1
theorem simplify_fraction (x : ℝ) (hx : x ≠ 0 ∧ x ≠ 1) :
  (x - 1) / x / ((2 * x - 2) / x^2) = x / 2 := by sorry

-- Part 2
theorem simplify_and_evaluate (a : ℝ) (ha : a ≠ -1) :
  (2 - (a - 1) / (a + 1)) / ((a^2 + 6*a + 9) / (a + 1)) = 1 / (a + 3) := by sorry

theorem evaluate_at_two :
  (2 - (2 - 1) / (2 + 1)) / ((2^2 + 6*2 + 9) / (2 + 1)) = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_simplify_and_evaluate_evaluate_at_two_l3218_321824


namespace NUMINAMATH_CALUDE_cost_per_page_is_five_cents_l3218_321807

def manuscript_copies : ℕ := 10
def binding_cost : ℚ := 5
def pages_per_manuscript : ℕ := 400
def total_cost : ℚ := 250

theorem cost_per_page_is_five_cents :
  let total_binding_cost : ℚ := manuscript_copies * binding_cost
  let copying_cost : ℚ := total_cost - total_binding_cost
  let total_pages : ℕ := manuscript_copies * pages_per_manuscript
  let cost_per_page : ℚ := copying_cost / total_pages
  cost_per_page = 5 / 100 := by sorry

end NUMINAMATH_CALUDE_cost_per_page_is_five_cents_l3218_321807


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l3218_321843

/-- The complex number z defined as (3 + i)i -/
def z : ℂ := (3 + Complex.I) * Complex.I

/-- Predicate to check if a complex number is in the second quadrant -/
def is_in_second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

/-- Theorem stating that z is in the second quadrant -/
theorem z_in_second_quadrant : is_in_second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l3218_321843


namespace NUMINAMATH_CALUDE_invitation_methods_l3218_321809

def total_teachers : ℕ := 10
def invited_teachers : ℕ := 6

theorem invitation_methods (total : ℕ) (invited : ℕ) : 
  total = total_teachers → invited = invited_teachers →
  (Nat.choose total invited) - (Nat.choose (total - 2) (invited - 2)) = 140 := by
  sorry

end NUMINAMATH_CALUDE_invitation_methods_l3218_321809


namespace NUMINAMATH_CALUDE_distance_P_to_x_axis_l3218_321861

/-- The distance from a point to the x-axis in a Cartesian coordinate system --/
def distance_to_x_axis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system --/
def P : ℝ × ℝ := (2, -3)

/-- Theorem: The distance from point P(2, -3) to the x-axis is 3 --/
theorem distance_P_to_x_axis :
  distance_to_x_axis P.2 = 3 := by sorry

end NUMINAMATH_CALUDE_distance_P_to_x_axis_l3218_321861


namespace NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_is_196_l3218_321854

theorem rectangle_area_difference : ℕ → Prop :=
  fun diff =>
    ∀ l w : ℕ,
      (l > 0 ∧ w > 0) →  -- Ensure positive side lengths
      (2 * l + 2 * w = 60) →  -- Perimeter condition
      ∃ l_max w_max l_min w_min : ℕ,
        (l_max > 0 ∧ w_max > 0 ∧ l_min > 0 ∧ w_min > 0) →
        (2 * l_max + 2 * w_max = 60) →
        (2 * l_min + 2 * w_min = 60) →
        (∀ l' w' : ℕ, (l' > 0 ∧ w' > 0) → (2 * l' + 2 * w' = 60) → 
          l' * w' ≤ l_max * w_max ∧ l' * w' ≥ l_min * w_min) →
        diff = l_max * w_max - l_min * w_min

theorem rectangle_area_difference_is_196 : rectangle_area_difference 196 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_rectangle_area_difference_is_196_l3218_321854


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3218_321879

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem fifth_term_of_arithmetic_sequence 
  (a d : ℤ) 
  (h1 : arithmetic_sequence a d 10 = 15) 
  (h2 : arithmetic_sequence a d 11 = 18) : 
  arithmetic_sequence a d 5 = 0 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l3218_321879


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3218_321887

theorem imaginary_part_of_z (z : ℂ) (h : z + z * Complex.I = 2) : 
  z.im = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3218_321887


namespace NUMINAMATH_CALUDE_prob_less_than_five_and_even_is_one_third_l3218_321876

/-- The probability of rolling a number less than 5 on a six-sided die -/
def prob_less_than_five : ℚ := 4 / 6

/-- The probability of rolling an even number on a six-sided die -/
def prob_even : ℚ := 3 / 6

/-- The probability of rolling a number less than 5 on the first die
    and an even number on the second die -/
def prob_less_than_five_and_even : ℚ := prob_less_than_five * prob_even

theorem prob_less_than_five_and_even_is_one_third :
  prob_less_than_five_and_even = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_prob_less_than_five_and_even_is_one_third_l3218_321876


namespace NUMINAMATH_CALUDE_line_passes_through_point_l3218_321856

/-- A line in the form kx - y + 1 - 3k = 0 always passes through (3, 1) -/
theorem line_passes_through_point :
  ∀ (k : ℝ), (3 * k : ℝ) - 1 + 1 - 3 * k = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_point_l3218_321856


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3218_321844

/-- A line is tangent to an ellipse if and only if it intersects the ellipse at exactly one point. -/
axiom tangent_iff_single_intersection {m : ℝ} :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) ↔
  (∃ x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6 ∧
    ∀ x' y' : ℝ, y' = m * x' + 2 ∧ 3 * x'^2 + 6 * y'^2 = 6 → x' = x ∧ y' = y)

/-- The theorem stating that if the line y = mx + 2 is tangent to the ellipse 3x^2 + 6y^2 = 6,
    then m^2 = 3/2. -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∃! x y : ℝ, y = m * x + 2 ∧ 3 * x^2 + 6 * y^2 = 6) → m^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3218_321844


namespace NUMINAMATH_CALUDE_n2o3_molecular_weight_is_76_02_l3218_321877

/-- The atomic weight of nitrogen in g/mol -/
def nitrogen_weight : ℝ := 14.01

/-- The atomic weight of oxygen in g/mol -/
def oxygen_weight : ℝ := 16.00

/-- The number of nitrogen atoms in N2O3 -/
def nitrogen_count : ℕ := 2

/-- The number of oxygen atoms in N2O3 -/
def oxygen_count : ℕ := 3

/-- The molecular weight of N2O3 in g/mol -/
def n2o3_molecular_weight : ℝ := nitrogen_weight * nitrogen_count + oxygen_weight * oxygen_count

/-- Theorem stating that the molecular weight of N2O3 is 76.02 g/mol -/
theorem n2o3_molecular_weight_is_76_02 : 
  n2o3_molecular_weight = 76.02 := by sorry

end NUMINAMATH_CALUDE_n2o3_molecular_weight_is_76_02_l3218_321877


namespace NUMINAMATH_CALUDE_complex_division_problem_l3218_321862

theorem complex_division_problem : (2 - I) / (2 + I) = 3/5 - 4/5 * I :=
by sorry

end NUMINAMATH_CALUDE_complex_division_problem_l3218_321862


namespace NUMINAMATH_CALUDE_equation_solutions_l3218_321883

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 2*x = 1 ↔ x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2) ∧
  (∀ x : ℝ, x^2 + 5*x + 6 = 0 ↔ x = -2 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3218_321883


namespace NUMINAMATH_CALUDE_platter_total_is_26_l3218_321870

/-- Represents the number of fruits of each type in the initial set --/
structure InitialFruits :=
  (green_apples : ℕ)
  (red_apples : ℕ)
  (yellow_apples : ℕ)
  (red_oranges : ℕ)
  (yellow_oranges : ℕ)
  (green_kiwis : ℕ)
  (purple_grapes : ℕ)
  (green_grapes : ℕ)

/-- Represents the ratio of apples in the platter --/
structure AppleRatio :=
  (green : ℕ)
  (red : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of fruits in the platter --/
def calculate_platter_total (initial : InitialFruits) (ratio : AppleRatio) : ℕ :=
  let green_apples := ratio.green
  let red_apples := ratio.red
  let yellow_apples := ratio.yellow
  let red_oranges := 1
  let yellow_oranges := 2
  let kiwis_and_grapes := min initial.green_kiwis initial.purple_grapes
  green_apples + red_apples + yellow_apples + red_oranges + yellow_oranges + 2 * kiwis_and_grapes

/-- Theorem stating that the total number of fruits in the platter is 26 --/
theorem platter_total_is_26 (initial : InitialFruits) (ratio : AppleRatio) : 
  initial.green_apples = 2 →
  initial.red_apples = 3 →
  initial.yellow_apples = 14 →
  initial.red_oranges = 4 →
  initial.yellow_oranges = 8 →
  initial.green_kiwis = 10 →
  initial.purple_grapes = 7 →
  initial.green_grapes = 5 →
  ratio.green = 2 →
  ratio.red = 4 →
  ratio.yellow = 3 →
  calculate_platter_total initial ratio = 26 := by
  sorry


end NUMINAMATH_CALUDE_platter_total_is_26_l3218_321870


namespace NUMINAMATH_CALUDE_polynomial_division_proof_l3218_321837

theorem polynomial_division_proof (x : ℚ) : 
  (3 * x^3 + 3 * x^2 - x - 2/3) * (3 * x + 5) + (-2/3) = 
  9 * x^4 + 18 * x^3 + 8 * x^2 - 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_proof_l3218_321837


namespace NUMINAMATH_CALUDE_inequality_proof_l3218_321885

def f (x : ℝ) := |2*x - 1| + |2*x + 1|

theorem inequality_proof (a b : ℝ) :
  (∀ x, -1 < x ∧ x < 1 → f x < 4) →
  -1 < a ∧ a < 1 →
  -1 < b ∧ b < 1 →
  |a + b| / |a*b + 1| < 1 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3218_321885


namespace NUMINAMATH_CALUDE_xyz_value_l3218_321825

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 16)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 4) :
  x * y * z = 4 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3218_321825


namespace NUMINAMATH_CALUDE_ball_attendees_l3218_321897

theorem ball_attendees :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l3218_321897


namespace NUMINAMATH_CALUDE_correct_average_mark_l3218_321888

theorem correct_average_mark (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 30 →
  initial_avg = 100 →
  wrong_mark = 70 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = 98 * n :=
by sorry

end NUMINAMATH_CALUDE_correct_average_mark_l3218_321888


namespace NUMINAMATH_CALUDE_square_side_length_equal_area_l3218_321831

/-- The side length of a square with the same area as a rectangle with length 18 and width 8 is 12 -/
theorem square_side_length_equal_area (length width : ℝ) (x : ℝ) :
  length = 18 →
  width = 8 →
  x ^ 2 = length * width →
  x = 12 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_equal_area_l3218_321831


namespace NUMINAMATH_CALUDE_dog_food_cup_weight_l3218_321891

/-- The weight of a cup of dog food in pounds -/
def cup_weight : ℝ := 0.25

/-- The number of dogs -/
def num_dogs : ℕ := 2

/-- The number of cups of dog food consumed by each dog per day -/
def cups_per_dog_per_day : ℕ := 12

/-- The number of days in a month -/
def days_per_month : ℕ := 30

/-- The number of bags of dog food bought per month -/
def bags_per_month : ℕ := 9

/-- The weight of each bag of dog food in pounds -/
def bag_weight : ℝ := 20

/-- Theorem stating that the weight of a cup of dog food is 0.25 pounds -/
theorem dog_food_cup_weight :
  cup_weight = (bags_per_month * bag_weight) / (num_dogs * cups_per_dog_per_day * days_per_month) :=
by sorry

end NUMINAMATH_CALUDE_dog_food_cup_weight_l3218_321891


namespace NUMINAMATH_CALUDE_task_probability_l3218_321850

/-- The probability that task 1 is completed on time -/
def prob_task1 : ℚ := 5/8

/-- The probability that task 2 is completed on time -/
def prob_task2 : ℚ := 3/5

/-- The probability that task 1 is completed on time but task 2 is not -/
def prob_task1_not_task2 : ℚ := prob_task1 * (1 - prob_task2)

theorem task_probability : prob_task1_not_task2 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_task_probability_l3218_321850


namespace NUMINAMATH_CALUDE_selection_methods_count_l3218_321817

-- Define the total number of students
def total_students : ℕ := 5

-- Define the number of students needed for each role
def translation_students : ℕ := 2
def transportation_students : ℕ := 1
def protocol_students : ℕ := 1

-- Define the total number of students to be selected
def selected_students : ℕ := translation_students + transportation_students + protocol_students

-- The theorem to be proved
theorem selection_methods_count : 
  (Nat.choose total_students translation_students) * 
  (Nat.choose (total_students - translation_students) transportation_students) * 
  (Nat.choose (total_students - translation_students - transportation_students) protocol_students) = 60 := by
  sorry


end NUMINAMATH_CALUDE_selection_methods_count_l3218_321817


namespace NUMINAMATH_CALUDE_associates_hired_l3218_321892

theorem associates_hired (initial_ratio_partners : ℕ) (initial_ratio_associates : ℕ)
  (current_partners : ℕ) (new_ratio_partners : ℕ) (new_ratio_associates : ℕ) :
  initial_ratio_partners = 2 →
  initial_ratio_associates = 63 →
  current_partners = 14 →
  new_ratio_partners = 1 →
  new_ratio_associates = 34 →
  ∃ (current_associates : ℕ) (hired_associates : ℕ),
    current_associates * initial_ratio_partners = current_partners * initial_ratio_associates ∧
    (current_associates + hired_associates) * new_ratio_partners = current_partners * new_ratio_associates ∧
    hired_associates = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_associates_hired_l3218_321892


namespace NUMINAMATH_CALUDE_quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l3218_321801

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_perpendicular_diagonals (q : Quadrilateral) : Prop := sorry
def has_bisecting_diagonals (q : Quadrilateral) : Prop := sorry
def is_square (q : Quadrilateral) : Prop := sorry

-- Theorem statement
theorem quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square :
  ∃ q : Quadrilateral, 
    has_perpendicular_diagonals q ∧ 
    has_bisecting_diagonals q ∧ 
    ¬ is_square q :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_with_perpendicular_bisecting_diagonals_not_necessarily_square_l3218_321801


namespace NUMINAMATH_CALUDE_nested_sqrt_equality_l3218_321818

theorem nested_sqrt_equality : Real.sqrt (64 * Real.sqrt (32 * Real.sqrt 16)) = 16 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_equality_l3218_321818


namespace NUMINAMATH_CALUDE_total_shells_count_l3218_321875

def purple_shells : ℕ := 13
def pink_shells : ℕ := 8
def yellow_shells : ℕ := 18
def blue_shells : ℕ := 12
def orange_shells : ℕ := 14

theorem total_shells_count :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_shells_count_l3218_321875


namespace NUMINAMATH_CALUDE_min_value_quadratic_min_value_attainable_l3218_321859

theorem min_value_quadratic (a b : ℝ) : a^2 + a*b + b^2 - a - 2*b ≥ -1 := by sorry

theorem min_value_attainable : ∃ (a b : ℝ), a^2 + a*b + b^2 - a - 2*b = -1 := by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_min_value_attainable_l3218_321859


namespace NUMINAMATH_CALUDE_range_of_a_l3218_321849

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 2 → a ≤ x + 1 / (x - 2)) → 
  ∃ s : ℝ, s = 4 ∧ ∀ y : ℝ, (∀ x : ℝ, x > 2 → y ≤ x + 1 / (x - 2)) → y ≤ s :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3218_321849


namespace NUMINAMATH_CALUDE_complex_inequality_implies_real_range_l3218_321842

theorem complex_inequality_implies_real_range (a : ℝ) :
  let z : ℂ := 3 + a * I
  (Complex.abs (z - 2) < 2) → (-Real.sqrt 3 < a ∧ a < Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_implies_real_range_l3218_321842


namespace NUMINAMATH_CALUDE_exclusive_or_implies_possible_p_true_q_false_l3218_321812

theorem exclusive_or_implies_possible_p_true_q_false (P Q : Prop) 
  (h1 : P ∨ Q) (h2 : ¬(P ∧ Q)) : 
  ∃ (p q : Prop), p = P ∧ q = Q ∧ p = true ∧ q = false :=
sorry

end NUMINAMATH_CALUDE_exclusive_or_implies_possible_p_true_q_false_l3218_321812


namespace NUMINAMATH_CALUDE_equivalent_discount_l3218_321855

theorem equivalent_discount (p : ℝ) (k : ℝ) : 
  (1 - k) * p = (1 - 0.05) * (1 - 0.10) * (1 - 0.15) * p ↔ k = 0.27325 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_discount_l3218_321855


namespace NUMINAMATH_CALUDE_range_of_m_for_inequality_l3218_321819

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- State the theorem
theorem range_of_m_for_inequality (h : ∀ x ≤ 5, f x ≤ 3) :
  {m : ℝ | ∀ x, f x + (x + 5) ≥ m} = Set.Iic 5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_inequality_l3218_321819


namespace NUMINAMATH_CALUDE_product_of_reciprocals_plus_one_ge_nine_l3218_321841

theorem product_of_reciprocals_plus_one_ge_nine (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  (1/a + 1) * (1/b + 1) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocals_plus_one_ge_nine_l3218_321841


namespace NUMINAMATH_CALUDE_mailman_theorem_l3218_321803

def mailman_problem (mails_per_block : ℕ) (houses_per_block : ℕ) : Prop :=
  mails_per_block / houses_per_block = 8

theorem mailman_theorem :
  mailman_problem 32 4 :=
by
  sorry

end NUMINAMATH_CALUDE_mailman_theorem_l3218_321803


namespace NUMINAMATH_CALUDE_cody_money_theorem_l3218_321884

def cody_money_problem (initial_money birthday_gift game_price discount friend_debt : ℝ) : Prop :=
  let total_before_purchase := initial_money + birthday_gift
  let discount_amount := game_price * discount
  let actual_game_cost := game_price - discount_amount
  let money_after_purchase := total_before_purchase - actual_game_cost
  let final_amount := money_after_purchase + friend_debt
  final_amount = 48.90

theorem cody_money_theorem :
  cody_money_problem 45 9 19 0.1 12 := by
  sorry

end NUMINAMATH_CALUDE_cody_money_theorem_l3218_321884


namespace NUMINAMATH_CALUDE_equation_solution_l3218_321880

theorem equation_solution : ∃! x : ℚ, x + 5/8 = 1/4 - 2/5 + 7/10 ∧ x = -3/40 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3218_321880


namespace NUMINAMATH_CALUDE_power_one_plus_power_five_quotient_l3218_321806

theorem power_one_plus_power_five_quotient (n : ℕ) :
  (1 : ℕ)^345 + 5^10 / 5^7 = 126 := by
  sorry

end NUMINAMATH_CALUDE_power_one_plus_power_five_quotient_l3218_321806


namespace NUMINAMATH_CALUDE_existence_of_x_l3218_321863

/-- A sequence of nonnegative integers satisfying the given conditions -/
def ValidSequence (a : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i ≥ 1 → j ≥ 1 → i + j ≤ 1997 →
    a i + a j ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1

/-- The theorem to be proved -/
theorem existence_of_x (a : ℕ → ℕ) (h : ValidSequence a) :
  ∃ x : ℝ, ∀ n : ℕ, 1 ≤ n → n ≤ 1997 → a n = ⌊n * x⌋ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_x_l3218_321863


namespace NUMINAMATH_CALUDE_files_remaining_l3218_321847

theorem files_remaining (music_files : ℕ) (video_files : ℕ) (deleted_files : ℕ)
  (h1 : music_files = 4)
  (h2 : video_files = 21)
  (h3 : deleted_files = 23) :
  music_files + video_files - deleted_files = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_files_remaining_l3218_321847


namespace NUMINAMATH_CALUDE_division_problem_l3218_321853

theorem division_problem (dividend quotient remainder : ℕ) (h : dividend = 162 ∧ quotient = 9 ∧ remainder = 9) :
  ∃ (divisor : ℕ), dividend = divisor * quotient + remainder ∧ divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3218_321853


namespace NUMINAMATH_CALUDE_marbles_ratio_l3218_321800

/-- Proves that the ratio of marbles given to Savanna to Miriam's current marbles is 3:1 -/
theorem marbles_ratio (initial : ℕ) (current : ℕ) (brother : ℕ) (sister : ℕ) 
  (h1 : initial = 300)
  (h2 : current = 30)
  (h3 : brother = 60)
  (h4 : sister = 2 * brother) : 
  (initial - current - brother - sister) / current = 3 := by
  sorry

end NUMINAMATH_CALUDE_marbles_ratio_l3218_321800


namespace NUMINAMATH_CALUDE_ellipse_equation_circle_diameter_property_l3218_321848

-- Define the ellipse C
def ellipse (a b x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the conditions
structure EllipseConditions (a b c : ℝ) :=
  (a_gt_b : a > b)
  (b_gt_zero : b > 0)
  (perimeter : 2*c + 2*a = 6)
  (focal_distance : 2*c*b = a*b)
  (pythagoras : a^2 = b^2 + c^2)

-- Theorem for part 1
theorem ellipse_equation (a b c : ℝ) (h : EllipseConditions a b c) :
  a = 2 ∧ b = Real.sqrt 3 ∧ c = 1 :=
sorry

-- Theorem for part 2
theorem circle_diameter_property (m : ℝ) :
  let a := 2
  let b := Real.sqrt 3
  ∀ x₀ y₀ : ℝ, 
    ellipse a b x₀ y₀ → 
    x₀ ≠ 2 → 
    x₀ ≠ -2 → 
    (m - 2) * (x₀ - 2) + (y₀^2 / (x₀ + 2)) * (m + 2) = 0 →
    m = 14 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_circle_diameter_property_l3218_321848


namespace NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l3218_321804

theorem cos_five_pi_sixth_plus_alpha (α : ℝ) (h : Real.sin (π / 3 + α) = 1 / 3) :
  Real.cos (5 * π / 6 + α) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_cos_five_pi_sixth_plus_alpha_l3218_321804


namespace NUMINAMATH_CALUDE_polynomial_property_l3218_321893

/-- Given a polynomial P(x) = ax^2 + bx + c where a, b, c are real numbers,
    if P(a) = bc, P(b) = ac, and P(c) = ab, then (a - b)(b - c)(c - a)(a + b + c) = 0 -/
theorem polynomial_property (a b c : ℝ) (P : ℝ → ℝ)
  (h_poly : ∀ x, P x = a * x^2 + b * x + c)
  (h_Pa : P a = b * c)
  (h_Pb : P b = a * c)
  (h_Pc : P c = a * b) :
  (a - b) * (b - c) * (c - a) * (a + b + c) = 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l3218_321893


namespace NUMINAMATH_CALUDE_petyas_journey_contradiction_l3218_321898

theorem petyas_journey_contradiction (S T : ℝ) (hS : S > 0) (hT : T > 0) : 
  ¬(∃ (S T : ℝ), 
    S / 2 = 4 * (T / 2) ∧ 
    S / 2 = 5 * (T / 2)) :=
by
  sorry

end NUMINAMATH_CALUDE_petyas_journey_contradiction_l3218_321898


namespace NUMINAMATH_CALUDE_mean_equality_implies_y_value_l3218_321865

theorem mean_equality_implies_y_value :
  let mean1 := (6 + 9 + 18) / 3
  let mean2 := (12 + y) / 2
  mean1 = mean2 → y = 10 := by
sorry

end NUMINAMATH_CALUDE_mean_equality_implies_y_value_l3218_321865


namespace NUMINAMATH_CALUDE_simplify_fraction_l3218_321802

theorem simplify_fraction : (48 : ℚ) / 72 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3218_321802


namespace NUMINAMATH_CALUDE_line_relationships_l3218_321827

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Defines when two lines are parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Defines when two lines coincide -/
def coincide (l1 l2 : Line2D) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ l1.a = k * l2.a ∧ l1.b = k * l2.b ∧ l1.c = k * l2.c

/-- Defines when two lines are perpendicular -/
def perpendicular (l1 l2 : Line2D) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- The main theorem about the relationships between two specific lines -/
theorem line_relationships (m : ℝ) :
  let l1 : Line2D := ⟨m + 3, 4, 3*m - 5⟩
  let l2 : Line2D := ⟨2, m + 5, -8⟩
  (parallel l1 l2 ↔ m = -7) ∧
  (coincide l1 l2 ↔ m = -1) ∧
  (perpendicular l1 l2 ↔ m = -13/3) := by
  sorry

end NUMINAMATH_CALUDE_line_relationships_l3218_321827


namespace NUMINAMATH_CALUDE_unique_solution_for_xy_equation_l3218_321882

theorem unique_solution_for_xy_equation :
  ∀ x y : ℤ, x > y ∧ y > 0 ∧ x + y + x * y = 99 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_xy_equation_l3218_321882


namespace NUMINAMATH_CALUDE_part_one_part_two_l3218_321811

-- Define the function f
def f (a x : ℝ) : ℝ := |x - a|

-- Part I
theorem part_one (x : ℝ) : 
  let a := 2
  (x ≤ -1/2 ∨ x ≥ 7/2) ↔ f a x ≥ 4 - |x - 1| :=
sorry

-- Part II
theorem part_two (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f 1 x ≤ 1 ↔ 0 ≤ x ∧ x ≤ 2) →
  1/m + 1/(2*n) = 1 →
  m + 2*n ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3218_321811


namespace NUMINAMATH_CALUDE_min_value_cubic_expression_l3218_321874

theorem min_value_cubic_expression (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^3 + y^3 - 5*x*y ≥ -125/27 := by sorry

end NUMINAMATH_CALUDE_min_value_cubic_expression_l3218_321874


namespace NUMINAMATH_CALUDE_toms_cat_surgery_savings_l3218_321808

/-- Calculates the savings made by having insurance for a pet's surgery --/
def calculate_insurance_savings (
  insurance_duration : ℕ
  ) (insurance_monthly_cost : ℝ
  ) (procedure_cost : ℝ
  ) (insurance_coverage_percentage : ℝ
  ) : ℝ :=
  let total_insurance_cost := insurance_duration * insurance_monthly_cost
  let out_of_pocket_cost := procedure_cost * (1 - insurance_coverage_percentage)
  let total_cost_with_insurance := out_of_pocket_cost + total_insurance_cost
  procedure_cost - total_cost_with_insurance

/-- Theorem stating that the savings made by having insurance for Tom's cat surgery is $3520 --/
theorem toms_cat_surgery_savings :
  calculate_insurance_savings 24 20 5000 0.8 = 3520 := by
  sorry

end NUMINAMATH_CALUDE_toms_cat_surgery_savings_l3218_321808


namespace NUMINAMATH_CALUDE_three_in_A_not_in_B_l3218_321805

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define the complement of A in U
def complement_A : Finset Nat := {2, 4}

-- Define the complement of B in U
def complement_B : Finset Nat := {3, 4}

-- Define set A
def A : Finset Nat := U \ complement_A

-- Define set B
def B : Finset Nat := U \ complement_B

-- Theorem statement
theorem three_in_A_not_in_B : 3 ∈ A ∧ 3 ∉ B := by
  sorry

end NUMINAMATH_CALUDE_three_in_A_not_in_B_l3218_321805


namespace NUMINAMATH_CALUDE_sixteen_divisors_problem_l3218_321860

theorem sixteen_divisors_problem (n : ℕ+) : 
  (∃ (d : Fin 16 → ℕ+), 
    (∀ i : Fin 16, d i ∣ n) ∧ 
    (∀ i j : Fin 16, i < j → d i < d j) ∧
    (d 0 = 1) ∧ 
    (d 15 = n) ∧ 
    (d 5 = 18) ∧ 
    (d 8 - d 7 = 17) ∧
    (∀ m : ℕ+, m ∣ n → ∃ i : Fin 16, d i = m)) →
  n = 1998 ∨ n = 3834 := by
sorry

end NUMINAMATH_CALUDE_sixteen_divisors_problem_l3218_321860


namespace NUMINAMATH_CALUDE_xy_reciprocal_problem_l3218_321852

theorem xy_reciprocal_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h1 : x * y = 1) (h2 : x / y = 36) : y = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_xy_reciprocal_problem_l3218_321852


namespace NUMINAMATH_CALUDE_average_score_five_students_l3218_321845

theorem average_score_five_students
  (initial_students : Nat)
  (initial_average : ℝ)
  (fifth_student_score : ℝ)
  (h1 : initial_students = 4)
  (h2 : initial_average = 85)
  (h3 : fifth_student_score = 90) :
  (initial_students * initial_average + fifth_student_score) / (initial_students + 1) = 86 :=
by sorry

end NUMINAMATH_CALUDE_average_score_five_students_l3218_321845


namespace NUMINAMATH_CALUDE_parabola_point_slope_l3218_321828

/-- A point on a parabola with given properties -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x
  distance_to_focus : Real.sqrt ((x - 1)^2 + y^2) = 5

/-- The theorem stating the absolute value of the slope -/
theorem parabola_point_slope (P : ParabolaPoint) : 
  |((P.y - 0) / (P.x - 1))| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_slope_l3218_321828


namespace NUMINAMATH_CALUDE_total_spent_calculation_l3218_321846

-- Define currency exchange rates
def gbp_to_usd : ℝ := 1.38
def eur_to_usd : ℝ := 1.12
def jpy_to_usd : ℝ := 0.0089

-- Define purchases
def speakers_cost : ℝ := 136.01
def cd_player_cost : ℝ := 139.38
def tires_cost_gbp : ℝ := 85.62
def tires_quantity : ℕ := 4
def printer_cables_cost_eur : ℝ := 12.54
def printer_cables_quantity : ℕ := 2
def blank_cds_cost_jpy : ℝ := 9800

-- Define sales tax rate
def sales_tax_rate : ℝ := 0.0825

-- Theorem statement
theorem total_spent_calculation :
  let usd_taxable := speakers_cost + cd_player_cost
  let usd_tax := usd_taxable * sales_tax_rate
  let usd_with_tax := usd_taxable + usd_tax
  let tires_usd := (tires_cost_gbp * tires_quantity) * gbp_to_usd
  let cables_usd := (printer_cables_cost_eur * printer_cables_quantity) * eur_to_usd
  let cds_usd := blank_cds_cost_jpy * jpy_to_usd
  usd_with_tax + tires_usd + cables_usd + cds_usd = 886.04 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_calculation_l3218_321846


namespace NUMINAMATH_CALUDE_rational_cube_sum_ratio_l3218_321890

theorem rational_cube_sum_ratio (r : ℚ) (hr : 0 < r) :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  r = (a^3 + b^3 : ℚ) / (c^3 + d^3 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_rational_cube_sum_ratio_l3218_321890


namespace NUMINAMATH_CALUDE_gummy_bear_problem_l3218_321851

/-- Calculates the number of gummy bear candies left to be shared with others. -/
def candies_left_to_share (initial_candies : ℕ) (siblings : ℕ) (candies_per_sibling : ℕ) (candies_to_eat : ℕ) : ℕ :=
  let remaining_after_siblings := initial_candies - siblings * candies_per_sibling
  let remaining_after_friend := remaining_after_siblings / 2
  remaining_after_friend - candies_to_eat

/-- Theorem stating that given the problem conditions, 19 candies are left to be shared. -/
theorem gummy_bear_problem :
  candies_left_to_share 100 3 10 16 = 19 := by
  sorry

end NUMINAMATH_CALUDE_gummy_bear_problem_l3218_321851


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3218_321821

/-- Given a line with equation 5x - 2y = 10, 
    the slope of a perpendicular line is -2/5 -/
theorem perpendicular_slope (x y : ℝ) : 
  (5 * x - 2 * y = 10) → 
  (∃ m : ℝ, m = -2/5 ∧ m * (5/2) = -1) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3218_321821


namespace NUMINAMATH_CALUDE_beetle_speed_l3218_321832

/-- Given an ant's average speed and a beetle that walks 10% less distance in the same time,
    prove that the beetle's speed is 1.8 km/h. -/
theorem beetle_speed (ant_distance : ℝ) (time : ℝ) (beetle_percentage : ℝ) :
  ant_distance = 1000 →
  time = 30 →
  beetle_percentage = 0.9 →
  let beetle_distance := ant_distance * beetle_percentage
  let beetle_speed_mpm := beetle_distance / time
  let beetle_speed_kmh := beetle_speed_mpm * 2 * 0.001
  beetle_speed_kmh = 1.8 := by
sorry

end NUMINAMATH_CALUDE_beetle_speed_l3218_321832


namespace NUMINAMATH_CALUDE_student_congress_sample_size_l3218_321839

/-- Represents a school with classes and students -/
structure School where
  num_classes : Nat
  students_per_class : Nat
  selected_students : Nat

/-- Defines the sample size for a given school -/
def sample_size (s : School) : Nat :=
  s.selected_students

/-- Theorem: The sample size for a school with 40 classes of 50 students each,
    and 150 selected students, is 150 -/
theorem student_congress_sample_size :
  let s : School := { num_classes := 40, students_per_class := 50, selected_students := 150 }
  sample_size s = 150 := by
  sorry


end NUMINAMATH_CALUDE_student_congress_sample_size_l3218_321839


namespace NUMINAMATH_CALUDE_car_distance_l3218_321864

theorem car_distance (x : ℝ) (h : 12 * x = 10 * (x + 2)) : 12 * x = 120 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l3218_321864


namespace NUMINAMATH_CALUDE_birds_count_l3218_321867

/-- The number of birds on the fence -/
def birds : ℕ := sorry

/-- Ten more than twice the number of birds on the fence is 50 -/
axiom birds_condition : 10 + 2 * birds = 50

/-- Prove that the number of birds on the fence is 20 -/
theorem birds_count : birds = 20 := by sorry

end NUMINAMATH_CALUDE_birds_count_l3218_321867


namespace NUMINAMATH_CALUDE_max_k_for_tangent_line_l3218_321830

/-- The maximum value of k for which the line y = kx - 2 has at least one point 
    where a line tangent to the circle x^2 + y^2 = 1 can be drawn -/
theorem max_k_for_tangent_line : 
  ∃ (k : ℝ), ∀ (k' : ℝ), 
    (∃ (x y : ℝ), y = k' * x - 2 ∧ 
      ∃ (m : ℝ), (y - m * x)^2 = (1 + m^2) * (1 - x^2)) → 
    k' ≤ k ∧ 
    k = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_max_k_for_tangent_line_l3218_321830


namespace NUMINAMATH_CALUDE_specific_conference_games_l3218_321820

/-- Calculates the number of games in a sports conference season -/
def conference_games (total_teams : ℕ) (division_size : ℕ) (intra_division_games : ℕ) : ℕ :=
  let teams_per_division := total_teams / 2
  let inter_division_games := division_size
  let games_per_team := (division_size - 1) * intra_division_games + inter_division_games
  (games_per_team * total_teams) / 2

/-- Theorem stating the number of games in the specific conference setup -/
theorem specific_conference_games :
  conference_games 16 8 2 = 176 := by
  sorry

end NUMINAMATH_CALUDE_specific_conference_games_l3218_321820


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_condition_l3218_321858

/-- If the ratios of sine and cosine of angles to their opposite sides are equal in a triangle, then it is an isosceles right triangle. -/
theorem isosceles_right_triangle_condition (A B C : Real) (a b c : Real) :
  (A + B + C = Real.pi) →
  (a > 0) →
  (b > 0) →
  (c > 0) →
  ((Real.sin A) / a = (Real.cos B) / b) →
  ((Real.sin A) / a = (Real.cos C) / c) →
  ((Real.cos B) / b = (Real.cos C) / c) →
  (A = Real.pi / 2 ∧ B = Real.pi / 4 ∧ C = Real.pi / 4) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_condition_l3218_321858


namespace NUMINAMATH_CALUDE_E_Z_eq_l3218_321822

variable (p : ℝ)

-- Assumption that p is a probability
axiom h_p_prob : 0 < p ∧ p < 1

-- Definition of the probability mass function for Y
def P_Y (k : ℕ) : ℝ := p * (1 - p) ^ (k - 1)

-- Definition of the probability mass function for Z
def P_Z (k : ℕ) : ℝ := 
  if k ≥ 2 then p * (1 - p) ^ (k - 1) + (1 - p) * p ^ (k - 1) else 0

-- Expected value of Y
axiom E_Y : ∑' k, k * P_Y p k = 1 / p

-- Theorem to prove
theorem E_Z_eq : ∑' k, k * P_Z p k = 1 / (p * (1 - p)) - 1 := by sorry

end NUMINAMATH_CALUDE_E_Z_eq_l3218_321822


namespace NUMINAMATH_CALUDE_equal_division_of_cakes_l3218_321889

theorem equal_division_of_cakes (total_cakes : ℕ) (num_children : ℕ) (cakes_per_child : ℕ) :
  total_cakes = 18 →
  num_children = 3 →
  total_cakes = num_children * cakes_per_child →
  cakes_per_child = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_division_of_cakes_l3218_321889


namespace NUMINAMATH_CALUDE_lamp_post_height_l3218_321868

/-- The height of a lamp post given specific conditions --/
theorem lamp_post_height (cable_ground_distance : ℝ) (person_distance : ℝ) (person_height : ℝ)
  (h1 : cable_ground_distance = 4)
  (h2 : person_distance = 3)
  (h3 : person_height = 1.6)
  (h4 : person_distance < cable_ground_distance) :
  ∃ (post_height : ℝ),
    post_height = (cable_ground_distance * person_height) / (cable_ground_distance - person_distance) ∧
    post_height = 6.4 := by
  sorry

end NUMINAMATH_CALUDE_lamp_post_height_l3218_321868


namespace NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3218_321834

theorem tic_tac_toe_tie_probability (ben_win_prob tom_win_prob tie_prob : ℚ) : 
  ben_win_prob = 1/4 → tom_win_prob = 2/5 → tie_prob = 1 - (ben_win_prob + tom_win_prob) → 
  tie_prob = 7/20 := by
  sorry

end NUMINAMATH_CALUDE_tic_tac_toe_tie_probability_l3218_321834


namespace NUMINAMATH_CALUDE_annual_growth_rate_l3218_321816

theorem annual_growth_rate (initial : ℝ) (final : ℝ) (years : ℕ) (x : ℝ) 
  (h1 : initial = 1000000)
  (h2 : final = 1690000)
  (h3 : years = 2)
  (h4 : x > 0)
  (h5 : (1 + x)^years = final / initial) :
  x = 0.3 := by
sorry

end NUMINAMATH_CALUDE_annual_growth_rate_l3218_321816


namespace NUMINAMATH_CALUDE_prime_factors_of_1998_l3218_321866

theorem prime_factors_of_1998 (a b c : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  a < b ∧ b < c ∧
  a * b * c = 1998 →
  (b + c)^a = 1600 := by
  sorry

end NUMINAMATH_CALUDE_prime_factors_of_1998_l3218_321866


namespace NUMINAMATH_CALUDE_consecutive_product_divisibility_l3218_321826

theorem consecutive_product_divisibility (k : ℤ) :
  let n := k * (k + 1) * (k + 2)
  (∃ m : ℤ, n = 7 * m) →
  (∃ m : ℤ, n = 6 * m) ∧
  (∃ m : ℤ, n = 14 * m) ∧
  (∃ m : ℤ, n = 21 * m) ∧
  (∃ m : ℤ, n = 42 * m) ∧
  ¬(∀ k : ℤ, ∃ m : ℤ, k * (k + 1) * (k + 2) = 28 * m) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_product_divisibility_l3218_321826


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l3218_321895

theorem cubic_equation_roots : ∃! (p n₁ n₂ : ℝ), 
  p > 0 ∧ n₁ < 0 ∧ n₂ < 0 ∧
  p^3 + 3*p^2 - 4*p + 12 = 0 ∧
  n₁^3 + 3*n₁^2 - 4*n₁ + 12 = 0 ∧
  n₂^3 + 3*n₂^2 - 4*n₂ + 12 = 0 ∧
  p ≠ n₁ ∧ p ≠ n₂ ∧ n₁ ≠ n₂ :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l3218_321895


namespace NUMINAMATH_CALUDE_constant_term_value_l3218_321881

theorem constant_term_value (x y : ℝ) (C : ℝ) : 
  5 * x + y = C →
  x + 3 * y = 1 →
  3 * x + 2 * y = 10 →
  C = 19 :=
by
  sorry

end NUMINAMATH_CALUDE_constant_term_value_l3218_321881


namespace NUMINAMATH_CALUDE_not_all_products_of_two_primes_l3218_321873

theorem not_all_products_of_two_primes (q : ℕ) (hq : Nat.Prime q) (hodd : Odd q) :
  ∃ k : ℕ, k ∈ Finset.range (q - 1) ∧ ¬∃ p₁ p₂ : ℕ, Nat.Prime p₁ ∧ Nat.Prime p₂ ∧ k^2 + k + q = p₁ * p₂ := by
  sorry

end NUMINAMATH_CALUDE_not_all_products_of_two_primes_l3218_321873


namespace NUMINAMATH_CALUDE_simplify_expression_l3218_321869

theorem simplify_expression (y : ℝ) : 2*y + 8*y^2 + 6 - (3 - 2*y - 8*y^2) = 16*y^2 + 4*y + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3218_321869
