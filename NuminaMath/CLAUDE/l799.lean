import Mathlib

namespace NUMINAMATH_CALUDE_least_multiple_of_13_greater_than_450_l799_79998

theorem least_multiple_of_13_greater_than_450 :
  (∀ n : ℕ, n * 13 > 450 → n * 13 ≥ 455) ∧ 455 % 13 = 0 ∧ 455 > 450 := by
  sorry

end NUMINAMATH_CALUDE_least_multiple_of_13_greater_than_450_l799_79998


namespace NUMINAMATH_CALUDE_min_value_cos_sum_l799_79907

theorem min_value_cos_sum (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ π/2) 
  (hy : 0 ≤ y ∧ y ≤ π/2) (hz : 0 ≤ z ∧ z ≤ π/2) :
  Real.cos (x - y) + Real.cos (y - z) + Real.cos (z - x) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_cos_sum_l799_79907


namespace NUMINAMATH_CALUDE_sacks_per_section_l799_79990

/-- Given an orchard with 8 sections that produces 360 sacks of apples daily,
    prove that each section produces 45 sacks per day. -/
theorem sacks_per_section (sections : ℕ) (total_sacks : ℕ) (h1 : sections = 8) (h2 : total_sacks = 360) :
  total_sacks / sections = 45 := by
  sorry

end NUMINAMATH_CALUDE_sacks_per_section_l799_79990


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l799_79986

/-- An isosceles triangle with congruent sides of length 6 and perimeter 20 has a base of length 8 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let congruent_side := 6
    let perimeter := 20
    (2 * congruent_side + base = perimeter) → base = 8

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 8 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l799_79986


namespace NUMINAMATH_CALUDE_k_range_l799_79962

-- Define the function h
def h (x : ℝ) : ℝ := 5 * x - 3

-- Define the function k as a composition of h
def k (x : ℝ) : ℝ := h (h (h x))

-- State the theorem
theorem k_range :
  ∀ x ∈ Set.Icc (-1 : ℝ) 3,
  ∃ y ∈ Set.Icc (-218 : ℝ) 282,
  k x = y ∧
  ∀ z, k x = z → z ∈ Set.Icc (-218 : ℝ) 282 :=
sorry

end NUMINAMATH_CALUDE_k_range_l799_79962


namespace NUMINAMATH_CALUDE_toms_reading_speed_l799_79955

/-- Tom's reading speed problem -/
theorem toms_reading_speed (normal_speed : ℕ) : 
  (2 * (3 * normal_speed) = 72) → normal_speed = 12 := by
  sorry

#check toms_reading_speed

end NUMINAMATH_CALUDE_toms_reading_speed_l799_79955


namespace NUMINAMATH_CALUDE_increasing_subsequence_exists_l799_79928

/-- Given a sequence of 2^n positive integers where each element is at most its index,
    there exists a monotonically increasing subsequence of length n+1. -/
theorem increasing_subsequence_exists (n : ℕ) (a : Fin (2^n) → ℕ)
  (h : ∀ k : Fin (2^n), a k ≤ k.val + 1) :
  ∃ (s : Fin (n + 1) → Fin (2^n)), Monotone (a ∘ s) :=
sorry

end NUMINAMATH_CALUDE_increasing_subsequence_exists_l799_79928


namespace NUMINAMATH_CALUDE_tony_bread_slices_left_l799_79932

/-- The number of slices of bread Tony uses in a week -/
def bread_used (weekday_slices : ℕ) (saturday_slices : ℕ) (sunday_slices : ℕ) : ℕ :=
  5 * weekday_slices + saturday_slices + sunday_slices

/-- The number of slices left from a loaf -/
def slices_left (total_slices : ℕ) (used_slices : ℕ) : ℕ :=
  total_slices - used_slices

/-- Theorem stating the number of slices left from Tony's bread usage -/
theorem tony_bread_slices_left :
  let weekday_slices := 2
  let saturday_slices := 5
  let sunday_slices := 1
  let total_slices := 22
  let used_slices := bread_used weekday_slices saturday_slices sunday_slices
  slices_left total_slices used_slices = 6 := by
  sorry

end NUMINAMATH_CALUDE_tony_bread_slices_left_l799_79932


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l799_79952

theorem exam_maximum_marks :
  let pass_percentage : ℚ := 45 / 100
  let fail_score : ℕ := 180
  let fail_margin : ℕ := 45
  let max_marks : ℕ := 500
  (pass_percentage * max_marks = fail_score + fail_margin) ∧
  (pass_percentage * max_marks = (fail_score + fail_margin : ℚ)) :=
by sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l799_79952


namespace NUMINAMATH_CALUDE_min_empty_cells_is_three_l799_79975

/-- Represents a triangular cell arrangement with grasshoppers -/
structure TriangularArrangement where
  up_cells : ℕ  -- Number of upward-pointing cells
  down_cells : ℕ  -- Number of downward-pointing cells
  has_more_up : up_cells = down_cells + 3

/-- The minimum number of empty cells after all grasshoppers have jumped -/
def min_empty_cells (arrangement : TriangularArrangement) : ℕ := 3

/-- Theorem stating that the minimum number of empty cells is always 3 -/
theorem min_empty_cells_is_three (arrangement : TriangularArrangement) :
  min_empty_cells arrangement = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_empty_cells_is_three_l799_79975


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l799_79976

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 4*a*x else (2*a + 3)*x - 4*a + 5

theorem f_increasing_iff_a_in_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ∈ Set.Icc (1/2) (3/2) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_in_range_l799_79976


namespace NUMINAMATH_CALUDE_city_fuel_efficiency_l799_79966

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on the highway
  city : ℝ     -- Miles per gallon in the city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- The conditions given in the problem -/
def problem_conditions (car : CarFuelEfficiency) : Prop :=
  car.highway * car.tank_size = 560 ∧
  car.city * car.tank_size = 336 ∧
  car.city = car.highway - 6

/-- The theorem to be proved -/
theorem city_fuel_efficiency (car : CarFuelEfficiency) :
  problem_conditions car → car.city = 9 := by
  sorry


end NUMINAMATH_CALUDE_city_fuel_efficiency_l799_79966


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l799_79918

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5 :=
sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  (x - 3)^2 + 2*x*(x - 3) = 0 ↔ x = 3 ∨ x = 1 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_l799_79918


namespace NUMINAMATH_CALUDE_candy_distribution_l799_79949

theorem candy_distribution (total_candy : ℕ) (num_bags : ℕ) (candy_per_bag : ℕ) :
  total_candy = 15 →
  num_bags = 5 →
  total_candy = num_bags * candy_per_bag →
  candy_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l799_79949


namespace NUMINAMATH_CALUDE_no_solution_when_k_is_seven_l799_79942

theorem no_solution_when_k_is_seven (k : ℝ) (h : k = 7) :
  ¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 5 ∧ (x^2 - 1) / (x - 3) = (x^2 - k) / (x - 5) :=
by
  sorry

end NUMINAMATH_CALUDE_no_solution_when_k_is_seven_l799_79942


namespace NUMINAMATH_CALUDE_complement_implies_set_l799_79993

def U : Set ℕ := {1, 3, 5, 7}

theorem complement_implies_set (M : Set ℕ) : 
  U = {1, 3, 5, 7} → (U \ M = {5, 7}) → M = {1, 3} := by
  sorry

end NUMINAMATH_CALUDE_complement_implies_set_l799_79993


namespace NUMINAMATH_CALUDE_cosine_difference_l799_79922

theorem cosine_difference (α β : Real) 
  (h1 : Real.sin α - Real.sin β = 1/2) 
  (h2 : Real.cos α - Real.cos β = 1/3) : 
  Real.cos (α - β) = 59/72 := by
  sorry

end NUMINAMATH_CALUDE_cosine_difference_l799_79922


namespace NUMINAMATH_CALUDE_temperature_conversion_l799_79931

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → k = 122 → t = 50 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l799_79931


namespace NUMINAMATH_CALUDE_largest_integer_quadratic_negative_l799_79909

theorem largest_integer_quadratic_negative : 
  (∀ m : ℤ, m > 7 → m^2 - 11*m + 24 ≥ 0) ∧ 
  (7^2 - 11*7 + 24 < 0) := by
sorry

end NUMINAMATH_CALUDE_largest_integer_quadratic_negative_l799_79909


namespace NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l799_79967

/-- Converts a base 9 number to base 10 --/
def base9ToBase10 (n : Nat) : Nat :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * 9^2 + tens * 9 + ones

/-- Checks if a number is a valid 3-digit base 9 number --/
def isValidBase9 (n : Nat) : Prop :=
  100 ≤ n ∧ n ≤ 888

theorem greatest_3digit_base9_divisible_by_7 :
  ∃ (n : Nat), isValidBase9 n ∧ 
               base9ToBase10 n % 7 = 0 ∧
               ∀ (m : Nat), isValidBase9 m ∧ base9ToBase10 m % 7 = 0 → base9ToBase10 m ≤ base9ToBase10 n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_3digit_base9_divisible_by_7_l799_79967


namespace NUMINAMATH_CALUDE_quadratic_equations_unique_solution_l799_79947

/-- Given two quadratic equations and their solution sets, prove the coefficients -/
theorem quadratic_equations_unique_solution :
  ∀ (p q r : ℝ),
  let A := {x : ℝ | x^2 + p*x - 2 = 0}
  let B := {x : ℝ | x^2 + q*x + r = 0}
  (A ∪ B = {-2, 1, 5}) →
  (A ∩ B = {-2}) →
  (p = -1 ∧ q = -3 ∧ r = -10) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_unique_solution_l799_79947


namespace NUMINAMATH_CALUDE_parabola_focus_point_slope_l799_79900

/-- The slope of a line between the focus of a parabola and a point on the parabola -/
theorem parabola_focus_point_slope (x y : ℝ) :
  y^2 = 4*x →  -- parabola equation
  x > 0 →  -- point is in the fourth quadrant
  y < 0 →  -- point is in the fourth quadrant
  x + 1 = 5 →  -- distance from point to directrix is 5
  (y - 0) / (x - 1) = -4/3 :=  -- slope of line AF
by sorry

end NUMINAMATH_CALUDE_parabola_focus_point_slope_l799_79900


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l799_79941

/-- The arc length of a sector with radius 8 cm and central angle 45° is 2π cm. -/
theorem arc_length_of_sector (r : ℝ) (θ_deg : ℝ) (l : ℝ) : 
  r = 8 → θ_deg = 45 → l = r * (θ_deg * π / 180) → l = 2 * π := by
  sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l799_79941


namespace NUMINAMATH_CALUDE_sector_perimeter_l799_79946

/-- Given a circular sector with area 2 cm² and central angle 4 radians, its perimeter is 6 cm. -/
theorem sector_perimeter (A : ℝ) (α : ℝ) (P : ℝ) : 
  A = 2 → α = 4 → P = 2 * Real.sqrt (2 / α) + Real.sqrt (2 / α) * α → P = 6 := by
  sorry

end NUMINAMATH_CALUDE_sector_perimeter_l799_79946


namespace NUMINAMATH_CALUDE_expression_simplification_l799_79930

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (1 + 1 / (x - 2)) / ((x^2 - 2*x + 1) / (x - 2)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l799_79930


namespace NUMINAMATH_CALUDE_skateboard_padding_cost_increase_l799_79982

/-- Calculates the percent increase in the combined cost of a skateboard and padding set. -/
theorem skateboard_padding_cost_increase 
  (skateboard_cost : ℝ) 
  (padding_cost : ℝ) 
  (skateboard_increase : ℝ) 
  (padding_increase : ℝ) : 
  skateboard_cost = 120 →
  padding_cost = 30 →
  skateboard_increase = 0.08 →
  padding_increase = 0.15 →
  let new_skateboard_cost := skateboard_cost * (1 + skateboard_increase)
  let new_padding_cost := padding_cost * (1 + padding_increase)
  let original_total := skateboard_cost + padding_cost
  let new_total := new_skateboard_cost + new_padding_cost
  (new_total - original_total) / original_total = 0.094 := by
  sorry

end NUMINAMATH_CALUDE_skateboard_padding_cost_increase_l799_79982


namespace NUMINAMATH_CALUDE_sqrt_division_equality_l799_79919

theorem sqrt_division_equality : Real.sqrt 3 / Real.sqrt 5 = Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_division_equality_l799_79919


namespace NUMINAMATH_CALUDE_range_of_c_sum_of_squares_inequality_l799_79973

-- Part I
theorem range_of_c (c : ℝ) (h1 : c > 0) 
  (h2 : ∀ x : ℝ, x + |x - 2*c| ≥ 2) : c ≥ 1 := by
  sorry

-- Part II
theorem sum_of_squares_inequality (p q r : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) 
  (h_sum : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_sum_of_squares_inequality_l799_79973


namespace NUMINAMATH_CALUDE_power_series_identity_l799_79933

/-- Given that (1 - hx)⁻¹ (1 - kx)⁻¹ = ∑(i≥0) aᵢ xⁱ, 
    prove that (1 + hkx)(1 - hkx)⁻¹ (1 - h²x)⁻¹ (1 - k²x)⁻¹ = ∑(i≥0) aᵢ² xⁱ -/
theorem power_series_identity 
  (h k : ℝ) (x : ℝ) (a : ℕ → ℝ) :
  (∀ x, (1 - h*x)⁻¹ * (1 - k*x)⁻¹ = ∑' i, a i * x^i) →
  (1 + h*k*x) * (1 - h*k*x)⁻¹ * (1 - h^2*x)⁻¹ * (1 - k^2*x)⁻¹ = ∑' i, (a i)^2 * x^i :=
by
  sorry

end NUMINAMATH_CALUDE_power_series_identity_l799_79933


namespace NUMINAMATH_CALUDE_lara_baking_cookies_l799_79963

/-- The number of baking trays Lara is using. -/
def num_trays : ℕ := 4

/-- The number of rows of cookies on each tray. -/
def rows_per_tray : ℕ := 5

/-- The number of cookies in each row. -/
def cookies_per_row : ℕ := 6

/-- The total number of cookies Lara is baking. -/
def total_cookies : ℕ := num_trays * rows_per_tray * cookies_per_row

theorem lara_baking_cookies : total_cookies = 120 := by
  sorry

end NUMINAMATH_CALUDE_lara_baking_cookies_l799_79963


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l799_79985

theorem quadratic_equation_roots : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (x₁^2 - 4*x₁ - 4 = 0) ∧ (x₂^2 - 4*x₂ - 4 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l799_79985


namespace NUMINAMATH_CALUDE_polynomial_expansion_l799_79968

theorem polynomial_expansion (x : ℝ) : 
  (7 * x + 3) * (5 * x^2 + 4) = 35 * x^3 + 15 * x^2 + 28 * x + 12 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l799_79968


namespace NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l799_79953

/-- 
Given a current speed and a man's speed against the current,
calculate the man's speed with the current.
-/
def mans_speed_with_current (current_speed : ℝ) (speed_against_current : ℝ) : ℝ :=
  speed_against_current + 2 * current_speed

/-- 
Theorem: Given a current speed of 2.5 km/hr and a speed against the current of 10 km/hr,
the man's speed with the current is 15 km/hr.
-/
theorem mans_speed_with_current_is_15 :
  mans_speed_with_current 2.5 10 = 15 := by
  sorry

#eval mans_speed_with_current 2.5 10

end NUMINAMATH_CALUDE_mans_speed_with_current_is_15_l799_79953


namespace NUMINAMATH_CALUDE_overlap_area_theorem_l799_79992

/-- Represents a square sheet of paper -/
structure Sheet :=
  (side : ℝ)
  (rotation : ℝ)

/-- Calculates the area of overlap between three rotated square sheets -/
def area_of_overlap (s1 s2 s3 : Sheet) : ℝ :=
  sorry

/-- The theorem stating the area of overlap for the given problem -/
theorem overlap_area_theorem :
  let s1 : Sheet := { side := 8, rotation := 0 }
  let s2 : Sheet := { side := 8, rotation := 45 }
  let s3 : Sheet := { side := 8, rotation := 90 }
  area_of_overlap s1 s2 s3 = 96 :=
sorry

end NUMINAMATH_CALUDE_overlap_area_theorem_l799_79992


namespace NUMINAMATH_CALUDE_negative_eight_interpretations_l799_79979

theorem negative_eight_interpretations :
  (-(- 8) = -(-8)) ∧
  (-(- 8) = -1 * (-8)) ∧
  (-(- 8) = |-8|) ∧
  (-(- 8) = 8) :=
by sorry

end NUMINAMATH_CALUDE_negative_eight_interpretations_l799_79979


namespace NUMINAMATH_CALUDE_right_triangle_area_l799_79914

theorem right_triangle_area (hypotenuse : ℝ) (angle : ℝ) : 
  hypotenuse = 10 * Real.sqrt 2 →
  angle = 45 * (π / 180) →
  (1 / 2) * (hypotenuse / Real.sqrt 2) * (hypotenuse / Real.sqrt 2) = 50 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l799_79914


namespace NUMINAMATH_CALUDE_circle_angle_sum_l799_79915

theorem circle_angle_sum (a b : ℝ) : 
  a + b + 110 + 60 = 360 → a + b = 190 := by
  sorry

end NUMINAMATH_CALUDE_circle_angle_sum_l799_79915


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l799_79945

theorem inverse_variation_problem (y z : ℝ) (h1 : y^4 * z^(1/4) = 162) (h2 : y = 6) : z = 1/4096 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l799_79945


namespace NUMINAMATH_CALUDE_quadratic_solution_l799_79916

theorem quadratic_solution (b : ℝ) : 
  ((-10 : ℝ)^2 + b * (-10) - 30 = 0) → b = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l799_79916


namespace NUMINAMATH_CALUDE_train_speed_proof_l799_79984

/-- Proves that a train with given parameters has a specific speed -/
theorem train_speed_proof (train_length : ℝ) (crossing_time : ℝ) (total_length : ℝ) :
  train_length = 150 →
  crossing_time = 30 →
  total_length = 225 →
  (total_length - train_length) / crossing_time * 3.6 = 45 := by
  sorry

#check train_speed_proof

end NUMINAMATH_CALUDE_train_speed_proof_l799_79984


namespace NUMINAMATH_CALUDE_star_equation_solution_l799_79956

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ :=
  a * Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem star_equation_solution :
  ∃ y : ℝ, star 3 y = 18 ∧ y = 30 := by
sorry

end NUMINAMATH_CALUDE_star_equation_solution_l799_79956


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l799_79959

theorem cubic_equation_solutions : 
  ∀ m n : ℤ, m^3 - n^3 = 2*m*n + 8 ↔ (m = 2 ∧ n = 0) ∨ (m = 0 ∧ n = -2) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l799_79959


namespace NUMINAMATH_CALUDE_milkshake_cost_calculation_l799_79996

/-- The cost of a milkshake given initial money, cupcake spending fraction, and remaining money --/
def milkshake_cost (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) : ℚ :=
  initial - initial * cupcake_fraction - remaining

theorem milkshake_cost_calculation (initial : ℚ) (cupcake_fraction : ℚ) (remaining : ℚ) 
  (h1 : initial = 10)
  (h2 : cupcake_fraction = 1/5)
  (h3 : remaining = 3) :
  milkshake_cost initial cupcake_fraction remaining = 5 := by
  sorry

#eval milkshake_cost 10 (1/5) 3

end NUMINAMATH_CALUDE_milkshake_cost_calculation_l799_79996


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l799_79908

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)
variable (parallel_plane_plane : Plane → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)

-- Theorem 1
theorem perpendicular_lines_from_perpendicular_planes
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : perpendicular_line_plane n β)
  (h3 : perpendicular_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

-- Theorem 2
theorem perpendicular_lines_from_perpendicular_parallel
  (m n : Line) (α β : Plane)
  (h1 : perpendicular_line_plane m α)
  (h2 : parallel_line_plane n β)
  (h3 : parallel_plane_plane α β) :
  perpendicular_line_line m n :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_perpendicular_lines_from_perpendicular_parallel_l799_79908


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l799_79934

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l799_79934


namespace NUMINAMATH_CALUDE_oldest_bride_age_l799_79958

theorem oldest_bride_age (bride_age groom_age : ℕ) : 
  bride_age = groom_age + 19 →
  bride_age + groom_age = 185 →
  bride_age = 102 := by
sorry

end NUMINAMATH_CALUDE_oldest_bride_age_l799_79958


namespace NUMINAMATH_CALUDE_max_lines_theorem_l799_79961

/-- Given n points on a plane where no three are collinear, 
    this function returns the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle with vertices 
    among the given points. -/
def max_lines_without_triangle (n : ℕ) : ℕ :=
  if n % 2 = 0 then
    n^2 / 4
  else
    (n^2 - 1) / 4

/-- Theorem stating the maximum number of lines that can be drawn 
    through pairs of points without forming a triangle, 
    given n points on a plane where no three are collinear and n ≥ 3. -/
theorem max_lines_theorem (n : ℕ) (h : n ≥ 3) :
  max_lines_without_triangle n = 
    if n % 2 = 0 then
      n^2 / 4
    else
      (n^2 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_max_lines_theorem_l799_79961


namespace NUMINAMATH_CALUDE_printer_pages_theorem_l799_79940

/-- Represents a printer with specific crumpling and blurring patterns -/
structure Printer where
  crumple_interval : Nat
  blur_interval : Nat

/-- Calculates the number of pages that are neither crumpled nor blurred -/
def good_pages (p : Printer) (total : Nat) : Nat :=
  total - (total / p.crumple_interval + total / p.blur_interval - total / (Nat.lcm p.crumple_interval p.blur_interval))

/-- Theorem: For a printer that crumples every 7th page and blurs every 3rd page,
    if 24 pages are neither crumpled nor blurred, then 42 pages were printed in total -/
theorem printer_pages_theorem (p : Printer) (h1 : p.crumple_interval = 7) (h2 : p.blur_interval = 3) :
  good_pages p 42 = 24 := by
  sorry

#eval good_pages ⟨7, 3⟩ 42  -- Should output 24

end NUMINAMATH_CALUDE_printer_pages_theorem_l799_79940


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l799_79927

theorem complex_magnitude_problem (z : ℂ) (h : (1 + 2*I)*z = 5) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l799_79927


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l799_79969

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_length : ℝ) : 
  total_length = 120 →
  difference = 22 →
  total_length = shorter_length + (shorter_length + difference) →
  shorter_length = 49 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l799_79969


namespace NUMINAMATH_CALUDE_quadratic_trinomial_existence_l799_79944

theorem quadratic_trinomial_existence (a b c : ℤ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (p q r : ℤ), p > 0 ∧ 
    (∀ x : ℤ, p * x^2 + q * x + r = x^3 - (x - a) * (x - b) * (x - c)) ∧
    (p * a^2 + q * a + r = a^3) ∧
    (p * b^2 + q * b + r = b^3) ∧
    (p * c^2 + q * c + r = c^3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_trinomial_existence_l799_79944


namespace NUMINAMATH_CALUDE_extended_pattern_black_tiles_l799_79929

/-- Represents a square pattern of tiles -/
structure SquarePattern :=
  (size : Nat)
  (blackTiles : Nat)

/-- Extends a square pattern by adding a border of black tiles -/
def extendPattern (pattern : SquarePattern) : SquarePattern :=
  { size := pattern.size + 2,
    blackTiles := pattern.blackTiles + (pattern.size + 2) * 4 - 4 }

theorem extended_pattern_black_tiles :
  let originalPattern : SquarePattern := { size := 5, blackTiles := 10 }
  let extendedPattern := extendPattern originalPattern
  extendedPattern.blackTiles = 34 := by
  sorry

end NUMINAMATH_CALUDE_extended_pattern_black_tiles_l799_79929


namespace NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l799_79957

theorem max_value_of_2x_plus_y (x y : ℝ) (h : 3 * x^2 + 2 * y^2 ≤ 6) :
  ∃ (M : ℝ), M = Real.sqrt 11 ∧ 2 * x + y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 3 * x₀^2 + 2 * y₀^2 ≤ 6 ∧ 2 * x₀ + y₀ = M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_2x_plus_y_l799_79957


namespace NUMINAMATH_CALUDE_mean_median_difference_l799_79995

/-- Represents the score distribution of students in a test --/
structure ScoreDistribution where
  score65 : Float
  score75 : Float
  score88 : Float
  score92 : Float
  score100 : Float
  total_percentage : Float
  h_total : total_percentage = score65 + score75 + score88 + score92 + score100

/-- Calculates the median score given a ScoreDistribution --/
def median (sd : ScoreDistribution) : Float :=
  sorry

/-- Calculates the mean score given a ScoreDistribution --/
def mean (sd : ScoreDistribution) : Float :=
  sorry

/-- The main theorem stating the difference between mean and median --/
theorem mean_median_difference (sd : ScoreDistribution) 
  (h1 : sd.score65 = 0.15)
  (h2 : sd.score75 = 0.20)
  (h3 : sd.score88 = 0.25)
  (h4 : sd.score92 = 0.10)
  (h5 : sd.score100 = 0.30)
  (h6 : sd.total_percentage = 1.0) :
  mean sd - median sd = -2 :=
sorry

end NUMINAMATH_CALUDE_mean_median_difference_l799_79995


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l799_79950

theorem imaginary_part_of_z (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (Complex.mk 1 a) = Real.sqrt 5) :
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l799_79950


namespace NUMINAMATH_CALUDE_divisible_by_five_l799_79971

theorem divisible_by_five (a b : ℕ) : 
  (5 ∣ a * b) → (5 ∣ a) ∨ (5 ∣ b) := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_five_l799_79971


namespace NUMINAMATH_CALUDE_unique_solution_abs_equation_l799_79983

theorem unique_solution_abs_equation :
  ∃! x : ℝ, |x - 20| + |x - 18| = |2*x - 36| :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_abs_equation_l799_79983


namespace NUMINAMATH_CALUDE_qin_jiushao_v3_l799_79917

/-- Qin Jiushao algorithm for polynomial evaluation -/
def qin_jiushao (f : ℤ → ℤ) (x : ℤ) : ℕ → ℤ
| 0 => 1
| 1 => qin_jiushao f x 0 * x + 47
| 2 => qin_jiushao f x 1 * x + 0
| 3 => qin_jiushao f x 2 * x - 37
| _ => 0

/-- The polynomial f(x) = x^5 + 47x^4 - 37x^2 + 1 -/
def f (x : ℤ) : ℤ := x^5 + 47*x^4 - 37*x^2 + 1

theorem qin_jiushao_v3 : qin_jiushao f (-1) 3 = 9 := by sorry

end NUMINAMATH_CALUDE_qin_jiushao_v3_l799_79917


namespace NUMINAMATH_CALUDE_repeating_decimal_division_l799_79936

theorem repeating_decimal_division :
  let a : ℚ := 54 / 99
  let b : ℚ := 18 / 99
  a / b = 3 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_division_l799_79936


namespace NUMINAMATH_CALUDE_equation_solution_l799_79903

theorem equation_solution : ∃ x : ℝ, (0.82 : ℝ)^3 - (0.1 : ℝ)^3 / (0.82 : ℝ)^2 + x + (0.1 : ℝ)^2 = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l799_79903


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_l799_79925

-- Define the quadratic equation
def quadratic_equation (m x : ℝ) : ℝ := x^2 - 2*m*x + m^2 - 1

-- Define the condition p
def condition_p (m : ℝ) : Prop := -1 < m ∧ m < 5

-- Define the condition q
def condition_q (m : ℝ) : Prop :=
  ∀ x, quadratic_equation m x = 0 → -2 < x ∧ x < 4

-- Theorem: p is a necessary but not sufficient condition for q
theorem p_necessary_not_sufficient :
  (∀ m, condition_q m → condition_p m) ∧
  ¬(∀ m, condition_p m → condition_q m) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_l799_79925


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l799_79938

theorem quadratic_solution_difference_squared : 
  ∀ Φ φ : ℝ, 
  Φ ≠ φ → 
  Φ^2 - 3*Φ + 1 = 0 → 
  φ^2 - 3*φ + 1 = 0 → 
  (Φ - φ)^2 = 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_squared_l799_79938


namespace NUMINAMATH_CALUDE_rotation_270_of_8_minus_4i_l799_79910

-- Define the rotation function
def rotate270 (z : ℂ) : ℂ := -z.im + z.re * Complex.I

-- State the theorem
theorem rotation_270_of_8_minus_4i :
  rotate270 (8 - 4 * Complex.I) = -4 - 8 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_rotation_270_of_8_minus_4i_l799_79910


namespace NUMINAMATH_CALUDE_square_sum_equals_29_l799_79913

theorem square_sum_equals_29 (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : 
  a^2 + b^2 = 29 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_29_l799_79913


namespace NUMINAMATH_CALUDE_hexagon_sixth_angle_l799_79972

/-- The sum of interior angles in a hexagon -/
def hexagon_angle_sum : ℝ := 720

/-- The five known angles in the hexagon -/
def known_angles : List ℝ := [108, 130, 142, 105, 120]

/-- Theorem: In a hexagon where five of the interior angles measure 108°, 130°, 142°, 105°, and 120°, the measure of the sixth angle is 115°. -/
theorem hexagon_sixth_angle :
  hexagon_angle_sum - (known_angles.sum) = 115 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_sixth_angle_l799_79972


namespace NUMINAMATH_CALUDE_minimum_oranges_l799_79978

theorem minimum_oranges : ∃ n : ℕ, n > 0 ∧ 
  (n % 5 = 1 ∧ n % 7 = 1 ∧ n % 10 = 1) ∧ 
  ∀ m : ℕ, m > 0 → (m % 5 = 1 ∧ m % 7 = 1 ∧ m % 10 = 1) → m ≥ 71 := by
  sorry

end NUMINAMATH_CALUDE_minimum_oranges_l799_79978


namespace NUMINAMATH_CALUDE_last_two_digits_of_sum_l799_79935

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a number encoded as ZARAZA -/
structure Zaraza where
  z : Digit
  a : Digit
  r : Digit
  ne_za : z ≠ a
  ne_zr : z ≠ r
  ne_ar : a ≠ r

/-- Represents a number encoded as ALMAZ -/
structure Almaz where
  a : Digit
  l : Digit
  m : Digit
  z : Digit
  ne_al : a ≠ l
  ne_am : a ≠ m
  ne_az : a ≠ z
  ne_lm : l ≠ m
  ne_lz : l ≠ z
  ne_mz : m ≠ z

/-- Convert Zaraza to a natural number -/
def zarazaToNat (x : Zaraza) : ℕ :=
  x.z.val * 100000 + x.a.val * 10000 + x.r.val * 1000 + x.a.val * 100 + x.z.val * 10 + x.a.val

/-- Convert Almaz to a natural number -/
def almazToNat (x : Almaz) : ℕ :=
  x.a.val * 10000 + x.l.val * 1000 + x.m.val * 100 + x.a.val * 10 + x.z.val

/-- The main theorem -/
theorem last_two_digits_of_sum (zar : Zaraza) (alm : Almaz) 
    (h1 : zarazaToNat zar % 4 = 0)
    (h2 : almazToNat alm % 28 = 0)
    (h3 : zar.z = alm.z ∧ zar.a = alm.a) :
    (zarazaToNat zar + almazToNat alm) % 100 = 32 := by
  sorry


end NUMINAMATH_CALUDE_last_two_digits_of_sum_l799_79935


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_progression_l799_79981

/-- Given an arithmetic progression with the 25th term equal to 173 and a common difference of 7,
    prove that the first term is 5. -/
theorem first_term_of_arithmetic_progression :
  ∀ (a : ℕ → ℤ),
    (∀ n : ℕ, a (n + 1) = a n + 7) →  -- Common difference is 7
    a 25 = 173 →                      -- 25th term is 173
    a 1 = 5 :=                        -- First term is 5
by
  sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_progression_l799_79981


namespace NUMINAMATH_CALUDE_probability_diff_two_meters_l799_79948

def bamboo_lengths : Finset ℕ := {1, 2, 3, 4}

def valid_pairs : Finset (ℕ × ℕ) :=
  {(1, 3), (3, 1), (2, 4), (4, 2)}

def total_pairs : Finset (ℕ × ℕ) :=
  bamboo_lengths.product bamboo_lengths

theorem probability_diff_two_meters :
  (valid_pairs.card : ℚ) / total_pairs.card = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_diff_two_meters_l799_79948


namespace NUMINAMATH_CALUDE_intersection_count_l799_79974

-- Define the lines
def line1 (x y : ℝ) : Prop := 3*x + 4*y - 12 = 0
def line2 (x y : ℝ) : Prop := 5*x - 2*y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem statement
theorem intersection_count :
  ∃ (p1 p2 : ℝ × ℝ), 
    p1 ≠ p2 ∧
    is_intersection p1.1 p1.2 ∧
    is_intersection p2.1 p2.2 ∧
    ∀ (p : ℝ × ℝ), is_intersection p.1 p.2 → p = p1 ∨ p = p2 :=
sorry

end NUMINAMATH_CALUDE_intersection_count_l799_79974


namespace NUMINAMATH_CALUDE_fourth_number_in_sequence_l799_79989

theorem fourth_number_in_sequence (a b c d : ℝ) : 
  a / b = 5 / 3 ∧ 
  b / c = 3 / 4 ∧ 
  a + b + c = 108 ∧ 
  d - c = c - b ∧ 
  c - b = b - a 
  → d = 45 := by
  sorry

end NUMINAMATH_CALUDE_fourth_number_in_sequence_l799_79989


namespace NUMINAMATH_CALUDE_weekly_savings_l799_79988

def hourly_rate_1 : ℚ := 20
def hourly_rate_2 : ℚ := 22
def subsidy : ℚ := 6
def hours_per_week : ℚ := 40

def weekly_cost_1 : ℚ := hourly_rate_1 * hours_per_week
def effective_hourly_rate_2 : ℚ := hourly_rate_2 - subsidy
def weekly_cost_2 : ℚ := effective_hourly_rate_2 * hours_per_week

theorem weekly_savings : weekly_cost_1 - weekly_cost_2 = 160 := by
  sorry

end NUMINAMATH_CALUDE_weekly_savings_l799_79988


namespace NUMINAMATH_CALUDE_sisters_phone_sale_total_l799_79906

def phone_price : ℕ := 400

theorem sisters_phone_sale_total (vivienne_phones aliyah_extra_phones : ℕ) :
  vivienne_phones = 40 →
  aliyah_extra_phones = 10 →
  (vivienne_phones + (vivienne_phones + aliyah_extra_phones)) * phone_price = 36000 :=
by sorry

end NUMINAMATH_CALUDE_sisters_phone_sale_total_l799_79906


namespace NUMINAMATH_CALUDE_simplify_expression_l799_79905

theorem simplify_expression (c : ℝ) : ((3 * c + 6) - 6 * c) / 3 = -c + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l799_79905


namespace NUMINAMATH_CALUDE_stratified_sampling_size_l799_79904

def workshop_A : ℕ := 120
def workshop_B : ℕ := 80
def workshop_C : ℕ := 60

def total_production : ℕ := workshop_A + workshop_B + workshop_C

def sample_size_C : ℕ := 3

theorem stratified_sampling_size :
  (workshop_C : ℚ) / total_production = sample_size_C / (13 : ℚ) := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_size_l799_79904


namespace NUMINAMATH_CALUDE_no_solution_mod_five_l799_79920

theorem no_solution_mod_five : ¬∃ (n : ℕ), n^2 % 5 = 1 ∧ n^3 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_mod_five_l799_79920


namespace NUMINAMATH_CALUDE_composite_n_fourth_plus_64_l799_79951

theorem composite_n_fourth_plus_64 : ∃ (n : ℕ), ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n^4 + 64 = a * b :=
sorry

end NUMINAMATH_CALUDE_composite_n_fourth_plus_64_l799_79951


namespace NUMINAMATH_CALUDE_initial_volume_calculation_l799_79923

/-- The initial volume of a solution in liters -/
def initial_volume : ℝ := 6

/-- The percentage of alcohol in the initial solution -/
def initial_alcohol_percentage : ℝ := 0.30

/-- The volume of pure alcohol added in liters -/
def added_alcohol : ℝ := 2.4

/-- The percentage of alcohol in the final solution -/
def final_alcohol_percentage : ℝ := 0.50

theorem initial_volume_calculation :
  initial_volume * initial_alcohol_percentage + added_alcohol =
  final_alcohol_percentage * (initial_volume + added_alcohol) :=
by sorry

end NUMINAMATH_CALUDE_initial_volume_calculation_l799_79923


namespace NUMINAMATH_CALUDE_total_purchase_ways_l799_79901

def oreo_flavors : ℕ := 7
def milk_types : ℕ := 4
def total_items : ℕ := 5

def ways_to_purchase : ℕ := sorry

theorem total_purchase_ways :
  ways_to_purchase = 13279 := by sorry

end NUMINAMATH_CALUDE_total_purchase_ways_l799_79901


namespace NUMINAMATH_CALUDE_solve_potato_problem_l799_79997

def potatoesProblem (initialPotatoes : ℕ) (ginaAmount : ℕ) : Prop :=
  let tomAmount := 2 * ginaAmount
  let anneAmount := tomAmount / 3
  let remainingPotatoes := initialPotatoes - (ginaAmount + tomAmount + anneAmount)
  remainingPotatoes = 47

theorem solve_potato_problem :
  potatoesProblem 300 69 := by
  sorry

end NUMINAMATH_CALUDE_solve_potato_problem_l799_79997


namespace NUMINAMATH_CALUDE_high_school_total_students_l799_79911

/-- Represents a high school with three grades and a stratified sampling method. -/
structure HighSchool where
  total_students : ℕ
  senior_students : ℕ
  sample_size : ℕ
  freshman_sample : ℕ
  sophomore_sample : ℕ

/-- The conditions of the problem -/
def problem_conditions : HighSchool where
  senior_students := 600
  sample_size := 90
  freshman_sample := 27
  sophomore_sample := 33
  total_students := 1800  -- This is what we want to prove

theorem high_school_total_students :
  ∀ (hs : HighSchool),
  hs.senior_students = 600 →
  hs.sample_size = 90 →
  hs.freshman_sample = 27 →
  hs.sophomore_sample = 33 →
  hs.total_students = 1800 :=
by
  sorry

#check high_school_total_students

end NUMINAMATH_CALUDE_high_school_total_students_l799_79911


namespace NUMINAMATH_CALUDE_interval_relation_l799_79965

theorem interval_relation : 
  (∀ x : ℝ, 3 < x ∧ x < 4 → 2 < x ∧ x < 5) ∧ 
  (∃ x : ℝ, 2 < x ∧ x < 5 ∧ ¬(3 < x ∧ x < 4)) :=
by sorry

end NUMINAMATH_CALUDE_interval_relation_l799_79965


namespace NUMINAMATH_CALUDE_product_of_numbers_l799_79964

theorem product_of_numbers (x y : ℝ) (sum_eq : x + y = 20) (sum_squares_eq : x^2 + y^2 = 200) :
  x * y = 100 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l799_79964


namespace NUMINAMATH_CALUDE_system_solution_l799_79937

theorem system_solution (x y z t : ℝ) : 
  (x^2 - 9*y^2 = 0 ∧ x + y + z = 0) ↔ 
  ((x = 3*t ∧ y = t ∧ z = -4*t) ∨ (x = -3*t ∧ y = t ∧ z = 2*t)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l799_79937


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l799_79977

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) / (1 - Complex.I) = Complex.I) : 
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l799_79977


namespace NUMINAMATH_CALUDE_solve_equation_l799_79943

theorem solve_equation (y : ℝ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l799_79943


namespace NUMINAMATH_CALUDE_purely_imaginary_z_implies_tan_theta_minus_pi_4_l799_79902

theorem purely_imaginary_z_implies_tan_theta_minus_pi_4 (θ : ℝ) :
  let z : ℂ := (Real.cos θ - 4/5) + (Real.sin θ - 3/5) * Complex.I
  (z.re = 0 ∧ z.im ≠ 0) → Real.tan (θ - π/4) = -7 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_implies_tan_theta_minus_pi_4_l799_79902


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l799_79939

theorem geometric_sequence_problem (a b c : ℝ) :
  (∀ q : ℝ, 1 * q = a ∧ a * q = b ∧ b * q = c ∧ c * q = 4) → b = 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l799_79939


namespace NUMINAMATH_CALUDE_awards_distribution_count_l799_79960

/-- The number of ways to distribute awards to students -/
def distribute_awards (num_awards num_students : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the correct number of ways to distribute 6 awards to 4 students -/
theorem awards_distribution_count :
  distribute_awards 6 4 = 3720 :=
by sorry

end NUMINAMATH_CALUDE_awards_distribution_count_l799_79960


namespace NUMINAMATH_CALUDE_cube_root_of_one_eighth_l799_79987

theorem cube_root_of_one_eighth (x : ℝ) : x^3 = 1/8 → x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_one_eighth_l799_79987


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l799_79921

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * time_s

/-- Proof that a train traveling at 60 km/hr and crossing a pole in 18 seconds has a length of approximately 300.06 meters -/
theorem train_length_proof (ε : ℝ) (h : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧ |train_length 60 18 - 300.06| < δ :=
sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l799_79921


namespace NUMINAMATH_CALUDE_anlu_temperature_difference_l799_79999

/-- Given a temperature range from -3°C to 3°C in Anlu on a winter day,
    the temperature difference is 6°C. -/
theorem anlu_temperature_difference :
  let min_temp : ℤ := -3
  let max_temp : ℤ := 3
  (max_temp - min_temp : ℤ) = 6 := by sorry

end NUMINAMATH_CALUDE_anlu_temperature_difference_l799_79999


namespace NUMINAMATH_CALUDE_contest_sequences_equal_combination_l799_79994

/-- Represents the number of players in each team -/
def team_size : ℕ := 7

/-- Represents the total number of players from both teams -/
def total_players : ℕ := 2 * team_size

/-- Represents the number of different possible sequences of matches in the contest -/
def match_sequences : ℕ := Nat.choose total_players team_size

theorem contest_sequences_equal_combination :
  match_sequences = 3432 := by
  sorry

end NUMINAMATH_CALUDE_contest_sequences_equal_combination_l799_79994


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l799_79912

def arithmetic_sequence (a : ℤ) (d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d

theorem eighth_term_of_sequence 
  (a : ℤ) (d : ℤ) 
  (h1 : arithmetic_sequence a d 4 = 23) 
  (h2 : arithmetic_sequence a d 6 = 47) : 
  arithmetic_sequence a d 8 = 71 := by
sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l799_79912


namespace NUMINAMATH_CALUDE_line_through_points_l799_79980

/-- A line passing through two points (1,3) and (4,-2) can be represented by y = mx + b, where m + b = 3 -/
theorem line_through_points (m b : ℚ) : 
  (3 = m * 1 + b) → (-2 = m * 4 + b) → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l799_79980


namespace NUMINAMATH_CALUDE_johns_payment_ratio_l799_79991

/-- Proves that the ratio of John's payment to the total cost for the first year is 1/2 --/
theorem johns_payment_ratio (
  num_members : ℕ)
  (join_fee : ℕ)
  (monthly_cost : ℕ)
  (johns_payment : ℕ)
  (h1 : num_members = 4)
  (h2 : join_fee = 4000)
  (h3 : monthly_cost = 1000)
  (h4 : johns_payment = 32000)
  : johns_payment / (num_members * (join_fee + 12 * monthly_cost)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_johns_payment_ratio_l799_79991


namespace NUMINAMATH_CALUDE_ratio_from_percentage_l799_79926

theorem ratio_from_percentage (x y : ℝ) (h : y = x * (1 - 0.909090909090909)) :
  x / y = 11 := by
sorry

end NUMINAMATH_CALUDE_ratio_from_percentage_l799_79926


namespace NUMINAMATH_CALUDE_complex_number_existence_l799_79970

theorem complex_number_existence : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧
  ∀ (z : ℂ), Complex.abs z = 1 → (1 + z + z^2 ≠ 0) →
    Complex.abs (Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c)) = d :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l799_79970


namespace NUMINAMATH_CALUDE_unique_desk_arrangement_l799_79924

theorem unique_desk_arrangement (total_desks : ℕ) (h_total : total_desks = 49) :
  ∃! (rows columns : ℕ),
    rows * columns = total_desks ∧
    rows ≥ 2 ∧
    columns ≥ 2 ∧
    (∀ r c : ℕ, r * c = total_desks → r ≥ 2 → c ≥ 2 → r = rows ∧ c = columns) :=
by sorry

end NUMINAMATH_CALUDE_unique_desk_arrangement_l799_79924


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l799_79954

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 6

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The total number of possible words -/
def total_words : ℕ := alphabet_size ^ word_length

/-- The number of words containing only vowels -/
def all_vowel_words : ℕ := vowel_count ^ word_length

/-- The number of words with at least one consonant -/
def words_with_consonant : ℕ := total_words - all_vowel_words

theorem words_with_consonant_count : words_with_consonant = 7744 := by
  sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l799_79954
