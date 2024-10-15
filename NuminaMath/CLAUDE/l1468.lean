import Mathlib

namespace NUMINAMATH_CALUDE_expression_simplification_l1468_146801

theorem expression_simplification (a : ℝ) (h1 : a^2 - 4*a + 3 = 0) (h2 : a ≠ 3) :
  (a^2 - 9) / (a^2 - 3*a) / ((a^2 + 9) / a + 6) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1468_146801


namespace NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1468_146858

/-- Given a regular pentagon with perimeter 60 inches and a rectangle with perimeter 80 inches
    where the length is twice the width, the ratio of the pentagon's side length to the rectangle's
    width is 9/10. -/
theorem pentagon_rectangle_ratio : 
  ∀ (pentagon_side rectangle_width : ℝ),
    pentagon_side * 5 = 60 →
    rectangle_width * 6 = 80 →
    pentagon_side / rectangle_width = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_pentagon_rectangle_ratio_l1468_146858


namespace NUMINAMATH_CALUDE_largest_square_tile_l1468_146891

theorem largest_square_tile (wall_width wall_length : ℕ) 
  (h1 : wall_width = 120) (h2 : wall_length = 96) : 
  Nat.gcd wall_width wall_length = 24 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_tile_l1468_146891


namespace NUMINAMATH_CALUDE_book_chapters_l1468_146849

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) 
  (h1 : total_pages = 1891) 
  (h2 : pages_per_chapter = 61) : 
  total_pages / pages_per_chapter = 31 := by
  sorry

end NUMINAMATH_CALUDE_book_chapters_l1468_146849


namespace NUMINAMATH_CALUDE_cherry_picking_time_l1468_146821

/-- The time spent picking cherries by 王芳 and 李丽 -/
def picking_time : ℝ := 0.25

/-- 王芳's picking rate in kg/hour -/
def wang_rate : ℝ := 8

/-- 李丽's picking rate in kg/hour -/
def li_rate : ℝ := 7

/-- Amount of cherries 王芳 gives to 李丽 after picking -/
def transfer_amount : ℝ := 0.25

theorem cherry_picking_time :
  wang_rate * picking_time - transfer_amount = li_rate * picking_time :=
by sorry

end NUMINAMATH_CALUDE_cherry_picking_time_l1468_146821


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1468_146823

theorem complex_equation_solution (z : ℂ) : z * (1 - I) = 3 - I → z = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1468_146823


namespace NUMINAMATH_CALUDE_cricket_bats_profit_percentage_l1468_146836

/-- Calculate the overall profit percentage for three cricket bats --/
theorem cricket_bats_profit_percentage
  (selling_price_A selling_price_B selling_price_C : ℝ)
  (profit_A profit_B profit_C : ℝ)
  (h1 : selling_price_A = 900)
  (h2 : selling_price_B = 1200)
  (h3 : selling_price_C = 1500)
  (h4 : profit_A = 300)
  (h5 : profit_B = 400)
  (h6 : profit_C = 500) :
  let total_cost_price := (selling_price_A - profit_A) + (selling_price_B - profit_B) + (selling_price_C - profit_C)
  let total_selling_price := selling_price_A + selling_price_B + selling_price_C
  let total_profit := total_selling_price - total_cost_price
  (total_profit / total_cost_price) * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cricket_bats_profit_percentage_l1468_146836


namespace NUMINAMATH_CALUDE_theater_bills_count_l1468_146874

-- Define the problem parameters
def total_tickets : ℕ := 300
def ticket_price : ℕ := 40
def total_revenue : ℕ := total_tickets * ticket_price

-- Define the variables for the number of each type of bill
def num_20_bills : ℕ := 238
def num_10_bills : ℕ := 2 * num_20_bills
def num_5_bills : ℕ := num_10_bills + 20

-- Define the theorem
theorem theater_bills_count :
  -- Conditions
  (20 * num_20_bills + 10 * num_10_bills + 5 * num_5_bills = total_revenue) →
  (num_10_bills = 2 * num_20_bills) →
  (num_5_bills = num_10_bills + 20) →
  -- Conclusion
  (num_20_bills + num_10_bills + num_5_bills = 1210) :=
by sorry

end NUMINAMATH_CALUDE_theater_bills_count_l1468_146874


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1468_146890

-- Define the y-intercept
def y_intercept : ℝ := 8

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 25 * y^2 = 100

-- Define the line equation
def line_eq (m x : ℝ) : ℝ := m * x + y_intercept

-- Theorem statement
theorem line_ellipse_intersection_slopes :
  ∀ m : ℝ, (∃ x : ℝ, ellipse_eq x (line_eq m x)) ↔ m^2 ≥ 2.4 :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l1468_146890


namespace NUMINAMATH_CALUDE_product_of_primes_summing_to_17_l1468_146807

theorem product_of_primes_summing_to_17 (p₁ p₂ p₃ p₄ : ℕ) : 
  p₁.Prime ∧ p₂.Prime ∧ p₃.Prime ∧ p₄.Prime ∧ 
  p₁ + p₂ + p₃ + p₄ = 17 → 
  p₁ * p₂ * p₃ * p₄ = 210 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_summing_to_17_l1468_146807


namespace NUMINAMATH_CALUDE_l_plaque_four_equal_parts_l1468_146879

/-- An L-shaped plaque -/
structure LPlaque where
  width : ℝ
  height : ℝ
  thickness : ℝ

/-- A straight cut on the plaque -/
inductive Cut
  | Vertical (x : ℝ)
  | Horizontal (y : ℝ)

/-- The result of applying cuts to an L-shaped plaque -/
def applyCuts (p : LPlaque) (cuts : List Cut) : List (Set (ℝ × ℝ)) :=
  sorry

/-- Check if all pieces have equal area -/
def equalAreas (pieces : List (Set (ℝ × ℝ))) : Prop :=
  sorry

/-- Main theorem: An L-shaped plaque can be divided into four equal parts using straight cuts -/
theorem l_plaque_four_equal_parts (p : LPlaque) :
  ∃ (cuts : List Cut), (applyCuts p cuts).length = 4 ∧ equalAreas (applyCuts p cuts) :=
sorry

end NUMINAMATH_CALUDE_l_plaque_four_equal_parts_l1468_146879


namespace NUMINAMATH_CALUDE_original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l1468_146873

-- Original proposition
theorem original_proposition (a b : ℝ) : a = b → a^2 = b^2 := by sorry

-- Converse is false
theorem converse_is_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

-- Inverse is false
theorem inverse_is_false : ¬ (∀ a b : ℝ, a ≠ b → a^2 ≠ b^2) := by sorry

-- Contrapositive is true
theorem contrapositive_is_true : ∀ a b : ℝ, a^2 ≠ b^2 → a ≠ b := by sorry

end NUMINAMATH_CALUDE_original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l1468_146873


namespace NUMINAMATH_CALUDE_largest_geometric_three_digit_l1468_146820

/-- Represents a three-digit number as a tuple of its digits -/
def ThreeDigitNumber := (Nat × Nat × Nat)

/-- Checks if all digits in a ThreeDigitNumber are distinct -/
def distinct_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ n.2.1 ∧ n.1 ≠ n.2.2 ∧ n.2.1 ≠ n.2.2

/-- Checks if the digits of a ThreeDigitNumber form a geometric sequence -/
def geometric_sequence (n : ThreeDigitNumber) : Prop :=
  ∃ r : Rat, r ≠ 0 ∧ n.2.1 = n.1 * r ∧ n.2.2 = n.2.1 * r

/-- Checks if a ThreeDigitNumber has no zero digits -/
def no_zero_digits (n : ThreeDigitNumber) : Prop :=
  n.1 ≠ 0 ∧ n.2.1 ≠ 0 ∧ n.2.2 ≠ 0

/-- Converts a ThreeDigitNumber to its integer representation -/
def to_int (n : ThreeDigitNumber) : Nat :=
  100 * n.1 + 10 * n.2.1 + n.2.2

/-- The main theorem stating that 842 is the largest number satisfying all conditions -/
theorem largest_geometric_three_digit :
  ∀ n : ThreeDigitNumber,
    distinct_digits n ∧ 
    geometric_sequence n ∧ 
    no_zero_digits n →
    to_int n ≤ 842 :=
  sorry

end NUMINAMATH_CALUDE_largest_geometric_three_digit_l1468_146820


namespace NUMINAMATH_CALUDE_prism_surface_area_l1468_146845

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  -- The diameter of the sphere
  sphere_diameter : ℝ
  -- The side length of the base of the prism
  base_side_length : ℝ
  -- The height of the prism
  height : ℝ
  -- All vertices are on the sphere surface
  vertices_on_sphere : sphere_diameter^2 = base_side_length^2 + base_side_length^2 + height^2

/-- The surface area of a right square prism -/
def surface_area (p : PrismOnSphere) : ℝ :=
  2 * p.base_side_length^2 + 4 * p.base_side_length * p.height

/-- Theorem: The surface area of the specific prism is 2 + 4√2 -/
theorem prism_surface_area :
  ∃ (p : PrismOnSphere),
    p.sphere_diameter = 2 ∧
    p.base_side_length = 1 ∧
    surface_area p = 2 + 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_prism_surface_area_l1468_146845


namespace NUMINAMATH_CALUDE_tree_arrangement_probability_l1468_146884

def maple_trees : ℕ := 4
def oak_trees : ℕ := 5
def birch_trees : ℕ := 6
def total_trees : ℕ := maple_trees + oak_trees + birch_trees

def valid_arrangements : ℕ := (Nat.choose 7 maple_trees) * 1

def total_arrangements : ℕ := (Nat.factorial total_trees) / 
  (Nat.factorial maple_trees * Nat.factorial oak_trees * Nat.factorial birch_trees)

theorem tree_arrangement_probability : 
  (valid_arrangements : ℚ) / total_arrangements = 7 / 166320 := by sorry

end NUMINAMATH_CALUDE_tree_arrangement_probability_l1468_146884


namespace NUMINAMATH_CALUDE_brothers_combined_age_theorem_l1468_146870

/-- Represents the ages of two brothers -/
structure BrothersAges where
  adam : ℕ
  tom : ℕ

/-- Calculates the number of years until the brothers' combined age reaches a target -/
def yearsUntilCombinedAge (ages : BrothersAges) (targetAge : ℕ) : ℕ :=
  (targetAge - (ages.adam + ages.tom)) / 2

/-- Theorem: The number of years until Adam and Tom's combined age is 44 is 12 -/
theorem brothers_combined_age_theorem (ages : BrothersAges) 
  (h1 : ages.adam = 8) 
  (h2 : ages.tom = 12) : 
  yearsUntilCombinedAge ages 44 = 12 := by
  sorry

end NUMINAMATH_CALUDE_brothers_combined_age_theorem_l1468_146870


namespace NUMINAMATH_CALUDE_floor_pi_minus_e_l1468_146824

theorem floor_pi_minus_e : ⌊π - Real.exp 1⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_floor_pi_minus_e_l1468_146824


namespace NUMINAMATH_CALUDE_paper_length_is_correct_l1468_146835

/-- The length of a rectangular sheet of paper satisfying given conditions -/
def paper_length : ℚ :=
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  (2 * second_sheet_length * second_sheet_width + area_difference) / (2 * width)

theorem paper_length_is_correct :
  let width : ℚ := 9
  let second_sheet_length : ℚ := 11
  let second_sheet_width : ℚ := 9/2
  let area_difference : ℚ := 100
  2 * paper_length * width = 2 * second_sheet_length * second_sheet_width + area_difference :=
by
  sorry

#eval paper_length

end NUMINAMATH_CALUDE_paper_length_is_correct_l1468_146835


namespace NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l1468_146862

/-- The probability of drawing a green or yellow marble from a bag -/
theorem green_or_yellow_marble_probability
  (green : ℕ) (yellow : ℕ) (white : ℕ)
  (h_green : green = 4)
  (h_yellow : yellow = 3)
  (h_white : white = 6) :
  (green + yellow) / (green + yellow + white) = 7 / 13 :=
by sorry

end NUMINAMATH_CALUDE_green_or_yellow_marble_probability_l1468_146862


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l1468_146856

theorem parabola_intersection_distance : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let roots := {x : ℝ | f x = 0}
  ∃ (a b : ℝ), a ∈ roots ∧ b ∈ roots ∧ a ≠ b ∧ |a - b| = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l1468_146856


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1468_146882

theorem complex_equation_sum (a b : ℝ) : 
  (a : ℂ) + b * Complex.I = (1 - Complex.I) * (2 + Complex.I) → a + b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1468_146882


namespace NUMINAMATH_CALUDE_minimize_quadratic_l1468_146866

/-- The quadratic function f(x) = x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 8*x + 7

/-- Theorem stating that -4 minimizes the quadratic function f(x) = x^2 + 8x + 7 for all real x -/
theorem minimize_quadratic :
  ∀ x : ℝ, f (-4) ≤ f x :=
by sorry

end NUMINAMATH_CALUDE_minimize_quadratic_l1468_146866


namespace NUMINAMATH_CALUDE_folded_square_area_ratio_l1468_146826

/-- The ratio of the area of a square paper folded along a line connecting points
    at 1/3 and 2/3 of one side to the area of the original square is 5/6. -/
theorem folded_square_area_ratio (s : ℝ) (h : s > 0) : 
  let A := s^2
  let B := s^2 - (1/2 * (s/3) * s)
  B / A = 5/6 := by sorry

end NUMINAMATH_CALUDE_folded_square_area_ratio_l1468_146826


namespace NUMINAMATH_CALUDE_paper_torn_fraction_l1468_146883

theorem paper_torn_fraction (perimeter : ℝ) (remaining_area : ℝ) : 
  perimeter = 32 → remaining_area = 48 → 
  (perimeter / 4)^2 - remaining_area = (1 / 4) * (perimeter / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_paper_torn_fraction_l1468_146883


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_l1468_146842

theorem sqrt_two_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_l1468_146842


namespace NUMINAMATH_CALUDE_time_fraction_proof_l1468_146804

/-- Given a 24-hour day and the current time being 6, 
    prove that the fraction of time left to time already completed is 3. -/
theorem time_fraction_proof : 
  let hours_in_day : ℕ := 24
  let current_time : ℕ := 6
  let time_left : ℕ := hours_in_day - current_time
  let time_completed : ℕ := current_time
  (time_left : ℚ) / time_completed = 3 := by
  sorry

end NUMINAMATH_CALUDE_time_fraction_proof_l1468_146804


namespace NUMINAMATH_CALUDE_system_solution_l1468_146899

theorem system_solution :
  ∃ (x y : ℚ), (4 * x = -10 - 3 * y) ∧ (6 * x = 5 * y - 32) ∧ (x = -73/19) ∧ (y = 34/19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1468_146899


namespace NUMINAMATH_CALUDE_josh_remaining_money_l1468_146855

/-- Calculates the remaining money after Josh's shopping trip. -/
def remaining_money (initial_amount hat_cost pencil_cost cookie_cost cookie_count : ℚ) : ℚ :=
  initial_amount - (hat_cost + pencil_cost + cookie_cost * cookie_count)

/-- Theorem stating that Josh has $3 left after his shopping trip. -/
theorem josh_remaining_money :
  remaining_money 20 10 2 1.25 4 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l1468_146855


namespace NUMINAMATH_CALUDE_pyramid_frustum_volume_ratio_l1468_146812

theorem pyramid_frustum_volume_ratio : 
  let base_edge : ℝ := 24
  let altitude : ℝ := 18
  let small_altitude : ℝ := altitude / 3
  let original_volume : ℝ := (1 / 3) * (base_edge ^ 2) * altitude
  let small_volume : ℝ := (1 / 3) * ((small_altitude / altitude) * base_edge) ^ 2 * small_altitude
  let frustum_volume : ℝ := original_volume - small_volume
  frustum_volume / original_volume = 32 / 33 := by sorry

end NUMINAMATH_CALUDE_pyramid_frustum_volume_ratio_l1468_146812


namespace NUMINAMATH_CALUDE_leak_drains_in_26_hours_l1468_146822

/-- Represents the time it takes for a leak to drain a tank, given the fill times with and without the leak -/
def leak_drain_time (pump_fill_time leak_fill_time : ℚ) : ℚ :=
  let pump_rate := 1 / pump_fill_time
  let combined_rate := 1 / leak_fill_time
  let leak_rate := pump_rate - combined_rate
  1 / leak_rate

/-- Theorem stating that given the specific fill times, the leak drains the tank in 26 hours -/
theorem leak_drains_in_26_hours :
  leak_drain_time 2 (13/6) = 26 := by sorry

end NUMINAMATH_CALUDE_leak_drains_in_26_hours_l1468_146822


namespace NUMINAMATH_CALUDE_vanya_age_l1468_146865

/-- Represents the ages of Vanya, his dad, and Seryozha -/
structure Ages where
  vanya : ℕ
  dad : ℕ
  seryozha : ℕ

/-- The conditions given in the problem -/
def age_relationships (ages : Ages) : Prop :=
  ages.vanya * 3 = ages.dad ∧
  ages.vanya = ages.seryozha * 3 ∧
  ages.dad = ages.seryozha + 40

/-- The theorem stating Vanya's age -/
theorem vanya_age (ages : Ages) : age_relationships ages → ages.vanya = 15 := by
  sorry

end NUMINAMATH_CALUDE_vanya_age_l1468_146865


namespace NUMINAMATH_CALUDE_solve_equation_l1468_146852

theorem solve_equation : ∃ y : ℕ, 400 + 2 * 20 * 5 + 25 = y ∧ y = 625 := by sorry

end NUMINAMATH_CALUDE_solve_equation_l1468_146852


namespace NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l1468_146861

theorem multiply_inverse_square_equals_cube (x : ℝ) : 
  x * (1/7)^2 = 7^3 → x = 16807 := by
sorry

end NUMINAMATH_CALUDE_multiply_inverse_square_equals_cube_l1468_146861


namespace NUMINAMATH_CALUDE_max_value_of_function_l1468_146886

theorem max_value_of_function (x y : ℝ) (h : x^2 + y^2 = 25) :
  (∀ a b : ℝ, a^2 + b^2 = 25 →
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) ≥
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50)) ∧
  (∃ a b : ℝ, a^2 + b^2 = 25 ∧
    Real.sqrt (8 * y - 6 * x + 50) + Real.sqrt (8 * y + 6 * x + 50) =
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) ∧
    Real.sqrt (8 * b - 6 * a + 50) + Real.sqrt (8 * b + 6 * a + 50) = 6 * Real.sqrt 10) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_function_l1468_146886


namespace NUMINAMATH_CALUDE_shadow_height_ratio_michaels_height_l1468_146854

/-- Given a flagpole and a person casting shadows at the same time, 
    calculate the person's height using the ratio of heights to shadows. -/
theorem shadow_height_ratio 
  (h₁ : ℝ) (s₁ : ℝ) (s₂ : ℝ) 
  (h₁_pos : h₁ > 0) (s₁_pos : s₁ > 0) (s₂_pos : s₂ > 0) :
  ∃ h₂ : ℝ, h₂ = (h₁ * s₂) / s₁ := by
  sorry

/-- Michael's height calculation based on the shadow ratio -/
theorem michaels_height 
  (flagpole_height : ℝ) (flagpole_shadow : ℝ) (michael_shadow : ℝ)
  (flagpole_height_eq : flagpole_height = 50)
  (flagpole_shadow_eq : flagpole_shadow = 25)
  (michael_shadow_eq : michael_shadow = 5) :
  ∃ michael_height : ℝ, michael_height = 10 := by
  sorry

end NUMINAMATH_CALUDE_shadow_height_ratio_michaels_height_l1468_146854


namespace NUMINAMATH_CALUDE_trigonometric_relation_and_triangle_property_l1468_146863

theorem trigonometric_relation_and_triangle_property (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  ∃ (A B C : ℝ), 
    (y / x = (Real.tan (9*π/20) * Real.cos (π/5) - Real.sin (π/5)) / (Real.cos (π/5) + Real.tan (9*π/20) * Real.sin (π/5))) ∧
    (Real.tan C = y / x) ∧
    (∀ A' B' : ℝ, Real.sin (2*A') + 2 * Real.cos B' ≤ B) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_relation_and_triangle_property_l1468_146863


namespace NUMINAMATH_CALUDE_class_size_problem_l1468_146813

theorem class_size_problem (x : ℕ) (n : ℕ) : 
  20 < x ∧ x < 30 ∧ 
  n = (0.20 : ℝ) * (5 * n) ∧
  n = (0.25 : ℝ) * (4 * n) ∧
  x = 8 * n + 2 →
  x = 26 := by sorry

end NUMINAMATH_CALUDE_class_size_problem_l1468_146813


namespace NUMINAMATH_CALUDE_picture_distance_from_right_end_l1468_146867

/-- Given a wall and a picture with specific dimensions and placement,
    calculate the distance from the right end of the wall to the nearest edge of the picture. -/
theorem picture_distance_from_right_end 
  (wall_width : ℝ) 
  (picture_width : ℝ) 
  (left_gap : ℝ) 
  (h1 : wall_width = 24)
  (h2 : picture_width = 4)
  (h3 : left_gap = 5) :
  wall_width - (left_gap + picture_width) = 15 := by
  sorry

#check picture_distance_from_right_end

end NUMINAMATH_CALUDE_picture_distance_from_right_end_l1468_146867


namespace NUMINAMATH_CALUDE_kanul_total_amount_l1468_146843

theorem kanul_total_amount (T : ℝ) : 
  T = 500 + 400 + 0.1 * T → T = 1000 := by
  sorry

end NUMINAMATH_CALUDE_kanul_total_amount_l1468_146843


namespace NUMINAMATH_CALUDE_power_equation_equality_l1468_146800

theorem power_equation_equality : 4^3 - 8 = 5^2 + 31 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_equality_l1468_146800


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangle_l1468_146846

/-- Represents a triangle with integer side lengths where two sides are equal -/
structure IsoscelesTriangle where
  a : ℕ  -- length of BC
  b : ℕ  -- length of AB and AC
  ab_eq_ac : b = b  -- AB = AC

/-- Represents the geometric configuration described in the problem -/
structure GeometricConfiguration (t : IsoscelesTriangle) where
  ω_center_is_incenter : Bool
  excircle_bc_internal : Bool
  excircle_ab_external : Bool
  excircle_ac_not_tangent : Bool

/-- The theorem statement -/
theorem min_perimeter_isosceles_triangle 
  (t : IsoscelesTriangle) 
  (config : GeometricConfiguration t) : 
  2 * t.b + t.a ≥ 20 := by
  sorry

#check min_perimeter_isosceles_triangle

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangle_l1468_146846


namespace NUMINAMATH_CALUDE_inequalities_satisfied_l1468_146850

theorem inequalities_satisfied
  (x y z : ℝ) (a b c : ℕ)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (hxa : x < a) (hyb : y < b) (hzc : z < c) :
  (x * y + y * z + z * x < a * b + b * c + c * a) ∧
  (x^2 + y^2 + z^2 < a^2 + b^2 + c^2) ∧
  (x * y * z < a * b * c) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_satisfied_l1468_146850


namespace NUMINAMATH_CALUDE_janice_throw_ratio_l1468_146881

/-- The height of Christine's first throw in feet -/
def christine_first : ℕ := 20

/-- The height of Janice's first throw in feet -/
def janice_first : ℕ := christine_first - 4

/-- The height of Christine's second throw in feet -/
def christine_second : ℕ := christine_first + 10

/-- The height of Christine's third throw in feet -/
def christine_third : ℕ := christine_second + 4

/-- The height of Janice's third throw in feet -/
def janice_third : ℕ := christine_first + 17

/-- The height of the highest throw in feet -/
def highest_throw : ℕ := 37

/-- The height of Janice's second throw in feet -/
def janice_second : ℕ := 2 * janice_first

theorem janice_throw_ratio :
  janice_second = 2 * janice_first ∧
  janice_third = highest_throw ∧
  janice_second < christine_third ∧
  janice_second > janice_first :=
by sorry

#check janice_throw_ratio

end NUMINAMATH_CALUDE_janice_throw_ratio_l1468_146881


namespace NUMINAMATH_CALUDE_arithmetic_progression_reciprocal_l1468_146893

/-- If a, b, and c form an arithmetic progression, and their reciprocals also form an arithmetic progression, then a = b = c. -/
theorem arithmetic_progression_reciprocal (a b c : ℝ) 
  (h1 : b - a = c - b)  -- a, b, c form an arithmetic progression
  (h2 : 1/b - 1/a = 1/c - 1/b)  -- reciprocals form an arithmetic progression
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_progression_reciprocal_l1468_146893


namespace NUMINAMATH_CALUDE_prob_two_slate_is_11_105_l1468_146897

-- Define the number of rocks for each type
def slate_rocks : ℕ := 12
def pumice_rocks : ℕ := 16
def granite_rocks : ℕ := 8

-- Define the total number of rocks
def total_rocks : ℕ := slate_rocks + pumice_rocks + granite_rocks

-- Define the probability of selecting two slate rocks
def prob_two_slate : ℚ := (slate_rocks : ℚ) / total_rocks * (slate_rocks - 1) / (total_rocks - 1)

theorem prob_two_slate_is_11_105 : prob_two_slate = 11 / 105 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_slate_is_11_105_l1468_146897


namespace NUMINAMATH_CALUDE_inverse_of_BP_squared_l1468_146808

/-- Given a 2x2 matrix B and a diagonal matrix P, prove that the inverse of (BP)² has a specific form. -/
theorem inverse_of_BP_squared (B P : Matrix (Fin 2) (Fin 2) ℚ) : 
  B⁻¹ = ![![3, 7], ![-2, -4]] →
  P = ![![1, 0], ![0, 2]] →
  ((B * P)^2)⁻¹ = ![![8, 28], ![-4, -12]] := by sorry

end NUMINAMATH_CALUDE_inverse_of_BP_squared_l1468_146808


namespace NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_product_l1468_146844

theorem sqrt_112_between_consecutive_integers_product : ∃ (n : ℕ), 
  n > 0 ∧ 
  n^2 < 112 ∧ 
  (n + 1)^2 > 112 ∧ 
  n * (n + 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_112_between_consecutive_integers_product_l1468_146844


namespace NUMINAMATH_CALUDE_parabola_shift_theorem_l1468_146818

/-- A parabola is a function of the form f(x) = a(x - h)^2 + k, where (h, k) is the vertex -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifting a parabola horizontally and vertically -/
def shift (p : Parabola) (dx dy : ℝ) : Parabola :=
  { a := p.a
    h := p.h - dx
    k := p.k - dy }

theorem parabola_shift_theorem (p : Parabola) :
  p.a = 3 ∧ p.h = 2 ∧ p.k = 1 →
  let p' := shift p 2 1
  p'.a = 3 ∧ p'.h = 0 ∧ p'.k = 0 := by
  sorry

#check parabola_shift_theorem

end NUMINAMATH_CALUDE_parabola_shift_theorem_l1468_146818


namespace NUMINAMATH_CALUDE_cannot_obtain_five_equal_numbers_l1468_146811

/-- Represents the set of numbers on the board -/
def BoardNumbers : Finset Int := {2, 3, 5, 7, 11}

/-- The operation of replacing two numbers with their arithmetic mean -/
def replaceWithMean (a b : Int) : Int := (a + b) / 2

/-- Predicate to check if two numbers have the same parity -/
def sameParity (a b : Int) : Prop := a % 2 = b % 2

/-- Theorem stating that it's impossible to obtain five equal numbers -/
theorem cannot_obtain_five_equal_numbers :
  ¬ ∃ (n : Int), ∃ (k : ℕ), ∃ (operations : Fin k → Int × Int),
    (∀ i, sameParity (operations i).1 (operations i).2) ∧
    (Finset.sum BoardNumbers id = 5 * n) ∧
    (∀ x ∈ BoardNumbers, x = n) :=
sorry

end NUMINAMATH_CALUDE_cannot_obtain_five_equal_numbers_l1468_146811


namespace NUMINAMATH_CALUDE_sequence_gcd_property_l1468_146809

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
  sorry

end NUMINAMATH_CALUDE_sequence_gcd_property_l1468_146809


namespace NUMINAMATH_CALUDE_students_without_A_l1468_146872

theorem students_without_A (total : ℕ) (history_A : ℕ) (math_A : ℕ) (both_A : ℕ) :
  total = 40 →
  history_A = 10 →
  math_A = 18 →
  both_A = 6 →
  total - ((history_A + math_A) - both_A) = 18 :=
by sorry

end NUMINAMATH_CALUDE_students_without_A_l1468_146872


namespace NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_l1468_146825

-- Equation 1
theorem quadratic_equation_1 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧ x₂ = -1 ∧ 
  (3 * x₁^2 + 2 * x₁ - 1 = 0) ∧ 
  (3 * x₂^2 + 2 * x₂ - 1 = 0) :=
sorry

-- Equation 2
theorem quadratic_equation_2 :
  ∃ x : ℝ, x = 3 ∧ 
  (x + 2) * (x - 3) = 5 * x - 15 :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_quadratic_equation_2_l1468_146825


namespace NUMINAMATH_CALUDE_nabla_sum_equals_32_l1468_146847

-- Define the ∇ operation
def nabla (k m : ℕ) : ℕ := k * (k - m)

-- State the theorem
theorem nabla_sum_equals_32 : nabla 5 1 + nabla 4 1 = 32 := by
  sorry

end NUMINAMATH_CALUDE_nabla_sum_equals_32_l1468_146847


namespace NUMINAMATH_CALUDE_statement_c_not_always_true_l1468_146838

theorem statement_c_not_always_true :
  ¬ ∀ (a b c : ℝ), a > b → a * c^2 > b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_statement_c_not_always_true_l1468_146838


namespace NUMINAMATH_CALUDE_stock_price_change_l1468_146805

/-- The final stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after specific changes -/
theorem stock_price_change : final_stock_price 80 1.2 0.3 = 123.2 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_change_l1468_146805


namespace NUMINAMATH_CALUDE_triangle_isosceles_condition_l1468_146848

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that if a/cos(A) = b/cos(B), then the triangle is isosceles. -/
theorem triangle_isosceles_condition (a b c A B C : ℝ) :
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →  -- Angles are in (0, π)
  A + B + C = π →  -- Sum of angles in a triangle
  a / Real.cos A = b / Real.cos B →  -- Given condition
  a = b  -- Conclusion: triangle is isosceles
  := by sorry

end NUMINAMATH_CALUDE_triangle_isosceles_condition_l1468_146848


namespace NUMINAMATH_CALUDE_square_area_calculation_l1468_146831

theorem square_area_calculation (side_length : ℝ) (h : side_length = 28) :
  side_length ^ 2 = 784 := by
  sorry

#check square_area_calculation

end NUMINAMATH_CALUDE_square_area_calculation_l1468_146831


namespace NUMINAMATH_CALUDE_unique_modulo_congruence_l1468_146880

theorem unique_modulo_congruence : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 10 ∧ n ≡ 99999 [ZMOD 11] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_modulo_congruence_l1468_146880


namespace NUMINAMATH_CALUDE_parabola_shift_l1468_146877

/-- The original parabola function -/
def original_parabola (x : ℝ) : ℝ := x^2 + 1

/-- The shifted parabola function -/
def shifted_parabola (x : ℝ) : ℝ := (x + 2)^2 - 2

/-- Theorem stating that the shifted parabola is equivalent to 
    shifting the original parabola 2 units left and 3 units down -/
theorem parabola_shift : 
  ∀ x : ℝ, shifted_parabola x = original_parabola (x + 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_l1468_146877


namespace NUMINAMATH_CALUDE_book_price_problem_l1468_146840

theorem book_price_problem (n : ℕ) (d : ℝ) (middle_price : ℝ) : 
  n = 40 → d = 3 → middle_price = 75 → 
  ∃ (first_price : ℝ), 
    (∀ i : ℕ, i ≤ n → 
      (first_price + d * (i - 1) = middle_price) ↔ i = n / 2) ∧
    first_price = 18 :=
by sorry

end NUMINAMATH_CALUDE_book_price_problem_l1468_146840


namespace NUMINAMATH_CALUDE_mathematical_run_disqualified_team_size_l1468_146878

theorem mathematical_run_disqualified_team_size 
  (initial_teams : ℕ) 
  (initial_average : ℕ) 
  (final_teams : ℕ) 
  (final_average : ℕ) 
  (h1 : initial_teams = 9)
  (h2 : initial_average = 7)
  (h3 : final_teams = initial_teams - 1)
  (h4 : final_average = 6) :
  initial_teams * initial_average - final_teams * final_average = 15 :=
by sorry

end NUMINAMATH_CALUDE_mathematical_run_disqualified_team_size_l1468_146878


namespace NUMINAMATH_CALUDE_cube_face_sum_l1468_146876

theorem cube_face_sum (a b c d e f g h : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f) = 2107 →
  a + b + c + d + e + f + g + h = 57 := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1468_146876


namespace NUMINAMATH_CALUDE_largest_binomial_equality_l1468_146833

theorem largest_binomial_equality : ∃ n : ℕ, (n ≤ 11 ∧ Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 n ∧ ∀ m : ℕ, m ≤ 11 → Nat.choose 10 4 + Nat.choose 10 5 = Nat.choose 11 m → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_largest_binomial_equality_l1468_146833


namespace NUMINAMATH_CALUDE_tangerine_boxes_count_l1468_146834

/-- Given information about apples and tangerines, prove the number of tangerine boxes --/
theorem tangerine_boxes_count
  (apple_boxes : ℕ)
  (apples_per_box : ℕ)
  (tangerines_per_box : ℕ)
  (total_fruits : ℕ)
  (h1 : apple_boxes = 19)
  (h2 : apples_per_box = 46)
  (h3 : tangerines_per_box = 170)
  (h4 : total_fruits = 1894)
  : ∃ (tangerine_boxes : ℕ), tangerine_boxes = 6 ∧ 
    apple_boxes * apples_per_box + tangerine_boxes * tangerines_per_box = total_fruits :=
by
  sorry


end NUMINAMATH_CALUDE_tangerine_boxes_count_l1468_146834


namespace NUMINAMATH_CALUDE_set_operations_l1468_146832

-- Define the sets A and B
def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

-- State the theorem
theorem set_operations :
  (A ∪ B = {0, 1, 2, 3}) ∧ (A ∩ B = {1, 2}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l1468_146832


namespace NUMINAMATH_CALUDE_subcommittee_count_l1468_146817

theorem subcommittee_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  n * (Nat.choose (n - 1) (k - 1)) = 12180 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1468_146817


namespace NUMINAMATH_CALUDE_ellipse_and_tangent_circle_l1468_146869

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

/-- Definition of the circle tangent to line l -/
def tangent_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + y^2 = 4 / 3

/-- Theorem statement -/
theorem ellipse_and_tangent_circle :
  ∀ (x y : ℝ),
  -- Conditions
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ a^2 - b^2 = 1) →  -- Ellipse properties
  (1^2 / 4 + (3/2)^2 / 3 = 1) →  -- Point (1, 3/2) lies on C
  (∃ (m : ℝ), m^2 = 2) →  -- Slope of line l
  -- Conclusions
  (ellipse_C x y ↔ x^2 / 4 + y^2 / 3 = 1) ∧
  (tangent_circle x y ↔ (x + 1)^2 + y^2 = 4 / 3) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_tangent_circle_l1468_146869


namespace NUMINAMATH_CALUDE_a_in_S_l1468_146802

theorem a_in_S (S T : Set ℕ) (a : ℕ) 
  (h1 : S = {1, 2})
  (h2 : T = {a})
  (h3 : S ∪ T = S) :
  a ∈ S := by
  sorry

end NUMINAMATH_CALUDE_a_in_S_l1468_146802


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l1468_146859

theorem exists_integer_sqrt_8m : ∃ m : ℕ+, ∃ k : ℕ, (8 * m.val : ℕ) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_8m_l1468_146859


namespace NUMINAMATH_CALUDE_parking_space_unpainted_side_l1468_146875

/-- Represents a rectangular parking space with three painted sides. -/
structure ParkingSpace where
  width : ℝ
  length : ℝ
  painted_sum : ℝ
  area : ℝ

/-- The length of the unpainted side of a parking space. -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

theorem parking_space_unpainted_side
  (p : ParkingSpace)
  (h1 : p.painted_sum = 37)
  (h2 : p.area = 126)
  (h3 : p.painted_sum = 2 * p.width + p.length)
  (h4 : p.area = p.width * p.length) :
  unpainted_side_length p = 9 := by
  sorry

end NUMINAMATH_CALUDE_parking_space_unpainted_side_l1468_146875


namespace NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1468_146885

theorem gcd_power_two_minus_one :
  Nat.gcd (2^1998 - 1) (2^1989 - 1) = 2^9 - 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_power_two_minus_one_l1468_146885


namespace NUMINAMATH_CALUDE_circle_translation_l1468_146806

-- Define the original circle
def original_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the translated circle
def translated_circle (x y : ℝ) : Prop := (x+1)^2 + (y-2)^2 = 1

-- Define the translation vector
def translation_vector : ℝ × ℝ := (-1, 2)

-- Theorem statement
theorem circle_translation :
  ∀ (x y : ℝ), original_circle x y ↔ translated_circle (x + translation_vector.1) (y + translation_vector.2) :=
by sorry

end NUMINAMATH_CALUDE_circle_translation_l1468_146806


namespace NUMINAMATH_CALUDE_car_distance_problem_l1468_146829

/-- Represents the distance traveled by a car given its speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem car_distance_problem (speed_x speed_y : ℝ) (initial_time : ℝ) :
  speed_x = 35 →
  speed_y = 65 →
  initial_time = 72 / 60 →
  ∃ t : ℝ, 
    distance speed_y t = distance speed_x initial_time + distance speed_x t ∧
    distance speed_x t = 49 := by
  sorry

#check car_distance_problem

end NUMINAMATH_CALUDE_car_distance_problem_l1468_146829


namespace NUMINAMATH_CALUDE_mother_father_age_ratio_l1468_146888

/-- Represents the ages and relationships in Darcie's family -/
structure Family where
  darcie_age : ℕ
  father_age : ℕ
  mother_age_ratio : ℚ
  darcie_mother_ratio : ℚ

/-- Theorem stating the ratio of mother's age to father's age -/
theorem mother_father_age_ratio (f : Family)
  (h1 : f.darcie_age = 4)
  (h2 : f.father_age = 30)
  (h3 : f.darcie_mother_ratio = 1 / 6)
  (h4 : f.mother_age_ratio * f.father_age = f.darcie_age / f.darcie_mother_ratio) :
  f.mother_age_ratio = 4 / 5 := by
  sorry


end NUMINAMATH_CALUDE_mother_father_age_ratio_l1468_146888


namespace NUMINAMATH_CALUDE_stating_equation_satisfied_l1468_146894

/-- 
Represents a number system with base X.
-/
structure BaseX where
  X : ℕ
  X_ge_two : X ≥ 2

/-- 
Represents a digit in the number system with base X.
-/
def Digit (b : BaseX) := {d : ℕ // d < b.X}

/--
Converts a number represented by digits in base X to its decimal value.
-/
def to_decimal (b : BaseX) (digits : List (Digit b)) : ℕ :=
  digits.foldr (fun d acc => acc * b.X + d.val) 0

/--
Theorem stating that the equation ABBC * CCA = CCCCAC is satisfied
in any base X ≥ 2 when A = 1, B = 0, and C = X - 1 or C = 1.
-/
theorem equation_satisfied (b : BaseX) :
  let A : Digit b := ⟨1, by sorry⟩
  let B : Digit b := ⟨0, by sorry⟩
  let C₁ : Digit b := ⟨b.X - 1, by sorry⟩
  let C₂ : Digit b := ⟨1, by sorry⟩
  (to_decimal b [A, B, B, C₁] * to_decimal b [C₁, C₁, A] = to_decimal b [C₁, C₁, C₁, C₁, A, C₁]) ∧
  (to_decimal b [A, B, B, C₂] * to_decimal b [C₂, C₂, A] = to_decimal b [C₂, C₂, C₂, C₂, A, C₂]) :=
by
  sorry


end NUMINAMATH_CALUDE_stating_equation_satisfied_l1468_146894


namespace NUMINAMATH_CALUDE_doubled_container_volume_l1468_146864

/-- A cylindrical container that can hold water -/
structure Container :=
  (volume : ℝ)
  (isOriginal : Bool)

/-- Double the dimensions of a container -/
def doubleContainer (c : Container) : Container :=
  { volume := 8 * c.volume, isOriginal := false }

theorem doubled_container_volume (c : Container) 
  (h1 : c.isOriginal = true) 
  (h2 : c.volume = 3) : 
  (doubleContainer c).volume = 24 := by
sorry

end NUMINAMATH_CALUDE_doubled_container_volume_l1468_146864


namespace NUMINAMATH_CALUDE_rotation_of_point_A_l1468_146892

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Rotates a point clockwise by π/2 around the origin -/
def rotate_clockwise_90 (p : Point2D) : Point2D :=
  ⟨p.y, -p.x⟩

theorem rotation_of_point_A : 
  let A : Point2D := ⟨2, 1⟩
  let B : Point2D := rotate_clockwise_90 A
  B.x = 1 ∧ B.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_rotation_of_point_A_l1468_146892


namespace NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_eq_7_l1468_146839

theorem log_sqrt10_1000sqrt10_eq_7 : Real.log (1000 * Real.sqrt 10) / Real.log (Real.sqrt 10) = 7 := by
  sorry

end NUMINAMATH_CALUDE_log_sqrt10_1000sqrt10_eq_7_l1468_146839


namespace NUMINAMATH_CALUDE_article_cost_l1468_146841

theorem article_cost (sell_price_1 sell_price_2 : ℝ) 
  (h1 : sell_price_1 = 380)
  (h2 : sell_price_2 = 420)
  (h3 : sell_price_2 - sell_price_1 = 0.05 * cost) : cost = 800 := by
  sorry

end NUMINAMATH_CALUDE_article_cost_l1468_146841


namespace NUMINAMATH_CALUDE_area_of_triangle_QPO_l1468_146851

-- Define the points
variable (A B C D P Q O N M : Point)
-- Define the area of the parallelogram
variable (k : ℝ)

-- Define the conditions
def is_parallelogram (A B C D : Point) : Prop := sorry

def bisects (P Q R : Point) : Prop := sorry

def intersects (L₁ L₂ P : Point) : Prop := sorry

def area (shape : Set Point) : ℝ := sorry

-- State the theorem
theorem area_of_triangle_QPO 
  (h1 : is_parallelogram A B C D)
  (h2 : bisects D N C)
  (h3 : intersects D P B)
  (h4 : bisects C M D)
  (h5 : intersects C Q A)
  (h6 : intersects D P O)
  (h7 : intersects C Q O)
  (h8 : area {A, B, C, D} = k) :
  area {Q, P, O} = 9/8 * k := sorry

end NUMINAMATH_CALUDE_area_of_triangle_QPO_l1468_146851


namespace NUMINAMATH_CALUDE_workshop_salary_calculation_l1468_146857

/-- Given a workshop with workers and technicians, calculate the average salary of non-technician workers. -/
theorem workshop_salary_calculation
  (total_workers : ℕ)
  (avg_salary_all : ℝ)
  (num_technicians : ℕ)
  (avg_salary_technicians : ℝ)
  (h_total_workers : total_workers = 21)
  (h_avg_salary_all : avg_salary_all = 8000)
  (h_num_technicians : num_technicians = 7)
  (h_avg_salary_technicians : avg_salary_technicians = 12000) :
  let num_rest := total_workers - num_technicians
  let total_salary := avg_salary_all * total_workers
  let total_salary_technicians := avg_salary_technicians * num_technicians
  let total_salary_rest := total_salary - total_salary_technicians
  total_salary_rest / num_rest = 6000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_calculation_l1468_146857


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1468_146895

/-- Given a parabola y = ax^2 + bx + c with vertex (h, k) and passing through (0, -k) where k ≠ 0,
    prove that b = 4k/h -/
theorem parabola_coefficient (a b c h k : ℝ) (hk : k ≠ 0) :
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 + k) →
  a * 0^2 + b * 0 + c = -k →
  b = 4 * k / h := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1468_146895


namespace NUMINAMATH_CALUDE_equation_represents_two_lines_l1468_146803

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Define what it means to be a straight line
def is_straight_line (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x : ℝ, f x = m * x + b

-- Theorem statement
theorem equation_represents_two_lines :
  ∃ f g : ℝ → ℝ, 
    (is_straight_line f ∧ is_straight_line g) ∧
    (∀ x y : ℝ, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_two_lines_l1468_146803


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1468_146814

theorem tan_pi_minus_alpha_eq_neg_two_implies_result (α : ℝ) 
  (h : Real.tan (π - α) = -2) : 
  1 / (Real.cos (2 * α) + Real.cos α ^ 2) = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_eq_neg_two_implies_result_l1468_146814


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l1468_146868

/-- A geometric sequence with its sum and common ratio -/
structure GeometricSequence where
  a : ℕ+ → ℝ
  S : ℕ+ → ℝ
  q : ℝ
  sum_formula : ∀ n : ℕ+, S n = (a 1) * (1 - q^n.val) / (1 - q)
  term_formula : ∀ n : ℕ+, a n = (a 1) * q^(n.val - 1)

/-- The theorem stating the general term of the specific geometric sequence -/
theorem geometric_sequence_general_term 
  (seq : GeometricSequence) 
  (h1 : seq.S 3 = 14) 
  (h2 : seq.q = 2) :
  ∀ n : ℕ+, seq.a n = 2^n.val :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l1468_146868


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l1468_146887

theorem complex_fraction_equality : Complex.I * 2 / (1 - Complex.I) = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l1468_146887


namespace NUMINAMATH_CALUDE_upstream_distance_is_48_l1468_146830

/-- Represents the problem of calculating the upstream distance rowed --/
def UpstreamRowingProblem (downstream_distance : ℝ) (time : ℝ) (stream_speed : ℝ) : Prop :=
  ∃ (upstream_distance : ℝ) (boat_speed : ℝ),
    downstream_distance = (boat_speed + stream_speed) * time ∧
    upstream_distance = (boat_speed - stream_speed) * time ∧
    upstream_distance = 48

/-- Theorem stating that given the problem conditions, the upstream distance is 48 km --/
theorem upstream_distance_is_48 :
  UpstreamRowingProblem 84 2 9 :=
sorry

end NUMINAMATH_CALUDE_upstream_distance_is_48_l1468_146830


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l1468_146889

theorem quadratic_root_difference : 
  let a : ℝ := 5 + 3 * Real.sqrt 2
  let b : ℝ := 2 + Real.sqrt 2
  let c : ℝ := -1
  let discriminant := b^2 - 4*a*c
  let root_difference := (2 * Real.sqrt discriminant) / (2 * a)
  root_difference = (2 * Real.sqrt (24 * Real.sqrt 2 + 180)) / 7 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l1468_146889


namespace NUMINAMATH_CALUDE_parrot_seed_consumption_l1468_146815

/-- Given a parrot that absorbs 40% of the seeds it consumes and absorbed 8 ounces of seeds,
    prove that the total amount of seeds consumed is 20 ounces and twice that amount is 40 ounces. -/
theorem parrot_seed_consumption (absorbed_percentage : ℝ) (absorbed_amount : ℝ) 
    (h1 : absorbed_percentage = 0.40)
    (h2 : absorbed_amount = 8) : 
  ∃ (total_consumed : ℝ), 
    total_consumed * absorbed_percentage = absorbed_amount ∧ 
    total_consumed = 20 ∧ 
    2 * total_consumed = 40 := by
  sorry


end NUMINAMATH_CALUDE_parrot_seed_consumption_l1468_146815


namespace NUMINAMATH_CALUDE_fraction_invariance_l1468_146819

theorem fraction_invariance (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2008 * (2 * x)) / (2007 * (2 * y)) = (2008 * x) / (2007 * y) := by
sorry

end NUMINAMATH_CALUDE_fraction_invariance_l1468_146819


namespace NUMINAMATH_CALUDE_extension_point_coordinates_l1468_146816

/-- Given points A and B, and a point P on the extension of segment AB such that |AP| = 2|PB|, 
    prove that P has specific coordinates. -/
theorem extension_point_coordinates (A B P : ℝ × ℝ) : 
  A = (2, 3) → 
  B = (4, -3) → 
  (∃ t : ℝ, t > 1 ∧ P = A + t • (B - A)) →
  ‖P - A‖ = 2 * ‖P - B‖ →
  P = (6, -9) := by
  sorry

end NUMINAMATH_CALUDE_extension_point_coordinates_l1468_146816


namespace NUMINAMATH_CALUDE_class_size_l1468_146837

theorem class_size (top_rank bottom_rank : ℕ) (h1 : top_rank = 17) (h2 : bottom_rank = 15) :
  top_rank + bottom_rank - 1 = 31 := by
  sorry

end NUMINAMATH_CALUDE_class_size_l1468_146837


namespace NUMINAMATH_CALUDE_dianes_trip_length_l1468_146810

theorem dianes_trip_length :
  ∀ (total_length : ℝ),
  (1/4 : ℝ) * total_length + 24 + (1/3 : ℝ) * total_length = total_length →
  total_length = 57.6 := by
sorry

end NUMINAMATH_CALUDE_dianes_trip_length_l1468_146810


namespace NUMINAMATH_CALUDE_ticket_problem_l1468_146896

/-- Represents the ticket distribution and pricing for a football match --/
structure TicketInfo where
  total : ℕ  -- Total number of tickets
  typeA : ℕ  -- Number of Type A tickets
  m : ℕ      -- Price parameter

/-- Conditions for the ticket distribution and pricing --/
def validTicketInfo (info : TicketInfo) : Prop :=
  info.total = 500 ∧
  info.typeA ≥ 3 * (info.total - info.typeA) ∧
  500 * (1 + (info.m + 10) / 100) * (info.m + 20) = 56000 ∧
  info.m > 0

theorem ticket_problem (info : TicketInfo) (h : validTicketInfo info) :
  info.typeA ≥ 375 ∧ info.m = 50 := by
  sorry


end NUMINAMATH_CALUDE_ticket_problem_l1468_146896


namespace NUMINAMATH_CALUDE_second_number_value_l1468_146898

theorem second_number_value (x y : ℕ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1468_146898


namespace NUMINAMATH_CALUDE_inequality_constraint_on_a_l1468_146860

theorem inequality_constraint_on_a (a : ℝ) : 
  (∀ x : ℝ, (Real.exp x - a * x) * (x^2 - a * x + 1) ≥ 0) → 
  0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_constraint_on_a_l1468_146860


namespace NUMINAMATH_CALUDE_clover_walking_distance_l1468_146827

/-- Clover's walking problem -/
theorem clover_walking_distance 
  (total_distance : ℝ) 
  (num_days : ℕ) 
  (walks_per_day : ℕ) :
  total_distance = 90 →
  num_days = 30 →
  walks_per_day = 2 →
  (total_distance / num_days) / walks_per_day = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_clover_walking_distance_l1468_146827


namespace NUMINAMATH_CALUDE_initial_odometer_reading_l1468_146828

/-- Calculates the initial odometer reading before a trip -/
theorem initial_odometer_reading
  (odometer_at_lunch : ℝ)
  (distance_traveled : ℝ)
  (h1 : odometer_at_lunch = 372)
  (h2 : distance_traveled = 159.7) :
  odometer_at_lunch - distance_traveled = 212.3 := by
sorry

end NUMINAMATH_CALUDE_initial_odometer_reading_l1468_146828


namespace NUMINAMATH_CALUDE_sum_of_bases_is_fifteen_l1468_146853

/-- Represents a fraction in a given base --/
structure FractionInBase where
  numerator : ℕ
  denominator : ℕ
  base : ℕ

/-- Converts a repeating decimal to a fraction --/
def repeatingDecimalToFraction (digits : ℕ) (base : ℕ) : FractionInBase :=
  { numerator := digits,
    denominator := base^2 - 1,
    base := base }

theorem sum_of_bases_is_fifteen :
  let R₁ : ℕ := 9
  let R₂ : ℕ := 6
  let F₁_in_R₁ := repeatingDecimalToFraction 48 R₁
  let F₂_in_R₁ := repeatingDecimalToFraction 84 R₁
  let F₁_in_R₂ := repeatingDecimalToFraction 35 R₂
  let F₂_in_R₂ := repeatingDecimalToFraction 53 R₂
  R₁ + R₂ = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_bases_is_fifteen_l1468_146853


namespace NUMINAMATH_CALUDE_power_division_expression_simplification_l1468_146871

-- Problem 1
theorem power_division (a : ℝ) : a^6 / a^2 = a^4 := by sorry

-- Problem 2
theorem expression_simplification (m : ℝ) : m^2 * m^4 - (2*m^3)^2 = -3*m^6 := by sorry

end NUMINAMATH_CALUDE_power_division_expression_simplification_l1468_146871
