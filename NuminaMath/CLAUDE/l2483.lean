import Mathlib

namespace NUMINAMATH_CALUDE_picture_placement_l2483_248321

theorem picture_placement (wall_width picture_width : ℝ) 
  (hw : wall_width = 19) 
  (hp : picture_width = 3) : 
  (wall_width - picture_width) / 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_picture_placement_l2483_248321


namespace NUMINAMATH_CALUDE_meeting_point_one_third_distance_l2483_248392

/-- Given two points in a 2D plane, this function calculates a point that is a fraction of the distance from the first point to the second point. -/
def intermediatePoint (x1 y1 x2 y2 t : ℝ) : ℝ × ℝ :=
  (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

/-- Theorem stating that the point (10, 5) is one-third of the way from (8, 3) to (14, 9). -/
theorem meeting_point_one_third_distance :
  intermediatePoint 8 3 14 9 (1/3) = (10, 5) := by
sorry

end NUMINAMATH_CALUDE_meeting_point_one_third_distance_l2483_248392


namespace NUMINAMATH_CALUDE_total_vegetables_l2483_248394

def vegetable_garden_problem (potatoes cucumbers tomatoes peppers carrots : ℕ) : Prop :=
  potatoes = 560 ∧
  cucumbers = potatoes - 132 ∧
  tomatoes = 3 * cucumbers ∧
  peppers = tomatoes / 2 ∧
  carrots = cucumbers + tomatoes

theorem total_vegetables (potatoes cucumbers tomatoes peppers carrots : ℕ) :
  vegetable_garden_problem potatoes cucumbers tomatoes peppers carrots →
  potatoes + cucumbers + tomatoes + peppers + carrots = 4626 := by
  sorry

end NUMINAMATH_CALUDE_total_vegetables_l2483_248394


namespace NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2483_248313

/-- Represents a horizontally positioned cylindrical tank -/
structure CylindricalTank where
  length : ℝ
  diameter : ℝ

/-- Calculates the depth of oil in a cylindrical tank given the surface area -/
def oilDepth (tank : CylindricalTank) (surfaceArea : ℝ) : ℝ :=
  sorry

theorem oil_depth_in_specific_tank :
  let tank : CylindricalTank := { length := 12, diameter := 8 }
  let surfaceArea : ℝ := 32
  oilDepth tank surfaceArea = 4 := by
  sorry

end NUMINAMATH_CALUDE_oil_depth_in_specific_tank_l2483_248313


namespace NUMINAMATH_CALUDE_bike_shop_profit_is_3000_l2483_248383

/-- Calculates the profit of a bike shop given various parameters. -/
def bike_shop_profit (tire_repair_charge : ℕ) (tire_repair_cost : ℕ) (tire_repairs : ℕ)
                     (complex_repair_charge : ℕ) (complex_repair_cost : ℕ) (complex_repairs : ℕ)
                     (retail_profit : ℕ) (fixed_expenses : ℕ) : ℕ :=
  (tire_repairs * (tire_repair_charge - tire_repair_cost)) +
  (complex_repairs * (complex_repair_charge - complex_repair_cost)) +
  retail_profit - fixed_expenses

/-- Theorem stating that the bike shop's profit is $3000 under given conditions. -/
theorem bike_shop_profit_is_3000 :
  bike_shop_profit 20 5 300 300 50 2 2000 4000 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_bike_shop_profit_is_3000_l2483_248383


namespace NUMINAMATH_CALUDE_even_function_property_l2483_248301

def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem even_function_property (f : ℝ → ℝ) 
  (h_even : EvenFunction f) 
  (h_nonneg : ∀ x ≥ 0, f x = x * (x + 1)) :
  ∀ x < 0, f x = -x * (1 - x) := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l2483_248301


namespace NUMINAMATH_CALUDE_calculate_expression_l2483_248359

theorem calculate_expression : ((15^10 / 15^9)^3 * 5^3) / 3^3 = 15625 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2483_248359


namespace NUMINAMATH_CALUDE_complex_number_theorem_l2483_248373

theorem complex_number_theorem (z : ℂ) (b : ℝ) :
  z = (Complex.I ^ 3) / (1 - Complex.I) →
  (∃ (y : ℝ), z + b = Complex.I * y) →
  b = -1/2 := by
sorry

end NUMINAMATH_CALUDE_complex_number_theorem_l2483_248373


namespace NUMINAMATH_CALUDE_expression_equals_24_l2483_248368

def arithmetic_expression (a b c d : ℕ) : Prop :=
  ∃ (e : ℕ → ℕ → ℕ → ℕ → ℕ), e a b c d = 24

theorem expression_equals_24 : arithmetic_expression 8 8 8 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_24_l2483_248368


namespace NUMINAMATH_CALUDE_angle_between_m_and_n_l2483_248337

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (9, 12)
def c : ℝ × ℝ := (4, -3)
def m : ℝ × ℝ := (2 * a.1 - b.1, 2 * a.2 - b.2)
def n : ℝ × ℝ := (a.1 + c.1, a.2 + c.2)

theorem angle_between_m_and_n :
  Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2))) = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_m_and_n_l2483_248337


namespace NUMINAMATH_CALUDE_area_of_trapezoid_psrt_l2483_248347

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid PSRT -/
structure Trapezoid where
  area : ℝ

/-- Represents the diagram configuration -/
structure DiagramConfig where
  pqr : Triangle
  smallestTriangles : Finset Triangle
  psrt : Trapezoid

/-- The main theorem statement -/
theorem area_of_trapezoid_psrt (config : DiagramConfig) : config.psrt.area = 53.5 :=
  by
  have h1 : config.pqr.area = 72 := by sorry
  have h2 : config.smallestTriangles.card = 9 := by sorry
  have h3 : ∀ t ∈ config.smallestTriangles, t.area = 2 := by sorry
  have h4 : ∀ t : Triangle, t ∈ config.smallestTriangles → t.area ≤ config.pqr.area := by sorry
  sorry

/-- Auxiliary definition for isosceles triangle -/
def isIsosceles (t : Triangle) : Prop := sorry

/-- Auxiliary definition for triangle similarity -/
def areSimilar (t1 t2 : Triangle) : Prop := sorry

/-- Additional properties of the configuration -/
axiom pqr_is_isosceles (config : DiagramConfig) : isIsosceles config.pqr
axiom all_triangles_similar (config : DiagramConfig) (t : Triangle) : 
  t ∈ config.smallestTriangles → areSimilar t config.pqr

end NUMINAMATH_CALUDE_area_of_trapezoid_psrt_l2483_248347


namespace NUMINAMATH_CALUDE_clarence_initial_oranges_l2483_248319

/-- Proves that Clarence's initial number of oranges is 5 -/
theorem clarence_initial_oranges :
  ∀ (initial total from_joyce : ℕ),
    initial + from_joyce = total →
    from_joyce = 3 →
    total = 8 →
    initial = 5 := by
  sorry

end NUMINAMATH_CALUDE_clarence_initial_oranges_l2483_248319


namespace NUMINAMATH_CALUDE_sum_of_extrema_equals_two_l2483_248304

-- Define the function f(x) = x ln |x| + 1
noncomputable def f (x : ℝ) : ℝ := x * Real.log (abs x) + 1

-- Theorem statement
theorem sum_of_extrema_equals_two :
  ∃ (max_val min_val : ℝ),
    (∀ x, f x ≤ max_val) ∧
    (∃ x, f x = max_val) ∧
    (∀ x, f x ≥ min_val) ∧
    (∃ x, f x = min_val) ∧
    max_val + min_val = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_extrema_equals_two_l2483_248304


namespace NUMINAMATH_CALUDE_eraser_ratio_l2483_248372

theorem eraser_ratio (andrea_erasers : ℕ) (anya_extra_erasers : ℕ) :
  andrea_erasers = 4 →
  anya_extra_erasers = 12 →
  (andrea_erasers + anya_extra_erasers) / andrea_erasers = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_eraser_ratio_l2483_248372


namespace NUMINAMATH_CALUDE_min_value_of_f_l2483_248340

theorem min_value_of_f (x : ℝ) (h : x > 0) : 
  let f := fun x => 1 / x^2 + 2 * x
  (∀ y > 0, f y ≥ 3) ∧ (∃ z > 0, f z = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2483_248340


namespace NUMINAMATH_CALUDE_cube_with_cylindrical_hole_volume_l2483_248311

/-- The volume of a cube with a cylindrical hole -/
theorem cube_with_cylindrical_hole_volume (cube_side : ℝ) (hole_diameter : ℝ) : 
  cube_side = 6 →
  hole_diameter = 3 →
  abs (cube_side ^ 3 - π * (hole_diameter / 2) ^ 2 * cube_side - 173.59) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_cube_with_cylindrical_hole_volume_l2483_248311


namespace NUMINAMATH_CALUDE_min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l2483_248306

/-- Represents the minimum number of hours required for all friends to know all news -/
def min_hours (N : ℕ) : ℕ :=
  if N = 64 then 6
  else if N = 55 then 7
  else if N = 100 then 7
  else 0  -- undefined for other values of N

/-- The theorem stating the minimum number of hours for specific N values -/
theorem min_hours_theorem :
  (min_hours 64 = 6) ∧ (min_hours 55 = 7) ∧ (min_hours 100 = 7) := by
  sorry

/-- Helper function to calculate the maximum number of friends who can know a piece of news after h hours -/
def max_friends_knowing (h : ℕ) : ℕ := 2^h

/-- Theorem stating that the minimum hours is sufficient for all friends to know all news -/
theorem min_hours_sufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing h ≥ N := by
  sorry

/-- Theorem stating that one less hour is insufficient for all friends to know all news -/
theorem less_hours_insufficient (N : ℕ) (h : ℕ) (h_eq : h = min_hours N) :
  max_friends_knowing (h - 1) < N := by
  sorry

end NUMINAMATH_CALUDE_min_hours_theorem_min_hours_sufficient_less_hours_insufficient_l2483_248306


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2483_248309

/-- The range of values for m where the line y = x + m intersects the ellipse x^2/4 + y^2/3 = 1 -/
theorem line_ellipse_intersection_range :
  let line (x m : ℝ) := x + m
  let ellipse (x y : ℝ) := x^2/4 + y^2/3 = 1
  let intersects (m : ℝ) := ∃ x, ellipse x (line x m)
  ∀ m, intersects m ↔ m ∈ Set.Icc (-Real.sqrt 7) (Real.sqrt 7) :=
by sorry


end NUMINAMATH_CALUDE_line_ellipse_intersection_range_l2483_248309


namespace NUMINAMATH_CALUDE_max_abs_z_l2483_248302

theorem max_abs_z (z : ℂ) (h : Complex.abs (z + 3 + 4*I) = 2) :
  ∃ (w : ℂ), Complex.abs w = 2 ∧ Complex.abs (w + 3 + 4*I) = 2 ∧
  ∀ (u : ℂ), Complex.abs (u + 3 + 4*I) = 2 → Complex.abs u ≤ Complex.abs w :=
sorry

end NUMINAMATH_CALUDE_max_abs_z_l2483_248302


namespace NUMINAMATH_CALUDE_expression_simplification_l2483_248356

theorem expression_simplification (x : ℝ) :
  x = (1/2)⁻¹ + (π - 1)^0 →
  ((x - 3) / (x^2 - 1) - 2 / (x + 1)) / (x / (x^2 - 2*x + 1)) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l2483_248356


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l2483_248375

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_eq : a + b + c = 12) 
  (sum_product_eq : a * b + a * c + b * c = 30) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 648 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l2483_248375


namespace NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l2483_248353

theorem smallest_six_digit_divisible_by_111 :
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 → n ≥ 100011 :=
by sorry

end NUMINAMATH_CALUDE_smallest_six_digit_divisible_by_111_l2483_248353


namespace NUMINAMATH_CALUDE_box_height_is_nine_l2483_248307

/-- A rectangular box with dimensions 6 × 6 × h -/
structure Box (h : ℝ) where
  length : ℝ := 6
  width : ℝ := 6
  height : ℝ := h

/-- A sphere with a given radius -/
structure Sphere (r : ℝ) where
  radius : ℝ := r

/-- Predicate to check if a sphere is tangent to three sides of a box -/
def tangent_to_three_sides (s : Sphere r) (b : Box h) : Prop :=
  sorry

/-- Predicate to check if two spheres are tangent -/
def spheres_tangent (s1 : Sphere r1) (s2 : Sphere r2) : Prop :=
  sorry

/-- The main theorem -/
theorem box_height_is_nine :
  ∀ (h : ℝ) (b : Box h) (large_sphere : Sphere 3) (small_spheres : Fin 8 → Sphere 1.5),
    (∀ i, tangent_to_three_sides (small_spheres i) b) →
    (∀ i, spheres_tangent large_sphere (small_spheres i)) →
    h = 9 :=
by sorry

end NUMINAMATH_CALUDE_box_height_is_nine_l2483_248307


namespace NUMINAMATH_CALUDE_paths_in_8x6_grid_l2483_248377

/-- The number of paths in a grid from bottom-left to top-right -/
def grid_paths (horizontal_steps : ℕ) (vertical_steps : ℕ) : ℕ :=
  Nat.choose (horizontal_steps + vertical_steps) vertical_steps

/-- Theorem: The number of paths in an 8x6 grid is 3003 -/
theorem paths_in_8x6_grid :
  grid_paths 8 6 = 3003 := by
  sorry

end NUMINAMATH_CALUDE_paths_in_8x6_grid_l2483_248377


namespace NUMINAMATH_CALUDE_checkerboard_valid_squares_l2483_248355

/-- Represents a square on the checkerboard -/
structure Square where
  size : Nat
  position : Nat × Nat

/-- The checkerboard -/
def Checkerboard : Type := Array (Array Bool)

/-- Creates a 10x10 checkerboard with alternating black and white squares -/
def create_checkerboard : Checkerboard := sorry

/-- Checks if a square contains at least 8 black squares -/
def has_at_least_8_black (board : Checkerboard) (square : Square) : Bool := sorry

/-- Counts the number of valid squares on the board -/
def count_valid_squares (board : Checkerboard) : Nat := sorry

theorem checkerboard_valid_squares :
  let board := create_checkerboard
  count_valid_squares board = 140 := by sorry

end NUMINAMATH_CALUDE_checkerboard_valid_squares_l2483_248355


namespace NUMINAMATH_CALUDE_no_linear_term_in_product_l2483_248341

theorem no_linear_term_in_product (m : ℚ) : 
  (∀ x : ℚ, (x - 2) * (x^2 + m*x + 1) = x^3 + (m-2)*x^2 + 0*x + (-2)) → m = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_in_product_l2483_248341


namespace NUMINAMATH_CALUDE_harry_lost_sea_creatures_l2483_248351

theorem harry_lost_sea_creatures (sea_stars seashells snails items_left : ℕ) 
  (h1 : sea_stars = 34)
  (h2 : seashells = 21)
  (h3 : snails = 29)
  (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 := by
  sorry

end NUMINAMATH_CALUDE_harry_lost_sea_creatures_l2483_248351


namespace NUMINAMATH_CALUDE_f_negative_l2483_248361

-- Define an odd function
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function for x > 0
def f_positive (x : ℝ) : ℝ :=
  x^2 - x + 1

-- Theorem statement
theorem f_negative (f : ℝ → ℝ) (h_odd : odd_function f) 
  (h_positive : ∀ x > 0, f x = f_positive x) :
  ∀ x < 0, f x = -x^2 - x - 1 := by
sorry

end NUMINAMATH_CALUDE_f_negative_l2483_248361


namespace NUMINAMATH_CALUDE_salary_change_l2483_248320

theorem salary_change (original_salary : ℝ) (increase_rate : ℝ) (decrease_rate : ℝ) : 
  increase_rate = 0.25 ∧ decrease_rate = 0.25 →
  (1 - decrease_rate) * (1 + increase_rate) * original_salary - original_salary = -0.0625 * original_salary := by
sorry

end NUMINAMATH_CALUDE_salary_change_l2483_248320


namespace NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l2483_248342

theorem product_of_sums_equal_difference_of_powers : 
  (2^1 + 3^1) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) * 
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := by
  sorry

end NUMINAMATH_CALUDE_product_of_sums_equal_difference_of_powers_l2483_248342


namespace NUMINAMATH_CALUDE_probability_ratio_l2483_248367

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of distinct numbers on the slips -/
def distinct_numbers : ℕ := 10

/-- The number of slips drawn -/
def drawn_slips : ℕ := 5

/-- The number of slips for each number -/
def slips_per_number : ℕ := 5

/-- The probability of drawing 5 slips with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

/-- The probability of drawing 2 slips with one number and 3 slips with a different number -/
def q : ℚ := (Nat.choose distinct_numbers 2 * Nat.choose slips_per_number 2 * Nat.choose slips_per_number 3 : ℚ) / (Nat.choose total_slips drawn_slips : ℚ)

theorem probability_ratio :
  q / p = 450 := by sorry

end NUMINAMATH_CALUDE_probability_ratio_l2483_248367


namespace NUMINAMATH_CALUDE_problem_solution_l2483_248379

theorem problem_solution (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) :
  (a^2 + b^2 = 7) ∧ (a < b → a - b = -Real.sqrt 5) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2483_248379


namespace NUMINAMATH_CALUDE_abc_inequality_l2483_248384

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a^2 / (1 + a^2) + b^2 / (1 + b^2) + c^2 / (1 + c^2) = 1) :
  a * b * c ≤ Real.sqrt 2 / 4 := by
sorry

end NUMINAMATH_CALUDE_abc_inequality_l2483_248384


namespace NUMINAMATH_CALUDE_brennans_pepper_theorem_l2483_248389

/-- The amount of pepper remaining after using some from an initial amount -/
def pepper_remaining (initial : ℝ) (used : ℝ) : ℝ :=
  initial - used

/-- Theorem: Given 0.25 grams of pepper initially and using 0.16 grams, 
    the remaining amount is 0.09 grams -/
theorem brennans_pepper_theorem :
  pepper_remaining 0.25 0.16 = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_brennans_pepper_theorem_l2483_248389


namespace NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l2483_248327

theorem smallest_quadratic_coefficient (a : ℕ) : 
  (∃ (b c : ℤ) (x₁ x₂ : ℝ), 
    x₁ ≠ x₂ ∧ 
    0 < x₁ ∧ x₁ < 1 ∧ 
    0 < x₂ ∧ x₂ < 1 ∧ 
    (a : ℝ) * x₁^2 + (b : ℝ) * x₁ + (c : ℝ) = 0 ∧ 
    (a : ℝ) * x₂^2 + (b : ℝ) * x₂ + (c : ℝ) = 0) →
  a ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_smallest_quadratic_coefficient_l2483_248327


namespace NUMINAMATH_CALUDE_largest_integer_less_than_93_remainder_4_mod_7_l2483_248343

theorem largest_integer_less_than_93_remainder_4_mod_7 :
  ∃ n : ℕ, n < 93 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 93 ∧ m % 7 = 4 → m ≤ n :=
by
  use 88
  sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_93_remainder_4_mod_7_l2483_248343


namespace NUMINAMATH_CALUDE_max_distance_in_parallelepiped_l2483_248390

/-- The maximum distance between two points in a 3x4x2 rectangular parallelepiped --/
theorem max_distance_in_parallelepiped :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 2
  ∃ (x₁ y₁ z₁ x₂ y₂ z₂ : ℝ),
    0 ≤ x₁ ∧ x₁ ≤ a ∧
    0 ≤ y₁ ∧ y₁ ≤ b ∧
    0 ≤ z₁ ∧ z₁ ≤ c ∧
    0 ≤ x₂ ∧ x₂ ≤ a ∧
    0 ≤ y₂ ∧ y₂ ≤ b ∧
    0 ≤ z₂ ∧ z₂ ≤ c ∧
    ∀ (x₃ y₃ z₃ x₄ y₄ z₄ : ℝ),
      0 ≤ x₃ ∧ x₃ ≤ a ∧
      0 ≤ y₃ ∧ y₃ ≤ b ∧
      0 ≤ z₃ ∧ z₃ ≤ c ∧
      0 ≤ x₄ ∧ x₄ ≤ a ∧
      0 ≤ y₄ ∧ y₄ ≤ b ∧
      0 ≤ z₄ ∧ z₄ ≤ c →
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 ≥ (x₃ - x₄)^2 + (y₃ - y₄)^2 + (z₃ - z₄)^2 ∧
      (x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2 = 29 := by
  sorry

end NUMINAMATH_CALUDE_max_distance_in_parallelepiped_l2483_248390


namespace NUMINAMATH_CALUDE_function_properties_l2483_248388

noncomputable def f (x m : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 + m

theorem function_properties :
  ∃ m : ℝ,
    (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x m ≤ 1) ∧
    (∃ x ∈ Set.Icc 0 (Real.pi / 4), f x m = 1) ∧
    m = -2 ∧
    (∀ x : ℝ, f x m ≥ -3) ∧
    (∀ k : ℤ, f ((2 * Real.pi / 3) + k * Real.pi) m = -3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l2483_248388


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2483_248329

/-- The sum of a geometric series with 6 terms, first term a, and common ratio r -/
def geometric_sum (a : ℚ) (r : ℚ) : ℚ :=
  a * (1 - r^6) / (1 - r)

/-- The problem statement -/
theorem geometric_series_sum :
  let a : ℚ := 1/5
  let r : ℚ := -1/2
  geometric_sum a r = 21/160 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2483_248329


namespace NUMINAMATH_CALUDE_vector_magnitude_proof_l2483_248366

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-2, 1)

theorem vector_magnitude_proof : ‖(2 • a) + b‖ = 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_proof_l2483_248366


namespace NUMINAMATH_CALUDE_mixed_games_count_l2483_248334

/-- Represents a chess competition with men and women players -/
structure ChessCompetition where
  womenCount : ℕ
  menCount : ℕ
  womenGames : ℕ
  menGames : ℕ

/-- Calculates the number of games between a man and a woman -/
def mixedGames (c : ChessCompetition) : ℕ :=
  c.womenCount * c.menCount

/-- Theorem stating the relationship between the number of games -/
theorem mixed_games_count (c : ChessCompetition) 
  (h1 : c.womenGames = 45)
  (h2 : c.menGames = 190)
  (h3 : c.womenGames = c.womenCount * (c.womenCount - 1) / 2)
  (h4 : c.menGames = c.menCount * (c.menCount - 1) / 2) :
  mixedGames c = 200 := by
  sorry


end NUMINAMATH_CALUDE_mixed_games_count_l2483_248334


namespace NUMINAMATH_CALUDE_salvadore_earnings_l2483_248310

/-- Given that Salvadore earned S dollars, Santo earned half of that, and their combined earnings were $2934, prove that Salvadore earned $1956. -/
theorem salvadore_earnings (S : ℝ) 
  (h1 : S + S / 2 = 2934) : S = 1956 := by
  sorry

end NUMINAMATH_CALUDE_salvadore_earnings_l2483_248310


namespace NUMINAMATH_CALUDE_roots_of_equation_l2483_248393

theorem roots_of_equation : 
  let f : ℝ → ℝ := λ x => (x^3 - 6*x^2 + 11*x - 6)*(x - 2)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3 := by
sorry

end NUMINAMATH_CALUDE_roots_of_equation_l2483_248393


namespace NUMINAMATH_CALUDE_total_teachers_is_210_l2483_248308

/-- Represents the number of teachers in each category and the sample size -/
structure TeacherData where
  senior : ℕ
  intermediate : ℕ
  sample_size : ℕ
  other_sampled : ℕ

/-- Calculates the total number of teachers given the data -/
def totalTeachers (data : TeacherData) : ℕ :=
  sorry

/-- Theorem stating that given the conditions, the total number of teachers is 210 -/
theorem total_teachers_is_210 (data : TeacherData) 
  (h1 : data.senior = 104)
  (h2 : data.intermediate = 46)
  (h3 : data.sample_size = 42)
  (h4 : data.other_sampled = 12)
  (h5 : ∀ (category : ℕ), (category : ℚ) / (totalTeachers data : ℚ) = (data.sample_size : ℚ) / (totalTeachers data : ℚ)) :
  totalTeachers data = 210 :=
sorry

end NUMINAMATH_CALUDE_total_teachers_is_210_l2483_248308


namespace NUMINAMATH_CALUDE_adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l2483_248322

/-- Represents the price of a book at a given position. -/
def book_price (c : ℕ) (n : ℕ) : ℕ := c + 2 * (n - 1)

/-- The theorem stating that the adjacent book is to the left of the middle book. -/
theorem adjacent_book_left_of_middle (c : ℕ) : 
  book_price c 31 = book_price c 16 + book_price c 15 :=
sorry

/-- The theorem stating that the adjacent book cannot be to the right of the middle book. -/
theorem adjacent_book_not_right_of_middle (c : ℕ) : 
  book_price c 31 ≠ book_price c 16 + book_price c 17 :=
sorry

end NUMINAMATH_CALUDE_adjacent_book_left_of_middle_adjacent_book_not_right_of_middle_l2483_248322


namespace NUMINAMATH_CALUDE_shannon_stones_l2483_248369

/-- The number of heart-shaped stones Shannon wants in each bracelet -/
def stones_per_bracelet : ℕ := 8

/-- The number of bracelets Shannon can make -/
def number_of_bracelets : ℕ := 6

/-- The total number of heart-shaped stones Shannon brought -/
def total_stones : ℕ := stones_per_bracelet * number_of_bracelets

theorem shannon_stones : total_stones = 48 := by
  sorry

end NUMINAMATH_CALUDE_shannon_stones_l2483_248369


namespace NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_in_range_l2483_248312

open Real

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x - (1/3) * sin (2*x) + a * sin x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 - (2/3) * cos (2*x) + a * cos x

/-- Theorem stating the range of 'a' for which f(x) is monotonically increasing -/
theorem f_monotone_increasing_iff_a_in_range :
  ∀ a : ℝ, (∀ x : ℝ, Monotone (f a)) ↔ a ∈ Set.Icc (-1/3) (1/3) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_iff_a_in_range_l2483_248312


namespace NUMINAMATH_CALUDE_cara_friends_photo_l2483_248345

theorem cara_friends_photo (n : ℕ) (k : ℕ) : n = 7 → k = 2 → Nat.choose n k = 21 := by
  sorry

end NUMINAMATH_CALUDE_cara_friends_photo_l2483_248345


namespace NUMINAMATH_CALUDE_adult_dogs_adopted_l2483_248380

/-- The number of adult dogs adopted given the costs and number of other animals -/
def num_adult_dogs (cat_cost puppy_cost adult_dog_cost total_cost : ℕ) 
                   (num_cats num_puppies : ℕ) : ℕ :=
  (total_cost - cat_cost * num_cats - puppy_cost * num_puppies) / adult_dog_cost

theorem adult_dogs_adopted :
  num_adult_dogs 50 150 100 700 2 2 = 3 := by
  sorry

end NUMINAMATH_CALUDE_adult_dogs_adopted_l2483_248380


namespace NUMINAMATH_CALUDE_cube_sum_positive_l2483_248382

theorem cube_sum_positive (x y z : ℝ) (h1 : x < y) (h2 : y < z) :
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_positive_l2483_248382


namespace NUMINAMATH_CALUDE_ratio_of_segments_l2483_248381

/-- Given points P, Q, R, and S on a line in that order, with PQ = 3, QR = 6, and PS = 20,
    the ratio of PR to QS is 9/17. -/
theorem ratio_of_segments (P Q R S : ℝ) : 
  P < Q ∧ Q < R ∧ R < S →  -- Points are in order on the line
  Q - P = 3 →              -- PQ = 3
  R - Q = 6 →              -- QR = 6
  S - P = 20 →             -- PS = 20
  (R - P) / (S - Q) = 9 / 17 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_segments_l2483_248381


namespace NUMINAMATH_CALUDE_infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l2483_248350

def IsFantastic (n : ℕ) : Prop :=
  ∃ (a b : ℚ), a > 0 ∧ b > 0 ∧ n = ⌊a + 1/a + b + 1/b⌋

theorem infinite_primes_no_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ (p : ℕ) (k : ℕ), p ∈ S → k > 0 → ¬IsFantastic (k * p)) :=
sorry

theorem infinite_primes_with_fantastic_multiple :
  ∃ (S : Set ℕ), Set.Infinite S ∧ (∀ p ∈ S, Prime p) ∧
    (∀ p ∈ S, ∃ (k : ℕ), k > 0 ∧ IsFantastic (k * p)) :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_no_fantastic_multiple_infinite_primes_with_fantastic_multiple_l2483_248350


namespace NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l2483_248364

/-- The circle with equation (x-a)^2 + (y+a)^2 = 4 -/
def Circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 + a)^2 = 4}

/-- A point is inside the circle if its distance from the center is less than the radius -/
def IsInside (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.1 - a)^2 + (p.2 + a)^2 < 4

/-- The theorem stating that if P(1,1) is inside the circle, then -1 < a < 1 -/
theorem point_inside_circle_implies_a_range :
  ∀ a : ℝ, IsInside (1, 1) a → -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_point_inside_circle_implies_a_range_l2483_248364


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2483_248360

/-- Given that ω = 10 + 3i, prove that |ω² + 10ω + 104| = 212 -/
theorem complex_absolute_value (ω : ℂ) (h : ω = 10 + 3*I) :
  Complex.abs (ω^2 + 10*ω + 104) = 212 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2483_248360


namespace NUMINAMATH_CALUDE_horner_v3_equals_16_l2483_248362

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 7x^7 + 5x^5 + 4x^4 + 2x^2 + x + 2 -/
def f : List ℝ := [7, 0, 5, 4, 2, 1, 2]

/-- v_3 is the fourth intermediate value in Horner's method -/
def v_3 (coeffs : List ℝ) (x : ℝ) : ℝ :=
  (coeffs.take 4).foldl (fun acc a => acc * x + a) 0

theorem horner_v3_equals_16 :
  v_3 f 1 = 16 := by sorry

end NUMINAMATH_CALUDE_horner_v3_equals_16_l2483_248362


namespace NUMINAMATH_CALUDE_strips_intersection_angle_l2483_248398

/-- A strip is defined as the region between two parallel lines. -/
structure Strip where
  width : ℝ

/-- The intersection of two strips forms a parallelogram. -/
structure StripIntersection where
  strip1 : Strip
  strip2 : Strip
  area : ℝ

/-- The angle between two strips is the angle between their defining lines. -/
def angleBetweenStrips (intersection : StripIntersection) : ℝ := sorry

theorem strips_intersection_angle (intersection : StripIntersection) :
  intersection.strip1.width = 1 →
  intersection.strip2.width = 1 →
  intersection.area = 2 →
  angleBetweenStrips intersection = 30 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_strips_intersection_angle_l2483_248398


namespace NUMINAMATH_CALUDE_inequality_solution_l2483_248333

theorem inequality_solution (x : ℝ) : 
  (2 / (x - 2) - 5 / (x - 3) + 5 / (x - 4) - 2 / (x - 5) < 1 / 15) ↔ 
  ((1 < x ∧ x < 2) ∨ (3 < x ∧ x < 6) ∨ (8 < x ∧ x < 10)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2483_248333


namespace NUMINAMATH_CALUDE_soccer_enjoyment_fraction_l2483_248349

theorem soccer_enjoyment_fraction (total : ℝ) (h_total : total > 0) :
  let enjoy_soccer := 0.7 * total
  let dont_enjoy_soccer := 0.3 * total
  let say_enjoy := 0.75 * enjoy_soccer
  let enjoy_but_say_dont := 0.25 * enjoy_soccer
  let say_dont_enjoy := 0.85 * dont_enjoy_soccer
  let total_say_dont := say_dont_enjoy + enjoy_but_say_dont
  enjoy_but_say_dont / total_say_dont = 35 / 86 := by
sorry

end NUMINAMATH_CALUDE_soccer_enjoyment_fraction_l2483_248349


namespace NUMINAMATH_CALUDE_fort_blocks_count_l2483_248305

/-- Represents the dimensions of a rectangular structure -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular structure given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the specifications of the fort -/
structure FortSpecs where
  outerDimensions : Dimensions
  wallThickness : ℕ
  floorThickness : ℕ

/-- Calculates the inner dimensions of the fort given its specifications -/
def innerDimensions (specs : FortSpecs) : Dimensions :=
  { length := specs.outerDimensions.length - 2 * specs.wallThickness,
    width := specs.outerDimensions.width - 2 * specs.wallThickness,
    height := specs.outerDimensions.height - specs.floorThickness }

/-- Calculates the number of blocks needed for the fort -/
def blocksNeeded (specs : FortSpecs) : ℕ :=
  volume specs.outerDimensions - volume (innerDimensions specs)

theorem fort_blocks_count : 
  let fortSpecs : FortSpecs := 
    { outerDimensions := { length := 20, width := 15, height := 8 },
      wallThickness := 2,
      floorThickness := 1 }
  blocksNeeded fortSpecs = 1168 := by sorry

end NUMINAMATH_CALUDE_fort_blocks_count_l2483_248305


namespace NUMINAMATH_CALUDE_power_of_three_even_tens_digit_l2483_248300

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem power_of_three_even_tens_digit (n : ℕ) (h : n ≥ 3) :
  Even (tens_digit (3^n)) := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_even_tens_digit_l2483_248300


namespace NUMINAMATH_CALUDE_brianna_marbles_l2483_248303

/-- The number of marbles Brianna lost through the hole in the bag. -/
def L : ℕ := sorry

/-- The total number of marbles Brianna started with. -/
def total : ℕ := 24

/-- The number of marbles Brianna had remaining. -/
def remaining : ℕ := 10

theorem brianna_marbles : 
  L + 2 * L + L / 2 = total - remaining ∧ L = 4 := by sorry

end NUMINAMATH_CALUDE_brianna_marbles_l2483_248303


namespace NUMINAMATH_CALUDE_negation_of_not_even_numbers_l2483_248378

theorem negation_of_not_even_numbers (a b : ℤ) : 
  ¬(¬Even a ∧ ¬Even b) ↔ (Even a ∨ Even b) :=
sorry

end NUMINAMATH_CALUDE_negation_of_not_even_numbers_l2483_248378


namespace NUMINAMATH_CALUDE_total_profit_is_3872_l2483_248328

/-- Represents the investment and duration for each person -/
structure Investment where
  amount : ℕ
  duration : ℕ

/-- Calculates the total profit given the investments and profit difference -/
def calculateTotalProfit (suresh rohan sudhir : Investment) (profitDifference : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the total profit is 3872 given the problem conditions -/
theorem total_profit_is_3872 :
  let suresh : Investment := ⟨18000, 12⟩
  let rohan : Investment := ⟨12000, 9⟩
  let sudhir : Investment := ⟨9000, 8⟩
  let profitDifference : ℕ := 352
  calculateTotalProfit suresh rohan sudhir profitDifference = 3872 :=
by sorry

end NUMINAMATH_CALUDE_total_profit_is_3872_l2483_248328


namespace NUMINAMATH_CALUDE_xw_value_l2483_248399

theorem xw_value (x w : ℝ) (h1 : 7 * x = 28) (h2 : x + w = 9) : x * w = 20 := by
  sorry

end NUMINAMATH_CALUDE_xw_value_l2483_248399


namespace NUMINAMATH_CALUDE_tan_sin_intersection_count_l2483_248395

open Real

theorem tan_sin_intersection_count :
  let f : ℝ → ℝ := λ x => tan x - sin x
  ∃! (s : Finset ℝ), s.card = 5 ∧ (∀ x ∈ s, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0) ∧
    (∀ x, -2*π ≤ x ∧ x ≤ 2*π ∧ f x = 0 → x ∈ s) :=
by
  sorry

end NUMINAMATH_CALUDE_tan_sin_intersection_count_l2483_248395


namespace NUMINAMATH_CALUDE_probability_both_truth_l2483_248358

theorem probability_both_truth (prob_A prob_B : ℝ) 
  (h1 : prob_A = 0.8) (h2 : prob_B = 0.6) :
  prob_A * prob_B = 0.48 := by
sorry

end NUMINAMATH_CALUDE_probability_both_truth_l2483_248358


namespace NUMINAMATH_CALUDE_probability_non_expired_bottle_l2483_248397

theorem probability_non_expired_bottle (total_bottles : ℕ) (expired_bottles : ℕ) 
  (h1 : total_bottles = 5) (h2 : expired_bottles = 1) : 
  (total_bottles - expired_bottles : ℚ) / total_bottles = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_non_expired_bottle_l2483_248397


namespace NUMINAMATH_CALUDE_book_has_120_pages_l2483_248371

/-- Represents a book reading plan. -/
structure ReadingPlan where
  pagesPerNight : ℕ
  totalDays : ℕ

/-- Calculates the total number of pages in a book given a reading plan. -/
def totalPages (plan : ReadingPlan) : ℕ :=
  plan.pagesPerNight * plan.totalDays

/-- Theorem stating that the book has 120 pages given the specified reading plan. -/
theorem book_has_120_pages :
  ∃ (plan : ReadingPlan),
    plan.pagesPerNight = 12 ∧
    plan.totalDays = 10 ∧
    totalPages plan = 120 := by
  sorry


end NUMINAMATH_CALUDE_book_has_120_pages_l2483_248371


namespace NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_theorem_l2483_248374

-- Define a triangle as a structure with three sides
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0

-- State the Triangle Inequality Theorem
theorem triangle_inequality (t : Triangle) : 
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b := by
  sorry

-- Define the property we want to prove
def sum_of_two_sides_greater_than_third (t : Triangle) : Prop :=
  (t.a + t.b > t.c) ∧ (t.b + t.c > t.a) ∧ (t.c + t.a > t.b)

-- Prove that the Triangle Inequality Theorem holds for all triangles
theorem triangle_inequality_theorem :
  ∀ t : Triangle, sum_of_two_sides_greater_than_third t := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_triangle_inequality_theorem_l2483_248374


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2483_248387

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → (6 * s^2 = 54) → s^3 = 27 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2483_248387


namespace NUMINAMATH_CALUDE_money_left_after_distributions_l2483_248318

/-- Calculates the amount of money left after distributions --/
theorem money_left_after_distributions (income : ℝ) : 
  income = 1000 → 
  income * (1 - 0.2 - 0.2) * (1 - 0.1) = 540 := by
  sorry

#check money_left_after_distributions

end NUMINAMATH_CALUDE_money_left_after_distributions_l2483_248318


namespace NUMINAMATH_CALUDE_figure_area_calculation_l2483_248396

theorem figure_area_calculation (total_area : ℝ) (y : ℝ) : 
  total_area = 1300 →
  (3 * y)^2 + (6 * y)^2 + (1/2 * 3 * y * 6 * y) = total_area →
  y = Real.sqrt 1300 / Real.sqrt 54 := by
  sorry

end NUMINAMATH_CALUDE_figure_area_calculation_l2483_248396


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2483_248325

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem first_term_of_geometric_sequence (a : ℕ → ℝ) :
  IsGeometricSequence a →
  a 2 = 16 →
  a 4 = 128 →
  a 0 = 4 := by
  sorry

end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l2483_248325


namespace NUMINAMATH_CALUDE_sara_initial_quarters_l2483_248352

/-- The number of quarters Sara had initially -/
def initial_quarters : ℕ := sorry

/-- The number of quarters Sara's dad gave her -/
def quarters_from_dad : ℕ := 49

/-- The total number of quarters Sara has after receiving quarters from her dad -/
def total_quarters : ℕ := 70

/-- Theorem stating that Sara initially had 21 quarters -/
theorem sara_initial_quarters : initial_quarters = 21 := by
  sorry

end NUMINAMATH_CALUDE_sara_initial_quarters_l2483_248352


namespace NUMINAMATH_CALUDE_arthur_muffins_arthur_muffins_proof_l2483_248344

theorem arthur_muffins : ℕ → Prop :=
  fun initial_muffins =>
    initial_muffins + 48 = 83 → initial_muffins = 35

-- Proof
theorem arthur_muffins_proof : arthur_muffins 35 := by
  sorry

end NUMINAMATH_CALUDE_arthur_muffins_arthur_muffins_proof_l2483_248344


namespace NUMINAMATH_CALUDE_number_division_sum_l2483_248354

theorem number_division_sum : ∃! N : ℕ, ∃ Q : ℕ, N = 11 * Q ∧ Q + N + 11 = 71 := by
  sorry

end NUMINAMATH_CALUDE_number_division_sum_l2483_248354


namespace NUMINAMATH_CALUDE_current_velocity_l2483_248365

/-- Velocity of current given rowing speed and round trip time -/
theorem current_velocity (rowing_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  rowing_speed = 5 →
  distance = 2.4 →
  total_time = 1 →
  ∃ v : ℝ,
    v > 0 ∧
    (distance / (rowing_speed - v) + distance / (rowing_speed + v) = total_time) ∧
    v = 1 := by
  sorry

end NUMINAMATH_CALUDE_current_velocity_l2483_248365


namespace NUMINAMATH_CALUDE_blue_spools_count_l2483_248314

/-- The number of spools needed to make one beret -/
def spools_per_beret : ℕ := 3

/-- The number of red yarn spools -/
def red_spools : ℕ := 12

/-- The number of black yarn spools -/
def black_spools : ℕ := 15

/-- The total number of berets that can be made -/
def total_berets : ℕ := 11

/-- The number of blue yarn spools -/
def blue_spools : ℕ := total_berets * spools_per_beret - (red_spools + black_spools)

theorem blue_spools_count : blue_spools = 6 := by
  sorry

end NUMINAMATH_CALUDE_blue_spools_count_l2483_248314


namespace NUMINAMATH_CALUDE_simplify_expression_l2483_248339

theorem simplify_expression : (81 ^ (1/4) - Real.sqrt 12.25) ^ 2 = 1/4 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2483_248339


namespace NUMINAMATH_CALUDE_solution_set_equals_given_values_l2483_248386

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- The set of solutions to the equation n = 2S(n)³ + 8 -/
def SolutionSet : Set ℕ := {n : ℕ | n > 0 ∧ n = 2 * (S n)^3 + 8}

/-- The theorem stating that the solution set contains exactly 10, 2008, and 13726 -/
theorem solution_set_equals_given_values : 
  SolutionSet = {10, 2008, 13726} := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_given_values_l2483_248386


namespace NUMINAMATH_CALUDE_farm_animals_l2483_248317

/-- Given a farm with cows and horses, prove the number of horses -/
theorem farm_animals (cow_count : ℕ) (horse_count : ℕ) : 
  (cow_count : ℚ) / horse_count = 7 / 2 → cow_count = 21 → horse_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l2483_248317


namespace NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2483_248326

def taxi_cost (initial_cost : ℝ) (cost_per_mile : ℝ) (distance : ℝ) : ℝ :=
  initial_cost + cost_per_mile * distance

theorem ten_mile_taxi_cost :
  taxi_cost 2 0.3 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ten_mile_taxi_cost_l2483_248326


namespace NUMINAMATH_CALUDE_fraction_of_girls_l2483_248357

theorem fraction_of_girls (total_students : ℕ) (boys : ℕ) (h1 : total_students = 160) (h2 : boys = 60) :
  (total_students - boys : ℚ) / total_students = 5 / 8 := by
  sorry

#check fraction_of_girls

end NUMINAMATH_CALUDE_fraction_of_girls_l2483_248357


namespace NUMINAMATH_CALUDE_family_weight_difference_l2483_248335

theorem family_weight_difference (m₂ : ℝ) (x₁ x₂ : ℝ) :
  let m₁ := m₂ + 1
  let family₁_before := 10 * m₁
  let family₂_before := 10 * m₂
  let family₁_after := (10 * m₁ - x₁) / 9
  let family₂_after := (10 * m₂ - x₂) / 9
  (family₁_before = family₂_before + 10) →
  ((family₁_after = family₂_after + 1) ∨ (family₂_after = family₁_after + 1)) →
  (x₁ - x₂ = 1 ∨ x₂ - x₁ = 19) :=
by sorry

end NUMINAMATH_CALUDE_family_weight_difference_l2483_248335


namespace NUMINAMATH_CALUDE_intersection_point_l2483_248315

/-- The line equation -/
def line (x y z : ℝ) : Prop :=
  (x - 2) / 4 = (y - 1) / (-3) ∧ (y - 1) / (-3) = (z + 3) / (-2)

/-- The plane equation -/
def plane (x y z : ℝ) : Prop :=
  3 * x - y + 4 * z = 0

/-- The theorem stating that (6, -2, -5) is the unique point of intersection -/
theorem intersection_point : ∃! (x y z : ℝ), line x y z ∧ plane x y z ∧ x = 6 ∧ y = -2 ∧ z = -5 := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_l2483_248315


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l2483_248323

/-- Parabola defined by y² = 8x -/
def Parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- Focus of the parabola -/
def Focus : ℝ × ℝ := (2, 0)

/-- Line passing through the focus with slope k -/
def Line (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 2)

/-- Point M -/
def M : ℝ × ℝ := (-2, 2)

/-- Intersection points of the line and the parabola -/
def Intersects (k : ℝ) (A B : ℝ × ℝ) : Prop :=
  Parabola A.1 A.2 ∧ Parabola B.1 B.2 ∧
  Line k A.1 A.2 ∧ Line k B.1 B.2

/-- Vector dot product -/
def DotProduct (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem parabola_line_intersection (k : ℝ) :
  ∃ A B : ℝ × ℝ, Intersects k A B ∧
  DotProduct (A.1 + 2, A.2 - 2) (B.1 + 2, B.2 - 2) = 0 →
  k = 2 :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l2483_248323


namespace NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_twice_original_radius_l2483_248376

/-- Represents a tetrahedron with an insphere and four smaller tetrahedrons -/
structure Tetrahedron where
  r : ℝ  -- radius of the insphere of the original tetrahedron
  r₁ : ℝ  -- radius of the insphere of the first smaller tetrahedron
  r₂ : ℝ  -- radius of the insphere of the second smaller tetrahedron
  r₃ : ℝ  -- radius of the insphere of the third smaller tetrahedron
  r₄ : ℝ  -- radius of the insphere of the fourth smaller tetrahedron

/-- The sum of the radii of the inspheres of the four smaller tetrahedrons is equal to twice the radius of the insphere of the original tetrahedron -/
theorem sum_of_smaller_radii_eq_twice_original_radius (t : Tetrahedron) :
  t.r₁ + t.r₂ + t.r₃ + t.r₄ = 2 * t.r := by
  sorry

end NUMINAMATH_CALUDE_sum_of_smaller_radii_eq_twice_original_radius_l2483_248376


namespace NUMINAMATH_CALUDE_special_lines_intersect_l2483_248391

/-- Given a triangle ABC with incircle center I and excircle center I_A -/
structure Triangle :=
  (A B C I I_A : EuclideanSpace ℝ (Fin 2))

/-- Line passing through orthocenters of triangles formed by vertices, incircle center, and excircle center -/
def special_line (T : Triangle) (v : Fin 3) : Set (EuclideanSpace ℝ (Fin 2)) :=
  sorry

/-- The theorem states that the three special lines intersect at a single point -/
theorem special_lines_intersect (T : Triangle) :
  ∃! P, P ∈ (special_line T 0) ∧ P ∈ (special_line T 1) ∧ P ∈ (special_line T 2) :=
sorry

end NUMINAMATH_CALUDE_special_lines_intersect_l2483_248391


namespace NUMINAMATH_CALUDE_problem_1997_2000_l2483_248324

theorem problem_1997_2000 : 1997 * (2000 / 2000) - 2000 * (1997 / 1997) = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1997_2000_l2483_248324


namespace NUMINAMATH_CALUDE_quarterback_throws_l2483_248332

/-- Proves that given the specified conditions, the quarterback stepped back to throw 80 times. -/
theorem quarterback_throws (p_no_throw : ℝ) (p_sack_given_no_throw : ℝ) (num_sacks : ℕ) :
  p_no_throw = 0.3 →
  p_sack_given_no_throw = 0.5 →
  num_sacks = 12 →
  ∃ (total_throws : ℕ), total_throws = 80 ∧ 
    (p_no_throw * p_sack_given_no_throw * total_throws : ℝ) = num_sacks := by
  sorry

#check quarterback_throws

end NUMINAMATH_CALUDE_quarterback_throws_l2483_248332


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l2483_248316

theorem negation_of_absolute_value_inequality :
  (¬ ∀ x : ℝ, |x - 1| ≥ 2) ↔ (∃ x : ℝ, |x - 1| < 2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_inequality_l2483_248316


namespace NUMINAMATH_CALUDE_three_from_fifteen_combination_l2483_248330

theorem three_from_fifteen_combination : (Nat.choose 15 3) = 455 := by sorry

end NUMINAMATH_CALUDE_three_from_fifteen_combination_l2483_248330


namespace NUMINAMATH_CALUDE_relay_race_ratio_l2483_248346

theorem relay_race_ratio (total_members : Nat) (other_members : Nat) (other_distance : ℝ) (total_distance : ℝ) :
  total_members = 5 →
  other_members = 4 →
  other_distance = 3 →
  total_distance = 18 →
  (total_distance - other_members * other_distance) / other_distance = 2 := by
  sorry

end NUMINAMATH_CALUDE_relay_race_ratio_l2483_248346


namespace NUMINAMATH_CALUDE_apple_eating_contest_difference_l2483_248363

/-- Represents the result of an apple eating contest -/
structure ContestResult where
  numStudents : Nat
  applesCounts : List Nat
  maxEater : Nat
  minEater : Nat

/-- Theorem stating the difference between the maximum and minimum number of apples eaten -/
theorem apple_eating_contest_difference (result : ContestResult)
  (h1 : result.numStudents = 8)
  (h2 : result.applesCounts.length = result.numStudents)
  (h3 : result.maxEater ∈ result.applesCounts)
  (h4 : result.minEater ∈ result.applesCounts)
  (h5 : ∀ x ∈ result.applesCounts, x ≤ result.maxEater ∧ x ≥ result.minEater) :
  result.maxEater - result.minEater = 8 :=
by sorry

end NUMINAMATH_CALUDE_apple_eating_contest_difference_l2483_248363


namespace NUMINAMATH_CALUDE_equal_numbers_exist_l2483_248336

theorem equal_numbers_exist (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b^2 + c^2 = a^2 + b + c^2 ∧ a^2 + b + c^2 = a^2 + b^2 + c) :
  a = b ∨ b = c ∨ a = c :=
by sorry

end NUMINAMATH_CALUDE_equal_numbers_exist_l2483_248336


namespace NUMINAMATH_CALUDE_mobius_rest_stop_time_l2483_248348

/-- Proves that the rest stop time for each half of the trip is 1 hour given the conditions of Mobius's journey --/
theorem mobius_rest_stop_time 
  (distance : ℝ) 
  (speed_with_load : ℝ) 
  (speed_without_load : ℝ) 
  (total_trip_time : ℝ) 
  (h1 : distance = 143) 
  (h2 : speed_with_load = 11) 
  (h3 : speed_without_load = 13) 
  (h4 : total_trip_time = 26) : 
  (total_trip_time - (distance / speed_with_load + distance / speed_without_load)) / 2 = 1 := by
sorry

end NUMINAMATH_CALUDE_mobius_rest_stop_time_l2483_248348


namespace NUMINAMATH_CALUDE_tank_capacity_l2483_248370

/-- The capacity of a tank with specific leak and inlet properties -/
theorem tank_capacity 
  (leak_empty_time : ℝ) 
  (inlet_rate : ℝ) 
  (combined_empty_time : ℝ) 
  (h1 : leak_empty_time = 6) 
  (h2 : inlet_rate = 2.5) 
  (h3 : combined_empty_time = 8) : 
  ∃ C : ℝ, C = 3600 / 7 ∧ 
    C / leak_empty_time - inlet_rate * 60 = C / combined_empty_time :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l2483_248370


namespace NUMINAMATH_CALUDE_original_denominator_problem_l2483_248338

theorem original_denominator_problem (d : ℚ) : 
  (3 : ℚ) / d ≠ 0 →
  (11 : ℚ) / (d + 8) = 2 / 5 →
  d = 39 / 2 :=
by sorry

end NUMINAMATH_CALUDE_original_denominator_problem_l2483_248338


namespace NUMINAMATH_CALUDE_male_associate_or_full_tenured_percentage_l2483_248385

structure University where
  total_professors : ℕ
  women_professors : ℕ
  tenured_professors : ℕ
  associate_or_full_professors : ℕ
  women_or_tenured_professors : ℕ
  male_associate_or_full_professors : ℕ

def University.valid (u : University) : Prop :=
  u.women_professors = (70 * u.total_professors) / 100 ∧
  u.tenured_professors = (70 * u.total_professors) / 100 ∧
  u.associate_or_full_professors = (50 * u.total_professors) / 100 ∧
  u.women_or_tenured_professors = (90 * u.total_professors) / 100 ∧
  u.male_associate_or_full_professors = (80 * u.associate_or_full_professors) / 100

theorem male_associate_or_full_tenured_percentage (u : University) (h : u.valid) :
  (u.tenured_professors - u.women_professors + (u.women_or_tenured_professors - u.total_professors)) * 100 / u.male_associate_or_full_professors = 50 := by
  sorry

end NUMINAMATH_CALUDE_male_associate_or_full_tenured_percentage_l2483_248385


namespace NUMINAMATH_CALUDE_square_number_ratio_l2483_248331

theorem square_number_ratio (k : ℕ) (h : k ≥ 2) :
  ∀ a b : ℕ, a ≠ 0 → b ≠ 0 →
  (a^2 + b^2) / (a * b + 1) = k^2 ↔ a = k ∧ b = k^3 := by
sorry

end NUMINAMATH_CALUDE_square_number_ratio_l2483_248331
