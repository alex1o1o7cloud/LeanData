import Mathlib

namespace NUMINAMATH_CALUDE_plane_through_point_parallel_to_plane_l2008_200830

/-- A plane in 3D space --/
structure Plane where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  a_pos : a > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

/-- A point in 3D space --/
structure Point where
  x : ℤ
  y : ℤ
  z : ℤ

def Plane.contains (p : Plane) (pt : Point) : Prop :=
  p.a * pt.x + p.b * pt.y + p.c * pt.z + p.d = 0

def Plane.isParallelTo (p1 p2 : Plane) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ p1.a = k * p2.a ∧ p1.b = k * p2.b ∧ p1.c = k * p2.c

theorem plane_through_point_parallel_to_plane 
  (given_plane : Plane) 
  (point : Point) :
  ∃ (result_plane : Plane), 
    result_plane.contains point ∧ 
    result_plane.isParallelTo given_plane ∧
    result_plane.a = 3 ∧ 
    result_plane.b = -2 ∧ 
    result_plane.c = 4 ∧ 
    result_plane.d = -19 := by
  sorry

end NUMINAMATH_CALUDE_plane_through_point_parallel_to_plane_l2008_200830


namespace NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l2008_200816

theorem inverse_expression_equals_one_fifth :
  (2 - 3 * (2 - 3)⁻¹)⁻¹ = (1 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_expression_equals_one_fifth_l2008_200816


namespace NUMINAMATH_CALUDE_satisfying_numbers_are_741_234_975_468_l2008_200826

-- Define a structure for three-digit numbers
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

-- Define the property of middle digit being the arithmetic mean
def isMiddleDigitMean (n : ThreeDigitNumber) : Prop :=
  2 * n.tens = n.hundreds + n.ones

-- Define divisibility by 13
def isDivisibleBy13 (n : ThreeDigitNumber) : Prop :=
  (100 * n.hundreds + 10 * n.tens + n.ones) % 13 = 0

-- Define the set of numbers satisfying both conditions
def satisfyingNumbers : Set ThreeDigitNumber :=
  {n | isMiddleDigitMean n ∧ isDivisibleBy13 n}

-- The theorem to prove
theorem satisfying_numbers_are_741_234_975_468 :
  satisfyingNumbers = {
    ⟨7, 4, 1, by norm_num, by norm_num, by norm_num⟩,
    ⟨2, 3, 4, by norm_num, by norm_num, by norm_num⟩,
    ⟨9, 7, 5, by norm_num, by norm_num, by norm_num⟩,
    ⟨4, 6, 8, by norm_num, by norm_num, by norm_num⟩
  } := by sorry


end NUMINAMATH_CALUDE_satisfying_numbers_are_741_234_975_468_l2008_200826


namespace NUMINAMATH_CALUDE_slope_less_than_two_l2008_200897

/-- Theorem: For two different points on a linear function, if the product of differences is negative, then the slope is less than 2 -/
theorem slope_less_than_two (k a b c d : ℝ) : 
  a ≠ c →  -- A and B are different points
  b = k * a - 2 * a - 1 →  -- A is on the line
  d = k * c - 2 * c - 1 →  -- B is on the line
  (c - a) * (d - b) < 0 →  -- Given condition
  k < 2 := by
  sorry


end NUMINAMATH_CALUDE_slope_less_than_two_l2008_200897


namespace NUMINAMATH_CALUDE_parallelepiped_covering_l2008_200828

/-- A parallelepiped constructed from 4 identical unit cubes stacked vertically -/
structure Parallelepiped :=
  (height : ℕ)
  (width : ℕ)
  (depth : ℕ)
  (is_valid : height = 4 ∧ width = 1 ∧ depth = 1)

/-- A square with side length n -/
structure Square (n : ℕ) :=
  (side_length : ℕ)
  (is_valid : side_length = n)

/-- The covering of the parallelepiped -/
structure Covering (p : Parallelepiped) :=
  (vertical_square : Square 4)
  (top_square : Square 1)
  (bottom_square : Square 1)

/-- The theorem stating that the parallelepiped can be covered by three squares -/
theorem parallelepiped_covering (p : Parallelepiped) :
  ∃ (c : Covering p),
    (c.vertical_square.side_length ^ 2 = 2 * p.height * p.width + 2 * p.height * p.depth) ∧
    (c.top_square.side_length ^ 2 = p.width * p.depth) ∧
    (c.bottom_square.side_length ^ 2 = p.width * p.depth) :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_covering_l2008_200828


namespace NUMINAMATH_CALUDE_equal_adjacent_sides_not_imply_square_l2008_200869

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define properties of quadrilaterals
def has_equal_adjacent_sides (q : Quadrilateral) : Prop :=
  sorry

def is_square (q : Quadrilateral) : Prop :=
  sorry

-- Theorem statement
theorem equal_adjacent_sides_not_imply_square :
  ∃ q : Quadrilateral, has_equal_adjacent_sides q ∧ ¬ is_square q :=
sorry

end NUMINAMATH_CALUDE_equal_adjacent_sides_not_imply_square_l2008_200869


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2008_200858

/-- Given that the solution set of x^2 + ax + b < 0 is (1, 2), 
    prove that the solution set of bx^2 + ax + 1 > 0 is (-∞, 1/2) ∪ (1, +∞) -/
theorem solution_set_inequality (a b : ℝ) : 
  (∀ x, x^2 + a*x + b < 0 ↔ 1 < x ∧ x < 2) →
  (∀ x, b*x^2 + a*x + 1 > 0 ↔ x < (1/2) ∨ x > 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2008_200858


namespace NUMINAMATH_CALUDE_range_of_t_largest_circle_range_of_t_with_P_inside_l2008_200822

-- Define the circle equation
def circle_equation (x y t : ℝ) : Prop :=
  x^2 + y^2 - 2*(t+3)*x + 2*(1-4*t^2)*y + 16*t^4 + 9 = 0

-- Define the point P
def point_P (t : ℝ) : ℝ × ℝ := (3, 4*t^2)

-- Theorem for the range of t
theorem range_of_t : ∀ t : ℝ, circle_equation x y t → -1/7 < t ∧ t < 1 :=
sorry

-- Theorem for the circle with the largest area
theorem largest_circle : ∃ t : ℝ, 
  ∀ x y : ℝ, circle_equation x y t → (x - 24/7)^2 + (y + 13/49)^2 = 16/7 :=
sorry

-- Theorem for the range of t when P is inside the circle
theorem range_of_t_with_P_inside : 
  ∀ t : ℝ, (∀ x y : ℝ, circle_equation x y t → 
    (point_P t).1^2 + (point_P t).2^2 < (x - (t+3))^2 + (y - (4*t^2-1))^2) →
  0 < t ∧ t < 3/4 :=
sorry

end NUMINAMATH_CALUDE_range_of_t_largest_circle_range_of_t_with_P_inside_l2008_200822


namespace NUMINAMATH_CALUDE_train_length_proof_l2008_200853

/-- Proves that the length of a train is 250 meters given specific conditions --/
theorem train_length_proof (bridge_length : ℝ) (time_to_pass : ℝ) (train_speed_kmh : ℝ) :
  bridge_length = 150 →
  time_to_pass = 41.142857142857146 →
  train_speed_kmh = 35 →
  ∃ (train_length : ℝ), train_length = 250 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_proof_l2008_200853


namespace NUMINAMATH_CALUDE_wrapping_paper_area_l2008_200888

/-- The area of a square sheet of wrapping paper required to wrap a rectangular box -/
theorem wrapping_paper_area (w h : ℝ) (hw : w > h) : 
  let l := 2 * w
  let box_diagonal := Real.sqrt (l^2 + w^2 + h^2)
  box_diagonal^2 = 5 * w^2 + h^2 := by
  sorry

end NUMINAMATH_CALUDE_wrapping_paper_area_l2008_200888


namespace NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2008_200827

open Real

theorem function_inequality_implies_parameter_bound (a : ℝ) :
  (∃ x : ℝ, x > 1 ∧ -a * x^2 + log x > -a) → a < 1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_parameter_bound_l2008_200827


namespace NUMINAMATH_CALUDE_silverware_per_setting_l2008_200803

/-- Proves the number of pieces of silverware per setting for a catering event -/
theorem silverware_per_setting :
  let silverware_weight : ℕ := 4  -- weight of each piece of silverware in ounces
  let plate_weight : ℕ := 12  -- weight of each plate in ounces
  let plates_per_setting : ℕ := 2  -- number of plates per setting
  let tables : ℕ := 15  -- number of tables
  let settings_per_table : ℕ := 8  -- number of settings per table
  let backup_settings : ℕ := 20  -- number of backup settings
  let total_weight : ℕ := 5040  -- total weight of all settings in ounces

  let total_settings : ℕ := tables * settings_per_table + backup_settings
  let plate_weight_per_setting : ℕ := plate_weight * plates_per_setting

  ∃ (silverware_per_setting : ℕ),
    silverware_per_setting * silverware_weight * total_settings +
    plate_weight_per_setting * total_settings = total_weight ∧
    silverware_per_setting = 3 :=
by sorry

end NUMINAMATH_CALUDE_silverware_per_setting_l2008_200803


namespace NUMINAMATH_CALUDE_abc_product_l2008_200812

theorem abc_product (a b c : ℕ) : 
  a * b * c + a * b + b * c + a * c + a + b + c = 164 → a * b * c = 80 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2008_200812


namespace NUMINAMATH_CALUDE_union_with_complement_l2008_200851

/-- Given sets A and B in the real number line, prove that their union with the complement of B is equal to the set of all real numbers less than -2 or greater than or equal to 0. -/
theorem union_with_complement (A B : Set ℝ) 
  (hA : A = {x | (x + 2) * (x - 1) > 0})
  (hB : B = {x | -3 ≤ x ∧ x < 0}) :
  A ∪ (Set.univ \ B) = {x | x < -2 ∨ x ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_union_with_complement_l2008_200851


namespace NUMINAMATH_CALUDE_inequality_solution_l2008_200838

-- Define the inequality function
def f (y : ℝ) : Prop :=
  y^3 / (y + 2) ≥ 3 / (y - 2) + 9/4

-- Define the solution set
def S : Set ℝ :=
  {y | y ∈ Set.Ioo (-2) 2 ∨ y ∈ Set.Ici 3}

-- State the theorem
theorem inequality_solution :
  {y : ℝ | f y} = S :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2008_200838


namespace NUMINAMATH_CALUDE_course_selection_ways_l2008_200892

theorem course_selection_ways (n : ℕ) (k : ℕ) (m : ℕ) :
  n = 6 → k = 3 → m = 1 →
  (n.choose m) * ((n - m).choose (k - m)) * ((k).choose m) = 180 :=
by sorry

end NUMINAMATH_CALUDE_course_selection_ways_l2008_200892


namespace NUMINAMATH_CALUDE_balloon_descent_rate_l2008_200807

/-- Hot air balloon ascent and descent rates problem -/
theorem balloon_descent_rate 
  (rise_rate : ℝ) 
  (total_pull_time : ℝ) 
  (release_time : ℝ) 
  (max_height : ℝ) 
  (h_rise_rate : rise_rate = 50)
  (h_total_pull_time : total_pull_time = 30)
  (h_release_time : release_time = 10)
  (h_max_height : max_height = 1400) :
  ∃ (descent_rate : ℝ),
    max_height = rise_rate * total_pull_time - descent_rate * release_time ∧
    descent_rate = 10 := by
  sorry

end NUMINAMATH_CALUDE_balloon_descent_rate_l2008_200807


namespace NUMINAMATH_CALUDE_gcd_of_rope_lengths_l2008_200867

theorem gcd_of_rope_lengths : Nat.gcd 825 1275 = 75 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_rope_lengths_l2008_200867


namespace NUMINAMATH_CALUDE_room_breadth_is_ten_l2008_200876

/-- Proves that the breadth of a room is 10 feet given specific conditions -/
theorem room_breadth_is_ten (room_length : ℝ) (tile_size : ℝ) (blue_tile_count : ℕ) : 
  room_length = 20 ∧ 
  tile_size = 2 ∧ 
  blue_tile_count = 16 →
  ∃ (room_breadth : ℝ),
    room_breadth = 10 ∧
    (room_length - 2 * tile_size) * (room_breadth - 2 * tile_size) * (2/3) = 
      (blue_tile_count : ℝ) * tile_size^2 :=
by sorry


end NUMINAMATH_CALUDE_room_breadth_is_ten_l2008_200876


namespace NUMINAMATH_CALUDE_event_probability_l2008_200834

theorem event_probability (p : ℝ) : 
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by sorry

end NUMINAMATH_CALUDE_event_probability_l2008_200834


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2008_200837

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculate the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  trapezoid_area t = 1336 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2008_200837


namespace NUMINAMATH_CALUDE_mark_ordered_100_nuggets_l2008_200861

/-- The number of chicken nuggets in a box -/
def nuggets_per_box : ℕ := 20

/-- The cost of one box of chicken nuggets in dollars -/
def cost_per_box : ℕ := 4

/-- The amount Mark paid for chicken nuggets in dollars -/
def mark_paid : ℕ := 20

/-- The number of chicken nuggets Mark ordered -/
def mark_ordered : ℕ := (mark_paid / cost_per_box) * nuggets_per_box

theorem mark_ordered_100_nuggets : mark_ordered = 100 := by
  sorry

end NUMINAMATH_CALUDE_mark_ordered_100_nuggets_l2008_200861


namespace NUMINAMATH_CALUDE_line_passes_through_first_and_fourth_quadrants_l2008_200810

-- Define the line y = kx + b
def line (k b x : ℝ) : ℝ := k * x + b

-- Define the condition bk < 0
def condition (b k : ℝ) : Prop := b * k < 0

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Theorem statement
theorem line_passes_through_first_and_fourth_quadrants (b k : ℝ) :
  condition b k →
  (∃ x y : ℝ, y = line k b x ∧ first_quadrant x y) ∧
  (∃ x y : ℝ, y = line k b x ∧ fourth_quadrant x y) :=
sorry

end NUMINAMATH_CALUDE_line_passes_through_first_and_fourth_quadrants_l2008_200810


namespace NUMINAMATH_CALUDE_box_2_3_neg1_l2008_200848

def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

theorem box_2_3_neg1 : box 2 3 (-1) = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_box_2_3_neg1_l2008_200848


namespace NUMINAMATH_CALUDE_cubic_function_identity_l2008_200836

/-- Given a cubic function g(x) = px³ + qx² + rx + s where g(3) = 4,
    prove that 6p - 3q + r - 2s = 60p + 15q + 7r - 8 -/
theorem cubic_function_identity (p q r s : ℝ) 
  (h : 27 * p + 9 * q + 3 * r + s = 4) :
  6 * p - 3 * q + r - 2 * s = 60 * p + 15 * q + 7 * r - 8 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_identity_l2008_200836


namespace NUMINAMATH_CALUDE_cube_surface_area_difference_l2008_200877

theorem cube_surface_area_difference : 
  let large_cube_volume : ℝ := 343
  let small_cube_count : ℕ := 343
  let small_cube_volume : ℝ := 1
  let large_cube_side : ℝ := large_cube_volume ^ (1/3)
  let large_cube_surface_area : ℝ := 6 * large_cube_side ^ 2
  let small_cube_side : ℝ := small_cube_volume ^ (1/3)
  let small_cube_surface_area : ℝ := 6 * small_cube_side ^ 2
  let total_small_cubes_surface_area : ℝ := small_cube_count * small_cube_surface_area
  total_small_cubes_surface_area - large_cube_surface_area = 1764 := by
sorry


end NUMINAMATH_CALUDE_cube_surface_area_difference_l2008_200877


namespace NUMINAMATH_CALUDE_work_completion_rate_l2008_200866

/-- Given workers A and B, where A can finish a work in 6 days and B can do the same work in half the time taken by A, prove that A and B working together can finish 1/2 of the work in one day. -/
theorem work_completion_rate (days_a : ℕ) (days_b : ℕ) : 
  days_a = 6 →
  days_b = days_a / 2 →
  (1 : ℚ) / days_a + (1 : ℚ) / days_b = (1 : ℚ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_rate_l2008_200866


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2008_200850

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, x^2 + b*x - 2*a > 0 ↔ 2 < x ∧ x < 3) →
  a + b = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2008_200850


namespace NUMINAMATH_CALUDE_simplify_expression_l2008_200874

theorem simplify_expression :
  (2 + 3) * (2^3 + 3^3) * (2^9 + 3^9) * (2^27 + 3^27) = 3^41 - 2^41 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2008_200874


namespace NUMINAMATH_CALUDE_problem_solution_l2008_200843

theorem problem_solution (a b m : ℚ) 
  (h1 : 2 * a = m) 
  (h2 : 5 * b = m) 
  (h3 : a + b = 2) : 
  m = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2008_200843


namespace NUMINAMATH_CALUDE_line_intersecting_ellipse_slope_l2008_200896

/-- The slope of a line intersecting an ellipse -/
theorem line_intersecting_ellipse_slope (m : ℝ) : 
  (∃ x y : ℝ, 25 * x^2 + 16 * y^2 = 400 ∧ y = m * x + 8) ↔ m^2 ≥ 39/16 :=
sorry

end NUMINAMATH_CALUDE_line_intersecting_ellipse_slope_l2008_200896


namespace NUMINAMATH_CALUDE_root_location_l2008_200886

/-- Given a function f(x) = a^x + x - b with a root x₀ ∈ (n, n+1), where n is an integer,
    and constants a and b satisfying 2^a = 3 and 3^b = 2, prove that n = -1. -/
theorem root_location (a b : ℝ) (n : ℤ) (x₀ : ℝ) :
  (2 : ℝ) ^ a = 3 →
  (3 : ℝ) ^ b = 2 →
  (∃ x₀, x₀ ∈ Set.Ioo n (n + 1) ∧ a ^ x₀ + x₀ - b = 0) →
  n = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_location_l2008_200886


namespace NUMINAMATH_CALUDE_inequality_holds_l2008_200887

-- Define an even function that is increasing on (-∞, 0]
def EvenIncreasingNegative (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

-- State the theorem
theorem inequality_holds (f : ℝ → ℝ) (h : EvenIncreasingNegative f) :
  ∀ a : ℝ, f 1 > f (a^2 + 2*a + 3) :=
sorry

end NUMINAMATH_CALUDE_inequality_holds_l2008_200887


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l2008_200806

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_power_sum : i^25 + i^125 = 2*i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l2008_200806


namespace NUMINAMATH_CALUDE_unique_divisor_with_remainder_l2008_200856

theorem unique_divisor_with_remainder (n : ℕ) (r : ℕ) : 
  (∃! u : ℕ, u > r ∧ n % u = r) ↔ n = 11 ∧ r = 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisor_with_remainder_l2008_200856


namespace NUMINAMATH_CALUDE_square_fraction_implies_properties_l2008_200860

theorem square_fraction_implies_properties (n : ℕ+) 
  (h : ∃ m : ℕ, (n * (n + 1)) / 3 = m ^ 2) :
  (∃ k : ℕ, n = 3 * k) ∧ 
  (∃ b : ℕ, n + 1 = b ^ 2) ∧ 
  (∃ a : ℕ, n / 3 = a ^ 2) := by
sorry

end NUMINAMATH_CALUDE_square_fraction_implies_properties_l2008_200860


namespace NUMINAMATH_CALUDE_find_divisor_l2008_200849

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = quotient * divisor + remainder →
  dividend = 265 →
  quotient = 12 →
  remainder = 1 →
  divisor = 22 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2008_200849


namespace NUMINAMATH_CALUDE_quadrilateral_not_necessarily_plane_l2008_200890

-- Define the types of figures we're considering
inductive Figure
| Triangle
| Trapezoid
| Parallelogram
| Quadrilateral

-- Define what it means for a figure to be a plane figure
def is_plane_figure (f : Figure) : Prop :=
  match f with
  | Figure.Triangle => True
  | Figure.Trapezoid => True
  | Figure.Parallelogram => True
  | Figure.Quadrilateral => false

-- Theorem stating that only quadrilaterals are not necessarily plane figures
theorem quadrilateral_not_necessarily_plane :
  ∀ f : Figure, ¬(is_plane_figure f) ↔ f = Figure.Quadrilateral :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_not_necessarily_plane_l2008_200890


namespace NUMINAMATH_CALUDE_edge_pairs_determine_plane_l2008_200846

/-- A regular octahedron -/
structure RegularOctahedron where
  /-- The number of edges in a regular octahedron -/
  num_edges : ℕ
  /-- The number of edges that intersect with any given edge -/
  num_intersecting_edges : ℕ
  /-- Property: A regular octahedron has 12 edges -/
  edge_count : num_edges = 12
  /-- Property: Each edge intersects with 8 other edges -/
  intersecting_edge_count : num_intersecting_edges = 8

/-- The number of unordered pairs of edges that determine a plane in a regular octahedron -/
def num_edge_pairs_determine_plane (o : RegularOctahedron) : ℕ :=
  (o.num_edges * o.num_intersecting_edges) / 2

/-- Theorem: The number of unordered pairs of edges that determine a plane in a regular octahedron is 48 -/
theorem edge_pairs_determine_plane (o : RegularOctahedron) :
  num_edge_pairs_determine_plane o = 48 := by
  sorry

end NUMINAMATH_CALUDE_edge_pairs_determine_plane_l2008_200846


namespace NUMINAMATH_CALUDE_right_triangle_adjacent_side_l2008_200825

theorem right_triangle_adjacent_side (h a o : ℝ) (h_positive : h > 0) (a_positive : a > 0) (o_positive : o > 0) 
  (hypotenuse : h = 8) (opposite : o = 5) (pythagorean : h^2 = a^2 + o^2) : a = Real.sqrt 39 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_adjacent_side_l2008_200825


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l2008_200847

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l2008_200847


namespace NUMINAMATH_CALUDE_sum_of_fractions_l2008_200855

theorem sum_of_fractions : 2 / 5 + 3 / 8 = 31 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l2008_200855


namespace NUMINAMATH_CALUDE_father_daughter_age_inconsistency_l2008_200832

theorem father_daughter_age_inconsistency :
  ¬ ∃ (x : ℕ), 
    (40 : ℝ) = 2 * 40 ∧ 
    (40 : ℝ) - x = 3 * (40 - x) :=
by sorry

end NUMINAMATH_CALUDE_father_daughter_age_inconsistency_l2008_200832


namespace NUMINAMATH_CALUDE_other_number_calculation_l2008_200800

theorem other_number_calculation (a b : ℕ+) 
  (h1 : Nat.lcm a b = 7700)
  (h2 : Nat.gcd a b = 11)
  (h3 : a = 308) :
  b = 275 := by
  sorry

end NUMINAMATH_CALUDE_other_number_calculation_l2008_200800


namespace NUMINAMATH_CALUDE_group_size_l2008_200845

theorem group_size (iceland : ℕ) (norway : ℕ) (both : ℕ) (neither : ℕ) 
  (h1 : iceland = 25)
  (h2 : norway = 23)
  (h3 : both = 21)
  (h4 : neither = 23) :
  iceland + norway - both + neither = 50 := by
  sorry

end NUMINAMATH_CALUDE_group_size_l2008_200845


namespace NUMINAMATH_CALUDE_expression_equals_one_l2008_200882

theorem expression_equals_one :
  (4 * 6) / (12 * 13) * (7 * 12 * 13) / (4 * 6 * 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_one_l2008_200882


namespace NUMINAMATH_CALUDE_system_solution_l2008_200885

theorem system_solution :
  ∀ (x y z : ℝ),
    ((x = 38 ∧ y = 4 ∧ z = 9) ∨ (x = 110 ∧ y = 2 ∧ z = 33)) →
    (x * y - 2 * y = x + 106) ∧
    (y * z + 3 * y = z + 39) ∧
    (z * x + 3 * x = 2 * z + 438) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2008_200885


namespace NUMINAMATH_CALUDE_tank_filling_time_l2008_200859

theorem tank_filling_time (fill_rate : ℝ → ℝ → ℝ) :
  (∀ (n : ℝ), fill_rate n (8 * n / 3) = 1) →
  fill_rate 2 12 = 1 :=
by sorry

end NUMINAMATH_CALUDE_tank_filling_time_l2008_200859


namespace NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l2008_200833

theorem smallest_sum_arithmetic_geometric (A B C D : ℤ) : 
  (∃ r : ℚ, B - A = C - B ∧ C = r * B ∧ D = r * C) →  -- Arithmetic and geometric sequence conditions
  C = (3/2) * B →                                    -- Given ratio
  A + B + C + D ≥ 21 :=                              -- Smallest possible sum
by sorry

end NUMINAMATH_CALUDE_smallest_sum_arithmetic_geometric_l2008_200833


namespace NUMINAMATH_CALUDE_digit_problem_l2008_200875

def Digit := Fin 9

def isConsecutive (a b : Digit) : Prop :=
  (a.val + 1 = b.val) ∨ (b.val + 1 = a.val)

def divides6DigitNumber (x : Nat) (a b c d e f : Digit) : Prop :=
  ∀ (perm : Fin 6 → Fin 6), x ∣ (100000 * (perm 0).val + 10000 * (perm 1).val + 1000 * (perm 2).val + 100 * (perm 3).val + 10 * (perm 4).val + (perm 5).val)

theorem digit_problem (a b c d e f : Digit) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ d ≠ e ∧ d ≠ f ∧ e ≠ f) :
  (∃ (x y : Digit), x ≠ y ∧ isConsecutive x y) ∧
  (∀ x : Nat, divides6DigitNumber x a b c d e f ↔ x = 1 ∨ x = 3 ∨ x = 9) :=
by sorry

end NUMINAMATH_CALUDE_digit_problem_l2008_200875


namespace NUMINAMATH_CALUDE_vampire_population_growth_l2008_200811

/-- Represents the vampire population growth in Willowton over two nights -/
theorem vampire_population_growth 
  (initial_population : ℕ) 
  (initial_vampires : ℕ) 
  (first_night_converts : ℕ) 
  (subsequent_night_increase : ℕ) : 
  initial_population ≥ 300 → 
  initial_vampires = 3 → 
  first_night_converts = 7 → 
  subsequent_night_increase = 1 → 
  (initial_vampires * first_night_converts + initial_vampires) * 
    (first_night_converts + subsequent_night_increase) + 
    (initial_vampires * first_night_converts + initial_vampires) = 216 := by
  sorry

end NUMINAMATH_CALUDE_vampire_population_growth_l2008_200811


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2008_200801

-- Define the hyperbola equation
def hyperbola_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y + 1)^2) - Real.sqrt ((x - 7)^2 + (y + 1)^2) = 4

-- Define the positive asymptote slope
def positive_asymptote_slope (h : ℝ → ℝ → Prop) : ℝ :=
  1

-- Theorem statement
theorem hyperbola_asymptote_slope :
  positive_asymptote_slope hyperbola_equation = 1 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2008_200801


namespace NUMINAMATH_CALUDE_square_side_lengths_l2008_200863

theorem square_side_lengths (s t : ℝ) : 
  (4 * s = 5 * (4 * t)) → -- perimeter of one square is 5 times the other
  (s + t = 60) →          -- sum of side lengths is 60
  ((s = 50 ∧ t = 10) ∨ (s = 10 ∧ t = 50)) := by
sorry

end NUMINAMATH_CALUDE_square_side_lengths_l2008_200863


namespace NUMINAMATH_CALUDE_special_triangle_properties_l2008_200868

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- Angles of the triangle
  A : ℝ
  B : ℝ
  C : ℝ
  -- Vector m
  m : ℝ × ℝ
  -- Vector n
  n : ℝ × ℝ
  -- Conditions
  m_def : m = (Real.cos B, Real.cos C)
  n_def : n = (2 * a + c, b)
  perpendicular : m.1 * n.1 + m.2 * n.2 = 0
  b_value : b = Real.sqrt 13
  a_c_sum : a + c = 4

/-- The main theorem about the special triangle -/
theorem special_triangle_properties (t : SpecialTriangle) :
  -- 1. The area of the triangle
  (1 / 2 : ℝ) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 4 ∧
  -- 2. The range of sin²A + sin²C
  (∀ x, ((Real.sin t.A)^2 + (Real.sin t.C)^2 = x) → (1 / 2 ≤ x ∧ x < 3 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_special_triangle_properties_l2008_200868


namespace NUMINAMATH_CALUDE_office_canteen_chairs_l2008_200894

/-- The number of round tables in the office canteen -/
def num_round_tables : ℕ := 2

/-- The number of rectangular tables in the office canteen -/
def num_rectangular_tables : ℕ := 2

/-- The number of chairs at each round table -/
def chairs_per_round_table : ℕ := 6

/-- The number of chairs at each rectangular table -/
def chairs_per_rectangular_table : ℕ := 7

/-- The total number of chairs in the office canteen -/
def total_chairs : ℕ := num_round_tables * chairs_per_round_table + num_rectangular_tables * chairs_per_rectangular_table

theorem office_canteen_chairs : total_chairs = 26 := by
  sorry

end NUMINAMATH_CALUDE_office_canteen_chairs_l2008_200894


namespace NUMINAMATH_CALUDE_dart_probability_l2008_200808

/-- The probability of a dart landing within a circle inscribed in a regular hexagonal dartboard -/
theorem dart_probability (s : ℝ) (h : s > 0) : 
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let circle_area := π * s^2
  circle_area / hexagon_area = 2 * π / (3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_dart_probability_l2008_200808


namespace NUMINAMATH_CALUDE_type_A_nutrition_l2008_200840

/-- Represents the nutritional content of food types A and B -/
structure FoodNutrition where
  protein_A : ℝ  -- Protein content per gram of type A food
  protein_B : ℝ  -- Protein content per gram of type B food
  iron_A : ℝ     -- Iron content per gram of type A food
  iron_B : ℝ     -- Iron content per gram of type B food

/-- Represents the meal composition and nutritional requirements -/
structure MealComposition where
  weight_A : ℝ    -- Weight of type A food in grams
  weight_B : ℝ    -- Weight of type B food in grams
  protein_req : ℝ -- Required protein in units
  iron_req : ℝ    -- Required iron in units

/-- Theorem stating the nutritional content of type A food -/
theorem type_A_nutrition 
  (food : FoodNutrition) 
  (meal : MealComposition) 
  (h1 : food.iron_A = 2 * food.protein_A)
  (h2 : food.iron_B = (4/7) * food.protein_B)
  (h3 : meal.weight_A * food.protein_A + meal.weight_B * food.protein_B = meal.protein_req)
  (h4 : meal.weight_A * food.iron_A + meal.weight_B * food.iron_B = meal.iron_req)
  (h5 : meal.weight_A = 28)
  (h6 : meal.weight_B = 30)
  (h7 : meal.protein_req = 35)
  (h8 : meal.iron_req = 40)
  : food.protein_A = 0.5 ∧ food.iron_A = 1 := by
  sorry

#check type_A_nutrition

end NUMINAMATH_CALUDE_type_A_nutrition_l2008_200840


namespace NUMINAMATH_CALUDE_inequality_comparison_l2008_200864

theorem inequality_comparison :
  (-14 ≤ 0) ∧
  (-2.1 ≤ -2.01) ∧
  (1/2 ≥ -1/3) ∧
  (-0.6 > -4/5) :=
by sorry

end NUMINAMATH_CALUDE_inequality_comparison_l2008_200864


namespace NUMINAMATH_CALUDE_impossible_placement_l2008_200815

/-- A function representing the placement of integers on a 35x35 table -/
def TablePlacement := Fin 35 → Fin 35 → ℤ

/-- The property that all integers in the table are different -/
def AllDifferent (t : TablePlacement) : Prop :=
  ∀ i j k l, t i j = t k l → (i = k ∧ j = l)

/-- The property that adjacent cells differ by at most 18 -/
def AdjacentDifference (t : TablePlacement) : Prop :=
  ∀ i j k l, (i = k ∧ (j.val + 1 = l.val ∨ j.val = l.val + 1)) ∨
             (j = l ∧ (i.val + 1 = k.val ∨ i.val = k.val + 1)) →
             |t i j - t k l| ≤ 18

/-- The main theorem stating the impossibility of the required placement -/
theorem impossible_placement : ¬∃ t : TablePlacement, AllDifferent t ∧ AdjacentDifference t := by
  sorry

end NUMINAMATH_CALUDE_impossible_placement_l2008_200815


namespace NUMINAMATH_CALUDE_hyperbola_center_l2008_200842

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 432 * y - 783 = 0

/-- The center of a hyperbola -/
def is_center (c : ℝ × ℝ) (eq : ℝ → ℝ → Prop) : Prop :=
  ∃ (a b : ℝ), ∀ (x y : ℝ),
    eq x y ↔ (x - c.1)^2 / a^2 - (y - c.2)^2 / b^2 = 1

/-- Theorem: The center of the given hyperbola is (3, 6) -/
theorem hyperbola_center : is_center (3, 6) hyperbola_eq :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l2008_200842


namespace NUMINAMATH_CALUDE_betty_sugar_purchase_l2008_200839

def min_sugar_purchase (s : ℝ) : Prop :=
  ∀ f : ℝ, (f ≥ 8 + (3/4) * s) ∧ (f ≤ 3 * s) → s ≥ 4

theorem betty_sugar_purchase :
  ∃ s : ℝ, min_sugar_purchase s ∧ ∀ t : ℝ, t < s → ¬ min_sugar_purchase t :=
sorry

end NUMINAMATH_CALUDE_betty_sugar_purchase_l2008_200839


namespace NUMINAMATH_CALUDE_xy_not_6_sufficient_not_necessary_l2008_200857

theorem xy_not_6_sufficient_not_necessary :
  (∃ x y : ℝ, (x ≠ 2 ∨ y ≠ 3) ∧ x * y = 6) ∧
  (∀ x y : ℝ, x * y ≠ 6 → (x ≠ 2 ∨ y ≠ 3)) :=
by sorry

end NUMINAMATH_CALUDE_xy_not_6_sufficient_not_necessary_l2008_200857


namespace NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2008_200802

/-- The probability of selecting a yellow jelly bean from a jar containing red, orange, yellow, and green jelly beans -/
theorem yellow_jelly_bean_probability
  (p_red : ℝ)
  (p_orange : ℝ)
  (p_green : ℝ)
  (h_red : p_red = 0.1)
  (h_orange : p_orange = 0.4)
  (h_green : p_green = 0.25)
  (h_sum : p_red + p_orange + p_green + (1 - p_red - p_orange - p_green) = 1) :
  1 - p_red - p_orange - p_green = 0.25 := by
sorry

end NUMINAMATH_CALUDE_yellow_jelly_bean_probability_l2008_200802


namespace NUMINAMATH_CALUDE_geometric_quadratic_no_roots_l2008_200883

/-- A function representing a quadratic equation with coefficients forming a geometric sequence -/
def geometric_quadratic (a b c : ℝ) : ℝ → ℝ := 
  fun x => a * x^2 + b * x + c

/-- Proposition: A quadratic function with coefficients forming a geometric sequence has no real roots -/
theorem geometric_quadratic_no_roots (a b c : ℝ) (h : b^2 = a*c) :
  ∀ x : ℝ, geometric_quadratic a b c x ≠ 0 := by
  sorry

#check geometric_quadratic_no_roots

end NUMINAMATH_CALUDE_geometric_quadratic_no_roots_l2008_200883


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l2008_200873

theorem third_root_of_cubic (c d : ℚ) :
  (∃ x : ℚ, c * x^3 + (c - 3*d) * x^2 + (2*d + 4*c) * x + (12 - 2*c) = 0) ∧
  (c * 1^3 + (c - 3*d) * 1^2 + (2*d + 4*c) * 1 + (12 - 2*c) = 0) ∧
  (c * (-3)^3 + (c - 3*d) * (-3)^2 + (2*d + 4*c) * (-3) + (12 - 2*c) = 0) →
  c * 4^3 + (c - 3*d) * 4^2 + (2*d + 4*c) * 4 + (12 - 2*c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l2008_200873


namespace NUMINAMATH_CALUDE_problem_solution_l2008_200821

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |3*x - 2|

-- Define the solution set of f(x) ≤ 5
def solution_set (a : ℝ) : Set ℝ := {x : ℝ | -4*a/5 ≤ x ∧ x ≤ 3*a/5}

-- State the theorem
theorem problem_solution (a b : ℝ) : 
  (∀ x : ℝ, f x ≤ 5 ↔ x ∈ solution_set a) →
  (a = 1 ∧ b = 2) ∧
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ (m : ℝ)^2 - 3*m) ∧
  (∃ m : ℝ, m = (3 + Real.sqrt 21) / 2 ∧
    ∀ m' : ℝ, (∀ x : ℝ, |x - 1| + |x + 2| ≥ m'^2 - 3*m') → m' ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2008_200821


namespace NUMINAMATH_CALUDE_octagon_area_l2008_200899

theorem octagon_area (r : ℝ) (h : r = 4) : 
  let octagon_area := 8 * (1/2 * r * r * Real.sin (π/4))
  octagon_area = 32 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_octagon_area_l2008_200899


namespace NUMINAMATH_CALUDE_bus_stop_distance_unit_l2008_200884

/-- Represents units of length measurement -/
inductive LengthUnit
  | Millimeter
  | Decimeter
  | Meter
  | Kilometer

/-- The distance between two bus stops in an arbitrary unit -/
def bus_stop_distance : ℕ := 3000

/-- Predicate to determine if a unit is appropriate for measuring bus stop distances -/
def is_appropriate_unit (unit : LengthUnit) : Prop :=
  match unit with
  | LengthUnit.Meter => True
  | _ => False

theorem bus_stop_distance_unit :
  is_appropriate_unit LengthUnit.Meter :=
sorry

end NUMINAMATH_CALUDE_bus_stop_distance_unit_l2008_200884


namespace NUMINAMATH_CALUDE_find_P_l2008_200841

theorem find_P : ∃ P : ℝ, (1/3 : ℝ) * (1/6 : ℝ) * P = (1/4 : ℝ) * (1/8 : ℝ) * 64 + (1/5 : ℝ) * (1/10 : ℝ) * 100 → P = 72 := by
  sorry

end NUMINAMATH_CALUDE_find_P_l2008_200841


namespace NUMINAMATH_CALUDE_simplify_expression_l2008_200871

theorem simplify_expression : 
  (Real.sqrt (Real.sqrt 81) - Real.sqrt (8 + 1/2))^2 = (70 - 12 * Real.sqrt 34) / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2008_200871


namespace NUMINAMATH_CALUDE_car_city_efficiency_approx_36_l2008_200835

/-- Represents the fuel efficiency of a car -/
structure CarFuelEfficiency where
  highway : ℝ  -- Miles per gallon on highway
  city : ℝ     -- Miles per gallon in city
  tank_size : ℝ -- Size of the fuel tank in gallons

/-- Conditions for the car's fuel efficiency -/
def efficiency_conditions (c : CarFuelEfficiency) : Prop :=
  c.highway * c.tank_size = 690 ∧
  c.city * c.tank_size = 420 ∧
  c.city = c.highway - 23

/-- Theorem stating that under given conditions, the car's city fuel efficiency is approximately 36 mpg -/
theorem car_city_efficiency_approx_36 (c : CarFuelEfficiency) 
  (h : efficiency_conditions c) : 
  ∃ ε > 0, |c.city - 36| < ε :=
sorry

end NUMINAMATH_CALUDE_car_city_efficiency_approx_36_l2008_200835


namespace NUMINAMATH_CALUDE_hidden_dots_on_dice_l2008_200805

theorem hidden_dots_on_dice (dice_count : Nat) (face_count : Nat) (visible_faces : Nat) (visible_sum : Nat) : 
  dice_count = 3 →
  face_count = 6 →
  visible_faces = 7 →
  visible_sum = 22 →
  (dice_count * face_count * (face_count + 1) / 2) - visible_sum = 41 := by
sorry

end NUMINAMATH_CALUDE_hidden_dots_on_dice_l2008_200805


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2008_200889

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2008_200889


namespace NUMINAMATH_CALUDE_natural_numbers_satisfying_condition_l2008_200865

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (q r : ℕ), n = 8 * q + r ∧ r < 8 ∧ q + r = 13

theorem natural_numbers_satisfying_condition :
  {n : ℕ | satisfies_condition n} = {108, 100, 92, 84, 76, 68, 60, 52, 44} :=
by sorry

end NUMINAMATH_CALUDE_natural_numbers_satisfying_condition_l2008_200865


namespace NUMINAMATH_CALUDE_complex_number_problem_l2008_200879

open Complex

theorem complex_number_problem :
  let z : ℂ := (1 - I)^2 + 3 + 6*I
  ∃ (a b : ℝ),
    z = 3 + 4*I ∧
    Complex.abs z = 5 ∧
    z^2 + a*z + b = -8 + 20*I ∧
    a = -1 ∧
    b = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l2008_200879


namespace NUMINAMATH_CALUDE_work_completion_time_l2008_200862

theorem work_completion_time
  (x_alone_time : ℝ)
  (y_alone_time : ℝ)
  (x_alone_days : ℝ)
  (h1 : x_alone_time = 20)
  (h2 : y_alone_time = 12)
  (h3 : x_alone_days = 4)
  : ∃ (total_time : ℝ), total_time = 10 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l2008_200862


namespace NUMINAMATH_CALUDE_filter_kit_solution_l2008_200817

def filter_kit_problem (last_filter_price : ℝ) : Prop :=
  let kit_price : ℝ := 72.50
  let filter1_price : ℝ := 12.45
  let filter2_price : ℝ := 14.05
  let discount_rate : ℝ := 0.1103448275862069
  let total_individual_price : ℝ := 2 * filter1_price + 2 * filter2_price + last_filter_price
  let amount_saved : ℝ := total_individual_price - kit_price
  amount_saved = discount_rate * total_individual_price

theorem filter_kit_solution :
  ∃ (last_filter_price : ℝ), filter_kit_problem last_filter_price ∧ last_filter_price = 28.50 := by
  sorry

end NUMINAMATH_CALUDE_filter_kit_solution_l2008_200817


namespace NUMINAMATH_CALUDE_problem_statement_l2008_200872

theorem problem_statement (a b c : ℚ) 
  (h : (3*a - 2*b + c - 4)^2 + (a + 2*b - 3*c + 6)^2 + (2*a - b + 2*c - 2)^2 ≤ 0) : 
  2*a + b - 4*c = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2008_200872


namespace NUMINAMATH_CALUDE_phone_price_increase_l2008_200870

/-- Proves that the percentage increase in the phone's price was 40% given the auction conditions --/
theorem phone_price_increase (tv_initial_price : ℝ) (tv_price_increase_ratio : ℝ) 
  (phone_initial_price : ℝ) (total_amount : ℝ) :
  tv_initial_price = 500 →
  tv_price_increase_ratio = 2 / 5 →
  phone_initial_price = 400 →
  total_amount = 1260 →
  let tv_final_price := tv_initial_price * (1 + tv_price_increase_ratio)
  let phone_final_price := total_amount - tv_final_price
  let phone_price_increase := (phone_final_price - phone_initial_price) / phone_initial_price
  phone_price_increase = 0.4 := by
sorry

end NUMINAMATH_CALUDE_phone_price_increase_l2008_200870


namespace NUMINAMATH_CALUDE_parking_garage_weekly_rate_l2008_200819

theorem parking_garage_weekly_rate :
  let monthly_rate : ℕ := 35
  let months_per_year : ℕ := 12
  let weeks_per_year : ℕ := 52
  let yearly_savings : ℕ := 100
  let weekly_rate : ℚ := (monthly_rate * months_per_year + yearly_savings) / weeks_per_year
  weekly_rate = 10 := by
sorry

end NUMINAMATH_CALUDE_parking_garage_weekly_rate_l2008_200819


namespace NUMINAMATH_CALUDE_houses_in_block_l2008_200824

/-- Proves that the number of houses in a block is 12, given the conditions of mail distribution. -/
theorem houses_in_block (total_mail : ℕ) (mail_per_house : ℕ) (skip_pattern : ℕ) :
  total_mail = 128 →
  mail_per_house = 16 →
  skip_pattern = 3 →
  ∃ (houses : ℕ), houses = 12 ∧ 
    houses * (skip_pattern - 1) * mail_per_house = total_mail * (skip_pattern - 1) / skip_pattern :=
by sorry

end NUMINAMATH_CALUDE_houses_in_block_l2008_200824


namespace NUMINAMATH_CALUDE_hot_dog_price_l2008_200804

-- Define variables for hamburger and hot dog prices
variable (h d : ℚ)

-- Define the equations based on the given conditions
def day1_equation : Prop := 3 * h + 4 * d = 10
def day2_equation : Prop := 2 * h + 3 * d = 7

-- Theorem statement
theorem hot_dog_price 
  (eq1 : day1_equation h d) 
  (eq2 : day2_equation h d) : 
  d = 1 := by sorry

end NUMINAMATH_CALUDE_hot_dog_price_l2008_200804


namespace NUMINAMATH_CALUDE_train_length_l2008_200818

/-- The length of a train given its passing times -/
theorem train_length (pole_time platform_time platform_length : ℝ) 
  (h1 : pole_time = 15)
  (h2 : platform_time = 40)
  (h3 : platform_length = 100) : ℝ :=
by
  -- The length of the train is 60 meters
  exact 60

end NUMINAMATH_CALUDE_train_length_l2008_200818


namespace NUMINAMATH_CALUDE_simple_interest_problem_l2008_200898

/-- Proves that given a sum P at simple interest for 10 years, 
    if increasing the interest rate by 5% results in Rs. 400 more interest, 
    then P = 800. -/
theorem simple_interest_problem (P R : ℝ) 
  (h1 : P > 0) 
  (h2 : R > 0) 
  (h3 : (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400) : 
  P = 800 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l2008_200898


namespace NUMINAMATH_CALUDE_coin_flip_problem_l2008_200844

theorem coin_flip_problem (total_coins : ℕ) (two_ruble : ℕ) (five_ruble : ℕ) :
  total_coins = 14 →
  two_ruble > 0 →
  five_ruble > 0 →
  two_ruble + five_ruble = total_coins →
  ∃ (k : ℕ), k > 0 ∧ 3 * k = 2 * two_ruble + 5 * five_ruble →
  five_ruble = 4 ∨ five_ruble = 8 ∨ five_ruble = 12 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_problem_l2008_200844


namespace NUMINAMATH_CALUDE_ab_value_l2008_200878

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 2) (h2 : a + b = 3) : a * b = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2008_200878


namespace NUMINAMATH_CALUDE_distance_lima_caracas_l2008_200880

theorem distance_lima_caracas : 
  let caracas : ℂ := 0
  let lima : ℂ := 960 + 1280 * Complex.I
  Complex.abs (lima - caracas) = 1600 := by
sorry

end NUMINAMATH_CALUDE_distance_lima_caracas_l2008_200880


namespace NUMINAMATH_CALUDE_g_of_3_equals_19_l2008_200823

def g (x : ℝ) : ℝ := x^2 + 3*x + 1

theorem g_of_3_equals_19 : g 3 = 19 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_19_l2008_200823


namespace NUMINAMATH_CALUDE_number_difference_l2008_200831

theorem number_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_eq : x - y = 19) 
  (square_diff_eq : x^2 - y^2 = 190) : 
  x - y = 19 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l2008_200831


namespace NUMINAMATH_CALUDE_circle_area_comparison_l2008_200809

theorem circle_area_comparison (R : ℝ) (h : R > 0) :
  let large_circle_area := π * R^2
  let small_circle_radius := R / 2
  let small_circle_area := π * small_circle_radius^2
  let total_small_circles_area := 4 * small_circle_area
  large_circle_area = total_small_circles_area := by
sorry

end NUMINAMATH_CALUDE_circle_area_comparison_l2008_200809


namespace NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2008_200852

def triangular_sum (n : ℕ) : ℕ := n * (n + 1) / 2

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10 + sum_of_digits (n / 10))

theorem sum_of_digits_of_triangular_array_rows :
  ∃ (N : ℕ), triangular_sum N = 2080 ∧ sum_of_digits N = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_triangular_array_rows_l2008_200852


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_l2008_200895

theorem smallest_integer_gcd_lcm (m n x : ℕ) : 
  m > 0 → n > 0 → x > 0 →
  m = 60 ∨ n = 60 →
  Nat.gcd m n = x + 3 →
  Nat.lcm m n = x * (x + 3) →
  (m = 60 → n ≥ 45) ∧ (n = 60 → m ≥ 45) :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_l2008_200895


namespace NUMINAMATH_CALUDE_max_bullet_speed_correct_l2008_200893

/-- Represents a ring moving on a segment --/
structure MovingRing where
  segmentLength : ℝ
  speed : ℝ

/-- Represents the game setup --/
structure GameSetup where
  ringCD : MovingRing
  ringEF : MovingRing
  ringGH : MovingRing
  AO : ℝ
  OP : ℝ
  PQ : ℝ

/-- The maximum bullet speed that allows passing through all rings --/
def maxBulletSpeed (setup : GameSetup) : ℝ :=
  4.5

/-- Theorem stating the maximum bullet speed --/
theorem max_bullet_speed_correct (setup : GameSetup) 
  (h1 : setup.ringCD.segmentLength = 20)
  (h2 : setup.ringEF.segmentLength = 20)
  (h3 : setup.ringGH.segmentLength = 20)
  (h4 : setup.ringCD.speed = 5)
  (h5 : setup.ringEF.speed = 9)
  (h6 : setup.ringGH.speed = 27)
  (h7 : setup.AO = 45)
  (h8 : setup.OP = 20)
  (h9 : setup.PQ = 20) :
  maxBulletSpeed setup = 4.5 := by
  sorry

#check max_bullet_speed_correct

end NUMINAMATH_CALUDE_max_bullet_speed_correct_l2008_200893


namespace NUMINAMATH_CALUDE_trigonometric_identity_l2008_200881

theorem trigonometric_identity (α : ℝ) :
  (2 * (Real.cos ((9 / 4) * π - α))^2) / (1 + Real.cos ((π / 2) + 2 * α)) -
  (Real.sin (α + (7 / 4) * π) / Real.sin (α + (π / 4))) *
  (1 / Real.tan ((3 / 4) * π - α)) =
  (4 * Real.sin (2 * α)) / (Real.cos (2 * α))^2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l2008_200881


namespace NUMINAMATH_CALUDE_total_cost_is_183_l2008_200814

/-- Represents the jewelry store inventory and pricing --/
structure JewelryStore where
  necklaceCapacity : ℕ
  currentNecklaces : ℕ
  ringCapacity : ℕ
  currentRings : ℕ
  braceletCapacity : ℕ
  currentBracelets : ℕ
  necklacePrice : ℕ
  ringPrice : ℕ
  braceletPrice : ℕ

/-- Calculates the total cost to fill the displays --/
def totalCost (store : JewelryStore) : ℕ :=
  (store.necklaceCapacity - store.currentNecklaces) * store.necklacePrice +
  (store.ringCapacity - store.currentRings) * store.ringPrice +
  (store.braceletCapacity - store.currentBracelets) * store.braceletPrice

/-- The specific jewelry store in the problem --/
def problemStore : JewelryStore := {
  necklaceCapacity := 12
  currentNecklaces := 5
  ringCapacity := 30
  currentRings := 18
  braceletCapacity := 15
  currentBracelets := 8
  necklacePrice := 4
  ringPrice := 10
  braceletPrice := 5
}

/-- Theorem stating that the total cost to fill the displays is $183 --/
theorem total_cost_is_183 : totalCost problemStore = 183 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_183_l2008_200814


namespace NUMINAMATH_CALUDE_horner_v4_for_f_at_neg_two_l2008_200813

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 3x^6 + 5x^5 + 6x^4 + 20x^3 - 8x^2 + 35x + 12 -/
def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

/-- Theorem: v_4 in Horner's method for f(x) when x = -2 is -16 -/
theorem horner_v4_for_f_at_neg_two :
  let x : ℝ := -2
  let v0 : ℝ := 3
  let v1 : ℝ := horner_step 5 x v0
  let v2 : ℝ := horner_step 6 x v1
  let v3 : ℝ := horner_step 20 x v2
  let v4 : ℝ := horner_step (-8) x v3
  v4 = -16 := by sorry

end NUMINAMATH_CALUDE_horner_v4_for_f_at_neg_two_l2008_200813


namespace NUMINAMATH_CALUDE_lattice_point_theorem_four_point_counterexample_l2008_200820

-- Define a lattice point as a pair of integers
def LatticePoint := ℤ × ℤ

-- Define the set of all lattice points
def S : Set LatticePoint := Set.univ

-- Define a function to check if a point is between two other points
def isBetween (p q r : LatticePoint) : Prop :=
  ∃ t : ℚ, 0 < t ∧ t < 1 ∧ 
    (t * p.1 + (1 - t) * q.1 = r.1) ∧
    (t * p.2 + (1 - t) * q.2 = r.2)

-- State the theorem
theorem lattice_point_theorem (A B C : LatticePoint) 
  (hA : A ∈ S) (hB : B ∈ S) (hC : C ∈ S) 
  (hAB : A ≠ B) (hBC : B ≠ C) (hAC : A ≠ C) :
  ∃ D : LatticePoint, D ∈ S ∧ D ≠ A ∧ D ≠ B ∧ D ≠ C ∧
    (∀ P : LatticePoint, P ∈ S → 
      ¬(isBetween A D P) ∧ ¬(isBetween B D P) ∧ ¬(isBetween C D P)) :=
sorry

-- Counter-example for 4 points
theorem four_point_counterexample :
  ∃ A B C D : LatticePoint, 
    A ∈ S ∧ B ∈ S ∧ C ∈ S ∧ D ∈ S ∧
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    ∀ E : LatticePoint, E ∈ S →
      (isBetween A B E) ∨ (isBetween A C E) ∨ (isBetween A D E) ∨
      (isBetween B C E) ∨ (isBetween B D E) ∨ (isBetween C D E) :=
sorry

end NUMINAMATH_CALUDE_lattice_point_theorem_four_point_counterexample_l2008_200820


namespace NUMINAMATH_CALUDE_second_x_intercept_of_quadratic_l2008_200854

/-- Given a quadratic function with vertex at (5, -3) and one x-intercept at (1, 0),
    the x-coordinate of the second x-intercept is 9. -/
theorem second_x_intercept_of_quadratic 
  (f : ℝ → ℝ) 
  (a b c : ℝ) 
  (h1 : ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = -3) : 
  ∃ x, f x = 0 ∧ x = 9 := by
sorry

end NUMINAMATH_CALUDE_second_x_intercept_of_quadratic_l2008_200854


namespace NUMINAMATH_CALUDE_derivative_f_at_1_l2008_200829

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 4 * x^2 + 10 * x - 5

-- State the theorem
theorem derivative_f_at_1 : 
  (deriv f) 1 = 11 := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_1_l2008_200829


namespace NUMINAMATH_CALUDE_problem_statement_l2008_200891

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + a + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x + |x^2 - 1|

def A (a : ℝ) : Set ℝ := {x | f a x ≤ 0}

theorem problem_statement (a : ℝ) :
  (∀ x ∈ A a, x ∈ Set.Icc 0 3) →
  -1 < a ∧ a ≤ 11/5 ∧
  (∃ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 3 ∧ g a x₁ = 0 ∧ g a x₂ = 0) →
  1 + Real.sqrt 3 < a ∧ a < 19/5 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2008_200891
